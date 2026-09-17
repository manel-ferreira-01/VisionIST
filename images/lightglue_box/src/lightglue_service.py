"""lightglue box — SuperPoint / DISK features + LightGlue matching.

A standard shared-envelope box (one RPC: ``Process``) focused on one job,
done well and fast: robust feature matching between image pairs.  The box
is a *thin* wrapper over the LightGlue network — it returns what the network
returns (per-image keypoints/descriptors/scores, and the matcher's
``matches`` indices); turning matches into epipolar geometry (F, E, poses,
triangulation, ...) is left to the caller, which holds the camera model.

Commands in the ``config_json`` section
``{"lightglue": {"command": ..., "parameters": {...}}}``:

* **``match``** (the default when no command is given) — extract features
  (SuperPoint or DISK) for each input image; with exactly two images in,
  also run the LightGlue matcher and return its ``matches`` indices (plus
  ``confidence``).  Stateless.
* **``reset``** — accepted by every standard box; on this box it is a plain
  no-op (nothing is held between calls).

Contract (see docs/gRPC_Services_Reference.md):

* response ``config_json`` is **namespaced**: ``{"lightglue": {"status": …}}``
  with ``status`` in ``done | empty_request | error`` (the human reason in
  ``"error"`` on failure), plus ``runtime`` and box-specific fields.
* the response declares its payload encoding (``"encoding": {field: codec}``):
  every array output is an ``np.save`` (``.npy``) blob declared ``numpy`` —
  ``boxes_client`` decodes them to ``np.ndarray`` keeping shape and dtype.

GPU lifecycle (fleet convention): models load lazily on first use, live on
``parameters.device`` (default: CUDA if visible, else CPU), and a watchdog
thread parks them on CPU after ``_IDLE_TIMEOUT`` seconds of inactivity, so
``empty_cache()`` reclaims VRAM.  Model pairs are cached per
(extractor, max_keypoints, filter_threshold), so repeat calls with the same parameters skip
all setup work.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import threading
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'
_IDLE_TIMEOUT = 60  # seconds: models park on CPU after this much idle time

# Per-command parameter defaults (``parameters`` in the config section).
_MATCH_DEFAULTS = {
    "feature_extractor": "SUPERPOINT",   # SUPERPOINT | DISK
    "max_keypoints": 1024,
}
_SUPPORTED_EXTRACTORS = ("SUPERPOINT", "DISK")

def torch_device_default():
    """CUDA if visible, else CPU (lazy: torch is optional for CPU boxes)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _done(extra, data=None, encoding=None):
    """Standard success envelope: namespaced status + declared encoding."""
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"lightglue": section}),
        data=data or {},
    )


def _empty_request():
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"lightglue": {"status": "empty_request"}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps(
            {"lightglue": {"status": "error", "error": message}}))


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def np_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize with np.save — keeps the array's dtype and shape in the
    blob, so the client's ``numpy`` codec (``np.load``) restores it exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def pad_and_stack(arrays, pad_value=0.0):
    """Pad a list of arrays to the same shape and stack on axis 0."""
    if not arrays:
        return np.zeros((0, 0))
    max_shape = np.max([np.array(a.shape) for a in arrays], axis=0)
    padded = []
    for a in arrays:
        pad_width = [(0, int(m - s)) for s, m in zip(a.shape, max_shape)]
        padded.append(np.pad(a, pad_width, mode='constant',
                             constant_values=pad_value))
    return np.stack(padded, axis=0)


def _num(parameters, key, default, cast, non_negative=True):
    """Coerce ``parameters[key]`` (falling back to ``default``) to a number."""
    v = parameters.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.{key} must be a number, got {v!r}") from e
    if non_negative and v < 0:
        raise ValueError(f"parameters.{key} must be >= 0, got {v!r}")
    return v


def _parse_match_parameters(parameters):
    fx = str(parameters.get("feature_extractor")
             or _MATCH_DEFAULTS["feature_extractor"]).upper()
    if fx not in _SUPPORTED_EXTRACTORS:
        raise ValueError(
            f"parameters.feature_extractor must be one of "
            f"{list(_SUPPORTED_EXTRACTORS)}, got {fx!r}")
    spec = {
        "extractor": fx,
        "max_keypoints": _num(parameters, "max_keypoints",
                              int(_MATCH_DEFAULTS["max_keypoints"]), int),
    }
    if spec["max_keypoints"] < 8:
        raise ValueError(f"parameters.max_keypoints must be >= 8, "
                         f"got {spec['max_keypoints']}")

    # Optional LightGlue match-confidence filter (None = its default, 0.1).
    # Baked into the matcher conf: in this LightGlue version it is a
    # constructor parameter (``filter_threshold``), not a forward kwarg.
    if "filter_threshold" in parameters and parameters["filter_threshold"] is not None:
        ct = float(_num(parameters, "filter_threshold", 0.1, float))
        if ct > 1:
            raise ValueError(f"parameters.filter_threshold must be in [0, 1], "
                             f"got {ct}")
        spec["filter_threshold"] = ct
    else:
        spec["filter_threshold"] = None
    return spec


# ---------------------------------------------
# Service Definition
# ---------------------------------------------
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        # Model cache: (extractor, max_keypoints, filter_threshold) -> pair
        self._models = {}
        self._device = torch_device_default()  # device models currently live on
        self._models_lock = threading.Lock()

        self._last_request_time = time.time()
        # Watchdog: park the (lazy) models back on CPU when idle,
        # same GPU lifecycle as the rest of the fleet.
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop,
                                                 daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------- device
    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._models_lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda"):
                    logging.info("Idle timeout reached: moving models to CPU")
                    for extractor, matcher in self._models.values():
                        extractor.to("cpu")
                        matcher.to("cpu")
                    self._device = "cpu"
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def _get_models(self, extractor, max_keypoints, filter_threshold, device):
        """Return (extractor, matcher) for the LightGlue feature set
        ``extractor`` (``"superpoint"`` | ``"disk"``), on ``device``, with
        ``filter_threshold`` baked in (None = LightGlue default).

        Builds are cached per (extractor, max_keypoints, filter_threshold)
        and models are only moved between devices on demand, so repeated
        calls with the same parameters skip all setup."""
        with self._models_lock:
            self._last_request_time = time.time()
            target = str(device)
            key = (extractor, max_keypoints, filter_threshold)
            model = self._models.get(key)
            if model is None:
                from lightglue import LightGlue, SuperPoint, DISK
                if extractor == "disk":
                    feat = DISK(max_num_keypoints=max_keypoints).eval()
                else:
                    feat = SuperPoint(max_num_keypoints=max_keypoints).eval()
                lg_conf = ({"filter_threshold": filter_threshold}
                           if filter_threshold is not None else {})
                matcher = LightGlue(features=extractor, **lg_conf).eval()
                feat.to(target)
                matcher.to(target)
                model = (feat, matcher)
                self._models[key] = model
                logging.info(f"LightGlue initialized: {extractor} "
                             f"(max_keypoints={max_keypoints}, "
                             f"filter_threshold={filter_threshold}) on {target}")

            current = next(iter(model[0].parameters())).device
            want = target.split(":")[0]
            want_index = None if ":" not in target \
                else int(target.split(":")[1])
            if current.type != want or \
                    (want_index is not None and current.index != want_index):
                logging.info(f"Moving LightGlue models to {target}")
                for m in model:
                    m.to(target)
            return model

    # ------------------------------------------------------------- inputs
    @staticmethod
    def _decode_images(image_bytes_list):
        """JPEG/PNG bytes -> BGR ndarray list; raises on any bad frame."""
        frames = []
        for i, raw in enumerate(image_bytes_list):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(image_bytes_list)} (not a supported raster?)")
            frames.append(img)
        return frames

    # ------------------------------------------------------------ matching
    @staticmethod
    def _img2tensor(img_bgr, device):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    def _run_match(self, imgs_in, spec, device):
        """LightGlue: 1 image -> its features; 2 images -> + the matcher's
        output (``matches`` indices and ``confidence``).

        Returns (data_dict, config_extra)."""
        extractor, matcher = self._get_models(
            spec["extractor"].lower(), spec["max_keypoints"],
            spec["filter_threshold"], device)

        import torch
        images_in = [self._img2tensor(img, device) for img in imgs_in]
        with torch.no_grad():
            feats = [extractor.extract(img) for img in images_in]   # one dict per image
            if len(imgs_in) == 2:
                match_res = matcher(
                    {"image0": feats[0], "image1": feats[1]})

        def _to_np(v):
            if isinstance(v, (list, tuple)):
                v = v[0]                              # LightGlue batches as lists
            t = v.detach().cpu() if hasattr(v, "cpu") else v
            return np.asarray(t, dtype=np.float32)

        out = {}
        n_images = len(imgs_in)
        # Per-image arrays: stack over images, padding ragged N as needed.
        # (newer lightglue calls the per-keypoint scores ``keypoint_scores``,
        # older ones ``score`` — accept both)
        for key, source_keys, reshape in (
                ("keypoints", ("keypoints",), lambda x: x.reshape(-1, 2)),
                ("descriptors", ("descriptors",),
                 lambda x: x.reshape(-1, x.shape[-1])),
                ("scores", ("keypoint_scores", "score"), lambda x: x.reshape(-1))):
            per_image = []
            for f in feats:
                x = next((f.get(k) for k in source_keys if f.get(k) is not None), None)
                per_image.append(_to_np(x) if x is not None else None)
            if any(x is None for x in per_image):
                continue                      # this extractor omits it
            out[key] = pad_and_stack([reshape(x) for x in per_image])

        if n_images == 2:
            for key in ("matches", "confidence"):
                v = match_res.get(key)
                if v is None:
                    v = next((match_res.get(k) for k in ("confidence", "scores")
                              if match_res.get(k) is not None), None) if key == "confidence" else None
                if v is None:
                    continue
                a = _to_np(v)
                if key == "matches":
                    out[key] = a.astype(np.int64).reshape(-1, 2)
                else:
                    out[key] = a.astype(np.float32).reshape(-1)

        extra = {"num_matches": int(out["matches"].shape[0])} \
            if "matches" in out else {}
        return out, extra

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()
        self._last_request_time = time.time()  # keep the idle watchdog honest

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get("lightglue"), dict):
                return _error("config section 'lightglue' missing or not an object")
            lg = config["lightglue"]
            command = lg.get("command") or "match"
            parameters = lg.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            # --- reset: plain no-op (this box is fully stateless) ---------
            if command == "reset":
                return _done({"action": "reset"})

            if command != "match":
                return _error(
                    f"unknown command {command!r} (expected 'match' or 'reset')")

            images = unwrap_value(request.data["images"]) if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()
            if len(images) > 2:
                return _error(
                    f"LightGlue matching is pairwise: expected 1 or 2 images, "
                    f"got {len(images)}")

            spec = _parse_match_parameters(parameters)

            device = torch_device_default()
            req_device = str(parameters.get("device") or "").strip().lower()
            if req_device:
                device = req_device

            imgs_in = self._decode_images(images)
            data, extra = self._run_match(imgs_in, spec, device)

            data = {key: wrap_value(np_to_bytes(arr))
                    for key, arr in data.items()}
            encoding = {key: "numpy" for key in data}
            return _done(
                {
                    "matcher": f"LightGlue ({spec['extractor'].lower()})",
                    "feature_extractor": spec["extractor"],
                    "max_keypoints": spec["max_keypoints"],
                    "device": device,
                    "num_images": len(imgs_in),
                    **extra,
                    "runtime": time.time() - start_time,
                },
                data=data, encoding=encoding)
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


# ---------------------------------------------
# Server setup
# ---------------------------------------------
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if port <= 0:
            logging.error("Port must be positive")
            return None
        return port
    except ValueError:
        logging.exception("Invalid port value")
        return None


def run_server(server):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ],
    )
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
