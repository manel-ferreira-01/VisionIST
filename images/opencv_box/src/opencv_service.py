"""opencv box — classic computer-vision utilities behind the shared envelope.

A standard shared-envelope box (one RPC: ``Process``) with three commands in
the ``config_json`` section ``{"opencv": {"command": ..., "parameters": {...}}}``:

* **``match``** (the default when no command is given) — extract features
  (SIFT / ORB via OpenCV, or SuperPoint / DISK via LightGlue) for each input
  image; with exactly two images in, also compute the matches plus a RANSAC
  fundamental matrix.  Stateless.
* **``similarity_check``** — stateful frame-change detector for video
  streams: Lucas–Kanade mean point displacement (``parameters.method:
  "motion"``, the default) or SSIM (``"ssim"``) of the incoming frame
  against the last *changed* frame.  The reference frame is the box's only
  state (``command: reset`` clears it; ``match`` is stateless).
* **``reset``** — clear the ``similarity_check`` state.  Accepted by every
  standard box; here it is a real no-op for the stateless ``match`` path.

Contract (see docs/gRPC_Services_Reference.md):

* response ``config_json`` is **namespaced**: ``{"opencv": {"status": …}}``
  with ``status`` in ``done | empty_request | error`` (the human reason in
  ``"error"`` on failure), plus ``runtime`` and box-specific fields.
* the response declares its payload encoding (``"encoding": {field: codec}``):
  all feature outputs are ``np.save`` (``.npy``) blobs declared ``numpy`` —
  ``boxes_client`` decodes them to ``np.ndarray`` keeping their shape;
  ``similarity_check`` echoes the changed frame as raw JPEG bytes
  (declared ``identity``).
* ``parameters.device`` (optional; ``"cpu"`` / ``"cuda"`` / ``"cuda:0"``)
  wins over the default (CUDA if visible) for the LightGlue models.
* GPU lifecycle (fleet convention): LightGlue loads lazily on first use,
  lives on the requested device, and a watchdog thread parks it on CPU after
  ``_IDLE_TIMEOUT`` seconds of inactivity, so ``empty_cache()`` reclaims VRAM.

Migration note: the box's pre-contract extra RPC ``similarity_check(Envelope)``
is gone — callers now send ``Process`` with ``command: "similarity_check"``.
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
_IDLE_TIMEOUT = 60  # seconds: LightGlue models park on CPU after this much idle time

# Per-command parameter defaults (``parameters`` in the config section).
_MATCH_DEFAULTS = {
    "feature_extractor": "SIFT",   # SIFT | ORB | SUPERPOINT | DISK | (LIGHTGLUE)
    "ratio_thresh": 0.75,          # Lowe's ratio test (FLANN path)
    "max_keypoints": 500,          # raised to 2048 automatically for LightGlue
}
_MATCH_LG_MAX_KEYPOINTS = 2048
_SIM_DEFAULTS = {
    "method": "motion",            # "motion" (Lucas–Kanade) or "ssim"
    "motion_thresh": 1.5,          # px mean displacement (motion method)
    "ssim_thresh": 0.90,           # SSIM gate (ssim method)
    "blur_kernel": 5,
    "max_corners": 200,
    "quality_level": 0.01,
    "min_distance": 5,
    "block_size": 7,
}


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
        config_json=json.dumps({"opencv": section}),
        data=data or {},
    )


def _empty_request():
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"opencv": {"status": "empty_request"}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"opencv": {"status": "error", "error": message}}))


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
    """Pad a list of 2-D arrays to the same shape and stack on axis 0."""
    if not arrays:
        return np.zeros((0, 0))
    max_shape = np.max([np.array(a.shape) for a in arrays], axis=0)
    padded = []
    for a in arrays:
        pad_width = [(0, int(m - s)) for s, m in zip(a.shape, max_shape)]
        padded.append(np.pad(a, pad_width, mode='constant', constant_values=pad_value))
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
    fx = str(parameters.get("feature_extractor") or _MATCH_DEFAULTS["feature_extractor"])
    return {
        "feature_extractor": fx,
        "ratio_thresh": _num(parameters, "ratio_thresh",
                             float(_MATCH_DEFAULTS["ratio_thresh"]), float),
        "max_keypoints": _num(parameters, "max_keypoints",
                              int(_MATCH_DEFAULTS["max_keypoints"]), int),
    }


def _parse_sim_parameters(parameters):
    spec = {
        "motion_thresh": _num(parameters, "motion_thresh",
                              float(_SIM_DEFAULTS["motion_thresh"]), float),
        "ssim_thresh": _num(parameters, "ssim_thresh",
                            float(_SIM_DEFAULTS["ssim_thresh"]), float),
        "blur_kernel": int(_num(parameters, "blur_kernel",
                                int(_SIM_DEFAULTS["blur_kernel"]), int)),
        "max_corners": _num(parameters, "max_corners",
                            int(_SIM_DEFAULTS["max_corners"]), int),
        "quality_level": _num(parameters, "quality_level",
                              float(_SIM_DEFAULTS["quality_level"]), float),
        "min_distance": _num(parameters, "min_distance",
                             int(_SIM_DEFAULTS["min_distance"]), int),
        "block_size": _num(parameters, "block_size",
                           int(_SIM_DEFAULTS["block_size"]), int),
    }
    method = str(parameters.get("method") or _SIM_DEFAULTS["method"]).lower()
    if method not in ("motion", "ssim"):
        raise ValueError(
            f"parameters.method must be 'motion' or 'ssim', got {parameters.get('method')!r}")
    spec["method"] = method
    return spec


def _parse_extractor(fx_param):
    """feature_extractor parameter -> (name, is_lightglue).

    Accepts the pre-contract values the LightGlue path was wired for
    (SUPERPOINT / DISK / LIGHTGLUE) in either case."""
    fx_upper = str(fx_param).upper()
    if 'SUPERPOINT' in fx_upper or 'DISK' in fx_upper or 'LIGHTGLUE' in fx_upper:
        return ('superpoint', True)
    return (fx_upper, False)


# ---------------------------------------------
# Service Definition
# ---------------------------------------------
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        # similarity_check state (the box's only state; ``match`` is stateless)
        self._sim_lock = threading.Lock()
        self._prev_gray = None
        self._prev_points = None
        self._frames_checked = 0

        # LightGlue models (lazy initialized on first LightGlue ``match``)
        self._lg_extractor = None
        self._lg_matcher = None
        self._lg_type = None        # "superpoint" | "disk" the models were built as
        self._lg_placed = None      # device the models currently live on (None = not loaded)
        self._lg_lock = threading.Lock()

        self._last_request_time = time.time()
        # Watchdog: park the (lazy) LightGlue models back on CPU when idle,
        # same GPU lifecycle as the rest of the fleet.
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------- device
    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._lg_lock:
                idle_time = time.time() - self._last_request_time
                if (idle_time > _IDLE_TIMEOUT
                        and self._lg_placed is not None
                        and self._lg_placed.startswith("cuda")
                        and self._lg_extractor is not None):
                    logging.info("Idle timeout reached: moving LightGlue models to CPU")
                    self._lg_extractor.to("cpu")
                    self._lg_matcher.to("cpu")
                    self._lg_placed = "cpu"
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def _place_lightglue(self, extractor_type, device, max_kpts):
        """Load (first use) / move the LightGlue models to ``device``.

        ``extractor_type`` ("superpoint" | "disk") picks the feature model;
        switching types after the fact rebuilds the pair.  May have been
        parked on CPU by the watchdog in the meantime."""
        with self._lg_lock:
            self._last_request_time = time.time()
            target = str(device)
            if self._lg_extractor is None or self._lg_type != extractor_type:
                from lightglue import LightGlue, SuperPoint, DISK
                if extractor_type == "disk":
                    self._lg_extractor = DISK(max_num_keypoints=max_kpts).eval().to(target)
                    self._lg_matcher = LightGlue(features='disk').eval().to(target)
                else:
                    self._lg_extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(target)
                    self._lg_matcher = LightGlue(features='superpoint').eval().to(target)
                self._lg_type = extractor_type
                self._lg_placed = target
                logging.info(f"LightGlue initialized: {extractor_type} on {target}")
            elif self._lg_placed != target:
                logging.info(f"Moving LightGlue models to {target}")
                self._lg_extractor.to(target)
                self._lg_matcher.to(target)
                self._lg_placed = target

    def _clear_similarity_state(self):
        """``command: reset`` — drop the reference frame and tracked points."""
        with self._sim_lock:
            self._prev_gray = None
            self._prev_points = None
            self._frames_checked = 0
            logging.info("similarity_check state cleared (reset)")

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
    def _match_opencv(self, imgs_in, spec):
        """SIFT/ORB + FLANN + RANSAC fundamental matrix. -> plain arrays."""
        extractor_name = spec["feature_extractor"]
        if extractor_name == "SIFT":
            detector = cv2.SIFT_create(nfeatures=spec["max_keypoints"])
        else:
            detector = cv2.ORB_create(nfeatures=spec["max_keypoints"])

        keypoints_list, descriptors_list = [], []
        for img in imgs_in:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, desc = detector.detectAndCompute(gray, None)
            desc = np.zeros((0, 128), np.float32) if desc is None else desc.astype(np.float32)
            kp_arr = cv2.KeyPoint_convert(kps)
            if kp_arr.size == 0:
                kp_arr = np.zeros((0, 2))
            keypoints_list.append(kp_arr.astype(np.float32))
            descriptors_list.append(desc)

        out = {
            "keypoints": pad_and_stack(keypoints_list, pad_value=0.0),
            "descriptors": pad_and_stack(descriptors_list, pad_value=0.0),
        }
        if len(imgs_in) != 2:
            # Single image: extraction only (no matching section).
            return out

        # The matching-section field names are the pre-contract ones the old
        # callers (and fleet/supervisor_demo.py) already read: matches_inliers_*.

        descA, descB = descriptors_list[0], descriptors_list[1]
        if extractor_name == "SIFT":
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        else:
            flann = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                dict(checks=50))

        good_matches = []
        if descA.shape[0] and descB.shape[0]:
            try:
                knn_matches = flann.knnMatch(descA, descB, k=2)
            except cv2.error:
                # OpenCV 5.x can reject plain 2-D float32 matrices in
                # FLANN's knnMatch ("Unsupported format") — brute force is
                # equivalent for the candidate counts used here.
                bf = cv2.BFMatcher()
                knn_matches = bf.knnMatch(descA, descB, k=2)
            good_matches = [m for m, n in knn_matches
                            if m.distance < spec["ratio_thresh"] * n.distance]

        out["matches_inliers_a"] = np.zeros((0, 2))
        out["matches_inliers_b"] = np.zeros((0, 2))
        out["fundamental_matrix"] = np.zeros((0, 0))
        if len(good_matches) >= 8:
            ptsA = np.float32([keypoints_list[0][m.queryIdx] for m in good_matches
                               if m.queryIdx < len(keypoints_list[0])])
            ptsB = np.float32([keypoints_list[1][m.trainIdx] for m in good_matches
                               if m.trainIdx < len(keypoints_list[1])])
            if len(ptsA) >= 8:
                F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.5, 0.999)
                if F is not None and mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    out["matches_inliers_a"] = ptsA[inlier_mask]
                    out["matches_inliers_b"] = ptsB[inlier_mask]
                    out["fundamental_matrix"] = F
        return out

    def _match_lightglue(self, imgs_in, spec, device):
        """SuperPoint/DISK + LightGlue matching + RANSAC fundamental (needs
        exactly two images)."""
        max_kpts = max(spec["max_keypoints"], 64)
        self._place_lightglue(
            "disk" if str(spec["feature_extractor"]).lower() == "disk" else "superpoint",
            device, max_kpts)

        import torch

        # Helper to format image tensor for SuperPoint [1, C, H, W]
        def img2tensor(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        feats0 = self._lg_extractor.extract(img2tensor(imgs_in[0]))
        feats1 = self._lg_extractor.extract(img2tensor(imgs_in[1]))

        # Match features
        matches_res = self._lg_matcher({'image0': feats0, 'image1': feats1})

        # Safely unpack per-image entries (removing the batch dim if present)
        def _first(t):
            if isinstance(t, list):
                return t[0]
            if np.ndim(t) == 3:
                return t[0]
            return t

        kps0 = _first(feats0['keypoints']).detach().cpu().numpy().astype(np.float32)
        kps1 = _first(feats1['keypoints']).detach().cpu().numpy().astype(np.float32)
        matches = _first(matches_res['matches']).detach().cpu().numpy()

        pts_a = np.zeros((0, 2), dtype=np.float32)
        pts_b = np.zeros((0, 2), dtype=np.float32)
        if len(matches) > 0:
            m_idx0, m_idx1 = matches[:, 0], matches[:, 1]
            # Mask valid keypoint index matches
            valid = (m_idx0 < len(kps0)) & (m_idx1 < len(kps1))
            pts_a = kps0[m_idx0[valid]]
            pts_b = kps1[m_idx1[valid]]

        # Pre-contract field names (matches_inliers_*), kept for old callers.
        out = {
            "matches_inliers_a": np.zeros((0, 2), dtype=np.float32),
            "matches_inliers_b": np.zeros((0, 2), dtype=np.float32),
            "fundamental_matrix": np.zeros((0, 0), dtype=np.float32),
        }
        if len(pts_a) >= 8:
            F_calc, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.99)
            if F_calc is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                out["matches_inliers_a"] = pts_a[inlier_mask]
                out["matches_inliers_b"] = pts_b[inlier_mask]
                out["fundamental_matrix"] = F_calc

        # Stack keypoints into a padded array (2, max_N, 2)
        max_len = max(len(kps0), len(kps1), 1)
        out["keypoints"] = pad_and_stack([kps0, kps1], pad_value=0.0)
        out["descriptors"] = np.zeros((2, max_len, 256), dtype=np.float32)
        return out

    # ------------------------------------------------------ similarity check
    def _command_similarity_check(self, img_bytes, spec, start_time):
        """One frame-change decision (see module docstring). The reference
        frame (and tracked points for the motion method) is kept until the
        next ``changed`` frame or a ``reset``."""
        # --- decode image ---
        raw_frame = image_bytes = img_bytes if isinstance(img_bytes, (bytes, bytearray)) \
            else bytes(img_bytes)

        # Try decoding using OpenCV first
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback: PIL can handle cases OpenCV rejects (e.g. some 4-ch PNGs)
            try:
                from PIL import Image
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
        if img is None:
            raise ValueError("could not decode the input frame (not a supported raster?)")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        gray = cv2.GaussianBlur(gray, (spec["blur_kernel"], spec["blur_kernel"]), 0)

        with self._sim_lock:
            self._frames_checked += 1
            frames = self._frames_checked

            # --- First frame: no reference yet — always "changed".          ---
            if self._prev_gray is None:
                self._prev_gray = gray
                self._prev_points = PipelineService._good_features(gray, spec)
                return _done(
                    {"metric": 0.0, "metric_type": spec["method"],
                     "changed": True, "first_frame": True, "num_frames": frames,
                     "runtime": time.time() - start_time},
                    data={"images": wrap_value([raw_frame])},
                    encoding={"images": "identity"})

            if spec["method"] == "ssim":
                # === SSIM SIMILARITY CHECK ===
                from skimage.metrics import structural_similarity as ssim
                metric_val = float(ssim(gray, self._prev_gray))
                changed = metric_val < spec["ssim_thresh"]
                metric_name = "ssim"
            else:
                # === LUCAS–KANADE MOTION DETECTION ===
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._prev_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                metric_val = 0.0
                if next_points is not None and status is not None:
                    next_points = np.asarray(next_points, dtype=np.float32)
                    pts0 = np.asarray(self._prev_points, dtype=np.float32)
                    # status may be (N,) or (N,1) depending on the OpenCV build
                    keep = np.asarray(status, dtype=bool).ravel()
                    if keep.any():
                        disp = np.linalg.norm(
                            next_points[keep] - pts0[keep], axis=1)
                        metric_val = float(np.mean(disp))
                changed = metric_val > spec["motion_thresh"]
                metric_name = "motion"
                # update tracked features for the next frame
                self._prev_points = PipelineService._good_features(gray, spec)

            if changed:
                # New reference frame; subsequent frames gate against it.
                self._prev_gray = gray
            else:
                # Keep comparing against the last *changed* frame.
                logging.info("No change — keeping previous reference frame.")

        extra = {"metric": metric_val, "metric_type": metric_name,
                 "changed": bool(changed), "num_frames": frames,
                 "runtime": time.time() - start_time}
        if changed:
            # Echo the (changed) frame for downstream publishing.
            return _done(extra,
                         data={"images": wrap_value([raw_frame])},
                         encoding={"images": "identity"})
        return _done(extra)

    @staticmethod
    def _good_features(gray, spec):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(spec["max_corners"]),
            qualityLevel=spec["quality_level"],
            minDistance=int(spec["min_distance"]),
            blockSize=int(spec["block_size"]),
        )

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()
        self._last_request_time = time.time()  # keep the idle watchdog honest

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get("opencv"), dict):
                return _error("config section 'opencv' missing or not an object")
            ocv = config["opencv"]
            command = ocv.get("command") or "match"
            parameters = ocv.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            # --- reset: stateless on this box except similarity_check state -
            if command == "reset":
                self._clear_similarity_state()
                return _done({"action": "reset"})

            if command not in ("match", "similarity_check"):
                return _error(
                    f"unknown command {command!r} "
                    "(expected one of: match | similarity_check | reset)")

            images = unwrap_value(request.data["images"]) if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()

            device = torch_device_default()
            req_device = str(parameters.get("device") or "").strip().lower()
            if req_device:
                device = req_device
            imgs_in = self._decode_images(images)

            if command == "similarity_check":
                spec = _parse_sim_parameters(parameters)
                # The *last* supplied frame is the one being gated.
                return self._command_similarity_check(images[-1], spec, start_time)

            # --------------------------------------------------------------- match
            spec = _parse_match_parameters(parameters)
            extractor_name, use_lightglue = _parse_extractor(spec["feature_extractor"])

            if use_lightglue:
                if len(imgs_in) != 2:
                    return _error(
                        f"feature_extractor {spec['feature_extractor']!r} (LightGlue) "
                        f"needs exactly 2 images, got {len(imgs_in)}")
                try:
                    result = self._match_lightglue(imgs_in, spec, device)
                except ImportError:
                    return _error(
                        "LightGlue is not available in this image build "
                        "(use feature_extractor SIFT or ORB instead)")
                except Exception as e:
                    logging.error(f"LightGlue execution failed: {e}")
                    raise ValueError(f"LightGlue matching failed: {e}") from None
                matcher = f"LightGlue ({extractor_name})"
            else:
                if extractor_name not in ("SIFT", "ORB"):
                    return _error(
                        f"unknown feature_extractor {spec['feature_extractor']!r} "
                        "(supported: SIFT, ORB, SUPERPOINT, DISK)")
                result = self._match_opencv(imgs_in, spec)
                matcher = "FLANN"

            data = {key: wrap_value(np_to_bytes(arr)) for key, arr in result.items()}
            encoding = {key: "numpy" for key in data}
            n_inliers = int(result["matches_inliers_a"].shape[0]) \
                if "matches_inliers_a" in result else 0
            return _done(
                {
                    "matcher": matcher,
                    "feature_extractor": str(parameters.get("feature_extractor")
                                             or spec["feature_extractor"]),
                    "device": device,
                    "num_images": len(imgs_in),
                    "num_inliers": n_inliers,
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
