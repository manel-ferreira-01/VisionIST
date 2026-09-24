# vggt — 3D reconstruction box

Meta's **VGGT** model wrapped into the shared gRPC `Process(Envelope)`
contract: feed a short sequence of same-size RGB frames, get back dense
3D reconstruction — depth maps, world-space points with confidence, camera
intrinsics/extrinsics, and a ready-to-visualize `.glb` mesh.

Stateless. GPU lifecycle: loads on CPU at startup → moves to the GPU when a
request arrives (explicitly, or whenever CUDA is visible) → watchdog returns
it to CPU after ~60 s idle.

## Layout

| Path | Description |
| --- | --- |
| `src/vggt_service.py` | gRPC server (the only contract RPC: `Process`) |
| `src/vggt/` | VGGT codebase (from `facebookresearch/vggt`) — baked in at build time; for local dev, clone it there (see below) |
| `protos/` | shared `Envelope` contract + `wrap_value`/`unwrap_value` helpers |
| `docker/Dockerfile` | multi-stage build (protos → vendor → weights → CUDA runtime) |
| `test/test_vggt.py` | standalone smoke test against a running box |

## Request contract

### `config_json`

Namespaced under the box key `vggt` (the pre-convention `aispgradio` section
and the flat top-level `parameters` form are still accepted for old callers):

```json
{
  "vggt": {
    "command": "reconstruct",
    "parameters": {
      "conf_threshold": 30,
      "device": "cuda:0"
    }
  }
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `command` | `reconstruct` | informational — the box is stateless; `reset` is accepted as a no-op, every other command runs the reconstruction |
| `parameters.conf_threshold` | `30` | GLB mesh confidence threshold (higher ⇒ sparser mesh) |
| `parameters.device` | *(auto)* | `"cpu"` / `"cuda"` / `"cuda:N"` — wins when set; otherwise GPU when visible |

### `data`

| Field | Type | Meaning |
| --- | --- | --- |
| `images` | `BytesList` (list of `b`) | RGB frames, **all must share height/width** |

### Response

`config_json`:

```json
{
  "vggt": {
    "status": "done",          // done | empty_request | error
    "runtime": 12.3,
    "num_images": 3,
    "device": "cuda:0",
    "encoding": { "world_points": "torch", "…": "torch", "glb_file": "identity" }
  }
}
```

`data` fields (declared in `encoding`):

| Field | Declared codec | Decoded value |
| --- | --- | --- |
| `world_points` | `torch` | dense points, `(N, 3)` |
| `world_points_conf` | `torch` | `(N,)` confidence per point |
| `depth` | `torch` | `(n_frames, H, W)` |
| `depth_conf` | `torch` | `(n_frames, H, W)` confidence aligned with depth |
| `extrinsic` | `torch` | `(n_frames, 4, 4)` camera-to-world |
| `intrinsic` | `torch` | `(n_frames, 3, 3)` |
| `images` | `torch` | normalized CHW frames exactly as fed to VGGT |
| `glb_file` | `identity` | raw binary `.glb` (glTF) — write to disk / open in any 3D viewer |

Status values follow the fleet contract: `done` (success), `empty_request`
(no `images`), `error` (missing/invalid config or inference failure — the
reason is in `"error"`).

## Calling

### `visionist_client` (preferred)

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")
res = b.run(
    data={"images": ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg"]},
    config={"vggt": {"command": "reconstruct",
                     "parameters": {"conf_threshold": 30}}},
)
print(res.config)                    # {"vggt": {"status": "done", ...}}
print(res.world_points.shape)        # torch tensors, decoded via the declared codec
with open("scene.glb", "wb") as f:
    f.write(res.glb_file)            # raw bytes (identity codec)
```

### Raw gRPC

```python
import grpc, json
import pipeline_pb2, pipeline_pb2_grpc, aux   # vendored from protos/

stub = pipeline_pb2_grpc.PipelineServiceStub(
    grpc.insecure_channel("localhost:8061",
                          options=[("grpc.max_send_message_length", -1),
                                   ("grpc.max_receive_message_length", -1)]))
images = [open(p, "rb").read() for p in ("frame_00.jpg", "frame_01.jpg")]
resp = stub.Process(pipeline_pb2.Envelope(
    config_json=json.dumps({"vggt": {"command": "reconstruct",
                                     "parameters": {"conf_threshold": 30}}}),
    data={"images": aux.wrap_value(images)},
))
print(json.loads(resp.config_json))
```

## Model weights & codebase

The service wants two things that are **not committed** to this repo (both
are fetched at Docker build time; the build needs network for them):

1. **VGGT codebase** (`src/vggt/` at runtime) — a shallow clone of
   `facebookresearch/vggt` (pin via `--build-arg VGGT_CODE_REV=<tag|sha>`).
2. **VGGT 1B checkpoint** — the file `model.pt` from the `facebook/VGGT-1B`
   Hugging Face hub repo, installed as `vggt-1b.pt` in the working directory.
   If it is ever absent at runtime, the service downloads it on startup
   (`HF_HOME`/`HF_HUB_CACHE` are pre-set inside the image for caching).

For *local* (non-Docker) dev runs:

```bash
git clone --depth 1 https://github.com/facebookresearch/vggt src/vggt
pip install -r requirements.txt -r src/vggt/requirements.txt
wget -O vggt-1b.pt https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
python src/vggt_service.py        # from the box root, PORT env as for any box
```

## Docker workflow

```bash
# build (from the vggt box root)
docker build --tag boxes/vggt -f docker/Dockerfile .

# run (AI4EU spec: port 8061)
docker run --rm --gpus all -p 8061:8061 boxes/vggt
```

## Testing

```bash
# standalone smoke test (server already up on :8061)
python test/test_vggt.py
BOX_HOST=10.0.0.5:8061 python test/test_vggt.py
```

Caveats: same-size frames only (the box validates and errors otherwise);
first call after an idle period includes the CPU→GPU move; the `.glb` can be
several MB — both sides run with `-1` message limits.

## Known upstream VGGT bug (and how this box avoids it)

`vggt/layers/rope.py :: PositionGetter` caches position grids keyed **only
by ``(height, width)``** but stores the tensor on the *first caller's*
device, and the ``PositionGetter`` object is a plain attribute (not an
``nn.Module``), so ``model.to(device)`` never moves the cache. Consequence:
a CUDA forward followed by a CPU forward crashes with
``Expected all tensors to be on the same device … cpu and cuda:0`` in
``aggregator.py`` (the ``torch.cat`` of ``pos_special``/``pos``). This
bit *only after a GPU inference* — a fresh CPU-only process never hits it.

This service clears the cache on **every device move**
(``_clear_position_caches()`` in ``vggt_service.py``), so device switches
work in both directions (the smoke test exercises cuda → cpu → cuda).
Any other consumer of the vanilla VGGT codebase that moves the model
between devices must do the same (or patch the cache key to include the
device). Worth an upstream issue on ``facebookresearch/vggt``.
