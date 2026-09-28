# UniMatch Box

One job, done well: **dense geometry between images** — optical flow, stereo
disparity, and metric depth — with the [UniMatch](https://github.com/autonomousvision/unimatch)
model family (TPAMI'23): one CNN-Transformer network used for three dense
estimation tasks. The box is a *thin* wrapper over the network: it returns
what the network returns, restored to the input images' resolution. Same
one-RPC contract as every standard box:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

| Command | What it does | State |
|---|---|---|
| `flow` (default) | forward optical flow **image 1 → image 2** (`exactly 2 images`): `flow (H, W, 2)`. Optional: `bwd_flow`, forward-backward occlusion masks `occ_fwd`/`occ_bwd` | none |
| `stereo` | disparity of a **rectified left/right pair** (`exactly 2 images`): `disparity (H, W)` in pixels of the left image | none |
| `depth` | **metric depth** from 1 image (monocular) or a reference/target pair, given camera `intrinsics` and (optional) relative pose: `depth (H, W)` in metres | none |
| `reset` | stateless box: standard no-op, acknowledged for the shared contract | — |

## Directory structure

```
unimatch/
├── docker/
│   └── Dockerfile             # multi-stage; vendors UniMatch (pinned commit);
                               #    bakes in the flow checkpoint (~20 MB)
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── unimatch_service.py    # PipelineService.Process(Envelope)
├── test/
│   ├── test_unimatch.py       # gRPC test against a running box
│   ├── flow_0.jpg flow_1.jpg  # test fixtures
│   └── stereo_0.png stereo_1.png
├── pretrained/                # checkpoints (see "Weights" below)
├── requirements.txt
└── README.md
```

## Build

```bash
cd images/unimatch
docker build --tag visionist-local/unimatch -f docker/Dockerfile .
```

The UniMatch package itself is **vendored into the image** (git clone of
`autonomousvision/unimatch` at a pinned commit — the build is self-contained).
PyTorch comes from the default PyPI wheels, which ship the CUDA runtime
libraries, so the box is GPU-capable without a CUDA base image (same
convention as the lightglue box).

## Run

```bash
docker run --rm -p 8061:8061 -e PORT=8061 visionist-local/unimatch
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 visionist-local/unimatch
```

In the fleet it runs as service `unimatch` on host port **9070**
(`fleet/docker-compose.yml`).

## Weights

One checkpoint per task, expected under `PRETRAINED_DIR` (env, default
`./pretrained`). The image **bakes in the flow model**; the other two are
fetched on first use when addressed, or you pre-stage them
([model zoo](https://github.com/autonomousvision/unimatch/blob/main/MODEL_ZOO.md)):

| task     | default checkpoint                          | in the image? |
|----------|---------------------------------------------|---------------|
| `flow`   | `gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth` | ✅ baked in |
| `stereo` | `gmstereo-scale1-sceneflow-124a438f.pth`     | ✗ on demand |
| `depth`  | `gmdepth-scale1-scannet-d3d1efb5.pth`        | ✗ on demand |

Which checkpoint a request uses is set with `parameters.model` — a file
**name** in the pretrained dir, an existing **local path**, or a
**checkpoint URL** (downloaded once into the dir). Without it, the default
per-task file applies — and if that is missing, the box answers a clean
`error` naming the expected file (no crash, no traceback):

```json
{"unimatch": {"status": "error",
  "error": "no checkpoint for task 'stereo': expected 'gmstereo-scale1-sceneflow-124a438f.pth' …"}}
```

To run all three tasks out of the box, pass `--build-arg` to bake them in
or mount the dir:

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 \
  -v /path/to/pretrained:/workspace/pretrained visionist-local/unimatch
```

## Service usage

### Request

```json
// config_json — namespaced under the box key
{
  "unimatch": {
    "command": "flow",            // "flow" (default) | "stereo" | "depth" | "reset"
    "parameters": {
      "device": "cuda",           // optional: "cpu" | "cuda" | "cuda:N"
                                  //    (default: CUDA if visible, else CPU)
      "model": "gmflow-…4c3a6e9a.pth",   // optional: name / path / URL
      "pred_bidir_flow": true,     // flow: also return bwd_flow
      "fwd_bwd_check": true,       // flow: forward-backward occlusion masks
      "occ_alpha": 0.01,           // fwd_bwd_check thresholds (defaults 0.01 / 0.5)
      "occ_beta": 0.5,
      "intrinsics": [fx, fy, cx, cy],  // depth: REQUIRED (3x3, 4x4, or 4 values)
      "pose": [[…4x4…]],          // depth: relative pose ref -> target
      "pose_ref": [[…]], "pose_tgt": [[…]],   // …or both world poses (4x4);
      "min_depth": 0.5,           // depth: candidate range in metres (0.5–10 default)
      "max_depth": 10.0,
      "num_depth_candidates": 64, // depth: default 64
      "inference_size": [320, 576],   // optional [h, w]; always wins over auto
      "padding_factor": 16,       // default 16 (model input is a multiple of it)
      "num_scales": 1             // checkpoint family (with per-scale lists below)
    }
  }
}
```

`data`:

| field    | kind | meaning / count by command |
|----------|------|----------------------------|
| `images` | `bb` | `flow`: **exactly 2** (img 1 → img 2); `stereo`: **exactly 2** (left, right, rectified); `depth`: **1** (monocular) or **2** (reference, target) |

### Response

```json
// config_json — success
{
  "unimatch": {
    "status": "done",
    "command": "flow",
    "model": "gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth",
    "num_images": 2,
    "device": "cuda",
    "runtime": 0.42,
    "encoding": { "flow": "numpy", "bwd_flow": "numpy" }
  }
}
```

Extras in the config: `"monocular": true` for single-image `depth` calls;
`"auto_resized": true` when the input was downscaled to the area cap (see
[semantics](#semantics-worth-knowing)). Status vocabulary per the shared
contract: `done` | `empty_request` (no `data.images`) | `error`
(reason in `"error"` — wrong image count, unknown command, malformed
`intrinsics`/pose, missing checkpoint, …).

`data` — all declared `numpy` (`np.save` blobs; `visionist_client` hands
them back as `np.ndarray` with shape and dtype, float32):

| field      | shape   | command | description |
|------------|---------|---------|-------------|
| `flow`     | `(H, W, 2)` | flow | forward flow image 1 → 2, in pixel displacement (`x, y`), at the original image resolution |
| `bwd_flow` | `(H, W, 2)` | flow | optional (`pred_bidir_flow`): backward flow |
| `occ_fwd`  | `(H, W)`  | flow | optional (`fwd_bwd_check`): forward occlusion mask, 0/1 |
| `occ_bwd`  | `(H, W)`  | flow | optional (`fwd_bwd_check`): backward occlusion mask, 0/1 |
| `disparity`| `(H, W)`  | stereo | disparity in pixels of the left image |
| `depth`    | `(H, W)`  | depth  | metric depth in metres |

### `depth` inputs, precisely

- **`intrinsics` (required)** — `3x3` matrix, `4x4` matrix, or
  `[fx, fy, cx, cy]`, in the resolution of the image **as sent**. The box
  scales them automatically to its padded model input.
- **Pose (optional — identity when absent), either form:**
  - `pose` — the 4x4 **relative** pose, world-referenced, computed ref →
    target the way the model-zoo demos do it: `inv(pose_tgt) @ pose_ref`; or
  - `pose_ref` + `pose_tgt` — both 4x4 in world coordinates (reference and
    target views); the box computes the relative pose.
- **Range** — `min_depth`/`max_depth` (metres, defaults `0.5`/`10`) and
  `num_depth_candidates` (default `64`) bound the inverse-depth candidates.

### Semantics worth knowing

- **All outputs are at the input resolution.** The model runs at a padded
  multiple of `padding_factor` (16) — or `parameters.inference_size`, which
  always wins — and predictions are restored (bilinear + scale, occlusion
  masks nearest) back to the sent size. Flow handles **portrait input** by
  transposing to landscape on the model side and un-transposing the result,
  following the model repo.
- **Automatic input cap.** The global-attention stages are O((HW)²); requests
  larger than `UNIMATCH_MAX_INPUT_AREA` (env, default **~1.6 MP**; overridable
  per request with `parameters.max_input_area`) are downscaled automatically
  **and reported** with `"auto_resized": true` in the response config, so a
  shared GPU won't be felled by a 29 GB matmul. An explicit
  `inference_size` disables the cap for that request.
- **`flow` is stateless and one-directional by default.** `fwd_bwd_check`
  needs `pred_bidir_flow: true` (the backward flow is what it is checked
  against); the occlusion masks are binary 0/1 float32.
- **`stereo` expects a rectified pair** (standard stereo assumption:
  horizontal epipolar lines); inputs are ImageNet-normalized by the box —
  send raw JPEG/PNG bytes, do not pre-normalize.
- **`depth` is monocular or relative-stereoscopic** depending on the pose
  you give it; a single image with identity pose is the "single-view" case.
  It needs real intrinsics — there is no guess mode.
- **Models are cached per `(checkpoint, num_scales, reg_refine, task)`**, so
  the first call per combination pays load cost; switching tasks then is a
  second load, not a re-download.
- **Architecture knobs** (defaults match the scale1 zoo models):
  `num_scales`, `attn_type` (default `swin`), `attn_splits_list`,
  `corr_radius_list` (flow/stereo), `prop_radius_list`, `reg_refine`,
  `num_reg_refine`. For `scale2` checkpoints the per-scale lists must all
  have length `num_scales` — the box validates this and answers a clear
  `error` otherwise.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib, numpy as np

b = Visionist("localhost:8061")                 # or 10.0.0.5:9070 in the fleet

# --- optical flow ----------------------------------------------------------
res = b.run(
    data   = {"images": [pathlib.Path("img1.jpg"), pathlib.Path("img2.jpg")]},
    config = {"unimatch": {"command": "flow",
                           "parameters": {"pred_bidir_flow": True,
                                          "fwd_bwd_check": True}}},
)
sec  = res.config["unimatch"]
flow = res.flow              # decoded: np.ndarray (H, W, 2) — px displacement
bwd  = res.bwd_flow          # (H, W, 2)
occ  = res.occ_fwd           # (H, W) binary 0/1
print(sec["status"], flow.shape, occ.mean())

# --- stereo disparity (2-rectified pair; needs the stereo checkpoint) ------
res = b.run(
    data   = {"images": [pathlib.Path("left.png"), pathlib.Path("right.png")]},
    config = {"unimatch": {"command": "stereo"}},
)
disp = res.disparity         # (H, W) — pixels of the left image

# --- metric depth (needs intrinsics; optional poses) ------------------------
res = b.run(
    data   = {"images": [pathlib.Path("ref.jpg"), pathlib.Path("tgt.jpg")]},
    config = {"unimatch": {"command": "depth", "parameters": {
        "intrinsics": [525.0, 525.0, 321.0, 243.0],   # at the sent resolution
        "pose_ref": pose_ref_4x4, "pose_tgt": pose_tgt_4x4,
        "min_depth": 0.5, "max_depth": 20.0}}},
)
print(res.depth.min(), res.depth.max(), res.depth.mean())
```

The `numpy` codec decodes every array field automatically — no manual
`np.load`. (Raw path: `np.load(io.BytesIO(bytes(aux.unwrap_value(res.raw.data[f]))))`.)

## GPU behaviour

Models load **lazily on first use** (per checkpoint + task), live on
`parameters.device` (default: CUDA if visible, else CPU); the fleet watchdog
parks all loaded models back on CPU after ~60 s of inactivity and calls
`empty_cache()`, so an idle box holds no VRAM. Checkpoints are kept in the
`pretrained` dir and reloaded from CPU on the next request — only the model
weights' first load is the slow path.

## Testing

```bash
# running box (standard smoke test, like the rest of the fleet)
python images/unimatch/test/test_unimatch.py
BOX_HOST=10.0.0.5:9070 python images/unimatch/test/test_unimatch.py
```

With the standard image (flow checkpoint only) the test asserts: `flow`
shape/finiteness, `bwd_flow` + occlusion masks, `stereo`/`depth` answering
**clean errors naming the expected checkpoint file**, the
`empty_request`/`error` contract, and the `reset` acknowledgement. With all
three checkpoints staged, run stereo/depth calls directly as above.
