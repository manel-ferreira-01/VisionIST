# OpenCV Box

Classic computer-vision utilities behind the **shared envelope** interface:
feature extraction & matching, and a stateful frame-change gate for video
streams. Same one-RPC contract as every standard box:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

| Command | What it does | State |
|---|---|---|
| `match` (default) | keypoints + descriptors per image; with exactly **2** images also matches (`FLANN` or `LightGlue`) + RANSAC fundamental matrix. Stateless | none |
| `similarity_check` | gates an incoming frame against the last *changed* frame — Lucas–Kanade mean displacement (`method: "motion"`, default) or SSIM (`method: "ssim"`); `changed` frames are echoed back in `data.images` | reference frame + tracked points (cleared by `reset`) |
| `reset` | clear the `similarity_check` state (harmless no-op for `match`) | — |

> **Migration note.** This box predates the shared contract: it used to
> serve `similarity_check` as a *second RPC* with flat `"status":
> "success"` responses. That is gone. The RPC surface is now the standard
> one (`Process`), `similarity_check` is a **command** of `Process`, and
> responses are namespaced under `"opencv"` with the
> `done | empty_request | error` status vocabulary. Old callers: send the
> same envelope through `Process` with `"command": "similarity_check"`.

## Directory structure

```
opencv_box/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── opencv_service.py      # PipelineService.Process(Envelope)
├── test/
│   ├── test_opencv.py         # smoke test against a running box
│   ├── smoke_inprocess.py     # drives the service in-process (no box build)
│   ├── test.ipynb             # manual playground
│   ├── 00.jpg                 # test fixture
│   └── 01.jpg                 # test fixture
├── requirements.txt
└── README.md
```

## Build

```bash
cd images/opencv_box
docker build --tag sipgisr/opencvbox --build-arg SERVICE_NAME=opencv -f docker/Dockerfile .
```

LightGlue (SuperPoint/DISK extractors) is installed from source in the
image — `match` with `feature_extractor: SUPERPOINT`/`DISK` needs GPU or
CPU with it present; builds without it answer a clean `error` and the
`SIFT`/`ORB` path still works.

## Run

```bash
docker run --rm -p 8061:8061 -e PORT=8061 sipgisr/opencvbox          # CPU
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/opencvbox
```

## Service usage

### Request

```json
// config_json — namespaced under the box key
{
  "opencv": {
    "command": "match",            // "match" (default) | "similarity_check" | "reset"
    "parameters": {
      "feature_extractor": "SIFT", // match: SIFT | ORB | SUPERPOINT | DISK
      "ratio_thresh": 0.75,        // match (FLANN): Lowe's ratio
      "max_keypoints": 500,        // match (LightGlue: 2048 default)
      "method": "motion",          // similarity_check: "motion" | "ssim"
      "motion_thresh": 1.5,        // similarity_check (motion): px displacement
      "ssim_thresh": 0.90,         // similarity_check (ssim)
      "blur_kernel": 5,            // similarity_check
      "max_corners": 200,          // similarity_check (tracked features)
      "quality_level": 0.01,
      "min_distance": 5,
      "block_size": 7,
      "device": "cuda"             // optional: "cpu" | "cuda" | "cuda:N"
                                   //    (wins over the auto lifecycle; only
                                   //    matters when LightGlue is in use)
    }
  }
}
```

`data`:

| field    | kind | meaning |
|----------|------|---------|
| `images` | `bb` | `match`: 1 image → extraction only, 2 images → + matching & fundamental matrix. `similarity_check`: the latest frame (the last entry if several are sent) |

### Response

`config_json` carries a **namespaced status** (never flat):

```json
// match, two images
{
  "opencv": {
    "status": "done",
    "matcher": "FLANN",              // or "LightGlue (superpoint)"
    "feature_extractor": "SIFT",
    "device": "cuda",
    "num_images": 2,
    "num_inliers": 517,
    "runtime": 0.31,
    "encoding": {
      "keypoints": "numpy", "descriptors": "numpy",
      "matches_inliers_a": "numpy", "matches_inliers_b": "numpy",
      "fundamental_matrix": "numpy"
    }
  }
}

// similarity_check, a frame changed
{
  "opencv": {
    "status": "done",
    "metric": 4.13,                  // px displacement (motion) or SSIM (ssim)
    "metric_type": "motion",
    "changed": true,
    "first_frame": true,             // true only right after startup/reset
    "num_frames": 1,                 // frames gated since startup/reset
    "runtime": 0.01,
    "encoding": { "images": "identity" }
  }
}

// reset
{ "opencv": { "status": "done", "action": "reset" } }
```

Status vocabulary per the shared contract: `done` (success),
`empty_request` (no `data.images`), `error` (reason in `"error"` — bad
command, bad parameter, undecodable frame, unknown extractor, …).

`data`:

| field | kind | description |
|---|---|---|
| `keypoints` | `b` (`numpy`) | `np.save` blob: per-image `(N, 2)` `(x, y)`, padded to a common `N` and stacked → `(num_images, N, 2)`. `np.load(io.BytesIO(blob))` restores it with shape & dtype |
| `descriptors` | `b` (`numpy`) | per-image descriptor matrix, same padding: `(num_images, N, 128)` (SIFT) / `(num_images, N, 32)` (ORB) / `(2, N, 256)` (LightGlue — placeholder zeros) |
| `matches_inliers_a` / `matches_inliers_b` | `b` (`numpy`) | two-image calls only: RANSAC inlier points of image A / image B, `(K, 2)` |
| `fundamental_matrix` | `b` (`numpy`) | two-image calls only: the estimated 3×3 F (empty `(0, 0)` when matches were insufficient) |
| `images` | `bb` (`identity`) | `similarity_check` only, and only when `changed`: the incoming frame bytes, for downstream publishing |

The declared `numpy` fields decode to `np.ndarray` in `boxes_client` (the
codec restores the `np.save` array — see `docs/CODECS.md`).

### Semantics worth knowing

- **`match`** is stateless and idempotent. One image returns extraction
  only; exactly two return the matching section. LightGlue extractors
  accept **exactly two** images (an error otherwise) — they replace the
  old SIFT-only path when present in the image.
- **`similarity_check`** compares the incoming frame against the last
  *changed* frame (unchanged frames keep the reference, so small drifting
  changes still trip the gate). The **first frame after startup or after
  `reset` is always reported `changed: true` with `first_frame: true`**.
  Frames are downscaled to 320×240 and blurred before comparing — cheap by
  design.
- **`reset`** clears the reference frame and tracked points
  (`num_frames` restarts at 1). It is the standard box reset and is a
  harmless no-op for the stateless `match` command.

## Call with boxes_client

```python
from boxes_client import Box
import pathlib

b = Box("localhost:8061")

# --- matching ---------------------------------------------------------
res = b.run(
    data   = {"images": [pathlib.Path("images/opencv_box/test/00.jpg"),
                         pathlib.Path("images/opencv_box/test/01.jpg")]},
    config = {"opencv": {"command": "match",
                         "parameters": {"feature_extractor": "SIFT",
                                        "max_keypoints": 1000}}},
)
print(res.encoding)          # {'keypoints': 'numpy', …}
print(res.config["opencv"])  # status / matcher / num_inliers / runtime
kp = res.keypoints           # already decoded: np.ndarray (2, N, 2)
F  = res.fundamental_matrix  # (3, 3) — F @ pts_a  ~  pts_b (up to scale)

# --- frame gating (video stream) ---------------------------------------
for frame in frames:                          # e.g. decoded from a video
    r = b.run(
        data   = {"images": [frame]},
        config = {"opencv": {"command": "similarity_check",
                             "parameters": {"motion_thresh": 3.0}}},
    )
    sec = r.config["opencv"]
    if sec["changed"]:
        publish(sec["metric"], r.images[0])   # r.images = the changed frame

# --- clear the gate -----------------------------------------------------
b.run(config={"opencv": {"command": "reset"}})
```

## GPU behaviour

The OpenCV SIFT/ORB path is pure CPU. LightGlue (SuperPoint/DISK) is loaded
**lazily on first use** and lives on `parameters.device` (default: CUDA if
visible, else CPU); the fleet watchdog parks it back on CPU after ~60 s of
inactivity and releases the cache, so an idle box holds no VRAM.

## Testing

```bash
# running box (standard smoke test, like the rest of the fleet)
python images/opencv_box/test/test_opencv.py
BOX_HOST=10.0.0.5:8061 python images/opencv_box/test/test_opencv.py

# no box build needed — drives src/opencv_service.py in-process
# (needs numpy, opencv-python(-headless), scikit-image)
cd images/opencv_box && python test/smoke_inprocess.py
```

`test/test.ipynb` is the manual playground (match + similarity gating).
