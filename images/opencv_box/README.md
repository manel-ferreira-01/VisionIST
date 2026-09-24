# OpenCV Box

Classic computer-vision utilities behind the **shared envelope** interface:
feature extraction & matching. Same one-RPC contract as every standard box:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

| Command | What it does | State |
|---|---|---|
| `match` (default) | keypoints + descriptors per image; with exactly **2** images also matches (`FLANN` or `LightGlue`) + RANSAC fundamental matrix. Stateless | none |
| `reset` | standard no-op (this box is stateless) | — |

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
    "command": "match",            // "match" (default) | "reset"
    "parameters": {
      "feature_extractor": "SIFT", // SIFT | ORB | SUPERPOINT | DISK
      "ratio_thresh": 0.75,        // FLANN: Lowe's ratio
      "max_keypoints": 500,        // (LightGlue: 2048 default)
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
| `images` | `bb` | 1 image → extraction only, 2 images → + matching & fundamental matrix |

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

The declared `numpy` fields decode to `np.ndarray` in `visionist_client` (the
codec restores the `np.save` array — see `docs/CODECS.md`).

### Semantics worth knowing

- **`match`** is stateless and idempotent. One image returns extraction
  only; exactly two return the matching section. LightGlue extractors
  accept **exactly two** images (an error otherwise) — they replace the
  old SIFT-only path when present in the image.
- **`reset`** is the standard box reset; on this box it is a plain no-op,
  since `match` is the only command and it is stateless.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")

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
# (needs numpy, opencv-python(-headless) locally)
cd images/opencv_box && python test/smoke_inprocess.py
```
