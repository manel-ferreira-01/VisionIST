# LightGlue Box

One job, done well and fast: **robust feature matching between image pairs**
with [LightGlue](https://github.com/cvg/LightGlue) — SuperPoint or DISK
features, matched by the LightGlue network. The box is a *thin* wrapper over
the network: it returns what the network returns. Turning matches into
epipolar geometry (F, E, camera poses, triangulation, ...) is left to the
caller, which holds the camera model. Same one-RPC contract as every
standard box:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

| Command | What it does | State |
|---|---|---|
| `match` (default) | SuperPoint / DISK features per image; with exactly **2** images also the LightGlue matcher's output: `matches` indices (+ `confidence`) | none |
| `reset` | standard no-op (this box is stateless) | — |

## Directory structure

```
lightglue_box/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── lightglue_service.py   # PipelineService.Process(Envelope)
├── test/
│   ├── test_lightglue.py      # gRPC test against a running box
│   ├── smoke_inprocess.py     # drives the service in-process (no box build)
│   ├── 00.jpg                 # test fixture
│   └── 01.jpg                 # test fixture
├── requirements.txt
└── README.md
```

## Build

```bash
cd images/lightglue_box
docker build --tag sipgisr/lightgluebox --build-arg SERVICE_NAME=lightglue -f docker/Dockerfile .
```

LightGlue is installed from source in the image (it is not on PyPI); the
SuperPoint / DISK extractors and the matcher come with it.

## Run

```bash
docker run --rm -p 8061:8061 -e PORT=8061 sipgisr/lightgluebox          # CPU
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/lightgluebox
```

## Service usage

### Request

```json
// config_json — namespaced under the box key
{
  "lightglue": {
    "command": "match",             // "match" (default) | "reset"
    "parameters": {
      "feature_extractor": "SUPERPOINT",  // SUPERPOINT | DISK
      "max_keypoints": 1024,
      "filter_threshold": 0.5,      // optional match-confidence filter (0..1,
                                    //    default: LightGlue's own, 0.1 — higher
                                    //    = fewer, stronger matches)
      "device": "cuda"              // optional: "cpu" | "cuda" | "cuda:N"
                                    //    (default: CUDA if visible, else CPU)
    }
  }
}
```

`data`:

| field    | kind | meaning |
|----------|------|---------|
| `images` | `bb` | 1 image → its features; **2** images → additionally the matcher's output. Matching is pairwise: more than 2 is an error |

### Response

```json
// two images — config_json
{
  "lightglue": {
    "status": "done",
    "matcher": "LightGlue (superpoint)",     // or "LightGlue (disk)"
    "feature_extractor": "SUPERPOINT",
    "max_keypoints": 1024,
    "device": "cuda",
    "num_images": 2,
    "num_matches": 312,
    "runtime": 0.17,
    "encoding": {
      "keypoints": "numpy", "descriptors": "numpy", "scores": "numpy",
      "matches": "numpy", "confidence": "numpy"
    }
  }
}
```

`data` (all declared `numpy`; `np.save` blobs — `boxes_client` hands them
back as `np.ndarray` with shape and dtype):

| field         | shape    | description |
|---------------|----------|-------------|
| `keypoints`   | `(I, N, 2)` | `(x, y)` per image, padded to a common `N` |
| `descriptors` | `(I, N, 256)` | per-image descriptors (SuperPoint here; same for DISK) |
| `scores`      | `(I, N)`  | per-keypoint detector scores (0.0 padding where images had fewer keypoints) |
| `matches`     | `(K, 2)`  | two-image calls: LightGlue match indices into `keypoints` — `matches[i, 0]` → image A, `matches[i, 1]` → image B |
| `confidence`  | `(K,)`    | two-image calls: per-match confidence |

From indices to point pairs: `pA = keypoints[0][matches[:, 0]]`,
`pB = keypoints[1][matches[:, 1]]`.

Status vocabulary per the shared contract: `done` (success),
`empty_request` (no `data.images`), `error` (reason in `"error"` — 3+ images,
unknown extractor, bad parameter, undecodable frame, …).

### Semantics worth knowing

- **`match`** is stateless and idempotent. One image returns its features;
  exactly two return the features plus the matcher's output — nothing more,
  nothing less.
- Model pairs are cached per `(extractor, max_keypoints, filter_threshold)`,
  so only the first call per combination pays setup cost.
- **Speed knobs**, in order of leverage: fewer `max_keypoints`, higher
  `filter_threshold`, `device: cpu` for a no-GPU box.

## Call with boxes_client

```python
from boxes_client import Box
import pathlib

b = Box("localhost:8061")

res = b.run(
    data   = {"images": [pathlib.Path("a.jpg"), pathlib.Path("b.jpg")]},
    config = {"lightglue": {"parameters": {
        "feature_extractor": "SUPERPOINT", "max_keypoints": 1024}}},
)

kp = res.keypoints            # already decoded: np.ndarray (2, N, 2)
M  = res.matches              # (K, 2) indices — the network's output
pA = kp[0][M[:, 0]]           # matched points in image A
pB = kp[1][M[:, 1]]           # matched points in image B
# F / E / poses / triangulation: your camera model, your cv2, your code.
```

## GPU behaviour

The models load **lazily on first use**, live on `parameters.device`
(default: CUDA if visible, else CPU); the fleet watchdog parks them back on
CPU after ~60 s of inactivity and releases the cache, so an idle box holds
no VRAM.

## Testing

```bash
# running box (standard smoke test, like the rest of the fleet)
python images/lightglue_box/test/test_lightglue.py
BOX_HOST=10.0.0.5:8061 python images/lightglue_box/test/test_lightglue.py

# no box build needed — drives src/lightglue_service.py in-process
# (needs numpy, opencv-python(-headless), torch, lightglue)
cd images/lightglue_box && python test/smoke_inprocess.py
```
