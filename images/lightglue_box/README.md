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
| `match` (default) | SuperPoint / DISK features per image; with exactly **2** images also the LightGlue matcher's output: `matches` indices (+ `confidence`). Stateless | none |
| `stream` | **sliding-window** stream matching: one new frame per call (`session_id`), matched against the last `window` stored frames → per-reference `matches_j`/`confidence_j`. Window = a ring of cached features per session | features ring per `session_id` (reset / TTL) |
| `reset` | without `session_id`: clear **all** stream sessions; with one: clear just that session | — |
| `list` | active stream sessions with frame counts (operator helper, like the other multi-session boxes) | — |

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
    "command": "match",             // "match" (default) | "stream" | "reset" | "list"
    "parameters": {
      "feature_extractor": "SUPERPOINT",  // SUPERPOINT | DISK
      "max_keypoints": 1024,
      "filter_threshold": 0.5,      // optional match-confidence filter (0..1,
                                    //    default: LightGlue's own, 0.1 — higher
                                    //    = fewer, stronger matches)
      "window": 3,                  // stream only: references the new frame is
                                    //    matched against (1..16, default 3)
      "session_id": "cam1",         // stream / reset only (default "default")
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

### `stream` response (sliding window per `session_id`)

One new frame per call. The box keeps the *extracted features* of the last
frames per session (on CPU) and matches the new frame against the last
`window` of them (default 3, max 16). Config carries `session`,
`requested_window`, the actual `window` filled (`J`), `num_frames`,
`num_matches_j` per reference — and `first_frame: true` for the session's
first step (features stored, no matches yet).

`data` (all declared `numpy`):

| field         | shape              | description |
|---------------|--------------------|-------------|
| `keypoints`   | `(J+1, N, 2)`      | references **most-recent → oldest**, then the new frame (row `J`) |
| `kp_counts`   | `(J+1,)`           | valid keypoints per row (rows are padded to a common `N`) |
| `matches_j`   | `(K_j, 2)`         | reference `j` (1 = previous frame) ↔ new frame indices: col 0 → `keypoints[j-1]`, col 1 → `keypoints[J]` |
| `confidence_j`| `(K_j,)`           | per-match confidence for reference `j` |

From indices to point pairs: `pA = keypoints[0][matches[:, 0]]`,
`pB = keypoints[1][matches[:, 1]]`.

Status vocabulary per the shared contract: `done` (success),
`empty_request` (no `data.images`), `error` (reason in `"error"` — 3+ images,
unknown extractor, bad parameter, undecodable frame, …).

### Semantics worth knowing

- **`match`** is stateless and idempotent. One image returns its features;
  exactly two return the features plus the matcher's output — nothing more,
  nothing less.
- **`stream`** is a sliding window over *features*, not pixels: each call,
  the new frame's features are stored and it is matched against the last
  `window` stored frames — so successive calls overlap by `window − 1`
  frames' worth of context. One extractor pass + `window` matcher passes
  per call. The cached features are the box's only per-session state
  (≤ 16 × ~1 MB of CPU RAM), reaped after `LIGHTGLUE_SESSION_TTL` idle
  seconds (default 1800; set 0 to keep sessions forever).
- **`reset`** clears stream sessions (one, or all without `session_id`);
  **`list`** shows the active ones. Both follow the other multi-session
  boxes (tapnext / yolo).
- Model pairs are cached per `(extractor, max_keypoints, filter_threshold)`,
  so only the first call per combination pays setup cost.
- **Speed knobs**, in order of leverage: fewer `max_keypoints`, higher
  `filter_threshold`, smaller `window`, `device: cpu` for a no-GPU box.

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

# --- sliding-window stream (per session_id; window = refs back) ----------
sid = {"session_id": "cam1", "window": 3}
for frame in frames:                            # e.g. a webcam / decoder loop
    r = b.run(
        data   = {"images": [frame]},
        config = {"lightglue": {"command": "stream", "parameters": sid}},
    )
    sec = r.config["lightglue"]
    if sec.get("first_frame"):
        continue                                # first step only stores features
    J, kp = sec["window"], r.keypoints          # rows: refs (newest first), then new
    for j in range(1, J + 1):                   # j = 1 is the previous frame
        m     = getattr(r, f"matches_{j}")
        p_ref = kp[j - 1][m[:, 0]]
        p_new = kp[J][m[:, 1]]
# stop a stream:  b.run(config={"lightglue": {"command": "reset",
#                                              "parameters": {"session_id": "cam1"}}})
# or all of them: b.run(config={"lightglue": {"command": "reset"}})
```

## GPU behaviour

The models load **lazily on first use**, live on `parameters.device`
(default: CUDA if visible, else CPU); the fleet watchdog parks them back on
CPU after ~60 s of inactivity and releases the cache, so an idle box holds
no VRAM. Stream sessions cache their feature rings on CPU (megabytes), so
they neither pin VRAM nor survive a model move — they are reaped by their
own TTL (`LIGHTGLUE_SESSION_TTL`, default 1800 s).

## Testing

```bash
# running box (standard smoke test, like the rest of the fleet)
python images/lightglue_box/test/test_lightglue.py
BOX_HOST=10.0.0.5:8061 python images/lightglue_box/test/test_lightglue.py

# no box build needed — drives src/lightglue_service.py in-process
# (needs numpy, opencv-python(-headless), torch, lightglue)
cd images/lightglue_box && python test/smoke_inprocess.py
```
