# TAPNext Tracker gRPC Service

A gRPC service for TAPNext point tracking with streaming support, following the vggt image pattern.

## Features

- Frame-by-frame tracking with state preservation
- Grid-based query point detection on first frame
- Server-side track accumulation until reset
- **Video input**: `data.video` — the box decodes the video into ordered frames
  on the server side and tracks them (same shape as yolo's `data.video`);
  `frame_step` / `max_frames` control the sampling
- **Growing point set** (optional `new_tracks: true`): as tracked points get
  lost, empty grid cells are re-seeded with fresh points — each seed is its
  own TAPNext *generation* (surviving points keep their long-range memory),
  with fresh track ids, optional retirement of dead generations, budgets, and
  optional trajectory backfill (see [Growing point set](#growing-point-set-new_tracks-true))
- **Multi-session (multi-tenant)**: one box serves many independent users at
  once — each `session_id` holds its own state, its own reset, and its own
  accumulation (see [Sessions](#sessions-sharing-one-box-with-many-users))
- CUDA-accelerated inference using JAX/PyTorch backend
- Configurable grid size for query point density

## Directory Structure

```
tapnext_tracker/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto     # Shared proto definition
│   └── aux.py             # Helper functions for Value wrapping
├── src/
│   └── tapnext_service.py # Main service implementation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Docker with CUDA support (nvidia-docker2 or Docker with GPU support)
- TAPNext checkpoint file: `bootstapnext_ckpt.npz` (740MB)

Checkpoint download:
```bash
wget https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
```

## Building the Docker Image

Build without checkpoint (service checks for checkpoint at runtime):

```bash
docker build --tag sipgisr/tapnexttracker \
    --build-arg SERVICE_NAME=tapnext \
    -f docker/Dockerfile .
```

Or mount the checkpoint at runtime:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz \
    sipgisr/tapnexttracker
```

## Service Usage

### gRPC Protocol

The service uses the shared `PipelineService` interface with `Envelope` messages.

#### Reset Tracking State (one-time initialization)

Reset clears previous tracking state before starting a new sequence. Reset is
**scoped to a session**: without a `session_id` it clears the shared `default`
session; pass one and only that user's state is touched (others keep running):

```python
import grpc
import pipeline_pb2 as proto
from aux import wrap_value, unwrap_value

# Initialize gRPC client
channel = grpc.insecure_channel('localhost:8061')
stub = proto.PipelineServiceStub(channel)

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {"command": "reset"}          # + "session_id": "alice-2025" to scope
    })
)
response = stub.Process(request)
```

#### Tracking Request

Track points in a single frame. The service automatically tracks across sequential frames:

```python
# Prepare frame data
with open('frame.jpg', 'rb') as f:
    frame_bytes = f.read()

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32}
        }
    }),
    data={"images": [wrap_value(frame_bytes)]}
)

response = stub.Process(request)
tracks = unwrap_value(response.data["tracks"])
visibles = unwrap_value(response.data["visibles"])
```

Each frame you send continues from the previous tracking state. No reset flag needed between frames.

#### Video Tracking Request

Instead of an explicit frame list you may send a **single video file**. The box
decodes it into ordered frames on the server side (mirroring the yolo box) and
feeds them through the exact same per-frame tracker:

```python
video_bytes = open("cozinha.mp4", "rb").read()   # or pathlib.Path(...) via visionist_client

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32, "frame_step": 4, "max_frames": 24}
        }
    }),
    data={"video": wrap_value(video_bytes)}
)
```

`data.images` and `data.video` are **mutually exclusive** — sending both returns a
clear error. The response is identical to the image case: `tracks`, `visibles`,
`observation_matrix` (plus the `frames_processed` count in the config).

### Sessions (sharing one box with many users)

The box is **multi-tenancy-ready**: every request can carry a `session_id`
(inside the `tapnext` config section). All requests with the same `session_id`
see the same tracking state; different ids see only their own — state,
accumulation, and reset are all scoped per session. No login or registry is
involved: the id is an opaque **capability string** — knowing it is enough to
run against that session, and it is the only way to name one.

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:9063")

# Student "alice" tracks — her session is created on first use
res = b.run(
    data   = {"images": [pathlib.Path("frame1.jpg")]},
    config = {"tapnext": {"command": "track",
                          "parameters": {"grid_size": 32},
                          "session_id": "alice-2025"}},
)

# Student "bob" on the SAME box, same GPU: completely independent
b.run(data={"images": [pathlib.Path("frame1.jpg")]},
      config={"tapnext": {"command": "track",
                          "parameters": {"grid_size": 32},
                          "session_id": "bob-2025"}})

# Reset is scoped: only alice's session restarts; bob's keeps going
b.run(config={"tapnext": {"command": "reset", "session_id": "alice-2025"}})

# Operator view: which sessions are alive (sid + frames + idle time, no state)
b.run(config={"tapnext": {"command": "list"}})
```

| Behaviour | Single session (no `session_id`) | Named sessions |
|---|---|---|
| State/accumulation | one shared `default` session | one per `session_id` |
| `reset` | clears the `default` session | clears only that session |
| Back-compat | old clients keep working unchanged | new key, ignored by old boxes |

Notes:
- Omitting `session_id` (or sending `null`) runs in the shared **`default`**
  session — the pre-multi-session behaviour, so existing callers are untouched.
- A session stays alive until it is reset, reaped after `TAPNEXT_SESSION_TTL`
  idle time (default 1800 s — see env vars), or the box is restarted.
  Set it to 0 to keep sessions forever (e.g. overnight work).
- The `session_id` is a capability: anyone who can reach the box port and knows
  an id can continue or reset that session (fine on a trusted LAN — do not
  publish unknown ids publicly if you don't trust the audience). `list` lets
  someone who reaches the box enumerate the active ids.
- The response config echoes which session answered (`"session": "<sid>"`).

#### Sessions in raw gRPC (without the client)

```python
request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32},
            "session_id": "alice-2025"
        }
    }),
    data={"images": [wrap_value(frame_bytes)]}
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | string | "track" | "track" for inference, "reset" to clear *this session's* tracking state, "list" to see active sessions (operator) |
| `session_id` | string | `"default"` | Opaque session tag; gives the user a private state. Share an id = share a session |
| `grid_size` | int | 32 | Number of grid points per dimension (grid_size × grid_size total) |
| `frame_step` | int | 1 | Video input: sample every Nth frame (1 = every frame) |
| `max_frames` | int | 0 | Cap on the number of frames to track (video input; 0 = no cap) |

| Parameter | Type | Default | Description (growing point set — only read when `new_tracks` is set) |
|-----------|------|---------|-------------|
| `new_tracks` | bool | `false` | Enable the growing path (legacy single-grid behaviour otherwise) |
| `cell_size` | int | 20 | Grid cell size in 256×256 model space; **empty** cells get re-seeded |
| `min_dist` | float | 4.0 | A cell is "occupied" if a *visible* tracked point is closer than this |
| `add_interval` | int | 5 | Re-seed check is made every N frames (0 never) |
| `max_new_per_seed` | int | 256 | Cap on new points per seed |
| `max_total_points` | int | 4096 | Hard cap on total live track ids (growth stops when hit) |
| `retire_after_invisible` | int | 0 (off) | A generation that is fully invisible for N straight frames is dropped — it stops costing model calls and VRAM from the next frame. 0 = generations live until reset (costs scale with the video) |
| `backfill` | bool | `false` | A new generation is initialized over the session's stored prefix, so its columns are written into earlier frames (trajectory from frame 0) — costs O(length of prefix) per seed, and the box must then keep decoded frames per session (≈200 KB/frame). Default: a new point simply does not exist before its birth frame |

### Response Format

Response includes all tracked points and the Tomasi-Kanade observation matrix:

```json
{
  "tapnext": {
    "status": "done",
    "session": "alice-2025",
    "runtime": 0.45,
    "frames_processed": 1,
    "num_points": 1024
  }
}
```

`"session"` echoes the id that answered (or `"default"`), so a client can
confirm which session it just talked to.

A `list` request returns a config-only envelope (no `data`):

```json
{
  "tapnext": {
    "status": "done",
    "action": "list",
    "sessions": [
      {"session": "alice-2025", "frames_processed": 42, "num_tracks": 1024, "idle_seconds": 3.2},
      {"session": "bob-2025",   "frames_processed": 7,  "num_tracks": 1024, "idle_seconds": 190.5}
    ]
  }
}
```

Response data contains:
- `tracks`: Float32 array of shape (frame_count, num_points, 2)
- `visibles`: Float32 array of track visibility logits  
- `observation_matrix`: Tomasi-Kanade P matrix (2×frames × num_points) for factorization

### Growing point set (`new_tracks: true`)

By default the box seeds `grid_size²` points on the session's first frame and
never re-adds points — `tracks` is a uniform `(F, N, 2)` and `num_points` is
constant. With `"new_tracks": true` the point set **grows**:

- every `add_interval` frames the box looks at the 256×256 grid and re-seeds
  **empty cells** (no *visible* tracked point within `min_dist`) with up to
  `max_new_per_seed` new points, capped at `max_total_points` total ids.
  If all points die, the next seed horizon re-samples the (empty) grid —
  the tracker self-heals instead of idling.
- each seed is a new **generation** with fresh track ids assigned after the
  existing ones; column `i` of `tracks` **is** track id `i` in *every* frame
  (stable for the whole session — survivors are never re-indexed).
- a point seeded at frame `k` does not exist before `k`: frames `0…k-1`
  carry `0.0` in `tracks`, `0` in `visibles`, `NaN` in `observation_matrix`
  for that column. `visibles` is the validity mask; **`data.birth_frames`**
  (a typed list, one first-frame per track id) and the config's compact
  **`"birth_hist"`** (frame → how many points were seeded then) separate
  "not yet born" from "born but lost".
- optional `retire_after_invisible` drops a generation after N fully
  invisible consecutive frames (its last recorded positions stay; it costs
  nothing after — this is what keeps long videos affordable).
- optional `backfill` writes a new generation's recorded trajectory into the
  earlier frames instead (its columns are finite from frame 0; costs O(prefix)
  per seed).

The response shape in growing mode:

```json
{
  "tapnext": {
    "status": "done",
    "session": "alice-2025",
    "frames_processed": 60,
    "runtime": 12.3,
    "num_points": 1120,              // max total track ids (may have grown)
    "new_tracks": true,
    "num_generations": 4,            // live generations right now
    "birth_hist": {"0": 1024, "30": 32, "55": 32},   // compact growth summary
    "encoding": { "tracks": "torch", "visibles": "torch",
                  "observation_matrix": "torch" }
  }
}
```

plus one more `data` field: `birth_frames` — a float list, one first-seeded
frame per track id (column `i` of `tracks` = track id `i`).

`tracks (F, N_max, 2)`, `visibles (F, N_max)`, `observation_matrix (2F, N_max)`
— padded exactly as above. Callers that want a per-frame dense tensor can just
mask with `visibles` / `birth_frames` ≤ frame.

Cost note: unlike a "merge and re-init" scheme, generations do **not** lose
their long-range TAPNext memory — surviving points keep their full LRU
history — at the cost of one model forward per *live* generation per frame.
On long sequences, set `retire_after_invisible` (and keep `max_total_points`
bounded); the box's per-request logging counts re-seeds and retirements.

## Docker Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | /workspace | Workspace directory inside container |
| `SERVICE_NAME` | tapnext | Service identifier (affects proto and service file lookup) |

## Deployment

### Local Development

```bash
docker build --tag tapnexttracker -f docker/Dockerfile .
docker run --gpus all -p 8061:8061 tapnexttracker
```

### Production with GPU

```bash
docker run --gpus all \
    -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz \
    -e PORT=8061 \
    --name tapnext tracker \
    tapnexttracker
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8061 | Server listening port |
| `TORCH_HOME` | /workspace/.cache | PyTorch model cache directory |
| `HF_HOME` | /workspace/.cache | HuggingFace cache directory |
| `TAPNEXT_SESSION_TTL` | 1800 | Seconds a session may sit idle before it (and its per-session GPU state) is reaped. 1800 keeps a classroom's pauses alive on a shared GPU while abandoned sessions still get reclaimed. Set it higher for longer student pauses; 0 disables reaping (sessions persist until reset/restart). |

## Performance Notes

- First inference call loads the model (~740MB checkpoint)
- Model stays on GPU after loading unless idle for 120 seconds
- Grid detection generates (grid_size × grid_size) query points per frame
- Track state persists across sequential frames for a session until that session
  is reset or reaped
- Growing sessions: per-frame cost = one forward per live generation; bound it
  with `max_total_points` / `retire_after_invisible` (see above)
- With multiple sessions the shared model is loaded once; each session adds a
  per-session state cost that grows with the video length. Idle sessions are
  reaped after `TAPNEXT_SESSION_TTL` (default 1800 s); raise it or set 0 if
  student work must persist longer. Reap more aggressively on a crowded GPU.

## Testing

Two test entry points:

```bash
# In-process session-isolation suite — NO GPU, NO tapnet wheel, NO Docker.
# Injects a deterministic stub model and exercises Process() directly (plus a
# real gRPC round-trip). This is the fastest way to prove multi-session
# isolation/regression.
cd images/tapnext_tracker/test
python test_tapnext_sessions.py

# In-process growing-point-set suite (new_tracks path) — same stub-model
# approach: growth, stable ids, retirement, backfill, budgets, reset/parking,
# isolation, and legacy back-compat, with exact forward-call counts
cd images/tapnext_tracker/test
python test_tapnext_growing.py

# Live smoke test against a running box (needs the real image + a GPU):
docker build --tag my_tapnext -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -v /path/to/ckpt:/workspace/bootstapnext_ckpt.npz my_tapnext &
cd images/tapnext_tracker/test
python test_tapnext.py          # sequential tracking over gRPC
```

## Troubleshooting

### Checkpoint Not Found

Ensure checkpoint is at `/workspace/bootstapnext_ckpt.npz`:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz tapnexttracker
```

### CUDA Out of Memory

Reduce batch size by processing frames individually or use smaller grid_size.
