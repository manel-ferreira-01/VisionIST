# visionist-client

A thin Python client for calling deployed AI **"boxes"** by `IP:port`.

A *box* is one of the gRPC services in [`/images/`](../images/) built to the
shared

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

interface. Point this client at **any one box** and send an `Envelope`:

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")                 # a local box
# b = Visionist("10.0.0.5:8061")                # ...or a remote one

res = b.run(data={"images": ["frame.jpg"]},
            config={"my_box": {"command": "do_thing", "parameters": {}}})
print(res.fields)     # decoded payload, whatever the box returned
print(res.config)     # parsed config_json
```

Box-specific one-liners live in an *optional* convenience layer, e.g. the
tapnext tracker:

```python
from visionist_client import trace
res = trace(b, images=["frame.jpg"], grid_size=30)
```

No registry, no central server — the client connects directly to the box, so
local and remote boxes are the same call. Boxes stay independent, addressable,
composable units (the *client* does the calling, preserving the distributed
nature of the fleet).

## Install

```bash
pip install visionist-client
# optional, to decode tapnext/vggt tensor payloads to numpy:
pip install "visionist-client[torch]"
```

(From a repo checkout, instead: `pip install -e visionist_client`.)

`zstandard` is a base dependency (the `zstd_pickle` codec needs it); `torch`
stays optional. Without the codec's library, declared payloads degrade to raw
`bytes` with a warning (still usable — e.g.
`torch.load(BytesIO(res.tracks), weights_only=False)`).

The full design of the declared-encoding contract lives in
[docs/CODECS.md](../docs/CODECS.md).

## Core API (box-agnostic) + optional conveniences

`Visionist` is deliberately box-agnostic: it builds and sends an `Envelope` and reads
a `Result` back, and knows **no box, field, or model**. Per-box sugar lives in a
separate convenience layer, so adding one never touches the core.

### 1. Generic — `Visionist.run(data, config, method, reset_first)`
The workhorse. No assumption about field names or payload types:

```python
b.run(
    data    = {"images":    [img1, img2]},           # any field names; any types
    config  = {"my_box":    {"command": "do_thing", "parameters": {...}}},
    method  = "Process",                              # default; only boxes with an extra named RPC need anything else
)
```

`data` is a dict of `field_name -> value`. Values are coerced per the rules
below and wrapped into the shared `Value` oneof via the vendored `aux.wrap_value`.

### 2. Convenience — `trace(box, images, grid_size=None, reset_first=True)`
An **optional** one-liner for the *tapnext* box, living in
`visionist_client.conveniences` (not the core). It reads image files (or accepts
pre-encoded bytes) and just calls `Visionist.run()` with the right shape:

```python
b = Visionist("localhost:8061")
res = trace(b, images=["f1.jpg", "f2.jpg", "f3.jpg"], grid_size=30)
np.save("tracks.npy", res.tracks.numpy())
```

The core `Visionist` knows nothing about tapnext — `trace` *is* the only tapnext
knowledge, and it's safe to delete without touching the generic client. Add a
sibling convenience (`segment`, `embed`, `detect`, …) for other boxes the same
way; never put a box name in `box.py`.

### 3. Convenience — `track_stream(box, frames, *, window=3, min_alive=2, mode="backbone", session_id="default", reset_first=True, **params)`
An **optional** helper for the *lightglue* box: feed a sequence of frames
(one per call, time-ordered) through the box's `stream` command in a single
session, then build a **Tomasi‑Kanade observation matrix** — shape
`(2F, kept)`, `P[2f]` = x / `P[2f+1]` = y per frame, **one column per
tracked point**, `NaN` where the point is absent.

```python
from visionist_client import track_stream
demo = track_stream(b, frames=["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"],
                    window=3, min_alive=2)   # mode: "backbone" | "greedy"
demo.obs_matrix        # np.ndarray (2F, kept)
print(demo)            # summary: components, kept/dropped, gap-bridged, per-frame
```

- **the `mode` toggle** — `"backbone"` (default): union-find candidate sets +
  longest-path peeling; a point that blinks out for one frame is **re-linked
  via Δ=2/3 matches** (NaN gap, one track). `"greedy"`: Δ=1 chains only; a
  point that fails to match the next frame dies and re-appears as a new track.
- `min_alive` — drop tracks seen in fewer frames (the "really big blobs"
  resolve into clean per-point tracks; one node per frame is guaranteed).
- The box-agnostic machinery (union-find, peeling, matrix) lives in
  `visionist_client.tracking` — pure functions over match edges + per-frame
  keypoints, unit-tested without a box (`tests/tracking_smoke.py`). The
  convenience only knows the lightglue request/response shape.

### `Visionist.reset(config_key=None)`
Sends `{config_key: {"command": "reset"}}` on `Process`. `config_key` is the
box's section name (or the one from the constructor). Stateful boxes clear
state; stateless boxes typically ignore it. Pass the box name explicitly, or
construct with `Visionist(host, config_key="tapnext")`.

### `Visionist.info()`
Asks the box (via gRPC reflection) whether it serves `pipeline.PipelineService`.
Useful to check box reachability and shape before committing to a call.

## Value coercion (for `Visionist.run` / convenience `data` values)

| You pass | What's sent |
|---|---|
| `bytes` / `bytearray` / `memoryview` | `b` (single bytes) or `bb` (list) — for pre-encoded binary payloads (images, tensors, etc.) |
| `pathlib.Path` | file bytes (read locally, sent as `b`) |
| `str` | **literal string** (NOT a file path) — sent as `s` |
| `int` | coerced to `float` (proto `f` is a float) |
| list of the above (homogeneous) | corresponding list: `BytesList` / `StringList` / `FloatList` |

That's why a *file* has to be passed as `pathlib.Path` (or a `bytes` you loaded
yourself) — the client serializes it and ships it. A *literal string* is just
`str` — no confusion.

```python
import pathlib
# file:
b.run(data={"images": [pathlib.Path("frame.jpg")]}, config={...})

# preencoded:
b.run(data={"images": [open("frame.jpg","rb").read()]}, config={...})

# literal text (e.g. a hypothetical text-only envelope box):
b.run(data={"sentences": ["hello", "world"]}, config={...})

# list of floats:
b.run(data={"vals": [1, 2, 3]}, config={...})       # -> FloatList
```

## Result object

`Result` wraps the raw response `Envelope`:

- `fields` — dict of `field_name -> decoded value`
- `.tracks`, `.visibles`, … — direct field access via `__getattr__`
- `config` — parsed `config_json`
- `encoding` — the box's **declared** payload encoding (codec-name string,
  `{field: codec}` map, or `None`), so you can see *why* a field is decoded
  or raw
- `raw` — the undecoded `Envelope` proto
- `as_dict()` — JSON-friendly version (numpy arrays -> `tolist`)

### Decoding: declared first, legacy guess as fallback

The authoritative path is the box's **declaration**: the response `config_json`
carries a generic `"encoding"` key — a codec name for all `bytes` fields, or a
`{field_name: codec_name}` map:

```json
{ "lang_sam": { "status": "done", "encoding": "zstd_pickle" } }
```

``encoding`` scans the parsed config top-level, then each section (first hit
wins). Named codecs (`[src/visionist_client/codec.py](src/visionist_client/codec.py)`,
registry `CODECS` / `decode_with`, full design in [docs/CODECS.md](../docs/CODECS.md)):

| name          | payload                               | decoded to              |
|---------------|---------------------------------------|-------------------------|
| `identity`    | raw bytes (the default)              | `bytes` unchanged       |
| `json`        | UTF-8 JSON                            | `list`/`dict`           |
| `torch`       | `torch.save()` tensor / dict          | `Tensor` / `dict`       |
| `numpy`       | raw float32 buffer                    | `np.ndarray`            |
| `zstd_pickle` | `zstd.compress(pickle.dumps(obj))`    | decoded Python          |

Unknown names or a missing codec library degrade to **raw bytes + a
warning** — never an exception.

The legacy guess chain
(`[src/visionist_client/decode_util.py](src/visionist_client/decode_util.py)`):
**JSON → torch (if installed) → numpy → raw bytes**, kept only for boxes that
declared nothing. Guessing is approximate (the numpy branch will happily
reinterpret any 4-byte-aligned blob); boxes are expected to declare.

## What's supported

| Box | In scope | Notes |
|-----|----------|-------|
| tapnext (`Process`) | ✅ | v1 target; `trace(box, ...)` convenience over `Visionist.run` |
| lightglue_box (`Process`) | ✅ | `match` (features + `matches`/`confidence`) and `stream` (per-`session_id` sliding window); `track_stream(box, frames, ...) -> TrackResult` convenience turns a `stream` run into a cleaned observation matrix (`2F x tracks`, NaN gaps) — the box-agnostic machinery (union-find, longest-path peeling, `min_alive`) lives in `visionist_client.tracking`, toggle with `mode="backbone" | "greedy"` |
| vggt, moege_box, clip, lang_segm, **yolo** (`Process`) | ✅ envelope shape | call via `Visionist.run(...)` with the box-specific `config`; `yolo` always tracks (per-session `track_id` in `detections`; `session_id`/`reset`/`list` like tapnext) and declares per-field `encoding` (`detections` json, `annotated` identity) |
| opencv_box (`Process`) | ✅ | standard envelope: `match` / `similarity_check` / `reset` commands; `numpy` fields declared and decoded (np.save blobs), similarity frames as `identity` |
| cotracker (`Forward`) | ⏸ pending | use `Visionist.run` after it's migrated to the shared envelope (client needs no changes) |

### Method dispatch caveat

`Visionist.run(..., method=...)` resolves to a method on the **client-stub**, which is
built from the shared `pipeline.proto`. Today the shared proto defines only
`Process`, so `method=` is currently useful only for `Process`. Boxes that add
extra `PipelineService` RPCs will need either (a) to be migrated to a *single*
`Process` with a `command` field (the pattern this fleet uses — opencv_box already
is), or (b) to have their own proto vendored into the client. This is the only
remaining "per-box" knowledge in the client — everything else (field names,
payload types, config shape) is fully generic.

## Run the tests

```bash
# in-process fake box (no GPU, no real box needed)
python visionist_client/tests/fake_box_smoke.py

# the box-agnostic tracker (synthetic edges — no box, GPU, or gRPC needed)
python visionist_client/tests/tracking_smoke.py

# real tapnext box at BOX_HOST:PORT
BOX_HOST=localhost:8061 python visionist_client/tests/live_tapnext.py
```

## Notes / design

- **Client-side only.** The box stays a box: independent, restartable,
  composable. Distribution is preserved because the *client* dials the box
  directly, so "local box" and "remote box" are the same call.
- **Auto-protocol.** `Visionist.info()` uses gRPC reflection to confirm the box
  serves `pipeline.PipelineService`. If the box does not serve reflection,
  calls still work — `info()` just reports `reflection: False`.
- **Declared-first decoding.** Boxes declare `"encoding"` in the response
  config; the named codec decodes it. Undeclared payloads fall back to the
  legacy JSON → torch → numpy → raw-bytes guess chain. Nothing raises; you
  always get a `Result` (unknown codec / missing library → raw bytes +
  warning).
- **Forward boxes deferred.** cotracker / textEmbedding use bespoke
  `Forward` messages; they will work through this client with no changes once
  migrated to the shared envelope (clip is already migrated).
- **Association is caller-side.** Stream boxes (lightglue, tapnext) return
  what the model returns; turning those outputs into tracks is the caller's
  job and lives here, not in the boxes. `visionist_client.tracking` is the
  box-agnostic association core (union-find, longest-path peeling, `min_alive`,
  the observation matrix) — pure functions over match edges + per-frame
  keypoints, unit-tested with no box. `track_stream` is the lightglue-shaped
  I/O wrapper around it (same rule as `trace` for tapnext): boxes stay thin
  wrappers over their networks, tracking policy stays with the caller.
