# boxes-client

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
from boxes_client import Box

b = Box("localhost:8061")                 # a local box
# b = Box("10.0.0.5:8061")                # ...or a remote one

res = b.run(data={"images": ["frame.jpg"]},
            config={"my_box": {"command": "do_thing", "parameters": {}}})
print(res.fields)     # decoded payload, whatever the box returned
print(res.config)     # parsed config_json
```

Box-specific one-liners live in an *optional* convenience layer, e.g. the
tapnext tracker:

```python
from boxes_client import trace
res = trace(b, images=["frame.jpg"], grid_size=30)
```

No registry, no central server — the client connects directly to the box, so
local and remote boxes are the same call. Boxes stay independent, addressable,
composable units (the *client* does the calling, preserving the distributed
nature of the fleet).

## Install

```bash
pip install boxes-client
# optional, to decode tapnext/vggt tensor payloads to numpy:
pip install "boxes-client[torch]"
```

(From a repo checkout, instead: `pip install -e boxes_client`.)

`zstandard` is a base dependency (the `zstd_pickle` codec needs it); `torch`
stays optional. Without the codec's library, declared payloads degrade to raw
`bytes` with a warning (still usable — e.g.
`torch.load(BytesIO(res.tracks), weights_only=False)`).

The full design of the declared-encoding contract lives in
[docs/CODECS.md](../docs/CODECS.md).

## Core API (box-agnostic) + optional conveniences

`Box` is deliberately box-agnostic: it builds and sends an `Envelope` and reads
a `Result` back, and knows **no box, field, or model**. Per-box sugar lives in a
separate convenience layer, so adding one never touches the core.

### 1. Generic — `Box.run(data, config, method, reset_first)`
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
`boxes_client.conveniences` (not the core). It reads image files (or accepts
pre-encoded bytes) and just calls `Box.run()` with the right shape:

```python
b = Box("localhost:8061")
res = trace(b, images=["f1.jpg", "f2.jpg", "f3.jpg"], grid_size=30)
np.save("tracks.npy", res.tracks.numpy())
```

The core `Box` knows nothing about tapnext — `trace` *is* the only tapnext
knowledge, and it's safe to delete without touching the generic client. Add a
sibling convenience (`segment`, `embed`, `detect`, …) for other boxes the same
way; never put a box name in `box.py`.

### `Box.reset(config_key=None)`
Sends `{config_key: {"command": "reset"}}` on `Process`. `config_key` is the
box's section name (or the one from the constructor). Stateful boxes clear
state; stateless boxes typically ignore it. Pass the box name explicitly, or
construct with `Box(host, config_key="tapnext")`.

### `Box.info()`
Asks the box (via gRPC reflection) whether it serves `pipeline.PipelineService`.
Useful to check box reachability and shape before committing to a call.

## Value coercion (for `Box.run` / convenience `data` values)

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
wins). Named codecs (`[src/boxes_client/codec.py](src/boxes_client/codec.py)`,
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
(`[src/boxes_client/decode_util.py](src/boxes_client/decode_util.py)`):
**JSON → torch (if installed) → numpy → raw bytes**, kept only for boxes that
declared nothing. Guessing is approximate (the numpy branch will happily
reinterpret any 4-byte-aligned blob); boxes are expected to declare.

## What's supported

| Box | In scope | Notes |
|-----|----------|-------|
| tapnext (`Process`) | ✅ | v1 target; `trace(box, ...)` convenience over `Box.run` |
| vggt, moege_box, clip, lang_segm, **yolo** (`Process`) | ✅ envelope shape | call via `Box.run(...)` with the box-specific `config`; `yolo` always tracks (per-session `track_id` in `detections`; `session_id`/`reset`/`list` like tapnext) and declares per-field `encoding` (`detections` json, `annotated` identity) |
| opencv_box (`Process`) | ✅ | standard envelope: `match` / `similarity_check` / `reset` commands; `numpy` fields declared and decoded (np.save blobs), similarity frames as `identity` |
| cotracker (`Forward`) | ⏸ pending | use `Box.run` after it's migrated to the shared envelope (client needs no changes) |

### Method dispatch caveat

`Box.run(..., method=...)` resolves to a method on the **client-stub**, which is
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
python boxes_client/tests/fake_box_smoke.py

# real tapnext box at BOX_HOST:PORT
BOX_HOST=localhost:8061 python boxes_client/tests/live_tapnext.py
```

## Notes / design

- **Client-side only.** The box stays a box: independent, restartable,
  composable. Distribution is preserved because the *client* dials the box
  directly, so "local box" and "remote box" are the same call.
- **Auto-protocol.** `Box.info()` uses gRPC reflection to confirm the box
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
