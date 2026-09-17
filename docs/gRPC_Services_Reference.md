# Envelope & Box Contract Reference

How boxes actually speak to each other and to `boxes_client`. This is the
contract — when you build a box or a client, match **this**, not the historical
examples.

## The proto

Shared by every standard box (canonical copy: [`protos/pipeline.proto`](../protos/pipeline.proto)):

```protobuf
message FloatList  { repeated float  values = 1; }
message StringList { repeated string values = 1; }
message BytesList  { repeated bytes  values = 1; }

message Value {
  oneof kind {
    bytes  b       = 1;   // single blobs (bytes, pickle, torch tensor, …)
    string s       = 2;
    float  f       = 5;
    BytesList  bb  = 6;
    StringList ss  = 7;
    FloatList  ff  = 10;
  }
}

message Envelope {
  string config_json       = 1;  // JSON string; per-box section
  map<string, Value> data   = 2; // named payload fields
}

service PipelineService {
  rpc Process(Envelope) returns (Envelope);
}
```

`wrap_value` / `unwrap_value` live in [`protos/aux.py`](../protos/aux.py) and
are vendored into each box at build time. They do exactly two things: coerce
Python objects (scalar / homogeneous list) into the `Value` oneof and back.
Use them; don't hand-roll the oneof.

## `config_json` contract

`config_json` is always a **JSON object namespaced under the box's key**:

```json
{
  "lang_sam": {
    "command": "segment",
    "parameters": { "box_threshold": 0.3, "text_threshold": 0.25, "device": "cuda:0" },
    "text_prompt": ["an excavator", "the wood pile"]
  }
}
```

- `command` — what to do. Every standard box accepts `"reset"` (stateless boxes
  no-op it; tapnext clears its tracker).
- `parameters` — box-specific knobs (thresholds, grid sizes, `device`, …).
- Any box-specific top-level fields (e.g. `lang_sam.text_prompt`) sit next to
  them.

### Box key per box

| Box (`images/…`) | Box key | Notes |
|---|---|---|
| tapnext_tracker | `tapnext` | stateful & **multi-session**: `session_id` (opaque string, omitted → shared `default` session) keys all state; `reset` clears that session only; `list` gives the operator the active sessions; input is `data.images` (a list of frames) **or** a single decoded `data.video` (the box decodes the video to ordered frames — mutually exclusive with `images`, `frame_step`/`max_frames` to sample); response declares `tracks`/`visibles`/`observation_matrix` as `torch` |
| clip | `clip` | stateless |
| textEmbedding | `sbert` | stateless |
| lang_segm | `lang_sam` | aliases accepted: `lang_segm`, `aispgradio`, or the legacy flat form |
| opencv_box | `opencv` | `match` (default) / `reset` as commands of `Process`; `match` is stateless (1 image: extraction, 2: + matches & fundamental matrix); response declares `keypoints`/`descriptors`/`matches_inliers_a`/`matches_inliers_b`/`fundamental_matrix` as `numpy` (np.save blobs) |
| vggt | `vggt` | stateless; legacy `aispgradio` section / flat form accepted; response declares per-field `encoding` (torch tensors, identity GLB) |
| moge_box | `moge` | stateless; **CUDA-only** (`cpu` rejected); outputs a per-image dict — `points (H,W,3)` / `depth (H,W)` / `normal (H,W,3)` / `intrinsics (3,3)` / `mask (H,W)` (OpenCV camera coords), encoding `zstd_pickle` |
| yolo | `yolo` | **multi-session tracker state** (tapnext contract): always tracks — every box in `detections` carries a per-session `track_id` (stable within a `session_id`, independent across sessions); `reset` scopes to the calling session, `list` lists active ones; `data.images` and/or a single decoded `data.video` (`frame_step`/`max_frames`); response declares `detections` as `json`, `annotated` as `identity` |
| lightglue_box | `lightglue` | stateless; `match` (default) / `reset` as commands of `Process`; 1 image → its `keypoints`/`descriptors`/`scores`, 2 images → + the LightGlue matcher's `matches` (index pairs) and `confidence`; all declared `numpy` (np.save blobs); `SUPERPOINT` (default) / `DISK`, `max_keypoints`, `filter_threshold`, `device` |

**Legacy flat form**: boxes written before the convention read
`config_json` directly (`{"parameters": {...}, "stream": 2}`); `lang_segm`
deliberately still accepts that shape for old callers. New code should use the
namespaced form.

## `data` fields

Free-form map; field names are per-box agreements, not proto-level. What every
box in this repo converges on:

| Field | Type | Meaning |
|---|---|---|
| `images` | `BytesList` (list of `b`) | input images, typically JPEG/PNG bytes |
| `texts` / `sentences` | `StringList` | input text (clip, sbert) |
| `results` | `b` (single bytes) | **heavy outputs**, see below |
| `tracks`, `visibles` | `b` | tensor payloads (tapnext: `torch.save`-format bytes, decoded by the client if torch is present) |
| `glb`, `frames`, … | `b` | box-specific binaries |

### Heavy results: `zstd(compress) + pickle`

For list-shaped results that can't fit in JSON (LangSAM masks),
the convention is:

```python
blob = zstandard.ZstdCompressor().compress(pickle.dumps(out_list))
# server:
return Envelope(data={"results": wrap_value(blob)})

# client:
out_list = pickle.loads(zstandard.ZstdDecompressor().decompress(bytes(blob)))
```

LangSAM (`lang_segm`) produces it — a cross-box contract: any consumer box
(or the client) can decode it.

**Declared encoding (the contract, not a guess).** Boxes describe their
payload in the response `config_json` with the generic `"encoding"` key:

- a **string** codec name → applies to every `bytes` field:
  `"lang_sam": {"status": "done", "encoding": "zstd_pickle"}`
- an **object** → a `{field_name: codec_name}` map for mixed responses:
  `"clip": {"status": "done", "encoding": {"image_emb": "torch", …}}`

Codec vocabulary (generic names, pure `bytes -> object`; full design in
[CODECS.md](CODECS.md)):

| name          | input                            | output                        |
|---------------|----------------------------------|-------------------------------|
| `identity`    | raw bytes                        | `bytes` (unchanged; the default) |
| `json`        | UTF-8 JSON                       | `list`/`dict`                 |
| `torch`       | `torch.save()` tensor / dict     | `Tensor` / `dict[Tensor]`     |
| `numpy`       | `numpy.save` (`.npy`) blob, or plain float32 buffer (legacy) | `np.ndarray`                  |
| `zstd_pickle` | `zstd.compress(pickle.dumps(...))` | decoded Python (usually `list`) |

`boxes_client` decodes declared fields directly (`res.encoding` exposes
what was declared); a box that declares nothing — or a codec whose library
is missing — still returns usable raw `bytes`, and old clients that ignore
the key keep decoding by their legacy JSON → torch → numpy → raw guess
chain.

## Response `config_json`

Standard boxes answer `Process` with a namespaced status:

```json
"lang_sam": {
  "status": "done",               // done | empty_request | error
  "runtime": 4.21,                // seconds when available
  "num_images": 2,
  "num_prompts": 2
}
```

- **`done`** — success; payload in `data.results`.
- **`empty_request`** — the `images` field was missing/empty.
- **`error`** — missing config, no prompt, or inference failure; the human
  readable reason is in `"error"`.
  Config-only envelopes (a caller's "no work, continue" ping) are answered
  by every standard box with `status: empty_request`.

## Devices & GPU behaviour (as deployed)

- `parameters.device`: optional string (`"cpu"` / `"cuda"` / `"cuda:0"`).
  When present it wins explicitly.
- When absent, GPU boxes use CUDA if visible, else CPU.
- GPU boxes fall back to CPU after `_IDLE_TIMEOUT` (usually 60 s); the move is
  in-place on the torch modules and releases the SAM2 predictor's feature
  cache so `empty_cache()` reclaims VRAM.
- After the first GPU use, a fixed VRAM floor (~0.9 GB with this stack) is the
  CUDA context — not a leak.

## Calling a box

### Primary: `boxes_client`

```python
from boxes_client import Box
import pathlib

b = Box("localhost:8061")                 # or 10.0.0.5:8061
# quick reachability check (uses gRPC reflection)
print(b.info())

res = b.run(
    data   = {"images": [pathlib.Path("frame.jpg"), pathlib.Path("frame2.jpg")]},
    config = {"lang_sam": {
        "command": "segment",
        "parameters": {"box_threshold": 0.3, "text_threshold": 0.25},
        "text_prompt": ["a car", "the road"],
    }},
    # method   = "Process",                # default
    # reset_first = True,                   # sends {"lang_sam": {"command":"reset"}} first
)
print(res.config)        # {"lang_sam": {"status": "done", "runtime": …, …}}
print(res.results)       # decoded: list of LangSAM.out dicts
```

Client rules (full doc: [`boxes_client/README.md`](../boxes_client/README.md)):

- `str` in `data` = **literal** string (not a file). For files, pass
  `pathlib.Path` or raw `bytes`.
- Homogeneous lists become `BytesList` / `StringList` / `FloatList`; scalars
  become `b` / `s` / `f`.
- Decoding: the box **declares** its payload encoding in the response
  config (`"encoding"`, above); declared fields are decoded by the named
  codec, undeclared ones by the legacy JSON → torch → numpy → raw-bytes
  chain. The client never raises; `res.encoding` shows what was declared.

### Raw (without the client)

```python
import grpc, json
import pipeline_pb2, pipeline_pb2_grpc, aux   # vendored from protos/

stub = pipeline_pb2_grpc.PipelineServiceStub(
    grpc.insecure_channel("localhost:8061",
                          options=[("grpc.max_send_message_length", -1),
                                   ("grpc.max_receive_message_length", -1)]))
req = pipeline_pb2.Envelope(
    config_json=json.dumps({"clip": {"command": "encode",
                                     "parameters": {"model": "ViT-B/32"}}}),
    data={"images": aux.wrap_value([open("frame.jpg","rb").read()])},
)
resp = stub.Process(req)
print(json.loads(resp.config_json))
```

## Testing a new box standalone

```bash
# 1) build & start
docker build --tag my_box -f images/<name>/docker/Dockerfile images/<name>/
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host my_box

# 2) smoke test (each box ships one under test/)
python images/<name>/test/test_<name>.py

# 3) call from the client (fastest sanity check)
python - <<'PY'
from boxes_client import Box
import pathlib
b = Box("localhost:8061")
print(b.info())
print(b.run(data={"images":[pathlib.Path("test.jpg")]},
            config={"my_box":{"command":"segment","parameters":{}}}).config)
PY
```

## Checklist for a *new* standard box

- [ ] `protos/pipeline.proto` copied from `protos/` at the repo root
- [ ] `aux.py` vendored (wrap/unwrap)
- [ ] `Process(Envelope) -> Envelope` implemented
- [ ] `config_json` namespaced under the box key (`{"my_box": {...}}`)
- [ ] `parameters` read with safe defaults; `command: reset` accepted
- [ ] `data.images` / `data.texts` / … documented in the box README
- [ ] Heavy results → `zstd(pickle)` into `data.results`
- [ ] `status` in `done | empty_request | error`, `error` in JSON on failure
- [ ] Large message limits set (`-1`), reflection enabled, PORT env respected
- [ ] GPU boxes: CPU at startup → auto-GPU on request → idle-timeout fallback
      (+ release of any per-inference caches before `empty_cache()`)
- [ ] Standalone test under `test/`, and a README with a worked example
