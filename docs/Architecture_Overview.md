# Architecture Overview

## What this repo is

**boxes** is a fleet of independent, Dockerized AI inference services ("boxes"),
each speaking one shared gRPC interface. Nothing here is a monolith: every box
is its own image, its own process, its own port mapping — you can start it,
stop it, move it to another machine, or run several of them side by side
without the others noticing.

```
   caller (visionist_client or any orchestration layer)
        │  gRPC  Envelope → Envelope
        ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ clip box  │   │ lang_segm │   │ tapnext   │   … one Docker image each
   └───────────┘   └───────────┘   └───────────┘
        each box:   proto + service.py + model, listening on :8061
```

The calling side does the orchestration: `visionist_client` (or any other
process) dials the box it needs, directly, by `ip:port`. That keeps
the boxes independent and restartable and preserves the distributed nature of
the fleet.

## The shared contract

Every "standard" box serves

```
service PipelineService { rpc Process(Envelope) returns (Envelope); }
```

The `Envelope` carries everything: a `config_json` string (JSON metadata,
per-box) and a `data` map of named, typed values (images, tensors, blobs).
Full request/response contract: see
[gRPC_Services_Reference](gRPC_Services_Reference.md).

(All standard boxes serve the single shared `Process` RPC;
box-specific behaviour is selected with a `command` field.)

## Box layout (every box is the same shape)

```
images/<name>/
├── protos/
│   ├── pipeline.proto          # the shared envelope (see protos/ at repo root)
│   ├── pipeline_pb2.py         # generated at build time
│   ├── pipeline_pb2_grpc.py
│   └── aux.py                  # wrap_value / unwrap_value helpers (see protos/aux.py)
├── src/
│   └── <name>_service.py       # the actual gRPC servicer
├── docker/
│   └── Dockerfile              # multi-stage: builder → runtime (CUDA base for GPU boxes)
├── test/
│   └── test_*.py               # standalone smoke test against a running box
├── requirements.txt
└── README.md                   # ← box-specific request/response reference; read this first
```

The **box README is the authoritative source for that box's request shape**
(config keys, fields, how to decode `results`). The central docs here describe
the shared pieces; the per-box READMEs describe the box-specific ones.

## Conventions every box follows

| Convention | Value |
|---|---|
| Port | **8061** (AI4EU spec); overridable with the `PORT` env var |
| Message limits | `grpc.max_*_message_length = -1` (payloads carry images/tensors) |
| Reflection | gRPC reflection enabled — `Visionist.info()` works |
| User | non-root `runner` inside the container |
| GPU lifecycle | CPU at startup → move to GPU on first request (when CUDA is visible) → watchdog falls back to CPU after ~60 s idle. See [GPU lifecycle](#gpu-memory-lifecycle) |
| Config | JSON section namespaced under the box key (`{"clip": {...}}`, `{"lang_sam": {...}}`, …) |
| Heavy results | `data.results` often carries `zstd.compress(pickle.dumps(list))` |
| Payload encoding | declared in the response `config_json` (`"encoding"`, a codec name or `{field: codec}` map); codecs are generic (`identity`/`json`/`torch`/`numpy`/`zstd_pickle`); boxes that declare nothing return raw `bytes` by default. See [CODECS.md](CODECS.md) |
| Stateless request | `{"command": "reset"}` is accepted on every standard box; stateless boxes no-op it. The stateful boxes (tapnext, yolo) are **multi-session**: `session_id` in the config section keys all state (omitted → one shared `default` session), so `reset` only clears the calling session |

### GPU memory lifecycle

GPU boxes (clip, textEmbedding, lang_segm) load their model **on CPU at
startup** so the container starts fast and with no VRAM. On the first request,
if CUDA is visible, the model is moved to the GPU **in place** (`.to(device)`
on the torch modules — cheap, no checkpoint reload). A watchdog thread moves
it back to CPU after ~60 s of inactivity and releases the SAM2 predictor's
cached feature maps, so `empty_cache()` can actually recover the memory.

Expect a **fixed VRAM floor of a few hundred MB (≈0.9 GB with this stack)**
after *any* GPU use: that is the CUDA context + cuDNN/cuBLAS workspaces,
inherent to a process that initialized CUDA. It is not a leak and it does not
grow with traffic.

`tapnext_tracker` loads straight to CUDA at startup when available (it runs
per-frame); `opencv_box` does the same lazy loading for its LightGlue models
(created on first use, parked on CPU when idle); `vggt` uses the same lazy
in-place moves (its 1B weights come from the baked image or an on-startup
download — see its README). All variants respect an explicit
`parameters.device` in the request when provided.

## Calling boxes

The primary client is [`visionist_client`](../visionist_client/README.md) — a thin
Python package that dials one box:

```python
from visionist_client import Visionist
b = Visionist("localhost:8061")
res = b.run(data={"images": ["frame.jpg"]},
            config={"tapnext": {"command": "track", "parameters": {"grid_size": 30}}})
print(res.config, res.tracks)
```

No registry, no central server: local and remote boxes are the same call.
Orchestration (calling several boxes in sequence) is the caller's job —
`fleet/hello.py` is the minimal multi-box walkthrough.

## Retired

An earlier orchestration approach based on the **Maestro** pipeline YAML
(`kind: pipeline / stage / link`, docker-compose stacks in a `pipelines/`
directory) is **not in use** and no longer documented here. If you find old
config files for it, treat them as historical artifacts.
