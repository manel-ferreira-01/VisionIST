# Quick Start Guide

Assumes Docker. `pip install -e visionist_client` for the client (the
recommended way to talk to any box).

## 1. Run an existing box

Pick any box from `images/` (each has a README with its exact request shape):

```bash
cd images/lang_segm
docker build --tag my_lang_segm -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host my_lang_segm
```

Every box listens on **8061** (AI4EU spec), so running several at once means
mapping each container to a different host port:

```bash
docker run -d -p 9061:8061 ...   # lang_segm on host :9061
docker run -d -p 9062:8061 ...   # clip on     host :9062
```

## 2. Call it

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")
print(b.info())   # reachability + reflection check
res = b.run(
    data   = {"images": [pathlib.Path("dog.jpg")]},
    config = {"lang_sam": {
        "command": "segment",
        "parameters": {"box_threshold": 0.3, "text_threshold": 0.25},
        "text_prompt": ["a dog"],
    }},
)
print(res.config)   # {"lang_sam": {"status": "done", "runtime": …, …}}
print(res)          # fields are best-effort decoded (JSON→torch→numpy→bytes)
```

Every box also ships a `test/test_*.py` smoke test you can point at a running
box:

```bash
python images/lang_segm/test/test_lang_sam.py
BOX_HOST=10.0.0.5:9061 python images/lang_segm/test/test_lang_sam.py
```

## 3. Build a box of your own

### 3.1 Laying out

```
images/<name>/
├── protos/            # pipeline.proto + aux.py, from the repo root protos/
├── src/<name>_service.py
├── docker/Dockerfile
├── test/test_<name>.py
├── requirements.txt
└── README.md          # request/response reference — required
```

```bash
mkdir -p images/<name>/{protos,src,docker,test}
cp protos/pipeline.proto images/<name>/protos/
cp protos/aux.py         images/<name>/protos/
```

### 3.2 The service (the standard pattern)

`images/<name>/src/<name>_service.py` — CPU-at-start, auto-GPU on request,
idle fallback to CPU. `tapnext`/`clip`/`lang_segm` follow this shape.

```python
import concurrent.futures as futures
import grpc, json, logging, os, sys, threading, time
import torch
sys.path.append("./protos")
import pipeline_pb2, pipeline_pb2_grpc
from aux import wrap_value, unwrap_value

_IDLE_TIMEOUT = 60  # seconds

class Box(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self._model = load_model(device="cpu")          # CPU first: fast start, no VRAM
        self._device = "cpu"
        self._last_request_time = time.time()
        self._lock = threading.Lock()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._lock:
                if time.time() - self._last_request_time > _IDLE_TIMEOUT \
                        and self._device.startswith("cuda"):
                    self._move_unlocked("cpu")

    def _move_unlocked(self, target):
        with self._lock:
            ...  # self._model.to(target); self._device = target; torch.cuda.empty_cache()

    def Process(self, request, context):
        try:
            cfg = json.loads(request.config_json)
            box_cfg = cfg.get("my_box", {})                       # ← your box key
            if box_cfg.get("command") == "reset":                 # always accept
                return pipeline_pb2.Envelope(config_json=json.dumps(
                    {"my_box": {"status": "done", "action": "reset"}}))

            parameters = box_cfg.get("parameters", {}) or {}
            images = unwrap_value(request.data["images"]) if "images" in request.data else None
            if not images:
                return pipeline_pb2.Envelope(config_json=json.dumps(
                    {"my_box": {"status": "empty_request"}}))

            with self._lock:                                      # auto-GPU, matches clip/sbert/lang_segm
                self._last_request_time = time.time()
                target = (parameters.get("device") or "").lower()
                if not target and torch.cuda.is_available():
                    target = "cuda"
                if target and target != self._device:
                    self._model.to(torch.device(target)); self._device = target

            out_list = self._model.predict(images)                # your real workload
            import zstandard as zstd, pickle
            blob = zstd.ZstdCompressor().compress(pickle.dumps(out_list))
            return pipeline_pb2.Envelope(
                config_json=json.dumps({"my_box": {"status": "done", "num_images": len(images)}}),
                data={"results": wrap_value(blob)})
        except Exception as e:
            logging.exception("inference failed")
            return pipeline_pb2.Envelope(config_json=json.dumps(
                {"my_box": {"status": "error", "error": str(e)}}))

if __name__ == "__main__":
    import grpc_reflection.v1alpha.reflection as grpc_reflection
    logging.basicConfig(level=logging.INFO)
    server = grpc.server(futures.ThreadPoolExecutor(), options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ])
    pb2_grpc.add_PipelineServiceServicer_to_server(Box(), server)
    grpc_reflection.enable_server_reflection(
        (pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
         grpc_reflection.SERVICE_NAME), server)
    server.add_insecure_port(f"[::]:{os.getenv('PORT', '8061')}")
    server.start()
    server.wait_for_termination()
```

### 3.3 Dockerfile (minimal, follows the repo convention)

```dockerfile
ARG WORKSPACE=/workspace

FROM python:3.10-slim AS builder
RUN pip install --upgrade pip && pip install grpcio grpcio-tools protobuf
COPY protos ${WORKSPACE}/
WORKDIR ${WORKSPACE}
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

FROM nvidia/cuda:12.2.2-base-ubuntu22.04        # python:3.10-slim for CPU boxes
ARG USER=runner
RUN addgroup --system runner-group && \
    adduser --system --no-create-home --ingroup runner-group runner && \
    mkdir ${WORKSPACE} && chown -R runner:runner ${WORKSPACE}

COPY requirements.txt .
RUN apt update -y && apt install -y pip && apt-get clean && \
    pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY --from=builder ${WORKSPACE}/*.py ${WORKSPACE}/
COPY src/${NAME}_service.py ${WORKSPACE}/service.py
COPY protos/pipeline.proto /

EXPOSE 8061
WORKDIR ${WORKSPACE}
USER runner
CMD ["python3", "service.py"]
```

### 3.4 Test + smoke-test

```bash
cd images/<name>
docker build --tag my_box -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host my_box

# smoke test
python test/test_<name>.py

# client check
python - <<'PY'
from visionist_client import Visionist, pathlib
b = Visionist("localhost:8061")
print(b.info())
print(b.run(data={"images": [pathlib.Path("x.jpg")]},
            config={"my_box": {"command": "do_it", "parameters": {}}}).config)
PY
```

### 3.5 Ship it

Write `images/<name>/README.md` — the per-box reference (what to put in
`config`, what to expect in `config_json.status`, how to decode `results`).
That one file tells you whether the box is done.

## Troubleshooting

| Symptom | First check |
|---|---|
| `Connection refused` on `:8061` | `docker logs <ctr>`; did the container actually start? |
| `status: error`, `no config_json` | Your request had an empty `config_json`. |
| `status: error`, `no images in data` | You passed an `str` where a file was expected — use `pathlib.Path` or raw `bytes`. |
| `status` not `done` and no `results` | Look at the box log; the client's `res.config["<box_key>"]["error"]` usually has the reason. |
| CUDA OOM while the model is "on CPU" | The box may have fallen back to CPU after `_IDLE_TIMEOUT`; the model re-migrates on next request. |
| Residual VRAM after fallback | Expected — see [Architecture → GPU memory lifecycle](Architecture_Overview.md). |

## Where to go next

- **Box contract & conventions**: [gRPC_Services_Reference](gRPC_Services_Reference.md)
- **Whole-box mental model**: [Architecture_Overview](Architecture_Overview.md)
- **Docker build templates**: [Docker_Image_Template_Guide](Docker_Image_Template_Guide.md)
- **Client details**: [`visionist_client/README.md`](../visionist_client/README.md)
- **A box's exact API**: `images/<name>/README.md`
