# Docker Image Template Guide

Templates for the Docker images used in the boxes fleet. The service-code
conventions shown here match the boxes in [`../images/`](../images/) — when in
doubt, read the closest existing box.

## Directory StructureTemplate

```
/images/my_service/
├── protos/               # Protocol buffer definitions
│   ├── my_service.proto  # Your service definition
│   └── aux.py            # Helper functions (wrap_value, unwrap_value)
├── src/                  # Service implementation code
│   └── my_service.py     # Main gRPC server
├── docker/               # Docker build files
│   ├── Dockerfile        # Multi-stage build
│   └── README.md         # Build instructions for this image
└── requirements.txt      # Python dependencies
```

---

## Template 1: Standard CPU Service

**Use when:** No GPU required (small models, preprocessing, ML frameworks without CUDA)

### protos/my_service.proto

```protobuf
syntax = "proto3";

package pipeline;

message Envelope {
  string config_json = 1;
  map<string, Value> data = 2;
}

message Value {
  oneof kind { 
    bytes b       = 1;
    string s      = 2;
    float f       = 5;
  }
}

service PipelineService {
  rpc Process(Envelope) returns (Envelope);
}
```

### src/my_service.py

```python
import sys
sys.path.append('./protos')
import pipeline_pb2 as pb2
import pipeline_pb2_grpc as pb2_grpc
from aux import wrap_value, unwrap_value

_PORT_DEFAULT = 8061

class MyService(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        # Initialize your service (load models, etc.)
        self.model = load_my_model()
    
    def Process(self, request, context):
        try:
            box_cfg = json.loads(request.config_json).get("my_service", {})
            parameters = box_cfg.get("parameters", {}) or {}
            if box_cfg.get("command") == "reset":          # accept on every box
                return pb2.Envelope(config_json=json.dumps(
                    {"my_service": {"status": "done", "action": "reset"}}))
            if "images" not in request.data:
                return pb2.Envelope(config_json=json.dumps(
                    {"my_service": {"status": "empty_request"}}))
            images = unwrap_value(request.data["images"])

            results = self.run_inference(images, parameters)
            
            return pb2.Envelope(
                config_json=json.dumps({"status": "success"}),
                data={"results": wrap_value(results)}
            )
        except Exception as e:
            logging.error(f"Processing failed: {e}")
            return pb2.Envelope()

# Add server setup (see full template below)
```

### docker/Dockerfile

```dockerfile
ARG WORKSPACE=/workspace
ARG SERVICE_NAME=my_service

FROM python:3.10-slim AS builder
RUN pip install grpcio grpcio-tools protobuf
COPY protos /workspace/
WORKDIR /workspace
RUN python -m grpc_tools.protoc --python_out=. --grpc_python_out=. my_service.proto

FROM python:3.10-slim
ARG USER=runner
RUN addgroup --system runner-group && \
    adduser --system --ingroup runner-group runner && \
    mkdir /workspace && chown runner:runner /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=builder /workspace/*.py /workspace/
COPY src/my_service.py /workspace/service.py
COPY protos/pipeline.proto /

USER runner
EXPOSE 8061
CMD ["python", "/workspace/service.py"]
```

### requirements.txt

```txt
grpcio
protobuf
grpcio-reflection
grpcio-status
numpy
# Add your dependencies here
```

---

## Template 2: CUDA/GPU Service

**Use when:** PyTorch/TensorFlow with GPU acceleration needed

### docker/Dockerfile (GPU version)

```dockerfile
ARG WORKSPACE=/workspace
ARG SERVICE_NAME=my_service

FROM python:3.10-slim AS builder
RUN pip install grpcio grpcio-tools protobuf
COPY protos /workspace/
WORKDIR /workspace
RUN python -m grpc_tools.protoc --python_out=. --grpc_python_out=. my_service.proto

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-base-ubuntu22.04
ARG USER=runner
RUN addgroup --system runner-group && \
    adduser --system --ingroup runner-group runner && \
    mkdir /workspace && chown runner:runner /workspace

COPY requirements.txt .
RUN apt update -y && \
    apt install -y pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip -r requirements.txt

COPY --from=builder /workspace/*.py /workspace/
COPY src/my_service.py /workspace/service.py
COPY protos/pipeline.proto /

USER runner
EXPOSE 8061

# Disable idle timeout GPU management (optional)
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "/workspace/service.py"]
```

### Service Code with GPU Management

```python
import threading
import time
import torch

IDLE_TIMEOUT = 60  # seconds

class MyService(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self._model = load_model_to_cpu()
        self._device = "cpu"
        self._last_request_time = time.time()
        
        # Watchdog to move back to CPU when idle
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
    
    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._lock:
                idle = time.time() - self._last_request_time
                if idle > IDLE_TIMEOUT and self._device.startswith("cuda"):
                    self._model.cpu()
                    torch.cuda.empty_cache()
                    self._device = "cpu"
    
    def Process(self, request, context):
        with self._lock:
            self._last_request_time = time.time()
            
            # Move to GPU if available
            if self._device == "cpu" and torch.cuda.is_available():
                self._model.cuda()
                self._device = "cuda"
        
        return self.run_inference(request)

    def set_device(self, target):
        with self._lock:
            self._last_request_time = time.time()
            target = target.lower()
            
            if target.startswith("cuda") and not torch.cuda.is_available():
                return self._device
            
            if target == self._device:
                return self._device
            
            # Reload model on new device
            new_model = load_model_to(target)
            del self._model
            torch.cuda.empty_cache()
            self._model = new_model
            self._device = target
            return self._device
```

---

## Template 3: YOLO-Specific Service

**Use when:** Ultralytics YOLO models (detection, tracking)

### docker/Dockerfile (YOLO variant)

```dockerfile
FROM ultralytics/ultralytics:latest

ARG USER=runner
ARG WORKSPACE=/workspace

RUN addgroup --system runner-group && \
    adduser --system --ingroup runner-group runner && \
    mkdir /workspace && chown runner:runner /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy service code
COPY src/yolo_service.py /workspace/service.py

USER runner
EXPOSE 8061

CMD ["python", "/workspace/service.py"]
```

### Service Code (YOLO)

```python
from ultralytics import YOLO
import cv2
import numpy as np
import json

class PipelineService(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    # the shared contract: ONE `Process` RPC, dispatched on `command`
    def Process(self, request, context):
        section = (json.loads(request.config_json or "{}") or {}).get("yolo", {})
        command = section.get("command", "detect")
        if command == "reset":
            return self._status({"status": "done", "action": "reset"})
        if "images" not in request.data:
            return self._status({"status": "empty_request"})
        images = unwrap_value(request.data["images"])
        params = section.get("parameters", {}) or {}

        annotated = []
        dets = []
        for i, img_bytes in enumerate(images):
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                              cv2.IMREAD_COLOR)[..., (2, 1, 0)]
            if command == "track":
                results = self.model.track(source=img, persist=True, **params)
            else:
                results = self.model(img, **params)
            annotated.append(cv2.imencode('.jpg',
                results[0].plot(img=np.ascontiguousarray(results[0].orig_img)))[1].tobytes())
            dets.append({"image_index": i, "boxes": len(results[0].boxes),
                         **({"track_id": [int(b.id) for b in results[0].boxes]}
                            if command == "track" else {})})

        return pb2.Envelope(
            data={"images": wrap_value(annotated),
                  "detections": wrap_value(json.dumps(dets).encode())},
            config_json=json.dumps({"yolo": {
                "status": "done", "command": command,
                "num_images": len(images),
                "encoding": {"images": "identity", "detections": "json"}}}))

    @staticmethod
    def _status(section):
        return pb2.Envelope(config_json=json.dumps({"yolo": section}))
```

---

## Helper Functions: aux.py

Every box vendors **the same** `aux.py` — the canonical copy lives at
[`protos/aux.py`](../protos/aux.py) in the repo root (each box keeps its own
copy in `protos/`, copied in at build time). It coerces Python objects to the
`Value` oneof and back:

```python
import pipeline_pb2

def wrap_value(obj):
    """Wrap a Python object into a pipeline.Value"""
    if isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)

    elif isinstance(obj, list):
        if all(isinstance(v, float) for v in obj):
            return pipeline_pb2.Value(ff=pipeline_pb2.FloatList(values=obj))
        elif all(isinstance(v, str) for v in obj):
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=obj))
        elif all(isinstance(v, (bytes, bytearray)) for v in obj):
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=obj))
    raise TypeError(f"Cannot wrap object of type {type(obj)}: {obj}")


def unwrap_value(val: pipeline_pb2.Value):
    """Unwrap a pipeline.Value into a plain Python object"""
    kind = val.WhichOneof("kind")
    if kind == "f":   return val.f
    if kind == "s":   return val.s
    if kind == "b":   return val.b
    if kind == "ff":  return list(val.ff.values)
    if kind == "ss":  return list(val.ss.values)
    if kind == "bb":  return list(val.bb.values)
    return None
```

---

## Complete Service Template (Standard)

```python
import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import json
import logging
import os
import pickle
import time
import zstandard as zstd

import sys
sys.path.append('./protos')
import pipeline_pb2 as pb2
import pipeline_pb2_grpc as pb2_grpc
from aux import wrap_value, unwrap_value

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class MyService(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        # Load models here
        self.model = load_my_model()
    
    def Process(self, request, context):
        # Implement your logic
        try:
            box_cfg = json.loads(request.config_json).get("my_service", {})
            parameters = box_cfg.get("parameters", {}) or {}
            if box_cfg.get("command") == "reset":
                return pb2.Envelope(config_json=json.dumps(
                    {"my_service": {"status": "done", "action": "reset"}}))
            if "images" not in request.data:
                return pb2.Envelope(config_json=json.dumps(
                    {"my_service": {"status": "empty_request"}}))
            images = unwrap_value(request.data["images"])

            out_list = self.run_inference(images, parameters)

            # Heavy results convention: zstd(pickle(list))
            return pb2.Envelope(
                config_json=json.dumps({"my_service": {"status": "done",
                                                       "num_images": len(images)}}),
                data={"results": wrap_value(
                    zstd.ZstdCompressor().compress(pickle.dumps(out_list)))}
            )
        except Exception as e:
            logging.error(f"Error: {e}")
            return pb2.Envelope(config_json=json.dumps(
                {"my_service": {"status": "error", "error": str(e)}}))

def run_server(server):
    port = int(os.getenv('PORT', _PORT_DEFAULT))
    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'Server started at {target}')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1)
        ]
    )
    
    pb2_grpc.add_PipelineServiceServicer_to_server(MyService(), server)
    
    service_names = (
        pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)
    
    run_server(server)
```

---

## Build & Deploy

### Build the image:

```bash
cd /home/manuelf/boxes/images/my_service
docker build -t myregistry/my_service:latest -f docker/Dockerfile .
```

### Push to registry (optional):

```bash
docker push myregistry/my_service:latest
```

### Use it from the caller:

The box is addressable by `ip:port` — `visionist_client` dials it directly:

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")
res = b.run(
    data   = {"images": [pathlib.Path("test.jpg")]},
    config = {"my_service": {"command": "process", "parameters": {}}},
)
print(res.config)    # {"my_service": {"status": "done", …}}
```

---

## Service Checklist

Before building:
- [ ] Proto file defines all required methods
- [ ] `aux.py` has wrap_value/unwrap_value helpers
- [ ] Dockerfile uses multi-stage build (builder + runtime)
- [ ] Non-root user created for security
- [ ] Port 8061 exposed
- [ ] Requirements include grpcio and protobuf

Before deploying:
- [ ] Service tested standalone (`python my_service.py`)
- [ ] gRPC reflection enabled
- [ ] Large message sizes configured (options with -1)
- [ ] GPU memory management if using CUDA

---

## Publishing to GitHub Container Registry (GHCR)

The fleet's images are published from this repo by the
`publish boxes` workflow (`.github/workflows/publish-boxes.yml`), not by hand.
Local `docker build` is for development; the registry is the source of truth
for `docker pull`.

### Publish

```bash
git tag visionist-v0.1.0
git push origin visionist-v0.1.0
```

That builds and pushes every publishable box to
`ghcr.io/<owner>/<box>:0.1.0` **and** `:latest` (public repo ⇒ free storage).
You can also trigger the workflow from Actions with an arbitrary version string
(`0.1.0-dev1`) without creating a tag.

### Reproducible builds

Every Dockerfile's `FROM` base is **pinned by digest**
(`…@sha256:…`), so a published tag produces the same image no matter when or
where it is rebuilt. When you bump a base image, update the digest in the
Dockerfile **and** commit that change so the published tag and the source
agree.

### Pulling / pinning a fleet

```bash
docker pull ghcr.io/<owner>/tapnext_tracker:0.1.0
docker run --rm --gpus all -p 8063:8061 ghcr.io/<owner>/tapnext_tracker:0.1.0
```

For a bit-for-bit reproducible fleet, pin the **digest** instead of the tag —
the `report digests` job prints a ready block of `ghcr.io/<owner>/<box>@sha256:…`
lines you can drop into `fleet/docker-compose.yml`.

### What is (and isn't) published

- Published: `clip`, `tapnext_tracker`, `lang_segm`, `textembedding`,
  `opencv_box`, `yolo`, `vggt`, `moge_box` (the 8 in the workflow matrix).
- `vggt` is the heaviest build: its Dockerfile clones
  `facebookresearch/vggt` and downloads the ~5 GB `VGGT-1B` checkpoint at
  build time (both were previously missing from the repo; the build is now
  self-contained).
- Removed: `folder_wd` and `gradio_display` were retired and deleted.

### Costs

- Public repo ⇒ GHCR storage and bandwidth are free; pulling needs no login.
- Builds run on public runners (2 concurrent). A full publish is ~15-30 min and
  consumes part of the free monthly runner-minute quota — run it per release,
  not per commit.
- No shared build cache is used (the GHA cache quota is far too small for the
  8 GB CUDA base layers); that is why a publish rebuilds from scratch.

