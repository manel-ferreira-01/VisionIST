# CLIP Box (text + image embeddings)

A gRPC box that encodes images and/or text with CLIP and returns the image
embedding, the text embedding, and the cross-modal similarity logits.

The box speaks the shared **envelope** interface, so it is addressable through
`visionist_client` exactly like tapnext:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

## Directory structure

```
clip/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── clip_service.py        # PipelineService.Process(Envelope)
├── test/
│   └── test_clip.py           # in-process / remote smoke test (dummy images)
├── requirements.txt
└── README.md
```

## Build

```bash
docker build --tag sipgisr/clip --build-arg SERVICE_NAME=clip -f docker/Dockerfile .
```

The first call lazily downloads the `ViT-B/32` weights (see `HF_HOME` /
`download_root`).

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/clip
```

## Service usage

### Request

Two fields in `data`, and a small `config`:

- `data["images"]` -- list of image bytes (JPEG/PNG).
- `data["texts"]` -- list of strings.
- `config["clip"]["command"]` -- `"encode"` (default) or `"reset"` (no-op,
  the box is stateless, but `reset_first` from `visionist_client` stays safe).
- `config["clip"]["parameters"]["model"]` -- accepted for API consistency; the
  box runs the checkpoint loaded at startup (`ViT-B/32`).

### Response

`config_json` carries `{clip:{status, runtime, num_images, num_texts}}` and
`data` carries three `torch.save`-decoded payloads:

| field        | shape                  | description                     |
|--------------|------------------------|---------------------------------|
| `image_emb`  | `[num_images, 512]`    | CLIP image embeddings           |
| `text_emb`   | `[num_texts, 512]`     | CLIP text embeddings            |
| `similarity` | `[num_images, num_texts]` | cross-modal logits          |

## Call with visionist_client

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")
res = b.run(
    data={
        "images": ["car.jpg", "dog.jpg"],        # local paths -> bytes
        "texts": ["a dog", "a cat", "a race car", "grass"],
    },
    config={"clip": {"command": "encode"}},
)
import torch, io
img  = torch.load(io.BytesIO(res.image_emb),    weights_only=False)
txt  = torch.load(io.BytesIO(res.text_emb),     weights_only=False)
sim  = torch.load(io.BytesIO(res.similarity),   weights_only=False)
print(img.shape, txt.shape, sim.shape)
```

Or use the vendored protos directly:

```python
import grpc, json, pipeline_pb2, pipeline_pb2_grpc, aux

channel = grpc.insecure_channel("localhost:8061")
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

req = pipeline_pb2.Envelope(
    config_json=json.dumps({"clip": {"command": "encode"}}),
    data={
        "images": aux.wrap_value([open("car.jpg","rb").read(), open("dog.jpg","rb").read()]),
        "texts":  aux.wrap_value(["a dog", "a cat", "a race car"]),
    },
)
resp = stub.Process(req)
print(aux.unwrap_value(resp.data["similarity"]))  # bytes -> torch.load(...)
```

## Test

A tapnext-style script (no notebook) that builds an `Envelope` with dummy
images and sample texts, calls `Process`, and prints the shape of each returned
tensor:

```bash
# server already running at :8061
python test/test_clip.py
# or
BOX_HOST=10.0.0.5:8061 python test/test_clip.py
```
