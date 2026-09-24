# SBERT Box (text embeddings)

A gRPC box that encodes a list of sentences with sentence-transformers
(`all-MiniLM-L6-v2` by default) and returns the embeddings plus the
sentence-similarity matrix.

The box speaks the shared **envelope** interface, so it is addressable through
`visionist_client` exactly like tapnext / clip:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

## Directory structure

```
textEmbedding/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── sbert_service.py       # PipelineService.Process(Envelope)
├── test/
│   └── test_sbert.py          # in-process / remote smoke test
├── requirements.txt
└── README.md
```

## Build

```bash
docker build --tag sipgisr/textembedding --build-arg SERVICE_NAME=sbert -f docker/Dockerfile .
```

The first call lazily downloads the `all-MiniLM-L6-v2` weights (see `HF_HOME` /
`HF_HUB_CACHE`).

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/textembedding
```

## Service usage

### Request

One field in `data`, and a small `config`:

- `data["texts"]` -- list of strings.
- `config["sbert"]["command"]` -- `"encode"` (default) or `"reset"` (no-op,
  the box is stateless, but `reset_first` from `visionist_client` stays safe).

### Response

`config_json` carries `{sbert:{status, runtime, num_texts}}` and `data` carries
two `torch.save`-decoded payloads:

| field          | shape                  | description                  |
|----------------|------------------------|------------------------------|
| `embeddings`   | `[num_texts, D]`       | sentence embeddings          |
| `similarities` | `[num_texts, num_texts]` | pairwise cosine similarity |

## Call with visionist_client

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")
res = b.run(
    data={
        "texts": [
            "o manel comeu um gelado.",
            "o carro anda rapido",
            "a bea bebeu agua muito fria",
        ],
    },
    config={"sbert": {"command": "encode"}},
)
import torch, io
emb = torch.load(io.BytesIO(res.embeddings),    weights_only=False)
sim = torch.load(io.BytesIO(res.similarities),  weights_only=False)
print(emb.shape, sim.shape)
```

Or use the vendored protos directly:

```python
import grpc, json, pipeline_pb2, pipeline_pb2_grpc, aux

channel = grpc.insecure_channel("localhost:8061")
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

req = pipeline_pb2.Envelope(
    config_json=json.dumps({"sbert": {"command": "encode"}}),
    data={
        "texts": aux.wrap_value([
            "o manel comeu um gelado.",
            "o carro anda rapido",
        ]),
    },
)
resp = stub.Process(req)
print(aux.unwrap_value(resp.data["embeddings"]))  # bytes -> torch.load(...)
```

## Test

A tapnext/clip-style script (no notebook) that builds an `Envelope` with sample
sentences, calls `Process`, and prints the shape of each returned tensor plus
the similarity matrix:

```bash
# server already running at :8061
python test/test_sbert.py
# or
BOX_HOST=10.0.0.5:8061 python test/test_sbert.py
```
