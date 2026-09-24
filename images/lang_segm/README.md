# LangSAM Box (language-guided segmentation)

A gRPC box that runs **LangSAM** (Language Segment Anything — SAM 2.1
`sam2.1_hiera_small`) to segment image regions described by a natural-language
prompt.

The box speaks the shared **envelope** interface, so it is addressable through
`visionist_client` exactly like tapnext, clip, and sbert:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

## Directory structure

```
lang_segm/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── lang_sam_service.py    # PipelineService.Process(Envelope)
├── test/
│   └── test_lang_sam.py       # standalone envelope smoke test
└── requirements.txt
```

## Build

The Dockerfile clones `lang-segment-anything`
([luca-medeiros/lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything))
at build time, so just build from the box root:

```bash
docker build --tag sipgisr/lang_sam -f docker/Dockerfile .
```

The model checkpoints (SAM 2.1 + Hiera encoder + prompter) are downloaded
lazily on first use and cached under `HF_HOME` / `TORCH_HOME`
(`/workspace/.cache` inside the container).

## Run

```bash
# GPU
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host sipgisr/lang_sam
```

The box loads the model **on CPU at startup** and moves it lazily to
whichever device the request asks for (or to `cuda` automatically); after
60 s idle, a watchdog re-instantiates the model back on CPU to free GPU
memory.

## Service usage

### Request

- `data["images"]` — list of image bytes (JPEG/PNG).
- `config["lang_sam"]["command"]` — `"segment"` (default) or `"reset"`
  (no-op; the box is stateless, but `reset_first` from `visionist_client`
  stays safe).
- `config["lang_sam"]["parameters"]` — optional:
  - `device` — explicit target, e.g. `"cpu"` or `"cuda:0"`. When omitted
    (the default), the box runs on `"cuda"` if a GPU is visible, else `"cpu"`.
  - `box_threshold` — Grounding-DINO box threshold (default `0.3`).
  - `text_threshold` — Grounding-DINO text threshold (default `0.25`).
- `config["lang_sam"]["text_prompt"]` — list of prompt strings, e.g.
  `["an excavator", "the wood pile"]`. They are joined into one prompt,
  which is applied to **every** image (LangSAM pairs one prompt with each
  image 1:1).

Accepted config aliases: the section may also be named `lang_segm` or
`aispgradio` (legacy pipeline format), or given in the flat top-level
format `{"parameters": {...}, "text_prompt": [...]}`.

```json
{
  "lang_sam": {
    "command": "segment",
    "parameters": {"device": "cuda:0"},
    "text_prompt": ["an excavator", "the wood pile"]
  }
}
```

### Response

`config_json` carries `{"lang_sam": {"status", "runtime", "num_images",
"num_prompts"}}` and `data` carries one field:

| field     | type                                    | description |
|-----------|-----------------------------------------|-------------|
| `results` | `zstd.compress(pickle.dumps(list))`     | one `LangSAM.predict()` output dict per input image (masks, bboxes, scores, …) |

The response config declares the encoding — `"encoding": "zstd_pickle"` —
under the `lang_sam` section, so `visionist_client` decodes it for you (the
client never guesses). With raw bytes (without the client, or with
an old client/image that predates the declaration) use the generic
`zstd_pickle` contract:

```python
import pickle, zstandard as zstd
out_list = pickle.loads(zstd.ZstdDecompressor().decompress(bytes_blob))
# out_list[0]["masks"][0] -> np.ndarray (H, W) bool
```

Statuses reported in `config["lang_sam"]["status"]`:

| status          | meaning |
|-----------------|---------|
| `done`          | inference finished; `data["results"]` present |
| `done` + `action: reset` | `reset` command acknowledged (no-op) |
| `empty_request` | images field empty |
| `error`         | `config` missing/invalid, no `text_prompt`, or inference failure — see `error` field |

Image-less envelopes (config only) are echoed back unchanged — some pipeline
stages forward bare envelopes and that behavior is preserved.

## Call with visionist_client

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")
res = b.run(
    data={"images": ["car.jpg"]},                     # local paths -> bytes
    config={"lang_sam": {
        "command": "segment",
        "parameters": {"device": "cpu"},
        "text_prompt": ["a car", "the road"],
    }},
)
print(res.config)    # {"lang_sam": {"status": "done", "encoding": "zstd_pickle", ...}}
print(res.encoding)  # "zstd_pickle" — the box's declared payload encoding

# The client decodes `results` via the declared codec: it's a plain list.
out_list = res.results
print([len(o["masks"]) for o in out_list])

# (Manual unwrap — only if talking to an image that predates the
#  encoding declaration, or without visionist_client:)
# import pickle, zstandard as zstd
# out_list = pickle.loads(zstd.ZstdDecompressor().decompress(res.results))
```

Or use the vendored protos directly:

```python
import grpc, json
import pipeline_pb2, pipeline_pb2_grpc, aux

channel = grpc.insecure_channel("localhost:8061")
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

req = pipeline_pb2.Envelope(
    config_json=json.dumps({"lang_sam": {
        "command": "segment",
        "text_prompt": ["a square"],
    }}),
    data={"images": aux.wrap_value([open("test.jpg", "rb").read()])},
)
resp = stub.Process(req)
print(json.loads(resp.config_json))
```

## Test

`Envelope` with matching prompts, calls `Process`, decodes `results` and
prints the per-image mask shapes, area fractions and scores. It also
**paints each mask area in its own color** over the original photo and saves
annotated PNGs in the test folder (plus an RGB legend per mask), so the
segmentation can be checked visually: `output_dog_langsam.png`,
`output_car_langsam.png`.

Expected: at least one mask on the dog image (prompt includes "a dog") and
ideally one on the car image ("a race car"); both prompts are joined into a
single LangSAM prompt, so some overlap/noise is normal.

```bash
# server already up on :8061
python test/test_lang_sam.py
# or a remote box:
BOX_HOST=10.0.0.5:8061 python test/test_lang_sam.py
# then open test/output_dog_langsam.png / test/output_car_langsam.png
```
