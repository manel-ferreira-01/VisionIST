# visionist-webui

> **Status: current** — backend (66 tests, live-verified) and the
> SPA (`web/`: fleet page, def-driven console, pipeline page, all 14
> visualizers) are built, and a session of fixes landed: tab-state isolation,
> video input for tapnext, per-frame track visibility, labeled heatmaps, input
> mosaic. **Pipelines** (several boxes chained server-side) landed with
> `sfm_video` (lightglue + MoGe + sfm), live-verified on `cozinha.mp4`.
> `tsc --noEmit && vite build` is clean; the built `web/dist` is served by
> the FastAPI app. Working state / run-loop / gotchas / next steps:
> [../docs/Webui_Guide.md](../docs/Webui_Guide.md).

A declarative web layer over a fleet of **boxes**: a box-agnostic core,
YAML box definitions, HTTP API, and `visionist_client` under the hood —
plus **pipelines** for tasks that chain several boxes (see
[Pipelines](#pipelines-several-boxes-one-task)).

```
   browser / curl ──HTTP──▶ webui (FastAPI, core/*, boxes_agnostic)
                                  │
                                  ▼  visionist_client.Visionist.run(...)
   box by IP:port (Process Envelope) ── clip · tapnext · lang_sam · sbert · vggt · yolo
```

**Design rule (inherited from `visionist_client`):** the core is *smart about
shape, dumb about content*. Nothing in `src/webui/` names a box. All box
knowledge lives in [`boxes/*.yaml`](boxes/) — the same data-driven contract,
so *adding a box = one YAML file*, never code.

## Scope (status)

| Box | In the webui | Why |
|---|---|---|
| clip | ✅ | standard envelope |
| tapnext | ✅ | standard envelope + multi-session |
| lang_segm | ✅ | standard envelope |
| textEmbedding (sbert) | ✅ | standard envelope |
| vggt | ✅ | standard envelope (legacy flat config, supported via `flat_config`) |
| yolo | ✅ | standard envelope; detection over images and/or a decoded video |
| lightglue | ✅ | standard envelope; `match` / `stream` (session id inside `parameters`: `session.placement: parameters`) |
| sfm | ✅ | standard envelope; `.npy` inputs (mode A tracks + depths + intrinsics, mode B matrices) → `scene` view |
| **opencv_box** | ⏸ **no definition yet** | now a standard envelope box (`match` / `reset` on `Process`) — YAML definition still to be added |

The skip is deliberate: the webui stays **contract-only** (one stub, `Process`,
for every box). opencv_box now speaks the shared envelope (`match` /
`reset` commands); its YAML definition is the only thing
missing — drop it into `boxes/` when wanted, no code changes needed. Defs that
request a non-`Process` `method` are refused with a clear error (`build_call`).

## Quick start

Assumes a running fleet (see [`fleet/`](../fleet/) — `docker compose up -d`)
and Python ≥ 3.10.

```bash
# 1) deps (client first, editable, from the repo)
pip install -e visionist_client
pip install -e webui            # pulls fastapi/uvicorn/pydantic/yaml/...

# 2) seed the fleet (optional but convenient)
mkdir -p data
cat > data/fleet.json <<'EOF'
{"entries": [
  {"id": "clip",        "name": "clip",        "addr": "127.0.0.1:9061", "def_id": "clip"},
  {"id": "sbert",       "name": "sbert",       "addr": "127.0.0.1:9062", "def_id": "sbert"},
  {"id": "tapnext",     "name": "tapnext",     "addr": "127.0.0.1:9063", "def_id": "tapnext"},
  {"id": "lang-sam",    "name": "lang_sam",    "addr": "127.0.0.1:9064", "def_id": "lang_sam"}
]}
EOF

# 3) run
visionist-webui                       # = uvicorn, WEBUI_HOST/PORT (default 127.0.0.1:8080)

# 4) talk to it
curl -s localhost:8080/                        # health + def ids
curl -s localhost:8080/api/defs                # the contract the UI renders from
curl -s -X POST localhost:8080/api/fleet/lang-sam/probe -H 'content-type: application/json' -d '{"timeout": 3}'
```

### Calling a box

```bash
# upload -> reference -> call
TOKEN=$(curl -s -F "file=@dog.jpg" localhost:8080/api/upload | jq -r .ref)

curl -s -X POST localhost:8080/api/call -H 'content-type: application/json' -d "{
  \"fleet_id\": \"lang-sam\",
  \"data\":     {\"images\": [\"$TOKEN\"]},
  \"section\":  {\"text_prompt\": [\"a dog\", \"the wood pile\"]},
  \"parameters\": {\"box_threshold\": 0.4}
}"
```

Response shape (JSON): `status` / `error` / `runtime` from the box's response
config, `fields` as JSON-safe values (arrays inline, heavy payloads as
`/api/file/<token>` artifacts), `config_extra` (the box's whole status
section), `declared_encoding` (the codec the box declared, verbatim),
`duration_ms`, `artifacts`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/defs` | all box definitions + widget/visualizer vocabulary |
| GET | `/api/defs/{id}` | one definition |
| GET | `/api/fleet` | fleet entries (+ last probe) |
| POST | `/api/fleet` | add `{name, addr, def_id?, note?}` |
| PATCH | `/api/fleet/{id}` | rename / move / re-point def |
| DELETE | `/api/fleet/{id}` | remove |
| POST | `/api/fleet/{id}/probe?timeout=` | reachability + gRPC reflection (`Visionist.info()`) |
| POST | `/api/upload` | multipart file → `{"ref": "@upl_…"}` |
| GET | `/api/file/{token}` | fetch any artifact (upload or result payload) |
| POST | `/api/call` | `{fleet_id, data, parameters, section, command, action, session_id}` |
| GET | `/api/pipelines` | pipeline definitions (+ `missing`: used boxes without a fleet entry) |
| POST | `/api/pipelines/{id}/run` | `{data, parameters, timeout?}` → `202` job snapshot (`id`, `status`, …) |
| GET | `/api/pipelines/jobs/{job}` | job snapshot: `status`, `steps` (calls / total / duration), `fields` (emitted so far), `info`, `error` |
| POST | `/api/pipelines/jobs/{job}/cancel` | stop at the next step / box-call boundary |

### Wire rules (where client-coercion foot-guns become UI policy)

* **files** travel as `"@<token>"` refs (uploaded first) → `bytes` in the
  envelope;
* **bare strings in `data` are always literals** (the client's `s` kind) —
  the definition's `widget`/`kind` decide what a field *is*, so the caller
  never guesses;
* the request is validated **against the box definition** — unknown
  parameters, section keys, data fields, commands, or actions are 400 with
  `known: [...]`;
* `GET /api/file/{token}` honours **`Range` → `206 Partial Content`**
  (`Accept-Ranges: bytes` is advertised), so HTML5 `<video>` seeks work
  without re-downloading the artifact;
* **reset semantics**: `command: reset` *is* the reset (no `reset_first`);
  stateful boxes (tapnext) are **never** auto-reset — that would kill a live
  sequence; stateless boxes get the client's safe no-op `reset_first`.

## Box definitions (the contract)

One YAML per box — form layout in, visualizers out. Full field reference:
[`src/webui/core/schema.py`](src/webui/core/schema.py) (it's small and has
docstrings). Vocabulary:

* **widgets** — `image_upload · video_frames · file_upload · tags ·
  text_repeat · slider · select · number · json`
* **visualizers** — `json (fallback) · table · image_grid · overlay
  (box/mask/point/flow layers) · matrix · tensor · field_map · glb · video · points
  · scene · tracks_player · download`
  (`scene` is a reconstruction: points (+ colors), camera `[R|t]` frustums +
  path, optional dense cloud colorable by frame; `params` name the keys, read
  from the field when it is a dict, else from the response's top-level fields)
  (`points` renders point clouds straight from typed arrays with three.js
  `THREE.Points` — no GLB encoding; `glb` is for real glTF binaries like
  the vggt scene)
* **result field `"*"`** — wildcard fallback, so the UI can never get stuck
  on a field a definition forgot (the "client always returns something" rule).

The per-box **README stays authoritative**; definitions carry `docs:` links
to them, and `note:` fields record where the webui's view might lag.

### Result serialization rules

* small JSON-able values (scalars; arrays ≤ 65 536 elements) → **inline**
* numeric arrays/masks beyond that (bool masks, float32 tensors) → **buffer
  artifact** with `dtype`/`shape` (the SPA can draw them as typed arrays)
* opaque bytes (GLB, images…) → **file artifact** with sniffed MIME
  (`glTF` → `model/gltf-binary`, JPEG/PNG/MP4…)
* exotic objects → **pickle artifact + note** (never a 500)

## Pipelines (several boxes, one task)

A box console is one `Process` call. Tasks like *SfM from a video* are many
calls with Python in between (lightglue tracks → MoGe depth → sfm). Those are
**pipeline modules** in [`pipelines/`](pipelines/) — the per-task
counterpart of a box YAML (the core still names no box):

```python
DEF = {                      # PipelineDef: same widgets / visualizers as a box def
    "id": "sfm_video", "name": "…",
    "uses": ["lightglue", "moge", "sfm"],     # box def ids, resolved through the fleet
    "inputs": [...], "parameters": [...], "results": [...],
}

def run(ctx, data, params):  # data: uploads resolved to bytes; params: defaults filled
    with ctx.step("moge · depth per frame", total=len(frames)):   # progress = box calls
        moge = ctx.box("moge")                 # a visionist_client.Visionist (counted, cancellable)
        ...
        ctx.emit("depths", maps)               # a result field, shown as soon as it exists
    ctx.info(tracks=1691)                      # small facts for the result header
```

* a run is a **job** on a small thread pool (`WEBUI_PIPELINE_WORKERS`, default 2);
  the SPA polls `GET /api/pipelines/jobs/{job}` and renders each result
  block when its field is emitted (same visualizers as the consoles);
* a failing box / exception marks its step `error` (message + traceback in
  the job); fields emitted before stay; **cancel** stops at the next step or
  box call;
* `run` is refused (400, `missing: [...]`) while a used box has no fleet entry;
* modules load at startup, fail-fast like the YAMLs (`DEF` validated, `uses`
  must be known box defs); `_*.py` files are skipped (helpers);
* jobs and emitted artifacts are in memory (artifact TTL applies).

Shipped: [`pipelines/sfm_video.py`](pipelines/sfm_video.py) — frames (a video
sampled in the browser: frame count + time range) → lightglue
`track_stream` → MoGe per frame → sfm (`linear` by default) → a `scene` with
the tracks, camera frustums and every depth map corrected with `Z = d·λ + o`
and back-projected (if the calibration is right, the frames' clouds overlap),
a per-frame table and depth previews. The webui port of the sfm cell of
`notebooks/boxes_walkthrough.ipynb`; ~20 s for 20 frames.

## Tests

```bash
cd webui
python -m pytest tests/ -q          # 66 tests: registry, caller (pure), API e2e, pipelines
```

E2E tests spin up real fake boxes over gRPC (the `fake_box_smoke.py` pattern)
and drive them through the full HTTP → core → `visionist_client` → box →
serialize path — including a box that answers `status: error` in-band.
`test_pipelines.py` runs the shipped `sfm_video` against fake lightglue /
moge / sfm boxes (steps, call counts, emitted fields, a failing box, cancel).
Live browser check of the pipeline page (real fleet, dev server on 8090):
`node web/.sfm_pipeline_e2e.cjs <dir of frame JPEGs>`.

## Layout

```
webui/
├── boxes/                    # ← the only per-box knowledge in the whole webui
│   ├── clip.yaml  lang_sam.yaml  lightglue.yaml  moge.yaml  sbert.yaml
│   ├── sfm.yaml  tapnext.yaml  unimatch.yaml  vggt.yaml  yolo.yaml
├── pipelines/                # ← per-task glue over several boxes (DEF + run)
│   └── sfm_video.py
├── src/webui/
│   ├── app.py                # FastAPI factory (env-driven)
│   ├── config.py             # WEBUI_* env, defaults
│   ├── core/                 # box-agnostic core
│   │   ├── schema.py         # Pydantic models + widget/visualizer vocabulary
│   │   ├── registry.py       # load+validate boxes/*.yaml (fail fast)
│   │   ├── caller.py         # build_call() pure / execute() wire (the only gRPC)
│   │   ├── serialize.py      # Result -> JSON + artifacts (never raises)
│   │   ├── artifact.py       # token store (TTL + cap), uploads & heavy fields
│   │   ├── fleet.py          # fleet.json CRUD + Visionist.info() probes
│   │   └── pipeline.py       # pipeline loading, RunContext, JobRunner
│   └── api/                  # FastAPI routes: defs / fleet / call / pipelines
└── tests/                    # registry · caller · API e2e (fake boxes, real gRPC)
```

## Environment

| Var | Default | Meaning |
|---|---|---|
| `WEBUI_HOST` / `WEBUI_PORT` | `127.0.0.1` / `8080` | bind |
| `WEBUI_DATA_DIR` | `./data` | `fleet.json` lives here |
| `WEBUI_BOXES_DIR` | `webui/boxes` | definitions dir |
| `WEBUI_PIPELINES_DIR` | `webui/pipelines` | pipeline modules (missing dir = no pipelines) |
| `WEBUI_PIPELINE_WORKERS` | `2` | concurrent pipeline jobs |
| `WEBUI_ARTIFACT_TTL` | `3600` s | token lifetime (uploads + result artifacts) |
| `WEBUI_MAX_ARTIFACT_BYTES` | `2000000000` | store cap (FIFO eviction) |
| `WEBUI_MAX_UPLOAD_BYTES` | `67108864` | per-upload cap |

Notes: tokens are **capability strings** (like tapnext's `session_id`) —
trusted-LAN tooling, not public multi-tenant infrastructure. Artifacts are
in-memory by design; `fleet.json` is the only durable state.

## Roadmap

1. **done** — core + definitions + HTTP API + tests (this scaffold)
2. **done** — SPA (`web/`): fleet dashboard (probe dots, add/delete/probe,
   console links), def-driven console (action/command/params/section/inputs/
   session + call history with re-render)
3. **done** — visualizers: `overlay` (lang_sam mask+boxes), `tracks_player`
   (tapnext, steps assembled from call history), `matrix` (clip/sbert),
   `tensor`, `glb` (vggt, three.js) + `image_grid`/`table`/`json`/`download`
4. **next** — polish: vggt camera auto-fit on a real reconstruction
   (needs a live vggt box), side-by-side prompts on lang_sam
5. **done** — standard `yolo` box (image + video detection) added to the webui
   via [`boxes/yolo.yaml`](boxes/yolo.yaml) → `video` (annotated_video) +
   `image_grid` (annotated, image input only — gated by `only_if_missing:`)
   + `table` (per-frame detections); for video input the mp4 is the view and the
   per-frame JPEGs surface via the artifacts list
