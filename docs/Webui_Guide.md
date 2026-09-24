# visionist-webui — guide & working state

> The declarative web layer over the fleet: a box-agnostic core, YAML box
> definitions, a FastAPI HTTP API, and a def-driven SPA.
> **This replaces `webui/STATUS.md`** (retired; see `webui/README.md` for the
> feature documentation — this file is the *working-state/session handoff*).

## 1. What it is

```
   browser / curl ──HTTP──▶ webui (FastAPI, box-agnostic core)
                                  │  visionist_client.Visionist.run(...)
   box by IP:port (Process Envelope) ── clip · tapnext · lang_sam · sbert · vggt · yolo
```

**Design rule (inherited from `visionist_client`):** the core is *smart about
shape, dumb about content*. Nothing in `webui/src/webui/` or `webui/web/src/`
names a box. **All box knowledge lives in `webui/boxes/*.yaml`** — adding a
box = one YAML file, never code. The per-box README under `images/` stays the
authoritative request-shape source; defs link to them via `docs:`.

**Scope is contract-only:** every box is served through the one shared
`Process` RPC. `opencv_box` is deliberately excluded (pre-contract second
RPC); re-adding it later = new YAML def only (`build_call` refuses
non-`Process` methods with a clear error until then).

## 2. Current state (verified)

| Layer | State |
|---|---|
| Backend | **52/52 tests green** (`python3 -m pytest tests/ -q` from `webui/`), live-verified against the running fleet (clip, lang_sam, tapnext; error paths 400/502) + an API-level vggt round trip (`fake_vggt`: namespaced call, torch tensors with full shape, GLB served as `model/gltf-binary`) + a live MoGe round trip through the `points` visualizer (701k reprojected points, photo-colored, no page errors — `webui/web/.moge_points_e2e.cjs`). The standard `yolo` box is covered by `boxes/yolo.yaml` + `test_yolo_detection_def` (registry) — a defs-only addition, no code. |
| Frontend | `tsc --noEmit && vite build` clean; `web/dist` auto-mounted by the FastAPI app (API + `/docs` keep priority) |
| Session fixes applied | ✅ tab-switch state leakage (console now remounts per def), ✅ video input for tapnext (`video_frames` widget), ✅ tapnext tracks `(y,x)` order corrected + per-frame visibility toggle, ✅ labeled/legend heatmaps (clip), ✅ input mosaic |

Still **unverified in-browser / live**: vggt GLB orbit + tensor cards
(the API path is covered by `fake_vggt`, but camera auto-fit still needs
a real reconstruction), pixel-level pass of `overlay`, history click-through,
and a first human pass of yolo's `video` player (API + def + serialization
verified end-to-end incl. a real 1920×1080 annotated mp4).
opencv_box intentionally out of scope (pre-contract second RPC); the new
standard `yolo` box covers object detection via `boxes/yolo.yaml`.

## 3. Run / build / test loop

```bash
# deps (client first, editable)
pip install -e visionist_client && pip install -e webui

# backend tests
cd webui && python3 -m pytest tests/ -q            # 52 passed in ~4 s

# frontend: typecheck + build (dist/ is served by the app)
cd webui/web && npx tsc --noEmit && npx vite build  # warn: three.js >500 kB chunk (cosmetic)

# run (SPA served from web/dist; root "/" = API health JSON, open /index.html#/…)
WEBUI_DATA_DIR=$PWD/data WEBUI_PORT=8090 \
  python3 -m uvicorn webui.app:factory --host 127.0.0.1 --port 8090
```

Fleet seed (local docker fleet): `clip 9061 · sbert 9062 · tapnext 9063 ·
lang_sam 9064 · vggt 9066 · moge 9067` (see `webui/data/fleet.json`).

## 4. Layout

```
webui/
├── boxes/                    # ← the ONLY per-box knowledge in the whole webui
│   ├── clip.yaml  lang_sam.yaml  sbert.yaml  tapnext.yaml  vggt.yaml
├── src/webui/
│   ├── app.py                # FastAPI factory (env-driven; mounts web/dist)
│   ├── config.py             # WEBUI_* env, defaults
│   ├── core/                 # box-agnostic
│   │   ├── schema.py         # Pydantic models + widget/visualizer vocab
│   │   ├── registry.py       # load+validate boxes/*.yaml (fail fast)
│   │   ├── caller.py         # build_call() pure / execute() wire (only gRPC)
│   │   ├── serialize.py      # Result -> JSON + artifacts (never raises)
│   │   ├── artifact.py       # token store (TTL + cap)
│   │   └── fleet.py          # fleet.json CRUD + Visionist.info() probes
│   └── api/                  # routes: defs / fleet / upload / file / call
├── web/                      # SPA (React + TS + Vite)
│   └── src/
│       ├── App.tsx           # hash router + topbar (fleet + tab per def)
│       ├── pages/            # FleetPage (probe dots, add/delete) + ConsolePage
│       ├── form/widgets.tsx  # def-driven form widgets
│       ├── viz/              # json/table/image_grid/overlay/matrix/tensor/
│       │                     #   glb/tracks_player/download
│       ├── resolvers.ts      # inline/buffer/file artifact → typed values
│       └── api.ts            # REST client + wire types
└── tests/                    # registry · caller · API e2e (fake boxes, real gRPC)
```

## 5. The contract the UI renders from

* **widgets** — `image_upload · video_frames · file_upload · tags ·
  text_repeat · slider · select · number · json`
* **visualizers** — `json (fallback) · table · image_grid · overlay
  (box/mask/point/flow) · matrix · tensor · field_map · glb · video ·
  points · tracks_player · download`
* result field `"*"` = wildcard fallback, so the UI can never get stuck on a
  field a definition forgot.

Def-driven extras (generic, no box names in code):

* **`video_frames` widget** — video file (mp4/webm/mov) or images. Video →
  N evenly-spaced frames (8–128, user-selected) extracted in the *browser*
  (seek+canvas, ≤960 px, JPEG), each uploaded as an image `"@token"`; wire
  shape is the same image list, so boxes never know a video happened. Used
  by tapnext (the box tracks the whole list in one call/session).
* **`matrix` result `params`** — `{row_labels: <input field>, col_labels:
  <input field>}` names the form fields that label the axes. Labels are
  snapshotted **per call** into call history (so history re-renders stay
  correct); image fields label rows as `img 1..n`. Heatmap: viridis color
  scale + min→max legend, axis labels (rotated when wide, all kept),
  hover readout `row × col → value`. Used by clip.
* **`tracks_player`** — steps assembled from the console's call history:
  one step per uploaded frame, points from `tracks (F, T, 2)` **as (y, x) —
  the box contract** (webui flips to (x,y) for drawing), visibility from
  `visibles` **per frame** — points flagged invisible are dropped (and
  trails into them stop). Session quick-actions (`reset`/`list`) available
  from the def's `session.actions`.
* **`select` params with no `default`** — render their def `placeholder`
  text as an extra first option (value `""`), so the form honestly shows
  "unset/auto" instead of silently displaying the first value. Unset → the
  parameter is **omitted** from the wire (the box's auto/default applies).
  Don't fake an auto mode by sending a magic value — the box treats a
  missing key as *use the default*.
* **Conditional result block** — `only_if_missing: <field>` on a result def
  renders that block **only when the named field is absent from the response**
  (preferred-field fallback, generic: no box names in code). Used by yolo: the
  per-frame `annotated` grid shows only when the box returned no
  `annotated_video` (i.e. image input) — for video input the mp4 stays the
  view and the JPEGs surface via the artifacts list.
* **`input_mosaic: false`** (box def, default `true`) — hides the console's
  "inputs · N images uploaded" mosaic. For defs whose result already shows the
  input, that block is pure duplication: yolo sets it (`annotated` frames are
  the input with boxes drawn); other boxes keep the mosaic, which is the
  only place the sent frames (e.g. `video_frames` extraction) are visible.
* **`video` result def** — a video file artifact (`video/mp4`/…) plays in a
  native `<video controls>` player (no codec work: the browser does it).
  Used by yolo: a video in → `annotated_video` out — the same annotated
  frames the box also returns as per-frame `annotated` JPEGs (see the
  conditional block above), re-encoded as one mp4 at the source video's fps.
  The box encodes **H.264** (PyAV bundles FFmpeg incl.
  libx264) because browsers can't decode `mp4v`; `/api/file` honours
  `Range`/`206` so the player's seeks don't re-download the clip.
* **`points` result def** — per-item point cloud, orbit/zoom, rendered
  directly with three.js `THREE.Points` — a point cloud is typed arrays,
  so there is **no GLB/glTF encoding at all** (no writer, no blobs, no
  binary layout to debug; `glb` is only for real glTF binaries like the
  vggt scene).  Def params pick the position source: `depth` (+ `intrinsics`,
  `projection: normalized` (default, MoGe convention) | `pixel`) or a
  `points` (`(…, 3)`) fallback; `mask` (same grid, keep > 0.5) filters;
  `base: <input field>` names the uploaded images whose per-pixel RGB colors
  the points (fallback: xyz-range colors).  Non-finite/≤ 0 depths dropped.
  Used by moge (every depth pixel back-projected, photo-colored).

## 6. Wire rules & serialization (what the SPA must handle)

* files travel as `"@<token>"` refs (uploaded first) → `bytes` in the
  envelope; bare strings in `data` are always literals;
* request validated **against the def** — unknown params/section/fields/
  commands → 400 with `known: [...]`;
* reset semantics: `command: reset` *is* the reset; stateful boxes are never
  auto-reset; `reset` calls are exempt from required-field validation in
  `build_call` (a stateless box must accept a data-less reset — that's the
  fleet-wide convention, and `reset_first` relies on it);
* serialized values (`core/serialize.py`, never raises — degrades):
  - `{kind:"array", dtype, shape, values}` — inline (≤ 65 536 elements)
  - `{kind:"buffer", url, dtype, shape, size}` — fetch → typed buffer.
    **Bandwidth rule:** the tensor visualizer shows buffers as *metadata
    only* (dtype/shape/size + "download raw"); no preview/stat fetches,
    no matter the size. Inline `{kind:"array"}` values (already in the
    JSON) still get stats + head preview — that costs nothing. Other
    visualizers (overlay, matrix) fetch only what they must render.
  - `{kind:"file", url, mime, size}` — images/GLB/etc.
    (sniffed: `glTF` → `model/gltf-binary`, JPEG/PNG/MP4)
  - everything else plain JSON (exotics → pickle + note).

## 7. Environment gotchas (learned the hard way)

* **Backend changes need a restart** — defs load at startup. Kill by port
  (`ss -ltnp | grep 8090 | grep -oP 'pid=\K[0-9]+' | xargs -r kill`);
  do **not** `pkill -f "uvicorn webui"` (matches your own shell).
* Every frontend change needs `npx tsc --noEmit && npx vite build` (the
  served app is `web/dist`) — and a **hard reload** in the browser to beat
  the old JS bundle.
* `tapnext` wants `data.images` as a **list** (even single frame) — the box
  checks `isinstance`.
* **Grid blowout** — the console's `.grid-2` (form | results) must keep
  `.grid-2 > * { min-width: 0 }`: a `1fr` track never shrinks below an item's
  min-content, and the results table (`nowrap` cells) has a min-content wider
  than the viewport. Without that, the *page* scrolls horizontally; with it,
  the table scrolls inside its own `.tablewrap`. Table cells stay unbroken —
  `overflow-wrap: anywhere` is explicitly excluded for `table.data`
* First GPU call on a box takes a few seconds (CPU→GPU model migration);
  the API `timeout` (s) is honored per call.
* tapnext sessions: a **fresh session ID is generated on every page load /
  remount** — the viewer (call history) doesn't survive a reload, so a
  persisted ID would mix a dead box session with a fresh viewer and the
  tracks would span both (nonsense).  Viewer and box therefore start in
  lockstep; use the console **Reset** / **regenerate** to restart
  mid-sequence.  Server-side sessions are reaped after
  `TAPNEXT_SESSION_TTL` (default 1800 s) regardless.
* lang_sam decoded item keys: `boxes / masks / scores / text_labels /
  mask_scores` (a live test caught an early `bbox` typo).
* test asset: `images/clip/test/dog.jpg`.

## 8. Next steps (open)

1. ~~vggt GLB camera auto-fit~~ — now covered: live reconstruction through
   the panel path (typed buffers incl. 4.3 MB world_points, GLB served as
   `model/gltf-binary`). Remaining: in-browser eyeball pass of the glb orbit
   + `overlay` pixel check, history click-through (all code-built and
   data-verified).
2. ~~MoGe 3D point cloud~~ — done the easy way: new `points` visualizer
   (`viz/PointCloud.tsx`) renders the back-projected cloud with three.js
   `THREE.Points` directly from the typed arrays — the hand-rolled client-
   side GLB writer was dropped (point clouds need no 3D file format).
   Live e2e: `node webui/web/.moge_points_e2e.cjs` (canvas check + screenshot).
3. In-browser human pass: `overlay` pixel check, glb orbit, history
   click-through (all code-built and data-verified, just not eyeballed).
4. ~~`image_grid` for yologpt once it migrates~~ — the repo now ships a
   standard `yolo` box (`images/yolo`, envelope-conformant, ultralytics) with a
   webui def (`boxes/yolo.yaml`): `video` over `annotated_video`, `table` over
   the per-frame detection JSON (no JPEG grid — the per-frame `annotated`
   JPEGs surface via the `*` fallback / artifacts list).
5. Optional: code-split the three.js chunk (currently one ~780 kB bundle).
