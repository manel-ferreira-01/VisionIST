# VisionIST — documentation

A fleet of independent, Dockerized AI inference services ("boxes") speaking one
shared gRPC envelope, called by `visionist_client` or by orchestration-layer boxes.

## Start here

- **[Architecture_Overview](Architecture_Overview.md)** — what a box is, the
  conventions every box follows, GPU memory lifecycle, what's retired (Maestro).
- **[Quick_Start_Guide](Quick_Start_Guide.md)** — run a box, call it, build your
  own from scratch, test it.
- **[gRPC_Services_Reference](gRPC_Services_Reference.md)** — the envelope
  contract: `config_json` shape per box, `data` fields, `results` encoding,
  devices, calling via `visionist_client` or raw stubs.
- **[Docker_Image_Template_Guide](Docker_Image_Template_Guide.md)** — Dockerfile
  templates (CPU / CUDA / YOLO) used by the boxes in this repo.
- **[CODECS](CODECS.md)** — self-describing payload decoding: the `"encoding"`
  contract boxes declare in their response config, the 5 named codecs, the
  legacy fallback, and design rationale.
- **[Webui_Guide](Webui_Guide.md)** — the declarative web layer over the
  fleet: webui state, run/build/test loop, def vocabulary, wire rules,
  gotchas, next steps (replaces the retired `webui/STATUS.md`).

## Boxes in this repo

| Box | Type | What it does |
|-----|------|--------------|
| clip | GPU | CLIP image/text embeddings |
| tapnext_tracker | GPU | Point tracking with TAPNext (stateful) |
| lang_segm | GPU | Text-guided segmentation (LangSAM) |
| textEmbedding | GPU / CPU | Sentence-BERT text embeddings |
| vggt | GPU | 3D reconstruction from image sequences |
| yolo | GPU / CPU | Object detection **+ tracking** (YOLOv8n via ultralytics) on images and videos — per-session track ids (multi-session, tapnext contract) |
| lightglue_box | GPU / CPU | Feature matching with LightGlue — SuperPoint/DISK features; for image pairs, the matcher's `matches` (+ `confidence`) |
| opencv_box | GPU / CPU | Feature extraction & matching (SIFT / ORB / LightGlue) |
| moge_box | GPU | **MoGe-3: Monocular 3D geometry (depth/points/normals)** |

**The per-box README is the authoritative source for that box's request shape**
(config keys, fields, status, how to decode `results`):
[`images/<name>/README.md`](../images/).

## The client

- [`visionist_client/README.md`](../visionist_client/README.md) — the Python client
  (`Visionist(ip:port).run(data, config)`), coercion rules, result decoding, `info()`.
