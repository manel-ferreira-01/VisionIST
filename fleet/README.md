# boxes fleet

Spin up several boxes at once and drive them all with **one** generic client —
the concrete proof that the core is box-agnostic.

## What's here

| file              | purpose |
|-------------------|---------|
| `docker-compose.yml` | launches 8 real boxes (clip, textEmbedding, tapnext, lang_segm, opencv, vggt, moge, yolo) on host ports 9061–9068, all on the shared `PipelineService` interface |
| `hello.py`        | the *minimal* end-user interface: one list of `(name, address, data, config)` specs, one `Box.run(...)` per box |

## Use

```bash
docker compose up -d          # start the fleet (GPU, or drop the `x-gpu` block for CPU)
docker compose ps             # confirm they are Up
python hello.py               # reach every box with the one generic pattern
docker compose down           # stop
```

## What this demonstrates

- **Agnostic core.** `hello.py` contains *no box-specific code*. Each row is
  just `data` + a `config` dict. The box names appear only as the
  *caller-chosen* section key (`clip`, `tapnext`, …) and as addresses — the
  client never branches on them. Swap the model behind a box, or add a new box,
  and nothing in the calling code changes.
- **One interface, many payloads.** The same `Box.run(data, config)` carries
  images, text, tensors and zstd+pickled mask blobs — whatever each box returns
  comes back in `res.fields`, best-effort decoded.
- **Conveniences are optional and separate.** `trace(box, ...)` (tapnext) is
  shown at the end; it composes the same core. Delete it and nothing breaks.

## Ports

| box         | host port | config section | what it does |
|-------------|-----------|----------------|--------------|
| clip        | 9061      | `clip`         | image+text embeddings + similarity |
| textembedding | 9062    | `sbert`        | text embeddings |
| tapnext     | 9063      | `tapnext`      | point tracking (stateful; `reset` supported) |
| lang_segm   | 9064      | `lang_sam`     | text-guided segmentation |
| opencv      | 9065      | `opencv`       | feature matching + frame gating — `match` / `similarity_check` / `reset` commands on `Process` |
| vggt        | 9066      | `vggt`         | 3D reconstruction (points/depth/cameras + GLB) — heavy image, ~12 GB pull |
| moge        | 9067      | `moge`         | MoGe-3 single-view geometry (metric depth/points/normals) — CUDA-only |
| yolo        | 9068      | `yolo`         | YOLO detection **+ tracking** on images and/or a decoded video (ultralytics; per-session track ids — `session_id`/`reset`/`list` like tapnext) |
