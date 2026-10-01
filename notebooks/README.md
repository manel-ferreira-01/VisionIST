# Notebooks — Python walkthrough of the fleet

`boxes_walkthrough.ipynb` runs the raw `visionist_client` (no webui) against
the local docker fleet: **yolo** detection on a video, **tapnext** point
tracking + observation matrix, **lightglue** feature matching, **unimatch**
flow/stereo, and an **sfm** cell that chains lightglue tracks + **moge** depth
into the sfm box (camera poses + 3D points) — rendered inline when you run it.

Run it with the fleet up (`cd fleet && docker compose up -d`), boxes on
their default host ports 9061–9071 (the sfm cell needs lightglue 9069,
moge 9067 and sfm 9071):

```bash
pip install visionist_client plotly   # or: pip install -e ../visionist_client; plotly for the 3D view
jupyter notebook boxes_walkthrough.ipynb
```

Cells are outputs-cleared on purpose — re-run them top to bottom. The
lightglue cells read test images from `../images/lightglue_box/test/`.
