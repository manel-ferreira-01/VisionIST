# Notebooks — Python walkthrough of the fleet

`boxes_walkthrough.ipynb` runs the raw `visionist_client` (no webui) against
the local docker fleet: **yolo** detection on a video, **tapnext** point
tracking + observation matrix, **lightglue** feature matching (rendered
inline when you run it).

Run it with the fleet up (`cd fleet && docker compose up -d`), boxes on
their default host ports 9061–9069:

```bash
pip install visionist_client          # or: pip install -e ../visionist_client
jupyter notebook boxes_walkthrough.ipynb
```

Cells are outputs-cleared on purpose — re-run them top to bottom. The
lightglue cells read test images from `../images/lightglue_box/test/`.
