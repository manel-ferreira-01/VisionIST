# MoGe-3 Box

**Monocular Geometry Estimation** — estimates 3D geometry (point maps, depth maps, surface normals, camera intrinsics) from single images using Microsoft's MoGe-3 model.

---

## Model

- **Version**: MoGe-3 (Fine-Detail Monocular Geometry with Self-Guided Sparse Volumetric Refinement)
- **Default**: `Ruicheng/moge-3-vitl` (370M params, fast inference)
- **Alternative**: `Ruicheng/moge-3-vitg` (1.25B params, best quality)
- **License**: MIT (except DINOv2 modules: Apache 2.0)

---

## Quick Start

### Run the box

```bash
cd images/moge_box

# Build
docker build -t moge-3-box -f docker/Dockerfile .

# Run (GPU required for inference)
docker run --rm --gpus all -p 9067:8061 -e PORT=8061 --ipc=host moge-3-box
```

### Test it

```bash
# Full test suite (8 tests)
python test/test_moge_box.py

# Or with custom host
BOX_HOST=localhost:9067 python test/test_moge_box.py
```

---

## API Reference

### Request Format

```jsonc
{
  "moge": {
    "command": "infer",           // "infer" (default) or "reset"
    "parameters": {
      "fov_x": 45.0,              // optional: camera FOV in degrees (0-180)
      "refine_steps": 3,          // MoGe-3 sparse refinement steps [0-∞], default=3
      "resolution_level": 9,      // inference resolution [0-9], default=9 (highest)
      "fp16": true,               // use FP16 for faster inference
      "device": "cuda:0"          // optional: force device ("cuda", "cuda:0"; "cpu" is rejected — MoGe-3 is CUDA-only)
    }
  }
}
```

**Images**: In `data.images` as a list of images (JPEG/PNG bytes, `pathlib.Path`, or file paths)

**Reset Command**: `{"moge": {"command": "reset"}}` — accepted on all boxes, clears cached state if any

### Response Format

```json
{
  "moge": {
    "status": "done",            // "done" | "empty_request" | "error"
    "runtime": 2.34,             // inference time in seconds
    "num_images": 1,             // number of images processed
    "device": "cuda:0",          // device used for inference
    "encoding": "zstd_pickle"    // payload encoding (for client decoding)
  }
}
```

**Results** (decoded from `data.results`):

When decoded (automatic with `visionist_client`), `results` is a list where each entry is:

```python
{
  "points": np.ndarray(H, W, 3),      # metric point map in camera coordinates
  "depth": np.ndarray(H, W),          # metric depth map
  "intrinsics": np.ndarray(3, 3),     # normalized camera intrinsics matrix
  "mask": np.ndarray(H, W),           # binary mask (valid pixels = 1, invalid = 0)
  "normal": np.ndarray(H, W, 3)       # surface normal map in camera coordinates
}
```

**Coordinate System** (OpenCV camera coordinates):
- `x`: right
- `y`: down
- `z`: forward (into the scene)

**Output Resolution**: Output maps match the **input image resolution** (H, W).

### Raw Response Bytes

If using raw gRPC without the client, `results` arrives as:
```python
bytes = zstandard.decompress(raw_bytes)
results = pickle.loads(bytes)
```

The `encoding` field in response config declares `"zstd_pickle"` so the client auto-decodes it.

---

## Usage Examples

### Basic Inference (single image)

```python
from visionist_client import Visionist
import pathlib

box = Visionist("localhost:9067")

res = box.run(
    data={"images": [pathlib.Path("image.jpg")]},
    config={"moge": {
        "command": "infer",
        "parameters": {
            "refine_steps": 3,
            "resolution_level": 9,
            "fp16": True
        }
    }}
)

# Decoded results (automatic with visionist_client)
print(res.config)  # {"moge": {"status": "done", "runtime": 2.34, ...}}
print(res.results)  # [{'points': (H,W,3), 'depth': (H,W), ...}]

# Access individual outputs
depth = res.results[0]["depth"]
points = res.results[0]["points"]
normal = res.results[0]["normal"]

print(f"Depth shape: {depth.shape}")
print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
print(f"Point cloud size: {points.shape[0] * points.shape[1]} points")
```

### With Camera FOV (improves accuracy)

```python
res = box.run(
    data={"images": [pathlib.Path("image.jpg")]},
    config={"moge": {
        "parameters": {
            "fov_x": 60.0,          # known camera FOV in degrees
            "refine_steps": 3
        }
    }}
)
```

### Batch Processing (multiple images)

```python
images = [
    pathlib.Path("img1.jpg"),
    pathlib.Path("img2.jpg"),
    pathlib.Path("img3.jpg")
]

res = box.run(
    data={"images": images},
    config={"moge": {"parameters": {"refine_steps": 3}}}
)

print(f"Processed {len(res.results)} images")
for i, result in enumerate(res.results):
    print(f"  Image {i+1}: depth range = {result['depth'].min():.2f}m - {result['depth'].max():.2f}m")
```

### Batch Processing (multiple images)

```python
images = [
    pathlib.Path("img1.jpg"),
    pathlib.Path("img2.jpg"),
    pathlib.Path("img3.jpg")
]

res = box.run(
    data={"images": images},
    config={"moge": {"parameters": {"refine_steps": 3}}}
)

print(f"Processed {len(res.results)} images")
for i, result in enumerate(res.results):
    print(f"  Image {i+1}: depth range = {result['depth'].min():.2f}m - {result['depth'].max():.2f}m")
```

### CPU Inference — not supported

MoGe-3's sparse volumetric refinement uses CUDA-only Triton kernels (flex_gemm),
so `"device": "cpu"` is rejected up front with a clean error:

```json
{"moge": {"status": "error", "error": "CPU inference is not supported for MoGe-3 ..."}}
```

Use the default auto-GPU path (the box auto-selects CUDA when available).

### Reset Box State

```python
# Reset command (always works, even if model not loaded)
res = box.run(
    data={},
    config={"moge": {"command": "reset"}}
)
print(res.config)  # {"moge": {"status": "done", "action": "reset"}}
```

---

## GPU Behavior

### Memory Lifecycle
- **Startup**: Model loads on CPU (no VRAM needed)
- **First request**: Auto-moves to GPU (in-place `.to()` on torch modules)
- **Idle**: Falls back to CPU after **60s** of inactivity
- **Fixed VRAM**: ~2-3 GB after first GPU use (model weights + CUDA context, not a leak)

### Tips
- Use `fp16: true` for **~2x faster** inference with minimal quality loss
- `resolution_level` controls token count (0-9, default 9), affects speed/quality trade-off
- `refine_steps`: 0 for fastest, 3 for best (default), higher = slower

---

## Installation (Local Dev)

### Dependencies

```bash
# Install client first
pip install -e visionist_client

# Install MoGe-3 box requirements (requires CUDA 12.2 or adjust Dockerfile)
pip install git+https://github.com/microsoft/MoGe.git
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122
pip install opencv-python zstandard
```

### Run Locally

```bash
cd images/moge_box
python src/moge_service.py
```

Box will start on port **8061** and download the model automatically on first request.

---

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `Connection refused` on port | Box not running | `docker logs <container>` or check service startup |
| `status: error, "Model not loaded"` | Model download failed | Check network/HuggingFace access, retry |
| `CUDA out of memory` | GPU VRAM exhausted | Use smaller `resolution_level` (e.g., 5-7), enable `fp16` |
| Very slow first inference | Model loading + warmup | First request is slow; subsequent requests are faster |
| `Empty images list` | No images in request | Pass `data={"images": [path]}` not `data={"images": path}` |
| Tied up at startup | Model download in progress | Wait 30-90s for initial model download (HF Hub) |

---

## Model Versions

| Version | Model | Params | Quality | Speed |
|---------|-------|--------|---------|-------|
| **MoGe-3 (gigantic)** | `Ruicheng/moge-3-vitg` | 1.25B | ⭐⭐⭐⭐⭐ | ★★☆☆☆ |
| **MoGe-3 (large)** | `Ruicheng/moge-3-vitl` | 370M | ⭐⭐⭐⭐ | ★★★☆☆ |
| MoGe-2 (large) | `Ruicheng/moge-2-vitl` | 326M | ⭐⭐⭐ | ★★★★☆ |
| MoGe-2 (small) | `Ruicheng/moge-2-vits` | 35M | ⭐⭐ | ★★★★★ |

To switch model, edit `docker/Dockerfile` argument `MODEL_REPO` and rebuild.

---

## Output Explanations

### Point Map (`points`)
- Shape: `(H, W, 3)`
- **OpenCV camera coordinates**: `(x=right, y=down, z=forward)`
- Metric units (meters, assuming FOV or calibrated camera)
- Invalid pixels have depth ≈ 0 or are masked out

### Depth Map (`depth`)
- Shape: `(H, W)`
- Metric distance from camera (meters)
- Values ≈ 0 or masked indicate invalid/occluded regions

### Intrinsics (`intrinsics`)
- Shape: `(3, 3)` camera intrinsic matrix
- Normalized (for the normalized coordinate system used by model)
- Can be combined with FOV parameter for real-world coordinates

### Normal Map (`normal`)
- Shape: `(H, W, 3)`
- Unit vectors in OpenCV camera coordinates
- Use for shading, occlusion detection, surface analysis

### Mask (`mask`)
- Shape: `(H, W)`
- Binary: `1` for valid pixels, `0` for invalid/masked
- Recommended to filter point cloud: `points[mask == 1]`

---

## License & Citation

**License**: MIT (except DINOv2 modules under Apache 2.0)

**Citation**:
```bibtex
@misc{kong2026finedetailmonoculargeometryestimation,
      title={Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement},
      author={Lingyu Kong and Ruicheng Li and Ruicheng Wang and Sicheng Xu and Chengtang Yao and Jianfeng Xiang and Jiaolong Yang},
      year={2026},
      eprint={2607.17967},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**Project**: https://qft-333.github.io/moge3page/
**Models**: https://huggingface.co/Ruicheng/moge-3-vitl

---

## Integration with Other Boxes

### Example: Compose with `clip` for semantic understanding

```python
from visionist_client import Visionist
import pathlib

# 1. Get embeddings for text prompts
clip_box = Visionist("localhost:9061")  # clip box
clip_res = clip_box.run(
    data={"images": [pathlib.Path("image.jpg")],
          "texts": ["building", "tree", "person"]},
    config={"clip": {"command": "encode"}}
)

# 2. Get geometry from image
moge_box = Visionist("localhost:9067")
moge_res = moge_box.run(
    data={"images": [pathlib.Path("image.jpg")]},
    config={"moge": {"parameters": {"refine_steps": 3}}}
)

# 3. Combine for semantic 3D understanding
depth = moge_res.results[0]["depth"]
points = moge_res.results[0]["points"]
```

---

## Next Steps

- Try the Gradio demo: `moge app --version v3 --pretrained Ruicheng/moge-3-vitl`
- Experiment with `resolution_level` [0-9] to balance speed/quality
- Combine with `vggt` box for multi-view 3D reconstruction
- Use `opencv_box` for camera intrinsics estimation if FOV unknown
