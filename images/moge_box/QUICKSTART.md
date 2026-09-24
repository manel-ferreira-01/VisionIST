# MoGe-3 Box - Quick Start

## What It Does
Estimates accurate 3D geometry from single images:
- **Point maps** (H×W×3) - metric 3D coordinates
- **Depth maps** (H×W) - metric distances in meters
- **Normal maps** (H×W×3) - surface normals
- **Camera intrinsics** (3×3) - normalized camera parameters

## Run the Box

```bash
# 1. Build
docker build -t moge-3-box -f docker/Dockerfile .

# 2. Run (requires GPU)
docker run --rm --gpus all -p 9067:8061 -e PORT=8061 --ipc=host moge-3-box
```

First run downloads the model (~2GB for moge-3-vitl) from HuggingFace.

## Call It

```python
from visionist_client import Visionist
import pathlib

box = Visionist("localhost:9067")

res = box.run(
    data={"images": [pathlib.Path("image.jpg")]},
    config={"moge": {
        "command": "infer",
        "parameters": {
            "refine_steps": 3,        # MoGe-3 sparse refinement
            "resolution_level": 9,    # 0-9, default=9 (highest quality)
            "fp16": True              # Use FP16 for 2x speed
        }
    }}
)

# Results (auto-decoded from zstd+pickle)
print(res.results[0].keys())
# dict_keys(['points', 'depth', 'intrinsics', 'mask', 'normal'])

print(f"Depth range: {res.results[0]['depth'].min():.2f}m - {res.results[0]['depth'].max():.2f}m")
print(f"Point cloud size: {res.results[0]['points'].shape}")
```

## Test It

```bash
# Run the full test suite (8 tests)
python test/test_moge_box.py

# With custom host
BOX_HOST=localhost:9067 python test/test_moge_box.py

# Ships a fixture at images/moge_box/test/test.jpg (clip box photos are also
# picked up automatically when running inside the repo).
```

## Output Format

Each image produces:
```python
{
    "points": (H, W, 3) np.ndarray     # metric 3D points [x,y,z] in camera coords
    "depth": (H, W) np.ndarray         # metric depth in meters
    "intrinsics": (3, 3) np.ndarray    # normalized camera intrinsics
    "mask": (H, W) np.ndarray          # 1=valid, 0=invalid pixels  
    "normal": (H, W, 3) np.ndarray     # surface normals in camera coords
}
```

## GPU Memory

- **First request**: Model auto-moves CPU→GPU
- **Idle fallback**: Returns to CPU after 60s
- **VRAM usage**: ~2-3 GB fixed (model + CUDA context)
- **Tip**: Use `fp16: True` for speed, adjust `resolution_level` (0-9) for quality/speed tradeoff

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `fov_x` | auto | 0-180° | Camera horizontal FOV (improves accuracy) |
| `refine_steps` | 3 | 0+ | MoGe-3 sparse refinement steps |
| `resolution_level` | 9 | 0-9 | Inference resolution (0=fast, 9=quality) |
| `fp16` | true | bool | Use FP16 for faster inference |
| `device` | auto | cuda/cuda:N | Force device ("cpu" is rejected — MoGe-3 is CUDA-only) |

## Model Options

Edit `docker/Dockerfile` argument `MODEL_REPO`:
- `Ruicheng/moge-3-vitl` (default, 370M, fast)
- `Ruicheng/moge-3-vitg` (1.25B, best quality, slower)

## Example: Compose with Other Boxes

```python
from visionist_client import Visionist
import pathlib

# Get geometry
moge = Visionist("localhost:9067")
geometry = moge.run(
    data={"images": [pathlib.Path("img.jpg")]},
    config={"moge": {"parameters": {"refine_steps": 3}}}
)

# Get semantic understanding
clip = Visionist("localhost:9061")
embeddings = clip.run(
    data={"images": [pathlib.Path("img.jpg")]},
    config={"clip": {"command": "encode"}}
)

# Combine: find depth of text-prompted objects
points = geometry.results[0]["points"]
depth_map = geometry.results[0]["depth"]
print(f"Scene depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")
```

## Troubleshooting

```bash
# Check if box is running
curl localhost:9067/health  # or use box.info()

# View logs
docker logs <container_id>

# Model download failed?
# Check network access to HuggingFace, or download manually:
python -c "from moge.model.v3 import MoGeModel; MoGeModel.from_pretrained('Ruicheng/moge-3-vitl')"

# Out of memory?
# Reduce resolution_level: 5-7 instead of 9
```

## Full API Reference

See `README.md` for complete documentation including:
- Request/response formats
- All parameters and options
- Error handling
- Advanced usage examples
- License and citation
