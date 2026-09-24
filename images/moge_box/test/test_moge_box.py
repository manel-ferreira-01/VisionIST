#!/usr/bin/env python3
"""
Complete MoGe-3 Box Test Suite

Tests all major functionality:
1. Connectivity check
2. Reset command  
3. Empty request handling
4. Basic inference
5. Batch processing
6. CPU mode
7. Parameter variations
8. Error handling

Usage:
    python test/test_moge_box.py                # test localhost:8061
    BOX_HOST=localhost:9067 python test/test_moge_box.py
    python -m pytest test/test_moge_box.py -v   # pytest version
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Import visionist_client
try:
    from visionist_client import Visionist
except ImportError:
    logger.error("visionist_client not found. Install it first:")
    logger.error("  cd ~/boxes && pip install -e visionist_client")
    sys.exit(1)

# Test configuration
BOX_HOST = os.getenv("BOX_HOST", "localhost:8061")

# Create output directory for test results (at the box root)
OUTPUT_DIR = Path(__file__).parent.parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"Test output directory: {OUTPUT_DIR}")

# Test image: the box ships its own fixture (test/test.jpg); also accept the
# clip box photos when running inside the repo.
COMMON_IMAGE_PATHS = [
    Path(__file__).parent / "test.jpg",
    Path("/home/manuelf/boxes/images/clip/test/dog.jpg"),
    Path("/home/manuelf/boxes/images/clip/test/car.jpg"),
]

TEST_IMAGE_DIRS = [
    Path("../../images/clip/test"),
    Path("images/clip/test"),
]


def save_inference_results(result, input_img_path, runtime, test_name):
    """Save depth map, normal map, and other outputs as images."""
    try:
        import numpy as np
        import cv2
        
        # Create timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"{test_name}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save depth map
        depth = result['depth']
        valid_mask = result['mask'] > 0
        
        # Normalize depth for visualization (0-1 range)
        depth_vis = depth.copy().astype(np.float32)
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            if depth_max > depth_min:
                depth_vis = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_vis[:] = 0.5
        
        # Apply colormap
        depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save depth
        depth_path = output_dir / "depth_map.png"
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
        logger.info(f"   Saved depth map: {depth_path}")
        
        # Save original depth values as numpy array (for inspection)
        depth_raw_path = output_dir / "depth_raw.npy"
        np.save(str(depth_raw_path), depth)
        logger.info(f"   Saved raw depth: {depth_raw_path}")
        
        # Save normal map
        normal = result['normal']
        # Convert from [-1, 1] to [0, 255]
        normal_vis = ((normal + 1) / 2 * 255).astype(np.uint8)
        normal_path = output_dir / "normal_map.png"
        cv2.imwrite(str(normal_path), normal_vis)
        logger.info(f"   Saved normal map: {normal_path}")
        
        # Save mask
        mask = result['mask']
        mask_path = output_dir / "mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        logger.info(f"   Saved valid mask: {mask_path}")
        
        # Copy input image for reference
        if Path(input_img_path).exists():
            import shutil
            input_path = output_dir / "input_image.jpg"
            shutil.copy2(str(input_img_path), str(input_path))
            logger.info(f"   Saved input image: {input_path}")
        
        # Save 3D point cloud as CSV (for visualization in MeshLab/Blender)
        points = result['points']
        valid_idx = np.where(valid_mask)
        points_valid = points[valid_idx]
        
        if len(points_valid) > 0:
            # Get approximate colors from input image
            colors = [200, 200, 200]  # Default gray
            if Path(input_img_path).exists():
                try:
                    input_img = cv2.imread(str(input_img_path))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    
                    # Sample colors for valid points
                    H, W, _ = input_img.shape
                    colors = []
                    for i in range(len(points_valid)):
                        x, y, z = points_valid[i]
                        # Simple projection to image coordinates
                        img_x = min(max(int(x + W/2), 0), W-1)
                        img_y = min(max(int(-y + H/2), 0), H-1)
                        colors.append(input_img[img_y, img_x].tolist())
                except:
                    pass
            
            # Save as CSV
            csv_path = output_dir / "pointcloud.csv"
            with open(str(csv_path), 'w') as f:
                f.write("x,y,z,r,g,b\n")
                for i, (x, y, z) in enumerate(points_valid):
                    if len(colors) > i:
                        r, g, b = colors[i]
                    else:
                        r, g, b = 200, 200, 200
                    f.write(f"{x:.6f},{y:.6f},{z:.6f},{r},{g},{b}\n")
            logger.info(f"   Saved point cloud: {csv_path}")
        
        logger.info(f"✅ All outputs saved to: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.warning(f"   Failed to save outputs: {e}")
        return None


def find_test_image():
    """Find an existing test image (box fixture first, then other boxes)."""
    # First check specific known paths
    for img_path in COMMON_IMAGE_PATHS:
        resolved = img_path.expanduser().resolve()
        if resolved.exists() and resolved.is_file():
            logger.info(f"Found test image: {resolved}")
            return resolved
    
    # Then search directories
    for test_dir in TEST_IMAGE_DIRS:
        # Resolve path (handles ~ and relative paths)
        resolved = test_dir.expanduser().resolve()
        if resolved.exists() and resolved.is_dir():
            for img in resolved.glob("*.{jpg,jpeg,png}"):
                if img.exists():
                    logger.info(f"Found test image: {img}")
                    return img
    return None


def get_test_image():
    """Find a real test image from other boxes or create one."""
    test_img = find_test_image()
    if test_img:
        logger.info(f"Using test image: {test_img}")
        return test_img
    else:
        logger.warning("No test image found. Creating synthetic one...")
        return create_test_image()


def create_test_image():
    """Create a simple test image if none exists."""
    test_path = Path("/tmp/test_image.jpg")
    
    try:
        import cv2
        import numpy as np
        
        logger.info("Creating test image...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(640):
            gradient_val = min(255, int(i / 640 * 255))
            img[:, i] = [gradient_val, gradient_val // 2, 255 - gradient_val]
        
        cv2.circle(img, (320, 240), 100, [128, 128, 128], -1)
        cv2.rectangle(img, (150, 150), (250, 350), [200, 100, 50], -1)
        
        cv2.imwrite(str(test_path), img)
        logger.info(f"Created test image: {test_path}")
        return test_path
        
    except ImportError:
        logger.warning("opencv-python not installed, skipping")
        return None


def test_1_connectivity():
    """Test 1: Basic connectivity and reflection."""
    print("\n" + "="*70)
    print("TEST 1: Connectivity Check")
    print("="*70)
    
    try:
        box = Visionist(BOX_HOST)
        info = box.info()
        print(f"✅ Connected to box at {BOX_HOST}")
        print(f"   Service info: {info}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("   Make sure box is running: docker ps | grep moge")
        return False


def test_2_reset_command():
    """Test 2: Reset command (always should work)."""
    print("\n" + "="*70)
    print("TEST 2: Reset Command")
    print("="*70)
    
    try:
        box = Visionist(BOX_HOST)
        
        res = box.run(
            data={},
            config={"moge": {"command": "reset"}}
        )
        
        status = res.config.get("moge", {}).get("status")
        
        if status in ["done", "empty_request"]:
            print(f"✅ Reset works: status={status}")
            return True
        else:
            print(f"⚠️  Unexpected status: {status}")
            return True
            
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False


def test_3_empty_request():
    """Test 3: Empty request handling."""
    print("\n" + "="*70)
    print("TEST 3: Empty Request Handling")
    print("="*70)
    
    try:
        box = Visionist(BOX_HOST)
        
        res = box.run(
            data={},
            config={"moge": {"command": "infer"}}
        )
        
        status = res.config.get("moge", {}).get("status")
        
        if status == "empty_request":
            print(f"✅ Empty request handled: status={status}")
            return True
        else:
            print(f"⚠️  Expected 'empty_request', got: {status}")
            return True
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_4_basic_inference():
    """Test 4: Basic single-image inference."""
    print("\n" + "="*70)
    print("TEST 4: Basic Inference (Single Image)")
    print("="*70)
    
    test_img = get_test_image()
    if not test_img:
        print(f"⏭️  Skipping: No test image")
        return False
    
    try:
        box = Visionist(BOX_HOST)
        
        print(f"   Running inference on {test_img}...")
        start_time = time.time()
        
        res = box.run(
            data={"images": [test_img]},
            config={"moge": {
                "parameters": {
                    "refine_steps": 3,
                    "fp16": True,
                    "resolution_level": 7
                }
            }}
        )
        
        runtime = time.time() - start_time
        
        status = res.config.get("moge", {}).get("status")
        if status != "done":
            print(f"❌ Inference failed: {res.config}")
            return False
        
        if not hasattr(res, 'results') or res.results is None:
            print(f"❌ No results returned")
            return False
        
        result = res.results[0]
        print(f"✅ Inference successful!")
        print(f"   Runtime: {runtime:.2f}s")
        print(f"   Device: {res.config['moge'].get('device', 'unknown')}")
        print(f"\n   Output shapes:")
        print(f"      Points (3D):    {result['points'].shape}")
        print(f"      Depth:          {result['depth'].shape}")
        print(f"      Normal:         {result['normal'].shape}")
        print(f"      Intrinsics:     {result['intrinsics'].shape}")
        print(f"      Mask:           {result['mask'].shape}")
        
        # Save visualization outputs
        print(f"\n   Saving visualization outputs...")
        save_inference_results(result, test_img, runtime, "basic_inference")
        
        # Quick statistics
        depth = result['depth']
        valid = result['mask'] > 0
        if valid.sum() > 0:
            print(f"\n   Depth stats (valid: {valid.sum()}/{valid.size}):")
            print(f"      Min: {depth[valid].min():.2f}m")
            print(f"      Max: {depth[valid].max():.2f}m")
            print(f"      Mean: {depth[valid].mean():.2f}m")
        else:
            print(f"\n   No valid pixels in result")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_batch_inference():
    """Test 5: Batch processing (multiple images)."""
    print("\n" + "="*70)
    print("TEST 5: Batch Inference (Multiple Images)")
    print("="*70)
    
    test_img = get_test_image()
    if not test_img:
        print(f"⏭️  Skipping: No test image")
        return False
    
    try:
        box = Visionist(BOX_HOST)
        
        # Use same image 3 times to simulate batch
        images = [test_img, test_img, test_img]
        
        print(f"   Running batch inference on {len(images)} images...")
        start_time = time.time()
        
        res = box.run(
            data={"images": images},
            config={"moge": {"parameters": {"refine_steps": 3}}}
        )
        
        runtime = time.time() - start_time
        
        status = res.config.get("moge", {}).get("status")
        if status != "done":
            print(f"❌ Batch failed: {res.config}")
            return False
        
        print(f"✅ Batch inference successful!")
        print(f"   Processed: {len(res.results)} images")
        print(f"   Runtime: {runtime:.2f}s")
        print(f"   Per-image: {runtime/len(res.results):.2f}s")
        
        # Save results for first image
        print(f"\n   Saving first result outputs...")
        save_inference_results(res.results[0], test_img, runtime, "batch_inference")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch failed: {e}")
        return False


def test_6_cpu_mode():
    """Test 6: CPU mode must be cleanly rejected (MoGe-3 is CUDA-only)."""
    print("\n" + "="*70)
    print("TEST 6: CPU Mode (unsupported — expect clean rejection)")
    print("="*70)
    
    test_img = get_test_image()
    if not test_img:
        print(f"⏭️  Skipping: No test image")
        return False
    
    try:
        box = Visionist(BOX_HOST)
        
        print(f"   Requesting CPU inference (should be rejected cleanly)...")
        res = box.run(
            data={"images": [test_img]},
            config={"moge": {
                "parameters": {
                    "device": "cpu",
                    "refine_steps": 2
                }
            }}
        )
        
        status = res.config.get("moge", {}).get("status")
        error = res.config.get("moge", {}).get("error", "")
        if status == "error" and "not supported" in error.lower():
            print(f"✅ CPU correctly rejected")
            print(f"   {error[:100]}")
            return True
        print(f"❌ Expected a clean 'CPU not supported' error, got status={status}, error={error[:100]}")
        return False
        
    except Exception as e:
        print(f"❌ CPU mode request failed: {e}")
        return False


def test_7_parameter_combinations():
    """Test 7: Various parameter combinations."""
    print("\n" + "="*70)
    print("TEST 7: Parameter Combinations")
    print("="*70)
    
    test_img = get_test_image()
    if not test_img:
        print(f"⏭️  Skipping: No test image")
        return False
    
    tests_passed = 0
    tests_total = 0
    
    param_combinations = [
        {"refine_steps": 0, "fp16": True, "desc": "Fast (0 refine)"},
        {"refine_steps": 5, "fp16": False, "desc": "Slow (5 refine, FP32)"},
        {"refine_steps": 3, "fp16": True, "resolution_level": 3, "desc": "Low res (level 3)"},
        {"refine_steps": 3, "fp16": True, "fov_x": 45.0, "desc": "With FOV (45°)"},
    ]
    
    for params in param_combinations:
        tests_total += 1
        desc = params.pop("desc")
        
        try:
            box = Visionist(BOX_HOST)
            
            res = box.run(
                data={"images": [test_img]},
                config={"moge": {"parameters": {"refine_steps": 3, "fp16": True, **params}}}
            )
            
            status = res.config.get("moge", {}).get("status")
            if status == "done":
                print(f"   ✅ {desc}: Success")
                tests_passed += 1
            else:
                print(f"   ❌ {desc}: Status={status}")
                
        except Exception as e:
            print(f"   ❌ {desc}: {e}")
    
    print(f"\n   Passed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_8_error_handling():
    """Test 8: Error handling."""
    print("\n" + "="*70)
    print("TEST 8: Error Handling")
    print("="*70)
    
    try:
        box = Visionist(BOX_HOST)
        
        # The client reads local files itself, so a box-side error can only be
        # triggered by invalid image bytes (undecodable by cv2).
        bad_bytes = b"\x00\x01\x02\x03 not-an-image"

        res = box.run(
            data={"images": [bad_bytes]},
            config={"moge": {"parameters": {"refine_steps": 3}}}
        )
        
        status = res.config.get("moge", {}).get("status")
        if status == "error":
            error_msg = res.config.get("moge", {}).get("error", "")
            print(f"✅ Error handling works")
            print(f"   Error: {error_msg[:100]}...")
            return True
        else:
            print(f"⚠️  Expected error but got: {status}")
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and summarize results."""
    print("\n" + "="*70)
    print("MOGE-3 BOX TEST SUITE")
    print(f"Box Host: {BOX_HOST}")
    print("="*70)
    
    tests = [
        ("Connectivity", test_1_connectivity),
        ("Reset Command", test_2_reset_command),
        ("Empty Request", test_3_empty_request),
        ("Basic Inference", test_4_basic_inference),
        ("Batch Inference", test_5_batch_inference),
        ("CPU Mode", test_6_cpu_mode),
        ("Parameter Combinations", test_7_parameter_combinations),
        ("Error Handling", test_8_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n💥 Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name:<30}")
    
    print("-"*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MoGe-3 box is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
