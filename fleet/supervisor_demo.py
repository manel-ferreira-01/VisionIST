#!/usr/bin/env python3
"""Supervisor demo — one box-agnostic client, six boxes, same call shape.

The whole point to show: there is NO per-box SDK. Every box is reached with
the identical two-argument call

        Visionist(address).run(data={...}, config={...})

only the address and the box's own ``config`` section differ. Nothing about
CLIP / SBERT / TAPNext / LangSAM / OpenCV lives in the client; the box itself
selects what to do from the ``config`` key it understands.

    Run the fleet (docker compose up -d), then:

        python fleet/supervisor_demo.py
"""

import pathlib
import sys
import traceback

# Make visionist_client importable whether installed or run from the repo.
ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = ROOT / "visionist_client" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visionist_client import Visionist  # noqa: E402

# Real test assets in the repo.
DOG = ROOT / "images/clip/test/dog.jpg"
CAR = ROOT / "images/clip/test/car.jpg"


def shape(v) -> str:
    """One-line, human description of a decoded value."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"bytes[{len(v)}]"
    if isinstance(v, (list, tuple)):
        return f"list[{len(v)}]"
    if hasattr(v, "shape"):
        return f"array{tuple(v.shape)}"
    return type(v).__name__


def as_nested(v):
    """Tensor / ndarray / list -> plain nested Python list (values unchanged)."""
    if hasattr(v, "tolist"):            # torch.Tensor / np.ndarray
        v = v.tolist()
    return [e for e in v] if isinstance(v, (list, tuple)) else v


def flat_list(v):
    """Any nested tensor/array/list -> a single flat list of floats."""
    out: list[float] = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                rec(e)
        else:
            out.append(float(x))

    rec(as_nested(v))
    return out


def banner(i: int, name: str, addr: str):
    print(f"\n{'=' * 70}\n{i}. {name:<9} {addr}\n{'-' * 70}")


def one(i, name, addr, fn):
    banner(i, name, addr)
    try:
        fn()
    except Exception:
        print(f"  !! {name} failed:")
        traceback.print_exc(limit=3)
    print()


def demo_clip():
    """Image + text -> per-pair similarity scores (one CLIP call)."""
    b = Visionist("localhost:9061")
    try:
        res = b.run(
            data={"images": [DOG], "texts": ["a dog", "the ocean"]},
            config={"clip": {"command": "process", "parameters": {}}},
        )
        print("  status      :", res.config.get("clip", {}).get("status"))
        print("  image_emb   :", shape(res.image_emb))
        print("  text_emb    :", shape(res.text_emb))
        scores = flat_list(res.similarity)
        print("  similarity  :", [f"{x:.3f}" for x in scores])
        labels = ["a dog", "the ocean"]
        best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        print(f"  -> best text match: {labels[best]}")
    finally:
        b.close()


def demo_textemb():
    """Pure-text sentence embeddings + a similarity matrix (no images)."""
    b = Visionist("localhost:9062")
    try:
        texts = ["a dog", "a car", "grass, sky, bark"]
        res = b.run(data={"texts": texts},
                    config={"sbert": {"command": "encode", "parameters": {}}})
        print("  status      :", res.config.get("sbert", {}).get("status"))
        print("  texts in    :", texts)
        print("  embeddings  :", shape(res.embeddings))
        print("  similarities:")
        for row, t in zip(as_nested(res.similarities), texts):
            print(f"    {t:<16}" + "  ".join(f"{float(x):6.3f}" for x in row))
    finally:
        b.close()


def demo_tapnext():
    """Multi-frame point tracking -> tracks over time (stateful: reset first)."""
    b = Visionist("localhost:9063")
    try:
        b.reset("tapnext")   # tapnext accumulates across calls -> start clean
        res = b.run(data={"images": [DOG, DOG, DOG]},
                    config={"tapnext": {"command": "track", "parameters": {"grid_size": 30}}})
        print("  status          :", res.config.get("tapnext", {}).get("status"))
        print("  frames in       :", 3)
        print("  tracks          :", shape(res.tracks), "  (frames, points, x/y)")
        print("  visibles        :", shape(res.visibles))
        print("  observation_mat :", shape(res.observation_matrix))
    finally:
        b.close()


def demo_lang_segm():
    """Text-guided segmentation -> decoded results (the zstd_pickle codec)."""
    b = Visionist("localhost:9064")
    try:
        res = b.run(
            data={"images": [DOG]},
            config={"lang_sam": {
                "command": "segment",
                "parameters": {"box_threshold": 0.3, "text_threshold": 0.25},
                "text_prompt": ["a dog"],
            }},
        )
        print("  status      :", res.config.get("lang_sam", {}).get("status"))
        print("  declared enc:", res.encoding)          # "zstd_pickle" — the new contract
        print("  results type:", type(res.results).__name__)   # list, NOT bytes
        print("  images out  :", len(res.results))
        for k, o in enumerate(res.results):
            n_masks = len(o.get("masks", []))
            n_boxes = len(o.get("bboxes", []))
            print(f"    image {k}: {n_masks} mask(s), {n_boxes} box(es)")
    finally:
        b.close()


def demo_opencv():
    """Feature matching between two images -> keypoints + inlier matches."""
    b = Visionist("localhost:9065")
    try:
        res = b.run(data={"images": [DOG, CAR]},
                    config={"opencv": {"command": "match", "parameters": {}}})
        print("  status          :", (res.config.get("opencv", {}) or {}).get("status")
                                             or res.config)
        for f in ("keypoints", "descriptors", "matches_inliers_a",
                  "matches_inliers_b", "fundamental_matrix"):
            if f in res.fields:
                print(f"  {f:18}:", shape(res.fields[f]))
    finally:
        b.close()


def demo_moge():
    """Monocular 3D geometry -> per-image metric depth/points/normals (CUDA-only box)."""
    b = Visionist("localhost:9067")
    try:
        res = b.run(data={"images": [DOG]},
                    config={"moge": {"command": "infer", "parameters": {}}})
        print("  status          :", (res.config.get("moge", {}) or {}).get("status")
                                             or res.config)
        for k, item in enumerate(res.results):
            print(f"  image {k}         : depth {getattr(item.get('depth'), 'shape', '?')}, "
                  f"points {getattr(item.get('points'), 'shape', '?')}")
    finally:
        b.close()


def main() -> int:
    print("boxes fleet — one agnostic client, six boxes, one call shape\n")
    one(1, "clip",       "localhost:9061", demo_clip)
    one(2, "textemb",    "localhost:9062", demo_textemb)
    one(3, "tapnext",    "localhost:9063", demo_tapnext)
    one(4, "lang_segm",  "localhost:9064", demo_lang_segm)
    one(5, "opencv",     "localhost:9065", demo_opencv)
    one(6, "moge",       "localhost:9067", demo_moge)
    print("Every call above is the same shape:  Visionist(addr).run(data, config).")
    print("That is the entire end-user surface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
