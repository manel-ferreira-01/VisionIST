#!/usr/bin/env python3
"""The minimal end-user interface to the box fleet -- this is the whole point.

Every box speaks the SAME ``pipeline.PipelineService`` interface, so one thin,
box-agnostic client reaches the whole fleet. There is no per-box SDK, no
per-box import, no box-specific code in the caller. An end user just:

    Visionist(address).run(data={...}, config={...})

...and that's it. ``Visionist`` does not know it is talking to CLIP, TAPNext,
LangSAM or SBERT -- the box itself does, through the ``config`` section it
understands (e.g. ``"clip"``, ``"tapnext"``, ``"lang_sam"``, ``"sbert"``).

Run the fleet first (see docker-compose.yml), then:  python hello.py

Each row below is a full end-user request: (name, address, data, config).
Swap a model behind a box, or add a new box, and this file does not change.
"""

import pathlib
import sys

# --- make visionist_client importable whether installed or run from the repo ---
_REPO = pathlib.Path(__file__).resolve().parent.parent
_SRC  = _REPO / "visionist_client" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visionist_client import Visionist, trace  # noqa: E402

# Test payloads (real assets in the repo).
DOG, CAR = _REPO / "images/clip/test/dog.jpg", _REPO / "images/clip/test/car.jpg"
VGGT_FRAMES = (_REPO / "images/vggt/test/images/00.jpg",
               _REPO / "images/vggt/test/images/01.jpg")

# -- THE ENTIRE end-user surface: a list of (name, address, data, config) ----
FLEET = [
    (
        "clip",       "localhost:9061",
        {"images": [DOG], "texts": ["a dog in a garden"]},
        {"clip":    {"command": "process", "parameters": {}}},
    ),
    (
        "textemb",    "localhost:9062",
        {"texts": ["a dog", "a car", "grass, sky, bark"]},
        {"sbert":   {"command": "encode",  "parameters": {}}},
    ),
    (
        "tapnext",    "localhost:9063",
        {"images": [DOG, DOG, DOG]},
        {"tapnext": {"command": "track",   "parameters": {"grid_size": 30}}},
    ),
    (
        "lang_segm",  "localhost:9064",
        {"images": [DOG]},
        {"lang_sam": {"command": "segment",
                      "parameters": {"box_threshold": 0.3, "text_threshold": 0.25},
                      "text_prompt": ["a dog"]}},
    ),
    (
        "opencv",     "localhost:9065",
        {"images": [DOG, CAR]},
        {"opencv":  {"command": "match",   "parameters": {}}},
    ),
    (
        "vggt",       "localhost:9066",
        {"images": list(VGGT_FRAMES)},
        {"vggt":    {"command": "reconstruct", "parameters": {"conf_threshold": 30}}},
    ),
    (
        "moge",       "localhost:9067",
        {"images": [DOG]},
        {"moge":    {"command": "infer",   "parameters": {}}},
    ),
    (
        "yolo",       "localhost:9068",
        {"images": [DOG, CAR]},
        {"yolo":    {"command": "detect", "parameters": {"save_annotated": False}}},
    ),
]


def shape(v):
    """One-line human description of a decoded field value."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"bytes[{len(v)}]"
    if isinstance(v, (list, tuple)):
        return f"list[{len(v)}]"
    if hasattr(v, "shape"):
        return f"array{tuple(v.shape)}"
    return type(v).__name__


def reach(name, address, data, config, reset=False):
    """One generic request to one box. This is literally all the client does."""
    key = next(iter(config))                      # the box's own section name
    box = Visionist(address)
    try:
        info = box.info(timeout=8)
        if not info.get("reachable"):
            return name, address, "UNREACHABLE", ""
        if reset:
            box.reset(key)                         # generic: caller supplies the key
        res = box.run(data=data, config=config)    # <-- the one generic call
        cfg = res.config if isinstance(res.config, dict) else {}
        section = cfg.get(key, {}) if isinstance(cfg.get(key), dict) else {}
        status = section.get("status") or cfg.get("status") or "-"
        fields = ", ".join(f"{k}={shape(v)}" for k, v in res.fields.items())
        return name, address, status, fields
    except Exception as e:  # a bad box must not kill the whole tour
        return name, address, "ERROR", f"{type(e).__name__}: {e}"
    finally:
        box.close()


def main():
    print("End-user tour of the fleet -- ONE generic call per box:\n")
    print(f"{'box':<10} {'address':<18} {'status':<14} fields (decoded)")
    print("-" * 78)
    ok = 0
    for name, address, data, config in FLEET:
        # tapnext accumulates tracks across calls -> reset first (its box name,
        # supplied by the caller, not by the core).
        reset = name == "tapnext"
        n, addr, status, fields = reach(name, address, data, config, reset=reset)
        mark = "  " if status == "done" or "success" in status.lower() else "!!"
        if "done" in str(status).lower() or "success" in str(status).lower():
            ok += 1
        print(f"{mark} {name:<9} {addr:<18} {status:<14} {fields}")

    print("-" * 78)

    # --- optional: the tapnext one-liner convenience (same generic core) ----
    b = Visionist(FLEET[2][1], config_key="tapnext")
    try:
        info = b.info(timeout=8)
        if info.get("reachable"):
            r = trace(b, images=[DOG, DOG, DOG], grid_size=30)
            print(f"\n[bonus] trace(box, ...)  -> taps={shape(r.fields.get('tracks'))} "
                  f"(convenience layer over the same Visionist.run)")
    finally:
        b.close()

    print(f"\n{ok}/{len(FLEET)} boxes answered with a 'done/success' status.")
    return 0 if ok == len(FLEET) else 1


if __name__ == "__main__":
    sys.exit(main())
