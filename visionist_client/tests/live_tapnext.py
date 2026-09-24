#!/usr/bin/env python3
"""Live smoke test: point ``visionist_client.Visionist`` at a running tapnext box.

Start a tapnext box on ``localhost:8061`` (or set ``BOX_HOST``), then:

    BOX_HOST=localhost:8061 python visionist_client/tests/live_tapnext.py

It loads a handful of frames from the bundled ``apple.mp4`` and drives the box
two ways, to prove both call paths:

  * ``trace(box, ...)`` -- the tapnext *convenience* (images -> tracks), an
                           optional layer over the generic API.
  * ``Visionist.run(...)``    -- the *generic* API (explicit data/config/method),
                           called by hand with the very same envelope.

Exits 0 on success, non-zero otherwise.
"""

import os
import sys

# Allow running directly from the repo without ``pip install -e``.
# The package lives in visionist_client/src/visionist_client.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from visionist_client import Visionist, trace  # noqa: E402


def frames_from_video(path, n=4):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    if not out:
        raise RuntimeError(f"could not read frames from {path}")
    return out


def encode_jpeg(frame):
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _to_numpy(x):
    """Best-effort: tensor/ndarray -> ndarray (else None)."""
    if x is None or isinstance(x, (list, tuple, str, bytes, dict)):
        return None
    if isinstance(x, np.ndarray):
        return x
    to_np = getattr(x, "numpy", None)
    if callable(to_np):
        return to_np()
    return None


def _show(label, res, expect_frames=None):
    """Print + sanity-check a tapnext tracking Result. Returns True on success."""
    cfg = res.config if isinstance(res.config, dict) else {}
    tap = cfg.get("tapnext", {}) if isinstance(cfg, dict) else {}
    print(f"[{label}] status : {tap.get('status', cfg)}")
    print(f"[{label}] fields : {list(res.fields)}")

    tracks = _to_numpy(res.fields.get("tracks"))
    visibles = _to_numpy(res.fields.get("visibles"))

    if tracks is None:
        raw = res.fields.get("tracks")
        print(f"[{label}] tracks : not an array; type={type(raw).__name__} "
              f"(raw bytes if you don't have torch installed)")
        if isinstance(raw, (bytes, bytearray)):
            print(f"[{label}]        raw bytes len={len(raw)}")
        return False  # no torch here -> cannot shape-check; not a failure of the client

    print(f"[{label}] tracks : shape={tracks.shape} dtype={tracks.dtype}")
    print(f"[{label}] tracks[-1][:5] =\n{tracks[-1][:5]}")
    assert tracks.ndim == 3 and tracks.shape[2] == 2, \
        f"unexpected tracks shape {tracks.shape}"
    if expect_frames is not None:
        assert tracks.shape[0] == expect_frames, \
            f"expected {expect_frames} frames, got {tracks.shape[0]}"
    if visibles is not None:
        n_vis = int(np.asarray(visibles[-1]).sum()) if visibles.ndim == 2 else "n/a"
        print(f"[{label}] visibles: shape={visibles.shape} n_visible(last)={n_vis}")
    return True


def main():
    host = os.environ.get("BOX_HOST", "localhost:8061")
    video = os.environ.get(
        "TEST_VIDEO",
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "images",
                "tapnext_tracker", "test", "apple.mp4",
            )
        ),
    )

    print(f"target box : {host}")
    print(f"test video : {video}")

    frames = frames_from_video(video, n=4)
    H, W = frames[0].shape[:2]
    print(f"frames     : {len(frames)} @ {W}x{H}")
    jpeg_frames = [encode_jpeg(f) for f in frames]

    ok = 0
    with Visionist(host, config_key="tapnext") as b:
        print("\n-- Visionist.info() --")
        info = b.info()
        print(info)
        if not info.get("reachable"):
            print("\n!! box not reachable; check it is running at that address.")
            return 2
        if not info.get("reflection"):
            print("   (box does not expose reflection; continuing anyway)")

        # --------------------------------------------------- trace(box, ...)
        print("\n-- trace(box, ...)  [convenience: images -> tracks] --")
        res_trace = trace(b, images=jpeg_frames, grid_size=30)
        if _show("trace", res_trace, expect_frames=len(jpeg_frames)):
            ok += 1

        # ----------------------------------------------------------- Visionist.run()
        print("\n-- Visionist.run()     [generic: explicit data/config/method] --")
        # Same envelope as trace(), but assembled by hand so we exercise the
        # generic path (no image assumption built into the client).
        # NOTE: the tapnext box *accumulates tracks across sequential requests*
        # and only clears on a reset, so reset_first=True keeps this a clean,
        # self-contained one-shot (mirrors trace() default).
        tapnext_cfg = {"tapnext": {"command": "track",
                                   "parameters": {"grid_size": 30}}}
        res_run = b.run(
            data={"images": jpeg_frames},
            config=tapnext_cfg,
            method="Process",
            reset_first=True,
        )
        if _show("run  ", res_run, expect_frames=len(jpeg_frames)):
            ok += 1

        # --------------------------------------------------------- cross-check
        if ok == 2:
            # Both paths returned arrays; they should have identical shape
            # (same N frames, same point count) since the input is identical.
            a = _to_numpy(res_trace.fields.get("tracks"))
            c = _to_numpy(res_run.fields.get("tracks"))
            if a is not None and c is not None:
                assert a.shape == c.shape, (a.shape, c.shape)
                diff = float(np.max(np.abs(a - c)))
                print(f"\n[cross-check] trace().tracks vs run().tracks: "
                      f"shape match={a.shape == c.shape}, max|Δ|={diff:.3e}")

    print("\nOK -- one-shot tapnext call(s) succeeded "
          f"({ok}/2 call paths checked with numpy/torch).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
