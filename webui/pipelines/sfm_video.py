"""sfm_video — camera poses + 3D points from a video, three boxes chained.

The webui port of the sfm cell of ``notebooks/boxes_walkthrough.ipynb``:

1. lightglue — ``visionist_client.track_stream`` over the frames (one
   ``stream`` call per frame in a job-private session) -> the observation
   matrix (2F x P, pixel x/y rows, NaN where a track is unseen);
2. moge      — one ``infer`` call per frame -> metric depth (invalid pixels
   NaN) + normalized intrinsics (median over the frames: one camera);
3. sfm       — tracks + depths + intrinsics -> per-frame [R | t], the points
   and the per-frame depth affine Z = d * lambda + o;
4. scene     — the 3D view: sparse points coloured from the frames, camera
   frustums, and every depth map depth-corrected with its frame's affine and
   back-projected into the reference camera (if the calibration is right,
   the dense clouds of all frames overlap).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

DEF = {
    "id": "sfm_video",
    "name": "SfM from video · lightglue + MoGe + sfm",
    "uses": ["lightglue", "moge", "sfm"],
    "docs": "../../images/sfm_box/README.md",
    "note": (
        "Frames in time order (a video is sampled in the browser). lightglue "
        "tracks them (sliding-window stream), MoGe gives each frame a metric "
        "depth map, the sfm box solves cameras + points + a per-frame depth "
        "affine. Frames should overlap: a short clip (a few seconds) with "
        "camera motion works best."),
    "inputs": [{
        "field": "images", "widget": "video_frames", "kind": "bb",
        "multiple": True, "required": True,
        "helper": ("a video (choose the frame count and time range) or image "
                   "files, in time order — at least 3 frames"),
    }],
    "parameters": [
        {"key": "solver", "widget": "select", "values": ["linear", "pairs", "completion"],
         "default": "linear",
         "placeholder": "linear: 2-frame rotations + global LM (best on partial tracks)"},
        {"key": "min_alive", "widget": "slider", "min": 2, "max": 32, "step": 1, "default": 4,
         "placeholder": "keep tracks seen in at least this many frames"},
        {"key": "window", "widget": "slider", "min": 1, "max": 8, "step": 1, "default": 3,
         "placeholder": "lightglue: previous frames each new frame is matched against"},
        {"key": "max_keypoints", "widget": "number", "min": 128, "max": 8192, "step": 128,
         "default": 1024},
        {"key": "mode", "widget": "select", "values": ["backbone", "greedy"],
         "default": "backbone",
         "placeholder": "track association: backbone bridges 1-frame gaps, greedy does not"},
        {"key": "resolution_level", "widget": "slider", "min": 0, "max": 9, "step": 1,
         "default": 9, "placeholder": "MoGe inference resolution (0 = fast, 9 = best)"},
        {"key": "dense_points", "widget": "number", "min": 0, "max": 2000000, "step": 50000,
         "default": 300000,
         "placeholder": "budget of back-projected depth points in the 3D view (0 = none)"},
    ],
    "results": [
        {"field": "scene", "visualizer": "scene",
         "caption": ("reconstruction — tracks (large), depth maps corrected with Z = d·λ + o "
                     "and back-projected (small), camera frustums + path; first camera = origin"),
         "params": {"points": "points", "colors": "colors", "cameras": "cameras",
                    "dense_points": "dense_points", "dense_colors": "dense_colors",
                    "dense_frames": "dense_frames", "intrinsics": "intrinsics",
                    "image_size": "image_size"}},
        {"field": "frames", "visualizer": "table",
         "caption": "per frame: tracks seen, depth affine, rotation from the first camera",
         "params": {"filename": "sfm_frames.csv"}},
        {"field": "track_depth", "visualizer": "field_map",
         "caption": "MoGe depth at every track (frames × tracks) — blank = unseen"},
        {"field": "depths", "visualizer": "field_map", "params": {"prop": "depth"},
         "caption": "MoGe metric depth per frame (preview, downsampled)"},
        {"field": "sfm", "visualizer": "json", "caption": "sfm box status"},
        {"field": "*", "visualizer": "json"},
    ],
}

_PREVIEW_MAX = 320          # depth previews: longest side (the full maps stay server-side)


def _npy(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a))
    return buf.getvalue()


def _rgb(jpg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))


def _preview(d: np.ndarray) -> np.ndarray:
    s = max(1, int(np.ceil(max(d.shape) / _PREVIEW_MAX)))
    return np.ascontiguousarray(d[::s, ::s])


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray, shape) -> np.ndarray:
    """RGB (0..1) of ``img`` at pixel coords (u, v) given on a grid of
    ``shape`` (H, W) — the image may have another resolution."""
    H, W = shape
    x = np.clip((u * img.shape[1] / W).astype(int), 0, img.shape[1] - 1)
    y = np.clip((v * img.shape[0] / H).astype(int), 0, img.shape[0] - 1)
    return img[y, x].astype(np.float32) / 255.0


def run(ctx, data, params):
    from visionist_client import track_stream

    frames = list(data["images"])
    F = len(frames)
    if F < 3:
        raise ValueError(f"need at least 3 frames, got {F}")
    min_alive = min(int(params["min_alive"]), F)

    # ---- 1. lightglue: tracks ---------------------------------------------
    sid = f"webui-{ctx.job_id}"
    with ctx.step("lightglue · stream tracks", total=F + 2):
        lightglue = ctx.box("lightglue")
        tr = track_stream(lightglue, frames, window=int(params["window"]),
                          min_alive=min_alive, mode=params["mode"], session_id=sid,
                          max_keypoints=int(params["max_keypoints"]))
        lightglue.run(config={"lightglue": {"command": "reset",
                                            "parameters": {"session_id": sid}}})
        obs = np.asarray(tr.obs_matrix, dtype=np.float64)          # (2F, P)
        if obs.shape[1] == 0:
            raise ValueError("lightglue found no tracks across the frames "
                             f"(min_alive={min_alive}) — frames too far apart?")
        ctx.info(tracks=int(obs.shape[1]),
                 complete_tracks=int(np.isfinite(obs).all(0).sum()),
                 missing_in=round(float(np.isnan(obs).mean()), 3))

    # ---- 2. moge: depth + intrinsics per frame ----------------------------
    with ctx.step("moge · depth per frame", total=F):
        moge = ctx.box("moge")
        depths, Ks = [], []
        for jpg in frames:
            out = moge.run(data={"images": [jpg]},
                           config={"moge": {"command": "infer", "parameters": {
                               "resolution_level": int(params["resolution_level"])}}})
            cfg = out.config.get("moge", {})
            if cfg.get("status") != "done":
                raise RuntimeError(f"moge: {cfg.get('error') or cfg.get('status')}")
            r = out.results[0]
            d = np.asarray(r["depth"], dtype=np.float32).copy()
            if "mask" in r:
                d[np.asarray(r["mask"]) == 0] = np.nan              # invalid -> missing entries
            depths.append(d)
            Ks.append(np.asarray(r["intrinsics"], dtype=np.float64))
        if len({d.shape for d in depths}) != 1:
            raise RuntimeError(f"moge depth maps differ in size: {sorted({d.shape for d in depths})}")
        depths = np.stack(depths)                                    # (F, H, W)
        K_norm = np.median(np.stack(Ks), axis=0)
        H, W = depths.shape[1:]
        ctx.emit("depths", [{"depth": _preview(d)} for d in depths])

        # the frames the tracks live on may have another size than the maps
        img_h, img_w = _rgb(frames[0]).shape[:2]
        if (img_h, img_w) != (H, W):
            obs[0::2] *= W / img_w
            obs[1::2] *= H / img_h
        u = np.nan_to_num(obs[0::2]).round().astype(int).clip(0, W - 1)
        v = np.nan_to_num(obs[1::2]).round().astype(int).clip(0, H - 1)
        z = depths[np.arange(F)[:, None], v, u]
        z[~(np.isfinite(obs[0::2]) & np.isfinite(obs[1::2]))] = np.nan
        ctx.emit("track_depth", z.astype(np.float32))

    # ---- 3. sfm: reconstruct ----------------------------------------------
    with ctx.step("sfm · reconstruct", total=1):
        rec = ctx.box("sfm").run(
            data={"tracks": _npy(obs), "depths": _npy(depths), "intrinsics": _npy(K_norm)},
            config={"sfm": {"command": "reconstruct", "parameters": {
                "intrinsics_normalized": True, "solver": params["solver"]}}})
        cfg = rec.config.get("sfm", {})
        ctx.emit("sfm", {k: v for k, v in cfg.items() if k != "encoding"})
        if cfg.get("status") != "done":
            raise RuntimeError(f"sfm: {cfg.get('error') or cfg.get('status')}")
        ctx.info(solver=cfg.get("solver"), frames_kept=cfg.get("num_frames"),
                 points_kept=cfg.get("num_points"),
                 reprojection_px=(cfg.get("reprojection_error") or {}).get("median"))

    # ---- 4. scene: sparse + dense clouds, cameras --------------------------
    with ctx.step("scene · back-project depth"):
        cams = np.asarray(rec.cameras, dtype=np.float64)             # (F', 3, 4)
        pts = np.asarray(rec.points, dtype=np.float64)               # (P', 3)
        fids = np.asarray(rec.frame_ids, dtype=int)
        pids = np.asarray(rec.point_ids, dtype=int)
        observed = np.asarray(rec.observed, dtype=bool)              # (F', P')
        d_aff = np.asarray(rec.depth_scales, dtype=np.float64)
        o_aff = np.asarray(rec.depth_offsets, dtype=np.float64)
        K = K_norm.copy()
        K[0] *= W
        K[1] *= H
        rgb = [_rgb(f) for f in frames]

        # sparse colours: pixel under each track in the first kept frame it is seen
        first = np.argmax(observed, axis=0)
        fu = obs[2 * fids[first], pids]
        fv = obs[2 * fids[first] + 1, pids]
        colors = np.stack([_sample(rgb[fids[k]], np.array([a]), np.array([b]), (H, W))[0]
                           for k, a, b in zip(first, fu, fv)]) if len(pids) else np.zeros((0, 3))

        # dense: every kept depth map, corrected + back-projected (stride to budget)
        budget = int(params["dense_points"])
        dense_p, dense_c, dense_f = [], [], []
        if budget > 0:
            valid = np.isfinite(depths[fids]).sum()
            stride = max(1, int(np.ceil(np.sqrt(valid / budget))))
            vv, uu = np.mgrid[0:H:stride, 0:W:stride]
            rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                             np.ones_like(uu, dtype=np.float64)], -1)   # (h, w, 3)
            for k, f in enumerate(fids):
                lam = depths[f][::stride, ::stride]
                ok = np.isfinite(lam)
                Z = d_aff[k] * lam[ok] + o_aff[k]
                ok_z = Z > 0
                Xc = rays[ok][ok_z] * Z[ok_z, None]
                R, t = cams[k, :, :3], cams[k, :, 3]
                dense_p.append((Xc - t) @ R)                             # R^T (Xc - t)
                dense_c.append(_sample(rgb[f], uu[ok][ok_z], vv[ok][ok_z], (H, W)))
                dense_f.append(np.full(int(ok_z.sum()), k, dtype=np.float32))

        ctx.emit("scene", {
            "points": pts.astype(np.float32),
            "colors": colors.astype(np.float32),
            "cameras": cams.astype(np.float32),
            "intrinsics": K.astype(np.float32),
            "image_size": [int(W), int(H)],
            "dense_points": (np.concatenate(dense_p) if dense_p else np.zeros((0, 3))).astype(np.float32),
            "dense_colors": (np.concatenate(dense_c) if dense_c else np.zeros((0, 3))).astype(np.float32),
            "dense_frames": (np.concatenate(dense_f) if dense_f else np.zeros(0)).astype(np.float32),
        })

        rows = []
        for k, f in enumerate(fids):
            R = cams[k, :, :3]
            rows.append({
                "frame": int(f),
                "tracks_seen": int(np.isfinite(obs[2 * f]).sum()),
                "d": round(float(d_aff[k]), 4),
                "o": round(float(o_aff[k]), 4),
                "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
                "centre": [round(float(c), 4) for c in -R.T @ cams[k, :, 3]],
            })
        ctx.emit("frames", rows)
