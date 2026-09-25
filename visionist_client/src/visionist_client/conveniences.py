"""Box-specific *conveniences* -- NOT part of the agnostic core.

The core of ``visionist_client`` (``box.py`` / ``envelope.py`` / ``result.py`` /
``decode_util.py``) is deliberately box-agnostic: it only knows *how* to build
an ``Envelope`` and send it over the shared ``PipelineService`` interface, and
*how* to read a ``Result`` back. It knows **no box name, field name, or model**.

This module is the opposite, on purpose: it encodes knowledge about a *specific*
box and only composes the generic ``Visionist.run`` / ``Visionist.reset`` primitives.  Today:

* :func:`trace` -- the tapnext point-tracking one-liner;
* :func:`track_stream` -- a lightglue sliding-window stream turned into a
  cleaned observation matrix, with the box-agnostic graph/track machinery
  delegated to :mod:`visionist_client.tracking`.

If you add a convenience for another box (``segment``, ``embed``, ``detect``,
...), put it here or in a sibling ``conveniences/<box>.py`` -- **never** in
``box.py`` -- so the core stays agnostic and each box's sugar is isolated,
visible, and easy to delete.
"""

from typing import Any, List, Optional, Sequence, Union
import pathlib

import numpy as np

from . import tracking

__all__ = ["trace", "load_images", "track_stream"]


def load_images(images: Union[str, bytes, "pathlib.PurePath", Sequence]) -> List[bytes]:
    """Normalize an ``images`` argument into a list of ``bytes``.

    ``trace`` is the *image-box* convenience, so a bare ``str`` here is treated
    as a **local file path** to serialize (distinct from the generic ``run``
    contract where ``str`` means a literal string).  Accepts:

    * a single path (``str``/``pathlib.Path``) or pre-encoded ``bytes``
    * a list of any of the above
    """
    if images is None:
        return []
    if isinstance(images, (str, bytes, bytearray, pathlib.PurePath)):
        images = [images]
    out: List[bytes] = []
    for item in images:
        if isinstance(item, (bytes, bytearray, memoryview)):
            out.append(bytes(item))
        elif isinstance(item, (str, pathlib.PurePath)):
            out.append(pathlib.Path(item).read_bytes())
        else:
            raise TypeError(
                f"Unsupported image element: {type(item)!r}. "
                "Pass a path (str/pathlib.Path), pre-encoded bytes, or a list of those."
            )
    return out


def trace(
    box,
    images: Union[str, bytes, "pathlib.PurePath", Sequence],
    *,
    grid_size: int = None,
    reset_first: bool = True,
    config_key: str = "tapnext",
    **params: Any,
):
    """Point-tracking convenience for the **tapnext** box.

    Built purely on the generic core -- it is equivalent to::

        box.run(data={"images": [bytes, ...]},
                config={config_key: {"command": "track", "parameters": {...}}},
                method="Process")

    with an optional preceding ``box.reset(config_key)`` (tapnext *accumulates*
    tracks across sequential requests, so a reset keeps a one-shot clean).

    Parameters
    ----------
    box:
        A :class:`visionist_client.Visionist` pointed at a tapnext box.
    images:
        A single local path / pre-encoded bytes, or a list of either.
    grid_size:
        TAPNext grid size (added to ``parameters``).
    reset_first:
        Send ``reset`` first (default ``True``).
    config_key:
        The box's config section name (default ``"tapnext"``).
    **params:
        Extra ``parameters`` (e.g. ``threshold=...``).
    """
    images_bytes = load_images(images)
    if not images_bytes:
        raise ValueError("trace(): no images provided")

    parameters: Any = dict(params or {})
    if grid_size is not None:
        parameters["grid_size"] = int(grid_size)

    if reset_first:
        box.reset(config_key)

    return box.run(
        data={"images": images_bytes},
        config={config_key: {"command": "track", "parameters": parameters}},
        method="Process",
    )


def track_stream(
    box,
    frames: Union[bytes, "pathlib.PurePath", Sequence],
    *,
    window: int = 3,
    min_alive: int = 2,
    mode: str = "backbone",
    session_id: Optional[str] = "default",
    reset_first: bool = True,
    config_key: str = "lightglue",
    **params: Any,
):
    """LightGlue sliding-window stream -> a cleaned observation matrix.

    Feeds ``frames`` (one image each, time-ordered) through the lightglue
    box's ``stream`` command inside a single session — the box keeps a
    sliding window of stored features and, per call, matches the new frame
    against the last ``window`` of them (reporting ``matches_1..window``).
    This convenience collects those match edges and builds a Tomasi-Kanade
    observation matrix (``2F x P``: x and y per frame, one column per
    tracked point, NaN where absent) with the box-agnostic tracker in
    :mod:`visionist_client.tracking`.

    Built purely on the generic core (``box.run`` / ``box.reset``) plus the
    lightglue response contract; the heavy lifting (union-find, backbone
    peeling, matrix) lives in :mod:`visionist_client.tracking` and is
    unit-testable without a box.

    Parameters
    ----------
    box:
        A :class:`visionist_client.Visionist` pointed at a lightglue box.
    frames:
        A sequence of frames (one per step): local path (str / pathlib.Path),
        or pre-encoded bytes; a single one is accepted too.
    window:
        The box-side sliding window (number of stored references the new
        frame is matched against; 1..16, default 3).
    min_alive:
        Keep tracks observed in at least this many frames (default 2).
    mode:
        ``"backbone"`` (default) — union-find candidate sets + longest-path
        peeling; bridges 1-frame gaps via the box's Δ=2/3 matches. ``"greedy"``
        — Δ=1 chains only; a point that fails to match the next frame dies and
        a re-appearing one is a new track. See :mod:`visionist_client.tracking`.
    session_id:
        The box-side stream session key (so concurrent callers don't collide);
        default ``"default"``.
    reset_first:
        Reset that stream session before starting (default ``True``).
    config_key:
        The box's config section name (default ``"lightglue"``).
    **params:
        Extra ``parameters`` forwarded to the box (``feature_extractor``,
        ``max_keypoints``, ``filter_threshold``, ``device``, ...).

    Returns
    -------
    visionist_client.tracking.TrackResult
        With ``obs_matrix`` (``2F x kept``), ``tracks`` (lists of
        ``(frame, kpt)`` nodes), summary stats, and ``.summary()`` for a
        printable report.

    Example
    -------
    >>> res = track_stream(b, frames=["f0.jpg", "f1.jpg", "f2.jpg"], window=3)
    >>> res.obs_matrix.shape
    (6, <kept>)
    >>> print(res)  # doctest: +SKIP
    """
    frames_bytes = load_images(frames)
    if not frames_bytes:
        raise ValueError("track_stream(): no frames provided")
    n_frames = len(frames_bytes)

    parameters: Any = dict(params or {})
    parameters["session_id"] = session_id if session_id is not None else "default"
    parameters["window"] = int(window)

    if reset_first:
        if session_id is not None:
            # Reset the specific session (leave other callers' sessions intact).
            box.run(
                config={config_key: {"command": "reset",
                                     "parameters": {"session_id": parameters["session_id"]}}},
            )
        else:
            box.reset(config_key)

    edges = []
    frame_kpts: List = [None] * n_frames
    for k, fb in enumerate(frames_bytes):
        r = box.run(
            data={"images": [fb]},
            config={config_key: {"command": "stream", "parameters": parameters}},
        )
        sec = r.config.get(config_key, {}) if isinstance(r.config, dict) else {}
        if not sec.get("status"):
            raise RuntimeError(f"track_stream(): frame {k}: no {config_key!r} status in {r.config!r}")
        if sec.get("status") != "done":
            raise RuntimeError(f"track_stream(): frame {k} failed: {sec}")

        j_win = int(sec.get("window", 0))
        kp = r.fields.get("keypoints")
        if kp is None:
            raise RuntimeError(f"track_stream(): frame {k}: response has no 'keypoints'")
        frame_kpts[k] = kp[j_win]  # the new frame is the J-th (last) row

        for j in range(1, j_win + 1):
            arr = r.fields.get(f"matches_{j}")
            if arr is None:
                continue
            m = np.asarray(arr).astype(int)  # (K, 2) = [ref_kpt, new_kpt]
            for i in range(m.shape[0]):
                edges.append(((k - j, int(m[i, 0])), (k, int(m[i, 1]))))

    return tracking.build_observation(
        edges,
        frame_kpts,
        n_frames,
        min_alive=min_alive,
        mode=mode,
        window=int(window),
    )
