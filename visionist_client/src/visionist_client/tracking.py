"""Box-agnostic feature tracing: graph of match edges -> tracks -> observation matrix.

This module knows **no box, no gRPC, no field names**. It operates on two
inputs that any "stream matching" box can supply:

* **edges** -- a sequence of ``(ref_node, new_node)`` pairs, each a match
  between a keypoint in an earlier frame and a keypoint in a later one;
* **frame_kpts** -- ``frame_kpts[f][i] == (x, y)`` for keypoint ``i`` of
  frame ``f``.

It turns those into:

* **candidate point sets** -- connected components over the edges (union-find);
  a point occluded for a frame stays in the same set because a Δ=2/3 edge
  re-links it (gap-bridging);
* **tracks** -- one world point each, at most **one node per frame**, produced
  by repeatedly peeling the longest valid chain from each set (see
  :func:`peel_tracks`);
* an **observation matrix** ``P`` of shape ``(2*F, P)``: ``P[2*f]`` = x and
  ``P[2*f+1]`` = y of each track's point in frame ``f``, NaN where the point
  is absent (the tapnext / Tomasi-Kanade convention).

Why the peeling (and not "one component = one track")?  Union-find is
transitive, so two nearby-but-distinct keypoints that both match a shared
neighbor get glued into one set — a set with several nodes in the SAME frame
is **not one point** and cannot be an honest 2-D matrix column (a column
holds one (x, y) per frame).  Peeling extracts the longest one-point-per-frame
chain from each set; what was a single malformed blob becomes several valid
tracks, none of them thrown away.

Two association modes (:func:`build_observation`):

* ``"backbone"`` (default) -- peel using matches of **all** deltas. A point
  that blinks out for one frame is re-linked via a Δ=2/3 edge and stays one
  track (a NaN gap in its column).
* ``"greedy"`` -- peel using only **Δ=1** matches. A point that fails to match
  the immediate next frame dies, and its re-appearance is a new track (the
  behaviour of a naive one-step tracker).

Both consume the *same* edges; only which edges are traversable differ.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# A "node" is one observation: (frame_index, keypoint_index_in_that_frame).
Node = Tuple[int, int]
# An "edge" connects a reference node to a newer node: (ref_node, new_node).
Edge = Tuple[Node, Node]

# Valid association modes.
MODES = ("backbone", "greedy")


def union_find_groups(edges: Sequence[Edge]) -> List[List[Node]]:
    """Connected components over ``edges`` (union-find, path halving).

    Returns each component as a list of its nodes. Nodes with no edges are
    never present (a component needs at least one edge).
    """
    parent: Dict[Node, Node] = {}

    def find(x: Node) -> Node:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comp: Dict[Node, List[Node]] = collections.defaultdict(list)
    for n in list(parent):
        comp[find(n)].append(n)
    return list(comp.values())


def build_pred(
    edges: Sequence[Edge], max_delta: Optional[int] = None
) -> Dict[Node, List[Node]]:
    """Invert edges into a predecessor map: ``pred[new_node] = [ref_node, ...]``.

    If ``max_delta`` is given, only edges whose frame gap is ``<= max_delta``
    are included (``max_delta=1`` -> the greedy/Δ=1 graph; ``None`` -> backbone).
    """
    pred: Dict[Node, List[Node]] = collections.defaultdict(list)
    for a, b in edges:
        if max_delta is not None and (b[0] - a[0]) > max_delta:
            continue
        pred[b].append(a)
    return dict(pred)


def longest_path(remaining: Set[Node], pred: Dict[Node, List[Node]]) -> List[Node]:
    """Longest frame-increasing path over the subgraph induced by ``remaining``.

    Edges always go from an earlier to a later frame, so nodes are already in
    topological order when sorted, and a path automatically has **one node per
    frame** (frames strictly increase along it). Returns the path in time
    order (earliest frame first).
    """
    ns = sorted(remaining)
    best = dict.fromkeys(ns, 1)
    par: Dict[Node, Node] = {}
    for n in ns:
        for p in pred.get(n, ()):
            if p in best and best[p] + 1 > best[n]:
                best[n] = best[p] + 1
                par[n] = p
    end = max(ns, key=lambda n: (best[n], n[0]))
    path: List[Node] = []
    while True:
        path.append(end)
        if end not in par:
            break
        end = par[end]
    return path[::-1]


def peel_tracks(
    nodes: Sequence[Node],
    pred: Dict[Node, List[Node]],
    min_alive: int = 2,
) -> Tuple[List[List[Node]], List[List[Node]]]:
    """Repeatedly peel the longest one-point-per-frame chain from ``nodes``.

    Each extracted chain becomes a track.  Chains spanning ``>= min_alive``
    frames are kept, the rest are dropped (but still consumed, so the same
    point is not re-emitted). Returns ``(kept, dropped)``.
    """
    remaining = set(nodes)
    kept: List[List[Node]] = []
    dropped: List[List[Node]] = []
    while remaining:
        p = longest_path(remaining, pred)
        for x in p:
            remaining.discard(x)
        (kept if len(p) >= min_alive else dropped).append(p)
    return kept, dropped


def tracks_to_matrix(
    tracks: Sequence[Sequence[Node]],
    frame_kpts: Sequence[Sequence],
    n_frames: int,
) -> np.ndarray:
    """Assemble the ``(2*n_frames, len(tracks))`` observation matrix.

    ``frame_kpts[f][i]`` must be the ``(x, y)`` of keypoint ``i`` in frame
    ``f``. NaN marks frames a track does not occupy (including NaN gaps).
    """
    P = np.full((2 * n_frames, len(tracks)), np.nan, dtype=np.float32)
    for t, path in enumerate(tracks):
        for f, i in path:
            row = np.asarray(frame_kpts[f])[i]
            P[2 * f, t] = row[0]
            P[2 * f + 1, t] = row[1]
    return P


@dataclass
class TrackResult:
    """The output of :func:`build_observation` (and the lightglue convenience).

    ``obs_matrix`` is the final ``(2*F, kept_tracks)`` array; ``tracks`` are
    the kept tracks as lists of ``(frame, kpt)`` nodes. The remaining fields
    are summary stats (see :meth:`summary` for a printable report).
    """

    obs_matrix: np.ndarray
    tracks: List[List[Node]]
    num_frames: int
    mode: str
    min_alive: int
    window: Optional[int] = None
    n_components: int = 0
    n_merged_components: int = 0
    n_tracks_kept: int = 0
    n_tracks_dropped: int = 0
    n_gapped_tracks: int = 0
    per_frame_counts: List[int] = field(default_factory=list)
    longest_track_frames: int = 0

    def summary(self) -> List[str]:
        return [
            f"mode={self.mode} | min_alive={self.min_alive} | window={self.window} | frames={self.num_frames}",
            f"union-find components: {self.n_components} "
            f"({self.n_merged_components} glued >1 point into one frame)",
            f"tracks kept: {self.n_tracks_kept} | dropped (< min_alive): {self.n_tracks_dropped} "
            f"| gap-bridged: {self.n_gapped_tracks}",
            f"tracks per frame: {self.per_frame_counts}",
            f"longest track: {self.longest_track_frames} frames",
            f"obs_matrix: {self.obs_matrix.shape} (2F x tracks); NaN where a point is absent",
        ]

    def __str__(self) -> str:
        return "\n".join(self.summary())


def _distinct_frames(path: Sequence[Node]) -> int:
    return len({f for f, _ in path})


def build_observation(
    edges: Sequence[Edge],
    frame_kpts: Sequence[Sequence],
    n_frames: int,
    *,
    min_alive: int = 2,
    mode: str = "backbone",
    window: Optional[int] = None,
) -> TrackResult:
    """Turn match edges + per-frame keypoints into a cleaned observation matrix.

    Parameters
    ----------
    edges:
        ``(ref_node, new_node)`` match pairs (any deltas).
    frame_kpts:
        ``frame_kpts[f]`` indexable by keypoint index to ``(x, y)``.
    n_frames:
        Number of frames (``F``); sets the matrix height ``2*F``.
    min_alive:
        Keep tracks observed in at least this many frames.
    mode:
        ``"backbone"`` (default; all deltas, bridges 1-frame gaps) or
        ``"greedy"`` (Δ=1 only; dies + re-births).
    window:
        Echoed back for reporting (the box-side sliding window, if any).

    Returns
    -------
    TrackResult
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if min_alive < 1:
        raise ValueError(f"min_alive must be >= 1, got {min_alive!r}")

    groups = union_find_groups(edges)
    max_delta = None if mode == "backbone" else 1
    pred = build_pred(edges, max_delta=max_delta)

    kept: List[List[Node]] = []
    dropped: List[List[Node]] = []
    for nodes in groups:
        k, d = peel_tracks(nodes, pred, min_alive=min_alive)
        kept.extend(k)
        dropped.extend(d)
    kept.sort(key=lambda p: min(p))  # stable: earliest frame, then kpt index

    P = tracks_to_matrix(kept, frame_kpts, n_frames)

    n_merged = sum(
        1 for g in groups
        if max(collections.Counter(f for f, _ in g).values()) > 1
    )
    n_gapped = sum(
        1 for p in kept
        if (max(f for f, _ in p) - min(f for f, _ in p) + 1) > len(p)
    )
    per_frame = [0] * n_frames
    for p in kept:
        for f, _ in p:
            per_frame[f] += 1
    longest = max((_distinct_frames(p) for p in kept), default=0)

    return TrackResult(
        obs_matrix=P,
        tracks=kept,
        num_frames=n_frames,
        mode=mode,
        min_alive=min_alive,
        window=window,
        n_components=len(groups),
        n_merged_components=n_merged,
        n_tracks_kept=len(kept),
        n_tracks_dropped=len(dropped),
        n_gapped_tracks=n_gapped,
        per_frame_counts=per_frame,
        longest_track_frames=longest,
    )
