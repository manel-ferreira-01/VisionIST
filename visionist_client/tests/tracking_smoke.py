#!/usr/bin/env python3
"""Unit tests for the box-agnostic tracker (:mod:`visionist_client.tracking`).

All cases are synthetic (hand-made match edges + per-frame keypoints), so no
box, GPU, or gRPC is needed. They pin down the association semantics:

* union-find grouping of match edges;
* **backbone** peeling: one node per frame, longest chains first, merged
  blobs split into their constituent tracks, min_alive filtering;
* **greedy** (Δ=1) peeling: a gap breaks the chain (dies + re-births);
* the gap-bridging difference between the two modes on identical edges;
* observation-matrix layout (x / y rows, NaN where a track skips a frame).

Run:
    python visionist_client/tests/tracking_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from visionist_client import tracking
from visionist_client.tracking import (
    union_find_groups,
    build_pred,
    peel_tracks,
    longest_path,
    tracks_to_matrix,
    build_observation,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def one(frame, x, y, n=1):
    """A per-frame keypoint array whose index 0 is (x, y); `n` rows total."""
    a = np.zeros((max(n, 1), 2), dtype=np.float32)
    a[0] = (x, y)
    return a


# ---------------------------------------------------------------------------
# Scenario A -- two frame-0 keypoints both match one frame-1 keypoint
# (the classic union-find glue), which then continues to frame 2.
# ---------------------------------------------------------------------------
print("== A: merged blob -> backbone split ==")
fkA = [
    np.array([[100., 200.], [102., 201.]]),   # frame 0: kpt 0 and kpt 1
    np.array([[110., 210.]]),                 # frame 1: kpt 0
    np.array([[120., 220.]]),                 # frame 2: kpt 0
]
edgesA = [((0, 0), (1, 0)), ((0, 1), (1, 0)), ((1, 0), (2, 0))]

groups = union_find_groups(edgesA)
check("A one component", len(groups) == 1, str(groups))
check("A component has 4 nodes", len(groups[0]) == 4, str(groups[0]))

rA = build_observation(edgesA, fkA, 3, min_alive=2, mode="backbone")
check("A kept == 1 track", rA.n_tracks_kept == 1, str(rA.tracks))
check("A dropped == 1", rA.n_tracks_dropped == 1)
check("A merged component flagged", rA.n_merged_components == 1)
check("A kept track = (0,0)-(1,0)-(2,0)",
      set(rA.tracks[0]) == {(0, 0), (1, 0), (2, 0)}, str(rA.tracks))
check("A one node per frame", len({f for f, _ in rA.tracks[0]}) == len(rA.tracks[0]))
check("A per-frame counts [1,1,1]", rA.per_frame_counts == [1, 1, 1], str(rA.per_frame_counts))
check("A longest 3", rA.longest_track_frames == 3)
check("A no gaps", rA.n_gapped_tracks == 0)
np.testing.assert_array_equal(rA.obs_matrix.ravel(), [100, 200, 110, 210, 120, 220])

# min_alive sensitivity
rA1 = build_observation(edgesA, fkA, 3, min_alive=1, mode="backbone")
check("A min_alive=1 keeps both", rA1.n_tracks_kept == 2, str(rA1.tracks))
rA4 = build_observation(edgesA, fkA, 3, min_alive=4, mode="backbone")
check("A min_alive=4 drops all", rA4.n_tracks_kept == 0 and rA4.obs_matrix.shape == (6, 0))

# ---------------------------------------------------------------------------
# Scenario B -- a 1-frame gap: (0,0)->(1,0), then (1,0)->(3,0) (Δ=2) and
# (2,0)->(3,0).  Backbone must bridge the gap; greedy must break at it.
# ---------------------------------------------------------------------------
print("\n== B: gap bridging — backbone vs greedy ==")
fkB = [one(0, 100, 200), one(0, 300, 400), one(0, 500, 600), one(0, 700, 800)]
edgesB = [((0, 0), (1, 0)), ((1, 0), (3, 0)), ((2, 0), (3, 0))]

rB = build_observation(edgesB, fkB, 4, min_alive=2, mode="backbone")
check("B backbone kept == 1 (bridged track)", rB.n_tracks_kept == 1, str(rB.tracks))
check("B bridged track = (0,0)-(1,0)-(3,0)",
      set(rB.tracks[0]) == {(0, 0), (1, 0), (3, 0)}, str(rB.tracks))
check("B gap flagged", rB.n_gapped_tracks == 1)
check("B per-frame [1,1,0,1]", rB.per_frame_counts == [1, 1, 0, 1], str(rB.per_frame_counts))
expected = np.array([100, 200, 300, 400, np.nan, np.nan, 700, 800], dtype=np.float32)
got = rB.obs_matrix.ravel()
check("B matrix NaN gap", bool(np.isnan(got[4]) and np.isnan(got[5])) and
      np.allclose(got[~np.isnan(got)], expected[~np.isnan(expected)]), str(got))

rB_g = build_observation(edgesB, fkB, 4, min_alive=2, mode="greedy")
check("B greedy kept == 2 (broken at the gap)", rB_g.n_tracks_kept == 2, str(rB_g.tracks))
check("B greedy no gap-bridged track", rB_g.n_gapped_tracks == 0)
check("B greedy never joins (0,0) and (3,0)",
      all(not ({(0, 0), (3, 0)} <= set(t)) for t in rB_g.tracks), str(rB_g.tracks))
check("B greedy per-frame [1,1,1,1]", rB_g.per_frame_counts == [1, 1, 1, 1])
check("B greedy shapes", rB_g.obs_matrix.shape == (8, 2))

# ---------------------------------------------------------------------------
# Lower-level primitives
# ---------------------------------------------------------------------------
print("\n== primitives ==")
pred_all = build_pred(edgesB)
pred_d1 = build_pred(edgesB, max_delta=1)
check("pred all has 3 predecessors total", sum(len(v) for v in pred_all.values()) == 3)
check("pred Δ=1 drops the Δ=2 edge", sum(len(v) for v in pred_d1.values()) == 2, str(pred_d1))
check("longest_path on full set",
      longest_path(set((n for g in groups for n in g)), pred_all)[:2] and
      set(longest_path({(2, 0), (3, 0)}, pred_d1)) == {(2, 0), (3, 0)})
predA = build_pred(edgesA)
kept, dropped = peel_tracks(groups[0], predA, min_alive=2)
check("peel_tracks split", [len(k) for k in kept] == [3] and len(dropped) == 1)

# empty input
rE = build_observation([], [one(0, 1, 1)], 1, min_alive=2, mode="backbone")
check("empty edges -> (2,0) matrix", rE.obs_matrix.shape == (2, 0) and rE.n_components == 0)

# two disconnected components stay separate
edgesC = [((0, 0), (1, 0)), ((0, 1), (1, 1))]
check("two components", len(union_find_groups(edgesC)) == 2)

# TrackResult.summary / str
lines = rB.summary()
check("summary lines are str", all(isinstance(s, str) for s in lines) and len(lines) >= 5)
check("__str__ works", len(str(rB)) > 0)

# validation errors
try:
    build_observation(edgesA, fkA, 3, mode="nope")
    check("bad mode raises", False)
except ValueError:
    check("bad mode raises", True)
try:
    build_observation(edgesA, fkA, 3, min_alive=0)
    check("min_alive<1 raises", False)
except ValueError:
    check("min_alive<1 raises", True)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print("FAIL -- " + "; ".join(FAILURES))
    sys.exit(1)
print("PASS -- all tracking unit cases ok")
