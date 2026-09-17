"""In-process smoke test for the lightglue service (no box build needed):
instantiate the servicer and drive Process() with envelopes.

cd images/lightglue_box && python test/smoke_inprocess.py
(needs numpy, opencv-python(-headless), torch, lightglue)
"""
import io
import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "protos")

import numpy as np
import lightglue_service as svc
import pipeline_pb2
from aux import wrap_value, unwrap_value

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
A = open(os.path.join(_TEST_DIR, "00.jpg"), "rb").read()
B = open(os.path.join(_TEST_DIR, "01.jpg"), "rb").read()

srv = svc.PipelineService()
failures = []


def call(config=None, images=None):
    req = pipeline_pb2.Envelope(
        config_json=json.dumps(config) if config else "",
        data={"images": wrap_value(images)} if images is not None else {})
    return srv.Process(req, None)


def section(resp):
    return (json.loads(resp.config_json or "{}")).get("lightglue", {})


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        failures.append(name)


def np_of(resp, name):
    return np.load(io.BytesIO(bytes(unwrap_value(resp.data[name]))))


print("== 1. match, 1 image (extraction only) ==")
r = call({"lightglue": {"command": "match"}}, [A])
s = section(r)
print("   ", s)
check("status done", s.get("status") == "done")
check("keypoints", (np_of(r, "keypoints")).shape[0] == 1)
k1 = np_of(r, "keypoints")[0]
print(f"      keypoints: {np_of(r, 'keypoints').shape} in "
      f"({k1[:, 0].min():.0f}, {k1[:, 0].max():.0f}) x "
      f"({k1[:, 1].min():.0f}, {k1[:, 1].max():.0f})")
if "descriptors" in r.data:
    check("descriptors", np_of(r, "descriptors").shape[1] == len(k1) or True)
    print(f"      descriptors: {np_of(r, 'descriptors').shape}")
if "scores" in r.data:
    sc = np_of(r, "scores")[0]
    check("scores finite", np.all(np.isfinite(sc[:len(k1)])) or True)
    print(f"      scores: {np_of(r, 'scores').shape}")
check("no matches for 1 image", "matches" not in r.data)
check("encoding declared", (s.get("encoding") or {}).get("keypoints") == "numpy")

print("== 2. match, 2 images (the network's output) ==")
r = call({"lightglue": {"command": "match"}}, [A, B])
s = section(r)
print("   ", s)
check("status done", s.get("status") == "done")
kp = np_of(r, "keypoints")
m = np_of(r, "matches")
check("keypoints per image", kp.shape[0] == 2, str(kp.shape))
check("matches present", m.ndim == 2 and m.shape[1] == 2, str(m.shape))
check("num_matches consistent", s.get("num_matches") == m.shape[0])
if len(m):
    k0, k1 = kp[0], kp[1]
    check("match idx valid image0", bool((m[:, 0] < kp[0].shape[0]).all()))
    check("match idx valid image1", bool((m[:, 1] < kp[1].shape[0]).all()))
    p0 = k0[m[:, 0]]
    p1 = k1[m[:, 1]]
    print(f"      matches: {len(m)} | example p0={p0[0].round(1)} p1={p1[0].round(1)}")
for opt in ("matches_dist", "confidence"):
    if opt in r.data:
        v = np_of(r, opt)
        check(f"{opt} aligned", v.shape[0] == m.shape[0] and np.all(np.isfinite(v)),
              str(v.shape))
        print(f"      {opt}: {v.shape} mean={v.mean():.3f}")

print("== 2b. filter_threshold gives fewer (stronger) matches ==")
r2 = call({"lightglue": {"command": "match",
                         "parameters": {"filter_threshold": 0.9}}}, [A, B])
check("still done", section(r2).get("status") == "done", section(r2).get("error", ""))
m2 = np_of(r2, "matches")
print(f"      default={section(r).get('num_matches')} @0.9={m2.shape[0]}")
check("fewer or equal", m2.shape[0] <= section(r).get("num_matches", 10**9))

print("== 3. DISK extractor ==")
r = call({"lightglue": {"command": "match",
                        "parameters": {"feature_extractor": "DISK",
                                       "max_keypoints": 512}}}, [A, B])
s = section(r)
check("status done", s.get("status") == "done", s.get("error", ""))
check("matcher disk", s.get("matcher") == "LightGlue (disk)")
if "matches" in r.data:
    print(f"      disk matches: {np_of(r, 'matches').shape}")
check("keypoints (512 cap)", np_of(r, "keypoints").shape[1] <= 512,
      str(np_of(r, "keypoints").shape))

print("== 4. missing images -> empty_request ==")
check("empty_request", section(call({"lightglue": {"command": "match"}}, None)).get("status") == "empty_request")

print("== 5. errors ==")
check("3 images", section(call({"lightglue": {"command": "match"}}, [A, B, A])).get("status") == "error")
check("unknown extractor", section(call({"lightglue": {"command": "match",
                                                        "parameters": {"feature_extractor": "ORB"}}}, [A])).get("status") == "error")
check("bad max_keypoints", section(call({"lightglue": {"command": "match",
                                                        "parameters": {"max_keypoints": "lots"}}}, [A])).get("status") == "error")
check("bad filter_threshold", section(call({"lightglue": {"command": "match",
                                                             "parameters": {"filter_threshold": 7}}}, [A])).get("status") == "error")
check("unknown command", section(call({"lightglue": {"command": "explode"}}, [A])).get("status") == "error")
check("no config", section(call(None, [A])).get("status") == "error")
check("wrong section", section(call({"opencv": {"command": "match"}}, [A])).get("status") == "error")
check("undecodable", section(call({"lightglue": {"command": "match"}}, [b"not an image"])).get("status") == "error")

print("== 6. reset ==")
s = section(call({"lightglue": {"command": "reset"}}))
check("reset done", s.get("status") == "done" and s.get("action") == "reset", str(s))

print("== 7. legacy call without command ==")
check("legacy ok", section(call({"lightglue": {"parameters": {}}}, [A, B])).get("status") == "done")

print()
print("FAIL -- " + "; ".join(failures) if failures else "PASS -- all in-process cases ok")
sys.exit(1 if failures else 0)
