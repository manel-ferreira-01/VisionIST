#!/usr/bin/env python3
"""Test script for the lightglue gRPC service (shared envelope interface).

Connects to a running lightglue box (default localhost:8061) and:
  1. ``match`` two images            -> keypoints/descriptors/scores +
                                        matches/confidence (all ``numpy``)
  2. verifies the declared ``encoding`` map and index validity
  3. single-image extraction, DISK extractor, filter_threshold
  4. ``empty_request`` / ``error`` contract
  5. ``command: reset`` (standard no-op)

Run (from the repo root, server already up):
    python images/lightglue_box/test/test_lightglue.py
    BOX_HOST=10.0.0.5:8061 python images/lightglue_box/test/test_lightglue.py
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import aux  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402

import numpy as np  # noqa: E402


def load_local_image(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    print(f"loaded {os.path.basename(path)}: {len(data) / (1024*1024):.2f} MB")
    return data


def make_stub(target: str):
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    return pipeline_pb2_grpc.PipelineServiceStub(channel), channel


def section(response):
    cfg = json.loads(response.config_json or "{}")
    sec = cfg.get("lightglue")
    if not isinstance(sec, dict):
        print(f"  ERROR: response config not namespaced under 'lightglue': {cfg}")
    elif sec.get("status") == "error":
        print(f"  ERROR: {sec.get('error')}")
    return sec


def np_of(response, name) -> np.ndarray:
    return np.load(io.BytesIO(bytes(aux.unwrap_value(response.data[name]))),
                   allow_pickle=False)


def proc(stub, config, images=None):
    """One Process call, returning the decoded 'lightglue' section."""
    env = pipeline_pb2.Envelope(config_json=json.dumps(config))
    if images is not None:
        env = pipeline_pb2.Envelope(
            config_json=json.dumps(config),
            data={"images": aux.wrap_value(images)})
    return section(stub.Process(env))


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)

    failures = []
    a = load_local_image(os.path.join(_TEST_DIR, "00.jpg"))
    b = load_local_image(os.path.join(_TEST_DIR, "01.jpg"))

    # ----------------------------------------------------------------------
    # Case 1: match, 2 images -> network output                          #
    # ----------------------------------------------------------------------
    print("\n== case 1: match (SuperPoint), 2 images ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"lightglue": {"command": "match"}}),
        data={"images": aux.wrap_value([a, b])},
    ))
    sec = section(response)
    if sec is None:
        return 1
    if sec.get("status") != "done":
        print(f"  status not done: {sec}")
        return 1
    kp = np_of(response, "keypoints")
    m = np_of(response, "matches")
    conf = np_of(response, "confidence")
    print(f"  keypoints: {kp.shape} | matches: {m.shape} | confidence: {conf.shape}")
    if kp.shape[0] != 2:
        failures.append("case1 keypoints axis")
    if sec.get("num_matches") != m.shape[0]:
        failures.append("case1 num_matches")
    if not ((m[:, 0] < kp[0].shape[0]).all() and (m[:, 1] < kp[1].shape[0]).all()):
        failures.append("case1 match indices")
    if conf.shape[0] != m.shape[0] or not np.all(np.isfinite(conf)):
        failures.append("case1 confidence")
    enc = sec.get("encoding", {})
    for f in ("keypoints", "descriptors", "scores", "matches", "confidence"):
        if enc.get(f) != "numpy":
            failures.append(f"case1 encoding {f}: {enc.get(f)!r}")
    print(f"  runtime={sec.get('runtime'):.3f}s matcher={sec.get('matcher')} "
          f"device={sec.get('device')} inliers-free matches={m.shape[0]}")
    if len(m):
        pA, pB = kp[0][m[0, 0]], kp[1][m[0, 1]]
        print(f"  example pair: A={pA.round(1)} B={pB.round(1)}")

    # ----------------------------------------------------------------------
    # Case 2: single image (extraction only)                             #
    # ----------------------------------------------------------------------
    print("\n== case 2: match, 1 image (extraction only) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"lightglue": {"command": "match",
                                              "parameters": {"max_keypoints": 512}}}),
        data={"images": aux.wrap_value([a])},
    ))
    sec = section(response)
    if sec is None or sec.get("status") != "done":
        failures.append("case2 status")
    else:
        if "matches" in response.data:
            failures.append("case2 matches present")
        kp1 = np_of(response, "keypoints")
        if kp1.shape[0] != 1 or kp1.shape[1] > 512:
            failures.append(f"case2 keypoints {kp1.shape}")
        print(f"  keypoints: {kp1.shape}")

    # ----------------------------------------------------------------------
    # Case 3: DISK + filter_threshold                                    #
    # ----------------------------------------------------------------------
    print("\n== case 3: DISK + filter_threshold ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"lightglue": {"command": "match",
                                              "parameters": {"feature_extractor": "DISK",
                                                             "max_keypoints": 512,
                                                             "filter_threshold": 0.05}}}),
        data={"images": aux.wrap_value([a, b])},
    ))
    sec = section(response)
    if sec is None or sec.get("status") != "done":
        failures.append("case3 status")
    else:
        if sec.get("feature_extractor") != "DISK":
            failures.append("case3 extractor")
        print(f"  disk matches: {np_of(response, 'matches').shape} "
              f"(filter_threshold=0.05)")

    # ----------------------------------------------------------------------
    # Case 4: empty_request / errors                                     #
    # ----------------------------------------------------------------------
    print("\n== case 4: empty_request + errors ==")
    sec = proc(stub, {"lightglue": {"command": "match"}})
    if sec is None or sec.get("status") != "empty_request":
        failures.append("case4 empty_request")

    sec = proc(stub, {"lightglue": {"command": "explode"}})
    if sec is None or sec.get("status") != "error":
        failures.append("case4 unknown command")

    sec = proc(stub, {"lightglue": {"command": "match",
                                    "parameters": {"feature_extractor": "SIFT"}}}, [a])
    if sec is None or sec.get("status") != "error" or "SIFT" not in str(sec.get("error")):
        failures.append("case4 bad extractor")

    sec = proc(stub, {"lightglue": {"command": "match"}}, [a, b, a])
    if sec is None or sec.get("status") != "error" or "2 images" not in str(sec.get("error")):
        failures.append("case4 three images")

    sec = proc(stub, {"lightglue": {"command": "match"}}, [b"not an image"])
    if sec is None or sec.get("status") != "error":
        failures.append("case4 undecodable")

    # ----------------------------------------------------------------------
    # Case 5: reset (standard no-op)                                     #
    # ----------------------------------------------------------------------
    print("\n== case 5: reset command ==")
    sec = proc(stub, {"lightglue": {"command": "reset"}})
    if sec is None or sec.get("status") != "done" or sec.get("action") != "reset":
        failures.append("case5 reset")
    else:
        print(f"  ok: {sec}")

    channel.close()

    print("\n" + ("FAIL -- " + "; ".join(failures) if failures
                  else "PASS -- all lightglue test cases."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
