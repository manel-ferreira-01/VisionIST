#!/usr/bin/env python3
"""Test script for the opencv gRPC service (shared envelope interface).

Connects to a running opencv box and:
  1. ``match`` (SIFT) the two bundled test images -> keypoints/descriptors +
     matches_inliers_* + fundamental_matrix (all ``np.save`` blobs, declared
     ``numpy``), a single-image extraction call, and the legacy no-command
     form
  2. verifies the declared ``encoding`` map in the response config
  3. exercises lightglue (SuperPoint) when the image build has it (graceful
     error otherwise)
  4. ``empty_request`` / ``error`` contract (missing images, unknown command,
     bad parameters, undecodable frame)
  5. ``command: reset`` as a standard no-op

Run (from the repo or image root, server already up):
    python images/opencv_box/test/test_opencv.py
    BOX_HOST=10.0.0.5:8061 python images/opencv_box/test/test_opencv.py

Note: no box build required — ``smoke_inprocess.py`` drives the same
service code in-process (needs cv2 locally).
"""

import io
import json
import os
import sys

# Make the protos/ folder importable (same pattern as the clip test).
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


def check_status(response):
    cfg = json.loads(response.config_json or "{}")
    section = cfg.get("opencv")
    if not isinstance(section, dict):
        print(f"  ERROR: response config not namespaced under 'opencv': {cfg}")
        return None
    if section.get("status") == "error":
        print(f"  ERROR: {section.get('error')}")
    return section


def decode_np(value) -> np.ndarray:
    """The box declares these ``numpy`` (np.save blobs): decode back to
    an array, shape and dtype intact."""
    return np.load(io.BytesIO(bytes(value)), allow_pickle=False)


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)

    failures = []
    img_00 = load_local_image(os.path.join(_TEST_DIR, "00.jpg"))
    img_01 = load_local_image(os.path.join(_TEST_DIR, "01.jpg"))

    # ------------------------------------------------------------------ #
    # Case 1: match (SIFT) — two images -> full matching payload         #
    # ------------------------------------------------------------------ #
    print("\n== case 1: match (SIFT), 2 images ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({
            "opencv": {
                "command": "match",
                "parameters": {"feature_extractor": "SIFT", "max_keypoints": 1000},
            }
        }),
        data={"images": aux.wrap_value([img_00, img_01])},
    ))
    section = check_status(response)
    if section is None:
        return 1
    if section.get("status") != "done":
        print(f"  status not done: {section}")
        failures.append("case1 status")
    expected = ("keypoints", "descriptors", "matches_inliers_a",
                "matches_inliers_b", "fundamental_matrix")
    for f in expected:
        if f not in response.data:
            print(f"  missing field: {f}")
            failures.append(f"case1 {f} missing")
        else:
            arr = decode_np(aux.unwrap_value(response.data[f]))
            print(f"  {f:20s}: shape={arr.shape} dtype={arr.dtype}")
            if f in ("keypoints", "descriptors") and arr.shape[0] != 2:
                failures.append(f"case1 {f} per-image axis")
    enc = section.get("encoding", {})
    for f in expected:
        if enc.get(f) != "numpy":
            print(f"  {f} not declared numpy: {enc.get(f)!r}")
            failures.append(f"case1 encoding {f}")
    print(f"  runtime={section.get('runtime'):.3f}s matcher={section.get('matcher')} "
          f"inliers={section.get('num_inliers')}")

    # ------------------------------------------------------------------ #
    # Case 2: match — single image (extraction only) + legacy no-command #
    # ------------------------------------------------------------------ #
    print("\n== case 2: match, 1 image (extraction only) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "match",
                                           "parameters": {"max_keypoints": 300}}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "done":
        failures.append("case2 status")
    else:
        if "matches_inliers_a" in response.data or "fundamental_matrix" in response.data:
            print("  single image should not carry matching fields")
            failures.append("case2 matching fields")
        if "keypoints" not in response.data:
            failures.append("case2 keypoints missing")
        else:
            arr = decode_np(aux.unwrap_value(response.data["keypoints"]))
            if arr.shape[0] != 1:
                print(f"  single-image keypoints shape {arr.shape} != (1, N, 2)")
                failures.append("case2 keypoints shape")

    print("\n== case 2b: legacy call without command (compat) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"parameters": {"feature_extractor": "ORB"}}}),
        data={"images": aux.wrap_value([img_00, img_01])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "done":
        failures.append("case2b legacy")

    # ------------------------------------------------------------------ #
    # Case 3: lightglue (SuperPoint) — needs the shipped image build     #
    # ------------------------------------------------------------------ #
    print("\n== case 3: match (LightGlue/SuperPoint) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {
            "command": "match",
            "parameters": {"feature_extractor": "SUPERPOINT", "max_keypoints": 512},
        }}),
        data={"images": aux.wrap_value([img_00, img_01])},
    ))
    section = check_status(response)
    if section is None or section.get("status") == "error":
        err = (section or {}).get("error", "")
        if "LightGlue" in err:
            print(f"  SKIPPED — box build has no LightGlue ({err}); graceful, ok")
        else:
            failures.append("case3 lightglue")
    else:
        if section.get("matcher", "").startswith("LightGlue"):
            print(f"  matcher: {section.get('matcher')}")
        for f in ("keypoints", "matches_inliers_a", "fundamental_matrix"):
            if f in response.data:
                arr = decode_np(aux.unwrap_value(response.data[f]))
                print(f"  {f:20s}: shape={arr.shape}")
        print(f"  inliers: {section.get('num_inliers')}")

    # ------------------------------------------------------------------ #
    # Case 4: error / empty_request contract                              #
    # ------------------------------------------------------------------ #
    print("\n== case 4: empty_request + errors ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "match"}})))
    section = check_status(response)
    if section is None or section.get("status") != "empty_request":
        failures.append("case4 empty_request")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "explode"}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error" or "explode" not in str(section.get("error")):
        failures.append("case4 unknown command")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "match",
                                           "parameters": {"max_keypoints": "lots"}}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error":
        failures.append("case4 bad parameter")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "match"}}),
        data={"images": aux.wrap_value([b"not an image"])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error":
        failures.append("case4 undecodable")

    # ------------------------------------------------------------------ #
    # Case 5: reset (standard no-op on this fully stateless box)          #
    # ------------------------------------------------------------------ #
    print("\n== case 5: reset command ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"opencv": {"command": "reset"}})))
    section = check_status(response)
    if section is None or section.get("status") != "done" \
            or section.get("action") != "reset":
        failures.append("case5 reset")
    else:
        print(f"  ok: {section}")

    channel.close()

    print("\n" + ("FAIL -- " + "; ".join(failures) if failures
                  else "PASS -- all opencv test cases."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
