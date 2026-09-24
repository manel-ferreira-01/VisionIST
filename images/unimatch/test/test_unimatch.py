#!/usr/bin/env python3
"""Test script for the unimatch gRPC service (shared envelope interface).

Connects to a running unimatch box (default localhost:8061) and:
  1. ``flow`` on the bundled demo pair
     -> ``flow`` [H, W, 2] float32  (shape / finiteness / plausibility)
  2. ``flow`` with ``pred_bidir_flow`` + ``fwd_bwd_check``
     -> ``bwd_flow`` + ``occ_fwd``/``occ_bwd`` masks
  3. ``stereo`` / ``depth`` contract: with only the flow checkpoint baked in
     they must answer a clean ``error`` naming the expected weights file
     (no crash, no opaque traceback)
  4. ``empty_request`` / ``error`` contract (unknown command, 1 image)
  5. ``command: reset`` (standard no-op)

Run (from the repo root, server already up):
    python images/unimatch/test/test_unimatch.py
    BOX_HOST=10.0.0.5:8061 python images/unimatch/test/test_unimatch.py
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
    print(f"loaded {os.path.basename(path)}: {len(data) / (1024 * 1024):.2f} MB")
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
    sec = cfg.get("unimatch")
    if not isinstance(sec, dict):
        print(f"  !! response config not namespaced under 'unimatch': {cfg}")
    elif sec.get("status") == "error":
        print(f"  !! {sec.get('error')}")
    return sec if isinstance(sec, dict) else {}


def np_of(response, name) -> np.ndarray:
    return np.load(io.BytesIO(bytes(aux.unwrap_value(response.data[name]))),
                   allow_pickle=False)


def proc(stub, config, images=None):
    """One Process call, returning (decoded 'unimatch' section, Envelope)."""
    env = pipeline_pb2.Envelope(config_json=json.dumps(config))
    if images is not None:
        env = pipeline_pb2.Envelope(
            config_json=json.dumps(config),
            data={"images": aux.wrap_value(images)})
    res = stub.Process(env)
    return section(res), res


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)

    img0 = load_local_image(os.path.join(_TEST_DIR, "flow_0.jpg"))
    img1 = load_local_image(os.path.join(_TEST_DIR, "flow_1.jpg"))

    # ------------------------------------------------------------- 1. flow
    print("\n[1] command: flow (image pair)")
    sec, res = proc(stub, {"unimatch": {"command": "flow"}}, [img0, img1])
    assert sec.get("status") == "done", sec
    assert sec.get("num_images") == 2
    assert "flow" in res.data, f"missing 'flow' field: {list(res.data)}"
    flow = np_of(res, "flow")
    assert flow.ndim == 3 and flow.shape[-1] == 2, flow.shape
    assert np.isfinite(flow).all(), "flow contains non-finite values"
    assert flow.std() > 0, "flow is degenerate (all zeros)"
    print(f"  flow shape {flow.shape}, std {flow.std():.2f} px — OK")

    # -------------------------------------------- 2. bidirectional + occ
    print("\n[2] command: flow (pred_bidir_flow + fwd_bwd_check)")
    sec, res = proc(
        stub,
        {"unimatch": {"command": "flow",
                      "parameters": {"pred_bidir_flow": True,
                                     "fwd_bwd_check": True}}},
        [img0, img1])
    assert sec.get("status") == "done", sec
    bwd = np_of(res, "bwd_flow")
    occ_fwd = np_of(res, "occ_fwd")
    occ_bwd = np_of(res, "occ_bwd")
    assert bwd.shape == flow.shape
    assert occ_fwd.ndim == 2 and occ_bwd.ndim == 2
    assert set(np.unique(occ_fwd)).issubset({0.0, 1.0})
    frac = float(occ_fwd.mean())
    print(f"  bwd {bwd.shape}, occ_fwd mean {frac:.3f} — OK")

    # ---------------------------------------------- 3. stereo / depth (API)
    print("\n[3] command: stereo (no checkpoint baked in -> clean error)")
    sec, _ = proc(stub, {"unimatch": {"command": "stereo"}}, [img0, img1])
    assert sec.get("status") == "error" and "gmstereo" in sec.get("error", ""), \
        sec
    print(f"  clean error: {sec.get('error', '')[:90]}… — OK")

    print("\n[4] command: depth (no checkpoint baked in -> clean error)")
    sec, _ = proc(stub, {
        "unimatch": {"command": "depth",
                     "parameters": {
                         "intrinsics": [500.0, 500.0, 320.0, 240.0]}}},
        [img0])
    assert sec.get("status") == "error" and "gmdepth" in sec.get("error", ""), \
        sec
    print(f"  clean error: {sec.get('error', '')[:90]}… — OK")

    # ------------------------------------------------- 5. contract basics
    print("\n[5] empty_request / error contract")
    sec, _ = proc(stub, {"unimatch": {"command": "flow"}})
    assert sec.get("status") == "empty_request", sec
    sec, _ = proc(stub, {"unimatch": {"command": "nope"}}, [img0, img1])
    assert sec.get("status") == "error" and "unknown command" in \
        sec.get("error", ""), sec
    sec, _ = proc(stub, {"unimatch": {"command": "flow"}}, [img0])
    assert sec.get("status") == "error" and "2 images" in sec.get("error", ""), \
        sec
    print("  empty_request + error responses — OK")

    print("\n[6] command: reset (stateless no-op)")
    sec, _ = proc(stub, {"unimatch": {"command": "reset"}})
    assert sec.get("status") == "done" and sec.get("action") == "reset", sec
    print("  reset acknowledged — OK")

    print(f"\nAll unimatch box checks passed against {target}.")
    channel.close()


if __name__ == "__main__":
    main()
