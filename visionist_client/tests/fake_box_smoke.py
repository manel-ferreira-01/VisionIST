#!/usr/bin/env python3
"""In-process fake boxes (same protos the client uses) to verify
``visionist_client.Visionist`` end-to-end without real boxes running.

Two fake services are defined, both implementing the shared
``PipelineService.Process(Envelope) -> Envelope`` interface:

* **FakeTapnext**  -- reads ``data["images"]`` (bytes), returns tracks/visibles
  (mirrors the *real* tapnext box's request/response shape).
* **FakeSentences** -- reads ``data["sentences"]`` (strings), echoes them
  back in ``data["echoed"]`` and a JSON blob in ``data["meta"]``. This
  proves the client's generic path works for *non-image* boxes too.

Run:
    python visionist_client/tests/fake_box_smoke.py
"""

import io
import json
import os
import sys
import concurrent.futures as futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
try:
    import torch
except ImportError:  # optional extra -- the torch-dependent case 1 is then skipped
    torch = None

from visionist_client._pb_loader import get as _get_pb
from visionist_client import Visionist, trace


pb2, pb2_grpc, aux = _get_pb()


class FakeTapnext(pb2_grpc.PipelineServiceServicer):
    """Mimics the real tapnext box (images in, tracks/visibles out)."""

    def Process(self, request, context):
        cfg = json.loads(request.config_json) if request.config_json else {}
        t = cfg.get("tapnext", {})
        if t.get("command") == "reset":
            return pb2.Envelope(config_json=json.dumps(
                {"tapnext": {"status": "done", "action": "reset"}}))

        imgs = aux.unwrap_value(request.data.get("images"))
        if not isinstance(imgs, list):
            return pb2.Envelope(config_json=json.dumps(
                {"tapnext": {"status": "error", "error": "no images"}}))
        n = len(imgs)
        tracks = torch.full((n, 16, 2), float(n))
        vis = torch.ones((n, 16))
        b1, b2 = io.BytesIO(), io.BytesIO()
        torch.save(tracks, b1, pickle_protocol=4)
        torch.save(vis, b2, pickle_protocol=4)
        env = pb2.Envelope(config_json=json.dumps(
            {"tapnext": {"status": "done", "frames_processed": n}}))
        env.data["tracks"].CopyFrom(aux.wrap_value(b1.getvalue()))
        env.data["visibles"].CopyFrom(aux.wrap_value(b2.getvalue()))
        return env


class FakeSentences(pb2_grpc.PipelineServiceServicer):
    """A hypothetical non-image envelope box: takes ``sentences`` (strings),
    echoes them back. Proves the generic client path is not image-only."""

    def Process(self, request, context):
        cfg = json.loads(request.config_json) if request.config_json else {}
        scfg = cfg.get("sentences", {})
        if scfg.get("command") == "reset":
            return pb2.Envelope(config_json=json.dumps(
                {"sentences": {"status": "done", "action": "reset"}}))

        vals = aux.unwrap_value(request.data.get("sentences"))
        if not isinstance(vals, list) or not vals:
            return pb2.Envelope(config_json=json.dumps(
                {"sentences": {"status": "error", "error": "no sentences"}}))
        env = pb2.Envelope(config_json=json.dumps(
            {"sentences": {"status": "done", "n": len(vals)}}))
        env.data["echoed"].CopyFrom(aux.wrap_value(list(vals)))
        # A JSON payload -- exercises decode_util's JSON branch in Result.
        meta = json.dumps({"got": list(vals)}).encode("utf-8")
        env.data["meta"].CopyFrom(aux.wrap_value(meta))
        return env


def _serve(servicer, name):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_PipelineServiceServicer_to_server(servicer, server)
    names = (
        pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(names, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    print(f"[{name}] fake box @ 127.0.0.1:{port}")
    return server, port


def main() -> int:
    tap_next_server, tap_port = _serve(FakeTapnext(), "tapnext")
    sent_server,     sent_port = _serve(FakeSentences(), "sentences")
    failures = []
    try:
        # ------------------------------------------------------------- case 1
        # tapnext: existing behavior (images -> tracks/visibles).
        print("\n== case 1: tapnext trace (images in, tracks out) ==")
        if torch is None:
            print("  SKIP -- torch not installed (pip install 'visionist-client[torch]')")
            b = None
        else:
            b = Visionist(f"127.0.0.1:{tap_port}")
        if b is not None:
            try:
                info = b.info()
                assert info["reachable"], f"info: {info}"
                assert info["reflection"], f"info: {info}"
                frames = [b"jpeg-frame-%d" % i for i in range(4)]
                res = trace(b, images=frames, grid_size=10)
                assert res.config.get("tapnext", {}).get("status") == "done"
                assert res.tracks.shape == (4, 16, 2), res.tracks.shape
                assert res.visibles.shape == (4, 16)
                print("  OK  -- tracks", res.tracks.shape, "visibles", res.visibles.shape)
            finally:
                b.close()

        # ------------------------------------------------------------- case 2
        # Generic Visionist.run() against a *non-image* box (sentences).
        print("\n== case 2: generic run() on a non-image box (sentences) ==")
        b2 = Visionist(f"127.0.0.1:{sent_port}", config_key="sentences")
        try:
            # reset (config-only call: data is empty)
            r_reset = b2.reset()
            assert r_reset.config.get("sentences", {}).get("action") == "reset", r_reset.config
            print("  OK  -- reset:", r_reset.config)

            # track (generic run): data has a StringList, NOT a BytesList
            r = b2.run(
                data={"sentences": ["hello", "world", "boxes"]},
                config={"sentences": {"command": "track", "parameters": {}}},
                method="Process",
            )
            assert r.config.get("sentences", {}).get("status") == "done"
            echoed = r.fields["echoed"]
            assert echoed == ["hello", "world", "boxes"], echoed
            meta = r.fields["meta"]
            assert isinstance(meta, dict) and meta["got"] == ["hello", "world", "boxes"], meta
            print("  OK  -- echoed:", echoed, "meta(decoded):", meta)
        finally:
            b2.close()

        # ------------------------------------------------------------- case 3
        # int -> float coercion in the data payload.
        print("\n== case 3: int/float payload via Visionist.run ==")
        # We'll just build and inspect the Envelope the client would send,
        # rather than spin up another fake service (coercion lives in
        # envelope.build -> aux.wrap_value).
        from visionist_client.envelope import build
        env2 = build(data={"vals": [1, 2, 3]}, config={})
        assert env2.data["vals"].WhichOneof("kind") == "ff"
        vals = list(env2.data["vals"].ff.values)
        assert vals == [1.0, 2.0, 3.0], vals
        print("  OK  -- ints coerced to ff:", vals)

        # ------------------------------------------------------------- case 4
        # str in data (generic API) must be a *literal string*, not a path.
        print("\n== case 4: str in generic data = literal string ==")
        env3 = build(data={"s": "hello"}, config={})
        assert env3.data["s"].WhichOneof("kind") == "s"
        assert env3.data["s"].s == "hello"
        print("  OK  -- 's' is a literal string")

        print("\nPASS -- all fake-box smoke cases.")
        return 0
    finally:
        tap_next_server.stop(0)
        sent_server.stop(0)


if __name__ == "__main__":
    sys.exit(main())
