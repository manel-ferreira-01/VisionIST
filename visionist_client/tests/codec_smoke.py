#!/usr/bin/env python3
"""Codec tests: the *declared* payload-encoding contract (see ``CODECS.md``).

Runs in-process fake boxes (same shared protos) that declare an
``"encoding"`` field in their response ``config_json``, and checks the
client decodes the payload via the named codec instead of guessing:

1. ``encoding: "zstd_pickle"`` (lang_segm-style)   -> ``res.results`` is a
   decoded **list**, type ``list`` (not ``bytes``/``ndarray``).
2. ``encoding: "json"`` (string form)              -> JSON bytes field decodes.
3. per-field map ``encoding: {field: "json"}``     -> only that field decodes.
4. unknown codec name                             -> raw **bytes**, no raise,
   and ``res.encoding`` still exposes the declared name.
5. no ``encoding`` (legacy)                        -> old auto-chain unchanged
   (torch tensor decodes when torch is installed; JSON still auto-decodes).
6. unit: ``decode_with`` round-trips torch / zstd_pickle / numpy / identity,
   and degrades to raw bytes (never raises) when a codec's library is missing
   or the name is unknown.

``torch`` is an optional extra: the torch-dependent pieces of cases 5/6 are
skipped (not failed) when it is not installed, so the test also passes on a
plain ``pip install visionist-client``.

Run:
    python visionist_client/tests/codec_smoke.py
"""

import io
import json
import os
import sys
import warnings
import concurrent.futures as futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import pickle
import zstandard as zstandard
try:
    import torch
except ImportError:  # optional extra -- the torch-dependent cases are skipped
    torch = None

from visionist_client import Visionist
from visionist_client.codec import CODECS, decode_with
from visionist_client._pb_loader import get as _get_pb

pb2, pb2_grpc, aux = _get_pb()

# A deterministic payload in the style of LangSAM ``predict()`` outputs.
_OUT_LIST = [
    {"bboxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
     "scores": [0.9, 0.8], "text_labels": ["a dog"]},
    {"bboxes": [], "scores": [], "text_labels": []},
]


def _zstd_pickle(obj) -> bytes:
    return zstandard.ZstdCompressor().compress(pickle.dumps(obj))


class FakeLangSegm(pb2_grpc.PipelineServiceServicer):
    """Mirrors the real lang_segm contract: zstd(pickle) in ``data.results``
    and the *declared* ``"encoding"`` in the box section."""

    def Process(self, request, context):
        if not request.data.get("images"):
            return pb2.Envelope(config_json=json.dumps(
                {"codecbox": {"status": "empty_request"}}))
        n = len(aux.unwrap_value(request.data["images"]))
        env = pb2.Envelope(config_json=json.dumps({
            "codecbox": {
                "status": "done",
                "num_images": n,
                "encoding": "zstd_pickle",   # declared payload encoding
            }}))
        env.data["results"].CopyFrom(aux.wrap_value(_zstd_pickle(_OUT_LIST[:n])))
        return env


class FakeJsonBox(pb2_grpc.PipelineServiceServicer):
    """A box whose one heavy field is a JSON document."""

    def __init__(self, section, encoding, field):
        self.section = section
        self.encoding = encoding
        self.field = field

    def Process(self, request, context):
        payload = json.dumps({"got": 42, "items": [1, 2, 3]}).encode("utf-8")
        env = pb2.Envelope(config_json=json.dumps(
            {self.section: {"status": "done", "encoding": self.encoding}}))
        env.data[self.field].CopyFrom(aux.wrap_value(payload))
        return env


class FakeUnknownCodec(pb2_grpc.PipelineServiceServicer):
    """Declares a codec this client doesn't know -> graceful raw bytes."""

    def Process(self, request, context):
        env = pb2.Envelope(config_json=json.dumps(
            {"unknownbox": {"status": "done", "encoding": "snappy_fancy"}}))
        env.data["payload"].CopyFrom(aux.wrap_value(b"opaque-bytes"))
        return env


class FakeLegacy(pb2_grpc.PipelineServiceServicer):
    """Declares *nothing*: the pre-coercion auto-chain must still work."""

    def Process(self, request, context):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        blob = io.BytesIO()
        torch.save(t, blob, pickle_protocol=4)
        env = pb2.Envelope(config_json=json.dumps({"legacybox": {"status": "done"}}))
        env.data["tensors"].CopyFrom(aux.wrap_value(blob.getvalue()))
        meta = json.dumps({"k": "v"}).encode("utf-8")
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


def unit_roundtrips() -> int:
    """Pure unit tests for decode_with (no gRPC)."""
    print("\n== unit: decode_with round-trips ==")

    # identity
    assert decode_with(b"abc", None) == b"abc"
    assert decode_with(b"abc", "identity") == b"abc"

    # json
    assert decode_with(json.dumps([1, "x"]).encode(), "json") == [1, "x"]

    # numpy float32 buffer
    import numpy as np
    arr = np.array([1.5, -2.5, 3.25], dtype=np.float32)
    back = decode_with(arr.tobytes(), "numpy")
    assert isinstance(back, np.ndarray) and back.tolist() == [1.5, -2.5, 3.25], back

    # torch payloads (Tensor + dict) -- only when torch is installed
    if torch is not None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        blob = io.BytesIO(); torch.save(t, blob, pickle_protocol=4)
        out = decode_with(blob.getvalue(), "torch")
        assert isinstance(out, torch.Tensor) and out.tolist() == t.tolist(), out

        # torch dict payload (the codec keeps the object, the old guess dropped it)
        d = {"tracks": torch.zeros(2, 4)}
        blob = io.BytesIO(); torch.save(d, blob, pickle_protocol=4)
        out = decode_with(blob.getvalue(), "torch")
        assert isinstance(out, dict) and set(out) == {"tracks"}
        assert isinstance(out["tracks"], torch.Tensor) and torch.equal(
            out["tracks"], torch.zeros(2, 4))

    # zstd_pickle (the lang_segm format)
    out = decode_with(_zstd_pickle(_OUT_LIST), "zstd_pickle")
    assert out == _OUT_LIST and isinstance(out, list), type(out)

    # unknown codec -> raw bytes, no exception
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert decode_with(b"xyz", "snappy_fancy") == b"xyz"

    # missing library degrades to raw bytes + warning, no raise.
    # Simulated by poisoning sys.modules so `import torch` raises ImportError.
    # (Without torch installed, any bytes do -- the codec raises ImportError
    # before it ever parses the payload.)
    if torch is not None:
        blob = io.BytesIO(); torch.save(torch.zeros(1), blob, pickle_protocol=4)
    else:
        blob = io.BytesIO(b"opaque-torch-bytes")
    saved_torch, saved_zstd = sys.modules.get("torch"), sys.modules.get("zstandard")
    sys.modules["torch"] = None
    sys.modules["zstandard"] = None
    try:
        got = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = decode_with(blob.getvalue(), "torch")
        assert got == blob.getvalue(), (type(got), got)
        assert any("torch" in str(x.message) for x in w), w
        got = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = decode_with(_zstd_pickle(_OUT_LIST), "zstd_pickle")
        assert got == _zstd_pickle(_OUT_LIST)
        assert any("zstandard" in str(x.message) for x in w), w
    finally:
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        else:
            del sys.modules["torch"]
        if saved_zstd is not None:
            sys.modules["zstandard"] = saved_zstd
        else:
            del sys.modules["zstandard"]

    assert set(CODECS) == {"identity", "json", "torch", "numpy", "zstd_pickle"}
    print("  OK  -- all decode_with round-trips, degrade paths, registry names")
    return 0


def main() -> int:
    servers = []
    try:
        # ------------------------------------------------------------- case 1
        print("\n== case 1: declared zstd_pickle (lang_segm-style) ==")
        srv, port = _serve(FakeLangSegm(), "lang_segm")
        servers.append(srv)
        b = Visionist(f"127.0.0.1:{port}")
        try:
            res = b.run(
                data={"images": [b"jpeg-frame"]},
                config={"codecbox": {"command": "segment", "text_prompt": ["a dog"]}},
            )
            assert res.config["codecbox"]["status"] == "done"
            assert res.encoding == "zstd_pickle", res.encoding
            assert isinstance(res.results, list), type(res.results)
            assert type(res.results) is list and not isinstance(res.results, bytes)
            assert not hasattr(res.results, "shape"), "must not be bytes/ndarray"
            assert res.results == _OUT_LIST[:1], res.results
            print(f"  OK  -- results is a decoded list of {len(res.results)} dict(s)")
        finally:
            b.close()

        # ------------------------------------------------------------- case 2
        print("\n== case 2: declared json (string form) ==")
        srv, port = _serve(
            FakeJsonBox("jsonbox", "json", "meta"), "jsonbox-str")
        servers.append(srv)
        b = Visionist(f"127.0.0.1:{port}")
        try:
            res = b.run(data={}, config={"jsonbox": {"command": "work"}})
            assert res.encoding == "json"
            assert res.meta == {"got": 42, "items": [1, 2, 3]}, res.meta
            print("  OK  -- meta decoded to:", res.meta)
        finally:
            b.close()

        # ------------------------------------------------------------- case 3
        print("\n== case 3: declared json (per-field map) ==")
        srv, port = _serve(
            FakeJsonBox("mapbox", {"meta": "json"}, "meta"), "jsonbox-map")
        servers.append(srv)
        b = Visionist(f"127.0.0.1:{port}")
        try:
            res = b.run(data={}, config={"mapbox": {"command": "work"}})
            assert res.encoding == {"meta": "json"}
            assert res.meta == {"got": 42, "items": [1, 2, 3]}, res.meta
            print("  OK  -- per-field map honored")
        finally:
            b.close()

        # ------------------------------------------------------------- case 4
        print("\n== case 4: unknown codec -> raw bytes, no exception ==")
        srv, port = _serve(FakeUnknownCodec(), "unknownbox")
        servers.append(srv)
        b = Visionist(f"127.0.0.1:{port}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the degradation warning is expected
                res = b.run(data={}, config={"unknownbox": {"command": "work"}})
            assert res.encoding == "snappy_fancy", res.encoding
            assert res.payload == b"opaque-bytes" and isinstance(res.payload, bytes)
            print("  OK  -- raw bytes preserved; res.encoding exposed,",
                  "no exception")
        finally:
            b.close()

        # ------------------------------------------------------------- case 5
        print("\n== case 5: no encoding -> legacy auto-chain unchanged ==")
        if torch is None:
            print("  SKIP -- torch not installed (pip install 'visionist-client[torch]')")
        else:
            srv, port = _serve(FakeLegacy(), "legacybox")
            servers.append(srv)
            b = Visionist(f"127.0.0.1:{port}")
            try:
                res = b.run(data={}, config={"legacybox": {"command": "work"}})
                assert res.encoding is None
                assert isinstance(res.tensors, torch.Tensor) and res.tensors.shape == (2, 3)
                assert res.meta == {"k": "v"}
                print("  OK  -- torch tensor + JSON still auto-decoded")
            finally:
                b.close()

        # ------------------------------------------------------------- unit
        unit_roundtrips()

        print("\nPASS -- all codec smoke cases.")
        return 0
    finally:
        for s in servers:
            s.stop(0)


if __name__ == "__main__":
    sys.exit(main())
