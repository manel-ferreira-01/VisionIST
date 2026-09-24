"""Serialize a decoded ``visionist_client.Result`` into JSON + artifacts.

Decoding is *always* the box's job (its declared ``encoding`` codec, via
``visionist_client``).  This module only shapes the decoded objects for the SPA:

* small JSON-able values (scalars, ≤ 65 536 elements)   -> inline
* numeric arrays/masks/tensors beyond that              -> raw buffer artifact (``kind: "buffer"``)
* opaque bytes (GLB, images, …)                          -> file artifact (``kind: "file"``)
* exotic objects                                         -> pickle artifact (never raises)

The "client always returns something" rule survives all the way into the UI:
unknown structures degrade to a downloadable pickle with a note, not a 500.
"""

from __future__ import annotations

import io
import math
import pickle
from typing import Any, Optional

from .artifact import ArtifactStore

MAX_INLINE_ELEMENTS = 65_536     # per-array inline cap (tolist)
MAX_LIST_ITEMS = 10_000          # per-list recursion cap
MAX_DICT_KEYS = 10_000
MAX_DEPTH = 12

_JSON_TYPES = (str, bool, int, float, type(None))


# --------------------------------------------------------------------------
# MIME sniffing (enough for the common boxes)
# --------------------------------------------------------------------------

def sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"glTF":
        return "model/gltf-binary"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"PK\x03\x04":
        return "application/zip"
    if data[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


# --------------------------------------------------------------------------
# Serializers
# --------------------------------------------------------------------------

def _file_ref(token: str, content_type: str, size: int, extra: Optional[dict] = None) -> dict:
    out = {
        "kind": "file",
        "url": ArtifactStore.url(token),
        "mime": content_type,
        "size": size,
    }
    if extra:
        out["extra"] = extra
    return out


def _array_inline(v, dtype_name: str, shape: list, values: Any) -> dict:
    return {"kind": "array", "dtype": dtype_name, "shape": list(shape), "values": values}


def _buffer_ref(store: ArtifactStore, buf: bytes, dtype_name: str, shape: list) -> dict:
    tok = store.add(buf, "application/x-binary",
                    extra={"dtype": dtype_name, "shape": list(shape)})
    return {
        "kind": "buffer",
        "url": ArtifactStore.url(tok),
        "dtype": dtype_name,
        "shape": list(shape),
        "size": len(buf),
    }


def _pickle_ref(store: ArtifactStore, v: Any, note: str) -> dict:
    blob = pickle.dumps(v, protocol=4)
    tok = store.add(blob, "application/x-python-obj",
                    extra={"pytype": type(v).__name__, "note": note})
    return _file_ref(tok, "application/x-python-obj", len(blob),
                     extra={"pytype": type(v).__name__, "note": note})


_TORCH_DTYPE = {
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
}
_NUMPY_OK = ("float16", "float32", "float64", "int8", "uint8", "int16",
             "int32", "int64", "bool")


def _ser_tensor(v, store: ArtifactStore) -> dict:
    t = v.detach().cpu() if hasattr(v, "detach") else v
    try:
        import numpy as np
        arr = t.numpy()
    except Exception:
        arr = None
    dtype_name = _TORCH_DTYPE.get(str(v.dtype))
    shape = list(getattr(v, "shape", []))
    numel = 1
    for s in shape:
        numel *= int(s)
    if arr is not None:
        dname = arr.dtype.name
        if dname in _NUMPY_OK:
            if numel <= MAX_INLINE_ELEMENTS:
                return _array_inline(None, dname, shape, arr.tolist())
            try:
                return _buffer_ref(store, arr.tobytes(), dname, shape)
            except Exception:
                pass
    # fallback: inline if small, else pickle
    try:
        if numel <= MAX_INLINE_ELEMENTS:
            return _array_inline(None, dtype_name or "object", shape, t.tolist())
    except Exception:
        pass
    return _pickle_ref(store, v, "tensor that could not be array-ified")


def _ser_ndarray(v, store: ArtifactStore) -> dict:
    dname = v.dtype.name
    shape = list(v.shape)
    if dname in _NUMPY_OK:
        if v.size <= MAX_INLINE_ELEMENTS:
            return _array_inline(None, dname, shape, v.tolist())   # bools stay bools
        return _buffer_ref(store, v.tobytes(), dname, shape)
    if dname == "object":
        vals = []
        for row in (v.flat if v.ndim else v):  # type: ignore
            vals.append(_ser(row, store))
        return vals
    return _pickle_ref(store, v, f"object ndarray ({dname})")


def _ser(v: Any, store: ArtifactStore, depth: int = 0) -> Any:
    if v is None or isinstance(v, _JSON_TYPES):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        tok = store.add(b, sniff_mime(b))
        return _file_ref(tok, sniff_mime(b), len(b))
    if isinstance(v, dict):
        if len(v) > MAX_DICT_KEYS or depth > MAX_DEPTH:
            return _pickle_ref(store, v, f"dict with {len(v)} keys")
        return {k: _ser(val, store, depth + 1) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_LIST_ITEMS or depth > MAX_DEPTH:
            return _pickle_ref(store, list(v), f"list with {len(v)} items")
        return [_ser(x, store, depth + 1) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return _ser_ndarray(v, store)
    except ImportError:
        pass
    if hasattr(v, "tolist") and hasattr(v, "shape"):
        return _ser_tensor(v, store)
    # Unknown type: degrade, never raise.
    return _pickle_ref(store, v, f"object of type {type(v).__name__}")


# --------------------------------------------------------------------------
# Status extraction (generic contract keys, not box names)
# --------------------------------------------------------------------------

def _extract_status(config: Any) -> dict:
    out = {"status": None, "error": None, "runtime": None, "extra": {}}
    if not isinstance(config, dict):
        out["extra"] = config
        return out
    section = None
    for val in config.values():
        if isinstance(val, dict) and "status" in val:
            section = val
            break
    if section is None:
        out["extra"] = config
        return out
    out["status"] = section.get("status")
    if out["status"] == "error" or "error" in section:
        out["error"] = section.get("error")
    if "runtime" in section:
        out["runtime"] = section["runtime"]
    out["extra"] = {k: v for k, v in section.items()
                    if k not in ("status", "error", "runtime", "encoding")}
    # keep any sibling sections for display (rare, but e.g. echoed config)
    for k, v in config.items():
        if k not in out["extra"].keys() and not (isinstance(v, dict) and "status" in v):
            out["extra"][k] = v
    return out


def serialize_result(res: Any, store: ArtifactStore) -> dict:
    """``res`` is a ``visionist_client.Result``.  Returns a JSON-safe dict plus
    the artifacts it referenced."""
    fields = {}
    for name, value in (res.fields or {}).items():
        fields[name] = _ser(value, store)
    status = _extract_status(res.config)
    return {
        "fields": fields,
        "status": status["status"],
        "error": status["error"],
        "runtime": status["runtime"],
        "config_extra": status["extra"],
        "declared_encoding": res.encoding,
    }


__all__ = ["serialize_result", "sniff_mime",
           "MAX_INLINE_ELEMENTS", "MAX_LIST_ITEMS", "MAX_DICT_KEYS"]
