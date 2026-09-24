"""Best-effort decoding of Envelope data values — the *legacy auto* path.

Boxes send heavy payloads as raw ``bytes``. The old client *guessed* the
encoding; that guessing is kept only as a **legacy fallback** so
un-migrated boxes keep working. The documented path is the box-declared
``"encoding"`` keyword decoded by the named codecs in
:mod:`visionist_client.codec` (see ``CODECS.md``): when a field has a declared
codec, :mod:`visionist_client.result` uses :func:`codec.decode_with` directly
and this guessing never runs.

Guess chain (legacy): JSON (if it looks like JSON) -> torch (if installed)
-> numpy (if 4-byte aligned and not an image) -> raw ``bytes``. Nothing here
raises: the client always returns *something*.

.. note::
    The guessing is imperfect: the numpy branch fires on *any*
    4-byte-aligned blob and can return a plausible-looking but wrong float
    array. If you control the box, declare ``"encoding"`` in its response
    config instead of relying on guessing.

The codec *bodies* live once in :mod:`visionist_client.codec`; this module
wraps them with the legacy "try it, or fall through" behaviour.
"""

from .codec import _decode_json, _decode_numpy, _decode_torch  # noqa: F401  (shared bodies / back-compat)


def _try_numpy(buf: bytes):
    try:
        return _decode_numpy(buf)
    except Exception:
        pass
    try:
        # legacy extra: the uint8 fallback kept for pre-declaration behaviour
        import numpy as np
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.nbytes == 0:
            return None
        return arr
    except Exception:
        return None


def _try_torch(buf: bytes):
    """Legacy: only accepts a plain Tensor payload (the original contract);
    dict-of-tensors go through the declared ``torch`` codec instead."""
    try:
        import torch
    except ImportError:
        return None
    try:
        obj = _decode_torch(buf)
        if isinstance(obj, torch.Tensor):
            return obj
    except Exception:
        pass
    return None


def _try_json(buf: bytes):
    try:
        s = buf.decode("utf-8")
    except Exception:
        return None
    if not s or s[0] not in "{[":
        return None
    try:
        return _decode_json(buf)
    except Exception:
        return None


def decode_payload(buf):
    """Return the decoded value of ``buf``.

    Order (legacy fallback only — boxes are expected to declare
    ``"encoding"`` in the response config instead):
    JSON (if it looks like it) -> torch (tapnext / vggt) -> numpy
    (opencv_box ``np_to_bytes``) -> raw ``bytes``.
    Nothing raises: the client always returns *something*.
    """
    if buf is None:
        return None
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        return buf
    buf = bytes(buf)

    # JSON first: a JSON document is never a valid torch/numpy buffer.
    # _try_json only fires when the first byte is '{' or '[' — cheap and safe.
    j = _try_json(buf)
    if j is not None:
        return j

    # torch next — tapnext / vggt use torch.save.
    t = _try_torch(buf)
    if t is not None:
        return t

    # numpy next — opencv_box uses np_to_bytes; only attempt when the size is
    # at least one element and not clearly a JPEG.
    if len(buf) >= 4 and len(buf) % 4 == 0 and buf[:2] not in (b"\xff\xd8", b"\x89P"):
        arr = _try_numpy(buf)
        if arr is not None:
            return arr

    return buf
