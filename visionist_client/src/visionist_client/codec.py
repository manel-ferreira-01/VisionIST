"""Named payload codecs: the *declared* decoding path (see ``CODECS.md``).

The client is smart about *shape* and dumb about *content*: a box says how
its payload is encoded by adding an ``"encoding"`` field to its response
``config_json`` (a codec name for every ``bytes`` field, or a
``{field_name: codec_name}`` map for mixed responses). The core only knows the
generic ``encoding`` keyword and this registry of codecs — **never a box
name**.

Codec vocabulary (all pure ``bytes -> object``):

==================  =========================================================
name                behavior
==================  =========================================================
``identity``        raw bytes, unchanged (the default)
``json``            UTF-8 JSON document -> ``list``/``dict``/scalar
``torch``           ``torch.save`` bytes -> the unpickled object
                    (Tensor or dict of Tensors); needs ``torch``
``numpy``           ``numpy.save`` (``.npy``) blob -> the array with its
                    shape/dtype restored, or a plain raw float32 buffer ->
                    1-D ``np.ndarray`` (legacy)
``zstd_pickle``     ``zstd.compress(pickle.dumps(obj))`` -> ``obj``;
                    needs ``zstandard``
==================  =========================================================

A codec whose library is missing (``torch`` / ``zstandard``) degrades to
``identity`` (raw bytes) plus a warning — **never raises**: the client always
returns *something* usable.
"""

import io
import json
import pickle
import warnings

import numpy as np

__all__ = ["CODECS", "decode_with"]


def _decode_identity(buf: bytes) -> bytes:
    return buf


def _decode_json(buf: bytes):
    """Strict: declared ``json`` fields must parse (or we degrade in
    ``decode_with``); no "does it look like JSON?" sniffing."""
    s = buf.decode("utf-8")
    return json.loads(s)


def _decode_torch(buf: bytes):
    """``torch.save`` blob -> whatever object was saved (Tensor or dict).

    Propagates (rather than hides) decode failures so ``decode_with`` can
    degrade with a warning instead of silently guessing."""
    import torch  # noqa: W061  -- ImportError here means the lib is missing

    return torch.load(io.BytesIO(buf), weights_only=False, map_location="cpu")


def _decode_numpy(buf: bytes) -> "np.ndarray":
    """Preferred: a ``numpy.save`` (``.npy``) blob — the format opencv_box
    emits via ``np_to_bytes`` — decoded with its shape and dtype intact.
    Fallback: a plain raw float32 buffer -> 1-D array (the original
    contract; the codec_smoke round-trip)."""
    try:
        return np.load(io.BytesIO(buf), allow_pickle=False)
    except Exception:
        return np.frombuffer(buf, dtype=np.float32)


def _decode_zstd_pickle(buf: bytes):
    """``zstandard`` compress + ``pickle`` -> the decoded Python object
    (normally a list). The lang_segm -> folder_wd cross-box contract."""
    import zstandard  # noqa: W061  -- ImportError here means the lib is missing

    return pickle.loads(zstandard.ZstdDecompressor().decompress(buf))


#: The named registry: generic codec name -> pure ``bytes -> object`` function.
#: No box name appears here or below; boxes *select* a codec by declaring it.
CODECS = {
    "identity": _decode_identity,
    "json": _decode_json,
    "torch": _decode_torch,
    "numpy": _decode_numpy,
    "zstd_pickle": _decode_zstd_pickle,
}


def decode_with(payload, name):
    """Apply a *named* codec to ``payload`` (raw ``bytes``/``bytearray``).

    - ``name`` is ``None``/``"identity"`` -> raw bytes, unchanged (the
      agnostic default).
    - unknown name, or a codec whose library is missing, or a payload the
      codec cannot parse -> a warning is emitted and the **raw bytes** are
      returned. ``decode_with`` never raises.
    """
    if name is None or name == "identity":
        return bytes(payload)
    fn = CODECS.get(name)
    if fn is None:
        warnings.warn(
            f"codec {name!r} is not a known codec (known: {sorted(CODECS)}); "
            "returning raw bytes",
            stacklevel=2,
        )
        return bytes(payload)
    try:
        return fn(payload)
    except Exception as e:
        kind = "library is not installed" if isinstance(e, ImportError) else "decode failed"
        warnings.warn(
            f"codec {name!r}: {kind} ({type(e).__name__}: {e}); "
            "returning raw bytes",
            stacklevel=2,
        )
        return bytes(payload)
