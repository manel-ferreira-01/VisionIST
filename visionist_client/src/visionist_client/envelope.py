"""Envelope builders for the shared ``pipeline.PipelineService`` boxes.

Every box in the *envelope family* (tapnext, yolo, vggt, opencv_box, lang_segm,
...) speaks::

    service PipelineService {
      rpc Process( Envelope ) returns ( Envelope );   # the shared contract RPC
    }

``build()`` turns a flat Python ``data`` dict + a ``config`` dict into an
``Envelope`` message. It intentionally makes **no assumption about field names
or payload types** -- ``data={"images": ...}``, ``data={"sentences": ...}``
or any other keys are all the same shape to the client. Value types pick the
``Value`` oneof via ``aux.wrap_value``.

Value coercion (applied to each data value, and to each list element):

    bytes / bytearray / memoryview   -> bytes          binary payload (e.g. an image)
    pathlib.Path                     -> read file      serializes a local file to bytes
    int                              -> float          (proto ``f`` is a float)
    str                              -> literal string a *text* value, NOT a file path
    list of the above (homogeneous)  -> BytesList / StringList / FloatList

So: to ship a *file* use ``bytes`` or ``pathlib.Path`` (or :func:`load`); to send a
*literal string* use ``str``.
"""

import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union

from ._pb_loader import get as _get_pb


def load(path: Union[str, "pathlib.PurePath"]) -> bytes:
    """Read a local file into ``bytes`` so it can be shipped to a (possibly
    remote) box. Equivalent to ``pathlib.Path(path).read_bytes()``."""
    return pathlib.Path(path).read_bytes()


def _coerce_item(v: Any) -> Any:
    """Coerce one scalar into something ``aux.wrap_value`` can wrap."""
    if isinstance(v, pathlib.PurePath):
        return v.read_bytes()
    if isinstance(v, (bytearray, memoryview)):
        return bytes(v)
    if type(v) is int:
        return float(v)
    return v


def _coerce_value(v: Any) -> Any:
    """Coerce a value or a homogeneous list of values."""
    if isinstance(v, (list, tuple)):
        return [_coerce_item(x) for x in v]
    return _coerce_item(v)


def build(data: Optional[Dict[str, Any]] = None,
          config: Optional[Dict[str, Any]] = None):
    """Build an ``Envelope`` request.

    Parameters
    ----------
    data:
        ``{field_name: value}`` mapped into ``Envelope.data``. Values follow the
        coercion rules in the module docstring.
    config:
        Arbitrary dict serialized to ``Envelope.config_json`` (the box-specific
        control payload; shape depends on the box).
    """
    pb2, _grpc, aux = _get_pb()
    env = pb2.Envelope(config_json=json.dumps(config or {}))
    for k, v in (data or {}).items():
        env.data[k].CopyFrom(aux.wrap_value(_coerce_value(v)))
    return env


def _load_images(images: Union[str, bytes, Sequence]) -> List[bytes]:
    """Normalize an ``images`` argument into a list of bytes.

    Used by the image-box conveniences (:mod:`visionist_client.conveniences`), so a
    bare ``str`` here is treated as a **local file path** to serialize (distinct
    from the generic ``run`` contract where ``str`` means a literal string).
    Accepts:

    * a single path (str) or preencoded bytes
    * a list of paths / preencoded bytes
    * :class:`pathlib.Path` objects
    """
    if images is None:
        return []
    if isinstance(images, (str, bytes, bytearray, pathlib.PurePath)):
        images = [images]
    out: List[bytes] = []
    for item in images:
        if isinstance(item, (bytes, bytearray, memoryview)):
            out.append(bytes(item))
        elif isinstance(item, (str, pathlib.PurePath)):
            out.append(pathlib.Path(item).read_bytes())
        else:
            raise TypeError(
                f"Unsupported image element: {type(item)!r}. "
                "Pass a path (str/pathlib.Path), pre-encoded bytes, or a list of those."
            )
    return out


def reset_envelope(config_key: str):
    """Build a config-only ``Process`` request that clears server-side state.

    ``config_key`` names the box's config section (e.g. ``"tapnext"``,
    ``"clip"``); the caller decides -- the core has no default and names no box.
    """
    return build(config={config_key: {"command": "reset"}})
