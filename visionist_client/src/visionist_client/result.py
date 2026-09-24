"""Result object returned from ``Visionist`` calls."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from .codec import decode_with
from .decode_util import decode_payload


def _is_numeric_array(v):
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(v, np.ndarray)


def _find_encoding(config):
    """Locate the box's declared ``"encoding"`` contract key.

    Scans the parsed config top-level, then each one-level section (first hit
    wins). Returns the declared value — a codec-name ``str`` that applies to
    every ``bytes`` field, or a ``{field_name: codec_name}`` map for mixed
    responses — or ``None`` (undeclared: legacy auto-decode applies).
    ``encoding`` is a generic contract keyword, not a box name.
    """
    if not isinstance(config, dict):
        return None
    if "encoding" in config:
        return config["encoding"]
    for section in config.values():
        if isinstance(section, dict) and "encoding" in section:
            return section["encoding"]
    return None


def _codec_for(enc, field_name):
    """Resolve the declared codec name for one field (or ``None``)."""
    if enc is None:
        return None
    if isinstance(enc, str):
        return enc
    if isinstance(enc, dict):
        return enc.get(field_name)
    # A declared but unrecognized shape (e.g. a list) -> ignore, use legacy.
    return None


def _decode_any(buf, codec_name=None):
    """Decode a single Envelope Value payload.

    - list of byte payloads: decode each element
    - list of strings/floats: pass through
    - bytes: declared codec if given (never raises), else the legacy guess chain
    - scalars (str/float/int): pass through
    """
    if isinstance(buf, (list, tuple)):
        items = list(buf)
        if items and all(isinstance(x, (bytes, bytearray, memoryview)) for x in items):
            if codec_name is not None:
                return [decode_with(x, codec_name) for x in items]
            return [decode_payload(x) for x in items]
        return items
    if isinstance(buf, (bytes, bytearray, memoryview)):
        if codec_name is not None:
            return decode_with(buf, codec_name)
        return decode_payload(buf)
    return buf


@dataclass
class Result:
    """A decoded response from a box.

    ``fields`` maps every Envelope data field to its decoded value: the
    box-declared ``"encoding"`` codec when the box declares one
    (:mod:`visionist_client.codec`), else the legacy auto-decode
    (:mod:`visionist_client.decode_util`). Use ``raw`` for the undecoded Envelope.
    Field values can also be read directly, e.g. ``res.tracks``.
    """

    raw: Any = None
    config: Any = None
    fields: Dict[str, Any] = field(default_factory=dict)
    #: The box's declared payload encoding, exactly as found in the response
    #: config: a codec-name ``str``, a ``{field: codec}`` map, or ``None``
    #: (undeclared → legacy auto-decode). Exposed so callers can see *why*
    #: a field came back decoded or raw.
    encoding: Any = None

    @classmethod
    def from_envelope(cls, envelope):
        config = envelope.config_json
        if config:
            try:
                config = json.loads(config)
            except (ValueError, TypeError):
                pass

        # Declared payload encoding (generic contract keyword, not a box name).
        # ``None`` → behaviour is identical to before the codecs landed.
        enc = _find_encoding(config)

        fields = {}
        for name, val in envelope.data.items():
            kind = val.WhichOneof("kind") if hasattr(val, "WhichOneof") else None
            if kind == "b":
                buf = val.b
            elif kind == "bb":
                buf = list(val.bb.values)
            elif kind == "ss":
                buf = list(val.ss.values)
            elif kind == "ff":
                buf = list(val.ff.values)
            elif kind == "f":
                buf = val.f
            elif kind == "s":
                buf = val.s
            else:
                buf = val
            fields[name] = _decode_any(buf, codec_name=_codec_for(enc, name))
        return cls(raw=envelope, config=config, fields=fields, encoding=enc)

    def __getattr__(self, name):
        if name.startswith("_") or name in ("raw", "config", "fields"):
            raise AttributeError(name)
        d = self.__dict__.get("fields") or {}
        if name in d:
            return d[name]
        raise AttributeError(
            f"Result has no field {name!r}; available: {sorted(d)}"
        )

    def as_dict(self):
        return {
            "config": self.config,
            "fields": {
                k: (v.tolist() if _is_numeric_array(v) else v)
                for k, v in self.fields.items()
            },
        }

    def __repr__(self):
        cfg = self.config if isinstance(self.config, dict) else "<str>"
        return f"<Result fields=[{', '.join(sorted(self.fields))}] config={cfg}>"
