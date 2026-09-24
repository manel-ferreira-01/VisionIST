"""Box-specific *conveniences* -- NOT part of the agnostic core.

The core of ``visionist_client`` (``box.py`` / ``envelope.py`` / ``result.py`` /
``decode_util.py``) is deliberately box-agnostic: it only knows *how* to build
an ``Envelope`` and send it over the shared ``PipelineService`` interface, and
*how* to read a ``Result`` back. It knows **no box name, field name, or model**.

This module is the opposite, on purpose: it encodes knowledge about a *specific*
box and only composes the generic ``Visionist.run`` / ``Visionist.reset`` primitives.  The
tapnext point-tracking helper :func:`trace` lives here.

If you add a convenience for another box (``segment``, ``embed``, ``detect``,
...), put it here or in a sibling ``conveniences/<box>.py`` -- **never** in
``box.py`` -- so the core stays agnostic and each box's sugar is isolated,
visible, and easy to delete.
"""

from typing import Any, List, Sequence, Union
import pathlib

__all__ = ["trace", "load_images"]


def load_images(images: Union[str, bytes, "pathlib.PurePath", Sequence]) -> List[bytes]:
    """Normalize an ``images`` argument into a list of ``bytes``.

    ``trace`` is the *image-box* convenience, so a bare ``str`` here is treated
    as a **local file path** to serialize (distinct from the generic ``run``
    contract where ``str`` means a literal string).  Accepts:

    * a single path (``str``/``pathlib.Path``) or pre-encoded ``bytes``
    * a list of any of the above
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


def trace(
    box,
    images: Union[str, bytes, "pathlib.PurePath", Sequence],
    *,
    grid_size: int = None,
    reset_first: bool = True,
    config_key: str = "tapnext",
    **params: Any,
):
    """Point-tracking convenience for the **tapnext** box.

    Built purely on the generic core -- it is equivalent to::

        box.run(data={"images": [bytes, ...]},
                config={config_key: {"command": "track", "parameters": {...}}},
                method="Process")

    with an optional preceding ``box.reset(config_key)`` (tapnext *accumulates*
    tracks across sequential requests, so a reset keeps a one-shot clean).

    Parameters
    ----------
    box:
        A :class:`visionist_client.Visionist` pointed at a tapnext box.
    images:
        A single local path / pre-encoded bytes, or a list of either.
    grid_size:
        TAPNext grid size (added to ``parameters``).
    reset_first:
        Send ``reset`` first (default ``True``).
    config_key:
        The box's config section name (default ``"tapnext"``).
    **params:
        Extra ``parameters`` (e.g. ``threshold=...``).
    """
    images_bytes = load_images(images)
    if not images_bytes:
        raise ValueError("trace(): no images provided")

    parameters: Any = dict(params or {})
    if grid_size is not None:
        parameters["grid_size"] = int(grid_size)

    if reset_first:
        box.reset(config_key)

    return box.run(
        data={"images": images_bytes},
        config={config_key: {"command": "track", "parameters": parameters}},
        method="Process",
    )
