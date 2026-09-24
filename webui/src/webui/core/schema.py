"""Pydantic models for the box-definition YAML files (``webui/boxes/*.yaml``).

The definitions are the *only* place in the webui where a box gets named.
Everything else — the API, the serializer, the SPA widgets/visualizers —
operates on these generic models.  This mirrors ``visionist_client``'s rule:
the core stays **box-agnostic**; per-box knowledge is data, not code.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# Vocabulary (the "codec" of the GUI — small, generic, stable)
# --------------------------------------------------------------------------

#: Form widgets the SPA must implement.
WIDGETS: frozenset[str] = frozenset({
    "image_upload",    # file/drag-drop -> uploaded bytes ("@token")
    "video_frames",    # video file -> frames extracted in browser -> list of "@token" image refs
                       # (wire shape identical to image_upload with multiple: true)
    "file_upload",     # generic single file -> uploaded bytes
    "tags",            # list of literal strings
    "text_repeat",     # one text line per image (index-aligned)
    "slider",          # number with min/max/step
    "select",          # choose from values
    "number",          # free number
    "json",            # raw JSON editor (escape hatch)
})

#: Result visualizers the SPA must implement (keyed by ``kind``).
VISUALIZERS: frozenset[str] = frozenset({
    "json",           # fallback: decoded object as a JSON tree
    "table",          # list of objects / 2-D array -> data table
    "image_grid",     # list of image bytes
    "overlay",        # base image + typed layers (box/mask/point/flow)
    "matrix",         # similarity/heat matrices
    "field_map",      # per-item numeric map as an image (H×W heat or H×W×3 RGB)
    "flow_field",     # dense 2-D vector field (u,v per pixel, e.g. optical flow)
                       # -> quiver over the first base image (def `base`), fwd/bwd
                       # fields selectable
    "tensor",         # shape/dtype/summary of a numeric array
    "glb",            # glTF binary 3D model (three.js)
    "video",          # video file artifact (mp4/…) -> native <video controls>
    "points",         # per-item point cloud (depth+intrinsics or (…, 3)) → THREE.Points, no 3D file format
    "tracks_player",  # frames + tracks/visibles animation
    "download",       # raw file download
})

#: Layer types an ``overlay`` visualizer can draw on a base image.
OVERLAY_LAYERS: frozenset[str] = frozenset({
    "box",    # [x1, y1, x2, y2]  (or [[x1,y1,x2,y2], …] per item)
    "mask",   # (H, W) bool / float mask array
    "point",  # [x, y] per item
    "flow",   # optical-flow vectors (Nx2 or grid)
})

#: Envelope ``Value`` oneof kinds (contract hint for the UI).
VALUE_KINDS = ("b", "s", "f", "bb", "ss", "ff")


class _Def(BaseModel):
    """Common strict base: typo'd keys are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class InputField(_Def):
    """One ``Envelope.data`` field the form collects.

    ``kind`` is the Value-oneof it will be wrapped as (contract hint);
    ``widget`` is how the form collects it.  An uploaded file arrives at the
    API as an ``"@token"`` reference; a bare string is always a *literal*.
    """

    field: str
    widget: str = "tags"
    kind: str = "bb"
    multiple: bool = False
    required: bool = False
    default: Any = None
    helper: Optional[str] = None      # e.g. "video_frames" (SPA feature)
    constraint: Optional[str] = None  # human note shown in the form
    placeholder: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.widget not in WIDGETS:
            raise ValueError(f"unknown widget {self.widget!r} (one of {sorted(WIDGETS)})")
        if self.kind not in VALUE_KINDS:
            raise ValueError(f"unknown kind {self.kind!r} (one of {list(VALUE_KINDS)})")
        return self


class ParamDef(_Def):
    """One box ``parameters`` key (or section-level key)."""

    key: str
    widget: str = "number"
    default: Any = None
    required: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    values: Optional[list[str]] = None   # for widget: select
    placeholder: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.widget not in WIDGETS:
            raise ValueError(f"unknown widget {self.widget!r} (one of {sorted(WIDGETS)})")
        return self


class ActionDef(_Def):
    """A named command entry.

    The envelope contract only defines one RPC (``Process``), so actions are
    today *named commands* over that single RPC (with their own parameter
    sets); the field stays so pre-contract boxes (opencv_box) can be
    re-expressed the moment they migrate — and so any box can offer
    "action = X" forms later without an API change.  Non-``Process``
    ``method`` values are rejected by ``build_call`` until then.
    """

    name: str
    method: str = "Process"
    parameters: list[ParamDef] = Field(default_factory=list)
    note: Optional[str] = None


class CommandSpec(_Def):
    """Values for the payload's ``"command"`` key (contract: every standard
    box accepts ``"reset"``)."""

    values: list[str] = Field(default_factory=list)
    default: Optional[str] = None


class SessionDef(_Def):
    """Multi-session state (tapnext pattern): ``session_id`` keys all state.

    The id is an opaque capability string — the SPA may auto-generate one per
    browser (``auto_generate``) but it is always overridable.
    """

    key: str = "session_id"
    auto_generate: bool = False
    actions: list[str] = Field(default_factory=list)   # e.g. ["reset", "list"]
    note: Optional[str] = None


class LayerDef(_Def):
    """One drawable layer of an ``overlay`` visualizer."""

    prop: str                                # item property holding the geometry
    layer: str                               # one of OVERLAY_LAYERS
    color_by_index: bool = True
    opacity: Optional[float] = None


class ResultDef(_Def):
    """How to render one response field (``field: "*"`` is the wildcard
    fallback for anything else)."""

    field: str
    visualizer: str
    caption: Optional[str] = None
    base: Optional[str] = None               # input field with base images (overlay base, points colors)
    only_if_missing: Optional[str] = None    # render only when <field> is NOT in the response
                                              # (preferred-field fallback, e.g. annotated grid
                                              # only when the box returned no annotated_video)
    layers: list[LayerDef] = Field(default_factory=list)
    inputs: dict[str, str] = Field(default_factory=dict)  # e.g. {frame: images, tracks: tracks}
    note: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)   # escape hatch for a visualizer

    @model_validator(mode="after")
    def _check(self):
        if self.visualizer not in VISUALIZERS:
            raise ValueError(f"unknown visualizer {self.visualizer!r} (one of {sorted(VISUALIZERS)})")
        for ly in self.layers:
            if ly.layer not in OVERLAY_LAYERS:
                raise ValueError(f"unknown overlay layer {ly.layer!r} (one of {sorted(OVERLAY_LAYERS)})")
        return self


# --------------------------------------------------------------------------
# The box definition
# --------------------------------------------------------------------------

class BoxDef(_Def):
    """One box as the webui sees it: form in, visualizers out.

    Layout of the payload (what the caller assembles):

    * standard boxes  -> ``config.<box_key>.{command, parameters, <section keys>, session_id}``
    * ``flat_config`` -> everything at the top level of ``config_json``
      (legacy shapes: vggt)

    The box's ``docs`` link keeps the *box README* the authoritative source;
    the definition is a view over it, not a replacement.
    """

    id: str
    name: str
    box_key: Optional[str] = None          # config_json namespace key (None only with flat_config)
    flat_config: bool = False
    method: str = "Process"                # contract RPC; only "Process" is usable today
    docs: Optional[str] = None
    note: Optional[str] = None
    experimental: bool = False
    input_mosaic: bool = True              # hide the "inputs" image mosaic when the
                                           # result already shows the input (e.g. yolo's
                                           # annotated frames = input + boxes)

    inputs: list[InputField] = Field(default_factory=list)
    actions: list[ActionDef] = Field(default_factory=list)
    command: Optional[CommandSpec] = None
    parameters: list[ParamDef] = Field(default_factory=list)
    section: list[ParamDef] = Field(default_factory=list)
    session: Optional[SessionDef] = None
    results: list[ResultDef] = Field(default_factory=list)

    # ------------------------------------------------------------------ refs
    def input_fields(self) -> dict[str, InputField]:
        return {f.field: f for f in self.inputs}

    def known_params(self, action: Optional[ActionDef] = None) -> list[ParamDef]:
        out = list(self.parameters)
        if action is not None:
            out = out + list(action.parameters)
        return out

    def known_section_keys(self) -> list[str]:
        return [s.key for s in self.section]

    @model_validator(mode="after")
    def _check(self):
        if self.flat_config and self.box_key is not None:
            raise ValueError(
                "flat_config=true means box_key must be null "
                "(payload goes at the top level of config_json)")
        if not self.flat_config and self.box_key is None:
            raise ValueError("box_key is required unless flat_config=true")
        seen_inputs = set()
        for f in self.inputs:
            if f.field in seen_inputs:
                raise ValueError(f"duplicate input field {f.field!r}")
            seen_inputs.add(f.field)
        if self.actions:
            names = [a.name for a in self.actions]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate action names: {names}")
        if self.command is not None and self.command.values:
            d = self.command.default
            if d is not None and d not in self.command.values:
                raise ValueError(f"command default {d!r} not in values {self.command.values!r}")
        return self


__all__ = [
    "BoxDef", "InputField", "ParamDef", "ActionDef", "CommandSpec",
    "SessionDef", "LayerDef", "ResultDef",
    "WIDGETS", "VISUALIZERS", "OVERLAY_LAYERS", "VALUE_KINDS",
]
