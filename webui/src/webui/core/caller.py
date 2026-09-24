"""Assemble an Envelope call from a box definition + a wire request, and
execute it against a box.

Design (mirrors ``visionist_client``):

* :func:`build_call` is **pure** — it only knows the definition, the
  request, and upload tokens.  No network.  Fully unit-testable.
* :func:`execute` is the only function that touches the wire: only the
  contract RPC ``Process`` via ``visionist_client.Visionist``.  Boxes that still serve
  bespoke RPCs (opencv_box) are out of scope until they migrate to
  the shared envelope — :func:`build_call` refuses non-``Process`` methods
  with a clear error.

Payload layout

* standard boxes -> ``{box_key: {command?, parameters?, <section keys>, session_id?}}``
* ``flat_config`` boxes -> every key at the top level of ``config_json``

Reset semantics

* command is ``"reset"``      -> that call *is* the reset (no reset_first)
* stateful box (``session``)  -> never auto-reset (it would kill the sequence)
* stateless box with box_key  -> safe no-op ``reset_first`` (client default)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schema import BoxDef
from .artifact import ArtifactStore

#: A bare string starting with this prefix is an uploaded-token reference;
#: any other string is always a *literal* value (the client's contract).
TOKEN_PREFIX = "@"


class CallBuildError(ValueError):
    """The wire request does not match the box definition. ``detail`` is
    JSON-serializable for the API error body."""

    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.detail = detail or {}


# --------------------------------------------------------------------------
# Wire request (browser -> webui API)
# --------------------------------------------------------------------------

class CallRequest(BaseModel):
    """Box-agnostic call request.  Field *names* come from the definition;
    everything here is shape, not content."""

    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    section: dict[str, Any] = Field(default_factory=dict)
    command: Optional[str] = None
    action: Optional[str] = None          # named RPC entry from def.actions
    session_id: Optional[str] = None
    method: Optional[str] = None          # explicit RPC override


@dataclass(frozen=True)
class CallSpec:
    """Resolved, ready-to-send call (pure data)."""

    data: dict[str, Any]
    config: dict[str, Any]
    method: str = "Process"
    reset_first: bool = False
    action: Optional[str] = None
    session_id: Optional[str] = None


# --------------------------------------------------------------------------
# Token resolution (files live server-side as upload artifacts)
# --------------------------------------------------------------------------

def _resolve_one(v: Any, store: ArtifactStore) -> Any:
    if isinstance(v, str) and v.startswith(TOKEN_PREFIX):
        token = v[len(TOKEN_PREFIX):]
        return bytes(store.get(token).data)      # ArtifactMissing -> caller maps
    return v


def resolve_data(data: dict[str, Any], store: ArtifactStore) -> dict[str, Any]:
    out = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            out[k] = [_resolve_one(x, store) for x in v]
        else:
            out[k] = _resolve_one(v, store)
    return out


# --------------------------------------------------------------------------
# build_call
# --------------------------------------------------------------------------

def build_call(defn: BoxDef, req: CallRequest, store: ArtifactStore) -> CallSpec:
    """Validate the wire request against the definition and assemble the
    payload.  Raises :class:`CallBuildError` on any contract mismatch."""

    # ---- command ---------------------------------------------------------
    command = req.command
    if command is None and defn.command is not None:
        command = defn.command.default
    if command is None and defn.command is not None and defn.command.values:
        command = defn.command.values[0]
    if (
        command is not None
        and defn.command is not None
        and defn.command.values
        and command not in defn.command.values
    ):
        raise CallBuildError(
            f"unknown command {command!r}",
            {"known": defn.command.values},
        )

    # ---- action / method --------------------------------------------------
    action = None
    if defn.actions:
        names = [a.name for a in defn.actions]
        want = req.action or names[0]
        action = next((a for a in defn.actions if a.name == want), None)
        if action is None:
            raise CallBuildError(f"unknown action {want!r}", {"known": names})
        method = req.method or action.method
    else:
        method = req.method or defn.method

    if method != "Process":
        raise CallBuildError(
            f"method {method!r} is not the shared contract RPC. Box "
            f"{defn.id!r} has not migrated to the envelope (Process) contract "
            f"yet — support lands when the box does (see the 'Method dispatch "
            f"caveat' in the visionist_client README).",
            {"method": method, "def_id": defn.id},
        )

    # ---- parameters -------------------------------------------------------
    known = {p.key: p for p in defn.known_params(action)}
    unknown_p = sorted(set(req.parameters) - set(known))
    if unknown_p:
        raise CallBuildError(
            f"unknown parameter(s) {unknown_p}",
            {"known": sorted(known)},
        )
    params: dict[str, Any] = {}
    for key, p in known.items():
        if req.parameters.get(key) is not None:
            params[key] = req.parameters[key]
        elif p.default is not None:
            params[key] = p.default
    for key, p in known.items():
        if p.required and req.parameters.get(key) is None and p.default is None:
            raise CallBuildError(f"missing required parameter {key!r}")

    # ---- section-level keys ----------------------------------------------
    # A "reset" call never carries inputs: required checks (section keys and
    # data fields) are waived for it — stateful boxes clear their session,
    # stateless boxes no-op it (fleet convention: every box accepts reset).
    is_reset = (command == "reset")
    unknown_s = sorted(set(req.section) - set(defn.known_section_keys()))
    if unknown_s:
        raise CallBuildError(
            f"unknown section key(s) {unknown_s}",
            {"known": defn.known_section_keys()},
        )
    sec: dict[str, Any] = {}
    for s in defn.section:
        if req.section.get(s.key) is not None:
            sec[s.key] = req.section[s.key]
        elif s.default is not None:
            sec[s.key] = s.default
        elif s.required and not is_reset:
            raise CallBuildError(f"missing required section key {s.key!r}")

    # ---- data fields -------------------------------------------------------

    fields = defn.input_fields()
    unknown_d = sorted(set(req.data) - set(fields))
    if unknown_d:
        raise CallBuildError(
            f"unknown data field(s) {unknown_d}",
            {"known": sorted(fields)},
        )
    data: dict[str, Any] = {}
    for fname, f in fields.items():
        v = req.data.get(fname, f.default)
        if v is None:
            if f.required and not is_reset:
                raise CallBuildError(f"missing required data field {fname!r}")
            continue
        if not f.multiple and isinstance(v, (list, tuple)):
            if len(v) == 1:
                v = v[0]
            else:
                raise CallBuildError(
                    f"data field {fname!r} takes a single value, got a list of {len(v)}")
        data[fname] = resolve_data({fname: v}, store)[fname]

    # ---- session id --------------------------------------------------------
    session_id = req.session_id
    if session_id is not None and defn.session is None:
        raise CallBuildError(f"box {defn.id!r} has no session (drop session_id)")

    # ---- assemble config_json payload --------------------------------------
    body: dict[str, Any] = {}
    if command is not None:
        body["command"] = command
    if params:
        body["parameters"] = params
    body.update(sec)
    if session_id is not None:
        body[defn.session.key if defn.session else "session_id"] = session_id

    if defn.flat_config:
        config = body
    else:
        config = {defn.box_key: body}

    # ---- reset semantics ----------------------------------------------------
    if command == "reset":
        reset_first = False
    elif defn.session is not None:
        reset_first = False              # never break a live tracking session
    else:
        reset_first = defn.box_key is not None
    return CallSpec(
        data=data, config=config, method=method,
        reset_first=reset_first,
        action=action.name if action else req.action,
        session_id=session_id,
    )


# --------------------------------------------------------------------------
# execute — the only network-touching part
# --------------------------------------------------------------------------

def execute(spec: CallSpec, address: str, defn: BoxDef,
            timeout: float = 600.0) -> "Result":  # noqa: F821 (visionist_client.Result)
    """Send :class:`CallSpec` to the box at ``address`` and return the
    client's decoded :class:`Result`.

    Contract RPC only (``Process``): that's the point of the shared envelope
    — one stub for every box.
    """
    if spec.method != "Process":
        raise CallBuildError(
            f"box {defn.id!r} cannot be called via {spec.method!r} (pre-contract box)")
    from visionist_client import Visionist

    with Visionist(address, config_key=defn.box_key, timeout=timeout) as box:
        return box.run(
            data=spec.data,
            config=spec.config,
            method=spec.method,
            reset_first=spec.reset_first,
        )


__all__ = [
    "TOKEN_PREFIX", "CallRequest", "CallSpec", "CallBuildError",
    "build_call", "execute", "resolve_data",
]
