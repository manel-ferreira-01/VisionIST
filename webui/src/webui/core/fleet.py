"""Fleet bookkeeping: which boxes exist where (``data/fleet.json``).

The webui is a *caller* over the fleet, not an orchestrator — this file only
remembers addresses, probes reachability via ``visionist_client.Visionist.info()``
(gRPC reflection), and keeps the last probe result for the UI.  No box
content or pipeline logic lives here.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class FleetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    addr: str                                # host:port (host required)
    def_id: Optional[str] = None             # which box definition to pair with
    note: Optional[str] = None
    added: float = Field(default_factory=time.time)
    last_probe: Optional[dict[str, Any]] = None


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "box"


def _check_addr(addr: str) -> str:
    a = (addr or "").strip()
    if not a:
        raise ValueError("addr is empty (expected host:port)")
    host, _, port = a.rpartition(":")
    if not host:
        raise ValueError(f"addr {addr!r} has no host")
    if port and not port.isdigit():
        raise ValueError(f"addr {addr!r}: port {port!r} is not numeric")
    return a


def probe_box(addr: str, timeout: float = 5.0) -> dict[str, Any]:
    """Reachability + reflection probe via visionist_client (its info() is the
    canonical one)."""
    from visionist_client import Visionist
    out: dict[str, Any] = {"at": time.time(), "addr": addr}
    try:
        with Visionist(addr) as box:
            out.update(box.info(timeout=timeout))
    except Exception as e:  # noqa: BLE001 — probe must never raise
        out.update({"reachable": False, "reflection": False, "error": str(e)})
    return out


class Fleet:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, FleetEntry] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for e in raw.get("entries", []):
                entry = FleetEntry.model_validate(e)
                self._entries[entry.id] = entry

    def _save(self) -> None:
        payload = {"entries": [e.model_dump() for e in self._entries.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- CRUD
    def list(self) -> list[FleetEntry]:
        with self._lock:
            return list(self._entries.values())

    def _get_locked(self, entry_id: str) -> FleetEntry:
        """Lock-internal lookup (caller already holds ``_lock``)."""
        try:
            return self._entries[entry_id]
        except KeyError:
            known = ", ".join(self._entries) or "(empty)"
            raise KeyError(f"unknown fleet entry {entry_id!r} (known: {known})") from None

    def get(self, entry_id: str) -> FleetEntry:
        with self._lock:
            return self._get_locked(entry_id)

    def add(self, name: str, addr: str, def_id: Optional[str] = None,
            note: Optional[str] = None) -> FleetEntry:
        addr = _check_addr(addr)
        with self._lock:
            base = _slug(name or addr.split(":")[0])
            eid, n = base, 2
            while eid in self._entries:
                eid = f"{base}-{n}"
                n += 1
            entry = FleetEntry(id=eid, name=name or base, addr=addr,
                               def_id=def_id, note=note)
            self._entries[eid] = entry
            self._save()
            return entry

    def update(self, entry_id: str, **fields: Any) -> FleetEntry:
        allowed = {"name", "addr", "def_id", "note"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update field(s) {sorted(bad)}")
        if "addr" in fields:
            fields["addr"] = _check_addr(fields["addr"])
        with self._lock:
            entry = self._get_locked(entry_id)
            data = entry.model_dump()
            data.update(fields)
            entry = FleetEntry.model_validate(data)
            self._entries[entry_id] = entry
            self._save()
            return entry

    def remove(self, entry_id: str) -> None:
        with self._lock:
            self._get_locked(entry_id)
            del self._entries[entry_id]
            self._save()

    # --------------------------------------------------------------- probe
    def probe(self, entry_id: str, timeout: float = 5.0) -> FleetEntry:
        with self._lock:
            addr = self._get_locked(entry_id).addr
        result = probe_box(addr, timeout=timeout)   # network: not held under the lock
        with self._lock:
            self._get_locked(entry_id).last_probe = result
            self._save()
            return self._entries[entry_id]


__all__ = ["Fleet", "FleetEntry", "probe_box"]
