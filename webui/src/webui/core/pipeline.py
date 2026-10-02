"""Pipelines: several boxes chained by server-side glue.

A box console is one ``Process`` call.  Some tasks are several calls with
Python in between (lightglue tracks + MoGe depth -> sfm): those live as
**pipeline modules** in ``webui/pipelines/*.py``.  Each module exposes

* ``DEF`` — a dict validated as :class:`~webui.core.schema.PipelineDef`
  (the same form-in / visualizers-out vocabulary as a box YAML), and
* ``run(ctx, data, params)`` — the glue.  It reaches boxes only through
  ``ctx.box(def_id)`` (resolved through the fleet), groups its work in
  ``with ctx.step(...)`` blocks, and publishes results with
  ``ctx.emit(field, value)`` as soon as they exist.

The core still names no box: the pipeline module is the per-task knowledge,
exactly like a box YAML is the per-box knowledge.

Runs are **jobs** (several minutes for a long video): :class:`JobRunner`
executes them on a small thread pool and keeps a JSON snapshot per job
(steps, emitted fields, errors) that the API serves for polling.
"""

from __future__ import annotations

import importlib.util
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

from pydantic import ValidationError

from .artifact import ArtifactStore
from .caller import resolve_data
from .fleet import Fleet, FleetEntry
from .registry import Registry
from .schema import PipelineDef
from .serialize import serialize_value


class PipelineError(RuntimeError):
    """A pipeline module is invalid, or a run request does not match its
    definition.  ``detail`` is JSON-serializable for the API error body."""

    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.detail = detail or {}


class Cancelled(Exception):
    """Raised inside a run when its job was cancelled (at the next step /
    box call boundary)."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Pipeline:
    defn: PipelineDef
    run: Callable[..., Any]
    path: Path


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"webui_pipeline_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"{path}: not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pipeline(path: Path) -> Pipeline:
    try:
        mod = _load_module(path)
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001 — import errors name the file
        raise PipelineError(f"{path}: import failed: {type(e).__name__}: {e}") from e
    raw, run = getattr(mod, "DEF", None), getattr(mod, "run", None)
    if not isinstance(raw, dict) or not callable(run):
        raise PipelineError(f"{path}: a pipeline module needs a DEF dict and a run(ctx, data, params)")
    try:
        defn = PipelineDef.model_validate(raw)
    except ValidationError as e:
        raise PipelineError(f"{path}: {e}") from e
    return Pipeline(defn=defn, run=run, path=path)


class Pipelines:
    """Validated, unique-id set of pipelines."""

    def __init__(self, items: list[Pipeline]):
        self.items = items

    def __iter__(self) -> Iterator[Pipeline]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def get(self, pipeline_id: str) -> Pipeline:
        for p in self.items:
            if p.defn.id == pipeline_id:
                return p
        known = ", ".join(p.defn.id for p in self.items) or "(none)"
        raise KeyError(f"unknown pipeline {pipeline_id!r} (known: {known})")

    def to_list(self) -> list[dict]:
        return [p.defn.model_dump(mode="json") for p in self.items]


def load_pipelines(pipelines_dir: str | Path, registry: Registry) -> Pipelines:
    """Load every ``*.py`` (not ``_*.py``) in ``pipelines_dir``.  A missing
    directory means no pipelines; a bad module fails fast, like a bad box
    YAML.  Every ``uses`` id must be a known box definition."""
    d = Path(pipelines_dir)
    if not d.is_dir():
        return Pipelines([])
    items: list[Pipeline] = []
    seen: set[str] = set()
    box_ids = {b.id for b in registry}
    for path in sorted(d.glob("*.py")):
        if path.name.startswith("_"):
            continue
        p = load_pipeline(path)
        if p.defn.id in seen:
            raise PipelineError(f"duplicate pipeline id {p.defn.id!r} (in {path})")
        unknown = sorted(set(p.defn.uses) - box_ids)
        if unknown:
            raise PipelineError(f"{path}: uses unknown box definition(s) {unknown}",
                                {"known": sorted(box_ids)})
        seen.add(p.defn.id)
        items.append(p)
    return Pipelines(items)


# --------------------------------------------------------------------------
# Request validation (the BoxDef rules: unknown keys are errors, defaults fill)
# --------------------------------------------------------------------------

def prepare_request(defn: PipelineDef, data: dict[str, Any], parameters: dict[str, Any],
                    store: ArtifactStore) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate ``data`` / ``parameters`` against the definition, fill the
    defaults, resolve ``"@token"`` uploads to bytes."""
    known = {p.key: p for p in defn.parameters}
    unknown_p = sorted(set(parameters) - set(known))
    if unknown_p:
        raise PipelineError(f"unknown parameter(s) {unknown_p}", {"known": sorted(known)})
    params: dict[str, Any] = {}
    for key, p in known.items():
        v = parameters.get(key)
        if v is None or v == "":
            v = p.default
        if v is None and p.required:
            raise PipelineError(f"missing required parameter {key!r}")
        params[key] = v

    fields = defn.input_fields()
    unknown_d = sorted(set(data) - set(fields))
    if unknown_d:
        raise PipelineError(f"unknown data field(s) {unknown_d}", {"known": sorted(fields)})
    out: dict[str, Any] = {}
    for name, f in fields.items():
        v = data.get(name, f.default)
        if v is None or v == []:
            if f.required:
                raise PipelineError(f"missing required data field {name!r}")
            continue
        if not f.multiple and isinstance(v, (list, tuple)):
            if len(v) != 1:
                raise PipelineError(f"data field {name!r} takes a single value, got a list of {len(v)}")
            v = v[0]
        out[name] = resolve_data({name: v}, store)[name]
    return out, params


def resolve_entry(fleet: Fleet, registry: Registry, def_id: str) -> Optional[FleetEntry]:
    """The fleet entry serving box definition ``def_id`` (explicit ``def_id``
    first, then a name/id match — the console's rule)."""
    entries = fleet.list()
    for e in entries:
        if e.def_id == def_id:
            return e
    for e in entries:
        if e.def_id is None and registry.match(e.name) is not None and registry.match(e.name).id == def_id:
            return e
    return None


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

@dataclass
class StepState:
    name: str
    status: str = "running"            # running | done | error | cancelled
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    calls: int = 0                     # box calls made inside this step
    total: Optional[int] = None        # expected box calls (progress bar), if known
    message: Optional[str] = None

    def to_json(self) -> dict:
        end = self.finished or time.time()
        return {"name": self.name, "status": self.status, "calls": self.calls,
                "total": self.total, "message": self.message,
                "duration_ms": round((end - self.started) * 1000.0, 1)}


@dataclass
class Job:
    id: str
    pipeline: str
    status: str = "queued"             # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    steps: list[StepState] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)      # serialized, emit order
    info: dict[str, Any] = field(default_factory=dict)        # small JSON facts (ctx.info)
    inputs: dict[str, Any] = field(default_factory=dict)      # the request's "@token" refs
    artifacts: list[dict] = field(default_factory=list)
    error: Optional[dict] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_json(self) -> dict:
        with self.lock:
            end = self.finished or time.time()
            return {
                "id": self.id, "pipeline": self.pipeline, "status": self.status,
                "created": self.created, "started": self.started, "finished": self.finished,
                "duration_ms": round((end - (self.started or end)) * 1000.0, 1),
                "steps": [s.to_json() for s in self.steps],
                "fields": dict(self.fields),
                "info": dict(self.info),
                "inputs": dict(self.inputs),
                "artifacts": list(self.artifacts),
                "error": self.error,
            }


class _CountingBox:
    """Proxy over a ``Visionist``: every ``run`` counts towards the active
    step and is a cancellation point.  Everything else is forwarded."""

    def __init__(self, box, ctx: "RunContext"):
        self._box, self._ctx = box, ctx

    def run(self, *a, **kw):
        self._ctx.check()
        try:
            return self._box.run(*a, **kw)
        finally:
            self._ctx._count_call()

    def __getattr__(self, name):
        return getattr(self._box, name)


class RunContext:
    """What a pipeline's ``run`` sees: boxes, steps, emitted results."""

    def __init__(self, job: Job, fleet: Fleet, registry: Registry, store: ArtifactStore,
                 timeout: float):
        self.job = job
        self._fleet, self._registry, self._store = fleet, registry, store
        self._timeout = timeout
        self._boxes: list[Any] = []

    @property
    def job_id(self) -> str:
        return self.job.id

    # ---------------------------------------------------------------- boxes
    def box(self, def_id: str):
        """A ``visionist_client.Visionist`` for the fleet entry serving box
        definition ``def_id`` (counted, cancellable; closed after the run)."""
        entry = resolve_entry(self._fleet, self._registry, def_id)
        if entry is None:
            raise PipelineError(f"no fleet entry serves box definition {def_id!r}")
        from visionist_client import Visionist
        defn = self._registry.get(def_id)
        box = Visionist(entry.addr, config_key=defn.box_key, timeout=self._timeout)
        self._boxes.append(box)
        return _CountingBox(box, self)

    def close(self) -> None:
        for b in self._boxes:
            try:
                b.close()
            except Exception:  # noqa: BLE001
                pass
        self._boxes.clear()

    # ---------------------------------------------------------------- steps
    @contextmanager
    def step(self, name: str, total: Optional[int] = None):
        """Group work under a named, timed step shown in the UI.  ``total``
        is the expected number of box calls (progress bar)."""
        self.check()
        st = StepState(name=name, total=total)
        with self.job.lock:
            self.job.steps.append(st)
        try:
            yield st
        except Cancelled:
            st.status = "cancelled"
            raise
        except BaseException as e:
            st.status, st.message = "error", f"{type(e).__name__}: {e}"
            raise
        else:
            st.status = "done"
        finally:
            st.finished = time.time()

    def _count_call(self) -> None:
        with self.job.lock:
            running = [s for s in self.job.steps if s.status == "running"]
            if running:
                running[-1].calls += 1

    def check(self) -> None:
        """Cancellation point (also implicit at every step / box call)."""
        if self.job.cancel.is_set():
            raise Cancelled()

    # -------------------------------------------------------------- results
    def emit(self, name: str, value: Any) -> None:
        """Publish result field ``name`` now (serialized like a box field:
        small arrays inline, big ones as typed buffer artifacts)."""
        before = set(self._store.tokens())
        ser = serialize_value(value, self._store)
        produced = [self._store.public_view(t) for t in self._store.tokens() if t not in before]
        with self.job.lock:
            self.job.fields[name] = ser
            self.job.artifacts.extend(produced)

    def info(self, **facts: Any) -> None:
        """Small JSON facts for the result header (counts, statuses, …)."""
        ser = {k: serialize_value(v, self._store) for k, v in facts.items()}
        with self.job.lock:
            self.job.info.update(ser)


class JobRunner:
    """Runs pipeline jobs on a small thread pool; keeps the last ``keep``
    jobs (finished ones are dropped oldest-first)."""

    def __init__(self, pipelines: Pipelines, fleet: Fleet, registry: Registry,
                 store: ArtifactStore, workers: int = 2, keep: int = 50):
        self.pipelines, self.fleet, self.registry, self.store = pipelines, fleet, registry, store
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pipeline")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def missing_boxes(self, pipeline: Pipeline) -> list[str]:
        return [d for d in pipeline.defn.uses
                if resolve_entry(self.fleet, self.registry, d) is None]

    def submit(self, pipeline_id: str, data: dict[str, Any], parameters: dict[str, Any],
               timeout: float = 600.0) -> Job:
        p = self.pipelines.get(pipeline_id)                       # KeyError -> 404
        missing = self.missing_boxes(p)
        if missing:
            raise PipelineError(f"no fleet entry for box definition(s) {missing}",
                                {"missing": missing, "uses": p.defn.uses})
        resolved, params = prepare_request(p.defn, data, parameters, self.store)
        job = Job(id="job_" + uuid.uuid4().hex[:12], pipeline=p.defn.id,
                  inputs={k: v for k, v in data.items() if k in p.defn.input_fields()})
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
        self._pool.submit(self._run, p, job, resolved, params, timeout)
        return job

    def _run(self, p: Pipeline, job: Job, data: dict, params: dict, timeout: float) -> None:
        ctx = RunContext(job, self.fleet, self.registry, self.store, timeout)
        with job.lock:
            job.status, job.started = "running", time.time()
        status, error = "done", None
        try:
            if job.cancel.is_set():
                raise Cancelled()
            p.run(ctx, data, params)
        except Cancelled:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001 — report, never kill the worker
            status = "error"
            step = next((s.name for s in reversed(job.steps) if s.status == "error"), None)
            error = {"message": f"{type(e).__name__}: {e}", "step": step,
                     "trace": traceback.format_exc(limit=8)}
        finally:
            ctx.close()
            with job.lock:
                for s in job.steps:
                    if s.status == "running":
                        s.status, s.finished = ("cancelled" if status == "cancelled" else "error"), time.time()
                job.status, job.error, job.finished = status, error, time.time()

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(f"unknown job {job_id!r}") from None

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel.set()
        with job.lock:
            if job.status == "queued":
                job.status = "cancelled"
        return job

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _trim(self) -> None:
        done = [j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")]
        while len(self._jobs) > self._keep and done:
            self._jobs.pop(done.pop(0).id, None)

    def shutdown(self) -> None:
        for j in self.list():
            j.cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "Pipeline", "Pipelines", "PipelineError", "Cancelled", "Job", "JobRunner",
    "RunContext", "load_pipeline", "load_pipelines", "prepare_request", "resolve_entry",
]
