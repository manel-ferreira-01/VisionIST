"""webui.core — the box-agnostic core (schema, registry, caller, ...).

Nothing in this package names a box: per-box knowledge lives in the YAML
files in ``webui/boxes/``, per-task glue in the modules in ``webui/pipelines/``.
"""

from .schema import (
    BoxDef, PipelineDef, InputField, ParamDef, ActionDef, CommandSpec, SessionDef,
    LayerDef, ResultDef,
    WIDGETS, VISUALIZERS, OVERLAY_LAYERS, VALUE_KINDS,
)
from .registry import Registry, RegistryError, load_def, load_registry
from .artifact import Artifact, ArtifactMissing, ArtifactStore
from .caller import (
    TOKEN_PREFIX, CallRequest, CallSpec, CallBuildError,
    build_call, execute, resolve_data,
)
from .serialize import serialize_result, serialize_value, sniff_mime
from .fleet import Fleet, FleetEntry, probe_box
from .pipeline import (
    Cancelled, Job, JobRunner, Pipeline, PipelineError, Pipelines, RunContext,
    load_pipeline, load_pipelines,
)

__all__ = [
    "BoxDef", "PipelineDef", "InputField", "ParamDef", "ActionDef", "CommandSpec",
    "SessionDef", "LayerDef", "ResultDef",
    "WIDGETS", "VISUALIZERS", "OVERLAY_LAYERS", "VALUE_KINDS",
    "Registry", "RegistryError", "load_def", "load_registry",
    "Artifact", "ArtifactMissing", "ArtifactStore",
    "TOKEN_PREFIX", "CallRequest", "CallSpec", "CallBuildError",
    "build_call", "execute", "resolve_data",
    "serialize_result", "serialize_value", "sniff_mime",
    "Fleet", "FleetEntry", "probe_box",
    "Cancelled", "Job", "JobRunner", "Pipeline", "PipelineError", "Pipelines",
    "RunContext", "load_pipeline", "load_pipelines",
]
