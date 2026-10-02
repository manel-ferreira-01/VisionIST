"""Pipeline routes: list the pipelines, start a run (a job), poll / cancel it."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from ..core import ArtifactMissing, JobRunner, PipelineError
from .errors import WebUIError


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    timeout: float = Field(600.0, ge=1.0, le=3600)     # per box call


def create_pipelines_router(runner: JobRunner) -> APIRouter:
    r = APIRouter(prefix="/api/pipelines", tags=["pipelines"])

    @r.get("")
    def list_pipelines():
        out = []
        for p in runner.pipelines:
            d = p.defn.model_dump(mode="json")
            d["missing"] = runner.missing_boxes(p)        # uses without a fleet entry
            out.append(d)
        return {"pipelines": out}

    @r.post("/{pipeline_id}/run", status_code=202)
    def run(pipeline_id: str, body: RunBody):
        try:
            job = runner.submit(pipeline_id, body.data, body.parameters, timeout=body.timeout)
        except KeyError as e:
            raise WebUIError(404, {"message": str(e.args[0])}) from e
        except PipelineError as e:
            raise WebUIError(400, {"message": str(e), **e.detail}) from e
        except ArtifactMissing as e:
            raise WebUIError(400, {"message": f"unknown upload token in data: {e.args[0]}"}) from e
        return job.to_json()

    @r.get("/jobs/{job_id}")
    def job(job_id: str):
        try:
            return runner.get(job_id).to_json()
        except KeyError as e:
            raise WebUIError(404, {"message": str(e.args[0])}) from e

    @r.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return runner.cancel(job_id).to_json()
        except KeyError as e:
            raise WebUIError(404, {"message": str(e.args[0])}) from e

    return r


__all__ = ["create_pipelines_router"]
