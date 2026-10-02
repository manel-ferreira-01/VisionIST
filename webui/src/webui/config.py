"""Runtime configuration (env-driven, sane defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppEnv:
    host: str
    port: int
    data_dir: Path                       # fleet.json (+ future state)
    boxes_dir: Path                      # the YAML definitions
    artifact_ttl: float
    max_artifact_bytes: int
    max_upload_bytes: int
    pipelines_dir: Path | None = None    # pipeline modules (None: <project>/pipelines)
    pipeline_workers: int = 2            # concurrent pipeline jobs


def read_env() -> AppEnv:
    here = Path(__file__).resolve().parent            # webui/src/webui
    project = here.parent.parent                       # webui/
    return AppEnv(
        host=os.environ.get("WEBUI_HOST", "127.0.0.1"),
        port=_int("WEBUI_PORT", 8080),
        data_dir=Path(os.environ.get("WEBUI_DATA_DIR", "data")),
        boxes_dir=Path(os.environ.get("WEBUI_BOXES_DIR", str(project / "boxes"))),
        artifact_ttl=_float("WEBUI_ARTIFACT_TTL", 3600.0),
        max_artifact_bytes=_int("WEBUI_MAX_ARTIFACT_BYTES", 2_000_000_000),
        max_upload_bytes=_int("WEBUI_MAX_UPLOAD_BYTES", 64 * 1024 * 1024),
        pipelines_dir=Path(os.environ.get("WEBUI_PIPELINES_DIR", str(project / "pipelines"))),
        pipeline_workers=_int("WEBUI_PIPELINE_WORKERS", 2),
    )


__all__ = ["AppEnv", "read_env"]
