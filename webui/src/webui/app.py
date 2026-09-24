"""FastAPI app factory for visionist-webui.

Run:
    visionist-webui                                   # after: pip install -e webui
    uvicorn webui.app:factory                     # equivalent, auto-reload friendly

The SPA (Phase 2/3, ``web/``) builds into ``web/dist`` and is served at
``/`` when present; until then the API + OpenAPI docs are the surface
(``/docs``).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppEnv, read_env
from .core import ArtifactStore, Fleet, load_registry
from .api.errors import register_error_handlers
from .api.defs import create_defs_router
from .api.fleet import create_fleet_router
from .api.call import create_call_router


def create_app(env: AppEnv | None = None) -> FastAPI:
    env = env or read_env()

    registry = load_registry(env.boxes_dir)
    store = ArtifactStore(ttl=env.artifact_ttl, max_bytes=env.max_artifact_bytes)
    fleet = Fleet(env.data_dir / "fleet.json")

    app = FastAPI(
        title="visionist-webui",
        version=__version__,
        description=(
            "Declarative web layer over a fleet of 'boxes': the box-agnostic "
            "core + YAML box definitions + visionist_client under the hood. "
            "The core names no box — all box knowledge lives in boxes/*.yaml."
        ),
    )
    register_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "service": "visionist-webui",
            "version": __version__,
            "defs": [d.id for d in registry],
            "endpoints": {
                "defs": "GET /api/defs",
                "fleet": "GET /api/fleet",
                "call": "POST /api/call (see docs, or POST /api/upload first)",
            },
        }

    app.include_router(create_defs_router(registry))
    app.include_router(create_fleet_router(fleet, registry))
    app.include_router(create_call_router(registry, fleet, store,
                                          max_upload_bytes=env.max_upload_bytes))

    dist = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    return app


factory = create_app  # for `uvicorn webui.app:factory`


def main() -> None:
    import uvicorn
    env = read_env()
    uvicorn.run("webui.app:factory", host=env.host, port=env.port,
                reload=False)


if __name__ == "__main__":
    main()
