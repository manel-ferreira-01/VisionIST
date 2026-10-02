"""Web API routers (defs / fleet / call / pipelines)."""

from .errors import WebUIError, register_error_handlers
from .defs import create_defs_router
from .fleet import create_fleet_router
from .call import create_call_router
from .pipelines import create_pipelines_router

__all__ = [
    "WebUIError", "register_error_handlers",
    "create_defs_router", "create_fleet_router", "create_call_router", "create_pipelines_router",
]
