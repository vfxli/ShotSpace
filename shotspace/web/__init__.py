"""FastAPI web backend & workspace manager for ShotSpace."""

from .app import create_app
from .workspace import WorkspaceManager

__all__ = ["WorkspaceManager", "create_app"]
