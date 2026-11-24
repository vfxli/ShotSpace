"""SAM 3D Body solver supervisor, multi-GPU queue scheduler, GPU architecture diagnostics, and person selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .gpu_arch import GpuDeviceInfo, list_gpus, probe_torch_archs
from .select_people import SelectionMode, SelectionSpec, select_people

if TYPE_CHECKING:
    from .manager import SolverManager
    from .scheduler import JobCancelledError, MultiGpuScheduler, QueueJob


def __getattr__(name: str) -> Any:
    if name == "SolverManager":
        from .manager import SolverManager

        return SolverManager
    if name in ("JobCancelledError", "MultiGpuScheduler", "QueueJob"):
        from . import scheduler

        return getattr(scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GpuDeviceInfo",
    "JobCancelledError",
    "MultiGpuScheduler",
    "QueueJob",
    "SelectionMode",
    "SelectionSpec",
    "SolverManager",
    "list_gpus",
    "probe_torch_archs",
    "select_people",
]
