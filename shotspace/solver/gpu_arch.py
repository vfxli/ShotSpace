"""GPU architecture & CUDA compute-capability inspection (RTX 4090 `sm_89`, RTX 5090 `sm_120`, etc.)."""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

_PROBE_SCRIPT = """
import json
res = {"torch_version": "", "archs": None}
try:
    import torch
    res["torch_version"] = torch.__version__
    flags = torch._C._cuda_getArchFlags()
    if flags:
        res["archs"] = flags.split()
except Exception:
    pass
print(json.dumps(res))
"""


@dataclass(frozen=True)
class GpuDeviceInfo:
    index: int
    name: str
    compute_cap: str
    sm: str
    memory_total_mb: int
    memory_free_mb: int
    compatible: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _cc(token: str) -> tuple[int, int] | None:
    m = re.match(r"^(?:sm|compute)_(\d+?)(\d)[a-z]*$", token)
    return (int(m.group(1)), int(m.group(2))) if m else None


def cap_to_sm(compute_cap: str) -> str:
    major, _, minor = compute_cap.strip().partition(".")
    return f"sm_{int(major)}{int(minor or 0)}"


def is_sm_compatible(archs: tuple[str, ...] | None, target_sm: str) -> bool:
    if not archs:
        return False
    if target_sm in archs:
        return True
    target = _cc(target_sm)
    if target is None:
        return False
    for a in archs:
        cc = _cc(a)
        if cc is None:
            continue
        if a.startswith("sm_") and cc[0] == target[0] and cc[1] <= target[1]:
            return True
        if a.startswith("compute_") and cc <= target:
            return True
    return False


@functools.lru_cache(maxsize=8)
def _probe_torch_archs_cached(python_str: str, timeout: float = 30.0) -> tuple[str, tuple[str, ...] | None]:
    python = Path(python_str)
    if not python.is_file():
        return "", None
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("CUDA_DEVICE_ORDER", None)
    try:
        out = subprocess.run([str(python), "-c", _PROBE_SCRIPT], capture_output=True, text=True, timeout=timeout, env=env)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        archs = data.get("archs")
        return str(data.get("torch_version") or ""), (tuple(archs) if archs else None)
    except Exception:
        return "", None


def probe_torch_archs(python: Path, timeout: float = 30.0) -> tuple[str, tuple[str, ...] | None]:
    return _probe_torch_archs_cached(str(python), timeout)


def list_gpus(sam3d_python: Path | None = None) -> list[GpuDeviceInfo]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    _, archs = probe_torch_archs(sam3d_python) if sam3d_python else ("", None)
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=index,name,compute_cap,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    gpus: list[GpuDeviceInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        name = parts[1]
        cc = parts[2]
        sm = cap_to_sm(cc)
        mem_tot = int(float(parts[3]))
        mem_free = int(float(parts[4]))
        compat = is_sm_compatible(archs, sm) if archs else True
        gpus.append(
            GpuDeviceInfo(
                index=idx,
                name=name,
                compute_cap=cc,
                sm=sm,
                memory_total_mb=mem_tot,
                memory_free_mb=mem_free,
                compatible=compat,
            )
        )
    return gpus
