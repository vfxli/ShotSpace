"""Configuration and runtime asset path resolution for ShotSpace.

All model weights, pinned repositories, FBX SDK, and isolated worker virtual environments
live directly inside `<PROJECT_ROOT>/third_party/` (self-contained copies, zero symlinks
to external projects), with optional environment variable overrides for studio deployments
on Rocky Linux 9.8 + CUDA 12.8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_HEALED = False


def _ensure_portable_runtime_links(root: Path, tp: Path) -> None:
    """Idempotently repair GNU `tar --transform` prefixed relative symlinks (`ShotSpace/...`)
    and `.pth` paths across standalone offline `.venv` runtimes on first call."""
    global _RUNTIME_HEALED
    if _RUNTIME_HEALED:
        return
    _RUNTIME_HEALED = True
    if os.name == "nt":  # Windows environments are made in place by the installer: nothing to repair
        return

    probes = (
        root / ".venv" / "lib64",
        tp / "alembic" / ".venv" / "bin" / "python",
        tp / "fbx" / ".venv" / "bin" / "python",
        tp / "sam_3d_body" / ".venv" / "lib64",
        tp / "sam_3d_objects" / ".venv" / "lib64",
    )
    needs_symlink_fix = False
    for pb in probes:
        try:
            if os.path.islink(pb) and os.readlink(pb).startswith("ShotSpace/"):
                needs_symlink_fix = True
                break
        except OSError:
            pass

    if needs_symlink_fix:
        for scan_root in (root / ".venv", tp):
            if not scan_root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=False):
                for name in dirnames + filenames:
                    p = os.path.join(dirpath, name)
                    try:
                        if os.path.islink(p):
                            tgt = os.readlink(p)
                            if tgt.startswith("ShotSpace/"):
                                fixed = tgt[len("ShotSpace/") :]
                                tmp_p = f"{p}.__ss_fix_{os.getpid()}"
                                try:
                                    os.unlink(tmp_p)
                                except OSError:
                                    pass
                                os.symlink(fixed, tmp_p)
                                os.replace(tmp_p, p)
                    except OSError:
                        pass

    root_str = f"{root}\n"
    venv_dirs = [root / ".venv"]
    if tp.is_dir():
        venv_dirs.extend(p for p in tp.glob("*/.venv") if p.is_dir())
    for vdir in venv_dirs:
        for sp in vdir.glob("lib/python*/site-packages"):
            if not sp.is_dir():
                continue
            try:
                for pth in sp.glob("*shotspace*.pth"):
                    cur = pth.read_text(encoding="utf-8", errors="ignore")
                    if cur != root_str:
                        pth.write_text(root_str, encoding="utf-8")
                root_pth = sp / "shotspace_root.pth"
                if not root_pth.is_file() or root_pth.read_text(encoding="utf-8", errors="ignore") != root_str:
                    root_pth.write_text(root_str, encoding="utf-8")
            except OSError:
                pass

    try:
        from .elf_compat import heal_runtime_elf_tree

        heal_runtime_elf_tree(tp, max_glibc=(2, 34))
    except Exception:
        pass


def venv_python(venv: Path) -> Path:
    """The interpreter of a virtual environment: `bin/python` on Linux, `Scripts/python.exe` on Windows."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var, "").strip()
    return Path(os.path.abspath(Path(val).expanduser() if val else default))


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    third_party: Path
    workspaces: Path
    webui_dist: Path

    # SAM 3D Body (highest quality standard tier: ViTDet-H + DINOv3 + MHR)
    sam3d_root: Path
    sam3d_python: Path
    sam3d_repo: Path
    dinov3_repo: Path
    sam3d_weights: Path
    vitdet_weights: Path

    # SAM 3.1 + 3D Object Reconstruction & Rigid Pose Matching isolated environment
    sam_obj_root: Path
    sam_obj_python: Path
    sam3_repo: Path
    moge_repo: Path
    triposr_repo: Path
    sam3d_objects_repo: Path
    sam_obj_weights: Path
    sam31_weights: Path
    moge_weights: Path
    triposr_weights: Path
    sam3d_objects_weights: Path

    # PyAlembic isolated environment
    alembic_root: Path
    alembic_python: Path

    # Autodesk FBX SDK + fbxio isolated environment
    fbx_root: Path
    fbx_python: Path
    fbx_sdk: Path


def get_paths() -> RuntimePaths:
    tp = _env_path("SHOTSPACE_THIRD_PARTY", PROJECT_ROOT / "third_party")
    _ensure_portable_runtime_links(PROJECT_ROOT, tp)
    ws = _env_path("SHOTSPACE_WORKSPACES", PROJECT_ROOT / "workspaces")
    webui_dist = _env_path("SHOTSPACE_WEBUI_DIST", PROJECT_ROOT / "webui" / "dist")

    sam3d_root = _env_path("SHOTSPACE_SAM3D_ROOT", tp / "sam_3d_body")
    sam3d_python = _env_path("SHOTSPACE_SAM3D_PYTHON", venv_python(sam3d_root / ".venv"))
    sam3d_repo = _env_path("SHOTSPACE_SAM3D_REPO", sam3d_root / "repo")
    dinov3_repo = _env_path("SHOTSPACE_DINOV3_DIR", sam3d_root / "dinov3")
    weights_base = _env_path("SHOTSPACE_WEIGHTS_DIR", sam3d_root / "weights")
    sam3d_weights = weights_base / "sam-3d-body-dinov3"
    vitdet_weights = weights_base / "vitdet"

    sam_obj_root = _env_path("SHOTSPACE_SAM_OBJ_ROOT", tp / "sam_3d_objects")
    sam_obj_python = _env_path("SHOTSPACE_SAM_OBJ_PYTHON", venv_python(sam_obj_root / ".venv"))
    sam3_repo = _env_path("SHOTSPACE_SAM3_REPO", sam_obj_root / "sam3_repo")
    moge_repo = _env_path("SHOTSPACE_MOGE_REPO", sam_obj_root / "moge_repo")
    triposr_repo = _env_path("SHOTSPACE_TRIPOSR_REPO", sam_obj_root / "triposr_repo")
    sam3d_objects_repo = _env_path("SHOTSPACE_SAM3D_OBJECTS_REPO", sam_obj_root / "repo")
    sam_obj_weights = _env_path("SHOTSPACE_SAM_OBJ_WEIGHTS_DIR", sam_obj_root / "weights")
    sam31_weights = sam_obj_weights / "sam3.1"
    moge_weights = sam_obj_weights / "moge-vitl"
    triposr_weights = sam_obj_weights / "TripoSR"
    sam3d_objects_weights = sam_obj_weights / "sam-3d-objects"

    alembic_root = _env_path("SHOTSPACE_ALEMBIC_ROOT", tp / "alembic")
    alembic_python = _env_path("SHOTSPACE_ALEMBIC_PYTHON", venv_python(alembic_root / ".venv"))

    fbx_root = _env_path("SHOTSPACE_FBX_ROOT", tp / "fbx")
    fbx_python = _env_path("SHOTSPACE_FBX_PYTHON", venv_python(fbx_root / ".venv"))
    fbx_sdk = _env_path("SHOTSPACE_FBX_SDK", tp / "_fbx_sdk" / "2020.3.10")

    return RuntimePaths(
        root=PROJECT_ROOT,
        third_party=tp,
        workspaces=ws,
        webui_dist=webui_dist,
        sam3d_root=sam3d_root,
        sam3d_python=sam3d_python,
        sam3d_repo=sam3d_repo,
        dinov3_repo=dinov3_repo,
        sam3d_weights=sam3d_weights,
        vitdet_weights=vitdet_weights,
        sam_obj_root=sam_obj_root,
        sam_obj_python=sam_obj_python,
        sam3_repo=sam3_repo,
        moge_repo=moge_repo,
        triposr_repo=triposr_repo,
        sam3d_objects_repo=sam3d_objects_repo,
        sam_obj_weights=sam_obj_weights,
        sam31_weights=sam31_weights,
        moge_weights=moge_weights,
        triposr_weights=triposr_weights,
        sam3d_objects_weights=sam3d_objects_weights,
        alembic_root=alembic_root,
        alembic_python=alembic_python,
        fbx_root=fbx_root,
        fbx_python=fbx_python,
        fbx_sdk=fbx_sdk,
    )


DEFAULT_PORT = int(os.environ.get("SHOTSPACE_PORT", "8765"))
DEFAULT_GPU = os.environ.get("SHOTSPACE_GPU", "")
