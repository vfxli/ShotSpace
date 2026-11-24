"""Unified multi-format 3D camera / character reader and scene exporter (`.usd`/`.usda`/`.usdc`, `.abc`, `.fbx`).

Note: `ocio_color` and `plate` (`PyOpenColorIO` / `OpenImageIO`) are loaded lazily via `__getattr__` so isolated
`alembic_worker`, `fbx_worker`, and `sam3d_worker` subprocesses can import `shotspace.io.scene_arrays` /
`shotspace.io.plate_grade` with pure NumPy only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.camera import CameraTrack
from ..core.rigid_object import RigidObjectTrack
from ..core.skeleton import SkinnedCharacter
from ..core.units import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH
from .plate_grade import GradeParams, apply_grade_rgb, build_uint8_lut
from .scene_scan_io import SCAN_EXTS, SceneScanData, load_scene_scan

if TYPE_CHECKING:
    from .ocio_color import OcioSettings, convert_frame_to_srgb_u8, list_ocio_options, resolve_input_colorspace
    from .plate import (
        ARCHIVE_EXTS,
        SEQUENCE_EXTS,
        PlateInfo,
        extract_sequence_archive,
        ingest_sequence_dir,
        prepare_solver_srgb_frames,
        read_image_rgb,
        write_exr_half_rgb,
    )

USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}

_OCIO_ATTRS = {"OcioSettings", "convert_frame_to_srgb_u8", "list_ocio_options", "resolve_input_colorspace"}
_PLATE_ATTRS = {
    "ARCHIVE_EXTS",
    "SEQUENCE_EXTS",
    "PlateInfo",
    "extract_sequence_archive",
    "ingest_sequence_dir",
    "prepare_solver_srgb_frames",
    "read_image_rgb",
    "write_exr_half_rgb",
}


def __getattr__(name: str) -> Any:
    if name in _OCIO_ATTRS:
        from . import ocio_color

        return getattr(ocio_color, name)
    if name in _PLATE_ATTRS:
        from . import plate

        return getattr(plate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_DCC_DEFAULT_CAM_NAMES = {
    "persp",
    "perspshape",
    "top",
    "topshape",
    "front",
    "frontshape",
    "side",
    "sideshape",
    "bottom",
    "back",
    "left",
    "right",
    "ortho",
    "orthographic",
}


def pick_default_camera_path(cameras: list[dict]) -> str:
    """Choose the best default shot camera when a multi-camera file is first imported:
    prefers non-viewport (`persp`/`top`/`front`/`side`) animated cameras with real frame spans."""
    if not cameras:
        return ""
    if len(cameras) == 1:
        return str(cameras[0].get("path") or "")

    def _score(c: dict) -> tuple[int, int, int, float]:
        nm = str(c.get("name") or "").strip().lower()
        leaf = str(c.get("path") or "").rstrip("/").split("/")[-1].strip().lower()
        is_dcc_view = nm in _DCC_DEFAULT_CAM_NAMES or leaf in _DCC_DEFAULT_CAM_NAMES
        f_cnt = int(c.get("frame_count") or (c["frames"][1] - c["frames"][0] + 1 if c.get("frames") else 0))
        travel = float(c.get("travel_raw") or 0.0)
        return (0 if is_dcc_view else 1, 1 if f_cnt > 1 else 0, f_cnt, travel)

    best = max(cameras, key=_score)
    return str(best.get("path") or "")


def build_hierarchy_tree(
    cameras: list[dict] | None = None,
    models: list[dict] | None = None,
    characters: list[dict] | None = None,
) -> list[dict]:
    """Build a nested Outliner-style hierarchy tree from `/a/b/c` scene-graph paths across `.abc`, `.fbx`, and `.usd`."""
    nodes_by_path: dict[str, dict] = {}
    roots: list[dict] = []

    def _ensure_path(full_path: str) -> dict:
        clean = "/" + "/".join(seg for seg in str(full_path or "/node").split("/") if seg)
        if clean in nodes_by_path:
            return nodes_by_path[clean]
        segs = [s for s in clean.split("/") if s]
        cur_parent_list = roots
        cur_prefix = ""
        last_node: dict | None = None
        for seg in segs:
            cur_prefix = f"{cur_prefix}/{seg}"
            node = nodes_by_path.get(cur_prefix)
            if node is None:
                node = {
                    "name": seg,
                    "path": cur_prefix,
                    "kind": "group",
                    "camera": None,
                    "mesh": None,
                    "character": None,
                    "camera_count": 0,
                    "mesh_count": 0,
                    "face_count": 0,
                    "children": [],
                }
                nodes_by_path[cur_prefix] = node
                cur_parent_list.append(node)
            cur_parent_list = node["children"]
            last_node = node
        assert last_node is not None
        return last_node

    for cam in cameras or []:
        nd = _ensure_path(str(cam.get("path") or f"/{cam.get('name', 'camera')}"))
        nd["kind"] = "camera"
        nd["camera"] = {
            "name": cam.get("name", nd["name"]),
            "path": nd["path"],
            "frames": cam.get("frames", []),
            "frame_count": int(cam.get("frame_count") or 0),
            "focal_mm": cam.get("focal_mm", 50.0),
            "travel_raw": round(float(cam.get("travel_raw") or 0.0), 4),
        }

    for mdl in models or []:
        nd = _ensure_path(str(mdl.get("path") or f"/{mdl.get('name', 'mesh')}"))
        if nd["kind"] == "group":
            nd["kind"] = "mesh"
        f_cnt = int(mdl.get("face_count") or 0)
        v_cnt = int(mdl.get("vertex_count") or 0)
        nd["mesh"] = {
            "name": mdl.get("name", nd["name"]),
            "path": nd["path"],
            "vertex_count": v_cnt,
            "face_count": f_cnt,
        }

    for ch in characters or []:
        nd = _ensure_path(str(ch.get("path") or f"/{ch.get('name', 'character')}"))
        if nd["kind"] == "group":
            nd["kind"] = "character"
        nd["character"] = {"name": ch.get("name", nd["name"]), "path": nd["path"], "joints": int(ch.get("joints") or 0)}

    def _rollup(node: dict) -> tuple[int, int, int]:
        c_tot = 1 if node["camera"] is not None else 0
        m_tot = 1 if node["mesh"] is not None else 0
        f_tot = int(node["mesh"]["face_count"]) if node["mesh"] is not None else 0
        for child in node["children"]:
            cc, mc, fc = _rollup(child)
            c_tot += cc
            m_tot += mc
            f_tot += fc
        node["camera_count"] = c_tot
        node["mesh_count"] = m_tot
        node["face_count"] = f_tot
        return c_tot, m_tot, f_tot

    for r in roots:
        _rollup(r)
    return roots


def inspect_3d_file(path: str | Path, take: str = "") -> dict:
    p = Path(path).expanduser().resolve()
    ext = p.suffix.lower()
    if ext in USD_EXTS:
        from .usd_io import list_usd_contents

        res = list_usd_contents(p)
    elif ext == ".abc":
        from .abc_io import list_abc_contents

        res = list_abc_contents(p)
    elif ext == ".fbx":
        from .fbx_io import list_fbx_contents

        res = list_fbx_contents(p, take=take)
    else:
        raise ValueError(f"不支持的三维文件格式：{p.suffix}（仅支持 .usd/.usda/.usdc/.usdz, .abc, .fbx）")
    res["hierarchy"] = build_hierarchy_tree(res.get("cameras"), res.get("models"), res.get("characters"))
    res["default_camera_path"] = pick_default_camera_path(res.get("cameras") or [])
    return res


def load_camera_track(
    path: str | Path,
    camera_path: str = "",
    unit: str = "auto",
    abc_unit: str = "",
    abc_up: str = "y",
    fbx_take: str = "",
    frame_offset: int = 0,
    default_width: int = DEFAULT_WIDTH,
    default_height: int = DEFAULT_HEIGHT,
) -> CameraTrack:
    p = Path(path).expanduser().resolve()
    ext = p.suffix.lower()
    norm_unit = (unit or "").strip().lower()
    if norm_unit not in ("m", "cm", "auto"):
        norm_unit = "auto"

    if ext in USD_EXTS:
        from .usd_io import read_usd_camera

        cam = read_usd_camera(
            p,
            camera_path=camera_path,
            unit=norm_unit,
            default_width=default_width,
            default_height=default_height,
        )
    elif ext == ".abc":
        from .abc_io import read_abc_camera

        use_abc_unit = norm_unit if norm_unit in ("m", "cm") else (abc_unit.strip().lower() or "cm")
        cam = read_abc_camera(
            p,
            camera_path=camera_path,
            unit=use_abc_unit,
            up=abc_up,
            default_width=default_width,
            default_height=default_height,
        )
    elif ext == ".fbx":
        from .fbx_io import read_fbx_camera

        cam = read_fbx_camera(
            p,
            camera_path=camera_path,
            unit=norm_unit,
            take=fbx_take,
            default_width=default_width,
            default_height=default_height,
        )
    else:
        raise ValueError(f"不支持的摄影机文件格式：{p.suffix}")
    return cam.with_frame_offset(int(frame_offset)) if frame_offset else cam


def load_skinned_characters(path: str | Path, fbx_take: str = "") -> list[SkinnedCharacter]:
    p = Path(path).expanduser().resolve()
    ext = p.suffix.lower()
    if ext in USD_EXTS:
        from .usd_io import read_usd_characters

        return read_usd_characters(p)
    if ext == ".fbx":
        from .fbx_io import read_fbx_characters

        return read_fbx_characters(p, take=fbx_take)
    raise ValueError(f"仅支持读取 .usd/.usdc/.usda 或 .fbx 角色文件（收到 {p.suffix}）")


def export_scene(
    out_path: str | Path,
    characters: list[SkinnedCharacter],
    cameras: list[CameraTrack],
    shot_frames: list[int],
    fps: float = DEFAULT_FPS,
    unit: str = "cm",
    rigid_objects: list[RigidObjectTrack] | None = None,
    usd_mode: str = "skel",
    top_group: str | None = None,
) -> Path:
    """`top_group`: write everything under one top-level group of that name (USD: the root prim; Alembic / FBX: a
    group above the people, objects and camera).  Without it: `/shot` in USD, the root of the file elsewhere."""
    out = Path(out_path).expanduser().resolve()
    ext = out.suffix.lower()
    if ext in (".usd", ".usda", ".usdc"):
        from .usd_io import export_usd_scene

        return export_usd_scene(
            out,
            characters=characters,
            cameras=cameras,
            shot_frames=shot_frames,
            fps=fps,
            unit=unit,
            rigid_objects=rigid_objects,
            usd_mode=usd_mode,
            top_group=top_group,
        )
    if ext == ".abc":
        from .abc_io import export_abc_scene

        return export_abc_scene(
            out,
            characters=characters,
            cameras=cameras,
            shot_frames=shot_frames,
            fps=fps,
            unit=unit,
            rigid_objects=rigid_objects,
            top_group=top_group,
        )
    if ext == ".fbx":
        from .fbx_io import export_fbx_scene

        return export_fbx_scene(
            out,
            characters=characters,
            cameras=cameras,
            shot_frames=shot_frames,
            fps=fps,
            unit=unit,
            rigid_objects=rigid_objects,
            top_group=top_group,
        )
    raise ValueError(f"不支持的导出格式：{out.suffix}（支持 .usdc / .usda / .usd, .abc, .fbx）")


__all__ = [
    "ARCHIVE_EXTS",
    "GradeParams",
    "OcioSettings",
    "PlateInfo",
    "SEQUENCE_EXTS",
    "apply_grade_rgb",
    "build_uint8_lut",
    "convert_frame_to_srgb_u8",
    "export_scene",
    "extract_sequence_archive",
    "ingest_sequence_dir",
    "inspect_3d_file",
    "list_ocio_options",
    "load_camera_track",
    "load_skinned_characters",
    "prepare_solver_srgb_frames",
    "read_image_rgb",
    "resolve_input_colorspace",
    "write_exr_half_rgb",
]
