"""Alembic (`.abc`) high-level camera reader & scene exporter wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT, get_paths
from ..core.camera import CameraTrack
from ..core.rigid_object import RigidObjectTrack
from ..core.skeleton import SkinnedCharacter
from ..core.units import Axes, DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, rigid, to_cm
from .scene_arrays import load, pack_export_scene, text


def _run_alembic_worker(job: dict) -> dict:
    paths = get_paths()
    if not paths.alembic_python.is_file():
        raise RuntimeError(f"ABC 导出没有安装（重新运行安装程序并选择安装 ABC / FBX 导出）：{paths.alembic_python}")
    with tempfile.TemporaryDirectory(prefix="shotspace_abc_") as td:
        job_file = Path(td) / "job.json"
        res_file = Path(td) / "result.json"
        payload = {**job, "result": str(res_file)}
        job_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        abc_lib = paths.alembic_python.parent.parent / "lib"
        if abc_lib.is_dir():
            cur_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{abc_lib}:{cur_ld}" if cur_ld else str(abc_lib)
        worker_script = PROJECT_ROOT / "shotspace" / "io" / "workers" / "alembic_worker.py"
        proc = subprocess.run(
            [str(paths.alembic_python), str(worker_script), str(job_file)],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"Alembic 处理失败：{err}")
        return json.loads(res_file.read_text(encoding="utf-8")) if res_file.is_file() else {}


def _import_abc_arrays(path: str | Path, cameras_only: bool = False) -> tuple[dict, dict[str, list[dict]]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"未找到 Alembic 文件：{p}")
    with tempfile.TemporaryDirectory(prefix="shotspace_abc_in_") as td:
        out_npz = Path(td) / "scene.npz"
        _run_alembic_worker({"mode": "import", "input": str(p), "output": str(out_npz), "cameras_only": bool(cameras_only)})
        return load(out_npz)


def list_abc_contents(path: str | Path) -> dict:
    top, items = _import_abc_arrays(path)
    fps = float(top["fps"])
    fps_recorded = bool(top.get("fps_recorded", True))
    cameras = []
    for it in items["camera"]:
        frames = np.asarray(it["frames"], dtype=np.int64).reshape(-1)
        focals = np.asarray(it["focal_mm"], dtype=np.float64).reshape(-1)
        haps = np.asarray(it["h_aperture_mm"], dtype=np.float64).reshape(-1)
        res = np.asarray(it["resolution"], dtype=np.int64).tolist() if "resolution" in it else None
        c_world = np.asarray(it["world"], dtype=np.float64).reshape(-1, 4, 4) if "world" in it else np.zeros((0, 4, 4))
        travel_raw = (
            float(np.linalg.norm(np.diff(c_world[:, :3, 3], axis=0), axis=1).sum()) if len(c_world) > 1 else 0.0
        )
        cameras.append(
            {
                "name": text(it["name"]),
                "path": text(it["path"]),
                "frames": [int(frames[0]), int(frames[-1])] if len(frames) else [],
                "frame_count": int(len(frames)),
                "focal_mm": round(float(focals[0]), 3) if len(focals) else 50.0,
                "h_aperture_mm": round(float(haps[0]), 3) if len(haps) else 36.0,
                "travel_raw": round(travel_raw, 4),
                "resolution": res,
            }
        )
    models = []
    for it in items["model"]:
        m_frames = [int(f) for f in np.asarray(it["frames"], dtype=np.int64).reshape(-1)]
        vis = (
            [int(v) for v in np.asarray(it["visible"], dtype=np.int32).reshape(-1)]
            if "visible" in it
            else [1] * len(m_frames)
        )
        pts = np.asarray(it["points"], dtype=np.float32) if "points" in it else None
        world = np.asarray(it["world"], dtype=np.float64) if "world" in it else None
        cnts = np.asarray(it["counts"], dtype=np.int64).reshape(-1) if "counts" in it else np.zeros(0, dtype=np.int64)
        f_cnt = int(np.maximum(0, cnts - 2).sum()) if len(cnts) else 0
        is_rigid = bool(np.asarray(it.get("is_rigid_object", False)).reshape(-1)[0])
        if pts is not None and pts.ndim == 3 and pts.size > 0:
            v_cnt = int(pts.shape[1])
            centroids = pts.mean(axis=1).tolist()
        elif pts is not None and pts.ndim == 2 and pts.size > 0:
            v_cnt = int(pts.shape[0])
            local_c = pts.mean(axis=0)
            if world is not None and len(world) == len(m_frames):
                centroids = [(w[:3, :3] @ local_c + w[:3, 3]).tolist() for w in world]
            else:
                centroids = [local_c.tolist()] * len(m_frames)
        else:
            v_cnt = 0
            centroids = []
        models.append(
            {
                "name": text(it["name"]),
                "path": text(it["path"]),
                "frames": m_frames,
                "visible": vis,
                "is_rigid_object": is_rigid,
                "vertex_count": v_cnt,
                "face_count": f_cnt,
                "centroids": centroids,
                "world_transforms": world.tolist() if (is_rigid and world is not None) else [],
            }
        )
    return {
        "format": "abc",
        "fps": fps,
        "fps_recorded": fps_recorded,
        "cameras": cameras,
        "models": models,
        "characters": [],
        "takes": [],
        "auto_take": "",
    }


def read_abc_camera(
    path: str | Path,
    camera_path: str = "",
    unit: str = "cm",
    up: str = "y",
    default_width: int = DEFAULT_WIDTH,
    default_height: int = DEFAULT_HEIGHT,
) -> CameraTrack:
    p = Path(path).expanduser().resolve()
    top, items = _import_abc_arrays(p, cameras_only=True)
    cams = items["camera"]
    if not cams:
        raise ValueError(f"Alembic 文件 {p.name} 中未找到任何 ICamera 摄影机节点")
    chosen = None
    target_path = camera_path
    if not target_path and len(cams) > 1:
        from . import pick_default_camera_path

        cand_list = []
        for it in cams:
            fr = np.asarray(it["frames"], dtype=np.int64).reshape(-1)
            cw = np.asarray(it["world"], dtype=np.float64).reshape(-1, 4, 4) if "world" in it else np.zeros((0, 4, 4))
            tr = float(np.linalg.norm(np.diff(cw[:, :3, 3], axis=0), axis=1).sum()) if len(cw) > 1 else 0.0
            cand_list.append({"name": text(it["name"]), "path": text(it["path"]), "frame_count": len(fr), "travel_raw": tr})
        target_path = pick_default_camera_path(cand_list)

    if target_path:
        for it in cams:
            if text(it["path"]) == target_path or text(it["key"]) == target_path or text(it["name"]) == target_path:
                chosen = it
                break
        if chosen is None and camera_path:
            raise ValueError(f"Alembic 文件 {p.name} 中不存在路径为 {camera_path} 的摄影机")
    if chosen is None:
        chosen = cams[0]

    use_u = "cm" if unit.strip().lower() == "auto" else unit
    axes = Axes.of(to_cm(use_u), up)
    frames = tuple(int(f) for f in np.asarray(chosen["frames"], dtype=np.int64).reshape(-1))
    world = rigid(axes.matrices(np.asarray(chosen["world"], dtype=np.float64)))
    res = (
        tuple(int(v) for v in np.asarray(chosen["resolution"]).reshape(-1)[:2])
        if "resolution" in chosen
        else (default_width, default_height)
    )
    over_frac = np.asarray(chosen.get("overscan", np.zeros(4)), dtype=np.float64).reshape(4)
    overscan_px = (
        int(round(over_frac[0] * res[0])),
        int(round(over_frac[1] * res[1])),
        int(round(over_frac[2] * res[0])),
        int(round(over_frac[3] * res[1])),
    )
    props = json.loads(text(chosen["properties"])) if "properties" in chosen else {}

    return CameraTrack(
        name=text(chosen["name"]),
        path=text(chosen["path"]),
        frames=frames,
        cam_to_world=world,
        focal_mm=np.asarray(chosen["focal_mm"], dtype=np.float64),
        h_aperture_mm=np.asarray(chosen["h_aperture_mm"], dtype=np.float64),
        v_aperture_mm=np.asarray(chosen["v_aperture_mm"], dtype=np.float64),
        center_mm=np.asarray(chosen["center_mm"], dtype=np.float64),
        width=int(res[0]),
        height=int(res[1]),
        pixel_aspect=float(chosen.get("pixel_aspect", 1.0)),
        overscan=overscan_px,
        fps=float(top["fps"]),
        properties=props,
        source_file=str(p),
        own_frames=frames,
    )


def export_abc_scene(
    out_path: str | Path,
    characters: list[SkinnedCharacter],
    cameras: list[CameraTrack],
    shot_frames: list[int],
    fps: float = DEFAULT_FPS,
    unit: str = "cm",
    rigid_objects: list[RigidObjectTrack] | None = None,
    top_group: str | None = None,
) -> Path:
    out = Path(out_path).expanduser().resolve()
    sa = pack_export_scene(
        characters=characters,
        cameras=cameras,
        shot_frames=shot_frames,
        fps=fps,
        bake_meshes_for_alembic=True,
        unit=unit,
        rigid_objects=rigid_objects,
        top_group=top_group,
    )
    with tempfile.TemporaryDirectory(prefix="shotspace_abc_out_") as td:
        in_npz = Path(td) / "scene.npz"
        sa.save(in_npz)
        _run_alembic_worker({"mode": "export", "input": str(in_npz), "output": str(out)})
    return out
