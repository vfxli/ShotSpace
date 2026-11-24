"""Autodesk FBX (`.fbx`) high-level camera & character reader and scene exporter wrapper."""

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
from ..core.skeleton import SkinnedCharacter, cg_bone_names, cg_orientations
from ..core.units import Axes, DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, rigid, to_cm
from .scene_arrays import load, pack_export_scene, text, texts


def _run_fbx_worker(job: dict) -> dict:
    paths = get_paths()
    if not paths.fbx_python.is_file():
        raise RuntimeError(f"FBX 导出没有安装（重新运行安装程序并选择安装 ABC / FBX 导出）：{paths.fbx_python}")
    with tempfile.TemporaryDirectory(prefix="shotspace_fbx_") as td:
        job_file = Path(td) / "job.json"
        res_file = Path(td) / "result.json"
        payload = {**job, "result": str(res_file)}
        job_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        fbx_lib = paths.fbx_python.parent.parent / "lib"
        if fbx_lib.is_dir():
            cur_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{fbx_lib}:{cur_ld}" if cur_ld else str(fbx_lib)
        worker_script = PROJECT_ROOT / "shotspace" / "io" / "workers" / "fbx_worker.py"
        proc = subprocess.run(
            [str(paths.fbx_python), str(worker_script), str(job_file)],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"FBX 处理失败：{err}")
        return json.loads(res_file.read_text(encoding="utf-8")) if res_file.is_file() else {}


def _import_fbx_arrays(path: str | Path, take: str = "", cameras_only: bool = False) -> tuple[dict, dict[str, list[dict]], dict]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"未找到 FBX 文件：{p}")
    with tempfile.TemporaryDirectory(prefix="shotspace_fbx_in_") as td:
        out_npz = Path(td) / "scene.npz"
        meta = _run_fbx_worker({"mode": "import", "input": str(p), "output": str(out_npz), "take": take, "cameras_only": bool(cameras_only)})
        top, items = load(out_npz)
        return top, items, meta


def list_fbx_contents(path: str | Path, take: str = "") -> dict:
    top, items, meta = _import_fbx_arrays(path, take=take)
    fps = float(top["fps"])
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
    characters = [
        {"name": text(it["name"]), "path": text(it["path"]), "joints": len(texts(it["joints"]))}
        for it in items["character"]
    ]
    models = []
    for it in items["model"]:
        m_frames = [int(f) for f in np.asarray(it["frames"], dtype=np.int64).reshape(-1)]
        vis = (
            [int(v) for v in np.asarray(it["visible"], dtype=np.int32).reshape(-1)]
            if "visible" in it
            else [1] * len(m_frames)
        )
        world = np.asarray(it["world"], dtype=np.float64) if "world" in it else None
        pts = np.asarray(it["points"], dtype=np.float32) if "points" in it else None
        cnts = np.asarray(it["counts"], dtype=np.int64).reshape(-1) if "counts" in it else np.zeros(0, dtype=np.int64)
        f_cnt = int(np.maximum(0, cnts - 2).sum()) if len(cnts) else 0
        models.append(
            {
                "name": text(it["name"]),
                "path": text(it["path"]),
                "frames": m_frames,
                "visible": vis,
                "vertex_count": int(len(pts)) if pts is not None else 0,
                "face_count": f_cnt,
                "world_transforms": world.tolist() if world is not None else [],
            }
        )
    return {
        "format": "fbx",
        "fps": fps,
        "unit_to_cm": float(top.get("unit_cm", 1.0)),
        "cameras": cameras,
        "characters": characters,
        "models": models,
        "takes": meta.get("takes", []),
        "auto_take": meta.get("auto_take", ""),
        "chosen_take": meta.get("chosen_take", ""),
    }


def read_fbx_camera(
    path: str | Path,
    camera_path: str = "",
    unit: str = "auto",
    take: str = "",
    default_width: int = DEFAULT_WIDTH,
    default_height: int = DEFAULT_HEIGHT,
) -> CameraTrack:
    p = Path(path).expanduser().resolve()
    top, items, _ = _import_fbx_arrays(p, take=take, cameras_only=True)
    cams = items["camera"]
    if not cams:
        raise ValueError(f"FBX 文件 {p.name} 中未找到任何 FbxCamera 摄影机节点")
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
            raise ValueError(f"FBX 文件 {p.name} 中不存在路径为 {camera_path} 的摄影机")
    if chosen is None:
        chosen = cams[0]

    u_norm = (unit or "auto").strip().lower()
    unit_scale = float(top.get("unit_cm", 1.0)) if u_norm == "auto" else to_cm(u_norm)
    axes = Axes(unit_scale, np.asarray(top.get("axes", np.eye(3)), dtype=np.float64))
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


def read_fbx_characters(path: str | Path, take: str = "") -> list[SkinnedCharacter]:
    p = Path(path).expanduser().resolve()
    top, items, _ = _import_fbx_arrays(p, take=take)
    axes = Axes(float(top.get("unit_cm", 1.0)), np.asarray(top.get("axes", np.eye(3)), dtype=np.float64))
    out: list[SkinnedCharacter] = []
    for idx, it in enumerate(items["character"]):
        if not it["meshes"]:
            continue
        all_frames = [int(f) for f in np.asarray(it["frames"], dtype=np.int64).reshape(-1)]
        vis_arr = np.asarray(it.get("visible", np.ones(len(all_frames))), dtype=np.int32).reshape(-1)
        frames = all_frames

        raw_joints = texts(it["joints"])
        parents = np.asarray(it["parents"], dtype=np.int64)
        bind_w = axes.matrices(np.asarray(it["bind"], dtype=np.float64))
        anim_w = axes.matrices(np.asarray(it["anim"], dtype=np.float64))
        std_names = cg_bone_names(raw_joints, parents)
        bind_std, anim_std = cg_orientations(parents, bind_w, anim_w)

        m = it["meshes"][0]
        pts = axes.world_points(np.asarray(m["points"], dtype=np.float64)).astype(np.float32)
        counts = np.asarray(m["counts"], dtype=np.int32)
        f_idx = np.asarray(m["indices"], dtype=np.int32)
        if np.all(counts == 3):
            faces = f_idx.reshape(-1, 3)
        else:
            tris = []
            pos = 0
            for c in counts:
                poly = f_idx[pos : pos + c]
                for k in range(1, c - 1):
                    tris.append([poly[0], poly[k], poly[k + 1]])
                pos += c
            faces = np.asarray(tris, dtype=np.int32)

        uv = np.asarray(m["uv"], dtype=np.float32) if "uv" in m else None
        uv_faces = (
            np.asarray(m["uv_indices"], dtype=np.int32).reshape(-1, 3)
            if ("uv_indices" in m and np.all(counts == 3))
            else None
        )
        out.append(
            SkinnedCharacter(
                name=text(it["name"]) or f"person_{idx + 1:02d}",
                person_id=idx + 1,
                frames=frames,
                joint_names=std_names,
                parents=parents,
                bind_world=bind_std,
                anim_world=anim_std,
                rest_points=pts,
                faces=faces,
                joint_indices=np.asarray(m["joint_indices"], dtype=np.int32),
                joint_weights=np.asarray(m["joint_weights"], dtype=np.float32),
                uv=uv,
                uv_faces=uv_faces,
                visible=vis_arr,
                custom_data={"lab2shot:person_id": idx + 1},
            )
        )
    return out


def export_fbx_scene(
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
        bake_meshes_for_alembic=False,
        unit=unit,
        rigid_objects=rigid_objects,
        top_group=top_group,
    )
    with tempfile.TemporaryDirectory(prefix="shotspace_fbx_out_") as td:
        in_npz = Path(td) / "scene.npz"
        sa.save(in_npz)
        _run_fbx_worker({"mode": "export", "input": str(in_npz), "output": str(out)})
    return out
