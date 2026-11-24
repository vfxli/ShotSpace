"""Pure-NumPy `.npz` IPC format (`SceneArrays`) shared between ShotSpace core and isolated
Alembic / FBX worker subprocesses."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..core.camera import CameraTrack
from ..core.names import identifier, join_path, split_path, unique
from ..core.rigid_object import RigidObjectTrack
from ..core.skeleton import SkinnedCharacter, bake_lbs_points
from ..core.units import CM_TO_M

KINDS = ("model", "points", "gaussian", "curves", "camera", "character")
MESH_KEY = re.compile(r"mesh\d+_")


def plural(kind: str) -> str:
    return kind if kind in ("points", "curves") else kind + "s"


class SceneArrays:
    """Builds scene arrays item by item for `.npz` serialization."""

    def __init__(self, fps: float, unit_cm: float = 1.0, axes: np.ndarray | None = None) -> None:
        self.top: dict[str, np.ndarray] = {"fps": np.float64(fps), "unit_cm": np.float64(unit_cm)}
        if axes is not None:
            self.top["axes"] = np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.items: dict[str, list[dict]] = {k: [] for k in KINDS}

    def option(self, param: str, values: list[str], auto: str) -> None:
        self.top[f"options_{param}"] = np.array([str(v) for v in values])
        self.top[f"auto_{param}"] = np.array(str(auto))

    def add(self, kind: str, name: str, path: str, frames, world=None, **arrays) -> dict:
        item = {
            "name": np.array(str(name)),
            "path": np.array(str(path)),
            "frames": np.asarray(frames, dtype=np.int64).reshape(-1),
        }
        if "shown" not in arrays:
            item["shown"] = np.array(str(path))
        if world is not None:
            item["world"] = np.asarray(world, dtype=np.float64).reshape(-1, 4, 4)
        item.update({k: np.asarray(v) for k, v in arrays.items() if v is not None})
        if kind == "character":
            item["meshes"] = []
        self.items[kind].append(item)
        return item

    @staticmethod
    def add_mesh(character: dict, name: str, **arrays) -> None:
        character["meshes"].append(
            {"name": np.array(str(name)), **{k: np.asarray(v) for k, v in arrays.items() if v is not None}}
        )

    def arrays(self) -> dict[str, np.ndarray]:
        out = dict(self.top)
        for kind, items in self.items.items():
            out[f"n_{plural(kind)}"] = np.int64(len(items))
            for i, item in enumerate(items):
                for key, value in item.items():
                    if key != "meshes":
                        out[f"{kind}{i}_{key}"] = value
                if kind == "character":
                    out[f"{kind}{i}_n_meshes"] = np.int64(len(item["meshes"]))
                    for k, mesh in enumerate(item["meshes"]):
                        out.update({f"{kind}{i}_mesh{k}_{key}": v for key, v in mesh.items()})
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp.npz")
        np.savez(tmp, **self.arrays())
        tmp.replace(p)
        return p


def load(path: str | Path | dict) -> tuple[dict, dict[str, list[dict]]]:
    data = dict(np.load(path, allow_pickle=False)) if not isinstance(path, dict) else path
    top = {
        k: data[k]
        for k in data
        if k in ("fps", "fps_recorded", "unit_cm", "axes", "frames") or k.startswith(("options_", "auto_"))
    }
    items: dict[str, list[dict]] = {}
    for kind in KINDS:
        items[kind] = []
        for i in range(int(data.get(f"n_{plural(kind)}", 0))):
            head = f"{kind}{i}_"
            item = {
                k[len(head) :]: v
                for k, v in data.items()
                if k.startswith(head) and not MESH_KEY.match(k[len(head) :])
            }
            if kind == "character":
                item["meshes"] = [
                    {k[len(f"{head}mesh{m}_") :]: v for k, v in data.items() if k.startswith(f"{head}mesh{m}_")}
                    for m in range(int(item.pop("n_meshes", 0)))
                ]
            items[kind].append(item)
        taken: set[str] = set()
        for item in items[kind]:
            if "path" in item:
                item["key"] = np.array(unique(text(item["path"]), taken))
                taken.add(text(item["key"]))
    return top, items


def text(value) -> str:
    value = np.asarray(value)
    v = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def texts(value) -> list[str]:
    return [
        v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
        for v in np.asarray(value).reshape(-1).tolist()
    ]


def needed_samples(values) -> np.ndarray:
    """Which per-frame samples have to be written so that the written ones give back EVERY frame exactly.

    A person who is not in the picture stands still at the pose of the nearest solved frame, so for long stretches
    every frame carries the same values.  Those stretches need no key on every frame: the first sample, the last
    one, and both ends of each stretch of identical samples are enough - between two equal keys the value does not
    move, whatever the interpolation, and everywhere else every frame keeps its own key.  Returns a boolean mask
    over the first axis."""
    arr = np.asarray(values)
    n = int(arr.shape[0])
    keep = np.ones(n, dtype=bool)
    if n > 2:
        flat = arr.reshape(n, -1)
        same_as_before = np.all(flat[1:] == flat[:-1], axis=1)  # [n - 1]: sample i + 1 equals sample i
        keep[1:-1] = ~(same_as_before[:-1] & same_as_before[1:])
    return keep


def held_changes(flags) -> np.ndarray:
    """For a value that holds from key to key (visibility): the first sample and every sample that differs from
    the one before it.  Returns a boolean mask."""
    arr = np.asarray(flags).reshape(-1)
    keep = np.ones(len(arr), dtype=bool)
    if len(arr) > 1:
        keep[1:] = arr[1:] != arr[:-1]
    return keep


def shown_span(item: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every integer frame from item's first sample to its last, and 1/0 visibility (0 across detection gaps!)."""
    frames = np.asarray(item["frames"], dtype=np.int64).reshape(-1)
    given = np.asarray(item.get("visible", np.ones(len(frames))), dtype=np.int32).reshape(-1)
    span = np.arange(frames.min(), frames.max() + 1, dtype=np.int64)
    at = dict(zip(frames.tolist(), given.tolist()))
    return span, np.array([at.get(int(f), 0) for f in span], dtype=np.int32)


def pack_export_scene(
    characters: list[SkinnedCharacter],
    cameras: list[CameraTrack],
    shot_frames: list[int],
    fps: float,
    bake_meshes_for_alembic: bool = False,
    unit: str = "cm",
    rigid_objects: list[RigidObjectTrack] | None = None,
    top_group: str | None = None,
) -> SceneArrays:
    """Pack aligned `SkinnedCharacter`s and `CameraTrack`s into `SceneArrays` for FBX or Alembic export.
    - Supports both `unit="cm"` and `unit="m"` (`scale_len = 0.01`, `unit_cm = 100.0` when `unit == "m"`).
    - Alembic has no skeletal skinning schema, so `bake_meshes_for_alembic=True` bakes each character via LBS into
      a deforming `model` (`OPolyMesh`).
    - `top_group`: everything is written under one top-level group of that name (the product's deliveries).
      Without it the items hang at the root of the file.
    """
    is_meter = unit.strip().lower() == "m"
    scale_len = CM_TO_M if is_meter else 1.0
    sa = SceneArrays(fps=fps, unit_cm=100.0 if is_meter else 1.0)

    def shown_at(path: str) -> str:
        """Where the item hangs in the written file.  The writers drop a leading `/shot` (it only names the
        scene); with a top group everything that is left hangs under that group."""
        if not top_group:
            return path
        parts = split_path(path)
        if parts and parts[0] == "shot":
            parts = parts[1:]
        return join_path([identifier(top_group)] + parts)

    shot_set = sorted(int(f) for f in shot_frames) if shot_frames else []

    for cam in cameras:
        c_frames = list(cam.frames)
        world = cam.cam_to_world.copy()
        world[:, :3, 3] *= scale_len
        over_frac = (
            np.array(
                [
                    cam.overscan[0] / max(cam.width, 1),
                    cam.overscan[1] / max(cam.height, 1),
                    cam.overscan[2] / max(cam.width, 1),
                    cam.overscan[3] / max(cam.height, 1),
                ],
                dtype=np.float64,
            )
            if any(cam.overscan)
            else np.zeros(4, dtype=np.float64)
        )
        c_path = cam.path if cam.path.startswith("/") else f"/shot/{identifier(cam.name)}"
        sa.add(
            "camera",
            cam.name,
            c_path,
            c_frames,
            world=world,
            shown=shown_at(c_path),
            focal_mm=cam.focal_mm,
            h_aperture_mm=cam.h_aperture_mm,
            v_aperture_mm=cam.v_aperture_mm,
            center_mm=cam.center_mm,
            pixel_aspect=np.float64(cam.pixel_aspect),
            overscan=over_frac,
            properties=np.array(json.dumps(cam.properties or {}, ensure_ascii=False)),
            resolution=np.array([int(cam.width), int(cam.height)], dtype=np.int64),
        )

    for ch in characters:
        if not ch.frames:
            continue
        c_path = f"/shot/{identifier(ch.name)}"
        faces = np.asarray(ch.faces, dtype=np.int32)
        counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
        indices = faces.reshape(-1)
        uv_kwargs = {}
        if ch.uv is not None and ch.uv_faces is not None:
            uv_kwargs["uv"] = np.asarray(ch.uv, dtype=np.float32)
            uv_kwargs["uv_indices"] = np.asarray(ch.uv_faces, dtype=np.int32).reshape(-1)

        # Build full shot sequence frames + explicit per-frame visibility (0 on undetected / frozen fly-away frames!)
        ch_vis = ch.visible_mask.astype(np.int32)
        f_to_idx = {int(f): idx for idx, f in enumerate(ch.frames)}
        use_frames = shot_set if shot_set else [int(f) for f in ch.frames]

        if bake_meshes_for_alembic:
            baked = bake_lbs_points(
                ch.rest_points, ch.bind_world, ch.anim_world, ch.joint_indices, ch.joint_weights
            ) * np.float32(scale_len)
            full_pts = []
            vis_list = []
            for f in use_frames:
                if f in f_to_idx:
                    idx_f = f_to_idx[f]
                    full_pts.append(baked[idx_f])
                    vis_list.append(int(ch_vis[idx_f]))
                elif f < ch.frames[0]:
                    full_pts.append(baked[0])
                    vis_list.append(0)
                else:
                    nearest = min(range(len(ch.frames)), key=lambda k: abs(ch.frames[k] - f))
                    full_pts.append(baked[nearest])
                    vis_list.append(0)
            pts_arr = np.stack(full_pts, axis=0)
            vis_arr = np.array(vis_list, dtype=np.int32)
            vis_kw = {} if np.all(vis_arr > 0) else {"visible": vis_arr}
            world_id = np.repeat(np.eye(4, dtype=np.float64)[None], len(use_frames), axis=0)
            sa.add(
                "model",
                ch.name,
                c_path,
                use_frames,
                world=world_id,
                shown=shown_at(c_path),
                counts=counts,
                indices=indices,
                points=pts_arr,
                **uv_kwargs,
                **vis_kw,
            )
        else:
            # Native rigged SkinnedCharacter for FBX (Y-up, cm or m)
            bind_scaled = np.asarray(ch.bind_world, dtype=np.float64).copy()
            bind_scaled[:, :3, 3] *= scale_len
            anim_src = np.asarray(ch.anim_world, dtype=np.float64).copy()
            anim_src[:, :, :3, 3] *= scale_len
            pts_scaled = (np.asarray(ch.rest_points, dtype=np.float32) * np.float32(scale_len)).astype(np.float32)

            anim_list = []
            vis_list = []
            for f in use_frames:
                if f in f_to_idx:
                    idx_f = f_to_idx[f]
                    anim_list.append(anim_src[idx_f])
                    vis_list.append(int(ch_vis[idx_f]))
                elif f < ch.frames[0]:
                    anim_list.append(anim_src[0])
                    vis_list.append(0)
                else:
                    nearest = min(range(len(ch.frames)), key=lambda k: abs(ch.frames[k] - f))
                    anim_list.append(anim_src[nearest])
                    vis_list.append(0)
            anim_arr = np.stack(anim_list, axis=0)
            vis_arr = np.array(vis_list, dtype=np.int32)
            vis_kw = {} if np.all(vis_arr > 0) else {"visible": vis_arr}

            item = sa.add(
                "character",
                ch.name,
                c_path,
                use_frames,
                shown=shown_at(c_path),
                joints=np.array(ch.joint_names),
                parents=np.asarray(ch.parents, dtype=np.int64),
                bind=bind_scaled,
                anim=anim_arr,
                root_at_path=np.array(False),
                **vis_kw,
            )
            sa.add_mesh(
                item,
                "body",
                counts=counts,
                indices=indices,
                points=pts_scaled,
                joint_indices=np.asarray(ch.joint_indices, dtype=np.int32),
                joint_weights=np.asarray(ch.joint_weights, dtype=np.float32),
                **uv_kwargs,
            )

    for obj in rigid_objects or []:
        if not obj.frames:
            continue
        o_path = f"/shot/{identifier(obj.name)}"
        faces = np.asarray(obj.faces, dtype=np.int32)
        counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
        indices = faces.reshape(-1)
        pts_local = (np.asarray(obj.rest_points, dtype=np.float32) * np.float32(scale_len)).astype(np.float32)

        obj_vis = obj.visible_mask.astype(np.int32)
        f_to_idx = {int(f): idx for idx, f in enumerate(obj.frames)}
        use_frames = shot_set if shot_set else [int(f) for f in obj.frames]

        w_src = np.asarray(obj.world_transforms, dtype=np.float64).copy()
        w_src[:, :3, 3] *= scale_len

        world_list = []
        vis_list = []
        for f in use_frames:
            if f in f_to_idx:
                idx_f = f_to_idx[f]
                world_list.append(w_src[idx_f])
                vis_list.append(int(obj_vis[idx_f]))
            elif f < obj.frames[0]:
                world_list.append(w_src[0])
                vis_list.append(0)
            else:
                nearest = min(range(len(obj.frames)), key=lambda k: abs(obj.frames[k] - f))
                world_list.append(w_src[nearest])
                vis_list.append(0)
        world_arr = np.stack(world_list, axis=0)
        vis_arr = np.array(vis_list, dtype=np.int32)
        vis_kw = {} if np.all(vis_arr > 0) else {"visible": vis_arr}

        sa.add(
            "model",
            obj.name,
            o_path,
            use_frames,
            world=world_arr,
            shown=shown_at(o_path),
            counts=counts,
            indices=indices,
            points=pts_local,  # 2D [V, 3] static local proxy mesh at (0,0,0)!
            is_rigid_object=np.array(True),
            **vis_kw,
        )

    return sa
