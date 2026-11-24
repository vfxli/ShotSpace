"""Standalone Autodesk FBX worker running under `third_party/fbx/.venv/bin/python` (`fbxio` API >= 8).

Usage:
    python -m shotspace.io.workers.fbx_worker <job.json>

Supports:
  - `mode: "import"`: `.fbx` (+ optional `take`) -> `scene.npz` (cameras with `aim=True` + `-90° Y` `FBX_TO_CAMERA`,
    rigged characters for standalone re-alignment, takes listing with auto-longest animated take).
  - `mode: "export"`: `scene.npz` -> `.fbx` (Maya Y-up cm, `FbxSkeleton` + `FbxSkin` + `FbxPose`,
    cameras with `orthonormal` + `+90° Y` `CAMERA_TO_FBX`, stepped `Visibility` keys across gaps).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import fbxio
import numpy as np

from shotspace.core.names import join_path, split_path
from shotspace.core.poses import orthonormal
from shotspace.core.units import standard_fps
from shotspace.io.scene_arrays import SceneArrays, load, needed_samples, shown_span, text, texts

API_MIN = 8
OVERSCAN_PROP = "lab2shot:lens:overscan"

# Our camera (-Z forward, +Y up) = FBX camera (+X forward, +Y up) rotated -90 deg about Y
FBX_TO_CAMERA = np.array(
    [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
CAMERA_TO_FBX = np.linalg.inv(FBX_TO_CAMERA)
AXES = "xyz"
FRONT = {
    ("y", "even"): "x",
    ("y", "odd"): "z",
    ("z", "even"): "x",
    ("z", "odd"): "y",
    ("x", "even"): "y",
    ("x", "odd"): "z",
}


def axes_of(info: dict) -> np.ndarray:
    if not info["right_handed"]:
        raise ValueError("FBX 文件使用了左手坐标系，无法无镜像导入；请在 DCC 中以右手系（Maya Y-up）导出")
    e = np.eye(3, dtype=np.float64)
    up = info["up_sign"] * e[AXES.index(info["up"])]
    front = info["front_sign"] * e[AXES.index(FRONT[(info["up"], info["front_parity"])])]
    return np.stack([np.cross(up, front), up, front])


def choose_take(scene, wanted: str) -> tuple[list[str], str, str]:
    takes = scene.takes()
    names = [t["name"] for t in takes]
    if not takes:
        return names, "", ""
    animated = [t for t in takes if t["keys"] > 0]
    default = max(animated, key=lambda t: t["stop"] - t["start"])["name"] if animated else names[0]
    if wanted and wanted not in names:
        raise ValueError(f"FBX 中不存在动画层 {wanted!r}（可选：{names}）")
    chosen = wanted or default
    scene.set_take(chosen)
    return names, default, chosen


class Reader:
    def __init__(self, scene) -> None:
        self.scene = scene
        self.info = scene.info()
        self.nodes = scene.nodes()
        self.start = round(self.info["start"])
        self.keys = {n["id"]: scene.key_frames(n["id"]) for n in self.nodes}

    def path(self, i: int) -> str:
        ns = []
        while i >= 0:
            ns.append(self.nodes[i]["name"])
            i = self.nodes[i]["parent"]
        return join_path(reversed(ns))

    def ancestors(self, i: int) -> list[int]:
        out = []
        while i >= 0:
            out.append(i)
            i = self.nodes[i]["parent"]
        return out

    def frames(self, ids) -> np.ndarray:
        keys = [k for i in {a for n in ids for a in self.ancestors(n)} for k in self.keys[i]]
        if not keys:
            return np.array([self.start], dtype=np.int64)
        first, last = math.ceil(min(keys) - 1e-6), math.floor(max(keys) + 1e-6)
        return np.arange(first, max(first, last) + 1, dtype=np.int64)

    def shown(self, i: int, frames) -> np.ndarray:
        return np.all(
            [self.scene.visibility(a, np.asarray(frames, dtype=np.float64)) > 0.5 for a in self.ancestors(i)],
            axis=0,
        )

    def visible(self, i: int, frames) -> dict:
        s = self.shown(i, frames)
        return {} if s.all() else {"visible": s.astype(np.int32)}

    def world(self, i: int, frames, aim: bool = False) -> np.ndarray:
        return self.scene.global_matrices(i, np.asarray(frames, dtype=np.float64), aim)


def do_import(fbx_path: Path, out_npz: Path, wanted_take: str = "", cameras_only: bool = False) -> dict:
    if getattr(fbxio, "API", 1) < API_MIN:
        raise RuntimeError(f"fbxio C++ 扩展版本过低（需要 >= {API_MIN}）")
    scene = fbxio.open(str(fbx_path))
    takes, auto_take, chosen_take = choose_take(scene, wanted_take)
    r = Reader(scene)
    out = SceneArrays(r.info["fps"], r.info["unit_cm"], axes_of(r.info))
    if takes:
        out.option("take", takes, auto_take)

    # Read cameras (including Aim-target / constraint-driven cameras whose keys live on LookAt or sibling nodes!)
    t_start = int(round(r.info["start"]))
    t_stop = int(round(r.info["stop"]))
    cam_ids = [n["id"] for n in r.nodes if n["kind"] == "camera"]
    for i in cam_ids:
        frames = r.frames([i])
        if len(frames) <= 1 and t_stop > t_start and (t_stop - t_start) <= 20000:
            probes = np.unique(np.linspace(t_start, t_stop, min(9, t_stop - t_start + 1), dtype=np.int64))
            pw = r.world(i, probes, aim=True)
            pl = r.scene.camera(i, probes.astype(np.float64))
            pfoc = np.asarray(pl["focal_mm"], dtype=np.float64)
            if float(np.max(np.abs(pw - pw[:1]))) > 1e-6 or float(np.max(np.abs(pfoc - pfoc[:1]))) > 1e-4:
                frames = np.arange(t_start, t_stop + 1, dtype=np.int64)

        lens = r.scene.camera(i, frames.astype(np.float64))
        n = len(frames)
        extra = {}
        if lens["aspect_mode"] == "fixed_resolution" and lens["aspect_width"] > 0 and lens["aspect_height"] > 0:
            extra["resolution"] = np.array([round(lens["aspect_width"]), round(lens["aspect_height"])], dtype=np.int64)
        props = dict(lens["properties"])
        overscan = json.loads(props.pop(OVERSCAN_PROP, "[0, 0, 0, 0]"))
        focal_arr = np.asarray(lens["focal_mm"], dtype=np.float64)
        focal_arr = np.where(focal_arr > 0.05, focal_arr, 35.0)
        fw_mm = float(lens["film_width_mm"]) if float(lens["film_width_mm"]) > 0.05 else 36.0
        fh_mm = float(lens["film_height_mm"]) if float(lens["film_height_mm"]) > 0.05 else 20.25
        pa = float(lens["pixel_aspect"]) if float(lens["pixel_aspect"]) > 1e-4 else 1.0
        out.add(
            "camera",
            r.nodes[i]["name"],
            r.path(i),
            frames,
            r.world(i, frames, aim=True) @ FBX_TO_CAMERA,
            focal_mm=focal_arr,
            h_aperture_mm=np.full(n, fw_mm, dtype=np.float64),
            v_aperture_mm=np.full(n, fh_mm, dtype=np.float64),
            center_mm=np.asarray(lens["center_mm"], dtype=np.float64),
            pixel_aspect=np.float64(pa),
            overscan=np.asarray(overscan, dtype=np.float64),
            properties=np.array(json.dumps(props, ensure_ascii=False)),
            **extra,
            **r.visible(i, frames),
        )

    if cameras_only:
        out.save(out_npz)
        return {
            "takes": takes,
            "auto_take": auto_take,
            "chosen_take": chosen_take,
            "cameras": len(out.items["camera"]),
            "characters": 0,
        }

    # Read rigged characters (supports importing existing FBX solves for standalone Re-Align mode)
    joint_set = {n["id"] for n in r.nodes if n["kind"] == "skeleton"}
    above = {i: next((a for a in r.ancestors(i)[1:] if a in joint_set), -1) for i in joint_set}
    roots = [i for i in sorted(joint_set) if above[i] < 0]
    member = {}
    rigs = []
    for c, root in enumerate(roots):
        ids = [i for i in sorted(joint_set) if root in r.ancestors(i)]
        member.update({i: (c, k) for k, i in enumerate(ids)})
        rigs.append({"root": root, "ids": ids, "meshes": []})
    skins = {}
    for n in r.nodes:
        if n["kind"] != "mesh":
            continue
        skin = r.scene.skin(n["id"])
        links = [cl["link"] for cl in skin["clusters"]] if skin else []
        owners = [member[j][0] for j in links if j in member]
        if owners:
            skins[n["id"]] = skin
            rigs[max(set(owners), key=owners.count)]["meshes"].append(n["id"])
    pose = r.scene.bind_pose()
    for rig in rigs:
        ids = rig["ids"]
        index = {i: k for k, i in enumerate(ids)}
        parents = np.array([index.get(above[i], -1) for i in ids], dtype=np.int64)
        frames = r.frames(ids + rig["meshes"])
        anim = np.stack([r.world(i, frames) for i in ids], axis=1)
        bind = [None] * len(ids)
        for m_id in rig["meshes"]:
            for cl in skins[m_id]["clusters"]:
                k = index.get(cl["link"])
                if k is not None and bind[k] is None:
                    bind[k] = np.asarray(cl["transform_link"])
        for k, i in enumerate(ids):
            if bind[k] is None and i in pose:
                bind[k] = np.asarray(pose[i])
            if bind[k] is None:
                g = r.world(i, [r.start])[0]
                if parents[k] >= 0:
                    p_now = r.world(ids[parents[k]], [r.start])[0]
                    g = bind[parents[k]] @ np.linalg.inv(p_now) @ g
                bind[k] = g
        bind_arr = np.stack(bind)
        meshes_shown = {m_id: r.shown(m_id, frames) for m_id in rig["meshes"]}
        shown = r.shown(rig["root"], frames) & (
            np.any(list(meshes_shown.values()), axis=0) if meshes_shown else True
        )
        item = out.add(
            "character",
            r.nodes[rig["root"]]["name"],
            r.path(rig["root"]),
            frames,
            joints=np.array([r.nodes[i]["name"] for i in ids]),
            parents=parents,
            bind=bind_arr,
            anim=anim,
            root_at_path=True,
            **({} if shown.all() else {"visible": shown.astype(np.int32)}),
        )
        for m_id in rig["meshes"]:
            m = r.scene.mesh(m_id)
            clusters = [cl for cl in skins[m_id]["clusters"] if cl["link"] in index]
            place = np.asarray(clusters[0]["transform"]) @ r.scene.geometric(m_id)
            base = np.asarray(m["points"], dtype=np.float64)
            points = base @ place[:3, :3].T + place[:3, 3]
            influences = [[] for _ in range(len(base))]
            for cl in clusters:
                for v, w in zip(np.asarray(cl["indices"]), np.asarray(cl["weights"])):
                    if w > 0:
                        influences[int(v)].append((index[cl["link"]], float(w)))
            for v, inf in enumerate(influences):
                tot = sum(w for _, w in inf)
                if tot > 0:
                    inf[:] = [(j, w / tot) for j, w in inf]
                else:
                    inf[:] = [(int(np.argmin(np.linalg.norm(bind_arr[:, :3, 3] - points[v], axis=1))), 1.0)]
            width = max(1, max(len(x) for x in influences))
            j_idx = np.zeros((len(base), width), dtype=np.int32)
            j_wts = np.zeros((len(base), width), dtype=np.float32)
            for v, inf in enumerate(influences):
                for s, (j, w) in enumerate(inf):
                    j_idx[v, s], j_wts[v, s] = j, w
            m_kw = {
                "counts": np.asarray(m["counts"], dtype=np.int32),
                "indices": np.asarray(m["indices"], dtype=np.int32),
                "points": points.astype(np.float32),
                "joint_indices": j_idx,
                "joint_weights": j_wts,
            }
            if m["uv"] is not None:
                m_kw["uv"] = np.asarray(m["uv"], dtype=np.float32)
                m_kw["uv_indices"] = np.asarray(m["uv_indices"], dtype=np.int32)
            out.add_mesh(item, r.nodes[m_id]["name"], **m_kw)

    # Read unskinned rigid mesh objects (parent transform carries TRS + Visibility, child mesh carries static vertices)
    child_counts: dict[int, int] = {}
    for n in r.nodes:
        if n["parent"] >= 0:
            child_counts[n["parent"]] = child_counts.get(n["parent"], 0) + 1
    for n in r.nodes:
        if n["kind"] != "mesh" or n["id"] in skins:
            continue
        m_id = n["id"]
        p_id = n["parent"]
        xf_id = (
            p_id
            if (
                p_id >= 0
                and r.nodes[p_id]["kind"] not in ("camera", "skeleton", "mesh")
                and child_counts.get(p_id, 0) == 1
            )
            else m_id
        )
        frames = r.frames([xf_id, m_id])
        world = r.world(m_id, frames)
        shown = r.shown(m_id, frames)
        m = r.scene.mesh(m_id)
        geom = r.scene.geometric(m_id)
        base = np.asarray(m["points"], dtype=np.float64)
        pts = (base @ geom[:3, :3].T + geom[:3, 3]).astype(np.float32)
        out.add(
            "model",
            r.nodes[xf_id]["name"],
            r.path(xf_id),
            frames,
            world=world,
            counts=np.asarray(m["counts"], dtype=np.int32),
            indices=np.asarray(m["indices"], dtype=np.int32),
            points=pts,
            is_rigid_object=np.array(True),
            **({} if shown.all() else {"visible": shown.astype(np.int32)}),
        )

    out.save(out_npz)
    return {
        "fps": r.info["fps"],
        "unit_cm": r.info["unit_cm"],
        "takes": takes,
        "auto_take": auto_take,
        "chosen_take": chosen_take,
        "cameras": len(cam_ids),
        "characters": len(rigs),
    }


def do_export(in_npz: Path, out_fbx: Path) -> dict:
    if getattr(fbxio, "API", 1) < API_MIN:
        raise RuntimeError(f"fbxio C++ 扩展版本过低（需要 >= {API_MIN}）")
    top, items = load(in_npz)
    fps = standard_fps(float(top["fps"]))
    all_frames = [int(f) for kind in items.values() for it in kind for f in np.asarray(it["frames"]).reshape(-1)]
    start, stop = (min(all_frames), max(all_frames)) if all_frames else (1, 1)
    scene = fbxio.create(fps, float(start), float(stop))
    unit_cm = float(top.get("unit_cm", 1.0))
    if abs(unit_cm - 1.0) > 1e-6:
        scene.set_axes("maya_y", unit_cm)
    groups: dict[str, int] = {}

    def under(path: str) -> tuple[int, str]:
        parts = split_path(path)
        if parts and parts[0] == "shot":
            parts = parts[1:]
        parent, key = -1, ""
        for segment in parts[:-1]:
            key = f"{key}/{segment}"
            if key not in groups:
                groups[key] = scene.add_node(parent, segment or "node")
                scene.set_transform(groups[key], np.array([float(start)]), np.eye(4)[None])
            parent = groups[key]
        return parent, (parts[-1] if parts else "node")

    def hide(nodes: list[int], it: dict) -> None:
        span, shown = shown_span(it)
        if not shown.all():
            # Keys at BOTH ends of every shown / hidden stretch, none inside.  One key per change would be enough for
            # a curve that holds the previous key, but the FBX library writes stepped keys that take the value of
            # the NEXT key (`KeySet` with its default mode), and readers differ in how they treat that; between two
            # keys of the same value every reader agrees.
            ends = needed_samples(shown)
            for node in nodes:
                scene.set_visibility(node, span[ends].astype(np.float64), shown[ends].astype(np.float64))

    for it in items["camera"]:
        f = np.asarray(it["frames"], dtype=np.float64)
        parent, leaf = under(text(it.get("shown", it["path"])))
        node = scene.add_node(parent, leaf or "camera")
        world = np.asarray(it["world"], dtype=np.float64).reshape(-1, 4, 4)
        world[:, :3, :3] = orthonormal(world[:, :3, :3])
        scene.set_transform(node, f, world @ CAMERA_TO_FBX)
        focal = np.asarray(it["focal_mm"], dtype=np.float64).reshape(-1)
        res = np.asarray(it["resolution"]).reshape(-1) if "resolution" in it else np.zeros(2)
        props = json.loads(text(it["properties"])) if "properties" in it else {}
        overscan = np.asarray(it["overscan"], dtype=np.float64).reshape(4)
        if overscan.any():
            props[OVERSCAN_PROP] = json.dumps(overscan.tolist())
        scene.set_camera(
            node,
            f,
            focal if np.ptp(focal) > 0 else focal[:1],
            float(np.median(it["h_aperture_mm"])),
            float(np.median(it["v_aperture_mm"])),
            int(res[0]),
            int(res[1]),
            np.asarray(it["center_mm"], dtype=np.float64).reshape(-1, 2),
            float(it["pixel_aspect"]),
            props,
        )
        hide([node], it)

    for it in items["character"]:
        f = np.asarray(it["frames"], dtype=np.float64)
        holder, leaf = under(text(it.get("shown", it["path"])))
        joints, parents = texts(it["joints"]), np.asarray(it["parents"], dtype=np.int64).reshape(-1)
        anim = np.asarray(it["anim"], dtype=np.float64).reshape(len(f), -1, 4, 4)
        bind = np.asarray(it["bind"], dtype=np.float64)
        root_here = bool(joints) and bool(np.asarray(it.get("root_at_path", False)).reshape(-1)[0])
        group = holder if root_here else scene.add_node(holder, leaf or "character")
        if not root_here:
            scene.set_transform(group, f[:1], np.eye(4)[None])
        # frames on which the body does something: while it stands still (hidden, frozen at the nearest solved
        # pose) no keys are written - between two equal keys the curve does not move
        keyed = needed_samples(anim)
        ids = []
        for j, (name, p) in enumerate(zip(joints, parents)):
            at = (holder if root_here else group) if p < 0 else ids[p]
            ids.append(scene.add_node(at, name or "joint"))
            local = anim[:, j] if p < 0 else np.linalg.inv(anim[:, p]) @ anim[:, j]
            moving = len(f) > 1 and np.abs(local - local[:1]).max() > 1e-9
            scene.set_transform(ids[-1], f[keyed] if moving else f[:1], local[keyed] if moving else local[:1])
            scene.set_skeleton(ids[-1], p < 0, 1.0)
        pose_ids, pose_mats = list(ids), list(bind)
        for mesh in it["meshes"]:
            node = scene.add_node(group, text(mesh["name"]) or "body")
            scene.set_transform(node, f[:1], np.eye(4)[None])
            points = np.asarray(mesh["points"], dtype=np.float64)
            scene.set_mesh(
                node,
                points,
                np.asarray(mesh["counts"], dtype=np.int64),
                np.asarray(mesh["indices"], dtype=np.int64),
                mesh.get("uv"),
                mesh.get("uv_indices"),
                mesh.get("normals"),
            )
            idx, wts = np.asarray(mesh["joint_indices"]), np.asarray(mesh["joint_weights"])
            clusters = []
            for j in range(len(joints)):
                hit = (idx == j) & (wts > 0)
                if hit.any():
                    v, s = np.nonzero(hit)
                    clusters.append((ids[j], v.astype(np.int64), wts[v, s].astype(np.float64), np.eye(4), bind[j]))
            scene.add_skin(node, clusters)
            pose_ids.append(node)
            pose_mats.append(np.eye(4))
        scene.add_bind_pose(pose_ids, np.stack(pose_mats))
        hide(([] if root_here else [group]) + [ids[j] for j, p in enumerate(parents) if p < 0], it)
        for node, mesh in zip(pose_ids[len(ids) :], it["meshes"]):
            own = np.asarray(mesh.get("visible", np.ones(len(f))), dtype=np.int32).reshape(-1)
            hide(
                [node],
                {
                    "frames": it["frames"],
                    "visible": own * np.asarray(it.get("visible", np.ones(len(f))), dtype=np.int32).reshape(-1),
                },
            )

    for it in items["model"]:
        if "points" not in it:
            continue
        f = np.asarray(it["frames"], dtype=np.float64)
        parent, leaf = under(text(it.get("shown", it["path"])))
        xform_node = scene.add_node(parent, leaf or "object")
        world = np.asarray(it["world"], dtype=np.float64).reshape(-1, 4, 4).copy()
        world[:, :3, :3] = orthonormal(world[:, :3, :3])
        moving = len(f) > 1 and float(np.abs(world - world[:1]).max()) > 1e-5
        keyed = needed_samples(world)
        scene.set_transform(xform_node, f[keyed] if moving else f[:1], world[keyed] if moving else world[:1])
        mesh_node = scene.add_node(xform_node, "proxy_mesh")
        scene.set_transform(mesh_node, f[:1], np.eye(4)[None])
        pts = np.asarray(it["points"], dtype=np.float64)
        if pts.ndim == 3:
            pts = pts[0]
        scene.set_mesh(
            mesh_node,
            pts,
            np.asarray(it["counts"], dtype=np.int64),
            np.asarray(it["indices"], dtype=np.int64),
            it.get("uv"),
            it.get("uv_indices"),
            it.get("normals"),
        )
        hide([xform_node, mesh_node], it)

    out_fbx.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fbx.with_name(f".{out_fbx.stem}.{os.getpid()}.tmp.fbx")
    try:
        scene.save(str(tmp))
        tmp.replace(out_fbx)
    finally:
        tmp.unlink(missing_ok=True)
    return {"cameras": len(items["camera"]), "characters": len(items["character"])}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mode = job["mode"]
    if mode == "import":
        res = do_import(Path(job["input"]), Path(job["output"]), job.get("take", ""), cameras_only=bool(job.get("cameras_only", False)))
    elif mode == "export":
        res = do_export(Path(job["input"]), Path(job["output"]))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if "result" in job:
        Path(job["result"]).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
