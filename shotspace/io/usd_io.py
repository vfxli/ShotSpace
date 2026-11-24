"""OpenUSD (`.usd` / `.usda` / `.usdc` / `.usdz`) camera & character reader and scene exporter.

Enforces:
  - Flattened self-contained file safety check (`external_files` rejected outside `.usdz`)
  - Stage unit (`metersPerUnit -> cm`) and up-axis (`Z-up -> Y-up`) normalization + `rigid()` SVD cleanup on cameras
  - Negated `apertureOffset` (`center_mm = -apertureOffset`) matching Maya/Houdini principal point convention
  - `UsdSkel.Root` + `UsdSkel.Skeleton` + `UsdSkel.Animation` + `UsdGeom.Mesh` export with gap `visibility`
  - `unit="cm"` (`metersPerUnit = 0.01`) or `unit="m"` (`rescale_stage(0.01)`, `metersPerUnit = 1.0`, keeping focal &
    apertures in `mm`!)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdSkel, UsdUtils, Vt

from .. import __version__
from ..core import names
from ..core.camera import CameraTrack
from ..core.names import ORIGINAL
from ..core.poses import continuous, decompose, local_from_world, world_from_local
from ..core.rigid_object import RigidObjectTrack
from ..core.skeleton import SkinnedCharacter, bake_lbs_points, cg_bone_names, cg_orientations
from ..core.units import Axes, CM_TO_M, DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, FILMBACK_MM, rigid, standard_fps, to_cm
from .scene_arrays import held_changes, needed_samples

METERS_PER_UNIT = 0.01
ROOT_PATH = "/shot"


def _clip_assets(layer: Sdf.Layer) -> list[str]:
    found: list[str] = []

    def visit(p: Sdf.Path) -> None:
        if not p.IsPrimPath():
            return
        spec = layer.GetPrimAtPath(p)
        if spec is None or not spec.HasInfo("clips"):
            return
        for clip in dict(spec.GetInfo("clips")).values():
            for key in ("assetPaths", "manifestAssetPath", "templateAssetPath"):
                value = clip.get(key)
                seq = (
                    value
                    if isinstance(value, (list, tuple)) or type(value).__name__.endswith("Array")
                    else [value]
                )
                for a in seq:
                    t = getattr(a, "authoredPath", a) if a is not None else ""
                    if t:
                        found.append(str(t))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return found


def external_files(layer: Sdf.Layer, package: str = "", seen: set[str] | None = None) -> list[str]:
    seen = set() if seen is None else seen
    out: list[str] = []
    for asset in [*layer.GetCompositionAssetDependencies(), *_clip_assets(layer)]:
        if not asset:
            continue
        where = layer.ComputeAbsolutePath(asset)
        if package and where.startswith(package + "["):
            if where not in seen and (inner := Sdf.Layer.FindOrOpen(where)) is not None:
                seen.add(where)
                out += external_files(inner, package, seen)
        elif asset not in out:
            out.append(asset)
    return out


def open_usd_stage(path: str | Path) -> Usd.Stage:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"未找到 USD 文件：{p}")
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    try:
        layer = Sdf.Layer.FindOrOpen(str(p))
    except Tf.ErrorException as exc:
        raise ValueError(f"无法读取 USD 文件 {p.name}：{exc}") from exc
    finally:
        del delegate
    if layer is None:
        raise ValueError(f"无法打开 USD 文件 {p.name}")
    outside = external_files(layer, str(p) if p.suffix.lower() == ".usdz" else "")
    if outside:
        shown = ", ".join(outside[:5])
        raise ValueError(
            f"USD 文件 {p.name} 引用了外部未拍平资产（{shown}）。请在 Maya / Houdini 中执行 Flatten 拍平导出后再导入。"
        )
    try:
        return Usd.Stage.Open(layer)
    except Tf.ErrorException as exc:
        raise ValueError(f"无法解析 USD Stage {p.name}：{exc}") from exc


def _stage_axes(stage: Usd.Stage) -> Axes:
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or METERS_PER_UNIT)
    up = "z" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else "y"
    return Axes.of(mpu / METERS_PER_UNIT, up)


def _recorded_fps(stage: Usd.Stage) -> float:
    layer = stage.GetRootLayer()
    if layer.HasTimeCodesPerSecond():
        return standard_fps(float(layer.timeCodesPerSecond))
    if layer.HasFramesPerSecond():
        return standard_fps(float(layer.framesPerSecond))
    return DEFAULT_FPS


def _xform_times(prim: Usd.Prim) -> set[float]:
    times: set[float] = set()
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        if p.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                times.update(op.GetTimeSamples())
        p = p.GetParent()
    return times


def _attr_times(*attrs) -> set[float]:
    return {t for a in attrs if a for t in a.GetTimeSamples()}


def _span_frames(stage: Usd.Stage, times: set[float]) -> tuple[int, ...]:
    if not times:
        start = stage.GetStartTimeCode()
        return (int(round(start)) if math.isfinite(start) else 1,)
    first, last = math.ceil(min(times) - 1e-6), math.floor(max(times) + 1e-6)
    return tuple(range(first, last + 1)) if last >= first else (int(round(min(times))),)


def _camera_resolution(stage: Usd.Stage, prim: Usd.Prim) -> tuple[int, int] | None:
    for key in ("shotspace:resolution", "lab2shot:resolution"):
        res = prim.GetCustomDataByKey(key)
        if res is not None and len(res) >= 2:
            return int(res[0]), int(res[1])
    for p in stage.Traverse():
        attr = p.GetAttribute("resolution")
        if p.GetTypeName() == "RenderSettings" and attr and attr.Get() is not None:
            r = attr.Get()
            return int(r[0]), int(r[1])
    return None


def list_usd_contents(path: str | Path) -> dict:
    stage = open_usd_stage(path)
    fps = _recorded_fps(stage)
    axes = _stage_axes(stage)
    cameras = []
    characters = []
    models = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        p_str = str(prim.GetPath())
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            times = _xform_times(prim) | _attr_times(
                cam.GetFocalLengthAttr(),
                cam.GetHorizontalApertureAttr(),
                cam.GetVerticalApertureAttr(),
                cam.GetHorizontalApertureOffsetAttr(),
                cam.GetVerticalApertureOffsetAttr(),
            )
            frames = _span_frames(stage, times)
            t0 = Usd.TimeCode(frames[0])
            focal0 = float(cam.GetFocalLengthAttr().Get(t0) or 50.0)
            hap0 = float(cam.GetHorizontalApertureAttr().Get(t0) or FILMBACK_MM)
            res = _camera_resolution(stage, prim)
            xf_cam = UsdGeom.Xformable(prim)
            sample_f = frames if len(frames) <= 64 else frames[:: max(1, len(frames) // 64)]
            if len(sample_f) > 1:
                pts_c = np.array(
                    [np.array(xf_cam.ComputeLocalToWorldTransform(Usd.TimeCode(f)), dtype=np.float64)[3, :3] for f in sample_f],
                    dtype=np.float64,
                )
                travel_raw = float(np.linalg.norm(np.diff(pts_c, axis=0), axis=1).sum())
            else:
                travel_raw = 0.0
            cameras.append(
                {
                    "name": str(prim.GetCustomDataByKey(ORIGINAL) or prim.GetName()),
                    "path": p_str,
                    "frames": [frames[0], frames[-1]],
                    "frame_count": len(frames),
                    "focal_mm": round(focal0, 3),
                    "h_aperture_mm": round(hap0, 3),
                    "travel_raw": round(travel_raw, 4),
                    "resolution": list(res) if res else None,
                }
            )
        elif prim.IsA(UsdSkel.Root):
            skels = [UsdSkel.Skeleton(p) for p in Usd.PrimRange(prim) if p.IsA(UsdSkel.Skeleton)]
            if skels:
                j_count = sum(len(list(s.GetJointsAttr().Get() or [])) for s in skels)
                characters.append({"name": prim.GetName(), "path": p_str, "joints": j_count})
        elif prim.IsA(UsdGeom.Mesh):
            anc = prim.GetParent()
            in_skel = False
            cur = anc
            while cur and cur.IsValid() and not cur.IsPseudoRoot():
                if cur.IsA(UsdSkel.Root):
                    in_skel = True
                    break
                cur = cur.GetParent()
            if not in_skel:
                xf_prim = (
                    anc
                    if (anc and anc.IsValid() and anc.IsA(UsdGeom.Xform) and len(anc.GetChildren()) == 1)
                    else prim
                )
                times = _xform_times(prim) | _attr_times(
                    UsdGeom.Imageable(xf_prim).GetVisibilityAttr(),
                    UsdGeom.Imageable(prim).GetVisibilityAttr(),
                )
                frames = list(_span_frames(stage, times))
                img_xf = UsdGeom.Imageable(xf_prim)
                vis = [0 if img_xf.ComputeVisibility(Usd.TimeCode(f)) == UsdGeom.Tokens.invisible else 1 for f in frames]
                xf_api = UsdGeom.Xformable(xf_prim)
                w_mats = [axes.matrices(np.array(xf_api.ComputeLocalToWorldTransform(Usd.TimeCode(f)), dtype=np.float64).T).tolist() for f in frames]
                mesh_api = UsdGeom.Mesh(prim)
                t_m = Usd.TimeCode(frames[0]) if frames else Usd.TimeCode.Default()
                pts = mesh_api.GetPointsAttr().Get(t_m)
                if pts is None:
                    pts = mesh_api.GetPointsAttr().Get(Usd.TimeCode.Default())
                cnts = mesh_api.GetFaceVertexCountsAttr().Get(t_m)
                if cnts is None:
                    cnts = mesh_api.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
                cnts_arr = np.asarray(cnts, dtype=np.int64).reshape(-1) if cnts is not None else np.zeros(0, dtype=np.int64)
                f_cnt = int(np.maximum(0, cnts_arr - 2).sum()) if len(cnts_arr) else 0
                models.append(
                    {
                        "name": str(xf_prim.GetCustomDataByKey(ORIGINAL) or xf_prim.GetName()),
                        "path": str(xf_prim.GetPath()),
                        "frames": frames,
                        "visible": vis,
                        "vertex_count": len(pts) if pts is not None else 0,
                        "face_count": f_cnt,
                        "world_transforms": w_mats,
                    }
                )
    return {
        "format": "usd",
        "fps": fps,
        "unit_to_cm": axes.to_cm,
        "cameras": cameras,
        "characters": characters,
        "models": models,
        "takes": [],
        "auto_take": "",
    }


def read_usd_camera(
    path: str | Path,
    camera_path: str = "",
    unit: str = "auto",
    default_width: int = DEFAULT_WIDTH,
    default_height: int = DEFAULT_HEIGHT,
) -> CameraTrack:
    p = Path(path).expanduser().resolve()
    stage = open_usd_stage(p)
    fps = _recorded_fps(stage)
    stage_axes = _stage_axes(stage)
    up = "z" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else "y"
    u_norm = (unit or "auto").strip().lower()
    unit_scale = stage_axes.to_cm if u_norm == "auto" else to_cm(u_norm)
    axes = Axes.of(unit_scale, up)

    prim = stage.GetPrimAtPath(camera_path) if camera_path else None
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        cam_prims = [pr for pr in stage.Traverse(Usd.TraverseInstanceProxies()) if pr.IsA(UsdGeom.Camera)]
        if not cam_prims:
            raise ValueError(f"USD 文件 {p.name} 中未找到任何 UsdGeom.Camera 摄影机节点")
        if camera_path:
            raise ValueError(f"USD 文件 {p.name} 中不存在路径为 {camera_path} 的摄影机")
        if len(cam_prims) > 1:
            from . import pick_default_camera_path

            cands = []
            for pr in cam_prims:
                c_obj = UsdGeom.Camera(pr)
                ts = _xform_times(pr) | _attr_times(c_obj.GetFocalLengthAttr())
                frs = _span_frames(stage, ts)
                xf_c = UsdGeom.Xformable(pr)
                s_f = frs if len(frs) <= 64 else frs[:: max(1, len(frs) // 64)]
                if len(s_f) > 1:
                    pts_c = np.array(
                        [np.array(xf_c.ComputeLocalToWorldTransform(Usd.TimeCode(f)), dtype=np.float64)[3, :3] for f in s_f],
                        dtype=np.float64,
                    )
                    tr_raw = float(np.linalg.norm(np.diff(pts_c, axis=0), axis=1).sum())
                else:
                    tr_raw = 0.0
                cands.append(
                    {
                        "name": str(pr.GetCustomDataByKey(ORIGINAL) or pr.GetName()),
                        "path": str(pr.GetPath()),
                        "frame_count": len(frs),
                        "travel_raw": tr_raw,
                    }
                )
            best_p = pick_default_camera_path(cands)
            prim = stage.GetPrimAtPath(best_p) if best_p else cam_prims[0]
        else:
            prim = cam_prims[0]

    cam = UsdGeom.Camera(prim)
    times = _xform_times(prim) | _attr_times(
        cam.GetFocalLengthAttr(),
        cam.GetHorizontalApertureAttr(),
        cam.GetVerticalApertureAttr(),
        cam.GetHorizontalApertureOffsetAttr(),
        cam.GetVerticalApertureOffsetAttr(),
    )
    frames = _span_frames(stage, times)
    xf = UsdGeom.Xformable(prim)

    c2w_raw = []
    focals, haps, vaps, centers = [], [], [], []
    h_off_attr = cam.GetHorizontalApertureOffsetAttr()
    v_off_attr = cam.GetVerticalApertureOffsetAttr()

    for f in frames:
        tc = Usd.TimeCode(f)
        # USD matrices are row-vector; transpose to column-vector
        m_usd = np.array(xf.ComputeLocalToWorldTransform(tc), dtype=np.float64).T
        c2w_raw.append(m_usd)
        focals.append(float(cam.GetFocalLengthAttr().Get(tc) or 50.0))
        hap = float(cam.GetHorizontalApertureAttr().Get(tc) or FILMBACK_MM)
        haps.append(hap)
        if cam.GetVerticalApertureAttr().HasAuthoredValue():
            vaps.append(float(cam.GetVerticalApertureAttr().Get(tc)))
        else:
            vaps.append(hap * default_height / max(default_width, 1))
        # Negate aperture offsets -> lens principal point center_mm (+x right, +y up)
        hoff = float(h_off_attr.Get(tc) or 0.0)
        voff = float(v_off_attr.Get(tc) or 0.0)
        centers.append([-hoff, -voff])

    c2w = rigid(axes.matrices(np.stack(c2w_raw, axis=0)))
    res = _camera_resolution(stage, prim) or (default_width, default_height)

    pa_attr = prim.GetAttribute("pixelAspect")
    pixel_aspect = float(pa_attr.Get() or 1.0) if pa_attr and pa_attr.HasAuthoredValue() else 1.0
    ov_attr = prim.GetAttribute("overscan")
    overscan = tuple(int(v) for v in ov_attr.Get()) if ov_attr and ov_attr.HasAuthoredValue() else (0, 0, 0, 0)

    return CameraTrack(
        name=str(prim.GetCustomDataByKey(ORIGINAL) or prim.GetName()),
        path=str(prim.GetPath()),
        frames=frames,
        cam_to_world=c2w,
        focal_mm=np.array(focals, dtype=np.float64),
        h_aperture_mm=np.array(haps, dtype=np.float64),
        v_aperture_mm=np.array(vaps, dtype=np.float64),
        center_mm=np.array(centers, dtype=np.float64),
        width=int(res[0]),
        height=int(res[1]),
        pixel_aspect=pixel_aspect,
        overscan=overscan,
        fps=fps,
        source_file=str(p),
        own_frames=frames,
    )


def read_usd_characters(path: str | Path) -> list[SkinnedCharacter]:
    """Read `UsdSkel.Root` characters from a USD file into `SkinnedCharacter`s (cm, +Y up) for standalone re-alignment."""
    p = Path(path).expanduser().resolve()
    stage = open_usd_stage(p)
    axes = _stage_axes(stage)
    cache = UsdSkel.Cache()
    out: list[SkinnedCharacter] = []

    for idx, root_prim in enumerate(pr for pr in stage.Traverse() if pr.IsA(UsdSkel.Root)):
        skel_root = UsdSkel.Root(root_prim)
        cache.Populate(skel_root, Usd.PrimDefaultPredicate)
        bindings = cache.ComputeSkelBindings(skel_root, Usd.PrimDefaultPredicate)
        if not bindings:
            continue
        binding = bindings[0]
        skel = binding.GetSkeleton()
        skel_query = cache.GetSkelQuery(skel)
        if not skel_query:
            continue
        topo = skel_query.GetTopology()
        n_joints = topo.GetNumJoints()
        parents = np.array([topo.GetParent(j) for j in range(n_joints)], dtype=np.int64)
        joints_attr = list(skel.GetJointsAttr().Get() or [])
        given_names = list(skel.GetJointNamesAttr().Get() or [])
        j_names = (
            [str(n) for n in given_names]
            if len(given_names) == n_joints
            else [str(j).rsplit("/", 1)[-1] for j in joints_attr]
        )

        bind_vt = skel.GetBindTransformsAttr().Get()
        bind_local_skel = np.transpose(np.asarray(bind_vt, dtype=np.float64), (0, 2, 1))

        # Collect animation frames
        anim_src = UsdSkel.BindingAPI(skel.GetPrim()).GetAnimationSourceRel().GetTargets()
        anim_prim = stage.GetPrimAtPath(anim_src[0]) if anim_src else None
        times = _xform_times(root_prim) | _attr_times(UsdGeom.Imageable(root_prim).GetVisibilityAttr())
        if anim_prim:
            anim = UsdSkel.Animation(anim_prim)
            times |= _attr_times(anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr())
        all_frames = list(_span_frames(stage, times))

        # Read per-frame visibility across all_frames
        img = UsdGeom.Imageable(root_prim)
        vis_arr = np.array(
            [
                0 if img.ComputeVisibility(Usd.TimeCode(f)) == UsdGeom.Tokens.invisible else 1
                for f in all_frames
            ],
            dtype=np.int32,
        )
        vis_frames = all_frames

        root_xf = UsdGeom.Xformable(root_prim)
        anim_world_list = []
        for f in vis_frames:
            tc = Usd.TimeCode(f)
            rw = np.array(root_xf.ComputeLocalToWorldTransform(tc), dtype=np.float64).T
            jt_vt = skel_query.ComputeJointSkelTransforms(tc)
            jt_skel = np.transpose(np.asarray(jt_vt, dtype=np.float64), (0, 2, 1))
            anim_world_list.append(axes.matrices(rw[None] @ jt_skel))

        rw0 = np.array(root_xf.ComputeLocalToWorldTransform(Usd.TimeCode(vis_frames[0])), dtype=np.float64).T
        bind_world = axes.matrices(rw0[None] @ bind_local_skel)
        anim_world = np.stack(anim_world_list, axis=0)

        targets = binding.GetSkinningTargets()
        if not targets:
            continue
        m_prim = targets[0].GetPrim()
        mesh = UsdGeom.Mesh(m_prim)
        pts_raw = np.asarray(mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or mesh.GetPointsAttr().Get(Usd.TimeCode(vis_frames[0])), dtype=np.float64)
        gbt_attr = UsdSkel.BindingAPI(m_prim).GetGeomBindTransformAttr()
        gbt = np.array(gbt_attr.Get() if gbt_attr and gbt_attr.HasAuthoredValue() else Gf.Matrix4d(1.0), dtype=np.float64).T
        pts_world = axes.world_points((pts_raw @ gbt[:3, :3].T) + gbt[:3, 3]).astype(np.float32)

        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        f_idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
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

        b_api = UsdSkel.BindingAPI(m_prim)
        j_idx_pv = b_api.GetJointIndicesPrimvar()
        j_wt_pv = b_api.GetJointWeightsPrimvar()
        elem_size = j_idx_pv.GetElementSize()
        j_indices = np.asarray(j_idx_pv.Get(), dtype=np.int32).reshape(len(pts_world), elem_size)
        j_weights = np.asarray(j_wt_pv.Get(), dtype=np.float32).reshape(len(pts_world), elem_size)

        uv, uv_faces = None, None
        st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        if st and st.HasAuthoredValue():
            uv = np.asarray(st.Get(), dtype=np.float32)
            if st.IsIndexed() and np.all(counts == 3):
                uv_faces = np.asarray(st.GetIndices(), dtype=np.int32).reshape(-1, 3)

        pid = root_prim.GetCustomDataByKey("lab2shot:person_id") or root_prim.GetCustomDataByKey("shotspace:person_id") or (idx + 1)
        std_names = cg_bone_names(j_names, parents)
        bind_std, anim_std = cg_orientations(parents, bind_world, anim_world)
        out.append(
            SkinnedCharacter(
                name=str(root_prim.GetCustomDataByKey(ORIGINAL) or root_prim.GetName()),
                person_id=int(pid),
                frames=vis_frames,
                joint_names=std_names,
                parents=parents,
                bind_world=bind_std,
                anim_world=anim_std,
                rest_points=pts_world,
                faces=faces,
                joint_indices=j_indices,
                joint_weights=j_weights,
                uv=uv,
                uv_faces=uv_faces,
                visible=vis_arr,
                custom_data={"lab2shot:person_id": int(pid)},
            )
        )
    return out


# --------------------------------------------------------------------------- Export


def _usd_matrices(m: np.ndarray) -> Vt.Matrix4dArray:
    m = np.asarray(m, dtype=np.float64).reshape(-1, 4, 4)
    return Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(np.transpose(m, (0, 2, 1))))


def _extent(p: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(*map(float, p.min(0))), Gf.Vec3f(*map(float, p.max(0)))])


def _set_values(attr: Usd.Attribute, values, frames: list[int]) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) == 1 or np.allclose(arr, arr[0]):
        attr.Set(float(arr[0]))
    else:
        for f, v in zip(frames, arr):
            attr.Set(float(v), Usd.TimeCode(int(f)))


def _joint_paths(joint_names: list[str], parents: np.ndarray) -> list[str]:
    paths: list[str] = []
    taken: set[str] = set()
    for i, (name, parent) in enumerate(zip(joint_names, parents)):
        if int(parent) >= i:
            raise ValueError("Joint parents must precede children")
        raw = names.identifier(name) if int(parent) < 0 else f"{paths[int(parent)]}/{names.identifier(name)}"
        p = names.unique(raw, taken)
        taken.add(p)
        paths.append(p)
    return paths


def rescale_usd_stage(stage: Usd.Stage, factor: float) -> None:
    """Multiply all scene lengths in `stage` by `factor` (e.g. `0.01` for `cm -> m`) while leaving
    `focalLength` and `Aperture` in `mm` per DCC standard."""

    def scaled_matrix(m) -> Gf.Matrix4d:
        m = Gf.Matrix4d(m)
        m.SetTranslateOnly(m.ExtractTranslation() * factor)
        return m

    def scaled_matrices(values):
        return Vt.Matrix4dArray([scaled_matrix(m) for m in values])

    def each_sample(attr, fn):
        if not attr or not attr.HasAuthoredValue():
            return
        times = attr.GetTimeSamples()
        if times:
            for t in times:
                attr.Set(fn(attr.Get(t)), Usd.TimeCode(t))
        else:
            attr.Set(fn(attr.Get()))

    vec_scale = lambda v: type(v)([x * factor for x in v])  # noqa: E731

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointBased):
            geom = UsdGeom.PointBased(prim)
            each_sample(geom.GetPointsAttr(), vec_scale)
            each_sample(geom.GetExtentAttr(), vec_scale)
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            each_sample(skel.GetBindTransformsAttr(), scaled_matrices)
            each_sample(skel.GetRestTransformsAttr(), scaled_matrices)
        if prim.IsA(UsdSkel.Animation):
            each_sample(UsdSkel.Animation(prim).GetTranslationsAttr(), vec_scale)
        each_sample(prim.GetAttribute("primvars:skel:geomBindTransform"), scaled_matrix)
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            each_sample(cam.GetClippingRangeAttr(), lambda r: Gf.Vec2f(r[0] * factor, r[1] * factor))
            each_sample(cam.GetFocusDistanceAttr(), lambda d: d * factor)
        if prim.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    each_sample(op.GetAttr(), lambda v: v * factor)
                elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    each_sample(op.GetAttr(), scaled_matrix)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT / factor)


def export_usd_scene(
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
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # the one root prim of the file (its default prim): `/shot`, or the top group the product's deliveries name
    root_path = f"/{names.identifier(top_group)}" if top_group else ROOT_PATH
    fps_std = standard_fps(fps)
    mode_norm = (usd_mode or "skel").strip().lower()
    is_point_cache = mode_norm in ("cache", "point_cache", "points", "mesh")

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetTimeCodesPerSecond(fps_std)
    stage.SetFramesPerSecond(fps_std)
    if shot_frames:
        stage.SetStartTimeCode(min(shot_frames))
        stage.SetEndTimeCode(max(shot_frames))
    root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root_prim)
    stage.GetRootLayer().customLayerData = {
        "shotspace": {
            "version": __version__,
            "unit": unit,
            "fps": float(fps_std),
            "usd_mode": "cache" if is_point_cache else "skel",
        }
    }

    taken_Siblings: set[str] = set()

    # Write cameras
    for cam in cameras:
        leaf = names.unique(names.identifier(cam.name or "camera"), taken_Siblings)
        taken_Siblings.add(leaf)
        c_path = f"{root_path}/{leaf}"
        usd_cam = UsdGeom.Camera.Define(stage, c_path)
        if leaf != cam.name:
            usd_cam.GetPrim().SetCustomDataByKey(ORIGINAL, cam.name)
        usd_cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        _set_values(usd_cam.CreateHorizontalApertureAttr(), cam.h_aperture_mm, list(cam.frames))
        _set_values(usd_cam.CreateVerticalApertureAttr(), cam.v_aperture_mm, list(cam.frames))
        usd_cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, 1.0e6))
        _set_values(usd_cam.CreateFocalLengthAttr(), cam.focal_mm, list(cam.frames))
        if np.any(cam.center_mm):
            _set_values(usd_cam.CreateHorizontalApertureOffsetAttr(), -cam.center_mm[:, 0], list(cam.frames))
            _set_values(usd_cam.CreateVerticalApertureOffsetAttr(), -cam.center_mm[:, 1], list(cam.frames))
        if abs(cam.pixel_aspect - 1.0) > 1e-6:
            usd_cam.GetPrim().CreateAttribute("pixelAspect", Sdf.ValueTypeNames.Double, custom=True).Set(
                float(cam.pixel_aspect)
            )
        if any(cam.overscan):
            usd_cam.GetPrim().CreateAttribute("overscan", Sdf.ValueTypeNames.Int4, custom=True).Set(
                Gf.Vec4i(*[int(v) for v in cam.overscan])
            )
        xf_op = usd_cam.AddTransformOp()
        mats = np.asarray(cam.cam_to_world, dtype=np.float64)
        if len(mats) == 1:
            xf_op.Set(Gf.Matrix4d(mats[0].T.tolist()))
        else:
            for f, m in zip(cam.frames, mats):
                xf_op.Set(Gf.Matrix4d(m.T.tolist()), Usd.TimeCode(int(f)))
        usd_cam.GetPrim().SetCustomDataByKey("shotspace:resolution", Gf.Vec2i(int(cam.width), int(cam.height)))
        usd_cam.GetPrim().SetCustomDataByKey("lab2shot:resolution", Gf.Vec2i(int(cam.width), int(cam.height)))

    # Write characters: either rigged `UsdSkel` ("skel") or baked per-frame vertex `UsdGeom.Mesh` ("cache")
    for ch in characters:
        if not ch.frames:
            continue
        leaf = names.unique(names.identifier(ch.name or f"person_{ch.person_id:02d}"), taken_Siblings)
        taken_Siblings.add(leaf)
        r_path = f"{root_path}/{leaf}"

        ch_vis = ch.visible_mask
        f_to_idx = {int(f): idx for idx, f in enumerate(ch.frames)}
        all_out_frames = sorted(set(int(f) for f in (shot_frames or ch.frames)) | set(int(f) for f in ch.frames))
        anim_expanded = []
        vis_expanded = []
        for f in all_out_frames:
            if f in f_to_idx:
                idx_f = f_to_idx[f]
                anim_expanded.append(ch.anim_world[idx_f])
                vis_expanded.append(bool(ch_vis[idx_f]))
            elif f < ch.frames[0]:
                anim_expanded.append(ch.anim_world[0])
                vis_expanded.append(False)
            else:
                nearest = min(range(len(ch.frames)), key=lambda k: abs(ch.frames[k] - f))
                anim_expanded.append(ch.anim_world[nearest])
                vis_expanded.append(False)
        anim_world_out = np.stack(anim_expanded, axis=0)
        # frames on which the body does something: stretches where it stands still (hidden, frozen) get no samples
        keyed = needed_samples(anim_world_out)
        vis_keyed = held_changes(vis_expanded)

        if is_point_cache:
            char_xform = UsdGeom.Xform.Define(stage, r_path)
            if leaf != ch.name:
                char_xform.GetPrim().SetCustomDataByKey(ORIGINAL, ch.name)
            for k, v in (ch.custom_data or {}).items():
                char_xform.GetPrim().SetCustomDataByKey(str(k), v)
            char_xform.GetPrim().SetCustomDataByKey("shotspace:person_id", int(ch.person_id))

            mesh = UsdGeom.Mesh.Define(stage, f"{r_path}/body")
            faces = np.asarray(ch.faces, dtype=np.int32)
            counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1))))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            if ch.uv is not None and ch.uv_faces is not None:
                st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(ch.uv, dtype=np.float32)))
                st.SetIndices(Vt.IntArray.FromNumpy(np.asarray(ch.uv_faces, dtype=np.int32).reshape(-1)))

            baked_pts = bake_lbs_points(
                ch.rest_points, ch.bind_world, anim_world_out, ch.joint_indices, ch.joint_weights
            )
            pts_attr = mesh.CreatePointsAttr()
            ext_attr = mesh.CreateExtentAttr()
            for i, f in enumerate(all_out_frames):
                if not keyed[i]:
                    continue
                tc = Usd.TimeCode(int(f))
                p_frame = np.ascontiguousarray(baked_pts[i], dtype=np.float32)
                pts_attr.Set(Vt.Vec3fArray.FromNumpy(p_frame), tc)
                ext_attr.Set(_extent(p_frame), tc)

            if not all(vis_expanded):
                vis_root = UsdGeom.Imageable(char_xform.GetPrim()).CreateVisibilityAttr()
                vis_mesh = UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr()
                for i, (f, is_vis) in enumerate(zip(all_out_frames, vis_expanded)):
                    if not vis_keyed[i]:
                        continue  # visibility holds from one sample to the next
                    tok = "inherited" if is_vis else "invisible"
                    tc = Usd.TimeCode(int(f))
                    vis_root.Set(tok, tc)
                    vis_mesh.Set(tok, tc)
            continue

        skel_root = UsdSkel.Root.Define(stage, r_path)
        if leaf != ch.name:
            skel_root.GetPrim().SetCustomDataByKey(ORIGINAL, ch.name)
        for k, v in (ch.custom_data or {}).items():
            skel_root.GetPrim().SetCustomDataByKey(str(k), v)
        skel_root.GetPrim().SetCustomDataByKey("shotspace:person_id", int(ch.person_id))

        parents = np.asarray(ch.parents, dtype=np.int64)
        j_paths = Vt.TokenArray(_joint_paths(ch.joint_names, parents))
        skel = UsdSkel.Skeleton.Define(stage, f"{r_path}/skeleton")
        skel.CreateJointsAttr(j_paths)
        skel.CreateJointNamesAttr(Vt.TokenArray(names.sibling_unique(list(ch.joint_names), parents)))
        skel.CreateBindTransformsAttr(_usd_matrices(ch.bind_world))
        skel.CreateRestTransformsAttr(_usd_matrices(local_from_world(ch.bind_world, parents)))

        anim = UsdSkel.Animation.Define(stage, f"{r_path}/anim")
        anim.CreateJointsAttr(j_paths)
        local_anim = local_from_world(anim_world_out, parents)
        n_j = local_anim.shape[1]
        t, q, s = decompose(local_anim.reshape(-1, 4, 4))
        t = t.reshape(-1, n_j, 3)
        q = continuous(q.reshape(-1, n_j, 4))
        s = s.reshape(-1, n_j, 3)
        t_attr, r_attr, s_attr = anim.CreateTranslationsAttr(), anim.CreateRotationsAttr(), anim.CreateScalesAttr()
        for i, f in enumerate(all_out_frames):
            if not keyed[i]:
                continue
            tc = Usd.TimeCode(int(f))
            t_attr.Set(Vt.Vec3fArray.FromNumpy(t[i].astype(np.float32)), tc)
            r_attr.Set(Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for w, x, y, z in q[i]]), tc)
            s_attr.Set(Vt.Vec3hArray.FromNumpy(s[i].astype(np.float16)), tc)
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])

        mesh = UsdGeom.Mesh.Define(stage, f"{r_path}/body")
        faces = np.asarray(ch.faces, dtype=np.int32)
        counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1))))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        pts = np.ascontiguousarray(ch.rest_points, dtype=np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateExtentAttr(_extent(pts))
        if ch.uv is not None and ch.uv_faces is not None:
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(ch.uv, dtype=np.float32)))
            st.SetIndices(Vt.IntArray.FromNumpy(np.asarray(ch.uv_faces, dtype=np.int32).reshape(-1)))

        b_api = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
        b_api.CreateSkeletonRel().SetTargets([skel.GetPath()])
        k_inf = ch.joint_indices.shape[1]
        b_api.CreateJointIndicesPrimvar(False, k_inf).Set(
            Vt.IntArray.FromNumpy(np.asarray(ch.joint_indices, dtype=np.int32).reshape(-1))
        )
        b_api.CreateJointWeightsPrimvar(False, k_inf).Set(
            Vt.FloatArray.FromNumpy(np.asarray(ch.joint_weights, dtype=np.float32).reshape(-1))
        )
        b_api.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))

        # Visibility on BOTH SkelRoot and body Mesh so undetected / frozen frames auto-hide in all DCCs; it holds
        # from one sample to the next, so only the frames where it changes are written
        if not all(vis_expanded):
            vis_root = UsdGeom.Imageable(skel_root.GetPrim()).CreateVisibilityAttr()
            vis_mesh = UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr()
            for i, (f, is_vis) in enumerate(zip(all_out_frames, vis_expanded)):
                if not vis_keyed[i]:
                    continue
                tok = "inherited" if is_vis else "invisible"
                tc = Usd.TimeCode(int(f))
                vis_root.Set(tok, tc)
                vis_mesh.Set(tok, tc)

    # Write rigid 3D objects (animated parent Xform + static child Mesh centered at (0,0,0) for 1-click DCC asset swap!)
    for obj in rigid_objects or []:
        if not obj.frames:
            continue
        leaf = names.unique(names.identifier(obj.name or f"object_{obj.object_id:02d}"), taken_Siblings)
        taken_Siblings.add(leaf)
        o_path = f"{root_path}/{leaf}"
        obj_xform = UsdGeom.Xform.Define(stage, o_path)
        if leaf != obj.name:
            obj_xform.GetPrim().SetCustomDataByKey(ORIGINAL, obj.name)
        obj_xform.GetPrim().SetCustomDataByKey("shotspace:object_id", int(obj.object_id))

        obj_vis = obj.visible_mask
        f_to_idx = {int(f): idx for idx, f in enumerate(obj.frames)}
        all_out_frames = sorted(set(int(f) for f in (shot_frames or obj.frames)) | set(int(f) for f in obj.frames))
        mats_expanded = []
        vis_expanded = []
        for f in all_out_frames:
            if f in f_to_idx:
                idx_f = f_to_idx[f]
                mats_expanded.append(obj.world_transforms[idx_f])
                vis_expanded.append(bool(obj_vis[idx_f]))
            elif f < obj.frames[0]:
                mats_expanded.append(obj.world_transforms[0])
                vis_expanded.append(False)
            else:
                nearest = min(range(len(obj.frames)), key=lambda k: abs(obj.frames[k] - f))
                mats_expanded.append(obj.world_transforms[nearest])
                vis_expanded.append(False)
        mats_out = np.stack(mats_expanded, axis=0)
        keyed = needed_samples(mats_out)  # no samples while the object stands still (hidden, frozen)
        vis_keyed = held_changes(vis_expanded)

        t_vec, q_vec, _ = decompose(mats_out)
        q_vec = continuous(q_vec)
        t_moving = len(t_vec) > 1 and float(np.abs(t_vec - t_vec[:1]).max()) > 1e-5
        r_moving = len(q_vec) > 1 and float(np.abs(q_vec - q_vec[:1]).max()) > 1e-5
        t_op = obj_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        r_op = obj_xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        if not t_moving:
            tx, ty, tz = map(float, t_vec[0])
            t_op.Set(Gf.Vec3d(tx, ty, tz))
        else:
            for i, f in enumerate(all_out_frames):
                if not keyed[i]:
                    continue
                tx, ty, tz = map(float, t_vec[i])
                t_op.Set(Gf.Vec3d(tx, ty, tz), Usd.TimeCode(int(f)))
        if not r_moving:
            qw, qx, qy, qz = map(float, q_vec[0])
            r_op.Set(Gf.Quatf(qw, qx, qy, qz))
        else:
            for i, f in enumerate(all_out_frames):
                if not keyed[i]:
                    continue
                qw, qx, qy, qz = map(float, q_vec[i])
                r_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(int(f)))

        omesh = UsdGeom.Mesh.Define(stage, f"{o_path}/proxy_mesh")
        faces = np.asarray(obj.faces, dtype=np.int32)
        counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
        omesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        omesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1))))
        omesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        omesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        pts = np.ascontiguousarray(obj.rest_points, dtype=np.float32)
        omesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        omesh.CreateExtentAttr(_extent(pts))
        if obj.vertex_colors is not None and len(obj.vertex_colors) == len(pts):
            dc = UsdGeom.PrimvarsAPI(omesh).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )
            dc.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(obj.vertex_colors[:, :3], dtype=np.float32)))

        if not all(vis_expanded):
            vis_xf = UsdGeom.Imageable(obj_xform.GetPrim()).CreateVisibilityAttr()
            vis_m = UsdGeom.Imageable(omesh.GetPrim()).CreateVisibilityAttr()
            for i, (f, is_vis) in enumerate(zip(all_out_frames, vis_expanded)):
                if not vis_keyed[i]:
                    continue
                tok = "inherited" if is_vis else "invisible"
                tc = Usd.TimeCode(int(f))
                vis_xf.Set(tok, tc)
                vis_m.Set(tok, tc)

    if unit.strip().lower() == "m":
        rescale_usd_stage(stage, CM_TO_M)

    export_args = {"format": "usda"} if out.suffix.lower() == ".usda" else {"format": "usdc"}
    if not stage.GetRootLayer().Export(str(out), "", export_args):
        raise RuntimeError(f"导出 USD 文件失败：{out}")
    return out
