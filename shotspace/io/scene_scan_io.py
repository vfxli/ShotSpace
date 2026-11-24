"""Unified Static Scene Scan Mesh Reader (`.abc`, `.fbx`, `.usd`/`.usda`/`.usdc`/`.usdz`).

Supports studio matchmove workflows where the matchmoved camera and the pixel-registered static
scene scan (LiDAR / photogrammetry unified ground + static set mesh) are delivered as two separate
files in any combination of `.abc`, `.fbx`, or `.usd`/`.usda`/`.usdc`/`.usdz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.names import ORIGINAL
from ..core.units import Axes, to_cm

USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
SCAN_EXTS = {".abc", ".fbx", ".usd", ".usda", ".usdc", ".usdz"}


def _norm_scene_path(raw: str) -> str:
    segs = [s for s in str(raw or "").split("/") if s]
    return "/" + "/".join(segs) if segs else "/"


def path_matches_selection(mesh_path: str, selected_paths: list[str] | None) -> bool:
    """Return True when `selected_paths` is empty/None, or when `mesh_path` equals or is a descendant of any selected path."""
    if not selected_paths:
        return True
    mp = _norm_scene_path(mesh_path)
    for raw_s in selected_paths:
        sp = _norm_scene_path(raw_s)
        if sp == "/" or mp == sp or mp.startswith(sp + "/"):
            return True
    return False


@dataclass
class SceneScanData:
    name: str
    source_file: str
    format: str  # "abc" | "fbx" | "usd"
    unit: str  # "auto" | "cm" | "m"
    resolved_unit_to_cm: float
    mesh_count: int
    raw_vertex_count: int
    raw_face_count: int
    vertices_cm: np.ndarray  # [V, 3] float32 world-space Y-up cm (calibration resolution <= 150k tris)
    faces: np.ndarray  # [F, 3] int32 triangle indices
    display_vertices_cm: np.ndarray  # [Vd, 3] float32 lightweight viewport display mesh (<= 30k tris)
    display_faces: np.ndarray  # [Fd, 3] int32
    bbox_min_cm: list[float]
    bbox_max_cm: list[float]
    selected_paths: list[str] = field(default_factory=list)
    available_meshes: list[dict] = field(default_factory=list)
    hierarchy: list[dict] = field(default_factory=list)
    # Full-resolution mesh for the people solver's ground contact (None when the scan is small enough that
    # `vertices_cm` / `faces` already are the full mesh).  Reducing a scan to ~120k triangles moves the ground by
    # about half a centimetre on average and by several centimetres at kerbs and steps (measured), and one
    # centimetre of ground height is about 0.6 % of a person's distance to the camera.
    solver_vertices_cm: np.ndarray | None = None
    solver_faces: np.ndarray | None = None

    def summary(self) -> dict:
        span_cm = [round(b - a, 1) for a, b in zip(self.bbox_min_cm, self.bbox_max_cm)]
        return {
            "name": self.name,
            "source_file": self.source_file,
            "format": self.format,
            "unit": self.unit,
            "resolved_unit_to_cm": round(float(self.resolved_unit_to_cm), 4),
            "mesh_count": int(self.mesh_count),
            "total_mesh_count": int(len(self.available_meshes) or self.mesh_count),
            "selected_paths": list(self.selected_paths),
            "raw_vertex_count": int(self.raw_vertex_count),
            "raw_face_count": int(self.raw_face_count),
            "vertex_count": int(len(self.display_vertices_cm)),
            "face_count": int(len(self.display_faces)),
            "bbox_min_cm": [round(float(x), 2) for x in self.bbox_min_cm],
            "bbox_max_cm": [round(float(x), 2) for x in self.bbox_max_cm],
            "span_cm": span_cm,
        }


def _triangulate_polys(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vectorized pure-NumPy fan triangulation of arbitrary polygon `(counts, indices)` -> `[T, 3]` `int32`."""
    c = np.asarray(counts, dtype=np.int64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int32).reshape(-1)
    if len(c) == 0 or len(idx) < 3:
        return np.zeros((0, 3), dtype=np.int32)
    if np.all(c == 3) and len(idx) == 3 * len(c):
        return idx.reshape(-1, 3).copy()

    valid_poly = c >= 3
    if not np.any(valid_poly):
        return np.zeros((0, 3), dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(c)[:-1]])
    c_v = c[valid_poly]
    s_v = starts[valid_poly]
    n_tris_per_poly = c_v - 2
    poly_rep = np.repeat(np.arange(len(c_v), dtype=np.int64), n_tris_per_poly)
    tri_starts = np.concatenate([[0], np.cumsum(n_tris_per_poly)[:-1]])
    local_k = np.arange(len(poly_rep), dtype=np.int64) - tri_starts[poly_rep]  # 0 .. c-3
    base_s = s_v[poly_rep]
    i0 = idx[base_s]
    i1 = idx[base_s + local_k + 1]
    i2 = idx[base_s + local_k + 2]
    return np.stack([i0, i1, i2], axis=-1).astype(np.int32)


def _decimate_mesh_grid_numpy(
    verts: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fast, zero-dependency 3D spatial vertex-clustering decimator in pure NumPy.
    Preserves full surface coverage & topology on multi-hundred-thousand-polygon LiDAR scans in < 25 ms."""
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if len(f) <= target_faces or len(v) <= 12:
        return v, f

    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    span = np.maximum(vmax - vmin, 1e-3)
    # Choose grid resolution proportional to sqrt(target_faces) along the two dominant scene extents
    res = max(24, int(round(np.sqrt(float(target_faces) * 1.35))))
    step = float(np.max(span)) / float(res)

    for _ in range(4):
        coords = np.floor((v - vmin[None, :]) / max(step, 1e-5)).astype(np.int64)
        max_c = coords.max(axis=0) + 1
        keys = coords[:, 0] * (max_c[1] * max_c[2]) + coords[:, 1] * max_c[2] + coords[:, 2]
        _, inv = np.unique(keys, return_inverse=True)
        n_clusters = int(inv.max()) + 1 if len(inv) else 0
        if n_clusters < 4:
            return v, f
        f_mapped = inv[f]
        non_deg = (f_mapped[:, 0] != f_mapped[:, 1]) & (f_mapped[:, 1] != f_mapped[:, 2]) & (f_mapped[:, 0] != f_mapped[:, 2])
        f_new = f_mapped[non_deg].astype(np.int32)
        if len(f_new) == 0:
            return v, f
        # Deduplicate faces ignoring cyclic order while keeping original winding
        f_sorted = np.sort(f_new, axis=1)
        _, uniq_idx = np.unique(f_sorted, axis=0, return_index=True)
        f_new = f_new[np.sort(uniq_idx)]
        if len(f_new) <= target_faces * 1.25 or step >= float(np.max(span)) * 0.15:
            # Average vertex positions inside each cluster
            v_sum = np.zeros((n_clusters, 3), dtype=np.float64)
            v_cnt = np.bincount(inv, minlength=n_clusters).astype(np.float64)
            np.add.at(v_sum, inv, v.astype(np.float64))
            v_new = (v_sum / np.maximum(v_cnt[:, None], 1.0)).astype(np.float32)
            # Compact unreferenced vertices
            used = np.unique(f_new.ravel())
            remap = np.full(n_clusters, -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            return v_new[used], remap[f_new]
        step *= float(np.sqrt(len(f_new) / max(float(target_faces), 100.0))) * 1.08

    return v, f


def _read_usd_scan(
    p: Path, unit: str, selected_paths: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, float, int, list[dict]]:
    from pxr import Usd, UsdGeom, UsdSkel

    from .usd_io import _stage_axes, open_usd_stage

    stage = open_usd_stage(p)
    stage_axes = _stage_axes(stage)
    up = "z" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else "y"
    unit_scale = stage_axes.to_cm if unit == "auto" else to_cm(unit)
    axes = Axes.of(unit_scale, up)

    start_tc = stage.GetStartTimeCode()
    t_eval = Usd.TimeCode(start_tc) if np.isfinite(start_tc) else Usd.TimeCode.Default()
    xf_cache = UsdGeom.XformCache(t_eval)

    verts_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    available_meshes: list[dict] = []
    v_offset = 0
    mesh_count = 0

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        cur = prim.GetParent()
        in_skel = False
        while cur and cur.IsValid() and not cur.IsPseudoRoot():
            if cur.IsA(UsdSkel.Root):
                in_skel = True
                break
            cur = cur.GetParent()
        if in_skel:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(t_eval) == UsdGeom.Tokens.invisible:
            continue

        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get(t_eval)
        if pts_attr is None:
            pts_attr = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        cnt_attr = mesh.GetFaceVertexCountsAttr().Get(t_eval)
        if cnt_attr is None:
            cnt_attr = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        idx_attr = mesh.GetFaceVertexIndicesAttr().Get(t_eval)
        if idx_attr is None:
            idx_attr = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        if pts_attr is None or cnt_attr is None or idx_attr is None:
            continue

        pts_local = np.asarray(pts_attr, dtype=np.float64)
        tris = _triangulate_polys(np.asarray(cnt_attr, dtype=np.int32), np.asarray(idx_attr, dtype=np.int32))
        if len(pts_local) < 3 or len(tris) == 0:
            continue

        anc = prim.GetParent()
        xf_prim = (
            anc
            if (anc and anc.IsValid() and anc.IsA(UsdGeom.Xform) and len(anc.GetChildren()) == 1)
            else prim
        )
        m_path = _norm_scene_path(str(xf_prim.GetPath()))
        m_name = str(xf_prim.GetCustomDataByKey(ORIGINAL) or xf_prim.GetName())
        available_meshes.append(
            {
                "name": m_name,
                "path": m_path,
                "vertex_count": int(len(pts_local)),
                "face_count": int(len(tris)),
            }
        )
        if not path_matches_selection(m_path, selected_paths):
            continue

        l2w_raw = np.array(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
        pts_raw_world = (pts_local @ l2w_raw[:3, :3].T) + l2w_raw[:3, 3]
        pts_cm = axes.world_points(pts_raw_world).astype(np.float32)

        orientation = mesh.GetOrientationAttr().Get()
        if orientation == UsdGeom.Tokens.leftHanded:
            tris = tris[:, [0, 2, 1]]

        verts_parts.append(pts_cm)
        faces_parts.append(tris + v_offset)
        v_offset += len(pts_cm)
        mesh_count += 1

    if not available_meshes:
        raise ValueError(f"USD 场景扫描文件 {p.name} 中未找到任何有效静态网格（UsdGeom.Mesh）")
    if not verts_parts:
        raise ValueError("所选层级节点下不包含任何有效静态网格，请在层级树中至少勾选一个网格或父级组")
    return (
        np.concatenate(verts_parts, axis=0),
        np.concatenate(faces_parts, axis=0),
        float(unit_scale),
        mesh_count,
        available_meshes,
    )


def _read_abc_scan(
    p: Path, unit: str, up: str = "y", selected_paths: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, float, int, list[dict]]:
    from .abc_io import _import_abc_arrays
    from .scene_arrays import text

    _, items = _import_abc_arrays(p)
    models = items.get("model") or []
    if not models:
        raise ValueError(f"Alembic 场景扫描文件 {p.name} 中未找到任何多边形网格（IPolyMesh）")

    unit_scale = 1.0 if unit == "auto" else to_cm(unit)
    axes = Axes.of(unit_scale, up)

    verts_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    available_meshes: list[dict] = []
    v_offset = 0
    mesh_count = 0

    for it in models:
        if "points" not in it or "counts" not in it or "indices" not in it:
            continue
        pts_arr = np.asarray(it["points"], dtype=np.float64)
        pts_local = pts_arr[0] if pts_arr.ndim == 3 else pts_arr
        tris = _triangulate_polys(np.asarray(it["counts"], dtype=np.int32), np.asarray(it["indices"], dtype=np.int32))
        if len(pts_local) < 3 or len(tris) == 0:
            continue

        m_path = _norm_scene_path(text(it["path"]))
        m_name = text(it["name"])
        available_meshes.append(
            {
                "name": m_name,
                "path": m_path,
                "vertex_count": int(len(pts_local)),
                "face_count": int(len(tris)),
            }
        )
        if not path_matches_selection(m_path, selected_paths):
            continue

        world_arr = np.asarray(it["world"], dtype=np.float64) if "world" in it else np.eye(4, dtype=np.float64)[None]
        w0 = world_arr[0] if world_arr.ndim == 3 else world_arr
        pts_raw_world = (pts_local @ w0[:3, :3].T) + w0[:3, 3]
        pts_cm = axes.world_points(pts_raw_world).astype(np.float32)

        verts_parts.append(pts_cm)
        faces_parts.append(tris + v_offset)
        v_offset += len(pts_cm)
        mesh_count += 1

    if not available_meshes:
        raise ValueError(f"Alembic 场景扫描文件 {p.name} 中未解析到有效三角网格拓扑")
    if not verts_parts:
        raise ValueError("所选层级节点下不包含任何有效静态网格，请在层级树中至少勾选一个网格或父级组")
    return (
        np.concatenate(verts_parts, axis=0),
        np.concatenate(faces_parts, axis=0),
        float(unit_scale),
        mesh_count,
        available_meshes,
    )


def _read_fbx_scan(
    p: Path, unit: str, selected_paths: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, float, int, list[dict]]:
    from .fbx_io import _import_fbx_arrays
    from .scene_arrays import text

    top, items, _ = _import_fbx_arrays(p)
    models = items.get("model") or []
    if not models:
        raise ValueError(f"FBX 场景扫描文件 {p.name} 中未找到任何静态网格节点（FbxMesh）")

    file_unit_cm = float(top.get("unit_cm", 1.0))
    unit_scale = file_unit_cm if unit == "auto" else to_cm(unit)
    axes = Axes(unit_scale, np.asarray(top.get("axes", np.eye(3)), dtype=np.float64))

    verts_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    available_meshes: list[dict] = []
    v_offset = 0
    mesh_count = 0

    for it in models:
        if "points" not in it or "counts" not in it or "indices" not in it:
            continue
        pts_local = np.asarray(it["points"], dtype=np.float64)
        tris = _triangulate_polys(np.asarray(it["counts"], dtype=np.int32), np.asarray(it["indices"], dtype=np.int32))
        if len(pts_local) < 3 or len(tris) == 0:
            continue

        m_path = _norm_scene_path(text(it["path"]))
        m_name = text(it["name"])
        available_meshes.append(
            {
                "name": m_name,
                "path": m_path,
                "vertex_count": int(len(pts_local)),
                "face_count": int(len(tris)),
            }
        )
        if not path_matches_selection(m_path, selected_paths):
            continue

        world_arr = np.asarray(it["world"], dtype=np.float64) if "world" in it else np.eye(4, dtype=np.float64)[None]
        w0 = world_arr[0] if world_arr.ndim == 3 else world_arr
        pts_raw_world = (pts_local @ w0[:3, :3].T) + w0[:3, 3]
        pts_cm = axes.world_points(pts_raw_world).astype(np.float32)

        verts_parts.append(pts_cm)
        faces_parts.append(tris + v_offset)
        v_offset += len(pts_cm)
        mesh_count += 1

    if not available_meshes:
        raise ValueError(f"FBX 场景扫描文件 {p.name} 中未解析到有效三角网格拓扑")
    if not verts_parts:
        raise ValueError("所选层级节点下不包含任何有效静态网格，请在层级树中至少勾选一个网格或父级组")
    return (
        np.concatenate(verts_parts, axis=0),
        np.concatenate(faces_parts, axis=0),
        float(unit_scale),
        mesh_count,
        available_meshes,
    )


def load_scene_scan(
    path: str | Path,
    unit: str = "auto",
    up: str = "y",
    selected_paths: list[str] | None = None,
    max_calib_faces: int = 120_000,
    max_display_faces: int = 28_000,
    max_solver_faces: int = 6_000_000,
) -> SceneScanData:
    """Load a static world-space scene scan mesh from `.abc`, `.fbx`, or `.usd`/`.usda`/`.usdc`/`.usdz`,
    optionally restricted to specific hierarchy node paths (`selected_paths`)."""
    from . import build_hierarchy_tree

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"未找到场景扫描文件：{p}")
    ext = p.suffix.lower()
    use_unit = unit.strip().lower() if unit and unit.strip().lower() in ("auto", "cm", "m") else "auto"
    clean_sel = [_norm_scene_path(s) for s in (selected_paths or []) if str(s or "").strip()]

    if ext in USD_EXTS:
        verts_raw, faces_raw, resolved_scale, mesh_cnt, avail_meshes = _read_usd_scan(p, use_unit, clean_sel or None)
        fmt = "usd"
    elif ext == ".abc":
        verts_raw, faces_raw, resolved_scale, mesh_cnt, avail_meshes = _read_abc_scan(
            p, use_unit, up=up, selected_paths=clean_sel or None
        )
        fmt = "abc"
    elif ext == ".fbx":
        verts_raw, faces_raw, resolved_scale, mesh_cnt, avail_meshes = _read_fbx_scan(p, use_unit, clean_sel or None)
        fmt = "fbx"
    else:
        raise ValueError(f"不支持的场景扫描格式：{p.suffix}（支持 .abc, .fbx, .usd/.usda/.usdc/.usdz）")

    raw_v_cnt = int(len(verts_raw))
    raw_f_cnt = int(len(faces_raw))
    bmin = verts_raw.min(axis=0).astype(float).tolist()
    bmax = verts_raw.max(axis=0).astype(float).tolist()

    v_cal, f_cal = _decimate_mesh_grid_numpy(verts_raw, faces_raw, target_faces=max_calib_faces)
    v_disp, f_disp = _decimate_mesh_grid_numpy(v_cal, f_cal, target_faces=max_display_faces)
    v_sol: np.ndarray | None = None
    f_sol: np.ndarray | None = None
    if len(f_cal) < raw_f_cnt:
        v_sol, f_sol = _decimate_mesh_grid_numpy(verts_raw, faces_raw, target_faces=max_solver_faces)
        v_sol, f_sol = np.asarray(v_sol, dtype=np.float32), np.asarray(f_sol, dtype=np.int32)
    tree = build_hierarchy_tree([], avail_meshes, [])

    return SceneScanData(
        name=p.stem,
        source_file=str(p),
        format=fmt,
        unit=use_unit,
        resolved_unit_to_cm=resolved_scale,
        mesh_count=mesh_cnt,
        raw_vertex_count=raw_v_cnt,
        raw_face_count=raw_f_cnt,
        vertices_cm=v_cal,
        faces=f_cal,
        display_vertices_cm=v_disp,
        display_faces=f_disp,
        bbox_min_cm=bmin,
        bbox_max_cm=bmax,
        selected_paths=clean_sel,
        solver_vertices_cm=v_sol,
        solver_faces=f_sol,
        available_meshes=avail_meshes,
        hierarchy=tree,
    )
