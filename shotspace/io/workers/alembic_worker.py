"""Standalone PyAlembic worker running under `third_party/alembic/.venv/bin/python`.

Usage:
    python -m shotspace.io.workers.alembic_worker <job.json>

Supports:
  - `mode: "import"`: `.abc` -> `scene.npz` (cameras + meshes; reads `_ai_DCC_FPS` directly from top metadata
    to avoid GBK crash in `GetArchiveInfo()`, filters motion-blur subframes `SUBFRAME = 1e-3`, converts apertures
    & film offsets `cm -> mm` via `/ 0.1`, overscan `/ 2.0`, row-vector -> column-vector `.T`).
  - `mode: "export"`: `scene.npz` -> `.abc` (Ogawa archive with `CreateArchiveWithInfo` FPS hint, cameras with
    `* 0.1` `mm -> cm` aperture conversion, LBS-baked meshes with CW `reverse_faces` winding & `shown_span` visibility).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import imath
import imathnumpy
import numpy as np
from alembic import Abc, AbcCoreAbstract, AbcGeom, Util

from shotspace.core.names import ORIGINAL, identifier, join_path, split_path, unique
from shotspace.core.units import CM_PER_MM, DEFAULT_FPS, standard_fps
from shotspace.io.scene_arrays import SceneArrays, load, shown_span, text

WRAP = Abc.WrapExistingFlag.kWrapExisting
FACEVARYING = AbcGeom.GeometryScope.kFacevaryingScope
SUBFRAME = 1e-3
DCC_FPS = "_ai_DCC_FPS"
RESOLUTION = "lab2shot:resolution"


def m44_to_np(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def np_to_m44(a: np.ndarray):
    return imath.M44d(*(float(v) for v in np.asarray(a, dtype=np.float64).reshape(16)))


def filled(array_type, data: np.ndarray, dtype):
    data = np.ascontiguousarray(data, dtype=dtype)
    out = array_type(len(data))
    if len(data):
        imathnumpy.arrayToNumpy(out)[...] = data
    return out


def as_numpy(values, dtype=np.float64) -> np.ndarray:
    return np.array(imathnumpy.arrayToNumpy(values), dtype)


def reverse_faces(counts: np.ndarray, per_corner: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    face = np.repeat(np.arange(len(counts)), counts)
    offset = np.arange(len(per_corner)) - starts[face]
    return per_corner[starts[face] + counts[face] - 1 - offset]


def children(obj) -> list:
    return [obj.getChild(i) for i in range(obj.getNumChildren())]


def is_xform(obj) -> bool:
    return AbcGeom.IXform.matches(obj.getMetaData())


def is_camera(obj) -> bool:
    return AbcGeom.ICamera.matches(obj.getMetaData())


def is_mesh(obj) -> bool:
    return AbcGeom.IPolyMesh.matches(obj.getMetaData())


def all_objects(obj):
    for child in children(obj):
        yield child
        yield from all_objects(child)


def object_name(obj) -> str:
    schema = AbcGeom.IXform(obj, WRAP).getSchema() if is_xform(obj) else None
    props = schema.getUserProperties() if schema is not None else None
    if props is not None and props.valid() and props.getPropertyHeader(ORIGINAL) is not None:
        return Abc.IStringProperty(props, ORIGINAL).getValue()
    return obj.getName()


def item_path(obj) -> str:
    parent = obj.getParent()
    if (
        parent.valid()
        and parent.getFullName() != "/"
        and is_xform(parent)
        and len(children(parent)) == 1
    ):
        obj = parent
    chain = []
    while obj.valid() and obj.getFullName() != "/":
        chain.append(object_name(obj))
        obj = obj.getParent()
    return join_path(reversed(chain))


def parent_xforms(obj) -> list:
    chain = []
    parent = obj.getParent()
    while parent.valid() and parent.getFullName() != "/":
        if is_xform(parent):
            chain.append(AbcGeom.IXform(parent, WRAP).getSchema())
        parent = parent.getParent()
    return chain


def sample_times(schema) -> list[float]:
    ts = schema.getTimeSampling()
    return [ts.getSampleTime(i) for i in range(schema.getNumSamples())]


def world_matrix(xforms: list, sel) -> np.ndarray:
    acc = np.eye(4, dtype=np.float64)
    for xs in xforms:
        if xs.getNumSamples() == 0:
            continue
        sample = xs.getValue(sel)
        acc = acc @ m44_to_np(sample.getMatrix())
        if not sample.getInheritsXforms():
            break
    return acc.T


def visibilities(obj) -> list:
    out = []
    while obj.valid() and obj.getFullName() != "/":
        prop = AbcGeom.GetVisibilityProperty(obj)
        if prop is not None and prop.valid() and prop.getNumSamples():
            out.append(prop)
        obj = obj.getParent()
    return out


def item_frames(schema, xforms: list, fps: float, also=()) -> dict[int, float]:
    times = sorted({t for s in [schema, *xforms, *also] if s.getNumSamples() > 1 for t in sample_times(s)})
    times = times or sample_times(schema)[:1] or [0.0]
    by_frame: dict[int, float] = {}
    for t in times:
        f = int(round(t * fps))
        if f not in by_frame or abs(t * fps - f) < abs(by_frame[f] * fps - f):
            by_frame[f] = t
    on = {f: t for f, t in by_frame.items() if abs(t * fps - f) <= SUBFRAME}
    return dict(sorted((on or by_frame).items()))


def dcc_fps(archive) -> float:
    try:
        hint = float(archive.getTop().getMetaData().get(DCC_FPS) or 0.0)
    except ValueError:
        return 0.0
    return standard_fps(hint) if hint > 0 else 0.0


def sampled_fps(archive) -> float:
    rates = []
    for i in range(1, archive.getNumTimeSamplings()):
        kind = archive.getTimeSampling(i).getTimeSamplingType()
        if (kind.isUniform() or kind.isCyclic()) and kind.getTimePerCycle() > 0:
            rates.append(standard_fps(1.0 / kind.getTimePerCycle()))
    return max(set(rates), key=rates.count) if rates else 0.0


def user_properties(schema, fps: float) -> dict:
    props = schema.getUserProperties()
    out: dict = {}
    if not props.valid():
        return out
    for i in range(props.getNumProperties()):
        header = props.getPropertyHeader(i)
        try:
            name = header.getName()
        except UnicodeDecodeError:
            continue
        if not (name.startswith("lab2shot:") or name.startswith("shotspace:")) or name == ORIGINAL or not header.isScalar():
            continue
        pod = header.getDataType().getPod()
        if pod == Util.POD.kStringPOD:
            out[name] = Abc.IStringProperty(props, name).getValue()
    return out


def do_import(abc_path: Path, out_npz: Path, cameras_only: bool = False) -> dict:
    archive = Abc.IArchive(str(abc_path))
    recorded = dcc_fps(archive) or sampled_fps(archive)
    fps = recorded or DEFAULT_FPS
    out = SceneArrays(fps, unit_cm=1.0)
    out.top["fps_recorded"] = np.array(bool(recorded))

    for obj in all_objects(archive.getTop()):
        if not cameras_only and is_mesh(obj):
            schema = AbcGeom.IPolyMesh(obj, WRAP).getSchema()
            if schema.getNumSamples() == 0:
                continue
            samp0 = schema.getValue(Abc.ISampleSelector(0))
            cnts = as_numpy(samp0.getFaceCounts(), np.int32)
            raw_idx = as_numpy(samp0.getFaceIndices(), np.int32)
            if len(cnts) == 0 or len(raw_idx) == 0:
                continue
            path = item_path(obj)
            name = split_path(path)[-1]
            xforms, seen = parent_xforms(obj), visibilities(obj)
            by_frame = item_frames(schema, xforms, fps, seen)
            frames = np.array(list(by_frame), dtype=np.int64)
            selectors = [Abc.ISampleSelector(float(t)) for t in by_frame.values()]
            world = np.stack([world_matrix(xforms, sel) for sel in selectors])
            is_static_mesh = schema.getNumSamples() <= 1
            idx_ccw = reverse_faces(cnts, raw_idx).astype(np.int32) if (len(cnts) and len(raw_idx)) else raw_idx
            if is_static_mesh:
                pts = as_numpy(samp0.getPositions(), np.float32)
            else:
                pts = np.stack([as_numpy(schema.getValue(sel).getPositions(), np.float32) for sel in selectors])
            if pts.size == 0:
                continue
            vis_kw = {}
            if seen:
                on = np.ones(len(selectors), dtype=bool)
                for prop in seen:
                    on &= np.array([int(prop.getValue(sel)) != int(AbcGeom.ObjectVisibility.kVisibilityHidden) for sel in selectors])
                if not on.all():
                    vis_kw["visible"] = on.astype(np.int32)
            out.add(
                "model",
                name,
                path,
                frames,
                world,
                points=pts,
                counts=cnts,
                indices=idx_ccw,
                is_rigid_object=np.array(bool(is_static_mesh)),
                **vis_kw,
            )
            continue
        if not is_camera(obj):
            continue
        schema = AbcGeom.ICamera(obj, WRAP).getSchema()
        path = item_path(obj)
        name = split_path(path)[-1]
        xforms, seen = parent_xforms(obj), visibilities(obj)
        by_frame = item_frames(schema, xforms, fps, seen)
        frames = np.array(list(by_frame), dtype=np.int64)
        selectors = [Abc.ISampleSelector(float(t)) for t in by_frame.values()]
        world = np.stack([world_matrix(xforms, sel) for sel in selectors])

        lens = {k: [] for k in ("focal", "hap", "vap", "hoff", "voff", "squeeze", "over")}
        if schema.getNumSamples() == 0:
            for _ in selectors:
                lens["focal"].append(35.0)
                lens["hap"].append(36.0)
                lens["vap"].append(20.25)
                lens["hoff"].append(0.0)
                lens["voff"].append(0.0)
                lens["squeeze"].append(1.0)
                lens["over"].append([0.0, 0.0, 0.0, 0.0])
        else:
            for sel in selectors:
                cam = schema.getValue(sel)
                foc = float(cam.getFocalLength())
                hap_mm = float(cam.getHorizontalAperture()) / CM_PER_MM
                vap_mm = float(cam.getVerticalAperture()) / CM_PER_MM
                lens["focal"].append(foc if foc > 0.05 else 35.0)
                lens["hap"].append(hap_mm if hap_mm > 0.05 else 36.0)
                lens["vap"].append(vap_mm if vap_mm > 0.05 else 20.25)
                lens["hoff"].append(float(cam.getHorizontalFilmOffset()) / CM_PER_MM)
                lens["voff"].append(float(cam.getVerticalFilmOffset()) / CM_PER_MM)
                sq = float(cam.getLensSqueezeRatio())
                lens["squeeze"].append(sq if sq > 1e-4 else 1.0)
                lens["over"].append(
                    [cam.getOverScanLeft(), cam.getOverScanTop(), cam.getOverScanRight(), cam.getOverScanBottom()]
                )
        props = user_properties(schema, fps)
        size = json.loads(props.pop(RESOLUTION, props.pop("shotspace:resolution", "null")))
        out.add(
            "camera",
            name,
            path,
            frames,
            world,
            focal_mm=np.array(lens["focal"], dtype=np.float64),
            h_aperture_mm=np.array(lens["hap"], dtype=np.float64),
            v_aperture_mm=np.array(lens["vap"], dtype=np.float64),
            center_mm=np.array([lens["hoff"], lens["voff"]], dtype=np.float64).T,
            pixel_aspect=np.float64(lens["squeeze"][0]),
            overscan=np.array(lens["over"][0], dtype=np.float64) / 2.0,
            properties=np.array(json.dumps(props, ensure_ascii=False)),
            resolution=np.array(size, dtype=np.int64) if size else None,
        )

    out.save(out_npz)
    return {"fps": fps, "fps_recorded": bool(recorded), "cameras": len(out.items["camera"])}


# ------------------------------------------------------------------ export


class Names:
    def __init__(self) -> None:
        self.used: dict[object, set[str]] = {}

    def __call__(self, parent_key: object, name: str) -> str:
        used = self.used.setdefault(parent_key, set())
        out = unique(identifier(name or "node"), used)
        used.add(out)
        return out

    @staticmethod
    def keep(schema, written: str, name: str) -> None:
        if written != name:
            Abc.OStringProperty(schema.getUserProperties(), ORIGINAL).setValue(name)


def matrix_sample(column_vector: np.ndarray):
    s = AbcGeom.XformSample()
    s.setMatrix(np_to_m44(np.asarray(column_vector, dtype=np.float64).T))
    return s


class Samplings:
    def __init__(self, archive, fps: float):
        self.archive, self.fps, self.made = archive, fps, {}

    def __call__(self, frames) -> int:
        key = tuple(int(f) for f in frames)
        if key not in self.made:
            if len(key) > 1 and any(b - a != 1 for a, b in zip(key, key[1:])):
                T = AbcCoreAbstract.TimeSamplingType
                times = AbcCoreAbstract.TimeVector()
                times[:] = [f / self.fps for f in key]
                ts = AbcCoreAbstract.TimeSampling(T(T.AcyclicNumSamples(), T.AcyclicTimePerCycle()), times)
            else:
                ts = AbcCoreAbstract.TimeSampling(1.0 / self.fps, key[0] / self.fps)
            self.made[key] = self.archive.addTimeSampling(ts)
        return self.made[key]


def placed(top, names: Names, item: dict, tsidx: int, groups: dict, samplings):
    parts = split_path(text(item.get("shown", item["path"])))
    if parts and parts[0] == "shot":
        parts = parts[1:]
    parent, key = top, "/"
    for segment in parts[:-1]:
        inner = f"{key}{segment}/"
        if inner not in groups:
            groups[inner] = AbcGeom.OXform(parent, names(key, segment), 0)
            groups[inner].getSchema().set(matrix_sample(np.eye(4)))
            names.keep(groups[inner].getSchema(), groups[inner].getName(), segment)
        parent, key = groups[inner], inner
    shown = parts[-1] if parts else text(item["name"])
    name = names(key, shown)
    world = np.asarray(item["world"], dtype=np.float64).reshape(-1, 4, 4)
    is_const_xf = len(world) <= 1 or float(np.abs(world - world[:1]).max()) <= 1e-5
    xf = AbcGeom.OXform(parent, name, 0 if is_const_xf else tsidx)
    for m in (world[:1] if is_const_xf else world):
        xf.getSchema().set(matrix_sample(m))
    span, on_span = shown_span(item)
    if not on_span.all():
        vis = AbcGeom.CreateVisibilityProperty(xf, samplings(span))
        for on in on_span:
            vis.setValue(
                int(
                    AbcGeom.ObjectVisibility.kVisibilityDeferred
                    if on
                    else AbcGeom.ObjectVisibility.kVisibilityHidden
                )
            )
    names.keep(xf.getSchema(), name, shown)
    return xf, name


def _write_archive(items: dict[str, list[dict]], tmp: Path, fps: float) -> dict:
    fps = standard_fps(fps)
    archive = Abc.CreateArchiveWithInfo(str(tmp), fps, "ShotSpace", "ShotSpace Alembic Export")
    top = archive.getTop()
    samplings = Samplings(archive, fps)
    names = Names()
    groups: dict = {}

    for item in items["camera"]:
        tsidx = samplings(item["frames"])
        xf, name = placed(top, names, item, tsidx, groups, samplings)
        lens = {k: np.asarray(item[f"{k}_mm"], dtype=np.float64).reshape(-1) for k in ("focal", "h_aperture", "v_aperture")}
        center = np.asarray(item["center_mm"], dtype=np.float64).reshape(-1, 2)
        over = np.asarray(item["overscan"], dtype=np.float64).reshape(4) * 2.0
        n = max(max(len(v) for v in lens.values()), len(center))
        schema = AbcGeom.OCamera(xf, f"{name}Shape", tsidx).getSchema()
        props = json.loads(text(item["properties"])) if "properties" in item else {}
        if "resolution" in item:
            res_list = [int(v) for v in np.asarray(item["resolution"]).reshape(-1)[:2]]
            Abc.OStringProperty(schema.getUserProperties(), RESOLUTION).setValue(json.dumps(res_list))
        for k_prop, v_prop in props.items():
            if isinstance(v_prop, str):
                Abc.OStringProperty(schema.getUserProperties(), k_prop).setValue(v_prop)
        for f in range(n):
            cam = AbcGeom.CameraSample()
            cam.setFocalLength(float(lens["focal"][min(f, len(lens["focal"]) - 1)]))
            cam.setHorizontalAperture(float(lens["h_aperture"][min(f, len(lens["h_aperture"]) - 1)]) * CM_PER_MM)
            cam.setVerticalAperture(float(lens["v_aperture"][min(f, len(lens["v_aperture"]) - 1)]) * CM_PER_MM)
            cam.setHorizontalFilmOffset(float(center[min(f, len(center) - 1), 0]) * CM_PER_MM)
            cam.setVerticalFilmOffset(float(center[min(f, len(center) - 1), 1]) * CM_PER_MM)
            cam.setLensSqueezeRatio(float(item["pixel_aspect"]))
            cam.setOverScanLeft(float(over[0]))
            cam.setOverScanTop(float(over[1]))
            cam.setOverScanRight(float(over[2]))
            cam.setOverScanBottom(float(over[3]))
            schema.set(cam)

    for item in items["model"]:
        tsidx = samplings(item["frames"])
        xf, name = placed(top, names, item, tsidx, groups, samplings)
        counts = np.asarray(item["counts"], dtype=np.int32)
        indices = np.asarray(item["indices"], dtype=np.int32)
        points = np.asarray(item["points"], dtype=np.float32)
        points = points[None] if points.ndim == 2 else points
        extra = {}
        if "uv" in item and "uv_indices" in item:
            extra["iUVs"] = AbcGeom.OV2fGeomParamSample(
                filled(imath.V2fArray, np.asarray(item["uv"]).reshape(-1, 2), np.float32),
                filled(imath.UnsignedIntArray, reverse_faces(counts, np.asarray(item["uv_indices"])), np.uint32),
                FACEVARYING,
            )
        shape_tsidx = 0 if len(points) == 1 else tsidx
        shape = AbcGeom.OPolyMesh(xf, f"{name}Shape", shape_tsidx)
        span, on_span = shown_span(item)
        if not on_span.all():
            shape_vis = AbcGeom.CreateVisibilityProperty(shape, samplings(span))
            for on in on_span:
                shape_vis.setValue(
                    int(
                        AbcGeom.ObjectVisibility.kVisibilityDeferred
                        if on
                        else AbcGeom.ObjectVisibility.kVisibilityHidden
                    )
                )
        mesh = shape.getSchema()
        topo_i = filled(imath.IntArray, reverse_faces(counts, indices), np.int32)
        topo_c = filled(imath.IntArray, counts, np.int32)
        for f, p in enumerate(points):
            P = filled(imath.V3fArray, p, np.float32)
            mesh.set(
                AbcGeom.OPolyMeshSchemaSample(P, topo_i, topo_c, **extra)
                if f == 0
                else AbcGeom.OPolyMeshSchemaSample(P)
            )

    return {"cameras": len(items["camera"]), "models": len(items["model"])}


def do_export(in_npz: Path, out_abc: Path) -> dict:
    top, items = load(in_npz)
    fps = float(top["fps"])
    out_abc.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_abc.with_name(f".{out_abc.name}.{os.getpid()}.tmp")
    try:
        counts = _write_archive(items, tmp, fps)
        gc.collect()  # Ensure C++ OArchive flushes & closes file descriptor before rename
        tmp.replace(out_abc)
        return counts
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mode = job["mode"]
    if mode == "import":
        res = do_import(Path(job["input"]), Path(job["output"]), cameras_only=bool(job.get("cameras_only", False)))
    elif mode == "export":
        res = do_export(Path(job["input"]), Path(job["output"]))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if "result" in job:
        Path(job["result"]).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
