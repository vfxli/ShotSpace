"""Data collection: every job leaves its data in the table of the day (v0.2.8).

Why: the real shots exist only at the studio.  What a job was given and what it produced is written, as numbers,
into one table per day on the machine that ran it (`data_collect/shotspace_data_YYYYMMDD.csv`).  Nothing is sent
anywhere; the tables stay where they are until somebody copies them.  They hold no pictures and no file paths.

The table (UTF-8 CSV, opens in a spreadsheet program):
  - one row per job (`记录类型` = 任务): outcome, timings, plate, settings, the camera of every frame (position,
    rotation, focal length, intrinsics), the scan and the height of its ground at a grid of points;
  - one row per person a people solve computed (`记录类型` = 人物): the readings, the solved body points of every
    frame in world space, the height of the scan's ground where the person is, and - when the network looked at
    the person in this job - the network's own per-frame outputs, i.e. everything the whole-shot solve starts from;
  - one row per object an object solve computed (`记录类型` = 物体).
A cell that holds one value per frame is a list of numbers separated by spaces (`nan` = no value); rows of one job
share `作业编号`.  `eval/collected_table.py` reads a table back into arrays.

Conventions: world space is the camera file's (Y up), centimetres.  `摄影机旋转Rij` is the camera-to-world rotation,
row i column j, of a camera that looks down its own -Z with +Y up.  `网络_位置` is the network's translation of the
body in its camera (metres, X right, Y down, Z forward).  Body points are the model's keypoints.

Recording must never disturb a job: everything here is called through `record_job`, which catches every error and
writes it to `collect_errors.log` next to the tables."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 3  # 2: 拆分自 (v0.2.9); 3: 检测框修正帧 (v0.2.12)
TABLE_PREFIX = "shotspace_data_"

# (keypoint index of the body model, name in the table)
BODY_POINTS: tuple[tuple[int, str], ...] = (
    (69, "颈"), (0, "鼻"), (1, "左眼"), (2, "右眼"), (3, "左耳"), (4, "右耳"),
    (5, "左肩"), (6, "右肩"), (7, "左肘"), (8, "右肘"), (62, "左腕"), (41, "右腕"),
    (9, "左髋"), (10, "右髋"), (11, "左膝"), (12, "右膝"), (13, "左踝"), (14, "右踝"),
    (15, "左大脚趾"), (16, "左小脚趾"), (17, "左脚跟"), (18, "右大脚趾"), (19, "右小脚趾"), (20, "右脚跟"),
)
N_KEYPOINTS, N_SHAPE, N_SCALE, N_BODY_POSE = 70, 45, 28, 133
GRID_CELL_CM, GRID_MAX_SIDE, GRID_MARGIN_CM = 50.0, 41, 300.0

COMMON = ["表格版本", "记录类型", "日期", "时间", "软件版本", "解算规则版本", "任务编号", "任务名称", "作业编号", "作业类型"]

TASK_SCALARS = [
    "作业结果", "错误信息", "排队等待_秒", "耗时_秒", "人物检测耗时_秒", "逐帧估计耗时_秒", "整段解算耗时_秒",
    "显卡编号", "显卡型号",
    "画面宽_像素", "画面高_像素", "帧数", "起始帧", "结束帧", "帧率", "画面格式", "位深", "色彩空间", "显示变换",
    "曝光", "亮度", "对比", "饱和", "Gamma",
    "检测阈值", "检测到人数", "手动框人数", "选择方式", "平滑", "手部细节",
    "已解算人数", "本次新增人数", "本次重新解算人数", "本次跳过人数", "已解算物体数", "警告",
    "摄影机格式", "摄影机单位", "摄影机移动距离_厘米", "是否变焦", "像素宽高比",
    "扫描", "扫描状态", "扫描格式", "扫描单位", "扫描面数", "扫描网格数",
    "扫描尺寸X_厘米", "扫描尺寸Y_厘米", "扫描尺寸Z_厘米",
    "地面采样间距_厘米", "地面采样点数", "地面高度中位_厘米", "地面高度最低_厘米", "地面高度最高_厘米",
]
TASK_ARRAYS = (
    ["摄影机帧号", "摄影机位置X_厘米", "摄影机位置Y_厘米", "摄影机位置Z_厘米"]
    + [f"摄影机旋转R{i}{j}" for i in range(3) for j in range(3)]
    + ["焦距_毫米", "胶片口宽_毫米", "胶片口高_毫米", "内参FX_像素", "内参FY_像素", "内参CX_像素", "内参CY_像素"]
    + ["地面采样X_厘米", "地面采样Z_厘米", "地面采样高度_厘米"]
)

PERSON_SCALARS = [
    "人物编号", "手动框", "拆分自", "检测框修正帧", "本次计算", "画面贴合_像素", "画面贴合90_像素", "触地偏移_厘米", "触地帧占比",
    "体型比例", "尺度来源", "尺度权重", "已解算帧数", "隐藏帧数", "模型身高_厘米",
]
PERSON_ARRAYS = (
    ["帧号", "可见", "已解算", "逐帧画面误差_像素", "触地左", "触地右",
     "框X0_像素", "框Y0_像素", "框X1_像素", "框Y1_像素", "到摄影机距离_厘米", "所在位置地面高度_厘米",
     "骨盆X_厘米", "骨盆Y_厘米", "骨盆Z_厘米"]
    + [f"{name}{axis}_厘米" for _i, name in BODY_POINTS for axis in "XYZ"]
    + [f"体型参数_{i:02d}" for i in range(N_SHAPE)]
    + [f"比例参数_{i:02d}" for i in range(N_SCALE)]
)
NET_ARRAYS = (
    ["网络_帧号", "网络_检测分数", "网络_位置X_米", "网络_位置Y_米", "网络_位置Z_米",
     "网络_整体旋转X", "网络_整体旋转Y", "网络_整体旋转Z", "网络_有轮廓", "网络_轮廓面积_像素",
     "网络_轮廓框X0", "网络_轮廓框Y0", "网络_轮廓框X1", "网络_轮廓框Y1",
     "网络_框X0", "网络_框Y0", "网络_框X1", "网络_框Y1"]
    + [f"网络_体型_{i:02d}" for i in range(N_SHAPE)]
    + [f"网络_比例_{i:02d}" for i in range(N_SCALE)]
    + [f"网络_姿态_{i:03d}" for i in range(N_BODY_POSE)]
    + [f"网络_关键点X_{i:02d}" for i in range(N_KEYPOINTS)]
    + [f"网络_关键点Y_{i:02d}" for i in range(N_KEYPOINTS)]
    + [f"网络_关键点在轮廓内_{i:02d}" for i in range(N_KEYPOINTS)]
)
OBJECT_SCALARS = ["物体编号", "物体名称", "物体尺寸X_厘米", "物体尺寸Y_厘米", "物体尺寸Z_厘米"]
OBJECT_ARRAYS = (
    ["物体帧号", "物体可见", "物体位置X_厘米", "物体位置Y_厘米", "物体位置Z_厘米"]
    + [f"物体旋转R{i}{j}" for i in range(3) for j in range(3)]
)

COLUMNS: list[str] = COMMON + TASK_SCALARS + TASK_ARRAYS + PERSON_SCALARS + PERSON_ARRAYS + NET_ARRAYS + OBJECT_SCALARS + OBJECT_ARRAYS
assert len(COLUMNS) == len(set(COLUMNS)), "column names of the data table must be unique"

GUIDE = f"""入镜 · ShotSpace 数据收集表 字段说明（表格版本 {SCHEMA_VERSION}）

文件：每天一张表，{TABLE_PREFIX}年月日.csv（UTF-8，可用表格软件打开）。只保存在本机，不联网、不外传。
内容：每个作业的数据（数字）。不含画面，不含文件路径。

行：
  记录类型 = 任务  每个作业一行：结果、耗时、画面规格、设置、摄影机逐帧数据、扫描与地面高度采样。
  记录类型 = 人物  人物解算中本次计算的每个人一行：质检读数、逐帧解算结果、所在位置的扫描地面高度；
                   本次经过网络估计的人（本次计算 = 新增）另含网络逐帧原始输出。
  记录类型 = 物体  物体解算中每个物体一行：逐帧位置与旋转。
  同一个作业的各行「作业编号」相同。

单元格：一个单元格内有多个数字时，以空格分隔，按帧排列；nan 表示该帧没有数值。
  任务行按「摄影机帧号」排列；人物行的解算结果按「帧号」排列，网络输出按「网络_帧号」排列；物体行按「物体帧号」排列。

约定：
  世界空间为摄影机文件的空间，Y 轴向上，单位厘米。
  摄影机旋转Rij：摄影机到世界的旋转矩阵第 i 行第 j 列；摄影机自身朝 -Z 方向看，+Y 向上。
  内参 FX / FY / CX / CY：像素单位的焦距与主点。
  地面采样：扫描模型（解算使用的那一份）在水平网格点上的地面高度；网格覆盖人物活动范围并向外扩 {GRID_MARGIN_CM / 100:.0f} 米，
            间距不小于 {GRID_CELL_CM:.0f} 厘米，每边最多 {GRID_MAX_SIDE} 个点；没有人物时覆盖整个扫描。
  所在位置地面高度：每一帧人物骨盆正下方的扫描地面高度。
  骨盆：左右髋的中点。其余身体点为人体模型的关键点。隐藏帧的位置保持在最近一个已解算帧的世界位置。
  拆分自：跟踪轨迹中途换了人时，轨迹被拆成多个人物；拆出的人物在此列记录原轨迹的编号。
  检测框修正帧：这些帧上该人物的轮廓带进了远离人物的碎块，检测框被撑大（例如下边缘突然伸到画面底部）；
            v0.2.12 起这些帧的轮廓在估计前去掉了碎块，框按前后帧一致的位置记录（空 = 没有这样的帧）。
  网络_位置：网络估计的人体在摄影机坐标中的位置，单位米，X 向右、Y 向下、Z 向前。
  网络_姿态 / 网络_体型 / 网络_比例 / 网络_整体旋转：人体模型参数（弧度或无量纲）。
  网络_关键点X / Y：网络估计的 70 个关键点在画面中的像素位置；网络_关键点在轮廓内：该关键点是否落在人物轮廓内（1 / 0）。
  手部姿态参数未记录。
"""


# ------------------------------------------------------------------ numbers -> cells


def _nums(values, fmt: str) -> str:
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0:
        return ""
    digits = int(fmt[2:-1]) if fmt.startswith("%.") and fmt.endswith("f") else 6
    a = np.round(a, digits) + 0.0  # rounded first and without a negative zero, so that no cell reads "-0.00"
    return " ".join(np.char.mod(fmt, a).tolist())


def _ints(values) -> str:
    a = np.asarray(values).ravel()
    return " ".join(str(int(v)) for v in a.tolist())


def _round(value, digits: int = 3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(v) else round(v, digits) + 0.0


# ------------------------------------------------------------------ the tables on disk


class DataCollector:
    """Appends rows to the table of the day.  One per task manager; safe to call from several job threads."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self._lock = threading.Lock()

    def _header_line(self) -> str:
        import io

        buf = io.StringIO()
        csv.writer(buf).writerow(COLUMNS)
        return buf.getvalue()

    def table_path(self, day: datetime) -> Path:
        """Today's table.  A table whose columns are not this version's (the software was updated during the
        day) is left as it is and a new one is begun next to it."""
        stem = f"{TABLE_PREFIX}{day.strftime('%Y%m%d')}"
        header = self._header_line().strip("\r\n")
        for n in range(1, 50):
            path = self.dir / (f"{stem}.csv" if n == 1 else f"{stem}_{n}.csv")
            if not path.exists():
                return path
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    first = fh.readline().strip("\r\n")
            except OSError:
                continue
            if first == header:
                return path
        return self.dir / f"{stem}_{int(time.time())}.csv"

    def append(self, rows: list[dict], now: datetime | None = None) -> Path | None:
        if not rows:
            return None
        now = now or datetime.now()
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            guide = self.dir / "字段说明.txt"
            try:
                if not guide.is_file() or guide.read_text(encoding="utf-8") != GUIDE:
                    guide.write_text(GUIDE, encoding="utf-8")
            except OSError:
                pass
            path = self.table_path(now)
            is_new = not path.exists()
            with path.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", restval="")
                if is_new:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return path

    def log_error(self, where: str) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with (self.dir / "collect_errors.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now().isoformat(timespec='seconds')} {where}\n{traceback.format_exc()}\n")
        except Exception:  # noqa: BLE001
            pass

    def today(self) -> dict:
        """For the admin page: where the tables are and how large today's is."""
        try:
            stem = f"{TABLE_PREFIX}{datetime.now().strftime('%Y%m%d')}"
            files = sorted(self.dir.glob(f"{stem}*.csv")) if self.dir.is_dir() else []
            size = sum(f.stat().st_size for f in files)
            total = sum(f.stat().st_size for f in self.dir.glob(f"{TABLE_PREFIX}*.csv")) if self.dir.is_dir() else 0
            return {"data_collect_dir": str(self.dir), "data_collect_today_mb": round(size / 1e6, 2), "data_collect_total_mb": round(total / 1e6, 2)}
        except OSError:
            return {"data_collect_dir": str(self.dir), "data_collect_today_mb": 0.0, "data_collect_total_mb": 0.0}


# ------------------------------------------------------------------ what a job leaves behind


class _Ground:
    """The scan's ground as the solver sees it: height of the supporting surface at points of the floor plan."""

    def __init__(self, scan_npz: Path) -> None:
        from ..solver.trajectory import GroundMesh

        full = scan_npz.with_name("scan_solver.npz")  # scans reduced for the view keep their full mesh for the solver
        with np.load(full if full.is_file() else scan_npz, allow_pickle=False) as d:
            verts = np.asarray(d["vertices_cm"], dtype=np.float64) * 0.01
            faces = np.asarray(d["faces"], dtype=np.int64)
        self.mesh = GroundMesh(verts, faces)
        self.ok = len(self.mesh) > 0
        tri = self.mesh.tri if self.ok else np.zeros((0, 3, 3))
        self.lo_cm = tri.reshape(-1, 3).min(axis=0) * 100.0 if self.ok else np.zeros(3)
        self.hi_cm = tri.reshape(-1, 3).max(axis=0) * 100.0 if self.ok else np.zeros(3)

    def heights_cm(self, x_cm: np.ndarray, z_cm: np.ndarray, near_y_cm: np.ndarray | float) -> np.ndarray:
        """Height of the walkable surface under (x, z) that lies nearest to `near_y_cm`; nan where there is none."""
        x = np.asarray(x_cm, dtype=np.float64).ravel()
        z = np.asarray(z_cm, dtype=np.float64).ravel()
        y = np.broadcast_to(np.asarray(near_y_cm, dtype=np.float64), x.shape).astype(np.float64)
        out = np.full(len(x), np.nan)
        if not self.ok or not len(x):
            return out
        good = np.isfinite(x) & np.isfinite(z) & np.isfinite(y)
        if not good.any():
            return out
        pts = np.stack([x[good], y[good], z[good]], axis=1) * 0.01
        nrm, off, ok = self.mesh.support(pts, max_gap_m=1.0e6)
        h = -(off + nrm[:, 0] * pts[:, 0] + nrm[:, 2] * pts[:, 2]) / np.where(np.abs(nrm[:, 1]) < 1e-9, 1e-9, nrm[:, 1])
        res = np.where(ok, h * 100.0, np.nan)
        out[good] = res
        return out


class _Task:
    """What the task holds at the moment its job ends (read once per record)."""

    def __init__(self, wm, sw, meta: dict) -> None:
        self.wm, self.sw, self.meta = wm, sw, meta
        self.plate = None
        self.cam = None
        self.frames: list[int] = []
        self.c2w = np.zeros((0, 4, 4))
        try:
            self.plate = wm._load_plate_info(sw) if meta.get("plate") else None
        except Exception:  # noqa: BLE001 - a task without a readable plate still gets its row
            self.plate = None
        try:
            cam = wm._load_camera_from_entry(meta.get("camera"), meta.get("plate"))
            if cam is not None:
                self.frames = [int(f) for f in (self.plate.frames if self.plate is not None else cam.frames)]
                self.cam = cam.resample_to(self.frames)
                self.c2w = np.asarray(self.cam.cam_to_world, dtype=np.float64)
        except Exception:  # noqa: BLE001
            self.cam = None
        self.frame_index = {int(f): i for i, f in enumerate(self.frames)}
        self.cut_from = {
            int(p["id"]): int(p["split_from"]) for p in ((meta.get("detection") or {}).get("people") or []) if p.get("split_from") is not None
        }
        # frames whose outline took in a piece far from the person (repaired before the network saw them)
        self.repaired: dict[int, str] = {}
        try:
            det_now, _run = wm._detection_on_disk(sw)
            for p in (det_now or {}).get("people") or []:
                if p.get("repaired"):
                    self.repaired[int(p["id"])] = " ".join(str(f) for f in sorted(int(k) for k in p["repaired"]))
        except Exception:  # noqa: BLE001 - the column stays empty
            self.repaired = {}
        self._ground: _Ground | None = None
        self._ground_tried = False

    @property
    def ground(self) -> _Ground | None:
        if not self._ground_tried:
            self._ground_tried = True
            scan_npz = self.sw.root / "scene_scan" / "scan_cache.npz"
            if self.meta.get("scene_scan") and scan_npz.is_file():
                g = _Ground(scan_npz)
                self._ground = g if g.ok else None
        return self._ground


def _camera_cells(task: _Task) -> dict:
    cam = task.cam
    if cam is None:
        return {}
    entry = task.meta.get("camera") or {}
    summary = entry.get("summary") or {}
    c2w = task.c2w
    width = int(task.plate.width if task.plate is not None else cam.width)
    height = int(task.plate.height if task.plate is not None else cam.height)
    k_map = cam.intrinsics_per_frame(width, height)
    k = np.stack([np.asarray(k_map[f], dtype=np.float64) for f in task.frames]) if task.frames else np.zeros((0, 3, 3))
    cells = {
        "摄影机格式": Path(str(entry.get("file_path") or "")).suffix.lower().lstrip("."),
        "摄影机单位": str(entry.get("unit") or ""),
        "摄影机移动距离_厘米": _round(summary.get("travel_cm"), 2),
        "是否变焦": "是" if summary.get("is_zoom") else "否",
        "像素宽高比": _round(getattr(cam, "pixel_aspect", 1.0), 5),
        "摄影机帧号": _ints(task.frames),
        "摄影机位置X_厘米": _nums(c2w[:, 0, 3], "%.3f"),
        "摄影机位置Y_厘米": _nums(c2w[:, 1, 3], "%.3f"),
        "摄影机位置Z_厘米": _nums(c2w[:, 2, 3], "%.3f"),
        "焦距_毫米": _nums(cam.focal_mm, "%.4f"),
        "胶片口宽_毫米": _nums(cam.h_aperture_mm, "%.4f"),
        "胶片口高_毫米": _nums(cam.v_aperture_mm, "%.4f"),
        "内参FX_像素": _nums(k[:, 0, 0], "%.3f"),
        "内参FY_像素": _nums(k[:, 1, 1], "%.3f"),
        "内参CX_像素": _nums(k[:, 0, 2], "%.3f"),
        "内参CY_像素": _nums(k[:, 1, 2], "%.3f"),
    }
    for i in range(3):
        for j in range(3):
            cells[f"摄影机旋转R{i}{j}"] = _nums(c2w[:, i, j], "%.7f")
    return cells


def _scan_cells(task: _Task, people_xz_cm: np.ndarray, people_ground_cm: np.ndarray) -> dict:
    entry = task.meta.get("scene_scan")
    if not entry:
        return {"扫描": "无"}
    summary = entry.get("summary") or {}
    span = list(summary.get("span_cm") or [])
    cells = {
        "扫描": "已导入",
        "扫描状态": "对齐后" if task.wm._scan_align_on(task.meta) else "对齐前",
        "扫描格式": str(summary.get("format") or ""),
        "扫描单位": str(entry.get("unit") or ""),
        "扫描面数": summary.get("raw_face_count", ""),
        "扫描网格数": summary.get("mesh_count", ""),
    }
    for axis, value in zip("XYZ", span[:3]):
        cells[f"扫描尺寸{axis}_厘米"] = _round(value, 1)
    ground = task.ground
    if ground is None:
        return cells
    # the grid: where the people are (plus a margin), or the whole scan when nobody was solved
    if len(people_xz_cm):
        lo = people_xz_cm.min(axis=0) - GRID_MARGIN_CM
        hi = people_xz_cm.max(axis=0) + GRID_MARGIN_CM
        lo = np.maximum(lo, ground.lo_cm[[0, 2]])
        hi = np.minimum(hi, ground.hi_cm[[0, 2]])
        if np.any(hi <= lo):  # the people are not over the scan at all: describe the scan itself
            lo, hi = ground.lo_cm[[0, 2]], ground.hi_cm[[0, 2]]
    else:
        lo, hi = ground.lo_cm[[0, 2]], ground.hi_cm[[0, 2]]
    cell = max(GRID_CELL_CM, float(np.max(hi - lo)) / (GRID_MAX_SIDE - 1))
    xs = np.arange(lo[0], hi[0] + 0.5 * cell, cell)[:GRID_MAX_SIDE]
    zs = np.arange(lo[1], hi[1] + 0.5 * cell, cell)[:GRID_MAX_SIDE]
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    finite = people_ground_cm[np.isfinite(people_ground_cm)] if len(people_ground_cm) else np.zeros(0)
    near = float(np.median(finite)) if len(finite) else float(ground.lo_cm[1])
    gh = ground.heights_cm(gx.ravel(), gz.ravel(), near)
    got = gh[np.isfinite(gh)]
    cells.update(
        {
            "地面采样间距_厘米": _round(cell, 1),
            "地面采样点数": int(len(gh)),
            "地面采样X_厘米": _nums(gx.ravel(), "%.1f"),
            "地面采样Z_厘米": _nums(gz.ravel(), "%.1f"),
            "地面采样高度_厘米": _nums(gh, "%.2f"),
            "地面高度中位_厘米": _round(np.median(got), 2) if len(got) else "",
            "地面高度最低_厘米": _round(got.min(), 2) if len(got) else "",
            "地面高度最高_厘米": _round(got.max(), 2) if len(got) else "",
        }
    )
    return cells


def _person_row(task: _Task, base: dict, entry: dict, kind: str, with_net: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    """-> (row, where the person is on the floor plan [n, 2] cm, the scan's ground height there [n] cm)."""
    pid = int(entry["id"])
    solve_dir = task.sw.root / "solve_raw"
    qc = dict(entry.get("qc") or {})
    manual_ids = {int(t.get("id", -1)) for t in (task.meta.get("manual_people") or [])}
    row = {
        **base,
        "记录类型": "人物",
        "人物编号": pid,
        "手动框": "是" if pid in manual_ids else "否",
        # the track this person was cut out of (it jumped to another person): the tracker's record knows it for good
        "拆分自": task.cut_from.get(pid, ""),
        "检测框修正帧": task.repaired.get(pid, ""),
        "本次计算": kind,
        "扫描状态": ("对齐后" if task.wm._scan_align_on(task.meta) else "对齐前") if task.meta.get("scene_scan") else "",
        "画面贴合_像素": qc.get("fit_px", ""),
        "画面贴合90_像素": qc.get("fit_px_p90", ""),
        "触地偏移_厘米": qc.get("foot_hold_cm", ""),
        "触地帧占比": qc.get("planted_pct", ""),
        "体型比例": qc.get("body_scale", ""),
        # as the interface words it: the body's size is the network's own, or was set by the scan's ground
        "尺度来源": {"network": "模型", "ground": "扫描"}.get(str(qc.get("scale_source", "")).split(" ")[0], str(qc.get("scale_source", ""))),
        "尺度权重": qc.get("scale_weight", ""),
        "已解算帧数": qc.get("solved_frames", ""),
        "隐藏帧数": qc.get("hidden_frames", ""),
    }
    xz, ground_h = np.zeros((0, 2)), np.zeros(0)
    with np.load(solve_dir / "cam_space" / f"person_{pid:02d}.npz", allow_pickle=False) as z:
        frames = np.asarray(z["frames"], dtype=np.int64)
        visible = np.asarray(z["visible"]).astype(bool)
        solved = np.isin(frames, np.asarray(z["solved_frames"], dtype=np.int64))
        row.update({"帧号": _ints(frames), "可见": _ints(visible), "已解算": _ints(solved)})
        if "rest_vertices" in z.files:
            rest_y = np.asarray(z["rest_vertices"], dtype=np.float64)[:, 1]
            row["模型身高_厘米"] = _round(rest_y.max() - rest_y.min(), 2)
        if "reproj_px" in z.files:
            row["逐帧画面误差_像素"] = _nums(z["reproj_px"], "%.2f")
        if "foot_contact" in z.files:
            fc = np.asarray(z["foot_contact"])
            row.update({"触地左": _ints(fc[:, 0]), "触地右": _ints(fc[:, 1])})
        if "bbox" in z.files:
            bb = np.asarray(z["bbox"], dtype=np.float64)
            for col, name in enumerate(("框X0_像素", "框Y0_像素", "框X1_像素", "框Y1_像素")):
                row[name] = _nums(bb[:, col], "%.1f")
        for key, name, count in (("shape_params", "体型参数", N_SHAPE), ("scale_params", "比例参数", N_SCALE)):
            if key in z.files:
                vals = np.asarray(z[key], dtype=np.float64).ravel()
                for i in range(min(count, len(vals))):
                    row[f"{name}_{i:02d}"] = _round(vals[i], 5)
        if "keypoints_cam_cm" in z.files and task.cam is not None:
            kp_cam = np.asarray(z["keypoints_cam_cm"], dtype=np.float64)
            anchors = np.asarray(z["anchor_frames"], dtype=np.int64) if "anchor_frames" in z.files else frames
            # A visible frame is placed by its own camera.  A hidden frame holds the body of its anchor frame and
            # stays where that frame put it in the WORLD (as in the view and the export).
            placed_by = np.where(visible, frames, anchors)
            cam_i = np.array([task.frame_index.get(int(f), -1) for f in placed_by])
            own_i = np.array([task.frame_index.get(int(f), -1) for f in frames])
            kp_w = np.full_like(kp_cam, np.nan)
            ok = cam_i >= 0
            if ok.any():
                rot, pos = task.c2w[cam_i[ok], :3, :3], task.c2w[cam_i[ok], :3, 3]
                kp_w[ok] = np.einsum("tij,tkj->tki", rot, kp_cam[ok]) + pos[:, None, :]
            pelvis = 0.5 * (kp_w[:, 9] + kp_w[:, 10])
            for axis, col in zip("XYZ", range(3)):
                row[f"骨盆{axis}_厘米"] = _nums(pelvis[:, col], "%.2f")
                for index, name in BODY_POINTS:
                    row[f"{name}{axis}_厘米"] = _nums(kp_w[:, index, col], "%.2f")
            dist = np.full(len(frames), np.nan)
            own_ok = own_i >= 0
            dist[own_ok] = np.linalg.norm(pelvis[own_ok] - task.c2w[own_i[own_ok], :3, 3], axis=1)
            row["到摄影机距离_厘米"] = _nums(dist, "%.2f")
            seen = visible & np.isfinite(pelvis).all(axis=1)
            xz = pelvis[seen][:, [0, 2]]
            if task.ground is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # a frame without a camera has no points at all
                    lowest = np.nanmin(kp_w[:, :, 1], axis=1)  # the body's lowest point: the level the person stands on
                under = task.ground.heights_cm(pelvis[:, 0], pelvis[:, 2], lowest)
                row["所在位置地面高度_厘米"] = _nums(under, "%.2f")
                ground_h = under[seen]
    if with_net:
        raw_file = solve_dir / "raw_net" / f"person_{pid:02d}.npz"
        if raw_file.is_file():
            with np.load(raw_file, allow_pickle=False) as r:
                row["网络_帧号"] = _ints(r["frames"])
                if "det_scores" in r.files:
                    row["网络_检测分数"] = _nums(r["det_scores"], "%.4f")
                for key, names, fmt in (
                    ("pred_cam_t", ("网络_位置X_米", "网络_位置Y_米", "网络_位置Z_米"), "%.5f"),
                    ("global_rot", ("网络_整体旋转X", "网络_整体旋转Y", "网络_整体旋转Z"), "%.5f"),
                    ("mask_box", ("网络_轮廓框X0", "网络_轮廓框Y0", "网络_轮廓框X1", "网络_轮廓框Y1"), "%.1f"),
                    ("bbox", ("网络_框X0", "网络_框Y0", "网络_框X1", "网络_框Y1"), "%.1f"),
                ):
                    if key in r.files:
                        arr = np.asarray(r[key], dtype=np.float64)
                        for col, name in enumerate(names):
                            row[name] = _nums(arr[:, col], fmt)
                if "has_mask" in r.files:
                    row["网络_有轮廓"] = _ints(r["has_mask"])
                if "mask_area" in r.files:
                    row["网络_轮廓面积_像素"] = _nums(r["mask_area"], "%.0f")
                for key, name, count, width, fmt in (
                    ("shape_params", "网络_体型", N_SHAPE, 2, "%.5f"),
                    ("scale_params", "网络_比例", N_SCALE, 2, "%.5f"),
                    ("body_pose_params", "网络_姿态", N_BODY_POSE, 3, "%.5f"),
                ):
                    if key in r.files:
                        arr = np.asarray(r[key], dtype=np.float64)
                        for i in range(min(count, arr.shape[1])):
                            row[f"{name}_{i:0{width}d}"] = _nums(arr[:, i], fmt)
                if "pred_keypoints_2d" in r.files:
                    k2 = np.asarray(r["pred_keypoints_2d"], dtype=np.float64)
                    for i in range(min(N_KEYPOINTS, k2.shape[1])):
                        row[f"网络_关键点X_{i:02d}"] = _nums(k2[:, i, 0], "%.2f")
                        row[f"网络_关键点Y_{i:02d}"] = _nums(k2[:, i, 1], "%.2f")
                if "kp_in_mask" in r.files:
                    km = np.asarray(r["kp_in_mask"])
                    for i in range(min(N_KEYPOINTS, km.shape[1])):
                        row[f"网络_关键点在轮廓内_{i:02d}"] = _ints(km[:, i])
    return row, xz, ground_h


def _object_rows(task: _Task, base: dict) -> list[dict]:
    """Solved objects as the view and the export show them (world space)."""
    wm, sw, meta = task.wm, task.sw, task.meta
    raw_objs = wm._load_cam_space_objects(sw, meta.get("active_object_ids") or None)
    if not raw_objs or task.cam is None:
        return []
    width = int(task.plate.width if task.plate is not None else task.cam.width)
    height = int(task.plate.height if task.plate is not None else task.cam.height)
    chars = wm._load_cam_space_chars(sw, meta.get("active_person_ids") or None)
    frames = wm._shot_frames(meta, task.cam, chars, raw_objs)
    scan_cache = wm._load_cached_scene_scan(sw, meta) if wm._scan_align_on(meta) else None
    _chars, objs = wm._align_scene(task.cam.resample_to(frames), width, height, frames, chars, raw_objs, scan_cache)
    rows = []
    for obj in objs:
        tf = np.asarray(obj.world_transforms, dtype=np.float64)
        pts = np.asarray(obj.rest_points, dtype=np.float64)
        size = (pts.max(axis=0) - pts.min(axis=0)) if len(pts) else np.zeros(3)
        row = {
            **base,
            "记录类型": "物体",
            "扫描状态": ("对齐后" if wm._scan_align_on(meta) else "对齐前") if meta.get("scene_scan") else "",
            "物体编号": int(obj.object_id),
            "物体名称": str(obj.name),
            "物体帧号": _ints(obj.frames),
            "物体可见": _ints(np.asarray(obj.visible)) if getattr(obj, "visible", None) is not None else "",
            "物体位置X_厘米": _nums(tf[:, 0, 3], "%.2f"),
            "物体位置Y_厘米": _nums(tf[:, 1, 3], "%.2f"),
            "物体位置Z_厘米": _nums(tf[:, 2, 3], "%.2f"),
        }
        for axis, value in zip("XYZ", size):
            row[f"物体尺寸{axis}_厘米"] = _round(value, 2)
        for i in range(3):
            for j in range(3):
                row[f"物体旋转R{i}{j}"] = _nums(tf[:, i, j], "%.6f")
        rows.append(row)
    return rows


def _task_row(task: _Task, base: dict, qjob, outcome: str, error: str, info: dict, people_xz: np.ndarray, people_ground: np.ndarray) -> dict:
    meta, plate = task.meta, task.plate
    now = time.time()
    started = getattr(qjob, "started_ts", None)
    created = getattr(qjob, "created_ts", None)
    times = dict(info.get("times") or {})
    counts = dict(info.get("counts") or {})
    grade = meta.get("grade") or {}
    sel = meta.get("selection") or {}
    sp = meta.get("solve_params") or {}
    row = {
        **base,
        "记录类型": "任务",
        "作业结果": outcome,
        "错误信息": str(error or "")[:500],
        "排队等待_秒": _round((started - created) if (started and created) else np.nan, 1),
        "耗时_秒": _round((now - started) if started else np.nan, 1),
        "人物检测耗时_秒": _round(times.get("detect", np.nan), 1),
        "逐帧估计耗时_秒": _round(times.get("net", np.nan), 1),
        "整段解算耗时_秒": _round(times.get("post", np.nan), 1),
        "显卡编号": "" if getattr(qjob, "gpu_index", None) is None else int(qjob.gpu_index),
        "曝光": grade.get("exposure", ""), "亮度": grade.get("brightness", ""), "对比": grade.get("contrast", ""),
        "饱和": grade.get("saturation", ""), "Gamma": grade.get("gamma", ""),
        "检测阈值": (meta.get("detect_params") or {}).get("threshold", ""),
        "检测到人数": len((meta.get("detection") or {}).get("people") or []),
        "手动框人数": len(meta.get("manual_people") or []),
        "选择方式": str(sel.get("mode") or ""),
        "平滑": sp.get("smoothing", ""),
        "手部细节": "开" if sp.get("hand_refine", True) else "关",
        "已解算人数": len(meta.get("solved_people") or []),
        "本次新增人数": counts.get("net", ""),
        "本次重新解算人数": counts.get("post", ""),
        "本次跳过人数": counts.get("skipped", ""),
        "已解算物体数": len(meta.get("solved_objects") or []),
        "警告": " | ".join(str(w) for w in (meta.get("solve_warnings") or []))[:1000],
    }
    try:
        slot = task.wm.scheduler._slots.get(int(qjob.gpu_index)) if getattr(qjob, "gpu_index", None) is not None else None
        if slot is not None:
            row["显卡型号"] = str(getattr(slot.info, "name", ""))
    except Exception:  # noqa: BLE001
        pass
    if plate is not None:
        row.update(
            {
                "画面宽_像素": int(plate.width), "画面高_像素": int(plate.height), "帧数": len(plate.frames),
                "起始帧": int(plate.frames[0]) if plate.frames else "", "结束帧": int(plate.frames[-1]) if plate.frames else "",
                "帧率": _round(plate.fps, 4), "画面格式": str(plate.file_ext).lstrip("."), "位深": str(plate.bit_depth),
                "色彩空间": str(plate.resolved_colorspace), "显示变换": str(plate.ocio_view),
            }
        )
    row.update(_camera_cells(task))
    row.update(_scan_cells(task, people_xz, people_ground))
    return row


def record_job(wm, sw, qjob, job_type: str, outcome: str, error: str = "") -> Path | None:
    """Write what the job that just ended leaves behind into the table of the day.  Never raises."""
    collector: DataCollector | None = getattr(wm, "data_collector", None)
    if collector is None:
        return None
    try:
        from .. import __version__
        from ..solver.trajectory import SOLVER_REVISION

        now = datetime.now()
        meta = sw.read_meta()
        info = getattr(qjob, "_record", None) or {}
        base = {
            "表格版本": SCHEMA_VERSION,
            "日期": now.strftime("%Y-%m-%d"),
            "时间": now.strftime("%H:%M:%S"),
            "软件版本": __version__,
            "解算规则版本": SOLVER_REVISION,
            "任务编号": str(meta.get("id") or sw.shot_id),
            "任务名称": str(meta.get("name") or ""),
            "作业编号": str(getattr(qjob, "id", "")),
            "作业类型": job_type,
        }
        task = _Task(wm, sw, meta)
        detail_rows: list[dict] = []
        xz_all: list[np.ndarray] = []
        ground_all: list[np.ndarray] = []
        if job_type == "人物解算":
            listed = {int(sp["id"]): sp for sp in (meta.get("solved_people") or [])}
            valid = {str(k) for k in ((meta.get("cache_signatures") or {}).get("people_post") or {})}
            for pid, kind in sorted((info.get("people") or {}).items()):
                entry = listed.get(int(pid))
                if entry is None or str(int(pid)) not in valid:
                    continue  # not in the task (any more), or without a finished result
                try:
                    row, xz, ground_h = _person_row(task, base, entry, str(kind), with_net=(str(kind) == "新增"))
                    detail_rows.append(row)
                    xz_all.append(xz)
                    ground_all.append(ground_h)
                except Exception:  # noqa: BLE001 - one person that cannot be read must not lose the others
                    collector.log_error(f"person {pid} of {sw.shot_id}")
        elif job_type == "物体解算" and outcome == "完成":
            try:
                detail_rows.extend(_object_rows(task, base))
            except Exception:  # noqa: BLE001
                collector.log_error(f"objects of {sw.shot_id}")
        people_xz = np.concatenate(xz_all) if xz_all else np.zeros((0, 2))
        people_ground = np.concatenate(ground_all) if ground_all else np.zeros(0)
        try:
            task_row = _task_row(task, base, qjob, outcome, error, info, people_xz, people_ground)
        except Exception:  # noqa: BLE001 - the job's row is written even if its details cannot be read
            collector.log_error(f"task row of {sw.shot_id}")
            task_row = {**base, "记录类型": "任务", "作业结果": outcome, "错误信息": str(error or "")[:500]}
        return collector.append([task_row] + detail_rows, now)
    except Exception:  # noqa: BLE001
        collector.log_error(f"{job_type} of {getattr(sw, 'shot_id', '?')}")
        return None


def default_directory(workspaces_dir: Path) -> Path:
    """Next to the folder of the tasks (for an installation: `<ShotSpace>/data_collect`).  It is not inside it:
    tasks are removed after the retention period, the tables are kept."""
    env = str(os.environ.get("SHOTSPACE_DATA_COLLECT_DIR") or "").strip()
    return Path(env).expanduser().resolve() if env else Path(workspaces_dir).resolve().parent / "data_collect"
