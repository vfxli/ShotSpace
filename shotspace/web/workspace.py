"""Disk-persisted Multi-User Shot Task System for ShotSpace (`WorkspaceManager`).

Exact 5-Step Camera-First Production Flow:
  Step 1. Input Image Sequence (`.exr` default `ACEScg -> OCIO sRGB`, `.dpx`, `.tif`, `.png`, `.jpg`, `.zip`...) +
          Real-time Color Grading Preview (`Exposure EV`, `Brightness`, `Gamma`).
  Step 2. Import Tracked Camera (mandatory, `.abc` / `.fbx` / `.usd`, default `m`) +
          Optional Static Scene Scan (omit when ground is not visible in frame).
  Step 3. Detect & Select Person(s) (`ViTDet-H` dual-threshold + Re-ID) OR Interactive Object Click & Track (`SAM 3.1`).
  Step 4. 3D Solve (`SAM 3D Body` conditioned natively on per-frame camera intrinsics `K(t)` + optional `SAM 3.1`
          silhouette mask guidance, OR `SAM 3D Objects` + 6DoF pose).
  Step 5. Instant Format Download of 4 Pre-Generated Deliverables (`USD 骨架`, `USD 点缓存`, `FBX`, `ABC`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable
import uuid

import numpy as np

from ..config import PROJECT_ROOT, get_paths
from ..core.platform_compat import replace_file
from ..core.camera import CameraTrack
from ..core.camera_space import (
    AlignmentReport,
    align_standard_character_to_camera,
    place_world_solved_character,
    align_standard_object_to_camera,
    calibrate_scene_with_scan,
    camera_spread_cm,
    standard_camera_focal_px,
)
from ..core.names import identifier
from ..core.rigid_object import RigidObjectTrack
from ..core.skeleton import SkinnedCharacter
from ..core.units import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, FILMBACK_MM
from ..io import (
    GradeParams,
    OcioSettings,
    PlateInfo,
    export_scene,
    ingest_sequence_dir,
    inspect_3d_file,
    load_camera_track,
    load_scene_scan,
    prepare_solver_srgb_frames,
)
from ..server_config import ConfigStore
from ..solver import (
    JobCancelledError,
    MultiGpuScheduler,
    QueueJob,
    SelectionSpec,
    SolverManager,
    select_people,
)
from ..solver.select_people import (
    build_manual_detected_people,
    normalize_manual_people,
    parse_person_ids,
)
from ..solver.box_repair import with_repairs
from ..solver.trajectory import SCAN_SOLVER_REVISION, SOLVER_REVISION
from .data_collect import DataCollector, default_directory, record_job
from .machines import MachineNames, decide_owner, machine_key, machine_label, normalise_ip, read_owner_records, same_machine
from .view_encode import build_viewport_bundle


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _parse_iso_ts(s: str) -> float:
    if not s:
        return time.time()
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except Exception:
        return time.time()


# The tag the people tracker writes into `boxes.json` (`people_tracker.ENGINE_TAG`; the worker module cannot be
# imported here, a test keeps the two equal).  Boxes without it come from the v0.1.x detector: they have no
# per-person outlines the present solver needs, so they are not used for solving - people are looked for anew.
TRACKER_ENGINE = "sam31_native_people_v1"
# People are estimated and solved this many at a time (the number the body network takes in one call,
# `sam3d_worker.PEOPLE_BATCH`; a test keeps the two equal).  After every such batch its people are part of the task:
# they show in the view, and a cancelled job keeps them.
PEOPLE_PER_PASS = 16


def _human_duration(seconds: float) -> str:
    minutes = max(1, int(round(seconds / 60.0)))
    return f"{minutes} 分钟" if minutes < 90 else f"{minutes // 60} 小时 {minutes % 60} 分"


def _json_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _standard_origin_camera(frames: list[int], plate_meta: dict | None) -> CameraTrack:
    """Build SAM 3D Body's canonical origin standard camera (`I_4` at `(0,0,0)`, `-Z` forward, `+Y` up,
    `f_px = hypot(W, H)`)."""
    f_list = frames or [1001]
    n = len(f_list)
    w = int((plate_meta or {}).get("width") or DEFAULT_WIDTH)
    h = int((plate_meta or {}).get("height") or DEFAULT_HEIGHT)
    fps = float((plate_meta or {}).get("fps") or DEFAULT_FPS)
    hap = FILMBACK_MM
    vap = hap * h / max(w, 1)
    f_mm = hap * standard_camera_focal_px(w, h) / max(w, 1)
    return CameraTrack(
        name="origin_cam",
        path="/origin_cam",
        frames=tuple(f_list),
        cam_to_world=np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0),
        focal_mm=np.full(n, f_mm, dtype=np.float64),
        h_aperture_mm=np.full(n, hap, dtype=np.float64),
        v_aperture_mm=np.full(n, vap, dtype=np.float64),
        center_mm=np.zeros((n, 2), dtype=np.float64),
        width=w,
        height=h,
        pixel_aspect=1.0,
        overscan=(0, 0, 0, 0),
        fps=fps,
        source_file="",
        own_frames=tuple(f_list),
    )




class ShotWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta_path = root / "shot.json"
        self.lock = threading.RLock()
        self.export_lock = threading.RLock()
        self.subscribers: list[Callable[[dict], None]] = []

    @property
    def shot_id(self) -> str:
        return self.root.name

    def read_meta(self) -> dict:
        with self.lock:
            if not self.meta_path.is_file():
                raise FileNotFoundError(f"Shot metadata missing: {self.meta_path}")
            return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_meta(self, data: dict) -> dict:
        with self.lock:
            data["updated_at"] = _now_iso()
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / ".shot.json.tmp"
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            replace_file(tmp, self.meta_path)
            return data

    def update_meta(self, **patch) -> dict:
        with self.lock:
            cur = self.read_meta()
            cur.update(patch)
            return self.write_meta(cur)

    def broadcast(self, event: dict) -> None:
        for cb in list(self.subscribers):
            try:
                cb(event)
            except Exception:
                pass


def default_shot_meta(shot_id: str, name: str, owner: dict | None = None) -> dict:
    now = _now_iso()
    return {
        "id": shot_id,
        "name": name,
        # the computer the task belongs to (v0.2.12, `machines.py`): {"ip", "name", "how", "at"}, or {"public": True}
        "owner": dict(owner) if owner else {"public": True, "how": "无来源", "at": now},
        "status": "idle",
        "active_job_id": None,
        "stage_message": "请在第 1 步上传序列图",
        "progress": {"done": 0, "total": 0, "stage": ""},
        "error": None,
        "created_at": now,
        "updated_at": now,
        "solved_at": None,
        "plate": None,
        "ocio": {"input_colorspace": "auto", "ocio_view": "aces_sdr"},
        "grade": {"exposure": 0.0, "brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0},
        "detect_params": {"threshold": 0.8},
        "manual_people": [],
        "detection": None,
        "selection": {"mode": "largest", "count": 2, "ids": "1", "picks": [], "least_frames": 3},
        "solve_params": {"hand_refine": True, "use_mask": False, "lock_shape": True, "smoothing": 0.5},
        "solved_state": None,
        "cache_signatures": {
            "detect": "",          # the picture the people were looked for in
            "detect_run": "",      # id of the tracker run whose numbering is in use
            "person_masks_sam3": "",
            "people_net": {},      # per solved person: what the cached network outputs were computed from
            "people_post": {},     # per solved person: what the whole-shot solve was computed from
            "solve_det": "",       # tracker run the solved people were numbered by
            "obj_track": "",
            "obj_solve": "",
        },
        "work_mode": "human",
        "solved_people": [],
        "active_person_ids": [],
        "object_prompts": [{"id": 1, "name": "object_01", "points": []}],
        "object_tracking": None,
        "object_solve_params": {"smoothing": 0.5},
        "solved_objects": [],
        "active_object_ids": [],
        "camera": None,
        "scene_scan": None,
        "scan_align": False,       # 对齐后: the scan is in use (set by the button of step 4, never by an import)
        "scan_align_sig": "",      # the scan the results of that state were last brought up to date with
        "scan_mode_results": {},   # results kept for the other state of that switch (see `_switch_scan_state`)
        "alignment_report": None,
        "has_view": False,
        "exports": [],
        "export_person_ids": [],
        "export_cache": {},
    }


SCAN_SOLVER_NPZ = "scan_solver.npz"
# The one top-level group of every delivered file: the people, the objects and the camera hang under it.
EXPORT_TOP_GROUP = "ShotSpace"


def _write_scan_cache(cache_npz: Path, scan) -> None:
    """Scan cache of a task: the reduced meshes (viewport, legacy calibration) in `scan_cache.npz` and, for scans
    that had to be reduced, the full-resolution mesh for the people solver's ground contact next to it."""
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        vertices_cm=scan.vertices_cm,
        faces=scan.faces,
        display_vertices_cm=scan.display_vertices_cm,
        display_faces=scan.display_faces,
    )
    solver_npz = cache_npz.with_name(SCAN_SOLVER_NPZ)
    if getattr(scan, "solver_vertices_cm", None) is not None and getattr(scan, "solver_faces", None) is not None:
        tmp = solver_npz.with_name(".scan_solver.tmp.npz")
        np.savez(tmp, vertices_cm=scan.solver_vertices_cm, faces=scan.solver_faces)
        tmp.replace(solver_npz)
    else:
        solver_npz.unlink(missing_ok=True)


def manual_masks_from_tracker(
    manual_people: list[dict], tracked_people: list[dict], tracker_masks_npz: Path, min_iou: float = 0.5
) -> tuple[dict[str, np.ndarray], dict[int, tuple[int, int]]]:
    """Outlines for manually boxed people, borrowed from the tracking stage.

    On every frame of a manual person the tracked person whose box agrees best with the artist's box (IoU of at
    least `min_iou`) lends its outline for that frame, whatever identity the tracker gave it: the artist's boxes
    are the identity.  Returns (npz entries keyed like the tracker's `p{id}_f{frame}_{bits,shape,rect}`,
    {manual id: (frames with an outline, frames with a box)})."""
    entries: dict[str, np.ndarray] = {}
    cover: dict[int, tuple[int, int]] = {}
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tp in tracked_people:
        if bool(tp.get("is_manual")):
            continue
        for fk, bx in (tp.get("boxes") or {}).items():
            by_frame.setdefault(int(fk), []).append((int(tp["id"]), np.asarray(bx, dtype=np.float64)))
    have_masks = Path(tracker_masks_npz).is_file()
    tm = np.load(tracker_masks_npz, allow_pickle=False) if have_masks else None
    try:
        for mp in manual_people:
            mid = int(mp["id"])
            boxes = mp.get("boxes") or {}
            got = 0
            for fk, bx in boxes.items():
                f = int(fk)
                cands = by_frame.get(f)
                if tm is None or not cands:
                    continue
                b = np.asarray(bx, dtype=np.float64)
                cb = np.stack([c[1] for c in cands])
                iw = np.clip(np.minimum(b[2], cb[:, 2]) - np.maximum(b[0], cb[:, 0]), 0.0, None)
                ih = np.clip(np.minimum(b[3], cb[:, 3]) - np.maximum(b[1], cb[:, 1]), 0.0, None)
                inter = iw * ih
                union = (b[2] - b[0]) * (b[3] - b[1]) + (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1]) - inter
                iou = inter / np.maximum(union, 1e-9)
                k = int(np.argmax(iou))
                src = f"p{cands[k][0]}_f{f}"
                if iou[k] < min_iou or any(f"{src}_{suffix}" not in tm.files for suffix in ("bits", "shape", "rect")):
                    continue
                for suffix in ("bits", "shape", "rect"):
                    entries[f"p{mid}_f{f}_{suffix}"] = tm[f"{src}_{suffix}"]
                got += 1
            cover[mid] = (got, len(boxes))
    finally:
        if tm is not None:
            tm.close()
    return entries, cover


class WorkspaceManager:
    def __init__(
        self,
        workspaces_dir: Path | None = None,
        config_store: ConfigStore | None = None,
        scheduler: MultiGpuScheduler | None = None,
    ) -> None:
        paths = get_paths()
        self.config_store = config_store or ConfigStore()
        self._explicit_workspaces_dir = workspaces_dir.resolve() if workspaces_dir is not None else None
        if self._explicit_workspaces_dir is not None:
            resolved_base = self._explicit_workspaces_dir
        else:
            cfg_cache = (self.config_store.load().cache_dir or "").strip()
            resolved_base = Path(cfg_cache).expanduser().resolve() if cfg_cache else paths.workspaces.resolve()
        self.base_dir = resolved_base
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # The daily tables of the data collection are kept next to the installation's own task folder - not inside
        # it (tasks are removed after the retention period) and not on a cache path an admin may switch to.
        self.data_collector = DataCollector(
            default_directory(self._explicit_workspaces_dir if self._explicit_workspaces_dir is not None else paths.workspaces)
        )
        self.scheduler = scheduler or MultiGpuScheduler(
            self.config_store,
            history_file=self.base_dir / "_queue_history.json",
        )
        self._shots: dict[str, ShotWorkspace] = {}
        self._lock = threading.RLock()
        # whose task is whose (v0.2.12): names of the computers, and where the records of the old tasks are
        self.machines = MachineNames(seen_file=self.base_dir / "_machines_seen.json")
        # stretched frames of the tracked people, per task: ((detection file stamp), {person: (boxes, regions)})
        self._repair_cache: dict[str, tuple[tuple, dict]] = {}
        # the owner of each task, per task: ((task record stamp), owner) - every call on a task asks for it, and the
        # record of a crowd task is megabytes of JSON
        self._owner_cache: dict[str, tuple[tuple, dict]] = {}
        self._log_files_override: list[Path] | None = None
        self._discover_existing()
        self.cleanup_expired()
        self._reaper = threading.Thread(target=self._retention_loop, name="shotspace-retention", daemon=True)
        self._reaper.start()

    def switch_cache_dir(self, raw_cache_dir: str) -> dict:
        """Persist and live-switch the active workspace/cache storage directory (e.g. to a studio NAS mount)."""
        if self.scheduler.has_active_or_queued_jobs():
            raise ValueError("当前有正在运行或排队的计算任务，请等待任务结束或取消后再切换缓存路径")
        with self._lock:
            cfg = self.config_store.set_cache_dir(raw_cache_dir)
            default_ws = get_paths().workspaces.resolve()
            new_base = Path(cfg.cache_dir).expanduser().resolve() if cfg.cache_dir else default_ws
            new_base.mkdir(parents=True, exist_ok=True)
            self._explicit_workspaces_dir = None
            self.base_dir = new_base
            self._shots.clear()
            self.scheduler.switch_history_file(self.base_dir / "_queue_history.json")
            self.machines.seen_file = self.base_dir / "_machines_seen.json"
            self._discover_existing()
            return self.storage_stats()

    def _retention_loop(self) -> None:
        while True:
            time.sleep(3600)
            try:
                self.cleanup_expired()
            except Exception:
                pass

    def _discover_existing(self) -> None:
        with self._lock:
            live_names = set()
            ownerless: list[ShotWorkspace] = []
            for child in sorted(self.base_dir.iterdir()):
                if child.is_dir() and not child.name.startswith("_") and (child / "shot.json").is_file():
                    live_names.add(child.name)
                    if child.name not in self._shots:
                        sw = ShotWorkspace(child)
                        try:
                            m = sw.read_meta()
                            changed = False
                            if "camera" not in m and m.get("target_camera"):
                                m["camera"] = m.pop("target_camera")
                                m.pop("source_camera", None)
                                m.pop("reuse_shot_id", None)
                                changed = True
                            if m.get("status") in ("queued", "running") and self.scheduler.job_for_shot(sw.shot_id) is None:
                                m["status"] = "ready" if m.get("has_view") else "idle"
                                m["active_job_id"] = None
                                changed = True
                            if m.get("solved_people") and not m.get("active_person_ids"):
                                m["active_person_ids"] = [int(p["id"]) for p in m["solved_people"]]
                                changed = True
                            if m.get("solved_objects") and not m.get("active_object_ids"):
                                m["active_object_ids"] = [int(o["id"]) for o in m["solved_objects"]]
                                changed = True
                            if (m.get("solved_people") or m.get("solved_objects")) and not m.get("solved_at"):
                                m["solved_at"] = m.get("updated_at") or m.get("created_at")
                                changed = True
                            if "work_mode" not in m:
                                m["work_mode"] = "object" if (m.get("solved_objects") and not m.get("solved_people")) else "human"
                                changed = True
                            if "object_prompts" not in m:
                                m["object_prompts"] = [{"id": 1, "name": "object_01", "points": []}]
                                changed = True
                            if "object_solve_params" not in m:
                                m["object_solve_params"] = {"smoothing": 0.5}
                                changed = True
                            if "ocio" not in m:
                                m["ocio"] = {"input_colorspace": "auto", "ocio_view": "aces_sdr"}
                                changed = True
                            if isinstance(m.get("solve_params"), dict) and "use_mask" not in m["solve_params"]:
                                m["solve_params"]["use_mask"] = False
                                changed = True
                            if "root_filter" in m:
                                # the Root filter is gone since v0.2.2: tasks export exactly what was solved
                                m.pop("root_filter", None)
                                changed = True
                            if "prebuilt_exports" in m or "exports_ready" in m:
                                # v0.2.1 and older built all export files ahead of time; the record of those is void
                                m.pop("prebuilt_exports", None)
                                m.pop("exports_ready", None)
                                changed = True
                            if not isinstance(m.get("export_person_ids"), list):
                                m["export_person_ids"] = []
                                changed = True
                            if "manual_people" not in m or not isinstance(m.get("manual_people"), list):
                                m["manual_people"] = []
                                changed = True
                            if changed:
                                sw.write_meta(m)
                            if not isinstance(m.get("owner"), dict):
                                ownerless.append(sw)
                        except Exception:
                            pass
                        self._shots[child.name] = sw
            for dead in list(self._shots):
                if dead not in live_names:
                    self._shots.pop(dead, None)
            if ownerless:
                self._give_old_tasks_an_owner(ownerless)

    # ------------------------------------------------------------------ whose task is whose (v0.2.12)

    def _log_files(self) -> list[Path]:
        """The service log, oldest part first (`shotspace.sh` keeps the previous part as `.old`)."""
        if self._log_files_override is not None:
            return list(self._log_files_override)
        env = str(os.environ.get("SHOTSPACE_LOG_FILE") or "").strip()
        log = Path(env).expanduser() if env else PROJECT_ROOT / "shotspace.log"
        return [log.with_name(log.name + ".old"), log]

    def _give_old_tasks_an_owner(self, tasks: list[ShotWorkspace]) -> None:
        """Tasks made before v0.2.12 have no owner: each gets the computer the service's records name, or becomes a
        public task when that cannot be decided with certainty (`machines.decide_owner`).  Written once per task;
        what was decided is listed in `_task_owners_v0212.txt` next to the tasks."""
        try:
            history = list(self.scheduler._history)
        except Exception:  # noqa: BLE001
            history = []
        records = read_owner_records(self._log_files(), history)
        now = _now_iso()
        lines: list[str] = []
        for sw in tasks:
            try:
                with sw.lock:
                    m = sw.read_meta()
                    if isinstance(m.get("owner"), dict):
                        continue
                    created = m.get("created_at") or ""
                    owner, why = decide_owner(sw.shot_id, _parse_iso_ts(created) if created else None, records)
                    owner.update(how="升级时按记录" if not owner.get("public") else "升级时未能确定", why=why, at=now)
                    sw.update_meta(owner=owner)
                lines.append(f"{sw.shot_id}\t{m.get('name', '')}\t{machine_label(owner)}\t{why}")
            except Exception:  # noqa: BLE001 - one unreadable task must not stop the others
                continue
        if lines:
            try:
                report = self.base_dir / "_task_owners_v0212.txt"
                head = (
                    f"# {now} 升级到按电脑隔离（v0.2.12）时，旧任务的归属（任务编号、名称、归属、依据）\n"
                    f"# 依据的记录：{', '.join(records.log_files) or '无日志'}；队列历史 {sum(len(v) for v in records.jobs.values())} 条作业\n"
                )
                with report.open("a", encoding="utf-8") as fh:
                    fh.write(head + "\n".join(lines) + "\n")
            except OSError:
                pass

    def owner_of(self, meta: dict) -> dict:
        owner = meta.get("owner")
        return dict(owner) if isinstance(owner, dict) else {"public": True}

    def _owner_now(self, sw: ShotWorkspace) -> dict:
        """The task's owner, read again only when its record changed on disk."""
        try:
            st = sw.meta_path.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return self.owner_of(sw.read_meta())
        hit = self._owner_cache.get(sw.shot_id)
        if hit is not None and hit[0] == key:
            return dict(hit[1])
        owner = self.owner_of(sw.read_meta())
        self._owner_cache[sw.shot_id] = (key, owner)
        return dict(owner)

    def may_use(self, shot_id: str, viewer: dict) -> bool:
        """May this computer see and use the task?  Its own tasks and the public ones.  A task that does not exist
        is not refused here (the call answers that it is not found).  A computer that is recognised has its record
        completed: a name learnt since, an address that changed."""
        try:
            sw = self.get_shot(shot_id)
        except KeyError:
            return True
        owner = self._owner_now(sw)
        if owner.get("public"):
            return True
        if not same_machine(owner, viewer):
            return False
        learnt = {}
        if viewer.get("name") and not owner.get("name"):
            learnt["name"] = viewer["name"]
        if viewer.get("ip") and normalise_ip(owner.get("ip") or "") != viewer["ip"]:
            learnt["ip"] = viewer["ip"]
        if learnt:
            with sw.lock:
                cur = self.owner_of(sw.read_meta())
                if same_machine(cur, viewer):
                    sw.update_meta(owner={**cur, **learnt})
        return True

    def claim_shot(self, shot_id: str, viewer: dict) -> dict:
        """A public task goes into this computer's own list."""
        sw = self.get_shot(shot_id)
        with sw.lock:
            owner = self.owner_of(sw.read_meta())
            if not owner.get("public"):
                if same_machine(owner, viewer):
                    return self.enrich_shot_meta(sw.read_meta())
                raise ValueError("这是另一台电脑的任务")
            sw.update_meta(owner={"ip": viewer.get("ip", ""), "name": viewer.get("name", ""), "how": "从公共任务移入", "at": _now_iso()})
        return self.enrich_shot_meta(sw.read_meta())

    def make_public(self, shot_id: str) -> dict:
        """Admin: a task becomes unclassified (every computer sees it; its computer moves it back with one click)."""
        sw = self.get_shot(shot_id)
        with sw.lock:
            before = self.owner_of(sw.read_meta())
            sw.update_meta(owner={"public": True, "how": "管理员设为未分类", "was": machine_label(before), "at": _now_iso()})
        return {"id": shot_id, "owner": machine_label(self.owner_of(sw.read_meta()))}

    def assign_shot(self, shot_id: str, ip: str, name: str = "") -> dict:
        """Admin: the task belongs to the computer with this address (and machine name, when known) from now on."""
        ip, name = normalise_ip(ip), str(name or "").strip()
        if not ip and not name:
            raise ValueError("请填写电脑的 IP 地址或机器名")
        if not name and ip:
            # a computer that has opened the service is known by its name as well
            name = str((self.machines.seen.get(ip) or {}).get("name") or "")
        sw = self.get_shot(shot_id)
        with sw.lock:
            before = self.owner_of(sw.read_meta())
            sw.update_meta(owner={"ip": ip, "name": name, "how": "管理员分配", "was": machine_label(before), "at": _now_iso()})
        return {"id": shot_id, "owner": machine_label(self.owner_of(sw.read_meta()))}

    def admin_delete_shot(self, shot_id: str) -> dict:
        """Admin: a task is deleted, whoever it belongs to (a running job of it is cancelled first)."""
        self.get_shot(shot_id)
        self.delete_shot(shot_id)
        return {"id": shot_id, "deleted": True}

    def machines_overview(self) -> list[dict]:
        """Admin: every computer that owns tasks or has opened the service - how many tasks it has, and when it
        last opened the service (a computer that opened it and has none is how a task given to the wrong
        computer shows).  The unclassified tasks come first, as one entry."""
        groups: dict[str, dict] = {}
        unclassified = 0
        for t in self.all_tasks_with_owner():
            if t["is_public"]:
                unclassified += 1
                continue
            g = groups.setdefault(t["owner_key"], {"key": t["owner_key"], "ip": t["owner_ip"], "name": t["owner_name"], "task_count": 0, "last_seen": None})
            g["task_count"] += 1
            if t["owner_name"] and not g["name"]:
                g["name"] = t["owner_name"]
        for ip, seen in dict(self.machines.seen).items():
            own = {"ip": ip, "name": str(seen.get("name") or "")}
            key = machine_key(own)
            g = groups.get(key)
            if g is None and own["name"]:
                # seen by name now, while its tasks still carry only the address
                g = groups.get(machine_key({"ip": ip}))
            if g is None:
                g = groups.setdefault(key, {"key": key, "ip": ip, "name": own["name"], "task_count": 0, "last_seen": None})
            last = float(seen.get("last_seen") or 0.0)
            if last and (g["last_seen"] is None or last > g["last_seen"]):
                g["last_seen"] = last
                g["ip"] = ip
            if own["name"] and not g["name"]:
                g["name"] = own["name"]
        out = []
        for g in groups.values():
            label = f"{g['name']} ({g['ip']})" if (g["name"] and g["ip"]) else (g["name"] or g["ip"])
            out.append({
                **g, "label": label,
                "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(g["last_seen"])) if g["last_seen"] else "",
            })
        out.sort(key=lambda g: (-g["task_count"], g["label"]))
        return [{"key": "", "label": "未分类", "ip": "", "name": "", "task_count": unclassified, "last_seen": "", "unclassified": True}] + out

    def all_tasks_with_owner(self) -> list[dict]:
        """Admin: every task with the computer it belongs to."""
        out = []
        with self._lock:
            self._discover_existing()
            shots = list(self._shots.values())
        for sw in shots:
            try:
                m = sw.read_meta()
            except Exception:  # noqa: BLE001
                continue
            owner = self.owner_of(m)
            qjob = self.scheduler.job_for_shot(m["id"])
            out.append({
                "id": m["id"], "name": m.get("name", ""), "created_at": m.get("created_at", ""),
                "updated_at": m.get("updated_at", ""), "is_public": bool(owner.get("public")),
                "owner_label": machine_label(owner), "owner_how": str(owner.get("how") or ""),
                "owner_why": str(owner.get("why") or ""), "owner_key": machine_key(owner),
                "owner_ip": str(owner.get("ip") or ""), "owner_name": str(owner.get("name") or ""),
                "people_count": len(m.get("solved_people") or []),
                "status": qjob["status"] if qjob else m.get("status", "idle"),
            })
        out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return out

    def cleanup_expired(self) -> dict:
        days = int(self.config_store.load().retention_days)
        if days <= 0:
            return {"deleted_count": 0, "deleted_ids": [], "retention_days": days}
        cutoff = time.time() - days * 86400.0
        deleted: list[str] = []
        with self._lock:
            self._discover_existing()
            for sid, sw in list(self._shots.items()):
                if self.scheduler.job_for_shot(sid) is not None:
                    continue
                try:
                    m = sw.read_meta()
                    ref_ts = _parse_iso_ts(m.get("solved_at") or m.get("updated_at") or m.get("created_at") or "")
                    if ref_ts < cutoff:
                        shutil.rmtree(sw.root, ignore_errors=True)
                        self._shots.pop(sid, None)
                        deleted.append(sid)
                except Exception:
                    continue
        return {"deleted_count": len(deleted), "deleted_ids": deleted, "retention_days": days}

    def storage_stats(self) -> dict:
        cfg = self.config_store.load()
        days = int(cfg.retention_days)
        default_ws = str(get_paths().workspaces.resolve())
        with self._lock:
            self._discover_existing()
            total_bytes = _dir_size_bytes(self.base_dir)
            solved_cnt = 0
            for sw in self._shots.values():
                try:
                    m_cur = sw.read_meta()
                    if m_cur.get("solved_people") or m_cur.get("solved_objects"):
                        solved_cnt += 1
                except Exception:
                    pass
            try:
                du = shutil.disk_usage(self.base_dir)
                free_gb = round(du.free / (1024 ** 3), 1)
                total_gb = round(du.total / (1024 ** 3), 1)
            except OSError:
                free_gb = 0.0
                total_gb = 0.0
            return {
                **self.data_collector.today(),
                "cache_dir": str(self.base_dir),
                "configured_cache_dir": (cfg.cache_dir or "").strip(),
                "default_cache_dir": default_ws,
                "is_custom_cache_dir": str(self.base_dir) != default_ws,
                "disk_free_gb": free_gb,
                "disk_total_gb": total_gb,
                "retention_days": days,
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "shots_total": len(self._shots),
                "solved_shots": solved_cnt,
            }

    def _expires_in_days(self, meta: dict, retention_days: int) -> float | None:
        if retention_days <= 0:
            return None
        ref_ts = _parse_iso_ts(meta.get("solved_at") or meta.get("updated_at") or meta.get("created_at") or "")
        rem = (ref_ts + retention_days * 86400.0 - time.time()) / 86400.0
        return round(max(0.0, rem), 1)

    def _has_solve_cache(self, sw: ShotWorkspace) -> bool:
        cam_space_dir = sw.root / "solve_raw" / "cam_space"
        return cam_space_dir.is_dir() and any(cam_space_dir.glob("person_*.npz"))

    def _has_object_solve_cache(self, sw: ShotWorkspace) -> bool:
        obj_dir = sw.root / "solve_objects" / "cam_space"
        return obj_dir.is_dir() and any(obj_dir.glob("object_*.npz"))

    def _has_any_solve_cache(self, sw: ShotWorkspace) -> bool:
        return self._has_solve_cache(sw) or self._has_object_solve_cache(sw)

    def enrich_shot_meta(self, meta: dict) -> dict:
        sid = meta["id"]
        qjob = self.scheduler.job_for_shot(sid)
        ret_days = int(self.config_store.load().retention_days)
        sw = self.get_shot(sid)
        has_human_solve = self._has_solve_cache(sw)
        has_obj_solve = self._has_object_solve_cache(sw)
        has_solve = has_human_solve or has_obj_solve
        has_cam = bool(meta.get("camera"))
        scan_now = ""
        if meta.get("scene_scan") and meta.get("scan_align_sig"):
            try:
                scan_now = self._scan_sig(meta.get("scene_scan"), sw.root / "scene_scan" / "scan_cache.npz")
            except OSError:
                scan_now = ""
        owner = self.owner_of(meta)
        shown_detection = meta.get("detection")
        if isinstance(shown_detection, dict) and shown_detection.get("people"):
            plate_m = meta.get("plate") or {}
            size_m = (int(plate_m.get("width") or 0), int(plate_m.get("height") or 0))
            shown_detection = {
                **shown_detection,
                "people": self._box_repairs(
                    sw, list(shown_detection["people"]), size_m if size_m[0] > 0 else None, str(shown_detection.get("run_id") or "")
                ),
            }
        return {
            **meta,
            "detection": shown_detection,
            "is_public": bool(owner.get("public")),
            "owner_label": machine_label(owner),
            # the two things the buttons of step 4 show: which state is on show (对齐前 / 对齐后), and whether the
            # results of 对齐后 were computed with the scan as it is now
            "scan_align": self._scan_align_on(meta),
            "scan_align_computed": bool(scan_now) and scan_now == str(meta.get("scan_align_sig") or ""),
            "queue_job": qjob,
            "has_solve_cache": has_solve,
            "has_human_solve_cache": has_human_solve,
            "has_object_solve_cache": has_obj_solve,
            "space_mode": "aligned" if (has_solve and has_cam) else ("origin" if has_solve else "none"),
            "expires_in_days": self._expires_in_days(meta, ret_days),
            "retention_days": ret_days,
        }

    def list_shots(self, viewer: dict | None = None) -> list[dict]:
        """The tasks of this computer and the public ones (`viewer=None`: all - used by the service itself)."""
        ret_days = int(self.config_store.load().retention_days)
        with self._lock:
            self._discover_existing()
            items = []
            for sw in self._shots.values():
                try:
                    m = sw.read_meta()
                    owner = self.owner_of(m)
                    if viewer is not None and not owner.get("public") and not same_machine(owner, viewer):
                        continue
                    qjob = self.scheduler.job_for_shot(m["id"])
                    has_human_solve = self._has_solve_cache(sw)
                    has_obj_solve = self._has_object_solve_cache(sw)
                    has_solve = has_human_solve or has_obj_solve
                    solved_map = {int(p["id"]): p for p in (m.get("solved_people") or [])}
                    plate_frames_m = [int(f) for f in ((m.get("plate") or {}).get("frames") or [])]
                    det_people = list(((m.get("detection") or {}).get("people") or [])) + build_manual_detected_people(
                        m.get("manual_people"), plate_frames_m
                    )
                    people_info = []
                    seen_pids: set[int] = set()
                    for sp in (m.get("solved_people") or []):
                        pid = int(sp["id"])
                        seen_pids.add(pid)
                        f_list = sp.get("frames") or []
                        people_info.append(
                            {
                                "id": pid,
                                "frame_count": len(f_list),
                                "first_frame": int(f_list[0]) if f_list else 1,
                                "last_frame": int(f_list[-1]) if f_list else 1,
                                "solved": True,
                            }
                        )
                    for dp in det_people:
                        pid = int(dp["id"])
                        if pid in seen_pids:
                            continue
                        b_keys = sorted(int(k) for k in (dp.get("boxes") or {}).keys())
                        people_info.append(
                            {
                                "id": pid,
                                "frame_count": int(dp.get("frame_count") or len(b_keys)),
                                "first_frame": b_keys[0] if b_keys else 1,
                                "last_frame": b_keys[-1] if b_keys else 1,
                                "solved": pid in solved_map,
                            }
                        )
                    items.append(
                        {
                            "id": m["id"],
                            "name": m["name"],
                            "status": qjob["status"] if qjob else m.get("status", "idle"),
                            "queue_position": qjob.get("queue_position") if qjob else None,
                            "gpu_index": qjob.get("gpu_index") if qjob else None,
                            "has_plate": bool(m.get("plate")),
                            "has_view": bool(m.get("has_view", False)),
                            "has_solve_cache": has_solve,
                            "has_human_solve_cache": has_human_solve,
                            "has_object_solve_cache": has_obj_solve,
                            "has_camera": bool(m.get("camera")),
                            "has_scene_scan": bool(m.get("scene_scan")),
                            "work_mode": m.get("work_mode", "human"),
                            "detected_count": len(det_people),
                            "people_count": len(m.get("solved_people") or []),
                            "objects_count": len(m.get("solved_objects") or []),
                            "tracked_objects_count": len(((m.get("object_tracking") or {}).get("objects")) or []),
                            "people_info": people_info,
                            "active_person_ids": m.get("active_person_ids") or [],
                            "plate": m.get("plate"),
                            "camera": (m.get("camera") or {}).get("summary"),
                            "expires_in_days": self._expires_in_days(m, ret_days),
                            "updated_at": m.get("updated_at", ""),
                            "created_at": m.get("created_at", ""),
                            "is_public": bool(owner.get("public")),
                            "owner_label": machine_label(owner),
                        }
                    )
                except Exception:
                    continue
            items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return items

    def get_shot(self, shot_id: str) -> ShotWorkspace:
        with self._lock:
            if shot_id not in self._shots:
                cand = self.base_dir / shot_id
                if cand.is_dir() and (cand / "shot.json").is_file():
                    self._shots[shot_id] = ShotWorkspace(cand)
                else:
                    raise KeyError(f"未找到任务：{shot_id}")
            return self._shots[shot_id]

    def create_shot(self, name: str = "新任务", shot_id: str | None = None, owner: dict | None = None) -> ShotWorkspace:
        """A new task, belonging to `owner` (the computer that asks for it: {"ip", "name"})."""
        with self._lock:
            sid = shot_id or f"shot_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            sdir = self.base_dir / sid
            sdir.mkdir(parents=True, exist_ok=True)
            sw = ShotWorkspace(sdir)
            own = {"ip": normalise_ip(owner.get("ip") or ""), "name": str(owner.get("name") or ""), "how": "创建", "at": _now_iso()} if owner else None
            sw.write_meta(default_shot_meta(sid, name=name, owner=own))
            self._shots[sid] = sw
            return sw

    def delete_shot(self, shot_id: str, viewer: dict | None = None) -> None:
        if viewer is not None:
            try:
                if self.owner_of(self.get_shot(shot_id).read_meta()).get("public"):
                    raise ValueError("未分类任务不能在这里删除：是你的任务就先「移到我的任务」再删除，或者请管理员在后台删除")
            except KeyError:
                return
        self.scheduler.cancel_shot_jobs(shot_id)
        with self._lock:
            sw = self._shots.pop(shot_id, None)
            target = sw.root if sw else (self.base_dir / shot_id)
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)

    # ------------------------------------------------------------------ Step 1: Image Sequence + OCIO + Grade

    def begin_sequence_upload(self, shot_id: str) -> Path:
        sw = self.get_shot(shot_id)
        staging_dir = sw.root / "uploads" / "sequence_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir

    def sequence_upload_dir(self, shot_id: str) -> Path:
        sw = self.get_shot(shot_id)
        staging_dir = sw.root / "uploads" / "sequence_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir

    def finalize_sequence_upload(
        self,
        shot_id: str,
        ocio: OcioSettings | None = None,
        fps: float = DEFAULT_FPS,
    ) -> dict:
        sw = self.get_shot(shot_id)
        staging_dir = sw.root / "uploads" / "sequence_staging"
        final_seq_dir = sw.root / "uploads" / "sequence"
        plate_staging = sw.root / ".plate_staging"
        if plate_staging.exists():
            shutil.rmtree(plate_staging, ignore_errors=True)
        cur = sw.read_meta()
        ocio_cfg = ocio or OcioSettings.from_dict(cur.get("ocio"))
        try:
            # Validate all frames & bake previews in staging first so intercepted uploads NEVER corrupt existing shot state!
            staged_info = ingest_sequence_dir(staging_dir, plate_staging, ocio=ocio_cfg, fps=fps)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            shutil.rmtree(plate_staging, ignore_errors=True)
            raise

        with sw.lock:
            if final_seq_dir.exists():
                shutil.rmtree(final_seq_dir, ignore_errors=True)
            final_seq_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir.replace(final_seq_dir)

            p_dir = sw.root / "plate"
            if p_dir.exists():
                shutil.rmtree(p_dir, ignore_errors=True)
            plate_staging.replace(p_dir)

            staging_prefix = str(staging_dir.resolve())
            final_seq_prefix = str(final_seq_dir.resolve())
            p_stage_prefix = str(plate_staging.resolve())
            p_final_prefix = str(p_dir.resolve())

            def _remap_path(p_str: str) -> str:
                if p_str.startswith(staging_prefix):
                    return final_seq_prefix + p_str[len(staging_prefix) :]
                if p_str.startswith(p_stage_prefix):
                    return p_final_prefix + p_str[len(p_stage_prefix) :]
                return p_str

            p_info = PlateInfo(
                source_name=staged_info.source_name,
                width=staged_info.width,
                height=staged_info.height,
                fps=staged_info.fps,
                frames=staged_info.frames,
                raw_files={f: _remap_path(v) for f, v in staged_info.raw_files.items()},
                preview_files={f: _remap_path(v) for f, v in staged_info.preview_files.items()},
                file_ext=staged_info.file_ext,
                bit_depth=staged_info.bit_depth,
                input_colorspace=staged_info.input_colorspace,
                resolved_colorspace=staged_info.resolved_colorspace,
                ocio_view=staged_info.ocio_view,
                warnings=staged_info.warnings,
            )
            self._save_plate_full(p_dir, p_info)
            self._reset_people_for_new_plate(sw)
            meta = sw.update_meta(
                plate=p_info.to_dict(),
                ocio=ocio_cfg.to_dict(),
                manual_people=[],
                detection=None,
                object_tracking=None,
                cache_signatures={"detect": "", "detect_run": "", "people_net": {}, "people_post": {}, "solve_det": "", "obj_track": "", "obj_solve": ""},
                stage_message=f"序列图已通过检测并加载（{p_info.frame_count} 帧 · {p_info.width}×{p_info.height} · {p_info.resolved_colorspace} → sRGB）",
            )
        self._rebuild_preview_if_possible(sw)
        return self.enrich_shot_meta(meta)

    def set_sequence_dir(
        self,
        shot_id: str,
        seq_dir: str | Path,
        ocio: OcioSettings | None = None,
        fps: float = DEFAULT_FPS,
    ) -> dict:
        sw = self.get_shot(shot_id)
        cur = sw.read_meta()
        ocio_cfg = ocio or OcioSettings.from_dict(cur.get("ocio"))
        p_dir = sw.root / "plate"
        p_dir.mkdir(parents=True, exist_ok=True)
        p_info = ingest_sequence_dir(seq_dir, p_dir, ocio=ocio_cfg, fps=fps)
        self._save_plate_full(p_dir, p_info)
        self._reset_people_for_new_plate(sw)
        meta = sw.update_meta(
            plate=p_info.to_dict(),
            ocio=ocio_cfg.to_dict(),
            detection=None,
            object_tracking=None,
            cache_signatures={"detect": "", "detect_run": "", "people_net": {}, "people_post": {}, "solve_det": "", "obj_track": "", "obj_solve": ""},
            stage_message=f"序列图已加载（{p_info.frame_count} 帧 · {p_info.resolved_colorspace} → sRGB）",
        )
        self._rebuild_preview_if_possible(sw)
        return self.enrich_shot_meta(meta)

    def update_ocio_settings(self, shot_id: str, ocio: OcioSettings) -> dict:
        sw = self.get_shot(shot_id)
        p_info = self._load_plate_info(sw)
        if not p_info.raw_files:
            raise ValueError("当前任务尚无序列图")
        first_raw = Path(next(iter(p_info.raw_files.values())))
        seq_dir = first_raw.parent
        p_dir = sw.root / "plate"
        rebuilt = ingest_sequence_dir(seq_dir, p_dir, ocio=ocio, fps=p_info.fps)
        self._save_plate_full(p_dir, rebuilt)
        meta = sw.update_meta(
            plate=rebuilt.to_dict(),
            ocio=ocio.to_dict(),
            stage_message=f"色彩空间已更新：{rebuilt.resolved_colorspace} → sRGB",
        )
        return self.enrich_shot_meta(meta)

    @staticmethod
    def _save_plate_full(p_dir: Path, p_info: PlateInfo) -> None:
        (p_dir / "plate_full.json").write_text(
            json.dumps(
                {
                    **p_info.to_dict(),
                    "raw_files": {str(k): v for k, v in p_info.raw_files.items()},
                    "preview_files": {str(k): v for k, v in p_info.preview_files.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_plate_info(self, sw: ShotWorkspace) -> PlateInfo:
        pf = sw.root / "plate" / "plate_full.json"
        if not pf.is_file():
            raise ValueError("请先在第 1 步上传序列图")
        d = json.loads(pf.read_text(encoding="utf-8"))
        raw_map = d.get("raw_files") or d.get("frame_files") or {}
        return PlateInfo(
            source_name=d.get("source_name") or Path(d.get("source_path", "sequence")).name,
            width=int(d["width"]),
            height=int(d["height"]),
            fps=float(d["fps"]),
            frames=[int(f) for f in d["frames"]],
            raw_files={int(k): str(v) for k, v in raw_map.items()},
            preview_files={int(k): str(v) for k, v in d["preview_files"].items()},
            file_ext=d.get("file_ext") or Path(next(iter(raw_map.values()), "frame.exr")).suffix.lower(),
            bit_depth=d.get("bit_depth", "uint8"),
            input_colorspace=d.get("input_colorspace", "auto"),
            resolved_colorspace=d.get("resolved_colorspace", "sRGB Encoded Rec.709 (sRGB)"),
            ocio_view=d.get("ocio_view", "aces_sdr"),
            warnings=list(d.get("warnings") or []),
        )

    def get_preview_frame_path(self, shot_id: str, frame: int) -> Path:
        sw = self.get_shot(shot_id)
        p_info = self._load_plate_info(sw)
        if frame in p_info.preview_files:
            return Path(p_info.preview_files[frame])
        if not p_info.frames:
            raise FileNotFoundError("No sequence frames")
        nearest = min(p_info.frames, key=lambda f: abs(f - frame))
        return Path(p_info.preview_files[nearest])

    def get_shot_thumbnail_path(self, shot_id: str) -> Path:
        from ..io.plate import read_image_rgb, write_jpeg_rgb
        from ..io.plate_grade import apply_grade_rgb

        sw = self.get_shot(shot_id)
        meta = sw.read_meta()
        g = GradeParams.from_dict(meta.get("grade"))
        sig = _json_hash({"plate": meta.get("plate"), "grade": g.to_dict()})
        thumb = sw.root / "plate" / f"thumb_{sig[:8]}.jpg"
        if thumb.is_file():
            return thumb

        p_info = self._load_plate_info(sw)
        mid_f = p_info.frames[len(p_info.frames) // 2]
        rgb, _, _ = read_image_rgb(p_info.preview_files[mid_f])
        graded = apply_grade_rgb(rgb, g) if not g.is_identity else np.asarray(rgb, dtype=np.uint8)
        h, w = graded.shape[:2]
        step = max(1, w // 360)
        small = graded[::step, ::step, :3]
        return write_jpeg_rgb(thumb, small, quality=86)

    def get_person_thumbnail_path(self, shot_id: str, person_id: int) -> Path:
        from ..io.plate import read_image_rgb, write_jpeg_rgb
        from ..io.plate_grade import apply_grade_rgb

        sw = self.get_shot(shot_id)
        meta = sw.read_meta()
        g = GradeParams.from_dict(meta.get("grade"))
        p_info = self._load_plate_info(sw)
        det_people = list(((meta.get("detection") or {}).get("people") or [])) + build_manual_detected_people(
            meta.get("manual_people"), p_info.frames
        )
        target_p = next((p for p in det_people if int(p["id"]) == int(person_id)), None)
        if not target_p or not target_p.get("boxes"):
            return self.get_shot_thumbnail_path(shot_id)
        sig = _json_hash(
            {
                "pid": person_id,
                "grade": g.to_dict(),
                "det": (meta.get("cache_signatures") or {}).get("detect"),
                "manual_boxes": target_p.get("boxes") if target_p.get("is_manual") else None,
            }
        )
        thumb = sw.root / "detect" / f"person_{int(person_id):02d}_{sig[:8]}.jpg"
        if thumb.is_file():
            return thumb

        # Pick frame with largest bounding-box area for crispest portrait crop
        best_f, best_box = max(
            ((int(fk), bv) for fk, bv in target_p["boxes"].items()),
            key=lambda kv: max(0.0, (kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1])),
        )
        fpath = p_info.preview_files.get(best_f) or p_info.preview_files[p_info.frames[0]]
        rgb, _, _ = read_image_rgb(fpath)
        graded = apply_grade_rgb(rgb, g) if not g.is_identity else np.asarray(rgb, dtype=np.uint8)
        h, w = graded.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in best_box]
        bw, bh = max(16.0, x2 - x1), max(16.0, y2 - y1)
        pad_x, pad_y = bw * 0.22, bh * 0.15
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y)))
        crop = graded[cy1:cy2, cx1:cx2, :3]
        if crop.size == 0:
            crop = graded
        step = max(1, max(crop.shape[0], crop.shape[1]) // 180)
        return write_jpeg_rgb(thumb, crop[::step, ::step, :3], quality=88)

    # ------------------------------------------------------------------ Step 4: Import Tracked Camera & Align (Y-up ONLY)

    def set_camera(
        self,
        shot_id: str,
        file_path: str | Path | None,
        camera_path: str = "",
        unit: str = "auto",
        fbx_take: str = "",
        frame_offset: int = 0,
    ) -> dict:
        """Step 4: Import or replace the tracked camera (`.abc` / `.fbx` / `.usd`, `Y-up`).
        Immediately aligns SAM 3D Body's cached origin-camera-space human motion into the imported camera's
        3D world space without re-running GPU inference.
        """
        sw = self.get_shot(shot_id)
        if file_path is None or str(file_path).strip() == "":
            has_solve = self._has_any_solve_cache(sw)
            sw.update_meta(
                camera=None,
                stage_message="已切回原点标准相机空间" if has_solve else "未导入摄影机",
            )
            self._rebuild_preview_if_possible(sw)
            return self.enrich_shot_meta(sw.read_meta())

        p = Path(file_path).expanduser().resolve()
        cur = sw.read_meta()
        plate_meta = cur.get("plate") or {}
        def_w = int(plate_meta.get("width") or DEFAULT_WIDTH)
        def_h = int(plate_meta.get("height") or DEFAULT_HEIGHT)
        use_unit = unit.strip().lower() if unit and unit.strip().lower() in ("m", "cm", "auto") else "auto"

        listing = inspect_3d_file(p, take=fbx_take)
        # Strip heavy per-frame model arrays from stored camera listing metadata while keeping full hierarchy!
        slim_listing = {
            "format": listing.get("format", ""),
            "fps": listing.get("fps", 24.0),
            "unit_to_cm": float(listing.get("unit_to_cm") or 1.0),
            "cameras": listing.get("cameras") or [],
            "default_camera_path": listing.get("default_camera_path") or "",
            "hierarchy": listing.get("hierarchy") or [],
            "takes": listing.get("takes") or [],
            "auto_take": listing.get("auto_take") or "",
        }
        cam_track = load_camera_track(
            p,
            camera_path=camera_path,
            unit=use_unit,
            abc_up="y",
            fbx_take=fbx_take,
            frame_offset=frame_offset,
            default_width=def_w,
            default_height=def_h,
        )
        p_frames = plate_meta.get("frames") or []
        used_offset = int(frame_offset)
        if (
            used_offset == 0
            and p_frames
            and cam_track.frames
            and (cam_track.frames[-1] < int(p_frames[0]) or cam_track.frames[0] > int(p_frames[-1]))
        ):
            used_offset = int(p_frames[0]) - int(cam_track.frames[0])
            cam_track = cam_track.with_frame_offset(used_offset)
        if cur.get("plate") and cam_track.fps > 0:
            cur_plate = dict(cur["plate"])
            cur_plate["fps"] = cam_track.fps
            sw.update_meta(plate=cur_plate)

        if use_unit == "m":
            resolved_unit_to_cm = 100.0
        elif use_unit == "cm":
            resolved_unit_to_cm = 1.0
        else:
            resolved_unit_to_cm = float(slim_listing.get("unit_to_cm") or 1.0)

        cam_entry = {
            "file_path": str(p),
            "camera_path": cam_track.path,
            "unit": use_unit,
            "resolved_unit_to_cm": resolved_unit_to_cm,
            "fbx_take": fbx_take,
            "frame_offset": used_offset,
            "listing": slim_listing,
            "summary": cam_track.summary(),
        }
        has_solve = self._has_any_solve_cache(sw)
        msg = (
            f"已更新摄影机：{cam_track.name}"
            if has_solve
            else f"已导入摄影机：{cam_track.name}（可继续第 3 步三维解算）"
        )
        sw.update_meta(
            camera=cam_entry,
            status="ready" if has_solve else cur.get("status", "idle"),
            error=None,
            stage_message=msg,
        )
        report = self._rebuild_preview_if_possible(sw)
        if has_solve:
            self._resolve_after_scene_change(sw, shot_id)
        sw.broadcast({"event": "done", "step": "camera_align", "alignment": report})
        return self.enrich_shot_meta(sw.read_meta())

    def update_uploaded_camera_options(
        self,
        shot_id: str,
        camera_path: str = "",
        unit: str = "",
        fbx_take: str = "",
        frame_offset: int = 0,
    ) -> dict:
        sw = self.get_shot(shot_id)
        meta = sw.read_meta()
        existing = meta.get("camera")
        if not existing or not existing.get("file_path"):
            raise ValueError("尚未导入摄影机文件")
        use_unit = unit.strip().lower() if unit and unit.strip().lower() in ("m", "cm", "auto") else str(existing.get("unit") or "m")
        return self.set_camera(
            shot_id,
            file_path=existing["file_path"],
            camera_path=camera_path or existing.get("camera_path", ""),
            unit=use_unit,
            fbx_take=fbx_take if fbx_take != "" else existing.get("fbx_take", ""),
            frame_offset=frame_offset,
        )

    def clear_camera(self, shot_id: str) -> dict:
        return self.set_camera(shot_id, file_path=None)

    def _record_job(self, sw: ShotWorkspace, qjob: QueueJob, job_type: str, outcome: str, error: str | None = None) -> None:
        """Data collection: what the job that just ended was given and what it produced goes into the table of
        the day (`data_collect.py`).  Recording never disturbs a job: every error is caught and logged there."""
        try:
            record_job(self, sw, qjob, job_type, outcome, str(error or ""))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _track_part(person: dict, first: int, last: int, new_id: int) -> dict:
        """One stretch of a tracked person as a person of its own (a track that followed several people was cut):
        the boxes, scores and areas of the frames `first`..`last`.  A part with a new number says whom it was cut
        from, and keeps using that track's outlines."""
        src = int(person["id"])
        piece = dict(person)
        for key, val in person.items():
            if isinstance(val, dict) and key in ("boxes", "raw_boxes", "scores", "areas", "confirmed", "repaired"):
                piece[key] = {k: v for k, v in val.items() if first <= int(k) <= last}
        piece["id"] = int(new_id)
        piece["frame_count"] = len(piece.get("boxes") or {})
        if "prominence" in person:
            # how much of the track's weight in the picture (the sum of its outline areas) lies in this stretch
            whole_area = float(sum((person.get("areas") or {}).values())) if isinstance(person.get("areas"), dict) else 0.0
            share = (
                float(sum(piece["areas"].values())) / whole_area
                if whole_area > 0.0
                else len(piece.get("boxes") or {}) / max(1, len(person.get("boxes") or {}))
            )
            piece["prominence"] = round(float(person["prominence"]) * share, 1)
        if int(new_id) != src:
            piece["split_from"] = int(person.get("split_from", src))
            piece["mask_id"] = int(person.get("mask_id", src))
        return piece

    def _is_whole_shot_task(self, sw: ShotWorkspace) -> bool:
        """Were the task's people solved by the whole-shot solver (v0.2.x)?  Only those can be solved again from
        cached network outputs.  A task solved by v0.1.x keeps its instant re-placement: solving it anew means
        tracking people again, which renumbers them, so that is left to the user pressing the solve button."""
        raw_files = sorted((sw.root / "solve_raw" / "raw_net").glob("person_*.npz"))
        if not raw_files:
            return False
        try:
            with np.load(raw_files[0], allow_pickle=False) as rz:
                return "kp_in_mask" in rz.files
        except Exception:  # noqa: BLE001
            return False

    def _resolve_after_scene_change(self, sw: ShotWorkspace, shot_id: str, client_ip: str = "", exclude: set[int] | None = None) -> bool:
        """What solved people were solved AGAINST has changed (the camera move; whether the scan is in use): the
        people whose result is no longer the result of today's inputs are solved again from the cached per-frame
        network outputs (a queue job of seconds per person, no network inference).  Returns False when nobody
        has to be solved again."""
        meta = sw.read_meta()
        if not meta.get("camera") or not self._has_solve_cache(sw) or meta.get("active_job_id"):
            return False
        if not self._is_whole_shot_task(sw):
            return False
        try:
            if not [pid for pid in self._people_to_solve_again(sw) if pid not in (exclude or set())]:
                return False
        except Exception:  # noqa: BLE001 - when in doubt the job decides (it computes nothing that is up to date)
            pass
        try:
            self.enqueue_solve(shot_id, only_existing=True, client_ip=client_ip)
            return True
        except Exception:  # noqa: BLE001 - fall back to showing the cached result under the new camera
            return False

    # ------------------------------------------------------------------ Step 4: the scan is used when the user says so

    @staticmethod
    def _scan_align_on(meta: dict) -> bool:
        """Is the scan in use - the state the interface calls 对齐后?  Importing a scan does not turn it on, the
        button of step 4 does.  A task from before v0.2.7 has no such setting: there an imported scan was
        always in use."""
        if not meta.get("scene_scan"):
            return False
        val = meta.get("scan_align")
        return True if val is None else bool(val)

    @staticmethod
    def _scan_sig(scan_entry: dict | None, scan_npz: Path) -> str:
        """The scan as the solve sees it: what was imported with which options, and the reduced file made of it."""
        st = scan_npz.stat()
        return _json_hash({"scan": scan_entry, "mtime": int(st.st_mtime), "size": st.st_size})

    @staticmethod
    def _camera_sigs(cam: CameraTrack, plate, shot_fps: float) -> tuple[CameraTrack, str, str]:
        """(camera on the plate's frames, signature of its intrinsics, signature of its move)."""
        cam_res = cam.resample_to(plate.frames)
        k_map = cam_res.intrinsics_per_frame(plate.width, plate.height)
        intr_sig = _json_hash(
            {
                "w": plate.width,
                "h": plate.height,
                "k": [[round(float(v), 3) for v in k_map[f].ravel()] for f in sorted(k_map)],
            }
        )
        ext_sig = _json_hash(
            {"c2w": [[round(float(v), 4) for v in m.ravel()] for m in cam_res.cam_to_world], "fps": round(shot_fps, 4)}
        )
        return cam_res, intr_sig, ext_sig

    @staticmethod
    def _person_net_sig(person: dict, is_manual: bool, detect_run: str, cam_intr_sig: str, hand_refine: bool) -> str:
        """What one person's cached network outputs depend on: their boxes (for a tracked person: of this very
        detection run), the camera's intrinsics, the hand option and the prompt rule.  The grade is deliberately
        absent: a result stays the result of the picture it was estimated from, and a new grade reaches tracked
        people only through a new detection, which clears them anyway."""
        # measurement runs only: a batch mode or precision forced by environment must not reuse another
        # setting's cached outputs
        forced_mode = str(os.environ.get("SHOTSPACE_SAM3D_BATCH_MODE") or "").strip().lower()
        forced_precision = str(os.environ.get("SHOTSPACE_SAM3D_PRECISION") or "").strip().lower()
        return _json_hash(
            {
                "detect_run": "" if is_manual else detect_run,
                "person": person,
                "cam_intr": cam_intr_sig,
                "hand_refine": hand_refine,
                "prompt": "mask_v3",
                **({"batch_mode": forced_mode} if forced_mode else {}),
                **({"precision": forced_precision} if forced_precision else {}),
            }
        )

    @staticmethod
    def _person_post_sig(net_sig: str, smoothing: float, cam_ext_sig: str, scan_sig: str, has_scan: bool) -> str:
        """What one person's whole-shot solve depends on: their network outputs, the smoothing, the solver, the
        camera move and - when the scan is in use - the scan."""
        return _json_hash(
            {"net": net_sig, "smoothing": round(smoothing, 5), "solver": f"whole_shot_v2.{SOLVER_REVISION}",
             "camera": cam_ext_sig, "scan": scan_sig, **({"scan_solver": SCAN_SOLVER_REVISION} if has_scan else {})}
        )

    def _people_to_solve_again(self, sw: ShotWorkspace) -> list[int]:
        """Solved people whose result on show is NOT the result of today's inputs - their boxes, the camera, the
        smoothing, the solver, and the scan when it is in use.  These are the people the next solve job would
        solve again: the answer is computed with that job's own signature functions from the same sources, so
        'nobody' here means such a job would compute nothing and need not be queued."""
        self._sync_detection_record(sw)
        meta = sw.read_meta()
        solved_ids = [int(sp["id"]) for sp in (meta.get("solved_people") or [])]
        if not solved_ids:
            return []
        plate = self._load_plate_info(sw)
        cam = self._load_camera_from_entry(meta.get("camera"), meta.get("plate"))
        if cam is None:
            return []  # without a camera nothing can be solved
        sigs = dict(meta.get("cache_signatures") or {})
        sp = meta.get("solve_params") or {}
        hand_refine = bool(sp.get("hand_refine", True))
        smoothing = float(sp.get("smoothing", 0.5))
        shot_fps = float(getattr(cam, "fps", 0.0) or getattr(plate, "fps", 0.0) or 24.0)
        scan_npz = sw.root / "scene_scan" / "scan_cache.npz"
        has_scan = self._scan_align_on(meta) and scan_npz.is_file()
        det_disk, _run = self._detection_on_disk(sw)
        have_boxes = (
            det_disk is not None
            and bool(sigs.get("detect"))
            and str(det_disk.get("engine") or "") == TRACKER_ENGINE
        )
        manual_people = build_manual_detected_people(normalize_manual_people(meta.get("manual_people")), plate.frames)
        manual_ids = {int(mp["id"]) for mp in manual_people}
        by_id = {int(p["id"]): p for p in (list(det_disk.get("people") or []) if have_boxes else []) + manual_people}
        detect_run = str(sigs.get("detect_run") or "")
        same_detection = str(sigs.get("solve_det") or "") == detect_run
        _cam_res, cam_intr_sig, cam_ext_sig = self._camera_sigs(cam, plate, shot_fps)
        scan_sig = self._scan_sig(meta.get("scene_scan"), scan_npz) if has_scan else ""
        net_prev = {str(k): v for k, v in dict(sigs.get("people_net") or {}).items()}
        post_prev = {str(k): v for k, v in dict(sigs.get("people_post") or {}).items()}
        again: list[int] = []
        for pid in solved_ids:
            person = by_id.get(pid)
            if person is None or not (same_detection or pid in manual_ids):
                again.append(pid)  # their boxes are gone or renumbered: the job has to deal with them
                continue
            net_now = self._person_net_sig(person, pid in manual_ids, detect_run, cam_intr_sig, hand_refine)
            post_now = self._person_post_sig(net_now, smoothing, cam_ext_sig, scan_sig, has_scan)
            if (
                net_prev.get(str(pid)) != net_now
                or post_prev.get(str(pid)) != post_now
                or not (sw.root / "solve_raw" / "raw_net" / f"person_{pid:02d}.npz").is_file()
                or not (sw.root / "solve_raw" / "cam_space" / f"person_{pid:02d}.npz").is_file()
            ):
                again.append(pid)
        return again

    @staticmethod
    def _kept_dir(sw: ShotWorkspace, mode: str) -> Path:
        return sw.root / "solve_raw" / f"cam_space_{mode}"

    def _trim_kept_results(self, sw: ShotWorkspace, gone: Callable[[int], bool]) -> None:
        """Forget the results kept for the other state of the scan switch for every person `gone` says yes to.
        Whoever leaves the task, or has their network outputs replaced, must not come back with an old result."""
        for mode in ("noscan", "scan"):
            kept_dir = self._kept_dir(sw, mode)
            if kept_dir.is_dir():
                for old_file in kept_dir.glob("person_*.npz"):
                    if gone(int(old_file.stem.split("_")[-1])):
                        old_file.unlink(missing_ok=True)
        with sw.lock:
            kept = dict(sw.read_meta().get("scan_mode_results") or {})
            if not kept:
                return
            new_kept = {
                **kept,
                "post": {str(k): v for k, v in dict(kept.get("post") or {}).items() if not gone(int(k))},
                "people": {str(k): v for k, v in dict(kept.get("people") or {}).items() if not gone(int(k))},
                "pair": {str(k): v for k, v in dict(kept.get("pair") or {}).items() if not gone(int(k))},
            }
            sw.update_meta(scan_mode_results=new_kept if new_kept["post"] else {})

    def _switch_scan_state(self, sw: ShotWorkspace, use_scan: bool) -> set[int]:
        """Change whether the scan is in use (对齐前 <-> 对齐后) and keep what was solved in the state that is left.

        Returns the people whose result that comes back is the other half of the very solve that was on show
        (v0.2.12: a solve computes both states, `scan_mode_results["pair"]`): they need nothing computed, even when
        both halves are older than today's settings - then the solve button says so, as without a scan.

        Every person's valid result goes aside with its signature and its readings; the result kept from the last
        time the task was in the other state comes back in its place.  Going back and forth therefore computes
        nothing.  Whether a result that comes back still is the result of today's inputs is not decided here: it
        comes back with the signature it was solved under, and that signature is what the next solve - and
        `_people_to_solve_again` - compare with today's.  A person without a kept result keeps the old one on
        show, without a signature, until they are solved in the new state."""
        with sw.lock:
            meta = sw.read_meta()
            was = self._scan_align_on(meta)
            if was == bool(use_scan):
                if meta.get("scan_align") is None:
                    sw.update_meta(scan_align=was)
                return set()
            old_mode, new_mode = ("scan", "noscan") if was else ("noscan", "scan")
            cam_dir = sw.root / "solve_raw" / "cam_space"
            old_dir, new_dir = self._kept_dir(sw, old_mode), self._kept_dir(sw, new_mode)
            sigs = dict(meta.get("cache_signatures") or {})
            post = {str(k): v for k, v in dict(sigs.get("people_post") or {}).items()}
            kept = dict(meta.get("scan_mode_results") or {})
            back = kept if str(kept.get("mode") or "") == new_mode else {}
            back_post = {str(k): v for k, v in dict(back.get("post") or {}).items()}
            back_people = {str(k): v for k, v in dict(back.get("people") or {}).items()}
            back_pair = {str(k): v for k, v in dict(back.get("pair") or {}).items()}
            scan_npz_now = sw.root / "scene_scan" / "scan_cache.npz"
            scan_now = self._scan_sig(meta.get("scene_scan"), scan_npz_now) if (meta.get("scene_scan") and scan_npz_now.is_file()) else ""
            # results kept for 对齐后 count as the other half only when they were computed with the scan of now
            back_scan_ok = new_mode != "scan" or (bool(scan_now) and str(back.get("scan_sig") or "") == scan_now)
            solved = [dict(sp) for sp in (meta.get("solved_people") or [])]
            # 1. While files change places no signature stands: an interruption then costs a re-solve from the
            #    cached network outputs, and can never leave a result filed under the wrong state.
            sigs["people_post"] = {}
            sw.update_meta(cache_signatures=sigs, scan_mode_results={})
            if old_dir.is_dir():
                for stale in old_dir.glob("person_*.npz"):
                    stale.unlink(missing_ok=True)
            # 2. the files
            aside_post: dict[str, str] = {}
            aside_people: dict[str, dict] = {}
            aside_pair: dict[str, str] = {}
            new_post: dict[str, str] = {}
            new_solved: list[dict] = []
            halves: set[int] = set()
            for sp in solved:
                pid = int(sp["id"])
                key, name = str(pid), f"person_{pid:02d}.npz"
                cur_file = cam_dir / name
                comes_back = key in back_post and key in back_people and (new_dir / name).is_file()
                if key in post and cur_file.is_file():
                    old_dir.mkdir(parents=True, exist_ok=True)
                    if comes_back:
                        cur_file.replace(old_dir / name)
                    else:
                        # Nothing comes back for this person: the old result stays on show until they are solved
                        # in the new state, so it has to be in both places.  A second name for the same file costs
                        # no time and no space (a result is ~70 MB per person and 300 frames); the workers never
                        # write into a result file, they replace it (`_atomic_save_npz`), which leaves the kept
                        # name untouched.  Where the storage cannot do that, the file is copied.
                        (old_dir / name).unlink(missing_ok=True)
                        try:
                            os.link(cur_file, old_dir / name)
                        except OSError:
                            tmp = old_dir / f".{name}.tmp"
                            shutil.copyfile(cur_file, tmp)
                            tmp.replace(old_dir / name)
                    aside_post[key] = post[key]
                    aside_people[key] = sp
                    if comes_back and back_scan_ok and back_pair.get(key) and back_pair.get(key) == post[key]:
                        aside_pair[key] = back_post[key]
                        halves.add(pid)
                if comes_back:
                    (new_dir / name).replace(cur_file)
                    new_post[key] = back_post[key]
                    new_solved.append(dict(back_people[key]))
                else:
                    new_solved.append(sp)
            if new_dir.is_dir():
                for stale in new_dir.glob("person_*.npz"):
                    stale.unlink(missing_ok=True)
            # 3. one write: the state, the people as they now stand, and what was put aside
            sigs["people_post"] = new_post
            aside = {"mode": old_mode, "post": aside_post, "people": aside_people, "pair": aside_pair}
            if old_mode == "scan":
                aside["scan_sig"] = str(meta.get("scan_align_sig") or "")
            sw.update_meta(
                cache_signatures=sigs,
                solved_people=new_solved,
                scan_align=bool(use_scan),
                scan_mode_results=aside if aside_post else {},
            )
            return halves

    def set_scan_alignment(self, shot_id: str, enabled: bool, client_ip: str = "") -> dict:
        """Step 4, the two buttons.  `enabled=True` (按扫描对齐 / 对齐后): the solved people are put on the scan -
        whoever has no up-to-date result with the scan is solved again from the cached network outputs, no network
        inference - and tracked objects take their depth from it.  `enabled=False` (对齐前): the task as it is
        without the scan.  Both results are kept, so switching between them computes nothing."""
        sw = self.get_shot(shot_id)
        meta = sw.read_meta()
        if meta.get("active_job_id"):
            raise ValueError("任务正在计算，请等待完成或取消后再切换")
        if enabled and not meta.get("scene_scan"):
            raise ValueError("尚未导入场景扫描")
        if enabled and (meta.get("solved_people") or []) and not meta.get("camera"):
            raise ValueError("请先导入摄影机")
        halves = self._switch_scan_state(sw, bool(enabled))
        report = self._rebuild_preview_if_possible(sw)
        self._clear_export_files(sw)
        cur = sw.read_meta()
        again: list[int] = []
        if cur.get("camera") and self._has_solve_cache(sw) and self._is_whole_shot_task(sw):
            # the other half of the solve on show came back: nothing to compute for them (v0.2.12)
            again = [pid for pid in self._people_to_solve_again(sw) if pid not in halves]
        if again:
            # a failure to queue is the caller's to see: the state is switched, its people are not solved yet
            self.enqueue_solve(shot_id, only_existing=True, client_ip=client_ip)
        else:
            # nobody has to be solved: the state on show is complete as it stands
            patch: dict = {"stage_message": "已按扫描对齐" if enabled else "已切换为对齐前"}
            scan_npz = sw.root / "scene_scan" / "scan_cache.npz"
            if enabled and cur.get("scene_scan") and scan_npz.is_file():
                patch["scan_align_sig"] = self._scan_sig(cur.get("scene_scan"), scan_npz)
            sw.update_meta(**patch)
            sw.broadcast({"event": "done", "step": "scene_scan_align", "alignment": report})
        return self.enrich_shot_meta(sw.read_meta())

    def set_scene_scan(
        self,
        shot_id: str,
        file_path: str | Path | None,
        unit: str = "auto",
        selected_paths: list[str] | None = None,
    ) -> dict:
        """Step 4 (separate from camera): Import or replace the unified static scene scan mesh
        (`.abc` / `.fbx` / `.usd`). Immediately calibrates character foot-to-ground contact and
        stationary/dynamic object depth against the scan in camera space without re-running GPU inference."""
        sw = self.get_shot(shot_id)
        cache_npz = sw.root / "scene_scan" / "scan_cache.npz"
        if file_path is None or str(file_path).strip() == "":
            # Without a scan there is only the state without it: the task goes back to 对齐前 (results kept for
            # that state come back as they are), and what was solved with the scan is not kept.
            before = sw.read_meta()
            was_on = self._scan_align_on(before)
            if was_on and before.get("active_job_id"):
                raise ValueError("任务正在计算，请等待完成或取消后再移除场景扫描")
            halves: set[int] = set()
            if was_on:
                halves = self._switch_scan_state(sw, False)
            self._trim_kept_results(sw, lambda _pid: True)
            cache_npz.unlink(missing_ok=True)
            cache_npz.with_name(SCAN_SOLVER_NPZ).unlink(missing_ok=True)
            cur_m = sw.read_meta()
            has_cam = bool(cur_m.get("camera"))
            sw.update_meta(
                scene_scan=None,
                scan_align=False,
                scan_align_sig="",
                stage_message="已移除场景扫描（保留摄影机空间）" if has_cam else "已移除场景扫描",
            )
            self._rebuild_preview_if_possible(sw)
            if was_on:
                self._clear_export_files(sw)
                # people who were only ever solved with the scan have no result without it yet
                self._resolve_after_scene_change(sw, shot_id, exclude=halves)
            return self.enrich_shot_meta(sw.read_meta())

        p = Path(file_path).expanduser().resolve()
        use_unit = unit.strip().lower() if unit and unit.strip().lower() in ("auto", "cm", "m") else "auto"
        scan = load_scene_scan(p, unit=use_unit, selected_paths=selected_paths)
        _write_scan_cache(cache_npz, scan)
        scan_entry = {
            "file_path": str(p),
            "unit": scan.unit,
            "selected_paths": scan.selected_paths,
            "listing": {
                "format": scan.format,
                "meshes": scan.available_meshes,
                "hierarchy": scan.hierarchy,
            },
            "summary": scan.summary(),
        }
        cur_m = sw.read_meta()
        if str((cur_m.get("scan_mode_results") or {}).get("mode") or "") == "scan":
            # results kept for 对齐后 were computed with the scan that is replaced now
            self._trim_kept_results(sw, lambda _pid: True)
            cur_m = sw.read_meta()
        has_cam = bool(cur_m.get("camera"))
        msg = (
            f"已加载场景扫描：{scan.name}（{scan.format.upper()}）"
            if has_cam
            else f"已加载场景扫描：{scan.name}（请导入摄影机）"
        )
        # Importing or changing a scan computes nothing and does not change which state is on show: the people
        # are put on the scan when the user presses 按扫描对齐 (`set_scan_alignment`).  The state is read BEFORE
        # the new entry is written: a task from before v0.2.7 has no setting, and its scan was in use.
        sw.update_meta(scene_scan=scan_entry, scan_align=self._scan_align_on(cur_m), error=None, stage_message=msg)
        report = self._rebuild_preview_if_possible(sw)
        sw.broadcast({"event": "done", "step": "scene_scan_align", "alignment": report})
        return self.enrich_shot_meta(sw.read_meta())

    def update_uploaded_scene_scan_options(
        self,
        shot_id: str,
        unit: str = "",
        selected_paths: list[str] | None = None,
    ) -> dict:
        sw = self.get_shot(shot_id)
        meta = sw.read_meta()
        existing = meta.get("scene_scan")
        if not existing or not existing.get("file_path"):
            raise ValueError("尚未导入场景扫描文件")
        use_unit = unit.strip().lower() if unit and unit.strip().lower() in ("auto", "cm", "m") else str(existing.get("unit") or "m")
        use_sel = selected_paths if selected_paths is not None else existing.get("selected_paths")
        return self.set_scene_scan(shot_id, file_path=existing["file_path"], unit=use_unit, selected_paths=use_sel)

    def clear_scene_scan(self, shot_id: str) -> dict:
        return self.set_scene_scan(shot_id, file_path=None)

    def _load_cached_scene_scan(self, sw: ShotWorkspace, meta: dict | None = None) -> dict | None:
        m = meta if meta is not None else sw.read_meta()
        scan_meta = m.get("scene_scan")
        if not scan_meta or not scan_meta.get("file_path"):
            return None
        cache_npz = sw.root / "scene_scan" / "scan_cache.npz"
        summary = scan_meta.get("summary") or {}
        name = str(summary.get("name") or Path(scan_meta["file_path"]).stem)
        if not cache_npz.is_file():
            p = Path(scan_meta["file_path"])
            if not p.is_file():
                return None
            scan = load_scene_scan(
                p,
                unit=str(scan_meta.get("unit") or "auto"),
                selected_paths=scan_meta.get("selected_paths"),
            )
            _write_scan_cache(cache_npz, scan)
        with np.load(cache_npz, allow_pickle=False) as d:
            return {
                "name": name,
                "vertices_cm": np.asarray(d["vertices_cm"], dtype=np.float32),
                "faces": np.asarray(d["faces"], dtype=np.int32),
                "display_vertices_cm": np.asarray(d["display_vertices_cm"], dtype=np.float32),
                "display_faces": np.asarray(d["display_faces"], dtype=np.int32),
            }

    def _load_camera_from_entry(self, entry: dict | None, plate_meta: dict | None) -> CameraTrack | None:
        if not entry or not entry.get("file_path"):
            return None
        def_w = int((plate_meta or {}).get("width") or DEFAULT_WIDTH)
        def_h = int((plate_meta or {}).get("height") or DEFAULT_HEIGHT)
        use_unit = str(entry.get("unit") or "auto")
        return load_camera_track(
            entry["file_path"],
            camera_path=entry.get("camera_path", ""),
            unit=use_unit,
            abc_up="y",
            fbx_take=entry.get("fbx_take", ""),
            frame_offset=int(entry.get("frame_offset", 0)),
            default_width=def_w,
            default_height=def_h,
        )

    def _shot_frames(
        self,
        meta: dict,
        cam: CameraTrack | None,
        chars: list[SkinnedCharacter],
        objs: list[RigidObjectTrack] | None = None,
    ) -> list[int]:
        if meta.get("plate") and meta["plate"].get("frames"):
            return [int(f) for f in meta["plate"]["frames"]]
        if chars:
            all_cf = sorted({int(f) for c in chars for f in c.frames})
            if all_cf:
                return list(range(all_cf[0], all_cf[-1] + 1))
        if objs:
            all_of = sorted({int(f) for o in objs for f in o.frames})
            if all_of:
                return list(range(all_of[0], all_of[-1] + 1))
        if cam and cam.frames:
            return [int(f) for f in cam.frames]
        return [1001]

    def _load_cam_space_chars(self, sw: ShotWorkspace, active_ids: list[int] | None = None) -> list[SkinnedCharacter]:
        cam_space_dir = sw.root / "solve_raw" / "cam_space"
        if not cam_space_dir.is_dir():
            return []
        allow = {int(x) for x in active_ids} if active_ids else None
        chars: list[SkinnedCharacter] = []
        for npz_path in sorted(cam_space_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[-1])
            if allow is not None and pid not in allow:
                continue
            # one unreadable or numerically invalid result must not take the other people of the shot with it
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if not np.all(np.isfinite(np.asarray(data["joint_world"], dtype=np.float64))):
                        continue
                    chars.append(SkinnedCharacter.from_mhr_npz(pid, data))
            except Exception:  # noqa: BLE001
                continue
        return chars

    def _load_cam_space_objects(self, sw: ShotWorkspace, active_ids: list[int] | None = None) -> list[RigidObjectTrack]:
        obj_dir = sw.root / "solve_objects" / "cam_space"
        if not obj_dir.is_dir():
            return []
        allow = {int(x) for x in active_ids} if active_ids else None
        objs: list[RigidObjectTrack] = []
        for npz_path in sorted(obj_dir.glob("object_*.npz")):
            oid = int(npz_path.stem.split("_")[-1])
            if allow is not None and oid not in allow:
                continue
            with np.load(npz_path, allow_pickle=False) as data:
                objs.append(RigidObjectTrack.from_npz(oid, data))
        return objs

    @staticmethod
    def _align_scene(
        cam_res: CameraTrack,
        plate_w: int,
        plate_h: int,
        shot_frames: list[int],
        raw_chars: list[SkinnedCharacter],
        raw_objs: list[RigidObjectTrack],
        scan_cache: dict | None,
    ) -> tuple[list[SkinnedCharacter], list[RigidObjectTrack]]:
        """World-space scene shown in the viewport and written to the export files.

        People solved over the whole shot (current solver) are only placed by the exact camera: their trajectory,
        body scale and ground contact were already solved against this camera and scan.  Results cached by older
        versions keep their original alignment path so previously solved tasks still open."""

        def _is_world_solved(c: SkinnedCharacter) -> bool:
            return bool((c.custom_data or {}).get("shotspace:world_solved", False))

        aligned: dict[int, SkinnedCharacter] = {}
        legacy_raw = [c for c in raw_chars if not _is_world_solved(c)]
        for c in raw_chars:
            if _is_world_solved(c):
                aligned[id(c)] = place_world_solved_character(c, cam_res, shot_frames)
        legacy_al = [align_standard_character_to_camera(c, cam_res, plate_w, plate_h, shot_frames) for c in legacy_raw]
        ref_chars = [aligned[id(c)] for c in raw_chars if id(c) in aligned] + legacy_al
        aligned_objs = [
            align_standard_object_to_camera(o, cam_res, plate_w, plate_h, shot_frames, reference_characters=ref_chars)
            for o in raw_objs
        ]
        if scan_cache is not None and (legacy_raw or raw_objs):
            legacy_al, aligned_objs = calibrate_scene_with_scan(
                scan_cache["vertices_cm"], scan_cache["faces"], cam_res, plate_w, plate_h, shot_frames,
                legacy_raw, legacy_al, raw_objs, aligned_objs,
            )
        for c, c_al in zip(legacy_raw, legacy_al):
            aligned[id(c)] = c_al
        return [aligned[id(c)] for c in raw_chars], aligned_objs

    def _rebuild_preview_if_possible(self, sw: ShotWorkspace) -> dict | None:
        meta = sw.read_meta()
        plate_meta = meta.get("plate")
        cam = self._load_camera_from_entry(meta.get("camera"), plate_meta)
        all_chars = self._load_cam_space_chars(sw, meta.get("active_person_ids") or None)
        all_objs = self._load_cam_space_objects(sw, meta.get("active_object_ids") or None)
        scan_cache = self._load_cached_scene_scan(sw, meta)

        if cam is None and scan_cache is None and not all_chars and not all_objs:
            sw.update_meta(has_view=False, alignment_report=None)
            return None

        shot_frames = self._shot_frames(meta, cam, all_chars, all_objs)
        plate_w = int((plate_meta or {}).get("width") or (cam.width if cam else DEFAULT_WIDTH))
        plate_h = int((plate_meta or {}).get("height") or (cam.height if cam else DEFAULT_HEIGHT))
        scan_view_payload: dict | None = (
            {
                "name": scan_cache["name"],
                "vertices": scan_cache["display_vertices_cm"],
                "faces": scan_cache["display_faces"],
            }
            if scan_cache is not None
            else None
        )

        if cam is not None:
            cam_res = cam.resample_to(shot_frames)
            cameras_out = [("aligned", cam_res)]
            aligned_chars, aligned_objs = self._align_scene(
                cam_res, plate_w, plate_h, shot_frames, all_chars, all_objs,
                scan_cache if self._scan_align_on(meta) else None,  # shown always, used only in the state 对齐后
            )
            report = AlignmentReport(
                mode="direct",
                fit_requested="per_frame",
                frames=shot_frames,
                common_authored_frames=len(cam_res.authored_frames),
                scale=1.0,
                target_spread_cm=round(camera_spread_cm(cam_res.cam_to_world), 2),
            )
            # Check frame range mismatch or partial missing camera frames relative to sequence frames
            if cam.frames and shot_frames:
                from ..io.plate import format_frame_ranges

                if cam.frames[-1] < shot_frames[0] or cam.frames[0] > shot_frames[-1]:
                    report.warnings.append(
                        f"摄影机帧范围 ({cam.frames[0]}–{cam.frames[-1]}) 与序列帧 ({shot_frames[0]}–{shot_frames[-1]}) 不重叠，请检查帧偏移"
                    )
                elif len(cam.own_frames or cam.frames) > 1:
                    authored_set = set(int(f) for f in (cam.own_frames or cam.frames))
                    missing_cam = sorted(set(int(f) for f in shot_frames) - authored_set)
                    if missing_cam:
                        report.warnings.append(
                            f"摄影机缺少部分序列帧采样（共缺 {len(missing_cam)} 帧：{format_frame_ranges(missing_cam)}），缺失区间已按邻近帧保持/插值"
                        )
            report_dict = report.to_dict()
        else:
            # Step 3 completed, before Step 4 camera import: origin standard camera space!
            origin_cam = _standard_origin_camera(shot_frames, plate_meta)
            cameras_out = [("origin", origin_cam)]
            aligned_chars = all_chars
            aligned_objs = all_objs
            report_dict = None

        build_viewport_bundle(
            out_dir=sw.root / "aligned",
            shot_frames=shot_frames,
            plate_meta=plate_meta,
            cameras=cameras_out,
            characters=aligned_chars,
            alignment_report=report_dict,
            rigid_objects=aligned_objs,
            scene_scan=scan_view_payload,
        )
        sw.update_meta(has_view=True, alignment_report=report_dict)
        return report_dict

    # ------------------------------------------------------------------ Step 2 & Step 3: Queue-Backed Detect & Solve

    def update_settings(self, shot_id: str, patch: dict) -> dict:
        sw = self.get_shot(shot_id)
        cur = sw.read_meta()
        active_changed = (
            ("active_person_ids" in patch and patch["active_person_ids"] != cur.get("active_person_ids"))
            or ("active_object_ids" in patch and patch["active_object_ids"] != cur.get("active_object_ids"))
        )
        for k in (
            "name",
            "work_mode",
            "ocio",
            "grade",
            "detect_params",
            "manual_people",
            "selection",
            "solve_params",
            "active_person_ids",
            "export_person_ids",
            "object_prompts",
            "object_solve_params",
            "active_object_ids",
        ):
            if k in patch:
                if k == "manual_people":
                    cur[k] = normalize_manual_people(patch[k] if isinstance(patch[k], list) else [])
                elif k == "export_person_ids":
                    solved_now = {int(sp["id"]) for sp in (cur.get("solved_people") or [])}
                    wanted = sorted({int(x) for x in (patch[k] if isinstance(patch[k], list) else [])} & solved_now)
                    # "everybody" is stored as an empty list, so people solved later are included by default
                    cur[k] = [] if len(wanted) == len(solved_now) else wanted
                elif isinstance(cur.get(k), dict) and isinstance(patch[k], dict):
                    cur[k] = {**cur[k], **patch[k]}
                else:
                    cur[k] = patch[k]
        sw.write_meta(cur)
        if active_changed and self._has_any_solve_cache(sw):
            self._rebuild_preview_if_possible(sw)
        return self.enrich_shot_meta(sw.read_meta())

    def enqueue_detect(
        self,
        shot_id: str,
        threshold: float | None = None,
        grade: dict | None = None,
        client_ip: str = "",
    ) -> dict:
        """Step 3: Detect & track people across the graded sequence via 2048px ViTDet-H + SAM 3.1 Memory Bank."""
        sw = self.get_shot(shot_id)
        self._refuse_if_task_has_job(shot_id)
        if grade is not None or threshold is not None:
            patch: dict = {}
            if grade is not None:
                patch["grade"] = GradeParams.from_dict(grade).to_dict()
            if threshold is not None:
                patch["detect_params"] = {"threshold": float(threshold)}
            self.update_settings(shot_id, patch)

        meta = sw.read_meta()
        self._load_plate_info(sw)

        def _state_hook(qjob: QueueJob) -> None:
            if qjob.status == "queued":
                sw.update_meta(status="queued", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "queued", "job": qjob.to_dict()})
            elif qjob.status == "running":
                sw.update_meta(status="running", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "status", "status": "running", "job": qjob.to_dict()})
            elif qjob.status == "cancelled":
                self._after_aborted_people_job(sw)
                self._record_job(sw, qjob, "人物检测", "已取消")
                cur = sw.read_meta()
                sw.update_meta(
                    status="ready" if cur.get("has_view") else "idle",
                    active_job_id=None,
                    stage_message="任务已取消",
                )
                sw.broadcast({"event": "cancelled", "job": qjob.to_dict()})
            elif qjob.status == "failed":
                self._after_aborted_people_job(sw)
                self._record_job(sw, qjob, "人物检测", "失败", qjob.error)
                sw.update_meta(status="error", active_job_id=None, error=qjob.error, stage_message=qjob.message)
                sw.broadcast({"event": "error", "error": qjob.error})

        def _runner(gpu_solver: SolverManager, qjob: QueueJob, is_cancelled: Callable[[], bool]) -> None:
            cur_meta = sw.read_meta()
            cur_plate = self._load_plate_info(sw)
            g_obj = GradeParams.from_dict(cur_meta.get("grade"))
            thresh = float((cur_meta.get("detect_params") or {}).get("threshold", 0.8))
            t_job = time.time()

            solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
            if is_cancelled():
                raise JobCancelledError()

            def _on_ev(ev: dict) -> None:
                if is_cancelled():
                    raise JobCancelledError()
                if ev.get("event") == "stage":
                    msg = f"[GPU {qjob.gpu_index}] {ev.get('message', '')}"
                    qjob.stage = ev.get("stage", "")
                    qjob.message = msg
                    sw.update_meta(stage_message=msg)
                elif ev.get("event") == "progress":
                    qjob.done = int(ev["done"])
                    qjob.total = int(ev["total"])
                    qjob.stage = ev.get("stage", "")
                    sw.update_meta(progress={"done": qjob.done, "total": qjob.total, "stage": qjob.stage})
                sw.broadcast(ev)

            boxes_path = sw.root / "detect" / "boxes.json"
            gpu_solver.detect_people(
                cur_plate,
                boxes_path,
                solver_frame_files=solver_frames,
                threshold=thresh,
                on_event=_on_ev,
            )
            # the boxes on disk are the new ones from here on: the bookkeeping below must not be skipped
            det_data = self._detection_on_disk(sw)[0] or json.loads(boxes_path.read_text(encoding="utf-8"))
            det_sig = self._detect_sig(cur_plate, g_obj, thresh)
            _, cleared = self._record_detection(sw, det_data, det_sig, thresh, g_obj, cur_plate)
            done_msg = f"检测跟踪完成：共检出 {len(det_data.get('people', []))} 人"
            if cleared:
                done_msg += f"；人物编号已重新分配，已解算的 {cleared} 人已清除"
            qjob._record = {"times": {"detect": time.time() - t_job}}
            self._record_job(sw, qjob, "人物检测", "完成")
            qjob.message = done_msg
            sw.update_meta(
                status="ready" if sw.read_meta().get("has_view") else "idle",
                active_job_id=None,
                stage_message=done_msg,
            )
            sw.broadcast({"event": "done", "step": "detect", "people_count": len(det_data.get("people", []))})

        job, pos = self.scheduler.submit(
            shot_id=shot_id,
            shot_name=meta.get("name", shot_id),
            job_type="detect",
            runner=_runner,
            on_state=_state_hook,
            client_ip=client_ip,
        )
        return {"queued": True, "job": job.to_dict(position=pos), "shot": self.enrich_shot_meta(sw.read_meta())}

    def enqueue_solve(
        self,
        shot_id: str,
        params_patch: dict | None = None,
        client_ip: str = "",
        wait: bool = False,
        only_existing: bool = False,
    ) -> dict:
        """Step 3: solve people.  The people selected now are ADDED to the ones solved before: only people who are
        new, or whose inputs changed, go through the network and the whole-shot solve; everybody else is kept as is.
        `only_existing=True` solves nobody new (used when the camera or the scan changed under solved people)."""
        sw = self.get_shot(shot_id)
        self._refuse_if_task_has_job(shot_id)
        if params_patch:
            self.update_settings(shot_id, params_patch)
        meta = sw.read_meta()
        self._load_plate_info(sw)
        if not meta.get("camera") or not (meta.get("camera") or {}).get("file_path"):
            raise ValueError("请先在第 2 步导入摄影机后再开始 SAM 3D Body 三维计算")

        done_event = threading.Event()
        final_err: list[str] = []

        def _state_hook(qjob: QueueJob) -> None:
            if qjob.status == "queued":
                sw.update_meta(status="queued", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "queued", "job": qjob.to_dict()})
            elif qjob.status == "running":
                sw.update_meta(status="running", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "status", "status": "running", "job": qjob.to_dict()})
            elif qjob.status == "cancelled":
                self._after_aborted_people_job(sw)
                self._record_job(sw, qjob, "人物解算", "已取消")
                shutil.rmtree(sw.root / "solve_raw" / ".other_half_tmp", ignore_errors=True)
                cur = sw.read_meta()
                sw.update_meta(
                    status="ready" if cur.get("has_view") else "idle",
                    active_job_id=None,
                    stage_message=(
                        "解算已完成；同时计算另一状态（对齐前 / 对齐后）的结果时已取消，切换时再算"
                        if getattr(qjob, "_main_done", False) else "任务已取消"
                    ),
                )
                sw.broadcast({"event": "cancelled", "job": qjob.to_dict()})
                done_event.set()
            elif qjob.status == "failed":
                self._after_aborted_people_job(sw)
                self._record_job(sw, qjob, "人物解算", "失败", qjob.error)
                shutil.rmtree(sw.root / "solve_raw" / ".other_half_tmp", ignore_errors=True)
                sw.update_meta(status="error", active_job_id=None, error=qjob.error, stage_message=qjob.message)
                sw.broadcast({"event": "error", "error": qjob.error})
                final_err.append(qjob.error or "解算失败")
                done_event.set()
            elif qjob.status == "completed":
                done_event.set()

        def _runner(gpu_solver: SolverManager, qjob: QueueJob, is_cancelled: Callable[[], bool]) -> None:
            cur_meta = sw.read_meta()
            cur_plate = self._load_plate_info(sw)
            cur_cam = self._load_camera_from_entry(cur_meta.get("camera"), cur_meta.get("plate"))
            if cur_cam is None:
                raise ValueError("请先在第 2 步导入摄影机后再开始 SAM 3D Body 三维计算")

            g_obj = GradeParams.from_dict(cur_meta.get("grade"))
            thresh = float((cur_meta.get("detect_params") or {}).get("threshold", 0.8))
            sel_dict = cur_meta.get("selection") or {}
            sel_spec = SelectionSpec(
                mode=sel_dict.get("mode", "largest"),
                count=int(sel_dict.get("count", 2)),
                ids=str(sel_dict.get("ids", "1")),
                picks=sel_dict.get("picks") or [],
                least_frames=int(sel_dict.get("least_frames", 3)),
            )
            sp = cur_meta.get("solve_params") or {}
            hand_refine = bool(sp.get("hand_refine", True))
            # Every person is solved with their own mask (it tells the network WHICH person in an overlapping
            # crowd to reconstruct) and with one body for the whole shot; neither is optional any more.
            use_mask = True
            lock_shape = True
            smoothing = float(sp.get("smoothing", 0.5))
            shot_fps = float(getattr(cur_cam, "fps", 0.0) or getattr(cur_plate, "fps", 0.0) or 24.0)
            scan_npz = sw.root / "scene_scan" / "scan_cache.npz"
            # an imported scan is used only once the user has put the people on it (step 4, 按扫描对齐)
            has_scan = self._scan_align_on(cur_meta) and scan_npz.is_file()
            # v0.2.12: with a scan imported, whoever this job solves is solved for BOTH states (对齐前 and 对齐后),
            # so that switching between them later computes nothing (`_other_half`)
            scan_file_sig = (
                self._scan_sig(cur_meta.get("scene_scan"), scan_npz) if (cur_meta.get("scene_scan") and scan_npz.is_file()) else ""
            )
            other_tmp = sw.root / "solve_raw" / ".other_half_tmp"
            shutil.rmtree(other_tmp, ignore_errors=True)

            solve_warnings: list[str] = []
            sw.update_meta(solve_warnings=[])
            # for the data collection: how long the stages took and whose result this job computed (and committed)
            job_rec: dict = {"times": {}, "people": {}, "counts": {}, "clock": time.time()}
            qjob._record = job_rec

            def _tick(stage_key: str) -> None:
                """The time since the last report belongs to the stage that reports now."""
                t_now = time.time()
                job_rec["times"][stage_key] = job_rec["times"].get(stage_key, 0.0) + (t_now - job_rec["clock"])
                job_rec["clock"] = t_now

            def _note(text: str) -> None:
                """A non-fatal problem the user should see after the job (kept with the task, shown in step 3)."""
                if text and text not in solve_warnings:
                    solve_warnings.append(text)
                    sw.update_meta(solve_warnings=solve_warnings[-12:])

            def _on_ev(ev: dict) -> None:
                if is_cancelled():
                    raise JobCancelledError()
                if ev.get("event") == "stage":
                    msg = f"[GPU {qjob.gpu_index}] {ev.get('message', '')}"
                    qjob.stage = ev.get("stage", "")
                    qjob.message = msg
                    sw.update_meta(stage_message=msg)
                elif ev.get("event") == "progress":
                    qjob.done = int(ev["done"])
                    qjob.total = int(ev["total"])
                    qjob.stage = ev.get("stage", "")
                    sw.update_meta(progress={"done": qjob.done, "total": qjob.total, "stage": qjob.stage})
                elif ev.get("event") == "warning":
                    _note(str(ev.get("message", "")))
                sw.broadcast(ev)

            self._sync_detection_record(sw)
            cur_meta = sw.read_meta()
            sigs = dict(cur_meta.get("cache_signatures") or {})
            det_sig = self._detect_sig(cur_plate, g_obj, thresh)
            boxes_path = sw.root / "detect" / "boxes.json"
            solver_frames: dict[int, str] | None = None
            manual_tracks_norm = normalize_manual_people(cur_meta.get("manual_people"))
            manual_people_eval = build_manual_detected_people(manual_tracks_norm, cur_plate.frames)
            manual_id_set = {int(mp["id"]) for mp in manual_people_eval}
            wanted_ids_if_mode = parse_person_ids(sel_spec.ids) if sel_spec.mode == "ids" else set()
            only_manual_ids = (
                sel_spec.mode == "ids"
                and bool(manual_people_eval)
                and bool(wanted_ids_if_mode)
                and wanted_ids_if_mode.issubset(manual_id_set)
            )
            # People are looked for ONCE per plate.  The numbers the tracker gives them are what the artist picks
            # from and what solved results are stored under, so a solve never looks for people again behind the
            # artist's back (a new grade does not do it either): only "重新检测选人框" does, or a new plate.
            # The one exception: a detection that found nobody is worthless, a changed grade may retry it.
            det_disk, _disk_run = self._detection_on_disk(sw)
            have_boxes = (
                det_disk is not None
                and bool(sigs.get("detect"))
                and str(det_disk.get("engine") or "") == TRACKER_ENGINE
            )
            stored_det = det_disk if have_boxes else {"people": []}
            skip_auto_detect = bool(manual_people_eval) and (only_manual_ids or not have_boxes)
            retry_empty = have_boxes and not (stored_det.get("people") or []) and sigs.get("detect") != det_sig

            if skip_auto_detect or only_existing:
                # `only_existing`: the people are already solved, their boxes are the ones they were solved from
                det_data = stored_det
            elif not have_boxes or retry_empty:
                solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
                job_rec["clock"] = time.time()
                gpu_solver.detect_people(
                    cur_plate,
                    boxes_path,
                    solver_frame_files=solver_frames,
                    threshold=thresh,
                    on_event=_on_ev,
                )
                _tick("detect")
                # the boxes on disk are the new ones from here on: record them before anything can interrupt
                det_data = self._detection_on_disk(sw)[0] or json.loads(boxes_path.read_text(encoding="utf-8"))
                sigs, cleared = self._record_detection(sw, det_data, det_sig, thresh, g_obj, cur_plate)
                if cleared:
                    _note(f"重新检测后人物编号已重新分配，已解算的 {cleared} 人已清除")
                cur_meta = sw.read_meta()
                if is_cancelled():
                    raise JobCancelledError()
            else:
                det_data = stored_det

            all_people = list(det_data.get("people") or []) + manual_people_eval
            if not all_people and not only_existing:
                raise ValueError("未检测到人物，请调高曝光/Gamma 或使用「+ 手动添加框」框选目标人物")
            by_id = {int(p["id"]): p for p in all_people}

            # ---- who is solved by this job --------------------------------------------------------------------
            # Solving ADDS people: everybody solved before stays, and the people selected now are solved as well.
            # Only the tracker's numbering can break that: after a new detection the numbers belong to other
            # people, so results solved from the previous detection cannot be carried over (manual tracks can).
            detect_run = str(sigs.get("detect_run") or "")
            prev_solved = [dict(sp) for sp in (cur_meta.get("solved_people") or [])]
            same_detection = str(sigs.get("solve_det") or "") == detect_run
            kept_ids = [
                int(sp["id"]) for sp in prev_solved
                if int(sp["id"]) in by_id and (same_detection or int(sp["id"]) in manual_id_set)
            ]
            lost_ids = {int(sp["id"]) for sp in prev_solved} - set(kept_ids)
            if lost_ids:
                self._forget_solved_people(sw, lost_ids)
                gone_boxes = sorted(pid for pid in lost_ids if pid not in by_id)
                renumbered = sorted(pid for pid in lost_ids if pid in by_id)
                if renumbered:
                    _note(f"重新检测后人物编号已重新分配，已解算的 {len(renumbered)} 人已清除")
                if gone_boxes:
                    _note("以下已解算人物的选框已不存在，结果已移除：" + "、".join(f"#{pid}" for pid in gone_boxes))
                prev_solved = [sp for sp in prev_solved if int(sp["id"]) not in lost_ids]
                sigs = {**(sw.read_meta().get("cache_signatures") or {}), **{k: v for k, v in sigs.items() if k not in ("people_net", "people_post")}}

            if only_existing:
                chosen_people = []
            else:
                chosen_people, _ = select_people(all_people, sel_spec)
                if not chosen_people and not kept_ids:
                    raise ValueError("当前选人条件未匹配到人物，请检查编号、点选位置或手动框有效帧")
            target_ids = sorted(set(kept_ids) | {int(p["id"]) for p in chosen_people})
            if not target_ids:
                report = self._rebuild_preview_if_possible(sw)
                self._clear_export_files(sw)
                done_msg = "没有需要解算的人物"
                qjob.message = done_msg
                cur_now = sw.read_meta()
                sw.update_meta(status="ready" if cur_now.get("has_view") else "idle", active_job_id=None, error=None, stage_message=done_msg)
                sw.broadcast({"event": "done", "step": "solve", "alignment": report})
                return

            # The signature functions are the task manager's own (`_people_to_solve_again` answers with the same
            # ones whether a job like this one would compute anything).
            cam_res_intr, cam_intr_sig, cam_ext_sig = self._camera_sigs(cur_cam, cur_plate, shot_fps)
            scan_sig = self._scan_sig(cur_meta.get("scene_scan"), scan_npz) if has_scan else ""

            def _net_sig(person: dict) -> str:
                return self._person_net_sig(person, int(person["id"]) in manual_id_set, detect_run, cam_intr_sig, hand_refine)

            def _post_sig(net_sig_of_person: str) -> str:
                return self._person_post_sig(net_sig_of_person, smoothing, cam_ext_sig, scan_sig, has_scan)

            raw_net_dir = sw.root / "solve_raw" / "raw_net"
            cam_space_dir = sw.root / "solve_raw" / "cam_space"
            net_prev = {str(k): v for k, v in dict(sigs.get("people_net") or {}).items()}
            post_prev = {str(k): v for k, v in dict(sigs.get("people_post") or {}).items()}
            net_now = {str(pid): _net_sig(by_id[pid]) for pid in target_ids}
            post_now = {pid_s: _post_sig(sig) for pid_s, sig in net_now.items()}

            # A task solved by v0.2.1 carries ONE signature for its whole selection.  When that signature still
            # matches today's inputs, the cached results are as valid as they were and are taken over person by
            # person, so adding somebody to such a task does not send the earlier people through the network again.
            legacy_net = str(sigs.get("solve_net") or "")
            legacy_state = cur_meta.get("solved_state") or {}
            legacy_ids = [int(x) for x in (legacy_state.get("selected_ids") or [])]
            if legacy_net and legacy_ids and all(pid in by_id for pid in legacy_ids):
                legacy_people = [by_id[pid] for pid in legacy_ids]
                legacy_manual = [lp for lp in legacy_people if bool(lp.get("is_manual"))]
                # v0.2.1 had the grade inside its signature.  The grade the task was solved with is on record, so a
                # grade touched since then does not throw the cached results away (it does not for new tasks either).
                legacy_dets = [det_sig]
                try:
                    solved_det = _json_hash(
                        {
                            "engine": "people_tracker_v2",
                            "ocio": legacy_state.get("ocio_cs") or cur_plate.resolved_colorspace,
                            "view": legacy_state.get("ocio_view") or cur_plate.ocio_view,
                            "grade": GradeParams.from_dict(legacy_state.get("grade")).to_dict(),
                            "threshold": float(legacy_state.get("threshold", thresh)),
                            "frames": len(cur_plate.frames),
                        }
                    )
                    if solved_det not in legacy_dets:
                        legacy_dets.append(solved_det)
                except Exception:  # noqa: BLE001 - an unreadable record only means the caches are not taken over
                    pass
                tracker_state = "tracker" if (sw.root / "detect" / "person_masks_sam3.npz").is_file() else "none"
                matched = False
                for legacy_det in legacy_dets:
                    legacy_masks = [tracker_state]
                    if legacy_manual:
                        legacy_masks.insert(0, "merged:" + _json_hash({"detect": legacy_det, "people": legacy_manual, "v": 2}))
                    for legacy_mask_state in legacy_masks:
                        legacy_now = _json_hash(
                            {
                                "detect_env": legacy_det,
                                "chosen_ids_boxes": legacy_people,
                                "cam_intr": cam_intr_sig,
                                "hand_refine": hand_refine,
                                "use_mask": use_mask,
                                "prompt": "mask_v3",
                                "masks": legacy_mask_state,
                            }
                        )
                        if legacy_now != legacy_net:
                            continue
                        legacy_post_now = _json_hash(
                            {"net": legacy_now, "smoothing": round(smoothing, 5), "solver": f"whole_shot_v2.{SOLVER_REVISION}",
                             "camera": cam_ext_sig, "scan": scan_sig, **({"scan_solver": SCAN_SOLVER_REVISION} if has_scan else {})}
                        )
                        for pid in legacy_ids:
                            if str(pid) in net_now and str(pid) not in net_prev:
                                net_prev[str(pid)] = net_now[str(pid)]
                                if legacy_post_now == str(sigs.get("postprocess") or "") and str(pid) not in post_prev:
                                    post_prev[str(pid)] = post_now[str(pid)]
                        matched = True
                        break
                    if matched:
                        break
            need_net: list[dict] = []   # the network has to look at these people (new, or their inputs changed)
            post_only: list[int] = []   # cached network outputs are valid, only the whole-shot solve is redone
            for pid in target_ids:
                raw_ok = (raw_net_dir / f"person_{pid:02d}.npz").is_file()
                cam_ok = (cam_space_dir / f"person_{pid:02d}.npz").is_file()
                if net_prev.get(str(pid)) != net_now[str(pid)] or not raw_ok:
                    need_net.append(by_id[pid])
                elif post_prev.get(str(pid)) != post_now[str(pid)] or not cam_ok:
                    post_only.append(pid)
            # Before anything is recomputed, the signatures of exactly those results are withdrawn (and the v0.2.1
            # whole-selection signature is retired, its still-valid part now lives in the per-person entries).
            # If the job is interrupted, a file rewritten half-way through the list can then never pass for a
            # valid cache of the earlier settings; the results themselves stay on disk until they are replaced.
            redo_net = {int(person["id"]) for person in need_net}
            redo_any = redo_net | set(post_only)
            sigs["people_net"] = {k: v for k, v in net_prev.items() if int(k) in target_ids and int(k) not in redo_net}
            sigs["people_post"] = {k: v for k, v in post_prev.items() if int(k) in target_ids and int(k) not in redo_any}
            for legacy_key in ("solve_net", "postprocess"):
                sigs.pop(legacy_key, None)
            sw.update_meta(cache_signatures=sigs)
            if redo_net:
                # a result kept for the other state of the scan switch was solved from the network outputs that
                # are about to be replaced
                self._trim_kept_results(sw, lambda pid: pid in redo_net)

            # Per-person masks for the network: tracked people bring theirs from the tracking stage; manually
            # keyframed people get one from their boxes.  The two sets are merged so nobody is solved from a bare box.
            masks_npz_for_solve: Path | None = None
            tracker_masks = sw.root / "detect" / "person_masks_sam3.npz"
            manual_chosen = [p for p in need_net if bool(p.get("is_manual"))]
            if manual_chosen:
                manual_masks = sw.root / "detect" / "person_masks_manual.npz"
                manual_sig = _json_hash({"detect": det_sig, "detect_run": detect_run, "people": manual_chosen, "v": 2})
                paths_cfg = get_paths()
                can_run_sam3 = (
                    hasattr(gpu_solver, "segment_people_masks")
                    and paths_cfg.sam_obj_python.is_file()
                    and (paths_cfg.sam31_weights / "sam3.1_multiplex.pt").is_file()
                )
                if sigs.get("person_masks_manual") != manual_sig or not manual_masks.is_file():
                    # 1) where the artist's box sits on a person the tracker already outlined, use that outline
                    manual_entries, manual_cover = manual_masks_from_tracker(
                        manual_chosen, list(det_data.get("people") or []), tracker_masks
                    )
                    # 2) people the tracker never saw (the usual reason for drawing a box) are outlined from the box
                    need_seg = [
                        mp for mp in manual_chosen
                        if manual_cover.get(int(mp["id"]), (0, 1))[0] < 0.8 * max(1, manual_cover.get(int(mp["id"]), (0, 1))[1])
                    ]
                    if need_seg and can_run_sam3:
                        seg_npz = sw.root / "detect" / "person_masks_manual_seg.npz"
                        try:
                            if solver_frames is None:
                                solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
                            gpu_solver.segment_people_masks(
                                plate=cur_plate,
                                selected_people=need_seg,
                                output_masks_npz=seg_npz,
                                solver_frame_files=solver_frames,
                                on_event=_on_ev,
                            )
                            with np.load(seg_npz, allow_pickle=False) as sm:
                                seg_frames = {k.rsplit("_", 1)[0] for k in sm.files}
                                for stem in seg_frames:
                                    if f"{stem}_bits" not in manual_entries and all(
                                        f"{stem}_{suffix}" in sm.files for suffix in ("bits", "shape", "rect")
                                    ):
                                        for suffix in ("bits", "shape", "rect"):
                                            manual_entries[f"{stem}_{suffix}"] = sm[f"{stem}_{suffix}"]
                        except JobCancelledError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            # an outline is a refinement: without it the person is still solved from the artist's box
                            _note(f"手动框人物的轮廓提取失败，已改为只按框解算：{type(exc).__name__}: {str(exc)[:160]}")
                    if is_cancelled():
                        raise JobCancelledError()
                    tmp_manual = manual_masks.with_name(".person_masks_manual.tmp.npz")
                    np.savez_compressed(tmp_manual, **manual_entries)
                    tmp_manual.replace(manual_masks)
                    sigs["person_masks_manual"] = manual_sig
                merged: dict[str, np.ndarray] = {}
                manual_keys = {f"p{int(mp['id'])}_" for mp in manual_chosen}
                if tracker_masks.is_file():
                    with np.load(tracker_masks, allow_pickle=False) as tm:
                        merged.update({k: tm[k] for k in tm.files if not any(k.startswith(mk) for mk in manual_keys)})
                if manual_masks.is_file():
                    with np.load(manual_masks, allow_pickle=False) as mm:
                        merged.update({k: mm[k] for k in mm.files if any(k.startswith(mk) for mk in manual_keys)})
                if merged:
                    masks_npz_for_solve = sw.root / "detect" / "person_masks_solve.npz"
                    np.savez_compressed(masks_npz_for_solve, **merged)
            elif tracker_masks.is_file():
                masks_npz_for_solve = tracker_masks

            # ---- solve what is new or changed, leave everybody else alone --------------------------------------
            # People go through the network and the whole-shot solve in batches of PEOPLE_PER_PASS.  After every
            # batch its people are committed to the task (list, signatures, view), so a long job shows its results
            # as they come and a cancelled job keeps what was finished.
            prev_by_id = {int(sp["id"]): sp for sp in prev_solved}
            fresh: dict[int, dict] = {}
            finished: set[int] = set()  # people whose work in this job is over, solved or failed
            cut_people: list[int] = []  # people that came out of tracks which followed more than one person
            for stale in raw_net_dir.glob(".split_*.npz"):  # parts of a cut that an interrupted job never took over
                stale.unlink(missing_ok=True)

            def _next_person_id() -> int:
                """The first number nobody carries: a track that is cut needs numbers for its further people."""
                return max([0] + list(by_id) + list(target_ids)) + 1

            def _take_in_cuts(result: dict) -> None:
                """A track that jumped from one person to another was cut by the solve (`track_split`) and comes
                back as several people.  They become people of the task like any other: the tracker's record
                gets the parts in place of the track, each part gets its own cached network outputs, and the
                signatures say that those are what today's boxes give - so nobody is estimated or cut twice."""
                cuts = [c for c in (result.get("splits") or []) if int(c.get("from", -1)) in by_id and int(c["from"]) not in manual_id_set]
                if not cuts:
                    return
                people_list = list(det_data.get("people") or [])
                where = {int(p["id"]): i for i, p in enumerate(people_list)}
                firsts: list[tuple[int, Path]] = []
                for cut in cuts:
                    src = int(cut["from"])
                    if src not in where:
                        continue
                    whole = people_list[where[src]]
                    names = []
                    for part in cut["parts"]:
                        part_id = int(part["id"])
                        piece = self._track_part(whole, int(part["first"]), int(part["last"]), part_id)
                        tmp_raw = raw_net_dir / f".split_{src:02d}_{part_id:02d}.npz"
                        names.append(f"#{part_id}")
                        if part_id == src:
                            people_list[where[src]] = piece
                            firsts.append((src, tmp_raw))
                        else:
                            if tmp_raw.is_file():
                                tmp_raw.replace(raw_net_dir / f"person_{part_id:02d}.npz")
                            people_list.append(piece)
                            cut_people.append(part_id)
                            finished.add(part_id)
                            if src in redo_net:
                                redo_net.add(part_id)
                            redo_any.add(part_id)
                        by_id[part_id] = piece
                    _note(f"人物 #{src} 的轨迹存在位置突变，已拆分为 {'、'.join(names)}")
                    self._trim_kept_results(sw, lambda pid, src=src: pid == src)
                # the tracker's record first (file, then task), then the first parts' own cached outputs: should
                # the job die in between, a track whose boxes changed without its outputs is simply estimated again
                det_data["people"] = people_list
                tmp_boxes = boxes_path.with_name(".boxes.tmp.json")
                tmp_boxes.write_text(json.dumps(det_data, ensure_ascii=False), encoding="utf-8")
                tmp_boxes.replace(boxes_path)
                rec = dict(sw.read_meta().get("detection") or {})
                rec["people"] = people_list
                sw.update_meta(detection=rec)
                for src, tmp_raw in firsts:
                    if tmp_raw.is_file():
                        tmp_raw.replace(raw_net_dir / f"person_{src:02d}.npz")
                for cut in cuts:
                    for part in cut["parts"]:
                        key = str(int(part["id"]))
                        net_now[key] = _net_sig(by_id[int(part["id"])])
                        post_now[key] = _post_sig(net_now[key])
                target_ids[:] = sorted(set(target_ids) | set(cut_people))
                for stale in raw_net_dir.glob(".split_*.npz"):
                    stale.unlink(missing_ok=True)
            def _other_half(ids: list[int]) -> str:
                """v0.2.12 - for the people this job solved, the result of the OTHER state of the scan switch
                (对齐前 when 对齐后 is on show, and the other way round), computed now by the very solve the switch
                would otherwise queue (`postprocess_cached` from the cached network outputs, with or without the
                scan, the same camera, smoothing and frames), so the two are the same numbers and switching later
                computes nothing.  Only when a scan is imported: a shot without one is solved exactly as before.
                Whoever already has a valid result for that state keeps it.  Returns a few words for the closing
                message."""
                if not scan_file_sig or not ids:
                    return ""
                other_scan = not has_scan
                other_mode, word = ("scan", "对齐后") if other_scan else ("noscan", "对齐前")
                kept_dir = self._kept_dir(sw, other_mode)
                sig_other = {
                    str(pid): self._person_post_sig(net_now[str(pid)], smoothing, cam_ext_sig, scan_file_sig if other_scan else "", other_scan)
                    for pid in target_ids if str(pid) in net_now
                }
                kept0 = dict(sw.read_meta().get("scan_mode_results") or {})
                kept_post0 = (
                    {str(k): v for k, v in dict(kept0.get("post") or {}).items()} if str(kept0.get("mode") or "") == other_mode else {}
                )
                need = [
                    pid for pid in ids
                    if not (kept_post0.get(str(pid)) == sig_other.get(str(pid)) and (kept_dir / f"person_{pid:02d}.npz").is_file())
                ]
                computed: dict[int, dict] = {}
                if need:
                    other_tmp.mkdir(parents=True, exist_ok=True)

                    def handler(ev: dict) -> None:
                        if is_cancelled():
                            raise JobCancelledError()
                        kind = ev.get("event")
                        if kind == "warning":
                            _note(str(ev.get("message", "")))
                            sw.broadcast(ev)
                            return
                        if kind != "progress":
                            return
                        done_n, total_n = int(ev.get("done") or 0), max(1, int(ev.get("total") or 1))
                        text = f"同时计算{word}的结果：第 {done_n}/{total_n} 人"
                        qjob.stage, qjob.done, qjob.total = "other_half", int(round(1000 * (done_n - 1) / total_n)), 1000
                        qjob.message = f"[GPU {qjob.gpu_index}] {text}"
                        sw.update_meta(stage_message=qjob.message, progress={"done": qjob.done, "total": qjob.total, "stage": "other_half"})
                        sw.broadcast({"event": "stage", "stage": "other_half", "message": text})
                        sw.broadcast({"event": "progress", "stage": "other_half", "done": qjob.done, "total": qjob.total})

                    job_rec["clock"] = time.time()
                    res_other = gpu_solver.postprocess_cached(
                        raw_net_dir=raw_net_dir,
                        cam_space_dir=other_tmp,
                        shot_frames=cur_plate.frames,
                        lock_shape=lock_shape,
                        smoothing=smoothing,
                        fps=shot_fps,
                        scan_npz=scan_npz if other_scan else None,
                        cam_to_world={int(f): cam_res_intr.cam_to_world[i] for i, f in enumerate(cam_res_intr.frames)},
                        person_ids=need,
                        on_event=handler,
                        new_id_start=None,  # the tracks were cut by the solve on show already
                    )
                    _tick("post")  # the other half is a whole-shot solve as well
                    if is_cancelled():
                        raise JobCancelledError()
                    computed = {int(e["id"]): e for e in (res_other.get("people") or [])}
                # filed in one write: the results, their signatures, and which result on show each is the other half of
                with sw.lock:
                    m_now = sw.read_meta()
                    kept = dict(m_now.get("scan_mode_results") or {})
                    if str(kept.get("mode") or "") != other_mode:
                        kept = {}
                        if kept_dir.is_dir():
                            for stale in kept_dir.glob("person_*.npz"):
                                stale.unlink(missing_ok=True)
                    k_post = {str(k): v for k, v in dict(kept.get("post") or {}).items()}
                    k_people = {str(k): v for k, v in dict(kept.get("people") or {}).items()}
                    k_pair = {str(k): v for k, v in dict(kept.get("pair") or {}).items()}
                    kept_dir.mkdir(parents=True, exist_ok=True)
                    on_show_post = {str(k): v for k, v in dict((m_now.get("cache_signatures") or {}).get("people_post") or {}).items()}
                    for pid in ids:
                        key, name = str(pid), f"person_{pid:02d}.npz"
                        if pid in need:
                            if pid in computed and (other_tmp / name).is_file():
                                (other_tmp / name).replace(kept_dir / name)
                                k_post[key], k_people[key] = sig_other[key], dict(computed[pid])
                            else:
                                (kept_dir / name).unlink(missing_ok=True)
                                k_post.pop(key, None)
                                k_people.pop(key, None)
                                k_pair.pop(key, None)
                                continue
                        if on_show_post.get(key):
                            k_pair[key] = on_show_post[key]
                    new_kept = {"mode": other_mode, "post": k_post, "people": k_people, "pair": k_pair}
                    if other_scan:
                        new_kept["scan_sig"] = scan_file_sig
                    patch: dict = {"scan_mode_results": new_kept if k_post else {}}
                    if other_scan:
                        # step 4 shows the switch right away when every person has a result on the scan of now
                        everyone = [int(sp["id"]) for sp in (m_now.get("solved_people") or [])]
                        complete = bool(everyone) and all(
                            (kept_dir / f"person_{pid:02d}.npz").is_file()
                            and (k_post.get(str(pid)) == sig_other.get(str(pid)) or (k_pair.get(str(pid)) and k_pair.get(str(pid)) == on_show_post.get(str(pid))))
                            for pid in everyone
                        )
                        patch["scan_align_sig"] = scan_file_sig if complete else ""
                    sw.update_meta(**patch)
                shutil.rmtree(other_tmp, ignore_errors=True)
                missed = [pid for pid in need if pid not in computed]
                if missed:
                    _note(f"以下人物的{word}结果没有算出，切换到{word}时再算：" + "、".join(f"#{pid}" for pid in missed[:20]))
                return f"，同时算好{word}的结果" if len(missed) < len(need) else ""

            old_choice = [int(x) for x in (cur_meta.get("export_person_ids") or [])]
            cam_meta_cur = cur_meta.get("camera") or {}
            job_people = len(need_net) + len(post_only)
            t_work = time.time()

            def _commit(final: bool) -> tuple[list[dict], dict | None]:
                """Write the task's people as they stand: everybody finished in this job with the new result,
                everybody else with the result they had (a person still waiting to be solved again keeps the old
                result on show, without a valid signature)."""
                people_now: list[dict] = []
                for pid in target_ids:
                    if pid in fresh:
                        people_now.append(fresh[pid])
                    elif pid not in finished and pid in prev_by_id and (cam_space_dir / f"person_{pid:02d}.npz").is_file():
                        if not (final and pid in redo_any):
                            people_now.append(prev_by_id[pid])
                ids_now = [int(sp["id"]) for sp in people_now]
                if final:
                    # Whoever is not in the final list has no valid result: a person whose solve failed just
                    # now, or a file left behind by an interrupted job.  Their files go, so the list, the view
                    # and the export can never disagree about who is in the task.
                    for sub in (raw_net_dir, cam_space_dir):
                        if sub.is_dir():
                            for old_file in sub.glob("person_*.npz"):
                                if int(old_file.stem.split("_")[-1]) not in ids_now:
                                    old_file.unlink(missing_ok=True)
                    self._trim_kept_results(sw, lambda pid: pid not in ids_now)
                sigs["people_net"] = {str(pid): net_now[str(pid)] for pid in ids_now if pid in fresh or pid not in redo_net}
                sigs["people_post"] = {str(pid): post_now[str(pid)] for pid in ids_now if pid in fresh or pid not in redo_any}
                sigs["solve_det"] = detect_run
                newly = [pid for pid in ids_now if pid not in prev_by_id]
                # people added now are exported by default, an earlier choice among the others is kept
                export_choice = sorted((set(old_choice) | set(newly)) & set(ids_now)) if old_choice else []
                if len(export_choice) == len(ids_now):
                    export_choice = []
                sw.update_meta(
                    cache_signatures=sigs,
                    solved_people=people_now,
                    active_person_ids=ids_now,
                    export_person_ids=export_choice,
                    solved_at=_now_iso(),
                    solved_state={
                        "cam_file": str(cam_meta_cur.get("file_path") or ""),
                        "cam_path": str(cam_meta_cur.get("camera_path") or ""),
                        "cam_take": str(cam_meta_cur.get("fbx_take") or ""),
                        "grade": g_obj.to_dict(),
                        "ocio_cs": cur_plate.resolved_colorspace,
                        "ocio_view": cur_plate.ocio_view,
                        "threshold": thresh,
                        "selected_ids": ids_now,
                        "manual_tracks": manual_tracks_norm,
                        "hand_refine": hand_refine,
                        "use_mask": use_mask,
                        "lock_shape": lock_shape,
                        "smoothing": round(smoothing, 4),
                    },
                    # every person of the task now has the result with the scan as it is: step 4 shows 已对齐
                    **({"scan_align_sig": scan_sig} if (final and has_scan) else {}),
                )
                job_rec["people"] = {int(pid): ("新增" if pid in redo_net else "重新解算") for pid in fresh}
                view_report = self._rebuild_preview_if_possible(sw)
                if not final:
                    sw.broadcast({"event": "partial", "step": "solve", "solved_ids": ids_now})
                return people_now, view_report

            def _people_events(stage_word: str, done_before: int, batch_no: int, batch_count: int, batch_size: int):
                """Progress of a long job in the artist's terms: how many people in all, how many are finished,
                which batch is being worked on and how far it is, and roughly how long the rest will take."""

                def handler(ev: dict) -> None:
                    if is_cancelled():
                        raise JobCancelledError()
                    kind, stage = ev.get("event"), str(ev.get("stage") or "")
                    if kind == "warning":
                        _note(str(ev.get("message", "")))
                        sw.broadcast(ev)
                        return
                    if stage not in ("sam3d_infer", "mhr_postprocess") or kind not in ("stage", "progress"):
                        _on_ev(ev)
                        return
                    _tick("net" if stage == "sam3d_infer" else "post")
                    if kind == "stage":
                        return  # the worker's own wording; the line below is rebuilt from its progress numbers
                    done, total = int(ev.get("done") or 0), max(1, int(ev.get("total") or 1))
                    if stage == "sam3d_infer":
                        part = 0.8 * done / total
                        now = f"{int(round(100.0 * done / total))}%"
                        word = "逐帧估计"
                    else:
                        part = (0.8 if stage_word == "逐帧估计" else 0.0) + (0.2 if stage_word == "逐帧估计" else 1.0) * (done - 1) / total
                        now = f"第 {done}/{total} 人"
                        word = "整段解算"
                    whole = min(1.0, (done_before + batch_size * part) / max(1, job_people))
                    text = f"{word}：共 {job_people} 人"
                    if job_people > 1:
                        text += f"，已完成 {done_before} 人"
                    if batch_count > 1:
                        text += f" · 第 {batch_no}/{batch_count} 批（{batch_size} 人）{now}"
                    else:
                        text += f" · {now}"
                    spent = time.time() - t_work
                    if whole >= 0.03 and spent >= 20.0 and whole < 0.999:
                        text += f"，预计剩余约 {_human_duration(spent * (1.0 - whole) / whole)}"
                    qjob.stage, qjob.done, qjob.total = stage, int(round(1000 * whole)), 1000
                    qjob.message = f"[GPU {qjob.gpu_index}] {text}"
                    sw.update_meta(stage_message=qjob.message, progress={"done": qjob.done, "total": qjob.total, "stage": stage})
                    sw.broadcast({"event": "stage", "stage": stage, "message": text})
                    sw.broadcast({"event": "progress", "stage": stage, "done": qjob.done, "total": qjob.total})

                return handler

            if need_net:
                if solver_frames is None:
                    solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
                passes = [need_net[i : i + PEOPLE_PER_PASS] for i in range(0, len(need_net), PEOPLE_PER_PASS)]
                job_rec["clock"] = time.time()
                for pass_no, group in enumerate(passes, start=1):
                    solve_res = gpu_solver.solve_people(
                        plate=cur_plate,
                        selected_people=group,
                        intrinsics_camera=cur_cam,
                        raw_net_dir=raw_net_dir,
                        cam_space_dir=cam_space_dir,
                        solver_frame_files=solver_frames,
                        hand_refine=hand_refine,
                        use_mask=use_mask,
                        person_masks_npz=masks_npz_for_solve,
                        lock_shape=lock_shape,
                        smoothing=smoothing,
                        fps=shot_fps,
                        scan_npz=scan_npz if has_scan else None,
                        on_event=_people_events("逐帧估计", len(finished), pass_no, len(passes), len(group)),
                        new_id_start=_next_person_id(),
                    )
                    _tick("post")  # a batch ends with the whole-shot solve of its last person
                    fresh.update({int(e["id"]): e for e in solve_res.get("people", [])})
                    finished.update(int(person["id"]) for person in group)
                    _take_in_cuts(solve_res)
                    if pass_no < len(passes) or post_only:
                        _commit(final=False)  # the batch is complete: it stays even if the job is cancelled now
                    if is_cancelled():
                        raise JobCancelledError()
            if post_only:
                job_rec["clock"] = time.time()
                post_res = gpu_solver.postprocess_cached(
                    raw_net_dir=raw_net_dir,
                    cam_space_dir=cam_space_dir,
                    shot_frames=cur_plate.frames,
                    lock_shape=lock_shape,
                    smoothing=smoothing,
                    fps=shot_fps,
                    scan_npz=scan_npz if has_scan else None,
                    cam_to_world={int(f): cam_res_intr.cam_to_world[i] for i, f in enumerate(cam_res_intr.frames)},
                    person_ids=post_only,
                    on_event=_people_events("整段解算", len(finished), 1, 1, len(post_only)),
                    new_id_start=_next_person_id(),
                )
                _tick("post")
                fresh.update({int(e["id"]): e for e in post_res.get("people", [])})
                finished.update(post_only)
                _take_in_cuts(post_res)

            if is_cancelled():
                raise JobCancelledError()

            solved_list, report = _commit(final=True)
            self._clear_export_files(sw)
            n_net, n_post = len(need_net), len(post_only)
            # the results on show are final from here on: a cancel now only stops the other half
            qjob._main_done = True
            also = ""
            if n_net or n_post:
                try:
                    also = _other_half(sorted(fresh))
                except JobCancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - the solve on show is complete and stays; only the other half is missing
                    shutil.rmtree(other_tmp, ignore_errors=True)
                    if is_cancelled():
                        # a cancel ends the worker, which is how its call ends: it is the cancel, not a failure
                        raise JobCancelledError() from exc
                    _note(f"另一状态（对齐前 / 对齐后）的结果没有算出，切换时再算：{type(exc).__name__}: {str(exc)[:160]}")
            if n_net or n_post:
                what = "、".join(x for x in (f"新增 {n_net} 人" if n_net else "", f"重新解算 {n_post} 人" if n_post else "") if x)
                n_skipped = sum(1 for pid in finished if pid not in fresh)
                if n_skipped:
                    # the count of people worked on and the count in the task differ: say why
                    what += f"，其中 {n_skipped} 人因可解算帧数不足已跳过"
                if cut_people:
                    # more people than were asked for: say where they come from
                    what += f"，轨迹拆分出 {len(cut_people)} 人"
                done_msg = f"解算完成：共 {len(solved_list)} 人（本次{what} · GPU {qjob.gpu_index}）{also}"
                # recorded while the task still belongs to this job (nobody can start another solve on it yet)
                job_rec["counts"] = {"net": n_net, "post": n_post, "skipped": n_skipped}
                self._record_job(sw, qjob, "人物解算", "完成")
            else:
                done_msg = f"结果已是最新：共 {len(solved_list)} 人"
            qjob.message = done_msg
            sw.update_meta(
                status="ready",
                active_job_id=None,
                error=None,
                stage_message=done_msg,
            )
            sw.broadcast({"event": "done", "step": "solve", "alignment": report})

        job, pos = self.scheduler.submit(
            shot_id=shot_id,
            shot_name=meta.get("name", shot_id),
            job_type="solve",
            runner=_runner,
            on_state=_state_hook,
            client_ip=client_ip,
        )
        if wait:
            done_event.wait()
            if final_err:
                raise RuntimeError(final_err[0])
        return {"queued": True, "job": job.to_dict(position=pos), "shot": self.enrich_shot_meta(sw.read_meta())}

    def enqueue_object_track(
        self,
        shot_id: str,
        object_prompts: list[dict] | None = None,
        grade: dict | None = None,
        client_ip: str = "",
        wait: bool = False,
    ) -> dict:
        """Object Mode Step 2: Interactive 2D object segmentation & tracking across the sequence via SAM 3.1."""
        sw = self.get_shot(shot_id)
        self._refuse_if_task_has_job(shot_id)
        patch: dict = {"work_mode": "object"}
        if object_prompts is not None:
            patch["object_prompts"] = object_prompts
        if grade is not None:
            patch["grade"] = GradeParams.from_dict(grade).to_dict()
        self.update_settings(shot_id, patch)

        meta = sw.read_meta()
        self._load_plate_info(sw)

        done_event = threading.Event()
        final_err: list[str] = []

        def _state_hook(qjob: QueueJob) -> None:
            if qjob.status == "queued":
                sw.update_meta(status="queued", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "queued", "job": qjob.to_dict()})
            elif qjob.status == "running":
                sw.update_meta(status="running", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "status", "status": "running", "job": qjob.to_dict()})
            elif qjob.status == "cancelled":
                cur = sw.read_meta()
                sw.update_meta(
                    status="ready" if cur.get("has_view") else "idle",
                    active_job_id=None,
                    stage_message="任务已取消",
                )
                sw.broadcast({"event": "cancelled", "job": qjob.to_dict()})
                self._record_job(sw, qjob, "物体跟踪", "已取消")
                done_event.set()
            elif qjob.status == "failed":
                sw.update_meta(status="error", active_job_id=None, error=qjob.error, stage_message=qjob.message)
                sw.broadcast({"event": "error", "error": qjob.error})
                self._record_job(sw, qjob, "物体跟踪", "失败", qjob.error)
                final_err.append(qjob.error or "物体跟踪失败")
                done_event.set()
            elif qjob.status == "completed":
                self._record_job(sw, qjob, "物体跟踪", "完成")
                done_event.set()

        def _runner(gpu_solver: SolverManager, qjob: QueueJob, is_cancelled: Callable[[], bool]) -> None:
            cur_meta = sw.read_meta()
            cur_plate = self._load_plate_info(sw)
            g_obj = GradeParams.from_dict(cur_meta.get("grade"))
            prompts = cur_meta.get("object_prompts") or []

            solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
            if is_cancelled():
                raise JobCancelledError()

            def _on_ev(ev: dict) -> None:
                if is_cancelled():
                    raise JobCancelledError()
                if ev.get("event") == "stage":
                    msg = f"[GPU {qjob.gpu_index}] {ev.get('message', '')}"
                    qjob.stage = ev.get("stage", "")
                    qjob.message = msg
                    sw.update_meta(stage_message=msg)
                elif ev.get("event") == "progress":
                    qjob.done = int(ev["done"])
                    qjob.total = int(ev["total"])
                    qjob.stage = ev.get("stage", "")
                    sw.update_meta(progress={"done": qjob.done, "total": qjob.total, "stage": qjob.stage})
                sw.broadcast(ev)

            track_dir = sw.root / "object_track"
            track_res = gpu_solver.track_objects(
                plate=cur_plate,
                objects_spec=prompts,
                output_dir=track_dir,
                solver_frame_files=solver_frames,
                on_event=_on_ev,
            )
            if is_cancelled():
                raise JobCancelledError()

            tr_sig = _json_hash(
                {
                    "engine": "ocio_v2",
                    "ocio": cur_plate.resolved_colorspace,
                    "view": cur_plate.ocio_view,
                    "grade": g_obj.to_dict(),
                    "prompts": prompts,
                    "frames": len(cur_plate.frames),
                }
            )
            sigs = {**(cur_meta.get("cache_signatures") or {}), "obj_track": tr_sig}
            n_objs = len(track_res.get("objects") or [])
            done_msg = f"SAM 3.1 物体跟踪完成：共跟踪 {n_objs} 个物体"
            qjob.message = done_msg
            sw.update_meta(
                status="ready" if cur_meta.get("has_view") else "idle",
                active_job_id=None,
                error=None,
                stage_message=done_msg,
                object_tracking=track_res,
                cache_signatures=sigs,
            )
            sw.broadcast({"event": "done", "step": "object_track", "objects_count": n_objs})

        job, pos = self.scheduler.submit(
            shot_id=shot_id,
            shot_name=meta.get("name", shot_id),
            job_type="detect",
            runner=_runner,
            on_state=_state_hook,
            client_ip=client_ip,
        )
        if wait:
            done_event.wait()
            if final_err:
                raise RuntimeError(final_err[0])
        return {"queued": True, "job": job.to_dict(position=pos), "shot": self.enrich_shot_meta(sw.read_meta())}

    @staticmethod
    def _refuse_if_objects_not_installed() -> None:
        """Object matching is optional (and Linux only): say so plainly instead of failing inside the worker."""
        paths = get_paths()
        have_recon = (paths.sam3d_objects_weights / "checkpoints" / "pipeline.yaml").is_file() or (
            paths.triposr_weights / "model.ckpt").is_file()
        if not (paths.moge_weights / "model.pt").is_file() or not have_recon:
            raise ValueError("物体匹配没有安装：请重新运行安装程序并选择安装物体匹配（只支持 Linux）。")

    def enqueue_object_solve(
        self,
        shot_id: str,
        params_patch: dict | None = None,
        client_ip: str = "",
        wait: bool = False,
    ) -> dict:
        """Object Mode Step 4: Reconstruct static 3D proxy mesh + solve 6DoF rigid pose with Step 2 camera intrinsics,
        then pre-generate all 4 export files."""
        sw = self.get_shot(shot_id)
        self._refuse_if_task_has_job(shot_id)
        self._refuse_if_objects_not_installed()
        patch: dict = {"work_mode": "object"}
        if params_patch:
            patch.update(params_patch)
        self.update_settings(shot_id, patch)
        meta = sw.read_meta()
        self._load_plate_info(sw)
        if not meta.get("camera") or not (meta.get("camera") or {}).get("file_path"):
            raise ValueError("请先在第 2 步导入摄影机后再开始物体三维与位姿计算")

        done_event = threading.Event()
        final_err: list[str] = []

        def _state_hook(qjob: QueueJob) -> None:
            if qjob.status == "queued":
                sw.update_meta(status="queued", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "queued", "job": qjob.to_dict()})
            elif qjob.status == "running":
                sw.update_meta(status="running", active_job_id=qjob.id, error=None, stage_message=qjob.message)
                sw.broadcast({"event": "status", "status": "running", "job": qjob.to_dict()})
            elif qjob.status == "cancelled":
                cur = sw.read_meta()
                sw.update_meta(
                    status="ready" if cur.get("has_view") else "idle",
                    active_job_id=None,
                    stage_message="任务已取消",
                )
                sw.broadcast({"event": "cancelled", "job": qjob.to_dict()})
                self._record_job(sw, qjob, "物体解算", "已取消")
                done_event.set()
            elif qjob.status == "failed":
                sw.update_meta(status="error", active_job_id=None, error=qjob.error, stage_message=qjob.message)
                sw.broadcast({"event": "error", "error": qjob.error})
                self._record_job(sw, qjob, "物体解算", "失败", qjob.error)
                final_err.append(qjob.error or "物体三维解算失败")
                done_event.set()
            elif qjob.status == "completed":
                self._record_job(sw, qjob, "物体解算", "完成")
                done_event.set()

        def _runner(gpu_solver: SolverManager, qjob: QueueJob, is_cancelled: Callable[[], bool]) -> None:
            cur_meta = sw.read_meta()
            cur_plate = self._load_plate_info(sw)
            cur_cam = self._load_camera_from_entry(cur_meta.get("camera"), cur_meta.get("plate"))
            if cur_cam is None:
                raise ValueError("请先在第 2 步导入摄影机后再开始物体三维与位姿计算")
            g_obj = GradeParams.from_dict(cur_meta.get("grade"))
            prompts = cur_meta.get("object_prompts") or []
            osp = cur_meta.get("object_solve_params") or {}
            smoothing = float(osp.get("smoothing", 0.5))

            def _on_ev(ev: dict) -> None:
                if is_cancelled():
                    raise JobCancelledError()
                if ev.get("event") == "stage":
                    msg = f"[GPU {qjob.gpu_index}] {ev.get('message', '')}"
                    qjob.stage = ev.get("stage", "")
                    qjob.message = msg
                    sw.update_meta(stage_message=msg)
                elif ev.get("event") == "progress":
                    qjob.done = int(ev["done"])
                    qjob.total = int(ev["total"])
                    qjob.stage = ev.get("stage", "")
                    sw.update_meta(progress={"done": qjob.done, "total": qjob.total, "stage": qjob.stage})
                sw.broadcast(ev)

            solver_frames = prepare_solver_srgb_frames(cur_plate, g_obj, sw.root / "plate" / "solver_srgb")
            if is_cancelled():
                raise JobCancelledError()

            sigs = dict(cur_meta.get("cache_signatures") or {})
            tr_sig = _json_hash(
                {
                    "engine": "ocio_v2",
                    "ocio": cur_plate.resolved_colorspace,
                    "view": cur_plate.ocio_view,
                    "grade": g_obj.to_dict(),
                    "prompts": prompts,
                    "frames": len(cur_plate.frames),
                }
            )
            track_dir = sw.root / "object_track"
            if sigs.get("obj_track") != tr_sig or not any(track_dir.glob("masks_object_*.npz")):
                track_res = gpu_solver.track_objects(
                    plate=cur_plate,
                    objects_spec=prompts,
                    output_dir=track_dir,
                    solver_frame_files=solver_frames,
                    on_event=_on_ev,
                )
                sigs["obj_track"] = tr_sig
                sw.update_meta(object_tracking=track_res, cache_signatures=sigs)

            if is_cancelled():
                raise JobCancelledError()

            cam_space_dir = sw.root / "solve_objects" / "cam_space"
            solve_res = gpu_solver.solve_objects(
                plate=cur_plate,
                track_dir=track_dir,
                cam_space_dir=cam_space_dir,
                intrinsics_camera=cur_cam,
                solver_frame_files=solver_frames,
                smoothing=smoothing,
                on_event=_on_ev,
            )
            if is_cancelled():
                raise JobCancelledError()

            solved_objs = solve_res.get("objects") or []
            active_oids = [int(o["id"]) for o in solved_objs]
            sigs["obj_solve"] = _json_hash(
                {
                    "tr": tr_sig,
                    "smoothing": round(smoothing, 5),
                    "cam_focal": round(float(np.median(cur_cam.focal_mm)), 4),
                }
            )
            sw.update_meta(
                cache_signatures=sigs,
                solved_objects=solved_objs,
                active_object_ids=active_oids,
                solved_at=_now_iso(),
            )
            report = self._rebuild_preview_if_possible(sw)
            self._clear_export_files(sw)
            done_msg = f"物体三维解算完成（{len(solved_objs)} 个物体 · GPU {qjob.gpu_index}）"
            qjob.message = done_msg
            sw.update_meta(
                status="ready",
                active_job_id=None,
                error=None,
                stage_message=done_msg,
            )
            sw.broadcast({"event": "done", "step": "object_solve", "alignment": report})

        job, pos = self.scheduler.submit(
            shot_id=shot_id,
            shot_name=meta.get("name", shot_id),
            job_type="solve",
            runner=_runner,
            on_state=_state_hook,
            client_ip=client_ip,
        )
        if wait:
            done_event.wait()
            if final_err:
                raise RuntimeError(final_err[0])
        return {"queued": True, "job": job.to_dict(position=pos), "shot": self.enrich_shot_meta(sw.read_meta())}

    def _refuse_if_task_has_job(self, shot_id: str) -> None:
        """A request for a task that already has a job (waiting or running) is refused BEFORE anything of it is
        written.  Up to v0.2.10 the request's settings were saved first and the queue refused it afterwards, so
        the job that was waiting then ran with the settings of the refused request."""
        job = self.scheduler.job_for_shot(shot_id)
        if job is not None:
            raise RuntimeError("该镜头已有任务在队列中等待" if job.get("status") == "queued" else "该镜头正在执行解算中")

    def cancel_shot_task(self, shot_id: str) -> dict:
        self.scheduler.cancel_shot_jobs(shot_id)
        sw = self.get_shot(shot_id)
        return self.enrich_shot_meta(sw.read_meta())

    # ------------------------------------------------------------------ Step 5: Pre-Generated 4-File Instant Export & Download

    @staticmethod
    def _default_export_unit(meta: dict, unit: str = "") -> str:
        req_unit = (unit or "").strip().lower()
        if req_unit in ("m", "cm"):
            return req_unit
        cam_entry = meta.get("camera") or {}
        cam_listing = cam_entry.get("listing") or {}
        cam_unit_cm = float(cam_entry.get("resolved_unit_to_cm") or cam_listing.get("unit_to_cm") or 100.0)
        return "m" if abs(cam_unit_cm - 100.0) < 1e-2 else "cm"

    @staticmethod
    def _safe_export_stem(meta: dict, shot_id: str, prefix: str = "") -> str:
        raw_stem = (prefix or meta.get("name") or shot_id).strip()
        safe_stem = "".join("_" if ch in '/\\:*?"<>|\r\n\t' else ch for ch in raw_stem).strip(" ._") or shot_id
        win_reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if safe_stem.upper() in win_reserved:
            safe_stem = f"_{safe_stem}"
        return safe_stem

    @staticmethod
    def _people_tag(person_ids: list[int], total: int) -> str:
        """File-name tag of a partial export, plain ASCII and different for every choice:
        persons 1 and 3 -> "p1p3", persons 1 to 4 and 7 -> "p1to4p7"; a very long list -> "p23of40_<hash>"."""
        ids = sorted({int(i) for i in person_ids})
        parts: list[str] = []
        i = 0
        while i < len(ids):
            j = i
            while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
                j += 1
            parts.append(f"p{ids[i]}to{ids[j]}" if j - i >= 2 else "".join(f"p{v}" for v in ids[i : j + 1]))
            i = j + 1
        tag = "".join(parts)
        if len(tag) > 40:
            tag = f"p{len(ids)}of{int(total)}_{_json_hash(ids)[:6]}"
        return tag

    @staticmethod
    def _resolve_export_format_spec(safe_stem: str, raw_fmt: str) -> tuple[str, str, str]:
        key = raw_fmt.strip().lower().lstrip(".")
        if key == "usd_skel":
            return ("usd_skel", f"{safe_stem}_skel.usd", "skel")
        if key == "usd_cache":
            return ("usd_cache", f"{safe_stem}_cache.usd", "cache")
        if key == "fbx":
            return ("fbx", f"{safe_stem}.fbx", "skel")
        if key == "abc":
            return ("abc", f"{safe_stem}.abc", "skel")
        if key in ("usdc", "usd", "usda"):
            return (key, f"{safe_stem}.{key}", "skel")
        raise ValueError(f"不支持的导出格式：{raw_fmt}")

    def _compute_export_sig(self, meta: dict, use_unit: str, use_fps: float, safe_stem: str) -> str:
        cam_e = meta.get("camera") or {}
        scan_e = meta.get("scene_scan") or {}
        cam_fp = str(cam_e.get("file_path") or "")
        scan_fp = str(scan_e.get("file_path") or "")
        try:
            cam_mtime = round(Path(cam_fp).stat().st_mtime, 3) if cam_fp and Path(cam_fp).is_file() else 0.0
        except OSError:
            cam_mtime = 0.0
        try:
            scan_mtime = round(Path(scan_fp).stat().st_mtime, 3) if scan_fp and Path(scan_fp).is_file() else 0.0
        except OSError:
            scan_mtime = 0.0
        return _json_hash(
            {
                "v": 3,
                "stem": safe_stem,
                "unit": use_unit,
                "fps": round(use_fps, 3),
                "sigs": meta.get("cache_signatures") or {},
                "solved_at": meta.get("solved_at") or "",
                "active_people": sorted(int(x) for x in (meta.get("active_person_ids") or [])),
                "export_people": self._export_people_of(meta),
                "active_objects": sorted(int(x) for x in (meta.get("active_object_ids") or [])),
                "scan_align": self._scan_align_on(meta),
                "top": EXPORT_TOP_GROUP,
                "cam": {
                    "fp": cam_fp,
                    "mt": cam_mtime,
                    "path": cam_e.get("camera_path", ""),
                    "unit": cam_e.get("unit", ""),
                    "take": cam_e.get("fbx_take", ""),
                    "offset": int(cam_e.get("frame_offset") or 0),
                },
                "scan": {
                    "fp": scan_fp,
                    "mt": scan_mtime,
                    "unit": scan_e.get("unit", ""),
                    "sel": scan_e.get("selected_paths") or [],
                },
            }
        )

    def _reset_people_for_new_plate(self, sw: ShotWorkspace) -> None:
        """A new plate replaces the pictures: what was derived from the old ones goes with them - the tracker's
        boxes and outlines (their numbers mean nothing in the new pictures) and the people solved from them."""
        for name in (
            "boxes.json", "person_masks_sam3.npz", "person_masks_manual.npz",
            "person_masks_manual_seg.npz", "person_masks_solve.npz",
        ):
            (sw.root / "detect" / name).unlink(missing_ok=True)
        self._forget_solved_people(sw, None)
        for sub in ("raw_net", "cam_space"):
            for old_file in (sw.root / "solve_raw" / sub).glob("person_*.npz"):
                old_file.unlink(missing_ok=True)
        self._trim_kept_results(sw, lambda _pid: True)
        self._clear_export_files(sw)

    def _forget_solved_people(self, sw: ShotWorkspace, person_ids: set[int] | None, stage_message: str = "") -> list[int]:
        """Remove the results of the given solved people (None = everybody): result files, cached network outputs,
        their place in the export choice.  The other people are untouched.  Returns the ids removed."""
        meta = sw.read_meta()
        solved = list(meta.get("solved_people") or [])
        drop = {int(sp["id"]) for sp in solved} if person_ids is None else {int(x) for x in person_ids}
        drop &= {int(sp["id"]) for sp in solved} | {
            int(f.stem.split("_")[-1]) for f in (sw.root / "solve_raw" / "cam_space").glob("person_*.npz")
        }
        if not drop:
            return []
        for pid in drop:
            for sub in ("raw_net", "cam_space"):
                (sw.root / "solve_raw" / sub / f"person_{pid:02d}.npz").unlink(missing_ok=True)
        self._trim_kept_results(sw, lambda pid: pid in drop)
        keep = [sp for sp in solved if int(sp["id"]) not in drop]
        keep_ids = [int(sp["id"]) for sp in keep]
        sigs = dict(meta.get("cache_signatures") or {})
        for key in ("people_net", "people_post"):
            sigs[key] = {k: v for k, v in dict(sigs.get(key) or {}).items() if int(k) not in drop}
        chosen = [int(x) for x in (meta.get("export_person_ids") or []) if int(x) in keep_ids]
        patch: dict = {
            "solved_people": keep,
            "active_person_ids": keep_ids,
            "export_person_ids": [] if len(chosen) == len(keep_ids) else chosen,
            "cache_signatures": sigs,
        }
        if stage_message:
            patch["stage_message"] = stage_message
        sw.update_meta(**patch)
        return sorted(drop)

    def _after_aborted_people_job(self, sw: ShotWorkspace) -> None:
        """After a detection or solve job was cancelled or failed: the record of the detection is brought in line
        with the tracker's files, result files of people who never made it into the list (they were being added)
        are removed, and a listed person without a result file leaves the list.  The list, the view and the
        export then always show the same people.  Results that existed before the job stay."""
        try:
            self._sync_detection_record(sw)
        except Exception:  # noqa: BLE001 - tidying up after a failure must never raise a second one
            pass
        try:
            meta = sw.read_meta()
            cam_dir = sw.root / "solve_raw" / "cam_space"
            listed = {int(sp["id"]) for sp in (meta.get("solved_people") or [])}
            missing = {pid for pid in listed if not (cam_dir / f"person_{pid:02d}.npz").is_file()}
            stray = 0
            for sub in ("raw_net", "cam_space"):
                for old_file in (sw.root / "solve_raw" / sub).glob("person_*.npz"):
                    if int(old_file.stem.split("_")[-1]) not in listed:
                        old_file.unlink(missing_ok=True)
                        stray += 1
            self._trim_kept_results(sw, lambda pid: pid not in listed)
            if missing:
                self._forget_solved_people(sw, missing)
            if missing or stray:
                self._rebuild_preview_if_possible(sw)
                self._clear_export_files(sw)
        except Exception:  # noqa: BLE001 - tidying up after a failure must never raise a second one
            pass

    @staticmethod
    def _detect_sig(plate, g_obj, thresh: float) -> str:
        """The picture the people were looked for in (colour pipeline, grade, length)."""
        return _json_hash(
            {
                "engine": "people_tracker_v2",
                "ocio": plate.resolved_colorspace,
                "view": plate.ocio_view,
                "grade": g_obj.to_dict(),
                "threshold": thresh,
                "frames": len(plate.frames),
            }
        )

    def _box_repairs(self, sw: ShotWorkspace, people: list[dict], size: tuple[int, int] | None, run_id: str = "") -> list[dict]:
        """The tracked people with their stretched frames repaired (`box_repair.with_repairs`: a frame whose outline
        took in a piece far from the person).  The repairs depend on the tracker's boxes only; they are kept per
        task and detection file, so a long list is not looked through again on every call.  People without such
        a frame come back unchanged."""
        boxes_path = sw.root / "detect" / "boxes.json"
        try:
            st = boxes_path.stat()
            key = (st.st_mtime_ns, st.st_size, size, str(run_id), len(people))
        except OSError:
            key = None
        cached = self._repair_cache.get(sw.shot_id)
        if key is None or cached is None or cached[0] != key:
            found: dict[int, tuple[dict, dict]] = {}
            for person in people:
                fixed = with_repairs(person, size)
                if fixed is not person and fixed.get("repaired"):
                    found[int(person["id"])] = ({k: fixed["boxes"][k] for k in fixed["repaired"]}, dict(fixed["repaired"]))
            if key is not None:
                self._repair_cache[sw.shot_id] = (key, found)
        else:
            found = cached[1]
        out = []
        for person in people:
            fix = found.get(int(person.get("id", -1)))
            if fix is None or "repaired" in person or person.get("is_manual"):
                out.append(person)
                continue
            boxes = dict(person.get("boxes") or {})
            boxes.update(fix[0])
            out.append({**person, "boxes": boxes, "repaired": dict(fix[1])})
        return out

    def _detection_on_disk(self, sw: ShotWorkspace) -> tuple[dict | None, str]:
        """The tracker's result on disk as (boxes payload, run id), or (None, "") when there is none or when its
        two files do not belong together.  Since v0.2.2 the tracker writes the id of its run into both
        `boxes.json` and the outlines (entry `__run_id__`); files of older versions carry no id and are taken
        as they are.  Since v0.2.12 the people come with their stretched frames repaired (`_box_repairs`)."""
        boxes_path = sw.root / "detect" / "boxes.json"
        if not boxes_path.is_file():
            return None, ""
        try:
            det = json.loads(boxes_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None, ""
        if not isinstance(det, dict):
            return None, ""
        run_id = str(det.get("run_id") or "")
        if run_id and (det.get("people") or []):
            try:
                with np.load(boxes_path.with_name("person_masks_sam3.npz"), allow_pickle=False) as mz:
                    if "__run_id__" not in mz.files or str(mz["__run_id__"]) != run_id:
                        return None, ""
            except Exception:  # noqa: BLE001
                return None, ""
        if det.get("people"):
            plate = sw.read_meta().get("plate") or {}
            size = (int(plate.get("width") or 0), int(plate.get("height") or 0)) if plate else None
            det["people"] = self._box_repairs(sw, list(det["people"]), size if size and size[0] > 0 else None, run_id)
        return det, run_id

    def _sync_detection_record(self, sw: ShotWorkspace) -> None:
        """Bring the task's record of the detection in line with the tracker's files on disk.  The two differ only
        when a job was interrupted at the very moment the tracker finished: then the files are a complete new run
        the task has not recorded (it is taken over like any finished run, which clears the people solved from the
        old numbering), or they do not belong together (they are discarded, so people are looked for again)."""
        meta = sw.read_meta()
        sigs = dict(meta.get("cache_signatures") or {})
        det, run_id = self._detection_on_disk(sw)
        boxes_path = sw.root / "detect" / "boxes.json"
        if det is not None and run_id == str(sigs.get("detect_run") or ""):
            return
        if det is None and not boxes_path.is_file() and not sigs.get("detect"):
            return
        if det is not None and meta.get("plate"):
            plate = self._load_plate_info(sw)
            g_obj = GradeParams.from_dict(meta.get("grade"))
            thresh = float((meta.get("detect_params") or {}).get("threshold", 0.8))
            self._record_detection(sw, det, self._detect_sig(plate, g_obj, thresh), thresh, g_obj, plate)
            return
        for name in ("boxes.json", "person_masks_sam3.npz"):
            (sw.root / "detect" / name).unlink(missing_ok=True)
        sigs.update({"detect": "", "detect_run": ""})
        sigs.pop("person_masks_sam3", None)
        sw.update_meta(detection=None, cache_signatures=sigs)
        stale = self._stale_tracked_solved_ids(meta)
        if stale and self._forget_solved_people(sw, set(stale)):
            self._rebuild_preview_if_possible(sw)
            self._clear_export_files(sw)

    def _record_detection(self, sw: ShotWorkspace, det_data: dict, det_sig: str, thresh: float, g_obj, plate) -> tuple[dict, int]:
        """Store a finished detection run.  The tracker has numbered the people anew, so results solved from the
        previous run belong to numbers that now mean somebody else: they are cleared in the same step (manually
        boxed people carry their own numbers and stay).  Returns (signatures, how many solved people were cleared)."""
        meta = sw.read_meta()
        boxes_path = sw.root / "detect" / "boxes.json"
        # the run id is the tracker's own (it is inside its files); a tracker that writes none leaves it empty
        sigs = {**(meta.get("cache_signatures") or {}), "detect": det_sig, "detect_run": str(det_data.get("run_id") or "")}
        if boxes_path.with_name("person_masks_sam3.npz").is_file():
            sigs["person_masks_sam3"] = det_sig
        sw.update_meta(
            detection={
                **det_data,
                "threshold": thresh,
                "grade": g_obj.to_dict(),
                "ocio_cs": plate.resolved_colorspace,
                "ocio_view": plate.ocio_view,
            },
            cache_signatures=sigs,
        )
        stale = self._stale_tracked_solved_ids(meta)
        gone = self._forget_solved_people(sw, set(stale)) if stale else []
        if gone:
            self._rebuild_preview_if_possible(sw)
            self._clear_export_files(sw)
        return dict(sw.read_meta().get("cache_signatures") or {}), len(gone)

    @staticmethod
    def _stale_tracked_solved_ids(meta: dict) -> list[int]:
        """Solved people that came from the tracker (not from a manual box track)."""
        manual_ids = {int(t.get("id", -1)) for t in (meta.get("manual_people") or [])}
        return [int(sp["id"]) for sp in (meta.get("solved_people") or []) if int(sp["id"]) not in manual_ids]

    def remove_solved_people(self, shot_id: str, person_ids: list[int] | None = None) -> dict:
        """Take solved people out of the task again (all of them when `person_ids` is empty)."""
        sw = self.get_shot(shot_id)
        if sw.read_meta().get("active_job_id"):
            raise ValueError("任务正在计算，请等它完成或取消后再移除人物")
        gone = self._forget_solved_people(sw, {int(x) for x in person_ids} if person_ids else None)
        if gone:
            left = len(sw.read_meta().get("solved_people") or [])
            sw.update_meta(stage_message=f"已移除 {len(gone)} 个已解算人物，还剩 {left} 人")
            self._rebuild_preview_if_possible(sw)
            self._clear_export_files(sw)
        return self.enrich_shot_meta(sw.read_meta())

    def _clear_export_files(self, sw: ShotWorkspace) -> None:
        """Export files are built when they are downloaded and kept only as a cache of that request.  After a new
        solve none of them is valid any more, so they are removed instead of being left to be mistaken for results."""
        exp_dir = sw.root / "exports"
        with sw.export_lock:
            if exp_dir.is_dir():
                for f in exp_dir.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                        except OSError:
                            pass
            sw.update_meta(exports=[], export_cache={})

    @staticmethod
    def _export_people_of(meta: dict) -> list[int]:
        """Solved people chosen for export, in id order; an empty choice means everybody."""
        solved = [int(p["id"]) for p in (meta.get("solved_people") or [])]
        active = {int(x) for x in (meta.get("active_person_ids") or [])} or set(solved)
        chosen = {int(x) for x in (meta.get("export_person_ids") or [])}
        pool = [pid for pid in solved if pid in active]
        picked = [pid for pid in pool if pid in chosen]
        return picked if chosen else pool

    def export_deliverables(
        self,
        shot_id: str,
        formats: list[str],
        unit: str = "",
        fps: float | None = None,
        prefix: str = "",
    ) -> list[dict]:
        sw = self.get_shot(shot_id)
        with sw.export_lock:
            meta = sw.read_meta()
            plate_meta = meta.get("plate")
            if not meta.get("camera"):
                raise ValueError("请先完成第 2 步导入跟踪摄影机后再导出")
            if not self._has_any_solve_cache(sw):
                raise ValueError("请先完成第 3 步人体或物体三维解算后再导出")

            use_unit = self._default_export_unit(meta, unit)
            cam_summary = (meta.get("camera") or {}).get("summary") or {}
            use_fps = float(
                fps
                if fps is not None and fps > 0
                else (plate_meta or {}).get("fps") or cam_summary.get("fps") or DEFAULT_FPS
            )
            safe_stem = self._safe_export_stem(meta, shot_id, prefix=prefix)
            # only some of the solved people: the file says so in its name, so two different choices never
            # end up as two different files of the same name on the artist's disk
            export_people = self._export_people_of(meta)
            solved_ids = [int(p["id"]) for p in (meta.get("solved_people") or [])]
            is_subset = bool(solved_ids) and len(export_people) < len(solved_ids)
            if is_subset:
                if not export_people:
                    raise ValueError("请至少选择一个要导出的人物")
                safe_stem = f"{safe_stem}_{self._people_tag(export_people, len(solved_ids))}"
            specs = [self._resolve_export_format_spec(safe_stem, fmt) for fmt in formats]
            exp_dir = sw.root / "exports"
            exp_dir.mkdir(parents=True, exist_ok=True)

            is_default_fps = fps is None or abs(
                use_fps - float((plate_meta or {}).get("fps") or cam_summary.get("fps") or DEFAULT_FPS)
            ) < 1e-3
            use_cache = (not prefix.strip()) and is_default_fps
            export_sig = self._compute_export_sig(meta, use_unit, use_fps, safe_stem) if use_cache else ""

            slot = dict((meta.get("export_cache") or {}).get(use_unit) or {})
            cached_files: dict[str, dict] = dict(slot.get("files") or {}) if (use_cache and slot.get("sig") == export_sig) else {}

            ready_by_key: dict[str, dict] = {}
            missing_specs: list[tuple[str, str, str]] = []
            for key, fname, usd_mode in specs:
                rec = cached_files.get(key)
                fpath = exp_dir / fname
                if use_cache and rec and rec.get("filename") == fname and fpath.is_file() and fpath.stat().st_size > 0:
                    ready_by_key[key] = rec
                else:
                    missing_specs.append((key, fname, usd_mode))

            if not missing_specs:
                return [ready_by_key[key] for key, _, _ in specs]

            raw_chars = self._load_cam_space_chars(sw, export_people or (meta.get("active_person_ids") or None))
            raw_objs = self._load_cam_space_objects(sw, meta.get("active_object_ids") or None)
            if not raw_chars and not raw_objs:
                raise ValueError("请先完成第 3 步人体或物体三维计算后再导出")

            cam = self._load_camera_from_entry(meta.get("camera"), plate_meta)
            if cam is None:
                raise ValueError("请先完成第 2 步导入摄影机后再导出")

            shot_frames = self._shot_frames(meta, cam, raw_chars, raw_objs)
            plate_w = int((plate_meta or {}).get("width") or cam.width or DEFAULT_WIDTH)
            plate_h = int((plate_meta or {}).get("height") or cam.height or DEFAULT_HEIGHT)

            cam_res = cam.resample_to(shot_frames)
            scan_cache = self._load_cached_scene_scan(sw, meta) if self._scan_align_on(meta) else None
            aligned_chars, aligned_objs = self._align_scene(
                cam_res, plate_w, plate_h, shot_frames, raw_chars, raw_objs, scan_cache
            )

            def _build_one(spec_item: tuple[str, str, str]) -> tuple[str, dict]:
                key, fname, usd_mode = spec_item
                out_file = exp_dir / fname
                export_scene(
                    out_file,
                    characters=aligned_chars,
                    cameras=[cam_res],
                    shot_frames=shot_frames,
                    fps=use_fps,
                    unit=use_unit,
                    rigid_objects=aligned_objs,
                    usd_mode=usd_mode,
                    top_group=EXPORT_TOP_GROUP,
                )
                return (
                    key,
                    {
                        "filename": fname,
                        "format": key,
                        "unit": use_unit,
                        "fps": round(use_fps, 3),
                        "size_bytes": out_file.stat().st_size,
                        "created_at": _now_iso(),
                    },
                )

            if len(missing_specs) == 1 or len(aligned_chars) >= 10:
                built_pairs = [_build_one(sp_it) for sp_it in missing_specs]
            else:
                with ThreadPoolExecutor(max_workers=min(4, len(missing_specs))) as pool:
                    built_pairs = list(pool.map(_build_one, missing_specs))

            for key, rec in built_pairs:
                ready_by_key[key] = rec
                if use_cache:
                    cached_files[key] = rec

            with sw.lock:
                cur_m = sw.read_meta()
                records = list(cur_m.get("exports") or [])
                for _, rec in built_pairs:
                    records = [r for r in records if r.get("filename") != rec["filename"]]
                    records.insert(0, rec)
                patch_meta: dict = {"exports": records}
                if use_cache:
                    cache = dict(cur_m.get("export_cache") or {})
                    cache[use_unit] = {"sig": export_sig, "files": cached_files, "updated_at": _now_iso()}
                    patch_meta["export_cache"] = cache
                sw.update_meta(**patch_meta)

            return [ready_by_key[key] for key, _, _ in specs]

    def export_single_path(self, shot_id: str, fmt: str, unit: str = "") -> Path:
        produced = self.export_deliverables(shot_id, formats=[fmt], unit=unit)
        sw = self.get_shot(shot_id)
        return sw.root / "exports" / produced[0]["filename"]
