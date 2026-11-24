"""Multi-User FIFO Task Queue & Multi-GPU Worker Pool Scheduler (`MultiGpuScheduler`).

Features:
  - Discovers all CUDA GPUs on the machine and creates an independent resident worker slot per GPU (`0`, `1`, ...).
  - Concurrent multi-GPU execution: when multiple GPUs are enabled, multiple artists' tasks execute in parallel!
  - Admin GPU control: individual GPUs can be enabled/disabled dynamically in `/admin`:
      * Disabling an idle GPU immediately offloads/terminates its resident worker so VRAM returns to 0 MB.
      * Disabling ALL GPUs keeps incoming tasks safely in `"queued"` state — NO task starts until at least one
        compatible GPU is re-enabled!
  - Supports cancelling queued jobs and aborting running jobs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import get_paths
from ..server_config import ConfigStore
from .gpu_arch import GpuDeviceInfo, list_gpus
from .manager import SolverManager


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class QueueJob:
    id: str
    shot_id: str
    shot_name: str
    job_type: str  # "detect" | "solve"
    client_ip: str = ""
    status: str = "queued"  # "queued" | "running" | "completed" | "failed" | "cancelled"
    gpu_index: int | None = None
    stage: str = ""
    message: str = "排队等待中..."
    done: int = 0
    total: int = 0
    error: str | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    created_ts: float = field(default_factory=time.time)
    started_ts: float | None = None
    finished_ts: float | None = None

    def to_dict(self, position: int | None = None) -> dict:
        d = asdict(self)
        d.pop("created_ts", None)
        d.pop("started_ts", None)
        d.pop("finished_ts", None)
        if position is not None:
            d["queue_position"] = position
        if self.started_ts is not None:
            end = self.finished_ts if self.finished_ts is not None else time.time()
            d["duration_sec"] = round(max(0.0, end - self.started_ts), 1)
        else:
            d["duration_sec"] = None
        d["wait_sec"] = round(
            max(0.0, (self.started_ts or self.finished_ts or time.time()) - self.created_ts), 1
        )
        return d


class JobCancelledError(RuntimeError):
    pass


class GpuSlot:
    def __init__(self, info: GpuDeviceInfo, enabled: bool) -> None:
        self.info = info
        self.enabled = enabled and info.compatible
        self.solver = SolverManager(default_gpu=str(info.index))
        self.current_job: QueueJob | None = None
        self.last_used_ts: float = 0.0
        self.has_loaded_vram: bool = False

    def stop_worker(self) -> None:
        self.has_loaded_vram = False
        self.solver.shutdown()


def _query_live_gpu_metrics() -> dict[int, dict]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        out = subprocess.check_output(
            [
                smi,
                "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=4,
        )
        res: dict[int, dict] = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                idx = int(parts[0])
                res[idx] = {
                    "memory_used_mb": int(float(parts[1])),
                    "memory_free_mb": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "utilization_pct": int(float(parts[4])) if parts[4].isdigit() else 0,
                    "temperature_c": int(float(parts[5])) if parts[5].isdigit() else 0,
                }
        return res
    except Exception:
        return {}


# Runner signature: runner(gpu_solver: SolverManager, job: QueueJob, is_cancelled: Callable[[], bool]) -> None
JobRunner = Callable[[SolverManager, QueueJob, Callable[[], bool]], None]
JobStateHook = Callable[[QueueJob], None]


class MultiGpuScheduler:
    MAX_HISTORY = 100

    def __init__(self, config_store: ConfigStore, history_file: Path | None = None) -> None:
        self.config_store = config_store
        self.history_file = history_file
        self._cond = threading.Condition()
        self._queue: deque[tuple[QueueJob, JobRunner, JobStateHook | None]] = deque()
        self._running: dict[str, tuple[QueueJob, GpuSlot, JobStateHook | None]] = {}
        self._cancelled_ids: set[str] = set()
        self._history: deque[dict] = deque(maxlen=self.MAX_HISTORY)
        self._slots: dict[int, GpuSlot] = {}
        self._shutdown = False

        self._load_history()
        self._init_slots()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="shotspace-scheduler", daemon=True)
        self._dispatcher.start()

    def _load_history(self) -> None:
        if self.history_file and self.history_file.is_file():
            try:
                items = json.loads(self.history_file.read_text(encoding="utf-8"))
                for it in reversed(items[: self.MAX_HISTORY]):
                    self._history.appendleft(it)
            except Exception:
                pass

    def _save_history_unlocked(self) -> None:
        if not self.history_file:
            return
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.history_file.with_name(f".{self.history_file.name}.tmp")
            tmp.write_text(json.dumps(list(self._history), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.history_file)
        except Exception:
            pass

    def _init_slots(self) -> None:
        paths = get_paths()
        gpus = list_gpus(paths.sam3d_python)
        disabled = set(self.config_store.load().disabled_gpus)
        with self._cond:
            for g in gpus:
                self._slots[g.index] = GpuSlot(g, enabled=(g.index not in disabled) and g.compatible)

    def enabled_gpu_count(self) -> int:
        with self._cond:
            return sum(1 for s in self._slots.values() if s.enabled and s.info.compatible)

    def has_active_or_queued_jobs(self) -> bool:
        with self._cond:
            return bool(self._running or self._queue)

    def switch_history_file(self, new_history_file: Path) -> None:
        with self._cond:
            self.history_file = new_history_file
            self._history.clear()
            self._load_history()

    def total_gpu_count(self) -> int:
        with self._cond:
            return len(self._slots)

    def set_gpu_enabled(self, gpu_index: int, enabled: bool) -> dict:
        to_stop: SolverManager | None = None
        with self._cond:
            slot = self._slots.get(int(gpu_index))
            if slot is None:
                raise KeyError(f"未找到 GPU {gpu_index}")
            if enabled and not slot.info.compatible:
                raise ValueError(f"GPU {gpu_index} ({slot.info.name}) CUDA 架构不兼容，无法启用")
            slot.enabled = bool(enabled)
            disabled_list = [idx for idx, s in self._slots.items() if not s.enabled]
            self.config_store.set_disabled_gpus(disabled_list)
            if not slot.enabled and slot.current_job is None:
                to_stop = slot.solver
            # Update waiting message on queued jobs if all GPUs were just disabled or re-enabled
            any_enabled = any(s.enabled for s in self._slots.values())
            for qjob, _, hook in self._queue:
                qjob.message = (
                    "排队等待空闲显卡..."
                    if any_enabled
                    else "排队等待中（后台已暂停全部显卡，开启显卡后自动开始）"
                )
                if hook:
                    try:
                        hook(qjob)
                    except Exception:
                        pass
            self._cond.notify_all()
        if to_stop is not None:
            to_stop.shutdown()
        return self.overview()

    def set_all_gpus_enabled(self, enabled: bool) -> dict:
        to_stop: list[SolverManager] = []
        with self._cond:
            for slot in self._slots.values():
                if enabled and slot.info.compatible:
                    slot.enabled = True
                elif not enabled:
                    slot.enabled = False
                    if slot.current_job is None:
                        to_stop.append(slot.solver)
            disabled_list = [idx for idx, s in self._slots.items() if not s.enabled]
            self.config_store.set_disabled_gpus(disabled_list)
            any_enabled = any(s.enabled for s in self._slots.values())
            for qjob, _, hook in self._queue:
                qjob.message = (
                    "排队等待空闲显卡..."
                    if any_enabled
                    else "排队等待中（后台已暂停全部显卡，开启显卡后自动开始）"
                )
                if hook:
                    try:
                        hook(qjob)
                    except Exception:
                        pass
            self._cond.notify_all()
        for solver in to_stop:
            solver.shutdown()
        return self.overview()

    def submit(
        self,
        shot_id: str,
        shot_name: str,
        job_type: str,
        runner: JobRunner,
        on_state: JobStateHook | None = None,
        client_ip: str = "",
    ) -> tuple[QueueJob, int]:
        with self._cond:
            # Reject duplicate active/queued job on the exact same shot
            for qj, _, _ in self._queue:
                if qj.shot_id == shot_id:
                    raise RuntimeError("该镜头已有任务在队列中等待")
            for rj, _, _ in self._running.values():
                if rj.shot_id == shot_id:
                    raise RuntimeError("该镜头正在执行解算中")

            any_enabled = any(s.enabled for s in self._slots.values())
            msg = (
                "已加入队列，等待分配显卡..."
                if any_enabled
                else "排队等待中（后台已暂停全部显卡，开启显卡后自动开始）"
            )
            job = QueueJob(
                id=f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}",
                shot_id=shot_id,
                shot_name=shot_name,
                job_type=job_type,
                client_ip=client_ip,
                status="queued",
                message=msg,
            )
            self._queue.append((job, runner, on_state))
            pos = len(self._queue)
            if on_state:
                try:
                    on_state(job)
                except Exception:
                    pass
            self._cond.notify_all()
            return job, pos

    def cancel_job(self, job_id: str) -> bool:
        kill_solver: SolverManager | None = None
        cancelled_job: QueueJob | None = None
        hook_to_call: JobStateHook | None = None

        with self._cond:
            for i, (qj, _, hook) in enumerate(self._queue):
                if qj.id == job_id:
                    self._queue.remove(self._queue[i])
                    qj.status = "cancelled"
                    qj.message = "任务已取消排队"
                    qj.finished_at = _now_iso()
                    qj.finished_ts = time.time()
                    self._history.appendleft(qj.to_dict())
                    self._save_history_unlocked()
                    cancelled_job = qj
                    hook_to_call = hook
                    break

            if cancelled_job is None and job_id in self._running:
                rj, slot, hook = self._running[job_id]
                self._cancelled_ids.add(job_id)
                kill_solver = slot.solver

            self._cond.notify_all()

        if cancelled_job is not None and hook_to_call is not None:
            try:
                hook_to_call(cancelled_job)
            except Exception:
                pass
            return True

        if kill_solver is not None:
            kill_solver.shutdown()
            return True

        return False

    def cancel_shot_jobs(self, shot_id: str) -> bool:
        target_ids: list[str] = []
        with self._cond:
            for qj, _, _ in self._queue:
                if qj.shot_id == shot_id:
                    target_ids.append(qj.id)
            for rj, _, _ in self._running.values():
                if rj.shot_id == shot_id:
                    target_ids.append(rj.id)
        ok = False
        for jid in target_ids:
            ok = self.cancel_job(jid) or ok
        return ok

    def job_for_shot(self, shot_id: str) -> dict | None:
        with self._cond:
            for idx, (qj, _, _) in enumerate(self._queue, start=1):
                if qj.shot_id == shot_id:
                    return qj.to_dict(position=idx)
            for rj, _, _ in self._running.values():
                if rj.shot_id == shot_id:
                    return rj.to_dict(position=0)
        return None

    def public_summary(self) -> dict:
        with self._cond:
            enabled = sum(1 for s in self._slots.values() if s.enabled)
            busy = sum(1 for s in self._slots.values() if s.current_job is not None)
            return {
                "queued_count": len(self._queue),
                "running_count": len(self._running),
                "enabled_gpu_count": enabled,
                "busy_gpu_count": busy,
                "total_gpu_count": len(self._slots),
                "all_gpus_disabled": len(self._slots) > 0 and enabled == 0,
            }

    def overview(self) -> dict:
        live_metrics = _query_live_gpu_metrics()
        with self._cond:
            gpus_out = []
            for idx in sorted(self._slots):
                slot = self._slots[idx]
                m = live_metrics.get(idx, {})
                gpus_out.append(
                    {
                        "index": slot.info.index,
                        "name": slot.info.name,
                        "compute_cap": slot.info.compute_cap,
                        "sm": slot.info.sm,
                        "compatible": slot.info.compatible,
                        "enabled": slot.enabled,
                        "memory_total_mb": m.get("memory_total_mb", slot.info.memory_total_mb),
                        "memory_used_mb": m.get(
                            "memory_used_mb",
                            max(0, slot.info.memory_total_mb - slot.info.memory_free_mb),
                        ),
                        "memory_free_mb": m.get("memory_free_mb", slot.info.memory_free_mb),
                        "utilization_pct": m.get("utilization_pct", 0),
                        "temperature_c": m.get("temperature_c", 0),
                        "state": (
                            "busy"
                            if slot.current_job is not None
                            else ("idle" if slot.enabled else "disabled")
                        ),
                        "current_job": slot.current_job.to_dict(position=0) if slot.current_job else None,
                    }
                )
            queued_out = [qj.to_dict(position=i) for i, (qj, _, _) in enumerate(self._queue, start=1)]
            running_out = [rj.to_dict(position=0) for rj, _, _ in self._running.values()]
            history_out = list(self._history)
            enabled_cnt = sum(1 for s in self._slots.values() if s.enabled)
            return {
                "summary": {
                    "queued_count": len(queued_out),
                    "running_count": len(running_out),
                    "enabled_gpu_count": enabled_cnt,
                    "total_gpu_count": len(self._slots),
                    "all_gpus_disabled": len(self._slots) > 0 and enabled_cnt == 0,
                },
                "gpus": gpus_out,
                "queued": queued_out,
                "running": running_out,
                "history": history_out,
            }

    def record_external_history(self, entry: dict) -> None:
        with self._cond:
            self._history.appendleft(entry)
            self._save_history_unlocked()

    # ------------------------------------------------------------------ Dispatcher Loop

    def _find_free_enabled_slot_unlocked(self) -> GpuSlot | None:
        # Prefer highest-free-memory idle enabled GPU (e.g. RTX 5090 32GB first, then RTX 4090 24GB)
        candidates = [s for s in self._slots.values() if s.enabled and s.info.compatible and s.current_job is None]
        if not candidates:
            return None
        candidates.sort(key=lambda s: s.info.memory_total_mb, reverse=True)
        return candidates[0]

    def _dispatch_loop(self) -> None:
        idle_offload_sec = float(os.environ.get("SHOTSPACE_IDLE_OFFLOAD_SEC", "180"))
        while not self._shutdown:
            idle_to_offload: list[GpuSlot] = []
            with self._cond:
                while not self._shutdown:
                    if self._queue and self._find_free_enabled_slot_unlocked() is not None:
                        break
                    if idle_offload_sec > 0 and not self._queue:
                        now_ts = time.time()
                        for s in self._slots.values():
                            if (
                                s.current_job is None
                                and s.has_loaded_vram
                                and s.last_used_ts > 0
                                and (now_ts - s.last_used_ts) >= idle_offload_sec
                            ):
                                s.has_loaded_vram = False
                                idle_to_offload.append(s)
                        if idle_to_offload:
                            break
                    self._cond.wait(timeout=2.0)
                if self._shutdown:
                    return
                if idle_to_offload:
                    for s_idle in idle_to_offload:
                        threading.Thread(target=s_idle.solver.offload, daemon=True).start()
                    continue
                slot = self._find_free_enabled_slot_unlocked()
                if slot is None or not self._queue:
                    continue
                job, runner, hook = self._queue.popleft()
                job.status = "running"
                job.gpu_index = slot.info.index
                job.started_at = _now_iso()
                job.started_ts = time.time()
                job.message = f"已分配 GPU {slot.info.index} ({slot.info.name})，正在启动..."
                slot.current_job = job
                slot.has_loaded_vram = True
                self._running[job.id] = (job, slot, hook)

                # Notify remaining queued jobs of updated positions
                for qj, _, qhook in self._queue:
                    if qhook:
                        try:
                            qhook(qj)
                        except Exception:
                            pass

            if hook:
                try:
                    hook(job)
                except Exception:
                    pass

            threading.Thread(
                target=self._execute_on_slot,
                args=(slot, job, runner, hook),
                name=f"shotspace-gpu-{slot.info.index}",
                daemon=True,
            ).start()

    def _execute_on_slot(
        self,
        slot: GpuSlot,
        job: QueueJob,
        runner: JobRunner,
        hook: JobStateHook | None,
    ) -> None:
        def _is_cancelled() -> bool:
            with self._cond:
                return job.id in self._cancelled_ids

        try:
            runner(slot.solver, job, _is_cancelled)
            with self._cond:
                if job.id in self._cancelled_ids:
                    job.status = "cancelled"
                    job.message = "任务已被终止"
                else:
                    job.status = "completed"
                    job.message = job.message or "执行完成"
        except Exception as exc:
            with self._cond:
                if job.id in self._cancelled_ids or isinstance(exc, JobCancelledError):
                    job.status = "cancelled"
                    job.error = None
                    job.message = "任务已被管理员或用户终止"
                else:
                    job.status = "failed"
                    job.error = str(exc)
                    job.message = f"失败：{exc}"
        finally:
            should_stop_worker = False
            with self._cond:
                job.finished_at = _now_iso()
                job.finished_ts = time.time()
                self._running.pop(job.id, None)
                self._cancelled_ids.discard(job.id)
                slot.current_job = None
                slot.last_used_ts = time.time()
                if not slot.enabled:
                    slot.has_loaded_vram = False
                    should_stop_worker = True
                self._history.appendleft(job.to_dict())
                self._save_history_unlocked()
                self._cond.notify_all()

            if should_stop_worker:
                slot.solver.shutdown()

            if hook:
                try:
                    hook(job)
                except Exception:
                    pass
