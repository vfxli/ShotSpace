"""Resident SAM 3D Body GPU worker supervisor (`SolverManager`).

Keeps one warm `--serve` worker process alive per selected GPU so consecutive `detect`, `solve`, and
`postprocess` requests avoid cold-loading ViTDet / DINOv3 / MHR (~10-15 s saved per operation), while streaming
structured progress events to FastAPI SSE / CLI listeners.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import DEFAULT_GPU, PROJECT_ROOT, get_paths
from ..core.camera import CameraTrack
from ..io.plate import PlateInfo
from ..io.plate_grade import GradeParams

EVENT_PREFIX = "@SHOTSPACE@"
ProgressCallback = Callable[[dict], None]


class SolverManager:
    def __init__(self, default_gpu: str = DEFAULT_GPU) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._proc_gpu: str | None = None
        self._obj_proc: subprocess.Popen | None = None
        self._obj_proc_gpu: str | None = None
        self.default_gpu = str(default_gpu).strip()

    def _resolve_gpu(self, gpu: str | int | None) -> str:
        if gpu is not None and str(gpu).strip() != "":
            return str(gpu).strip()
        if self.default_gpu:
            return self.default_gpu
        # Default to GPU 0 (or highest-VRAM compatible card if available)
        return "0"

    @staticmethod
    def _offload_proc(proc: subprocess.Popen | None, timeout_sec: float = 20.0) -> None:
        if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
            return
        done = threading.Event()

        def _wait_ack() -> None:
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps({"cmd": "offload"}) + "\n")
                proc.stdin.flush()
                for line in proc.stdout:
                    if line.startswith(EVENT_PREFIX):
                        ev = json.loads(line[len(EVENT_PREFIX) :])
                        if ev.get("event") == "offloaded":
                            break
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=_wait_ack, daemon=True)
        t.start()
        if not done.wait(timeout=timeout_sec):
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass

    def _ensure_worker(self, gpu_id: str) -> subprocess.Popen:
        # Fully terminate object worker before running ViTDet / SAM 3D Body on the same GPU so 0 MB CUDA context leaks
        self._stop_obj_worker_unlocked()
        if self._proc is not None and self._proc.poll() is None and self._proc_gpu == gpu_id:
            return self._proc
        self._stop_worker_unlocked()

        paths = get_paths()
        if not paths.sam3d_python.is_file():
            raise FileNotFoundError(f"未找到 SAM 3D Body Python 环境：{paths.sam3d_python}")
        worker_script = PROJECT_ROOT / "shotspace" / "solver" / "workers" / "sam3d_worker.py"
        env = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MOMENTUM_ENABLED": "0",
            "HF_HUB_OFFLINE": "1",
            "SHOTSPACE_SAM3D_REPO": str(paths.sam3d_repo),
            "SHOTSPACE_DINOV3_DIR": str(paths.dinov3_repo),
            "SHOTSPACE_WEIGHTS_DIR": str(paths.sam3d_weights.parent),
            "SHOTSPACE_PARENT_PID": str(os.getpid()),
        }
        self._proc = subprocess.Popen(
            [str(paths.sam3d_python), str(worker_script), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._proc_gpu = gpu_id
        return self._proc

    def _ensure_obj_worker(self, gpu_id: str) -> subprocess.Popen:
        # Fully terminate ViTDet / SAM 3D Body worker before running SAM 3.1 / 3D Object worker so 24GB GPUs (4090) have 100% free VRAM
        self._stop_worker_unlocked()
        if self._obj_proc is not None and self._obj_proc.poll() is None and self._obj_proc_gpu == gpu_id:
            return self._obj_proc
        self._stop_obj_worker_unlocked()

        paths = get_paths()
        if not paths.sam_obj_python.is_file():
            raise FileNotFoundError(f"未找到物体匹配 Python 环境：{paths.sam_obj_python}")
        worker_script = PROJECT_ROOT / "shotspace" / "solver" / "workers" / "sam_object_worker.py"
        py_paths = [
            str(PROJECT_ROOT),
            str(paths.sam3_repo),
            str(paths.moge_repo),
            str(paths.sam3d_objects_repo),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(py_paths),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1",
            "SHOTSPACE_SAM3_REPO": str(paths.sam3_repo),
            "SHOTSPACE_SAM_OBJ_WEIGHTS_DIR": str(paths.sam_obj_weights),
            "SHOTSPACE_PARENT_PID": str(os.getpid()),
        }
        self._obj_proc = subprocess.Popen(
            [str(paths.sam_obj_python), str(worker_script), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._obj_proc_gpu = gpu_id
        return self._obj_proc

    def _stop_worker_unlocked(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None and self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._proc_gpu = None

    def _stop_obj_worker_unlocked(self) -> None:
        if self._obj_proc is not None:
            try:
                if self._obj_proc.poll() is None and self._obj_proc.stdin:
                    self._obj_proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                    self._obj_proc.stdin.flush()
                    self._obj_proc.wait(timeout=3)
            except Exception:
                try:
                    self._obj_proc.kill()
                except Exception:
                    pass
            self._obj_proc = None
            self._obj_proc_gpu = None

    def shutdown(self) -> None:
        """Terminate or kill worker subprocesses immediately and reap them so GPU VRAM is 100% freed before returning."""
        for proc in (self._proc, self._obj_proc):
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        if self._lock.acquire(blocking=False):
            try:
                self._proc = None
                self._proc_gpu = None
                self._obj_proc = None
                self._obj_proc_gpu = None
            finally:
                self._lock.release()

    def offload(self) -> None:
        with self._lock:
            self._offload_proc(self._proc)
            self._offload_proc(self._obj_proc)

    def _run_job(self, job_payload: dict, gpu: str | int | None = None, on_event: ProgressCallback | None = None) -> dict:
        gpu_id = self._resolve_gpu(gpu)
        with self._lock:
            proc = self._ensure_worker(gpu_id)
            assert proc.stdin is not None and proc.stdout is not None
            with tempfile.TemporaryDirectory(prefix="shotspace_sam3d_") as td:
                job_file = Path(td) / "job.json"
                res_file = Path(td) / "result.json"
                full_job = {**job_payload, "result_file": str(res_file)}
                job_file.write_text(json.dumps(full_job, ensure_ascii=False), encoding="utf-8")

                proc.stdin.write(json.dumps({"cmd": "job", "job": str(job_file)}) + "\n")
                proc.stdin.flush()

                log_tail: list[str] = []
                job_end: dict | None = None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line.startswith(EVENT_PREFIX):
                        ev = json.loads(line[len(EVENT_PREFIX) :])
                        if ev.get("event") == "job_end":
                            job_end = ev
                            break
                        if on_event is not None:
                            on_event(ev)
                    else:
                        if line.strip():
                            log_tail.append(line)
                            if len(log_tail) > 40:
                                log_tail.pop(0)

                if job_end is None:
                    self._stop_worker_unlocked()
                    tail = "\n".join(log_tail[-15:])
                    raise RuntimeError(f"SAM 3D Body 推理进程意外退出：\n{tail}")
                if not job_end.get("ok"):
                    err_msg = job_end.get("error") or "\n".join(log_tail[-10:])
                    raise RuntimeError(f"SAM 3D Body 执行失败：{err_msg}")
                return json.loads(res_file.read_text(encoding="utf-8")) if res_file.is_file() else {}

    def detect_people(
        self,
        plate: PlateInfo,
        output_boxes: Path,
        solver_frame_files: dict[int, str] | None = None,
        grade: GradeParams | None = None,
        threshold: float = 0.8,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> dict:
        fmap = solver_frame_files if solver_frame_files is not None else plate.preview_files
        use_grade = GradeParams() if solver_frame_files is not None else (grade or GradeParams())
        frames_list = [{"frame": int(f), "path": fmap[int(f)]} for f in plate.frames]

        # People are found and identified for the whole shot by SAM 3.1's native video concept tracking
        # (`workers/people_tracker.py`): one mask per person per frame, boxes are the tight bounds of the visible
        # mask. `threshold` (the former ViTDet score threshold) is accepted for compatibility and not used.
        paths = get_paths()
        if not paths.sam_obj_python.is_file() or not (paths.sam31_weights / "sam3.1_multiplex.pt").is_file():
            raise FileNotFoundError(f"未找到 SAM 3.1 人物跟踪所需的环境或权重：{paths.sam31_weights / 'sam3.1_multiplex.pt'}")
        sam3_masks_npz = output_boxes.with_name("person_masks_sam3.npz")
        for stale_name in ("person_masks.npz", "vitdet_raw_dets.npz"):
            output_boxes.with_name(stale_name).unlink(missing_ok=True)
        output_boxes.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._run_obj_job(
                {
                    "command": "track_people",
                    "frames": frames_list,
                    "grade": use_grade.to_dict(),
                    "image_size": [int(plate.width), int(plate.height)],
                    "threshold": float(threshold),
                    "output_boxes": str(output_boxes),
                    "output_masks_npz": str(sam3_masks_npz),
                },
                gpu=gpu,
                on_event=on_event,
            )
        except RuntimeError as exc:
            raise RuntimeError(str(exc).replace("物体解算失败", "人物跟踪失败").replace("物体跟踪与三维解算进程", "人物跟踪进程")) from exc

    def solve_people(
        self,
        plate: PlateInfo,
        selected_people: list[dict],
        intrinsics_camera: CameraTrack | None = None,
        raw_net_dir: Path = Path("."),
        cam_space_dir: Path = Path("."),
        solver_frame_files: dict[int, str] | None = None,
        grade: GradeParams | None = None,
        hand_refine: bool = True,
        use_mask: bool = True,
        person_masks_npz: Path | None = None,
        lock_shape: bool = True,
        smoothing: float = 0.5,
        fps: float = 24.0,
        scan_npz: Path | None = None,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
        new_id_start: int | None = None,
    ) -> dict:
        """`new_id_start`: the first free person number.  With it, a track that jumps from one person to another
        is cut and comes back as several people (`shotspace.solver.track_split`); the result then lists the cuts
        under `splits`.  Without it tracks are solved as they are."""
        fmap = solver_frame_files if solver_frame_files is not None else plate.preview_files
        use_grade = GradeParams() if solver_frame_files is not None else (grade or GradeParams())
        frames_list = [{"frame": int(f), "path": fmap[int(f)]} for f in plate.frames]
        has_real_cam = intrinsics_camera is not None
        if intrinsics_camera is not None:
            cam_res = intrinsics_camera.resample_to(plate.frames)
            k_map = cam_res.intrinsics_per_frame(plate.width, plate.height)
            k_frames = sorted(k_map)
            k_stack = np.stack([k_map[f] for f in k_frames], axis=0)
            f_to_ci = {int(f): idx for idx, f in enumerate(cam_res.frames)}
            c2w_stack = np.stack([cam_res.cam_to_world[f_to_ci[f]] for f in k_frames], axis=0).astype(np.float64)
        else:
            # SAM 3D Body's standard origin camera: f_std = hypot(W, H), principal point = (W/2, H/2)
            k_frames = [int(f) for f in plate.frames]
            f_std = float(np.hypot(float(plate.width), float(plate.height)))
            k_one = np.array(
                [
                    [f_std, 0.0, float(plate.width) * 0.5],
                    [0.0, f_std, float(plate.height) * 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            k_stack = np.repeat(k_one[None], len(k_frames), axis=0)
            c2w_stack = np.repeat(np.eye(4, dtype=np.float64)[None], len(k_frames), axis=0)

        raw_net_dir.mkdir(parents=True, exist_ok=True)
        intr_file = raw_net_dir / "camera_intrinsics.npz"
        np.savez(intr_file, frames=np.asarray(k_frames, dtype=np.int64), K=k_stack, cam_to_world=c2w_stack)

        return self._run_job(
            {
                "command": "solve",
                "frames": frames_list,
                "image_size": [int(plate.width), int(plate.height)],
                "people": selected_people,
                "camera_intrinsics_npz": str(intr_file),
                "has_real_camera": has_real_cam,
                "use_mask": bool(use_mask),
                "person_masks_npz": str(person_masks_npz) if person_masks_npz is not None else None,
                "raw_net_dir": str(raw_net_dir),
                "cam_space_dir": str(cam_space_dir),
                "grade": use_grade.to_dict(),
                "hand_refine": bool(hand_refine),
                "lock_shape": bool(lock_shape),
                "smoothing": float(smoothing),
                "fps": float(fps),
                "scan_npz": str(scan_npz) if scan_npz is not None and Path(scan_npz).is_file() else None,
                "mask_box_pad": 0.12,
                "new_id_start": None if new_id_start is None else int(new_id_start),
            },
            gpu=gpu,
            on_event=on_event,
        )

    def postprocess_cached(
        self,
        raw_net_dir: Path,
        cam_space_dir: Path,
        shot_frames: list[int] | None = None,
        lock_shape: bool = True,
        smoothing: float = 0.5,
        fps: float = 24.0,
        scan_npz: Path | None = None,
        cam_to_world: dict[int, np.ndarray] | None = None,
        person_ids: list[int] | None = None,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
        new_id_start: int | None = None,
    ) -> dict:
        """Re-run the whole-shot solve from the cached per-frame network outputs (seconds per person): used when
        the scan, the camera move or the smoothing changes, so the GPU network never has to run again.
        `person_ids` limits it to those people; the results of everybody else are left untouched."""
        cam_file = None
        if cam_to_world:
            cam_file = Path(raw_net_dir) / "camera_extrinsics_override.npz"
            fr = sorted(int(f) for f in cam_to_world)
            np.savez(cam_file, frames=np.asarray(fr, dtype=np.int64), cam_to_world=np.stack([cam_to_world[f] for f in fr]))
        return self._run_job(
            {
                "command": "postprocess",
                "raw_net_dir": str(raw_net_dir),
                "cam_space_dir": str(cam_space_dir),
                "shot_frames": [int(f) for f in shot_frames] if shot_frames else None,
                "lock_shape": bool(lock_shape),
                "smoothing": float(smoothing),
                "fps": float(fps),
                "scan_npz": str(scan_npz) if scan_npz is not None and Path(scan_npz).is_file() else None,
                "camera_extrinsics_npz": str(cam_file) if cam_file is not None else None,
                "person_ids": [int(x) for x in person_ids] if person_ids is not None else None,
                "new_id_start": None if new_id_start is None else int(new_id_start),
            },
            gpu=gpu,
            on_event=on_event,
        )

    def _run_obj_job(
        self,
        job_payload: dict,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> dict:
        gpu_id = self._resolve_gpu(gpu)
        with self._lock:
            proc = self._ensure_obj_worker(gpu_id)
            assert proc.stdin is not None and proc.stdout is not None
            with tempfile.TemporaryDirectory(prefix="shotspace_samobj_") as td:
                job_file = Path(td) / "job.json"
                res_file = Path(td) / "result.json"
                full_job = {**job_payload, "result_file": str(res_file)}
                job_file.write_text(json.dumps(full_job, ensure_ascii=False), encoding="utf-8")

                proc.stdin.write(json.dumps({"cmd": "job", "job": str(job_file)}) + "\n")
                proc.stdin.flush()

                log_tail: list[str] = []
                job_end: dict | None = None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line.startswith(EVENT_PREFIX):
                        ev = json.loads(line[len(EVENT_PREFIX) :])
                        if ev.get("event") == "job_end":
                            job_end = ev
                            break
                        if on_event is not None:
                            on_event(ev)
                    else:
                        if line.strip():
                            log_tail.append(line)
                            if len(log_tail) > 40:
                                log_tail.pop(0)

                if job_end is None:
                    self._stop_obj_worker_unlocked()
                    tail = "\n".join(log_tail[-15:])
                    raise RuntimeError(f"物体跟踪与三维解算进程意外退出：\n{tail}")
                if not job_end.get("ok"):
                    err_msg = job_end.get("error") or "\n".join(log_tail[-10:])
                    raise RuntimeError(f"物体解算失败：{err_msg}")
                return json.loads(res_file.read_text(encoding="utf-8")) if res_file.is_file() else {}

    def track_objects(
        self,
        plate: PlateInfo,
        objects_spec: list[dict],
        output_dir: Path,
        solver_frame_files: dict[int, str] | None = None,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> dict:
        fmap = solver_frame_files if solver_frame_files is not None else plate.preview_files
        frames_list = [{"frame": int(f), "path": fmap[int(f)]} for f in plate.frames]
        return self._run_obj_job(
            {
                "command": "track_objects",
                "frames": frames_list,
                "image_size": [int(plate.width), int(plate.height)],
                "objects": objects_spec,
                "output_dir": str(output_dir),
            },
            gpu=gpu,
            on_event=on_event,
        )

    def segment_people_masks(
        self,
        plate: PlateInfo,
        selected_people: list[dict],
        output_masks_npz: Path,
        solver_frame_files: dict[int, str] | None = None,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> dict:
        fmap = solver_frame_files if solver_frame_files is not None else plate.preview_files
        frames_list = [{"frame": int(f), "path": fmap[int(f)]} for f in plate.frames]
        return self._run_obj_job(
            {
                "command": "segment_people",
                "frames": frames_list,
                "image_size": [int(plate.width), int(plate.height)],
                "people": selected_people,
                "output_masks_npz": str(output_masks_npz),
            },
            gpu=gpu,
            on_event=on_event,
        )

    def solve_objects(
        self,
        plate: PlateInfo,
        track_dir: Path,
        cam_space_dir: Path,
        intrinsics_camera: CameraTrack | None = None,
        object_ids: list[int] | None = None,
        solver_frame_files: dict[int, str] | None = None,
        smoothing: float = 0.5,
        gpu: str | int | None = None,
        on_event: ProgressCallback | None = None,
    ) -> dict:
        fmap = solver_frame_files if solver_frame_files is not None else plate.preview_files
        frames_list = [{"frame": int(f), "path": fmap[int(f)]} for f in plate.frames]
        cam_focal_px = None
        if intrinsics_camera is not None:
            cam_res = intrinsics_camera.resample_to(plate.frames)
            k_map = cam_res.intrinsics_per_frame(plate.width, plate.height)
            if k_map:
                cam_focal_px = float(np.median([float(m[0, 0]) for m in k_map.values()]))
        return self._run_obj_job(
            {
                "command": "solve_objects",
                "frames": frames_list,
                "image_size": [int(plate.width), int(plate.height)],
                "camera_focal_px": cam_focal_px,
                "track_dir": str(track_dir),
                "cam_space_dir": str(cam_space_dir),
                "object_ids": [int(x) for x in object_ids] if object_ids else None,
                "smoothing": float(smoothing),
            },
            gpu=gpu,
            on_event=on_event,
        )
