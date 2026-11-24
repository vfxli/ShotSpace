"""Standalone SAM 3.1 + Meta `SAM 3D Objects` (`3dfy_v9`) Highest-Precision 3D Object Worker
(`third_party/sam_3d_objects/.venv/bin/python`).

Implements ShotSpace's Production Object Matching Mode (`物体匹配模式`):
  1. `track_objects`: Interactive multi-object 2D segmentation & sequence tracking via `SAM 3.1`
     (`sam3.1_multiplex.pt`) from positive (`label=1`) and negative (`label=0`) viewport clicks on any frame(s).
  2. `solve_objects`:
     - Reconstructs a full-precision, amodally completed Y-up static 3D mesh + metric scale + 3D layout pose
       directly with Meta's genuine `SAM 3D Objects` (`ss_generator` 25-step DiT + `slat_generator` 25-step DiT +
       `slat_decoder_mesh` 256^3 deformable FlexiCubes + `ScaleShiftInvariant` 3D layout pose decoder + `MoGe-ViT-L`).
     - Tracks 6DoF poses via **Rigid-Mesh Surface Z-Buffer Lifting + Keyframe & Sequential FB-LK `solvePnPRansac`**
       with strict inlier spread & angular step verification (zero monocular depth or rotation jitter).
     - Jointly refines `(R_gl, t_gl)` across all reliable frames via **Batched PyTorch GPU Differentiable 2D-3D
       Silhouette Distance-Transform + Contour Chamfer + Unclipped-BBox + 2nd-Order Smoothness Optimization**,
       locking every visible frame's 3D projection directly onto the 2D object silhouette (`< 1 px` bbox error).
     - Strips tiny border-truncated slivers (`visible = 0`) and holds stationary poses at anchor frames so entering/exiting
       objects never fly away.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import ctypes
import errno
import gc
import ipaddress
import json
import math
import platform
import socket
import sys
import tempfile
import threading
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import roma
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

from shotspace.core.camera_space import _fit_cam_translation_to_mask_box, _rotation_between_vectors
from shotspace.core.names import identifier
from shotspace.core.platform_compat import link_file, pid_alive
from shotspace.core.poses import matrix_to_quat, quat_to_matrix
from shotspace.core.smoothing import (
    gaussian_smooth,
    segments,
    smooth_quats,
    smoothing_sigma,
)
from shotspace.core.units import CV_TO_GL, M_TO_CM
from shotspace.io.plate_grade import GradeParams, apply_grade_rgb, build_uint8_lut

EVENT_PREFIX = "@SHOTSPACE@"

_CHANNEL = None
_SINK = None
_MODELS: dict[str, Any] = {}


def _pin_stdout() -> None:
    global _CHANNEL
    if _CHANNEL is None:
        _CHANNEL = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)


def emit(event: str, **payload: Any) -> None:
    out = _CHANNEL or sys.stdout
    out.write(EVENT_PREFIX + json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
    out.flush()


@contextmanager
def quiet():
    global _SINK
    if _SINK is None:
        _SINK = open(os.devnull, "w")
    with redirect_stdout(_SINK):
        yield


def _forbid_network() -> None:
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _is_remote(sock: socket.socket, address) -> str | None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return None
        host = str(address[0])
        try:
            return None if host == "localhost" or ipaddress.ip_address(host.split("%")[0]).is_loopback else host
        except ValueError:
            return host

    def _refused_connect(sock, address):
        if host := _is_remote(sock, address):
            raise ConnectionRefusedError(errno.ECONNREFUSED, f"ShotSpace offline worker blocked connection to {host}")
        return orig_connect(sock, address)

    def _refused_connect_ex(sock, address):
        if _is_remote(sock, address):
            return errno.ECONNREFUSED
        return orig_connect_ex(sock, address)

    socket.socket.connect = _refused_connect
    socket.socket.connect_ex = _refused_connect_ex


def _weights_dir() -> Path:
    env_w = os.environ.get("SHOTSPACE_SAM_OBJ_WEIGHTS_DIR", "").strip()
    if env_w:
        return Path(env_w).resolve()
    return (Path(__file__).resolve().parents[3] / "third_party" / "sam_3d_objects" / "weights").resolve()


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()
    if platform.libc_ver()[0] == "glibc":
        try:
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim(0)
        except OSError:
            pass


def unload_models() -> None:
    from shotspace.solver.workers.sam3d_objects_bridge import move_tree_tensors, stage_pipeline_modules

    sam3_pred = _MODELS.pop("sam3", None)
    if sam3_pred is not None:
        try:
            if hasattr(sam3_pred, "bf16_context") and sam3_pred.bf16_context is not None:
                sam3_pred.bf16_context.__exit__(None, None, None)
        except Exception:
            pass
        try:
            move_tree_tensors(sam3_pred, "cpu")
        except Exception:
            pass
        del sam3_pred
    pipe = _MODELS.pop("sam3d_objects", None)
    _MODELS.pop("moge", None)
    if pipe is not None:
        try:
            stage_pipeline_modules(pipe, "cpu")
        except Exception:
            pass
        del pipe
    for val in list(_MODELS.values()):
        try:
            move_tree_tensors(val, "cpu")
        except Exception:
            pass
    try:
        torch.set_autocast_enabled("cuda", False)
    except Exception:
        pass
    _MODELS.clear()
    _free_memory()


def _get_sam3_predictor():
    if "sam3" in _MODELS:
        return _MODELS["sam3"]
    unload_models()
    wdir = _weights_dir()
    sam3_repo = os.environ.get("SHOTSPACE_SAM3_REPO", "").strip() or str((wdir.parent / "sam3_repo").resolve())
    if sam3_repo and sam3_repo not in sys.path:
        sys.path.insert(0, sam3_repo)
    ckpt = wdir / "sam3.1" / "sam3.1_multiplex.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到 SAM 3.1 权重：{ckpt}")
    emit("stage", stage="load_sam3", message="正在加载跟踪模型（SAM 3.1）...")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    with quiet():
        pred = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(ckpt),
            max_num_objects=12,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        if hasattr(pred, "bf16_context") and pred.bf16_context is not None:
            try:
                pred.bf16_context.__exit__(None, None, None)
            except Exception:
                pass
        # Fix upstream SAM 3.1 OOM on 24GB GPUs (RTX 4090):
        # 1) Force `postprocess_batch_size = 1` so full-frame boolean/float grids never accumulate in multi-frame batches;
        # 2) Patch `tracker.model.init_state` so `offload_state_to_cpu=True` and `offload_video_to_cpu=True` are ALWAYS
        #    honored (`storage_device = cpu`), preventing `N_frames * 27 MB` of spatial memory banks from leaking in VRAM!
        if hasattr(pred, "model"):
            pred.model.postprocess_batch_size = 1
            trk_mod = getattr(getattr(pred.model, "tracker", None), "model", None)
            if trk_mod is not None and hasattr(trk_mod, "init_state") and not getattr(trk_mod, "_shotspace_cpu_offload_patched", False):
                _orig_trk_init_state = trk_mod.init_state

                def _patched_trk_init_state(*args: Any, **kwargs: Any) -> Any:
                    kwargs["offload_video_to_cpu"] = True
                    kwargs["offload_state_to_cpu"] = True
                    return _orig_trk_init_state(*args, **kwargs)

                trk_mod.init_state = _patched_trk_init_state
                trk_mod._shotspace_cpu_offload_patched = True

            # 3) Prevent `Sam3MultiplexTracking._construct_initial_input_batch` -> `recursive_to` from copying the entire
            #    `[N_frames, 3, 1008, 1008]` float32 video tensor onto GPU VRAM (`12.2 MB * N_frames` -> 5 GB on 400 frames!):
            if hasattr(pred.model, "_construct_initial_input_batch") and not getattr(
                pred.model, "_shotspace_img_batch_patched", False
            ):
                _orig_construct_batch = pred.model._construct_initial_input_batch

                class _CpuStreamingFrames:
                    """Opaque indexable CPU fp16 frame store ignored by `recursive_to`, streaming 1 frame at a time to GPU."""

                    def __init__(self, src_images: Any) -> None:
                        if isinstance(src_images, torch.Tensor):
                            self._cpu = src_images.to(device="cpu", dtype=torch.float16)
                        else:
                            self._cpu = src_images

                    def __len__(self) -> int:
                        return len(self._cpu)

                    def __getitem__(self, idx: Any) -> torch.Tensor:
                        item = self._cpu[idx]
                        if isinstance(item, torch.Tensor):
                            return item.to(device="cuda", dtype=torch.float32, non_blocking=True)
                        return item

                def _patched_construct_batch(inference_state: dict, images: Any) -> Any:
                    wrapped = _CpuStreamingFrames(images) if isinstance(images, torch.Tensor) else images
                    return _orig_construct_batch(inference_state, wrapped)

                pred.model._construct_initial_input_batch = _patched_construct_batch
                pred.model._shotspace_img_batch_patched = True
    # 4) With the tracker state kept on the CPU (point 2), upstream crashes with "Expected all tensors to be on the
    #    same device" as soon as its per-frame detector adds or re-conditions an object on a later frame.  The fix
    #    lives next to the people tracker that first measured it; every other use of the model needs it too
    #    (text-prompted objects, silhouettes of manually boxed people).
    from shotspace.solver.workers.people_tracker import patch_cpu_state_merge

    patch_cpu_state_merge(pred)
    _MODELS["sam3"] = pred
    return pred


def _prepare_sam3_frame_dir(
    frame_paths: list[str],
    tmp_seq_dir: Path,
    grade: GradeParams | None = None,
    lut: np.ndarray | None = None,
    max_side: int = 1344,
) -> tuple[int, int]:
    """Write sequence frames for SAM 3.1 capped at `max_side = 1344` (`> 1008` internal SAM 3.1 vision backbone grid).
    Preserves 100% of SAM 3.1's `1008x1008` token accuracy while cutting 2K/4K GPU postprocessing & mask memory by up to 8x!"""
    work_w, work_h = 0, 0
    is_ident = grade is None or grade.is_identity
    for idx, src_p in enumerate(frame_paths):
        src = Path(src_p).resolve()
        ext = src.suffix.lower()
        dst = tmp_seq_dir / f"{idx:05d}.jpg"
        with Image.open(src) as im_raw:
            orig_w, orig_h = im_raw.size
            need_resize = max(orig_w, orig_h) > max_side
            if not need_resize and ext in (".jpg", ".jpeg") and is_ident and src.is_file():
                work_w, work_h = orig_w, orig_h
                link_file(src, dst)
                continue
            rgb = im_raw.convert("RGB")
            if need_resize:
                sc = float(max_side) / float(max(orig_w, orig_h))
                work_w = max(2, int(round(orig_w * sc)))
                work_h = max(2, int(round(orig_h * sc)))
                arr = cv2.resize(np.asarray(rgb, dtype=np.uint8), (work_w, work_h), interpolation=cv2.INTER_AREA)
            else:
                work_w, work_h = orig_w, orig_h
                arr = np.asarray(rgb, dtype=np.uint8)
            if not is_ident and lut is not None:
                arr = cv2.LUT(arr, lut)
            Image.fromarray(arr).save(dst, format="JPEG", quality=94)
    return work_w, work_h


def _close_sam3_session(pred: Any, session_id: str) -> None:
    try:
        pred.handle_request({"type": "close_session", "session_id": session_id, "clear_cache_threshold": 0})
    except Exception:
        pass
    _free_memory()


def _sam3_propagate_all_frames(pred: Any, session_id: str, start_idx: int, n_frames: int, **extra_kwargs: Any):
    """Safely propagate SAM 3.1 across all `0 .. n_frames-1` frames with constant O(1) GPU VRAM:
      1) Prunes multi-keyframe `add_prompt` backbone entries from `feature_cache` before propagation;
      2) Enables `evict_cached_frame_outputs=True` so yielded per-frame masks never accumulate in VRAM;
      3) Selects `forward` / `backward` / `both` cleanly so `start_idx == n_frames - 1` never hits `propagation_fetch` KeyError."""
    try:
        sess = pred._all_inference_states.get(session_id)
        fcache = sess["state"].get("feature_cache") if (sess and isinstance(sess.get("state"), dict)) else None
        if isinstance(fcache, dict):
            for k in [k for k in list(fcache.keys()) if isinstance(k, int)]:
                fcache.pop(k, None)
    except Exception:
        pass
    _free_memory()

    s_clamped = max(0, min(max(0, n_frames - 1), int(start_idx)))
    if n_frames <= 1 or s_clamped <= 0:
        direction = "forward"
        s_clamped = 0
    elif s_clamped >= n_frames - 1:
        direction = "backward"
        s_clamped = n_frames - 1
    else:
        direction = "both"
    return pred.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": s_clamped,
            "evict_cached_frame_outputs": True,
            **extra_kwargs,
        }
    )


def _get_sam3d_objects_pipeline():
    if "sam3d_objects" in _MODELS:
        return _MODELS["sam3d_objects"]
    if "sam3" in _MODELS:
        unload_models()
    wdir = _weights_dir()
    sam_obj_root = wdir.parent
    cfg_file = wdir / "sam-3d-objects" / "checkpoints" / "pipeline.yaml"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"未找到 Meta SAM 3D Objects 权重配置：{cfg_file}")
    emit("stage", stage="load_sam3d_objects", message="正在加载 Meta SAM 3D Objects 最高精度三维重建与空间位姿模型...")
    from shotspace.solver.workers.sam3d_objects_bridge import load_sam3d_objects_pipeline

    with quiet():
        pipeline = load_sam3d_objects_pipeline(sam_obj_root, device="cuda")
    _MODELS["sam3d_objects"] = pipeline
    _MODELS["moge"] = pipeline.depth_model.model
    return pipeline


def _get_moge_model():
    if "sam3d_objects" in _MODELS:
        return _MODELS["sam3d_objects"].depth_model.model
    if "moge" in _MODELS:
        return _MODELS["moge"]
    pipeline = _get_sam3d_objects_pipeline()
    return pipeline.depth_model.model


# ------------------------------------------------------------------ Mask Cleaning, Packing & Contours


def _clean_binary_mask(
    mask_bool: np.ndarray,
    min_area: int = 20,
    ref_pts: list[tuple[float, float]] | None = None,
    prev_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Keep the target object's main connected component (plus only紧邻 <=14px occlusion-split pieces),
    100% rejecting stray false-positive islands on neighboring objects."""
    if not np.any(mask_bool):
        return mask_bool
    labeled, n_cc = cc_label(mask_bool)
    if n_cc <= 1:
        return mask_bool if int(mask_bool.sum()) >= min_area else np.zeros_like(mask_bool, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    max_c = int(counts.max())
    if max_c < min_area:
        return np.zeros_like(mask_bool, dtype=bool)

    cand_ids = [int(cid) for cid in np.flatnonzero(counts >= min_area)]
    if not cand_ids:
        return np.zeros_like(mask_bool, dtype=bool)

    comp_boxes: dict[int, tuple[int, int, int, int]] = {}
    comp_cents: dict[int, tuple[float, float]] = {}
    for cid in cand_ids:
        ys, xs = np.nonzero(labeled == cid)
        comp_boxes[cid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        comp_cents[cid] = (float(xs.mean()), float(ys.mean()))

    h, w = mask_bool.shape[:2]
    main_id = int(np.argmax(counts))

    if ref_pts:
        hit_ids = []
        for rx, ry in ref_pts:
            xi = int(np.clip(round(rx), 0, w - 1))
            yi = int(np.clip(round(ry), 0, h - 1))
            lid = int(labeled[yi, xi])
            if lid in comp_boxes:
                hit_ids.append(lid)
        if hit_ids:
            main_id = max(hit_ids, key=lambda cid: int(counts[cid]))
        else:
            rx0, ry0 = ref_pts[0]
            main_id = min(
                cand_ids,
                key=lambda cid: math.hypot(comp_cents[cid][0] - rx0, comp_cents[cid][1] - ry0),
            )
    elif prev_mask is not None and np.any(prev_mask):
        pm_u8 = prev_mask.astype(np.uint8)
        pm_dil = cv2.dilate(pm_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1) > 0
        pys, pxs = np.nonzero(prev_mask)
        pcx, pcy = float(pxs.mean()), float(pys.mean())
        viable = [cid for cid in cand_ids if int(counts[cid]) >= max(min_area, int(0.10 * max_c))] or cand_ids
        main_id = max(
            viable,
            key=lambda cid: (
                float(np.logical_and(labeled == cid, pm_dil).sum()) * 4.0
                + float(counts[cid]) * 0.35
                - math.hypot(comp_cents[cid][0] - pcx, comp_cents[cid][1] - pcy) * 12.0
            ),
        )

    mx1, my1, mx2, my2 = comp_boxes[main_id]
    keep_ids = [main_id]
    for cid in cand_ids:
        if cid == main_id:
            continue
        if int(counts[cid]) < max(min_area, int(0.10 * int(counts[main_id]))):
            continue
        cx1, cy1, cx2, cy2 = comp_boxes[cid]
        gap_x = max(0, mx1 - cx2, cx1 - mx2)
        gap_y = max(0, my1 - cy2, cy1 - my2)
        if math.hypot(gap_x, gap_y) <= 14.0:
            keep_ids.append(cid)
    return np.isin(labeled, keep_ids)


def _pack_masks(masks_bool: np.ndarray) -> dict[str, Any]:
    m = np.asarray(masks_bool, dtype=np.uint8)
    packed = np.packbits(m, axis=-1)
    return {"packed": packed, "shape": np.asarray(m.shape, dtype=np.int64)}


def _unpack_masks(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    n, h, w = [int(x) for x in shape]
    unpacked = np.unpackbits(packed, axis=-1)[..., :w]
    return unpacked.reshape(n, h, w).astype(bool)


def _extract_mask_contours(mask_bool: np.ndarray, max_contours: int = 3) -> list[list[list[float]]]:
    m_u8 = (np.asarray(mask_bool, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    polys: list[list[list[float]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 12.0:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.4, 0.004 * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) >= 3:
            pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
            polys.append(pts)
    return polys


def _sample_contour_points_2d(mask_bool: np.ndarray, n_pts: int = 128) -> np.ndarray:
    """Uniformly sample `n_pts` `(u, v)` boundary coordinates along the exterior contour of `mask_bool`."""
    m_u8 = (np.asarray(mask_bool, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((n_pts, 2), dtype=np.float32)
    all_pts = np.concatenate([c.reshape(-1, 2) for c in contours if len(c) >= 4], axis=0)
    if len(all_pts) == 0:
        all_pts = contours[0].reshape(-1, 2)
    idx = np.linspace(0, len(all_pts) - 1, n_pts, dtype=np.int64)
    return all_pts[idx].astype(np.float32)


def _mask_bbox_xyxy(mask_bool: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


# ------------------------------------------------------------------ Stage 1: SAM 3.1 Interactive Object Tracking


def run_track_objects(job: dict) -> dict:
    frames_spec = job["frames"]
    shot_frames = [int(it["frame"]) for it in frames_spec]
    frame_paths = [str(it["path"]) for it in frames_spec]
    img_w, img_h = [int(x) for x in job["image_size"]]
    objects_spec = job.get("objects") or []
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not objects_spec:
        raise ValueError("请先在画面中点击选择至少一个物体")

    valid_objs = []
    for obj in objects_spec:
        pts = [p for p in (obj.get("points") or []) if int(p.get("label", 1)) in (0, 1)]
        if pts and any(int(p.get("label", 1)) == 1 for p in pts):
            valid_objs.append(
                {
                    "id": int(obj["id"]),
                    "name": str(obj.get("name") or f"object_{int(obj['id']):02d}"),
                    "points": pts,
                }
            )
    if not valid_objs:
        raise ValueError("每个待跟踪物体至少需要一个左键包含点（正样本点）")

    frame_to_idx = {f: i for i, f in enumerate(shot_frames)}
    n_frames = len(shot_frames)

    pred = _get_sam3_predictor()
    emit("stage", stage="sam3_track", message=f"正在使用 SAM 3.1 跟踪 {len(valid_objs)} 个物体（共 {n_frames} 帧）...")

    with tempfile.TemporaryDirectory(prefix="shotspace_sam3_seq_") as tmp_seq:
        tmp_seq_dir = Path(tmp_seq)
        _prepare_sam3_frame_dir(frame_paths, tmp_seq_dir, max_side=1344)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sid = pred.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(tmp_seq_dir),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": True,
                }
            )["session_id"]
            try:
                seed_indices: list[int] = []
                for obj in valid_objs:
                    oid = int(obj["id"])
                    by_frame: dict[int, list[dict]] = {}
                    for pt in obj["points"]:
                        f_num = int(pt["frame"])
                        f_idx = frame_to_idx.get(f_num)
                        if f_idx is None:
                            f_idx = min(range(n_frames), key=lambda k: abs(shot_frames[k] - f_num))
                        by_frame.setdefault(f_idx, []).append(pt)

                    for f_idx, f_pts in sorted(by_frame.items()):
                        seed_indices.append(f_idx)
                        rel_pts = [
                            [
                                float(np.clip(float(p["x"]) / max(img_w, 1), 0.0, 1.0)),
                                float(np.clip(float(p["y"]) / max(img_h, 1), 0.0, 1.0)),
                            ]
                            for p in f_pts
                        ]
                        labels = [int(p.get("label", 1)) for p in f_pts]
                        pred.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": sid,
                                "frame_index": int(f_idx),
                                "points": rel_pts,
                                "point_labels": labels,
                                "clear_old_points": True,
                                "obj_id": oid,
                            }
                        )

                start_idx = seed_indices[len(seed_indices) // 2] if seed_indices else 0
                raw_per_obj: dict[int, dict[str, Any]] = {
                    int(o["id"]): {
                        "masks": np.zeros((n_frames, img_h, img_w), dtype=bool),
                        "probs": np.zeros(n_frames, dtype=np.float32),
                    }
                    for o in valid_objs
                }

                seen_frames: set[int] = set()
                for resp in _sam3_propagate_all_frames(pred, sid, int(start_idx), n_frames):
                    f_idx = int(resp["frame_index"])
                    outs = resp["outputs"]
                    out_ids = [int(x) for x in np.asarray(outs["out_obj_ids"]).tolist()]
                    out_probs = np.asarray(outs["out_probs"], dtype=np.float32)
                    out_masks = np.asarray(outs["out_binary_masks"], dtype=bool)
                    for k, oid in enumerate(out_ids):
                        if oid not in raw_per_obj:
                            continue
                        m_k = out_masks[k]
                        if m_k.shape != (img_h, img_w):
                            m_k = cv2.resize(m_k.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0
                        if int(m_k.sum()) >= 20:
                            raw_per_obj[oid]["masks"][f_idx] = m_k
                            raw_per_obj[oid]["probs"][f_idx] = float(out_probs[k])
                    seen_frames.add(f_idx)
                    emit("progress", stage="sam3_track", done=min(len(seen_frames), n_frames), total=n_frames)
            finally:
                _close_sam3_session(pred, sid)
                try:
                    torch.set_autocast_enabled("cuda", False)
                except Exception:
                    pass

    # Temporally coherent connected-component cleaning outward from each object's seed click frame(s)
    for obj in valid_objs:
        oid = int(obj["id"])
        masks_arr = raw_per_obj[oid]["masks"]
        probs_arr = raw_per_obj[oid]["probs"]
        pos_by_fidx: dict[int, list[tuple[float, float]]] = {}
        for pt in obj["points"]:
            if int(pt.get("label", 1)) != 1:
                continue
            f_num = int(pt["frame"])
            fi = frame_to_idx.get(f_num)
            if fi is None:
                fi = min(range(n_frames), key=lambda k: abs(shot_frames[k] - f_num))
            pos_by_fidx.setdefault(fi, []).append((float(pt["x"]), float(pt["y"])))

        pivot_fi = sorted(pos_by_fidx.keys())[len(pos_by_fidx) // 2] if pos_by_fidx else start_idx
        if np.any(masks_arr[pivot_fi]):
            masks_arr[pivot_fi] = _clean_binary_mask(masks_arr[pivot_fi], min_area=20, ref_pts=pos_by_fidx.get(pivot_fi))
            if not np.any(masks_arr[pivot_fi]):
                probs_arr[pivot_fi] = 0.0

        prev_m = masks_arr[pivot_fi] if np.any(masks_arr[pivot_fi]) else None
        for fi in range(pivot_fi + 1, n_frames):
            if not np.any(masks_arr[fi]):
                continue
            masks_arr[fi] = _clean_binary_mask(
                masks_arr[fi], min_area=20, ref_pts=pos_by_fidx.get(fi), prev_mask=prev_m
            )
            if np.any(masks_arr[fi]):
                prev_m = masks_arr[fi]
            else:
                probs_arr[fi] = 0.0

        prev_m = masks_arr[pivot_fi] if np.any(masks_arr[pivot_fi]) else None
        for fi in range(pivot_fi - 1, -1, -1):
            if not np.any(masks_arr[fi]):
                continue
            masks_arr[fi] = _clean_binary_mask(
                masks_arr[fi], min_area=20, ref_pts=pos_by_fidx.get(fi), prev_mask=prev_m
            )
            if np.any(masks_arr[fi]):
                prev_m = masks_arr[fi]
            else:
                probs_arr[fi] = 0.0

    for old_f in out_dir.glob("masks_object_*.npz"):
        old_f.unlink(missing_ok=True)

    tracked_summary: list[dict] = []
    for obj in valid_objs:
        oid = int(obj["id"])
        oname = identifier(obj["name"] or f"object_{oid:02d}")
        masks_arr = raw_per_obj[oid]["masks"]
        probs_arr = raw_per_obj[oid]["probs"]
        present_arr = masks_arr.reshape(n_frames, -1).sum(axis=1) >= 20

        boxes_dict: dict[str, list[float]] = {}
        contours_dict: dict[str, list[list[list[float]]]] = {}
        boxes_arr = np.full((n_frames, 4), -1.0, dtype=np.float32)
        for i, f_num in enumerate(shot_frames):
            if not present_arr[i]:
                continue
            bx = _mask_bbox_xyxy(masks_arr[i])
            if bx is None:
                present_arr[i] = False
                continue
            boxes_arr[i] = bx
            boxes_dict[str(f_num)] = [round(v, 1) for v in bx]
            cnts = _extract_mask_contours(masks_arr[i])
            if cnts:
                contours_dict[str(f_num)] = cnts

        packed_info = _pack_masks(masks_arr)
        npz_file = out_dir / f"masks_object_{oid:02d}.npz"
        np.savez_compressed(
            npz_file,
            object_id=np.int64(oid),
            name=np.array(oname),
            frames=np.asarray(shot_frames, dtype=np.int64),
            present=present_arr.astype(np.uint8),
            probs=probs_arr.astype(np.float32),
            boxes=boxes_arr,
            packed_masks=packed_info["packed"],
            mask_shape=packed_info["shape"],
        )
        valid_f = [shot_frames[i] for i in range(n_frames) if present_arr[i]]
        tracked_summary.append(
            {
                "id": oid,
                "name": oname,
                "frame_count": len(valid_f),
                "first_frame": int(valid_f[0]) if valid_f else shot_frames[0],
                "last_frame": int(valid_f[-1]) if valid_f else shot_frames[-1],
                "boxes": boxes_dict,
                "contours": contours_dict,
                "points": obj["points"],
            }
        )

    summary_payload = {"frames": shot_frames, "image_size": [img_w, img_h], "objects": tracked_summary}
    (out_dir / "track_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


# Outlines of manually boxed people (`run_segment_people`)
MANUAL_MAX_SEEDS = 24  # keyframes of one person given to the tracker as box prompts
MANUAL_BOX_GROW = 0.15  # the outline is kept inside the artist's box grown by this share of its size
MANUAL_PEOPLE_PER_SESSION = 8  # the interactive predictor is built for 12 objects per session


def run_segment_people(job: dict) -> dict:
    """Outlines for manually boxed people: SAM 3.1's interactive tracker, seeded with the artist's own boxes.

    Each person is one tracked object.  The two corners of the artist's box on a keyframe are given to the tracker
    as a box prompt (point labels 2 and 3), on several keyframes spread over the person's range, and the tracker
    carries the outline through the frames in between.  Measured on six people of Panoptic pizza1 (120 frames)
    against the whole-shot tracker's outlines of the same people: mask IoU 0.98, 7.6 GB of VRAM - whereas the
    text-prompted pass this replaces ran the detector on every frame and overflowed a 24 GB card.
    Job: frames [{frame, path}], image_size [w, h], people [{id, boxes {frame: xyxy}, key_frames [frame, ...]}],
    output_masks_npz (same `p{id}_f{frame}_{bits,shape,rect}` layout the whole-shot tracker writes)."""
    frames_spec = job["frames"]
    shot_frames = [int(it["frame"]) for it in frames_spec]
    frame_paths = [str(it["path"]) for it in frames_spec]
    img_w, img_h = [int(x) for x in job["image_size"]]
    people_spec = job.get("people") or []
    out_npz = Path(job["output_masks_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(shot_frames)
    frame_to_idx = {f: i for i, f in enumerate(shot_frames)}

    person_boxes: dict[int, dict[int, list[float]]] = {}
    person_keys: dict[int, list[int]] = {}
    for p in people_spec:
        pid = int(p["id"])
        bmap = {int(fk): [float(x) for x in bv] for fk, bv in (p.get("boxes") or {}).items() if int(fk) in frame_to_idx}
        if not bmap:
            continue
        person_boxes[pid] = bmap
        boxed = sorted(bmap)
        keys = sorted({int(f) for f in (p.get("key_frames") or []) if int(f) in bmap})
        if not keys:
            # no keyframe list (older callers): the first and last boxed frame and one every second in between
            keys = sorted(set(boxed[::24]) | {boxed[0], boxed[-1]})
        if len(keys) > MANUAL_MAX_SEEDS:
            pick = np.unique(np.round(np.linspace(0, len(keys) - 1, MANUAL_MAX_SEEDS)).astype(int))
            keys = [keys[i] for i in pick]
        person_keys[pid] = keys

    if not person_boxes or n_frames == 0:
        np.savez_compressed(out_npz)
        return {"masked_pairs": 0}

    pred = _get_sam3_predictor()
    emit(
        "stage",
        stage="sam3_person_mask",
        message=f"正在按手动框提取人物轮廓（{len(person_boxes)} 人，{n_frames} 帧）...",
    )

    packed_archive: dict[str, np.ndarray] = {}
    stored = 0

    def _store(pid: int, fnum: int, mask_bool: np.ndarray) -> bool:
        """Keep the outline inside the artist's box grown by MANUAL_BOX_GROW (an interpolated box lags a moving
        person a little; anything further out is somebody else)."""
        bx = person_boxes[pid][fnum]
        gw, gh = MANUAL_BOX_GROW * (bx[2] - bx[0]), MANUAL_BOX_GROW * (bx[3] - bx[1])
        x1 = max(0, min(img_w - 1, int(math.floor(bx[0] - gw))))
        y1 = max(0, min(img_h - 1, int(math.floor(bx[1] - gh))))
        x2 = max(x1 + 1, min(img_w, int(math.ceil(bx[2] + gw))))
        y2 = max(y1 + 1, min(img_h, int(math.ceil(bx[3] + gh))))
        crop = mask_bool[y1:y2, x1:x2]
        if crop.size == 0 or int(crop.sum()) < 24:
            return False
        crop = _clean_binary_mask(crop, min_area=20)
        if int(crop.sum()) < 24:
            return False
        ch, cw = crop.shape
        if max(ch, cw) > 512:
            sc = 512.0 / float(max(ch, cw))
            nh, nw = max(1, int(round(ch * sc))), max(1, int(round(cw * sc)))
            crop_u8 = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            crop_u8 = crop.astype(np.uint8)
        packed_archive[f"p{pid}_f{fnum}_bits"] = np.packbits(crop_u8.reshape(-1))
        packed_archive[f"p{pid}_f{fnum}_shape"] = np.asarray(crop_u8.shape, dtype=np.int16)
        packed_archive[f"p{pid}_f{fnum}_rect"] = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        return True

    pids = sorted(person_boxes)
    groups = [pids[i : i + MANUAL_PEOPLE_PER_SESSION] for i in range(0, len(pids), MANUAL_PEOPLE_PER_SESSION)]
    done_frames = 0
    total_frames = n_frames * len(groups)
    with tempfile.TemporaryDirectory(prefix="shotspace_sam3_people_") as tmp_seq:
        tmp_seq_dir = Path(tmp_seq)
        _prepare_sam3_frame_dir(frame_paths, tmp_seq_dir, max_side=1344)
        for group in groups:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sid = pred.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(tmp_seq_dir),
                        "offload_video_to_cpu": True,
                        "offload_state_to_cpu": True,
                    }
                )["session_id"]
                try:
                    seed_idx: list[int] = []
                    for pid in group:
                        for fnum in person_keys[pid]:
                            x1, y1, x2, y2 = person_boxes[pid][fnum]
                            pts = [
                                [float(np.clip(x1 / max(img_w, 1), 0.0, 1.0)), float(np.clip(y1 / max(img_h, 1), 0.0, 1.0))],
                                [float(np.clip(x2 / max(img_w, 1), 0.0, 1.0)), float(np.clip(y2 / max(img_h, 1), 0.0, 1.0))],
                            ]
                            pred.handle_request(
                                {
                                    "type": "add_prompt",
                                    "session_id": sid,
                                    "frame_index": int(frame_to_idx[fnum]),
                                    "points": pts,
                                    "point_labels": [2, 3],  # the two corners of a box prompt
                                    "clear_old_points": True,
                                    "obj_id": int(pid),
                                }
                            )
                            seed_idx.append(frame_to_idx[fnum])
                    pivot_idx = sorted(seed_idx)[len(seed_idx) // 2] if seed_idx else 0
                    seen: set[int] = set()
                    for resp in _sam3_propagate_all_frames(pred, sid, int(pivot_idx), n_frames):
                        f_idx = int(resp["frame_index"])
                        fnum = shot_frames[f_idx]
                        outs = resp["outputs"]
                        out_ids = [int(x) for x in np.asarray(outs["out_obj_ids"]).tolist()]
                        out_masks = np.asarray(outs["out_binary_masks"], dtype=bool)
                        for k_i, oid in enumerate(out_ids):
                            if oid not in person_boxes or fnum not in person_boxes[oid]:
                                continue
                            mk = out_masks[k_i]
                            if mk.shape != (img_h, img_w):
                                mk = cv2.resize(mk.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0
                            if _store(oid, fnum, mk):
                                stored += 1
                        if f_idx not in seen:
                            seen.add(f_idx)
                            emit("progress", stage="sam3_person_mask", done=min(done_frames + len(seen), total_frames), total=total_frames)
                    done_frames += n_frames
                finally:
                    _close_sam3_session(pred, sid)
                    try:
                        torch.set_autocast_enabled("cuda", False)
                    except Exception:
                        pass

    tmp_npz = out_npz.with_name(f".{out_npz.stem}.tmp.npz")
    np.savez_compressed(tmp_npz, **packed_archive)
    tmp_npz.replace(out_npz)
    return {"masked_pairs": int(stored)}


def _ray_look_rotation_gl(cx_px: float, cy_px: float, img_w: int, img_h: int, f_std: float) -> np.ndarray:
    """Build SO(3) rotation `R_gl` (`det = +1`) so canonical `-Z_local` aligns with the viewing ray to `(cx_px, cy_px)`
    and `+Y_local` stays upright in image space."""
    fwd = np.array([(cx_px - img_w * 0.5) / f_std, -(cy_px - img_h * 0.5) / f_std, -1.0], dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    z_ax = -fwd  # +Z_local points back toward the camera
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_ax = np.cross(up_world, z_ax)
    if np.linalg.norm(x_ax) < 1e-6:
        x_ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    return np.column_stack([x_ax, y_ax, z_ax])


def _detect_reliable_object_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    present: np.ndarray,
    probs: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, int]:
    """Identify reliable object frames (stripping tiny border-clipped slivers < 32% of interior size so entering/exiting
    objects never fly away) and pick the highest-quality interior anchor frame."""
    n = len(present)
    valid_idx = np.flatnonzero(present)
    if len(valid_idx) == 0:
        raise ValueError("所选物体未跟踪到有效掩码帧，请补充点击正样本点后重新跟踪")

    margin = max(6.0, 0.012 * min(img_w, img_h))
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.float64)
    bw = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    bh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])

    touches_border = np.zeros(n, dtype=bool)
    touches_border[valid_idx] = (
        (boxes[valid_idx, 0] <= margin)
        | (boxes[valid_idx, 1] <= margin)
        | (boxes[valid_idx, 2] >= img_w - margin)
        | (boxes[valid_idx, 3] >= img_h - margin)
    )

    interior_idx = valid_idx[~touches_border[valid_idx]]
    ref_idx = interior_idx if len(interior_idx) >= 2 else valid_idx
    med_area = float(np.median(areas[ref_idx]))
    med_bw = float(np.median(bw[ref_idx]))
    med_bh = float(np.median(bh[ref_idx]))

    reliable = present.copy()
    if len(interior_idx) >= 1:
        for idx in valid_idx:
            if touches_border[idx]:
                # Strip clipped entering/exiting border masks immediately (< 85% area or < 86% width/height)
                # so clipped masks never cause 2–3 frames of apparent depth sliding right before exiting!
                if areas[idx] < 0.85 * med_area or bw[idx] < 0.86 * med_bw or bh[idx] < 0.86 * med_bh:
                    reliable[idx] = False
            else:
                # Strip spurious tiny false-positive islands (< 25% of median interior area)
                if areas[idx] < 0.25 * med_area:
                    reliable[idx] = False

    # Peel degraded 1–4 leading/trailing frames on each contiguous object visibility segment (occlusion / exit tail)
    rel_cur = np.flatnonzero(reliable)
    if len(rel_cur) >= 4:
        for run in segments([int(i) for i in rel_cur], max_gap=1):
            seg_i = list(run)
            if len(seg_i) < 3:
                continue
            core_i = seg_i[2:-2] if len(seg_i) >= 7 else seg_i
            ref_a = float(np.percentile(areas[core_i], 65))
            ref_p = float(np.percentile(probs[core_i], 65))
            ref_w = float(np.percentile(bw[core_i], 65))
            ref_h = float(np.percentile(bh[core_i], 65))
            for _ in range(4):
                if len(seg_i) < 3:
                    break
                kt = seg_i[-1]
                if (
                    (touches_border[kt] and len(interior_idx) >= 1 and areas[kt] < 0.94 * ref_a)
                    or areas[kt] < 0.76 * ref_a
                    or bw[kt] < 0.82 * ref_w
                    or bh[kt] < 0.82 * ref_h
                    or probs[kt] < min(0.75, 0.85 * ref_p)
                ):
                    reliable[kt] = False
                    seg_i.pop()
                else:
                    break
            for _ in range(4):
                if len(seg_i) < 3:
                    break
                kh = seg_i[0]
                if (
                    (touches_border[kh] and len(interior_idx) >= 1 and areas[kh] < 0.94 * ref_a)
                    or areas[kh] < 0.76 * ref_a
                    or bw[kh] < 0.82 * ref_w
                    or bh[kh] < 0.82 * ref_h
                    or probs[kh] < min(0.75, 0.85 * ref_p)
                ):
                    reliable[kh] = False
                    seg_i.pop(0)
                else:
                    break

    # Strip sudden 1-frame mask bbox explosions relative to local 7-frame window
    rel_cur = np.flatnonzero(reliable)
    if len(rel_cur) >= 5:
        for pos_k, idx in enumerate(rel_cur):
            lo = max(0, pos_k - 3)
            hi = min(len(rel_cur), pos_k + 4)
            nb = rel_cur[lo:hi]
            loc_w = float(np.median(bw[nb]))
            loc_h = float(np.median(bh[nb]))
            if bw[idx] > 1.42 * loc_w or bh[idx] > 1.42 * loc_h:
                reliable[idx] = False

    if not np.any(reliable):
        best_single = int(valid_idx[np.argmax(areas[valid_idx])])
        reliable[best_single] = True

    rel_indices = np.flatnonzero(reliable)
    best_anchor = int(
        max(
            rel_indices,
            key=lambda i: (
                0.0 if touches_border[i] else 1.0,
                float(probs[i]) * math.sqrt(max(areas[i], 1.0)),
            ),
        )
    )
    return reliable, best_anchor


def _clean_interior_depth_cm(depth_cm: np.ndarray, mask_bool: np.ndarray) -> float | None:
    """Compute robust front-surface depth in cm inside eroded object mask after removing depth-discontinuity edges."""
    m_u8 = mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(m_u8, k, iterations=1) > 0
    if int(eroded.sum()) < 16:
        eroded = mask_bool

    # Strip depth-jump edges (flying boundary pixels)
    d_safe = np.where(np.isfinite(depth_cm), depth_cm, 0.0).astype(np.float32)
    gx = cv2.Sobel(d_safe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_safe, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    valid = eroded & np.isfinite(depth_cm) & (depth_cm > 15.0) & (depth_cm < 50000.0)
    if int(valid.sum()) < 8:
        return None
    med0 = float(np.median(depth_cm[valid]))
    smooth_valid = valid & (grad < max(12.0, 0.08 * med0))
    vals = depth_cm[smooth_valid] if int(smooth_valid.sum()) >= 8 else depth_cm[valid]
    # Keep 15th..80th percentile to reject background bleed-through
    q80 = float(np.quantile(vals, 0.80))
    vals_fg = vals[vals <= q80]
    return float(np.median(vals_fg)) if len(vals_fg) > 0 else float(np.median(vals))


def _sample_keypoints_in_mask(gray: np.ndarray, mask_bool: np.ndarray, max_pts: int = 240) -> np.ndarray:
    m_u8 = mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(m_u8, k, iterations=1) > 0
    if int(eroded.sum()) < 20:
        eroded = mask_bool
    er_u8 = eroded.astype(np.uint8) * 255

    pts_list: list[np.ndarray] = []
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_pts // 2, qualityLevel=0.012, minDistance=4, mask=er_u8)
    if corners is not None and len(corners) > 0:
        pts_list.append(corners.reshape(-1, 2).astype(np.float32))

    ys, xs = np.nonzero(eroded)
    if len(xs) > 0:
        rem = max(48, max_pts - (len(pts_list[0]) if pts_list else 0))
        step = max(1, int(math.sqrt(len(xs) / rem)))
        grid_pts = np.stack([xs[::step], ys[::step]], axis=-1).astype(np.float32)
        if len(grid_pts) > rem:
            grid_pts = grid_pts[np.linspace(0, len(grid_pts) - 1, rem, dtype=np.int64)]
        pts_list.append(grid_pts)

    if not pts_list:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(pts_list, axis=0)[:max_pts]


def _pnp_between_frames_rigid_surface(
    gray_src: np.ndarray,
    gray_dst: np.ndarray,
    mask_src: np.ndarray,
    mask_dst: np.ndarray,
    surf_pts_local: np.ndarray,
    surf_nrm_local: np.ndarray,
    r_src_gl: np.ndarray,
    t_src_gl: np.ndarray,
    r_guess_gl: np.ndarray,
    t_guess_gl: np.ndarray,
    k_std: np.ndarray,
    f_std: float,
    img_w: int,
    img_h: int,
    max_deg_step: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Track 2D keypoints from `src -> dst` via FB-checked pyramidal LK, lift them onto the front-facing
    surface of the **rigid 3D proxy mesh** at `(r_src_gl, t_src_gl)`, and solve `(r_dst_gl, t_dst_gl)` via
    `cv2.solvePnPRansac`. Zero per-frame monocular depth drift!"""
    # Project front-facing mesh surface samples at `src` frame
    p_cam_gl = (surf_pts_local @ r_src_gl.T) + t_src_gl[None, :]
    n_cam_gl = surf_nrm_local @ r_src_gl.T
    # Front-facing toward camera at origin: dot(n_cam_gl, -p_cam_gl) > 0
    front = (np.einsum("ni,ni->n", n_cam_gl, -p_cam_gl) > 0.0) & (p_cam_gl[:, 2] < -12.0)
    if int(front.sum()) < 30:
        front = p_cam_gl[:, 2] < -12.0
    if int(front.sum()) < 16:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    idx_front = np.flatnonzero(front)
    pf = p_cam_gl[idx_front]
    depth_f = -pf[:, 2]
    uf = img_w * 0.5 + f_std * (pf[:, 0] / depth_f)
    vf = img_h * 0.5 - f_std * (pf[:, 1] / depth_f)

    # Sort back-to-front then splat onto a 2D pixel index buffer so closest front surface wins!
    order = np.argsort(-depth_f)
    ui = np.round(uf[order]).astype(np.int32)
    vi = np.round(vf[order]).astype(np.int32)
    in_b = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
    ui, vi, ord_v = ui[in_b], vi[in_b], idx_front[order[in_b]]
    if len(ord_v) < 16:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    surf_map = np.full((img_h, img_w), -1, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ux = np.clip(ui + dx, 0, img_w - 1)
            vy = np.clip(vi + dy, 0, img_h - 1)
            surf_map[vy, ux] = ord_v

    pts_src = _sample_keypoints_in_mask(gray_src, mask_src, max_pts=260)
    if len(pts_src) < 12:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    lk_kw = dict(
        winSize=(23, 23),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    pts_dst, st_f, _ = cv2.calcOpticalFlowPyrLK(gray_src, gray_dst, pts_src[:, None, :], None, **lk_kw)
    if pts_dst is None:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0
    pts_bck, st_b, _ = cv2.calcOpticalFlowPyrLK(gray_dst, gray_src, pts_dst, None, **lk_kw)
    if pts_bck is None:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    uv0 = pts_src
    uv1 = pts_dst.reshape(-1, 2)
    fb_err = np.linalg.norm(uv0 - pts_bck.reshape(-1, 2), axis=-1)

    u0i = np.clip(np.round(uv0[:, 0]).astype(np.int32), 0, img_w - 1)
    v0i = np.clip(np.round(uv0[:, 1]).astype(np.int32), 0, img_h - 1)
    u1i = np.clip(np.round(uv1[:, 0]).astype(np.int32), 0, img_w - 1)
    v1i = np.clip(np.round(uv1[:, 1]).astype(np.int32), 0, img_h - 1)
    s_idx = surf_map[v0i, u0i]

    good = (
        (st_f.reshape(-1) == 1)
        & (st_b.reshape(-1) == 1)
        & (fb_err < 1.15)
        & mask_dst[v1i, u1i]
        & (s_idx >= 0)
    )
    n_good = int(good.sum())
    if n_good < 24:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    # Require 2D keypoints to span >= 18% of mask width & height so collinear edge points never flip 3D PnP
    ys_m, xs_m = np.nonzero(mask_dst)
    if len(xs_m) >= 16:
        mw = max(8.0, float(xs_m.max() - xs_m.min()))
        mh = max(8.0, float(ys_m.max() - ys_m.min()))
        ug_sp = float(uv1[good, 0].max() - uv1[good, 0].min())
        vg_sp = float(uv1[good, 1].max() - uv1[good, 1].min())
        if ug_sp < 0.18 * mw or vg_sp < 0.18 * mh:
            return r_guess_gl.copy(), t_guess_gl.copy(), 0

    # Convert local OpenGL points `p_local_gl` into OpenCV convention (`diag(1,-1,-1)`) for `cv2.solvePnPRansac`
    sign_mat = CV_TO_GL[:, None] * CV_TO_GL[None, :]
    p_local_cv = (surf_pts_local[s_idx[good]] * CV_TO_GL[None, :]).astype(np.float64)
    uv1_g = uv1[good].astype(np.float64)

    r_init_cv = (r_guess_gl * sign_mat).astype(np.float64)
    t_init_cv = (t_guess_gl * CV_TO_GL).reshape(3, 1).astype(np.float64)
    rvec0, _ = cv2.Rodrigues(r_init_cv)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        p_local_cv,
        uv1_g,
        k_std,
        None,
        rvec=rvec0,
        tvec=t_init_cv,
        useExtrinsicGuess=True,
        iterationsCount=180,
        reprojectionError=2.0,
        confidence=0.995,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    n_inl = len(inliers) if (ok and inliers is not None) else 0
    if n_inl < 22:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    r_cv, _ = cv2.Rodrigues(rvec)
    r_gl = (r_cv * sign_mat).astype(np.float64)
    t_gl = (tvec.reshape(3) * CV_TO_GL).astype(np.float64)

    d_rot = r_guess_gl.T @ r_gl
    deg = math.degrees(math.acos(float(np.clip((np.trace(d_rot) - 1.0) * 0.5, -1.0, 1.0))))
    z_ref = max(20.0, float(-t_guess_gl[2]))
    if deg > max_deg_step or -t_gl[2] <= 15.0 or abs((-t_gl[2]) - z_ref) / z_ref > 0.25:
        return r_guess_gl.copy(), t_guess_gl.copy(), 0

    return r_gl, t_gl, n_inl


def _refine_6dof_silhouette_gpu(
    rest_points_cm: np.ndarray,
    r_init_gl: np.ndarray,
    t_init_gl: np.ndarray,
    masks_rel: np.ndarray,
    boxes_rel: np.ndarray,
    segs: list[tuple[int, int]],
    f_std: float,
    img_w: int,
    img_h: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched PyTorch CUDA joint optimization of `(R_gl[0..M-1], t_gl[0..M-1])` minimizing:
      1. Un-clipped 2D mask bounding-box alignment (`u_min, v_min, u_max, v_max`)
      2. Differentiable outside-mask Euclidean distance transform (`F.grid_sample`)
      3. 2D mask-contour to projected-mesh Chamfer distance (pulls silhouette & 3D orientation tight!)
      4. 2nd-order temporal acceleration smoothness on `t_gl` and `R_gl` + soft PnP rotation prior.
    """
    m_cnt = len(r_init_gl)
    if m_cnt == 0:
        return r_init_gl, t_init_gl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subsample up to 650 outer hull/surface vertices for fast batched silhouette projection
    if len(rest_points_cm) > 650:
        norms = np.linalg.norm(rest_points_cm, axis=-1)
        top_hull = np.argsort(-norms)[:450]
        uni_idx = np.linspace(0, len(rest_points_cm) - 1, 200, dtype=np.int64)
        sel_v = np.unique(np.concatenate([top_hull, uni_idx]))
        verts_sub = rest_points_cm[sel_v]
    else:
        verts_sub = rest_points_cm

    # Build downsampled outside-mask distance transforms (<= 960px long side) + 96 contour points per frame
    ds = max(1, int(math.ceil(max(img_w, img_h) / 960.0))) if max(img_w, img_h) >= 480 else 1
    h_ds, w_ds = max(16, img_h // ds), max(16, img_w // ds)
    dt_stack = np.zeros((m_cnt, 1, h_ds, w_ds), dtype=np.float32)
    contours_stack = np.zeros((m_cnt, 96, 2), dtype=np.float32)
    edge_valid = np.zeros((m_cnt, 4), dtype=np.float32)
    margin = max(6.0, 0.012 * min(img_w, img_h))

    for k in range(m_cnt):
        mk = masks_rel[k]
        mk_small = (
            cv2.resize(mk.astype(np.uint8), (w_ds, h_ds), interpolation=cv2.INTER_NEAREST) > 0
            if ds > 1
            else mk
        )
        inv_u8 = (~mk_small).astype(np.uint8)
        dt = cv2.distanceTransform(inv_u8, cv2.DIST_L2, 3) * float(ds)
        dt_stack[k, 0] = dt
        contours_stack[k] = _sample_contour_points_2d(mk, n_pts=96)
        bx = boxes_rel[k]
        edge_valid[k] = [
            1.0 if bx[0] > margin else 0.0,
            1.0 if bx[1] > margin else 0.0,
            1.0 if bx[2] < img_w - margin else 0.0,
            1.0 if bx[3] < img_h - margin else 0.0,
        ]

    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        v_t = torch.from_numpy(verts_sub.astype(np.float32)).to(device)  # (V, 3)
        r0_t = torch.from_numpy(r_init_gl.astype(np.float32)).to(device)  # (M, 3, 3)
        t0_t = torch.from_numpy(t_init_gl.astype(np.float32)).to(device)  # (M, 3)
        boxes_t = torch.from_numpy(boxes_rel.astype(np.float32)).to(device)  # (M, 4)
        ev_t = torch.from_numpy(edge_valid).to(device)  # (M, 4)
        dt_t = torch.from_numpy(dt_stack).to(device)  # (M, 1, Hd, Wd)
        cnt_t = torch.from_numpy(contours_stack).to(device)  # (M, 96, 2)
        interior_frame_mask = (ev_t.sum(dim=-1) >= 3.9).float()  # (M,)

        d_trans = torch.zeros_like(t0_t, requires_grad=True)
        d_rotvec = torch.zeros((m_cnt, 3), dtype=torch.float32, device=device, requires_grad=True)

        opt = torch.optim.Adam([{"params": [d_trans], "lr": 1.4}, {"params": [d_rotvec], "lr": 0.004}])
        w_smooth = 0.35 + 1.45 * float(smoothing)

        for step in range(55):
            opt.zero_grad()
            r_delta = roma.rotvec_to_rotmat(d_rotvec)
            r_cur = torch.bmm(r_delta, r0_t)  # (M, 3, 3)
            t_cur = t0_t + d_trans  # (M, 3)

            # Project vertices: (M, V, 3)
            p_cam = torch.matmul(v_t.unsqueeze(0), r_cur.transpose(1, 2)) + t_cur.unsqueeze(1)
            depth = torch.clamp(-p_cam[..., 2], min=12.0)
            u_proj = img_w * 0.5 + f_std * (p_cam[..., 0] / depth)
            v_proj = img_h * 0.5 - f_std * (p_cam[..., 1] / depth)
            uv_proj = torch.stack([u_proj, v_proj], dim=-1)  # (M, V, 2)

            # 1. Soft-extrema & exact extrema bbox alignment on un-truncated edges
            u_min = u_proj.min(dim=1).values
            u_max = u_proj.max(dim=1).values
            v_min = v_proj.min(dim=1).values
            v_max = v_proj.max(dim=1).values
            pred_box = torch.stack([u_min, v_min, u_max, v_max], dim=-1)
            loss_box = (F.smooth_l1_loss(pred_box, boxes_t, reduction="none", beta=2.0) * ev_t).mean()

            # 2. Outside-mask Distance Transform loss via bilinear `grid_sample`
            grid_x = (u_proj / max(img_w - 1, 1)) * 2.0 - 1.0
            grid_y = (v_proj / max(img_h - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (M, V, 1, 2)
            dt_vals = F.grid_sample(dt_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
            loss_outside = dt_vals.squeeze(1).squeeze(-1).mean()

            # 3. 2D Contour-to-Projected-Mesh Chamfer loss on un-truncated interior frames
            dists = torch.cdist(cnt_t, uv_proj)  # (M, 96, V)
            min_c2m = dists.min(dim=-1).values.mean(dim=-1)  # (M,)
            if float(interior_frame_mask.sum().item()) > 0.5:
                loss_chamfer = (min_c2m * interior_frame_mask).sum() / interior_frame_mask.sum()
            else:
                loss_chamfer = min_c2m.mean()

            # 4. Temporal 2nd-difference acceleration smoothness per contiguous segment + strong rotation prior
            loss_smooth = torch.tensor(0.0, device=device)
            for s0, s1 in segs:
                if s1 - s0 >= 3:
                    acc_t = t_cur[s0 + 2 : s1] - 2.0 * t_cur[s0 + 1 : s1 - 1] + t_cur[s0 : s1 - 2]
                    acc_r = d_rotvec[s0 + 2 : s1] - 2.0 * d_rotvec[s0 + 1 : s1 - 1] + d_rotvec[s0 : s1 - 2]
                    loss_smooth = loss_smooth + w_smooth * (0.28 * (acc_t**2).mean() + 180.0 * (acc_r**2).mean())
                elif s1 - s0 == 2:
                    vel_t = t_cur[s0 + 1] - t_cur[s0]
                    loss_smooth = loss_smooth + 0.04 * w_smooth * (vel_t**2).mean()

            # Keep orientation faithful to SAM 3D Objects + verified PnP (zero silhouette-noise wobble!)
            loss_prior = 65.0 * (d_rotvec**2).mean() + 0.008 * (d_trans[:, 2] ** 2).mean()

            loss = 1.6 * loss_box + 0.85 * loss_outside + 0.45 * loss_chamfer + loss_smooth + loss_prior
            loss.backward()
            opt.step()

        with torch.no_grad():
            r_final = torch.bmm(roma.rotvec_to_rotmat(d_rotvec), r0_t).detach().cpu().numpy().astype(np.float64)
            t_final = (t0_t + d_trans).detach().cpu().numpy().astype(np.float64)

    # Smooth ray depth `-t_final[:, 2]` per segment, then lock 2D `(tx, ty)` to < 0.1 px without depth chatter
    z_sig = max(1.35, 0.9 + float(smoothing))
    for s0, s1 in segs:
        if s1 - s0 >= 4:
            t_final[s0:s1, 2:3] = gaussian_smooth(t_final[s0:s1, 2:3], z_sig)

    for k in range(m_cnt):
        verts_rot = rest_points_cm.astype(np.float64) @ r_final[k].T
        t_final[k] = _fit_cam_translation_to_mask_box(
            verts_rot, t_final[k], boxes_rel[k], f_std, img_w * 0.5, img_h * 0.5, img_w, img_h, update_depth=False
        )

    return r_final, t_final


def run_solve_objects(job: dict) -> dict:
    frames_spec = job["frames"]
    shot_frames = [int(it["frame"]) for it in frames_spec]
    frame_paths = {int(it["frame"]): str(it["path"]) for it in frames_spec}
    img_w, img_h = [int(x) for x in job["image_size"]]
    track_dir = Path(job["track_dir"])
    out_dir = Path(job["cam_space_dir"])
    smoothing = float(job.get("smoothing", 0.5))
    selected_ids = set(int(x) for x in (job.get("object_ids") or []))

    npz_files = sorted(track_dir.glob("masks_object_*.npz"))
    if not npz_files:
        raise ValueError("尚未运行物体跟踪，请先在第 2 步点击物体并执行跟踪")
    if selected_ids:
        npz_files = [f for f in npz_files if int(f.stem.split("_")[-1]) in selected_ids]
    if not npz_files:
        raise ValueError("未找到选中的已跟踪物体")

    if "sam3" in _MODELS:
        unload_models()

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_f in out_dir.glob("object_*.npz"):
        old_f.unlink(missing_ok=True)

    f_std = float(math.hypot(float(img_w), float(img_h)))
    if job.get("camera_focal_px") and float(job["camera_focal_px"]) > 10.0:
        f_std = float(job["camera_focal_px"])
    fov_x_deg = float(np.clip(math.degrees(2.0 * math.atan((float(img_w) * 0.5) / f_std)), 8.0, 145.0))
    k_std = np.array(
        [
            [f_std, 0.0, float(img_w) * 0.5],
            [0.0, f_std, float(img_h) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n_frames = len(shot_frames)
    pipeline = _get_sam3d_objects_pipeline()
    moge = pipeline.depth_model.model
    from shotspace.solver.workers.sam3d_objects_bridge import reconstruct_object_sam3d, stage_pipeline_modules

    obj_tracks_loaded = []
    need_frame_idx: set[int] = set()
    anchor_frame_idx: set[int] = set()
    for npz_p in npz_files:
        with np.load(npz_p, allow_pickle=False) as d:
            oid = int(d["object_id"])
            oname = str(d["name"])
            present = d["present"].astype(bool)
            probs = d["probs"].astype(np.float32)
            boxes = d["boxes"].astype(np.float32)
            masks = _unpack_masks(d["packed_masks"], d["mask_shape"])
        reliable, anchor_idx = _detect_reliable_object_masks(masks, boxes, present, probs, img_w, img_h)
        for fi in np.flatnonzero(reliable):
            need_frame_idx.add(int(fi))
        anchor_frame_idx.add(int(anchor_idx))
        obj_tracks_loaded.append(
            {
                "id": oid,
                "name": oname,
                "masks": masks,
                "boxes": boxes,
                "probs": probs,
                "reliable": reliable,
                "anchor_idx": anchor_idx,
                "raw_z_front": np.full(n_frames, np.nan, dtype=np.float64),
            }
        )

    anchor_rgb_cache: dict[int, np.ndarray] = {}
    gray_cache: dict[int, np.ndarray] = {}

    sorted_need = sorted(need_frame_idx)
    stage_pipeline_modules(pipeline, "moge")
    emit("stage", stage="moge_depth", message=f"正在计算相机空间度量深度与点云（{len(sorted_need)} 帧）...")
    for step_i, fi in enumerate(sorted_need):
        f_num = shot_frames[fi]
        im_rgb = np.asarray(Image.open(frame_paths[f_num]).convert("RGB"), dtype=np.uint8)
        if im_rgb.shape[:2] != (img_h, img_w):
            im_rgb = cv2.resize(im_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
        if fi in anchor_frame_idx:
            anchor_rgb_cache[fi] = im_rgb
        gray_cache[fi] = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

        img_t = torch.from_numpy(im_rgb).permute(2, 0, 1).float().to(pipeline.device) / 255.0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            m_out = moge.infer(img_t, fov_x=fov_x_deg, resolution_level=7, apply_mask=True)
        depth_cm_frame = m_out["depth"].detach().float().cpu().numpy().astype(np.float32) * float(M_TO_CM)
        del img_t, m_out
        for item in obj_tracks_loaded:
            if item["reliable"][fi]:
                zf = _clean_interior_depth_cm(depth_cm_frame, item["masks"][fi])
                if zf is not None:
                    item["raw_z_front"][fi] = zf
        emit("progress", stage="moge_depth", done=step_i + 1, total=len(sorted_need))
    stage_pipeline_modules(pipeline, "cpu")

    solved_summaries: list[dict] = []
    for obj_idx, item in enumerate(obj_tracks_loaded):
        oid = item["id"]
        oname = item["name"]
        masks = item["masks"]
        boxes = item["boxes"]
        reliable = item["reliable"].copy()
        anchor_idx = item["anchor_idx"]

        emit(
            "stage",
            stage="reconstruct_mesh",
            message=f"正在使用 Meta SAM 3D Objects 最高精度重建物体 #{oid} ({oname})（锚点帧 {shot_frames[anchor_idx]}）...",
        )
        with quiet():
            sam3d_res = reconstruct_object_sam3d(
                pipeline,
                anchor_rgb_cache[anchor_idx],
                masks[anchor_idx],
                f_std=f_std,
                seed=42,
            )
        rest_points_cm = sam3d_res["rest_points_cm"]
        faces_tri = sam3d_res["faces"]
        vcols = sam3d_res["vertex_colors"]
        r_anchor_gl = sam3d_res["r_gl"]
        t_anchor_gl = _fit_cam_translation_to_mask_box(
            rest_points_cm.astype(np.float64) @ r_anchor_gl.T,
            sam3d_res["t_gl_cm"],
            boxes[anchor_idx],
            f_std,
            img_w * 0.5,
            img_h * 0.5,
            img_w,
            img_h,
            update_depth=True,
        )

        rel_indices = np.flatnonzero(reliable)
        raw_z_front = item["raw_z_front"]

        if not np.isfinite(raw_z_front[anchor_idx]):
            finite_z = raw_z_front[np.isfinite(raw_z_front)]
            raw_z_front[anchor_idx] = float(np.median(finite_z)) if len(finite_z) > 0 else float(np.linalg.norm(t_anchor_gl))
        z_anchor_front = float(raw_z_front[anchor_idx])

        # Sample dense surface points + outward normals on the SAM 3D Objects mesh for drift-free 6DoF tracking
        tm_cm = trimesh.Trimesh(vertices=rest_points_cm, faces=faces_tri, process=False)
        surf_pts, face_idx = trimesh.sample.sample_surface(tm_cm, 8000, seed=42)
        surf_pts_local = np.asarray(surf_pts, dtype=np.float64)
        surf_nrm_local = np.asarray(tm_cm.face_normals[face_idx], dtype=np.float64)

        # Build geometric silhouette-matched initial translations + ray-transported SAM 3D Objects orientations
        r_guess_all = np.repeat(np.eye(3, dtype=np.float64)[None], n_frames, axis=0)
        t_guess_all = np.zeros((n_frames, 3), dtype=np.float64)
        areas = masks.reshape(n_frames, -1).sum(axis=1).astype(np.float64)
        area_anchor = max(20.0, float(areas[anchor_idx]))
        d_anchor_center = float(np.linalg.norm(t_anchor_gl))

        diags_rel = np.hypot(boxes[rel_indices, 2] - boxes[rel_indices, 0], boxes[rel_indices, 3] - boxes[rel_indices, 1])
        cov_diag = float(np.std(diags_rel) / max(1.0, float(np.median(diags_rel)))) if len(diags_rel) >= 3 else 0.25
        w_moge = 0.20 if cov_diag < 0.12 else 0.45

        for fi in rel_indices:
            bx = boxes[fi]
            cx_f, cy_f = 0.5 * float(bx[0] + bx[2]), 0.5 * float(bx[1] + bx[3])
            d_app = d_anchor_center * math.sqrt(area_anchor / max(20.0, float(areas[fi])))
            if np.isfinite(raw_z_front[fi]):
                d_moge = float(raw_z_front[fi]) * (d_anchor_center / max(20.0, z_anchor_front))
                d_est = w_moge * d_moge + (1.0 - w_moge) * d_app
            else:
                d_est = d_app
            fwd_f = np.array([(cx_f - img_w * 0.5) / f_std, -(cy_f - img_h * 0.5) / f_std, -1.0], dtype=np.float64)
            fwd_f /= np.linalg.norm(fwd_f)
            t_est = fwd_f * d_est
            r_ray_f = _rotation_between_vectors(t_anchor_gl, t_est) @ r_anchor_gl
            r_guess_all[fi] = r_ray_f
            v_rot_f = rest_points_cm.astype(np.float64) @ r_ray_f.T
            t_guess_all[fi] = _fit_cam_translation_to_mask_box(
                v_rot_f, t_est, bx, f_std, img_w * 0.5, img_h * 0.5, img_w, img_h
            )

        # Strip sudden depth spikes before tracking/optimization
        if len(rel_indices) >= 5:
            z_vals = -t_guess_all[rel_indices, 2]
            med_z = float(np.median(z_vals))
            mad_z = max(20.0, float(np.median(np.abs(z_vals - med_z))) * 1.4826)
            for fi in rel_indices:
                if fi != anchor_idx and abs((-t_guess_all[fi, 2]) - med_z) > max(5.0 * mad_z, 0.90 * med_z):
                    reliable[fi] = False

        rel_indices = np.flatnonzero(reliable)
        rel_frames_list = [shot_frames[int(i)] for i in rel_indices]
        runs = segments(rel_frames_list, max_gap=2)
        segs: list[tuple[int, int]] = []
        cursor_pos = 0
        for run in runs:
            segs.append((cursor_pos, cursor_pos + len(run)))
            cursor_pos += len(run)

        emit("stage", stage="track_6dof", message=f"正在解算并轮廓对齐物体 #{oid} ({oname}) 的 6DoF 位姿...")
        r_gl_all = r_guess_all.copy()
        t_gl_all = t_guess_all.copy()
        r_gl_all[anchor_idx] = r_anchor_gl
        t_gl_all[anchor_idx] = t_anchor_gl

        for s_start, s_end in segs:
            seg_idx = rel_indices[s_start:s_end]
            seed_pos = int(np.where(seg_idx == anchor_idx)[0][0]) if anchor_idx in seg_idx else len(seg_idx) // 2
            seed_fi = int(seg_idx[seed_pos])

            # Forward propagation (with direct seed-keyframe lock when close to seed!)
            for p_i in range(seed_pos + 1, len(seg_idx)):
                prev_fi = int(seg_idx[p_i - 1])
                curr_fi = int(seg_idx[p_i])
                # Predict initial guess by applying `prev_fi`'s relative spin onto `curr_fi`'s viewing ray
                ray_delta = _rotation_between_vectors(t_gl_all[prev_fi], t_guess_all[curr_fi])
                r_pred = ray_delta @ r_gl_all[prev_fi]
                t_pred = _fit_cam_translation_to_mask_box(
                    rest_points_cm.astype(np.float64) @ r_pred.T,
                    t_guess_all[curr_fi],
                    boxes[curr_fi],
                    f_std,
                    img_w * 0.5,
                    img_h * 0.5,
                    img_w,
                    img_h,
                )

                # Try direct seed -> curr keyframe PnP first (drift-free!) when within 12 frames
                used_direct = False
                if abs(curr_fi - seed_fi) <= 12:
                    r_dir, t_dir, n_inl = _pnp_between_frames_rigid_surface(
                        gray_cache[seed_fi],
                        gray_cache[curr_fi],
                        masks[seed_fi],
                        masks[curr_fi],
                        surf_pts_local,
                        surf_nrm_local,
                        r_gl_all[seed_fi],
                        t_gl_all[seed_fi],
                        r_pred,
                        t_pred,
                        k_std,
                        f_std,
                        img_w,
                        img_h,
                        max_deg_step=24.0,
                    )
                    if n_inl >= 26:
                        r_gl_all[curr_fi], t_gl_all[curr_fi] = r_dir, t_dir
                        used_direct = True

                if not used_direct and curr_fi - prev_fi <= 2:
                    r_seq, t_seq, _ = _pnp_between_frames_rigid_surface(
                        gray_cache[prev_fi],
                        gray_cache[curr_fi],
                        masks[prev_fi],
                        masks[curr_fi],
                        surf_pts_local,
                        surf_nrm_local,
                        r_gl_all[prev_fi],
                        t_gl_all[prev_fi],
                        r_pred,
                        t_pred,
                        k_std,
                        f_std,
                        img_w,
                        img_h,
                        max_deg_step=10.0,
                    )
                    r_gl_all[curr_fi], t_gl_all[curr_fi] = r_seq, t_seq
                elif not used_direct:
                    r_gl_all[curr_fi], t_gl_all[curr_fi] = r_pred, t_pred

            # Backward propagation from seed
            for p_i in range(seed_pos - 1, -1, -1):
                prev_fi = int(seg_idx[p_i + 1])
                curr_fi = int(seg_idx[p_i])
                ray_delta = _rotation_between_vectors(t_gl_all[prev_fi], t_guess_all[curr_fi])
                r_pred = ray_delta @ r_gl_all[prev_fi]
                t_pred = _fit_cam_translation_to_mask_box(
                    rest_points_cm.astype(np.float64) @ r_pred.T,
                    t_guess_all[curr_fi],
                    boxes[curr_fi],
                    f_std,
                    img_w * 0.5,
                    img_h * 0.5,
                    img_w,
                    img_h,
                )

                used_direct = False
                if abs(curr_fi - seed_fi) <= 12:
                    r_dir, t_dir, n_inl = _pnp_between_frames_rigid_surface(
                        gray_cache[seed_fi],
                        gray_cache[curr_fi],
                        masks[seed_fi],
                        masks[curr_fi],
                        surf_pts_local,
                        surf_nrm_local,
                        r_gl_all[seed_fi],
                        t_gl_all[seed_fi],
                        r_pred,
                        t_pred,
                        k_std,
                        f_std,
                        img_w,
                        img_h,
                        max_deg_step=24.0,
                    )
                    if n_inl >= 26:
                        r_gl_all[curr_fi], t_gl_all[curr_fi] = r_dir, t_dir
                        used_direct = True

                if not used_direct and prev_fi - curr_fi <= 2:
                    r_seq, t_seq, _ = _pnp_between_frames_rigid_surface(
                        gray_cache[prev_fi],
                        gray_cache[curr_fi],
                        masks[prev_fi],
                        masks[curr_fi],
                        surf_pts_local,
                        surf_nrm_local,
                        r_gl_all[prev_fi],
                        t_gl_all[prev_fi],
                        r_pred,
                        t_pred,
                        k_std,
                        f_std,
                        img_w,
                        img_h,
                        max_deg_step=10.0,
                    )
                    r_gl_all[curr_fi], t_gl_all[curr_fi] = r_seq, t_seq
                elif not used_direct:
                    r_gl_all[curr_fi], t_gl_all[curr_fi] = r_pred, t_pred

        # Smooth initial rotations & depth before GPU silhouette bundle adjustment
        r_gl_rel = r_gl_all[rel_indices].copy()
        t_gl_rel = t_gl_all[rel_indices].copy()
        sigma = smoothing_sigma(smoothing)
        if sigma > 0.0 and len(rel_indices) >= 3:
            q_rel = matrix_to_quat(r_gl_rel)
            for s0, s1 in segs:
                if s1 - s0 >= 3:
                    q_rel[s0:s1] = smooth_quats(q_rel[s0:s1], sigma * 0.8)
                    # Smooth depth Z more strongly than X/Y so 2D screen lock never lags!
                    t_gl_rel[s0:s1, 2:3] = gaussian_smooth(t_gl_rel[s0:s1, 2:3], sigma)
            r_gl_rel = quat_to_matrix(q_rel)

        # Batched PyTorch GPU 2D-3D Silhouette + Contour Chamfer + BBox + Smoothness Joint Refinement!
        r_gl_rel, t_gl_rel = _refine_6dof_silhouette_gpu(
            rest_points_cm,
            r_gl_rel,
            t_gl_rel,
            masks[rel_indices],
            boxes[rel_indices],
            segs,
            f_std,
            img_w,
            img_h,
            smoothing,
        )

        # Expand across ALL `shot_frames`: hold nearest reliable anchor pose on `visible == 0` frames!
        world_transforms = np.repeat(np.eye(4, dtype=np.float64)[None], n_frames, axis=0)
        visible_arr = np.zeros(n_frames, dtype=np.uint8)
        anchor_frames_arr = np.zeros(n_frames, dtype=np.int64)

        for k_i, fi in enumerate(rel_indices):
            world_transforms[fi, :3, :3] = r_gl_rel[k_i]
            world_transforms[fi, :3, 3] = t_gl_rel[k_i]
            visible_arr[fi] = 1
            anchor_frames_arr[fi] = shot_frames[int(fi)]

        for fi in range(n_frames):
            if visible_arr[fi] == 1:
                continue
            nearest_rel_fi = int(rel_indices[np.argmin(np.abs(rel_indices - fi))])
            world_transforms[fi] = world_transforms[nearest_rel_fi]
            anchor_frames_arr[fi] = shot_frames[nearest_rel_fi]

        npz_out = out_dir / f"object_{oid:02d}.npz"
        solved_f_list = [shot_frames[int(fi)] for fi in rel_indices]
        np.savez_compressed(
            npz_out,
            object_id=np.int64(oid),
            name=np.array(oname),
            frames=np.asarray(shot_frames, dtype=np.int64),
            solved_frames=np.asarray(solved_f_list, dtype=np.int64),
            visible=visible_arr,
            anchor_frames=anchor_frames_arr,
            mask_boxes=boxes.astype(np.float32),
            rest_points=rest_points_cm.astype(np.float32),
            faces=faces_tri.astype(np.int32),
            vertex_colors=vcols.astype(np.float32),
            world_transforms=world_transforms.astype(np.float64),
            source_focal_px=np.float64(f_std),
        )

        solved_summaries.append(
            {
                "id": oid,
                "name": oname,
                "file": npz_out.name,
                "frames": shot_frames,
                "solved_frames": solved_f_list,
                "visible_count": int(visible_arr.sum()),
                "vertex_count": int(len(rest_points_cm)),
                "face_count": int(len(faces_tri)),
            }
        )
        emit("progress", stage="solve_objects", done=obj_idx + 1, total=len(obj_tracks_loaded))

    return {"objects": solved_summaries}


def run_track_people(job: dict) -> dict:
    """Whole-shot person masks + identities from SAM 3.1's native video concept tracking (see `people_tracker`)."""
    from shotspace.solver.workers import people_tracker

    return people_tracker.run_track_people(job, get_predictor=_get_sam3_predictor, emit=emit, free_memory=_free_memory)


# ------------------------------------------------------------------ Resident Server & Entry Point


def _watch_parent(parent_pid: int) -> None:
    if parent_pid <= 1:
        return
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, 15)
        except Exception:
            pass

    def _poll() -> None:
        while True:
            if not pid_alive(parent_pid):  # never os.kill(pid, 0): on Windows that ends the process
                os._exit(0)
            threading.Event().wait(2.0)

    threading.Thread(target=_poll, daemon=True).start()


def _dispatch_job(job_file: str) -> None:
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    cmd = job["command"]
    if cmd == "track_objects":
        res = run_track_objects(job)
    elif cmd == "solve_objects":
        res = run_solve_objects(job)
    elif cmd == "segment_people":
        res = run_segment_people(job)
    elif cmd == "track_people":
        res = run_track_people(job)
    else:
        raise ValueError(f"Unknown object worker command: {cmd}")

    if res_file := job.get("result_file"):
        Path(res_file).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    _pin_stdout()
    _forbid_network()
    if p_pid := int(os.environ.get("SHOTSPACE_PARENT_PID", "0") or "0"):
        _watch_parent(p_pid)

    args = sys.argv[1:]
    if "--serve" in args:
        commands = os.fdopen(os.dup(0), "r", encoding="utf-8")
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)

        for raw in commands:
            raw = raw.strip()
            if not raw:
                continue
            msg = json.loads(raw)
            cmd = msg.get("cmd")
            if cmd == "exit":
                unload_models()
                break
            if cmd == "offload":
                unload_models()
                emit("offloaded")
                continue
            if cmd == "job":
                try:
                    _dispatch_job(msg["job"])
                    _free_memory()
                    emit("job_end", ok=True)
                except BaseException as exc:
                    tb_str = traceback.format_exc()
                    sys.stderr.write(tb_str)
                    sys.stderr.flush()
                    _free_memory()
                    emit("job_end", ok=False, error=f"{type(exc).__name__}: {exc}\n{tb_str[-800:]}")
    elif "--job" in args:
        idx = args.index("--job")
        _dispatch_job(args[idx + 1])
    else:
        raise SystemExit("Usage: sam_object_worker.py [--serve | --job <job.json>]")


if __name__ == "__main__":
    main()
