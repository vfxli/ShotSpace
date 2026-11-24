"""SAM 3D Body GPU worker (`third_party/sam_3d_body/.venv/bin/python`), resident or one-shot.

  solve        per-frame SAM 3D Body for every tracked person, prompted with the person's own mask (so the network
               reconstructs the right person in an overlapping crowd) under the exact per-frame intrinsics of the
               imported camera; raw outputs are cached (`raw_net/person_XX.npz`); then the whole-shot solve.
  postprocess  re-run only the whole-shot solve from the cached raw outputs (camera move, scan or smoothing changed).

The whole-shot solve itself lives in `shotspace.solver.trajectory` and `shotspace.solver.workers.body_solve`.
"""

from __future__ import annotations

import os

# Disable native Momentum dependency before importing sam_3d_body; MHR uses TorchScript mhr_model.pt
os.environ.setdefault("MOMENTUM_ENABLED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", os.environ.get("SHOTSPACE_WEIGHTS_DIR", ""))

import ctypes
import errno
import gc
import ipaddress
import itertools
import json
import platform
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from shotspace.core.poses import quat_to_matrix
from shotspace.solver.box_repair import clean_outline
from shotspace.core.smoothing import expand_to_full_shot_with_visibility
from shotspace.core.units import CV_TO_GL, M_TO_CM
from shotspace.io.plate_grade import GradeParams, apply_grade_rgb, build_uint8_lut

EVENT_PREFIX = "@SHOTSPACE@"
CHUNK = 64
# People of one frame go through the network together, this many per call.  Measured on an RTX 4090 at 1080p with
# the backbone in half precision (see BODY_PRECISION): 1 per call 2.2 person-frames per second, 4 per call 5.8,
# 16 per call 8.7 to 9.3, 32 per call 10.6; peak memory of the call 3.4 GB, 3.6 GB, 4.6 GB, 5.8 GB.  (In float32:
# 1.7, 3.6, 4.4 to 4.7, 4.9 person-frames per second and 5.1, 5.5, 7.0, 8.9 GB.)  A call has a fixed cost of about
# 0.37 s (the decoder evaluates the body model at every layer) that a batch shares; beyond 16 the three backbone
# passes dominate and a bigger batch gains little.
PEOPLE_BATCH = 16
BATCH_STEPS = (16, 8, 4, 2, 1)
# How the people of one frame are handed to SAM 3D Body.  Its full (body + hands) inference treats a call with ONE
# person differently from a call with several (`sam3d_body.run_inference`, "if keypoint_prompt.shape[0] > 1"): with
# several, hand/elbow prompts that are not usable are replaced by dummy prompts and the body is estimated a second
# time for everybody; with one, unusable prompts are dropped and the second estimate is skipped when none is left.
# Batched as it comes, which of the two a person gets on a frame depends on who else is in the batch (how many people
# were selected, who is in the picture on that frame, the batch size after an out-of-memory step-down), and a person
# estimated alone differs from the same person estimated with others (hidden people most: 20 % of the frames by more
# than 5 px in some keypoint, up to 6 cm at the pelvis after the whole-shot solve).  Against ground truth the two
# paths are equally good (Panoptic pizza1 c23: 20.93 px / 10.57 cm one per call, 21.00 px / 10.63 cm several per
# call; EMDB: 8.28 / 3.39 against 8.27 / 3.34).
#   "consistent": people are batched, and `patch_per_person_prompts` makes the network handle the prompts of every
#                 person in the batch exactly the way it handles a person who is alone in the call.  A person's
#                 estimate then depends on nobody else, and the card is used fully.  The product setting.
#   "single":     every person in a call of their own (how v0.2.2 did it: same result, about a third of the speed).
#   "multi":      the network's own several-people path, never fewer than two people per call.  For measurements.
#   "auto":       the network's own paths, batches as they come (the behaviour up to v0.2.1).  For measurements.
# The environment variable SHOTSPACE_SAM3D_BATCH_MODE overrides it for a measurement run.
PERSON_BATCH_MODE = "consistent"
_PROMPT_PATCH = {"on": False}
# Arithmetic of the network's backbone (DINOv3-H+, 841 M parameters; its three passes per person - body, left hand,
# right hand - are almost all of the work).  Three types were measured on 255 person-frames of Panoptic pizza1 c23,
# 16 people per call with `patch_per_person_prompts`, against float32 with every person in a call of their own:
#   bfloat16 (as released, 8 significant bits):  2D keypoints 0.14 px (median), 7.7 px (99 %), 56 px (largest);
#                                                 35 person-frames off by more than 5 px; 8.3 person-frames per second
#   float16  (the network's own other half type,  0.03 px (median), 1.9 px (99 %), 8.3 px (largest); 2 beyond 5 px;
#             11 significant bits):               8.7 person-frames per second, 4.6 GB
#   float32:                                      0.002 px (median), 0.17 px (99 %), 0.4 px (largest); none;
#                                                 4.4 person-frames per second, 7.0 GB
# bfloat16 rounds so coarsely that on a hard frame (a hidden hand or foot) the estimate tips over; float16 does not,
# and against ground truth it is the same as float32 (Panoptic pizza1 c23: 20.69 px / 10.030 cm against
# 20.69 px / 10.032 cm; EMDB: 8.28 px / 3.421 cm against 8.27 px / 3.420 cm) at twice the speed.
#   "float16": the product setting     "full": float32     "half": bfloat16 as released; the last two for measurements
# The environment variable SHOTSPACE_SAM3D_PRECISION overrides it for a measurement run.
BODY_PRECISION = "float16"


def body_precision() -> str:
    return str(os.environ.get("SHOTSPACE_SAM3D_PRECISION") or BODY_PRECISION).strip().lower()


def patch_per_person_prompts(model) -> None:
    """Make a batched call treat every person the way a call with that person alone would.

    The only place where SAM 3D Body's full inference lets the people of a call influence one another is the second
    body estimate, prompted with the wrist and elbow positions from the hand passes: for a batch it pads everybody's
    prompts to the same count with dummy prompts (label -2).  Here the padded prompts are taken out again: people
    are grouped by WHICH of their prompts are usable, each group is sent through the network's own
    `run_keypoint_prompt` with just those prompts (what the one-person path does), and people without any usable
    prompt keep their first estimate (the one-person path skips the second estimate for them).  The heavy parts of
    the call - backbone, body decoder, both hand passes - stay batched and are untouched."""
    if getattr(model, "_shotspace_prompt_patch", False):
        return
    original = model.run_keypoint_prompt

    def run_keypoint_prompt(batch, output, keypoint_prompt):
        dummy = keypoint_prompt[..., -1] == -2  # [people, prompts]
        if not _PROMPT_PATCH["on"] or not bool(dummy.any()):
            return original(batch, output, keypoint_prompt)
        n = int(keypoint_prompt.shape[0])
        first = output["mhr"]
        per_person = lambda v: torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == n  # noqa: E731
        merged = {k: (v.clone() if per_person(v) else v) for k, v in first.items()}
        all_idx, all_ray = list(model.body_batch_idx), batch.get("ray_cond")
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(dummy.tolist()):
            groups.setdefault(tuple(bool(d) for d in row), []).append(i)
        try:
            for pattern, members in groups.items():
                usable = [j for j, is_dummy in enumerate(pattern) if not is_dummy]
                if not usable:
                    continue
                sel = torch.as_tensor(members, device=keypoint_prompt.device, dtype=torch.long)
                sub = dict(output)
                sub["image_embeddings"] = output["image_embeddings"][sel]
                if output.get("condition_info") is not None:
                    sub["condition_info"] = output["condition_info"][sel]
                sub["mhr"] = {k: (v[sel] if per_person(v) else v) for k, v in first.items()}
                model.body_batch_idx = [all_idx[i] for i in members]
                if all_ray is not None:
                    batch["ray_cond"] = all_ray[sel]
                sub, _ = original(batch, sub, keypoint_prompt[sel][:, usable])
                for k, v in sub["mhr"].items():
                    if k in merged and per_person(merged[k]) and torch.is_tensor(v) and int(v.shape[0]) == len(members):
                        merged[k][sel] = v.to(merged[k].dtype)
        finally:
            model.body_batch_idx = all_idx
            if all_ray is not None:
                batch["ray_cond"] = all_ray
        output.update({"mhr": merged})
        return output, keypoint_prompt

    model.run_keypoint_prompt = run_keypoint_prompt
    model._shotspace_prompt_patch = True


_OOM_MARKERS = ("out of memory", "cuda driver error: device not ready")

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
            raise ConnectionRefusedError(errno.ECONNREFUSED, f"ShotSpace offline worker blocked network connection to {host}")
        return orig_connect(sock, address)

    def _refused_connect_ex(sock, address):
        if _is_remote(sock, address):
            return errno.ECONNREFUSED
        return orig_connect_ex(sock, address)

    socket.socket.connect = _refused_connect
    socket.socket.connect_ex = _refused_connect_ex


def _install_local_hub(dinov3_dir: str) -> None:
    import torch.hub

    original = getattr(torch.hub.load, "_shotspace_orig", torch.hub.load)

    def _load(repo_or_dir, model, *args, **kwargs):
        name = str(repo_or_dir).split(":")[0].rstrip("/").split("/")[-1]
        if name == "dinov3":
            for k in ("source", "force_reload", "trust_repo", "skip_validation"):
                kwargs.pop(k, None)
            return original(str(dinov3_dir), model, *args, source="local", **kwargs)
        if Path(str(repo_or_dir)).is_dir():
            return original(repo_or_dir, model, *args, **kwargs)
        raise RuntimeError(f"Offline worker refused torch.hub.load({repo_or_dir!r})")

    _load._shotspace_orig = original
    torch.hub.load = _load


def _limit_gpu_memory(margin_mb: int = 512) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    # Never artificially cap PyTorch below 92% of physical VRAM due to transient external allocations
    cap = max(free + torch.cuda.memory_reserved() - margin_mb * 2**20, int(total * 0.92))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))


def _is_oom(exc: BaseException) -> bool:
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "OutOfMemoryError" or any(m in str(cur).lower() for m in _OOM_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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


@contextmanager
def keep_gpu_cache():
    real = torch.cuda.empty_cache

    def _empty():
        if sys._getframe(1).f_globals.get("__name__", "").endswith("sam_3d_body_estimator"):
            return
        real()

    torch.cuda.empty_cache = _empty
    try:
        yield
    finally:
        torch.cuda.empty_cache = real
        real()


def _move_model_tensors(obj: Any, target_device: str | torch.device) -> None:
    from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

    atoms = (
        str, bytes, int, float, bool, complex, type(None), type,
        ModuleType, FunctionType, BuiltinFunctionType, MethodType, Path, torch.device, torch.dtype,
    )
    seen, stack = set(), [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if id(cur) in seen or depth > 24 or isinstance(cur, atoms):
            continue
        seen.add(id(cur))
        if isinstance(cur, torch.Tensor):
            if cur.device.type != "meta":
                if cur.is_inference():
                    with torch.inference_mode():
                        cur.data = cur.data.to(target_device)
                else:
                    cur.data = cur.data.to(target_device)
            continue
        if isinstance(cur, torch.nn.Module):
            items = [*cur.parameters(recurse=False), *cur.buffers(recurse=False), *cur.children()]
            if not isinstance(cur, torch.jit.ScriptModule):
                items += [v for k, v in vars(cur).items() if k not in ("_parameters", "_buffers", "_modules")]
        elif isinstance(cur, dict):
            items = list(cur.values())
        elif isinstance(cur, (list, tuple, set, frozenset)):
            items = list(cur)
        else:
            items = list(getattr(cur, "__dict__", {}).values())
        stack.extend((it, depth + 1) for it in items)


def get_sam3d_body(sam3d_weights: Path, device: torch.device):
    if "body" not in _MODELS:
        emit("stage", stage="load_sam3d", message="正在加载 SAM 3D Body (DINOv3 + MHR) 模型...")
        from sam_3d_body import load_sam_3d_body

        with quiet():
            _MODELS["body"] = load_sam_3d_body(
                str(sam3d_weights / "model.ckpt"),
                device=device,
                mhr_path=str(sam3d_weights / "assets" / "mhr_model.pt"),
            )
        if body_precision() == "float16":
            # the network's own other half-precision type (`FP16_TYPE: float16` in its code)
            body_model = _MODELS["body"][0]
            body_model.backbone.to(torch.float16)
            body_model.backbone_dtype = torch.float16
        if body_precision() == "full":
            body_model = _MODELS["body"][0]
            body_model.backbone.float()
            body_model.backbone_dtype = torch.float32
            # Exact float32 products.  TF32 (operands rounded to 10 bits, 44 % faster on an RTX card) was measured
            # and rejected: against exact products the 2D keypoints move by 0.44 px (median), 4 px (99 %), up to
            # 57 px, and a person in a batch no longer equals the same person alone.  This is PyTorch's default;
            # it is set here so that nothing else in the process can switch it on unnoticed.
            torch.backends.cuda.matmul.allow_tf32 = False
    else:
        _move_model_tensors(_MODELS["body"], device)
    return _MODELS["body"]


def offload_all() -> None:
    for val in _MODELS.values():
        _move_model_tensors(val, "cpu")
    _free_memory()


# ------------------------------------------------------------------ Frame reading + Grade


def _read_graded_rgb(path: str, grade: GradeParams, lut: np.ndarray) -> np.ndarray:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".exr":
        raw = cv2.imread(str(p), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise RuntimeError(f"无法读取 EXR 帧：{p}")
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, None], 3, axis=2)
        rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        return apply_grade_rgb(rgb, grade, lut)

    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"无法读取图像帧：{p}")
    if raw.ndim == 2:
        raw = np.repeat(raw[:, :, None], 3, axis=2)
    elif raw.shape[2] == 4:
        raw = raw[:, :, :3]
    rgb = raw[:, :, ::-1]  # BGR -> RGB
    if rgb.dtype == np.uint8:
        return cv2.LUT(rgb, lut) if not grade.is_identity else np.ascontiguousarray(rgb)
    return apply_grade_rgb(rgb, grade, lut)


# ------------------------------------------------------------------ MHR Rig & Postprocess


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def skel_state_to_matrix(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    t, q, s = state[..., :3], state[..., 3:7], state[..., 7:8]
    r = quat_to_matrix(q[..., [3, 0, 1, 2]])  # (x, y, z, w) -> (w, x, y, z)
    m = np.zeros(state.shape[:-1] + (4, 4), dtype=np.float64)
    m[..., :3, :3] = r * s[..., None]
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m


def extract_rig_data(head) -> dict:
    mhr = head.mhr
    character = mhr.character_torch
    lbs = character.linear_blend_skinning
    n_verts = int(lbs.num_vertices)
    bind_world = np.linalg.inv(skel_state_to_matrix(_np(lbs.inverse_bind_pose)))

    verts = _np(lbs.vert_indices_flattened).astype(np.int64)
    joints = _np(lbs.skin_indices_flattened).astype(np.int64)
    weights = _np(lbs.skin_weights_flattened).astype(np.float64)
    counts = np.bincount(verts, minlength=n_verts)
    k = int(counts.max())
    order = np.argsort(verts, kind="stable")
    slot = np.arange(len(verts)) - np.repeat(np.cumsum(counts) - counts, counts)
    skin_idx = np.zeros((n_verts, k), dtype=np.int32)
    skin_w = np.zeros((n_verts, k), dtype=np.float32)
    skin_idx[verts[order], slot] = joints[order]
    skin_w[verts[order], slot] = weights[order]

    mesh = character.mesh
    return {
        "joint_names": np.array(list(mhr.get_joint_names())),
        "parents": _np(character.skeleton.joint_parents).astype(np.int64),
        "bind_world": bind_world,
        "faces": _np(mesh.faces).astype(np.int32),
        "uv": _np(mesh.texcoords).astype(np.float32),
        "uv_faces": _np(mesh.texcoord_faces).astype(np.int32),
        "skin_indices": skin_idx,
        "skin_weights": skin_w,
    }


@torch.no_grad()
def evaluate_mhr_chunks(head, global_rot, body, hand, scale, shape) -> dict:
    device = head.scale_mean.device
    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    joint_world, vertices = [], []
    for lo in range(0, len(global_rot), CHUNK):
        sl = slice(lo, lo + CHUNK)
        n = len(global_rot[sl])
        shape_t = to_t(shape[sl])
        expr = torch.zeros(n, head.num_face_comps, device=device)
        _, model_params = head.mhr_forward(
            global_trans=torch.zeros(n, 3, device=device),
            global_rot=to_t(global_rot[sl]),
            body_pose_params=to_t(body[sl]),
            hand_pose_params=to_t(hand[sl]),
            scale_params=to_t(scale[sl]),
            shape_params=shape_t,
            expr_params=expr,
            return_model_params=True,
        )
        verts, skel_state = head.mhr(shape_t, model_params, expr)
        joint_world.append(skel_state_to_matrix(_np(skel_state)))
        vertices.append(_np(verts).astype(np.float64))
    return {"joint_world": np.concatenate(joint_world), "vertices": np.concatenate(vertices)}


@torch.no_grad()
def evaluate_rest_vertices(head, shape_vec: np.ndarray) -> np.ndarray:
    device = head.scale_mean.device
    coeffs = torch.as_tensor(np.asarray(shape_vec)[None], dtype=torch.float32, device=device)
    return _np(head.mhr.character_torch.blend_shape.forward(coeffs))[0]


# ------------------------------------------------------------------ whole-shot solve (solver v2)


def solvable_frames(raw_data: dict, min_visible: float = 0.25, min_run: int = 3, end_factor: float = 0.6,
                    fps: float = 24.0) -> np.ndarray:
    """Frames of one tracked person that carry enough of the body to be solved.

    A frame is used when the network produced a finite estimate in front of the camera and, when the tracker's
    mask is available, at least `min_visible` of the body keypoints (head, torso, limbs, feet) land on the person's
    own visible pixels inside the picture.  Entering / leaving the frame and full occlusions therefore end the
    solved range on the last frame that really shows the person.  Frames whose distance estimate cannot belong to
    the same person as their neighbours are dropped as well, and so are isolated islands shorter than `min_run`
    frames.  Last, the weak first and last frames of every run are cut off (`trim_weak_ends`: below `end_factor`
    times the share of the body this person usually shows; all of them while the person is cut by the picture
    edge, at most a third of a second when somebody stands in front; 0 switches it off)."""
    from shotspace.solver.trajectory import BODY_KP, depth_outliers, drop_short_runs, trim_weak_ends

    cam_t = np.asarray(raw_data["pred_cam_t"], dtype=np.float64)
    ok = np.all(np.isfinite(cam_t), axis=1) & (cam_t[:, 2] > 0.2) & (cam_t[:, 2] < 500.0)
    vis = None
    if "kp_in_mask" in raw_data and "has_mask" in raw_data and np.any(np.asarray(raw_data["has_mask"]) > 0):
        vis = np.asarray(raw_data["kp_in_mask"], dtype=bool)[:, list(BODY_KP)].mean(axis=1)
        has = np.asarray(raw_data["has_mask"]).reshape(-1) > 0
        ok &= ~has | (vis >= min_visible)
        vis = np.where(has, vis, 1.0)  # a frame without an outline has nothing to judge it by
    frames = np.asarray(raw_data["frames"]).reshape(-1).astype(np.int64)
    ok = drop_short_runs(ok, frames, min_run)
    ok &= ~depth_outliers(cam_t[:, 2], frames, ok)
    ok = drop_short_runs(ok, frames, min_run)
    if vis is not None and end_factor > 0.0:
        # entering and leaving: the first and last frames of a run, where only a sliver of the person shows, drift
        most = max(3, int(round(float(fps) / 3.0)))
        edge = None
        if "mask_box" in raw_data and "image_size" in raw_data:
            # the person's visible pixels reach the border of the picture: they are partly outside it
            mb = np.asarray(raw_data["mask_box"], dtype=np.float64).reshape(-1, 4)
            pic_w, pic_h = (float(v) for v in np.asarray(raw_data["image_size"]).reshape(-1)[:2])
            edge = (mb[:, 0] <= 2.0) | (mb[:, 1] <= 2.0) | (mb[:, 2] >= pic_w - 2.0) | (mb[:, 3] >= pic_h - 2.0)
        ok = drop_short_runs(
            trim_weak_ends(ok, frames, vis, factor=end_factor, max_frames=most, at_picture_edge=edge), frames, min_run
        )
    return ok


def postprocess_person_v2(
    raw_data: dict,
    head,
    rig: dict,
    shot_frames: list[int] | None = None,
    fps: float = 24.0,
    ground=None,
    smoothing: float = 0.5,
    refine_pose: bool = True,
) -> dict | None:
    """Whole-shot solve of one tracked person (see `shotspace.solver.trajectory`): one body for the shot, a
    world-space trajectory fitted to every frame's 2D evidence through the exact camera, planted feet tied to
    single world points and to the scan surface, then joint angles refined against the same evidence."""
    from shotspace.solver.trajectory import SolveConfig
    from shotspace.solver.workers.body_solve import keypoints_cam_cm, refine_pose_shot, solve_body_shot

    det_frames = [int(f) for f in np.asarray(raw_data["frames"]).reshape(-1)]
    use = solvable_frames(raw_data, fps=float(fps))
    if int(use.sum()) < 2:
        return None
    sm = float(np.clip(smoothing, 0.0, 1.0))
    gain = 0.5 / max(sm, 0.08)  # slider 0.5 = calibrated default; higher = smoother, lower = follows evidence more
    cfg = SolveConfig(fps=float(fps), sigma_acc_cm=0.9 * gain)
    kp_vis = np.asarray(raw_data["kp_in_mask"], dtype=bool) if "kp_in_mask" in raw_data else None
    from shotspace.solver.trajectory import drop_short_runs

    sol = None
    for _attempt in range(3):
        cand = solve_body_shot(raw_data, head, fps=float(fps), ground=ground, pose_sigma=2.0 * sm, cfg=cfg,
                               use_frames=use, kp_visible=kp_vis)
        tau = np.asarray(cand["cam_t_m"], dtype=np.float64)
        # a frame the solve put behind or onto the camera, or could not place at all, carried no usable evidence:
        # it is taken out (it will be exported hidden) and the person is solved again without it
        bad = ~np.all(np.isfinite(tau), axis=1) | (tau[:, 2] < 0.3)
        if not bad.any():
            sol = cand
            break
        use = np.array(use, dtype=bool, copy=True)
        use[np.asarray(cand["index"])[bad]] = False
        use = drop_short_runs(use, np.asarray(det_frames, dtype=np.int64), 3)
        if int(use.sum()) < 2:
            return None
    if sol is None:
        return None
    if refine_pose:
        sol = refine_pose_shot(sol, raw_data, head, fps=float(fps), ground=ground, w_smooth=1.0 / gain)

    s_body = float(sol["body_scale"])
    ev = evaluate_mhr_chunks(head, sol["global_rot"], sol["body"], sol["hand"], sol["scale_params"], sol["shape"])
    offset_cm = (sol["cam_t_m"] * CV_TO_GL) * M_TO_CM
    jw_valid = ev["joint_world"].copy()
    jw_valid[:, :, :3, 3] = jw_valid[:, :, :3, 3] * s_body + offset_cm[:, None, :]
    verts_valid = ev["vertices"] * s_body + offset_cm[:, None, :]

    v_frames = [int(f) for f in sol["frames"]]
    full_frames = sorted(set(int(f) for f in shot_frames)) if shot_frames else list(range(det_frames[0], det_frames[-1] + 1))
    src_vi, visible, anchor_frames = expand_to_full_shot_with_visibility(full_frames, v_frames, np.ones(len(v_frames)))
    bind = np.array(rig["bind_world"], dtype=np.float64, copy=True)
    bind[:, :3, 3] *= s_body
    kp_cam = keypoints_cam_cm(sol)
    contact = np.asarray(sol["contact"], dtype=np.uint8)
    idx = sol["index"]

    # quality read-outs the user can quote: how well the solved body sits on the per-frame 2D evidence (pixels),
    # and how far planted feet stray from their footprint in the world (cm)
    from shotspace.solver.trajectory import KP_L_FOOT, KP_R_FOOT, footprint_groups, keypoints_world_m

    c2w_used = np.asarray(raw_data["cam_to_world"], dtype=np.float64)[idx]
    kp_w = keypoints_world_m(sol["kp_local_m"], sol["cam_t_m"], s_body, c2w_used)
    stray = []
    for side, pieces in footprint_groups(np.asarray(sol["contact"], dtype=bool), sol["frames"], kp_w, float(fps)):
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in pieces])
        for kp in (KP_L_FOOT, KP_R_FOOT)[side]:
            stray.append(np.linalg.norm(kp_w[rows, kp] - np.median(kp_w[rows, kp], axis=0), axis=1))
    foot_hold_cm = float(np.percentile(np.concatenate(stray), 95) * 100.0) if stray else -1.0
    planted_pct = float(np.asarray(sol["contact"], dtype=bool).any(axis=1).mean() * 100.0)
    qc = {
        "fit_px": round(float(np.median(sol["reproj_px"])), 2),
        "fit_px_p90": round(float(np.percentile(sol["reproj_px"], 90)), 2),
        "foot_hold_cm": round(foot_hold_cm, 2),
        "planted_pct": round(planted_pct, 1),
        "body_scale": round(s_body, 4),
        "scale_source": str(sol["info"].get("scale_source", "network")),
        "scale_weight": float(sol["info"].get("scale_weight", 0.0)),
        "solved_frames": int(len(v_frames)),
        "hidden_frames": int(len(full_frames) - len(v_frames)),
    }
    return {
        "frames": np.asarray(full_frames, dtype=np.int64),
        "visible": visible.astype(np.int32),
        "solved_frames": np.asarray(v_frames, dtype=np.int64),
        "anchor_frames": anchor_frames.astype(np.int64),
        "rejected_count": np.int64(len(det_frames) - len(v_frames)),
        "joint_world": jw_valid[src_vi].astype(np.float32),
        "vertices": verts_valid[src_vi].astype(np.float32),
        "bind_world": bind,
        "rest_vertices": (evaluate_rest_vertices(head, sol["shape"][0]) * s_body).astype(np.float32),
        "faces": rig["faces"],
        "uv": rig["uv"],
        "uv_faces": rig["uv_faces"],
        "parents": rig["parents"],
        "joint_names": rig["joint_names"],
        "skin_indices": rig["skin_indices"],
        "skin_weights": rig["skin_weights"],
        "shape_params": sol["shape"][0].astype(np.float32),
        "scale_params": sol["scale_params"][0].astype(np.float32),
        "has_real_camera": np.int32(1),
        "smoothing_strength": np.float32(smoothing),
        "keypoints_2d": np.asarray(raw_data["pred_keypoints_2d"], dtype=np.float32)[idx][src_vi],
        "bbox": np.asarray(raw_data["bbox"], dtype=np.float32)[idx][src_vi],
        "solver_version": np.int32(2),
        "body_scale": np.float32(s_body),
        "keypoints_cam_cm": kp_cam[src_vi].astype(np.float32),
        "foot_contact": contact[src_vi],
        "reproj_px": np.asarray(sol["reproj_px"], dtype=np.float32)[src_vi],
        "scale_source": np.asarray(str(sol["info"].get("scale_source", "network"))),
        "qc_json": np.asarray(json.dumps(qc, ensure_ascii=False)),
    }


def solve_track(pid: int, raw_pack: dict, head, rig: dict, raw_net_dir: Path, cam_space_dir: Path, new_ids, fps: float,
                **kwargs) -> tuple[list[dict], dict | None]:
    """Whole-shot solve of one track -> (the people solved from it, the record of a cut or None).

    A track that jumps from one person to another is cut first (`shotspace.solver.track_split`): each part is
    solved as a person of its own.  The first part keeps the track's number, the others take new numbers from
    `new_ids`.  The cached network outputs of the parts are written next to the track's own under a temporary name
    (`.split_<track>_<person>.npz`); the task manager puts them in place when it takes the cut over, so a job that
    dies half-way leaves the track as it was.  `new_ids=None` (an older task manager) and manually boxed people:
    never cut."""
    from shotspace.solver.track_split import people_of_track

    parts, dropped = [raw_pack], 0
    if new_ids is not None and not int(np.asarray(raw_pack.get("is_manual", 0))):
        try:
            parts, dropped = people_of_track(raw_pack, solvable_frames(raw_pack, fps=float(fps)), float(fps))
        except Exception:  # noqa: BLE001 - a track that cannot be judged is solved as it is
            traceback.print_exc()
            parts, dropped = [raw_pack], 0
    cut = len(parts) > 1
    record = {"from": int(pid), "parts": [], "dropped": int(dropped)} if cut else None
    people: list[dict] = []
    for k, part in enumerate(parts):
        part_id = int(pid) if k == 0 else int(next(new_ids))
        if cut:
            _atomic_save_npz(raw_net_dir / f".split_{int(pid):02d}_{part_id:02d}.npz", **part)
            part_frames = np.asarray(part["frames"]).reshape(-1)
            record["parts"].append({"id": part_id, "first": int(part_frames[0]), "last": int(part_frames[-1]), "frames": int(len(part_frames))})
        mhr_data = _solve_person_safely(part_id, part, head, rig, fps=fps, **kwargs)
        if mhr_data is None:
            continue
        cam_file = cam_space_dir / f"person_{part_id:02d}.npz"
        _atomic_save_npz(cam_file, **mhr_data)
        entry = {
            "id": part_id,
            "file": cam_file.name,
            "frames": [int(f) for f in mhr_data["solved_frames"].tolist()],
            "rejected_count": int(mhr_data["rejected_count"]),
            "qc": json.loads(str(mhr_data["qc_json"])),
        }
        if k > 0:
            entry["split_from"] = int(pid)
        people.append(entry)
    return people, record


def _new_ids(job: dict):
    """Numbers for the people a cut track turns into, or None when the task manager did not give a first one."""
    start = job.get("new_id_start")
    return None if start is None else itertools.count(int(start))


def _solve_person_safely(pid: int, raw_pack: dict, head, rig: dict, **kwargs) -> dict | None:
    """One person's whole-shot solve.  A failure of one person (degenerate track, numerical trouble) must not
    lose everybody else in a crowd: it is reported and that person is skipped."""
    try:
        res = postprocess_person_v2(raw_pack, head, rig, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            raise
        traceback.print_exc()
        emit("warning", message=f"人物 {pid} 的整段解算失败，已跳过该人物：{type(exc).__name__}: {exc}")
        return None
    if res is None:
        emit("warning", message=f"人物 {pid} 可解算帧数不足，已跳过")
        return None
    for key in ("joint_world", "vertices", "keypoints_cam_cm"):
        if key in res and not np.all(np.isfinite(np.asarray(res[key], dtype=np.float64))):
            emit("warning", message=f"人物 {pid} 的解算结果含无效数值，已跳过该人物")
            return None
    return res


def load_ground_mesh(scan_npz: str | None):
    """Imported scene scan (world cm) -> supporting-surface query in metres, or None."""
    if not scan_npz or not Path(scan_npz).is_file():
        return None
    from shotspace.solver.trajectory import GroundMesh

    # scans that had to be reduced for the viewport keep their full-resolution mesh next to the cache
    full = Path(scan_npz).with_name("scan_solver.npz")
    with np.load(full if full.is_file() else scan_npz, allow_pickle=False) as d:
        verts, faces = np.asarray(d["vertices_cm"], dtype=np.float64) * 0.01, np.asarray(d["faces"], dtype=np.int64)
    mesh = GroundMesh(verts, faces)
    return mesh if len(mesh) else None


class _PersonTrack:
    """Boxes and scores of one tracked person, keyed by frame number."""

    def __init__(self, boxes: dict[int, np.ndarray], scores: dict[int, float]) -> None:
        self.boxes = boxes
        self.scores = scores


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _batches(present: list[tuple[int, np.ndarray]], size: int) -> list[list[tuple[int, np.ndarray]]]:
    ordered = sorted(present, key=lambda pb: float((pb[1][2] - pb[1][0]) * (pb[1][3] - pb[1][1])), reverse=True)
    s = max(1, int(size))
    return [ordered[i : i + s] for i in range(0, len(ordered), s)]


# ------------------------------------------------------------------ Job Handlers


def run_solve(job: dict, device: torch.device, weights_dir: Path) -> dict:
    from sam_3d_body import SAM3DBodyEstimator

    grade = GradeParams.from_dict(job.get("grade"))
    lut = build_uint8_lut(grade)
    frames_spec: list[tuple[int, str]] = [(int(item["frame"]), str(item["path"])) for item in job["frames"]]

    # Selected people tracks: [{"id": 1, "is_manual": False, "boxes": {"1001": [x1,y1,x2,y2], ...}}, ...]
    people_in = job["people"]
    manual_pids: set[int] = {int(p["id"]) for p in people_in if bool(p.get("is_manual", False))}
    # a person that was cut out of another track (`track_split`) is outlined under that track's number
    mask_ids: dict[int, int] = {int(p["id"]): int(p.get("mask_id", p["id"])) for p in people_in}
    selected: list[tuple[int, _PersonTrack]] = [
        (
            int(p["id"]),
            _PersonTrack(
                boxes={int(k): np.asarray(v, dtype=np.float64) for k, v in (p.get("boxes") or {}).items()},
                scores={int(k): float(v) for k, v in (p.get("scores") or {}).items()},
            ),
        )
        for p in people_in
    ]
    if not selected:
        raise ValueError("未选中任何可解算的人物轨迹")

    track_map = {pid: tr for pid, tr in selected}
    shot_frames = [int(f) for f, _ in frames_spec]
    img_sz_raw = job.get("image_size") or [0, 0]
    img_sz: list[int] = [int(img_sz_raw[0]), int(img_sz_raw[1])]

    # Load exact per-frame [3, 3] camera intrinsics AND [4, 4] extrinsics (100% from imported camera, ZERO blind FOV!)
    intr_npz = np.load(job["camera_intrinsics_npz"])
    intr_frames = [int(f) for f in intr_npz["frames"]]
    intr_mats = np.asarray(intr_npz["K"], dtype=np.float32)  # [F, 3, 3]
    c2w_mats_in = (
        np.asarray(intr_npz["cam_to_world"], dtype=np.float64)
        if "cam_to_world" in intr_npz
        else np.repeat(np.eye(4, dtype=np.float64)[None], len(intr_frames), axis=0)
    )
    k_by_frame = {f: torch.as_tensor(intr_mats[i : i + 1], dtype=torch.float32) for i, f in enumerate(intr_frames)}
    k_np_by_frame = {f: intr_mats[i] for i, f in enumerate(intr_frames)}
    c2w_by_frame = {f: c2w_mats_in[i] for i, f in enumerate(intr_frames)}
    known_k_frames = sorted(k_by_frame)

    def cam_int_for(f: int) -> torch.Tensor:
        if f in k_by_frame:
            return k_by_frame[f]
        nearest = min(known_k_frames, key=lambda kf: abs(kf - f))
        return k_by_frame[nearest]

    def cam_k_np_for(f: int) -> np.ndarray:
        if f in k_np_by_frame:
            return k_np_by_frame[f]
        nearest = min(known_k_frames, key=lambda kf: abs(kf - f))
        return k_np_by_frame[nearest]

    def cam_c2w_for(f: int) -> np.ndarray:
        if f in c2w_by_frame:
            return c2w_by_frame[f]
        nearest = min(known_k_frames, key=lambda kf: abs(kf - f))
        return c2w_by_frame[nearest]

    model, cfg = get_sam3d_body(weights_dir / "sam-3d-body-dinov3", device)
    patch_per_person_prompts(model)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=None, fov_estimator=None)

    inference_type = "full" if bool(job.get("hand_refine", True)) else "body"
    use_mask = bool(job.get("use_mask", False))
    has_real_camera = bool(job.get("has_real_camera", True))
    masks_archive = None
    masks_npz_path = Path(job["person_masks_npz"]) if job.get("person_masks_npz") else None
    if use_mask and masks_npz_path is not None and masks_npz_path.is_file():
        masks_archive = dict(np.load(masks_npz_path, allow_pickle=False))

    mask_rect: dict[int, tuple[int, int, int, int]] = {}  # id(full-frame mask) -> the rectangle it is non-zero in

    def _full_mask_for(pid: int, fnum: int, bx: np.ndarray, h_full: int, w_full: int) -> np.ndarray | None:
        """Full-frame binary mask of one tracked person.  Masks written by the people tracker carry the plate
        rectangle they cover (`_rect`), so they are placed exactly where they were measured; legacy masks without
        it are stretched onto the given box."""
        if masks_archive is None:
            return None
        pid = mask_ids.get(pid, pid)
        kb = f"p{pid}_f{fnum}_bits"
        ks = f"p{pid}_f{fnum}_shape"
        if kb not in masks_archive or ks not in masks_archive:
            return None
        mh, mw = int(masks_archive[ks][0]), int(masks_archive[ks][1])
        if mh <= 0 or mw <= 0:
            return None
        bits = np.unpackbits(masks_archive[kb])[: mh * mw].reshape(mh, mw)
        kr = f"p{pid}_f{fnum}_rect"
        rect = masks_archive[kr] if kr in masks_archive else bx
        x1 = max(0, min(w_full - 1, int(round(float(rect[0])))))
        y1 = max(0, min(h_full - 1, int(round(float(rect[1])))))
        x2 = max(x1 + 1, min(w_full, int(round(float(rect[2])))))
        y2 = max(y1 + 1, min(h_full, int(round(float(rect[3])))))
        full_m = np.zeros((h_full, w_full), dtype=np.uint8)
        resized = (
            cv2.resize(bits.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            if (mh != (y2 - y1) or mw != (x2 - x1))
            else bits.astype(np.uint8)
        )
        if int(resized.sum()) < 16:
            return None
        full_m[y1:y2, x1:x2] = resized
        mask_rect[id(full_m)] = (x1, y1, x2, y2)
        return full_m

    def _mask_stats(fm: np.ndarray | None, kp2d: np.ndarray, bx: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Which predicted keypoints land on the person's own visible pixels (tolerance ~1.5 % of the box height),
        plus the visible area and its tight box.  Without a mask everything counts as visible.

        The outline is non-zero only inside the rectangle it was pasted into, so everything is computed on that
        rectangle grown by the tolerance: the same numbers as on the whole frame, at a fraction of the cost (on a 4K
        plate the whole-frame version took 28 ms per person and frame)."""
        n_kp = kp2d.shape[0]
        if fm is None:
            return np.ones(n_kp, dtype=np.uint8), float((bx[2] - bx[0]) * (bx[3] - bx[1])), np.asarray(bx, dtype=np.float32)
        hh, ww = fm.shape
        rx1, ry1, rx2, ry2 = mask_rect.get(id(fm), (0, 0, ww, hh))
        ys, xs = np.nonzero(fm[ry1:ry2, rx1:rx2])
        ys, xs = ys + ry1, xs + rx1
        m_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        rad = int(max(2, round(0.015 * float(m_box[3] - m_box[1]))))
        gx1, gy1 = max(0, rx1 - rad - 1), max(0, ry1 - rad - 1)
        gx2, gy2 = min(ww, rx2 + rad + 1), min(hh, ry2 + rad + 1)
        dil = cv2.dilate(
            np.ascontiguousarray(fm[gy1:gy2, gx1:gx2]),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)),
        )
        u = np.clip(np.round(kp2d[:, 0]).astype(np.int64), 0, ww - 1)
        v = np.clip(np.round(kp2d[:, 1]).astype(np.int64), 0, hh - 1)
        inside = (kp2d[:, 0] >= 0) & (kp2d[:, 0] < ww) & (kp2d[:, 1] >= 0) & (kp2d[:, 1] < hh)
        in_grown = (u >= gx1) & (u < gx2) & (v >= gy1) & (v < gy2)
        hit = np.zeros(n_kp, dtype=bool)
        hit[in_grown] = dil[v[in_grown] - gy1, u[in_grown] - gx1].astype(bool)
        return (hit & inside).astype(np.uint8), float(len(xs)), m_box

    # v0.2.12: frames whose outline took in a piece far from the person (`shotspace.solver.box_repair`): the outline
    # is cleaned before the network sees it, the box becomes the box of what is left
    repair_regions: dict[int, dict[int, np.ndarray]] = {
        int(p["id"]): {int(k): np.asarray(v, dtype=np.float64) for k, v in (p.get("repaired") or {}).items()}
        for p in people_in
        if p.get("repaired") and not bool(p.get("is_manual", False))
    }

    box_pad = float(job.get("mask_box_pad", 0.0))
    amodal_boxes: dict[int, dict[int, np.ndarray]] = {
        int(k): {int(f): np.asarray(b, dtype=np.float64) for f, b in v.items()} for k, v in (job.get("prompt_boxes") or {}).items()
    }

    def _prompt_box(pid: int, fnum: int, bx: np.ndarray) -> np.ndarray:
        """Crop box given to the network together with the person's mask: an explicit full-body box when the caller
        supplies one (second pass), otherwise the visible-mask box enlarged by `mask_box_pad` so a partly hidden
        person keeps some context around the visible part.

        The box is scaled about its own centre, by the same factor in both directions, and only as far as the
        nearest picture border allows.  Enlarging just the sides that have room moves the person off the centre of
        the crop, and the network then misjudges someone cut by the frame edge: measured on Panoptic pizza1 with a
        virtual pan (person cut at the bottom edge) 9.0 cm / 40 px with the one-sided enlargement against
        3.5 cm / 26 px with the tight box - while away from the borders the enlargement helps (8.4 against 9.2 cm
        on the plain clip)."""
        if pid in amodal_boxes and fnum in amodal_boxes[pid]:
            return amodal_boxes[pid][fnum]
        if box_pad <= 0:
            return bx
        bw, bh = max(float(bx[2] - bx[0]), 1e-6), max(float(bx[3] - bx[1]), 1e-6)
        room = min(float(bx[0]) / bw, (float(img_sz[0]) - 1.0 - float(bx[2])) / bw,
                   float(bx[1]) / bh, (float(img_sz[1]) - 1.0 - float(bx[3])) / bh)
        grow = max(0.0, min(float(box_pad), room))
        return np.array([bx[0] - grow * bw, bx[1] - grow * bh, bx[2] + grow * bw, bx[3] + grow * bh], dtype=np.float64)

    per_person: dict[int, dict[int, dict]] = {pid: {} for pid, _ in selected}
    cur_batch = [PEOPLE_BATCH]
    batch_mode = str(os.environ.get("SHOTSPACE_SAM3D_BATCH_MODE") or PERSON_BATCH_MODE).strip().lower()
    _PROMPT_PATCH["on"] = batch_mode == "consistent"

    def _estimate(rgb_img: np.ndarray, boxes: np.ndarray, masks: np.ndarray | None, k_cam) -> list[dict]:
        """One call into the network for the given people of one frame, in the configured batch mode."""

        def _call(b: np.ndarray, m: np.ndarray | None) -> list[dict]:
            with quiet():
                if m is not None:
                    return estimator.process_one_image(
                        rgb_img, bboxes=b, masks=m, cam_int=k_cam, use_mask=True, inference_type=inference_type
                    )
                return estimator.process_one_image(rgb_img, bboxes=b, cam_int=k_cam, inference_type=inference_type)

        n = int(len(boxes))
        if batch_mode == "single" and n > 1:
            outs: list[dict] = []
            for i in range(n):
                outs.extend(_call(boxes[i : i + 1], None if masks is None else masks[i : i + 1]))
            return outs
        if batch_mode == "multi" and n == 1:
            # the same person twice: the call takes the several-people path, the second copy is thrown away
            return list(_call(np.concatenate([boxes, boxes]), None if masks is None else np.concatenate([masks, masks])))[:1]
        return list(_call(boxes, masks))

    emit(
        "stage",
        stage="sam3d_infer",
        message=f"逐帧估计：正在估计所选 {len(selected)} 人（{len(frames_spec)} 帧）...",
    )
    total = len(frames_spec)
    with keep_gpu_cache():
        for idx, (fnum, fpath) in enumerate(frames_spec, start=1):
            present = [(pid, t.boxes[fnum]) for pid, t in selected if fnum in t.boxes]
            _COMPACT_OUT_KEYS = (
                "shape_params",
                "scale_params",
                "body_pose_params",
                "hand_pose_params",
                "global_rot",
                "pred_cam_t",
                "pred_keypoints_2d",
                "bbox",
            )
            if present:
                rgb = _read_graded_rgb(fpath, grade, lut)
                if img_sz[0] <= 0 or img_sz[1] <= 0:
                    img_sz = [int(rgb.shape[1]), int(rgb.shape[0])]
                k_t = cam_int_for(fnum)
                while True:
                    size = cur_batch[0]
                    try:
                        pairs = []
                        fm_by_pid: dict[int, np.ndarray | None] = {}
                        mask_rect.clear()
                        used_box: dict[int, np.ndarray] = {}
                        if use_mask and masks_archive is not None:
                            with_m, without_m = [], []
                            for pid, bx in present:
                                fm = _full_mask_for(pid, fnum, bx, img_sz[1], img_sz[0])
                                region = repair_regions.get(pid, {}).get(fnum)
                                if region is not None and fm is not None:
                                    cleaned = clean_outline(fm, region, mask_rect.get(id(fm)))
                                    if cleaned is None:
                                        # nothing of the outline lies where the person is: on this frame the tracker
                                        # did not see them (it outlined something else) - the frame is left out
                                        continue
                                    fm, bx = cleaned
                                    mask_rect[id(fm)] = (int(bx[0]), int(bx[1]), int(np.ceil(bx[2])), int(np.ceil(bx[3])))
                                used_box[pid] = bx
                                fm_by_pid[pid] = fm
                                if fm is not None:
                                    with_m.append((pid, _prompt_box(pid, fnum, bx), fm))
                                else:
                                    without_m.append((pid, bx))
                            for lo_c in range(0, len(with_m), max(1, size)):
                                sub = with_m[lo_c : lo_c + max(1, size)]
                                outs = _estimate(rgb, np.stack([b for _, b, _ in sub]), np.stack([m for _, _, m in sub]), k_t)
                                pairs.extend(zip([pid for pid, _, _ in sub], outs))
                            for chunk in _batches(without_m, size):
                                outs = _estimate(rgb, np.stack([b for _, b in chunk]), None, k_t)
                                pairs.extend(zip([pid for pid, _ in chunk], outs))
                        else:
                            for chunk in _batches(present, size):
                                outs = _estimate(rgb, np.stack([b for _, b in chunk]), None, k_t)
                                pairs.extend(zip([pid for pid, _ in chunk], outs))
                        # Retain ONLY compact MHR parameters (< 2 KB/frame vs 660 KB with pred_vertices) so 50+ people never bloat RAM!
                        box_of = {**dict(present), **used_box}
                        for pid, out_d in pairs:
                            rec = {kk: np.asarray(out_d[kk], dtype=np.float32) for kk in _COMPACT_OUT_KEYS if kk in out_d}
                            kin, area, m_box = _mask_stats(fm_by_pid.get(pid), rec["pred_keypoints_2d"], box_of[pid])
                            rec["kp_in_mask"], rec["mask_area"], rec["mask_box"] = kin, np.float32(area), m_box
                            rec["has_mask"] = np.uint8(1 if fm_by_pid.get(pid) is not None else 0)
                            per_person[pid][fnum] = rec
                        break
                    except RuntimeError as exc:
                        if not _is_oom(exc):
                            raise
                        _free_memory()
                        smaller = [s for s in BATCH_STEPS if s < size]
                        if not smaller:
                            raise RuntimeError(f"GPU 显存不足：即使单人批大小 (batch=1) 也在第 {fnum} 帧发生 OOM") from exc
                        cur_batch[0] = smaller[0]
                        emit("warning", message=f"第 {fnum} 帧显存受限，自动将每帧并发批大小从 {size} 降至 {smaller[0]}")
            if idx == 1 or idx % 2 == 0 or idx == total:
                emit("progress", stage="sam3d_infer", done=idx, total=total, frame=fnum)

    # Save raw network outputs so future smoothing / lock_shape tweaks run in ~0.2 s via `run_postprocess`!
    raw_net_dir = Path(job["raw_net_dir"])
    cam_space_dir = Path(job["cam_space_dir"])
    raw_net_dir.mkdir(parents=True, exist_ok=True)
    cam_space_dir.mkdir(parents=True, exist_ok=True)

    emit("stage", stage="mhr_postprocess", message="整段解算：正在逐人进行整段解算...")
    # Offload DINOv3-Huge backbone to CPU before multi-person GPU trajectory BA so MHR has full 24GB VRAM headroom!
    if hasattr(model, "backbone"):
        _move_model_tensors(model.backbone, "cpu")
        _free_memory()
    head = model.head_pose
    rig = extract_rig_data(head)
    smoothing = float(job.get("smoothing", 0.5))
    fps = float(job.get("fps") or 24.0)
    ground = load_ground_mesh(job.get("scan_npz"))

    solved_people = []
    splits: list[dict] = []
    new_ids = _new_ids(job)
    non_empty_pids = [pid for pid, outs in per_person.items() if outs]
    for p_step, pid in enumerate(non_empty_pids, start=1):
        outputs = per_person[pid]
        emit(
            "stage",
            stage="mhr_postprocess",
            message=f"整段解算：第 {p_step}/{len(non_empty_pids)} 人...",
        )
        emit("progress", stage="mhr_postprocess", done=p_step, total=len(non_empty_pids))
        p_frames = sorted(outputs)
        tr_scores = track_map[pid].scores
        raw_kpts2d = np.stack([np.asarray(outputs[f]["pred_keypoints_2d"], dtype=np.float32) for f in p_frames])
        raw_pack = {
            "frames": np.asarray(p_frames, dtype=np.int64),
            "shot_frames": np.asarray(shot_frames, dtype=np.int64),
            "is_manual": np.int32(1 if pid in manual_pids else 0),
            "image_size": np.asarray(img_sz, dtype=np.int32),
            "has_real_camera": np.int32(1 if has_real_camera else 0),
            "det_scores": np.asarray([float(tr_scores.get(f, 1.0)) for f in p_frames], dtype=np.float32),
            "cam_K": np.stack([cam_k_np_for(f) for f in p_frames]).astype(np.float32),
            "cam_to_world": np.stack([cam_c2w_for(f) for f in p_frames]).astype(np.float64),
            "shape_params": np.stack([np.asarray(outputs[f]["shape_params"], dtype=np.float32) for f in p_frames]),
            "scale_params": np.stack([np.asarray(outputs[f]["scale_params"], dtype=np.float32) for f in p_frames]),
            "body_pose_params": np.stack([np.asarray(outputs[f]["body_pose_params"], dtype=np.float32) for f in p_frames]),
            "hand_pose_params": np.stack([np.asarray(outputs[f]["hand_pose_params"], dtype=np.float32) for f in p_frames]),
            "global_rot": np.stack([np.asarray(outputs[f]["global_rot"], dtype=np.float32) for f in p_frames]),
            "pred_cam_t": np.stack([np.asarray(outputs[f]["pred_cam_t"], dtype=np.float32) for f in p_frames]),
            "pred_keypoints_2d": raw_kpts2d,
            "bbox": np.stack([np.asarray(outputs[f]["bbox"], dtype=np.float32) for f in p_frames]),
            "kp_in_mask": np.stack([np.asarray(outputs[f]["kp_in_mask"], dtype=np.uint8) for f in p_frames]),
            "mask_area": np.asarray([float(outputs[f]["mask_area"]) for f in p_frames], dtype=np.float32),
            "mask_box": np.stack([np.asarray(outputs[f]["mask_box"], dtype=np.float32) for f in p_frames]),
            "has_mask": np.asarray([int(outputs[f]["has_mask"]) for f in p_frames], dtype=np.uint8),
        }
        raw_file = raw_net_dir / f"person_{pid:02d}.npz"
        _atomic_save_npz(raw_file, **raw_pack)

        people_of_it, cut = solve_track(
            pid, raw_pack, head, rig, raw_net_dir, cam_space_dir, new_ids, fps,
            shot_frames=shot_frames, ground=ground, smoothing=smoothing,
        )
        solved_people.extend(people_of_it)
        if cut is not None:
            splits.append(cut)

    return {"people": solved_people, "splits": splits}


def run_postprocess(job: dict, device: torch.device, weights_dir: Path) -> dict:
    """Fast ~0.2 s re-evaluation of MHR from cached `raw_net/person_XX.npz` when only `lock_shape` or `smoothing` changes!"""
    raw_net_dir = Path(job["raw_net_dir"])
    cam_space_dir = Path(job["cam_space_dir"])
    cam_space_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_net_dir.glob("person_*.npz"))
    if job.get("person_ids") is not None:
        wanted = {int(v) for v in job["person_ids"]}
        raw_files = [rf for rf in raw_files if int(rf.stem.split("_")[-1]) in wanted]
    if not raw_files:
        raise FileNotFoundError(f"未找到缓存的网络原始输出：{raw_net_dir}")

    model, _ = get_sam3d_body(weights_dir / "sam-3d-body-dinov3", device)
    head = model.head_pose
    rig = extract_rig_data(head)
    smoothing = float(job.get("smoothing", 0.5))
    fps = float(job.get("fps") or 24.0)
    ground = load_ground_mesh(job.get("scan_npz"))
    job_shot_frames = [int(f) for f in job["shot_frames"]] if job.get("shot_frames") else None
    cam_override: dict[int, np.ndarray] = {}
    if job.get("camera_extrinsics_npz") and Path(job["camera_extrinsics_npz"]).is_file():
        with np.load(job["camera_extrinsics_npz"], allow_pickle=False) as ce:
            cam_override = {int(f): np.asarray(m, dtype=np.float64) for f, m in zip(ce["frames"], ce["cam_to_world"])}

    emit("stage", stage="mhr_postprocess", message="整段解算：按当前摄影机、扫描与平滑设置重新解算...")
    solved_people = []
    splits: list[dict] = []
    new_ids = _new_ids(job)
    for p_step, rf in enumerate(raw_files, start=1):
        pid = int(rf.stem.split("_")[-1])
        raw_pack = dict(np.load(rf, allow_pickle=False))
        if cam_override:
            # the camera move was re-imported: place the cached per-frame estimates under the new extrinsics
            c2w_new = np.array(raw_pack["cam_to_world"], dtype=np.float64, copy=True)
            for i, f in enumerate(raw_pack["frames"]):
                if int(f) in cam_override:
                    c2w_new[i] = cam_override[int(f)]
            raw_pack["cam_to_world"] = c2w_new
        sf = job_shot_frames or (
            [int(f) for f in raw_pack["shot_frames"].tolist()] if "shot_frames" in raw_pack else None
        )
        emit("progress", stage="mhr_postprocess", done=p_step, total=len(raw_files))
        people_of_it, cut = solve_track(
            pid, raw_pack, head, rig, raw_net_dir, cam_space_dir, new_ids, fps, shot_frames=sf, ground=ground, smoothing=smoothing
        )
        solved_people.extend(people_of_it)
        if cut is not None:
            splits.append(cut)
    return {"people": solved_people, "splits": splits}


def execute_job(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sam3d_repo = os.environ["SHOTSPACE_SAM3D_REPO"]
    dinov3_dir = os.environ["SHOTSPACE_DINOV3_DIR"]
    weights_dir = Path(os.environ["SHOTSPACE_WEIGHTS_DIR"])
    if sam3d_repo not in sys.path:
        sys.path.insert(0, sam3d_repo)

    _forbid_network()
    _install_local_hub(dinov3_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 NVIDIA CUDA GPU，无法运行 SAM 3D Body")
    _limit_gpu_memory(768)
    device = torch.device("cuda")

    cmd = job["command"]
    if cmd == "solve":
        res = run_solve(job, device, weights_dir)
    elif cmd == "postprocess":
        res = run_postprocess(job, device, weights_dir)
    else:
        raise ValueError(f"Unknown SAM3D worker command: {cmd!r}")

    if "result_file" in job:
        rf = Path(job["result_file"])
        rf.parent.mkdir(parents=True, exist_ok=True)
        rf.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("done", result=res)


def serve_loop() -> None:
    _pin_stdout()
    commands = os.fdopen(os.dup(0), "r", encoding="utf-8")
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)

    parent_pid = int(os.environ.get("SHOTSPACE_PARENT_PID") or os.getppid())

    def _watch_parent() -> None:
        if os.name == "nt":
            # a Windows venv's python.exe is a launcher that starts the real interpreter: the direct parent is that
            # launcher, never the service - ask whether the service itself still runs
            from shotspace.core.platform_compat import pid_alive

            while pid_alive(parent_pid):
                time.sleep(2.0)
            os._exit(1)
        while os.getppid() == parent_pid:
            time.sleep(2.0)
        os._exit(1)

    threading.Thread(target=_watch_parent, daemon=True).start()

    for line in commands:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        action = msg.get("cmd")
        if action == "exit":
            break
        if action == "offload":
            offload_all()
            emit("offloaded")
            continue
        if action == "job":
            try:
                execute_job(msg["job"])
                _free_memory()
                emit("job_end", ok=True, error=None)
            except BaseException as exc:
                traceback.print_exc()
                sys.stderr.flush()
                _free_memory()
                emit("job_end", ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    _pin_stdout()
    if sys.argv[1:] == ["--serve"]:
        serve_loop()
    elif len(sys.argv) == 2:
        execute_job(sys.argv[1])
    else:
        raise SystemExit("Usage: sam3d_worker.py <job.json> | sam3d_worker.py --serve")
