"""Proprietary Camera Space Conversion & Dual-Camera Alignment Engine.

Supports three production scenarios:
  A. Direct Camera-Space Placement (`source_camera=None`):
       SAM 3D Body origin camera-space output -> target camera world via T(f) = C_dst(f).
  B. Dual-Camera Per-Frame Exact Fit (`fit="per_frame"`):
       Checks normalized lens FOV match within 2% (`LENS_TOLERANCE = 0.02`), then applies
       T(f) = C_dst(f) @ inv(C_src(f)) frame by frame.
  C. Dual-Camera Whole-Shot Scaled Similarity Fit (`fit="whole"`):
       Fits constant similarity T_fix = [s*R, t; 0, 1] over common authored frames `both`:
         1. R from SVD chordal mean of camera orientations (never degenerates on dolly/pan);
         2. s from least-squares on optical centres ONLY when target RMS spread >= 10.0 cm (`MOVED_CM`),
            preventing handheld tripod noise from shrinking characters to 3%;
         3. abs(s) prevents negative mirror flip;
         4. Applies scale s to characters/meshes (preserving perspective projection size), while stripping
            scale from moved cameras (`cbrt(|det|)`) so DCC cameras remain strictly rigid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

from .camera import CameraTrack
from .poses import rotation_deg, scaled_align
from .rigid_object import RigidObjectTrack
from .skeleton import SkinnedCharacter

LENS_TOLERANCE = 0.02  # 2% normalized focal/aperture ratio tolerance
MOVED_CM = 10.0  # Target camera RMS spread threshold (cm) before fitting scale
WARN_RESIDUAL_CM = 25.0  # RMS centre error above which two tracks likely come from mismatched shots


class LensMismatchError(ValueError):
    """Raised in `per_frame` mode when source and target cameras have > 2% FOV discrepancy."""

    def __init__(self, diff_pct: float) -> None:
        self.diff_pct = float(diff_pct)
        super().__init__(
            f"来源摄影机与目标摄影机的视场角（焦距/片门比）相差 {diff_pct:.2f}%（超过 2% 阈值），"
            f"不存在能使三维角色同时贴合两台相机画面的刚体变换。请改用「整段平滑（whole）」模式，或直接使用目标摄影机重新解算动作。"
        )


@dataclass
class AlignmentReport:
    mode: Literal["direct", "same", "per_frame", "whole"]
    fit_requested: str
    frames: list[int]
    common_authored_frames: int
    scale: float = 1.0
    rotation_deg: float = 0.0
    translation_cm: float = 0.0
    rms_pos_error_cm: float = 0.0
    rms_ang_error_deg: float = 0.0
    lens_diff_pct: float = 0.0
    target_spread_cm: float = 0.0
    scale_locked_still: bool = False
    track_reversed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignmentTransform:
    """Computed camera-space transform: either per-frame [F, 4, 4] rigid matrices or constant [4, 4] similarity."""

    frames: tuple[int, ...]
    matrices: np.ndarray  # [4, 4] (constant whole-shot) or [F, 4, 4] (per-frame)
    report: AlignmentReport

    @property
    def is_per_frame(self) -> bool:
        return self.matrices.ndim == 3

    def matrix_for_frame_map(self) -> dict[int, np.ndarray]:
        if self.matrices.ndim == 2:
            return {int(f): self.matrices for f in self.frames}
        return {int(f): self.matrices[i] for i, f in enumerate(self.frames)}

    def apply_to_character(self, ch: SkinnedCharacter) -> SkinnedCharacter:
        """Transform a SkinnedCharacter into target world space (including uniform scale `s` when `whole`)."""
        f_map = self.matrix_for_frame_map()
        fallback = self.matrices if self.matrices.ndim == 2 else self.matrices[0]
        mats = np.stack([f_map.get(int(f), fallback) for f in ch.frames], axis=0)  # [Fc, 4, 4]

        if self.matrices.ndim == 2:
            fix = self.matrices
            s = float(np.cbrt(abs(np.linalg.det(fix[:3, :3]))))
            bind_new = fix[None] @ ch.bind_world
            anim_new = fix[None, None] @ ch.anim_world
            pts_new = (ch.rest_points.astype(np.float64) @ fix[:3, :3].T + fix[:3, 3]).astype(np.float32)
            # Note: in UsdSkel / FBX skinning, world mesh vertices in bind pose + world bind_transforms + world anim_transforms
            # satisfy: v_skinned(t) = anim_new(t) @ inv(bind_new) @ pts_new
            #                       = (fix @ anim(t)) @ (inv(bind) @ inv(fix)) @ (fix @ pts)
            #                       = fix @ (anim(t) @ inv(bind) @ pts), which scales & places the character 100% identically!
            return replace(ch, bind_world=bind_new, anim_world=anim_new, rest_points=pts_new,
                           custom_data={**ch.custom_data, "shotspace:scale": round(s, 6)})
        else:
            # Per-frame rigid placement: bind_world and rest_points stay in canonical rig space,
            # while anim_world[t, j] = T(t) @ anim_world[t, j]
            anim_new = mats[:, None, :, :] @ ch.anim_world
            return replace(ch, anim_world=anim_new)

    def apply_to_camera(self, cam: CameraTrack) -> CameraTrack:
        """Move a camera into target world space, stripping scale `s` from its orientation basis."""
        resampled = cam.resample_to(self.frames) if self.frames and cam.frames != self.frames else cam
        m = self.matrices
        moved = (m[None] if m.ndim == 2 else m) @ resampled.cam_to_world
        scale = np.cbrt(np.abs(np.linalg.det(moved[:, :3, :3])))
        moved[:, :3, :3] /= np.maximum(scale[:, None, None], 1e-12)
        return replace(resampled, cam_to_world=moved)


def camera_spread_cm(poses: np.ndarray) -> float:
    """RMS distance (cm) of camera optical centres from their centroid."""
    c = np.asarray(poses, dtype=np.float64)[:, :3, 3]
    if len(c) == 0:
        return 0.0
    return float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))


def lens_diff_ratio(a: CameraTrack, b: CameraTrack) -> float:
    """Maximum relative discrepancy in normalized focal ratio (focal_mm / h_aperture_mm) across frames."""
    fa = np.asarray(a.focal_mm, dtype=np.float64) / np.where(a.h_aperture_mm > 0, a.h_aperture_mm, np.nan)
    fb = np.asarray(b.focal_mm, dtype=np.float64) / np.where(b.h_aperture_mm > 0, b.h_aperture_mm, np.nan)
    fa_b, fb_b = np.broadcast_arrays(fa, fb)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = float(np.nanmax(np.abs(fa_b / fb_b - 1.0))) if fa_b.size else 0.0
    return diff if np.isfinite(diff) else 0.0


def same_camera(a: CameraTrack, b: CameraTrack) -> bool:
    if a.frames != b.frames:
        return False
    return bool(
        np.allclose(a.cam_to_world, b.cam_to_world, atol=1e-6)
        and lens_diff_ratio(a, b) < 1e-9
    )


def standard_camera_focal_px(width: int, height: int) -> float:
    """SAM 3D Body's built-in standard pinhole focal length in pixels (`hypot(width, height)`)."""
    return float(np.hypot(float(width), float(height)))


def align_standard_character_to_camera(
    ch: SkinnedCharacter,
    target_camera: CameraTrack,
    plate_width: int,
    plate_height: int,
    shot_frames: list[int] | tuple[int, ...] | None = None,
) -> SkinnedCharacter:
    """Transform a solved `SAM 3D Body` character into `target_camera`'s 3D world space via pure rigid `SE(3)` transforms:
      1. When `shotspace:has_real_camera` is True (standard camera-first workflow), `SAM 3D Body` was already run
         natively under `target_camera`'s exact per-frame intrinsics `K(f)`, so `c2w(f) @ anim_cam(f)` is 100%
         perspective-exact (`0.00 px` 2D drift). For legacy origin-camera caches, applies viewing-ray rotation +
         focal-ratio translation around the pelvis.
      2. Performs ray-constrained world-space trajectory denoising AFTER `c2w(f)` (eliminating camera-motion swim
         while clamping cross-ray delta <= 0.85 px and along-ray delta <= 4.5% so 2D silhouette edges stay razor-sharp).
      3. Freezes `visible == 0` frames in 3D world space at their peeled valid anchor frame.
    """
    from .poses import matrix_to_quat, quat_to_matrix
    from .smoothing import gaussian_smooth, segments, smooth_quats, smoothing_sigma

    if not ch.frames:
        return ch
    w = max(1, int(plate_width or target_camera.width or 1920))
    h = max(1, int(plate_height or target_camera.height or 1080))
    f_std = standard_camera_focal_px(w, h)

    all_frames = sorted(set(int(f) for f in (shot_frames or ch.frames)) | set(int(f) for f in ch.frames))
    cam_res = target_camera.resample_to(all_frames)
    f_to_cam_idx = {int(f): idx for idx, f in enumerate(cam_res.frames)}
    k_map = cam_res.intrinsics_per_frame(w, h)

    vis_mask = ch.visible_mask
    f_to_char_idx = {int(f): idx for idx, f in enumerate(ch.frames)}
    anim_new = np.empty_like(ch.anim_world, dtype=np.float64)

    has_real_cam = bool((ch.custom_data or {}).get("shotspace:has_real_camera", False))
    smooth_strength = float((ch.custom_data or {}).get("shotspace:smoothing_strength", 0.5))
    pelvis_j = next(
        (j for j, jn in enumerate(ch.joint_names) if jn.lower() in ("hips", "root", "pelvis")),
        0,
    )

    # Pass 1: Transform genuinely visible frames into world space
    valid_indices = [t for t, ok in enumerate(vis_mask) if ok]
    if not valid_indices:
        valid_indices = list(range(len(ch.frames)))

    for t in valid_indices:
        fi = int(ch.frames[t])
        c_idx = f_to_cam_idx[fi]
        c2w = cam_res.cam_to_world[c_idx]

        if has_real_cam:
            # Exact native intrinsics match: `ch.anim_world[t]` is already in `target_camera`'s OpenGL camera space!
            anim_new[t] = c2w[None, :, :] @ ch.anim_world[t]
        else:
            k_mat = k_map[fi]
            f_cam = max(float(k_mat[0, 0]), 1.0)
            cx = float(k_mat[0, 2])
            cy = float(k_mat[1, 2])

            p0 = np.asarray(ch.anim_world[t, pelvis_j, :3, 3], dtype=np.float64)
            d0 = max(float(-p0[2]), 1.0)
            d_new = (f_cam / f_std) * d0
            x_new = p0[0] - (d_new / f_cam) * (cx - w * 0.5)
            y_new = p0[1] + (d_new / f_cam) * (cy - h * 0.5)
            z_new = -d_new
            p_cam = np.array([x_new, y_new, z_new], dtype=np.float64)

            r_delta = _rotation_between_vectors(p0, p_cam)
            t_cam = np.eye(4, dtype=np.float64)
            t_cam[:3, :3] = r_delta
            t_cam[:3, 3] = p_cam - (r_delta @ p0)
            m_world = c2w @ t_cam
            anim_new[t] = m_world[None, :, :] @ ch.anim_world[t]

    # Pass 1b: World-space ray-constrained trajectory & root orientation stabilization
    # (zero camera-shake lag, strong viewing-ray depth damping, <= 1.15 px lateral lock, and clean SO(3) pelvis stabilization!)
    sigma = smoothing_sigma(smooth_strength)
    if has_real_cam and sigma > 0.05 and len(valid_indices) >= 3:
        v_frames = [int(ch.frames[t]) for t in valid_indices]
        f_to_t = {int(ch.frames[t]): t for t in valid_indices}
        for run in segments(v_frames, max_gap=1):
            if len(run) < 3:
                continue
            seg_t = [f_to_t[f] for f in run]
            p_seg = anim_new[seg_t, pelvis_j, :3, 3].copy()
            r_seg = anim_new[seg_t, pelvis_j, :3, :3].copy()

            p_smooth_lat = gaussian_smooth(p_seg, 1.35 * sigma)
            p_smooth_ray = gaussian_smooth(p_seg, 2.25 * sigma)
            q_smooth = smooth_quats(matrix_to_quat(r_seg), 1.30 * sigma)
            r_smooth = quat_to_matrix(q_smooth)

            for k_s, t in enumerate(seg_t):
                fi = int(ch.frames[t])
                c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
                cam_o = c2w[:3, 3]
                f_cam = max(float(k_map[fi][0, 0]), 1.0)

                vec = p_seg[k_s] - cam_o
                d_cur = float(np.linalg.norm(vec))
                if d_cur < 20.0:
                    continue
                ray_u = vec / d_cur
                dw_ray = p_smooth_ray[k_s] - p_seg[k_s]
                dw_lat = p_smooth_lat[k_s] - p_seg[k_s]
                along = float(np.dot(dw_ray, ray_u))
                lat_vec = dw_lat - float(np.dot(dw_lat, ray_u)) * ray_u

                # Clamp along-ray depth smoothing to <= 6.5% (eliminates monocular Z breathing) and lateral to <= 1.15 px
                along_clamped = float(np.clip(along, -0.065 * d_cur, 0.065 * d_cur))
                lat_norm = float(np.linalg.norm(lat_vec))
                max_lat_cm = (1.15 / f_cam) * d_cur
                if lat_norm > max_lat_cm and lat_norm > 1e-8:
                    lat_vec = lat_vec * (max_lat_cm / lat_norm)

                p_final = p_seg[k_s] + along_clamped * ray_u + lat_vec

                # Stabilize world-space pelvis orientation on SO(3) (`<= 6.0 deg` despike cap preserves fast turns & 2D limbs)
                r_cur = r_seg[k_s]
                r_rel = r_smooth[k_s] @ r_cur.T
                ang_rad = math.acos(float(np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)))
                max_rad = math.radians(6.0)
                if ang_rad > max_rad and ang_rad > 1e-6:
                    q_rel = matrix_to_quat(r_rel[None])[0]
                    if q_rel[0] < 0:
                        q_rel = -q_rel
                    frac = max_rad / ang_rad
                    half = frac * ang_rad * 0.5
                    sin_orig = math.sqrt(max(1.0 - float(q_rel[0] ** 2), 1e-12))
                    q_clamped = np.array(
                        [math.cos(half), *(q_rel[1:] * (math.sin(half) / sin_orig))], dtype=np.float64
                    )
                    r_rel = quat_to_matrix(q_clamped[None])[0]

                m_delta = np.eye(4, dtype=np.float64)
                m_delta[:3, :3] = r_rel
                m_delta[:3, 3] = p_final - (r_rel @ p_seg[k_s])
                anim_new[t] = m_delta[None, :, :] @ anim_new[t]

    # Pass 2: Freeze hidden frames in 3D world space at their anchor valid frame (zero flying with camera!)
    for t, ok in enumerate(vis_mask):
        if ok:
            continue
        anc_t = None
        if ch.anchor_frames is not None and t < len(ch.anchor_frames):
            anc_f = int(ch.anchor_frames[t])
            if anc_f in f_to_char_idx and vis_mask[f_to_char_idx[anc_f]]:
                anc_t = f_to_char_idx[anc_f]
        if anc_t is None:
            anc_t = min(valid_indices, key=lambda vi: abs(int(ch.frames[vi]) - int(ch.frames[t])))
        anim_new[t] = anim_new[anc_t]

    return replace(ch, anim_world=anim_new)


def place_world_solved_character(
    ch: SkinnedCharacter,
    target_camera: CameraTrack,
    shot_frames: list[int] | tuple[int, ...] | None = None,
) -> SkinnedCharacter:
    """Place a character whose trajectory was solved over the whole shot (solver v2) into the world.

    The solve already ran against the exact per-frame camera and the scan, so the only thing left to do is to
    apply the exact `cam_to_world(f)` per frame.  Nothing is filtered or re-scaled here: any further change would
    break the camera-view match the solve produced.  Hidden frames stay frozen in WORLD space at their anchor
    frame so they never travel with the camera."""
    if not ch.frames:
        return ch
    all_frames = sorted(set(int(f) for f in (shot_frames or ch.frames)) | set(int(f) for f in ch.frames))
    cam_res = target_camera.resample_to(all_frames)
    f_to_cam = {int(f): i for i, f in enumerate(cam_res.frames)}
    f_to_ch = {int(f): i for i, f in enumerate(ch.frames)}
    vis = ch.visible_mask
    valid = [t for t, ok in enumerate(vis) if ok] or list(range(len(ch.frames)))
    anim = np.empty_like(ch.anim_world, dtype=np.float64)
    for t in valid:
        anim[t] = cam_res.cam_to_world[f_to_cam[int(ch.frames[t])]][None, :, :] @ ch.anim_world[t]
    for t, ok in enumerate(vis):
        if ok:
            continue
        anc_t = None
        if ch.anchor_frames is not None and t < len(ch.anchor_frames):
            anc_f = int(ch.anchor_frames[t])
            if anc_f in f_to_ch and vis[f_to_ch[anc_f]]:
                anc_t = f_to_ch[anc_f]
        if anc_t is None:
            anc_t = min(valid, key=lambda vi: abs(int(ch.frames[vi]) - int(ch.frames[t])))
        anim[t] = anim[anc_t]
    return replace(ch, anim_world=anim)


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal SO(3) rotation matrix mapping 3D vector `a` onto `b` (`det = +1`)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return np.eye(3, dtype=np.float64)
    ua = a / na
    ub = b / nb
    v = np.cross(ua, ub)
    c = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-degree flip around any orthogonal axis
        orth = np.array([1.0, 0.0, 0.0]) if abs(ua[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(ua, orth)
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3, dtype=np.float64)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _fit_cam_translation_to_mask_box(
    verts_rot: np.ndarray,
    p_init: np.ndarray,
    box_xyxy: np.ndarray,
    f_cam: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    update_depth: bool = True,
) -> np.ndarray:
    """Refine OpenGL camera-space translation `p = [tx, ty, tz]` (`tz < 0`) in 6 fast analytical steps
    so projected 3D vertices `verts_rot + p` match un-truncated edges of `box_xyxy` to sub-pixel accuracy."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if x2 - x1 < 4.0 or y2 - y1 < 4.0 or len(verts_rot) == 0:
        return p_init.copy()

    margin = max(6.0, 0.012 * min(w, h))
    in_l = x1 > margin
    in_t = y1 > margin
    in_r = x2 < (w - margin)
    in_b = y2 < (h - margin)

    p = p_init.astype(np.float64).copy()
    v_f64 = np.asarray(verts_rot, dtype=np.float64)
    max_vz = float(v_f64[:, 2].max()) if len(v_f64) else 0.0

    for _ in range(6):
        min_d = max(15.0, max_vz + 8.0)
        if -p[2] < min_d:
            p[2] = -min_d
        vc = v_f64 + p[None, :]
        inv_d = 1.0 / np.maximum(-vc[:, 2], 5.0)
        u = cx + f_cam * (vc[:, 0] * inv_d)
        v = cy - f_cam * (vc[:, 1] * inv_d)
        u_min, u_max = float(u.min()), float(u.max())
        v_min, v_max = float(v.min()), float(v.max())
        pw = max(1.0, u_max - u_min)
        ph = max(1.0, v_max - v_min)
        d_cur = -p[2]

        # Update depth ONLY when `update_depth=True` and full width or full height is un-clipped inside the image
        ratios = []
        if update_depth:
            if in_l and in_r:
                ratios.append(pw / max(4.0, x2 - x1))
            if in_t and in_b:
                ratios.append(ph / max(4.0, y2 - y1))
        if ratios:
            scale_ratio = float(np.clip(sum(ratios) / len(ratios), 0.7, 1.45))
            d_next = max(min_d, d_cur * scale_ratio)
            p[0] *= d_next / d_cur
            p[1] *= d_next / d_cur
            p[2] = -d_next
            vc = v_f64 + p[None, :]
            inv_d = 1.0 / np.maximum(-vc[:, 2], 5.0)
            u = cx + f_cam * (vc[:, 0] * inv_d)
            v = cy - f_cam * (vc[:, 1] * inv_d)
            u_min, u_max = float(u.min()), float(u.max())
            v_min, v_max = float(v.min()), float(v.max())
            d_cur = -p[2]

        # Shift X/Y to lock un-clipped edges
        du = 0.0
        if in_l and in_r:
            du = 0.5 * (x1 + x2) - 0.5 * (u_min + u_max)
        elif in_l:
            du = x1 - u_min
        elif in_r:
            du = x2 - u_max

        dv = 0.0
        if in_t and in_b:
            dv = 0.5 * (y1 + y2) - 0.5 * (v_min + v_max)
        elif in_t:
            dv = y1 - v_min
        elif in_b:
            dv = y2 - v_max

        p[0] += (du / f_cam) * d_cur
        p[1] -= (dv / f_cam) * d_cur
        if abs(du) < 0.05 and abs(dv) < 0.05 and (not ratios or abs(ratios[0] - 1.0) < 1e-3):
            break

    return p


def _try_triangulate_static_world_object(
    obj: RigidObjectTrack,
    mats_per_frame: np.ndarray,
    interior_indices: list[int],
    valid_indices: list[int],
    cam_res: CameraTrack,
    f_to_cam_idx: dict[int, int],
    k_map: dict[int, np.ndarray],
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Detect whether an object is stationary in 3D world space (or spinning in place on a fixed world pivot):
      - Supports BOTH wide-parallax moving cameras (`max_ang_deg >= 2.2°`, multi-view ray triangulation) AND
        low-parallax panning / tilting / tripod cameras (`max_ang_deg < 2.2°`, world pose consistency fit).
      - Subcase A (`is_pure_static = True`: chair, table, teapot, bottle): returns 100% constant `mats_static`
        across all frames so exported USD / FBX / ABC contain ZERO animation keyframes!
      - Subcase B (`is_pure_static = False`: turntable-spinning prop at fixed world pivot): keeps world position
        `p_static` 100% constant (`translate` static) while preserving smooth per-frame rotation."""
    pool_raw = interior_indices if len(interior_indices) >= 4 else valid_indices
    if obj.mask_boxes is None or len(pool_raw) < 4:
        return None

    raw_bw = np.array([float(obj.mask_boxes[t, 2] - obj.mask_boxes[t, 0]) for t in pool_raw])
    raw_bh = np.array([float(obj.mask_boxes[t, 3] - obj.mask_boxes[t, 1]) for t in pool_raw])
    med_bw, med_bh = float(np.median(raw_bw)), float(np.median(raw_bh))
    clean_int = [
        t
        for k, t in enumerate(pool_raw)
        if abs(raw_bw[k] - med_bw) < 0.38 * med_bw and abs(raw_bh[k] - med_bh) < 0.38 * med_bh
    ]
    if len(clean_int) < 4:
        return None

    margin = max(6.0, 0.012 * min(w, h))
    origins = []
    rays_w = []
    centers_2d = []
    boxes_2d = []
    diags_2d = []
    for t in clean_int:
        fi = int(obj.frames[t])
        c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
        w2c = np.linalg.inv(c2w)
        k_mat = k_map[fi]
        f_cam = max(float(k_mat[0, 0]), 1.0)
        cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
        bx = obj.mask_boxes[t]
        p_c_solved = w2c[:3, :3] @ (mats_per_frame[t, :3, 3] - c2w[:3, 3])
        if p_c_solved[2] < -15.0:
            uc = cx + f_cam * (p_c_solved[0] / (-p_c_solved[2]))
            vc = cy - f_cam * (p_c_solved[1] / (-p_c_solved[2]))
        else:
            uc, vc = 0.5 * float(bx[0] + bx[2]), 0.5 * float(bx[1] + bx[3])
        d_cam = np.array([(uc - cx) / f_cam, -(vc - cy) / f_cam, -1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        d_w = c2w[:3, :3] @ d_cam
        d_w /= np.linalg.norm(d_w)
        origins.append(c2w[:3, 3])
        rays_w.append(d_w)
        centers_2d.append((uc, vc, f_cam, cx, cy, c2w))
        boxes_2d.append(bx)
        diags_2d.append(math.hypot(float(bx[2] - bx[0]), float(bx[3] - bx[1])))

    r_arr = np.stack(rays_w, axis=0)
    dots = np.clip(r_arr @ r_arr.T, -1.0, 1.0)
    max_ang_deg = float(np.degrees(np.arccos(np.min(dots))))

    orig_arr = np.stack(origins, axis=0)
    b_dists = np.linalg.norm(orig_arr[:, None, :] - orig_arr[None, :, :], axis=-1)
    max_baseline = float(b_dists.max()) if b_dists.size else 0.0
    med_d_orig = float(
        np.median([np.linalg.norm(mats_per_frame[t, :3, 3] - origins[k]) for k, t in enumerate(clean_int)])
    )
    baseline_deg = math.degrees(2.0 * math.atan2(max_baseline * 0.5, max(med_d_orig, 20.0)))

    def _eval_cand_rms(pt: np.ndarray) -> float:
        sq = 0.0
        for uc, vc, f_cam, cx, cy, c2w in centers_2d:
            p_c = c2w[:3, :3].T @ (pt - c2w[:3, 3])
            if p_c[2] >= -20.0:
                return 1e9
            u_p = cx + f_cam * (p_c[0] / (-p_c[2]))
            v_p = cy - f_cam * (p_c[1] / (-p_c[2]))
            sq += (u_p - uc) ** 2 + (v_p - vc) ** 2
        return math.sqrt(sq / max(1, len(centers_2d)))

    mid_k = len(origins) // 2
    o_mid, d_mid = origins[mid_k], rays_w[mid_k]
    best_pol = np.median(mats_per_frame[clean_int, :3, 3], axis=0).copy()
    best_rms = _eval_cand_rms(best_pol)
    if max_baseline > 1.0:
        for s_try in np.geomspace(0.25, 12.0, 36):
            pt_try = o_mid + d_mid * (float(s_try) * max(med_d_orig, 25.0))
            r_try = _eval_cand_rms(pt_try)
            if r_try < best_rms:
                best_rms = r_try
                best_pol = pt_try

    p_pol = best_pol.copy()
    pol_ok = True
    for _ in range(6):
        step_w = np.zeros(3, dtype=np.float64)
        for uc, vc, f_cam, cx, cy, c2w in centers_2d:
            p_c = c2w[:3, :3].T @ (p_pol - c2w[:3, 3])
            if p_c[2] >= -15.0:
                pol_ok = False
                break
            u_p = cx + f_cam * (p_c[0] / (-p_c[2]))
            v_p = cy - f_cam * (p_c[1] / (-p_c[2]))
            step_w += c2w[:3, :3] @ np.array(
                [((uc - u_p) / f_cam) * (-p_c[2]), -((vc - v_p) / f_cam) * (-p_c[2]), 0.0],
                dtype=np.float64,
            )
        if not pol_ok:
            break
        p_pol = p_pol + step_w / max(1, len(centers_2d))

    p_tri = p_pol if pol_ok else None
    if baseline_deg >= 4.5 and max_ang_deg >= 3.5:
        eye3 = np.eye(3, dtype=np.float64)
        a_sum = np.zeros((3, 3), dtype=np.float64)
        b_sum = np.zeros(3, dtype=np.float64)
        for o_i, d_i in zip(origins, rays_w):
            proj = eye3 - np.outer(d_i, d_i)
            a_sum += proj
            b_sum += proj @ o_i
        try:
            p_ray = np.linalg.solve(a_sum, b_sum)
            d_ray_med = float(np.median([np.linalg.norm(p_ray - o_i) for o_i in origins]))
            r_ray = _eval_cand_rms(p_ray)
            r_tri_cur = _eval_cand_rms(p_tri) if p_tri is not None else 1e9
            if 0.35 * med_d_orig <= d_ray_med <= 2.85 * med_d_orig and r_ray <= r_tri_cur + 0.5:
                p_tri = p_ray
            elif r_ray <= r_tri_cur - 1.5 and d_ray_med > 2.85 * med_d_orig:
                # Moving object or large-baseline camera whose rays intersect farther than 2.85x monocular depth
                p_tri = p_ray
        except np.linalg.LinAlgError:
            pass
    if p_tri is None:
        return None

    px_errs = []
    scale_ratios = []
    size_dist_prods = []
    v_f64 = np.asarray(obj.rest_points, dtype=np.float64)
    for idx_k, (uc, vc, f_cam, cx, cy, c2w) in enumerate(centers_2d):
        p_c = np.linalg.inv(c2w)[:3, :3] @ (p_tri - c2w[:3, 3])
        if p_c[2] >= -20.0:
            return None
        u_p = cx + f_cam * (p_c[0] / (-p_c[2]))
        v_p = cy - f_cam * (p_c[1] / (-p_c[2]))
        px_errs.append(math.hypot(u_p - uc, v_p - vc))

        t_orig = clean_int[idx_k]
        p_orig_w = mats_per_frame[t_orig, :3, 3]
        d_orig = float(np.linalg.norm(p_orig_w - c2w[:3, 3]))
        d_tri = float(np.linalg.norm(p_tri - c2w[:3, 3]))
        if d_orig > 10.0:
            scale_ratios.append(d_tri / d_orig)
        size_dist_prods.append((diags_2d[idx_k] / f_cam) * d_tri)

    rms_px = float(np.sqrt(np.mean(np.square(px_errs))))
    max_px = float(np.max(px_errs))
    prod_arr = np.asarray(size_dist_prods, dtype=np.float64)
    cov_size = float(np.std(prod_arr) / max(1e-6, float(np.median(prod_arr))))
    med_diag = float(np.median(diags_2d))
    rms_limit = max(6.5, 0.032 * med_diag)
    if rms_px > rms_limit or max_px > max(13.5, 2.3 * rms_limit) or cov_size > 0.28 or not scale_ratios:
        return None

    alpha = float(np.median(scale_ratios))
    if alpha < 0.35 or alpha > 2.85:
        return None
    new_rest = (v_f64 * alpha).astype(np.float32)

    def _max_pairwise_rot_deg(rots: np.ndarray) -> float:
        # Pairwise trace(R_a @ R_b^T) across first/quarter/mid/three-quarter/last samples
        m = len(rots)
        sel = rots[np.unique(np.linspace(0, m - 1, min(m, 12), dtype=int))]
        tr = np.einsum("aij,bij->ab", sel, sel)
        return float(np.max(np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))))

    r_worlds = mats_per_frame[clean_int, :3, :3]
    r_ref = r_worlds[len(r_worlds) // 2]
    world_rot_spread_deg = _max_pairwise_rot_deg(r_worlds)

    c2w_rots = np.stack([c[5][:3, :3] for c in centers_2d], axis=0)
    cam_orbit_deg = _max_pairwise_rot_deg(c2w_rots)

    r_stds = obj.world_transforms[clean_int, :3, :3]
    cam_rot_spread_deg = _max_pairwise_rot_deg(r_stds)

    # Polish `p_tri` so `(r_ref, p_tri)` aligns cleanly on all un-truncated 2D bbox edges
    p_static = p_tri.copy()
    for _ in range(5):
        du_acc = np.zeros(3, dtype=np.float64)
        for idx_k, (uc, vc, f_cam, cx, cy, c2w) in enumerate(centers_2d):
            w2c = np.linalg.inv(c2w)
            r_c = w2c[:3, :3] @ r_ref
            p_c = w2c[:3, :3] @ (p_static - c2w[:3, 3])
            pts_c = (new_rest.astype(np.float64) @ r_c.T) + p_c[None, :]
            inv_d = 1.0 / np.maximum(-pts_c[:, 2], 5.0)
            u_arr = cx + f_cam * (pts_c[:, 0] * inv_d)
            v_arr = cy - f_cam * (pts_c[:, 1] * inv_d)
            bx = boxes_2d[idx_k]
            in_l, in_t = bx[0] > margin, bx[1] > margin
            in_r, in_b = bx[2] < (w - margin), bx[3] < (h - margin)
            err_u = (
                0.5 * (bx[0] + bx[2]) - 0.5 * (float(u_arr.min()) + float(u_arr.max()))
                if (in_l and in_r)
                else (bx[0] - float(u_arr.min()) if in_l else (bx[2] - float(u_arr.max()) if in_r else 0.0))
            )
            err_v = (
                0.5 * (bx[1] + bx[3]) - 0.5 * (float(v_arr.min()) + float(v_arr.max()))
                if (in_t and in_b)
                else (bx[1] - float(v_arr.min()) if in_t else (bx[3] - float(v_arr.max()) if in_b else 0.0))
            )
            d_cur = -p_c[2]
            delta_c = np.array([(err_u / f_cam) * d_cur, -(err_v / f_cam) * d_cur, 0.0], dtype=np.float64)
            du_acc += c2w[:3, :3] @ delta_c
        p_static = p_static + du_acc / max(1, len(centers_2d))

    # Measure static world pose 2D error across all clean frames on un-truncated edges
    static_cent_errs = []
    for idx_k, (uc, vc, f_cam, cx, cy, c2w) in enumerate(centers_2d):
        w2c = np.linalg.inv(c2w)
        r_c = w2c[:3, :3] @ r_ref
        p_c = w2c[:3, :3] @ (p_static - c2w[:3, 3])
        pts_c = (new_rest.astype(np.float64) @ r_c.T) + p_c[None, :]
        inv_d = 1.0 / np.maximum(-pts_c[:, 2], 5.0)
        u_arr = cx + f_cam * (pts_c[:, 0] * inv_d)
        v_arr = cy - f_cam * (pts_c[:, 1] * inv_d)
        bx = boxes_2d[idx_k]
        in_l, in_t = bx[0] > margin, bx[1] > margin
        in_r, in_b = bx[2] < (w - margin), bx[3] < (h - margin)
        cu = (
            0.5 * (float(u_arr.min()) + float(u_arr.max())) - 0.5 * (bx[0] + bx[2])
            if (in_l and in_r)
            else (float(u_arr.min()) - bx[0] if in_l else (float(u_arr.max()) - bx[2] if in_r else 0.0))
        )
        cv = (
            0.5 * (float(v_arr.min()) + float(v_arr.max())) - 0.5 * (bx[1] + bx[3])
            if (in_t and in_b)
            else (float(v_arr.min()) - bx[1] if in_t else (float(v_arr.max()) - bx[3] if in_b else 0.0))
        )
        static_cent_errs.append(math.hypot(cu, cv))
    static_rms = float(np.sqrt(np.mean(np.square(static_cent_errs))))

    net_world_spin_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(r_worlds[-1] @ r_worlds[0].T) - 1.0) * 0.5, -1.0, 1.0)))
    )
    is_spinning_turntable = (
        net_world_spin_deg > 24.0
        and world_rot_spread_deg > 25.0
        and cam_rot_spread_deg > max(24.0, 1.25 * cam_orbit_deg)
    )
    if not is_spinning_turntable and static_rms <= max(6.8, 0.032 * med_diag):
        # Purely fixed 3D scene object -> 100% constant world matrix (ZERO animation keyframes!)
        m_static = np.eye(4, dtype=np.float64)
        m_static[:3, :3] = r_ref
        m_static[:3, 3] = p_static
        return np.repeat(m_static[None], len(obj.frames), axis=0), new_rest, True

    # Fixed-pivot / turntable spinning prop: keep world translation `p_static` 100% constant while rotating smoothly!
    mats_anchored = mats_per_frame.copy()
    for t in valid_indices:
        r_w = mats_per_frame[t, :3, :3] if is_spinning_turntable else r_ref
        m_w = np.eye(4, dtype=np.float64)
        m_w[:3, :3] = r_w
        m_w[:3, 3] = p_static
        mats_anchored[t] = m_w
    return mats_anchored, new_rest, False


def _calibrate_dynamic_object_depth_with_characters(
    obj: RigidObjectTrack,
    p_cam_list: list[np.ndarray],
    r_cam_list: list[np.ndarray],
    c2w_list: list[np.ndarray],
    valid_indices: list[int],
    k_map: dict[int, np.ndarray],
    w: int,
    h: int,
    reference_characters: list[SkinnedCharacter],
) -> tuple[np.ndarray, np.ndarray] | None:
    """When a shot has solved `SAM 3D Body` characters (`reference_characters`) and a dynamic/low-parallax
    object comes into 2D hand/body contact (e.g. throwing, catching, holding, kicking), calibrate the object's
    monocular scale-depth ambiguity directly against the contacting 3D human joints so the object sits
    physically inside the actor's hands in 3D space while preserving sub-pixel 2D projection!"""
    if not reference_characters or obj.mask_boxes is None or len(valid_indices) == 0:
        return None

    # Pre-index visible character joints per frame
    char_frame_lookup: list[tuple[SkinnedCharacter, dict[int, int], list[int]]] = []
    for ch in reference_characters:
        vis_c = ch.visible_mask
        f_map = {int(f): idx for idx, (f, ok) in enumerate(zip(ch.frames, vis_c)) if ok}
        hand_j = [
            j
            for j, jn in enumerate(ch.joint_names)
            if any(tok in jn.lower() for tok in ("wrist", "hand", "thumb", "index", "middle", "palm"))
        ]
        if not hand_j:
            hand_j = list(range(len(ch.joint_names)))
        char_frame_lookup.append((ch, f_map, hand_j))

    contact_ratios: list[float] = []
    contact_z_targets: dict[int, float] = {}

    for k_v, t in enumerate(valid_indices):
        fi = int(obj.frames[t])
        bx = obj.mask_boxes[t]
        if bx[2] <= bx[0] + 4.0 or bx[3] <= bx[1] + 4.0:
            continue
        d_obj_cur = float(-p_cam_list[k_v][2])
        if d_obj_cur <= 10.0:
            continue

        c2w = c2w_list[k_v]
        w2c_r = c2w[:3, :3].T
        cam_o = c2w[:3, 3]
        k_mat = k_map[fi]
        f_cam = max(float(k_mat[0, 0]), 1.0)
        cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
        diag_px = math.hypot(float(bx[2] - bx[0]), float(bx[3] - bx[1]))
        pad_px = max(14.0, 0.55 * diag_px)

        best_dist = 1e9
        best_d_hand = None
        for ch, f_map, hand_j in char_frame_lookup:
            ch_t = f_map.get(fi)
            if ch_t is None:
                continue
            pts_w = ch.anim_world[ch_t, hand_j, :3, 3]  # [Jh, 3]
            pts_c = (pts_w - cam_o[None, :]) @ w2c_r.T
            valid_z = pts_c[:, 2] < -15.0
            if not np.any(valid_z):
                continue
            pts_c = pts_c[valid_z]
            d_j = -pts_c[:, 2]
            u_j = cx + f_cam * (pts_c[:, 0] / d_j)
            v_j = cy - f_cam * (pts_c[:, 1] / d_j)

            # Distance from 2D joint `(u_j, v_j)` to expanded object bbox `[x1, y1, x2, y2]`
            dx = np.maximum(0.0, np.maximum(bx[0] - u_j, u_j - bx[2]))
            dy = np.maximum(0.0, np.maximum(bx[1] - v_j, v_j - bx[3]))
            dist_box = np.hypot(dx, dy)
            min_i = int(np.argmin(dist_box))
            if float(dist_box[min_i]) <= pad_px and float(dist_box[min_i]) < best_dist:
                best_dist = float(dist_box[min_i])
                close_mask = dist_box <= (float(dist_box[min_i]) + 18.0)
                best_d_hand = float(np.median(d_j[close_mask]))

        if best_d_hand is not None and best_d_hand > 15.0:
            contact_ratios.append(best_d_hand / d_obj_cur)
            contact_z_targets[k_v] = -best_d_hand

    if len(contact_ratios) < 1:
        return None

    scale_factor = float(np.clip(np.median(contact_ratios), 0.25, 4.5))
    raw_ext = float(np.max(obj.rest_points.max(axis=0) - obj.rest_points.min(axis=0)))
    if raw_ext * scale_factor < 3.5 or raw_ext * scale_factor > 380.0:
        return None

    new_rest = (np.asarray(obj.rest_points, dtype=np.float64) * scale_factor).astype(np.float32)
    z_scaled = np.array([float(p_cam_list[k_v][2]) * scale_factor for k_v in range(len(valid_indices))], dtype=np.float64)

    # Pin hand-contact frames to exact 3D hand depth, and linearly interpolate free-flight ballistic depth
    # (`d^2 Z / dt^2 = 0` under gravity) across mid-air passes between two contact events!
    if contact_z_targets:
        c_keys = sorted(contact_z_targets.keys())
        c_zvals = [float(contact_z_targets[k]) for k in c_keys]
        c_diffs = [float(contact_z_targets[k] - z_scaled[k]) for k in c_keys]
        n_v = len(valid_indices)
        idx_arr = np.arange(n_v, dtype=np.float64)
        if len(c_keys) >= 2 and (c_keys[-1] - c_keys[0]) >= 3:
            # Inside `[c_keys[0] .. c_keys[-1]]`: exact contact depth at held frames + linear ballistic Z in free flight
            z_interp = np.interp(idx_arr, np.asarray(c_keys, dtype=np.float64), np.asarray(c_zvals, dtype=np.float64))
            k0, k1 = c_keys[0], c_keys[-1]
            z_scaled[k0 : k1 + 1] = z_interp[k0 : k1 + 1]
            if k0 > 0:
                z_scaled[:k0] += c_diffs[0]
            if k1 + 1 < n_v:
                z_scaled[k1 + 1 :] += c_diffs[-1]
        else:
            z_scaled += float(np.median(c_diffs))

    return new_rest, z_scaled


def align_standard_object_to_camera(
    obj: RigidObjectTrack,
    target_camera: CameraTrack,
    plate_width: int,
    plate_height: int,
    shot_frames: list[int] | tuple[int, ...] | None = None,
    reference_characters: list[SkinnedCharacter] | None = None,
) -> RigidObjectTrack:
    """Align a rigid 3D object solved in origin standard camera space (`I_4`, `f_src = hypot(W, H)`)
    into `target_camera`'s 3D world space via pure rigid `SE(3)` transforms (`scale = (1, 1, 1)`):
      1. Transforms all `visible == 1` frames into `target_camera` world space with viewing-ray rotation
         compensation + exact sub-pixel 2D mask bbox polish in `target_camera`'s lens.
      2. Automatically detects genuinely static 3D scene props under moving cameras via multi-view ray
         triangulation (`< 4 px` RMS), locking them stationary in world space while rescaling static `rest_points`.
      3. Freezes all `visible == 0` frames in **3D world space** at their anchor valid frame so hidden objects
         never swing or fly around with camera motion!
    """
    if not obj.frames:
        return obj
    w = max(1, int(plate_width or target_camera.width or 1920))
    h = max(1, int(plate_height or target_camera.height or 1080))
    f_arr = np.asarray(obj.source_focal_px, dtype=np.float64).reshape(-1)
    f_val = float(f_arr[0]) if f_arr.size > 0 else 0.0
    f_src = f_val if f_val > 1.0 else standard_camera_focal_px(w, h)

    all_frames = sorted(set(int(f) for f in (shot_frames or obj.frames)) | set(int(f) for f in obj.frames))
    cam_res = target_camera.resample_to(all_frames)
    f_to_cam_idx = {int(f): idx for idx, f in enumerate(cam_res.frames)}
    k_map = cam_res.intrinsics_per_frame(w, h)

    vis_mask = obj.visible_mask
    f_to_obj_idx = {int(f): idx for idx, f in enumerate(obj.frames)}
    mats_new = np.empty_like(obj.world_transforms, dtype=np.float64)

    valid_indices = [t for t, ok in enumerate(vis_mask) if ok]
    if not valid_indices:
        valid_indices = list(range(len(obj.frames)))

    has_boxes = obj.mask_boxes is not None and len(obj.mask_boxes) == len(obj.frames)
    margin = max(6.0, 0.012 * min(w, h))
    interior_indices: list[int] = []

    for t in valid_indices:
        fi = int(obj.frames[t])
        c_idx = f_to_cam_idx[fi]
        c2w = cam_res.cam_to_world[c_idx]
        k_mat = k_map[fi]
        f_cam = max(float(k_mat[0, 0]), 1.0)
        cx = float(k_mat[0, 2])
        cy = float(k_mat[1, 2])

        m_std = np.asarray(obj.world_transforms[t], dtype=np.float64)
        p0 = m_std[:3, 3]
        d0 = max(float(-p0[2]), 1.0)
        d_new = (f_cam / f_src) * d0
        x_new = p0[0] - (d_new / f_cam) * (cx - w * 0.5)
        y_new = p0[1] + (d_new / f_cam) * (cy - h * 0.5)
        z_new = -d_new
        p_cam = np.array([x_new, y_new, z_new], dtype=np.float64)

        # Compensate viewing-ray angle change when `f_cam != f_src` so the camera sees the identical object face!
        r_delta = _rotation_between_vectors(p0, p_cam)
        r_cam = r_delta @ m_std[:3, :3]

        if has_boxes and obj.mask_boxes is not None:
            bx = obj.mask_boxes[t]
            if bx[2] > bx[0] + 4.0 and bx[3] > bx[1] + 4.0:
                verts_rot = np.asarray(obj.rest_points, dtype=np.float64) @ r_cam.T
                p_cam = _fit_cam_translation_to_mask_box(verts_rot, p_cam, bx, f_cam, cx, cy, w, h)
                if bx[0] > margin and bx[1] > margin and bx[2] < w - margin and bx[3] < h - margin:
                    interior_indices.append(t)

        m_cam = np.eye(4, dtype=np.float64)
        m_cam[:3, :3] = r_cam
        m_cam[:3, 3] = p_cam
        mats_new[t] = c2w @ m_cam

    # Check if object sits at a stationary 3D world position (supports both moving and panning/tripod cameras!)
    new_rest = obj.rest_points
    new_vis = obj.visible
    tri_done = False
    is_pure_static = False
    if has_boxes and len(valid_indices) >= 4:
        tri_res = _try_triangulate_static_world_object(
            obj, mats_new, interior_indices, valid_indices, cam_res, f_to_cam_idx, k_map, w, h
        )
        if tri_res is not None:
            mats_new, new_rest, is_pure_static = tri_res
            tri_done = True
            if is_pure_static:
                # Stationary 3D scene object stays permanently placed in world space with ZERO visibility/TRS keys!
                new_vis = np.ones(len(obj.frames), dtype=np.int32)
    if not tri_done and has_boxes and len(valid_indices) >= 4:
        # Smooth camera-space depth `-z_cam` across contiguous visible frames to eliminate 1px bbox quantization chatter
        from .smoothing import gaussian_smooth

        z_cams = np.zeros(len(valid_indices), dtype=np.float64)
        c2w_list = []
        r_cam_list = []
        p_cam_list = []
        for k_v, t in enumerate(valid_indices):
            fi = int(obj.frames[t])
            c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
            w2c = np.linalg.inv(c2w)
            m_c = w2c @ mats_new[t]
            c2w_list.append(c2w)
            r_cam_list.append(m_c[:3, :3])
            p_cam_list.append(m_c[:3, 3])
            z_cams[k_v] = m_c[2, 3]

        if reference_characters:
            cal_res = _calibrate_dynamic_object_depth_with_characters(
                obj, p_cam_list, r_cam_list, c2w_list, valid_indices, k_map, w, h, reference_characters
            )
            if cal_res is not None:
                new_rest, z_cams = cal_res

        # Smooth contiguous chunks (frame gap <= 2)
        s0 = 0
        for k_v in range(1, len(valid_indices) + 1):
            if k_v == len(valid_indices) or int(obj.frames[valid_indices[k_v]]) - int(obj.frames[valid_indices[k_v - 1]]) > 2:
                if k_v - s0 >= 4:
                    z_cams[s0:k_v] = gaussian_smooth(z_cams[s0:k_v, None], 1.15)[:, 0]
                s0 = k_v

        rest_f64 = np.asarray(new_rest, dtype=np.float64)
        for k_v, t in enumerate(valid_indices):
            fi = int(obj.frames[t])
            k_mat = k_map[fi]
            f_cam = max(float(k_mat[0, 0]), 1.0)
            cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
            p_c = p_cam_list[k_v].copy()
            if p_c[2] < -5.0 and z_cams[k_v] < -5.0:
                ratio = z_cams[k_v] / p_c[2]
                p_c *= ratio
            bx = obj.mask_boxes[t]
            if bx[2] > bx[0] + 4.0 and bx[3] > bx[1] + 4.0:
                p_c = _fit_cam_translation_to_mask_box(
                    rest_f64 @ r_cam_list[k_v].T, p_c, bx, f_cam, cx, cy, w, h, update_depth=False
                )
            m_c = np.eye(4, dtype=np.float64)
            m_c[:3, :3] = r_cam_list[k_v]
            m_c[:3, 3] = p_c
            mats_new[t] = c2w_list[k_v] @ m_c

    if not is_pure_static:
        for t, ok in enumerate(vis_mask):
            if ok:
                continue
            anc_t = None
            if obj.anchor_frames is not None and t < len(obj.anchor_frames):
                anc_f = int(obj.anchor_frames[t])
                if anc_f in f_to_obj_idx and vis_mask[f_to_obj_idx[anc_f]]:
                    anc_t = f_to_obj_idx[anc_f]
            if anc_t is None:
                anc_t = min(valid_indices, key=lambda vi: abs(int(obj.frames[vi]) - int(obj.frames[t])))
            mats_new[t] = mats_new[anc_t]

    return replace(obj, world_transforms=mats_new, rest_points=new_rest, visible=new_vis)


def compute_camera_space_alignment(
    target_camera: CameraTrack,
    source_camera: CameraTrack | None = None,
    shot_frames: list[int] | tuple[int, ...] | None = None,
    fit: Literal["per_frame", "whole"] = "per_frame",
) -> AlignmentTransform:
    """Compute the exact camera-space conversion transform from `source_camera` (or solver origin camera)
    to `target_camera` across `shot_frames`."""
    frames = [int(f) for f in (shot_frames if shot_frames else target_camera.frames)]
    if not frames:
        frames = [0]

    dst = target_camera.resample_to(frames)
    dst_spread = camera_spread_cm(dst.cam_to_world)

    # Scenario A: Direct placement from SAM 3D Body origin camera space
    if source_camera is None:
        report = AlignmentReport(
            mode="direct",
            fit_requested="per_frame",
            frames=frames,
            common_authored_frames=len(dst.authored_frames),
            scale=1.0,
            target_spread_cm=round(dst_spread, 3),
        )
        return AlignmentTransform(tuple(frames), dst.cam_to_world.copy(), report)

    src_full = source_camera.resample_to(frames)

    # Identical camera fast-path
    if same_camera(src_full, dst):
        report = AlignmentReport(
            mode="same",
            fit_requested=fit,
            frames=frames,
            common_authored_frames=len(frames),
            scale=1.0,
            target_spread_cm=round(dst_spread, 3),
        )
        return AlignmentTransform(tuple(frames), np.eye(4, dtype=np.float64), report)

    diff = lens_diff_ratio(src_full, dst)
    diff_pct = diff * 100.0

    # Scenario B: Per-frame exact rigid fit
    if fit == "per_frame":
        if diff > LENS_TOLERANCE:
            raise LensMismatchError(diff_pct)
        mats = dst.cam_to_world @ np.linalg.inv(src_full.cam_to_world)
        report = AlignmentReport(
            mode="per_frame",
            fit_requested="per_frame",
            frames=frames,
            common_authored_frames=len(frames),
            scale=1.0,
            lens_diff_pct=round(diff_pct, 3),
            target_spread_cm=round(dst_spread, 3),
        )
        return AlignmentTransform(tuple(frames), mats, report)

    # Scenario C: Whole-shot smooth scaled similarity alignment (`scaled_align`)
    src_own = set(source_camera.authored_frames)
    dst_own = list(target_camera.authored_frames) or frames
    both = [int(f) for f in dst_own if not src_own or int(f) in src_own]
    if not both:
        both = frames

    fr_both = source_camera.resample_to(both)
    to_both = target_camera.resample_to(both)
    both_diff = lens_diff_ratio(fr_both, to_both)
    both_diff_pct = both_diff * 100.0

    warnings: list[str] = []
    if both_diff > LENS_TOLERANCE:
        warnings.append(
            f"来源与目标摄影机视场角相差 {both_diff_pct:.1f}%：整段平滑仅拟合运动轨迹，透视畸变差异可能导致边缘不完全贴合。"
        )

    to_spread = camera_spread_cm(to_both.cam_to_world)
    fr_spread = camera_spread_cm(fr_both.cam_to_world)
    moved = (to_spread >= MOVED_CM) and (fr_spread > 0.0)
    scale_locked = not moved

    fix, scale, _, turn_off_deg, left_cm, reversed_dir = scaled_align(
        fr_both.cam_to_world, to_both.cam_to_world, fit_scale=moved
    )

    if not moved:
        warnings.append(
            f"目标摄影机位移散布仅 {to_spread:.1f} cm（小于 {MOVED_CM:.0f} cm 阈值），已自动锁定缩放比例 s = 1.0 防止轨迹噪声缩坏人物；固定机位镜头建议使用「逐帧贴合」。"
        )
    if reversed_dir:
        warnings.append("检测到两台摄影机运动方向近似相反，已强制保持正尺度防止角色镜像翻转，请核对两台相机帧号顺序。")
    if left_cm > WARN_RESIDUAL_CM:
        warnings.append(
            f"整段对齐后光心位置残差达 {left_cm:.1f} cm（角度残差 {turn_off_deg:.2f}°），请核查来源相机与目标相机是否对应同一段实拍镜头。"
        )

    rot_only = fix[:3, :3] / max(scale, 1e-12)
    report = AlignmentReport(
        mode="whole",
        fit_requested="whole",
        frames=frames,
        common_authored_frames=len(both),
        scale=round(float(scale), 6),
        rotation_deg=round(float(rotation_deg(rot_only)), 3),
        translation_cm=round(float(np.linalg.norm(fix[:3, 3])), 3),
        rms_pos_error_cm=round(float(left_cm), 3),
        rms_ang_error_deg=round(float(turn_off_deg), 4),
        lens_diff_pct=round(float(both_diff_pct), 3),
        target_spread_cm=round(float(to_spread), 3),
        scale_locked_still=scale_locked,
        track_reversed=reversed_dir,
        warnings=warnings,
    )
    return AlignmentTransform(tuple(frames), fix, report)


# ------------------------------------------------------------------ Static Scene Scan Raycast Depth & Contact Calibration


def _rasterize_scan_depth_map_cm(
    verts_world_cm: np.ndarray,
    faces: np.ndarray,
    c2w: np.ndarray,
    k_mat: np.ndarray,
    plate_w: int,
    plate_h: int,
    max_grid_dim: int = 360,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fast vectorized barycentric Z-buffer + world upward-normal buffer of a world-space scene scan mesh.
    Returns `(z_buf_cm [Hg, Wg], abs_ny_buf [Hg, Wg], sx, sy)` (`inf` / `0` where empty)."""
    ds = max(1.0, float(max(plate_w, plate_h)) / float(max_grid_dim))
    wg = max(16, int(round(plate_w / ds)))
    hg = max(16, int(round(plate_h / ds)))
    sx = float(wg) / max(1.0, float(plate_w))
    sy = float(hg) / max(1.0, float(plate_h))

    z_buf = np.full((hg, wg), np.inf, dtype=np.float64)
    ny_buf = np.zeros((hg, wg), dtype=np.float32)
    if len(verts_world_cm) < 3 or len(faces) == 0:
        return z_buf, ny_buf, sx, sy

    v_w = np.asarray(verts_world_cm, dtype=np.float64)
    w2c_r = c2w[:3, :3].T
    cam_o = c2w[:3, 3]
    pts_c = (v_w - cam_o[None, :]) @ w2c_r.T

    tri_w = v_w[faces]  # [T, 3, 3]
    tri = pts_c[faces]  # [T, 3, 3]
    in_front = np.all(tri[..., 2] < -10.0, axis=1)
    if not np.any(in_front):
        return z_buf, ny_buf, sx, sy
    tp = tri[in_front]
    tw = tri_w[in_front]
    fn_raw = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
    fn_len = np.maximum(np.linalg.norm(fn_raw, axis=1), 1e-9)
    abs_ny_tri = np.abs(fn_raw[:, 1] / fn_len).astype(np.float32)

    fx = float(k_mat[0, 0]) * sx
    fy = float(k_mat[1, 1]) * sy
    cx = float(k_mat[0, 2]) * sx
    cy = float(k_mat[1, 2]) * sy

    d = -tp[..., 2]
    u = cx + fx * (tp[..., 0] / d)
    v = cy - fy * (tp[..., 1] / d)
    area2 = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (v[:, 1] - v[:, 0]) * (u[:, 2] - u[:, 0])
    non_deg = np.abs(area2) > 1e-4
    if not np.any(non_deg):
        return z_buf, ny_buf, sx, sy

    u, v, d, area2, abs_ny_tri = u[non_deg], v[non_deg], d[non_deg], area2[non_deg], abs_ny_tri[non_deg]
    umin = np.maximum(0, np.floor(u.min(axis=1)).astype(np.int32))
    umax = np.minimum(wg - 1, np.ceil(u.max(axis=1)).astype(np.int32))
    vmin = np.maximum(0, np.floor(v.min(axis=1)).astype(np.int32))
    vmax = np.minimum(hg - 1, np.ceil(v.max(axis=1)).astype(np.int32))

    valid_bb = (umax >= umin) & (vmax >= vmin) & ((umax - umin) <= wg) & ((vmax - vmin) <= hg)
    if not np.any(valid_bb):
        return z_buf, ny_buf, sx, sy
    u, v, d, area2, abs_ny_tri = u[valid_bb], v[valid_bb], d[valid_bb], area2[valid_bb], abs_ny_tri[valid_bb]
    umin, umax, vmin, vmax = umin[valid_bb], umax[valid_bb], vmin[valid_bb], vmax[valid_bb]

    bw = umax - umin + 1
    bh = vmax - vmin + 1
    counts = bw * bh
    order = np.arange(len(counts))
    chunk_start = 0
    z_flat = z_buf.ravel()
    ny_flat = ny_buf.ravel()
    while chunk_start < len(order):
        acc = 0
        chunk_end = chunk_start
        while chunk_end < len(order) and (acc + int(counts[chunk_end]) <= 2_500_000 or chunk_end == chunk_start):
            acc += int(counts[chunk_end])
            chunk_end += 1
        sl = slice(chunk_start, chunk_end)
        c_sl = counts[sl]
        if int(c_sl.sum()) > 0 and int(c_sl.sum()) <= 6_000_000:
            t_idx = np.repeat(np.arange(len(c_sl), dtype=np.int32), c_sl)
            starts = np.concatenate([[0], np.cumsum(c_sl)[:-1]])
            loc = np.arange(len(t_idx), dtype=np.int32) - starts[t_idx]
            bw_t = bw[sl][t_idx]
            px = umin[sl][t_idx] + (loc % bw_t)
            py = vmin[sl][t_idx] + (loc // bw_t)
            px_f = px.astype(np.float64) + 0.5
            py_f = py.astype(np.float64) + 0.5

            u0, u1, u2 = u[sl, 0][t_idx], u[sl, 1][t_idx], u[sl, 2][t_idx]
            v0, v1, v2 = v[sl, 0][t_idx], v[sl, 1][t_idx], v[sl, 2][t_idx]
            inv_a = 1.0 / area2[sl][t_idx]
            w0 = ((u1 - px_f) * (v2 - py_f) - (v1 - py_f) * (u2 - px_f)) * inv_a
            w1 = ((u2 - px_f) * (v0 - py_f) - (v2 - py_f) * (u0 - px_f)) * inv_a
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
            if np.any(inside):
                px_in = px[inside]
                py_in = py[inside]
                ti_in = t_idx[inside]
                inv_z = (
                    w0[inside] / d[sl, 0][ti_in]
                    + w1[inside] / d[sl, 1][ti_in]
                    + w2[inside] / d[sl, 2][ti_in]
                )
                z_val = 1.0 / np.maximum(inv_z, 1e-6)
                flat_idx = py_in * wg + px_in
                ny_val = abs_ny_tri[sl][ti_in]
                # Sort by descending depth so final overwrite keeps closest frontmost triangle's normal & depth!
                z_ord = np.argsort(-z_val)
                flat_s = flat_idx[z_ord]
                zv_s = z_val[z_ord]
                ny_s = ny_val[z_ord]
                np.minimum.at(z_flat, flat_s, zv_s)
                front_mask = zv_s <= z_flat[flat_s] + 1e-5
                ny_flat[flat_s[front_mask]] = ny_s[front_mask]
        chunk_start = chunk_end

    return z_buf, ny_buf, sx, sy


def calibrate_scene_with_scan(
    scan_verts_cm: np.ndarray,
    scan_faces: np.ndarray,
    cam_res: CameraTrack,
    plate_w: int,
    plate_h: int,
    shot_frames: list[int],
    raw_chars: list[SkinnedCharacter],
    aligned_chars: list[SkinnedCharacter],
    raw_objs: list[RigidObjectTrack],
    aligned_objs: list[RigidObjectTrack],
) -> tuple[list[SkinnedCharacter], list[RigidObjectTrack]]:
    """Calibrate solved `SAM 3D Body` characters and `SAM 3D Objects` props directly against an imported
    pixel-registered static scene scan (`scan_verts_cm`, `scan_faces`) in `cam_res` world space:
      1. Strictly guards against waist-up / no-ground shots and vertical walls: ONLY samples feet that project
         inside the un-cropped image frame AND hit an upward-facing walkable ground surface (`|n_y| >= 0.55`),
         accounting for exact shoe-sole thickness below foot/toe bones.
      2. Calibrates shot-wide true metric scale `s_char` (`0.00 px` change in 2D Camera-View silhouette!) PLUS
         tightly-bounded (`<= ±4.5%`) smoothed viewing-ray depth refinement so feet track the exact 3D scan ground
         without any artificial leg IK that would degrade 2D silhouette match accuracy.
      3. Raycasts stationary objects onto their exact scanned surface depth and enforces scan occlusion bounds."""
    from .smoothing import gaussian_smooth

    if len(scan_verts_cm) < 3 or len(scan_faces) == 0:
        return aligned_chars, aligned_objs

    f_to_cam_idx = {int(f): idx for idx, f in enumerate(cam_res.frames)}
    k_map = cam_res.intrinsics_per_frame(plate_w, plate_h)
    depth_cache: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}

    def _get_zmap(fi: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        if fi not in depth_cache:
            c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
            depth_cache[fi] = _rasterize_scan_depth_map_cm(scan_verts_cm, scan_faces, c2w, k_map[fi], plate_w, plate_h)
        return depth_cache[fi]

    # 1. Calibrate human characters via walkable-ground sole-to-scan raycast (preserving 100% 2D silhouette registration!)
    cal_chars: list[SkinnedCharacter] = []
    char_changed = False
    mx_px = max(8.0, 0.018 * plate_w)
    my_px = max(10.0, 0.022 * plate_h)

    for ch_raw, ch_al in zip(raw_chars, aligned_chars):
        vis = ch_al.visible_mask
        valid_t = [t for t, ok in enumerate(vis) if ok]
        foot_j = [
            j
            for j, jn in enumerate(ch_al.joint_names)
            if any(tok in jn.lower() for tok in ("foot", "toe", "ankle", "ball"))
        ]
        pelvis_j = next(
            (j for j, jn in enumerate(ch_al.joint_names) if jn.lower() in ("hips", "root", "pelvis")),
            0,
        )
        if len(valid_t) < 2 or not foot_j:
            cal_chars.append(ch_al)
            continue

        # Measure anatomical sole-bottom offset below lowest foot/toe bone in bind pose (typically ~3.2 cm)
        foot_j_set = set(foot_j)
        foot_v_mask = np.any(
            np.isin(ch_al.joint_indices, list(foot_j_set)) & (ch_al.joint_weights >= 0.25), axis=1
        )
        if np.any(foot_v_mask):
            min_bone_bind_y = float(np.min(ch_al.bind_world[foot_j, 1, 3]))
            min_vert_bind_y = float(np.percentile(ch_al.rest_points[foot_v_mask, 1], 3))
            sole_drop_cm = float(np.clip(min_bone_bind_y - min_vert_bind_y, 0.0, 9.0))
        else:
            sole_drop_cm = 3.2

        # Collect per-frame walkable ground ray hits across ALL visible frames where the supporting foot is inside the image
        per_t_hit: dict[int, tuple[float, float]] = {}  # t -> (d_scan_cm, d_sole_cm)
        for t in valid_t:
            fi = int(ch_al.frames[t])
            if fi not in f_to_cam_idx:
                continue
            c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
            w2c_r = c2w[:3, :3].T
            cam_o = c2w[:3, 3]
            k_mat = k_map[fi]
            f_cam = max(float(k_mat[0, 0]), 1.0)
            cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])

            # Pick lowest world-Y foot/toe joint at frame `t` and step down to actual shoe sole bottom `(0, -sole_drop_cm, 0)`
            j_low = foot_j[int(np.argmin(ch_al.anim_world[t, foot_j, 1, 3]))]
            pw_sole = ch_al.anim_world[t, j_low, :3, 3].copy()
            pw_sole[1] -= sole_drop_cm

            # Guard against mid-stride high foot lift when the other foot wasn't lowest: check pelvis-to-foot extension
            pw_pelvis = ch_al.anim_world[t, pelvis_j, :3, 3]
            leg_drop = float(pw_pelvis[1] - pw_sole[1])
            if leg_drop < 35.0:
                continue

            pc = w2c_r @ (pw_sole - cam_o)
            if pc[2] >= -20.0:
                continue
            d_sole = float(-pc[2])
            u_px = cx + f_cam * (pc[0] / d_sole)
            v_px = cy - f_cam * (pc[1] / d_sole)

            # Strict in-frame check: skips waist-up / medium / close-up shots where feet are outside the frame!
            if not (mx_px <= u_px <= plate_w - mx_px and my_px <= v_px <= plate_h - my_px):
                continue

            z_buf, ny_buf, sx, sy = _get_zmap(fi)
            hg, wg = z_buf.shape
            ug = int(np.clip(round(u_px * sx), 0, wg - 1))
            vg = int(np.clip(round(v_px * sy), 0, hg - 1))
            z_patch = z_buf[max(0, vg - 2) : min(hg, vg + 3), max(0, ug - 2) : min(wg, ug + 3)]
            ny_patch = ny_buf[max(0, vg - 2) : min(hg, vg + 3), max(0, ug - 2) : min(wg, ug + 3)]
            # Only accept walkable upward-facing scan surfaces (`|n_y| >= 0.55`), never vertical walls!
            walkable = np.isfinite(z_patch) & (ny_patch >= 0.55)
            if int(np.sum(walkable)) >= 2:
                d_scan = float(np.median(z_patch[walkable]))
                if 25.0 < d_scan < 15000.0:
                    per_t_hit[t] = (d_scan, d_sole)

        if len(per_t_hit) >= 2:
            # Filter out airborne jump frames where ground ray shoots >16% behind median character scale
            all_ratios = np.array([ ds / dc for (ds, dc) in per_t_hit.values() ], dtype=np.float64)
            med_r0 = float(np.median(all_ratios))
            inlier_ratios = all_ratios[np.abs(all_ratios - med_r0) <= 0.16 * med_r0]
            s_char = float(np.clip(np.median(inlier_ratios) if len(inlier_ratios) else med_r0, 0.35, 3.2))

            bind_s = ch_raw.bind_world.copy()
            bind_s[:, :3, 3] *= s_char
            anim_s = ch_raw.anim_world.copy()
            anim_s[:, :, :3, 3] *= s_char
            rest_s = (np.asarray(ch_raw.rest_points, dtype=np.float64) * s_char).astype(np.float32)
            ch_scaled_raw = replace(
                ch_raw,
                bind_world=bind_s,
                anim_world=anim_s,
                rest_points=rest_s,
                custom_data={**ch_raw.custom_data, "shotspace:scan_scale": round(s_char, 5)},
            )
            ch_cal = align_standard_character_to_camera(ch_scaled_raw, cam_res, plate_w, plate_h, shot_frames)

            # Smooth residual viewing-ray depth refinement (`[0.955 .. 1.045]` around `s_char`, strong sigma=3.5 eliminates foot-switch ripple!)
            hit_ts = sorted(t for t, (ds, dc) in per_t_hit.items() if abs((ds / (dc * s_char)) - 1.0) <= 0.12)
            if len(hit_ts) >= 2:
                from .smoothing import despike_hampel_1d

                hit_res = np.array([per_t_hit[t][0] / (per_t_hit[t][1] * s_char) for t in hit_ts], dtype=np.float64)
                if len(hit_res) >= 5:
                    hit_res = despike_hampel_1d(hit_res[:, None], threshold_mad=2.5)[:, 0]
                vt_arr = np.asarray(valid_t, dtype=np.float64)
                interp_res = np.interp(vt_arr, np.asarray(hit_ts, dtype=np.float64), hit_res)
                if len(interp_res) >= 3:
                    interp_res = gaussian_smooth(interp_res[:, None], 3.5)[:, 0]
                interp_res = np.clip(interp_res, 0.955, 1.045)

                anim_world_refined = ch_cal.anim_world.copy()
                for k_v, t in enumerate(valid_t):
                    alpha_r = float(interp_res[k_v])
                    if abs(alpha_r - 1.0) < 1e-4:
                        continue
                    fi = int(ch_cal.frames[t])
                    cam_o = cam_res.cam_to_world[f_to_cam_idx[fi], :3, 3]
                    p_pel = anim_world_refined[t, pelvis_j, :3, 3]
                    shift_w = (alpha_r - 1.0) * (p_pel - cam_o)
                    anim_world_refined[t, :, :3, 3] += shift_w[None, :]

                # Keep hidden frames frozen at their updated valid anchor pose
                f_to_cidx = {int(f): idx for idx, f in enumerate(ch_cal.frames)}
                for t, ok in enumerate(vis):
                    if ok:
                        continue
                    anc_t = None
                    if ch_cal.anchor_frames is not None and t < len(ch_cal.anchor_frames):
                        af = int(ch_cal.anchor_frames[t])
                        if af in f_to_cidx and vis[f_to_cidx[af]]:
                            anc_t = f_to_cidx[af]
                    if anc_t is None:
                        anc_t = min(valid_t, key=lambda vi: abs(int(ch_cal.frames[vi]) - int(ch_cal.frames[t])))
                    anim_world_refined[t] = anim_world_refined[anc_t]
                ch_cal = replace(ch_cal, anim_world=anim_world_refined)

            cal_chars.append(ch_cal)
            char_changed = True
            continue
        cal_chars.append(ch_al)

    # Re-align objects against scan-calibrated characters first if any character scale updated
    cur_objs = (
        [
            align_standard_object_to_camera(o, cam_res, plate_w, plate_h, shot_frames, reference_characters=cal_chars)
            for o in raw_objs
        ]
        if (char_changed and cal_chars)
        else list(aligned_objs)
    )

    # 2. Calibrate stationary objects to exact scanned surface depth & enforce scan occlusion bound on dynamic objects
    cal_objs: list[RigidObjectTrack] = []
    for raw_o, al_o in zip(raw_objs, cur_objs):
        vis = al_o.visible_mask
        valid_t = [t for t, ok in enumerate(vis) if ok]
        if not valid_t or raw_o.mask_boxes is None:
            cal_objs.append(al_o)
            continue

        pos_all = al_o.world_transforms[:, :3, 3]
        is_static = float(np.max(np.linalg.norm(pos_all - pos_all[0:1], axis=-1))) < 0.25
        sample_t = (
            [valid_t[i] for i in np.linspace(0, len(valid_t) - 1, min(6, len(valid_t)), dtype=int)]
            if is_static
            else valid_t
        )

        if is_static:
            surf_ratios: list[float] = []
            v_f64 = np.asarray(al_o.rest_points, dtype=np.float64)
            for t in sample_t:
                fi = int(al_o.frames[t])
                bx = raw_o.mask_boxes[t]
                if fi not in f_to_cam_idx or bx[2] <= bx[0] + 6.0 or bx[3] <= bx[1] + 6.0:
                    continue
                c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
                w2c = np.linalg.inv(c2w)
                m_c = w2c @ al_o.world_transforms[t]
                pts_c = (v_f64 @ m_c[:3, :3].T) + m_c[:3, 3][None, :]
                valid_vz = pts_c[:, 2] < -12.0
                if not np.any(valid_vz):
                    continue
                d_obj_front = float(np.percentile(-pts_c[valid_vz, 2], 25))

                z_buf, _ny_buf, sx, sy = _get_zmap(fi)
                hg, wg = z_buf.shape
                bw_px, bh_px = float(bx[2] - bx[0]), float(bx[3] - bx[1])
                cx_b, cy_b = 0.5 * float(bx[0] + bx[2]), 0.5 * float(bx[1] + bx[3])
                x1g = int(np.clip(round((cx_b - 0.25 * bw_px) * sx), 0, wg - 1))
                x2g = int(np.clip(round((cx_b + 0.25 * bw_px) * sx), x1g + 1, wg))
                y1g = int(np.clip(round((cy_b - 0.25 * bh_px) * sy), 0, hg - 1))
                y2g = int(np.clip(round((cy_b + 0.25 * bh_px) * sy), y1g + 1, hg))
                patch = z_buf[y1g:y2g, x1g:x2g]
                fin = patch[np.isfinite(patch)]
                if fin.size >= 4:
                    d_scan_front = float(np.percentile(fin, 30))
                    if 20.0 < d_scan_front < 15000.0:
                        surf_ratios.append(d_scan_front / d_obj_front)

            if len(surf_ratios) >= 2:
                alpha_s = float(np.clip(np.median(surf_ratios), 0.35, 3.2))
                if abs(alpha_s - 1.0) > 0.015:
                    mid_t = valid_t[len(valid_t) // 2]
                    fi_mid = int(al_o.frames[mid_t])
                    o_mid = cam_res.cam_to_world[f_to_cam_idx[fi_mid], :3, 3]
                    p_old = al_o.world_transforms[0, :3, 3]
                    p_new = o_mid + alpha_s * (p_old - o_mid)
                    mats_s = al_o.world_transforms.copy()
                    mats_s[:, :3, 3] = p_new[None, :]
                    rest_s = (v_f64 * alpha_s).astype(np.float32)
                    cal_objs.append(replace(al_o, world_transforms=mats_s, rest_points=rest_s))
                    continue
            cal_objs.append(al_o)
        else:
            # Dynamic object: enforce non-penetration in front of scanned static background walls/floors
            mats_d = al_o.world_transforms.copy()
            r_half = 0.35 * float(np.max(al_o.rest_points.max(axis=0) - al_o.rest_points.min(axis=0)))
            changed_d = False
            for t in valid_t:
                fi = int(al_o.frames[t])
                bx = raw_o.mask_boxes[t]
                if fi not in f_to_cam_idx or bx[2] <= bx[0] + 6.0 or bx[3] <= bx[1] + 6.0:
                    continue
                z_buf, _ny_buf, sx, sy = _get_zmap(fi)
                hg, wg = z_buf.shape
                ug = int(np.clip(round(0.5 * float(bx[0] + bx[2]) * sx), 0, wg - 1))
                vg = int(np.clip(round(0.5 * float(bx[1] + bx[3]) * sy), 0, hg - 1))
                patch = z_buf[max(0, vg - 2) : min(hg, vg + 3), max(0, ug - 2) : min(wg, ug + 3)]
                fin = patch[np.isfinite(patch)]
                if fin.size == 0:
                    continue
                d_bg = float(np.percentile(fin, 80))
                c2w = cam_res.cam_to_world[f_to_cam_idx[fi]]
                w2c = np.linalg.inv(c2w)
                m_c = w2c @ mats_d[t]
                d_cur = float(-m_c[2, 3])
                max_allowed = max(25.0, d_bg - r_half)
                if d_cur > max_allowed * 1.03:
                    m_c[:3, 3] *= max_allowed / d_cur
                    mats_d[t] = c2w @ m_c
                    changed_d = True
            cal_objs.append(replace(al_o, world_transforms=mats_d) if changed_d else al_o)

    return cal_chars, cal_objs
