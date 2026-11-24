"""Whole-shot world-space trajectory solve for one person under an exact camera.

The per-frame network estimate (SAM 3D Body) gives a body pose and a camera-space position for every frame
independently.  Depth along the viewing ray is the weakly observed quantity of a single image: a few percent of
depth error is a few pixels of apparent size in the camera view but tens of centimetres in the world.  This module
replaces "per-frame guess + filtering" by one least-squares solve over the whole shot:

  unknowns   per-frame body translation tau_t (camera space) and ONE body scale s for the whole shot
  evidence   2D keypoints of every frame, seen through the exact intrinsics K_t
  motion     the pelvis moves smoothly in WORLD space (exact extrinsics C_t), planted feet do not move in the world
  scene      planted feet rest on the supplied ground / scan surface

Every term is linear in (tau, s) once the perspective division is re-linearised, so each iteration is a sparse
banded linear system solved exactly (no step sizes, no clamps, no local minima of an iterative optimiser).
Matching the camera view and not sliding in the world are the same solution here, not a trade-off: the solve only
moves the body where the image does not constrain it.

Conventions
  camera space   OpenCV axes (x right, y down, z forward), metres
  world          ShotSpace convention: +Y up, right-handed; `c2w_cm` is the GL camera (looks down -Z) in cm
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bumped whenever the solve changes its results: cached solves of an older revision are solved again (from the
# cached network outputs) the next time the task is solved.
SOLVER_REVISION = 10
# The same for the part of the solve that only runs when a scan is loaded.  Kept apart so that a change to it does
# not send every person of a task WITHOUT a scan through the solve again for a result that cannot differ.
SCAN_SOLVER_REVISION = 1

CV_TO_GL = np.diag([1.0, -1.0, -1.0])

# MHR70 keypoint indices (see sam_3d_body/metadata/mhr70.py)
KP_L_HIP, KP_R_HIP = 9, 10
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ANKLE, KP_R_ANKLE = 13, 14
KP_L_FOOT = (15, 16, 17)  # left big toe, left small toe, left heel
KP_R_FOOT = (18, 19, 20)  # right big toe, right small toe, right heel
BODY_KP = tuple(range(0, 21)) + (41, 62)  # head, torso, legs, feet + right/left wrist


@dataclass
class GroundPlane:
    """Supporting plane `normal . p + offset = 0` in world metres (normal points up)."""

    normal: np.ndarray
    offset: float

    def support(self, pts_m: np.ndarray, max_gap_m: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.broadcast_to(self.normal, pts_m.shape).copy()
        d = np.full(len(pts_m), float(self.offset))
        return n, d, np.ones(len(pts_m), dtype=bool)


@dataclass
class SolveConfig:
    fps: float = 24.0
    sigma_px_rel: float = 0.012  # 2D keypoint noise as a fraction of the person's pixel height
    sigma_px_min: float = 1.0
    huber_k: float = 2.5  # robust threshold in sigmas
    sigma_acc_cm: float = 0.9  # pelvis world acceleration noise per frame^2 at 30 fps
    sigma_anchor_cm: float = 0.6  # planted-foot keypoint distance from its world anchor point
    sigma_ground_cm: float = 1.0  # planted-foot height above the supporting surface
    sigma_scale: float = 0.03  # how far the true body size may be from the network's metric size (1 sigma)
    iterations: int = 6
    contact_speed_cm: float = 1.6  # foot is planted below this world speed per frame at 30 fps
    standing_speed_cm: float = 0.8  # pelvis slower than this (per frame at 30 fps): an UNSEEN foot counts as planted
    same_spot_cm: float = 8.0  # a foot planted again within this distance of its previous spot is the same footprint
    same_spot_gap_s: float = 4.0  # ... if it comes back within this time
    contact_min_frames: int = 3
    contact_passes: int = 2
    use_contact: bool = True
    # How far away a body is, the picture says only through how large the body appears: one pixel of size is a few
    # millimetres of distance for a person 2 m away and several centimetres for a person 12 m away.  Fitted frame by
    # frame, a far body therefore slides along the line of sight by centimetres per frame while the camera view
    # shows nothing.  The body's motion along the line of sight is made stiffer the farther away it is: by
    # (distance / ray_stiff_from_m)^2, at most ray_stiff_max, and not at all nearer than ray_stiff_from_m (there the
    # picture does determine the distance, and a stiff distance would only lag behind real movement).  It acts
    # only in passes that hold the person's average distance, so it cannot pull the person towards the camera.
    ray_stiff_from_m: float = 4.0
    ray_stiff_max: float = 30.0
    # With a scan, planted feet are also recognised in the picture (`contacts_on_ground`): seconds of the shortest
    # window, and how well one point of the scanned ground must explain the foot's image positions.  The tolerance is
    # what was measured for truly planted feet: the network's foot keypoint stays within 1 to 2 px of the true foot
    # from frame to frame (90 %: 2.4 px on large people, 5.8 px on 250 px people), whatever the person's size, while
    # a foot that swings travels 10 px and more per frame.
    ground_contact_window_s: float = 0.2
    ground_contact_windows: tuple = (1.0, 2.0, 3.0)  # window lengths tried, as multiples of the shortest
    ground_contact_fit_px: float = 3.0
    ground_contact_fit_sigma: float = 0.25
    # ... and the camera's own travel across the window must shift that point in the picture by at least this many
    # times the tolerance.  Without travel a foot that moves along the line of sight looks as still as a planted
    # one, and the speed test, which works with a camera that does not travel, is left to decide alone.
    ground_contact_min_shift: float = 3.0
    # A body scale measured from camera parallax is only taken when it is precise and plausible.  Noise in the
    # camera positions, or a "planted" foot that is not really still, biases it towards a larger body, and the
    # scatter between stance phases (its standard error) does not show that: measured on 3DPW, whose camera poses
    # are only approximate, it reported 1.25 +- 0.05 for a person whose true scale was within 2 % of the network's.
    parallax_max_se: float = 0.02
    parallax_max_dev: float = 0.15
    # hold the person's average distance at the level the image evidence gives (see `solve_trajectory`);
    # switchable only so the effect can be measured
    hold_depth_level: bool = True


@dataclass
class SolveResult:
    cam_t_m: np.ndarray  # [T, 3] camera-space translation of the body origin
    scale: float
    contact: np.ndarray  # [T, 2] bool, left / right foot planted
    reproj_px: np.ndarray  # [T] robust mean 2D error of the weighted keypoints
    info: dict = field(default_factory=dict)
    anchor_weight: np.ndarray | None = None  # [T, 2] robust weight of each planted foot (low = evidence says it moved)


def _world_basis(c2w_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M_t maps an OpenCV camera-space vector to world axes; o_t is the camera centre in world metres."""
    c2w = np.asarray(c2w_cm, dtype=np.float64)
    return c2w[:, :3, :3] @ CV_TO_GL, c2w[:, :3, 3] * 0.01


def keypoints_world_m(kp_local_m: np.ndarray, cam_t_m: np.ndarray, scale: float, c2w_cm: np.ndarray) -> np.ndarray:
    m, o = _world_basis(c2w_cm)
    p_cam = scale * kp_local_m + cam_t_m[:, None, :]
    return np.einsum("tij,tkj->tki", m, p_cam) + o[:, None, :]


def drop_short_runs(ok: np.ndarray, frames: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Clear runs of usable frames shorter than `min_run` (a run ends at a frame gap or at an unusable frame)."""
    ok = np.array(ok, dtype=bool, copy=True)
    frames = np.asarray(frames).reshape(-1).astype(np.int64)
    n = len(ok)
    start = 0
    for i in range(1, n + 1):
        if i == n or frames[i] - frames[i - 1] > 1 or ok[i] != ok[i - 1]:
            if ok[start] and (i - start) < min_run:
                ok[start:i] = False
            start = i
    return ok


def trim_weak_ends(ok: np.ndarray, frames: np.ndarray, visible_share: np.ndarray, factor: float = 0.6,
                   max_frames: int = 8, at_picture_edge: np.ndarray | None = None, ref_percentile: float = 75.0) -> np.ndarray:
    """Take the weak first and last frames off every run of usable frames - a few frames, not more.

    A run begins where the person comes into the picture (or out from behind somebody) and ends where they leave.
    On its first and last frames only a sliver of the body shows; the network fills in the rest, and at the end of
    a run no later (or earlier) frame holds the position, so the solved person drifts away for a few frames.
    Frames in the MIDDLE of a run with just as little visible are held from both sides by the solve and stay.

    "Little" is relative to the person: the limit is `factor` times the share of the body's keypoints this person
    usually shows in the shot (the `ref_percentile` over the usable frames), so somebody framed from the waist up
    for the whole shot loses nothing.  A frame gap ends a run, like an unusable frame does.

    How much is cut depends on why the end is weak.  While the person is cut by the picture edge
    (`at_picture_edge`) they are entering or leaving, and every weak frame goes, however long that takes.
    Otherwise somebody stands in front of them: at most `max_frames` are cut, because a person who stays half
    hidden for seconds at the start of the shot is not a moment of appearing and keeps those frames.

    Measured against ground truth (Panoptic, 30 fps, 10 frames for hidden ends): the frames removed are 5 % of
    the solved frames; half of them lie more than 4 to 5 cm further from the truth than that person usually does,
    one in ten more than 12 to 25 cm."""
    ok = np.array(ok, dtype=bool, copy=True)
    if not ok.any():
        return ok
    share = np.asarray(visible_share, dtype=np.float64).reshape(-1)
    frames = np.asarray(frames).reshape(-1).astype(np.int64)
    limit = float(factor) * float(np.percentile(share[ok], ref_percentile))
    edge = np.zeros(len(ok), dtype=bool) if at_picture_edge is None else np.asarray(at_picture_edge, dtype=bool).reshape(-1)
    n = len(ok)
    start = 0
    for i in range(1, n + 1):
        if i == n or frames[i] - frames[i - 1] > 1 or ok[i] != ok[i - 1]:
            if ok[start]:
                a, b = start, i - 1
                cut = 0
                while a <= b and share[a] < limit and (edge[a] or cut < max_frames):
                    ok[a] = False
                    a += 1
                    cut += 1
                cut = 0
                while b > a and share[b] < limit and (edge[b] or cut < max_frames):
                    ok[b] = False
                    b -= 1
                    cut += 1
            start = i
    return ok


def depth_outliers(depth_m: np.ndarray, frames: np.ndarray, ok: np.ndarray, window: int = 30,
                   max_ratio: float = 2.5, min_jump_m: float = 3.0) -> np.ndarray:
    """Frames whose per-frame distance estimate cannot belong to the same person as their neighbours.

    The network sometimes puts a tiny or half-hidden person at a wildly different distance for a few frames
    (measured on MOTSynth: a 44 px person at 22 m reported at 130-410 m).  A frame is an outlier when its distance
    differs from the median of the usable frames within `window` frames by more than a factor `max_ratio` AND by
    more than `min_jump_m` metres - nobody covers that between neighbouring frames, while a person really walking
    or running towards the camera stays inside both limits."""
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    frames = np.asarray(frames).reshape(-1).astype(np.int64)
    ok = np.asarray(ok, dtype=bool).reshape(-1)
    out = np.zeros(len(z), dtype=bool)
    idx_ok = np.nonzero(ok)[0]
    if len(idx_ok) < 3:
        return out
    f_ok, z_ok = frames[idx_ok], z[idx_ok]
    for i in idx_ok:
        near = np.abs(f_ok - frames[i]) <= window
        if int(near.sum()) < 3:
            continue
        med = float(np.median(z_ok[near]))
        if med <= 0:
            continue
        ratio = max(z[i] / med, med / max(z[i], 1e-9))
        out[i] = ratio > max_ratio and abs(z[i] - med) > min_jump_m
    return out


def fit_translation_per_frame(
    kp_local_m: np.ndarray, kp2d: np.ndarray, weight: np.ndarray, k_mats: np.ndarray, init_t: np.ndarray
) -> np.ndarray:
    """Independent per-frame translation that best reprojects the (fixed-shape) posed body onto the 2D keypoints.
    Linear (bearing) form re-weighted by depth so residuals are in pixels; 3 refinement passes."""
    t_out = np.array(init_t, dtype=np.float64, copy=True)
    fx, fy = k_mats[:, 0, 0], k_mats[:, 1, 1]
    cx, cy = k_mats[:, 0, 2], k_mats[:, 1, 2]
    a = (kp2d[..., 0] - cx[:, None]) / fx[:, None]
    b = (kp2d[..., 1] - cy[:, None]) / fy[:, None]
    for _ in range(3):
        z_est = np.maximum(kp_local_m[..., 2] + t_out[:, None, 2], 0.05)
        wx = weight * fx[:, None] / z_est
        wy = weight * fy[:, None] / z_est
        for t in range(len(t_out)):
            sel = weight[t] > 0
            if int(sel.sum()) < 3:
                continue
            xs, ys, zs = kp_local_m[t, sel, 0], kp_local_m[t, sel, 1], kp_local_m[t, sel, 2]
            at, bt = a[t, sel], b[t, sel]
            rows_a = np.stack([np.ones_like(at), np.zeros_like(at), -at], axis=1) * wx[t, sel, None]
            rows_b = np.stack([np.zeros_like(bt), np.ones_like(bt), -bt], axis=1) * wy[t, sel, None]
            rhs_a = -(xs - at * zs) * wx[t, sel]
            rhs_b = -(ys - bt * zs) * wy[t, sel]
            sol, *_ = np.linalg.lstsq(np.concatenate([rows_a, rows_b]), np.concatenate([rhs_a, rhs_b]), rcond=None)
            if np.all(np.isfinite(sol)) and sol[2] > 0.05:
                t_out[t] = sol
    return t_out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def detect_contacts(
    foot_world_m: np.ndarray, frames: np.ndarray, fps: float, speed_cm: float, min_frames: int
) -> np.ndarray:
    """Planted-foot flags `[T, 2]` from world-space foot keypoint speed (median of toe/heel points per foot).
    The threshold is stated at 30 fps and scaled with the real frame interval."""
    t_n = foot_world_m.shape[0]
    contact = np.zeros((t_n, 2), dtype=bool)
    if t_n < 3:
        return contact
    dt = np.diff(frames).astype(np.float64)
    thr_m = speed_cm * 0.01 * (30.0 / max(fps, 1.0))
    for side in range(2):
        pts = foot_world_m[:, side]  # [T, 3 keypoints, 3]
        step = np.linalg.norm(np.diff(pts, axis=0), axis=-1)  # [T-1, 3]
        speed = np.median(step, axis=1)
        slow_step = (speed < thr_m) & (dt <= 1.5)  # step i links frame i and i+1; never link through a frame gap
        for lo, hi in _runs(slow_step):
            if (hi - lo + 1) >= min_frames:  # steps lo..hi-1 cover frames lo..hi
                contact[lo : hi + 1, side] = True
    return contact


def contacts_on_ground(
    foot_px: np.ndarray, foot_seen: np.ndarray, k_mats: np.ndarray, c2w_cm: np.ndarray, foot_world_m: np.ndarray,
    frames: np.ndarray, sigma_px: np.ndarray, ground, sole_drop_m: float, cfg: "SolveConfig",
) -> np.ndarray:
    """Planted-foot flags `[T, 2]` read from the picture, the exact camera and the scanned ground.

    The line of sight through a foot's image position meets the scan in one place: where that foot is if it stands
    on the ground.  For a planted foot this place is the same on every frame of the stance; for a foot that swings
    it runs away within a few frames.  A foot is called planted over a window of frames when ONE point of the scan
    explains its image positions on all of them to within the keypoint noise.

    Nothing here depends on how far away the network put the person, so a solve that drifts along the line of sight
    (a travelling camera turns every distance error into apparent foot motion, and `detect_contacts`, which looks
    at that motion, then misses most planted feet) cannot hide a planted foot from this test.

    `foot_px` [T, 2, 3, 2]: image positions of the three keypoints of each foot; `foot_seen` [T, 2, 3]: the camera
    sees the keypoint (a hidden foot's position is a guess and is not judged here); `foot_world_m` [T, 2, 3, 3]:
    where the unanchored solve put them (only used to find the right part of the scan); `sigma_px` [T]."""
    t_n = len(frames)
    contact = np.zeros((t_n, 2), dtype=bool)
    if ground is None or t_n < 3:
        return contact
    m_w, cam_o = _world_basis(c2w_cm)
    k_all = np.asarray(k_mats, dtype=np.float64)
    k_inv = np.linalg.inv(k_all)
    frames = np.asarray(frames).reshape(-1).astype(np.int64)
    link = np.diff(frames) == 1
    sigma = np.asarray(sigma_px, dtype=np.float64).reshape(-1)
    px = np.asarray(foot_px, dtype=np.float64)
    ok = np.asarray(foot_seen, dtype=bool) & np.all(np.isfinite(px), axis=-1)
    ones = np.ones(px.shape[:-1] + (1,))
    d_w = np.einsum("tij,tskj->tski", m_w, np.einsum("tij,tskj->tski", k_inv, np.concatenate([np.nan_to_num(px), ones], axis=-1)))
    # where each line of sight meets the scan (the surface is taken as flat around the point; three rounds settle
    # on the level the foot is on)
    x = np.asarray(foot_world_m, dtype=np.float64).copy()
    hit = ok.copy()
    for _ in range(3):
        nrm, off, ok_g = ground.support(np.nan_to_num(x).reshape(-1, 3), max_gap_m=1.5)
        nrm, off, ok_g = nrm.reshape(x.shape), off.reshape(x.shape[:-1]), ok_g.reshape(x.shape[:-1])
        den = np.einsum("tski,tski->tsk", nrm, d_w)
        num = sole_drop_m - off - np.einsum("tski,ti->tsk", nrm, cam_o)
        lam = num / np.where(np.abs(den) > 1e-9, den, np.nan)
        hit = ok & ok_g & np.isfinite(lam) & (lam > 0.05)
        x = np.where(hit[..., None], cam_o[:, None, None, :] + lam[..., None] * d_w, x)
    n0 = max(cfg.contact_min_frames + 1, int(round(cfg.ground_contact_window_s * cfg.fps)))
    spans = [(a, a + n) for n in sorted({int(round(n0 * m)) for m in cfg.ground_contact_windows}) for a in range(0, t_n - n + 1)]
    for a, b in spans:
        if not link[a : b - 1].all():
            continue
        sg = float(np.mean(sigma[a:b]))
        for side in range(2):
            if contact[a:b, side].all():
                continue
            passed = 0
            for kp in range(3):
                if not hit[a:b, side, kp].all():
                    continue
                spot = np.median(x[a:b, side, kp], axis=0)  # one point of the scan for the whole window
                rel = spot[None, :] - cam_o[a:b]
                p_cam = np.einsum("tji,tj->ti", m_w[a:b], rel)
                if not np.all(p_cam[:, 2] > 0.05):
                    continue
                tol = cfg.ground_contact_fit_px + cfg.ground_contact_fit_sigma * sg
                r0, r1 = rel[0] / np.linalg.norm(rel[0]), rel[-1] / np.linalg.norm(rel[-1])
                travel_px = float(np.arccos(np.clip(np.dot(r0, r1), -1.0, 1.0))) * float(k_all[a, 0, 0])
                if travel_px < cfg.ground_contact_min_shift * tol:
                    continue  # the camera did not travel enough to tell a planted foot from one moving along the ray
                uv = np.einsum("tij,tj->ti", k_all[a:b], p_cam / p_cam[:, 2:3])[:, :2]
                rms = float(np.sqrt(np.mean(np.sum((uv - px[a:b, side, kp]) ** 2, axis=1))))
                passed += rms <= tol
            if passed >= 2:  # two of the foot's three keypoints: the heel rises while the toes are still down
                contact[a:b, side] = True
    return contact


def unseen_feet_of_a_standing_person(
    kp_world_m: np.ndarray, frames: np.ndarray, kp_seen: np.ndarray, fps: float, speed_cm: float, half_win: int = 5
) -> np.ndarray:
    """Planted flags `[T, 2]` for feet the camera cannot see (behind another person, below the frame edge).

    Where a foot is hidden the network only guesses it, and the guess wanders from frame to frame.  When the
    person's pelvis is not travelling (standing, turning, gesturing) the hidden feet are taken to be planted, so
    they are held on one world point instead of following the guess.  A person who walks keeps free hidden feet."""
    t_n = len(frames)
    out = np.zeros((t_n, 2), dtype=bool)
    if t_n < 3:
        return out
    pel = 0.5 * (kp_world_m[:, KP_L_HIP] + kp_world_m[:, KP_R_HIP])
    speed = np.zeros(t_n)
    for t in range(t_n):
        a, b = max(0, t - half_win), min(t_n - 1, t + half_win)
        span = max(1, int(frames[b] - frames[a]))
        d = pel[b] - pel[a]
        speed[t] = float(np.hypot(d[0], d[2])) / span
    standing = speed < speed_cm * 0.01 * (30.0 / max(fps, 1.0))
    for side, kps in enumerate((KP_L_FOOT, KP_R_FOOT)):
        hidden = np.mean(kp_seen[:, list(kps)], axis=1) < 0.5
        out[:, side] = hidden & standing
    return out


def _solve_normal(rows, cols, vals, rhs, n, eq: tuple[np.ndarray, float] | None = None):
    """Solve the sparse least-squares problem `min ||A x - b||` given COO triplets (already weighted).

    `eq = (g, c)` adds one exact linear equality `g . x = c` (Lagrange multiplier, solved with the same
    factorisation, so a dense `g` costs nothing in sparsity)."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        a = sp.coo_matrix((vals, (rows, cols)), shape=(len(rhs), n)).tocsr()
        ata = (a.T @ a).tocsc()
        atb = a.T @ rhs
        ata = ata + sp.identity(n, format="csc") * 1e-9
        if eq is None:
            return spla.spsolve(ata, atb)
        lu = spla.splu(ata)
        x0, y = lu.solve(np.asarray(atb, dtype=np.float64)), lu.solve(np.asarray(eq[0], dtype=np.float64))
    except ImportError:
        a = np.zeros((len(rhs), n), dtype=np.float64)
        np.add.at(a, (rows, cols), vals)
        ata = a.T @ a + np.eye(n) * 1e-9
        if eq is None:
            return np.linalg.solve(ata, a.T @ rhs)
        both = np.linalg.solve(ata, np.stack([a.T @ rhs, np.asarray(eq[0], dtype=np.float64)], axis=1))
        x0, y = both[:, 0], both[:, 1]
    gy = float(np.dot(eq[0], y))
    if not np.isfinite(gy) or abs(gy) < 1e-300:
        return x0
    return x0 - y * ((float(np.dot(eq[0], x0)) - float(eq[1])) / gy)


class _Rows:
    def __init__(self) -> None:
        self.r: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.n = 0

    def add(self, cols: np.ndarray, vals: np.ndarray, rhs: np.ndarray) -> None:
        """`cols`, `vals`: [R, C] per-row column indices and coefficients; `rhs`: [R]."""
        r_n, c_n = cols.shape
        idx = self.n + np.arange(r_n)
        self.r.append(np.repeat(idx, c_n))
        self.c.append(cols.reshape(-1))
        self.v.append(vals.reshape(-1))
        self.b.append(rhs)
        self.n += r_n

    def system(self):
        return np.concatenate(self.r), np.concatenate(self.c), np.concatenate(self.v), np.concatenate(self.b)


def stance_runs(contact: np.ndarray, frames: np.ndarray) -> list[tuple[int, int, int]]:
    """Planted phases `(side, lo, hi)` (frames lo..hi-1), never spanning a frame gap."""
    out = []
    gap_after = np.diff(frames) > 1
    for side in range(2):
        for lo, hi in _runs(contact[:, side]):
            cut = lo
            for i in range(lo, hi - 1):
                if gap_after[i]:
                    if i + 1 - cut >= 2:
                        out.append((side, cut, i + 1))
                    cut = i + 1
            if hi - cut >= 2:
                out.append((side, cut, hi))
    return out


def footprint_groups(
    contact: np.ndarray, frames: np.ndarray, kp_world_m: np.ndarray | None, fps: float,
    same_spot_m: float = 0.08, max_gap_s: float = 4.0,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Group stance phases of one foot that land on the same spot: `(side, [(lo, hi), ...])`.

    Contact detection on a noisy estimate breaks one long stance into pieces (a shifted weight, a few doubtful
    frames).  If every piece got its own anchor point the foot could hop a few centimetres at each break, which
    adds up to visible creeping of a person who is simply standing.  Pieces whose foot keypoints sit within
    `same_spot_m` of the first piece of the group (and return within `max_gap_s`) share ONE anchor; a real step
    lands much farther away and starts a new footprint."""
    runs = stance_runs(contact, frames)
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    ref: dict[int, tuple[int, np.ndarray, int]] = {}  # side -> (group index, reference centre, last frame number)
    for side, lo, hi in runs:
        centre = None
        if kp_world_m is not None:
            centre = np.median(kp_world_m[lo:hi][:, list((KP_L_FOOT, KP_R_FOOT)[side])].mean(axis=1), axis=0)
        if centre is not None and side in ref:
            gi, c_ref, f_last = ref[side]
            if (frames[lo] - f_last) <= max_gap_s * fps and float(np.linalg.norm(centre - c_ref)) <= same_spot_m:
                groups[gi][1].append((lo, hi))
                ref[side] = (gi, c_ref, int(frames[hi - 1]))
                continue
        groups.append((side, [(lo, hi)]))
        if centre is not None:
            ref[side] = (len(groups) - 1, centre, int(frames[hi - 1]))
    return groups


def solve_trajectory(
    frames: np.ndarray,
    k_mats: np.ndarray,
    c2w_cm: np.ndarray,
    kp_local_m: np.ndarray,
    kp2d: np.ndarray,
    kp_weight: np.ndarray,
    init_cam_t_m: np.ndarray,
    person_height_px: np.ndarray,
    cfg: SolveConfig | None = None,
    ground=None,
    sole_drop_m: float = 0.0,
    contact: np.ndarray | None = None,
    scale_init: float = 1.0,
    refine_scale: bool = False,
    obs_index: np.ndarray | None = None,
    kp_seen: np.ndarray | None = None,
    anchor_ref_world_m: np.ndarray | None = None,
    depth_level_ref: np.ndarray | None = None,
    anchor_rows: bool = True,
) -> SolveResult:
    """Solve per-frame translations `tau_t`, one body scale `s`, and one world anchor point per planted-foot
    keypoint and stance phase.

    `depth_level_ref` [T] (metres): when given, the person's average distance to the camera is held at this
    reference (`mean(tau_z / ref) == 1`, exactly).  How far away a body of a given size is, is decided by how large
    it appears; penalties on world-space motion (planted feet, smoothness) all get cheaper the closer the body is
    put to the camera, so they must not be allowed to move that level - the same reason they may not choose the
    body scale.  Measured without it on Panoptic pizza1: people pulled 0.1-1.4 % towards the camera, most for the
    person whose feet are never in the picture.
    `anchor_rows=False` keeps the planted feet on the ground / scan (when there is one) but does not tie them to a
    fixed world point: the solve that tells how far away the scene itself puts the person.

    `kp2d` / `kp_weight` may hold several independent 2D measurements of the same body keypoint (e.g. a dedicated
    2D pose estimator and the 3D network's own projection): column j observes keypoint `obs_index[j]`.

    frames            [T] increasing frame numbers (gaps allowed)
    k_mats            [T, 3, 3] exact pixel intrinsics
    c2w_cm            [T, 4, 4] exact camera-to-world (GL camera, cm)
    kp_local_m        [T, 70, 3] posed fixed-shape keypoints relative to the body origin, camera axes, metres
    kp2d, kp_weight   [T, 70, 2] observed pixels and [T, 70] weights in [0, 1] (0 = unusable)
    init_cam_t_m      [T, 3] network translation (initial depth)
    person_height_px  [T] apparent pixel height (sets the 2D noise level)
    ground            optional object with `.support(points_m) -> (normal [N,3], offset [N], ok [N])`
    contact           [T, 2] planted-foot flags (left, right); a planted foot keypoint is ONE world point for the
                      whole stance phase (an unknown of the solve), which is what stops slow drift as well as jitter
    scale_init        body scale relative to the network's metric size.  Smoothness and contact terms always
                      prefer "less motion", i.e. a smaller body closer to the camera, so they must never be allowed
                      to choose the scale: it is held at `scale_init` unless `refine_scale` is set AND ground rows
                      exist (heights above a known surface are the only absolute evidence inside this solve).
    """
    cfg = cfg or SolveConfig()
    frames = np.asarray(frames, dtype=np.int64)
    t_n = len(frames)
    k_mats = np.asarray(k_mats, dtype=np.float64)
    kp_local_m = np.asarray(kp_local_m, dtype=np.float64)
    kp2d = np.asarray(kp2d, dtype=np.float64)
    w_obs = np.clip(np.asarray(kp_weight, dtype=np.float64), 0.0, 1.0)
    m_w, o_w = _world_basis(c2w_cm)
    fx, fy = k_mats[:, 0, 0], k_mats[:, 1, 1]
    cx, cy = k_mats[:, 0, 2], k_mats[:, 1, 2]
    a_obs = (kp2d[..., 0] - cx[:, None]) / fx[:, None]
    b_obs = (kp2d[..., 1] - cy[:, None]) / fy[:, None]
    sigma_px = np.maximum(cfg.sigma_px_min, cfg.sigma_px_rel * np.asarray(person_height_px, dtype=np.float64))

    fps_k = (30.0 / max(cfg.fps, 1.0)) ** 2
    sig_acc = cfg.sigma_acc_cm * 0.01 * fps_k
    sig_anc = cfg.sigma_anchor_cm * 0.01
    sig_gnd = cfg.sigma_ground_cm * 0.01

    scale = float(scale_init)
    obs_index = np.arange(kp2d.shape[1]) if obs_index is None else np.asarray(obs_index, dtype=np.int64)
    obs_local = kp_local_m[:, obs_index]  # [T, Ko, 3] the 3D point each 2D observation column refers to
    tau = fit_translation_per_frame(scale * obs_local, kp2d, w_obs, k_mats, np.asarray(init_cam_t_m, dtype=np.float64) * scale)
    s_col = 3 * t_n
    pelvis_local = 0.5 * (kp_local_m[:, KP_L_HIP] + kp_local_m[:, KP_R_HIP])
    chest_local = 0.5 * (kp_local_m[:, KP_L_SHOULDER] + kp_local_m[:, KP_R_SHOULDER])
    foot_idx = np.array([KP_L_FOOT, KP_R_FOOT])  # [2, 3]
    dt = np.diff(frames).astype(np.float64)
    contact = np.zeros((t_n, 2), dtype=bool) if contact is None else np.asarray(contact, dtype=bool)

    # anchor bookkeeping: one unknown world point per (stance phase, foot keypoint)
    groups = footprint_groups(contact, frames, anchor_ref_world_m, cfg.fps, cfg.same_spot_cm * 0.01, cfg.same_spot_gap_s)
    runs = [(side, lo, hi) for side, pieces in groups for lo, hi in pieces]
    anc_t, anc_kp, anc_id = [], [], []
    n_anchor = 0
    for side, pieces in groups:
        for kp in foot_idx[side]:
            for lo, hi in pieces:
                anc_t.append(np.arange(lo, hi))
                anc_kp.append(np.full(hi - lo, kp))
                anc_id.append(np.full(hi - lo, n_anchor))
            n_anchor += 1
    if n_anchor:
        anc_t, anc_kp, anc_id = np.concatenate(anc_t), np.concatenate(anc_kp), np.concatenate(anc_id)
    a_col0 = s_col + 1
    n_unk = a_col0 + 3 * n_anchor
    rob = np.ones_like(w_obs)
    rob_a = np.ones(len(anc_t) if n_anchor else 0)
    anchors = np.zeros((n_anchor, 3))
    info: dict = {}
    level_eq = None
    if depth_level_ref is not None:
        ref_z = np.asarray(depth_level_ref, dtype=np.float64).reshape(-1)
        ok_ref = np.isfinite(ref_z) & (ref_z > 0.05)
        if len(ref_z) == t_n and int(ok_ref.sum()) >= 1:
            g = np.zeros(n_unk)
            g[3 * np.nonzero(ok_ref)[0] + 2] = 1.0 / (ref_z[ok_ref] * float(ok_ref.sum()))
            level_eq = (g, 1.0)

    for _ in range(cfg.iterations):
        rows = _Rows()
        n_ground_rows = 0
        z_est = np.maximum(scale * obs_local[..., 2] + tau[:, None, 2], 0.05)

        # 1. reprojection of every usable keypoint (pixel units after depth re-weighting)
        tt, kk = np.nonzero(w_obs > 0)
        if len(tt):
            wt = np.sqrt(w_obs[tt, kk] * rob[tt, kk]) / sigma_px[tt]
            x, y, z = obs_local[tt, kk, 0], obs_local[tt, kk, 1], obs_local[tt, kk, 2]
            for obs, f_len, lat, lat_col in ((a_obs, fx, x, 0), (b_obs, fy, y, 1)):
                ob = obs[tt, kk]
                w_r = wt * f_len[tt] / z_est[tt, kk]
                cols = np.stack([3 * tt + lat_col, 3 * tt + 2, np.full_like(tt, s_col)], axis=1)
                vals = np.stack([w_r, -ob * w_r, (lat - ob * z) * w_r], axis=1)
                rows.add(cols, vals, np.zeros(len(tt)))

        # 2. smooth world-space motion of the pelvis and chest (non-uniform second difference across short gaps)
        if t_n >= 3:
            h1, h2 = dt[:-1], dt[1:]
            idx = np.nonzero((h1 <= 3.0) & (h2 <= 3.0))[0] + 1
            if len(idx):
                c0 = 2.0 / (h1[idx - 1] * (h1[idx - 1] + h2[idx - 1]))
                c2 = 2.0 / (h2[idx - 1] * (h1[idx - 1] + h2[idx - 1]))
                c1 = -(c0 + c2)
                w_a = 0.7071 / sig_acc
                for local in (pelvis_local, chest_local):
                    for ax in range(3):
                        cols = np.concatenate(
                            [3 * (idx - 1)[:, None] + np.arange(3), 3 * idx[:, None] + np.arange(3),
                             3 * (idx + 1)[:, None] + np.arange(3), np.full((len(idx), 1), s_col)], axis=1)
                        v0 = m_w[idx - 1, ax, :] * c0[:, None]
                        v1 = m_w[idx, ax, :] * c1[:, None]
                        v2 = m_w[idx + 1, ax, :] * c2[:, None]
                        vs = (c0 * np.einsum("tj,tj->t", m_w[idx - 1, ax, :], local[idx - 1])
                              + c1 * np.einsum("tj,tj->t", m_w[idx, ax, :], local[idx])
                              + c2 * np.einsum("tj,tj->t", m_w[idx + 1, ax, :], local[idx + 1]))
                        rhs = -(c0 * o_w[idx - 1, ax] + c1 * o_w[idx, ax] + c2 * o_w[idx + 1, ax])
                        rows.add(cols, np.concatenate([v0, v1, v2, vs[:, None]], axis=1) * w_a, rhs * w_a)
                    if cfg.ray_stiff_max > 1.0 and level_eq is not None:
                        # the same acceleration taken along the line of sight to the body, weighted up with distance
                        p_w = np.einsum("tij,tj->ti", m_w, scale * local + tau) + o_w
                        ray = p_w[idx] - o_w[idx]
                        far = np.linalg.norm(ray, axis=1)
                        ray /= np.maximum(far[:, None], 1e-9)
                        k_t = np.clip((far / max(cfg.ray_stiff_from_m, 1e-6)) ** 2, 1.0, cfg.ray_stiff_max)
                        w_x = w_a * np.sqrt(np.maximum(k_t**2 - 1.0, 0.0))
                        if np.any(w_x > 0):
                            cols = np.concatenate(
                                [3 * (idx - 1)[:, None] + np.arange(3), 3 * idx[:, None] + np.arange(3),
                                 3 * (idx + 1)[:, None] + np.arange(3), np.full((len(idx), 1), s_col)], axis=1)
                            v0 = np.einsum("ta,taj->tj", ray, m_w[idx - 1]) * c0[:, None]
                            v1 = np.einsum("ta,taj->tj", ray, m_w[idx]) * c1[:, None]
                            v2 = np.einsum("ta,taj->tj", ray, m_w[idx + 1]) * c2[:, None]
                            vs = (c0 * np.einsum("ta,taj,tj->t", ray, m_w[idx - 1], local[idx - 1])
                                  + c1 * np.einsum("ta,taj,tj->t", ray, m_w[idx], local[idx])
                                  + c2 * np.einsum("ta,taj,tj->t", ray, m_w[idx + 1], local[idx + 1]))
                            rhs = -np.einsum("ta,ta->t", ray, c0[:, None] * o_w[idx - 1] + c1[:, None] * o_w[idx] + c2[:, None] * o_w[idx + 1])
                            rows.add(cols, np.concatenate([v0, v1, v2, vs[:, None]], axis=1) * w_x[:, None], rhs * w_x)

        # 3. a planted foot keypoint is one world point for its whole stance phase
        if n_anchor:
            w_c = np.sqrt(rob_a) / sig_anc
            for ax in range(3 if anchor_rows else 0):
                cols = np.concatenate(
                    [3 * anc_t[:, None] + np.arange(3), np.full((len(anc_t), 1), s_col), (a_col0 + 3 * anc_id + ax)[:, None]], axis=1)
                vs = np.einsum("tj,tj->t", m_w[anc_t, ax, :], kp_local_m[anc_t, anc_kp])
                vals = np.concatenate([m_w[anc_t, ax, :], vs[:, None], -np.ones((len(anc_t), 1))], axis=1)
                rows.add(cols, vals * w_c[:, None], -o_w[anc_t, ax] * w_c)
            # 4. ... and rests on the ground / scan surface
            if ground is not None:
                q = np.einsum("tij,tj->ti", m_w[anc_t], scale * kp_local_m[anc_t, anc_kp] + tau[anc_t]) + o_w[anc_t]
                nrm, off, ok_g = ground.support(q)
                if ok_g.any():
                    gt_i, nrm, off = anc_t[ok_g], nrm[ok_g], off[ok_g]
                    nm = np.einsum("ti,tij->tj", nrm, m_w[gt_i])
                    cols = np.concatenate([3 * gt_i[:, None] + np.arange(3), np.full((len(gt_i), 1), s_col)], axis=1)
                    vs = np.einsum("tj,tj->t", nm, kp_local_m[gt_i, anc_kp[ok_g]]) - sole_drop_m
                    rhs = -(off + np.einsum("ti,ti->t", nrm, o_w[gt_i]))
                    w_g = np.sqrt(rob_a[ok_g]) / sig_gnd
                    if kp_seen is not None:
                        # an unseen foot only gets a weak "do not sink or float" hint - and no say at all in the
                        # solve that sets the person's distance level (its position is the network's guess)
                        w_g = w_g * np.where(kp_seen[gt_i, anc_kp[ok_g]], 1.0, 0.2 if anchor_rows else 0.0)
                    rows.add(cols, np.concatenate([nm, vs[:, None]], axis=1) * w_g[:, None], rhs * w_g)
                    n_ground_rows = int(ok_g.sum())

        # 5. body scale: free only when ground rows give absolute evidence, otherwise pinned
        free_scale = bool(refine_scale and ground is not None and n_ground_rows >= 6)
        w_s = (1.0 / cfg.sigma_scale) if free_scale else 1e7
        rows.add(np.array([[s_col]]), np.array([[w_s]]), np.array([w_s * float(scale_init)]))

        r_i, c_i, v_i, b_i = rows.system()
        sol = _solve_normal(r_i, c_i, v_i, b_i, n_unk, eq=level_eq)
        if not np.all(np.isfinite(sol)):
            break
        tau = sol[: 3 * t_n].reshape(t_n, 3)
        scale = float(sol[s_col])
        anchors = sol[a_col0:].reshape(n_anchor, 3)

        # robust re-weighting from true pixel residuals and from anchor residuals
        p_obs = scale * obs_local + tau[:, None, :]
        zc = np.maximum(p_obs[..., 2], 0.05)
        du = fx[:, None] * p_obs[..., 0] / zc + cx[:, None] - kp2d[..., 0]
        dv = fy[:, None] * p_obs[..., 1] / zc + cy[:, None] - kp2d[..., 1]
        err = np.hypot(du, dv) / sigma_px[:, None]
        rob = np.where(err <= cfg.huber_k, 1.0, cfg.huber_k / np.maximum(err, 1e-9))
        if n_anchor and anchor_rows:
            p_cam = scale * kp_local_m + tau[:, None, :]
            q = np.einsum("tij,tj->ti", m_w[anc_t], p_cam[anc_t, anc_kp]) + o_w[anc_t]
            e_a = np.linalg.norm(q - anchors[anc_id], axis=1) / sig_anc
            rob_a = np.where(e_a <= cfg.huber_k, 1.0, cfg.huber_k / np.maximum(e_a, 1e-9))

    p_obs = scale * obs_local + tau[:, None, :]
    zc = np.maximum(p_obs[..., 2], 0.05)
    du = fx[:, None] * p_obs[..., 0] / zc + cx[:, None] - kp2d[..., 0]
    dv = fy[:, None] * p_obs[..., 1] / zc + cy[:, None] - kp2d[..., 1]
    e_px = np.hypot(du, dv)
    w_sum = np.maximum(w_obs.sum(axis=1), 1e-9)
    reproj = (e_px * w_obs).sum(axis=1) / w_sum
    # frames whose planted-foot evidence disagrees with the anchor (lift-off, pivot): reported so the caller can trim
    anchor_weight = np.ones((t_n, 2))
    if n_anchor:
        side_of = np.isin(anc_kp, foot_idx[1]).astype(np.int64)
        np.minimum.at(anchor_weight, (anc_t, side_of), rob_a)
    info["sigma_px_median"] = float(np.median(sigma_px))
    info["contact_frames"] = [int(contact[:, 0].sum()), int(contact[:, 1].sum())]
    info["stance_phases"] = len(runs)
    info["footprints"] = len(groups)
    return SolveResult(cam_t_m=tau, scale=scale, contact=contact, reproj_px=reproj, info=info, anchor_weight=anchor_weight)


# ------------------------------------------------------------------ ground geometry


class GroundMesh:
    """Supporting-surface query on a triangle mesh (the imported static scene scan), world metres, +Y up.

    Triangles are binned on an XZ grid.  For a query point the supporting triangle is the walkable triangle
    (|normal_y| >= `min_up`) whose XZ footprint contains the point and whose surface height is closest to the point's
    current height (so multi-level sets, stairs and platforms resolve to the level the foot is actually on)."""

    def __init__(self, verts_m: np.ndarray, faces: np.ndarray, cell_m: float = 0.5, min_up: float = 0.5) -> None:
        v = np.asarray(verts_m, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(nrm, axis=1)
        keep = area2 > 1e-12
        tri, nrm, area2 = tri[keep], nrm[keep], area2[keep]
        nrm = nrm / area2[:, None]
        flip = nrm[:, 1] < 0
        nrm[flip] *= -1.0
        walk = nrm[:, 1] >= min_up
        self.tri = tri[walk]
        self.nrm = nrm[walk]
        self.off = -np.einsum("ij,ij->i", self.nrm, self.tri[:, 0])
        self.cell = float(cell_m)
        # cell -> triangles, stored as sorted arrays so that a full-resolution scan (millions of triangles) is
        # indexed in seconds and without per-triangle Python objects
        self._keys = np.zeros(0, dtype=np.int64)
        self._start = np.zeros(1, dtype=np.int64)
        self._tris = np.zeros(0, dtype=np.int64)
        self._origin = np.zeros(2, dtype=np.int64)
        self._width = 1
        self.big = np.zeros(0, dtype=np.int64)
        if len(self.tri):
            xz = self.tri[:, :, [0, 2]]
            lo = np.floor(xz.min(axis=1) / self.cell).astype(np.int64)
            hi = np.floor(xz.max(axis=1) / self.cell).astype(np.int64)
            n_xz = hi - lo + 1
            span = n_xz.prod(axis=1)
            # very large triangles (a single ground quad) go to a global list instead of thousands of cells
            big = span > 4096
            self.big = np.nonzero(big)[0]
            small = np.nonzero(~big)[0]
            if len(small):
                self._origin = lo[small].min(axis=0)
                self._width = int(hi[small, 1].max() - self._origin[1] + 1)
                rep = np.repeat(small, span[small])
                first = np.cumsum(span[small]) - span[small]
                local = np.arange(len(rep), dtype=np.int64) - np.repeat(first, span[small])
                nz = n_xz[rep, 1]
                keys = (lo[rep, 0] + local // nz - self._origin[0]) * self._width + (lo[rep, 1] + local % nz - self._origin[1])
                order = np.argsort(keys, kind="stable")
                keys = keys[order]
                self._tris = rep[order]
                self._keys, start = np.unique(keys, return_index=True)
                self._start = np.append(start, len(keys)).astype(np.int64)

    def _cell_triangles(self, cx: int, cz: int) -> np.ndarray | None:
        dz = cz - int(self._origin[1])
        if dz < 0 or dz >= self._width or not len(self._keys):
            return None
        key = (cx - int(self._origin[0])) * self._width + dz
        j = int(np.searchsorted(self._keys, key))
        if j >= len(self._keys) or int(self._keys[j]) != key:
            return None
        return self._tris[self._start[j]:self._start[j + 1]]

    def __len__(self) -> int:
        return len(self.tri)

    def support(self, pts_m: np.ndarray, max_gap_m: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 3)
        n_out = np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1))
        d_out = np.zeros(len(pts))
        ok = np.zeros(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            cand = self._cell_triangles(int(np.floor(p[0] / self.cell)), int(np.floor(p[2] / self.cell)))
            idx = self.big if cand is None else np.concatenate([cand, self.big])
            if len(idx) == 0:
                continue
            t = self.tri[idx]
            a, b, c = t[:, 0, [0, 2]], t[:, 1, [0, 2]], t[:, 2, [0, 2]]
            v0, v1, v2 = b - a, c - a, p[[0, 2]][None, :] - a
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            den = np.where(np.abs(den) < 1e-14, 1e-14, den)
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            w = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            inside = (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1.0 + 1e-6)
            if not np.any(inside):
                continue
            ii = idx[inside]
            h = -(self.off[ii] + self.nrm[ii, 0] * p[0] + self.nrm[ii, 2] * p[2]) / self.nrm[ii, 1]
            j = int(np.argmin(np.abs(h - p[1])))
            if abs(h[j] - p[1]) > max_gap_m:
                continue
            n_out[i], d_out[i], ok[i] = self.nrm[ii[j]], self.off[ii[j]], True
        return n_out, d_out, ok


# ------------------------------------------------------------------ body scale from the scene


def fuse_with_prior(estimate: float, se: float, prior_sigma: float) -> tuple[float, float]:
    """Precision-weighted fusion of a scale measurement with the prior `1 +- prior_sigma` (the network's own
    metric size).  Returns (fused value, weight of the measurement in [0, 1])."""
    w_d, w_p = 1.0 / max(se, 1e-6) ** 2, 1.0 / prior_sigma**2
    return float((w_d * estimate + w_p) / (w_d + w_p)), float(w_d / (w_d + w_p))


def scale_from_ground(
    kp_world_m: np.ndarray, contact: np.ndarray, cam_o_m: np.ndarray, ground, sole_drop_m: float,
    frames: np.ndarray | None = None, height_sigma_m: float = 0.015, kp_seen: np.ndarray | None = None,
) -> tuple[float, int, float]:
    """Scale correction that puts planted feet on the supporting surface.

    Scaling the body about the camera centre leaves the camera view untouched, so for every planted-foot keypoint
    the factor r that moves it along its viewing ray onto the surface is a measurement of
    (true size / assumed size).  Returns (median r, number of stance phases used, standard error).  The standard
    error combines the scatter between stance phases with the systematic uncertainty of where exactly the sole
    sits relative to the surface (`height_sigma_m`: shoe sole, scan noise), which matters for grazing rays."""
    frames = np.arange(len(contact)) if frames is None else np.asarray(frames)
    per_phase, sens = [], []
    for side, lo, hi in stance_runs(contact, frames):
        vals = []
        for kp in (KP_L_FOOT, KP_R_FOOT)[side]:
            q = kp_world_m[lo:hi, kp]
            nrm, off, ok = ground.support(q, max_gap_m=1.5)
            v = q - cam_o_m[lo:hi]
            den = np.einsum("ij,ij->i", nrm, v)
            num = sole_drop_m - off - np.einsum("ij,ij->i", nrm, cam_o_m[lo:hi])
            good = ok & (den < -0.05)
            if kp_seen is not None:
                good &= kp_seen[lo:hi, kp]  # a foot hidden behind someone is a guess, not a measurement
            r = num[good] / den[good]
            keep = (r > 0.5) & (r < 2.0)
            vals.append(r[keep])
            sens.append(height_sigma_m / np.abs(den[good][keep]))
        vals = np.concatenate(vals) if vals else np.zeros(0)
        if len(vals):
            per_phase.append(float(np.median(vals)))
    if not per_phase:
        return 1.0, 0, float("inf")
    r_all = np.asarray(per_phase)
    med = float(np.median(r_all))
    spread = float(1.4826 * np.median(np.abs(r_all - med))) if len(r_all) >= 3 else 0.04
    se_stat = spread / np.sqrt(len(r_all))
    se_sys = float(np.median(np.concatenate(sens))) if sens and sum(len(x) for x in sens) else 0.05
    return med, int(len(r_all)), float(np.hypot(se_stat, se_sys))


def scale_from_parallax(
    kp_world_m: np.ndarray, contact: np.ndarray, cam_o_m: np.ndarray, frames: np.ndarray,
    kp_seen: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Scale correction from planted feet under camera translation (no ground needed).

    A planted point Q satisfies Q = o_t + a * v_t (v_t: camera-to-point vector at the assumed scale, a: correction).
    Hence within one stance v_t = const - (1/a) o_t.  Regressing v on the EXACT camera positions o gives an estimate
    of 1/a whose regressor carries no noise; regressing the other way, or minimising foot motion, would be biased
    towards a smaller body.  Returns (a, standard error of a from the scatter BETWEEN stance phases, phases used);
    a static camera yields an infinite standard error."""
    slopes, weights = [], []
    for side, lo, hi in stance_runs(contact, frames):
        if hi - lo < 4:
            continue
        o = cam_o_m[lo:hi] - cam_o_m[lo:hi].mean(axis=0)
        sxx = float(np.sum(o * o)) * 3.0
        if sxx < 1e-8:
            continue
        if kp_seen is not None and np.mean(kp_seen[lo:hi][:, list((KP_L_FOOT, KP_R_FOOT)[side])]) < 0.7:
            continue
        sxy = 0.0
        for kp in (KP_L_FOOT, KP_R_FOOT)[side]:
            v = kp_world_m[lo:hi, kp] - cam_o_m[lo:hi]
            sxy += float(np.sum((v - v.mean(axis=0)) * o))
        slopes.append(-sxy / sxx)
        weights.append(sxx)
    if len(slopes) < 3:
        return 1.0, float("inf"), len(slopes)
    sl, wt = np.asarray(slopes), np.asarray(weights)
    inv_a = float(np.sum(wt * sl) / np.sum(wt))
    se_inv = float(np.sqrt(np.sum((wt * (sl - inv_a)) ** 2)) / np.sum(wt)) * np.sqrt(len(sl) / (len(sl) - 1.0))
    if inv_a <= 0.3:
        return 1.0, float("inf"), len(slopes)
    return float(1.0 / inv_a), float(se_inv / inv_a**2), len(slopes)


def solve_person(
    frames: np.ndarray,
    k_mats: np.ndarray,
    c2w_cm: np.ndarray,
    kp_local_m: np.ndarray,
    kp2d: np.ndarray,
    kp_weight: np.ndarray,
    init_cam_t_m: np.ndarray,
    person_height_px: np.ndarray,
    cfg: SolveConfig | None = None,
    ground=None,
    sole_drop_m: float = 0.0,
    obs_index: np.ndarray | None = None,
    kp_seen: np.ndarray | None = None,
) -> SolveResult:
    """Full schedule for one person: smooth solve -> planted feet -> body scale from the scene -> anchored solve.
    `kp_seen` [T, 70] marks keypoints that are actually visible (on the person's own mask, inside the picture):
    only those may measure the body scale against the scene."""
    cfg = cfg or SolveConfig()
    _, cam_o = _world_basis(c2w_cm)
    foot_idx = np.array([KP_L_FOOT, KP_R_FOOT])
    args = (frames, k_mats, c2w_cm, kp_local_m, kp2d, kp_weight, init_cam_t_m, person_height_px)

    # pass 0: no contact, network scale -- the only state from which the body scale may be measured, because
    # anchoring planted feet afterwards removes exactly the signal (their apparent motion) the measurement needs
    res = solve_trajectory(*args, cfg=cfg, obs_index=obs_index)
    scale, scale_src, scale_n, scale_w = 1.0, "network", 0, 0.0
    contact = np.zeros((len(frames), 2), dtype=bool)
    if cfg.use_contact:
        kp_w = keypoints_world_m(kp_local_m, res.cam_t_m, res.scale, c2w_cm)
        contact = detect_contacts(kp_w[:, foot_idx], frames, cfg.fps, cfg.contact_speed_cm, cfg.contact_min_frames)
        if ground is not None and kp_seen is not None:
            # the picture, the exact camera and the scan as a second witness (see `contacts_on_ground`)
            oi = np.arange(kp2d.shape[1]) if obs_index is None else np.asarray(obs_index, dtype=np.int64)
            foot_px = np.full((len(frames), 2, 3, 2), np.nan)
            foot_seen = np.zeros((len(frames), 2, 3), dtype=bool)
            for side, kps in enumerate((KP_L_FOOT, KP_R_FOOT)):
                for j, kp in enumerate(kps):
                    cols = np.flatnonzero(oi == kp)
                    if len(cols):
                        foot_px[:, side, j] = kp2d[:, cols[-1]]  # the network's own projection
                        foot_seen[:, side, j] = (kp_weight[:, cols[-1]] > 0) & kp_seen[:, kp]
            sigma_px = np.maximum(cfg.sigma_px_min, cfg.sigma_px_rel * np.asarray(person_height_px, dtype=np.float64))
            contact = contact | contacts_on_ground(
                foot_px, foot_seen, k_mats, c2w_cm, kp_w[:, foot_idx], frames, sigma_px, ground, sole_drop_m, cfg)
        hidden_planted = np.zeros_like(contact)
        if kp_seen is not None:
            hidden_planted = unseen_feet_of_a_standing_person(kp_w, frames, kp_seen, cfg.fps, cfg.standing_speed_cm)
            contact = contact | hidden_planted
        if contact.any():
            if ground is not None:
                r, n_r, se = scale_from_ground(kp_w, contact, cam_o, ground, sole_drop_m, frames, kp_seen=kp_seen)
                src = "ground"
            else:
                r, se, n_r = scale_from_parallax(kp_w, contact, cam_o, frames, kp_seen=kp_seen)
                src = "parallax"
                if not (np.isfinite(se) and se <= cfg.parallax_max_se and abs(r - 1.0) <= cfg.parallax_max_dev):
                    n_r, se = 0, float("inf")
            if n_r > 0 and np.isfinite(se):
                scale, w_meas = fuse_with_prior(r, se, cfg.sigma_scale)
                scale_src, scale_n, scale_w = f"{src} ({w_meas:.0%} weight)", n_r, float(w_meas)
            scale = float(np.clip(scale, 0.6, 1.6))
            # The level the planted feet may not move: the distance of the un-anchored solution carried to the
            # chosen scale (size and distance scale together) - or, with a ground / scan, of the solve in which
            # the planted feet rest on it but are not yet tied to fixed points.  Feet on the ground are evidence
            # about distance; "the feet move less if the body is nearer" is not.
            level_ref = scale * np.asarray(res.cam_t_m, dtype=np.float64)[:, 2] / max(float(res.scale), 1e-9)
            if ground is not None and cfg.hold_depth_level:
                res_g = solve_trajectory(
                    *args, cfg=cfg, ground=ground, sole_drop_m=sole_drop_m, contact=contact,
                    scale_init=scale, refine_scale=False, obs_index=obs_index, kp_seen=kp_seen,
                    anchor_ref_world_m=kp_w, anchor_rows=False,
                )
                if np.all(np.isfinite(res_g.cam_t_m)):
                    level_ref = np.asarray(res_g.cam_t_m, dtype=np.float64)[:, 2]
            for pass_i in range(max(1, cfg.contact_passes)):
                res = solve_trajectory(
                    *args, cfg=cfg, ground=ground, sole_drop_m=sole_drop_m, contact=contact,
                    scale_init=scale, refine_scale=False, obs_index=obs_index, kp_seen=kp_seen,
                    anchor_ref_world_m=kp_w, depth_level_ref=level_ref if cfg.hold_depth_level else None,
                )
                if pass_i + 1 >= max(1, cfg.contact_passes):
                    break
                kp_w = keypoints_world_m(kp_local_m, res.cam_t_m, res.scale, c2w_cm)
                seen = detect_contacts(kp_w[:, foot_idx], frames, cfg.fps, cfg.contact_speed_cm, cfg.contact_min_frames)
                # keep phases the evidence supported; a planted foot that the image pulls away has lifted or pivoted
                # (hidden feet have no such evidence, they stay planted while the person stands)
                new_contact = ((seen | contact) & (res.anchor_weight >= 0.5)) | hidden_planted
                if np.array_equal(new_contact, contact):
                    break
                contact = new_contact
        elif scale != 1.0:
            res = solve_trajectory(*args, cfg=cfg, scale_init=scale, obs_index=obs_index)
    if cfg.ray_stiff_max > 1.0 and not contact.any():
        # no pass above held the person's average distance (nothing was anchored): one that does, so that the
        # distance of a far body is steadied here too
        res = solve_trajectory(
            *args, cfg=cfg, scale_init=float(res.scale), obs_index=obs_index, kp_seen=kp_seen,
            depth_level_ref=np.asarray(res.cam_t_m, dtype=np.float64)[:, 2],
        )
    res.contact = contact
    res.info.update({"scale_source": scale_src, "scale_samples": int(scale_n), "scale_weight": round(scale_w, 4)})
    return res
