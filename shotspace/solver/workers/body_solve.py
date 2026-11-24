"""Worker-side glue between SAM 3D Body's per-frame outputs and the whole-shot trajectory solve
(`shotspace.solver.trajectory`).  Runs inside the SAM 3D Body worker environment (torch + the MHR head)."""

from __future__ import annotations

import numpy as np
import torch

from shotspace.core.smoothing import gaussian_smooth, segments, smooth_angles, smooth_quats
from shotspace.solver.trajectory import (
    BODY_KP,
    KP_L_FOOT,
    KP_R_FOOT,
    SolveConfig,
    SolveResult,
    footprint_groups,
    keypoints_world_m,
    solve_person,
)

CHUNK = 64
CV_FLIP = np.array([1.0, -1.0, -1.0])


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def mhr_keypoints_local_m(head, global_rot, body, hand, scale, shape, n_kp: int = 70) -> np.ndarray:
    """Posed keypoints `[N, n_kp, 3]` relative to the body origin (`global_trans = 0`), OpenCV camera axes, metres."""
    device = head.scale_mean.device
    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    out = []
    for lo in range(0, len(global_rot), CHUNK):
        sl = slice(lo, lo + CHUNK)
        n = len(global_rot[sl])
        _, kp = head.mhr_forward(
            global_trans=torch.zeros((n, 3), device=device),
            global_rot=to_t(global_rot[sl]),
            body_pose_params=to_t(body[sl]),
            hand_pose_params=to_t(hand[sl]),
            scale_params=to_t(scale[sl]),
            shape_params=to_t(shape[sl]),
            expr_params=torch.zeros((n, head.num_face_comps), device=device),
            return_keypoints=True,
        )
        out.append(_np(kp[:, :n_kp]).astype(np.float64))
    return np.concatenate(out) * CV_FLIP[None, None, :]


# Weight of a keypoint the network placed outside the picture (a guess, not something it saw)
W_OUTSIDE_PICTURE = 0.15


def build_observations(
    kp2d_net: np.ndarray,
    image_size: tuple[int, int] | None,
    kp2d_pose: np.ndarray | None = None,
    pose_score: np.ndarray | None = None,
    kp_visible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D evidence for the solve: `(kp2d_obs [N, Ko, 2], weight [N, Ko], obs_index [Ko])`.

    Only body, feet and wrists carry the solve (fingers and face detail are too noisy to determine depth).
      * the 3D network's own projected keypoints: always available, full weight when they are the only source;
      * a dedicated 2D pose estimator (when supplied): more accurate on joints it can see, so it gets the full
        weight scaled by its confidence, and the network's keypoints stay as a weak second opinion that keeps
        occluded joints where the full-body prior put them.
    Keypoints the network had to extrapolate outside the picture, or (when known) behind an occluder, are
    down-weighted."""
    n = kp2d_net.shape[0]
    body = np.asarray(BODY_KP, dtype=np.int64)

    def base_weight(k2: np.ndarray) -> np.ndarray:
        w = np.ones((n, len(body)), dtype=np.float64)
        if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
            wid, hei = float(image_size[0]), float(image_size[1])
            out = (k2[..., 0] < 0) | (k2[..., 0] > wid - 1) | (k2[..., 1] < 0) | (k2[..., 1] > hei - 1)
            w[out] *= W_OUTSIDE_PICTURE
        return w

    net2 = np.asarray(kp2d_net, dtype=np.float64)[:, body]
    w_net = base_weight(net2)
    if kp_visible is not None:
        w_net[~np.asarray(kp_visible, dtype=bool)[:, body]] *= 0.35
    if kp2d_pose is None:
        return net2, w_net, body
    pose2 = np.asarray(kp2d_pose, dtype=np.float64)[:, body]
    conf = np.ones((n, len(body))) if pose_score is None else np.asarray(pose_score, dtype=np.float64)[:, body]
    w_pose = base_weight(pose2) * np.clip((conf - 0.5) / 0.4, 0.0, 1.0) ** 2
    w_pose[~np.all(np.isfinite(pose2), axis=-1)] = 0.0
    pose2 = np.nan_to_num(pose2)
    if kp_visible is not None:
        w_pose[~np.asarray(kp_visible, dtype=bool)[:, body]] *= 0.2
    return np.concatenate([pose2, net2], axis=1), np.concatenate([w_pose, 0.25 * w_net], axis=1), np.concatenate([body, body])


def _smooth_world_global_rot(euler: np.ndarray, c2w_rot: np.ndarray, sigma: float) -> np.ndarray:
    import roma

    r_cam = roma.euler_to_rotmat("ZYX", torch.as_tensor(euler, dtype=torch.float64)).numpy()
    r_world = c2w_rot @ r_cam
    q = smooth_quats(roma.rotmat_to_unitquat(torch.as_tensor(r_world)).numpy(), sigma)
    r_s = roma.unitquat_to_rotmat(torch.as_tensor(q)).numpy()
    return roma.rotmat_to_euler("ZYX", torch.as_tensor(np.swapaxes(c2w_rot, -1, -2) @ r_s)).numpy()


def solve_body_shot(
    raw: dict,
    head,
    fps: float,
    ground=None,
    pose_sigma: float = 1.0,
    cfg: SolveConfig | None = None,
    use_frames: np.ndarray | None = None,
    kp2d_pose: np.ndarray | None = None,
    pose_score: np.ndarray | None = None,
    kp_visible: np.ndarray | None = None,
) -> dict:
    """Whole-shot solve for one tracked person from cached raw network outputs.

    Returns the per-frame parameters that define the final animation (camera space, the body is placed into the
    world by the exact camera afterwards): fixed shape/scale parameters, smoothed pose, solved translation and
    the uniform body scale."""
    frames_all = np.asarray(raw["frames"]).reshape(-1).astype(np.int64)
    keep = np.ones(len(frames_all), dtype=bool) if use_frames is None else np.asarray(use_frames, dtype=bool)
    idx = np.nonzero(keep)[0]
    frames = frames_all[idx]
    shape = np.asarray(raw["shape_params"], dtype=np.float64)[idx]
    scale_p = np.asarray(raw["scale_params"], dtype=np.float64)[idx]
    body = np.asarray(raw["body_pose_params"], dtype=np.float64)[idx].copy()
    hand = np.asarray(raw["hand_pose_params"], dtype=np.float64)[idx].copy()
    grot = np.asarray(raw["global_rot"], dtype=np.float64)[idx].copy()
    cam_t = np.asarray(raw["pred_cam_t"], dtype=np.float64)[idx]
    kp2d_net = np.asarray(raw["pred_keypoints_2d"], dtype=np.float64)[idx]
    bbox = np.asarray(raw["bbox"], dtype=np.float64)[idx]
    k_mats = np.asarray(raw["cam_K"], dtype=np.float64)[idx]
    c2w = np.asarray(raw["cam_to_world"], dtype=np.float64)[idx]
    isz = np.asarray(raw["image_size"]).reshape(-1) if "image_size" in raw else np.zeros(2)
    image_size = (int(isz[0]), int(isz[1])) if len(isz) >= 2 else None

    # one body for the whole shot
    shape_lock = np.repeat(np.median(shape, axis=0)[None], len(idx), axis=0)
    scale_lock = np.repeat(np.median(scale_p, axis=0)[None], len(idx), axis=0)

    # light zero-phase smoothing of joint angles inside contiguous runs (root orientation smoothed in world space)
    if pose_sigma > 0 and len(frames) >= 3:
        f_to_i = {int(f): i for i, f in enumerate(frames)}
        for run in segments([int(f) for f in frames], max_gap=1):
            if len(run) < 3:
                continue
            ii = [f_to_i[f] for f in run]
            body[ii] = smooth_angles(body[ii], pose_sigma)
            hand[ii] = gaussian_smooth(hand[ii], pose_sigma)
            grot[ii] = _smooth_world_global_rot(grot[ii], c2w[ii, :3, :3], pose_sigma)

    kp_local = mhr_keypoints_local_m(head, grot, body, hand, scale_lock, shape_lock)
    obs2d, obs_w, obs_index = build_observations(
        kp2d_net, image_size,
        None if kp2d_pose is None else np.asarray(kp2d_pose, dtype=np.float64)[idx],
        None if pose_score is None else np.asarray(pose_score, dtype=np.float64)[idx],
        None if kp_visible is None else np.asarray(kp_visible, dtype=bool)[idx],
    )
    height_px = np.maximum(8.0, bbox[:, 3] - bbox[:, 1])

    # sole thickness: how far the foot keypoints sit above the lowest point of the foot in the rest pose
    sole_drop_m = 0.02

    cfg = cfg or SolveConfig(fps=fps)
    kp_seen = None
    if kp_visible is not None:
        kp_seen = np.asarray(kp_visible, dtype=bool)[idx]
        if image_size is not None and image_size[0] > 0:
            kp_seen = kp_seen & (kp2d_net[..., 0] >= 0) & (kp2d_net[..., 0] < image_size[0]) & (kp2d_net[..., 1] >= 0) & (kp2d_net[..., 1] < image_size[1])
    res: SolveResult = solve_person(
        frames, k_mats, c2w, kp_local, obs2d, obs_w, cam_t, height_px, cfg=cfg, ground=ground,
        sole_drop_m=sole_drop_m, obs_index=obs_index, kp_seen=kp_seen,
    )
    return {
        "index": idx,
        "frames": frames,
        "shape": shape_lock,
        "scale_params": scale_lock,
        "body": body,
        "hand": hand,
        "global_rot": grot,
        "cam_t_m": res.cam_t_m,
        "body_scale": float(res.scale),
        "contact": res.contact,
        "reproj_px": res.reproj_px,
        "kp_local_m": kp_local,
        "obs_kp2d": obs2d,
        "obs_weight": obs_w,
        "obs_index": obs_index,
        "height_px": height_px,
        "kp_seen": kp_seen,
        "info": res.info,
    }


def keypoints_cam_cm(sol: dict) -> np.ndarray:
    """Solved MHR70 keypoints in the product's camera space (GL axes, cm) `[N, 70, 3]`."""
    p_cv = sol["body_scale"] * sol["kp_local_m"] + sol["cam_t_m"][:, None, :]
    return p_cv * CV_FLIP[None, None, :] * 100.0


# ------------------------------------------------------------------ stage 2: pose refinement against the image


def _slice_solution(sol: dict, lo: int, hi: int) -> dict:
    out = dict(sol)
    for key in ("index", "frames", "shape", "scale_params", "body", "hand", "global_rot", "cam_t_m", "contact",
                "reproj_px", "kp_local_m", "obs_kp2d", "obs_weight", "height_px", "kp_seen"):
        if sol.get(key) is not None:
            out[key] = sol[key][lo:hi]
    return out


def refine_pose_shot(sol: dict, raw: dict, head, fps: float, ground=None, window: int = 320, overlap: int = 32, **kw) -> dict:
    """Pose refinement of the whole shot (see `_refine_window`).  Long shots are refined in overlapping windows so
    GPU memory stays bounded; each window is pinned to the already refined frames it shares with the previous one,
    so the result is continuous across window borders."""
    n = len(sol["index"])
    if n <= window:
        return _refine_window(sol, raw, head, fps, ground=ground, **kw)
    out = dict(sol)
    for key in ("body", "global_rot", "cam_t_m", "kp_local_m", "reproj_px"):
        out[key] = np.array(sol[key], dtype=np.float64, copy=True)
    start, log = 0, {}
    while True:
        end = min(n, start + window)
        sub = _slice_solution(sol, start, end)
        pin = None
        if start > 0:
            pin = {k: out[k][start : start + overlap] for k in ("body", "global_rot", "cam_t_m")}
        res = _refine_window(sub, raw, head, fps, ground=ground, pin=pin, **kw)
        keep_from = 0 if start == 0 else overlap // 2
        for key in ("body", "global_rot", "cam_t_m", "kp_local_m", "reproj_px"):
            out[key][start + keep_from : end] = res[key][keep_from:]
        log[f"{start}-{end}"] = res.get("refine_log")
        if end >= n:
            break
        start = end - overlap
    out["refine_log"] = log
    return out


def _refine_window(
    sol: dict,
    raw: dict,
    head,
    fps: float,
    ground=None,
    iters: int = 160,
    sigma_px_rel: float = 0.012,
    w_prior: float = 1.0,
    w_smooth: float = 1.0,
    lr: float = 0.01,
    sole_drop_m: float = 0.02,
    verbose: bool = False,
    pin: dict | None = None,
    hold_level: bool = True,
) -> dict:
    """Refine joint angles, root orientation and translation of the whole shot against the 2D evidence.

    Stage 1 moved the body rigidly per frame; whatever 2D error is left comes from the pose itself (and so does the
    remaining foot skating: a planted foot can only stay planted if the leg pose is consistent from frame to frame).
    All frames are optimised together: reprojection of every body keypoint through the exact intrinsics, planted
    feet fixed in the world and resting on the ground, smooth world-space motion, and a prior that keeps the pose
    near the network estimate where the image does not say otherwise.  The body shape and scale stay fixed."""
    idx = sol["index"]
    device = head.scale_mean.device
    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    n = len(idx)
    k_mats = np.asarray(raw["cam_K"], dtype=np.float64)[idx]
    c2w = np.asarray(raw["cam_to_world"], dtype=np.float64)[idx]
    kp2d_np = np.asarray(sol["obs_kp2d"], dtype=np.float64)
    frames = np.asarray(sol["frames"], dtype=np.int64)
    w_np = np.asarray(sol["obs_weight"], dtype=np.float64)
    obs_index = torch.as_tensor(np.asarray(sol["obs_index"], dtype=np.int64), device=device)
    sigma_px = np.maximum(1.0, sigma_px_rel * np.asarray(sol["height_px"], dtype=np.float64))

    body0, grot0 = to_t(sol["body"]), to_t(sol["global_rot"])
    hand, scale_p, shape = to_t(sol["hand"]), to_t(sol["scale_params"]), to_t(sol["shape"])
    tau0 = to_t(sol["cam_t_m"])
    s_body = float(sol["body_scale"])
    kp2d = to_t(kp2d_np)
    w_kp = to_t(w_np)
    sig = to_t(sigma_px)[:, None]
    fx, fy = to_t(k_mats[:, 0, 0])[:, None], to_t(k_mats[:, 1, 1])[:, None]
    cx, cy = to_t(k_mats[:, 0, 2])[:, None], to_t(k_mats[:, 1, 2])[:, None]
    flip = torch.tensor([1.0, -1.0, -1.0], device=device)
    m_w = to_t(c2w[:, :3, :3] @ np.diag([1.0, -1.0, -1.0]))
    o_w = to_t(c2w[:, :3, 3] * 0.01)
    contact = torch.as_tensor(np.asarray(sol["contact"], dtype=bool), device=device)
    dt = np.diff(frames)
    consec = torch.as_tensor(dt == 1, device=device)
    tri = torch.as_tensor((dt[:-1] == 1) & (dt[1:] == 1), device=device) if n >= 3 else None
    fk = (30.0 / max(fps, 1.0))
    sig_acc = 0.009 * fk * fk
    sig_anc = 0.006
    foot_np = np.array([KP_L_FOOT, KP_R_FOOT])

    # planted feet: one free world anchor per (stance phase, foot keypoint), initialised from the stage-1 result
    contact_np = np.asarray(sol["contact"], dtype=bool)
    kp_w0 = keypoints_world_m(sol["kp_local_m"], sol["cam_t_m"], s_body, c2w)
    a_t, a_kp, a_id, a_init = [], [], [], []
    for side, pieces in footprint_groups(contact_np, frames, kp_w0, fps):
        for kp in foot_np[side]:
            rows = np.concatenate([np.arange(lo, hi) for lo, hi in pieces])
            a_t.append(rows); a_kp.append(np.full(len(rows), kp)); a_id.append(np.full(len(rows), len(a_init)))
            a_init.append(np.median(kp_w0[rows, kp], axis=0))
    has_anchor = len(a_init) > 0
    if has_anchor:
        a_t = torch.as_tensor(np.concatenate(a_t), device=device)
        a_kp = torch.as_tensor(np.concatenate(a_kp), device=device)
        a_id = torch.as_tensor(np.concatenate(a_id), device=device)
        anchors = to_t(np.stack(a_init)).clone().requires_grad_(True)
        seen_np = sol.get("kp_seen")
        a_seen = (
            torch.ones(len(a_t), device=device)
            if seen_np is None
            else torch.where(torch.as_tensor(np.asarray(seen_np, dtype=bool), device=device)[a_t, a_kp], 1.0, 0.04)
        )

    d_body = torch.zeros_like(body0, requires_grad=True)
    d_rot = torch.zeros_like(grot0, requires_grad=True)
    d_tau = torch.zeros_like(tau0, requires_grad=True)
    groups = [{"params": [d_body, d_rot], "lr": lr}, {"params": [d_tau], "lr": lr * 0.3}]
    if has_anchor:
        groups.append({"params": [anchors], "lr": lr * 0.1})
    opt = torch.optim.Adam(groups)
    zeros_t = torch.zeros((n, 3), device=device)
    expr = torch.zeros((n, head.num_face_comps), device=device)
    lvl_lo = 0  # frames from here on hold the depth level (pinned frames of a window are held by the pin itself)
    if pin is not None:
        n_pin = len(pin["body"])
        lvl_lo = min(n_pin, max(0, n - 1))
        pin_body = to_t(pin["body"]) - body0[:n_pin]
        pin_rot = to_t(pin["global_rot"]) - grot0[:n_pin]
        pin_tau = to_t(pin["cam_t_m"]) - tau0[:n_pin]

    def forward():
        _, kp = head.mhr_forward(
            global_trans=zeros_t, global_rot=grot0 + d_rot, body_pose_params=body0 + d_body, hand_pose_params=hand,
            scale_params=scale_p, shape_params=shape, expr_params=expr, return_keypoints=True,
        )
        loc = kp[:, :70] * flip  # camera axes, metres, relative to the body origin
        p_cam = s_body * loc + (tau0 + d_tau)[:, None, :]
        return loc, p_cam

    log = {}
    diverged = False
    with torch.enable_grad():
        for it in range(iters):
            opt.zero_grad()
            loc, p_cam = forward()
            p_obs = p_cam[:, obs_index]
            z = p_obs[..., 2].clamp_min(0.05)
            du = (fx * p_obs[..., 0] / z + cx - kp2d[..., 0]) / sig
            dv = (fy * p_obs[..., 1] / z + cy - kp2d[..., 1]) / sig
            e2 = du * du + dv * dv
            hub = torch.where(e2 <= 6.25, e2, 5.0 * torch.sqrt(e2.clamp_min(1e-12)) - 6.25)
            l_2d = (hub * w_kp).sum() / n

            q = torch.einsum("tij,tkj->tki", m_w, p_cam) + o_w[:, None, :]
            l_smooth = q.new_zeros(())
            if tri is not None and bool(tri.any()):
                acc = (q[2:] - 2.0 * q[1:-1] + q[:-2])[tri]
                core = torch.tensor([5, 6, 9, 10], device=device)
                l_smooth = ((acc[:, core] / sig_acc) ** 2).sum() / n * 0.5 + ((acc / (3.0 * sig_acc)) ** 2).sum() / n * 0.5
            l_con = q.new_zeros(())
            l_gnd = q.new_zeros(())
            if has_anchor:
                pts = q[a_t, a_kp]
                r2 = (((pts - anchors[a_id]) / sig_anc) ** 2).sum(-1)
                l_con = torch.where(r2 <= 6.25, r2, 5.0 * torch.sqrt(r2.clamp_min(1e-12)) - 6.25).sum() / n
                if ground is not None:
                    nrm, off, ok = ground.support(pts.detach().cpu().numpy().astype(np.float64))
                    if ok.any():
                        h = (pts * to_t(nrm)).sum(-1) + to_t(off) - sole_drop_m * s_body
                        l_gnd = ((h / 0.01) ** 2 * a_seen * torch.as_tensor(ok, device=device, dtype=torch.float32)).sum() / n
            l_prior = w_prior * ((d_body / 0.12) ** 2).sum() / n + w_prior * ((d_rot / 0.10) ** 2).sum() / n
            l_tsm = d_body.new_zeros(())
            if tri is not None and bool(tri.any()):
                pa = (body0 + d_body)
                acc_p = (pa[2:] - 2.0 * pa[1:-1] + pa[:-2])[tri]
                acc_r = (d_rot[2:] - 2.0 * d_rot[1:-1] + d_rot[:-2])[tri]
                l_tsm = w_smooth * (((acc_p / 0.05) ** 2).sum() + ((acc_r / 0.03) ** 2).sum()) / n
            loss = l_2d + l_smooth + l_con + l_gnd + l_prior + l_tsm
            if pin is not None:
                loss = loss + (((d_body[:n_pin] - pin_body) / 0.004) ** 2).sum() / n + (
                    ((d_rot[:n_pin] - pin_rot) / 0.003) ** 2).sum() / n + (((d_tau[:n_pin] - pin_tau) / 0.001) ** 2).sum() / n
            if not bool(torch.isfinite(loss)):
                diverged = True
                break
            loss.backward()
            opt.step()
            if not all(bool(torch.isfinite(par).all()) for par in (d_body, d_rot, d_tau)):
                diverged = True
                break
            if hold_level and lvl_lo < n:
                # the refinement may reshape the body and move it frame by frame, but the person's average
                # distance to the camera stays where stage 1 put it (see `solve_trajectory`: motion penalties
                # get cheaper the closer the body is, which is not evidence about distance)
                with torch.no_grad():
                    rel = d_tau[lvl_lo:, 2] / tau0[lvl_lo:, 2].clamp_min(0.05)
                    d_tau[lvl_lo:, 2] -= tau0[lvl_lo:, 2] * rel.mean()
            if it in (0, iters - 1):
                log[it] = {k: float(v) for k, v in dict(l2d=l_2d, smooth=l_smooth, contact=l_con, ground=l_gnd,
                                                         prior=l_prior, tsm=l_tsm).items()}
                if verbose:
                    print(f"   refine it {it}: {log[it]}")

    if diverged:
        # never hand on a result with invalid numbers: the stage-1 solution stays as it is
        out = dict(sol)
        out["refine_log"] = {"diverged": True}
        return out
    with torch.no_grad():
        loc, p_cam = forward()
        p_obs = p_cam[:, obs_index]
        z = p_obs[..., 2].clamp_min(0.05)
        e_px = torch.hypot(fx * p_obs[..., 0] / z + cx - kp2d[..., 0], fy * p_obs[..., 1] / z + cy - kp2d[..., 1])
        reproj = (e_px * w_kp).sum(1) / w_kp.sum(1).clamp_min(1e-9)
    out = dict(sol)
    out["body"] = _np(body0 + d_body).astype(np.float64)
    out["global_rot"] = _np(grot0 + d_rot).astype(np.float64)
    out["cam_t_m"] = _np(tau0 + d_tau).astype(np.float64)
    out["kp_local_m"] = _np(loc).astype(np.float64)
    out["reproj_px"] = _np(reproj).astype(np.float64)
    out["refine_log"] = log
    return out
