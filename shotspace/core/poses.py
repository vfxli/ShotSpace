"""3D pose, quaternion, hierarchy, and camera-track alignment math (pure NumPy)."""

from __future__ import annotations

import bisect

import numpy as np

UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrices -> [..., 4] unit quaternions (w, x, y, z) with w >= 0."""
    r = np.asarray(r, dtype=np.float64)
    shape = r.shape[:-2]
    r = r.reshape(-1, 3, 3)
    tr = np.trace(r, axis1=1, axis2=2)
    cand = np.stack(
        [
            1.0 + tr,
            1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
            1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
            1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2],
        ],
        axis=1,
    )
    best = np.argmax(cand, axis=1)
    q = np.empty((len(r), 4), dtype=np.float64)
    for k in range(4):
        idx = best == k
        if not idx.any():
            continue
        m = r[idx]
        s = np.sqrt(np.maximum(cand[idx, k], 1e-12)) * 2.0
        if k == 0:
            q[idx] = np.stack([s / 4.0, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)
        elif k == 1:
            q[idx] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, s / 4.0, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)
        elif k == 2:
            q[idx] = np.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, s / 4.0, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)
        else:
            q[idx] = np.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, s / 4.0], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q.reshape(*shape, 4)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[..., 4] quaternions (w, x, y, z) -> [..., 3, 3] rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], -1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], -1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], -1),
        ],
        -2,
    )


def slerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, -1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    safe = np.where(small, 1.0, sin)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs along axis 0 so consecutive frames stay in the same hemisphere."""
    q = np.array(q, dtype=np.float64, copy=True)
    if len(q) < 2:
        return q
    sign = np.sign(np.sum(q[1:] * q[:-1], -1))
    sign[sign == 0] = 1.0
    q[1:] *= np.cumprod(sign, axis=0)[..., None]
    return q


def orthogonal(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    return u @ vt


def handedness(m: np.ndarray) -> np.ndarray:
    return np.where(np.linalg.det(np.asarray(m, dtype=np.float64)) < 0, -1.0, 1.0)


def orthonormal(m: np.ndarray) -> np.ndarray:
    q = orthogonal(m)
    return q * np.stack([handedness(q), np.ones(q.shape[:-2]), np.ones(q.shape[:-2])], -1)[..., None, :]


def signed_scales(m: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(np.asarray(m, dtype=np.float64), axis=-2)
    return s * np.stack([handedness(m), np.ones(s.shape[:-1]), np.ones(s.shape[:-1])], -1)


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[..., 4, 4] -> (translations [..., 3], unit quats wxyz [..., 4], signed scales [..., 3])."""
    m = np.asarray(m, dtype=np.float64)
    basis = m[..., :3, :3]
    return m[..., :3, 3], matrix_to_quat(orthonormal(basis)), signed_scales(basis)


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    q = matrix_to_quat(r)
    axis = q[..., 1:]
    sin_half = np.linalg.norm(axis, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, q[..., 0])
    safe = np.where(sin_half < 1e-12, 1.0, sin_half)
    return np.where((sin_half < 1e-12)[..., None], 0.0, axis / safe[..., None] * angle[..., None])


def aim_frame(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Orthonormal [3, 3] frame (columns): +X along `direction`, +Z orthogonalized from `reference`, +Y = Z x X."""
    x = direction / np.linalg.norm(direction)
    z = reference - np.dot(reference, x) * x
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], axis=1)


def world_from_local(local: np.ndarray, parents) -> np.ndarray:
    world = np.array(local, dtype=np.float64, copy=True)
    for j, p in enumerate(parents):
        if int(p) >= 0:
            world[..., j, :, :] = world[..., int(p), :, :] @ world[..., j, :, :]
    return world


def local_from_world(world: np.ndarray, parents) -> np.ndarray:
    world = np.asarray(world, dtype=np.float64)
    local = world.copy()
    for j, p in enumerate(parents):
        if int(p) >= 0:
            local[..., j, :, :] = np.linalg.inv(world[..., int(p), :, :]) @ world[..., j, :, :]
    return local


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of [N, 3, 3] rotations via SVD projection onto SO(3) (never degenerates on collinear/tripod tracks)."""
    u, _, vt = np.linalg.svd(np.asarray(rotations, dtype=np.float64).sum(0))
    d = np.eye(3, dtype=np.float64)
    d[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ d @ vt


def rotation_deg(r: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)))


def track_scale(xs: np.ndarray, xd: np.ndarray) -> tuple[float, bool]:
    """Least-squares positive scale fitting centred `xs` (already rotated) onto `xd`.
    Returns (|s|, reversed_direction). Never produces a negative mirror scale."""
    xs, xd = np.asarray(xs, dtype=np.float64), np.asarray(xd, dtype=np.float64)
    denom = float((xs ** 2).sum())
    s = float((xd * xs).sum() / denom) if denom > 1e-12 else 1.0
    if not np.isfinite(s) or s == 0.0:
        return 1.0, False
    return abs(s), bool(s < 0)


def scaled_align(
    src: np.ndarray,
    dst: np.ndarray,
    fit_scale: bool = True,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    """Whole-shot similarity alignment T = [s*R, t; 0, 1] such that dst[i] ~ T @ src[i].

    - Rotation R comes from camera orientations: chordal mean of dst_R @ src_R^T (immune to collinear/coincident centres).
    - Scale s > 0 and translation t come from least-squares on camera optical centres given R.
    - When `fit_scale=False` (target camera moved < 10 cm RMS), s is locked to 1.0.

    Returns:
      (T [4,4], scale, src_spread, rms_ang_error_deg, rms_pos_error_cm, track_reversed)
    """
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    r = mean_rotation(dst[:, :3, :3] @ np.swapaxes(src[:, :3, :3], -1, -2))
    cs, cd = src[:, :3, 3], dst[:, :3, 3]
    xs, xd = cs - cs.mean(0), cd - cd.mean(0)
    denom = float((xs ** 2).sum())
    src_spread = float(np.sqrt(denom / max(len(cs), 1)))
    if fit_scale:
        s, reversed_dir = track_scale(xs @ r.T, xd)
    else:
        s, reversed_dir = 1.0, False
    t = cd.mean(0) - s * (r @ cs.mean(0))
    big_t = np.eye(4, dtype=np.float64)
    big_t[:3, :3] = s * r
    big_t[:3, 3] = t

    rel = np.swapaxes(r @ src[:, :3, :3], -1, -2) @ dst[:, :3, :3]
    ang = np.degrees(np.linalg.norm(matrix_to_rotvec(rel), axis=-1))
    pos = np.linalg.norm(s * (cs @ r.T) + t - cd, axis=-1)
    return (
        big_t,
        float(s),
        src_spread,
        float(np.sqrt(np.mean(ang ** 2))),
        float(np.sqrt(np.mean(pos ** 2))),
        reversed_dir,
    )


def blend_poses(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    q = slerp(matrix_to_quat(a[:3, :3]), matrix_to_quat(b[:3, :3]), t)
    out = np.array(b, dtype=np.float64, copy=True)
    out[:3, :3] = quat_to_matrix(q)
    out[:3, 3] = (1.0 - t) * a[:3, 3] + t * b[:3, 3]
    return out


def interpolate_poses(known: dict[int, np.ndarray], frames) -> np.ndarray:
    """Interpolate/hold [F, 4, 4] rigid poses across `frames` from sparse or partial `known` samples."""
    keys = sorted(known)
    if not keys:
        return np.repeat(np.eye(4, dtype=np.float64)[None], len(frames), axis=0)
    out = []
    for f in (int(x) for x in frames):
        j = bisect.bisect_left(keys, f)
        if j < len(keys) and keys[j] == f:
            out.append(np.asarray(known[f], dtype=np.float64))
            continue
        a, b = keys[max(j - 1, 0)], keys[min(j, len(keys) - 1)]
        out.append(blend_poses(known[a], known[b], 0.0 if a == b else (f - a) / (b - a)))
    return np.stack(out)
