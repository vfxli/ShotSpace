"""Zero-phase temporal smoothing helpers, contiguous-run utilities, the optional root wave filter of the export
step, and expansion of solved frames onto the full shot with visibility flags (pure NumPy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .poses import continuous


def segments(frames: list[int], max_gap: int = 1) -> list[list[int]]:
    """Split sorted frame numbers into contiguous runs so smoothing never blurs across detection gaps."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= max_gap:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def smoothing_sigma(strength: float) -> float:
    """UI smoothing slider [0..1] -> Gaussian sigma in frames (default 0.5 -> 1.5 frames)."""
    return 3.0 * float(strength)


def despike_hampel_1d(values: np.ndarray, threshold_mad: float = 3.2) -> np.ndarray:
    """Remove isolated 1-frame neural/detector glitches without dulling genuine motion starts/stops."""
    arr = np.array(values, dtype=np.float64, copy=True)
    n = len(arr)
    if n < 5:
        return arr
    flat = arr.reshape(n, -1)
    for t in range(1, n - 1):
        mid = 0.5 * (flat[t - 1] + flat[t + 1])
        dev = flat[t] - mid
        lo = max(0, t - 3)
        hi = min(n, t + 4)
        win = flat[lo:hi]
        med = np.median(win, axis=0)
        mad = np.median(np.abs(win - med[None, :]), axis=0) * 1.4826 + 1e-6
        # Only clamp genuine reversals where `t` jumps away from both `t-1` and `t+1` in the same direction
        reversal = ((flat[t] - flat[t - 1]) * (flat[t] - flat[t + 1])) > 0.0
        spike = reversal & (np.abs(dev) > threshold_mad * mad)
        if np.any(spike):
            flat[t, spike] = mid[spike] + np.sign(dev[spike]) * (0.35 * threshold_mad * mad[spike])
    return flat.reshape(values.shape)


def _one_euro_alpha(cutoff: np.ndarray | float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-5))
    return 1.0 / (1.0 + tau)


def one_euro_zero_phase(
    values: np.ndarray,
    min_cutoff: float = 0.065,
    beta: float = 0.55,
    d_cutoff: float = 0.22,
    weights: np.ndarray | None = None,
    despike: bool = True,
) -> np.ndarray:
    """Zero-phase forward-backward velocity-adaptive One-Euro filter along axis 0:
      - Heavy damping (`min_cutoff`) when a joint/box/root is stationary or moving slowly -> kills standing jitter;
      - Automatic wide bandwidth (`min_cutoff + beta * |v_norm|`) during fast motion -> zero lag or motion smear;
      - Exact zero phase delay via symmetric forward + backward filtering + odd-reflection boundary anchoring.
    """
    n = len(values)
    if n < 3:
        return np.array(values, copy=True)
    src = despike_hampel_1d(values) if despike else np.array(values, dtype=np.float64, copy=True)
    flat = src.reshape(n, -1)
    w_arr = (
        np.clip(np.asarray(weights, dtype=np.float64).reshape(n), 0.1, 1.0)
        if (weights is not None and len(weights) == n)
        else np.ones(n, dtype=np.float64)
    )

    # Normalize velocity per channel by robust step scale so angles (rad), boxes (px), and positions (cm) behave uniformly
    diffs = np.abs(np.diff(flat, axis=0))
    scale = np.maximum(np.percentile(diffs, 75, axis=0) * 1.35, 1e-4)

    def _causal_pass(seq: np.ndarray, w_seq: np.ndarray) -> np.ndarray:
        out_p = np.empty_like(seq)
        out_p[0] = seq[0]
        dx_prev = np.zeros(seq.shape[1], dtype=np.float64)
        a_d = float(_one_euro_alpha(d_cutoff))
        for i in range(1, len(seq)):
            dx_raw = (seq[i] - out_p[i - 1]) / scale
            dx_hat = a_d * dx_raw + (1.0 - a_d) * dx_prev
            dx_prev = dx_hat
            cutoff = min_cutoff + beta * np.abs(dx_hat)
            alpha = _one_euro_alpha(cutoff) * (0.45 + 0.55 * w_seq[i])
            alpha = np.clip(alpha, 0.02, 0.98)
            out_p[i] = alpha * seq[i] + (1.0 - alpha) * out_p[i - 1]
        return out_p

    # Odd-symmetric point reflection padding at boundaries prevents endpoint pullback on linear motion!
    pad = min(6, n - 1)
    left_pad = 2.0 * flat[0:1] - flat[pad:0:-1]
    right_pad = 2.0 * flat[-1:] - flat[-2 : -2 - pad : -1]
    ext = np.concatenate([left_pad, flat, right_pad], axis=0)
    w_ext = np.concatenate([w_arr[pad:0:-1], w_arr, w_arr[-2 : -2 - pad : -1]], axis=0)

    fwd = _causal_pass(ext, w_ext)
    bwd = _causal_pass(fwd[::-1], w_ext[::-1])[::-1]
    res = bwd[pad : pad + n]
    return res.reshape(values.shape)


def gaussian_smooth(values: np.ndarray, sigma: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Zero-phase Gaussian filter along axis 0 with odd-symmetric point reflection at segment boundaries
    so linear walking motion has zero endpoint pullback or phase lag."""
    if sigma <= 0 or len(values) < 3:
        return np.array(values, copy=True)
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    n = len(flat)
    w_arr = (
        np.clip(np.asarray(weights, dtype=np.float64).reshape(n), 0.05, 1.0)
        if (weights is not None and len(weights) == n)
        else np.ones(n, dtype=np.float64)
    )
    radius = max(1, int(np.ceil(3.0 * sigma)))
    pad = min(radius, n - 1)
    left_pad = 2.0 * flat[0:1] - flat[pad:0:-1]
    right_pad = 2.0 * flat[-1:] - flat[-2 : -2 - pad : -1]
    ext = np.concatenate([left_pad, flat, right_pad], axis=0)
    w_ext = np.concatenate([w_arr[pad:0:-1], w_arr, w_arr[-2 : -2 - pad : -1]], axis=0)

    offs = np.arange(-radius, radius + 1, dtype=np.float64)
    base_k = np.exp(-0.5 * (offs / sigma) ** 2)
    out = np.empty_like(flat, dtype=np.float64)
    for i in range(n):
        c = i + pad
        lo = max(0, c - radius)
        hi = min(len(ext), c + radius + 1)
        k_sl = base_k[(lo - c + radius) : (hi - c + radius)] * w_ext[lo:hi]
        out[i] = (k_sl[:, None] * ext[lo:hi]).sum(0) / max(float(k_sl.sum()), 1e-12)
    return out.reshape(values.shape)


def smooth_angles(values: np.ndarray, sigma: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Unwrap radian angle channels along time, apply velocity-adaptive zero-phase One-Euro damping + Gaussian polish."""
    if sigma <= 0 or len(values) < 3:
        return np.array(values, copy=True)
    unwrapped = np.unwrap(np.asarray(values, dtype=np.float64), axis=0)
    strength = float(np.clip(sigma / 3.0, 0.05, 1.5))
    min_cut = max(0.03, 0.13 / (0.5 + strength))
    beta = 0.55 / (0.65 + 0.35 * strength)
    adap = one_euro_zero_phase(unwrapped, min_cutoff=min_cut, beta=beta, weights=weights, despike=True)
    return gaussian_smooth(adap, 0.48 * sigma, weights=weights)


def smooth_quats(q: np.ndarray, sigma: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Sign-align unit quaternions [F, 4], apply velocity-adaptive zero-phase One-Euro + Gaussian polish, and renormalize."""
    if sigma <= 0 or len(q) < 3:
        return np.array(q, copy=True)
    cont = continuous(q)
    strength = float(np.clip(sigma / 3.0, 0.05, 1.5))
    min_cut = max(0.03, 0.12 / (0.5 + strength))
    beta = 0.52 / (0.65 + 0.35 * strength)
    adap = one_euro_zero_phase(cont, min_cutoff=min_cut, beta=beta, weights=weights, despike=True)
    s = gaussian_smooth(adap, 0.50 * sigma, weights=weights)
    return s / np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-12)


def expand_to_full_shot_with_visibility(
    shot_frames: list[int],
    valid_frames: list[int],
    valid_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map `valid_frames` (length M >= 1) onto the entire `shot_frames` sequence (length F):

    Returns:
      - `src_valid_idx` `[F]` int64: index in `0 .. M-1` whose pose/position should be used at each shot frame
        (exact match when visible; frozen dead-still at the peeled high-confidence boundary frame when hidden!).
      - `visible` `[F]` int32: `1` on valid solved frames, `0` on undetected or peeled boundary frames (auto-hidden!).
      - `anchor_frames` `[F]` int64: the actual valid frame number `valid_frames[src_valid_idx[t]]` used at `t`.
    """
    m = len(valid_frames)
    if m == 0:
        raise ValueError("valid_frames must contain at least one frame")
    vf = np.asarray(valid_frames, dtype=np.int64)
    vq = np.asarray(valid_quality, dtype=np.float64).reshape(m)
    f_to_vi = {int(f): idx for idx, f in enumerate(valid_frames)}

    f_total = len(shot_frames)
    src_idx = np.zeros(f_total, dtype=np.int64)
    visible = np.zeros(f_total, dtype=np.int32)

    for t, f in enumerate(shot_frames):
        fi = int(f)
        if fi in f_to_vi:
            src_idx[t] = f_to_vi[fi]
            visible[t] = 1
        elif fi < vf[0]:
            src_idx[t] = 0
            visible[t] = 0
        elif fi > vf[-1]:
            src_idx[t] = m - 1
            visible[t] = 0
        else:
            # Mid-sequence occlusion gap: hold stationary at the preceding clean pre-occlusion frame (`left`)
            # unless `left` is substantially lower quality than `right`!
            right = int(np.searchsorted(vf, fi))
            left = max(0, right - 1)
            right = min(m - 1, right)
            src_idx[t] = left if vq[left] >= 0.85 * vq[right] else right
            visible[t] = 0

    anchor_frames = vf[src_idx]
    return src_idx, visible, anchor_frames
