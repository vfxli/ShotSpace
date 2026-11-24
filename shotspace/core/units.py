"""Unified internal coordinate & unit conventions for ShotSpace.

Internal standard:
  - Length unit: centimetres (cm, 1 m = 100 cm)
  - World axes: right-handed, +Y up
  - Camera local axes (GL / USD / Maya / Houdini): +X right, +Y up, looking down -Z
  - Matrices: 4x4 column-vector convention (p' = M @ p, translation in M[:3, 3])
  - Camera extrinsics: cam_to_world in SE(3), strictly rigid (det(R) = +1, no scale)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

M_TO_CM = 100.0
CM_TO_M = 0.01
CM_PER_MM = 0.1
CV_TO_GL = np.array([1.0, -1.0, -1.0], dtype=np.float64)

DEFAULT_FPS = 24.0
FILMBACK_MM = 36.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

STANDARD_FPS = (
    12.0,
    15.0,
    24000.0 / 1001.0,
    24.0,
    25.0,
    30000.0 / 1001.0,
    30.0,
    48.0,
    50.0,
    60000.0 / 1001.0,
    60.0,
    90.0,
    100.0,
    120.0,
    240.0,
)
FPS_SNAP = 2e-4

Z_UP_TO_Y_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def standard_fps(fps: float) -> float:
    """Snap written FPS (e.g. 23.976 -> 24000/1001) to exact industry standard rates."""
    fps = float(fps)
    if fps <= 0:
        return DEFAULT_FPS
    nearest = min(STANDARD_FPS, key=lambda s: abs(fps / s - 1.0))
    return nearest if abs(fps / nearest - 1.0) <= FPS_SNAP else fps


def to_cm(unit: str) -> float:
    unit_norm = unit.strip().lower()
    if unit_norm in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return 1.0
    if unit_norm in ("m", "meter", "meters", "metre", "metres"):
        return M_TO_CM
    if unit_norm in ("mm", "millimeter", "millimeters"):
        return 0.1
    raise ValueError(f"Unsupported length unit: {unit!r} (expected 'cm' or 'm')")


def focal_px(focal_mm, filmback_mm, width: int | float) -> np.ndarray:
    """Convert focal length (mm) and horizontal aperture (mm) to pixels at `width`."""
    return np.asarray(focal_mm, dtype=np.float64) / np.asarray(filmback_mm, dtype=np.float64) * float(width)


def focal_mm(focal_pixels, filmback_mm, width: int | float) -> np.ndarray:
    """Inverse of `focal_px`."""
    return np.asarray(focal_pixels, dtype=np.float64) * np.asarray(filmback_mm, dtype=np.float64) / float(width)


def fov_x_deg(focal_pixels, width: int | float) -> float:
    return float(np.degrees(2.0 * np.arctan(float(width) / (2.0 * np.asarray(focal_pixels, dtype=np.float64)))))


def rigid(m: np.ndarray) -> np.ndarray:
    """Strip any parent group scale/shear from [..., 4, 4] camera matrices via SVD,
    enforcing det(R) = +1 (proper right-handed SO(3) rotation)."""
    out = np.array(m, dtype=np.float64, copy=True)
    orig_shape = out.shape
    flat = out.reshape(-1, 4, 4)
    u, _, vt = np.linalg.svd(flat[:, :3, :3])
    r = u @ vt
    det = np.linalg.det(r)
    neg = det < 0
    if np.any(neg):
        u_fix = u[neg].copy()
        u_fix[:, :, -1] *= -1.0
        r[neg] = u_fix @ vt[neg]
    flat[:, :3, :3] = r
    flat[:, 3, :3] = 0.0
    flat[:, 3, 3] = 1.0
    return flat.reshape(orig_shape)


@dataclass(frozen=True)
class Axes:
    """Conversion from a source file's unit and up-axis to ShotSpace's (cm, +Y up, right-handed)."""

    to_cm: float = 1.0
    turn: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    @staticmethod
    def of(unit_to_cm: float, up: str = "y") -> Axes:
        up_norm = up.strip().lower()
        if up_norm not in ("y", "z"):
            raise ValueError(f"Unsupported up axis: {up!r} (expected 'y' or 'z')")
        return Axes(float(unit_to_cm), Z_UP_TO_Y_UP.copy() if up_norm == "z" else np.eye(3, dtype=np.float64))

    def matrices(self, m: np.ndarray) -> np.ndarray:
        out = np.array(m, dtype=np.float64, copy=True)
        out[..., :3, 3] *= self.to_cm
        turn4 = np.eye(4, dtype=np.float64)
        turn4[:3, :3] = self.turn
        return turn4 @ out

    def points(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=np.float64) * self.to_cm

    def world_points(self, p: np.ndarray) -> np.ndarray:
        return self.points(p) @ self.turn.T
