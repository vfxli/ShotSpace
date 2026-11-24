"""Pre-solve Plate Grading: Exposure (EV), Brightness, Contrast, Saturation, and Gamma.

Identical formula in Python (for ViTDet + SAM 3D Body + SAM 3.1/SAM 3D Objects inference) and GLSL (WebGL live preview):
    c = x * 2^exposure + brightness
    c = (c - 0.5) * contrast + 0.5
    c = max(c, 0.0) ^ (1.0 / max(gamma, 0.05))
    luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
    y = clip(luma + saturation * (c - luma), 0.0, 1.0)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GradeParams:
    exposure: float = 0.0  # EV stops, [-4.0, +4.0]
    brightness: float = 0.0  # additive offset, [-0.5, +0.5]
    contrast: float = 1.0  # pivot around 0.5, [0.2, 2.5]
    saturation: float = 1.0  # Rec.709 luma saturation, [0.0, 2.5]
    gamma: float = 1.0  # display/tone gamma, [0.2, 3.0]

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.exposure) < 1e-6
            and abs(self.brightness) < 1e-6
            and abs(self.contrast - 1.0) < 1e-6
            and abs(self.saturation - 1.0) < 1e-6
            and abs(self.gamma - 1.0) < 1e-6
        )

    def clamped(self) -> GradeParams:
        return GradeParams(
            exposure=float(np.clip(self.exposure, -4.0, 4.0)),
            brightness=float(np.clip(self.brightness, -0.5, 0.5)),
            contrast=float(np.clip(self.contrast, 0.2, 2.5)),
            saturation=float(np.clip(self.saturation, 0.0, 2.5)),
            gamma=float(np.clip(self.gamma, 0.2, 3.0)),
        )

    def to_dict(self) -> dict:
        c = self.clamped()
        return asdict(c)

    @classmethod
    def from_dict(cls, data: dict | None) -> GradeParams:
        if not data:
            return cls()
        return cls(
            exposure=float(data.get("exposure", 0.0)),
            brightness=float(data.get("brightness", 0.0)),
            contrast=float(data.get("contrast", 1.0)),
            saturation=float(data.get("saturation", 1.0)),
            gamma=float(data.get("gamma", 1.0)),
        ).clamped()


def _build_float_curve_lut(g: GradeParams) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    gain = float(2.0 ** g.exposure)
    inv_gamma = float(1.0 / max(g.gamma, 0.05))
    c = x * gain + float(g.brightness)
    c = (c - 0.5) * float(g.contrast) + 0.5
    return np.power(np.maximum(c, 0.0), inv_gamma).astype(np.float32)


def build_uint8_lut(grade: GradeParams) -> np.ndarray:
    """Build a 256-entry uint8 per-channel lookup table (when `saturation == 1.0`)."""
    g = grade.clamped()
    if g.is_identity:
        return np.arange(256, dtype=np.uint8)
    y = _build_float_curve_lut(g)
    return np.clip(np.rint(y * 255.0), 0.0, 255.0).astype(np.uint8)


def apply_grade_rgb(img: np.ndarray, grade: GradeParams, lut: np.ndarray | None = None) -> np.ndarray:
    """Apply `grade` to an RGB image (`uint8`, `uint16`, or `float32`/`float64` in [0..1] / HDR linear)
    and return an RGB `uint8` image ready for ViTDet, SAM 3D Body & SAM 3D Objects."""
    g = grade.clamped()
    arr = np.asarray(img)
    has_sat = abs(g.saturation - 1.0) >= 1e-6

    if arr.dtype == np.uint8:
        if g.is_identity:
            return arr
        if not has_sat:
            table = lut if lut is not None else build_uint8_lut(g)
            return table[arr]
        curve_f = _build_float_curve_lut(g)
        xf = curve_f[arr]
        luma = 0.2126 * xf[..., 0:1] + 0.7152 * xf[..., 1:2] + 0.0722 * xf[..., 2:3]
        xf = luma + np.float32(g.saturation) * (xf - luma)
        return np.clip(np.rint(xf * 255.0), 0.0, 255.0).astype(np.uint8)

    if arr.dtype == np.uint16:
        xf = arr.astype(np.float32) * (1.0 / 65535.0)
    else:
        xf = arr.astype(np.float32)

    if not g.is_identity:
        gain = np.float32(2.0 ** g.exposure)
        inv_gamma = np.float32(1.0 / max(g.gamma, 0.05))
        xf = xf * gain + np.float32(g.brightness)
        xf = (xf - np.float32(0.5)) * np.float32(g.contrast) + np.float32(0.5)
        xf = np.power(np.maximum(xf, 0.0), inv_gamma)
        if has_sat:
            luma = 0.2126 * xf[..., 0:1] + 0.7152 * xf[..., 1:2] + 0.0722 * xf[..., 2:3]
            xf = luma + np.float32(g.saturation) * (xf - luma)

    return np.clip(np.rint(xf * 255.0), 0.0, 255.0).astype(np.uint8)
