"""OpenColorIO (OCIO v2) Color Management Engine for ShotSpace.

Why OCIO is mandatory for SAM 3D Body & ViTDet-H:
  - Both ViTDet-H (`detectron2` COCO) and SAM 3D Body (`DINOv3` ImageNet normalization) strictly expect
    8-bit display-referred `sRGB` (`0..255`) images.
  - Undistorted VFX plates delivered from Nuke / 3DEqualizer / Flame as `.exr` are scene-linear `ACEScg`
    (`AP1` wide-gamut primaries, linear transfer) by studio convention; `.dpx` / `.cin` scans are `ADX10` log.
  - Naive clamping (`clip(exr, 0, 1) * 255`) crushes mid-grey (`0.18 -> 46/255` instead of `~118/255`), clips
    scene-linear highlights > 1.0, and leaves `AP1` primaries un-matrixed (distorting skin tones & edges).

Uses `PyOpenColorIO` built-in `studio-config-v2.2.0_aces-v1.3_ocio-v2.4` (Nuke's standard ACES 1.3 Studio Config,
100% offline, zero external files required; also honors `$OCIO` environment variable if set by a studio).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO

from .plate_grade import GradeParams, apply_grade_rgb, build_uint8_lut

BUILTIN_ACES_CONFIG = "studio-config-v2.2.0_aces-v1.3_ocio-v2.4"

# Curated VFX studio input colorspaces exposed in the UI (all verified in ACES 1.3 Studio Config)
STUDIO_INPUT_COLORSPACES: list[tuple[str, str]] = [
    ("auto", "自动识别 (.exr = ACEScg / .dpx = ADX10 / 其余 = sRGB)"),
    ("ACEScg", "ACEScg (VFX EXR 默认场景线性 · AP1)"),
    ("ACES2065-1", "ACES2065-1 (AP0 交换线性)"),
    ("ACEScct", "ACEScct (调色对数空间)"),
    ("Linear Rec.709 (sRGB)", "Linear Rec.709 / Utility - Linear - sRGB"),
    ("sRGB Encoded Rec.709 (sRGB)", "sRGB - Texture (8/16-bit PNG/JPG/TIF 标准)"),
    ("Gamma 2.2 Encoded Rec.709", "Rec.709 Gamma 2.2"),
    ("Camera Rec.709", "Rec.709 Camera (BT.1886)"),
    ("ADX10", "ADX10 (DPX / Cineon 10-bit 胶片对数)"),
    ("ARRI LogC3 (EI800)", "ARRI LogC3 (EI800 / Alexa)"),
    ("ARRI LogC4", "ARRI LogC4 (Alexa 35)"),
    ("Log3G10 REDWideGamutRGB", "RED Log3G10 / REDWideGamutRGB"),
    ("S-Log3 S-Gamut3.Cine", "Sony S-Log3 / S-Gamut3.Cine"),
    ("Raw", "Raw (不做色彩空间转换)"),
]

OCIO_VIEW_MODES: list[tuple[str, str]] = [
    ("aces_sdr", "ACES 1.0 SDR Video (实拍 HDR 影调映射 · 推荐)"),
    ("un_tone_mapped", "sRGB 直转 (Un-tone-mapped 线性逆变换)"),
]

SRGB_CS = "sRGB Encoded Rec.709 (sRGB)"
SRGB_DISPLAY = "sRGB - Display"
ACES_SDR_VIEW = "ACES 1.0 - SDR Video"


@dataclass(frozen=True)
class OcioSettings:
    input_colorspace: str = "auto"  # "auto" or any OCIO colorspace name
    ocio_view: str = "aces_sdr"  # "aces_sdr" | "un_tone_mapped"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> OcioSettings:
        if not data:
            return cls()
        return cls(
            input_colorspace=str(data.get("input_colorspace") or "auto"),
            ocio_view=str(data.get("ocio_view") or "aces_sdr"),
        )


@lru_cache(maxsize=2)
def get_ocio_config() -> OCIO.Config:
    env_ocio = os.environ.get("OCIO", "").strip()
    if env_ocio and Path(env_ocio).is_file():
        return OCIO.Config.CreateFromFile(env_ocio)
    return OCIO.Config.CreateFromBuiltinConfig(BUILTIN_ACES_CONFIG)


def resolve_input_colorspace(requested: str, ext: str, is_float: bool) -> str:
    """Resolve `"auto"` to the studio-standard colorspace for the sequence format."""
    req = (requested or "auto").strip()
    if req and req.lower() != "auto":
        return req
    ext_l = ext.lower()
    if ext_l == ".exr":
        return "ACEScg"
    if ext_l in (".dpx", ".cin"):
        return "ADX10"
    if ext_l == ".hdr":
        return "Linear Rec.709 (sRGB)"
    if is_float:
        return "ACEScg"
    return SRGB_CS


def is_scene_linear_or_log(colorspace: str) -> bool:
    return colorspace not in (
        SRGB_CS,
        "Gamma 2.2 Encoded Rec.709",
        "Gamma 1.8 Encoded Rec.709",
        "Camera Rec.709",
        "Raw",
    )


@lru_cache(maxsize=32)
def _cpu_processor_to_srgb(src_colorspace: str, ocio_view: str) -> OCIO.CPUProcessor:
    cfg = get_ocio_config()
    if src_colorspace in ("Raw", SRGB_CS):
        return cfg.getProcessor(SRGB_CS, SRGB_CS).getDefaultCPUProcessor()
    if ocio_view == "aces_sdr" and is_scene_linear_or_log(src_colorspace):
        dvt = OCIO.DisplayViewTransform(src_colorspace, SRGB_DISPLAY, ACES_SDR_VIEW)
        return cfg.getProcessor(dvt).getDefaultCPUProcessor()
    return cfg.getProcessor(src_colorspace, SRGB_CS).getDefaultCPUProcessor()


@lru_cache(maxsize=16)
def _cpu_processor_to_acescg(src_colorspace: str) -> OCIO.CPUProcessor:
    cfg = get_ocio_config()
    return cfg.getProcessor(src_colorspace, "ACEScg").getDefaultCPUProcessor()


@lru_cache(maxsize=8)
def srgb_to_acescg_processor() -> OCIO.CPUProcessor:
    cfg = get_ocio_config()
    return cfg.getProcessor(SRGB_CS, "ACEScg").getDefaultCPUProcessor()


def convert_frame_to_srgb_u8(
    pixels: np.ndarray,
    resolved_colorspace: str,
    ocio_view: str = "aces_sdr",
    grade: GradeParams | None = None,
) -> np.ndarray:
    """Convert raw image pixels (`uint8`, `uint16`, or `float32`) in `resolved_colorspace` to 8-bit `sRGB` (`uint8`)
    ready for SAM 3D Body, ViTDet-H, or WebGL backplate preview.

    Color-science order of operations:
      - Fast path: if `resolved_colorspace` is already `sRGB` / `Raw` and `pixels` is `uint8`, applies `GradeParams` LUT.
      - OCIO path (e.g. `.exr` `ACEScg`, `.dpx` `ADX10`, `ARRI LogC`, etc.):
          1. Normalize integer buffers (`uint8`/`uint16`) to `[0..1]` `float32`, keep float buffers (`EXR`) unclipped!
          2. If `grade.exposure != 0` on a scene-linear/log source: convert to `ACEScg` (if log), multiply by `2^EV`
             in scene-linear `ACEScg` BEFORE display tone-mapping (preserving >1.0 EXR highlights cleanly, matching Nuke!).
          3. Apply OCIO transform (`ACEScg -> sRGB Display / ACES 1.0 SDR Video` or `Un-tone-mapped sRGB`).
          4. Apply post-display `brightness` & `gamma` in `float32`, then quantize to `uint8` `[0..255]`.
    """
    g = (grade or GradeParams()).clamped()
    arr = np.asarray(pixels)

    if resolved_colorspace in (SRGB_CS, "Raw"):
        return apply_grade_rgb(arr, g)

    # Convert to contiguous RGB float32
    if arr.dtype == np.uint8:
        buf = np.ascontiguousarray(arr, dtype=np.float32) * np.float32(1.0 / 255.0)
    elif arr.dtype == np.uint16:
        buf = np.ascontiguousarray(arr, dtype=np.float32) * np.float32(1.0 / 65535.0)
    else:
        buf = np.array(arr, dtype=np.float32, copy=True, order="C")

    # Replace NaN / Inf in EXR renders so OCIO LUTs never produce black/NaN artifacts
    if not np.all(np.isfinite(buf)):
        buf = np.nan_to_num(buf, nan=0.0, posinf=65504.0, neginf=0.0)

    src_cs = resolved_colorspace
    if abs(g.exposure) >= 1e-6 and is_scene_linear_or_log(src_cs):
        if src_cs != "ACEScg":
            _cpu_processor_to_acescg(src_cs).applyRGB(buf)
            src_cs = "ACEScg"
        buf *= np.float32(2.0 ** g.exposure)
        rem_grade = GradeParams(
            exposure=0.0,
            brightness=g.brightness,
            contrast=g.contrast,
            saturation=g.saturation,
            gamma=g.gamma,
        )
    else:
        rem_grade = g

    proc = _cpu_processor_to_srgb(src_cs, ocio_view)
    proc.applyRGB(buf)
    buf = np.clip(buf, 0.0, 1.0)

    if not rem_grade.is_identity:
        return apply_grade_rgb(buf, rem_grade)
    return np.clip(np.rint(buf * 255.0), 0.0, 255.0).astype(np.uint8)


def list_ocio_options() -> dict:
    return {
        "config_name": os.environ.get("OCIO") or BUILTIN_ACES_CONFIG,
        "colorspaces": [{"id": cid, "label": label} for cid, label in STUDIO_INPUT_COLORSPACES],
        "views": [{"id": vid, "label": label} for vid, label in OCIO_VIEW_MODES],
    }
