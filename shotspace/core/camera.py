"""CameraTrack representation, per-frame lens intrinsics extraction, and overscan/shift handling."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace

import numpy as np

from .poses import interpolate_poses
from .units import DEFAULT_FPS, FILMBACK_MM, focal_px, rigid, standard_fps


@dataclass(frozen=True)
class CameraTrack:
    """Unified camera representation in ShotSpace's internal convention:
    cm, +Y up, right-handed, looking down -Z, rigid SE(3) column-vector [F, 4, 4] matrices.
    """

    name: str
    path: str
    frames: tuple[int, ...]
    cam_to_world: np.ndarray  # [F, 4, 4] rigid float64
    focal_mm: np.ndarray  # [F] float64
    h_aperture_mm: np.ndarray  # [F] float64
    v_aperture_mm: np.ndarray  # [F] float64
    center_mm: np.ndarray = field(default_factory=lambda: np.zeros((1, 2), dtype=np.float64))  # [F, 2] (+x right, +y up)
    width: int = 1920
    height: int = 1080
    pixel_aspect: float = 1.0
    overscan: tuple[int, int, int, int] = (0, 0, 0, 0)  # pixels (left, top, right, bottom) around width x height
    fps: float = DEFAULT_FPS
    plate: str = ""
    properties: dict = field(default_factory=dict)
    source_file: str = ""
    own_frames: tuple[int, ...] | None = None  # Genuinely authored sample frames before resampling/holding

    def __post_init__(self) -> None:
        frames_t = tuple(int(f) for f in self.frames)
        n = max(1, len(frames_t))
        c2w = np.asarray(self.cam_to_world, dtype=np.float64)
        if c2w.ndim == 2:
            c2w = np.repeat(c2w[None], n, axis=0)
        c2w = rigid(c2w)

        def _per_frame_1d(arr, fallback: float) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            if len(a) == 0:
                return np.full(n, fallback, dtype=np.float64)
            if len(a) == 1:
                return np.full(n, float(a[0]), dtype=np.float64)
            if len(a) != n:
                raise ValueError(f"Camera lens array length {len(a)} does not match frame count {n}")
            return a

        c_mm = np.asarray(self.center_mm, dtype=np.float64).reshape(-1, 2)
        if len(c_mm) == 0:
            c_mm = np.zeros((n, 2), dtype=np.float64)
        elif len(c_mm) == 1:
            c_mm = np.repeat(c_mm, n, axis=0)
        elif len(c_mm) != n:
            raise ValueError(f"Camera center_mm length {len(c_mm)} does not match frame count {n}")

        object.__setattr__(self, "frames", frames_t)
        object.__setattr__(self, "cam_to_world", c2w)
        object.__setattr__(self, "focal_mm", _per_frame_1d(self.focal_mm, 50.0))
        object.__setattr__(self, "h_aperture_mm", _per_frame_1d(self.h_aperture_mm, FILMBACK_MM))
        object.__setattr__(self, "v_aperture_mm", _per_frame_1d(self.v_aperture_mm, FILMBACK_MM * 9.0 / 16.0))
        object.__setattr__(self, "center_mm", c_mm)
        object.__setattr__(self, "overscan", tuple(int(v) for v in self.overscan))
        object.__setattr__(self, "fps", standard_fps(self.fps))
        if self.own_frames is None:
            object.__setattr__(self, "own_frames", frames_t)
        else:
            object.__setattr__(self, "own_frames", tuple(int(f) for f in self.own_frames))

    @property
    def authored_frames(self) -> tuple[int, ...]:
        return self.own_frames if self.own_frames is not None else self.frames

    def with_frame_offset(self, offset: int) -> CameraTrack:
        """Shift camera sample frame numbers by `offset` (e.g. +1000 to align 1..150 camera to 1001..1150 plate)."""
        if offset == 0:
            return self
        return replace(
            self,
            frames=tuple(f + offset for f in self.frames),
            own_frames=tuple(f + offset for f in self.authored_frames),
        )

    def resample_to(self, target_frames: list[int] | tuple[int, ...]) -> CameraTrack:
        """Evaluate camera poses and lens channels onto `target_frames` (interpolating between keys, holding ends),
        while preserving `own_frames` so whole-shot alignment knows which frames had genuine samples."""
        want = [int(f) for f in target_frames]
        if not want or tuple(want) == self.frames:
            return self
        if not self.frames:
            return replace(
                self,
                frames=tuple(want),
                cam_to_world=np.repeat(self.cam_to_world[:1], len(want), axis=0),
                focal_mm=np.repeat(self.focal_mm[:1], len(want)),
                h_aperture_mm=np.repeat(self.h_aperture_mm[:1], len(want)),
                v_aperture_mm=np.repeat(self.v_aperture_mm[:1], len(want)),
                center_mm=np.repeat(self.center_mm[:1], len(want), axis=0),
            )

        known_poses = {f: self.cam_to_world[i] for i, f in enumerate(self.frames)}
        new_poses = interpolate_poses(known_poses, want)

        keys = list(self.frames)

        def _interp_channel(vals: np.ndarray) -> np.ndarray:
            out = []
            for f in want:
                j = bisect.bisect_left(keys, f)
                if j < len(keys) and keys[j] == f:
                    out.append(vals[j])
                elif j == 0:
                    out.append(vals[0])
                elif j >= len(keys):
                    out.append(vals[-1])
                else:
                    t = (f - keys[j - 1]) / (keys[j] - keys[j - 1])
                    out.append((1.0 - t) * vals[j - 1] + t * vals[j])
            return np.asarray(out, dtype=np.float64)

        return replace(
            self,
            frames=tuple(want),
            cam_to_world=rigid(new_poses),
            focal_mm=_interp_channel(self.focal_mm),
            h_aperture_mm=_interp_channel(self.h_aperture_mm),
            v_aperture_mm=_interp_channel(self.v_aperture_mm),
            center_mm=_interp_channel(self.center_mm),
            own_frames=self.authored_frames,
        )

    def intrinsics_per_frame(self, canvas_width: int, canvas_height: int) -> dict[int, np.ndarray]:
        """Compute exact [3, 3] pixel intrinsics K(f) for SAM 3D Body for each frame in `self.frames`.
        Accounts for per-frame zoom (`focal_mm` / `h_aperture_mm`), Film Offset (`center_mm`), and `overscan`.
        """
        left, top, right, bottom = self.overscan
        if any(self.overscan) and self.width > 0 and self.height > 0:
            # Scale overscan proportionally if the plate was resized relative to recorded camera canvas
            expected_cw = self.width + left + right
            expected_ch = self.height + top + bottom
            sx = canvas_width / expected_cw if expected_cw > 0 else 1.0
            sy = canvas_height / expected_ch if expected_ch > 0 else 1.0
            plate_w = self.width * sx
            plate_h = self.height * sy
            cx0 = left * sx + plate_w * 0.5
            cy0 = top * sy + plate_h * 0.5
        else:
            plate_w = float(canvas_width)
            plate_h = float(canvas_height)
            cx0 = plate_w * 0.5
            cy0 = plate_h * 0.5

        f_px = focal_px(self.focal_mm, self.h_aperture_mm, plate_w)
        per_mm = plate_w / np.maximum(self.h_aperture_mm, 1e-6)
        cx = cx0 + self.center_mm[:, 0] * per_mm
        cy = cy0 - self.center_mm[:, 1] * per_mm * float(self.pixel_aspect)

        out: dict[int, np.ndarray] = {}
        for i, f in enumerate(self.frames):
            fx = float(f_px[i])
            k = np.array(
                [
                    [fx, 0.0, float(cx[i])],
                    [0.0, fx, float(cy[i])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            out[int(f)] = k
        return out

    def summary(self) -> dict:
        f_min, f_max = float(np.min(self.focal_mm)), float(np.max(self.focal_mm))
        pos = np.asarray(self.cam_to_world[:, :3, 3], dtype=np.float64)
        travel_cm = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
        span_cm = [round(float(v), 2) for v in (pos.max(axis=0) - pos.min(axis=0))] if len(pos) else [0.0, 0.0, 0.0]
        return {
            "name": self.name,
            "path": self.path,
            "frames": [int(self.frames[0]), int(self.frames[-1])] if self.frames else [],
            "frame_count": len(self.frames),
            "authored_count": len(self.authored_frames),
            "focal_mm": round(f_min, 3) if abs(f_max - f_min) < 1e-3 else [round(f_min, 3), round(f_max, 3)],
            "is_zoom": bool(abs(f_max - f_min) >= 1e-3),
            "h_aperture_mm": round(float(np.median(self.h_aperture_mm)), 3),
            "v_aperture_mm": round(float(np.median(self.v_aperture_mm)), 3),
            "center_mm": [round(float(v), 4) for v in np.median(self.center_mm, axis=0)],
            "pixel_aspect": round(float(self.pixel_aspect), 4),
            "width": int(self.width),
            "height": int(self.height),
            "fps": round(float(self.fps), 3),
            "travel_cm": round(travel_cm, 2),
            "span_cm": span_cm,
            "source_file": self.source_file,
        }
