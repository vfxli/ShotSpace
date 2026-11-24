"""Rigid 3D Object representation (`RigidObjectTrack`) for ShotSpace Object Matching Mode.

Stores a single static local proxy mesh centered at `(0, 0, 0)` (in `cm`, `+Y` up) together with
per-frame rigid `SE(3)` transforms (`world_transforms` `[F, 4, 4]`, `scale = (1, 1, 1)`) and
per-frame `visible` (`1` = visible, `0` = hidden/frozen), enabling 1-click asset replacement in Maya / Houdini / Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RigidObjectTrack:
    name: str
    object_id: int
    frames: list[int]
    world_transforms: np.ndarray  # [F, 4, 4] float64 column-vector SE(3) transforms (cm, +Y up)
    rest_points: np.ndarray  # [V, 3] float32 static local proxy mesh vertices centered at (0, 0, 0) (cm, +Y up)
    faces: np.ndarray  # [T, 3] int32 triangle indices
    vertex_colors: np.ndarray | None = None  # [V, 3] float32 RGB in [0, 1]
    visible: np.ndarray | None = None  # [F] int32: 1 on valid tracked frames, 0 on hidden/frozen frames
    anchor_frames: np.ndarray | None = None  # [F] int64: valid anchor frame held on hidden frames
    mask_boxes: np.ndarray | None = None  # [F, 4] float32: 2D SAM mask [x1, y1, x2, y2] in px (-1 when absent)
    source_focal_px: float = 0.0
    custom_data: dict[str, Any] = field(default_factory=dict)

    @property
    def visible_mask(self) -> np.ndarray:
        if self.visible is None or len(self.visible) != len(self.frames):
            return np.ones(len(self.frames), dtype=bool)
        return np.asarray(self.visible, dtype=np.int32).reshape(-1) > 0

    @property
    def solved_frames(self) -> list[int]:
        mask = self.visible_mask
        return [int(f) for f, ok in zip(self.frames, mask) if ok]

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "name": np.array(str(self.name)),
            "object_id": np.int64(self.object_id),
            "frames": np.asarray(self.frames, dtype=np.int64),
            "world_transforms": np.asarray(self.world_transforms, dtype=np.float64),
            "rest_points": np.asarray(self.rest_points, dtype=np.float32),
            "faces": np.asarray(self.faces, dtype=np.int32),
            "visible": self.visible_mask.astype(np.int32),
            "source_focal_px": np.float64(self.source_focal_px),
        }
        if self.vertex_colors is not None:
            out["vertex_colors"] = np.asarray(self.vertex_colors, dtype=np.float32)
        if self.anchor_frames is not None:
            out["anchor_frames"] = np.asarray(self.anchor_frames, dtype=np.int64)
        if self.mask_boxes is not None:
            out["mask_boxes"] = np.asarray(self.mask_boxes, dtype=np.float32)
        return out

    @classmethod
    def from_npz(cls, object_id: int, data: Any) -> "RigidObjectTrack":
        frames = [int(f) for f in np.asarray(data["frames"]).reshape(-1)]
        vis = (
            np.asarray(data["visible"], dtype=np.int32).reshape(-1)
            if "visible" in data
            else np.ones(len(frames), dtype=np.int32)
        )
        anc = (
            np.asarray(data["anchor_frames"], dtype=np.int64).reshape(-1)
            if "anchor_frames" in data
            else None
        )
        vcols = (
            np.asarray(data["vertex_colors"], dtype=np.float32)
            if "vertex_colors" in data
            else None
        )
        mboxes = (
            np.asarray(data["mask_boxes"], dtype=np.float32)
            if "mask_boxes" in data
            else None
        )
        raw_name = (
            str(np.asarray(data["name"]).item())
            if "name" in data
            else f"object_{int(object_id):02d}"
        )
        f_arr = np.asarray(data["source_focal_px"]).reshape(-1) if "source_focal_px" in data else np.array([0.0])
        f_px = float(f_arr[0]) if f_arr.size > 0 else 0.0
        return cls(
            name=raw_name,
            object_id=int(object_id),
            frames=frames,
            world_transforms=np.asarray(data["world_transforms"], dtype=np.float64),
            rest_points=np.asarray(data["rest_points"], dtype=np.float32),
            faces=np.asarray(data["faces"], dtype=np.int32),
            vertex_colors=vcols,
            visible=vis,
            anchor_frames=anc,
            mask_boxes=mboxes,
            source_focal_px=f_px,
            custom_data={"shotspace:object_id": int(object_id)},
        )
