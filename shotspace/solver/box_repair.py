"""A tracked person's outline that suddenly reaches far beyond the person for a frame or two (v0.2.12).

What the user saw on company footage: "检测框突然猛地变长就是下边缘会猛地向下几倍框大小，导致解算人上下猛地抖动一下" -
nobody else below, the area below already outside the picture.  The box of a frame is the box around the tracker's
outline (`people_tracker.TrackStore.add`); now and then the outline of one frame takes in a piece of something else
(a patch of ground, a shadow, the edge of the picture), and the box reaches several box heights further.  The body
network is then given that box and that outline: on the frames measured here it put the person at a fraction of
their distance (25-36 m on the frames around, 4.7 m on the stretched frame), and the whole-shot solve is pulled
towards it - the jerk.

The rule, on the boxes alone: a frame whose box is more than `reach` box heights LARGER than the box the frames
around it agree on (the median of up to `window` frames on each side, the frame itself left out), and reaches that
far beyond it, is a stretched frame.  A box that only moves (the track on a neighbour for a frame) is not one.  A
real change of the box is not one frame long either - a person who comes out from behind something keeps the new
box, and the median follows within a few frames.

For a stretched frame the body network gets the outline without what lies away from the person (`clean_outline`):
every separate piece of the outline that does not reach into the region the frames around agree on (their box
grown by `margin` box heights) is dropped - on the frames looked at here that piece was a speck on a building, at
the edge of the picture, on another person - and only if the outline still reaches beyond that region it is cut
there.  The person's box on that frame becomes the agreed box.  Nothing else changes: the other frames, and every
person without such a frame, are exactly as the tracker gave them.  numpy (and OpenCV for the outline, in the
worker)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepairConfig:
    window: int = 5          # frames on each side that say where the box usually is
    max_gap: int = 6         # ...at most this many frames away
    min_neighbours: int = 4  # fewer than this: the frame is not judged
    reach: float = 0.5       # a side this many box heights beyond the agreed box (and the box that much larger): stretched
    margin: float = 0.3      # the outline of a stretched frame is cut to the agreed box grown by this many box heights


def stretched_frames(boxes: dict, image_size: tuple[int, int] | None = None, cfg: RepairConfig | None = None) -> dict[int, dict]:
    """Stretched frames of one tracked person -> {frame: {"box": agreed box, "region": where the outline is cut to}}.

    `boxes` maps frame (int or str) to [x1, y1, x2, y2] in plate pixels."""
    cfg = cfg or RepairConfig()
    frames = sorted(int(f) for f in boxes)
    if len(frames) < cfg.min_neighbours + 1:
        return {}
    arr = np.asarray([np.asarray(boxes[f] if f in boxes else boxes[str(f)], dtype=np.float64) for f in frames])
    frame_arr = np.asarray(frames, dtype=np.int64)
    out: dict[int, dict] = {}
    for i, f in enumerate(frames):
        lo, hi = max(0, i - cfg.window), min(len(frames), i + cfg.window + 1)
        near = [j for j in range(lo, hi) if j != i and abs(int(frame_arr[j]) - f) <= cfg.max_gap]
        if len(near) < cfg.min_neighbours:
            continue
        agreed = np.median(arr[near], axis=0)
        h = float(agreed[3] - agreed[1])
        if h <= 1.0:
            continue
        b = arr[i]
        beyond = max(agreed[0] - b[0], b[2] - agreed[2], agreed[1] - b[1], b[3] - agreed[3]) / h
        grown = max((b[3] - b[1]) - h, (b[2] - b[0]) - (agreed[2] - agreed[0])) / h
        if beyond <= cfg.reach or grown <= cfg.reach:
            continue
        region = np.array([agreed[0] - cfg.margin * h, agreed[1] - cfg.margin * h, agreed[2] + cfg.margin * h, agreed[3] + cfg.margin * h])
        if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
            region = np.clip(region, 0.0, [image_size[0], image_size[1], image_size[0], image_size[1]])
        out[f] = {
            "box": [round(float(v), 2) for v in agreed],
            "region": [round(float(v), 2) for v in region],
        }
    return out


def with_repairs(person: dict, image_size: tuple[int, int] | None = None, cfg: RepairConfig | None = None) -> dict:
    """The tracked person with their stretched frames repaired: on those frames `boxes` holds the agreed box and
    `repaired` (frame -> region) tells the body network where to cut the outline.  Computed from the tracker's own
    boxes (`raw_boxes`), so it gives the same answer every time.  A person that carries `repaired` already (kept
    with the task's record) is returned as it is, so is a manually boxed person, and so is - the very same object -
    a person without a stretched frame."""
    if "repaired" in person or person.get("is_manual"):
        return person
    raw = person.get("raw_boxes") or person.get("boxes") or {}
    if not isinstance(raw, dict) or not raw:
        return person
    found = stretched_frames(raw, image_size, cfg)
    if not found:
        return person
    out = dict(person)
    boxes = dict(person.get("boxes") or {})
    for f, fix in found.items():
        boxes[str(f)] = fix["box"]
    out["boxes"] = boxes
    out["repaired"] = {str(f): fix["region"] for f, fix in sorted(found.items())}
    return out


def clean_outline(mask: np.ndarray, region, rect=None) -> tuple[np.ndarray, np.ndarray] | None:
    """The outline of a stretched frame without what lies away from the person -> (outline, its box), or None when
    nothing of it lies in `region` (the frame is then left as it was).

    Every separate piece (8-connected) that does not reach into `region` is dropped; whatever of the rest still lies
    outside `region` is cut away.  `rect` = (x1, y1, x2, y2): the outline is non-zero only there (speed)."""
    import cv2

    hh, ww = mask.shape[:2]
    rx1, ry1, rx2, ry2 = (0, 0, ww, hh) if rect is None else (max(0, int(rect[0])), max(0, int(rect[1])), min(ww, int(rect[2])), min(hh, int(rect[3])))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    sub = (mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
    gx1, gy1 = int(np.floor(region[0])) - rx1, int(np.floor(region[1])) - ry1
    gx2, gy2 = int(np.ceil(region[2])) - rx1, int(np.ceil(region[3])) - ry1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for k in range(1, n):
        x, y, w, h = (int(v) for v in stats[k, :4])
        if x < gx2 and x + w > gx1 and y < gy2 and y + h > gy1:
            piece = lab[max(0, gy1):max(0, gy2), max(0, gx1):max(0, gx2)] == k
            keep[k] = bool(piece.any())
    kept = keep[lab]
    # whatever of the kept pieces still lies outside the region is cut away
    inside = np.zeros_like(kept)
    inside[max(0, gy1):max(0, gy2), max(0, gx1):max(0, gx2)] = True
    kept &= inside
    ys, xs = np.nonzero(kept)
    if len(ys) < 16:
        return None
    out = np.zeros_like(mask)
    out[ry1:ry2, rx1:rx2] = kept.astype(mask.dtype)
    box = np.array([rx1 + xs.min(), ry1 + ys.min(), rx1 + xs.max() + 1, ry1 + ys.max() + 1], dtype=np.float64)
    return out, box
