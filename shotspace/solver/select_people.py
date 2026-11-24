"""Person selection rules (`largest`, `top`, `all`, `ids`, `picked`) and ghost-track filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SelectionMode = Literal["largest", "top", "all", "ids", "picked"]
ManualKeyframeStatus = Literal["active", "occluded", "off"]


def normalize_manual_people(raw_list: list[dict] | None) -> list[dict]:
    """Sanitize, deduplicate, and sort `manual_people` tracks & their per-frame stateful keyframes."""
    if not raw_list or not isinstance(raw_list, list):
        return []
    out: list[dict] = []
    seen_ids: set[int] = set()
    next_fallback_id = 101
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("id") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid <= 0 or pid in seen_ids:
            while next_fallback_id in seen_ids:
                next_fallback_id += 1
            pid = next_fallback_id
        seen_ids.add(pid)
        next_fallback_id = max(next_fallback_id, pid + 1)

        name = str(item.get("name") or f"手动 #{pid}").strip() or f"手动 #{pid}"
        enabled = bool(item.get("enabled", True))
        kf_by_frame: dict[int, dict] = {}
        for kf in item.get("keyframes") or []:
            if not isinstance(kf, dict):
                continue
            try:
                fnum = int(round(float(kf.get("frame"))))
            except (TypeError, ValueError):
                continue
            st = str(kf.get("status") or "active").strip().lower()
            if st not in ("active", "occluded", "off"):
                st = "active"
            if st == "active":
                raw_box = kf.get("box")
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                except (TypeError, ValueError):
                    continue
                lx1, lx2 = min(x1, x2), max(x1, x2)
                ly1, ly2 = min(y1, y2), max(y1, y2)
                if (lx2 - lx1) < 4.0 or (ly2 - ly1) < 6.0:
                    continue
                kf_by_frame[fnum] = {
                    "frame": fnum,
                    "status": "active",
                    "box": [round(lx1, 2), round(ly1, 2), round(lx2, 2), round(ly2, 2)],
                }
            else:
                kf_by_frame[fnum] = {"frame": fnum, "status": st}

        out.append(
            {
                "id": pid,
                "name": name,
                "enabled": enabled,
                "keyframes": [kf_by_frame[f] for f in sorted(kf_by_frame)],
            }
        )
    return out


def evaluate_manual_person_boxes(
    track: dict,
    shot_frames: list[int],
) -> tuple[dict[str, list[float]], dict[int, str]]:
    """Evaluate a stateful manual person track across `shot_frames`:
      - `"active"` keyframes linearly interpolate `[x1, y1, x2, y2]` to the next `"active"` keyframe,
        or hold `[x1, y1, x2, y2]` up to an `"occluded"` / `"off"` keyframe (or end of shot);
      - `"occluded"` suppresses 2D boxes starting at its frame until a later `"active"` keyframe re-activates the same ID;
      - `"off"` terminates 2D boxes starting at its frame onward (unless re-activated by a later `"active"` keyframe);
      - Frames before the first keyframe are `"inactive"` (no box).
    """
    norm_list = normalize_manual_people([track])
    if not norm_list:
        return {}, {int(f): "inactive" for f in shot_frames}
    norm = norm_list[0]
    kfs = norm["keyframes"]
    if not norm.get("enabled", True) or not kfs:
        return {}, {int(f): "inactive" for f in shot_frames}

    boxes_out: dict[str, list[float]] = {}
    statuses_out: dict[int, str] = {}
    kf_frames = [int(k["frame"]) for k in kfs]
    m = len(kfs)

    # bisect-free fast scan over sorted shot_frames
    idx = -1
    for raw_f in shot_frames:
        f = int(raw_f)
        while (idx + 1) < m and kf_frames[idx + 1] <= f:
            idx += 1
        if idx < 0:
            statuses_out[f] = "inactive"
            continue
        cur_kf = kfs[idx]
        st = cur_kf["status"]
        if st == "occluded":
            statuses_out[f] = "occluded"
            continue
        if st == "off":
            statuses_out[f] = "off"
            continue

        b0 = cur_kf["box"]
        if f == cur_kf["frame"]:
            boxes_out[str(f)] = [round(float(v), 2) for v in b0]
            statuses_out[f] = "key_active"
        elif idx == m - 1:
            boxes_out[str(f)] = [round(float(v), 2) for v in b0]
            statuses_out[f] = "hold_active"
        else:
            nxt_kf = kfs[idx + 1]
            if nxt_kf["status"] == "active" and "box" in nxt_kf:
                b1 = nxt_kf["box"]
                span = max(1, int(nxt_kf["frame"]) - int(cur_kf["frame"]))
                alpha = float(f - int(cur_kf["frame"])) / float(span)
                boxes_out[str(f)] = [
                    round((1.0 - alpha) * float(b0[c]) + alpha * float(b1[c]), 2) for c in range(4)
                ]
                statuses_out[f] = "interp_active"
            else:
                boxes_out[str(f)] = [round(float(v), 2) for v in b0]
                statuses_out[f] = "hold_active"

    return boxes_out, statuses_out


def build_manual_detected_people(manual_people: list[dict] | None, shot_frames: list[int]) -> list[dict]:
    """Convert enabled `manual_people` tracks with >=1 active frame into `DetectedPerson`-compatible dicts."""
    if not manual_people or not shot_frames:
        return []
    out: list[dict] = []
    for trk in normalize_manual_people(manual_people):
        if not trk.get("enabled", True):
            continue
        boxes, _ = evaluate_manual_person_boxes(trk, shot_frames)
        if not boxes:
            continue
        prom = float(sum(max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes.values()))
        out.append(
            {
                "id": int(trk["id"]),
                "name": str(trk.get("name") or f"手动 #{trk['id']}"),
                "is_manual": True,
                "prominence": round(prom, 1),
                "frame_count": len(boxes),
                "boxes": boxes,
                "scores": {fk: 0.99 for fk in boxes},
                # the frames the artist really drew a box on (the rest are interpolated)
                "key_frames": sorted(
                    int(k["frame"]) for k in trk.get("keyframes") or []
                    if k.get("status") == "active" and str(int(k["frame"])) in boxes
                ),
            }
        )
    return out


@dataclass(frozen=True)
class SelectionSpec:
    mode: SelectionMode = "largest"
    count: int = 3
    ids: str = ""  # e.g. "1, 3-5"
    picks: list[list[float]] | None = None  # [[frame, x_px, y_px], ...]
    least_frames: int = 1  # Ghost track filter: ignore tracks shorter than least_frames


def parse_person_ids(spec: str) -> set[int]:
    out: set[int] = set()
    for part in re.split(r"[,;\s]+", (spec or "").strip()):
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                out.update(range(min(lo, hi), max(lo, hi) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def select_people(people: list[dict], spec: SelectionSpec) -> tuple[list[dict], list[dict]]:
    """Filter `people` (`[{"id": 1, "prominence": ..., "boxes": {"1001": [x1,y1,x2,y2], ...}}, ...]`)
    according to `spec`, preserving prominence order.

    Returns `(chosen_people, missed_picks)`.
    """
    min_f = max(1, int(spec.least_frames or 1))
    auto_people = [p for p in people if not bool(p.get("is_manual", False))]
    manual_valid = [p for p in people if bool(p.get("is_manual", False)) and len(p.get("boxes") or {}) >= 1]

    auto_valid = [p for p in auto_people if len(p.get("boxes") or {}) >= min_f]
    if not auto_valid and auto_people:
        # If every track is shorter than least_frames (e.g. a 1-frame plate), keep non-empty tracks
        auto_valid = [p for p in auto_people if len(p.get("boxes") or {}) >= 1]

    valid = auto_valid + manual_valid
    missed: list[dict] = []
    mode = spec.mode

    if mode == "largest":
        return auto_valid[:1] + manual_valid, missed
    if mode == "top":
        return auto_valid[: max(1, int(spec.count))] + manual_valid, missed
    if mode == "all":
        return list(valid), missed
    if mode == "ids":
        wanted = parse_person_ids(spec.ids)
        return [p for p in valid if int(p["id"]) in wanted], missed
    if mode == "picked":
        if not (spec.picks or []) and manual_valid:
            return list(manual_valid), missed
        picked_ids: set[int] = set()
        for pk in spec.picks or []:
            if len(pk) < 3:
                continue
            f_str = str(int(round(pk[0])))
            px, py = float(pk[1]), float(pk[2])
            hits: list[tuple[float, int]] = []
            for p in valid:
                box = (p.get("boxes") or {}).get(f_str)
                if not box:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                if x1 <= px <= x2 and y1 <= py <= y2:
                    area = max((x2 - x1) * (y2 - y1), 0.0)
                    hits.append((area, int(p["id"])))
            if hits:
                # Choose smallest containing box so foreground/occluded clicks resolve accurately
                hits.sort(key=lambda pair: pair[0])
                picked_ids.add(hits[0][1])
            else:
                missed.append({"frame": int(round(pk[0])), "x": px, "y": py})
        return [p for p in valid if int(p["id"]) in picked_ids], missed

    raise ValueError(f"未知选人模式：{mode!r}")
