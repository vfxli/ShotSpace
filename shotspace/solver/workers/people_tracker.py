"""Whole-shot people tracking on SAM 3.1's native video concept tracking (text prompt ``"person"``).

Runs inside the SAM 3.1 / 3D-objects worker venv (``third_party/sam_3d_objects/.venv``).

Design:
  * Identity is decided by SAM 3.1's own memory-based multi-object tracker (Object Multiplex): the model detects every
    person on every frame, associates detections with its masklets and keeps one id per person for the whole shot.
    People entering late are picked up by the per-frame detector when they appear.
  * The primary output is one binary mask per person per frame. Boxes are the tight bounds of the visible mask;
    a person who is not visible on a frame has NO entry on that frame (nothing is extrapolated or smoothed).
  * No hand-written box association: the only post-processing is bookkeeping on the masks SAM 3.1 produced
    (dropping empty / never-confirmed masklets and ordering people by prominence).
"""

from __future__ import annotations

import json
import uuid
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from shotspace.core.platform_compat import link_file

ENGINE_TAG = "sam31_native_people_v1"
# entry of `person_masks_sam3.npz` holding the id of the run that wrote it (the same id is in `boxes.json`)
RUN_ID_KEY = "__run_id__"
PROMPT_TEXT = "person"

# SAM 3.1 runtime knobs used for people tracking (attribute name on `predictor.model` -> value).
# Values that differ from the upstream demo defaults are justified by measurements in `eval/track_*`.
PEOPLE_KNOBS: dict[str, Any] = {
    # the builder used for object mode caps tracking at 12 objects; a crowd needs far more
    "max_num_objects": 96,
    # upstream runs the detector on 16 frames at once, which overflows a 24 GB card (measured: 24.4 GB reserved
    # -> "CUDA driver error: device not ready" under WSL). 4 frames peak at ~11-12.5 GB with 25-50 people.
    "batched_grounding_batch_size": 4,
}


# Mask hygiene (not an identity decision): SAM 3.1 runs with its sprinkle removal disabled, so more than half of
# the returned masks carry a few stray pixels far away from the person, which would blow the tight box up to a
# large part of the frame. Connected components smaller than `SPECK_REL` of the person's largest visible part are
# specks and are dropped; genuinely split silhouettes (a body cut in two by an occluder) keep every sizeable part.
SPECK_REL = 0.02
SPECK_MIN_PX = 12


def remove_specks(mask: np.ndarray, rel: float = SPECK_REL, min_px: int = SPECK_MIN_PX) -> np.ndarray:
    """Drop connected components smaller than `max(min_px, rel * largest component)`; returns a bool mask."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = areas >= max(float(min_px), rel * float(areas.max()))
    if keep.all():
        return mask
    lut = np.zeros(n + 1, dtype=bool)
    lut[1:] = keep
    return lut[lab]


class TrackStore:
    """Compact per-(object, frame) store: tight box, pixel area, score and a bit-packed mask crop."""

    def __init__(self) -> None:
        self.boxes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
        self.areas: dict[int, dict[int, int]] = {}
        self.scores: dict[int, dict[int, float]] = {}
        self.crops: dict[int, dict[int, tuple[np.ndarray, int, int]]] = {}
        # frame -> True when SAM 3.1's per-frame detector saw this object on that frame (matched or newly detected)
        self.confirmed: dict[int, dict[int, bool]] = {}

    def add(self, oid: int, fidx: int, mask: np.ndarray, score: float, clean: bool = True, confirmed: bool = True) -> bool:
        ys = np.flatnonzero(mask.any(axis=1))
        if len(ys) == 0:
            return False
        xs = np.flatnonzero(mask.any(axis=0))
        y1, y2 = int(ys[0]), int(ys[-1]) + 1
        x1, x2 = int(xs[0]), int(xs[-1]) + 1
        crop = mask[y1:y2, x1:x2]
        if clean:
            cleaned = remove_specks(crop)
            if cleaned is not crop:
                ys2 = np.flatnonzero(cleaned.any(axis=1))
                xs2 = np.flatnonzero(cleaned.any(axis=0))
                if len(ys2) == 0:
                    return False
                y1, y2 = y1 + int(ys2[0]), y1 + int(ys2[-1]) + 1
                x1, x2 = x1 + int(xs2[0]), x1 + int(xs2[-1]) + 1
                crop = cleaned[int(ys2[0]) : int(ys2[-1]) + 1, int(xs2[0]) : int(xs2[-1]) + 1]
        self.boxes.setdefault(oid, {})[fidx] = (x1, y1, x2, y2)
        self.areas.setdefault(oid, {})[fidx] = int(crop.sum())
        self.scores.setdefault(oid, {})[fidx] = float(score)
        self.crops.setdefault(oid, {})[fidx] = (np.packbits(crop.reshape(-1)), y2 - y1, x2 - x1)
        self.confirmed.setdefault(oid, {})[fidx] = bool(confirmed)
        return True

    def copy_entry(self, new_oid: int, fidx: int, src: "TrackStore", src_oid: int) -> None:
        self.boxes.setdefault(new_oid, {})[fidx] = src.boxes[src_oid][fidx]
        self.areas.setdefault(new_oid, {})[fidx] = src.areas[src_oid][fidx]
        self.scores.setdefault(new_oid, {})[fidx] = src.scores[src_oid][fidx]
        self.crops.setdefault(new_oid, {})[fidx] = src.crops[src_oid][fidx]
        self.confirmed.setdefault(new_oid, {})[fidx] = src.confirmed.get(src_oid, {}).get(fidx, True)

    def shifted(self, offset: int) -> "TrackStore":
        """Same store with every frame index moved by `offset` (chunk-local -> shot index)."""
        out = TrackStore()
        for name in ("boxes", "areas", "scores", "crops", "confirmed"):
            src = getattr(self, name)
            setattr(out, name, {oid: {f + offset: v for f, v in fmap.items()} for oid, fmap in src.items()})
        return out

    def remove_frame(self, oid: int, fidx: int) -> None:
        for d in (self.boxes, self.areas, self.scores, self.crops, self.confirmed):
            d.get(oid, {}).pop(fidx, None)

    def object_ids(self) -> list[int]:
        return sorted(self.boxes)

    def mask_crop(self, oid: int, fidx: int) -> np.ndarray:
        bits, h, w = self.crops[oid][fidx]
        return np.unpackbits(bits)[: h * w].reshape(h, w).astype(bool)

    def drop(self, oid: int) -> None:
        for d in (self.boxes, self.areas, self.scores, self.crops, self.confirmed):
            d.pop(oid, None)


def patch_cpu_state_merge(pred: Any) -> None:
    """Make SAM 3.1's dense detect-and-track flow work with tracker state stored on the CPU.

    With `offload_state_to_cpu=True` (needed so VRAM does not grow with the number of frames) upstream moves each
    frame's compact tracker output to the CPU, but when the per-frame detector then adds new people (or
    re-conditions masklets) on that same frame it concatenates that CPU output with fresh GPU tensors and raises
    "Expected all tensors to be on the same device". Moving the frame's stored tensors back to the GPU right before
    the merge fixes it; upstream re-offloads them immediately afterwards when it rebuilds the compact output."""
    trk = getattr(getattr(pred.model, "tracker", None), "model", None)
    if trk is None or getattr(trk, "_shotspace_state_merge_patched", False):
        return

    def _wrap(fn: Callable) -> Callable:
        def inner(*args: Any, **kwargs: Any) -> Any:
            prev_out = kwargs.get("prev_output")
            if isinstance(prev_out, dict):
                for key, val in list(prev_out.items()):
                    if torch.is_tensor(val) and val.device.type != "cuda":
                        prev_out[key] = val.to("cuda", non_blocking=True)
            return fn(*args, **kwargs)

        return inner

    for name in ("add_new_masks_to_existing_state", "recondition_masks_in_existing_state"):
        if hasattr(trk, name):
            setattr(trk, name, _wrap(getattr(trk, name)))
    trk._shotspace_state_merge_patched = True


class DetectionMaskOutputs:
    """Repair collapsed tracker masks with the detector's own instance mask, and record per-frame scores.

    SAM 3.1 outputs, for an existing object, the mask its tracker propagated from memory. Measured on crowded
    night footage (MOT17-04) a propagated mask can collapse - a clearly visible person shrinks to a head-only
    blob and stays like that for 150 frames - while the per-frame detector keeps returning a clean whole-person
    instance mask with a 0.9+ score for that same person.

    Identity stays exactly what SAM 3.1's tracker decided. The output mask of an object is swapped for a
    detection only in the collapse case, i.e. when all of these hold:
      * the propagated mask lies inside the detection (>= `min_inside` of its low-res pixels),
      * it is much smaller than the detection (<= `max_area_ratio` of the detection's area),
      * the detection is the silhouette this object itself output within the last `max_gap` frames
        (mask IoU >= `min_iou`), so a stray fragment sitting on somebody else is never promoted to that person.
    Swapping healthy masks for detections was measured too (MOTS20-09) and rejected: it lowered mask precision
    (74.5 vs 79.9) because the detector returns only one part of a silhouette that an occluder cuts in two.
    Also records a per-frame score per object (detector score when repaired or newly detected, tracker object
    score otherwise)."""

    def __init__(
        self,
        model: Any,
        min_inside: float = 0.8,
        max_area_ratio: float = 0.6,
        min_iou: float = 0.5,
        max_gap: int = 5,
    ) -> None:
        self.model = model
        self.min_inside = float(min_inside)
        self.max_area_ratio = float(max_area_ratio)
        self.min_iou = float(min_iou)
        self.max_gap = int(max_gap)
        self.confirm_score = 0.25
        self.confirm_iom = 0.25
        self.enabled = False
        self.frame_scores: dict[int, dict[int, float]] = {}
        self.frame_matched: dict[int, set[int]] = {}
        self.frame_confirmed: dict[int, set[int]] = {}
        self._last_out: dict[int, tuple[int, Any, float]] = {}  # obj id -> (frame idx, low-res bool mask, area)
        if getattr(model, "_shotspace_det_outputs_patched", False):
            raise RuntimeError("already installed on this model; use DetectionMaskOutputs.get(model)")
        self._orig = model.build_outputs
        model.build_outputs = self._build_outputs
        model._shotspace_det_outputs_patched = True
        model._shotspace_det_outputs = self

    @staticmethod
    def get(model: Any) -> "DetectionMaskOutputs":
        inst = getattr(model, "_shotspace_det_outputs", None)
        return inst if inst is not None else DetectionMaskOutputs(model)

    def reset(self) -> None:
        self.frame_scores.clear()
        self.frame_matched.clear()
        self.frame_confirmed.clear()
        self._last_out.clear()

    def _build_outputs(self, *args: Any, **kwargs: Any) -> dict:
        obj_id_to_mask = self._orig(*args, **kwargs)
        if not self.enabled:
            return obj_id_to_mask
        import torch.nn.functional as F

        frame_idx = int(kwargs["frame_idx"])
        det_out = kwargs["det_out"]
        trk_low = kwargs["tracker_low_res_masks_global"]
        obj_ids = [int(o) for o in kwargs["tracker_metadata_prev"]["obj_ids_all_gpu"]]
        h_vid, w_vid = int(kwargs["orig_vid_height"]), int(kwargs["orig_vid_width"])
        scores_f: dict[int, float] = {}
        repaired: set[int] = set()
        trk_scores = kwargs.get("tracker_obj_scores_global")
        if trk_scores is not None and trk_scores.numel() == len(obj_ids):
            ts = trk_scores.float().sigmoid().flatten().tolist()
            scores_f.update({oid: float(v) for oid, v in zip(obj_ids, ts)})
        new_ids = [int(o) for o in kwargs["sam2_update_plan"]["new_det_obj_ids"]]
        new_inds = [int(i) for i in kwargs["sam2_update_plan"]["new_det_fa_inds"]]
        det_scores = det_out["scores"].float()
        confirmed: set[int] = set(new_ids)
        for oid, di in zip(new_ids, new_inds):
            scores_f[oid] = float(det_scores[di])
        for _det, trk_ids in (kwargs.get("det_to_matched_trk_obj_ids") or {}).items():
            confirmed.update(int(t) for t in np.asarray(trk_ids).reshape(-1).tolist())

        n_trk = min(int(trk_low.shape[0]), len(obj_ids))
        # "Seen by the detector" for the confirmation flag is deliberately more inclusive than SAM 3.1's own
        # association (score > 0.4, away from the frame border): a partly occluded or border-cut person still
        # draws a weaker detection, and that is enough to say the person is really there on this frame.
        weak = det_scores > self.confirm_score
        if n_trk > 0 and bool(weak.any()):
            w_idx = torch.nonzero(weak, as_tuple=True)[0]
            w_bool = det_out["mask"][w_idx].float() > 0
            t_low = trk_low[:n_trk].float()
            if t_low.shape[-2:] != w_bool.shape[-2:]:
                t_low = F.interpolate(t_low.unsqueeze(1), size=w_bool.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            tb = (t_low > 0).flatten(1).float()
            wb = w_bool.flatten(1).float()
            inter_w = wb @ tb.T
            iom = inter_w / torch.minimum(wb.sum(dim=1)[:, None], tb.sum(dim=1)[None, :]).clamp_min(1.0)
            seen = (iom >= self.confirm_iom).any(dim=0).tolist()
            confirmed.update(obj_ids[tj] for tj, ok in enumerate(seen) if ok)
        keep = det_scores > float(self.model.score_threshold_detection)
        det_idx = torch.nonzero(keep, as_tuple=True)[0]
        d_bool = (det_out["mask"][det_idx].float() > 0) if det_idx.numel() > 0 else None
        low_hw = d_bool.shape[-2:] if d_bool is not None else (trk_low.shape[-2:] if n_trk > 0 else None)
        out_low: dict[int, Any] = {}
        t_bool = None
        if n_trk > 0:
            trk_logits = trk_low[:n_trk].float()
            if trk_logits.shape[-2:] != low_hw:
                trk_logits = F.interpolate(trk_logits.unsqueeze(1), size=low_hw, mode="bilinear", align_corners=False).squeeze(1)
            t_bool = trk_logits > 0
        if t_bool is not None and d_bool is not None:
            d_bin = d_bool.flatten(1).float()
            t_bin = t_bool.flatten(1).float()
            inter = d_bin @ t_bin.T  # [n_det, n_trk] shared low-res pixels
            area_t = t_bin.sum(dim=1)
            area_d = d_bin.sum(dim=1)
            collapsed = (
                (inter >= self.min_inside * area_t.clamp_min(1.0)[None, :])
                & (area_t[None, :] > 0)
                & (area_t[None, :] <= self.max_area_ratio * area_d[:, None])
            )
            cand_d, cand_t = torch.nonzero(collapsed, as_tuple=True)
            pairs: list[tuple[int, int]] = []
            used_d: set[int] = set()
            used_t: set[int] = set()
            order = torch.argsort(inter[cand_d, cand_t], descending=True).tolist() if cand_d.numel() > 0 else []
            for k in order:
                di, tj = int(cand_d[k]), int(cand_t[k])
                if di in used_d or tj in used_t:
                    continue
                last = self._last_out.get(obj_ids[tj])
                if last is None or abs(frame_idx - last[0]) > self.max_gap or last[1].shape != d_bool[di].shape:
                    continue
                li = float((last[1] & d_bool[di]).sum())
                lu = float((last[1] | d_bool[di]).sum())
                if lu <= 0 or (li / lu) < self.min_iou:
                    continue
                used_d.add(di)
                used_t.add(tj)
                pairs.append((di, tj))
            if pairs:
                sel = det_idx[torch.tensor([p[0] for p in pairs], device=det_idx.device)]
                vid = F.interpolate(det_out["mask"][sel].float().unsqueeze(1), size=(h_vid, w_vid), mode="bilinear", align_corners=False) > 0
                for k, (di, tj) in enumerate(pairs):
                    oid = obj_ids[tj]
                    key = next((q for q in obj_id_to_mask if int(q) == oid), None)
                    if key is None:
                        continue
                    obj_id_to_mask[key] = vid[k]
                    out_low[oid] = d_bool[di]
                    scores_f[oid] = float(det_scores[det_idx[di]])
                    repaired.add(oid)
        if t_bool is not None:
            for tj in range(n_trk):
                oid = obj_ids[tj]
                if oid in out_low:
                    continue
                m = t_bool[tj]
                area = float(m.sum())
                if area <= 0:
                    continue
                last = self._last_out.get(oid)
                # a mask that just collapsed (and could not be repaired on this frame) must not replace the
                # reference silhouette, otherwise the next frame's detection could never be recognised
                if last is not None and abs(frame_idx - last[0]) <= self.max_gap and area < self.max_area_ratio * last[2]:
                    continue
                self._last_out[oid] = (frame_idx, m, area)
        for oid, m in out_low.items():
            self._last_out[oid] = (frame_idx, m, float(m.sum()))
        if d_bool is not None:
            for oid, di in zip(new_ids, new_inds):
                pos = torch.nonzero(det_idx == di, as_tuple=True)[0]
                if pos.numel() > 0:
                    m = d_bool[int(pos[0])]
                    self._last_out[oid] = (frame_idx, m, float(m.sum()))
        self.frame_scores[frame_idx] = scores_f
        self.frame_matched[frame_idx] = repaired
        self.frame_confirmed[frame_idx] = confirmed | repaired
        return obj_id_to_mask


def apply_knobs(model: Any, knobs: dict[str, Any]) -> dict[str, Any]:
    """Set attributes on the SAM 3.1 demo model, returning the previous values so the caller can restore them."""
    prev: dict[str, Any] = {}
    for key, val in knobs.items():
        if not hasattr(model, key):
            raise AttributeError(f"SAM 3.1 model has no knob {key!r}")
        prev[key] = getattr(model, key)
        setattr(model, key, val)
    return prev


# SAM 3.1 resamples every frame to its own 1008 x 1008 grid and predicts masks on a 4x coarser grid, so mask detail
# stops well below 2K. Frames larger than this are downscaled before tracking (masks and boxes are mapped back to
# full-plate pixels on output); it bounds VRAM/RAM/time on 4K+ plates without losing anything the model can see.
WORK_MAX_SIDE = 2016


def prepare_frame_dir(
    frame_paths: list[str],
    dst_dir: Path,
    lut: np.ndarray | None = None,
    max_side: int = WORK_MAX_SIDE,
) -> tuple[int, int, int, int]:
    """Expose the shot frames to SAM 3.1 as ``00000.jpg ..``; returns (work_w, work_h, full_w, full_h).

    JPEG sources that need no change are symlinked; everything else is written once as JPEG (quality 95)."""
    import cv2
    from PIL import Image

    dst_dir.mkdir(parents=True, exist_ok=True)
    work_w = work_h = full_w = full_h = 0
    for idx, src_p in enumerate(frame_paths):
        src = Path(src_p).resolve()
        dst = dst_dir / f"{idx:05d}.jpg"
        with Image.open(src) as im_raw:
            w0, h0 = im_raw.size
            if idx == 0:
                full_w, full_h = int(w0), int(h0)
                sc = min(1.0, float(max_side) / float(max(w0, h0)))
                work_w = int(round(w0 * sc)) if sc < 1.0 else int(w0)
                work_h = int(round(h0 * sc)) if sc < 1.0 else int(h0)
            same_size = (w0, h0) == (work_w, work_h)
            if same_size and lut is None and src.suffix.lower() in (".jpg", ".jpeg"):
                link_file(src, dst)
                continue
            arr = np.asarray(im_raw.convert("RGB"), dtype=np.uint8)
        if not same_size:
            arr = cv2.resize(arr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        if lut is not None:
            arr = cv2.LUT(arr, lut)
        Image.fromarray(arr).save(dst, format="JPEG", quality=95)
    return work_w, work_h, full_w, full_h


def unpack_person_mask(archive: Any, pid: int, frame: int, width: int, height: int) -> np.ndarray | None:
    """Full-plate boolean mask of person `pid` on `frame` from `person_masks_sam3.npz`, or None when absent.

    `archive` is the opened npz (or a dict of its arrays). Handles crops stored below plate resolution (4K+ plates)
    by resizing the crop onto its `_rect`; consumers should use this instead of pasting crops themselves."""
    kb, ks, kr = f"p{pid}_f{frame}_bits", f"p{pid}_f{frame}_shape", f"p{pid}_f{frame}_rect"
    if kb not in archive or ks not in archive or kr not in archive:
        return None
    mh, mw = int(archive[ks][0]), int(archive[ks][1])
    crop = np.unpackbits(np.asarray(archive[kb]))[: mh * mw].reshape(mh, mw)
    x1f, y1f, x2f, y2f = [float(v) for v in np.asarray(archive[kr]).reshape(-1)[:4]]
    x1 = max(0, min(width - 1, int(round(x1f))))
    y1 = max(0, min(height - 1, int(round(y1f))))
    x2 = max(x1 + 1, min(width, int(round(x2f))))
    y2 = max(y1 + 1, min(height, int(round(y2f))))
    if crop.shape != (y2 - y1, x2 - x1):
        import cv2

        crop = cv2.resize(crop.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = crop > 0
    return full


def _prune_feature_cache(pred: Any, session_id: str) -> None:
    try:
        sess = pred._all_inference_states.get(session_id)
        fcache = sess["state"].get("feature_cache") if (sess and isinstance(sess.get("state"), dict)) else None
        if isinstance(fcache, dict):
            for k in [k for k in list(fcache.keys()) if isinstance(k, int)]:
                fcache.pop(k, None)
    except Exception:
        pass


def iter_native_tracking(
    pred: Any,
    frames_dir: Path,
    n_frames: int,
    start_idx: int = 0,
    reverse: bool = False,
    prompt: str = PROMPT_TEXT,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Yield ``(frame_idx, obj_ids[N], det_scores[N], masks[N, H, W] bool, n_new_people_dropped_at_cap)`` for every
    frame, in processing order."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sid = pred.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frames_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )["session_id"]
        try:
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": int(start_idx), "text": prompt})
            _prune_feature_cache(pred, sid)
            stream = pred.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "propagation_direction": "backward" if reverse else "forward",
                    "start_frame_index": int(start_idx),
                    "evict_cached_frame_outputs": True,
                }
            )
            for resp in stream:
                outs = resp["outputs"]
                fs = outs.get("frame_stats") or {}
                yield (
                    int(resp["frame_index"]),
                    np.asarray(outs["out_obj_ids"]).astype(np.int64).reshape(-1),
                    np.asarray(outs["out_probs"], dtype=np.float32).reshape(-1),
                    np.asarray(outs["out_binary_masks"], dtype=bool),
                    int(fs.get("num_obj_dropped") or 0),
                )
        finally:
            try:
                pred.handle_request({"type": "close_session", "session_id": sid, "clear_cache_threshold": 0})
            except Exception:
                pass


# SAM 3.1 looks for instances of a noun phrase, and "person" is not always the phrase it responds to: on a ballet
# stage (DanceTrack 0026: a dozen dancers in white tutus) "person", "people" and "human" return ZERO detections while
# "dancer" returns 13 with scores up to 0.93. So before tracking, a handful of frames is probed with several words
# for people and the shot is tracked with "person" unless another word clearly finds more people.
PROMPT_CANDIDATES = ("person", "man", "woman", "child", "dancer", "athlete", "soldier", "performer")
PROMPT_PROBE_FRAMES = 5
PROMPT_KEEP_DEFAULT_RATIO = 0.8


def choose_prompt(
    pred: Any,
    frames_dir: Path,
    n_frames: int,
    candidates: tuple[str, ...] = PROMPT_CANDIDATES,
    n_probe: int = PROMPT_PROBE_FRAMES,
    min_score: float = 0.5,
) -> tuple[str, dict]:
    """Pick the noun phrase SAM 3.1 detects the people of this shot with.

    Runs the detector alone on `n_probe` evenly spaced frames for every candidate and counts instances scoring
    >= `min_score`. The first candidate ("person", the most general word) is kept whenever it finds at least
    `PROMPT_KEEP_DEFAULT_RATIO` of what the best candidate finds; otherwise the best candidate is used."""
    import tempfile

    idxs = sorted({int(round(v)) for v in np.linspace(0, max(0, n_frames - 1), min(n_probe, n_frames))})
    counts: dict[str, int] = {c: 0 for c in candidates}
    sums: dict[str, float] = {c: 0.0 for c in candidates}
    with tempfile.TemporaryDirectory(prefix="probe_", dir=str(frames_dir.parent)) as td:
        for fi in idxs:
            one = Path(td) / f"f{fi:05d}"
            one.mkdir()
            link_file((frames_dir / f"{fi:05d}.jpg").resolve(), one / "00000.jpg")
            for cand in candidates:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sid = pred.handle_request(
                        {"type": "start_session", "resource_path": str(one), "offload_video_to_cpu": True, "offload_state_to_cpu": True}
                    )["session_id"]
                    try:
                        out = pred.handle_request(
                            {"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": cand, "output_prob_thresh": min_score}
                        )["outputs"]
                    finally:
                        try:
                            pred.handle_request({"type": "close_session", "session_id": sid, "clear_cache_threshold": 0})
                        except Exception:
                            pass
                probs = np.asarray(out.get("out_probs", []), dtype=np.float64).reshape(-1)
                probs = probs[probs >= min_score]
                counts[cand] += int(len(probs))
                sums[cand] += float(probs.sum())
    default = candidates[0]
    best = max(candidates, key=lambda c: (counts[c], sums[c]))
    chosen = default if counts[default] >= PROMPT_KEEP_DEFAULT_RATIO * counts[best] else best
    report = {
        "probe_frames": idxs,
        "instances": counts,
        "score_sum": {c: round(v, 2) for c, v in sums.items()},
        "chosen": chosen,
    }
    return chosen, report


def _collect_range(
    pred: Any,
    frames_dir: Path,
    n_local: int,
    offset: int,
    reverse: bool,
    store: TrackStore,
    counters: dict,
    on_frame: Callable[[int], None] | None,
    use_detection_masks: bool,
    prompt: str = PROMPT_TEXT,
) -> None:
    """One SAM 3.1 session over the `n_local` frames in `frames_dir` (shot index = local index + `offset`);
    entries are added to `store` as they are produced, so a failure mid-way keeps everything already yielded."""
    start_idx = (n_local - 1) if reverse else 0
    n_done = 0
    hook = DetectionMaskOutputs.get(pred.model)
    hook.reset()
    hook.enabled = bool(use_detection_masks)
    try:
        for fidx, oids, probs, masks, n_drop in iter_native_tracking(
            pred, frames_dir, n_local, start_idx=start_idx, reverse=reverse, prompt=prompt
        ):
            counters["peak_objs"] = max(counters["peak_objs"], len(oids))
            counters["dropped_at_cap"] += n_drop
            f_scores = hook.frame_scores.pop(fidx, {}) if hook.enabled else {}
            f_matched = hook.frame_matched.pop(fidx, set()) if hook.enabled else set()
            f_conf = hook.frame_confirmed.pop(fidx, None) if hook.enabled else None
            for k, oid in enumerate(oids.tolist()):
                sc = f_scores.get(int(oid), float(probs[k]) if k < len(probs) else 1.0)
                is_conf = True if f_conf is None else (int(oid) in f_conf)
                if store.add(int(oid), fidx + offset, masks[k], sc, confirmed=is_conf):
                    counters["entries"] += 1
                    counters["repaired"] += int(int(oid) in f_matched)
            counters["last_frame"] = fidx + offset
            counters["frames"] += 1
            n_done += 1
            if on_frame is not None:
                on_frame(fidx + offset)
            if n_done % RAM_CHECK_EVERY == 0 and n_done >= 2 * RESTART_OVERLAP and n_local - n_done > RESTART_OVERLAP:
                total_gb, avail_gb = _system_ram_gb()
                rss = _rss_gb()
                counters["peak_rss_gb"] = max(counters.get("peak_rss_gb", 0.0), rss)
                if total_gb > 0 and (rss > RAM_SHARE_LIMIT * total_gb or avail_gb < RAM_FREE_FLOOR_GB):
                    raise _RamBudgetReached(f"rss {rss:.1f} GB, free {avail_gb:.1f} GB of {total_gb:.0f} GB")
    finally:
        hook.enabled = False
        hook.reset()


def _rss_gb() -> float:
    """Resident memory of this process in GB (0.0 when it cannot be read)."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        pass
    try:  # Windows (no /proc)
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return 0.0


def _system_ram_gb() -> tuple[float, float]:
    """(total, available) system RAM in GB ((0, 0) when it cannot be read)."""
    try:
        info: dict[str, float] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = float(rest.split()[0]) / 2**20
        return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)
    except Exception:
        pass
    try:  # Windows (no /proc)
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except Exception:
        return 0.0, 0.0


class _RamBudgetReached(RuntimeError):
    """Raised inside a session when host RAM runs low; the pass continues in a fresh session (state released)."""


# SAM 3.1 keeps every frame's tracker state until the session closes (measured: 29.6 GB resident for 2 x 300 frames
# with ~20 dancers). A session is therefore ended early - and continued in a new one, stitched on shared frames -
# once this process holds more than RAM_SHARE_LIMIT of the machine's RAM or less than RAM_FREE_FLOOR_GB stays free.
RAM_SHARE_LIMIT = 0.40
RAM_FREE_FLOOR_GB = 6.0
RAM_CHECK_EVERY = 8


RESTART_OVERLAP = 24
MAX_RESTARTS = 6


def collect_pass(
    pred: Any,
    frames_dir: Path,
    n_frames: int,
    reverse: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    use_detection_masks: bool = True,
    prompt: str = PROMPT_TEXT,
) -> tuple[TrackStore, dict]:
    """Track every person through the `n_frames` frames of `frames_dir` in one time direction.

    SAM 3.1's dense tracker can abort in the middle of a crowded shot (seen on MOT17-04, backward pass, frame 55 of
    299: "No points are provided; please add points first" - one of its internal object buckets was left without a
    conditioning frame). Out-of-memory errors are passed up to the caller; any other failure is survived by
    starting a fresh session `RESTART_OVERLAP` frames before the point reached (or, when the failure came sooner
    than that, halfway into what was reached) and joining the two parts on their shared frames by mask agreement
    (`stitch_chunks`), exactly like the pieces of a long shot."""
    import shutil
    import tempfile

    t0 = time.time()
    counters = {"peak_objs": 0, "dropped_at_cap": 0, "entries": 0, "repaired": 0, "frames": 0, "last_frame": None}
    seen: set[int] = set()

    def _on_frame(fi: int) -> None:
        seen.add(fi)
        if on_progress is not None:
            on_progress(len(seen), n_frames)

    total: TrackStore | None = None
    lo, hi = 0, n_frames  # shot-index range [lo, hi) the current session covers
    restarts: list[dict] = []
    scratch: list[Path] = []
    try:
        while True:
            part = TrackStore()
            if (lo, hi) == (0, n_frames):
                cur_dir = frames_dir
            else:
                cur_dir = Path(tempfile.mkdtemp(prefix="restart_", dir=str(frames_dir.parent)))
                scratch.append(cur_dir)
                for k, fi in enumerate(range(lo, hi)):
                    link_file((frames_dir / f"{fi:05d}.jpg").resolve(), cur_dir / f"{k:05d}.jpg")
            counters["last_frame"] = None
            failure: BaseException | None = None
            try:
                _collect_range(pred, cur_dir, hi - lo, lo, reverse, part, counters, _on_frame, use_detection_masks, prompt)
            except RuntimeError as exc:
                if _is_oom(exc):
                    raise
                failure = exc
            _release_host_memory()
            # join this part with what earlier sessions of this pass produced
            if total is None:
                total = part
            elif reverse:
                total, _ = stitch_chunks(part, total, restarts[-1]["overlap"][0], restarts[-1]["overlap"][1])
            else:
                total, _ = stitch_chunks(total, part, restarts[-1]["overlap"][0], restarts[-1]["overlap"][1])
            if failure is None:
                break
            reached = counters["last_frame"]
            progressed = reached is not None and (hi - lo) > RESTART_OVERLAP + 8
            is_budget = isinstance(failure, _RamBudgetReached)
            n_fail = sum(1 for r in restarts if not r.get("ram_budget"))
            if not progressed or (not is_budget and n_fail >= MAX_RESTARTS):
                if is_budget:
                    raise RuntimeError(f"内存不足：人物跟踪无法在这台机器的内存里继续（{failure}）") from failure
                raise failure
            # A session that failed within its first RESTART_OVERLAP frames cannot go back that far, and starting
            # it again on the same frame would fail the same way.  The next session then starts halfway into what
            # was reached: a different first frame gives SAM 3.1 different object buckets, and the frames both
            # sessions produced are still shared for joining the identities.
            if reverse:
                new_hi = min(hi, reached + RESTART_OVERLAP)
                if new_hi >= hi:
                    new_hi = hi - max(1, (hi - reached) // 2)
                    if new_hi <= reached:
                        raise failure
                overlap = [reached, new_hi]
                hi = new_hi
            else:
                new_lo = max(lo, reached - RESTART_OVERLAP + 1)
                if new_lo <= lo:
                    new_lo = lo + max(1, (reached - lo + 1) // 2)
                    if new_lo > reached:
                        raise failure
                overlap = [new_lo, reached + 1]
                lo = new_lo
            restarts.append(
                {
                    "at_frame": int(reached),
                    "overlap": overlap,
                    "ram_budget": is_budget,
                    "error": f"{type(failure).__name__}: {str(failure)[:160]}",
                }
            )
    finally:
        for d in scratch:
            shutil.rmtree(d, ignore_errors=True)
    assert total is not None
    stats = {
        "frames": len(seen),
        "peak_objects_per_frame": counters["peak_objs"],
        "seconds": round(time.time() - t0, 1),
        "mask_entries": counters["entries"],
        "repaired_mask_share": round(counters["repaired"] / max(1, counters["entries"]), 4),
        # new people SAM 3.1 detected but could not start tracking because `max_num_objects` was reached
        "new_people_dropped_at_cap": counters["dropped_at_cap"],
    }
    if restarts:
        stats["session_restarts"] = restarts
    stats["rss_after_gb"] = round(_rss_gb(), 2)
    return total, stats


def _release_host_memory() -> None:
    """Return freed tracker state to the OS between sessions (Python garbage, CUDA cache, glibc arenas)."""
    import ctypes
    import gc

    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    if os.name != "nt":
        try:
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim(0)
        except (OSError, TypeError):
            pass


def _crop_iou(store_a: TrackStore, a: int, store_b: TrackStore, b: int, fidx: int) -> float:
    """Mask IoU of two stored masks on the same frame (crops may span different rectangles)."""
    ax1, ay1, ax2, ay2 = store_a.boxes[a][fidx]
    bx1, by1, bx2, by2 = store_b.boxes[b][fidx]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    ma = store_a.mask_crop(a, fidx)[iy1 - ay1 : iy2 - ay1, ix1 - ax1 : ix2 - ax1]
    mb = store_b.mask_crop(b, fidx)[iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1]
    inter = int(np.count_nonzero(ma & mb))
    if inter == 0:
        return 0.0
    union = store_a.areas[a][fidx] + store_b.areas[b][fidx] - inter
    return inter / max(1, union)


def pass_agreement(store_a: TrackStore, store_b: TrackStore, iou_thr: float = 0.5) -> dict[tuple[int, int], tuple[int, int]]:
    """For every pair (track of pass A, track of pass B): (#frames where both exist, #frames where masks agree)."""
    frames_a: dict[int, list[int]] = {}
    for a, fmap in store_a.boxes.items():
        for f in fmap:
            frames_a.setdefault(f, []).append(a)
    out: dict[tuple[int, int], list[int]] = {}
    for b, fmap in store_b.boxes.items():
        for f, (bx1, by1, bx2, by2) in fmap.items():
            for a in frames_a.get(f, ()):
                ax1, ay1, ax2, ay2 = store_a.boxes[a][f]
                rec = out.setdefault((a, b), [0, 0])
                rec[0] += 1
                if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                    continue
                if _crop_iou(store_a, a, store_b, b, f) >= iou_thr:
                    rec[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def _frame_partners(store_a: TrackStore, store_b: TrackStore, iou_thr: float = 0.5) -> dict[int, dict[int, int]]:
    """For every track of pass A: frame -> the track of pass B whose mask coincides with it (IoU >= thr)."""
    frames_b: dict[int, list[int]] = {}
    for b, fmap in store_b.boxes.items():
        for f in fmap:
            frames_b.setdefault(f, []).append(b)
    out: dict[int, dict[int, int]] = {}
    for a, fmap in store_a.boxes.items():
        part: dict[int, int] = {}
        for f, (ax1, ay1, ax2, ay2) in fmap.items():
            best, best_iou = None, iou_thr
            for b in frames_b.get(f, ()):
                bx1, by1, bx2, by2 = store_b.boxes[b][f]
                if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                    continue
                v = _crop_iou(store_a, a, store_b, b, f)
                if v >= best_iou:
                    best, best_iou = b, v
            if best is not None:
                part[f] = best
        out[a] = part
    return out


class AppearanceProbe:
    """How much does a track's colour distribution change between the frames just before and just after a moment?

    Used only to arbitrate between the two tracking directions when they disagree about who is who: the track
    whose masked pixels change colour across the disputed moment is the one that moved to another person.
    0 = identical colour histograms, 1 = disjoint (Lab, 8 bins per channel, over the masked pixels of up to
    `n_side` frames on each side)."""

    def __init__(self, frame_path: Callable[[int], str | None], n_side: int = 6) -> None:
        self._frame_path = frame_path
        self.n_side = int(n_side)
        self._cache: dict[int, np.ndarray | None] = {}

    def _lab(self, fidx: int) -> np.ndarray | None:
        if fidx not in self._cache:
            if len(self._cache) > 48:
                self._cache.clear()
            import cv2

            path = self._frame_path(fidx)
            bgr = cv2.imread(path) if path else None
            self._cache[fidx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB) if bgr is not None else None
        return self._cache[fidx]

    def _hist(self, store: "TrackStore", tid: int, frames: list[int]) -> np.ndarray | None:
        acc = np.zeros(512, dtype=np.float64)
        for f in frames:
            lab = self._lab(f)
            if lab is None:
                continue
            x1, y1, x2, y2 = store.boxes[tid][f]
            if y2 > lab.shape[0] or x2 > lab.shape[1]:
                continue
            px = lab[y1:y2, x1:x2][store.mask_crop(tid, f)]
            if len(px) == 0:
                continue
            q = (px // 32).astype(np.int64)
            acc += np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512)
        tot = acc.sum()
        return acc / tot if tot > 0 else None

    def discontinuity(self, store: "TrackStore", tid: int, before: int, after: int) -> float | None:
        frames = sorted(store.boxes.get(tid, {}))
        fb = [f for f in frames if f <= before][-self.n_side :]
        fa = [f for f in frames if f >= after][: self.n_side]
        if len(fb) < 2 or len(fa) < 2:
            return None
        hb, ha = self._hist(store, tid, fb), self._hist(store, tid, fa)
        if hb is None or ha is None:
            return None
        return 1.0 - float(np.minimum(hb, ha).sum())


def _split_at_hops(
    store: TrackStore,
    partners: dict[int, dict[int, int]],
    other: TrackStore,
    min_run: int = 3,
    coexist_frames: int = 3,
    probe: AppearanceProbe | None = None,
) -> tuple[dict[tuple[int, int], list[int]], int, int, int]:
    """Cut every track of one pass where the other pass proves it changed person.

    Along a track, the coinciding track of the other pass is its witness. Suppose the witness changes from X to Y.
    This track is the one that hopped only if both witnesses are seen living on somewhere else across the change:
      * X still exists after the change, away from this track (its person did not vanish - this track left it), and
      * Y already existed before the change, away from this track (its person was there before - this track joined it).
    Then X and Y are two different people and the track is cut between the two witness runs. If only one of the two
    holds (X simply ended, or Y simply started), the other pass lost / re-found the person and this track is kept
    whole - it is what joins the other pass's fragments. A witness that shows up for fewer than `min_run` frames
    between two runs of another witness is a momentary overlap; those frames are dropped from the track.

    When two people swap in one direction only, the evidence above is symmetric: the hopping tracks AND the correct
    tracks of the other direction all see their witness change. `probe` settles which direction is right: a track is
    cut only if its own colours change across the moment more than its witnesses' colours do. Measured on the
    ground-truth-labelled cuts of MOTS20-09 / MOTS20-02 this kept all 5 cuts where the person really changes and
    waived all 10 measurable cuts where it is the same person.
    Returns ({(track, piece_index): [frames]}, number of cuts, number of dropped frames, cuts waived by `probe`)."""
    frames_of_other = {o: sorted(other.boxes[o]) for o in other.object_ids()}

    def lives_elsewhere(w: int, tid: int, frames: list[int]) -> bool:
        """Witness `w` has `coexist_frames`+ frames among `frames` where it does not coincide with track `tid`."""
        n = 0
        part = partners.get(tid, {})
        own = store.boxes[tid]
        for f in frames:
            if f in other.boxes[w] and not (f in own and part.get(f) == w):
                n += 1
                if n >= coexist_frames:
                    return True
        return False

    pieces: dict[tuple[int, int], list[int]] = {}
    n_cuts = n_dropped = n_waived = 0
    for tid in store.object_ids():
        frames = sorted(store.boxes[tid])
        part = partners.get(tid, {})
        runs: list[list] = []  # [witness, [frames]] in track order; frames without a witness are neutral
        for f in frames:
            w = part.get(f)
            if w is None:
                continue
            if runs and runs[-1][0] == w:
                runs[-1][1].append(f)
            else:
                runs.append([w, [f]])
        # a short visit of another witness between two runs of the same witness is a momentary overlap
        drop: set[int] = set()
        solid: list[list] = []
        for k, (w, fr) in enumerate(runs):
            nxt = runs[k + 1][0] if k + 1 < len(runs) else None
            if len(fr) < min_run and solid and solid[-1][0] != w and nxt == solid[-1][0]:
                drop.update(fr)
                continue
            if solid and solid[-1][0] == w:
                solid[-1][1].extend(fr)
            else:
                solid.append([w, list(fr)])
        cuts: list[float] = []
        for k in range(1, len(solid)):
            (w1, f1), (w2, f2) = solid[k - 1], solid[k]
            if w1 == w2:
                continue
            after = [f for f in frames_of_other[w1] if f > f1[-1]]
            before = [f for f in frames_of_other[w2] if f < f2[0]]
            if lives_elsewhere(w1, tid, after) and lives_elsewhere(w2, tid, before):
                if probe is not None:
                    d_self = probe.discontinuity(store, tid, f1[-1], f2[0])
                    d_wit = [probe.discontinuity(other, w, f1[-1], f2[0]) for w in (w1, w2)]
                    d_wit = [d for d in d_wit if d is not None]
                    if d_self is not None and d_wit and d_self <= float(np.mean(d_wit)):
                        n_waived += 1
                        continue
                cuts.append(0.5 * (f1[-1] + f2[0]))
        n_cuts += len(cuts)
        n_dropped += len(drop)
        cur: list[int] = []
        idx = 0
        ci = 0
        for f in frames:
            if f in drop:
                continue
            while ci < len(cuts) and f > cuts[ci]:
                if cur:
                    pieces[(tid, idx)] = cur
                    idx += 1
                    cur = []
                ci += 1
            cur.append(f)
        if cur:
            pieces[(tid, idx)] = cur
    return pieces, n_cuts, n_dropped, n_waived


def merge_passes(
    fwd: TrackStore,
    bwd: TrackStore,
    min_agree: int = 5,
    min_ratio: float = 0.7,
    require_both: bool = True,
    probe: AppearanceProbe | None = None,
) -> tuple[TrackStore, dict]:
    """Fuse a forward and a backward tracking pass into one set of identities.

    Each pass is SAM 3.1's own identity decision seen from one time direction. A person lost in one direction
    (re-detected under a new id after an occlusion, or picked up late) is usually carried through by the other,
    and a track that hops onto another person in one direction is contradicted by the other.
      1. Every track is cut where the other pass proves it changed person (`_split_at_hops`; with `probe`, the
         colours inside the masks decide which direction is the one that changed person when both look guilty).
      2. Two pieces from different passes are the same person when their MASKS coincide (IoU >= 0.5) on at least
         `min_ratio` of the frames where both exist (and on >= `min_agree` frames). Pieces are joined greedily,
         strongest agreement first, never putting two pieces of the same pass that exist at the same time into one
         identity (one person cannot have two masks on a frame).
      3. On every frame an identity takes the mask of its longest piece present there.
      4. An identity that only one direction produced is dropped (`require_both`): measured on MOTS20-09, the 11 such
         identities of 40 carried 17 correct person-frames against 176 false ones - they are ghosts and duplicates,
         a real person is found from both ends of the shot.
    Nothing but the masks SAM 3.1 produced enters these decisions."""
    part_f = _frame_partners(fwd, bwd)
    part_b = _frame_partners(bwd, fwd)
    pieces_f, cuts_f, drop_f, waived_f = _split_at_hops(fwd, part_f, bwd, probe=probe)
    pieces_b, cuts_b, drop_b, waived_b = _split_at_hops(bwd, part_b, fwd, probe=probe)

    nodes: dict[tuple[str, int, int], list[int]] = {}
    owner_f: dict[tuple[int, int], tuple[str, int, int]] = {}
    owner_b: dict[tuple[int, int], tuple[str, int, int]] = {}
    for (tid, k), fr in pieces_f.items():
        nodes[("f", tid, k)] = fr
        for f in fr:
            owner_f[(tid, f)] = ("f", tid, k)
    for (tid, k), fr in pieces_b.items():
        nodes[("b", tid, k)] = fr
        for f in fr:
            owner_b[(tid, f)] = ("b", tid, k)

    agree: dict[tuple, int] = {}
    for a, part in part_f.items():
        for f, b in part.items():
            na, nb = owner_f.get((a, f)), owner_b.get((b, f))
            if na is not None and nb is not None:
                agree[(na, nb)] = agree.get((na, nb), 0) + 1
    frame_sets = {n: set(fr) for n, fr in nodes.items()}
    parent = {n: n for n in nodes}
    members = {n: [n] for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def conflict(r1, r2) -> bool:
        for m1 in members[r1]:
            for m2 in members[r2]:
                if m1[0] == m2[0] and len(frame_sets[m1] & frame_sets[m2]) > 2:
                    return True
        return False

    edges = []
    for (na, nb), n_ag in agree.items():
        n_co = len(frame_sets[na] & frame_sets[nb])
        if n_ag >= min_agree and n_ag >= min_ratio * n_co:
            edges.append((n_ag, na, nb))
    edges.sort(key=lambda e: -e[0])
    joined = refused = 0
    for _n, na, nb in edges:
        r1, r2 = find(na), find(nb)
        if r1 == r2:
            continue
        if conflict(r1, r2):
            refused += 1
            continue
        parent[r2] = r1
        members[r1].extend(members.pop(r2))
        joined += 1

    out = TrackStore()
    new_id = 0
    one_direction_only = 0
    for root in sorted(members, key=lambda r: min(min(frame_sets[m]) for m in members[r])):
        if require_both and len({m[0] for m in members[root]}) < 2:
            one_direction_only += 1
            continue
        new_id += 1
        for src, tid, k in sorted(members[root], key=lambda m: -len(frame_sets[m])):
            st = fwd if src == "f" else bwd
            for f in nodes[(src, tid, k)]:
                if f in out.boxes.get(new_id, {}):
                    continue  # the longer piece already supplied this frame
                out.copy_entry(new_id, f, st, tid)
    stats = {
        "forward_tracks": len(fwd.object_ids()),
        "backward_tracks": len(bwd.object_ids()),
        "hop_cuts_forward": cuts_f,
        "hop_cuts_backward": cuts_b,
        "cuts_waived_by_appearance": waived_f + waived_b,
        "momentary_frames_dropped": drop_f + drop_b,
        "merged_identities": new_id,
        "joins": joined,
        "joins_refused_same_time": refused,
        "dropped_one_direction_only": one_direction_only,
    }
    return out, stats


def trim_unconfirmed_ends(store: TrackStore, max_gap: int = 1) -> int:
    """Make entries and exits clean: a visible stretch of a person starts on its first and ends on its last frame
    that SAM 3.1's per-frame detector confirmed.

    After a person leaves the frame or disappears behind somebody, the tracker can keep emitting a propagated mask
    for a few more frames (it is only suppressed once several frames passed without a matching detection); those
    frames are where a stale mask sits on background or on the occluder. Inside a stretch, frames the detector
    missed stay - the tracker bridging a blurred frame is exactly what it is for. Returns the number of removed
    (person, frame) entries."""
    removed = 0
    for oid in store.object_ids():
        frames = sorted(store.boxes[oid])
        conf = store.confirmed.get(oid, {})
        stretch: list[int] = []
        stretches: list[list[int]] = []
        for f in frames:
            if stretch and f - stretch[-1] > max_gap:
                stretches.append(stretch)
                stretch = []
            stretch.append(f)
        if stretch:
            stretches.append(stretch)
        for st in stretches:
            ok = [f for f in st if conf.get(f, True)]
            lo, hi = (ok[0], ok[-1]) if ok else (None, None)
            for f in st:
                if lo is None or f < lo or f > hi:
                    store.remove_frame(oid, f)
                    removed += 1
        if not store.boxes.get(oid):
            store.drop(oid)
    return removed


# Long shots are tracked in overlapping pieces: SAM 3.1 keeps every frame's tracker state for the whole session
# (measured ~15 GB of RAM for 300 frames x 20 people), so memory would grow without bound with shot length.
CHUNK_FRAMES = 360
CHUNK_OVERLAP = 48


def plan_chunks(n_frames: int, chunk: int = CHUNK_FRAMES, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """[start, end) frame-index ranges covering the shot; consecutive ranges share `overlap` frames."""
    if n_frames <= chunk + overlap:
        return [(0, n_frames)]
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(n_frames, start + chunk)
        if n_frames - end < overlap:
            end = n_frames
        out.append((start, end))
        if end >= n_frames:
            return out
        start = end - overlap


def stitch_chunks(prev: TrackStore, nxt: TrackStore, ov_start: int, ov_end: int, min_agree: int = 3, min_ratio: float = 0.5) -> tuple[TrackStore, dict]:
    """Join the identities of two consecutive chunks (both in shot frame indices) over their shared frames.

    A person of the earlier chunk and a person of the later chunk are the same when their masks coincide
    (IoU >= 0.5) on >= `min_ratio` of the shared frames where both exist (and on >= `min_agree` frames); pairs are
    taken one-to-one, strongest agreement first. In the shared range the earlier chunk supplies the first half
    and the later chunk the second half, so neither chunk's start-up frames are used where the other is settled."""
    counts: dict[tuple[int, int], list[int]] = {}
    in_prev: dict[int, list[int]] = {}
    for a, fmap in prev.boxes.items():
        for f in fmap:
            if ov_start <= f < ov_end:
                in_prev.setdefault(f, []).append(a)
    for b, fmap in nxt.boxes.items():
        for f in fmap:
            if not (ov_start <= f < ov_end):
                continue
            for a in in_prev.get(f, ()):
                rec = counts.setdefault((a, b), [0, 0])
                rec[0] += 1
                if _crop_iou(prev, a, nxt, b, f) >= 0.5:
                    rec[1] += 1
    pairs = sorted(((ag, a, b) for (a, b), (co, ag) in counts.items() if ag >= min_agree and ag >= min_ratio * co), reverse=True)
    a_of_b: dict[int, int] = {}
    used_a: set[int] = set()
    for _ag, a, b in pairs:
        if a in used_a or b in a_of_b:
            continue
        used_a.add(a)
        a_of_b[b] = a
    mid = (ov_start + ov_end) // 2
    out = TrackStore()
    for a in prev.object_ids():
        for f in prev.boxes[a]:
            if f < mid or a not in used_a:
                out.copy_entry(a, f, prev, a)
    next_id = (max(prev.object_ids()) if prev.object_ids() else 0) + 1
    for b in nxt.object_ids():
        if b in a_of_b:
            tgt = a_of_b[b]
            for f in nxt.boxes[b]:
                if f >= mid:
                    out.copy_entry(tgt, f, nxt, b)
        else:
            tgt = next_id
            next_id += 1
            for f in nxt.boxes[b]:
                out.copy_entry(tgt, f, nxt, b)
    return out, {"shared_frames": [ov_start, ov_end], "joined": len(a_of_b), "new_in_later_chunk": len(nxt.object_ids()) - len(a_of_b)}


# Measured (mask IoU, MOTS20-09 / MOTS20-02): one direction alone gives IDF1 74.9 / 78.2 with 17 / 19 identity switches;
# both directions fused give IDF1 83.1 / 81.1 with 4 / 9 switches. It costs a second pass; speed is not a goal here.
BIDIRECTIONAL_DEFAULT = True
# Measured on MOTS20-09 and MOT17-04: identities seen on fewer than 20 frames carry ~1 % / 0.2 % of all correctly
# tracked person-frames but make up a quarter to a third of the people list, and half of their frames are false.
# A body that is visible for less than ~0.4 s cannot be matchmoved; such people can still be added by hand.
MIN_TRACK_FRAMES = 10


def split_impossible_jumps(
    store: TrackStore, max_gap: int = 2, factor: float = 1.5, factor_alone: float = 2.5, return_within: int = 6
) -> tuple[int, int]:
    """Deal with an identity that "teleports": between two neighbouring frames nobody moves further than their own
    size.

    * It comes back within `return_within` frames (a few frames whose mask sat on something else): those frames
      are removed - the person is simply not seen there - and the identity stays whole.
    * It stays away (SAM 3.1 handed the id over to a different, far-away person; seen on small people in dense
      scenes, one id hopping across the picture and ending on somebody near the camera): the part after the jump
      becomes a separate person instead of dragging the first one's 3D path across the scene.

    A fast pan moves every box at once, so the displacement shared by the tracks present on both frames (their
    median) is taken out first; with fewer than three tracks to tell, a larger jump (`factor_alone`) is required.
    Returns (identities cut, stray frames removed)."""

    def centre(b) -> np.ndarray:
        return np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float64)

    def remove(oid: int, fidx: int) -> None:
        for name in ("boxes", "areas", "scores", "crops", "confirmed"):
            getattr(store, name).get(oid, {}).pop(fidx, None)

    cuts_made = stray = 0
    next_id = max(store.object_ids(), default=0) + 1
    for _round in range(4):  # removing stray frames creates new neighbours; a few rounds settle it
        moves: dict[tuple[int, int], list[np.ndarray]] = {}
        for oid in store.object_ids():
            fr = sorted(store.boxes[oid])
            for a, b in zip(fr[:-1], fr[1:]):
                if b - a <= max_gap:
                    moves.setdefault((a, b), []).append(centre(store.boxes[oid][b]) - centre(store.boxes[oid][a]))
        shared = {k: (np.median(np.stack(v), axis=0) if len(v) >= 3 else np.zeros(2)) for k, v in moves.items()}

        def jumps(oid: int, a: int, b: int) -> bool:
            box_a = np.asarray(store.boxes[oid][a], dtype=np.float64)
            box_b = np.asarray(store.boxes[oid][b], dtype=np.float64)
            shift = shared.get((a, b), np.zeros(2))
            box_b = box_b - np.array([shift[0], shift[1], shift[0], shift[1]])
            if min(box_a[2], box_b[2]) > max(box_a[0], box_b[0]) and min(box_a[3], box_b[3]) > max(box_a[1], box_b[1]):
                return False
            size = max(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
            limit = (factor if len(moves.get((a, b), ())) >= 3 else factor_alone) * size * (b - a)
            return float(np.linalg.norm(centre(box_b) - centre(box_a))) > limit

        changed = False
        for oid in list(store.object_ids()):
            fr = sorted(store.boxes[oid])
            i = 0
            while i + 1 < len(fr):
                a, b = fr[i], fr[i + 1]
                if b - a > max_gap or not jumps(oid, a, b):
                    i += 1
                    continue
                # does the identity return to where it was within a few frames?
                back = None
                for j in range(i + 2, len(fr)):
                    if fr[j] - a > return_within + max_gap:
                        break
                    box_a = store.boxes[oid][a]
                    box_j = store.boxes[oid][fr[j]]
                    if min(box_a[2], box_j[2]) > max(box_a[0], box_j[0]) and min(box_a[3], box_j[3]) > max(box_a[1], box_j[1]):
                        back = j
                        break
                if back is not None:
                    for f in fr[i + 1 : back]:
                        remove(oid, f)
                        stray += 1
                    fr = fr[: i + 1] + fr[back:]
                    changed = True
                    continue
                moved = fr[i + 1 :]
                for name in ("boxes", "areas", "scores", "crops", "confirmed"):
                    table = getattr(store, name)
                    src = table.get(oid, {})
                    for f in moved:
                        if f in src:
                            table.setdefault(next_id, {})[f] = src.pop(f)
                next_id += 1
                cuts_made += 1
                changed = True
                break  # the tail is a separate identity now and is examined in the next round
        if not changed:
            break
    return cuts_made, stray


def prune_tracks(store: TrackStore, n_frames: int, min_frames: int = MIN_TRACK_FRAMES) -> int:
    """Drop identities seen on fewer than `min_frames` frames (relaxed to a quarter of very short shots)."""
    need = max(1, min(min_frames, max(3, n_frames // 4)))
    drop = [o for o in store.object_ids() if len(store.boxes[o]) < need]
    for o in drop:
        store.drop(o)
    return len(drop)


def write_outputs(
    store: TrackStore,
    shot_frames: list[int],
    image_size: tuple[int, int],
    out_boxes: Path,
    out_masks: Path,
    grade_dict: dict | None = None,
    extra: dict | None = None,
    work_size: tuple[int, int] | None = None,
) -> dict:
    """Write `boxes.json` + `person_masks_sam3.npz` in the product contract (people ordered by summed mask area).

    `image_size` is the full plate size; `work_size` the size the masks were tracked at (defaults to the plate
    size). Boxes, rects and areas are always written in full-plate pixels; crops stay at working resolution."""
    full_w, full_h = int(image_size[0]), int(image_size[1])
    work_w, work_h = (int(work_size[0]), int(work_size[1])) if work_size else (full_w, full_h)
    sx = float(full_w) / float(work_w) if work_w > 0 else 1.0
    sy = float(full_h) / float(work_h) if work_h > 0 else 1.0
    order = sorted(store.object_ids(), key=lambda o: -sum(store.areas[o].values()))
    people = []
    packed: dict[str, np.ndarray] = {}
    for new_id, oid in enumerate(order, start=1):
        fmap = store.boxes[oid]
        boxes = {
            str(shot_frames[f]): [round(fmap[f][0] * sx, 2), round(fmap[f][1] * sy, 2), round(fmap[f][2] * sx, 2), round(fmap[f][3] * sy, 2)]
            for f in sorted(fmap)
        }
        people.append(
            {
                "id": new_id,
                "prominence": round(float(sum(store.areas[oid].values())) * sx * sy, 1),
                "frame_count": len(boxes),
                "boxes": boxes,
                "raw_boxes": dict(boxes),
                "scores": {str(shot_frames[f]): round(float(store.scores[oid][f]), 4) for f in sorted(fmap)},
                # visible mask area in full-plate pixels: the measurable occlusion signal for the 3D stage
                "areas": {str(shot_frames[f]): int(round(store.areas[oid][f] * sx * sy)) for f in sorted(fmap)},
                # 1 = SAM 3.1's detector saw the person on that frame, 0 = only the tracker's memory carried the mask
                "confirmed": {str(shot_frames[f]): int(bool(store.confirmed.get(oid, {}).get(f, True))) for f in sorted(fmap)},
            }
        )
        for f in sorted(fmap):
            bits, hh, ww = store.crops[oid][f]
            fnum = shot_frames[f]
            packed[f"p{new_id}_f{fnum}_bits"] = bits
            packed[f"p{new_id}_f{fnum}_shape"] = np.array([hh, ww], dtype=np.int16)
            packed[f"p{new_id}_f{fnum}_rect"] = np.asarray(boxes[str(fnum)], dtype=np.float32)
    out_boxes.parent.mkdir(parents=True, exist_ok=True)
    # Both files carry the id of this run, and each is written completely before it takes its final name: the
    # numbers in the boxes and the outlines stored under those numbers must come from the same run.  A job killed
    # while writing can then leave the previous pair, or at worst a pair the reader recognises as not matching.
    run_id = uuid.uuid4().hex
    payload = {
        "people": people,
        "grade": grade_dict or {},
        "image_size": [full_w, full_h],
        "work_size": [work_w, work_h],
        "engine": ENGINE_TAG,
        "run_id": run_id,
        **(extra or {}),
    }
    tmp_boxes = out_boxes.with_name(f".{out_boxes.name}.tmp")
    tmp_boxes.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if packed:
        packed[RUN_ID_KEY] = np.array(run_id)
        tmp_masks = out_masks.with_name(f".{out_masks.stem}.tmp.npz")
        np.savez_compressed(tmp_masks, **packed)
        tmp_masks.replace(out_masks)
    elif out_masks.exists():
        out_masks.unlink(missing_ok=True)
    tmp_boxes.replace(out_boxes)
    return {"people_count": len(people), "boxes_file": str(out_boxes), "run_id": run_id}


UI_DEFAULT_THRESHOLD = 0.8


def sensitivity_knobs(ui_threshold: Any) -> dict[str, float]:
    """Map the UI's person detection threshold onto SAM 3.1's score for STARTING a new person track.

    The UI default (0.8) leaves SAM 3.1 exactly at its own defaults (`new_det_thresh` 0.65, detections kept above
    0.4). Lower UI values let weaker detections start a track (more people picked up in dark / blurred plates, more
    risk of ghosts); higher values demand stronger detections."""
    try:
        t = float(ui_threshold)
    except (TypeError, ValueError):
        return {}
    if abs(t - UI_DEFAULT_THRESHOLD) < 1e-6:
        return {}
    new_det = float(np.clip(0.65 + (t - UI_DEFAULT_THRESHOLD) * 0.75, 0.30, 0.80))
    return {"new_det_thresh": round(new_det, 3), "score_threshold_detection": round(min(0.40, new_det - 0.10), 3)}


_OOM_MARKERS = ("out of memory", "cuda driver error: device not ready")
GROUNDING_BATCH_STEPS = (4, 2, 1)


def _is_oom(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "OutOfMemoryError" or any(m in str(cur).lower() for m in _OOM_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def run_track_people(
    job: dict,
    get_predictor: Callable[[], Any],
    emit: Callable[..., None] | None = None,
    free_memory: Callable[[], None] | None = None,
) -> dict:
    """Worker job `track_people`: whole-shot person masks + identities written in the product contract.

    job = {"frames": [{"frame": int, "path": str}, ...], "grade": dict | None,
           "output_boxes": path, "output_masks_npz": path, optional "knobs": {...}, "bidirectional": bool}
    """
    import tempfile

    def _emit(event: str, **payload: Any) -> None:
        if emit is not None:
            emit(event, **payload)

    frames_spec = job["frames"]
    shot_frames = [int(it["frame"]) for it in frames_spec]
    frame_paths = [str(it["path"]) for it in frames_spec]
    out_boxes = Path(job["output_boxes"])
    out_masks = Path(job["output_masks_npz"])
    n_frames = len(shot_frames)
    grade_dict = job.get("grade") or {}
    lut = None
    if grade_dict:
        from shotspace.io.plate_grade import GradeParams, build_uint8_lut

        g_obj = GradeParams.from_dict(grade_dict)
        if not g_obj.is_identity:
            lut = build_uint8_lut(g_obj)
    if n_frames == 0:
        return write_outputs(TrackStore(), shot_frames, (0, 0), out_boxes, out_masks, grade_dict)

    pred = get_predictor()
    patch_cpu_state_merge(pred)
    knobs = {**PEOPLE_KNOBS, **sensitivity_knobs(job.get("threshold")), **(job.get("knobs") or {})}
    bidirectional = bool(job.get("bidirectional", BIDIRECTIONAL_DEFAULT)) and n_frames >= 3
    n_pass = 2 if bidirectional else 1
    chunks = plan_chunks(n_frames, int(job.get("chunk_frames") or CHUNK_FRAMES), int(job.get("chunk_overlap") or CHUNK_OVERLAP))
    total_steps = n_pass * sum(e - s0 for s0, e in chunks)
    stats: dict[str, Any] = {"chunks": [list(c) for c in chunks], "bidirectional": bidirectional}
    t0 = time.time()
    done_steps = [0]
    prev = apply_knobs(pred.model, knobs)
    try:
        with tempfile.TemporaryDirectory(prefix="shotspace_people_") as td:
            _emit("stage", stage="sam3_person_track", message=f"人物检测：正在检测并跟踪画面中的人物（共 {n_frames} 帧）...")
            all_dir = Path(td) / "all"
            work_w, work_h, width, height = prepare_frame_dir(frame_paths, all_dir, lut=lut)
            stats["work_size"] = [work_w, work_h]
            steps = [b for b in GROUNDING_BATCH_STEPS if b <= int(pred.model.batched_grounding_batch_size)] or [1]

            prompt = str(job.get("prompt") or "auto").strip()
            if prompt.lower() == "auto":
                prompt, stats["prompt_probe"] = choose_prompt(pred, all_dir, n_frames)
                if prompt != PROMPT_TEXT:
                    _emit(
                        "warning",
                        message=f"按“{PROMPT_TEXT}”找到的人明显偏少，已自动改按“{prompt}”来识别人物",
                    )
            stats["prompt"] = prompt

            def _one_pass(frames_dir: Path, n_local: int, reverse: bool) -> tuple[TrackStore, dict]:
                base = done_steps[0]
                last = [0]

                def _progress(done: int, total: int) -> None:
                    if done == total or done - last[0] >= 4:
                        last[0] = done
                        _emit("progress", stage="sam3_person_track", done=base + done, total=total_steps)

                while True:
                    try:
                        res_pass = collect_pass(pred, frames_dir, n_local, reverse=reverse, on_progress=_progress, prompt=prompt)
                        done_steps[0] = base + n_local
                        return res_pass
                    except RuntimeError as exc:
                        if not _is_oom(exc):
                            raise
                        if free_memory is not None:
                            free_memory()
                        cur_b = int(pred.model.batched_grounding_batch_size)
                        smaller = [b for b in steps if b < cur_b]
                        if not smaller:
                            raise RuntimeError(
                                f"显存不足：人物跟踪即使每次只处理一帧仍然放不下（共 {n_frames} 帧），请关闭其他占用显卡的任务后重试"
                            ) from exc
                        pred.model.batched_grounding_batch_size = smaller[0]
                        stats["grounding_batch_after_oom"] = smaller[0]
                        _emit("warning", message="显存不足，已自动减少每次处理的帧数后重试")

            store: TrackStore | None = None
            chunk_stats: list[dict] = []
            dropped_at_cap = 0
            for ci, (c0, c1) in enumerate(chunks):
                if len(chunks) == 1:
                    cdir = all_dir
                else:
                    cdir = Path(td) / f"chunk_{ci:03d}"
                    cdir.mkdir()
                    for k, fi in enumerate(range(c0, c1)):
                        link_file((all_dir / f"{fi:05d}.jpg").resolve(), cdir / f"{k:05d}.jpg")
                    _emit(
                        "stage",
                        stage="sam3_person_track",
                        message=f"人物检测：正在检测并跟踪画面中的人物（第 {ci + 1}/{len(chunks)} 段，共 {n_frames} 帧）...",
                    )
                n_local = c1 - c0
                st_c, st_f = _one_pass(cdir, n_local, False)
                cs: dict[str, Any] = {"range": [c0, c1], "forward": st_f}
                dropped_at_cap += int(st_f.get("new_people_dropped_at_cap") or 0)
                if bidirectional and n_local >= 3:
                    if len(chunks) == 1:
                        _emit("stage", stage="sam3_person_track", message=f"人物检测：正在反向跟踪并核对人物身份（共 {n_frames} 帧）...")
                    st_b_store, st_b = _one_pass(cdir, n_local, True)
                    cs["backward"] = st_b
                    probe = AppearanceProbe(lambda fi, _d=cdir: str(_d / f"{fi:05d}.jpg"))
                    st_c, cs["merge"] = merge_passes(st_c, st_b_store, probe=probe)
                st_c = st_c.shifted(c0)
                if store is None:
                    store = st_c
                else:
                    store, cs["stitch"] = stitch_chunks(store, st_c, c0, chunks[ci - 1][1])
                chunk_stats.append(cs)
            assert store is not None
            stats["passes"] = chunk_stats
            if dropped_at_cap:
                stats["new_people_dropped_at_cap"] = dropped_at_cap
                _emit(
                    "warning",
                    message=f"画面中的人数超过了一次能跟踪的上限（{int(pred.model.max_num_objects)} 人），多出的人没有被跟踪，可以用「手动添加框」补上",
                )
    finally:
        apply_knobs(pred.model, prev)
    stats["trimmed_unconfirmed_entries"] = trim_unconfirmed_ends(store)
    cuts, stray = split_impossible_jumps(store)
    stats["cuts_at_impossible_jumps"], stats["stray_frames_removed"] = cuts, stray
    dropped = prune_tracks(store, n_frames)
    stats["dropped_short_tracks"] = dropped
    stats["seconds"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        stats["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    try:
        import resource

        stats["peak_ram_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    except Exception:
        pass
    res = write_outputs(
        store, shot_frames, (width, height), out_boxes, out_masks, grade_dict, extra={"stats": stats}, work_size=(work_w, work_h)
    )
    res["stats"] = stats
    return res
