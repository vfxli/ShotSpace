"""A track that jumps to another person is cut into the people it really followed (v0.2.9).

In a dense crowd the tracker's number can move from one person to a neighbour, or come back on somebody else after
the person was hidden for a while.  The per-frame network estimate then does what no person can do: one frame the
body is here, the next used frame it is somewhere else.  Left alone, the whole-shot solve gives two different
people one body and lets it fly across the set.

This module looks at the per-frame network estimate of ONE track, placed in the world by the exact camera, and cuts
the track where that happens.  Only the direction ACROSS the line of sight is judged: there the estimate is good to
a few centimetres, while along the line of sight it wanders by itself (which is what the whole-shot solve
steadies), so a step in distance alone proves nothing.

Two situations, two questions:
  - two used frames follow each other directly: is the body after the step where the frames before it lead to?
    (a few frames on each side vote, so one wild frame decides nothing);
  - used frames with unseen time between them (the person was hidden, or the frames were unusable): on top of
    that, for every unseen second, the distance of a brisk walk - or of twice the track's own speed - is allowed.
    Somebody who walked on while hidden can be anywhere within that reach, so only a body beyond it is a cut.

Nothing here needs the pictures or the network again: it works on the cached per-frame outputs.  numpy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CV_TO_GL = np.array([1.0, -1.0, -1.0])


@dataclass
class SplitConfig:
    window: int = 5               # used frames on each side that vote on where the body is
    side_span_s: float = 0.5      # ...and how far back / forward in time a side may reach
    min_jump_cm: float = 50.0     # nobody is cut for less than this (across the line of sight): half a body width
    noise_factor: float = 5.0     # ...nor for less than this many times the scatter of the track's own estimate
    max_speed_cm_s: float = 1200.0  # a side's own speed estimate is never believed beyond this
    walk_cm_s: float = 250.0      # across unseen time: anybody may have moved this fast (a brisk walk, a jog)
    speed_factor: float = 2.0     # ...or this many times as fast as the track itself moves next to the gap
    speed_span_s: float = 1.0     # how much of the track next to the gap says how fast it moves
    min_part_s: float = 0.4       # a part shorter than this (and than 3 frames) is not a person: its frames are dropped
    rejoin_max_s: float = 1.5     # parts with dropped frames between them are one person again only across this much
    rejoin_cm_s: float = 100.0    # ...unseen time, and only this close to where the first part leads (per second)


def track_positions(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per frame of the track: where the network put the body, in the world (cm), and where the camera is (cm)."""
    t_cam = np.asarray(raw["pred_cam_t"], dtype=np.float64) * CV_TO_GL * 100.0  # body in the camera, GL axes, cm
    c2w = np.asarray(raw["cam_to_world"], dtype=np.float64)
    pos = np.einsum("tij,tj->ti", c2w[:, :3, :3], t_cam) + c2w[:, :3, 3]
    return pos, c2w[:, :3, 3].copy()


def _velocity(t: np.ndarray, p: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """Median of the slopes between all pairs of frames (it ignores a wild frame); fewer than four frames give none."""
    if len(t) < 4:
        return np.zeros(3)
    ii, jj = np.triu_indices(len(t), k=1)
    dt = t[jj] - t[ii]
    ok = dt > 1e-9
    if not ok.any():
        return np.zeros(3)
    vel = np.median((p[jj[ok]] - p[ii[ok]]) / dt[ok, None], axis=0)
    speed = float(np.linalg.norm(vel))
    return vel * (cfg.max_speed_cm_s / speed) if speed > cfg.max_speed_cm_s else vel


def _side(t: np.ndarray, p: np.ndarray, cfg: SplitConfig) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Robust summary of a few frames: (time, position, velocity, each frame's offset from that line)."""
    t0 = float(np.median(t))
    p0 = np.median(p, axis=0)
    vel = _velocity(t, p, cfg)
    return t0, p0, vel, p - (p0 + vel[None, :] * (t - t0)[:, None])


def _off_ray(vec: np.ndarray, ray: np.ndarray) -> np.ndarray:
    """The part of a vector (or of each row) that lies across the line of sight."""
    return vec - np.tensordot(vec, ray, axes=([-1], [0]))[..., None] * ray


def _voters(used: np.ndarray, last: int, first: int, t: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """The used frames that vote before position `last` (inclusive) and from position `first` on."""
    before = used[max(0, last - cfg.window + 1):last + 1]
    after = used[first:first + cfg.window]
    before = before[t[before] >= t[used[last]] - cfg.side_span_s]
    after = after[t[after] <= t[used[first]] + cfg.side_span_s]
    return before, after


def _lead(used: np.ndarray, lo: int, hi: int, t: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
    where = used[max(0, lo):max(0, hi)]
    return where[(t[where] >= t_from) & (t[where] <= t_to)]


def _lead_velocity(where: np.ndarray, t: np.ndarray, pos: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """How the track moves next to a step, from up to a second of it.  A handful of frames says nothing reliable
    about speed (their scatter would pass for motion): then the body is taken as standing."""
    if len(where) < 6 or float(t[where[-1]] - t[where[0]]) < 0.25:
        return np.zeros(3)
    return _velocity(t[where], pos[where], cfg)


def step_evidence(used: np.ndarray, last: int, first: int, t: np.ndarray, pos: np.ndarray, cam: np.ndarray, fps: float,
                  cfg: SplitConfig) -> dict | None:
    """What the track does between the used positions `last` and `first` (first > last; used frames between them,
    if any, are ignored), all across the line of sight and in cm:
      away    how far the body after the step is from where the body before it leads to (the smaller of predicting
              forward and predicting backward, each with that side's own motion),
      base    what the track's own scatter - or the least jump that counts - allows,
      unseen  seconds between the two frames beyond one frame step,
      speed   how fast the track itself moves next to the step (cm/s),
      own     the step between the two frames themselves.
    None when a side has too few frames to say where the body is."""
    before, after = _voters(used, last, first, t, cfg)
    a, b = used[last], used[first]
    ray = pos[a] - cam[a]
    ray = ray / max(float(np.linalg.norm(ray)), 1e-9)
    if len(before) < 2 or len(after) < 2:
        return None
    tb, pb, _vb, rb = _side(t[before], pos[before], cfg)
    ta, pa, _va, ra = _side(t[after], pos[after], cfg)
    off = _off_ray(np.concatenate([rb, ra]), ray)
    scatter = 1.4826 * float(np.median(np.linalg.norm(off, axis=1))) if len(off) >= 4 else 0.0
    v_in = _lead_velocity(_lead(used, last - 4 * cfg.window, last + 1, t, t[a] - cfg.speed_span_s, t[a]), t, pos, cfg)
    v_out = _lead_velocity(_lead(used, first, first + 4 * cfg.window, t, t[b], t[b] + cfg.speed_span_s), t, pos, cfg)
    dt = ta - tb
    forward = float(np.linalg.norm(_off_ray(pa - (pb + v_in * dt), ray)))
    backward = float(np.linalg.norm(_off_ray(pb - (pa - v_out * dt), ray)))
    return {
        "away": min(forward, backward),
        "base": max(cfg.min_jump_cm, cfg.noise_factor * scatter),
        "unseen": max(0.0, float(t[b] - t[a]) - 1.0 / float(fps)),
        "speed": max(float(np.linalg.norm(_off_ray(v_in, ray))), float(np.linalg.norm(_off_ray(v_out, ray)))),
        "own": float(np.linalg.norm(_off_ray(pos[b] - pos[a], ray))),
    }


def breaks(ev: dict | None, cfg: SplitConfig) -> bool:
    """The body after the step is NOT where a person can have gone: farther from where its own motion leads than the
    track's scatter allows, plus - for every second nobody saw - a brisk walk, or twice the track's own speed."""
    if ev is None:
        return False
    if ev["own"] < 0.5 * ev["away"]:
        return False  # a jump happens in ONE step; a body that drifts off over several frames has not changed person
    return ev["away"] > ev["base"] + max(cfg.walk_cm_s, cfg.speed_factor * ev["speed"]) * ev["unseen"]


def continues(ev: dict | None, cfg: SplitConfig) -> bool:
    """The body after the step IS where the body before it leads to (asked before two parts with dropped frames
    between them are made one person again - which needs evidence for, not just none against)."""
    if ev is None:
        return False
    return ev["unseen"] <= cfg.rejoin_max_s and ev["away"] <= ev["base"] + cfg.rejoin_cm_s * ev["unseen"]


def find_cuts(frames: np.ndarray, pos: np.ndarray, cam: np.ndarray, use: np.ndarray, fps: float, cfg: SplitConfig | None = None) -> list[int]:
    """Positions k (into the used frames) after which the track is cut: the body from used[k+1] on is not where
    the body up to used[k] can have gone."""
    cfg = cfg or SplitConfig()
    used = np.nonzero(use)[0]
    if len(used) < 4:
        return []
    t = np.asarray(frames, dtype=np.float64) / float(fps)
    found: list[tuple[float, int]] = []
    for k in range(len(used) - 1):
        ev = step_evidence(used, k, k + 1, t, pos, cam, fps, cfg)
        if breaks(ev, cfg):
            found.append((ev["own"], k))
    # Next to a real break its neighbouring steps look broken too (their voters straddle it): of steps that close
    # to each other, the one whose own step is largest is the break.
    cuts: list[int] = []
    radius = max(1, (cfg.window - 1) // 2)
    for _size, k in sorted(found, reverse=True):
        if all(abs(k - c) > radius for c in cuts):
            cuts.append(k)
    return sorted(cuts)


def split_track(raw: dict, use: np.ndarray, fps: float, cfg: SplitConfig | None = None) -> list[np.ndarray]:
    """The parts of one track that each follow one person, as masks over the track's frames (subsets of `use`, in
    order of time).  One mask equal to `use` means the track is one person throughout.

    1. cut where the body jumps (`find_cuts`);
    2. a part too short to be a person (`min_part_s`) is dropped - a few frames on a neighbour, or the rubbish
       between two people;
    3. parts on both sides of dropped frames that continue each other are one person again."""
    cfg = cfg or SplitConfig()
    use = np.asarray(use, dtype=bool)
    frames = np.asarray(raw["frames"]).reshape(-1).astype(np.int64)
    used = np.nonzero(use)[0]
    if len(used) < 4:
        return [use.copy()]
    pos, cam = track_positions(raw)
    cuts = find_cuts(frames, pos, cam, use, fps, cfg)
    if not cuts:
        return [use.copy()]
    t = frames.astype(np.float64) / float(fps)
    bounds = [0] + [c + 1 for c in cuts] + [len(used)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]  # positions into `used`
    min_frames = max(3, int(round(cfg.min_part_s * float(fps))))
    kept = [s for s in spans if s[1] - s[0] >= min_frames]
    if not kept:
        kept = [max(spans, key=lambda s: s[1] - s[0])]  # nothing is long: the longest part is the track
    merged: list[list[tuple[int, int]]] = [[kept[0]]]
    for nxt in kept[1:]:
        cur_end = merged[-1][-1][1]
        if nxt[0] > cur_end and continues(step_evidence(used, cur_end - 1, nxt[0], t, pos, cam, fps, cfg), cfg):
            merged[-1].append(nxt)  # frames were dropped between the two, and they continue each other
        else:
            merged.append([nxt])
    out = []
    for group in merged:
        mask = np.zeros(len(use), dtype=bool)
        for lo, hi in group:
            mask[used[lo:hi]] = True
        out.append(mask)
    return out


# What a track's cached network outputs hold besides one value per frame.
NOT_PER_FRAME = ("shot_frames", "is_manual", "image_size", "has_real_camera")


def people_of_track(raw: dict, use: np.ndarray, fps: float, cfg: SplitConfig | None = None) -> tuple[list[dict], int]:
    """The cached network outputs of every person the track really followed, in order of time, and how many used
    frames were given to nobody.

    A track that is one person comes back as it is (the very same dict).  Otherwise every part gets its own stretch
    of the track - all frames from its first used frame to its last, without the frames that were dropped or belong
    to another part - so that each can be solved, cached and solved again as a person of its own."""
    use = np.asarray(use, dtype=bool)
    parts = split_track(raw, use, fps, cfg)
    given = np.zeros(len(use), dtype=bool)
    for mask in parts:
        given |= mask
    dropped = int((use & ~given).sum())
    if len(parts) == 1 and dropped == 0:
        return [raw], 0
    n = len(use)
    out = []
    for mask in parts:
        idx = np.nonzero(mask)[0]
        keep = np.zeros(n, dtype=bool)
        keep[idx[0]:idx[-1] + 1] = True
        keep &= ~(use & ~mask)  # used frames inside the stretch that are not this person's
        out.append(
            {
                key: (val[keep] if (key not in NOT_PER_FRAME and isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n) else val)
                for key, val in raw.items()
            }
        )
    return out, dropped

