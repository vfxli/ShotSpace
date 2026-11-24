import {
  DetectedPerson,
  ManualBoxKeyframe,
  ManualEvaluatedFrameStatus,
  ManualKeyframeStatus,
  ManualPersonTrack,
} from "../types";

export function normalizeManualTrack(raw: ManualPersonTrack): ManualPersonTrack {
  const kfByFrame = new Map<number, ManualBoxKeyframe>();
  for (const kf of raw.keyframes || []) {
    if (!kf || typeof kf.frame !== "number" || !Number.isFinite(kf.frame)) continue;
    const fnum = Math.round(kf.frame);
    const st: ManualKeyframeStatus =
      kf.status === "occluded" || kf.status === "off" ? kf.status : "active";
    if (st === "active") {
      if (!Array.isArray(kf.box) || kf.box.length < 4) continue;
      const [x1, y1, x2, y2] = kf.box.map(Number);
      if ([x1, y1, x2, y2].some((v) => !Number.isFinite(v))) continue;
      const lx1 = Math.min(x1, x2);
      const lx2 = Math.max(x1, x2);
      const ly1 = Math.min(y1, y2);
      const ly2 = Math.max(y1, y2);
      if (lx2 - lx1 < 4 || ly2 - ly1 < 6) continue;
      kfByFrame.set(fnum, {
        frame: fnum,
        status: "active",
        box: [
          Math.round(lx1 * 100) / 100,
          Math.round(ly1 * 100) / 100,
          Math.round(lx2 * 100) / 100,
          Math.round(ly2 * 100) / 100,
        ],
      });
    } else {
      kfByFrame.set(fnum, { frame: fnum, status: st });
    }
  }
  const sortedFrames = Array.from(kfByFrame.keys()).sort((a, b) => a - b);
  return {
    id: Math.max(1, Math.round(raw.id || 101)),
    name: (raw.name || `手动 #${raw.id || 101}`).trim() || `手动 #${raw.id || 101}`,
    enabled: raw.enabled !== false,
    keyframes: sortedFrames.map((f) => kfByFrame.get(f)!),
  };
}

export function normalizeManualPeopleList(list: ManualPersonTrack[] | undefined | null): ManualPersonTrack[] {
  if (!Array.isArray(list) || list.length === 0) return [];
  const seen = new Set<number>();
  let nextFallback = 101;
  const out: ManualPersonTrack[] = [];
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    let pid = Math.round(Number(item.id) || 0);
    if (pid <= 0 || seen.has(pid)) {
      while (seen.has(nextFallback)) nextFallback += 1;
      pid = nextFallback;
    }
    seen.add(pid);
    nextFallback = Math.max(nextFallback, pid + 1);
    out.push(normalizeManualTrack({ ...item, id: pid }));
  }
  return out;
}

export function evaluateManualPersonTrack(
  track: ManualPersonTrack,
  shotFrames: number[]
): {
  boxes: Record<string, [number, number, number, number]>;
  statuses: Record<number, ManualEvaluatedFrameStatus>;
  frameCount: number;
} {
  const norm = normalizeManualTrack(track);
  const kfs = norm.keyframes;
  const statuses: Record<number, ManualEvaluatedFrameStatus> = {};
  const boxes: Record<string, [number, number, number, number]> = {};

  if (norm.enabled === false || kfs.length === 0) {
    for (const f of shotFrames) statuses[f] = "inactive";
    return { boxes, statuses, frameCount: 0 };
  }

  const m = kfs.length;
  let idx = -1;
  for (const rawF of shotFrames) {
    const f = Math.round(rawF);
    while (idx + 1 < m && kfs[idx + 1].frame <= f) {
      idx += 1;
    }
    if (idx < 0) {
      statuses[f] = "inactive";
      continue;
    }
    const curKf = kfs[idx];
    if (curKf.status === "occluded") {
      statuses[f] = "occluded";
      continue;
    }
    if (curKf.status === "off") {
      statuses[f] = "off";
      continue;
    }
    const b0 = curKf.box!;
    if (f === curKf.frame) {
      boxes[String(f)] = [
        Math.round(b0[0] * 100) / 100,
        Math.round(b0[1] * 100) / 100,
        Math.round(b0[2] * 100) / 100,
        Math.round(b0[3] * 100) / 100,
      ];
      statuses[f] = "key_active";
    } else if (idx === m - 1) {
      boxes[String(f)] = [
        Math.round(b0[0] * 100) / 100,
        Math.round(b0[1] * 100) / 100,
        Math.round(b0[2] * 100) / 100,
        Math.round(b0[3] * 100) / 100,
      ];
      statuses[f] = "hold_active";
    } else {
      const nxtKf = kfs[idx + 1];
      if (nxtKf.status === "active" && nxtKf.box) {
        const b1 = nxtKf.box;
        const span = Math.max(1, nxtKf.frame - curKf.frame);
        const alpha = (f - curKf.frame) / span;
        boxes[String(f)] = [
          Math.round(((1 - alpha) * b0[0] + alpha * b1[0]) * 100) / 100,
          Math.round(((1 - alpha) * b0[1] + alpha * b1[1]) * 100) / 100,
          Math.round(((1 - alpha) * b0[2] + alpha * b1[2]) * 100) / 100,
          Math.round(((1 - alpha) * b0[3] + alpha * b1[3]) * 100) / 100,
        ];
        statuses[f] = "interp_active";
      } else {
        boxes[String(f)] = [
          Math.round(b0[0] * 100) / 100,
          Math.round(b0[1] * 100) / 100,
          Math.round(b0[2] * 100) / 100,
          Math.round(b0[3] * 100) / 100,
        ];
        statuses[f] = "hold_active";
      }
    }
  }

  return { boxes, statuses, frameCount: Object.keys(boxes).length };
}

export function buildManualDetectedPeople(
  manualPeople: ManualPersonTrack[],
  shotFrames: number[]
): {
  people: DetectedPerson[];
  statusesByPerson: Record<number, Record<number, ManualEvaluatedFrameStatus>>;
} {
  const people: DetectedPerson[] = [];
  const statusesByPerson: Record<number, Record<number, ManualEvaluatedFrameStatus>> = {};
  for (const trk of normalizeManualPeopleList(manualPeople)) {
    const ev = evaluateManualPersonTrack(trk, shotFrames);
    statusesByPerson[trk.id] = ev.statuses;
    if (trk.enabled === false || ev.frameCount === 0) continue;
    let prom = 0;
    for (const b of Object.values(ev.boxes)) {
      prom += Math.max(0, (b[2] - b[0]) * (b[3] - b[1]));
    }
    people.push({
      id: trk.id,
      name: trk.name || `手动 #${trk.id}`,
      is_manual: true,
      prominence: Math.round(prom * 10) / 10,
      frame_count: ev.frameCount,
      boxes: ev.boxes,
    });
  }
  return { people, statusesByPerson };
}

export function upsertManualKeyframe(
  track: ManualPersonTrack,
  kf: ManualBoxKeyframe
): ManualPersonTrack {
  const filtered = (track.keyframes || []).filter((k) => Math.round(k.frame) !== Math.round(kf.frame));
  return normalizeManualTrack({
    ...track,
    keyframes: [...filtered, kf],
  });
}

export function removeManualKeyframeAt(
  track: ManualPersonTrack,
  frame: number
): ManualPersonTrack {
  return normalizeManualTrack({
    ...track,
    keyframes: (track.keyframes || []).filter((k) => Math.round(k.frame) !== Math.round(frame)),
  });
}
