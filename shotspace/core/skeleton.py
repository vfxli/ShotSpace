"""CG skeleton standardization (Maya HumanIK / Mixamo naming + +X bone orientations)
and Linear Blend Skinning (LBS) point-cache baking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .names import unique
from .poses import UP, aim_frame, orthonormal

CG_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)
SIDES = ("l", "r")
SIDE_WORD = {"l": "Left", "r": "Right"}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

STANDARD_PART_NAMES: dict[str, str] = {
    "hips": "Hips",
    "head": "Head",
    "jaw": "Jaw",
    **{
        f"{s}.{p}": f"{SIDE_WORD[s]}{n}"
        for s in SIDES
        for p, n in (
            ("clavicle", "Shoulder"),
            ("upperarm", "Arm"),
            ("forearm", "ForeArm"),
            ("hand", "Hand"),
            ("thigh", "UpLeg"),
            ("shin", "Leg"),
            ("foot", "Foot"),
            ("toe", "ToeBase"),
            ("eye", "Eye"),
        )
    },
}
CHAIN_STEM = {"spine": "Spine", "neck": "Neck"}

_PREFIX = re.compile(r"^(?i:mixamorig\d*|biped\d*|bip\d*|cc_base|def|org|mch|jnt|jt|bn|bone|rig|sk|character\d*|body_world)")
_HELPER = {"twist", "roll", "bend", "helper", "ribbon", "corrective", "offset", "aux", "ik", "fk", "pole", "ctrl", "proc"}
_TWIST = _HELPER - {"bend"}
_END = {"end", "top", "nub", "site", "tip", "effector", "null"}
_FINGER = {"thumb", "index", "middle", "ring", "pinky", "pinkie", "finger", "fingers"}
_PALM = {"metacarpal", "metacarpals", "carpal", "palm", "meta"}
_FINGER_WORDS = {
    "thumb": {"thumb"},
    "index": {"index", "fore", "point"},
    "middle": {"middle", "mid"},
    "ring": {"ring"},
    "pinky": {"pinky", "pinkie", "little"},
}
_PHALANX = {"proximal", "intermediate", "middle", "mid", "distal"}
_LIMB = {
    "fore", "forearm", "lower", "lowerarm", "upper", "upperarm", "up", "low", "elbow", "knee",
    "shin", "calf", "leg", "thigh", "foot", "ankle", "hand", "wrist", "arm", "shoulder",
    "clavicle", "collar", "toe", "ball", "shldr", "upleg", "lowleg",
}


def _strip_prefix(name: str) -> str:
    m = _PREFIX.match(name)
    if m is None:
        return name
    head, rest = m.group(0), name[m.end():]
    if rest[:1] in ("_", "-", ".", " "):
        return rest[1:]
    starts_word = rest[:1].isupper() and (head[-1].islower() or head[-1].isdigit() or rest[1:2].islower())
    return rest if starts_word else name


def _tokens(name: str) -> list[str]:
    name = name.rsplit(":", 1)[-1]
    name = _strip_prefix(name) or name
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)
    return [p.lower() for p in parts]


def _side(words: list[str]) -> str | None:
    left = any(w in ("left", "l", "lf", "lft") for w in words)
    right = any(w in ("right", "r", "rt", "rgt") for w in words)
    return "l" if left and not right else "r" if right and not left else None


def _kind(words: list[str]) -> str | None:
    w = set(words)
    if w & _HELPER or w & _END:
        return None
    if w & {"hand", "wrist"} and not w & (_FINGER | _PALM):
        return "hand"
    if w & {"foot", "ankle"} and not w & {"toe", "toes", "ball"}:
        return "foot"
    if "head" in w:
        return "head"
    if "jaw" in w:
        return "jaw"
    if w & {"eye", "eyes"} and not w & {"lid", "brow", "ball"}:
        return "eye"
    if w & {"hips", "pelvis"} or (w == {"hip"}) or (w & {"hip"} and _side(words) is None):
        return "hips"
    return None


def _ancestors(j: int, parents) -> list[int]:
    out = []
    while j >= 0:
        out.append(j)
        j = int(parents[j])
    return out


def _path(top: int, bottom: int, parents) -> list[int]:
    up = _ancestors(bottom, parents)
    return list(reversed(up[: up.index(top)])) if top in up else []


def _base(words: list[str]) -> frozenset[str]:
    return frozenset(w for w in words if w not in _HELPER and not w.isdigit())


def _segments(path: list[int], words: list[list[str]]) -> list[list[int]]:
    out: list[list[int]] = []
    for i, j in enumerate(path):
        base = _base(words[j])
        variant = bool(out) and base >= (first := _base(words[out[-1][0]])) and not (base - first) & _LIMB
        twist = bool(set(words[j]) & _TWIST) and not (i + 1 < len(path) and _base(words[path[i + 1]]) == base)
        if out and (variant or twist):
            out[-1].append(j)
        else:
            out.append([j])
    return out


def _common_above(parents, joints) -> int:
    joints = [int(j) for j in joints]
    if not joints:
        return -1
    others = [set(_ancestors(j, parents)) for j in joints[1:]]
    return next((a for a in _ancestors(joints[0], parents) if all(a in o for o in others)), -1)


def _guess_parts_by_names(names: list[str], parents) -> dict:
    parents = np.asarray(parents, dtype=np.int64)
    words = [_tokens(n) for n in names]
    kinds = [_kind(w) for w in words]
    sides = [_side(w) for w in words]
    depth = [len(_ancestors(j, parents)) for j in range(len(names))]

    def top_most(kind: str, s: str | None = None) -> int | None:
        found = [j for j in range(len(names)) if kinds[j] == kind and (s is None or sides[j] == s)]
        return min(found, key=lambda j: depth[j]) if found else None

    def limb(path: list[int]) -> list[int]:
        return [seg[0] for seg in _segments(path, words)]

    def plain(path: list[int]) -> list[int]:
        return [j for j in path if not set(words[j]) & _HELPER] or list(path)

    out: dict = {}
    feet = {s: top_most("foot", s) for s in SIDES}
    hips = top_most("hips")
    if all(f is not None for f in feet.values()):
        marks = [j for j in (top_most("head"), top_most("hand", "l"), top_most("hand", "r")) if j is not None]
        top = _common_above(parents, marks) if len(marks) >= 2 else -1
        fork = _common_above(parents, [*feet.values(), *([top] if top >= 0 else [])])
        if hips is None or (fork >= 0 and hips not in _ancestors(fork, parents)):
            hips = fork if fork >= 0 else hips

    if hips is not None:
        out["hips"] = hips
        for s, foot in feet.items():
            leg = limb(_path(hips, foot, parents)) if foot is not None else []
            if len(leg) >= 3:
                out[f"{s}.thigh"], out[f"{s}.shin"], out[f"{s}.foot"] = leg[-3:]
                below = [c for c in range(len(names)) if foot in _ancestors(c, parents)[1:] and not set(words[c]) & (_END | _HELPER)]
                named_toes = [c for c in below if set(words[c]) & {"toe", "toes", "ball"}]
                toes = named_toes or [c for c in below if parents[c] == foot]
                if toes:
                    has_kids = {int(parents[c]) for c in range(len(names))}
                    out[f"{s}.toe"] = min(toes, key=lambda c: (depth[c], c not in has_kids))
        head = top_most("head")
        hands = [h for h in (top_most("hand", s) for s in SIDES) if h is not None]
        if head is not None:
            trunk = _path(hips, head, parents)[:-1]
            out["head"] = head
        elif len(hands) == 2:
            shared = [j for j in _ancestors(hands[0], parents) if j in _ancestors(hands[1], parents)]
            trunk = _path(hips, shared[0], parents) if shared else []
        else:
            trunk = []
        spine = plain(trunk)
        for s in SIDES:
            hand = top_most("hand", s)
            if hand is None or not spine:
                continue
            branch = next((j for j in _ancestors(hand, parents) if j in spine or j == hips), None)
            arm = limb(_path(branch, hand, parents)) if branch is not None else []
            if len(arm) >= 3:
                out[f"{s}.upperarm"], out[f"{s}.forearm"], out[f"{s}.hand"] = arm[-3:]
                if len(arm) >= 4:
                    out[f"{s}.clavicle"] = arm[-4]
                if branch in spine and "chest" not in out:
                    while "neck" in words[branch] and trunk.index(branch) > 0:
                        branch = trunk[trunk.index(branch) - 1]
                    out["chest"] = branch
        if "chest" in out:
            k = trunk.index(out["chest"])
            out["spine"], out["neck"] = plain(trunk[:k]), plain(trunk[k + 1 :])
        elif spine:
            out["spine"], out["neck"] = spine, []

    if (jaw := top_most("jaw")) is not None:
        out["jaw"] = jaw
    for s in SIDES:
        if (eye := top_most("eye", s)) is not None:
            out[f"{s}.eye"] = eye

    # Finger chains
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)
    skip = _HELPER | _END | _PALM

    def subtree_size(j: int) -> int:
        return 1 + sum(subtree_size(c) for c in kids.get(j, []))

    def finger_roots(j: int) -> list[int]:
        res = []
        for c in kids.get(j, []):
            res += finger_roots(c) if set(words[c]) & skip else [c]
        return res

    def which_finger(chain: list[int]) -> tuple[int, int] | None:
        for i, j in enumerate(chain):
            hit = {k for k, hints in enumerate(_FINGER_WORDS.values()) if set(words[j]) & hints}
            if len(hit) > 1:
                hit -= {k for k, (_, hints) in enumerate(_FINGER_WORDS.items()) if set(words[j]) & hints <= _PHALANX}
            if len(hit) == 1:
                return hit.pop(), i
        return None

    for s in SIDES:
        hand = out.get(f"{s}.hand")
        if hand is None:
            continue
        found_f: dict[int, list[int]] = {}
        for r in finger_roots(hand):
            chain = [r]
            while nxt := [c for c in kids.get(chain[-1], []) if not set(words[c]) & skip]:
                chain.append(max(nxt, key=subtree_size))
            while len(chain) > 3 and chain[-1] not in kids:
                chain.pop()
            got = which_finger(chain)
            if got is None:
                continue
            fk, start = got
            chain = chain[start:]
            if len(chain) > len(found_f.get(fk, [])):
                found_f[fk] = chain
        for fk, chain in found_f.items():
            out[f"{s}.{FINGERS[fk]}"] = chain

    return out


def _chain_name(part: str, k: int) -> str:
    s, _, name = part.rpartition(".")
    if name in FINGERS:
        return f"{SIDE_WORD[s]}Hand{name.title()}{k + 1}"
    stem = CHAIN_STEM.get(part, re.sub(r"[^A-Za-z0-9]", "", part.title()) or "Bone")
    return stem if k == 0 else f"{stem}{k}"


def cg_bone_names(names: list[str], parents) -> list[str]:
    """Map ML skeleton joint names (e.g. MHR 127 joints) to Maya HumanIK / Mixamo standard CG bone names."""
    raw = [str(n) for n in names]
    parts = _guess_parts_by_names(raw, parents)
    named: dict[int, str] = {}
    for part, where in parts.items():
        if isinstance(where, list):
            for k, j in enumerate(where):
                named[int(j)] = _chain_name(part, k)
        elif part == "chest":
            named[int(where)] = _chain_name("spine", len(parts.get("spine") or []))
        elif part in STANDARD_PART_NAMES:
            named[int(where)] = STANDARD_PART_NAMES[part]

    out = list(raw)
    for j, n in named.items():
        out[j] = n

    children: dict[int, list[int]] = {}
    for i, up in enumerate(parents):
        children.setdefault(int(up), []).append(i)
    for i, n in enumerate(out):
        if n != raw[i] or i in children:
            continue
        up = int(parents[i])
        if up >= 0 and out[up] != raw[up] and len(children.get(up, ())) == 1:
            out[i] = f"{out[up]}_End"

    # Deduplicate keeping recognized standard names first
    order = sorted(range(len(out)), key=lambda i: (i not in named, i))
    taken, seen, final = set(out), set(), list(out)
    for i in order:
        n = out[i]
        if n in seen:
            n = unique(n, taken)
            taken.add(n)
        seen.add(n)
        final[i] = n
    return final


def cg_orientations(parents, bind_world: np.ndarray, anim_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize joint orientations to Maya's +X-down-the-bone convention while keeping skinning
    matrices `anim_world @ inv(bind_world)` bit-identical."""
    parents = np.asarray(parents, dtype=np.int64)
    bind = np.asarray(bind_world, dtype=np.float64)
    pos = bind[:, :3, 3]
    kids: dict[int, list[int]] = {}
    for j, up in enumerate(parents):
        kids.setdefault(int(up), []).append(j)
    old = orthonormal(bind[:, :3, :3])
    new = old.copy()
    for j in range(len(parents)):
        child = kids.get(j, ())
        bone = pos[child[0]] - pos[j] if len(child) == 1 else np.zeros(3, dtype=np.float64)
        length = float(np.linalg.norm(bone))
        if length > 1e-9:
            aim = bone / length
            ref = UP if abs(float(aim @ CG_FORWARD)) > abs(float(aim @ UP)) else CG_FORWARD
            new[j] = aim_frame(aim, ref)
        else:
            new[j] = new[parents[j]] if parents[j] >= 0 else np.eye(3, dtype=np.float64)
    fix = np.zeros((len(parents), 4, 4), dtype=np.float64)
    fix[:, 3, 3] = 1.0
    fix[:, :3, :3] = np.swapaxes(old, 1, 2) @ new
    return bind @ fix, np.asarray(anim_world, dtype=np.float64) @ fix


@dataclass
class SkinnedCharacter:
    """Standard CG skinned character in ShotSpace world/camera coordinates (cm, +Y up, column-vector 4x4).

    `frames` spans the shot sequence; `visible` (`[F]` int32 `1`/`0`) marks genuinely solved reliable frames (`1`)
    versus undetected / border-fly-away frozen frames (`0`, which hold `anchor_frames` stationary and auto-hide in
    viewport and `.usd` / `.fbx` / `.abc` exports).
    """

    name: str
    person_id: int
    frames: list[int]
    joint_names: list[str]
    parents: np.ndarray  # [J] int64
    bind_world: np.ndarray  # [J, 4, 4] float64
    anim_world: np.ndarray  # [F, J, 4, 4] float64
    rest_points: np.ndarray  # [V, 3] float32/float64
    faces: np.ndarray  # [P, 3] int32
    joint_indices: np.ndarray  # [V, K] int32
    joint_weights: np.ndarray  # [V, K] float32
    uv: np.ndarray | None = None  # [U, 2] float32
    uv_faces: np.ndarray | None = None  # [P, 3] int32
    visible: np.ndarray | None = None  # [F] int32 (1 = visible, 0 = frozen & auto-hidden)
    anchor_frames: np.ndarray | None = None  # [F] int64 valid anchor frame number per frame
    custom_data: dict = field(default_factory=dict)

    @property
    def visible_mask(self) -> np.ndarray:
        if self.visible is None or len(self.visible) != len(self.frames):
            return np.ones(len(self.frames), dtype=bool)
        return np.asarray(self.visible).reshape(-1) > 0

    @property
    def solved_frames(self) -> list[int]:
        mask = self.visible_mask
        return [int(f) for f, ok in zip(self.frames, mask) if ok]

    @classmethod
    def from_mhr_npz(cls, person_id: int, data: dict | np.lib.npyio.NpzFile) -> SkinnedCharacter:
        raw_names = [str(n) for n in data["joint_names"]]
        parents = np.asarray(data["parents"], dtype=np.int64)
        bind_raw = np.asarray(data["bind_world"], dtype=np.float64)
        anim_raw = np.asarray(data["joint_world"], dtype=np.float64)
        names = cg_bone_names(raw_names, parents)
        bind, anim = cg_orientations(parents, bind_raw, anim_raw)
        frames = [int(f) for f in np.asarray(data["frames"]).reshape(-1)]
        vis = (
            np.asarray(data["visible"], dtype=np.int32).reshape(-1)
            if "visible" in data
            else np.ones(len(frames), dtype=np.int32)
        )
        anc = (
            np.asarray(data["anchor_frames"], dtype=np.int64).reshape(-1)
            if "anchor_frames" in data
            else np.asarray(frames, dtype=np.int64)
        )
        n_solved = int(np.sum(vis > 0))
        has_real_cam = bool(int(np.asarray(data["has_real_camera"]).reshape(-1)[0])) if "has_real_camera" in data else False
        smooth_val = float(np.asarray(data["smoothing_strength"]).reshape(-1)[0]) if "smoothing_strength" in data else 0.5
        return cls(
            name=f"person_{person_id:02d}",
            person_id=int(person_id),
            frames=frames,
            joint_names=names,
            parents=parents,
            bind_world=bind,
            anim_world=anim,
            rest_points=np.asarray(data["rest_vertices"], dtype=np.float32),
            faces=np.asarray(data["faces"], dtype=np.int32),
            joint_indices=np.asarray(data["skin_indices"], dtype=np.int32),
            joint_weights=np.asarray(data["skin_weights"], dtype=np.float32),
            uv=np.asarray(data["uv"], dtype=np.float32) if "uv" in data else None,
            uv_faces=np.asarray(data["uv_faces"], dtype=np.int32) if "uv_faces" in data else None,
            visible=vis,
            anchor_frames=anc,
            custom_data={
                "lab2shot:person_id": int(person_id),
                "shotspace:frames_solved": n_solved,
                "shotspace:has_real_camera": has_real_cam,
                "shotspace:smoothing_strength": round(smooth_val, 4),
                "shotspace:world_solved": bool("solver_version" in data and int(np.asarray(data["solver_version"]).reshape(-1)[0]) >= 2),
                "shotspace:body_scale": float(np.asarray(data["body_scale"]).reshape(-1)[0]) if "body_scale" in data else 1.0,
            },
        )


def bake_lbs_points(
    rest_points: np.ndarray,
    bind_world: np.ndarray,
    anim_world: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    chunk_frames: int = 32,
) -> np.ndarray:
    """Vectorized Linear Blend Skinning evaluation -> per-frame world vertices [F, V, 3] float32.
    Used for Alembic (.abc) deforming mesh export and WebGL >4-influence fallback."""
    pts = np.asarray(rest_points, dtype=np.float64)  # [V, 3]
    inv_bind = np.linalg.inv(np.asarray(bind_world, dtype=np.float64))  # [J, 4, 4]
    anim = np.asarray(anim_world, dtype=np.float64)  # [F, J, 4, 4]
    skin_mats = anim @ inv_bind[None]  # [F, J, 4, 4]

    idx = np.asarray(joint_indices, dtype=np.int64)  # [V, K]
    wts = np.asarray(joint_weights, dtype=np.float64)  # [V, K]
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)  # [V, 4]

    f_total = len(anim)
    out = np.empty((f_total, len(pts), 3), dtype=np.float32)
    for lo in range(0, f_total, chunk_frames):
        hi = min(lo + chunk_frames, f_total)
        # Blended [B, V, 3, 4] skinning matrix per vertex
        sm = skin_mats[lo:hi, :, :3, :]  # [B, J, 3, 4]
        blended = np.zeros((hi - lo, len(pts), 3, 4), dtype=np.float64)
        for k in range(idx.shape[1]):
            wk = wts[:, k]
            nz = wk > 0
            if not np.any(nz):
                continue
            blended[:, nz] += sm[:, idx[nz, k]] * wk[None, nz, None, None]
        out[lo:hi] = np.einsum("bvrc,vc->bvr", blended, pts_h).astype(np.float32)
    return out
