"""`shotspace.solver.track_split`: a track that jumps to another person is cut - and nothing else is.

Synthetic tracks: a body placed in the world, seen by a camera, with the scatter the per-frame network estimate
really has (a few centimetres across the line of sight, decimetres along it).  numpy only."""

from __future__ import annotations

import numpy as np

from shotspace.solver.track_split import NOT_PER_FRAME, SplitConfig, people_of_track, split_track

FPS = 24.0


def _camera(n: int, travel_cm_per_frame: float = 0.0, pan_deg_per_frame: float = 0.0) -> np.ndarray:
    """Camera-to-world per frame: at the origin looking down -Z, optionally travelling along X and panning."""
    c2w = np.repeat(np.eye(4)[None], n, axis=0)
    for i in range(n):
        a = np.radians(pan_deg_per_frame * i)
        c2w[i, :3, :3] = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
        c2w[i, 0, 3] = travel_cm_per_frame * i
    return c2w


def _track(frames, world_cm: np.ndarray, c2w: np.ndarray | None = None, seed: int = 0, across_cm: float = 4.0, along_cm: float = 40.0) -> dict:
    """Cached network outputs of a track whose body is at `world_cm` (per frame), with the estimate's scatter."""
    frames = np.asarray(frames, dtype=np.int64)
    n = len(frames)
    c2w = _camera(n) if c2w is None else c2w
    rng = np.random.default_rng(seed)
    cam = c2w[:, :3, 3]
    ray = world_cm - cam
    dist = np.linalg.norm(ray, axis=1, keepdims=True)
    ray = ray / dist
    side = np.cross(ray, np.array([0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    noisy = world_cm + ray * rng.normal(0.0, along_cm, (n, 1)) + side * rng.normal(0.0, across_cm, (n, 1)) + np.array([0.0, 1.0, 0.0]) * rng.normal(0.0, across_cm, (n, 1))
    in_cam_gl = np.einsum("tji,tj->ti", c2w[:, :3, :3], noisy - cam)  # world -> camera (rotation transposed)
    return {
        "frames": frames,
        "shot_frames": np.arange(int(frames[0]), int(frames[0]) + n, dtype=np.int64),   # same length as the track on purpose
        "is_manual": np.int32(0),
        "image_size": np.array([1920, 1080], dtype=np.int32),
        "has_real_camera": np.int32(1),
        "pred_cam_t": (in_cam_gl * np.array([1.0, -1.0, -1.0]) / 100.0).astype(np.float32),
        "cam_to_world": c2w.astype(np.float64),
        "bbox": np.zeros((n, 4), dtype=np.float32),
        "pred_keypoints_2d": np.zeros((n, 70, 2), dtype=np.float32),
    }


def _walk(n: int, start=(0.0, 0.0, -800.0), speed_cm_s=(140.0, 0.0, 0.0)) -> np.ndarray:
    t = np.arange(n)[:, None] / FPS
    return np.asarray(start)[None, :] + t * np.asarray(speed_cm_s)[None, :]


def _parts(raw: dict, use: np.ndarray | None = None) -> list[list[int]]:
    use = np.ones(len(raw["frames"]), dtype=bool) if use is None else use
    return [[int(f) for f in raw["frames"][m]] for m in split_track(raw, use, FPS)]


def test_one_person_is_never_cut() -> None:
    frames = np.arange(1001, 1121)
    for seed in range(8):
        # walking across the picture, standing, walking towards the camera - with the estimate's usual scatter
        for start, speed in (((-300.0, 0.0, -800.0), (140.0, 0.0, 0.0)), ((0.0, 0.0, -500.0), (0.0, 0.0, 0.0)), ((100.0, 0.0, -1500.0), (0.0, 0.0, 150.0))):
            raw = _track(frames, _walk(len(frames), start, speed), seed=seed)
            assert _parts(raw) == [frames.tolist()], (seed, start, speed)
    # far away the estimate scatters more across the line of sight too: still one person
    raw = _track(frames, _walk(len(frames), (0.0, 0.0, -3000.0), (100.0, 0.0, 0.0)), seed=3, across_cm=18.0, along_cm=200.0)
    assert _parts(raw) == [frames.tolist()]
    # a runner: 8 m/s across the picture
    raw = _track(frames, _walk(len(frames), (-1500.0, 0.0, -2000.0), (800.0, 0.0, 0.0)), seed=4)
    assert _parts(raw) == [frames.tolist()]
    # a step in DISTANCE alone (five metres along the line of sight, which the estimate does by itself) proves nothing
    world = _walk(len(frames), (0.0, 0.0, -1200.0), (0.0, 0.0, 0.0))
    world[60:, 2] -= 500.0
    assert _parts(_track(frames, world, seed=5)) == [frames.tolist()]
    # too short to judge
    assert _parts(_track(frames[:3], _walk(3))) == [frames[:3].tolist()]


def test_a_jump_to_a_neighbour_is_cut_where_it_happens() -> None:
    frames = np.arange(1001, 1121)
    for seed in range(8):
        world = _walk(len(frames), (-200.0, 0.0, -700.0), (60.0, 0.0, 0.0))
        world[60:, 0] += 90.0  # from frame 1061 on the track is on the person standing 90 cm to the side
        assert _parts(_track(frames, world, seed=seed)) == [frames[:60].tolist(), frames[60:].tolist()], seed
    # three people in a row
    world = _walk(len(frames), (0.0, 0.0, -600.0), (0.0, 0.0, 0.0))
    world[40:, 0] += 150.0
    world[80:, 0] -= 320.0
    assert _parts(_track(frames, world, seed=1)) == [frames[:40].tolist(), frames[40:80].tolist(), frames[80:].tolist()]
    # a camera that travels and pans changes nothing: the body is judged in the world
    c2w = _camera(len(frames), travel_cm_per_frame=10.0, pan_deg_per_frame=0.15)
    still = _walk(len(frames), (300.0, 0.0, -900.0), (0.0, 0.0, 0.0))
    assert _parts(_track(frames, still, c2w, seed=2)) == [frames.tolist()]
    jumped = still.copy()
    jumped[70:, 0] -= 200.0
    assert _parts(_track(frames, jumped, c2w, seed=2)) == [frames[:70].tolist(), frames[70:].tolist()]
    # a track whose own estimate scatters by decimetres is not cut for a step its scatter explains...
    world = _walk(len(frames), (0.0, 0.0, -2500.0), (0.0, 0.0, 0.0))
    world[60:, 0] += 60.0
    assert _parts(_track(frames, world, seed=6, across_cm=25.0, along_cm=200.0)) == [frames.tolist()]
    # ...but is for a step far beyond it
    world[60:, 0] += 400.0
    assert _parts(_track(frames, world, seed=6, across_cm=25.0, along_cm=200.0)) == [frames[:60].tolist(), frames[60:].tolist()]


def test_a_few_frames_on_somebody_else_do_not_make_a_person() -> None:
    frames = np.arange(1001, 1121)
    base = _walk(len(frames), (-100.0, 0.0, -700.0), (50.0, 0.0, 0.0))
    # two frames on a neighbour: one wild frame or two decide nothing, the track stays whole
    world = base.copy()
    world[50:52, 0] += 120.0
    assert _parts(_track(frames, world, seed=1)) == [frames.tolist()]
    # six frames on a neighbour (too short to be a person): those frames are dropped, both sides are one person again
    world = base.copy()
    world[50:56, 0] += 150.0
    kept = frames[:50].tolist() + frames[56:].tolist()
    assert _parts(_track(frames, world, seed=2)) == [kept]
    # rubbish between two DIFFERENT people: dropped, and the two stay two
    world = base.copy()
    world[50:56, 0] += 400.0
    world[56:, 0] -= 300.0
    assert _parts(_track(frames, world, seed=3)) == [frames[:50].tolist(), frames[56:].tolist()]


def test_unseen_time_allows_what_a_person_can_walk() -> None:
    frames = np.arange(1001, 1141)
    use = np.ones(len(frames), dtype=bool)
    use[40:70] = False  # hidden for 30 frames (1.25 s)
    for seed in range(6):
        # comes back two metres to the side: a person can have walked there
        world = _walk(len(frames), (0.0, 0.0, -900.0), (0.0, 0.0, 0.0))
        world[70:, 0] += 200.0
        assert _parts(_track(frames, world, seed=seed), use) == [frames[use].tolist()], seed
        # comes back six metres to the side: nobody standing there a second ago walked that
        world[70:, 0] += 400.0
        assert _parts(_track(frames, world, seed=seed), use) == [frames[:40].tolist(), frames[70:].tolist()], seed
    # somebody running at 6 m/s who is hidden for a while and runs on is one person
    world = _walk(len(frames), (-2000.0, 0.0, -2500.0), (600.0, 0.0, 0.0))
    assert _parts(_track(frames, world, seed=4), use) == [frames[use].tolist()]


def test_the_parts_come_back_as_tracks_of_their_own() -> None:
    frames = np.arange(1001, 1121)
    world = _walk(len(frames), (0.0, 0.0, -700.0), (0.0, 0.0, 0.0))
    world[50:56, 0] += 400.0   # rubbish
    world[56:, 0] -= 300.0     # somebody else
    raw = _track(frames, world, seed=1)
    use = np.ones(len(frames), dtype=bool)
    use[10:13] = False         # three unusable frames inside the first person: they stay with that person's stretch
    parts, dropped = people_of_track(raw, use, FPS)
    assert len(parts) == 2 and dropped == 6
    first, second = parts
    assert first["frames"].tolist() == frames[:50].tolist() and second["frames"].tolist() == frames[56:].tolist()
    for part in parts:
        n = len(part["frames"])
        for key, val in part.items():
            if key in NOT_PER_FRAME:
                assert np.array_equal(val, raw[key]), key          # what is not per frame is untouched
            else:
                assert len(val) == n, key                            # every per-frame array follows the frames
    assert np.array_equal(second["pred_cam_t"], raw["pred_cam_t"][56:]) and np.array_equal(second["cam_to_world"], raw["cam_to_world"][56:])
    # a track that is one person comes back as the very same object, nothing copied, nothing dropped
    whole = _track(frames, _walk(len(frames)), seed=2)
    same, none = people_of_track(whole, np.ones(len(frames), dtype=bool), FPS)
    assert len(same) == 1 and same[0] is whole and none == 0
    # the settings are plain numbers one can read
    cfg = SplitConfig()
    assert cfg.min_jump_cm == 50.0 and cfg.walk_cm_s == 250.0 and cfg.noise_factor == 5.0


def run_all() -> None:
    test_one_person_is_never_cut()
    test_a_jump_to_a_neighbour_is_cut_where_it_happens()
    test_a_few_frames_on_somebody_else_do_not_make_a_person()
    test_unseen_time_allows_what_a_person_can_walk()
    test_the_parts_come_back_as_tracks_of_their_own()
    print("track split tests passed")


if __name__ == "__main__":
    run_all()
