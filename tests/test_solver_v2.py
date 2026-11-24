"""Tests of the whole-shot solve (`shotspace.solver.trajectory`) and of how its results are placed in the world.

These check the MATH and the data contract on constructed scenes with a known answer.  They say nothing about
quality on real footage: that is measured against ground-truth datasets by the harness under `eval/`
(`eval/run_product.py`, `eval/solver_bench.py`)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shotspace.core.camera import CameraTrack
from shotspace.core.camera_space import place_world_solved_character
from shotspace.core.skeleton import SkinnedCharacter
from shotspace.solver.trajectory import (
    GroundMesh,
    GroundPlane,
    KP_L_FOOT,
    KP_R_FOOT,
    SolveConfig,
    fit_translation_per_frame,
    fuse_with_prior,
    keypoints_world_m,
    scale_from_parallax,
    solve_person,
    solve_trajectory,
    stance_runs,
)


def _walking_scene(true_scale: float = 1.06, t_n: int = 96, px_noise: float = 1.2, seed: int = 0):
    """A rigid 70-keypoint body walking across a dollying, panning camera, feet alternately planted.
    The 'network' believes the body has scale 1 and returns a noisy, biased depth."""
    rng = np.random.default_rng(seed)
    frames = np.arange(1001, 1001 + t_n)
    w, h, f = 1920, 1080, 1800.0
    k = np.repeat(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])[None], t_n, 0)
    c2w = np.repeat(np.eye(4)[None], t_n, 0)
    for i in range(t_n):
        ang = np.radians(-4 + 8 * i / t_n)
        c, s = np.cos(ang), np.sin(ang)
        c2w[i, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        c2w[i, :3, 3] = [-80 + 160 * i / t_n, 160.0, 0.0]
    tmpl = np.zeros((70, 3))
    tmpl[:, 1] = rng.uniform(0.0, 1.7, 70)
    tmpl[:, 0] = rng.uniform(-0.25, 0.25, 70)
    tmpl[:, 2] = rng.uniform(-0.12, 0.12, 70)
    tmpl[[9, 10]] = [[0.1, 0.92, 0], [-0.1, 0.92, 0]]
    tmpl[[5, 6]] = [[0.2, 1.45, 0], [-0.2, 1.45, 0]]
    tmpl[list(KP_L_FOOT)] = [[0.12, 0.02, 0.1], [0.16, 0.02, 0.08], [0.1, 0.02, -0.08]]
    tmpl[list(KP_R_FOOT)] = [[-0.12, 0.02, 0.1], [-0.16, 0.02, 0.08], [-0.1, 0.02, -0.08]]
    root = np.stack([np.linspace(-1.2, 1.2, t_n), np.zeros(t_n), np.full(t_n, -6.0)], 1)
    kp_world = true_scale * tmpl[None] + root[:, None, :]
    step = 12
    contact = np.zeros((t_n, 2), bool)
    for i in range(t_n):
        side = (i // step) % 2
        contact[i, side] = True
        a_i = (i // step) * step
        for kp in (KP_L_FOOT, KP_R_FOOT)[side]:
            kp_world[i, kp] = true_scale * tmpl[kp] + root[a_i]
        kp_world[i, list((KP_L_FOOT, KP_R_FOOT)[1 - side]), 1] += 0.08
    m = c2w[:, :3, :3] @ np.diag([1.0, -1.0, -1.0])
    o = c2w[:, :3, 3] * 0.01
    p_cam = np.einsum("tji,tkj->tki", m, kp_world - o[:, None, :])
    uv = np.stack([f * p_cam[..., 0] / p_cam[..., 2] + w / 2, f * p_cam[..., 1] / p_cam[..., 2] + h / 2], -1)
    uv_noisy = uv + rng.normal(0, px_noise, uv.shape)
    origin = np.einsum("tji,tj->ti", m, root - o)
    kp_local = (p_cam - origin[:, None, :]) / true_scale
    init_t = origin / true_scale * (1 + rng.normal(0, 0.03, (t_n, 1)))
    return dict(frames=frames, k=k, c2w=c2w, kp_local=kp_local, uv=uv_noisy, init_t=init_t, kp_world=kp_world,
                contact=contact, height_px=np.full(t_n, f * 1.7 / 6.0), true_scale=true_scale)


def _errors(sc: dict, tau: np.ndarray, scale: float) -> tuple[float, float]:
    kw = keypoints_world_m(sc["kp_local"], tau, scale, sc["c2w"])
    pel = 0.5 * (kw[:, 9] + kw[:, 10])
    pel_gt = 0.5 * (sc["kp_world"][:, 9] + sc["kp_world"][:, 10])
    slide = []
    for side, kps in enumerate((KP_L_FOOT, KP_R_FOOT)):
        st = sc["contact"][:-1, side] & sc["contact"][1:, side]
        slide.append(np.linalg.norm(np.diff(kw[:, kps[0]], axis=0), axis=1)[st])
    return float(np.linalg.norm(pel - pel_gt, axis=1).mean() * 100), float(np.concatenate(slide).mean() * 100)


def test_whole_shot_solve_recovers_scale_and_plants_feet() -> None:
    sc = _walking_scene()
    w = np.ones(sc["uv"].shape[:2])
    args = (sc["frames"], sc["k"], sc["c2w"], sc["kp_local"], sc["uv"], w, sc["init_t"], sc["height_px"])

    # per-frame reprojection fit alone: correct in the camera view, but the wrong body scale puts it at the wrong depth
    t_fit = fit_translation_per_frame(sc["kp_local"], sc["uv"], w, sc["k"], sc["init_t"])
    err_fit, slide_fit = _errors(sc, t_fit, 1.0)
    assert err_fit > 25.0, err_fit

    # no scan: the planted feet under the moving EXACT camera reveal the true scale
    res = solve_person(*args, cfg=SolveConfig(fps=24.0, sigma_scale=0.2))
    err, slide = _errors(sc, res.cam_t_m, res.scale)
    assert abs(res.scale - sc["true_scale"]) < 0.012, res.scale
    assert err < 3.0 and slide < 0.25, (err, slide)
    assert res.info["scale_source"].startswith("parallax")
    assert (res.contact == sc["contact"]).mean() > 0.9
    assert float(np.median(res.reproj_px)) < 3.0  # the camera view is not sacrificed

    # with the ground: feet rest on it
    plane = GroundPlane(np.array([0.0, 1.0, 0.0]), 0.0)
    res_g = solve_person(*args, cfg=SolveConfig(fps=24.0, sigma_scale=0.2), ground=plane, sole_drop_m=0.02)
    err_g, slide_g = _errors(sc, res_g.cam_t_m, res_g.scale)
    assert abs(res_g.scale - sc["true_scale"]) < 0.02 and err_g < 4.0 and slide_g < 0.25, (res_g.scale, err_g, slide_g)
    kw = keypoints_world_m(sc["kp_local"], res_g.cam_t_m, res_g.scale, sc["c2w"])
    planted_h = [kw[t, KP_L_FOOT[0] if sc["contact"][t, 0] else KP_R_FOOT[0], 1] for t in range(len(sc["frames"]))]
    assert abs(float(np.median(planted_h)) - 0.02) < 0.012

    # a static camera carries no parallax: the scale must stay at the network's value instead of drifting
    still = dict(sc)
    still["c2w"] = np.repeat(sc["c2w"][:1], len(sc["frames"]), 0)
    m = still["c2w"][:, :3, :3] @ np.diag([1.0, -1.0, -1.0])
    o = still["c2w"][:, :3, 3] * 0.01
    p_cam = np.einsum("tji,tkj->tki", m, sc["kp_world"] - o[:, None, :])
    f = sc["k"][0, 0, 0]
    uv = np.stack([f * p_cam[..., 0] / p_cam[..., 2] + 960, f * p_cam[..., 1] / p_cam[..., 2] + 540], -1)
    res_s = solve_person(sc["frames"], sc["k"], still["c2w"], sc["kp_local"], uv, w, sc["init_t"], sc["height_px"],
                         cfg=SolveConfig(fps=24.0))
    assert abs(res_s.scale - 1.0) < 1e-6, res_s.scale


def test_motion_terms_never_choose_the_scale() -> None:
    """Smoothness and planted feet both prefer 'less motion' = a smaller body nearer the camera.  With the scale
    measurement switched off the solve must keep the given scale exactly."""
    sc = _walking_scene(true_scale=1.0)
    w = np.ones(sc["uv"].shape[:2])
    res = solve_person(sc["frames"], sc["k"], sc["c2w"], sc["kp_local"], sc["uv"], w, sc["init_t"], sc["height_px"],
                       cfg=SolveConfig(fps=24.0, use_contact=False))
    assert abs(res.scale - 1.0) < 1e-6


def test_parallax_scale_is_fused_with_the_prior() -> None:
    val, weight = fuse_with_prior(1.10, 0.01, 0.03)
    assert 1.08 < val < 1.10 and weight > 0.85
    val, weight = fuse_with_prior(1.10, 0.20, 0.03)
    assert abs(val - 1.0) < 0.01 and weight < 0.05
    # hidden feet are not measurements
    sc = _walking_scene()
    kw = keypoints_world_m(sc["kp_local"], sc["init_t"], 1.0, sc["c2w"])
    cam_o = sc["c2w"][:, :3, 3] * 0.01
    unseen = np.zeros(kw.shape[:2], dtype=bool)
    a, se, n_used = scale_from_parallax(kw, sc["contact"], cam_o, sc["frames"], kp_seen=unseen)
    assert n_used == 0 and not np.isfinite(se) and a == 1.0


def test_stance_runs_never_span_frame_gaps() -> None:
    frames = np.array([1, 2, 3, 4, 8, 9, 10, 11, 12])
    contact = np.zeros((9, 2), bool)
    contact[:, 0] = True
    runs = stance_runs(contact, frames)
    assert runs == [(0, 0, 4), (0, 4, 9)], runs


def test_ground_mesh_picks_the_level_the_foot_is_on() -> None:
    # floor at y = 0, a 1 m high platform over x in [2, 4], and a vertical wall that must never count as ground
    v = np.array([
        [-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10],
        [2, 1, -1], [4, 1, -1], [4, 1, 1], [2, 1, 1],
        [0, 0, -5], [0, 3, -5], [0, 3, 5], [0, 0, 5],
    ], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6], [8, 9, 10], [8, 10, 11]])
    mesh = GroundMesh(v, f)
    assert len(mesh) == 4
    nrm, off, ok = mesh.support(np.array([[3.0, 1.05, 0.0], [3.0, 0.03, 0.0], [-5.0, 0.2, 0.0], [50.0, 0.0, 0.0]]))
    assert ok.tolist() == [True, True, True, False]
    heights = -(off[:3] + nrm[:3, 0] * np.array([3.0, 3.0, -5.0])) / nrm[:3, 1]
    assert np.allclose(heights, [1.0, 0.0, 0.0], atol=1e-9)
    assert np.all(nrm[:3, 1] > 0.99)


def _toy_character(vis: np.ndarray, world_solved: bool) -> SkinnedCharacter:
    f_n = len(vis)
    anim = np.repeat(np.eye(4)[None, None], f_n, 0).repeat(2, 1)
    anim[:, :, 2, 3] = -300.0  # 3 m in front of the camera
    anim[:, 1, 1, 3] = 90.0
    return SkinnedCharacter(
        name="person_01", person_id=1, frames=list(range(1001, 1001 + f_n)), joint_names=["Root", "Hips"],
        parents=np.array([-1, 0]), bind_world=np.repeat(np.eye(4)[None], 2, 0), anim_world=anim,
        rest_points=np.zeros((3, 3), np.float32), faces=np.array([[0, 1, 2]], np.int32),
        joint_indices=np.zeros((3, 1), np.int32), joint_weights=np.ones((3, 1), np.float32),
        visible=vis.astype(np.int32), anchor_frames=np.array([1003 if not v else 1001 + i for i, v in enumerate(vis)]),
        custom_data={"shotspace:world_solved": world_solved, "shotspace:has_real_camera": True},
    )


def test_world_solved_people_are_placed_by_the_exact_camera_only() -> None:
    f_n = 8
    frames = list(range(1001, 1001 + f_n))
    c2w = np.repeat(np.eye(4)[None], f_n, 0)
    c2w[:, 0, 3] = np.linspace(0.0, 700.0, f_n)  # the camera travels 7 m
    cam = CameraTrack(name="cam", path="/cam", frames=tuple(frames), cam_to_world=c2w, focal_mm=np.full(f_n, 35.0),
                      h_aperture_mm=np.full(f_n, 36.0), v_aperture_mm=np.full(f_n, 20.25), width=1920, height=1080)
    vis = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    ch = _toy_character(vis, world_solved=True)
    out = place_world_solved_character(ch, cam, frames)
    for t in np.nonzero(vis)[0]:
        assert np.allclose(out.anim_world[t], c2w[t][None] @ ch.anim_world[t], atol=1e-9)
    # hidden frames stay where the person was last solved, they do not ride along with the camera
    assert np.allclose(out.anim_world[3], out.anim_world[2]) and np.allclose(out.anim_world[4], out.anim_world[2])
    assert out.visible_mask.tolist() == vis.tolist()

    from shotspace.web.workspace import WorkspaceManager

    aligned, _objs = WorkspaceManager._align_scene(cam, 1920, 1080, frames, [ch], [], None)
    assert np.allclose(aligned[0].anim_world, out.anim_world)
    # a scan never rescales or shifts a world-solved person afterwards (it was already used inside the solve)
    scan = {"vertices_cm": np.array([[-900, 0, -900], [900, 0, -900], [900, 0, 900], [-900, 0, 900.0]]),
            "faces": np.array([[0, 2, 1], [0, 3, 2]])}
    aligned_s, _ = WorkspaceManager._align_scene(cam, 1920, 1080, frames, [ch], [], scan)
    assert np.allclose(aligned_s[0].anim_world, out.anim_world)
    assert replace(ch).custom_data["shotspace:world_solved"] is True


def test_solver_v2_result_file_is_marked_world_solved() -> None:
    f_n, j_n, v_n = 4, 3, 5
    data = {
        "frames": np.arange(1001, 1001 + f_n), "visible": np.ones(f_n, np.int32), "anchor_frames": np.arange(1001, 1001 + f_n),
        "joint_world": np.repeat(np.eye(4, dtype=np.float32)[None, None], f_n, 0).repeat(j_n, 1),
        "bind_world": np.repeat(np.eye(4)[None], j_n, 0), "rest_vertices": np.zeros((v_n, 3), np.float32),
        "faces": np.array([[0, 1, 2]], np.int32), "parents": np.array([-1, 0, 1]), "joint_names": np.array(["a", "b", "c"]),
        "skin_indices": np.zeros((v_n, 1), np.int32), "skin_weights": np.ones((v_n, 1), np.float32),
        "has_real_camera": np.int32(1), "smoothing_strength": np.float32(0.5),
    }
    assert SkinnedCharacter.from_mhr_npz(1, data).custom_data["shotspace:world_solved"] is False
    data.update(solver_version=np.int32(2), body_scale=np.float32(1.04))
    ch = SkinnedCharacter.from_mhr_npz(1, data)
    assert ch.custom_data["shotspace:world_solved"] is True and abs(ch.custom_data["shotspace:body_scale"] - 1.04) < 1e-6


def test_manual_people_borrow_the_outline_of_the_tracked_person_under_their_box() -> None:
    """A manually boxed person takes, frame by frame, the outline of whichever tracked person sits under the
    artist's box (the boxes are the identity, not the tracker's ids); frames nobody matches get no outline."""
    import tempfile
    from pathlib import Path

    from shotspace.web.workspace import manual_masks_from_tracker

    def entry(pid: int, frame: int, rect: list[float]) -> dict[str, np.ndarray]:
        bits = np.packbits(np.full((4, 4), pid % 2, dtype=np.uint8).reshape(-1))
        return {
            f"p{pid}_f{frame}_bits": bits,
            f"p{pid}_f{frame}_shape": np.asarray([4, 4], dtype=np.int16),
            f"p{pid}_f{frame}_rect": np.asarray(rect, dtype=np.float32),
        }

    # the tracker swapped the identities of two people between frame 1002 and 1003
    box_a, box_b = [100.0, 100.0, 200.0, 400.0], [400.0, 120.0, 520.0, 420.0]
    tracked = [
        {"id": 1, "boxes": {"1001": box_a, "1002": box_a, "1003": box_b, "1004": box_b}},
        {"id": 2, "boxes": {"1001": box_b, "1002": box_b, "1003": box_a}},
    ]
    with tempfile.TemporaryDirectory() as td:
        npz = Path(td) / "person_masks_sam3.npz"
        arrays: dict[str, np.ndarray] = {}
        for tp in tracked:
            for fk, bx in tp["boxes"].items():
                arrays.update(entry(int(tp["id"]), int(fk), bx))
        np.savez(npz, **arrays)
        # the artist boxed person A (slightly loose boxes) on 1001..1005; nobody is tracked under the box on 1004/1005
        loose_a = [92.0, 95.0, 210.0, 410.0]
        manual = [{"id": 101, "is_manual": True, "boxes": {str(f): loose_a for f in range(1001, 1006)}}]
        entries, cover = manual_masks_from_tracker(manual, tracked, npz)
        assert cover == {101: (3, 5)}, cover
        for frame, src in ((1001, 1), (1002, 1), (1003, 2)):
            assert np.array_equal(entries[f"p101_f{frame}_bits"], arrays[f"p{src}_f{frame}_bits"])
            assert np.allclose(entries[f"p101_f{frame}_rect"], box_a)
        assert not any(k.startswith("p101_f1004") or k.startswith("p101_f1005") for k in entries)
        # without a tracking result every frame is left to the box-prompted outline
        entries_none, cover_none = manual_masks_from_tracker(manual, [], Path(td) / "missing.npz")
        assert entries_none == {} and cover_none == {101: (0, 5)}


def _swaying_scene(t_n: int = 120, sway_m: float = 0.06, px_noise: float = 1.0, seed: int = 1) -> dict:
    """Someone standing on the spot whose body sways, seen by a static camera.  The per-frame estimate is a rigid
    body swaying as a whole; its feet are out of the picture, so their 2D 'evidence' is the estimate's own
    projection (low weight, not seen)."""
    rng = np.random.default_rng(seed)
    frames = np.arange(1001, 1001 + t_n)
    w, h, f = 1920, 1080, 1500.0
    k = np.repeat(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])[None], t_n, 0)
    c2w = np.repeat(np.eye(4)[None], t_n, 0)
    c2w[:, 1, 3] = 160.0
    tmpl = np.zeros((70, 3))
    tmpl[:, 1] = rng.uniform(0.0, 1.7, 70)
    tmpl[:, 0] = rng.uniform(-0.25, 0.25, 70)
    tmpl[:, 2] = rng.uniform(-0.12, 0.12, 70)
    tmpl[[9, 10]] = [[0.1, 0.92, 0], [-0.1, 0.92, 0]]
    tmpl[[5, 6]] = [[0.2, 1.45, 0], [-0.2, 1.45, 0]]
    tmpl[list(KP_L_FOOT)] = [[0.12, 0.02, 0.1], [0.16, 0.02, 0.08], [0.1, 0.02, -0.08]]
    tmpl[list(KP_R_FOOT)] = [[-0.12, 0.02, 0.1], [-0.16, 0.02, 0.08], [-0.1, 0.02, -0.08]]
    ph = np.linspace(0, 6 * np.pi, t_n)
    root = np.stack([sway_m * np.sin(ph), np.zeros(t_n), -4.0 + 0.5 * sway_m * np.cos(0.7 * ph)], 1)
    kp_world = tmpl[None] + root[:, None, :]
    m = c2w[:, :3, :3] @ np.diag([1.0, -1.0, -1.0])
    o = c2w[:, :3, 3] * 0.01
    p_cam = np.einsum("tji,tkj->tki", m, kp_world - o[:, None, :])
    uv = np.stack([f * p_cam[..., 0] / p_cam[..., 2] + w / 2, f * p_cam[..., 1] / p_cam[..., 2] + h / 2], -1)
    origin = np.einsum("tji,tj->ti", m, root - o)
    seen = np.ones((t_n, 70), bool)
    seen[:, list(KP_L_FOOT + KP_R_FOOT)] = False
    return dict(frames=frames, k=k, c2w=c2w, kp_local=p_cam - origin[:, None, :], uv=uv + rng.normal(0, px_noise, uv.shape),
                w=np.where(seen, 1.0, 0.2), seen=seen, init_t=origin * (1 + rng.normal(0, 0.02, (t_n, 1))),
                height_px=np.full(t_n, f * 1.7 / 4.0), true_z=origin[:, 2])


def test_planted_feet_do_not_pull_the_person_towards_the_camera() -> None:
    """Holding feet still is cheaper the closer the body is to the camera (the same sway is fewer centimetres
    there).  That is not evidence about distance: with the level held, the anchored solve keeps exactly the average
    distance the image evidence gave; with the level free it comes out measurably nearer."""
    sc = _swaying_scene()
    args = (sc["frames"], sc["k"], sc["c2w"], sc["kp_local"], sc["uv"], sc["w"], sc["init_t"], sc["height_px"])
    free = solve_trajectory(*args, cfg=SolveConfig(fps=24.0))

    held = solve_person(*args, cfg=SolveConfig(fps=24.0), kp_seen=sc["seen"])
    assert held.contact.all()  # the hidden feet of someone standing on the spot are held planted
    assert abs(float(np.mean(held.cam_t_m[:, 2] / free.cam_t_m[:, 2])) - 1.0) < 1e-6
    assert abs(float(np.mean(held.cam_t_m[:, 2] / sc["true_z"])) - 1.0) < 5e-4

    loose = solve_person(*args, cfg=SolveConfig(fps=24.0, hold_depth_level=False), kp_seen=sc["seen"])
    assert float(np.mean(loose.cam_t_m[:, 2] / free.cam_t_m[:, 2])) - 1.0 < -1e-3


def test_frames_with_an_impossible_distance_are_not_solved() -> None:
    """A tiny or half-hidden person is sometimes reported at a wildly different distance for a few frames.  Those
    frames are outliers; someone really approaching the camera, even fast, is not."""
    from shotspace.solver.trajectory import depth_outliers, drop_short_runs

    frames = np.arange(1001, 1061)
    ok = np.ones(60, dtype=bool)
    z = np.full(60, 22.0) + 0.3 * np.sin(np.arange(60))
    z[[20, 21]] = [321.0, 0.4]          # two wild frames in the middle
    z[57:] = [382.0, 413.0, 134.0]      # and a wild tail
    out = depth_outliers(z, frames, ok)
    assert out[[20, 21, 57, 58, 59]].all() and int(out.sum()) == 5, np.nonzero(out)[0]

    # running towards the camera: 12 m -> 2 m in 60 frames is real motion, nothing may be dropped
    z_run = np.linspace(12.0, 2.0, 60)
    assert not depth_outliers(z_run, frames, ok).any()
    # a close-up person stepping back and forth by a metre is real motion too
    z_close = 1.5 + 0.9 * np.sin(np.arange(60) / 6.0)
    assert not depth_outliers(z_close, frames, ok).any()

    # islands shorter than three frames go, and a frame gap ends a run
    gap_frames = np.array([1001, 1002, 1003, 1010, 1011, 1020, 1021, 1022, 1023])
    kept = drop_short_runs(np.ones(9, dtype=bool), gap_frames, 3)
    assert kept.tolist() == [True, True, True, False, False, True, True, True, True]


def test_the_weak_ends_of_a_solved_run_are_cut_off() -> None:
    """Entering and leaving the picture: the first and last frames of a run show only a sliver of the person and
    drift.  They are cut off; frames with as little visible in the MIDDLE of a run stay (the solve holds them from
    both sides), and a person who never shows more (framed from the waist up all shot) loses nothing."""
    from shotspace.solver.trajectory import trim_weak_ends

    frames = np.arange(1001, 1041)
    # walks in (share rising), fully visible, half hidden behind somebody for a while, walks out (share falling)
    share = np.concatenate([np.linspace(0.26, 0.95, 8), np.full(10, 0.95), np.full(6, 0.35), np.full(10, 0.95), np.linspace(0.9, 0.26, 6)])
    ok = np.ones(40, dtype=bool)
    out = trim_weak_ends(ok, frames, share)
    limit = 0.6 * np.percentile(share, 75)
    assert not out[0] and not out[1] and not out[-1], "the slivers at both ends must go"
    first, last = np.flatnonzero(out)[[0, -1]]
    assert share[first] >= limit and share[last] >= limit and np.all(share[:first] < limit) and np.all(share[last + 1:] < limit)
    assert out[first:last + 1].all(), "nothing in the middle of the run may be removed"
    assert out[18:24].all() and np.all(share[18:24] < limit), "the half-hidden stretch in the middle stays"

    # two runs separated by frames that are not usable: each run is trimmed at its own ends
    ok2 = np.ones(40, dtype=bool)
    ok2[18:24] = False
    out2 = trim_weak_ends(ok2, frames, share)
    assert not out2[18:24].any() and out2[8:18].all() and out2[24:34].all() and not out2[0] and not out2[-1]

    # a frame gap ends a run too
    gap_frames = np.concatenate([np.arange(1001, 1011), np.arange(1020, 1030)])
    gap_share = np.concatenate([np.full(8, 0.9), [0.3, 0.3], [0.3], np.full(9, 0.9)])
    out3 = trim_weak_ends(np.ones(20, dtype=bool), gap_frames, gap_share)
    assert out3.tolist() == [True] * 8 + [False, False] + [False] + [True] * 9

    # somebody who is cut at the waist for the whole shot: the limit follows what they usually show
    waist = np.full(40, 0.45)
    waist[:3] = 0.3
    out4 = trim_weak_ends(np.ones(40, dtype=bool), frames, waist)
    assert out4.all(), "0.30 is above 0.6 x 0.45: nothing to cut"
    waist[:3] = 0.2
    assert trim_weak_ends(np.ones(40, dtype=bool), frames, waist).tolist() == [False] * 3 + [True] * 37

    # "a few frames": somebody half hidden for a long stretch at the start is not entering - only the limit is cut
    long_weak = np.concatenate([np.full(20, 0.3), np.full(20, 0.95)])
    out5 = trim_weak_ends(np.ones(40, dtype=bool), frames, long_weak, max_frames=8)
    assert out5.tolist() == [False] * 8 + [True] * 32
    assert trim_weak_ends(np.ones(40, dtype=bool), frames, long_weak, max_frames=100).tolist() == [False] * 20 + [True] * 20
    # ...unless the person is cut by the picture edge on those frames: then they ARE entering, however slowly,
    # and every weak frame goes (the limit is for people hidden by somebody else)
    edge = np.zeros(40, dtype=bool)
    edge[:20] = True
    assert trim_weak_ends(np.ones(40, dtype=bool), frames, long_weak, max_frames=8, at_picture_edge=edge).tolist() == [False] * 20 + [True] * 20
    edge[:20] = False
    edge[:5] = True  # at the edge for 5 frames, then inside the picture but still half hidden: 5 + the limit counts on
    out6 = trim_weak_ends(np.ones(40, dtype=bool), frames, long_weak, max_frames=8, at_picture_edge=edge)
    assert out6.tolist() == [False] * 8 + [True] * 32
    # nothing usable, or everything weak: no crash, nothing invented
    assert not trim_weak_ends(np.zeros(5, dtype=bool), frames[:5], share[:5]).any()
    assert trim_weak_ends(np.ones(5, dtype=bool), frames[:5], np.full(5, 0.5)).all()


def test_planted_feet_are_recognised_in_the_picture_with_camera_and_scan() -> None:
    """With a scan, a foot is planted when ONE point of the scanned ground explains its image positions over a
    stretch of frames (`contacts_on_ground`).  The test must not care how far away the unanchored solve put the
    person: under a travelling camera that solve drifts along the line of sight, which is exactly when the old
    speed test loses the planted feet.  A foot that swings is not planted; without a scan nothing is said; and
    with a camera that does not travel nothing is said either (a foot moving along the line of sight would look
    as still as a planted one - there the speed test decides alone, as before)."""
    from shotspace.solver.trajectory import CV_TO_GL, GroundPlane, SolveConfig, contacts_on_ground, detect_contacts

    rng = np.random.default_rng(3)
    t_n, fps = 28, 20.0
    frames = np.arange(1001, 1001 + t_n)
    k = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    k_mats = np.repeat(k[None], t_n, axis=0)
    ground = GroundPlane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0)
    sole = 0.02
    for cam_step_m in (0.11, 0.0):  # a camera travelling 11 cm per frame, and a camera that stands still
        c2w = np.repeat(np.eye(4)[None], t_n, axis=0)
        c2w[:, 0, 3] = 100.0 * cam_step_m * np.arange(t_n)
        c2w[:, 1, 3] = 160.0  # 1.6 m above the ground, looking along -Z
        cam_o = c2w[:, :3, 3] * 0.01
        # left foot: planted on frames 0..13, then swings forward and up; right foot: swings all the time
        toe = np.array([0.5, sole, -6.0])
        left = np.repeat(np.stack([toe, toe + [0.07, 0.0, 0.0], toe + [0.03, 0.0, 0.22]])[None], t_n, axis=0)
        swing = np.clip(np.arange(t_n) - 13, 0, None)[:, None, None]
        left = left + swing * np.array([0.15, 0.0, 0.0]) + np.minimum(swing, 4) * np.array([0.0, 0.025, 0.0])
        right = left[0][None] + np.array([-0.3, 0.05, 0.1]) + np.arange(t_n)[:, None, None] * np.array([0.13, 0.0, 0.02])
        world = np.stack([left, right], axis=1)  # [T, 2, 3, 3] metres
        p_cv = (world - cam_o[:, None, None, :]) @ CV_TO_GL  # the camera has no rotation: world axes = GL camera axes
        px = np.stack([1400.0 * p_cv[..., 0] / p_cv[..., 2] + 960.0, 1400.0 * p_cv[..., 1] / p_cv[..., 2] + 540.0], axis=-1)
        px = px + rng.normal(0.0, 0.5, px.shape)
        assert np.all(p_cv[..., 2] > 1.0)
        # what the unanchored solve believes: every frame 6 to 14 % too far, differently on every frame
        wrong = cam_o[:, None, None, :] + (world - cam_o[:, None, None, :]) * (1.10 + 0.04 * np.sin(np.arange(t_n) * 1.3))[:, None, None, None]
        seen = np.ones((t_n, 2, 3), dtype=bool)
        cfg = SolveConfig(fps=fps)
        sigma_px = np.full(t_n, 3.0)
        got = contacts_on_ground(px, seen, k_mats, c2w, wrong, frames, sigma_px, ground, sole, cfg)
        if cam_step_m == 0.0:
            assert not got.any(), got.astype(int).tolist()
            continue
        assert got[:13, 0].all(), (cam_step_m, got[:, 0].astype(int).tolist())
        assert not got[16:, 0].any(), (cam_step_m, got[:, 0].astype(int).tolist())
        assert not got[:, 1].any(), (cam_step_m, got[:, 1].astype(int).tolist())
        # a foot the camera does not see is not judged
        hidden = seen.copy()
        hidden[:, 0] = False
        assert not contacts_on_ground(px, hidden, k_mats, c2w, wrong, frames, sigma_px, ground, sole, cfg).any()
        # no scan, no statement
        assert not contacts_on_ground(px, seen, k_mats, c2w, wrong, frames, sigma_px, None, sole, cfg).any()
        if cam_step_m > 0:
            # the speed test on the drifting solve misses this planted foot - the reason the new test exists
            by_speed = detect_contacts(wrong, frames, fps, cfg.contact_speed_cm, cfg.contact_min_frames)
            assert by_speed[:13, 0].mean() < 0.5, by_speed[:, 0].astype(int).tolist()


def run_all() -> None:
    test_planted_feet_are_recognised_in_the_picture_with_camera_and_scan()
    test_frames_with_an_impossible_distance_are_not_solved()
    test_the_weak_ends_of_a_solved_run_are_cut_off()
    test_planted_feet_do_not_pull_the_person_towards_the_camera()
    test_manual_people_borrow_the_outline_of_the_tracked_person_under_their_box()
    test_whole_shot_solve_recovers_scale_and_plants_feet()
    test_motion_terms_never_choose_the_scale()
    test_parallax_scale_is_fused_with_the_prior()
    test_stance_runs_never_span_frame_gaps()
    test_ground_mesh_picks_the_level_the_foot_is_on()
    test_world_solved_people_are_placed_by_the_exact_camera_only()
    test_solver_v2_result_file_is_marked_world_solved()


if __name__ == "__main__":
    run_all()
    print("All solver v2 tests passed!")
