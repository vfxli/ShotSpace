"""Automated regression test suite for ShotSpace core math, camera-space alignment,
plate grading, person selection, and USD / Alembic / FBX round-trip I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from shotspace.core.camera import CameraTrack
from shotspace.core.camera_space import (
    LensMismatchError,
    align_standard_character_to_camera,
    align_standard_object_to_camera,
    compute_camera_space_alignment,
    standard_camera_focal_px,
)
from shotspace.core.rigid_object import RigidObjectTrack
from shotspace.core.skeleton import SkinnedCharacter
from shotspace.core.smoothing import expand_to_full_shot_with_visibility
from shotspace.io import (
    GradeParams,
    OcioSettings,
    apply_grade_rgb,
    export_scene,
    ingest_sequence_dir,
    inspect_3d_file,
    load_camera_track,
    load_skinned_characters,
    write_exr_half_rgb,
)
from shotspace.server_config import ConfigStore
from shotspace.solver import MultiGpuScheduler, SelectionSpec, select_people


def test_plate_grade_lut_vs_float() -> None:
    g1 = GradeParams(exposure=1.75, brightness=-0.04, contrast=1.35, saturation=1.0, gamma=1.8)
    u8 = np.array([[[16, 96, 210], [0, 128, 255]]], dtype=np.uint8)
    fl = u8.astype(np.float32) / 255.0
    assert np.allclose(apply_grade_rgb(u8, g1), apply_grade_rgb(fl, g1), atol=1)

    g2 = GradeParams(exposure=0.4, brightness=0.05, contrast=1.25, saturation=1.6, gamma=1.15)
    assert np.allclose(apply_grade_rgb(u8, g2), apply_grade_rgb(fl, g2), atol=1)

    g_desat = GradeParams(saturation=0.0)
    gray = apply_grade_rgb(u8, g_desat)
    assert np.all(gray[..., 0] == gray[..., 1]) and np.all(gray[..., 1] == gray[..., 2])


def test_camera_space_and_multiformat_roundtrip() -> None:
    frames = [1001, 1002, 1003, 1004, 1005]
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 5, axis=0)
    c2w[:, 0, 3] = np.linspace(-25.0, 25.0, 5)
    c2w[:, 1, 3] = 155.0
    c2w[:, 2, 3] = 380.0
    cam = CameraTrack(
        name="shotCam",
        path="/shot/shotCam",
        frames=tuple(frames),
        cam_to_world=c2w,
        focal_mm=np.linspace(35.0, 50.0, 5),
        h_aperture_mm=np.full(5, 36.0),
        v_aperture_mm=np.full(5, 20.25),
        center_mm=np.tile([1.5, -0.6], (5, 1)),
        width=1920,
        height=1080,
        fps=24.0,
    )

    bind = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    bind[1, 1, 3] = 50.0
    bind[2, 1, 3] = 100.0
    anim = np.repeat(bind[None], 4, axis=0)
    anim[:, :, 2, 3] -= 280.0
    ch = SkinnedCharacter(
        name="person_01",
        person_id=1,
        frames=[1001, 1002, 1004, 1005],
        joint_names=["Hips", "Spine", "Head"],
        parents=np.array([-1, 0, 1], dtype=np.int64),
        bind_world=bind,
        anim_world=anim,
        rest_points=np.array([[0, 0, 0], [10, 0, 0], [0, 50, 0], [0, 100, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        joint_indices=np.array([[0, 1], [0, 1], [1, 2], [2, 1]], dtype=np.int32),
        joint_weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]], dtype=np.float32),
    )

    direct = compute_camera_space_alignment(cam, None, frames)
    placed = direct.apply_to_character(ch)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for ext in (".usdc", ".abc", ".fbx"):
            out_p = export_scene(root / f"scene{ext}", [placed], [cam], frames, fps=24.0, unit="cm")
            assert len(inspect_3d_file(out_p)["cameras"]) == 1
            back = load_camera_track(out_p)
            assert np.allclose(back.cam_to_world, cam.cam_to_world, atol=1e-3)
            assert np.allclose(back.focal_mm, cam.focal_mm, atol=1e-3)
            assert np.allclose(back.center_mm, cam.center_mm, atol=1e-3)
            if ext in (".usdc", ".fbx"):
                chars = load_skinned_characters(out_p)
                assert len(chars) == 1
                assert chars[0].frames == frames
                assert chars[0].solved_frames == [1001, 1002, 1004, 1005]
                assert chars[0].visible.tolist() == [1, 1, 0, 1, 1]

    # Dual-camera whole similarity fit (scale = 3.0)
    dst_c2w = c2w.copy()
    dst_c2w[:, :3, 3] = c2w[:, :3, 3] * 3.0 + np.array([120.0, 0.0, -80.0])
    dst_cam = CameraTrack("dstCam", "/dstCam", tuple(frames), dst_c2w, cam.focal_mm, cam.h_aperture_mm, cam.v_aperture_mm)
    whole = compute_camera_space_alignment(dst_cam, cam, frames, fit="whole")
    assert abs(whole.report.scale - 3.0) < 1e-4
    assert whole.report.rms_pos_error_cm < 1e-4

    # Still target camera (<10cm RMS spread) locks scale = 1.0
    still_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 5, axis=0)
    still_c2w[:, 0, 3] = np.linspace(0.0, 3.0, 5)
    still_cam = CameraTrack("stillCam", "/stillCam", tuple(frames), still_c2w, cam.focal_mm, cam.h_aperture_mm, cam.v_aperture_mm)
    still_fit = compute_camera_space_alignment(still_cam, cam, frames, fit="whole")
    assert still_fit.report.scale == 1.0 and still_fit.report.scale_locked_still

    # >2% FOV mismatch rejected in per_frame
    mismatch_cam = CameraTrack("wide", "/wide", tuple(frames), dst_c2w, cam.focal_mm * 1.1, cam.h_aperture_mm, cam.v_aperture_mm)
    try:
        compute_camera_space_alignment(mismatch_cam, cam, frames, fit="per_frame")
        raise AssertionError("Expected LensMismatchError")
    except LensMismatchError:
        pass


def test_selection_rules() -> None:
    people = [
        {"id": 1, "prominence": 8000, "boxes": {"1001": [50, 50, 350, 500], "1002": [50, 50, 350, 500]}},
        {"id": 2, "prominence": 1500, "boxes": {"1001": [120, 120, 190, 240], "1002": [120, 120, 190, 240]}},
    ]
    chosen, missed = select_people(people, SelectionSpec(mode="picked", picks=[[1001, 150, 180]], least_frames=2))
    assert [p["id"] for p in chosen] == [2] and not missed


def test_admin_auth_and_gpu_queue() -> None:
    import time

    with tempfile.TemporaryDirectory() as td:
        cs = ConfigStore(Path(td) / "server_config.json")
        cs.initialize(port=8765, password="TestPass123")
        assert cs.verify_password("TestPass123")
        tok = cs.create_token()
        assert cs.verify_token(tok)
        cs.change_password("TestPass123", "UpdatedPass999")
        assert not cs.verify_token(tok)

        sched = MultiGpuScheduler(cs)
        sched.set_all_gpus_enabled(False)
        executed: list[str] = []
        j1, pos1 = sched.submit("s1", "Task 1", "solve", lambda s, j, c: executed.append("s1"))
        time.sleep(0.25)
        assert pos1 == 1 and j1.status == "queued" and executed == []
        sched.cancel_job(j1.id)
        assert sched.overview()["summary"]["queued_count"] == 0


def test_ocio_exr_and_standard_camera_alignment() -> None:
    # 1. Verify ACEScg 16-bit half EXR sequence -> OCIO sRGB ingestion
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seq_dir = root / "seq"
        exr_img = np.full((64, 96, 3), 0.18, dtype=np.float32)  # 18% scene-linear mid-grey in ACEScg
        for f in (1001, 1002, 1003):
            write_exr_half_rgb(seq_dir / f"shot.{f:04d}.exr", exr_img)
        p_info = ingest_sequence_dir(seq_dir, root / "cache", ocio=OcioSettings("auto", "aces_sdr"))
        assert p_info.resolved_colorspace == "ACEScg"
        assert p_info.frames == [1001, 1002, 1003]
        assert "half" in p_info.bit_depth

        # 2. Verify Step 3 (SAM 3D Body origin standard camera) -> Step 4 (arbitrary lens + shift camera) rigid alignment
        w, h = 1280, 720
        f_std = standard_camera_focal_px(w, h)
        bind = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        bind[1, 1, 3] = 85.0  # Head 85 cm above pelvis
        anim = np.repeat(bind[None], 3, axis=0)
        anim[:, :, 0, 3] += 45.0
        anim[:, :, 1, 3] += -12.0
        anim[:, :, 2, 3] += -380.0  # 380 cm in front of standard camera
        ch = SkinnedCharacter(
            name="person_01",
            person_id=1,
            frames=[1001, 1002, 1003],
            joint_names=["Hips", "Head"],
            parents=np.array([-1, 0], dtype=np.int64),
            bind_world=bind,
            anim_world=anim,
            rest_points=np.array([[0, 0, 0], [0, 85, 0], [10, 0, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            joint_indices=np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32),
            joint_weights=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )
        # Pixel projection in standard camera
        p0_std = ch.anim_world[0, 0, :3, 3]
        u0_std = f_std * (p0_std[0] / (-p0_std[2])) + w * 0.5
        v0_std = f_std * (-p0_std[1] / (-p0_std[2])) + h * 0.5

        c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        c2w[:, 0, 3] = [100.0, 130.0, 160.0]
        c2w[:, 1, 3] = 165.0
        cam85 = CameraTrack(
            name="tele85",
            path="/shot/tele85",
            frames=(1001, 1002, 1003),
            cam_to_world=c2w,
            focal_mm=np.array([85.0, 85.0, 85.0]),
            h_aperture_mm=np.full(3, 36.0),
            v_aperture_mm=np.full(3, 20.25),
            center_mm=np.tile([2.4, -1.2], (3, 1)),
            width=w,
            height=h,
        )
        aligned = align_standard_character_to_camera(ch, cam85, w, h)
        # Bone length between Hips and Head must stay 100% rigid (85.0 cm)!
        bone_len = np.linalg.norm(aligned.anim_world[0, 1, :3, 3] - aligned.anim_world[0, 0, :3, 3])
        assert abs(bone_len - 85.0) < 1e-5
        # Reproject aligned Hips into cam85 pixel space -> must match (u0_std, v0_std) to 1e-4 px!
        p_cam = np.linalg.inv(c2w[0]) @ np.append(aligned.anim_world[0, 0, :3, 3], 1.0)
        k0 = cam85.intrinsics_per_frame(w, h)[1001]
        u_new = k0[0, 0] * (p_cam[0] / (-p_cam[2])) + k0[0, 2]
        v_new = k0[1, 1] * (-p_cam[1] / (-p_cam[2])) + k0[1, 2]
        assert abs(u_new - u0_std) < 1e-3 and abs(v_new - v0_std) < 1e-3

        # 3. Verify meter ("m") export across USD, ABC, and FBX!
        for ext in (".usdc", ".abc", ".fbx"):
            out_m = export_scene(root / f"meter_scene{ext}", [aligned], [cam85], [1001, 1002, 1003], fps=24.0, unit="m")
            assert out_m.is_file() and out_m.stat().st_size > 100
        # Reload meter FBX and verify unit_to_cm == 100.0 and camera matches!
        fbx_info = inspect_3d_file(root / "meter_scene.fbx")
        assert abs(fbx_info["unit_to_cm"] - 100.0) < 1e-3
        fbx_cam = load_camera_track(root / "meter_scene.fbx")
        assert np.allclose(fbx_cam.cam_to_world, cam85.cam_to_world, atol=1e-2)


def test_anti_flyaway_and_all_format_visibility_export() -> None:
    """Verify production anti-fly-away stabilizer (entry/exit border crop fly-away + mid-sequence Z spike),
    high-confidence anchor holding, 3D world-space freeze under moving camera, and per-frame visibility
    round-trip across USD (.usdc), FBX (.fbx), and Alembic (.abc)."""
    shot_frames = list(range(1001, 1021))  # 20 frames: 1001..1020
    # Detected on 1004..1012 and 1014..1018 (1001..1003 before entry, 1013 gap, 1019..1020 after exit)
    det_frames = [1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1014, 1015, 1016, 1017, 1018]
    n = len(det_frames)
    w, h = 1920, 1080

    # Normal interior bboxes (~240x620) vs border-clipped entry/exit slivers (~42x380 touching x=0 or x=1918)
    bboxes = np.zeros((n, 4), dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)
    cam_t = np.zeros((n, 3), dtype=np.float64)

    for i, f in enumerate(det_frames):
        if f in (1004, 1005):
            # Entering left border -> truncated sliver & flying away at Z = 19m!
            bboxes[i] = [0.0, 260.0, 42.0, 640.0]
            scores[i] = 0.49
            cam_t[i] = [-2.5, 0.1, 19.0]
        elif f == 1012:
            # Mid-sequence occlusion spike -> jumps from Z=3.6m to Z=14.0m in 1 frame!
            bboxes[i] = [880.0, 220.0, 1100.0, 820.0]
            scores[i] = 0.62
            cam_t[i] = [0.2, 0.0, 14.0]
        elif f in (1017, 1018):
            # Exiting right border -> truncated sliver & flying away at Z = 23m!
            bboxes[i] = [1880.0, 250.0, 1919.0, 620.0]
            scores[i] = 0.46
            cam_t[i] = [3.1, 0.1, 23.0]
        else:
            # Clean interior walking from left (-0.8m) to right (+0.8m) at steady Z = 3.6m
            prog = (f - 1006) / 10.0
            cx = 400.0 + prog * 1100.0
            bboxes[i] = [cx - 120.0, 200.0, cx + 120.0, 820.0]
            scores[i] = 0.96 if f in (1006, 1016) else 0.94
            cam_t[i] = [-0.8 + prog * 1.6, 0.0, 3.6]

    # Frames on which the person is really solvable (in the product this comes from the tracker masks:
    # `sam3d_worker.solvable_frames`); everything else must freeze in the world and be exported as hidden.
    expected_valid = [1006, 1007, 1008, 1009, 1010, 1011, 1014, 1015, 1016]
    valid_mask = np.array([f in expected_valid for f in det_frames])
    qual = np.ones(n, dtype=np.float64)
    valid_frames = list(expected_valid)

    src_vi, vis_full, anc_full = expand_to_full_shot_with_visibility(
        shot_frames, valid_frames, qual[valid_mask]
    )
    expected_vis = [1 if f in expected_valid else 0 for f in shot_frames]
    assert vis_full.tolist() == expected_vis
    # Pre-entry frames 1001..1005 must freeze at 1006; post-exit frames 1017..1020 must freeze at 1016
    assert all(int(anc_full[t]) == 1006 for t in range(5))
    assert all(int(anc_full[t]) == 1016 for t in range(16, 20))

    # Build SkinnedCharacter across all 20 shot frames and align to a MOVING camera
    bind = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    bind[1, 1, 3] = 85.0
    valid_pos_gl_cm = np.column_stack([cam_t[valid_mask, 0], -cam_t[valid_mask, 1], -cam_t[valid_mask, 2]]) * 100.0
    anim_valid = np.repeat(bind[None], len(valid_frames), axis=0)
    anim_valid[:, :, :3, 3] += valid_pos_gl_cm[:, None, :]
    anim_full = anim_valid[src_vi]

    ch = SkinnedCharacter(
        name="person_01",
        person_id=1,
        frames=shot_frames,
        joint_names=["Hips", "Head"],
        parents=np.array([-1, 0], dtype=np.int64),
        bind_world=bind,
        anim_world=anim_full,
        rest_points=np.array([[0, 0, 0], [0, 85, 0], [12, 0, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        joint_indices=np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32),
        joint_weights=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        visible=vis_full,
        anchor_frames=anc_full,
    )

    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(shot_frames), axis=0)
    c2w[:, 0, 3] = np.linspace(0.0, 300.0, len(shot_frames))  # Camera dollies 3 meters sideways!
    c2w[:, 1, 3] = 160.0
    cam = CameraTrack(
        name="trackCam",
        path="/shot/trackCam",
        frames=tuple(shot_frames),
        cam_to_world=c2w,
        focal_mm=np.full(len(shot_frames), 50.0),
        h_aperture_mm=np.full(len(shot_frames), 36.0),
        v_aperture_mm=np.full(len(shot_frames), 20.25),
        width=w,
        height=h,
    )
    aligned = align_standard_character_to_camera(ch, cam, w, h, shot_frames)

    # Hidden pre-entry frames 1001..1005 must stay 100% motionless in 3D world space at frame 1006's world pose!
    for t in range(5):
        assert np.allclose(aligned.anim_world[t], aligned.anim_world[5], atol=1e-6)
    # Hidden post-exit frames 1017..1020 must stay 100% motionless in 3D world space at frame 1016's world pose!
    for t in range(16, 20):
        assert np.allclose(aligned.anim_world[t], aligned.anim_world[15], atol=1e-6)
    # Entire 20-frame sequence never flies away
    hips_world = aligned.anim_world[:, 0, :3, 3]
    assert float(np.max(np.linalg.norm(hips_world, axis=1))) < 900.0

    # Export to USD (.usdc), FBX (.fbx), and Alembic (.abc) and verify visibility & frozen geometry round-trip!
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        usdc_p = export_scene(root / "vis_test.usdc", [aligned], [cam], shot_frames, fps=24.0, unit="cm")
        fbx_p = export_scene(root / "vis_test.fbx", [aligned], [cam], shot_frames, fps=24.0, unit="cm")
        abc_p = export_scene(root / "vis_test.abc", [aligned], [cam], shot_frames, fps=24.0, unit="cm")

        usd_chars = load_skinned_characters(usdc_p)
        assert usd_chars[0].frames == shot_frames
        assert usd_chars[0].visible.tolist() == expected_vis

        fbx_chars = load_skinned_characters(fbx_p)
        assert fbx_chars[0].frames == shot_frames
        assert fbx_chars[0].visible.tolist() == expected_vis

        abc_info = inspect_3d_file(abc_p)
        assert len(abc_info["models"]) == 1
        abc_m = abc_info["models"][0]
        assert abc_m["frames"] == shot_frames
        assert abc_m["visible"] == expected_vis
        centroids = np.asarray(abc_m["centroids"], dtype=np.float64)
        for t in range(5):
            assert np.allclose(centroids[t], centroids[5], atol=1e-4)
        for t in range(16, 20):
            assert np.allclose(centroids[t], centroids[15], atol=1e-4)


def test_rigid_object_alignment_and_maya_parent_transform_export() -> None:
    """Verify RigidObjectTrack camera alignment (exact pixel reprojection, unit scale, world-space freeze on
    hidden frames) and Maya-ready parent-transform TRS + static child proxy_mesh across USD, ABC, and FBX."""
    shot_frames = [1001, 1002, 1003, 1004, 1005, 1006]
    w, h = 1280, 720
    f_std = standard_camera_focal_px(w, h)

    # Static cube centered at local (0, 0, 0)
    rest_pts = np.array(
        [
            [-15, -10, -12],
            [15, -10, -12],
            [15, 10, -12],
            [-15, 10, -12],
            [-15, -10, 12],
            [15, -10, 12],
            [15, 10, 12],
            [-15, 10, 12],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
        ],
        dtype=np.int32,
    )
    vcols = np.full((8, 3), [0.9, 0.45, 0.2], dtype=np.float32)

    # Frames 1001 (hidden pre-entry), 1002..1005 (visible moving + rotating), 1006 (hidden post-exit)
    vis = np.array([0, 1, 1, 1, 1, 0], dtype=np.uint8)
    anc = np.array([1002, 1002, 1003, 1004, 1005, 1005], dtype=np.int64)
    wt = np.repeat(np.eye(4, dtype=np.float64)[None], 6, axis=0)
    for idx in range(1, 5):
        ang = 0.15 * (idx - 1)
        ca, sa = np.cos(ang), np.sin(ang)
        wt[idx, :3, :3] = [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]
        wt[idx, :3, 3] = [-40.0 + idx * 25.0, 15.0, -320.0]
    wt[0] = wt[1]
    wt[5] = wt[4]

    obj = RigidObjectTrack(
        name="prop_crate_01",
        object_id=1,
        frames=shot_frames,
        rest_points=rest_pts,
        faces=faces,
        world_transforms=wt,
        vertex_colors=vcols,
        visible=vis,
        anchor_frames=anc,
        source_focal_px=f_std,
    )

    # Projected center at frame 1003 (idx=2) in standard camera
    p_std = obj.world_transforms[2, :3, 3]
    u_std = f_std * (p_std[0] / (-p_std[2])) + w * 0.5
    v_std = f_std * (-p_std[1] / (-p_std[2])) + h * 0.5

    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 6, axis=0)
    c2w[:, 0, 3] = np.linspace(50.0, 200.0, 6)
    c2w[:, 1, 3] = 140.0
    cam = CameraTrack(
        name="mayaTrackCam",
        path="/shot/mayaTrackCam",
        frames=tuple(shot_frames),
        cam_to_world=c2w,
        focal_mm=np.full(6, 65.0),
        h_aperture_mm=np.full(6, 36.0),
        v_aperture_mm=np.full(6, 20.25),
        center_mm=np.tile([1.8, -0.9], (6, 1)),
        width=w,
        height=h,
    )
    aligned_obj = align_standard_object_to_camera(obj, cam, w, h, shot_frames)

    # 1. Pure rigid transform (unit scale = 1, det(R) = +1)
    for t in range(6):
        r_t = aligned_obj.world_transforms[t, :3, :3]
        assert np.allclose(r_t.T @ r_t, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(r_t) - 1.0) < 1e-6

    # 2. Exact pixel reprojection match in target camera
    p_cam = np.linalg.inv(c2w[2]) @ np.append(aligned_obj.world_transforms[2, :3, 3], 1.0)
    k2 = cam.intrinsics_per_frame(w, h)[1003]
    u_new = k2[0, 0] * (p_cam[0] / (-p_cam[2])) + k2[0, 2]
    v_new = k2[1, 1] * (-p_cam[1] / (-p_cam[2])) + k2[1, 2]
    assert abs(u_new - u_std) < 1e-3 and abs(v_new - v_std) < 1e-3

    # 3. Hidden frames 1001 & 1006 stay motionless in 3D world space despite moving camera!
    assert np.allclose(aligned_obj.world_transforms[0], aligned_obj.world_transforms[1], atol=1e-6)
    assert np.allclose(aligned_obj.world_transforms[5], aligned_obj.world_transforms[4], atol=1e-6)

    # 4. Export across USD, ABC, and FBX and verify parent-transform + static local mesh + visibility!
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for ext in (".usdc", ".abc", ".fbx"):
            p = export_scene(
                root / f"obj_scene{ext}",
                characters=[],
                cameras=[cam],
                shot_frames=shot_frames,
                fps=24.0,
                unit="cm",
                rigid_objects=[aligned_obj],
            )
            info = inspect_3d_file(p)
            assert len(info["models"]) == 1, f"{ext} missing rigid object model: {info}"
            m = info["models"][0]
            assert m["visible"] == vis.tolist(), f"{ext} vis mismatch: {m['visible']}"

        # Verify USD explicitly has animated parent Xform + static child proxy_mesh at (0,0,0)!
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(root / "obj_scene.usdc"))
        parent_prim = stage.GetPrimAtPath("/shot/prop_crate_01")
        mesh_prim = stage.GetPrimAtPath("/shot/prop_crate_01/proxy_mesh")
        assert parent_prim.IsValid() and mesh_prim.IsValid()
        mesh_pts_samples = UsdGeom.Mesh(mesh_prim).GetPointsAttr().GetTimeSamples()
        assert len(mesh_pts_samples) == 0, "Child proxy_mesh points must be static (zero time samples) for Maya swap!"
        pts_static = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float32)
        assert np.allclose(pts_static.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-4)

        # 5. Verify a FIXED / STATIONARY rigid object exports with ZERO animation keyframes across USD, FBX, and ABC!
        m_fixed = np.eye(4, dtype=np.float64)
        m_fixed[:3, 3] = [85.0, 24.0, -290.0]
        static_obj = RigidObjectTrack(
            name="fixed_table_01",
            object_id=2,
            frames=shot_frames,
            rest_points=rest_pts,
            faces=faces,
            world_transforms=np.repeat(m_fixed[None], len(shot_frames), axis=0),
            vertex_colors=vcols,
            visible=np.ones(len(shot_frames), dtype=np.int32),
            source_focal_px=f_std,
        )
        for ext in (".usdc", ".abc", ".fbx"):
            sp = export_scene(
                root / f"static_obj{ext}",
                characters=[],
                cameras=[cam],
                shot_frames=shot_frames,
                fps=24.0,
                unit="cm",
                rigid_objects=[static_obj],
            )
            s_info = inspect_3d_file(sp)
            assert len(s_info["models"]) == 1, f"{ext} missing static object"
            # Static object in USD/FBX/ABC has only 1 static sample (no per-frame animation keys)!
            assert len(s_info["models"][0]["frames"]) == 1, (
                f"{ext} wrote animation keyframes on a stationary object: {s_info['models'][0]['frames']}"
            )

        st_usd = Usd.Stage.Open(str(root / "static_obj.usdc"))
        st_xf = UsdGeom.Xformable(st_usd.GetPrimAtPath("/shot/fixed_table_01"))
        for op in st_xf.GetOrderedXformOps():
            assert len(op.GetTimeSamples()) == 0, f"USD static object op {op.GetOpName()} must have 0 timeSamples!"
        st_vis = UsdGeom.Imageable(st_usd.GetPrimAtPath("/shot/fixed_table_01")).GetVisibilityAttr()
        assert len(st_vis.GetTimeSamples()) == 0, "USD static object visibility must have 0 timeSamples!"


def test_admin_cache_dir_switch_and_protection() -> None:
    """Verify `/admin` cache directory live switching to a custom server/NAS directory, automatic creation,
    protected-folder rejection, and instant restore to default."""
    from shotspace.server_config import ConfigStore, validate_and_prepare_cache_dir
    from shotspace.web.workspace import WorkspaceManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg_file = root / "server_config.json"
        nas_dir = root / "mnt_company_nas" / "shotspace_cache"

        store = ConfigStore(cfg_file)
        store.initialize(port=8765, password="test_admin_pwd")
        wm = WorkspaceManager(config_store=store)

        # Protected code/weights directories must be rejected
        for bad in ("/", str(Path(__file__).resolve().parents[1] / "shotspace")):
            rejected = False
            try:
                validate_and_prepare_cache_dir(bad)
            except ValueError:
                rejected = True
            assert rejected, f"Expected protected directory {bad} to be rejected!"

        # Live switch to simulated company server path `nas_dir`
        stats = wm.switch_cache_dir(str(nas_dir))
        assert Path(stats["cache_dir"]) == nas_dir.resolve()
        assert stats["is_custom_cache_dir"] is True
        assert nas_dir.is_dir()
        assert store.load().cache_dir == str(nas_dir.resolve())

        # Create a shot in the new server cache path and verify it persists on `nas_dir`
        sw = wm.create_shot("NAS 缓存测试任务")
        assert sw.root.parent == nas_dir.resolve()
        assert (nas_dir / sw.shot_id / "shot.json").is_file()

        # Restore default path (`""`)
        restored = wm.switch_cache_dir("")
        assert restored["is_custom_cache_dir"] is False
        assert store.load().cache_dir == ""


def test_separate_scene_scan_abc_fbx_usd_import_and_calibration() -> None:
    """Verify separate camera (`.abc`/`.fbx`/`.usdc`) + static scene scan (`.abc`/`.fbx`/`.usdc`) import,
    multi-format world-space mesh loading, and raycast depth calibration onto scanned surfaces."""
    import json

    from shotspace.core.camera import CameraTrack
    from shotspace.core.camera_space import standard_camera_focal_px
    from shotspace.core.rigid_object import RigidObjectTrack
    from shotspace.io import export_scene, inspect_3d_file, load_scene_scan
    from shotspace.server_config import ConfigStore
    from shotspace.web.workspace import WorkspaceManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w, h, fps = 960, 540, 24.0
        frames = [1001, 1002, 1003, 1004, 1005, 1006]
        n = len(frames)
        f_std = standard_camera_focal_px(w, h)
        f_mm = 36.0 * f_std / w

        # Moving camera orbiting slightly around origin looking toward -Z
        c2w = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
        for i in range(n):
            c2w[i, 0, 3] = -20.0 + 8.0 * i
            c2w[i, 1, 3] = 110.0
            c2w[i, 2, 3] = 0.0
        cam = CameraTrack(
            name="matchmoveCam",
            path="/shot/matchmoveCam",
            frames=tuple(frames),
            cam_to_world=c2w,
            focal_mm=np.full(n, f_mm),
            h_aperture_mm=np.full(n, 36.0),
            v_aperture_mm=np.full(n, 20.25),
            width=w,
            height=h,
            fps=fps,
        )
        cam_abc = export_scene(root / "camera_only.abc", characters=[], cameras=[cam], shot_frames=frames, fps=fps, unit="cm")

        # Separate unified static scene scan mesh: ground quad (Y=0) + scanned prop block centered at (0, 80, -260)
        # Front face sits at Z = -245 cm (cube half-size = 15 cm around Z = -260 cm)
        floor_v = np.array(
            [[-250.0, 0.0, -50.0], [250.0, 0.0, -50.0], [250.0, 0.0, -450.0], [-250.0, 0.0, -450.0]],
            dtype=np.float32,
        )
        floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        s = 15.0
        box_v = np.array(
            [
                [-s, 80 - s, -260 - s],
                [s, 80 - s, -260 - s],
                [s, 80 + s, -260 - s],
                [-s, 80 + s, -260 - s],
                [-s, 80 - s, -260 + s],
                [s, 80 - s, -260 + s],
                [s, 80 + s, -260 + s],
                [-s, 80 + s, -260 + s],
            ],
            dtype=np.float32,
        )
        box_f = (
            np.array(
                [
                    [4, 5, 6],
                    [4, 6, 7],
                    [1, 0, 3],
                    [1, 3, 2],
                    [0, 4, 7],
                    [0, 7, 3],
                    [5, 1, 2],
                    [5, 2, 6],
                    [3, 7, 6],
                    [3, 6, 2],
                    [0, 1, 5],
                    [0, 5, 4],
                ],
                dtype=np.int32,
            )
            + 4
        )
        scan_verts = np.concatenate([floor_v, box_v], axis=0)
        scan_faces = np.concatenate([floor_f, box_f], axis=0)
        scan_obj_export = RigidObjectTrack(
            name="unified_lidar_scan",
            object_id=99,
            frames=tuple(frames),
            rest_points=scan_verts,
            faces=scan_faces,
            world_transforms=np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0),
            visible=np.ones(n, dtype=np.int32),
        )

        # Export the unified static scan in all 3 studio formats (.abc, .fbx, .usdc) WITHOUT camera!
        for ext in (".abc", ".fbx", ".usdc"):
            sp = export_scene(
                root / f"lidar_scan_only{ext}",
                characters=[],
                cameras=[],
                shot_frames=frames,
                fps=fps,
                unit="cm",
                rigid_objects=[scan_obj_export],
            )
            loaded = load_scene_scan(sp, unit="auto")
            assert loaded.raw_face_count == len(scan_faces), f"{ext} face count mismatch"
            assert abs(loaded.bbox_min_cm[0] - (-250.0)) < 1e-2, f"{ext} bbox_min X mismatch: {loaded.bbox_min_cm}"
            assert abs(loaded.bbox_max_cm[1] - 95.0) < 1e-2, f"{ext} bbox_max Y mismatch: {loaded.bbox_max_cm}"

        # Create a shot with a solved static object initially sitting on a STATIC tripod camera (no parallax!)
        # at wrong monocular depth Z = -160 cm instead of the true scanned Z = -260 cm!
        tripod_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
        tripod_c2w[:, 1, 3] = 110.0
        tripod_cam = CameraTrack(
            name="tripodCam",
            path="/shot/tripodCam",
            frames=tuple(frames),
            cam_to_world=tripod_c2w,
            focal_mm=np.full(n, f_mm),
            h_aperture_mm=np.full(n, 36.0),
            v_aperture_mm=np.full(n, 20.25),
            width=w,
            height=h,
            fps=fps,
        )
        tripod_usdc = export_scene(
            root / "tripod_cam.usdc", characters=[], cameras=[tripod_cam], shot_frames=frames, fps=fps, unit="cm"
        )

        store = ConfigStore(root / "cfg.json")
        store.initialize(port=8765, password="test")
        wm = WorkspaceManager(workspaces_dir=root / "ws", config_store=store)
        sw = wm.create_shot("独立相机与扫描校准测试")
        sw.update_meta(plate={"width": w, "height": h, "fps": fps, "frames": frames})

        # True box at (0, 80, -260) in world -> in tripod_cam (at Y=110) it is at (0, -30, -260) with half-size 15 cm.
        # Suppose monocular solver guessed scale = 160/260 -> cam pos (0, -30*160/260, -160) & half-size 15*160/260!
        wrong_ratio = 160.0 / 260.0
        local_cube = (box_v - np.array([[0.0, 80.0, -260.0]], dtype=np.float32)) * wrong_ratio
        m_wrong = np.eye(4, dtype=np.float64)
        m_wrong[:3, 3] = [0.0, -30.0 * wrong_ratio, -160.0]
        # Compute exact 2D projected bbox of the true object in `tripod_cam`
        pts_true_cam = box_v.astype(np.float64) - np.array([[0.0, 110.0, 0.0]])
        u_proj = w * 0.5 + f_std * (pts_true_cam[:, 0] / (-pts_true_cam[:, 2]))
        v_proj = h * 0.5 - f_std * (pts_true_cam[:, 1] / (-pts_true_cam[:, 2]))
        bx_2d = np.array([u_proj.min(), v_proj.min(), u_proj.max(), v_proj.max()], dtype=np.float32)

        raw_track = RigidObjectTrack(
            name="scanned_prop_01",
            object_id=1,
            frames=tuple(frames),
            rest_points=local_cube,
            faces=box_f - 4,
            world_transforms=np.repeat(m_wrong[None], n, axis=0),
            visible=np.ones(n, dtype=np.int32),
            source_focal_px=f_std,
            mask_boxes=np.repeat(bx_2d[None], n, axis=0),
        )
        obj_cache_dir = sw.root / "solve_objects" / "cam_space"
        obj_cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(obj_cache_dir / "object_01.npz", **raw_track.to_npz_dict())
        sw.update_meta(
            solved_objects=[{"id": 1, "name": "scanned_prop_01", "file": "object_01.npz", "frames": frames}],
            active_object_ids=[1],
        )

        # 1) Import camera ONLY (USD) -> depth stays at uncalibrated ~-160 cm
        wm.set_camera(sw.shot_id, tripod_usdc)
        exp_before = wm.export_single_path(sw.shot_id, "usdc")
        info_before = inspect_3d_file(exp_before)
        assert len(info_before["models"]) == 1

        # 2) Separately import scene scan (FBX): it is SHOWN, and nothing is put on it yet (v0.2.7: the scan is
        #    used when the user presses 按扫描对齐) -> the object stays at its uncalibrated depth
        from shotspace.io.abc_io import _import_abc_arrays

        def exported_depth() -> float:
            _, items = _import_abc_arrays(wm.export_single_path(sw.shot_id, "abc"))
            return float(np.asarray(items["model"][0]["world"], dtype=np.float64)[0][2, 3])

        depth_camera_only = exported_depth()
        meta_after = wm.set_scene_scan(sw.shot_id, root / "lidar_scan_only.fbx")
        assert meta_after["scene_scan"]["summary"]["format"] == "fbx"
        assert meta_after["scan_align"] is False and meta_after["scan_align_computed"] is False
        assert (sw.root / "aligned" / "view_scene.bin.gz").is_file()
        scene_json = json.loads((sw.root / "aligned" / "view_manifest.json").read_text(encoding="utf-8"))
        assert scene_json["scene_scan"] is not None
        assert abs(exported_depth() - depth_camera_only) < 1e-6 and abs(depth_camera_only - (-260.0)) > 50.0, depth_camera_only

        # 3) 按扫描对齐 -> raycast snaps object depth from -160 cm to scanned -260 cm; 对齐前 shows it as it was
        meta_on = wm.set_scan_alignment(sw.shot_id, True)
        assert meta_on["scan_align"] is True and meta_on["scan_align_computed"] is True and meta_on["status"] != "queued"
        depth_on_scan = exported_depth()
        assert wm.set_scan_alignment(sw.shot_id, False)["scan_align"] is False
        assert abs(exported_depth() - depth_camera_only) < 1e-6
        assert wm.set_scan_alignment(sw.shot_id, True)["scan_align"] is True
        assert abs(exported_depth() - depth_on_scan) < 1e-6

        exp_after = wm.export_single_path(sw.shot_id, "abc")
        _, abc_items = _import_abc_arrays(exp_after)
        exported_world = np.asarray(abc_items["model"][0]["world"], dtype=np.float64)[0]
        # Calibrated object center Z must match the scanned box at Z = -260 cm within ~3 cm, with 1 static sample!
        assert abs(float(exported_world[2, 3]) - (-260.0)) < 4.0, (
            f"Expected scan-calibrated Z near -260 cm, got {exported_world[2, 3]:.2f} cm"
        )
        assert len(abc_items["model"][0]["frames"]) == 1, "Scan-calibrated stationary object must have 0 animation keys!"


def test_first_run_startup_script_port_and_password_requirement() -> None:
    """Verify `shotspace.sh` enforces first-run port + `/admin` password setup when `server_config.json` is absent,
    rejects non-interactive unconfigured startup, and persists custom port & hashed password."""
    if __import__("os").name == "nt":
        print("  (skipped on Windows: this drives the Linux start script shotspace.sh through a terminal)")
        return
    import os
    import subprocess

    from shotspace.server_config import ConfigStore

    repo_root = Path(__file__).resolve().parents[1]
    sh_script = repo_root / "shotspace.sh"

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "server_config.json"
        env = {**os.environ, "SHOTSPACE_CONFIG_FILE": str(cfg_path)}

        # 1. Without `server_config.json`, non-interactive `shotspace.sh start --init-only` with no password MUST fail
        #    (closing stdin + running in a detached session without /dev/tty)
        r_fail = subprocess.run(
            ["bash", str(sh_script), "start", "--init-only"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            start_new_session=True,
        )
        assert r_fail.returncode != 0, f"Expected non-interactive first run without password to fail: {r_fail.stdout} {r_fail.stderr}"
        assert not cfg_path.exists()

        # 2. Simulate full end-to-end interactive TTY `shotspace.sh start` (NO --init-only!),
        #    verifying real daemon startup, `/api/health` 200 OK, `status`, `restart`, `doctor`, and `stop`!
        import json
        import pty
        import shutil
        import socket
        import urllib.request

        # Pick a free ephemeral localhost port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_free:
            s_free.bind(("127.0.0.1", 0))
            free_port = int(s_free.getsockname()[1])

        # Isolate APP_ROOT in temp dir symlinked to real `.venv` & `third_party` so we don't touch live server PID!
        mock_app = Path(td) / "ShotSpace"
        mock_app.mkdir(parents=True, exist_ok=True)
        for item_name in ("shotspace", "webui", "config", ".venv", "third_party"):
            src_item = repo_root / item_name
            if src_item.exists():
                (mock_app / item_name).symlink_to(src_item)
        shutil.copy2(sh_script, mock_app / "shotspace.sh")
        (mock_app / "shotspace.sh").chmod(0o755)

        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            ["bash", str(mock_app / "shotspace.sh"), "start"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        buf = b""
        sent_step = 0
        try:
            while proc.poll() is None:
                import select

                rlist, _, _ = select.select([master_fd], [], [], 0.2)
                if rlist:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if chunk:
                        buf += chunk
                        text_now = buf.decode("utf-8", errors="ignore")
                        if sent_step == 0 and "监听端口" in text_now:
                            os.write(master_fd, f"{free_port}\n".encode())
                            sent_step = 1
                        elif sent_step == 1 and "后台管理密码" in text_now:
                            os.write(master_fd, b"MyStudioPwd!\n")
                            sent_step = 2
                        elif sent_step == 2 and "再次输入确认" in text_now:
                            os.write(master_fd, b"MyStudioPwd!\n")
                            sent_step = 3
        finally:
            try:
                while True:
                    import select

                    rlist, _, _ = select.select([master_fd], [], [], 0.05)
                    if not rlist:
                        break
                    c = os.read(master_fd, 4096)
                    if not c:
                        break
                    buf += c
            except OSError:
                pass
            os.close(master_fd)

        try:
            rc = proc.wait(timeout=30)
            out_text = buf.decode("utf-8", errors="ignore")
            assert rc == 0, f"Full interactive `shotspace.sh start` failed (rc={rc}):\n{out_text}"
            assert "已成功启动" in out_text, f"Expected startup banner in output:\n{out_text}"

            # Verify live HTTP `/api/health` on the configured port
            with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/api/health", timeout=5.0) as resp:
                h_json = json.loads(resp.read().decode("utf-8"))
                assert h_json["status"] == "ok" and h_json["configured_port"] == free_port

            # Verify `status`, `restart`, and `doctor` all exit 0 cleanly!
            assert subprocess.run(["bash", str(mock_app / "shotspace.sh"), "status"], env=env, capture_output=True).returncode == 0
            r_res = subprocess.run(["bash", str(mock_app / "shotspace.sh"), "restart"], env=env, capture_output=True, text=True)
            assert r_res.returncode == 0, f"`shotspace.sh restart` failed:\n{r_res.stdout}\n{r_res.stderr}"
            r_doc = subprocess.run(["bash", str(mock_app / "shotspace.sh"), "doctor"], env=env, capture_output=True, text=True)
            assert r_doc.returncode == 0, f"`shotspace.sh doctor` failed:\n{r_doc.stdout}\n{r_doc.stderr}"
        finally:
            subprocess.run(["bash", str(mock_app / "shotspace.sh"), "stop"], env=env, capture_output=True)
            subprocess.run(["bash", str(sh_script), "relocate-venv"], capture_output=True)

        store = ConfigStore(cfg_path)
        loaded = store.load()
        assert loaded.initialized is True
        assert loaded.port == free_port
        assert store.verify_password("MyStudioPwd!") is True
        assert store.verify_password("wrong") is False


def test_windows_client_upload_zip_gbk_and_download_compatibility() -> None:
    """Verify a Windows 10/11 browser client accessing the Linux ShotSpace server handles:
      1. Windows backslash paths (`C:\\Users\\Artist\\镜头01\\frame_1001.png`) on sequence, camera & scan uploads;
      2. Chinese Windows Explorer GBK-encoded `.zip` archives with `\\` separators, `Thumbs.db`, `desktop.ini`, and `__MACOSX`;
      3. Windows reserved device names (`CON`, `AUX`) and illegal filename chars (`<>:"/\\|?*`) on export;
      4. RFC 5987 UTF-8 `Content-Disposition` headers for Chinese-named `.usdc` / `.fbx` / `.abc` downloads;
      5. Concurrent multi-user thumbnail & export downloads on the same shot without file race corruption.
    """
    import io
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from shotspace.io.plate import write_png_rgb
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "cfg.json")
        cfg.initialize(port=8765, password="WinUserTest123")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)

        # 1. Create a shot with Chinese + Windows-special chars in its title
        r = client.post("/api/shots", json={"name": '走廊镜头:01*测试?"<>|'})
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

        # Prepare a dummy 64x48 RGB PNG in memory
        png_path = root / "sample.png"
        rgb = np.full((48, 64, 3), 128, dtype=np.uint8)
        write_png_rgb(png_path, rgb)
        png_bytes = png_path.read_bytes()

        # 2. Upload frame using Windows full path with drive letter + backslashes
        assert client.post(f"/api/shots/{sid}/sequence/begin").status_code == 200
        r_f1 = client.post(
            f"/api/shots/{sid}/sequence/frame",
            params={"filename": r"D:\Projects\测试项目\镜头A\shot_1001.png"},
            content=png_bytes,
        )
        assert r_f1.status_code == 200 and r_f1.json()["filename"] == "shot_1001.png", r_f1.text

        # 3. Build a real Windows Explorer-style non-UTF8 (flag_bits=0, GBK-encoded) .zip with `\` paths & junk files!
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name_str, payload in [
                ("实拍序列\\shot_1002.png", png_bytes),
                ("实拍序列\\shot_1003.png", png_bytes),
                ("实拍序列\\Thumbs.db", b"junk"),
                ("实拍序列\\desktop.ini", b"junk"),
                ("__MACOSX\\shot_1002.png", b"corrupt_mac_fork"),
            ]:
                zi = zipfile.ZipInfo()
                # Simulate Chinese Windows CP936 zip header (flag_bits without 0x800 UTF-8 bit!)
                zi.filename = name_str.encode("gb18030").decode("cp437")
                zi.flag_bits = 0
                zf.writestr(zi, payload)

        r_zip = client.post(
            f"/api/shots/{sid}/sequence/frame",
            params={"filename": r"C:\Users\Administrator\Desktop\实拍序列包.zip"},
            content=zip_buf.getvalue(),
        )
        assert r_zip.status_code == 200 and r_zip.json()["extracted"] == 2, r_zip.text

        r_fin = client.post(
            f"/api/shots/{sid}/sequence/finalize",
            json={"input_colorspace": "auto", "ocio_view": "aces_sdr", "fps": 24.0},
        )
        assert r_fin.status_code == 200, r_fin.text
        assert r_fin.json()["plate"]["frames"] == [1001, 1002, 1003]

        # 4. Concurrent multi-user thumbnail fetch (8 Windows browser tabs hitting /thumbnail simultaneously)
        with ThreadPoolExecutor(max_workers=8) as pool:
            thumbs = list(pool.map(lambda _: client.get(f"/api/shots/{sid}/thumbnail"), range(8)))
        assert all(t.status_code == 200 and len(t.content) > 100 for t in thumbs)

        # 5. Upload camera & scene scan with Windows backslash paths + export with Chinese & Windows-reserved names
        cam_abc = Path(__file__).resolve().parent / "assets" / "gt_studio_benchmark_cam.abc"
        scan_abc = Path(__file__).resolve().parent / "assets" / "gt_studio_scene_scan.abc"
        r_cam = client.post(
            f"/api/shots/{sid}/camera/upload",
            params={"filename": r"E:\Matchmove\镜头A\gt_studio_cam.abc"},
            content=cam_abc.read_bytes(),
        )
        assert r_cam.status_code == 200, r_cam.text
        r_scan = client.post(
            f"/api/shots/{sid}/scene_scan/upload",
            params={"filename": r"E:\Lidar\现场扫描\gt_studio_scene_scan.abc"},
            content=scan_abc.read_bytes(),
        )
        assert r_scan.status_code == 200, r_scan.text

        # Seed a dummy rigid object in cam_space so we can test concurrent export & Windows download headers
        sw = app.state.wm.get_shot(sid)
        obj_dir = sw.root / "solve_objects" / "cam_space"
        obj_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            obj_dir / "object_01.npz",
            object_id=np.int64(1),
            name=np.array("prop_box"),
            frames=np.array([1001, 1002, 1003], dtype=np.int64),
            solved_frames=np.array([1001, 1002, 1003], dtype=np.int64),
            visible=np.array([1, 1, 1], dtype=np.uint8),
            anchor_frames=np.array([1001, 1002, 1003], dtype=np.int64),
            mask_boxes=np.tile(np.array([[20.0, 15.0, 44.0, 35.0]], dtype=np.float32), (3, 1)),
            rest_points=np.array([[-5, -5, -5], [5, -5, -5], [0, 5, -5], [0, 0, 5]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32),
            vertex_colors=np.ones((4, 3), dtype=np.float32) * 0.8,
            world_transforms=np.repeat(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -200.0], [0, 0, 0, 1]], dtype=np.float64)[None],
                3,
                axis=0,
            ),
            source_focal_px=np.float64(80.0),
        )

        # Test Windows reserved name "CON" -> must become "_CON.usdc"
        r_con = client.post(f"/api/shots/{sid}/export", json={"formats": ["usdc"], "prefix": "CON"})
        assert r_con.status_code == 200 and r_con.json()["exports"][0]["filename"] == "_CON.usdc"

        # Test concurrent Chinese filename downloads (`走廊镜头_01_测试_____.usdc`)
        with ThreadPoolExecutor(max_workers=4) as pool:
            dls = list(pool.map(lambda _: client.get(f"/api/shots/{sid}/download/usdc"), range(4)))
        assert all(d.status_code == 200 and len(d.content) > 200 for d in dls)
        cd = dls[0].headers.get("content-disposition", "")
        assert "utf-8''" in cd.lower(), f"Expected RFC 5987 UTF-8 filename header for Windows browser, got: {cd}"


def test_sequence_validation_interception_and_safe_staging() -> None:
    """Verify strict plate sequence validation & interception (`基础检测拦截`):
      - Missing frames (`缺帧` with exact frame ranges)
      - Per-frame resolution mismatch (`某帧分辨率不一致`)
      - Corrupted / 0-byte empty frames (`损坏帧 / 空文件`)
      - Duplicate frame numbers (`重复帧号`), unnumbered stills (`无帧号`), mixed sequence prefixes (`多序列混杂`)
      - Video file upload rejection & `< 2` single-frame mistake rejection
      - Staging isolation: a rejected replacement upload NEVER wipes or corrupts an already-loaded valid sequence!
    """
    from fastapi.testclient import TestClient
    from shotspace.io.plate import ingest_sequence_dir, write_exr_half_rgb, write_png_rgb
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rgb_ok = np.full((48, 64, 3), 120, dtype=np.uint8)
        rgb_wrong_res = np.full((72, 96, 3), 120, dtype=np.uint8)

        def _expect_intercept(subdir: str, setup_fn, must_contain: tuple[str, ...]) -> None:
            sdir = root / subdir
            sdir.mkdir(parents=True, exist_ok=True)
            setup_fn(sdir)
            try:
                ingest_sequence_dir(sdir, root / f"cache_{subdir}")
            except ValueError as exc:
                msg = str(exc)
                for token in must_contain:
                    assert token in msg, f"Expected {token!r} in interception message, got: {msg!r}"
                return
            raise AssertionError(f"Expected sequence {subdir!r} to be intercepted, but it succeeded!")

        # 1. Missing frames (`缺帧`: 1001, 1002, 1005 -> missing 1003–1004)
        _expect_intercept(
            "missing_frames",
            lambda d: [write_png_rgb(d / f"shot.{f:04d}.png", rgb_ok) for f in (1001, 1002, 1005)],
            ("缺帧", "1003–1004"),
        )

        # 2. Per-frame resolution mismatch (`某帧分辨率不一致`: frame 1002 is 96x72 vs 64x48, both PNG and EXR)
        def _setup_res_mismatch_png(d: Path) -> None:
            write_png_rgb(d / "shot.1001.png", rgb_ok)
            write_png_rgb(d / "shot.1002.png", rgb_wrong_res)
            write_png_rgb(d / "shot.1003.png", rgb_ok)

        _expect_intercept("res_mismatch_png", _setup_res_mismatch_png, ("分辨率不一致", "96×72", "64×48"))

        def _setup_res_mismatch_exr(d: Path) -> None:
            write_exr_half_rgb(d / "shot.1001.exr", rgb_ok.astype(np.float32) / 255.0)
            write_exr_half_rgb(d / "shot.1002.exr", rgb_wrong_res.astype(np.float32) / 255.0)

        _expect_intercept("res_mismatch_exr", _setup_res_mismatch_exr, ("分辨率不一致", "96×72", "64×48"))

        # 3. Corrupted / 0-byte frame
        def _setup_zero_byte(d: Path) -> None:
            write_png_rgb(d / "shot.1001.png", rgb_ok)
            (d / "shot.1002.png").write_bytes(b"")
            write_png_rgb(d / "shot.1003.png", rgb_ok)

        _expect_intercept("zero_byte", _setup_zero_byte, ("损坏或无法读取", "0 字节"))

        def _setup_corrupt_bytes(d: Path) -> None:
            write_png_rgb(d / "shot.1001.png", rgb_ok)
            (d / "shot.1002.png").write_bytes(b"corrupted_binary_header")

        _expect_intercept("corrupt_bytes", _setup_corrupt_bytes, ("损坏或无法读取", "shot.1002.png"))

        # 4. Duplicate frame numbers
        def _setup_dup_frame(d: Path) -> None:
            write_png_rgb(d / "shot.1001.png", rgb_ok)
            write_png_rgb(d / "shot.1002.png", rgb_ok)
            write_png_rgb(d / "sub" / "shot.01002.png", rgb_ok)

        _expect_intercept("dup_frame", _setup_dup_frame, ("重复帧号", "1002"))

        # 5. Mixed sequence prefixes
        def _setup_mixed_prefix(d: Path) -> None:
            write_png_rgb(d / "shotA.1001.png", rgb_ok)
            write_png_rgb(d / "shotA.1002.png", rgb_ok)
            write_png_rgb(d / "shotB.1003.png", rgb_ok)

        _expect_intercept("mixed_prefix", _setup_mixed_prefix, ("多个不同前缀",))

        # 6. Unnumbered still mixed into sequence
        def _setup_unnumbered(d: Path) -> None:
            write_png_rgb(d / "shot.1001.png", rgb_ok)
            write_png_rgb(d / "shot.1002.png", rgb_ok)
            write_png_rgb(d / "cleanplate_ref.png", rgb_ok)

        _expect_intercept("unnumbered", _setup_unnumbered, ("无帧号命名", "cleanplate_ref.png"))

        # 7. Video-only folder & single-frame mistake
        _expect_intercept("video_only", lambda d: (d / "take01.mov").write_bytes(b"fake_mov"), ("暂不支持直接导入视频文件",))
        _expect_intercept("single_frame", lambda d: write_png_rgb(d / "shot.1001.png", rgb_ok), ("至少需要连续 2 帧以上",))

        # 8. Safe staging rollback via HTTP API: rejected upload MUST NOT destroy existing valid shot sequence!
        cfg = ConfigStore(root / "cfg.json")
        cfg.initialize(port=8765, password="StageSafe123")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)

        sid = client.post("/api/shots", json={"name": "安全隔离拦截测试"}).json()["id"]
        png_ok_p = root / "ok.png"
        png_bad_res_p = root / "bad_res.png"
        write_png_rgb(png_ok_p, rgb_ok)
        write_png_rgb(png_bad_res_p, rgb_wrong_res)

        assert client.post(f"/api/shots/{sid}/sequence/begin").status_code == 200
        for f in (1001, 1002, 1003):
            assert client.post(
                f"/api/shots/{sid}/sequence/frame",
                params={"filename": f"good.{f:04d}.png"},
                content=png_ok_p.read_bytes(),
            ).status_code == 200
        r_good = client.post(f"/api/shots/{sid}/sequence/finalize", json={})
        assert r_good.status_code == 200 and r_good.json()["plate"]["frames"] == [1001, 1002, 1003]

        # Now attempt to upload a sequence with missing frame 1002 AND mismatched resolution on frame 1003
        assert client.post(f"/api/shots/{sid}/sequence/begin").status_code == 200
        client.post(f"/api/shots/{sid}/sequence/frame", params={"filename": "bad.1001.png"}, content=png_ok_p.read_bytes())
        client.post(f"/api/shots/{sid}/sequence/frame", params={"filename": "bad.1003.png"}, content=png_bad_res_p.read_bytes())
        r_bad = client.post(f"/api/shots/{sid}/sequence/finalize", json={})
        assert r_bad.status_code == 400 and "缺帧" in r_bad.json()["detail"]

        # Verify previous good plate [1001, 1002, 1003] is 100% preserved and playable!
        cur_shot = client.get(f"/api/shots/{sid}").json()
        assert cur_shot["plate"]["frames"] == [1001, 1002, 1003]
        assert client.get(f"/api/shots/{sid}/frames/1002").status_code == 200


def test_studio_meter_unit_and_multi_hierarchy_selection() -> None:
    """Verify studio-default meter ('m') camera & scan import (preventing 100x displacement shrinkage),
    Outliner hierarchy tree extraction across .abc/.fbx/.usd, smart default non-persp camera selection,
    and hierarchy-path subset filtering on multi-object scene scans."""
    from fastapi.testclient import TestClient

    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frames = [1001, 1002, 1003, 1004, 1005]

        # Build a multi-camera scene in raw METER numbers (e.g. 4.80 meters dolly travel!)
        # Note: export with unit="cm" so raw numbers in file are [0.0 .. 4.80] (simulating DCC exporting
        # meter-scale numeric coordinates even when file header is omitted in .abc or left at cm in .fbx/.usd!)
        persp_c2w = np.eye(4, dtype=np.float64)[None]
        persp_cam = CameraTrack(
            name="persp",
            path="/MayaViews/persp",
            frames=(1001,),
            cam_to_world=persp_c2w,
            focal_mm=np.array([35.0]),
            h_aperture_mm=np.array([36.0]),
            v_aperture_mm=np.array([20.25]),
        )

        wit_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 5, axis=0)
        wit_c2w[:, 0, 3] = np.linspace(-1.2, 0.0, 5)  # 1.20 m travel
        wit_c2w[:, 1, 3] = 1.65
        wit_cam = CameraTrack(
            name="witnessCam_B",
            path="/CamRig/WitnessGrp/witnessCam_B",
            frames=tuple(frames),
            cam_to_world=wit_c2w,
            focal_mm=np.full(5, 24.0),
            h_aperture_mm=np.full(5, 36.0),
            v_aperture_mm=np.full(5, 20.25),
        )

        shot_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 5, axis=0)
        shot_c2w[:, 2, 3] = np.linspace(6.0, 1.2, 5)  # 4.80 m dolly travel!
        shot_c2w[:, 1, 3] = 1.65
        shot_cam = CameraTrack(
            name="renderCam_01",
            path="/CamRig/MainShotGrp/renderCam_01",
            frames=tuple(frames),
            cam_to_world=shot_c2w,
            focal_mm=np.full(5, 50.0),
            h_aperture_mm=np.full(5, 36.0),
            v_aperture_mm=np.full(5, 20.25),
        )

        # Also build a multi-object static scene scan in raw METER numbers:
        # 1) /SetEnv/GroundGrp/floor_lidar: 10m x 10m floor at Y=0
        # 2) /SetEnv/GroundGrp/wall_lidar: 10m x 3m back wall at Z=-4m
        # 3) /SkyProxy/skydome_junk: huge 800m sky proxy box that MUST be excludable via hierarchy selector!
        floor_v = np.array([[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]], dtype=np.float32)
        quad_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        wall_v = np.array([[-5.0, 0.0, -4.0], [5.0, 0.0, -4.0], [5.0, 3.0, -4.0], [-5.0, 3.0, -4.0]], dtype=np.float32)
        sky_v = np.array([[-400.0, -400.0, -400.0], [400.0, -400.0, -400.0], [400.0, 400.0, 400.0], [-400.0, 400.0, 400.0]], dtype=np.float32)

        scan_objs = [
            RigidObjectTrack("floor_lidar", 1, [1001], np.eye(4)[None], floor_v, quad_f),
            RigidObjectTrack("wall_lidar", 2, [1001], np.eye(4)[None], wall_v, quad_f),
            RigidObjectTrack("skydome_junk", 3, [1001], np.eye(4)[None], sky_v, quad_f),
        ]

        cfg = ConfigStore(root / "cfg.json")
        cfg.initialize(port=8765, password="MeterHierarchy123")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)

        sid = client.post("/api/shots", json={"name": "米制与多层级选择测试"}).json()["id"]

        # Put a solved 170 cm human in cam_space so Step 5 export is enabled
        sw = app.state.wm.get_shot(sid)
        sw.update_meta(plate={"width": 1920, "height": 1080, "fps": 24.0, "frames": frames})
        bind = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        bind[1, 1, 3] = 170.0
        anim = np.repeat(bind[None], len(frames), axis=0)
        anim[:, :, 2, 3] = -350.0
        ch = SkinnedCharacter(
            name="person_01",
            person_id=1,
            frames=frames,
            joint_names=["Hips", "Head"],
            parents=np.array([-1, 0], dtype=np.int64),
            bind_world=bind,
            anim_world=anim,
            rest_points=np.array([[0, 0, 0], [0, 170, 0], [20, 0, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            joint_indices=np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32),
            joint_weights=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )
        cs_dir = sw.root / "solve_raw" / "cam_space"
        cs_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cs_dir / "person_01.npz",
            frames=np.array(frames, dtype=np.int64),
            joint_names=np.array(ch.joint_names),
            parents=ch.parents,
            bind_world=ch.bind_world,
            joint_world=ch.anim_world,
            rest_vertices=ch.rest_points,
            faces=ch.faces,
            skin_indices=ch.joint_indices,
            skin_weights=ch.joint_weights,
            visible=np.ones(len(frames), dtype=np.int32),
        )
        sw.update_meta(solved_people=[{"id": 1, "file": "person_01.npz", "frames": frames}], active_person_ids=[1])

        for ext in (".abc", ".fbx", ".usdc"):
            cam_file = export_scene(
                root / f"multi_cam{ext}",
                characters=[],
                cameras=[persp_cam, wit_cam, shot_cam],
                shot_frames=frames,
                fps=24.0,
                unit="cm",  # raw numbers in file = [0..4.80]
            )
            # Upload via default WebUI HTTP endpoint (no unit query param -> defaults to "m"!)
            r_up = client.post(
                f"/api/shots/{sid}/camera/upload",
                params={"filename": cam_file.name},
                content=cam_file.read_bytes(),
            )
            assert r_up.status_code == 200, r_up.text
            cam_meta = r_up.json()["camera"]
            assert cam_meta["unit"] == "m"
            assert len(cam_meta["listing"]["cameras"]) == 3
            assert len(cam_meta["listing"]["hierarchy"]) >= 1
            # Smart picker must skip static `persp` and pick `renderCam_01` (4.80 m -> 480.0 cm travel!)
            assert "renderCam_01" in cam_meta["summary"]["name"], cam_meta["summary"]
            assert abs(cam_meta["summary"]["travel_cm"] - 480.0) < 0.5, cam_meta["summary"]

            # Switch hierarchy camera node to witnessCam_B (1.20 m -> 120.0 cm travel!)
            wit_path = next(c["path"] for c in cam_meta["listing"]["cameras"] if "witnessCam_B" in c["name"])
            r_sw = client.post(
                f"/api/shots/{sid}/camera/options",
                json={"camera_path": wit_path, "unit": "m"},
            )
            assert r_sw.status_code == 200
            assert abs(r_sw.json()["camera"]["summary"]["travel_cm"] - 120.0) < 0.5

            # Test multi-mesh scene scan upload (defaults to "m"!) & hierarchy subset selection
            scan_file = export_scene(
                root / f"multi_scan{ext}",
                characters=[],
                cameras=[],
                shot_frames=[1001],
                fps=24.0,
                unit="cm",  # raw numbers in file = [-5..5] meters and [-400..400] skydome
                rigid_objects=scan_objs,
            )
            r_sc = client.post(
                f"/api/shots/{sid}/scene_scan/upload",
                params={"filename": scan_file.name},
                content=scan_file.read_bytes(),
            )
            assert r_sc.status_code == 200, r_sc.text
            sc_meta = r_sc.json()["scene_scan"]
            assert sc_meta["unit"] == "m"
            assert len(sc_meta["listing"]["meshes"]) == 3
            # Before filtering skydome, span X is 800m (80,000 cm)
            assert abs(sc_meta["summary"]["span_cm"][0] - 80000.0) < 10.0

            # Select ONLY floor_lidar via hierarchy path!
            floor_path = next(m["path"] for m in sc_meta["listing"]["meshes"] if "floor_lidar" in m["name"])
            r_sc_sel = client.post(
                f"/api/shots/{sid}/scene_scan/options",
                json={"unit": "m", "selected_paths": [floor_path]},
            )
            assert r_sc_sel.status_code == 200, r_sc_sel.text
            sc_filtered = r_sc_sel.json()["scene_scan"]["summary"]
            assert sc_filtered["mesh_count"] == 1 and sc_filtered["total_mesh_count"] == 3
            # Now span X is strictly 10.0m (1,000 cm), skydome completely excluded!
            assert abs(sc_filtered["span_cm"][0] - 1000.0) < 1.0

        # Finally verify Step 5 export defaults to "m" (matching meter camera 1:1!)
        r_exp = client.post(f"/api/shots/{sid}/export", json={"formats": ["usdc", "fbx", "abc"]})
        assert r_exp.status_code == 200
        for rec in r_exp.json()["exports"]:
            assert rec["unit"] == "m", f"Expected meter export, got {rec}"


def _upload_test_plate_and_camera(client, root: Path, sid: str, frames: list[int], moved: float = 0.0) -> None:
    """A tiny PNG plate plus an exact `.abc` camera, through the same HTTP calls the web page makes."""
    from shotspace.io.plate import write_png_rgb

    assert client.post(f"/api/shots/{sid}/sequence/begin").status_code == 200
    sample_png = root / "sample.png"
    write_png_rgb(sample_png, np.full((360, 640, 3), 90, dtype=np.uint8))
    png_bytes = sample_png.read_bytes()
    for f in frames:
        r_up = client.post(
            f"/api/shots/{sid}/sequence/frame?filename=shot.{f}.png",
            content=png_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r_up.status_code == 200
    assert client.post(f"/api/shots/{sid}/sequence/finalize", json={"fps": 24.0}).status_code == 200
    _upload_test_camera(client, root, sid, frames, moved=moved)


def _upload_test_camera(client, root: Path, sid: str, frames: list[int], moved: float = 0.0) -> None:
    n = len(frames)
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    c2w[:, 0, 3] = np.linspace(-40.0, 40.0, n) + moved
    c2w[:, 1, 3] = 150.0
    c2w[:, 2, 3] = 420.0
    cam = CameraTrack(
        name="trackCam",
        path="/trackCam",
        frames=tuple(frames),
        cam_to_world=c2w,
        focal_mm=np.full(n, 35.0),
        h_aperture_mm=np.full(n, 36.0),
        v_aperture_mm=np.full(n, 20.25),
        width=640,
        height=360,
        fps=24.0,
    )
    cam_abc = export_scene(root / f"track_cam_{int(moved)}.abc", [], [cam], frames, fps=24.0, unit="m")
    r_cam = client.post(
        f"/api/shots/{sid}/camera/upload?filename=track_cam.abc&unit=m",
        content=cam_abc.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r_cam.status_code == 200, r_cam.text


def _write_test_person(cam_dir: Path, pid: int, frames: list[int], x_cm: float, smoothing: float = 0.5, gen: int = 0, extra: dict | None = None) -> dict:
    """One solved person in the result layout of the whole-shot solver (camera space per frame, marked with
    `solver_version`, so the server only places it with the exact camera): two joints, one triangle, at `x_cm`."""
    n = len(frames)
    bind = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    bind[1, 1, 3] = 80.0
    anim = np.repeat(bind[None], n, axis=0)
    anim[:, :, 2, 3] = -380.0
    anim[:, :, 0, 3] = x_cm + np.linspace(-15.0, 15.0, n)[:, None]
    cam_dir.mkdir(parents=True, exist_ok=True)
    # like the real worker (`_atomic_save_npz`): a result file is never written into, it is replaced
    tmp_file = cam_dir / f".person_{pid:02d}.tmp.npz"
    arrays = dict(
        frames=np.array(frames, dtype=np.int64),
        solved_frames=np.array(frames, dtype=np.int64),
        visible=np.ones(n, dtype=np.int32),
        joint_names=np.array(["Hips", "Head"]),
        parents=np.array([-1, 0], dtype=np.int64),
        bind_world=bind,
        joint_world=anim.astype(np.float32),
        rest_vertices=np.array([[-10, 0, 0], [10, 0, 0], [0, 80, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        uv_faces=np.array([[0, 1, 2]], dtype=np.int32),
        skin_indices=np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32),
        skin_weights=np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32),
        has_real_camera=np.int32(1),
        solver_version=np.int32(2),
        smoothing_strength=np.float32(smoothing),
        gen=np.int64(gen),
    )
    arrays.update(extra or {})  # what a real result file holds besides (the data collection test)
    np.savez_compressed(tmp_file, **arrays)
    tmp_file.replace(cam_dir / f"person_{pid:02d}.npz")
    return {
        "id": int(pid),
        "file": f"person_{pid:02d}.npz",
        "frames": [int(f) for f in frames],
        "rejected_count": 0,
        "qc": {
            "fit_px": 1.0, "fit_px_p90": 2.0, "foot_hold_cm": 0.5, "planted_pct": 50.0, "body_scale": 1.0,
            "scale_source": "network", "scale_weight": 0.0, "solved_frames": n, "hidden_frames": 0,
        },
    }


def test_camera_first_workflow_on_demand_exports_and_glibc_compat() -> None:
    """1. ELF64 GLIBC <= 2.34 compatibility of the `.abc` / `.fbx` workers.
    2. Camera first: solving is refused with HTTP 400 until the step-2 camera is imported.
    3. Export files are built WHEN A DOWNLOAD IS ASKED FOR, one format at a time: nothing is built ahead of time,
       asking for one format builds exactly that file, and asking again with unchanged inputs returns the same file
       without building it again.  Changing what goes into the file (here: the unit) builds a new one.
    4. USD comes as binary Crate in two forms (`_skel.usd` UsdSkel, `_cache.usd` baked point cache), plus FBX, ABC.
    """
    import subprocess
    from fastapi.testclient import TestClient
    from pxr import Usd, UsdGeom, UsdSkel
    from shotspace.config import get_paths
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    paths = get_paths()
    for py_bin, mod_check in (
        (paths.alembic_python, "import alembic, imath, imathnumpy; print('ABC_OK')"),
        (paths.fbx_python, "import fbxio; print('FBX_OK')"),
    ):
        if py_bin.is_file():
            out = subprocess.check_output([str(py_bin), "-c", mod_check], text=True)
            assert "OK" in out

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        sid = client.post("/api/shots", json={"name": "cam_first_shot"}).json()["id"]
        frames = [1001, 1002, 1003]

        assert client.post(f"/api/shots/{sid}/sequence/begin").status_code == 200
        from shotspace.io.plate import write_png_rgb
        sample_png = root / "sample.png"
        write_png_rgb(sample_png, np.full((360, 640, 3), 90, dtype=np.uint8))
        for f in frames:
            assert client.post(
                f"/api/shots/{sid}/sequence/frame?filename=shot.{f}.png",
                content=sample_png.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            ).status_code == 200
        assert client.post(f"/api/shots/{sid}/sequence/finalize", json={"fps": 24.0}).status_code == 200

        # solving BEFORE the camera is imported must be refused
        r_no_cam = client.post(f"/api/shots/{sid}/solve", json={})
        assert r_no_cam.status_code == 400 and "摄影机" in r_no_cam.json()["detail"]
        r_obj_no_cam = client.post(f"/api/shots/{sid}/objects/solve", json={})
        # (an installation without the optional object matching says that first)
        assert r_obj_no_cam.status_code == 400 and (
            "摄影机" in r_obj_no_cam.json()["detail"] or "物体匹配没有安装" in r_obj_no_cam.json()["detail"])
        _upload_test_camera(client, root, sid, frames)

        wm = app.state.wm
        sw = wm.get_shot(sid)
        rec = _write_test_person(sw.root / "solve_raw" / "cam_space", 1, frames, x_cm=0.0)
        sw.update_meta(solved_people=[rec], active_person_ids=[1], solved_at="2026-11-22T23:59:00+0800")

        exp_dir = sw.root / "exports"
        built = lambda: sorted(f.name for f in exp_dir.glob("*") if f.is_file()) if exp_dir.is_dir() else []  # noqa: E731
        shot_meta = client.get(f"/api/shots/{sid}").json()
        assert built() == [] and not shot_meta.get("exports"), "nothing may be built before a download is asked for"
        assert "exports_ready" not in shot_meta and "prebuilt_exports" not in shot_meta

        # one download = one file
        r_skel = client.get(f"/api/shots/{sid}/download/usd_skel?unit=m")
        assert r_skel.status_code == 200 and r_skel.content[:8] == b"PXR-USDC"
        assert 'filename="cam_first_shot_skel.usd"' in r_skel.headers["content-disposition"]
        assert built() == ["cam_first_shot_skel.usd"], built()
        skel_usd = exp_dir / "cam_first_shot_skel.usd"

        # asking again does not build again (same file, untouched)
        mtime_ns = skel_usd.stat().st_mtime_ns
        assert client.get(f"/api/shots/{sid}/download/usd_skel?unit=m").status_code == 200
        assert skel_usd.stat().st_mtime_ns == mtime_ns and built() == ["cam_first_shot_skel.usd"]

        # the other formats, each on its own request
        for fmt, fname in (("usd_cache", "cam_first_shot_cache.usd"), ("fbx", "cam_first_shot.fbx"), ("abc", "cam_first_shot.abc")):
            before = set(built())
            r_dl = client.get(f"/api/shots/{sid}/download/{fmt}?unit=m")
            assert r_dl.status_code == 200 and len(r_dl.content) > 0, r_dl.text[:200]
            assert set(built()) - before == {fname}, (fmt, built())
        assert skel_usd.stat().st_mtime_ns == mtime_ns, "building another format must not rebuild this one"
        cache_usd = exp_dir / "cam_first_shot_cache.usd"
        assert cache_usd.read_bytes()[:8] == b"PXR-USDC"

        st_skel = Usd.Stage.Open(str(skel_usd))
        assert abs(UsdGeom.GetStageMetersPerUnit(st_skel) - 1.0) < 1e-6
        assert any(p.IsA(UsdSkel.Root) for p in st_skel.Traverse())
        st_cache = Usd.Stage.Open(str(cache_usd))
        assert abs(UsdGeom.GetStageMetersPerUnit(st_cache) - 1.0) < 1e-6
        assert not any(p.IsA(UsdSkel.Root) for p in st_cache.Traverse())
        mesh_prims = [p for p in st_cache.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(mesh_prims) == 1
        pts_1001 = np.asarray(UsdGeom.Mesh(mesh_prims[0]).GetPointsAttr().Get(1001.0), dtype=np.float64)
        pts_1003 = np.asarray(UsdGeom.Mesh(mesh_prims[0]).GetPointsAttr().Get(1003.0), dtype=np.float64)
        assert pts_1001.shape == (3, 3) and not np.allclose(pts_1001, pts_1003)
        del st_skel, st_cache

        # another unit is another file content: it is built anew, in that unit
        assert client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm").status_code == 200
        st_cm = Usd.Stage.Open(str(skel_usd))
        assert abs(UsdGeom.GetStageMetersPerUnit(st_cm) - 0.01) < 1e-6
        del st_cm

        # a new solve result makes every earlier file void: they are removed, the next download builds from the new data
        wm._clear_export_files(sw)
        assert built() == [] and not sw.read_meta().get("exports") and not sw.read_meta().get("export_cache")


def test_export_choice_of_people_and_no_root_filter() -> None:
    """Choosing which solved people go into the export:
      - nothing chosen = everybody (also people solved later), file name without a tag;
      - a partial choice exports exactly those people and says so in the file name (`_p1p3`), in every format;
      - choosing everybody again is stored as "everybody"; unknown ids are ignored; an export of nobody is refused.
    And the Root filter is gone: what is exported is exactly what was solved, a leftover `root_filter` setting of an
    older version changes nothing and cannot be set any more."""
    from fastapi.testclient import TestClient
    from pxr import Usd, UsdGeom, UsdSkel
    from shotspace.io.scene_scan_io import load_scene_scan
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        sid = client.post("/api/shots", json={"name": "choice"}).json()["id"]
        frames = list(range(1001, 1011))
        _upload_test_plate_and_camera(client, root, sid, frames)
        wm = app.state.wm
        sw = wm.get_shot(sid)
        cam_dir = sw.root / "solve_raw" / "cam_space"
        recs = [_write_test_person(cam_dir, pid, frames, x_cm=100.0 * pid) for pid in (1, 2, 3)]
        # person 2 jitters hard from frame to frame: exactly the motion the removed filter used to change
        with np.load(cam_dir / "person_02.npz", allow_pickle=False) as z:
            pack = {k: z[k] for k in z.files}
        pack["joint_world"] = pack["joint_world"].copy()
        pack["joint_world"][:, :, 2, 3] += (8.0 * (-1.0) ** np.arange(len(frames)))[:, None].astype(np.float32)
        np.savez_compressed(cam_dir / "person_02.npz", **pack)
        sw.update_meta(solved_people=recs, active_person_ids=[1, 2, 3], solved_at="2026-11-23T10:00:00+0800")
        exp_dir = sw.root / "exports"

        def skeleton_people(path: Path) -> list[int]:
            return sorted(int(ch.person_id) for ch in load_skinned_characters(path))

        # default: everybody, plain file name
        assert client.get(f"/api/shots/{sid}").json().get("export_person_ids") == []
        r_all = client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm")
        assert r_all.status_code == 200 and 'filename="choice_skel.usd"' in r_all.headers["content-disposition"]
        assert skeleton_people(exp_dir / "choice_skel.usd") == [1, 2, 3]

        # what is exported is exactly what was solved, placed by the exact camera: person 2's jitter included
        ch2 = [c for c in load_skinned_characters(exp_dir / "choice_skel.usd") if int(c.person_id) == 2][0]
        cam_now = wm._load_camera_from_entry(sw.read_meta().get("camera"), sw.read_meta().get("plate")).resample_to(frames)
        expect = np.einsum("fij,fkjl->fkil", cam_now.cam_to_world, pack["joint_world"].astype(np.float64))
        assert np.allclose(ch2.anim_world[:, :, :3, 3], expect[:, :, :3, 3], atol=2e-3), "export differs from the solve"
        assert float(np.abs(np.diff(ch2.anim_world[:, 0, 2, 3])).mean()) > 15.0, "the solved jitter was filtered away"

        # a leftover setting of an older version is ignored, and the setting cannot be made any more
        sw.update_meta(root_filter={"default": 0.9, "per_person": {"2": 1.0}})
        client.patch(f"/api/shots/{sid}", json={"root_filter": {"default": 1.0, "per_person": {"2": 1.0}}})
        rec_again = wm.export_deliverables(sid, formats=["usd_skel"], unit="cm", prefix="again")[0]
        ch2_again = [c for c in load_skinned_characters(exp_dir / rec_again["filename"]) if int(c.person_id) == 2][0]
        assert np.allclose(ch2_again.anim_world, ch2.anim_world, atol=1e-5)

        # a partial choice
        r_patch = client.patch(f"/api/shots/{sid}", json={"export_person_ids": [3, 1, 99]})
        assert r_patch.status_code == 200 and r_patch.json()["export_person_ids"] == [1, 3], r_patch.text[:200]
        for fmt, fname in (
            ("usd_skel", "choice_p1p3_skel.usd"), ("usd_cache", "choice_p1p3_cache.usd"),
            ("fbx", "choice_p1p3.fbx"), ("abc", "choice_p1p3.abc"),
        ):
            r_dl = client.get(f"/api/shots/{sid}/download/{fmt}?unit=cm")
            assert r_dl.status_code == 200, r_dl.text[:200]
            assert f'filename="{fname}"' in r_dl.headers["content-disposition"], r_dl.headers["content-disposition"]
            assert (exp_dir / fname).is_file()
        assert skeleton_people(exp_dir / "choice_p1p3_skel.usd") == [1, 3]
        st = Usd.Stage.Open(str(exp_dir / "choice_p1p3_skel.usd"))
        assert len([p for p in st.Traverse() if p.IsA(UsdSkel.Root)]) == 2
        del st
        # the other three formats hold the same two people (read back in the way each format allows)
        fbx_x = sorted(round(float(c.anim_world[0, 0, 0, 3]), 1) for c in load_skinned_characters(exp_dir / "choice_p1p3.fbx"))
        usd_x = sorted(round(float(c.anim_world[0, 0, 0, 3]), 1) for c in load_skinned_characters(exp_dir / "choice_p1p3_skel.usd"))
        assert len(fbx_x) == 2 and np.allclose(fbx_x, usd_x, atol=0.2), (fbx_x, usd_x)
        st_c = Usd.Stage.Open(str(exp_dir / "choice_p1p3_cache.usd"))
        assert sorted(str(pr.GetPath()) for pr in st_c.Traverse() if pr.IsA(UsdGeom.Mesh)) == ["/ShotSpace/person_01/body", "/ShotSpace/person_03/body"]
        del st_c
        # every delivery has ONE top-level group, named after the product: the people and the camera hang under it
        for usd_name in ("choice_p1p3_skel.usd", "choice_p1p3_cache.usd"):
            st_t = Usd.Stage.Open(str(exp_dir / usd_name))
            assert str(st_t.GetDefaultPrim().GetPath()) == "/ShotSpace", usd_name
            assert [str(pr.GetPath()) for pr in st_t.GetPseudoRoot().GetChildren()] == ["/ShotSpace"], usd_name
            assert sorted(pr.GetName() for pr in st_t.GetDefaultPrim().GetChildren()) == ["person_01", "person_03", "trackCam"], usd_name
            del st_t
        for other_name in ("choice_p1p3.abc", "choice_p1p3.fbx"):
            info_t = inspect_3d_file(exp_dir / other_name)
            assert [(h["path"] if isinstance(h, dict) else h) for h in info_t["hierarchy"]] == ["/ShotSpace"], (other_name, info_t["hierarchy"])
            assert [c["path"] for c in info_t["cameras"]] == ["/ShotSpace/trackCam"], (other_name, info_t["cameras"])
        assert sorted(m["path"] for m in inspect_3d_file(exp_dir / "choice_p1p3.abc")["models"]) == ["/ShotSpace/person_01", "/ShotSpace/person_03"]
        # ...and a file written without a top group (cameras and scans the tests build) is laid out as before
        plain = export_scene(root / "plain_layout.usda", [], [cam_now], frames, fps=24.0, unit="cm")
        st_p = Usd.Stage.Open(str(plain))
        assert str(st_p.GetDefaultPrim().GetPath()) == "/shot"
        del st_p
        assert [c["path"] for c in inspect_3d_file(export_scene(root / "plain_layout.abc", [], [cam_now], frames, fps=24.0, unit="cm"))["cameras"]] == ["/trackCam"]
        abc_geo = load_scene_scan(exp_dir / "choice_p1p3.abc")
        assert abc_geo.faces.shape[0] == 2 and abc_geo.vertices_cm.shape[0] == 6, "the Alembic must hold two people"
        # the full file built before is a different file and still holds everybody
        assert skeleton_people(exp_dir / "choice_skel.usd") == [1, 2, 3]

        # one person
        assert client.patch(f"/api/shots/{sid}", json={"export_person_ids": [2]}).json()["export_person_ids"] == [2]
        r_one = client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm")
        assert 'filename="choice_p2_skel.usd"' in r_one.headers["content-disposition"]
        assert skeleton_people(exp_dir / "choice_p2_skel.usd") == [2]

        # everybody chosen by hand = "everybody"
        assert client.patch(f"/api/shots/{sid}", json={"export_person_ids": [1, 2, 3]}).json()["export_person_ids"] == []
        r_full = client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm")
        assert 'filename="choice_skel.usd"' in r_full.headers["content-disposition"]

        # removing a solved person keeps the choice consistent
        assert client.patch(f"/api/shots/{sid}", json={"export_person_ids": [1, 2]}).json()["export_person_ids"] == [1, 2]
        r_rm = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [2]})
        assert r_rm.status_code == 200, r_rm.text[:200]
        m_rm = r_rm.json()
        assert [p["id"] for p in m_rm["solved_people"]] == [1, 3] and m_rm["export_person_ids"] == [1]
        assert not (cam_dir / "person_02.npz").exists()
        assert not any(exp_dir.glob("*")), "files built from the old set of people must be gone"
        r_after = client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm")
        assert 'filename="choice_p1_skel.usd"' in r_after.headers["content-disposition"]
        assert skeleton_people(exp_dir / "choice_p1_skel.usd") == [1]

        # clearing everybody: nothing left to export
        m_clear = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": None}).json()
        assert m_clear["solved_people"] == [] and m_clear["export_person_ids"] == []
        assert not any(cam_dir.glob("person_*.npz"))
        assert client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm").status_code == 400


def test_exports_write_no_keys_while_a_person_stands_still() -> None:
    """A person who is not in the picture is exported hidden and standing still at the nearest solved pose.  Those
    stretches carry no keys of their own: USD skeleton, USD point cache and FBX hold samples only where something
    changes, and visibility is keyed only where it flips.  Read back frame by frame, every format still gives the
    same pose on every frame as before (the solved pose where the person is solved, the frozen pose elsewhere).
    Alembic keeps a sample on every frame: its readers expect a regularly sampled point cache."""
    from fastapi.testclient import TestClient
    from pxr import Usd, UsdGeom, UsdSkel
    from shotspace.io.scene_arrays import held_changes, needed_samples
    from shotspace.io.scene_scan_io import load_scene_scan
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    # the rule itself: kept samples give back every value, stretches of equal samples keep only their two ends
    vals = np.array([0, 0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5], dtype=np.float64)
    kept = needed_samples(vals)
    assert kept.tolist() == [True, False, True, True, True, True, False, False, True, True, True, True, False, True]
    xs = np.arange(len(vals))
    assert np.array_equal(np.interp(xs, xs[kept], vals[kept]), vals)
    assert needed_samples(np.zeros((1, 3))).tolist() == [True] and needed_samples(np.zeros((2, 3))).tolist() == [True, True]
    assert held_changes([1, 1, 0, 0, 0, 1, 1, 0]).tolist() == [True, False, True, False, False, True, False, True]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        sid = client.post("/api/shots", json={"name": "still"}).json()["id"]
        frames = list(range(1001, 1041))  # 40 frames
        _upload_test_plate_and_camera(client, root, sid, frames)
        wm = app.state.wm
        sw = wm.get_shot(sid)
        cam_dir = sw.root / "solve_raw" / "cam_space"

        # person 1: solved on 1009-1018 and 1026-1032, hidden before, between and after
        rec = _write_test_person(cam_dir, 1, frames, x_cm=0.0)
        with np.load(cam_dir / "person_01.npz", allow_pickle=False) as z:
            pack = {k: z[k] for k in z.files}
        fr = np.asarray(frames)
        vis = ((fr >= 1009) & (fr <= 1018)) | ((fr >= 1026) & (fr <= 1032))
        seen = fr[vis]
        jw = pack["joint_world"].astype(np.float64).copy()
        wiggle = np.sin(np.arange(len(frames)) * 0.9)
        jw[:, :, 0, 3] += (25.0 * wiggle)[:, None]       # moves on every frame while solved
        jw[:, 1, 1, 3] += 6.0 * np.cos(np.arange(len(frames)) * 0.7)
        pack["joint_world"] = jw.astype(np.float32)
        pack["visible"] = vis.astype(np.int32)
        pack["anchor_frames"] = np.array([int(seen[np.argmin(np.abs(seen - f))]) for f in fr], dtype=np.int64)
        np.savez_compressed(cam_dir / "person_01.npz", **pack)
        # person 2: solved on every frame (nothing may be thinned there)
        rec2 = _write_test_person(cam_dir, 2, frames, x_cm=150.0)
        sw.update_meta(solved_people=[rec, rec2], active_person_ids=[1, 2], solved_at="2026-11-23T19:00:00+0800")

        # what every frame must show: the solved pose placed by the exact camera, or the frozen pose of the anchor
        cam_now = wm._load_camera_from_entry(sw.read_meta().get("camera"), sw.read_meta().get("plate")).resample_to(frames)
        world = np.einsum("fij,fkjl->fkil", cam_now.cam_to_world, pack["joint_world"].astype(np.float64))
        anchor_idx = np.array([frames.index(int(a)) for a in pack["anchor_frames"]])
        expect = np.where(vis[:, None, None, None], world, world[anchor_idx])
        still = ~needed_samples(expect)          # frames inside a stretch where nothing changes
        assert 15 <= int(still.sum()) <= 22, int(still.sum())
        n_keys = int((~still).sum())
        n_vis_keys = int(held_changes(vis).sum())

        exp_dir = sw.root / "exports"
        for fmt in ("usd_skel", "usd_cache", "fbx", "abc"):
            assert client.get(f"/api/shots/{sid}/download/{fmt}?unit=cm").status_code == 200, fmt

        # ---- USD skeleton
        st = Usd.Stage.Open(str(exp_dir / "still_skel.usd"))
        anim1 = UsdSkel.Animation(st.GetPrimAtPath("/ShotSpace/person_01/anim"))
        key_times = [int(v) for v in anim1.GetTranslationsAttr().GetTimeSamples()]
        assert len(key_times) == n_keys < len(frames), (len(key_times), n_keys)
        assert not any(frames[i] in key_times for i in np.flatnonzero(still)), "a key inside a stretch where the body stands still"
        assert len(anim1.GetRotationsAttr().GetTimeSamples()) == n_keys and len(anim1.GetScalesAttr().GetTimeSamples()) == n_keys
        vis_attr = UsdGeom.Imageable(st.GetPrimAtPath("/ShotSpace/person_01")).GetVisibilityAttr()
        assert len(vis_attr.GetTimeSamples()) == n_vis_keys == 5, vis_attr.GetTimeSamples()
        anim2 = UsdSkel.Animation(st.GetPrimAtPath("/ShotSpace/person_02/anim"))
        assert len(anim2.GetTranslationsAttr().GetTimeSamples()) == len(frames), "a person solved on every frame keeps every key"
        assert int(st.GetStartTimeCode()) == frames[0] and int(st.GetEndTimeCode()) == frames[-1]
        del st
        back = {int(c.person_id): c for c in load_skinned_characters(exp_dir / "still_skel.usd")}
        assert list(back[1].frames) == frames
        assert np.allclose(back[1].anim_world[:, :, :3, 3], expect[:, :, :3, 3], atol=2e-3), "USD skeleton: a frame changed"
        assert np.array_equal(np.asarray(back[1].visible_mask, dtype=bool), vis)

        # ---- USD point cache
        st = Usd.Stage.Open(str(exp_dir / "still_cache.usd"))
        pts_attr = UsdGeom.Mesh(st.GetPrimAtPath("/ShotSpace/person_01/body")).GetPointsAttr()
        assert len(pts_attr.GetTimeSamples()) == n_keys, len(pts_attr.GetTimeSamples())
        assert len(UsdGeom.Mesh(st.GetPrimAtPath("/ShotSpace/person_02/body")).GetPointsAttr().GetTimeSamples()) == len(frames)
        pts = np.stack([np.asarray(pts_attr.Get(Usd.TimeCode(f)), dtype=np.float64) for f in frames])
        assert np.allclose(pts, pts[anchor_idx][np.arange(len(frames))] * 0 + np.where(vis[:, None, None], pts, pts[anchor_idx]), atol=1e-5)
        assert np.allclose(pts[~vis], pts[anchor_idx][~vis], atol=1e-5), "USD point cache: a hidden frame is not the frozen pose"
        moved = np.linalg.norm(pts[1:] - pts[:-1], axis=-1).max(axis=1)
        assert np.all(moved[np.flatnonzero(vis[1:] & vis[:-1])] > 1.0), "USD point cache: solved frames must keep moving"
        img = UsdGeom.Imageable(st.GetPrimAtPath("/ShotSpace/person_01"))
        assert [img.ComputeVisibility(Usd.TimeCode(f)) != UsdGeom.Tokens.invisible for f in frames] == vis.tolist()
        del st

        # ---- FBX
        back = load_skinned_characters(exp_dir / "still.fbx")
        fb = sorted(back, key=lambda c: float(c.anim_world[0, 0, 0, 3]))[0]  # person 1 stands at the smaller x
        assert list(fb.frames) == frames, "FBX: read back from the first key to the last"
        assert np.allclose(fb.anim_world[:, :, :3, 3], expect[:, :, :3, 3], atol=2e-3), "FBX: a frame changed"
        assert np.array_equal(np.asarray(fb.visible_mask, dtype=bool), vis)

        # ---- Alembic keeps every frame
        geo = load_scene_scan(exp_dir / "still.abc")
        assert geo.faces.shape[0] == 2


class _StandInPeopleSolver:
    """Stands in for the GPU workers in the bookkeeping test: it writes the same files under the same names the
    real workers write, and keeps a log of what it was asked to do.  No network runs, so the test is about one
    thing only: WHO gets computed, kept, re-solved and removed."""

    def __init__(self) -> None:
        import threading

        self.log: list[tuple[str, list[int]]] = []
        self.with_scan: list[bool] = []            # per solve / resolve call: was it given the scan?
        self.gen = 0
        self.fail_after_writing: set[int] = set()   # these people are written to disk, then the job fails
        self.block_after_writing: set[int] = set()  # these people are written to disk, then the job hangs until killed
        self.skip: set[int] = set()                 # these people have too few usable frames: estimated, not solved
        self.resolve_fails: set[int] = set()        # a whole-shot solve WITH the scan from cached outputs of these people fails
        self.resolve_blocks: set[int] = set()       # ...or hangs until the job is cancelled
        self.blocked = threading.Event()
        self._kill = threading.Event()
        self.seen_lines: list[str] = []            # the task's status line right after each progress report
        self.read_line = lambda: ""                # set by the test: reads the task's status line
        self.rich = False                          # write everything a real result and network file holds
        self.cut: dict[int, int] = {}              # track -> frame from which it follows somebody else (cut there)
        self.id_starts: list[int | None] = []      # the first free person number each call was given

    def calls(self, kind: str) -> list[list[int]]:
        return [ids for k, ids in self.log if k == kind]

    def shutdown(self) -> None:  # what the queue calls to kill the worker of a cancelled job
        self._kill.set()

    def offload(self) -> None:
        pass

    def detect_people(self, plate, output_boxes, solver_frame_files=None, grade=None, threshold=0.8, gpu=None, on_event=None) -> dict:
        import json
        import uuid

        self.log.append(("detect", []))
        run_id = uuid.uuid4().hex
        frames = [int(f) for f in plate.frames]
        people, packed = [], {}
        for pid in (1, 2, 3):
            x0 = 20.0 + 120.0 * (pid - 1)
            boxes = {str(f): [x0, 60.0, x0 + 80.0, 300.0] for f in frames}
            people.append(
                {"id": pid, "prominence": 1000.0 - pid, "frame_count": len(frames), "boxes": boxes,
                 "raw_boxes": dict(boxes), "scores": {str(f): 0.9 for f in frames}}
            )
            for f in frames:
                packed[f"p{pid}_f{f}_bits"] = np.packbits(np.ones(240 * 80, dtype=np.uint8))
                packed[f"p{pid}_f{f}_shape"] = np.array([240, 80], dtype=np.int16)
                packed[f"p{pid}_f{f}_rect"] = np.asarray(boxes[str(f)], dtype=np.float32)
        packed["__run_id__"] = np.array(run_id)
        output_boxes.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_boxes.with_name("person_masks_sam3.npz"), **packed)
        output_boxes.write_text(
            json.dumps({"people": people, "run_id": run_id, "engine": "sam31_native_people_v1"}), encoding="utf-8"
        )
        return {"people_count": len(people), "run_id": run_id}

    def segment_people_masks(self, plate, selected_people, output_masks_npz, solver_frame_files=None, gpu=None, on_event=None) -> dict:
        self.log.append(("outline", sorted(int(p["id"]) for p in selected_people)))
        packed = {}
        for person in selected_people:
            for fk, b in person["boxes"].items():
                w, h = max(1, int(round(b[2] - b[0]))), max(1, int(round(b[3] - b[1])))
                stem = f"p{int(person['id'])}_f{int(fk)}"
                packed[f"{stem}_bits"] = np.packbits(np.ones(h * w, dtype=np.uint8))
                packed[f"{stem}_shape"] = np.array([h, w], dtype=np.int16)
                packed[f"{stem}_rect"] = np.asarray(b, dtype=np.float32)
        np.savez_compressed(output_masks_npz, **packed)
        return {}

    @staticmethod
    def rich_arrays(pid: int, frames: list[int]) -> tuple[dict, dict]:
        """(network file, result file) contents with the keys and shapes of the real ones, values that can be
        recomputed by a test: body point k of frame t sits at camera-space (10 pid + k, 100 + t, -400 - k) cm."""
        n = len(frames)
        t_idx = np.arange(n, dtype=np.float64)
        k_idx = np.arange(70, dtype=np.float64)
        kp_cam = np.stack(
            [np.broadcast_to(10.0 * pid + k_idx, (n, 70)), np.broadcast_to(100.0 + t_idx[:, None], (n, 70)), np.broadcast_to(-400.0 - k_idx, (n, 70))],
            axis=-1,
        ).astype(np.float32)
        visible = np.ones(n, dtype=np.int32)
        anchors = np.asarray(frames, dtype=np.int64).copy()
        if n >= 4:  # one hidden frame: it holds the body of the frame before it
            visible[2], anchors[2], kp_cam[2] = 0, frames[1], kp_cam[1]
        net = dict(
            shot_frames=np.asarray(frames, dtype=np.int64), is_manual=np.int32(0), image_size=np.array([640, 360], dtype=np.int32),
            det_scores=np.linspace(0.5, 0.9, n).astype(np.float32),
            pred_cam_t=np.stack([0.1 * pid + 0.01 * t_idx, 0.5 + 0.0 * t_idx, 4.0 + 0.1 * t_idx], axis=1).astype(np.float32),
            global_rot=np.stack([0.01 * t_idx, 0.02 * t_idx, 0.03 * t_idx], axis=1).astype(np.float32),
            shape_params=(0.001 * np.arange(45)[None] + 0.01 * t_idx[:, None]).astype(np.float32),
            scale_params=(0.002 * np.arange(28)[None] + 0.0 * t_idx[:, None]).astype(np.float32),
            body_pose_params=(0.001 * np.arange(133)[None] - 0.01 * t_idx[:, None]).astype(np.float32),
            hand_pose_params=np.zeros((n, 108), dtype=np.float32),
            pred_keypoints_2d=np.stack([100.0 + k_idx[None] + t_idx[:, None], 50.0 + 2.0 * k_idx[None] + 0.0 * t_idx[:, None]], axis=-1).astype(np.float32),
            bbox=np.stack([20.0 + t_idx, 60.0 + 0.0 * t_idx, 100.0 + t_idx, 300.0 + 0.0 * t_idx], axis=1).astype(np.float32),
            mask_area=(5000.0 + 10.0 * t_idx).astype(np.float32),
            mask_box=np.stack([22.0 + t_idx, 62.0 + 0.0 * t_idx, 98.0 + t_idx, 298.0 + 0.0 * t_idx], axis=1).astype(np.float32),
            has_mask=np.ones(n, dtype=np.uint8),
        )
        res = dict(
            visible=visible, anchor_frames=anchors, keypoints_cam_cm=kp_cam,
            solved_frames=np.asarray([f for f, v in zip(frames, visible) if v], dtype=np.int64),
            reproj_px=(1.0 + 0.5 * t_idx).astype(np.float32), foot_contact=np.stack([t_idx % 2, (t_idx + 1) % 2], axis=1).astype(np.uint8),
            bbox=net["bbox"], shape_params=np.asarray(net["shape_params"][0]), scale_params=np.asarray(net["scale_params"][0]),
            keypoints_2d=net["pred_keypoints_2d"], body_scale=np.float32(1.02), scale_source=np.array("network"),
        )
        return net, res

    def _write(self, raw_net_dir: Path, cam_space_dir: Path, pid: int, frames: list[int], smoothing: float, raw: bool) -> dict:
        self.gen += 1
        net_extra, res_extra = self.rich_arrays(pid, frames) if self.rich else ({}, {})
        if raw:
            raw_net_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                raw_net_dir / f"person_{pid:02d}.npz",
                frames=np.asarray(frames, dtype=np.int64),
                kp_in_mask=np.ones((len(frames), 70), dtype=np.uint8),
                gen=np.int64(self.gen),
                **net_extra,
            )
        return _write_test_person(cam_space_dir, pid, frames, x_cm=30.0 * (pid % 50), smoothing=smoothing, gen=self.gen, extra=res_extra)

    def solve_people(
        self, plate, selected_people, intrinsics_camera=None, raw_net_dir=Path("."), cam_space_dir=Path("."),
        solver_frame_files=None, hand_refine=True, use_mask=True, person_masks_npz=None, lock_shape=True,
        smoothing=0.5, fps=24.0, scan_npz=None, gpu=None, on_event=None, new_id_start=None,
    ) -> dict:
        ids = [int(p["id"]) for p in selected_people]
        self.log.append(("solve", sorted(ids)))
        self.with_scan.append(scan_npz is not None)
        self.id_starts.append(new_id_start)
        self.mask_ids = {int(p["id"]): int(p.get("mask_id", p["id"])) for p in selected_people}
        boxes_of = {int(p["id"]): sorted(int(k) for k in (p.get("boxes") or {})) for p in selected_people}
        n_frames = len(plate.frames)
        if on_event is not None:  # what the real worker sends while it estimates: frames done of this batch
            on_event({"event": "stage", "stage": "sam3d_infer", "message": "worker wording"})
            on_event({"event": "progress", "stage": "sam3d_infer", "done": n_frames // 2, "total": n_frames})
            self.seen_lines.append(self.read_line())
        out = []
        for n_done, pid in enumerate(ids, start=1):
            if on_event is not None:  # ...and while it solves the people of the batch one after the other
                on_event({"event": "progress", "stage": "mhr_postprocess", "done": n_done, "total": len(ids)})
                self.seen_lines.append(self.read_line())
            if pid in self.skip:
                # what the real worker does with a person it cannot solve: network outputs on disk, a warning,
                # no solved result and no entry in what it returns
                self._write(Path(raw_net_dir), Path(cam_space_dir), pid, [int(f) for f in plate.frames], float(smoothing), raw=True)
                (Path(cam_space_dir) / f"person_{pid:02d}.npz").unlink()
                if on_event is not None:
                    on_event({"event": "warning", "stage": "mhr_postprocess", "message": f"人物 {pid} 可解算帧数不足，已跳过"})
                continue
            # like the real worker: a person is estimated on the frames they have boxes in
            own_frames = [f for f in (int(f) for f in plate.frames) if f in set(boxes_of.get(pid) or [int(f) for f in plate.frames])]
            rec = self._write(Path(raw_net_dir), Path(cam_space_dir), pid, own_frames, float(smoothing), raw=True)
            if pid in self.fail_after_writing:
                raise RuntimeError("stand-in worker failure")
            if pid in self.block_after_writing:
                self.blocked.set()
                self._kill.wait(timeout=30.0)
                raise RuntimeError("stand-in worker was killed")
            out.append(rec)
        return self._with_cuts(Path(raw_net_dir), Path(cam_space_dir), out, float(smoothing), new_id_start)

    def _with_cuts(self, raw_net_dir: Path, cam_space_dir: Path, people: list[dict], smoothing: float, new_id_start) -> dict:
        """What the real worker does with a track that jumps to another person at frame `self.cut[track]`: the part
        before stays the track, the part from that frame on becomes a person with a new number; the cached network
        outputs of both parts are written under temporary names for the task manager to put in place."""
        splits, next_id = [], new_id_start
        for rec in list(people):
            pid, at = int(rec["id"]), self.cut.get(int(rec["id"]))
            frames = [int(f) for f in rec["frames"]]
            if at is None or new_id_start is None or not (frames[0] < at <= frames[-1]):
                continue
            new_id, next_id = int(next_id), int(next_id) + 1
            first, second = [f for f in frames if f < at], [f for f in frames if f >= at]
            for part_id, part_frames in ((pid, first), (new_id, second)):
                np.savez(
                    raw_net_dir / f".split_{pid:02d}_{part_id:02d}.npz",
                    frames=np.asarray(part_frames, dtype=np.int64), kp_in_mask=np.ones((len(part_frames), 70), dtype=np.uint8), gen=np.int64(self.gen),
                )
            people[people.index(rec)] = _write_test_person(cam_space_dir, pid, first, x_cm=30.0 * (pid % 50), smoothing=smoothing, gen=self.gen)
            people.append({**_write_test_person(cam_space_dir, new_id, second, x_cm=30.0 * (new_id % 50), smoothing=smoothing, gen=self.gen), "split_from": pid})
            splits.append({"from": pid, "dropped": 0, "parts": [
                {"id": pid, "first": first[0], "last": first[-1], "frames": len(first)},
                {"id": new_id, "first": second[0], "last": second[-1], "frames": len(second)},
            ]})
        return {"people": people, "splits": splits}

    def postprocess_cached(
        self, raw_net_dir, cam_space_dir, shot_frames=None, lock_shape=True, smoothing=0.5, fps=24.0,
        scan_npz=None, cam_to_world=None, person_ids=None, gpu=None, on_event=None, new_id_start=None,
    ) -> dict:
        on_disk = sorted(int(f.stem.split("_")[-1]) for f in Path(raw_net_dir).glob("person_*.npz"))
        ids = on_disk if person_ids is None else [pid for pid in on_disk if pid in {int(x) for x in person_ids}]
        self.log.append(("resolve", ids))
        self.with_scan.append(scan_npz is not None)
        self.id_starts.append(new_id_start)
        out = []
        for n_done, pid in enumerate(ids, start=1):
            if on_event is not None:
                on_event({"event": "progress", "stage": "mhr_postprocess", "done": n_done, "total": len(ids)})
            if pid in self.resolve_blocks and scan_npz is not None:
                self.blocked.set()
                self._kill.wait(timeout=30.0)
                raise RuntimeError("stand-in worker was killed")
            if pid in self.resolve_fails and scan_npz is not None:
                raise RuntimeError("stand-in whole-shot solve failure")
            # like the real worker: a person is solved on the frames their cached network outputs hold
            with np.load(Path(raw_net_dir) / f"person_{pid:02d}.npz", allow_pickle=False) as rz:
                own_frames = [int(f) for f in rz["frames"]]
            out.append(self._write(Path(raw_net_dir), Path(cam_space_dir), pid, own_frames, float(smoothing), raw=False))
            if pid in self.fail_after_writing:
                raise RuntimeError("stand-in worker failure")
        return self._with_cuts(Path(raw_net_dir), Path(cam_space_dir), out, float(smoothing), new_id_start)


def test_accumulating_people_solve() -> None:
    """Solving people ADDS to what is solved (it does not replace it), and only what is new or changed is computed:
      - the first solve looks for people once and solves the selected person; the next solve adds another person
        without touching the first; solving again with nothing new computes nothing;
      - a new smoothing (or camera move) re-solves EVERY solved person from the cached network outputs, no network;
      - people can be removed one by one or all at once; the export choice follows;
      - a manually boxed person is added the same way;
      - people are looked for once per plate: a changed grade does not make a solve do it again; "重新检测" does,
        and it clears the people solved from the old numbering (manual ones stay);
      - a failed or cancelled job loses nobody who was solved before and leaves no half-added person behind;
      - a task solved by v0.2.1 (one cache signature for the whole selection) is carried over without recomputing.
    """
    import json
    import time
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.solver.trajectory import SOLVER_REVISION
    from shotspace.web.app import create_app
    from shotspace.web.workspace import _json_hash

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        slots = list(wm.scheduler._slots.values())
        if not any(s.enabled for s in slots):
            print("  (skipped: this machine has no enabled GPU slot, the queue cannot run a job)")
            return
        fake = _StandInPeopleSolver()
        for s in slots:
            s.solver = fake

        def wait_idle(shot_id: str, timeout: float = 60.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if wm.scheduler.job_for_shot(shot_id) is None and not wm.get_shot(shot_id).read_meta().get("active_job_id"):
                    return
                time.sleep(0.03)
            raise AssertionError("the queued job did not finish")

        def new_shot(name: str, frames: list[int]):
            shot_id = client.post("/api/shots", json={"name": name}).json()["id"]
            _upload_test_plate_and_camera(client, root, shot_id, frames)
            return shot_id, wm.get_shot(shot_id)

        frames = list(range(1001, 1007))
        sid, sw = new_shot("accum", frames)
        raw_dir, cam_dir = sw.root / "solve_raw" / "raw_net", sw.root / "solve_raw" / "cam_space"
        meta = sw.read_meta
        solved = lambda: [int(sp["id"]) for sp in (meta().get("solved_people") or [])]  # noqa: E731
        on_disk = lambda d: sorted(int(f.stem.split("_")[-1]) for f in d.glob("person_*.npz"))  # noqa: E731

        def in_view_of(shot_id: str) -> list[int]:
            """The people the viewport is given to draw."""
            r_view = client.get(f"/api/shots/{shot_id}/view/manifest")
            assert r_view.status_code == 200, r_view.text[:200]
            return sorted(int(c["person_id"]) for c in r_view.json()["characters"])

        def in_view() -> list[int]:
            return in_view_of(sid)

        def stamp(pid: int, key: str = "gen") -> float:
            with np.load(cam_dir / f"person_{pid:02d}.npz", allow_pickle=False) as z:
                return float(z[key])

        def solve(ids: str | None = None, **patch) -> None:
            body = dict(patch)
            if ids is not None:
                body["selection"] = {"mode": "ids", "ids": ids}
            fake.log.clear()
            wm.enqueue_solve(sid, body or None, wait=True)

        # ---- 1. first solve: people are looked for (once), the selected person is solved
        solve("1")
        assert fake.log == [("detect", []), ("solve", [1])], fake.log
        assert solved() == [1] and on_disk(cam_dir) == [1] and on_disk(raw_dir) == [1]
        run_1 = meta()["cache_signatures"]["detect_run"]
        assert run_1 and run_1 == json.loads((sw.root / "detect" / "boxes.json").read_text(encoding="utf-8"))["run_id"]
        gen_1 = stamp(1)

        # ---- 2. another person is ADDED; the first one is neither recomputed nor rewritten
        solve("2")
        assert fake.log == [("solve", [2])], fake.log
        assert solved() == [1, 2] and stamp(1) == gen_1
        assert "共 2 人" in meta()["stage_message"] and "新增 1 人" in meta()["stage_message"], meta()["stage_message"]
        assert in_view() == [1, 2]

        # ---- 3. nothing new: nothing is computed (also with nobody selected)
        solve("2")
        assert fake.log == [] and solved() == [1, 2] and "已是最新" in meta()["stage_message"]
        solve("")
        assert fake.log == [] and solved() == [1, 2]

        # ---- 4. new smoothing: everybody solved is solved again from the cached network outputs
        solve(None, solve_params={"smoothing": 0.8})
        assert fake.log == [("resolve", [1, 2])], fake.log
        assert abs(stamp(1, "smoothing_strength") - 0.8) < 1e-6 and abs(stamp(2, "smoothing_strength") - 0.8) < 1e-6
        assert "重新解算 2 人" in meta()["stage_message"]

        # ---- 5. the export choice survives adding a person, the new person is in by default
        assert client.patch(f"/api/shots/{sid}", json={"export_person_ids": [1]}).json()["export_person_ids"] == [1]
        solve("3")
        assert fake.log == [("solve", [3])] and solved() == [1, 2, 3]
        assert meta()["export_person_ids"] == [1, 3]

        # ---- 6. remove one person: only that person goes, with files and cache
        r_rm = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [2]})
        assert r_rm.status_code == 200, r_rm.text[:200]
        assert solved() == [1, 3] and on_disk(cam_dir) == [1, 3] and on_disk(raw_dir) == [1, 3]
        assert meta()["export_person_ids"] == [] and set(meta()["cache_signatures"]["people_net"]) == {"1", "3"}
        assert in_view() == [1, 3]

        # ---- 7. a camera move imported afterwards: an automatic job re-solves the solved people, from cache,
        #         without looking for people again
        fake.log.clear()
        _upload_test_camera(client, root, sid, frames, moved=25.0)
        wait_idle(sid)
        assert fake.log == [("resolve", [1, 3])], fake.log
        assert solved() == [1, 3]

        # ---- 8. a manually boxed person: added like anybody else, only that person is outlined and estimated
        manual_track = {
            "id": 101, "name": "手动 #101", "enabled": True,
            "keyframes": [
                {"frame": 1001, "status": "active", "box": [480, 40, 580, 330]},
                {"frame": 1006, "status": "active", "box": [500, 40, 600, 330]},
            ],
        }
        solve("101", manual_people=[manual_track])
        assert fake.log == [("outline", [101]), ("solve", [101])], fake.log
        assert solved() == [1, 3, 101]

        # ---- 9. a changed grade does NOT make a solve look for people again (the numbers must stay what the
        #         artist sees); the person added now is the only one computed
        solve("2", grade={"exposure": 0.7})
        assert fake.log == [("solve", [2])], fake.log
        assert solved() == [1, 2, 3, 101] and meta()["cache_signatures"]["detect_run"] == run_1

        # ---- 10. a failing job: who was solved before stays, the half-added person leaves no file behind
        client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [2]})
        gens = {pid: stamp(pid) for pid in (1, 3, 101)}
        fake.fail_after_writing = {2}
        try:
            solve("2")
            raise AssertionError("the job was expected to fail")
        except RuntimeError as exc:
            assert "stand-in worker failure" in str(exc)
        fake.fail_after_writing = set()
        wait_idle(sid)
        assert solved() == [1, 3, 101] and on_disk(cam_dir) == [1, 3, 101] and on_disk(raw_dir) == [1, 3, 101]
        assert {pid: stamp(pid) for pid in (1, 3, 101)} == gens and meta()["status"] == "error"

        # ---- 11. a cancelled job: the same guarantee
        fake.block_after_writing = {2}
        fake.blocked.clear()
        fake._kill.clear()
        fake.log.clear()
        wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": "2"}})
        assert fake.blocked.wait(timeout=30.0), "the stand-in job never started"
        assert on_disk(cam_dir) == [1, 2, 3, 101]          # the half-added person's file is on disk right now
        client.post(f"/api/shots/{sid}/cancel")
        wait_idle(sid)
        fake.block_after_writing = set()
        assert solved() == [1, 3, 101] and on_disk(cam_dir) == [1, 3, 101] and on_disk(raw_dir) == [1, 3, 101]
        assert {pid: stamp(pid) for pid in (1, 3, 101)} == gens
        assert meta()["status"] in ("ready", "idle") and meta()["stage_message"] == "任务已取消"

        # ---- 12. an interrupted RE-solve keeps the old results and can never mistake a rewritten file for a
        #          valid cache: the signatures of what is being redone are withdrawn before the work starts
        fake.block_after_writing = {1}
        fake.blocked.clear()
        fake._kill.clear()
        wm.enqueue_solve(sid, {"solve_params": {"hand_refine": False}, "selection": {"mode": "ids", "ids": ""}})
        assert fake.blocked.wait(timeout=30.0)
        client.post(f"/api/shots/{sid}/cancel")
        wait_idle(sid)
        fake.block_after_writing = set()
        assert solved() == [1, 3, 101], "an interrupted re-solve must not lose anybody"
        assert "1" not in meta()["cache_signatures"]["people_net"], "person 1's file was rewritten: its old signature must be gone"
        solve("", solve_params={"hand_refine": True})     # back to the old setting: person 1 must NOT pass as cached
        assert ("solve", [1, 3, 101]) in fake.log or sorted(sum(fake.calls("solve"), [])) == [1, 3, 101], fake.log
        assert solved() == [1, 3, 101]

        # ---- 13. "重新检测选人框": the numbers are new, people solved from the old numbers are cleared,
        #          the manually boxed person stays
        fake.log.clear()
        r_det = client.post(f"/api/shots/{sid}/detect", json={})
        assert r_det.status_code == 200, r_det.text[:200]
        wait_idle(sid)
        assert fake.log == [("detect", [])]
        assert solved() == [101] and on_disk(cam_dir) == [101] and on_disk(raw_dir) == [101]
        run_2 = meta()["cache_signatures"]["detect_run"]
        assert run_2 and run_2 != run_1 and "已清除" in meta()["stage_message"]
        solve("1")
        assert fake.log == [("solve", [1])] and solved() == [1, 101]

        # ---- 14. the tracker's two files must come from the same run, and from the run the task has recorded
        boxes_path = sw.root / "detect" / "boxes.json"
        masks_path = sw.root / "detect" / "person_masks_sam3.npz"
        det_now = json.loads(boxes_path.read_text(encoding="utf-8"))
        # (a) a complete newer run on disk that the task has not recorded (the job was cut off right after the
        #     tracker finished): it is taken over, which clears the tracked person solved from the old numbers
        with np.load(masks_path, allow_pickle=False) as z:
            packed = {k: z[k] for k in z.files}
        packed["__run_id__"] = np.array("f" * 32)
        np.savez_compressed(masks_path, **packed)
        boxes_path.write_text(json.dumps({**det_now, "run_id": "f" * 32}), encoding="utf-8")
        solve("")
        assert fake.log == [] and solved() == [101] and meta()["cache_signatures"]["detect_run"] == "f" * 32
        # (b) boxes and outlines of different runs: unusable.  They are discarded; with a manual box in the task a
        #     solve does not start looking for people by itself, the artist does that with the detect button.
        packed["__run_id__"] = np.array("e" * 32)
        np.savez_compressed(masks_path, **packed)
        solve("3")
        assert fake.log == [] and solved() == [101], (fake.log, solved())
        assert meta()["detection"] is None and not meta()["cache_signatures"]["detect"]
        assert not boxes_path.exists() and not masks_path.exists()
        fake.log.clear()
        assert client.post(f"/api/shots/{sid}/detect", json={}).status_code == 200
        wait_idle(sid)
        solve("3")
        assert fake.log == [("solve", [3])] and solved() == [3, 101]
        assert meta()["cache_signatures"]["detect_run"] not in ("", "e" * 32, "f" * 32)

        # ---- 15. clear everything
        m_clear = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": None}).json()
        assert m_clear["solved_people"] == [] and on_disk(cam_dir) == [] and on_disk(raw_dir) == []
        assert in_view() == []

        # ---- 16. a task solved by v0.2.1: one signature for the whole selection.  Adding a person to it must
        #          not send the earlier people through the network again.
        sid, sw = new_shot("legacy", frames)
        raw_dir, cam_dir = sw.root / "solve_raw" / "raw_net", sw.root / "solve_raw" / "cam_space"
        meta = sw.read_meta
        solve("1,2")
        assert solved() == [1, 2]

        def to_v021_form(smoothing: float) -> None:
            """Rewrite the task's bookkeeping into what v0.2.1 stored (its signature formulas, frozen here)."""
            m = meta()
            plate = wm._load_plate_info(sw)
            cam = wm._load_camera_from_entry(m.get("camera"), m.get("plate"))
            from shotspace.io.plate_grade import GradeParams

            g_obj = GradeParams.from_dict(m.get("grade"))
            det_sig = wm._detect_sig(plate, g_obj, float((m.get("detect_params") or {}).get("threshold", 0.8)))
            det = json.loads((sw.root / "detect" / "boxes.json").read_text(encoding="utf-8"))
            det.pop("run_id", None)
            (sw.root / "detect" / "boxes.json").write_text(json.dumps(det), encoding="utf-8")
            by_id = {int(pp["id"]): pp for pp in det["people"]}
            cam_res = cam.resample_to(plate.frames)
            k_map = cam_res.intrinsics_per_frame(plate.width, plate.height)
            cam_intr = _json_hash({"w": plate.width, "h": plate.height,
                                   "k": [[round(float(v), 3) for v in k_map[f].ravel()] for f in sorted(k_map)]})
            fps = float(getattr(cam, "fps", 0.0) or getattr(plate, "fps", 0.0) or 24.0)
            cam_ext = _json_hash({"c2w": [[round(float(v), 4) for v in mm.ravel()] for mm in cam_res.cam_to_world],
                                  "fps": round(fps, 4)})
            net = _json_hash({"detect_env": det_sig, "chosen_ids_boxes": [by_id[1], by_id[2]], "cam_intr": cam_intr,
                              "hand_refine": True, "use_mask": True, "prompt": "mask_v3", "masks": "tracker"})
            post = _json_hash({"net": net, "smoothing": round(smoothing, 5), "solver": f"whole_shot_v2.{SOLVER_REVISION}",
                               "camera": cam_ext, "scan": ""})
            sigs = {k: v for k, v in m["cache_signatures"].items()
                    if k not in ("people_net", "people_post", "solve_det", "detect_run")}
            sigs.update({"solve_net": net, "postprocess": post})
            st = dict(m["solved_state"])
            st["selected_ids"] = [1, 2]
            sw.update_meta(cache_signatures=sigs, solved_state=st)

        to_v021_form(smoothing=0.5)
        gens = {pid: stamp(pid) for pid in (1, 2)}
        # ...also when the grade was touched after the v0.2.1 solve (v0.2.1 had the grade in its signature; the
        # grade the task was solved with is on record in `solved_state`)
        solve("3", grade={"exposure": 0.4})
        assert fake.log == [("solve", [3])], f"the people of the v0.2.1 task were computed again: {fake.log}"
        assert solved() == [1, 2, 3] and {pid: stamp(pid) for pid in (1, 2)} == gens
        assert "solve_net" not in meta()["cache_signatures"] and set(meta()["cache_signatures"]["people_net"]) == {"1", "2", "3"}

        # the same with a setting changed since: the v0.2.1 people are re-solved from their cached network outputs
        client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [3]})
        to_v021_form(smoothing=0.5)
        solve("3", solve_params={"smoothing": 0.2})
        assert sorted(fake.log) == [("resolve", [1, 2]), ("solve", [3])], fake.log
        assert solved() == [1, 2, 3] and abs(stamp(1, "smoothing_strength") - 0.2) < 1e-6

        # ---- 16b. boxes of the v0.1.x detector (no engine tag) are not used by the present solver: pressing solve
        #           on such a task looks for people anew, which clears the people solved from the old boxes
        import re as _re
        from shotspace.web.workspace import TRACKER_ENGINE

        worker_src = (Path(__file__).resolve().parents[1] / "shotspace/solver/workers/people_tracker.py").read_text(encoding="utf-8")
        assert _re.search(r'^ENGINE_TAG = "([^"]+)"', worker_src, _re.M).group(1) == TRACKER_ENGINE
        # Adding people one by one equals solving them together only while a person's estimate depends on nobody
        # else in the call.  Two settings of the body network make that so, and both were measured on ground truth:
        # the prompts of every person are handled the one-person way ("consistent"), and the backbone computes in
        # float16 (the released bfloat16 rounds so coarsely that the result depends on the batch; float16 equals
        # float32 against ground truth at twice its speed).
        body_src = (Path(__file__).resolve().parents[1] / "shotspace/solver/workers/sam3d_worker.py").read_text(encoding="utf-8")
        assert _re.search(r'^PERSON_BATCH_MODE = "([^"]+)"', body_src, _re.M).group(1) == "consistent"
        assert _re.search(r'^BODY_PRECISION = "([^"]+)"', body_src, _re.M).group(1) == "float16"
        old_boxes = json.loads((sw.root / "detect" / "boxes.json").read_text(encoding="utf-8"))
        old_boxes.pop("engine", None)
        old_boxes.pop("run_id", None)
        (sw.root / "detect" / "boxes.json").write_text(json.dumps(old_boxes), encoding="utf-8")
        sigs_old = {k: v for k, v in meta()["cache_signatures"].items() if k != "detect_run"}
        sw.update_meta(cache_signatures=sigs_old)
        solve("1")
        assert fake.log == [("detect", []), ("solve", [1])], fake.log
        assert solved() == [1] and meta()["detection"]["engine"] == TRACKER_ENGINE

        # ---- 16c. many people are worked through in batches: the status line counts people, every finished
        #           batch is committed, and cancelling keeps the batches that were finished
        import shotspace.web.workspace as ws_mod

        assert _re.search(r"^PEOPLE_BATCH = (\d+)", body_src, _re.M).group(1) == str(ws_mod.PEOPLE_PER_PASS)
        client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": None})
        per_pass = ws_mod.PEOPLE_PER_PASS
        ws_mod.PEOPLE_PER_PASS = 2
        try:
            fake.read_line = lambda: meta().get("stage_message") or ""
            fake.seen_lines.clear()
            solve("1,2,3")
            assert fake.log == [("solve", [1, 2]), ("solve", [3])], fake.log
            assert solved() == [1, 2, 3]
            lines = [ln.split("] ", 1)[-1] for ln in fake.seen_lines]
            assert lines[0].startswith("逐帧估计：共 3 人，已完成 0 人 · 第 1/2 批（2 人）50%"), lines
            assert lines[1].startswith("整段解算：共 3 人，已完成 0 人 · 第 1/2 批（2 人）第 1/2 人"), lines
            assert lines[3].startswith("逐帧估计：共 3 人，已完成 2 人 · 第 2/2 批（1 人）50%"), lines
            assert meta()["progress"]["total"] == 1000

            client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": None})
            fake.block_after_writing = {3}
            fake.blocked.clear()
            fake._kill.clear()
            fake.log.clear()
            wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": "1,2,3"}})
            assert fake.blocked.wait(timeout=30.0)
            assert solved() == [1, 2], "the first batch must already be part of the task while the second one runs"
            assert in_view() == [1, 2]
            client.post(f"/api/shots/{sid}/cancel")
            wait_idle(sid)
            fake.block_after_writing = set()
            assert solved() == [1, 2] and on_disk(cam_dir) == [1, 2] and on_disk(raw_dir) == [1, 2]
            assert set(meta()["cache_signatures"]["people_net"]) == {"1", "2"}
            solve("3")
            assert fake.log == [("solve", [3])] and solved() == [1, 2, 3]
            # one person alone: the line says so without batch numbers
            fake.seen_lines.clear()
            client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [3]})
            solve("3")
            assert fake.seen_lines[0].split("] ", 1)[-1].startswith("逐帧估计：共 1 人 · 50%"), fake.seen_lines

            # a person the solver cannot use (too few frames) is not part of the task, leaves no files behind, and
            # the closing line says that somebody was skipped instead of showing two counts that do not agree
            client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": None})
            fake.skip = {2}
            fake.log.clear()
            solve("1,2,3")
            assert fake.log == [("solve", [1, 2]), ("solve", [3])], fake.log
            assert solved() == [1, 3] and on_disk(cam_dir) == [1, 3] and on_disk(raw_dir) == [1, 3]
            assert set(meta()["cache_signatures"]["people_net"]) == {"1", "3"}
            closing = meta()["stage_message"]
            assert "共 2 人" in closing and "新增 3 人" in closing and "其中 1 人因可解算帧数不足已跳过" in closing, closing
            assert any("人物 2 可解算帧数不足" in w for w in meta().get("solve_warnings") or []), meta().get("solve_warnings")
            fake.skip = set()
            fake.log.clear()
            solve("1,2,3")
            assert fake.log == [("solve", [2])] and solved() == [1, 2, 3], (fake.log, solved())
            assert "跳过" not in meta()["stage_message"], meta()["stage_message"]
        finally:
            ws_mod.PEOPLE_PER_PASS = per_pass
            fake.read_line = lambda: ""
            fake.skip = set()

        # ---- 17. a new plate replaces the pictures: the old boxes, outlines and solved people go with them, and
        #          the next solve looks for people in the new pictures
        assert client.get(f"/api/shots/{sid}/download/usd_skel?unit=cm").status_code == 200
        _upload_test_plate_and_camera(client, root, sid, frames)
        wait_idle(sid)
        assert solved() == [] and on_disk(cam_dir) == [] and on_disk(raw_dir) == []
        assert meta()["detection"] is None and not (sw.root / "detect" / "boxes.json").exists()
        assert not any((sw.root / "exports").glob("*")) and in_view_of(sid) == []
        solve("2")
        assert fake.log == [("detect", []), ("solve", [2])], fake.log
        assert solved() == [2]


def test_scan_is_used_on_request_and_both_results_are_kept() -> None:
    """Step 4 (v0.2.7, v0.2.12): importing a scan computes nothing; 按扫描对齐 solves the solved people again with
    the scan from the cached network outputs; the results of both states (对齐前 / 对齐后) are kept, so switching
    between them computes nothing and puts exactly the earlier files back.
    v0.2.12: with a scan imported, a solve computes BOTH states of whoever it solves (the other one by the very
    solve the switch would run), so the switch never has to compute for them - not even when the settings changed
    since (then both halves are as old as each other, and the solve button says so).
      - a person added in one state is solved in the other one when the task is switched there - only that person;
      - results kept for the other state that are no longer the result of today's inputs (new smoothing, another
        scan) are solved again when the task is switched there, never shown as up to date;
      - a person removed from the task takes the kept result along;
      - removing the scan goes back to 对齐前 with the kept results; nothing solved with the scan stays;
      - an alignment job that fails loses nothing: 对齐前 is exactly what it was;
      - a task from before v0.2.7 (no setting, scan imported) counts as 对齐后: there the scan was always in use.
    """
    import time
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        slots = list(wm.scheduler._slots.values())
        if not any(s.enabled for s in slots):
            print("  (skipped: this machine has no enabled GPU slot, the queue cannot run a job)")
            return
        fake = _StandInPeopleSolver()
        for s in slots:
            s.solver = fake

        frames = list(range(1001, 1007))
        sid = client.post("/api/shots", json={"name": "scan switch"}).json()["id"]
        _upload_test_plate_and_camera(client, root, sid, frames)
        sw = wm.get_shot(sid)
        raw_dir, cam_dir = sw.root / "solve_raw" / "raw_net", sw.root / "solve_raw" / "cam_space"
        kept_before, kept_after = sw.root / "solve_raw" / "cam_space_noscan", sw.root / "solve_raw" / "cam_space_scan"
        meta = sw.read_meta
        shown = lambda: client.get(f"/api/shots/{sid}").json()  # noqa: E731 - what the interface is told
        solved = lambda: [int(sp["id"]) for sp in (meta().get("solved_people") or [])]  # noqa: E731
        on_disk = lambda d: sorted(int(f.stem.split("_")[-1]) for f in d.glob("person_*.npz")) if d.is_dir() else []  # noqa: E731
        post_sigs = lambda: dict(meta()["cache_signatures"]["people_post"])  # noqa: E731

        def stamps(d: Path = cam_dir) -> dict[int, float]:
            out = {}
            for pid in on_disk(d):
                with np.load(d / f"person_{pid:02d}.npz", allow_pickle=False) as z:
                    out[pid] = float(z["gen"])
            return out

        def in_view() -> list[int]:
            r_view = client.get(f"/api/shots/{sid}/view/manifest")
            assert r_view.status_code == 200, r_view.text[:200]
            return sorted(int(c["person_id"]) for c in r_view.json()["characters"])

        def wait_idle(timeout: float = 60.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if wm.scheduler.job_for_shot(sid) is None and not meta().get("active_job_id"):
                    return
                time.sleep(0.03)
            raise AssertionError("the queued job did not finish")

        def solve(ids: str | None = None, **patch) -> None:
            body = dict(patch)
            if ids is not None:
                body["selection"] = {"mode": "ids", "ids": ids}
            fake.log.clear()
            fake.with_scan.clear()
            wm.enqueue_solve(sid, body or None, wait=True)

        def scan_file(name: str, y_cm: float) -> Path:
            """A scanned ground: one quad at height `y_cm`."""
            ground = RigidObjectTrack(
                name="ground", object_id=99, frames=(1001,),
                rest_points=np.array([[-800, y_cm, -1200], [800, y_cm, -1200], [800, y_cm, 600], [-800, y_cm, 600]], dtype=np.float32),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                world_transforms=np.eye(4, dtype=np.float64)[None], visible=np.ones(1, dtype=np.int32),
            )
            return export_scene(root / name, characters=[], cameras=[], shot_frames=[1001], fps=24.0, unit="cm", rigid_objects=[ground])

        def import_scan(path: Path) -> dict:
            fake.log.clear()
            fake.with_scan.clear()
            r_up = client.post(f"/api/shots/{sid}/scene_scan/upload", params={"filename": path.name, "unit": "cm"}, content=path.read_bytes())
            assert r_up.status_code == 200, r_up.text[:300]
            return r_up.json()

        def align(enabled: bool, expect_job: bool) -> dict:
            """Press 按扫描对齐 / 对齐后 (`enabled`) or 对齐前; says whether a job had to be queued for it."""
            fake.log.clear()
            fake.with_scan.clear()
            r_al = client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": enabled})
            assert r_al.status_code == 200, r_al.text[:300]
            queued = r_al.json()["status"] in ("queued", "running") or r_al.json().get("queue_job") is not None
            assert queued == expect_job, (queued, expect_job, r_al.json()["stage_message"])
            wait_idle()
            if not expect_job:
                assert fake.log == [], fake.log
            assert in_view() == solved(), (in_view(), solved())
            return shown()

        solve("1,2")
        assert fake.log == [("detect", []), ("solve", [1, 2])] and fake.with_scan == [False], (fake.log, fake.with_scan)
        before, sig_before = stamps(), post_sigs()
        people_before = {int(sp["id"]): sp for sp in meta()["solved_people"]}

        # ---- 1. importing a scan: shown, nothing computed, nothing changed, nothing queued
        scan_a = scan_file("ground_a.abc", 0.0)
        m = import_scan(scan_a)
        assert m["status"] not in ("queued", "running") and wm.scheduler.job_for_shot(sid) is None
        wait_idle()
        assert fake.log == [] and stamps() == before and post_sigs() == sig_before
        assert m["scan_align"] is False and m["scan_align_computed"] is False
        assert client.get(f"/api/shots/{sid}/view/manifest").json()["scene_scan"] is not None
        # a solve in this state still does not use the scan, and has nothing to do
        solve("1,2")
        assert fake.log == [] and stamps() == before

        # ---- 2. 按扫描对齐: everybody solved again WITH the scan from the cached outputs; the results from before are kept
        m = align(True, expect_job=True)
        assert fake.log == [("resolve", [1, 2])] and fake.with_scan == [True], (fake.log, fake.with_scan)
        assert "重新解算 2 人" in m["stage_message"], m["stage_message"]
        assert m["scan_align"] is True and m["scan_align_computed"] is True
        after, sig_after = stamps(), post_sigs()
        assert all(after[pid] != before[pid] for pid in (1, 2)) and sig_after != sig_before
        assert stamps(kept_before) == before and on_disk(kept_after) == []
        assert on_disk(raw_dir) == [1, 2]
        # what is kept and what is on show are two different files now (the job REPLACED the files on show)
        assert all((kept_before / f"person_{pid:02d}.npz").stat().st_ino != (cam_dir / f"person_{pid:02d}.npz").stat().st_ino for pid in (1, 2))

        # ---- 3. 对齐前: at once, nothing computed - the very files and readings from before are back
        m = align(False, expect_job=False)
        assert m["scan_align"] is False and m["scan_align_computed"] is True and m["stage_message"] == "已切换为对齐前"
        assert stamps() == before and post_sigs() == sig_before
        assert {int(sp["id"]): sp for sp in meta()["solved_people"]} == people_before
        assert stamps(kept_after) == after and on_disk(kept_before) == []
        solve("1,2")
        assert fake.log == [], fake.log  # ...and they are valid results: a solve has nothing to do

        # ---- 4. 对齐后 again: at once
        m = align(True, expect_job=False)
        assert m["scan_align"] is True and stamps() == after and post_sigs() == sig_after
        assert stamps(kept_before) == before
        solve("1,2")
        assert fake.log == [], fake.log

        # ---- 5. the export follows the state on show
        exp_sig = lambda: wm._compute_export_sig(meta(), "cm", 24.0, "x")  # noqa: E731
        sig_on = exp_sig()
        align(False, expect_job=False)
        assert exp_sig() != sig_on
        align(True, expect_job=False)
        assert exp_sig() == sig_on

        # ---- 6. a person added in 对齐后 is solved with the scan AND, in the same job, without it (v0.2.12);
        #         对齐前 then computes nothing
        solve("3")
        assert fake.log == [("solve", [3]), ("resolve", [3])] and fake.with_scan == [True, False] and solved() == [1, 2, 3], (fake.log, fake.with_scan)
        assert "同时算好对齐前的结果" in meta()["stage_message"], meta()["stage_message"]
        after = stamps()
        assert sorted(stamps(kept_before)) == [1, 2, 3] and not (sw.root / "solve_raw" / ".other_half_tmp").exists()
        m = align(False, expect_job=False)
        assert {pid: stamps()[pid] for pid in (1, 2)} == before and solved() == [1, 2, 3]
        before = stamps()
        assert stamps(kept_after) == after
        align(True, expect_job=False)
        assert stamps() == after
        align(False, expect_job=False)
        assert stamps() == before

        # ---- 6b. the settings changed but nobody was solved since: both halves are as old as each other, the
        #          switch still computes nothing (the solve button says the results are not of the new setting)
        assert client.patch(f"/api/shots/{sid}", json={"solve_params": {"smoothing": 0.7}}).status_code == 200
        m = align(True, expect_job=False)
        assert stamps() == after and m["scan_align"] is True
        m = align(False, expect_job=False)
        assert stamps() == before
        assert client.patch(f"/api/shots/{sid}", json={"solve_params": {"smoothing": 0.5}}).status_code == 200

        # ---- 7. a new smoothing in 对齐前: the solve computes both states with it (v0.2.12), 对齐后 is at once
        #         of the new smoothing
        solve(None, solve_params={"smoothing": 0.8})
        assert fake.log == [("resolve", [1, 2, 3]), ("resolve", [1, 2, 3])] and fake.with_scan == [False, True], (fake.log, fake.with_scan)
        assert meta()["scan_align_sig"] and shown()["scan_align_computed"] is True   # step 4 shows the switch right away
        before = stamps()
        m = align(True, expect_job=False)
        after = stamps()
        assert all(abs(float(np.load(cam_dir / f"person_{pid:02d}.npz")["smoothing_strength"]) - 0.8) < 1e-6 for pid in (1, 2, 3))

        # ---- 8. another scan while 对齐后: nothing is computed by the import; step 4 says the alignment is not
        #         of this scan; pressing the button solves everybody again with it; 对齐前 was never touched
        m = import_scan(scan_file("ground_b.abc", 12.0))
        wait_idle()
        assert fake.log == [] and stamps() == after
        assert m["scan_align"] is True and m["scan_align_computed"] is False
        m = align(True, expect_job=True)
        assert fake.log == [("resolve", [1, 2, 3])] and fake.with_scan == [True] and m["scan_align_computed"] is True
        after = stamps()
        align(False, expect_job=False)
        assert stamps() == before

        # ---- 9. ...and the other way round: another scan while 对齐前 - the kept results of 对齐后 are stale
        m = import_scan(scan_file("ground_c.abc", 20.0))
        wait_idle()
        assert fake.log == [] and stamps() == before and m["scan_align"] is False and m["scan_align_computed"] is False
        m = align(True, expect_job=True)
        assert fake.log == [("resolve", [1, 2, 3])] and fake.with_scan == [True] and m["scan_align_computed"] is True
        after = stamps()

        # ---- 10. a person removed from the task takes the kept result along
        assert on_disk(kept_before) == [1, 2, 3]
        r_rm = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [2]})
        assert r_rm.status_code == 200, r_rm.text[:200]
        assert solved() == [1, 3] and on_disk(cam_dir) == [1, 3] and on_disk(kept_before) == [1, 3]
        assert set(meta()["scan_mode_results"]["post"]) == {"1", "3"} and set(meta()["scan_mode_results"]["people"]) == {"1", "3"}

        # ---- 11. an alignment job that fails loses nothing
        align(False, expect_job=False)
        assert stamps() == {pid: before[pid] for pid in (1, 3)}
        import_scan(scan_file("ground_d.abc", 30.0))
        fake.fail_after_writing = {3}
        r_fail = client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": True})
        assert r_fail.status_code == 200, r_fail.text[:200]
        wait_idle()
        fake.fail_after_writing = set()
        m = shown()
        assert m["status"] == "error" and m["scan_align"] is True and m["scan_align_computed"] is False
        assert solved() == [1, 3] and post_sigs() == {}, post_sigs()   # nothing passes for a result with the scan
        assert stamps(kept_before) == {pid: before[pid] for pid in (1, 3)}   # ...and a half-done job did not write into what is kept
        m = align(False, expect_job=False)                              # 对齐前 is exactly what it was
        assert stamps() == {pid: before[pid] for pid in (1, 3)} and m["scan_align"] is False
        m = align(True, expect_job=True)                                # and the alignment can simply be run again
        assert fake.log == [("resolve", [1, 3])] and fake.with_scan == [True] and m["scan_align_computed"] is True
        after = stamps()

        # ---- 12. removing the scan in 对齐后: back to 对齐前 with the kept results, nothing computed, nothing of
        #          the state with the scan stays behind
        fake.log.clear()
        r_del = client.delete(f"/api/shots/{sid}/scene_scan")
        assert r_del.status_code == 200, r_del.text[:200]
        wait_idle()
        m = shown()
        assert fake.log == [] and m["scene_scan"] is None and m["scan_align"] is False and m["scan_align_computed"] is False
        assert stamps() == {pid: before[pid] for pid in (1, 3)} and in_view() == [1, 3]
        assert on_disk(kept_before) == [] and on_disk(kept_after) == [] and not meta().get("scan_mode_results")
        r_no = client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": True})
        assert r_no.status_code == 400 and "尚未导入场景扫描" in r_no.text, r_no.text[:200]

        # ---- 13. a person added while 对齐后 is on show has a result without the scan already (v0.2.12):
        #          removing the scan computes nothing
        import_scan(scan_a)
        align(True, expect_job=True)
        solve("2")
        assert fake.log == [("solve", [2]), ("resolve", [2])] and fake.with_scan == [True, False] and solved() == [1, 2, 3], (fake.log, fake.with_scan)
        fake.log.clear()
        fake.with_scan.clear()
        assert client.delete(f"/api/shots/{sid}/scene_scan").status_code == 200
        wait_idle()
        assert fake.log == [], (fake.log, fake.with_scan)
        assert {pid: stamps()[pid] for pid in (1, 3)} == {pid: before[pid] for pid in (1, 3)} and solved() == [1, 2, 3]
        assert shown()["scan_align"] is False and on_disk(kept_after) == [] and on_disk(kept_before) == []
        #          ...and a person who has NO result without the scan (solved before v0.2.12, say) is solved without it
        import_scan(scan_a)
        align(True, expect_job=True)
        with sw.lock:
            m_old = sw.read_meta()
            kept_old = dict(m_old.get("scan_mode_results") or {})
            kept_old["post"] = {k: v for k, v in kept_old["post"].items() if k != "2"}
            kept_old["people"] = {k: v for k, v in kept_old["people"].items() if k != "2"}
            kept_old["pair"] = {k: v for k, v in (kept_old.get("pair") or {}).items() if k != "2"}
            sw.update_meta(scan_mode_results=kept_old)
        (kept_before / "person_02.npz").unlink()
        fake.log.clear()
        fake.with_scan.clear()
        assert client.delete(f"/api/shots/{sid}/scene_scan").status_code == 200
        wait_idle()
        assert fake.log == [("resolve", [2])] and fake.with_scan == [False], (fake.log, fake.with_scan)
        assert shown()["scan_align"] is False and solved() == [1, 2, 3]

        # ---- 13b. the other half cannot be computed (a failure) or is cancelled: the solve on show stays complete
        import_scan(scan_a)
        fake.resolve_fails = {3}
        solve(None, solve_params={"smoothing": 0.6})
        fake.resolve_fails = set()
        m = shown()
        assert m["status"] == "ready" and solved() == [1, 2, 3] and fake.with_scan == [False, True], (m["status"], fake.log, fake.with_scan)
        assert any("切换时再算" in w for w in meta().get("solve_warnings") or []), meta().get("solve_warnings")
        assert m["scan_align_computed"] is False                    # 对齐后 is not complete: step 4 offers 按扫描对齐
        on_show_now = stamps()
        fake.blocked.clear()
        fake._kill.clear()
        fake.resolve_blocks = {1}
        fake.log.clear()
        fake.with_scan.clear()
        wm.enqueue_solve(sid, {"solve_params": {"smoothing": 0.65}})
        assert fake.blocked.wait(10.0)
        assert client.post(f"/api/shots/{sid}/cancel").status_code == 200
        wait_idle()
        fake.resolve_blocks = set()
        m = shown()
        assert m["status"] == "ready" and "解算已完成" in m["stage_message"] and "已取消" in m["stage_message"], m["stage_message"]
        assert solved() == [1, 2, 3] and all(stamps()[pid] != on_show_now[pid] for pid in (1, 2, 3))   # the new results are on show
        assert not (sw.root / "solve_raw" / ".other_half_tmp").exists()
        m = align(True, expect_job=True)                              # 对齐后 is computed when the task goes there
        assert fake.with_scan == [True] and m["scan_align_computed"] is True
        align(False, expect_job=False)
        assert client.delete(f"/api/shots/{sid}/scene_scan").status_code == 200
        wait_idle()

        # ---- 14. the scan imported BEFORE anybody is solved: 按扫描对齐 only turns the state on (nothing to
        #          compute), and the people solved afterwards are solved with the scan straight away
        sid2 = client.post("/api/shots", json={"name": "scan first"}).json()["id"]
        _upload_test_plate_and_camera(client, root, sid2, frames)
        r_up = client.post(f"/api/shots/{sid2}/scene_scan/upload", params={"filename": scan_a.name, "unit": "cm"}, content=scan_a.read_bytes())
        assert r_up.status_code == 200 and r_up.json()["scan_align"] is False
        r_al = client.post(f"/api/shots/{sid2}/scene_scan/align", json={"enabled": True})
        assert r_al.status_code == 200 and r_al.json()["scan_align"] is True and r_al.json()["scan_align_computed"] is True
        assert r_al.json()["status"] not in ("queued", "running") and r_al.json()["stage_message"] == "已按扫描对齐"
        fake.log.clear()
        fake.with_scan.clear()
        wm.enqueue_solve(sid2, {"selection": {"mode": "ids", "ids": "1"}}, wait=True)
        assert fake.log == [("detect", []), ("solve", [1]), ("resolve", [1])] and fake.with_scan == [True, False], (fake.log, fake.with_scan)

        # ---- 15. a task from before v0.2.7: no setting, a scan imported -> the scan is in use, as it was
        sw2 = wm.get_shot(sid2)
        old_meta = sw2.read_meta()
        old_meta.pop("scan_align", None)
        old_meta.pop("scan_align_sig", None)
        old_meta.pop("scan_mode_results", None)
        sw2.write_meta(old_meta)
        assert client.get(f"/api/shots/{sid2}").json()["scan_align"] is True
        fake.log.clear()
        fake.with_scan.clear()
        wm.enqueue_solve(sid2, {"selection": {"mode": "ids", "ids": "1,2"}}, wait=True)
        assert fake.log == [("solve", [2]), ("resolve", [2])] and fake.with_scan == [True, False], (fake.log, fake.with_scan)   # person 1 is still valid
        r_off = client.post(f"/api/shots/{sid2}/scene_scan/align", json={"enabled": False})
        assert r_off.status_code == 200 and r_off.json()["scan_align"] is False
        t0 = time.time()
        while (wm.scheduler.job_for_shot(sid2) is not None or sw2.read_meta().get("active_job_id")) and time.time() - t0 < 60.0:
            time.sleep(0.03)
        # person 1 was solved before (no result without the scan, it is computed now); person 2 has it already
        assert fake.log[-1] == ("resolve", [1]) and fake.with_scan[-1] is False, (fake.log, fake.with_scan)
        assert sw2.read_meta()["scan_align"] is False


def test_a_track_that_changes_person_becomes_several_people() -> None:
    """Identity (v0.2.9): the solve cuts a track that jumps from one person to another (`track_split`, tested on its
    own) and gives back several people.  Here: what the task does with them.
      - the part before the jump keeps the track's number, the part after it gets the first free number; both are
        people of the task like any other - in the tracker's record (each with the boxes of its own frames), in the
        list, in the view, in the cached network outputs;
      - nobody is estimated or cut twice: a solve right afterwards has nothing to do, a new smoothing solves every
        part from ITS cached outputs;
      - the new person can be removed and solved again by its number; it is outlined under the old track's number;
      - a cut found while solving from cached outputs (a task solved before this version) is taken over the same way;
      - the closing line and the notes of the solve say what was cut.
    """
    import json
    import time
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        slots = list(wm.scheduler._slots.values())
        if not any(s.enabled for s in slots):
            print("  (skipped: this machine has no enabled GPU slot, the queue cannot run a job)")
            return
        fake = _StandInPeopleSolver()
        for s in slots:
            s.solver = fake

        frames = list(range(1001, 1007))
        sid = client.post("/api/shots", json={"name": "cut"}).json()["id"]
        _upload_test_plate_and_camera(client, root, sid, frames)
        sw = wm.get_shot(sid)
        raw_dir, cam_dir = sw.root / "solve_raw" / "raw_net", sw.root / "solve_raw" / "cam_space"
        meta = sw.read_meta
        solved = lambda: [int(sp["id"]) for sp in (meta().get("solved_people") or [])]  # noqa: E731
        on_disk = lambda d: sorted(int(f.stem.split("_")[-1]) for f in d.glob("person_*.npz"))  # noqa: E731
        detected = lambda: {int(p["id"]): p for p in meta()["detection"]["people"]}  # noqa: E731
        box_frames = lambda pid: sorted(int(k) for k in detected()[pid]["boxes"])  # noqa: E731

        def frames_in(folder: Path, pid: int) -> list[int]:
            with np.load(folder / f"person_{pid:02d}.npz", allow_pickle=False) as z:
                return [int(f) for f in z["frames"]]

        def in_view() -> list[int]:
            r_view = client.get(f"/api/shots/{sid}/view/manifest")
            assert r_view.status_code == 200, r_view.text[:200]
            return sorted(int(c["person_id"]) for c in r_view.json()["characters"])

        def wait_idle(timeout: float = 60.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if wm.scheduler.job_for_shot(sid) is None and not meta().get("active_job_id"):
                    return
                time.sleep(0.03)
            raise AssertionError("the queued job did not finish")

        def solve(ids: str | None = None, **patch) -> None:
            body = dict(patch)
            if ids is not None:
                body["selection"] = {"mode": "ids", "ids": ids}
            fake.log.clear()
            fake.id_starts.clear()
            wm.enqueue_solve(sid, body or None, wait=True)

        # ---- 1. track 2 follows somebody else from frame 1004 on: it comes back as #2 (1001-1003) and #4 (1004-1006)
        fake.cut = {2: 1004}
        solve("1,2")
        assert fake.log == [("detect", []), ("solve", [1, 2])] and fake.id_starts == [4], (fake.log, fake.id_starts)   # 3 people found: 4 is free
        assert solved() == [1, 2, 4] and on_disk(cam_dir) == [1, 2, 4] and on_disk(raw_dir) == [1, 2, 4] and in_view() == [1, 2, 4]
        assert sorted(detected()) == [1, 2, 3, 4]
        assert box_frames(2) == [1001, 1002, 1003] and box_frames(4) == [1004, 1005, 1006] and box_frames(1) == frames
        assert detected()[2]["frame_count"] == 3 and detected()[4]["frame_count"] == 3
        assert detected()[4]["split_from"] == 2 and detected()[4]["mask_id"] == 2 and "split_from" not in detected()[2]
        assert abs(detected()[2]["prominence"] + detected()[4]["prominence"] - 998.0) < 0.2   # the track's 1000 - 2, shared out
        assert frames_in(raw_dir, 2) == [1001, 1002, 1003] and frames_in(raw_dir, 4) == [1004, 1005, 1006]
        assert frames_in(cam_dir, 2) == [1001, 1002, 1003] and frames_in(cam_dir, 4) == [1004, 1005, 1006]
        assert not list(raw_dir.glob(".split_*")), list(raw_dir.glob(".split_*"))
        on_file = json.loads((sw.root / "detect" / "boxes.json").read_text(encoding="utf-8"))
        assert on_file["people"] == meta()["detection"]["people"] and on_file["run_id"] == meta()["cache_signatures"]["detect_run"]
        assert [sp.get("split_from") for sp in meta()["solved_people"]] == [None, None, 2]
        closing = meta()["stage_message"]
        assert "共 3 人" in closing and "新增 2 人" in closing and "轨迹拆分出 1 人" in closing, closing
        assert any("#2" in w and "#4" in w and "拆分" in w for w in meta().get("solve_warnings") or []), meta().get("solve_warnings")
        assert set(meta()["cache_signatures"]["people_net"]) == {"1", "2", "4"} == set(meta()["cache_signatures"]["people_post"])
        assert client.get(f"/api/shots/{sid}").json()["export_person_ids"] == []   # everybody is exported, the new person too

        # ---- 2. nothing is estimated or cut twice: the same solve again has nothing to do (the worker would cut
        #         track 2 again if it were handed the whole track once more)
        solve("1,2")
        assert fake.log == [] and solved() == [1, 2, 4] and "已是最新" in meta()["stage_message"], (fake.log, meta()["stage_message"])
        fake.cut = {}

        # ---- 3. a new smoothing: every part is solved again from ITS cached outputs
        solve(None, solve_params={"smoothing": 0.8})
        assert fake.log == [("resolve", [1, 2, 4])] and fake.id_starts == [5], (fake.log, fake.id_starts)
        assert frames_in(cam_dir, 2) == [1001, 1002, 1003] and frames_in(cam_dir, 4) == [1004, 1005, 1006] and solved() == [1, 2, 4]

        # ---- 4. the new person is a person: removed, then solved again by number - outlined as track 2
        r_rm = client.post(f"/api/shots/{sid}/solved_people/remove", json={"person_ids": [4]})
        assert r_rm.status_code == 200 and solved() == [1, 2] and on_disk(raw_dir) == [1, 2] and sorted(detected()) == [1, 2, 3, 4]
        solve("4")
        assert fake.log == [("solve", [4])] and fake.mask_ids == {4: 2} and solved() == [1, 2, 4], (fake.log, fake.mask_ids)
        assert frames_in(raw_dir, 4) == [1004, 1005, 1006] and frames_in(cam_dir, 4) == [1004, 1005, 1006]
        # the table of the day says whom a person was cut from - when it was cut, and every time it is solved after that
        import csv
        table_rows: list[dict] = []
        for table_file in sorted((root / "data_collect").glob("shotspace_data_*.csv")):
            with table_file.open("r", encoding="utf-8-sig", newline="") as fh:
                table_rows.extend(csv.DictReader(fh))
        of_4 = [r for r in table_rows if r["记录类型"] == "人物" and r["人物编号"] == "4"]
        assert [(r["拆分自"], r["本次计算"]) for r in of_4] == [("2", "新增"), ("2", "重新解算"), ("2", "新增")], [(r["拆分自"], r["本次计算"]) for r in of_4]
        assert all(r["拆分自"] == "" for r in table_rows if r["记录类型"] == "人物" and r["人物编号"] in ("1", "2"))

        # ---- 5. a cut found while solving from cached outputs (a task solved before this version): taken over the same
        fake.cut = {1: 1003}
        solve(None, solve_params={"smoothing": 0.5})
        assert fake.log == [("resolve", [1, 2, 4])] and fake.id_starts == [5], (fake.log, fake.id_starts)
        assert solved() == [1, 2, 4, 5] and on_disk(raw_dir) == [1, 2, 4, 5] and in_view() == [1, 2, 4, 5]
        assert box_frames(1) == [1001, 1002] and box_frames(5) == [1003, 1004, 1005, 1006] and detected()[5]["split_from"] == 1
        assert frames_in(raw_dir, 1) == [1001, 1002] and frames_in(raw_dir, 5) == [1003, 1004, 1005, 1006]
        assert "重新解算 3 人" in meta()["stage_message"] and "轨迹拆分出 1 人" in meta()["stage_message"], meta()["stage_message"]
        fake.cut = {}
        solve("1")
        assert fake.log == [] and solved() == [1, 2, 4, 5]
        wait_idle()

        # ---- 6. looking for people again starts from the tracker's own tracks: the parts are gone with the old numbers
        assert client.post(f"/api/shots/{sid}/detect", json={}).status_code == 200
        wait_idle()
        assert solved() == [] and sorted(detected()) == [1, 2, 3] and box_frames(2) == frames


def test_data_collection_writes_the_table_of_the_day() -> None:
    """Data collection (v0.2.8): every job that computed something, failed or was cancelled leaves its data in the
    table of the day - one row for the job, one per person it computed - and nothing else does:
      - the job's row holds the camera of every frame, the settings, the timings and the scan's ground heights;
      - a person's row holds the solved body points in WORLD space (a hidden frame stays where its anchor frame
        put it), the ground height where the person is, and for a person the network looked at in this job the
        network's own per-frame outputs; a person solved again from cached outputs has no network columns;
      - importing a scan, switching between the kept results and a solve with nothing to do write nothing;
      - a failing job gets its row with the error, and no row for the person it did not finish;
      - the table holds no file path;
      - a recording that goes wrong never disturbs the job.
    """
    import csv
    import time
    from datetime import datetime
    from fastapi.testclient import TestClient
    from shotspace import __version__
    from shotspace.server_config import ConfigStore
    from shotspace.web import data_collect as dc
    from shotspace.web import workspace as ws_mod
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        slots = list(wm.scheduler._slots.values())
        if not any(s.enabled for s in slots):
            print("  (skipped: this machine has no enabled GPU slot, the queue cannot run a job)")
            return
        fake = _StandInPeopleSolver()
        fake.rich = True
        for s in slots:
            s.solver = fake

        data_dir = root / "data_collect"
        assert wm.data_collector.dir == data_dir.resolve(), wm.data_collector.dir   # next to the task folder, not in it
        frames = list(range(1001, 1007))
        n = len(frames)
        sid = client.post("/api/shots", json={"name": "镜头, 带逗号"}).json()["id"]
        _upload_test_plate_and_camera(client, root, sid, frames)
        sw = wm.get_shot(sid)
        meta = sw.read_meta

        def wait_idle(timeout: float = 60.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if wm.scheduler.job_for_shot(sid) is None and not meta().get("active_job_id"):
                    return
                time.sleep(0.03)
            raise AssertionError("the queued job did not finish")

        def table() -> list[dict]:
            files = sorted(data_dir.glob(f"{dc.TABLE_PREFIX}*.csv")) if data_dir.is_dir() else []
            rows: list[dict] = []
            for f in files:
                with f.open("r", encoding="utf-8-sig", newline="") as fh:
                    rows.extend(csv.DictReader(fh))
            return rows

        def nums(cell: str) -> np.ndarray:
            return np.array([float(v) for v in cell.split()], dtype=np.float64)

        def solve(ids: str, **patch) -> None:
            wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": ids}, **patch}, wait=True)

        assert table() == []

        # ---- 1. a scan is imported: no job, nothing recorded
        ground = RigidObjectTrack(
            name="ground", object_id=99, frames=(1001,),
            rest_points=np.array([[-800, 12, -1200], [800, 12, -1200], [800, 12, 600], [-800, 12, 600]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            world_transforms=np.eye(4, dtype=np.float64)[None], visible=np.ones(1, dtype=np.int32),
        )
        scan = export_scene(root / "secret_ground.abc", characters=[], cameras=[], shot_frames=[1001], fps=24.0, unit="cm", rigid_objects=[ground])
        r_up = client.post(f"/api/shots/{sid}/scene_scan/upload", params={"filename": scan.name, "unit": "cm"}, content=scan.read_bytes())
        assert r_up.status_code == 200, r_up.text[:200]
        assert table() == []

        # ---- 2. the first solve (it looks for people, then solves two of them): one row for the job, one per person
        solve("1,2")
        rows = table()
        assert [r["记录类型"] for r in rows] == ["任务", "人物", "人物"], [r["记录类型"] for r in rows]
        job, p1, p2 = rows
        assert len({r["作业编号"] for r in rows}) == 1 and job["作业编号"]
        assert job["日期"] == datetime.now().strftime("%Y-%m-%d") and job["软件版本"] == __version__
        assert job["任务编号"] == sid and job["任务名称"] == "镜头, 带逗号" and job["表格版本"] == str(dc.SCHEMA_VERSION)
        assert job["作业类型"] == "人物解算" and job["作业结果"] == "完成" and job["错误信息"] == ""
        assert (job["本次新增人数"], job["本次重新解算人数"], job["本次跳过人数"], job["已解算人数"]) == ("2", "0", "0", "2")
        assert job["检测到人数"] == "3" and job["手动框人数"] == "0" and job["平滑"] == "0.5" and job["手部细节"] == "开"
        assert (job["画面宽_像素"], job["画面高_像素"], job["帧数"], job["起始帧"], job["结束帧"], job["画面格式"]) == ("640", "360", "6", "1001", "1006", "png")
        assert float(job["帧率"]) == 24.0 and float(job["耗时_秒"]) >= 0.0 and job["人物检测耗时_秒"] != "" and job["显卡编号"] != ""
        # the camera of every frame, as the product holds it (centimetres)
        assert job["摄影机帧号"].split() == [str(f) for f in frames] and job["摄影机格式"] == "abc" and job["摄影机单位"] == "m"
        cam_x = np.linspace(-40.0, 40.0, n)
        assert np.allclose(nums(job["摄影机位置X_厘米"]), cam_x, atol=2e-3) and np.allclose(nums(job["摄影机位置Y_厘米"]), 150.0, atol=2e-3)
        assert np.allclose(nums(job["摄影机位置Z_厘米"]), 420.0, atol=2e-3)
        for i in range(3):
            for j in range(3):
                assert np.allclose(nums(job[f"摄影机旋转R{i}{j}"]), 1.0 if i == j else 0.0, atol=1e-6), (i, j)
        assert np.allclose(nums(job["焦距_毫米"]), 35.0) and np.allclose(nums(job["胶片口宽_毫米"]), 36.0) and np.allclose(nums(job["胶片口高_毫米"]), 20.25)
        assert np.allclose(nums(job["内参FX_像素"]), 35.0 / 36.0 * 640.0, atol=2e-3) and np.allclose(nums(job["内参CX_像素"]), 320.0, atol=2e-3)
        assert np.allclose(nums(job["内参CY_像素"]), 180.0, atol=2e-3)
        # the scan: imported, not in use; the height of its ground on a grid around the people
        assert job["扫描"] == "已导入" and job["扫描状态"] == "对齐前" and job["扫描格式"] == "abc" and job["扫描面数"] == "2"
        gx, gz, gh = nums(job["地面采样X_厘米"]), nums(job["地面采样Z_厘米"]), nums(job["地面采样高度_厘米"])
        assert len(gx) == len(gz) == len(gh) == int(job["地面采样点数"]) >= 9 and np.allclose(gh, 12.0, atol=1e-2)
        assert float(job["地面采样间距_厘米"]) >= dc.GRID_CELL_CM and len(np.unique(gx)) <= dc.GRID_MAX_SIDE
        assert gx.min() >= -800.0 - 1e-6 and gx.max() <= 800.0 + 1e-6 and gz.min() >= -1200.0 - 1e-6 and gz.max() <= 600.0 + 1e-6
        assert (float(job["地面高度中位_厘米"]), float(job["地面高度最低_厘米"]), float(job["地面高度最高_厘米"])) == (12.0, 12.0, 12.0)
        # a person: readings, and every frame's body points in WORLD space = the frame's camera applied to the result
        for prow, pid in ((p1, 1), (p2, 2)):
            assert prow["人物编号"] == str(pid) and prow["本次计算"] == "新增" and prow["手动框"] == "否" and prow["扫描状态"] == "对齐前"
            assert prow["帧号"].split() == [str(f) for f in frames] and prow["可见"].split() == ["1", "1", "0", "1", "1", "1"]
            assert prow["已解算"].split() == ["1", "1", "0", "1", "1", "1"]
            assert (prow["画面贴合_像素"], prow["体型比例"], prow["尺度来源"], prow["隐藏帧数"]) == ("1.0", "1.0", "模型", "0")
            assert float(prow["模型身高_厘米"]) == 80.0
            t_idx = np.arange(n, dtype=np.float64)
            placed = t_idx.copy()
            placed[2] = 1.0  # the hidden frame: the body AND the camera of its anchor frame
            assert np.allclose(nums(prow["骨盆X_厘米"]), 10.0 * pid + 9.5 + cam_x[placed.astype(int)], atol=6e-3), prow["骨盆X_厘米"]
            assert np.allclose(nums(prow["骨盆Y_厘米"]), 100.0 + placed + 150.0, atol=6e-3)
            assert np.allclose(nums(prow["骨盆Z_厘米"]), -400.0 - 9.5 + 420.0, atol=6e-3)
            assert np.allclose(nums(prow["左踝X_厘米"]), 10.0 * pid + 13.0 + cam_x[placed.astype(int)], atol=6e-3)
            assert np.allclose(nums(prow["颈Z_厘米"]), -400.0 - 69.0 + 420.0, atol=6e-3)
            pelvis = np.stack([nums(prow[f"骨盆{a}_厘米"]) for a in "XYZ"], axis=1)
            cam_pos = np.stack([cam_x, np.full(n, 150.0), np.full(n, 420.0)], axis=1)
            assert np.allclose(nums(prow["到摄影机距离_厘米"]), np.linalg.norm(pelvis - cam_pos, axis=1), atol=2e-2)
            assert np.allclose(nums(prow["所在位置地面高度_厘米"]), 12.0, atol=1e-2)
            assert np.allclose(nums(prow["逐帧画面误差_像素"]), 1.0 + 0.5 * t_idx) and prow["触地左"].split() == ["0", "1", "0", "1", "0", "1"]
            assert np.allclose(nums(prow["框X1_像素"]), 100.0 + t_idx) and float(prow["体型参数_03"]) == 0.003 and float(prow["比例参数_02"]) == 0.004
            # what the network said, frame by frame: everything the whole-shot solve starts from
            assert prow["网络_帧号"].split() == [str(f) for f in frames]
            assert np.allclose(nums(prow["网络_位置X_米"]), 0.1 * pid + 0.01 * t_idx, atol=1e-5) and np.allclose(nums(prow["网络_位置Z_米"]), 4.0 + 0.1 * t_idx, atol=1e-5)
            assert np.allclose(nums(prow["网络_整体旋转Y"]), 0.02 * t_idx, atol=1e-5) and np.allclose(nums(prow["网络_检测分数"]), np.linspace(0.5, 0.9, n), atol=1e-4)
            assert np.allclose(nums(prow["网络_姿态_005"]), 0.005 - 0.01 * t_idx, atol=1e-5) and np.allclose(nums(prow["网络_姿态_132"]), 0.132 - 0.01 * t_idx, atol=1e-5)
            assert np.allclose(nums(prow["网络_体型_44"]), 0.044 + 0.01 * t_idx, atol=1e-5) and np.allclose(nums(prow["网络_比例_27"]), 0.054, atol=1e-5)
            assert np.allclose(nums(prow["网络_关键点X_10"]), 110.0 + t_idx, atol=1e-2) and np.allclose(nums(prow["网络_关键点Y_69"]), 188.0, atol=1e-2)
            assert prow["网络_关键点在轮廓内_00"].split() == ["1"] * n and prow["网络_有轮廓"].split() == ["1"] * n
            assert np.allclose(nums(prow["网络_轮廓框X0"]), 22.0 + t_idx) and np.allclose(nums(prow["网络_轮廓面积_像素"]), 5000.0 + 10.0 * t_idx)
            assert prow["作业结果"] == "" and prow["摄影机帧号"] == ""   # a person's row repeats nothing of the job's
        # every column the module declares is in the file, in order; the guide lies next to the table
        first_file = sorted(data_dir.glob(f"{dc.TABLE_PREFIX}*.csv"))[0]
        assert first_file.name == f"{dc.TABLE_PREFIX}{datetime.now().strftime('%Y%m%d')}.csv"
        with first_file.open("r", encoding="utf-8-sig", newline="") as fh:
            assert next(csv.reader(fh)) == dc.COLUMNS
        assert first_file.read_bytes()[:3] == b"\xef\xbb\xbf" and "表格版本" in (data_dir / "字段说明.txt").read_text(encoding="utf-8")

        # ---- 3. nothing to do, and switching between kept results: no rows
        solve("1,2")
        assert len(table()) == 3

        # ---- 4. 按扫描对齐 after a solve that computed both states already (the scan was imported before it,
        #         v0.2.12): nothing to compute, nothing recorded
        assert client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": True}).status_code == 200
        wait_idle()
        assert len(table()) == 3 and client.get(f"/api/shots/{sid}").json()["scan_align"] is True
        assert client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": False}).status_code == 200
        wait_idle()
        assert len(table()) == 3
        # ---- 4b. another scan (the same ground, scanned larger): 按扫描对齐 is a job that solves both again with it
        #          from cached outputs - rows without network columns
        ground2 = RigidObjectTrack(
            name="ground", object_id=99, frames=(1001,),
            rest_points=np.array([[-900, 12, -1300], [900, 12, -1300], [900, 12, 700], [-900, 12, 700]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            world_transforms=np.eye(4, dtype=np.float64)[None], visible=np.ones(1, dtype=np.int32),
        )
        scan2 = export_scene(root / "secret_ground2.abc", characters=[], cameras=[], shot_frames=[1001], fps=24.0, unit="cm", rigid_objects=[ground2])
        r_up2 = client.post(f"/api/shots/{sid}/scene_scan/upload", params={"filename": scan2.name, "unit": "cm"}, content=scan2.read_bytes())
        assert r_up2.status_code == 200, r_up2.text[:200]
        assert client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": True}).status_code == 200
        wait_idle()
        rows = table()
        assert [r["记录类型"] for r in rows[3:]] == ["任务", "人物", "人物"], [r["记录类型"] for r in rows]
        job2, q1, _q2 = rows[3:]
        assert job2["作业编号"] != job["作业编号"] and job2["扫描状态"] == "对齐后" and job2["人物检测耗时_秒"] == ""
        assert (job2["本次新增人数"], job2["本次重新解算人数"]) == ("0", "2") and job2["整段解算耗时_秒"] != ""
        assert q1["本次计算"] == "重新解算" and q1["扫描状态"] == "对齐后" and q1["网络_帧号"] == "" and q1["网络_姿态_005"] == ""
        assert q1["骨盆X_厘米"] == p1["骨盆X_厘米"] and q1["所在位置地面高度_厘米"] == p1["所在位置地面高度_厘米"]
        assert client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": False}).status_code == 200
        wait_idle()
        assert client.post(f"/api/shots/{sid}/scene_scan/align", json={"enabled": True}).status_code == 200
        wait_idle()
        assert len(table()) == 6   # switching between the two kept results computes nothing and records nothing

        # ---- 5. a job that fails: its row says so; the person it did not finish has no row
        fake.fail_after_writing = {3}
        try:
            solve("3")
            raise AssertionError("the job was expected to fail")
        except RuntimeError:
            pass
        fake.fail_after_writing = set()
        wait_idle()
        rows = table()
        assert len(rows) == 7 and rows[6]["记录类型"] == "任务" and rows[6]["作业结果"] == "失败", [r["记录类型"] for r in rows]
        assert "stand-in worker failure" in rows[6]["错误信息"] and rows[6]["已解算人数"] == "2"

        # ---- 6. a detection job on its own
        assert client.post(f"/api/shots/{sid}/detect", json={}).status_code == 200
        wait_idle()
        rows = table()
        assert len(rows) == 8 and (rows[7]["作业类型"], rows[7]["作业结果"], rows[7]["检测到人数"]) == ("人物检测", "完成", "3")
        assert rows[7]["人物检测耗时_秒"] != "" and rows[7]["摄影机帧号"].split() == [str(f) for f in frames]

        # ---- 7. no file path, no file name of what was uploaded
        text = "".join(f.read_text(encoding="utf-8-sig") for f in sorted(data_dir.glob(f"{dc.TABLE_PREFIX}*.csv")))
        for secret in (str(root), "secret_ground", "track_cam", "sample.png", "shot.1001"):
            assert secret not in text, secret

        # ---- 8. a recording that goes wrong does not disturb the job; the error is logged next to the tables
        real_person_row = dc._person_row
        dc._person_row = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken recording"))
        try:
            solve("1")
        finally:
            dc._person_row = real_person_row
        assert meta()["status"] == "ready" and [int(sp["id"]) for sp in meta()["solved_people"]] == [1]
        rows = table()
        assert len(rows) == 9 and rows[8]["记录类型"] == "任务" and rows[8]["作业结果"] == "完成"   # the job's row is still written
        assert "broken recording" in (data_dir / "collect_errors.log").read_text(encoding="utf-8")
        real_record = ws_mod.record_job
        ws_mod.record_job = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recording is gone"))
        try:
            solve("1,2")
        finally:
            ws_mod.record_job = real_record
        assert meta()["status"] == "ready" and [int(sp["id"]) for sp in meta()["solved_people"]] == [1, 2] and len(table()) == 9

        # ---- 9. a table of the same day with other columns (the software was updated during the day) is left
        #         alone, a second one is begun; the admin page is told where the tables are
        other = dc.DataCollector(root / "other_tables")
        day = datetime(2026, 11, 24, 12, 0, 0)
        (root / "other_tables").mkdir()
        (root / "other_tables" / f"{dc.TABLE_PREFIX}20261124.csv").write_text("旧列一,旧列二\r\n1,2\r\n", encoding="utf-8-sig", newline="")
        second = other.append([{"记录类型": "任务", "任务编号": "x"}], day)
        assert second is not None and second.name == f"{dc.TABLE_PREFIX}20261124_2.csv"
        assert other.append([{"记录类型": "任务", "任务编号": "y"}], day) == second
        assert (root / "other_tables" / f"{dc.TABLE_PREFIX}20261124.csv").read_bytes() == "\ufeff旧列一,旧列二\r\n1,2\r\n".encode("utf-8")
        stats = wm.storage_stats()
        assert stats["data_collect_dir"] == str(data_dir.resolve()) and stats["data_collect_today_mb"] > 0.0 and stats["data_collect_total_mb"] > 0.0


def test_v016_manual_keyframe_interpolation_states_and_crowd_dedup() -> None:
    """Verify v0.1.6 stateful manual person box keyframing (`active`, `occluded`, `off`),
    linear keyframe interpolation, 1-to-1 Python <-> TypeScript parity, visibility of occluded/off keyframes,
    and HTTP API persistence."""
    from fastapi.testclient import TestClient

    from shotspace.core.smoothing import expand_to_full_shot_with_visibility
    from shotspace.solver.select_people import (
        SelectionSpec,
        build_manual_detected_people,
        evaluate_manual_person_boxes,
        select_people,
    )
    from shotspace.web.app import create_app

    shot_frames = list(range(1001, 1021))  # 1001..1020 (20 frames)
    manual_track = {
        "id": 101,
        "name": "手动 #101",
        "enabled": True,
        "keyframes": [
            # Pre-entry 1001..1002 -> inactive
            {"frame": 1003, "status": "active", "box": [100.0, 200.0, 220.0, 560.0]},
            {"frame": 1007, "status": "active", "box": [180.0, 220.0, 300.0, 580.0]},
            # Fully occluded starting at 1010..1013 (1008..1009 hold keyframe 1007!)
            {"frame": 1010, "status": "occluded"},
            # Re-emerges under the SAME person ID #101 at 1014..1016!
            {"frame": 1014, "status": "active", "box": [340.0, 210.0, 460.0, 570.0]},
            {"frame": 1016, "status": "active", "box": [380.0, 210.0, 500.0, 570.0]},
            # Box invalidated / terminated starting at 1018..1020
            {"frame": 1018, "status": "off"},
        ],
    }

    boxes_py, statuses_py = evaluate_manual_person_boxes(manual_track, shot_frames)

    # 1. Pre-first-keyframe inactive
    assert "1001" not in boxes_py and statuses_py[1001] == "inactive"
    assert "1002" not in boxes_py and statuses_py[1002] == "inactive"
    # 2. Linear interpolation between 1003 and 1007 (midpoint 1005 -> alpha=0.5 -> [140, 210, 260, 570])
    assert boxes_py["1003"] == [100.0, 200.0, 220.0, 560.0] and statuses_py[1003] == "key_active"
    assert boxes_py["1005"] == [140.0, 210.0, 260.0, 570.0] and statuses_py[1005] == "interp_active"
    assert boxes_py["1007"] == [180.0, 220.0, 300.0, 580.0] and statuses_py[1007] == "key_active"
    # 3. Hold active on 1008..1009 up to `occluded` at 1010
    assert boxes_py["1009"] == [180.0, 220.0, 300.0, 580.0] and statuses_py[1009] == "hold_active"
    for f_occ in (1010, 1011, 1012, 1013):
        assert str(f_occ) not in boxes_py and statuses_py[f_occ] == "occluded"
    # 4. Re-activation & interpolation on 1014..1016, hold on 1017, then `off` on 1018..1020
    assert boxes_py["1015"] == [360.0, 210.0, 480.0, 570.0] and statuses_py[1015] == "interp_active"
    assert boxes_py["1017"] == [380.0, 210.0, 500.0, 570.0] and statuses_py[1017] == "hold_active"
    for f_off in (1018, 1019, 1020):
        assert str(f_off) not in boxes_py and statuses_py[f_off] == "off"

    # 5. Verify 1-to-1 bit-parity against TypeScript `webui/src/utils/manualBoxes.ts` via Node/Vite esbuild
    esbuild_bin = Path(__file__).resolve().parents[1] / "webui" / "node_modules" / ".bin" / "esbuild"
    if esbuild_bin.is_file():
        with tempfile.TemporaryDirectory() as td_js:
            bundle_mjs = Path(td_js) / "manualBoxes.mjs"
            subprocess.run(
                [
                    str(esbuild_bin),
                    str(Path(__file__).resolve().parents[1] / "webui" / "src" / "utils" / "manualBoxes.ts"),
                    "--bundle",
                    "--format=esm",
                    f"--outfile={bundle_mjs}",
                ],
                check=True,
                capture_output=True,
            )
            node_out = subprocess.check_output(
                [
                    "node",
                    "--input-type=module",
                    "-e",
                    (
                        f"import {{ evaluateManualPersonTrack }} from {str(bundle_mjs)!r};"
                        f"const res = evaluateManualPersonTrack({json.dumps(manual_track)}, {json.dumps(shot_frames)});"
                        "console.log(JSON.stringify(res));"
                    ),
                ],
                text=True,
            )
            ts_res = json.loads(node_out)
            assert ts_res["boxes"] == boxes_py
            assert {int(k): v for k, v in ts_res["statuses"].items()} == statuses_py

    # 6. Occluded / off keyframes are simply not solved: the remaining active frames expand onto the shot with
    #    `visible=0` everywhere else (no heuristic may reject an artist-authored active frame).
    active_frames = sorted(int(k) for k in boxes_py)
    _, vis_full, _ = expand_to_full_shot_with_visibility(shot_frames, active_frames, np.ones(len(active_frames)))
    vis_by_f = {f: int(vis_full[i]) for i, f in enumerate(shot_frames)}
    assert all(vis_by_f[f] == 0 for f in (1001, 1002, 1010, 1011, 1012, 1013, 1018, 1019, 1020))
    assert all(vis_by_f[f] == 1 for f in active_frames)

    # 7. Verify `select_people` auto-includes manual tracks alongside auto tracks AND supports manual-only selection
    manual_people_eval = build_manual_detected_people([manual_track], shot_frames)
    auto_people = [{"id": 1, "prominence": 50000.0, "boxes": {"1001": [10, 10, 110, 310], "1002": [10, 10, 110, 310], "1003": [10, 10, 110, 310]}}]
    chosen_both, _ = select_people(auto_people + manual_people_eval, SelectionSpec(mode="largest", least_frames=3))
    assert [p["id"] for p in chosen_both] == [1, 101]
    chosen_manual_only, _ = select_people(auto_people + manual_people_eval, SelectionSpec(mode="ids", ids="101", least_frames=3))
    assert [p["id"] for p in chosen_manual_only] == [101]

    # 9. Verify HTTP PATCH `/api/shots/{id}` persists & normalizes `manual_people`
    with tempfile.TemporaryDirectory() as td_api:
        root_api = Path(td_api)
        cfg = ConfigStore(root_api / "cfg.json")
        cfg.initialize(port=8765, password="ManualApi123")
        app = create_app(workspaces_dir=root_api / "ws", config_store=cfg)
        client = TestClient(app)
        sid = client.post("/api/shots", json={"name": "手动关键帧测试"}).json()["id"]
        r_patch = client.patch(f"/api/shots/{sid}", json={"manual_people": [manual_track]})
        assert r_patch.status_code == 200
        saved_mp = r_patch.json()["manual_people"]
        assert len(saved_mp) == 1 and saved_mp[0]["id"] == 101 and len(saved_mp[0]["keyframes"]) == 6


def test_v017_exr_all_five_grade_params_and_solver_srgb_rebake() -> None:
    """Verify v0.1.7 EXR (`ACEScg`) grading feeds ALL 5 sliders (`exposure`, `contrast`, `brightness`,
    `saturation`, `gamma`) into `prepare_solver_srgb_frames` (even when `exposure != 0` simultaneously with
    `contrast` & `saturation`!), and re-bakes `solver_srgb` immediately whenever any grade slider changes."""
    from shotspace.io.plate import ingest_sequence_dir, prepare_solver_srgb_frames, read_image_rgb, write_exr_half_rgb

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seq_dir = root / "exr_seq"
        # Chromatic HDR ACEScg test frame (dark shadow + mid-tone + color saturation + super-white >1.0 highlight)
        exr_frame = np.zeros((32, 48, 3), dtype=np.float32)
        exr_frame[:, :16] = [0.03, 0.02, 0.015]
        exr_frame[:, 16:32] = [0.24, 0.14, 0.08]
        exr_frame[:, 32:] = [1.40, 0.95, 0.65]
        for f in (1001, 1002):
            write_exr_half_rgb(seq_dir / f"crowd.{f:04d}.exr", exr_frame)

        p_info = ingest_sequence_dir(seq_dir, root / "plate", ocio=OcioSettings("auto", "aces_sdr"))
        assert p_info.resolved_colorspace == "ACEScg"
        srgb_dir = root / "plate" / "solver_srgb"

        def _bake_and_read(g: GradeParams) -> np.ndarray:
            fmap = prepare_solver_srgb_frames(p_info, g, srgb_dir)
            rgb, _, _ = read_image_rgb(fmap[1001])
            return np.asarray(rgb, dtype=np.int32)

        base = _bake_and_read(GradeParams())
        exp_only = _bake_and_read(GradeParams(exposure=1.4))
        assert not np.array_equal(exp_only, base) and exp_only.mean() > base.mean() + 15.0

        # Crucial regression check: with `exposure=1.4 != 0` on EXR, changing `contrast` or `saturation` MUST change solver PNGs!
        exp_contrast = _bake_and_read(GradeParams(exposure=1.4, contrast=1.55))
        assert np.abs(exp_contrast - exp_only).max() >= 15, "Contrast was ignored on EXR when exposure != 0!"

        exp_desat = _bake_and_read(GradeParams(exposure=1.4, contrast=1.55, saturation=0.0))
        assert np.all(exp_desat[..., 0] == exp_desat[..., 1]) and np.all(exp_desat[..., 1] == exp_desat[..., 2]), (
            "Saturation=0.0 was ignored on EXR when exposure != 0!"
        )

        exp_bright = _bake_and_read(GradeParams(exposure=1.4, brightness=0.12))
        assert exp_bright.mean() > exp_only.mean() + 8.0

        exp_gamma = _bake_and_read(GradeParams(exposure=1.4, gamma=1.6))
        assert not np.array_equal(exp_gamma, exp_only)


def test_a_stretched_box_is_repaired_before_the_network_sees_it() -> None:
    """v0.2.12 (`shotspace.solver.box_repair`): a frame whose box suddenly reaches several box heights further down
    (the outline took in a piece far from the person) reaches the network repaired - the box the frames around agree
    on, and the region the outline is to be cut to; the interface shows the repaired box; only that person is due
    for a new estimate after an upgrade, everybody else's results stay valid."""
    import json as _json
    import time
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        slots = list(wm.scheduler._slots.values())
        if not any(s.enabled for s in slots):
            print("  (skipped: this machine has no enabled GPU slot, the queue cannot run a job)")
            return
        fake = _StandInPeopleSolver()
        for s in slots:
            s.solver = fake
        seen: list[list[dict]] = []
        real_solve = fake.solve_people

        def recording_solve(plate, selected_people, **kw):
            seen.append([dict(p) for p in selected_people])
            return real_solve(plate, selected_people, **kw)

        fake.solve_people = recording_solve
        frames = list(range(1001, 1013))
        sid = client.post("/api/shots", json={"name": "stretched box"}).json()["id"]
        _upload_test_plate_and_camera(client, root, sid, frames)
        sw = wm.get_shot(sid)
        assert client.post(f"/api/shots/{sid}/detect", json={}).status_code == 200
        t0 = time.time()
        while (wm.scheduler.job_for_shot(sid) is not None or sw.read_meta().get("active_job_id")) and time.time() - t0 < 30:
            time.sleep(0.03)
        # the tracker's outline of person 2 takes in a speck at the bottom of the picture on frame 1006
        boxes_path = sw.root / "detect" / "boxes.json"
        det = _json.loads(boxes_path.read_text(encoding="utf-8"))
        p2 = next(p for p in det["people"] if int(p["id"]) == 2)
        clean_box = list(p2["boxes"]["1006"])
        stretched = [clean_box[0], clean_box[1], clean_box[2], clean_box[3] + 3.0 * (clean_box[3] - clean_box[1])]
        p2["boxes"]["1006"] = list(stretched)
        p2["raw_boxes"]["1006"] = list(stretched)
        boxes_path.write_text(_json.dumps(det), encoding="utf-8")
        meta = sw.read_meta()
        meta["detection"]["people"] = det["people"]
        sw.write_meta(meta)

        wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": "1,2"}}, wait=True)
        by_id = {int(p["id"]): p for p in seen[-1]}
        assert sorted(by_id) == [1, 2]
        assert by_id[2]["repaired"] and sorted(by_id[2]["repaired"]) == ["1006"], by_id[2].get("repaired")
        assert by_id[2]["boxes"]["1006"] == clean_box or np.allclose(by_id[2]["boxes"]["1006"], clean_box)
        assert by_id[2]["raw_boxes"]["1006"] == stretched                          # what the tracker said is kept
        assert all(by_id[2]["boxes"][k] == p2["boxes"][k] for k in p2["boxes"] if k != "1006")
        assert "repaired" not in by_id[1]                                            # person 1 exactly as the tracker gave them
        region = by_id[2]["repaired"]["1006"]
        h = clean_box[3] - clean_box[1]
        pw, ph = int(sw.read_meta()["plate"]["width"]), int(sw.read_meta()["plate"]["height"])
        expect = np.clip([clean_box[0] - 0.3 * h, clean_box[1] - 0.3 * h, clean_box[2] + 0.3 * h, clean_box[3] + 0.3 * h], 0.0, [pw, ph, pw, ph])
        assert np.allclose(region, expect, atol=0.05), (region, expect.tolist())          # grown, and never outside the picture
        # the interface draws the repaired box
        shown = client.get(f"/api/shots/{sid}").json()
        shown_p2 = next(p for p in shown["detection"]["people"] if int(p["id"]) == 2)
        assert np.allclose(shown_p2["boxes"]["1006"], clean_box)
        # nothing to do now: the results are of the repaired input
        fake.log.clear()
        wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": "1,2"}}, wait=True)
        assert fake.log == [], fake.log
        # a task solved before v0.2.12 (person 2 estimated from the stretched input): only person 2 is due again
        unrepaired = {k: v for k, v in by_id[2].items() if k != "repaired"}
        unrepaired["boxes"] = dict(p2["boxes"])
        sigs = dict(sw.read_meta()["cache_signatures"])
        meta_now = sw.read_meta()
        plate = wm._load_plate_info(sw)
        cam = wm._load_camera_from_entry(meta_now.get("camera"), meta_now.get("plate"))
        _cam, intr_sig, _ext = wm._camera_sigs(cam, plate, float(getattr(cam, "fps", 0.0) or 24.0))
        sigs["people_net"] = {**sigs["people_net"], "2": wm._person_net_sig(unrepaired, False, str(sigs.get("detect_run") or ""), intr_sig, True)}
        sw.update_meta(cache_signatures=sigs)
        assert wm._people_to_solve_again(sw) == [2]
        fake.log.clear()
        wm.enqueue_solve(sid, {"selection": {"mode": "ids", "ids": "1,2"}}, wait=True)
        assert fake.log == [("solve", [2])], fake.log


def test_each_computer_sees_only_its_own_tasks() -> None:
    """v0.2.12: a task belongs to the computer that created it (machine name + address).  Another computer neither
    sees it in its list nor opens, changes, computes or deletes it; a public task is seen by all and moved into one
    computer's list with a click; the admin page sees every task and can make one public."""
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = ConfigStore(root / "server_config.json")
        cfg.initialize(port=8765, password="testpassword")
        app = create_app(workspaces_dir=root / "ws", config_store=cfg)
        client = TestClient(app)
        wm = app.state.wm
        names = {"10.0.0.1": "WS-A", "10.0.0.9": "WS-A", "10.0.0.2": "WS-B"}   # WS-A gets another address later
        wm.machines.resolver = lambda ip: names.get(ip, "")
        A, B, C = {"X-Forwarded-For": "10.0.0.1"}, {"X-Forwarded-For": "10.0.0.2"}, {"X-Forwarded-For": "10.0.0.3"}
        ids = lambda who: [s["id"] for s in client.get("/api/shots", headers=who).json()["shots"]]  # noqa: E731

        ta = client.post("/api/shots", json={"name": "A 的镜头"}, headers=A).json()
        tb = client.post("/api/shots", json={"name": "B 的镜头"}, headers=B).json()
        tc = client.post("/api/shots", json={"name": "C 的镜头"}, headers=C).json()
        assert ta["owner"]["ip"] == "10.0.0.1" and ta["owner"]["name"] == "WS-A" and ta["owner_label"] == "WS-A (10.0.0.1)"
        assert tc["owner"] == {**tc["owner"], "ip": "10.0.0.3", "name": ""} and tc["owner_label"] == "10.0.0.3"
        assert ids(A) == [ta["id"]] and ids(B) == [tb["id"]] and ids(C) == [tc["id"]]
        listing = client.get("/api/shots", headers=A).json()
        assert listing["machine"] == {"ip": "10.0.0.1", "name": "WS-A"} and listing["shots"][0]["is_public"] is False

        # ---- another computer can do nothing with it
        sid = ta["id"]
        for method, path, body in (
            ("GET", f"/api/shots/{sid}", None), ("PATCH", f"/api/shots/{sid}", {"name": "x"}), ("DELETE", f"/api/shots/{sid}", None),
            ("POST", f"/api/shots/{sid}/solve", {}), ("POST", f"/api/shots/{sid}/detect", {}), ("POST", f"/api/shots/{sid}/cancel", None),
            ("GET", f"/api/shots/{sid}/thumbnail", None), ("GET", f"/api/shots/{sid}/view/manifest", None), ("GET", f"/api/shots/{sid}/events", None),
            ("GET", f"/api/shots/{sid}/download/usd_skel", None), ("POST", f"/api/shots/{sid}/claim", None), ("POST", f"/api/shots/{sid}/sequence/begin", None),
        ):
            r = client.request(method, path, json=body, headers=B)
            assert r.status_code == 403 and "另一台电脑" in r.text, (method, path, r.status_code, r.text[:120])
        assert client.get(f"/api/shots/{sid}", headers=A).status_code == 200 and client.get(f"/api/shots/{sid}", headers=A).json()["name"] == "A 的镜头"

        # ---- the same computer with another address: recognised by its name, its record follows
        assert ids({"X-Forwarded-For": "10.0.0.9"}) == [ta["id"]]
        assert client.get(f"/api/shots/{sid}", headers={"X-Forwarded-For": "10.0.0.9"}).status_code == 200
        assert wm.get_shot(sid).read_meta()["owner"]["ip"] == "10.0.0.9"
        assert ids(A) == [ta["id"]]                          # ...and still with the first address
        # a computer without a name is its address: another address is another computer
        assert ids({"X-Forwarded-For": "10.0.0.4"}) == []
        # a name that becomes known later completes the record
        names["10.0.0.3"] = "WS-C"
        wm.machines._cache.pop("10.0.0.3", None)
        assert client.get(f"/api/shots/{tc['id']}", headers=C).status_code == 200
        assert wm.get_shot(tc["id"]).read_meta()["owner"]["name"] == "WS-C"

        # ---- the admin page sees every task and can make one public
        token = client.post("/api/admin/login", json={"password": "testpassword"}).json()["token"]
        adm = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/admin/shots").status_code == 401
        everyone = {t["id"]: t for t in client.get("/api/admin/shots", headers=adm).json()["shots"]}
        assert set(everyone) == {ta["id"], tb["id"], tc["id"]} and everyone[tb["id"]]["owner_label"] == "WS-B (10.0.0.2)"
        assert client.post(f"/api/admin/shots/{sid}/unclassified").status_code == 401
        assert client.post(f"/api/admin/shots/{sid}/unclassified", headers=adm).status_code == 200
        assert ta["id"] in ids(A) and ta["id"] in ids(B) and ta["id"] in ids(C)
        shared = next(s for s in client.get("/api/shots", headers=B).json()["shots"] if s["id"] == sid)
        assert shared["is_public"] is True and shared["owner_label"] == "未分类"
        assert client.get(f"/api/shots/{sid}", headers=B).status_code == 200            # a public task opens anywhere
        r_del = client.delete(f"/api/shots/{sid}", headers=B)
        assert r_del.status_code == 400 and "移到我的任务" in r_del.text                 # ...but is not deleted by just anybody
        # ---- whose it is moves it into their list: gone from everybody else's
        r_claim = client.post(f"/api/shots/{sid}/claim", headers=B)
        assert r_claim.status_code == 200 and r_claim.json()["owner_label"] == "WS-B (10.0.0.2)" and r_claim.json()["is_public"] is False
        assert ids(A) == [] and sorted(ids(B)) == sorted([ta["id"], tb["id"]])
        assert client.get(f"/api/shots/{sid}", headers=A).status_code == 403
        assert client.delete(f"/api/shots/{sid}", headers=B).status_code == 200 and ids(B) == [tb["id"]]

        # ---- the admin page: how many tasks each computer has; who opened the service and has none
        overview = client.get("/api/admin/machines", headers=adm).json()["machines"]
        assert overview[0]["unclassified"] is True and overview[0]["task_count"] == 0
        by_name = {(m["name"] or m["ip"]): m for m in overview[1:]}
        assert by_name["WS-B"]["task_count"] == 1 and by_name["WS-C"]["task_count"] == 1 and by_name["WS-B"]["label"] == "WS-B (10.0.0.2)", by_name
        assert by_name["WS-A"]["task_count"] == 0 and by_name["WS-A"]["last_seen"], by_name     # A opened it, has none now
        assert by_name["WS-A"]["label"] == "WS-A (10.0.0.1)"         # shown with the address it came from last
        assert by_name["10.0.0.4"]["task_count"] == 0                # looked, never made a task
        assert len(by_name) == 4, sorted(by_name)                    # WS-A under two addresses is one computer
        # ---- the admin gives a task to a computer: from the list of computers, or by an address typed in
        wsa = by_name["WS-A"]
        assert client.post(f"/api/admin/shots/{tb['id']}/owner", json={"ip": wsa["ip"], "name": wsa["name"]}, headers=adm).status_code == 200
        assert ids(A) == [tb["id"]] and ids(B) == [] and wm.get_shot(tb["id"]).read_meta()["owner"]["how"] == "管理员分配"
        assert client.post(f"/api/admin/shots/{tb['id']}/owner", json={"ip": "10.0.0.77", "name": ""}, headers=adm).status_code == 200
        assert ids({"X-Forwarded-For": "10.0.0.77"}) == [tb["id"]] and ids(A) == []
        assert client.post(f"/api/admin/shots/{tb['id']}/owner", json={"ip": "", "name": ""}, headers=adm).status_code == 400
        assert client.post("/api/admin/shots/shot_does_not_exist/owner", json={"ip": "10.0.0.1"}, headers=adm).status_code == 404
        # ---- ...or deletes it, whoever it belongs to
        assert client.delete(f"/api/admin/shots/{tb['id']}").status_code == 401
        assert client.delete(f"/api/admin/shots/{tb['id']}", headers=adm).json()["deleted"] is True
        assert ids({"X-Forwarded-For": "10.0.0.77"}) == [] and tb["id"] not in {t["id"] for t in client.get("/api/admin/shots", headers=adm).json()["shots"]}
        # the computers that opened the service are remembered across a restart
        assert (root / "ws" / "_machines_seen.json").is_file()
        from shotspace.web.machines import MachineNames
        again = MachineNames(seen_file=root / "ws" / "_machines_seen.json")
        assert {"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.9"} <= set(again.seen)


def test_old_tasks_get_their_owner_from_the_records() -> None:
    """v0.2.12, tasks made before it (no owner): each is given to the computer the service's records name - the
    log (every request with its address) and the queue history (every job with the address that submitted it).
    What cannot be decided with certainty becomes a public task.  Decided once; nothing is lost."""
    import json as _json
    import os as _os
    from fastapi.testclient import TestClient
    from shotspace.server_config import ConfigStore
    from shotspace.web.app import create_app
    from shotspace.web.workspace import default_shot_meta

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws = root / "ws"

        def old_task(sid: str, created: str) -> None:
            meta = default_shot_meta(sid, name=f"旧任务 {sid}")
            meta.pop("owner")
            meta["created_at"] = created
            (ws / sid).mkdir(parents=True)
            (ws / sid / "shot.json").write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        old_task("shot_new_in_log", "2026-11-24T09:30:00+0800")     # created while the log was being written
        old_task("shot_before_log", "2026-11-20T10:00:00+0800")     # older than the log: only one computer worked on it
        old_task("shot_shared", "2026-11-20T11:00:00+0800")         # older than the log, two computers worked on it
        old_task("shot_only_jobs", "2026-11-19T11:00:00+0800")      # rolled out of the log, its jobs are in the history
        old_task("shot_no_record", "2026-11-18T11:00:00+0800")      # nothing anywhere
        (ws / "_queue_history.json").write_text(_json.dumps([
            {"id": "job_1", "shot_id": "shot_only_jobs", "client_ip": "10.0.0.7", "job_type": "solve", "status": "completed"},
            {"id": "job_2", "shot_id": "shot_only_jobs", "client_ip": "10.0.0.7", "job_type": "detect", "status": "completed"},
            {"id": "job_3", "shot_id": "shot_shared", "client_ip": "10.0.0.5", "job_type": "solve", "status": "completed"},
        ]), encoding="utf-8")
        log = root / "shotspace.log"
        (root / "shotspace.log.old").write_text(
            "==============================================================================\n"
            "[2026-11-24 08:00:00 +0800] Starting ShotSpace v0.2.11 | root=/x | listen=0.0.0.0:8795\n"
            'INFO:     10.0.0.5:5001 - "GET /api/shots/shot_shared HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.6:5002 - "POST /api/shots/shot_shared/solve HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.4:5003 - "POST /api/shots/shot_before_log/solve HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.8:5004 - "POST /api/shots/shot_before_log/solve HTTP/1.1" 400 Bad Request\n'   # refused: does not count
            'INFO:     10.0.0.8:5005 - "GET /api/shots/shot_before_log/thumbnail HTTP/1.1" 200 OK\n',          # only looked: does not count
            encoding="utf-8",
        )
        log.write_text(
            "[2026-11-24 09:00:00 +0800] Starting ShotSpace v0.2.11 | root=/x | listen=0.0.0.0:8795\n"
            'INFO:     10.0.0.2:5101 - "POST /api/shots HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.2:5102 - "GET /api/shots/shot_new_in_log/events HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.3:5103 - "POST /api/shots/shot_new_in_log/solve HTTP/1.1" 200 OK\n'
            'INFO:     10.0.0.2:5104 - "POST /api/shots/shot_new_in_log/sequence/begin HTTP/1.1" 200 OK\n',
            encoding="utf-8",
        )
        old_env = _os.environ.get("SHOTSPACE_LOG_FILE")
        _os.environ["SHOTSPACE_LOG_FILE"] = str(log)
        try:
            cfg = ConfigStore(root / "server_config.json")
            cfg.initialize(port=8765, password="testpassword")
            app = create_app(workspaces_dir=ws, config_store=cfg)
        finally:
            if old_env is None:
                _os.environ.pop("SHOTSPACE_LOG_FILE", None)
            else:
                _os.environ["SHOTSPACE_LOG_FILE"] = old_env
        wm = app.state.wm
        wm.machines.resolver = lambda ip: ""
        owner = lambda sid: wm.get_shot(sid).read_meta()["owner"]  # noqa: E731
        assert owner("shot_new_in_log")["ip"] == "10.0.0.2" and "创建" in owner("shot_new_in_log")["why"]
        assert owner("shot_before_log")["ip"] == "10.0.0.4" and "同一台电脑" in owner("shot_before_log")["why"]
        assert owner("shot_only_jobs")["ip"] == "10.0.0.7"
        assert owner("shot_shared").get("public") is True and "2 台电脑" in owner("shot_shared")["why"]
        assert owner("shot_no_record").get("public") is True and "没有找到" in owner("shot_no_record")["why"]
        report = (ws / "_task_owners_v0212.txt").read_text(encoding="utf-8")
        assert all(sid in report for sid in ("shot_new_in_log", "shot_before_log", "shot_shared", "shot_only_jobs", "shot_no_record"))
        # every old task can be reached: its own computer sees it, the undecided ones are public for everybody
        client = TestClient(app)
        ids = lambda ip: sorted(s["id"] for s in client.get("/api/shots", headers={"X-Forwarded-For": ip}).json()["shots"])  # noqa: E731
        assert ids("10.0.0.2") == ["shot_new_in_log", "shot_no_record", "shot_shared"]
        assert ids("10.0.0.4") == ["shot_before_log", "shot_no_record", "shot_shared"]
        assert ids("10.0.0.7") == ["shot_no_record", "shot_only_jobs", "shot_shared"]
        # decided once: a later start (with other records) changes nothing
        before = {sid: owner(sid) for sid in ("shot_new_in_log", "shot_before_log", "shot_shared", "shot_only_jobs", "shot_no_record")}
        log.write_text('INFO:     10.0.0.9:1 - "POST /api/shots/shot_shared/solve HTTP/1.1" 200 OK\n', encoding="utf-8")
        wm._shots.clear()
        wm._discover_existing()
        assert {sid: owner(sid) for sid in before} == before


def test_machine_name_from_the_windows_name_service() -> None:
    """The answer of a Windows computer to a NetBIOS name query is read correctly (a stand-in answers on a free
    port; the real query goes to port 137)."""
    import socket
    import struct
    import threading
    from shotspace.web.machines import _netbios_name

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def answer() -> None:
        data, addr = srv.recvfrom(1024)
        tid = data[:2]
        names = [("WORKGROUP", 0x00, 0x8400), ("ARTIST-PC-07", 0x00, 0x0400), ("ARTIST-PC-07", 0x20, 0x0400)]
        body = bytes([len(names)]) + b"".join(n.ljust(15).encode("ascii") + bytes([suffix]) + struct.pack(">H", fl) for n, suffix, fl in names)
        reply = tid + b"\x84\x00\x00\x00\x00\x01\x00\x00\x00\x00" + data[12:46] + b"\x00\x21\x00\x01" + b"\x00\x00\x00\x00" + struct.pack(">H", len(body) + 46) + body + b"\x00" * 46
        srv.sendto(reply, addr)

    t = threading.Thread(target=answer, daemon=True)
    t.start()
    assert _netbios_name("127.0.0.1", timeout=2.0, port=port) == "ARTIST-PC-07"
    t.join(2.0)
    srv.close()
    assert _netbios_name("127.0.0.1", timeout=0.2, port=port) == ""     # nobody answers: no name


if __name__ == "__main__":
    test_plate_grade_lut_vs_float()
    test_camera_space_and_multiformat_roundtrip()
    test_selection_rules()
    test_admin_auth_and_gpu_queue()
    test_ocio_exr_and_standard_camera_alignment()
    test_anti_flyaway_and_all_format_visibility_export()
    test_rigid_object_alignment_and_maya_parent_transform_export()
    test_admin_cache_dir_switch_and_protection()
    test_separate_scene_scan_abc_fbx_usd_import_and_calibration()
    test_first_run_startup_script_port_and_password_requirement()
    test_windows_client_upload_zip_gbk_and_download_compatibility()
    test_sequence_validation_interception_and_safe_staging()
    test_studio_meter_unit_and_multi_hierarchy_selection()
    test_camera_first_workflow_on_demand_exports_and_glibc_compat()
    test_export_choice_of_people_and_no_root_filter()
    test_exports_write_no_keys_while_a_person_stands_still()
    test_accumulating_people_solve()
    test_scan_is_used_on_request_and_both_results_are_kept()
    test_a_stretched_box_is_repaired_before_the_network_sees_it()
    test_each_computer_sees_only_its_own_tasks()
    test_old_tasks_get_their_owner_from_the_records()
    test_machine_name_from_the_windows_name_service()
    test_a_track_that_changes_person_becomes_several_people()
    test_data_collection_writes_the_table_of_the_day()
    test_v016_manual_keyframe_interpolation_states_and_crowd_dedup()
    test_v017_exr_all_five_grade_params_and_solver_srgb_rebake()
    import test_solver_v2

    test_solver_v2.run_all()
    import test_track_split

    test_track_split.run_all()
    # the box repair cleans outlines with OpenCV, which only the solver's Python has (that is where it runs)
    import subprocess as _sp
    from shotspace.config import get_paths as _gp

    _solver_py = _gp().sam3d_python
    if _solver_py.is_file():
        _sp.run([str(_solver_py), str(Path(__file__).with_name("test_box_repair.py"))], check=True,
                env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    else:
        print("  (box repair tests skipped: the solver's Python is not installed here)")
    print("All ShotSpace tests passed!")
