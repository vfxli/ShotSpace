"""`shotspace.solver.box_repair`: a frame whose outline took in a piece far from the person is repaired - and
nothing else is.  Synthetic boxes and outlines; numpy (+ OpenCV for the outline)."""

from __future__ import annotations

import numpy as np

from shotspace.solver.box_repair import RepairConfig, clean_outline, stretched_frames, with_repairs


def _walk(n: int = 60, x0: float = 400.0, y0: float = 300.0, w: float = 60.0, h: float = 180.0, vx: float = 3.0, jitter: float = 2.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        x = x0 + vx * i + rng.normal(0, jitter)
        y = y0 + rng.normal(0, jitter)
        out[str(1001 + i)] = [x, y, x + w + rng.normal(0, jitter), y + h + rng.normal(0, jitter)]
    return out


def test_a_normal_track_is_left_alone() -> None:
    for seed in range(10):
        assert stretched_frames(_walk(seed=seed), (1920, 1080)) == {}, seed
    # walking towards the camera: the box grows steadily (3 % a frame) - not a stretch
    boxes = {}
    for i in range(60):
        s = 1.03 ** i
        boxes[str(1001 + i)] = [500 - 20 * s, 400 - 60 * s, 500 + 20 * s, 400 + 60 * s]
    assert stretched_frames(boxes, (1920, 1080)) == {}
    # somebody comes out from behind a car: from one frame on the box is twice as tall and STAYS so
    boxes = _walk(seed=3)
    for i in range(30, 60):
        b = boxes[str(1001 + i)]
        boxes[str(1001 + i)] = [b[0], b[1], b[2], b[1] + 2.0 * (b[3] - b[1])]
    found = stretched_frames(boxes, (1920, 1080))
    assert len(found) <= 1, sorted(found)     # at most the very first frame of the change, never the lasting part
    # the box only MOVES for a frame (the track on a neighbour for a moment): not a stretch either
    boxes = _walk(seed=4)
    b = boxes["1030"]
    boxes["1030"] = [b[0] + 200.0, b[1], b[2] + 200.0, b[3]]
    assert stretched_frames(boxes, (1920, 1080)) == {}
    # too short to judge
    assert stretched_frames({str(1001 + i): [0, 0, 10, 30] for i in range(4)}) == {}


def test_a_stretched_frame_is_found_and_its_box_is_the_agreed_one() -> None:
    boxes = _walk(seed=1)
    for f, extra in (("1020", 2.0), ("1040", 0.7)):          # the bottom edge several / 0.7 box heights further down
        b = boxes[f]
        boxes[f] = [b[0], b[1], b[2], b[3] + extra * (b[3] - b[1])]
    b = boxes["1050"]
    boxes["1050"] = [b[0] - 400.0, b[1], b[2], b[3]]          # and once far to the side
    found = stretched_frames(boxes, (1920, 1080))
    assert sorted(found) == [1020, 1040, 1050], sorted(found)
    for f, fix in found.items():
        agreed, region = np.asarray(fix["box"]), np.asarray(fix["region"])
        h = agreed[3] - agreed[1]
        assert abs(h - 180.0) < 8.0 and np.all(region[:2] < agreed[:2]) and np.all(region[2:] > agreed[2:])
        assert np.allclose(region, agreed + 0.3 * h * np.array([-1, -1, 1, 1]), atol=0.05)
    # a small stretch (0.4 box heights) stays below the bar
    boxes = _walk(seed=2)
    b = boxes["1030"]
    boxes["1030"] = [b[0], b[1], b[2], b[3] + 0.4 * (b[3] - b[1])]
    assert stretched_frames(boxes, (1920, 1080)) == {}
    # the region never leaves the picture
    boxes = _walk(x0=5.0, vx=0.0, seed=5)
    b = boxes["1030"]
    boxes["1030"] = [b[0], b[1], b[2], b[3] + 3.0 * (b[3] - b[1])]
    reg = stretched_frames(boxes, (1920, 1080))[1030]["region"]
    assert reg[0] >= 0.0 and reg[3] <= 1080.0
    assert RepairConfig().reach == 0.5 and RepairConfig().margin == 0.3


def test_the_repaired_person_and_what_stays_untouched() -> None:
    raw = _walk(seed=6)
    b = raw["1025"]
    raw["1025"] = [b[0], b[1], b[2], 1080.0]                   # down to the bottom of the picture
    person = {"id": 7, "boxes": dict(raw), "raw_boxes": dict(raw), "scores": {k: 0.9 for k in raw}}
    fixed = with_repairs(person, (1920, 1080))
    assert fixed is not person and sorted(fixed["repaired"]) == ["1025"]
    assert fixed["boxes"]["1025"] != raw["1025"] and fixed["raw_boxes"]["1025"] == raw["1025"]
    assert all(fixed["boxes"][k] == raw[k] for k in raw if k != "1025")   # every other frame as the tracker gave it
    # the same answer every time, from the tracker's own boxes; a repaired person is not repaired twice
    assert with_repairs(dict(person), (1920, 1080)) == fixed and with_repairs(fixed, (1920, 1080)) is fixed
    # a person without such a frame is the very same object (their signature stays as it was)
    clean = {"id": 8, "boxes": _walk(seed=7), "raw_boxes": _walk(seed=7)}
    assert with_repairs(clean, (1920, 1080)) is clean
    # manually boxed people are never touched
    manual = {**person, "is_manual": True}
    assert with_repairs(manual, (1920, 1080)) is manual


def test_the_outline_loses_only_what_lies_away_from_the_person() -> None:
    H, W = 400, 600
    body = np.zeros((H, W), np.uint8)
    body[100:280, 200:260] = 1                                  # the person
    region = [182.0, 46.0, 278.0, 334.0]                         # their agreed box (200,100,260,280) grown by 0.3 h
    # a speck far below at the edge of the picture: dropped, the box is the person's again
    m = body.copy()
    m[392:398, 225:235] = 1
    out, box = clean_outline(m, region)
    assert np.array_equal(out, body) and box.tolist() == [200.0, 100.0, 260.0, 280.0]
    # a piece on somebody to the side: dropped; a piece of the person outside the agreed box but inside the region
    # (an arm): kept
    m = body.copy()
    m[150:200, 450:500] = 1
    m[120:140, 260:270] = 1
    out, box = clean_outline(m, region)
    assert out[150:200, 450:500].sum() == 0 and out[120:140, 260:270].sum() == 200 and box.tolist() == [200.0, 100.0, 270.0, 280.0]
    # a leak that hangs on to the person (the floor below the feet): cut at the region's edge
    m = body.copy()
    m[280:400, 215:245] = 1
    out, box = clean_outline(m, region)
    assert out[334:, :].sum() == 0 and out[280:334, 215:245].sum() == 54 * 30 and box[3] == 334.0
    # only the rectangle the outline lives in is looked at - same answer
    m = body.copy()
    m[392:398, 225:235] = 1
    out2, box2 = clean_outline(m, region, rect=(200, 100, 260, 398))
    assert np.array_equal(out2, body) and box2.tolist() == [200.0, 100.0, 260.0, 280.0]
    # nothing of the outline where the person is: None (the frame is left out)
    m = np.zeros((H, W), np.uint8)
    m[350:390, 450:520] = 1
    assert clean_outline(m, region) is None


def run_all() -> None:
    test_a_normal_track_is_left_alone()
    test_a_stretched_frame_is_found_and_its_box_is_the_agreed_one()
    test_the_repaired_person_and_what_stays_untouched()
    test_the_outline_loses_only_what_lies_away_from_the_person()
    print("box repair tests passed")


if __name__ == "__main__":
    run_all()
