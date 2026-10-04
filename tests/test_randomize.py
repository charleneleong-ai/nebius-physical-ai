import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.randomize import (
    TASK_LABELS,
    Randomization,
    SceneRandomizer,
    outline,
    radius,
    segment_distance,
)
from nebius_physical_ai.twin.scene import TwinLayout

LAYOUT = TwinLayout()
RANGES = Randomization()


@pytest.fixture
def randomizer() -> SceneRandomizer:
    return SceneRandomizer(LAYOUT, seed=7)


def test_cameras_aim_at_the_mat_from_within_range(randomizer: SceneRandomizer) -> None:
    centre = np.array([*LAYOUT.mat_to_world(LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm / 2), 0.0])
    for _ in range(200):
        cam = randomizer.camera()
        pos, right, up = np.array(cam["pos"]), np.array(cam["xyaxes"][:3]), np.array(cam["xyaxes"][3:])
        forward = -np.cross(right, up)  # MuJoCo cameras look down -z, z = x × y
        ray_to_mat = pos + forward * (-pos[2] / forward[2])  # where the optical axis meets the table
        assert np.linalg.norm(ray_to_mat - centre) < 0.2
        assert RANGES.elevation_deg[0] - 1e-6 <= np.degrees(np.arcsin(pos[2] / np.linalg.norm(pos - ray_to_mat)))
        assert up[2] > 0 and abs(right @ up) < 1e-9  # image up is world-up-ish, axes orthogonal
        assert RANGES.fovy_deg[0] <= cam["fovy"] <= RANGES.fovy_deg[1]


def test_cameras_spread_over_the_arc_but_never_sit_behind_the_arm(randomizer: SceneRandomizer) -> None:
    centre = np.array(LAYOUT.mat_to_world(LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm / 2))
    offsets = [np.array(randomizer.camera()["pos"][:2]) - centre for _ in range(400)]
    azimuths = np.degrees([np.arctan2(o[1], o[0]) for o in offsets])  # 0 = across the mat from the robot
    assert np.abs(azimuths).max() <= RANGES.azimuth_deg[1] + 15  # aim jitter blurs the edge a little
    assert np.histogram(azimuths, bins=6, range=RANGES.azimuth_deg)[0].min() > 30


def test_objects_keep_the_task_pair_and_leave_room_for_the_arm(randomizer: SceneRandomizer) -> None:
    sampler, layout = SceneSampler(LAYOUT, seed=3), LAYOUT
    base = np.array([layout.mat.width_mm / 2, layout.mat.height_mm + layout.base_offset_mm])
    seen = 0
    for _ in range(150):
        objects = randomizer.objects(sampler)
        pick, place, *distractors = objects
        assert (pick.label, place.label) == ("blue", "red")
        assert max(pick.length_mm, pick.width_mm) <= RANGES.pick_length_mm + 1e-6
        assert pick.height_mm + place.height_mm <= RANGES.pick_height_mm[1] + RANGES.place_height_mm[1]
        a, b = np.array([pick.x_mm, pick.y_mm]), np.array([place.x_mm, place.y_mm])
        for d in distractors:
            seen += 1
            p = np.array([d.x_mm, d.y_mm])
            assert d.label not in TASK_LABELS
            edge = outline(d, p)
            for o in objects[:2]:  # footprints never overlap the task blocks at spawn
                assert np.hypot(*(edge - [o.x_mm, o.y_mm]).T).min() > radius(o) + RANGES.clearance_mm
            for start, end in [(base, a), (base, b), (a, b)]:  # and stay out of the arm's way
                assert segment_distance(edge, start, end).min() > RANGES.corridor_mm
    assert seen > 50  # the constraints still leave room for clutter


def test_colours_stay_close_to_the_label(randomizer: SceneRandomizer) -> None:
    blues = np.array([randomizer.colour("blue") for _ in range(200)])
    assert (blues[:, 2] > blues[:, 0]).all() and (blues[:, 2] > blues[:, 1]).all()  # still unmistakably blue
    assert blues.std(axis=0).max() > 0.02  # but not one fixed shade


def test_real_objects_appear_as_distractors_with_their_measured_size() -> None:
    real = DetectedObject("glasses_case", 0.0, 0.0, 8100.0, length_mm=140.0, width_mm=58.0, height_mm=26.0)
    clash = DetectedObject("red", 0.0, 0.0, 900.0)  # would collide with the task's red block: dropped
    ranges = Randomization(distractors=(2, 2))
    randomizer = SceneRandomizer(LAYOUT, seed=1, ranges=ranges, real_objects=[real, clash])
    sampler = SceneSampler(LAYOUT, seed=1)
    placed = [d for _ in range(60) for d in randomizer.objects(sampler)[2:] if d.label == "glasses_case"]
    assert len(placed) > 15  # about half of the distractors that fit
    assert randomizer.real_objects == [real]
    assert all(126 <= d.length_mm <= 154 and 23 <= d.height_mm <= 29 for d in placed)  # measured size +-10%
