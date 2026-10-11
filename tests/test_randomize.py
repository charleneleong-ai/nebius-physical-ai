import numpy as np
import pytest

from nebius_physical_ai.capture.camera import CAMERA_SOURCES, DEFAULT_CAMERA_MIX, CameraPose, parse_camera_mix
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


def test_cameras_spread_over_the_whole_arc_including_behind_the_arm(randomizer: SceneRandomizer) -> None:
    centre = np.array(LAYOUT.mat_to_world(LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm / 2))
    offsets = [np.array(randomizer.camera()["pos"][:2]) - centre for _ in range(400)]
    azimuths = np.degrees([np.arctan2(o[1], o[0]) for o in offsets])  # 0 = across the mat from the robot
    assert np.abs(azimuths).max() > 170  # the real camera sits behind the arm, at about 180 degrees
    assert np.histogram(azimuths, bins=6, range=(-180, 180))[0].min() > 30


POSE = CameraPose(  # the camera calibrated from the mat clip: 76 degrees elevation, behind the arm
    position_mm=(176.2, 235.2, 418.6),
    forward=(0.00585, -0.23954, -0.97087),
    up=(0.00438, -0.97087, 0.23956),
    hfov_deg=69.0,
    image_size=(720, 1280),
    reprojection_px=1.3,
)
NO_JITTER = {"pose_position_mm": 0.0, "pose_rotation_deg": 0.0, "pose_hfov_deg": 0.0}


def posed(mix: tuple[float, float, float] = (1.0, 0.0, 0.0), seed: int = 42, **ranges: float) -> SceneRandomizer:
    return SceneRandomizer(LAYOUT, seed=seed, ranges=Randomization(**ranges), camera_pose=POSE, camera_mix=mix)


def forward_of(spec: dict) -> np.ndarray:
    right, up = np.array(spec["xyaxes"][:3]), np.array(spec["xyaxes"][3:])
    return -np.cross(right, up)  # MuJoCo cameras look down -z, z = x × y


class TestPoseCameras:
    def test_without_jitter_the_camera_is_exactly_the_layouts_own_view_of_the_pose(self) -> None:
        camera, expected = posed(**NO_JITTER).camera(), TwinLayout(front_camera=POSE).front_camera_spec()
        np.testing.assert_allclose(camera["pos"], expected["pos"], atol=1e-9)
        np.testing.assert_allclose(camera["xyaxes"], expected["xyaxes"], atol=1e-9)
        assert camera["fovy"] == pytest.approx(expected["fovy"])

    def test_jitter_stays_inside_the_design_specs_ranges(self) -> None:
        base = LAYOUT.camera_spec(POSE)
        randomizer, positions = posed(), []
        for _ in range(300):
            camera = randomizer.camera()
            positions.append(np.array(camera["pos"]))
            assert np.abs(np.array(camera["pos"]) - base["pos"]).max() <= 0.030 + 1e-9  # 3 cm on every axis
            cosine = forward_of(camera) @ forward_of(base) / np.linalg.norm(forward_of(camera))
            assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) <= 5 * np.sqrt(2) + 0.2  # yaw and pitch of 5 degrees
            assert abs(camera["fovy"] - base["fovy"]) <= 5.0  # an HFOV change of 5 degrees
            right, up = np.array(camera["xyaxes"][:3]), np.array(camera["xyaxes"][3:])
            assert abs(right @ up) < 1e-9 and up[2] > 0  # still a proper, upright camera
        assert np.std(positions, axis=0).min() > 0.01  # and it really moves, on every axis

    def test_the_same_seed_gives_the_same_cameras(self) -> None:
        a, b = posed((0.6, 0.3, 0.1), seed=42), posed((0.6, 0.3, 0.1), seed=42)
        assert [a.camera() for _ in range(40)] == [b.camera() for _ in range(40)]

    def test_the_mixture_shares_hold_over_many_draws(self) -> None:
        randomizer = posed(DEFAULT_CAMERA_MIX)
        for _ in range(2000):
            randomizer.camera()
        assert sum(randomizer.camera_counts.values()) == 2000
        for source, expected in zip(CAMERA_SOURCES, DEFAULT_CAMERA_MIX):
            assert abs(randomizer.camera_counts[source] / 2000 - expected) < 0.04

    def test_each_source_gives_its_own_kind_of_view(self) -> None:
        base = np.array(LAYOUT.camera_spec(POSE)["pos"])
        near_pose = [np.linalg.norm(np.array(posed((1, 0, 0)).camera()["pos"]) - base) for _ in range(30)]
        arc = posed((0, 1, 0))
        far_arc = [np.linalg.norm(np.array(arc.camera()["pos"]) - base) for _ in range(60)]
        assert max(near_pose) < 0.06 and np.median(far_arc) > 0.1
        assert posed((0, 0, 1)).camera() == LAYOUT.top_down_camera_spec()

    def test_the_source_of_the_latest_camera_is_recorded(self) -> None:
        randomizer = posed((0, 0, 1))
        randomizer.camera()
        assert randomizer.last_camera_source == "top_down" and randomizer.camera_counts == {"top_down": 1}

    def test_without_a_pose_the_mix_is_ignored_and_every_camera_is_an_arc(self) -> None:
        randomizer = SceneRandomizer(LAYOUT, seed=1, camera_mix=(0.0, 0.0, 1.0))
        for _ in range(50):
            randomizer.camera()
        assert randomizer.camera_counts == {"arc": 50}


class TestParseCameraMix:
    def test_three_shares_summing_to_one(self) -> None:
        assert parse_camera_mix("0.3,0.6,0.1") == (0.3, 0.6, 0.1) == DEFAULT_CAMERA_MIX
        assert parse_camera_mix(" 1, 0, 0 ") == (1.0, 0.0, 0.0)

    @pytest.mark.parametrize("text", ["0.5,0.5", "0.6,0.3,0.2", "0.5,0.4,0.05", "a,b,c", "-0.1,0.6,0.5", ""])
    def test_anything_else_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError, match="camera mix"):
            parse_camera_mix(text)


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
