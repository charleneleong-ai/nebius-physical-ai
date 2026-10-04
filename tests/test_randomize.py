import numpy as np
import pytest

from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.randomize import Randomization, SceneRandomizer
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


def test_cameras_cover_every_side_of_the_mat(randomizer: SceneRandomizer) -> None:
    centre = np.array(LAYOUT.mat_to_world(LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm / 2))
    azimuths = [np.arctan2(*(np.array(randomizer.camera()["pos"][:2]) - centre)[::-1]) for _ in range(400)]
    assert np.histogram(azimuths, bins=8, range=(-np.pi, np.pi))[0].min() > 20


def test_objects_keep_the_task_pair_and_space_out_distractors(randomizer: SceneRandomizer) -> None:
    sampler = SceneSampler(LAYOUT, seed=3)
    for _ in range(100):
        objects = randomizer.objects(sampler)
        pick, place, *distractors = objects
        assert (pick.label, place.label) == ("blue", "red")
        assert RANGES.pick_width_mm[0] <= pick.width_mm <= RANGES.pick_width_mm[1]
        assert RANGES.place_width_mm[0] <= place.width_mm <= RANGES.place_width_mm[1]
        assert len(distractors) <= RANGES.distractors[1]
        for d in distractors:
            assert d.label not in ("blue", "red")
            assert all(
                np.hypot(d.x_mm - o.x_mm, d.y_mm - o.y_mm) > RANGES.distractor_gap_mm for o in objects if o is not d
            )


def test_colours_stay_close_to_the_label(randomizer: SceneRandomizer) -> None:
    blues = np.array([randomizer.colour("blue") for _ in range(200)])
    assert (blues[:, 2] > blues[:, 0]).all() and (blues[:, 2] > blues[:, 1]).all()  # still unmistakably blue
    assert blues.std(axis=0).max() > 0.02  # but not one fixed shade
