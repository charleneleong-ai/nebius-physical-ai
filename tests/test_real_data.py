import numpy as np
import pytest

from nebius_physical_ai.real_data import (
    armnet_episodes,
    closing_frames,
    fit_frame,
    interpolate,
    resample_positions,
    to_twin_units,
)
from nebius_physical_ai.twin.scene import lerobot_to_qpos

LIMITS = np.array([[-1.9, 1.9], [-1.7, 1.7], [-1.7, 1.7], [-1.7, 1.7], [-2.7, 2.8], [-0.2, 1.7]])


EPISODES = {
    "episode_index": [0, 1, 2, 3, 4],
    "policy_type": [float("nan"), "teleoperated", "act", "pi0.5", "act"],  # pandas reads a missing type as NaN
    "success_class": ["successful", "successful", "successful", "failure", "suboptimal"],
}


class TestEpisodes:
    @pytest.mark.parametrize(("rollouts", "expected"), [(False, [0, 1]), (True, [0, 1, 2])])
    def test_teleop_plus_successful_rollouts_only(self, rollouts: bool, expected: list[int]):
        assert armnet_episodes(EPISODES, rollouts) == expected


class TestUnits:
    def test_clips_overshoot_and_keeps_the_rest(self):
        joints = np.array([[-103.0, 50.0, 0.0, 99.0, 155.0, -1.0], [10.0, -20.0, 30.0, -40.0, 50.0, 60.0]])
        out = to_twin_units(joints)
        np.testing.assert_array_equal(out[0], [-100, 50, 0, 99, 100, 0])
        np.testing.assert_array_equal(out[1], joints[1])

    def test_lerobot_to_qpos_spans_the_limits_and_broadcasts(self):
        joints = np.array([[-100, -100, -100, -100, -100, 0], [100, 100, 100, 100, 100, 100], [0, 0, 0, 0, 0, 50]])
        qpos = lerobot_to_qpos(joints, LIMITS)
        np.testing.assert_allclose(qpos[0], LIMITS[:, 0])
        np.testing.assert_allclose(qpos[1], LIMITS[:, 1])
        np.testing.assert_allclose(qpos[2], LIMITS.mean(axis=1))


class TestResample:
    def test_20_to_30_fps_keeps_duration_and_endpoints(self):
        positions = resample_positions(21, 20, 30)  # one second at 20 fps
        assert len(positions) == 31 and positions[0] == 0 and positions[-1] == 20

    def test_interpolation_is_linear_between_frames(self):
        values = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
        np.testing.assert_allclose(
            interpolate(values, np.array([0.0, 0.5, 1.5, 2.0])), [[0, 10], [1, 15], [3, 30], [4, 40]]
        )


class TestFrames:
    @pytest.mark.parametrize("hw", [(576, 1024), (720, 1280), (480, 640), (600, 600)])
    def test_any_aspect_becomes_the_twin_frame(self, hw: tuple[int, int]):
        assert fit_frame(np.zeros((*hw, 3), np.uint8)).shape == (480, 640, 3)

    def test_wide_frames_lose_their_sides_not_their_middle(self):
        image = np.zeros((576, 1024, 3), np.uint8)
        image[:, :128] = 255  # outside the centred 4:3 crop (768 wide)
        image[:, 500:524] = 100
        out = fit_frame(image)
        assert out[:, :40].max() == 0 and out[:, 300:340].max() == 100


class TestGraspOnsets:
    def test_finds_each_close_but_not_opens(self):
        gripper = np.array([60, 60, 55, 10, 5, 5, 40, 60, 50, 8, 8], dtype=float)
        np.testing.assert_array_equal(closing_frames(gripper), [3, 9])
