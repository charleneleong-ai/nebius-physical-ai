from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from nebius_physical_ai.twin.rl_env import GAMMA, POLICY_HZ, RESIDUAL_COST, PickPlaceEnv
from nebius_physical_ai.twin.rollout import CONTROL_HZ
from nebius_physical_ai.twin.scene import SO101_XML

pytestmark = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


@pytest.fixture
def residual_env() -> Iterator[PickPlaceEnv]:
    env = PickPlaceEnv(seed=5, residual=True)
    env.reset()
    yield env
    env.close()


def test_spaces_match_observation(residual_env: PickPlaceEnv) -> None:
    obs, *_ = residual_env.step(np.zeros(residual_env.action_space.shape))
    assert obs.shape == residual_env.observation_space.shape


def test_zero_residual_follows_the_expert_plan(residual_env: PickPlaceEnv) -> None:
    """With no correction the target is the scripted expert's own waypoint, so the zero policy is the expert."""
    for _ in range(5):
        residual_env.step(np.zeros(residual_env.action_space.shape))
        lo, hi = residual_env.limits.T
        np.testing.assert_allclose(residual_env.target, np.clip(residual_env.plan[residual_env.tick - 1], lo, hi))


def test_correction_costs_reward(residual_env: PickPlaceEnv) -> None:
    """The residual cost is charged on top of the shaping term, in proportion to the squared correction."""
    action = np.full(residual_env.action_space.shape, 0.5)
    before = residual_env.last_potential
    _, reward, terminated, *_ = residual_env.step(action)
    assert not terminated
    shaping = GAMMA * residual_env.last_potential - before
    assert reward == pytest.approx(shaping - RESIDUAL_COST * 0.25)


def test_standing_still_earns_nothing() -> None:
    """Potential-based shaping pays progress, not position: holding a pose returns at most zero over an episode."""
    env = PickPlaceEnv(seed=5)
    env.reset()
    rewards = [env.step(np.zeros(env.action_space.shape))[1] for _ in range(20)]
    env.close()
    assert -1.0 < sum(rewards) <= 0.0


def test_on_tick_sees_every_commanded_target(residual_env: PickPlaceEnv) -> None:
    """Recording hooks in per control tick, so a recovery has one frame for each target the arm was driven to."""
    seen: list[np.ndarray] = []
    residual_env.on_tick = seen.append
    for _ in range(3):
        residual_env.step(np.zeros(residual_env.action_space.shape))
    assert len(seen) == 3 * (CONTROL_HZ // POLICY_HZ)
    np.testing.assert_allclose(seen[-1], residual_env.target)
