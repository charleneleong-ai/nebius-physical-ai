import json

import numpy as np
import pytest

from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin.dagger import FailureSnapshot, Harvester, record_recovery
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.randomize import Randomization, SceneRandomizer
from nebius_physical_ai.twin.scene import SO101_XML, TwinLayout

LAYOUT = TwinLayout()
pytestmark = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


class StillPolicy:
    """Holds the arm where it is, so every rollout fails without touching anything; remembers the prompts it saw."""

    def __init__(self) -> None:
        self.prompts: set[str] = set()

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        self.prompts |= {o.task for o in observations}
        return np.stack([np.tile(o.joints, (8, 1)) for o in observations])


@pytest.fixture(scope="module")
def harvested() -> tuple[StillPolicy, list[FailureSnapshot], dict[str, int]]:
    policy, sampler = StillPolicy(), SceneSampler(LAYOUT, seed=3)
    randomizer = SceneRandomizer(LAYOUT, seed=3, ranges=Randomization(original_share=0.0), role_agnostic=True)
    snapshots, outcomes = Harvester(policy, randomizer, sampler, batch_size=2, max_steps=24).harvest(3)
    return policy, snapshots, dict(outcomes)


class TestHarvest:
    def test_every_failure_is_snapshotted_with_its_own_scene_and_prompt(self, harvested: tuple) -> None:
        policy, snapshots, outcomes = harvested
        assert outcomes == {"no_contact": 3} and len(snapshots) == 3
        assert policy.prompts == {s.prompt for s in snapshots} and len(policy.prompts) == 3  # one prompt per twin
        assert all(s.failure == "no_contact" and s.prompt.startswith(f"Pick {s.pick.split('_')[0]}") for s in snapshots)

    def test_a_snapshot_survives_json_and_restores_the_exact_state(self, harvested: tuple) -> None:
        snapshot = harvested[1][0]
        loaded = FailureSnapshot.from_record(json.loads(json.dumps(snapshot.record())))
        assert loaded == snapshot
        with loaded.restore(LAYOUT) as twin:
            np.testing.assert_allclose(twin.data.qpos, snapshot.qpos, atol=1e-6)

    def test_the_expert_recovers_from_where_the_policy_stopped(self, harvested: tuple) -> None:
        recovered = [(s, frames) for s in harvested[1] if (frames := record_recovery(s, LAYOUT, (24, 32)))]
        assert recovered  # an untouched scene is the easiest recovery there is
        snapshot, frames = recovered[0]
        with snapshot.restore(LAYOUT) as twin:
            np.testing.assert_allclose(frames[0].obs.joints, twin.to_lerobot(twin.arm_qpos()), atol=1e-4)
        assert frames[0].obs.task == snapshot.prompt
