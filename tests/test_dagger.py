import json
from collections import Counter

import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin.dagger import FailureSnapshot, Harvester, Rehearsed, record_recovery, select
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.randomize import Randomization, SceneRandomizer
from nebius_physical_ai.twin.scene import SO101_XML, BlockPhysics, SceneLook, TwinLayout

LAYOUT = TwinLayout()
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


class StillPolicy:
    """Holds the arm where it is, so every rollout fails without touching anything; remembers the prompts it saw."""

    def __init__(self) -> None:
        self.prompts: set[str] = set()

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        self.prompts |= {o.task for o in observations}
        return np.stack([np.tile(o.joints, (8, 1)) for o in observations])


def snapshot(scene: int, p: float, failure: str) -> FailureSnapshot:
    objects = [DetectedObject("blue", 100.0, 100.0, 900.0), DetectedObject("red", 200.0, 100.0, 1600.0)]
    return FailureSnapshot(objects, "blue", "red", "t", SceneLook(), BlockPhysics(), [], [], failure, scene, p)


@pytest.fixture(scope="module")
def harvested() -> tuple[StillPolicy, list[FailureSnapshot], dict[str, int]]:
    policy, sampler = StillPolicy(), SceneSampler(LAYOUT, seed=3)
    randomizer = SceneRandomizer(LAYOUT, seed=3, ranges=Randomization(original_share=0.0), role_agnostic=True)
    snapshots, outcomes = Harvester(policy, randomizer, sampler, batch_size=4, max_steps=24, variants=2).harvest(3)
    return policy, snapshots, dict(outcomes)


class TestSelect:
    """Which harvested failures are worth recording."""

    def test_the_most_uncertain_scene_wins(self) -> None:
        assert select([snapshot(0, 0.1, "pushed"), snapshot(1, 0.5, "pushed")], budget=1)[0].scene == 1

    @pytest.mark.parametrize(("p", "kept"), [(0.1, True), (0.9, True), (0.0, False), (1.0, False)])
    def test_the_band_is_inclusive(self, p: float, kept: bool) -> None:
        assert bool(select([snapshot(0, p, "pushed")], budget=1, band=(0.1, 0.9))) is kept

    def test_a_scene_is_capped_keeping_its_rarer_failure_types(self) -> None:
        pool = [snapshot(0, 0.5, "no_contact"), snapshot(0, 0.5, "no_contact"), snapshot(0, 0.5, "dropped")]
        pool += [snapshot(1, 0.5, "no_contact")]
        chosen = select(pool, budget=10, per_scene=2)
        assert sorted((s.scene, s.failure) for s in chosen) == [(0, "dropped"), (0, "no_contact"), (1, "no_contact")]

    def test_the_budget_is_shared_across_failure_types(self) -> None:
        pool = [snapshot(i, 0.5, "no_contact") for i in range(20)] + [
            snapshot(20 + i, 0.5, "dropped") for i in range(3)
        ]
        assert Counter(s.failure for s in select(pool, budget=6)) == {"no_contact": 3, "dropped": 3}


class ScriptedHarvester(Harvester):
    """Scene k succeeds in its first k of `variants` rollouts, without a simulator."""

    def __init__(self, variants: int) -> None:
        self.variants, self.batch_size = variants, 3

    def draw(self, scene: int) -> list[Rehearsed]:
        return [Rehearsed(scene, f"v{i}", None, None, None) for i in range(self.variants)]  # type: ignore[arg-type]

    def roll(self, batch: list[Rehearsed]) -> list[tuple[Rehearsed, str, FailureSnapshot | None]]:
        failed = [int(r.prompt[1:]) >= r.scene for r in batch]
        return [
            (r, "pushed" if f else "", snapshot(r.scene, 0.0, "pushed") if f else None) for r, f in zip(batch, failed)
        ]


def test_each_failure_carries_its_scenes_success_rate() -> None:
    snapshots, outcomes = ScriptedHarvester(variants=4).harvest(3)
    assert outcomes == {"success": 3, "pushed": 9}  # scenes 0, 1, 2 succeed 0, 1, 2 times of 4
    assert sorted({(s.scene, s.scene_p) for s in snapshots}) == [(0, 0.0), (1, 0.25), (2, 0.5)]


@needs_model
class TestHarvest:
    def test_every_failure_is_snapshotted_with_its_own_scene_and_prompt(self, harvested: tuple) -> None:
        policy, snapshots, outcomes = harvested
        assert outcomes == {"no_contact": 6} and len(snapshots) == 6  # 3 scenes x 2 variants
        assert policy.prompts == {s.prompt for s in snapshots} and len(policy.prompts) == 3  # one prompt per scene
        assert sorted(s.scene for s in snapshots) == [0, 0, 1, 1, 2, 2] and {s.scene_p for s in snapshots} == {0.0}
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
