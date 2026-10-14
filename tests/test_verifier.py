from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.policy.verifier import Verifier
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.rollout import BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout


def test_verifier_picks_the_top_score_and_survives_a_round_trip(tmp_path: Path) -> None:
    verifier = Verifier(width=4)
    with torch.no_grad():  # score = the first feature, so the pick is known
        for layer in (0, 2):
            verifier.net[layer].weight.copy_(
                torch.eye(verifier.net[layer].out_features, verifier.net[layer].in_features)
            )
            verifier.net[layer].bias.zero_()
        verifier.net[4].weight.zero_()
        verifier.net[4].weight[0, 0] = 1.0
    x = np.zeros((2, 3, 4), dtype=np.float32)
    x[0, 2, 0], x[1, 1, 0] = 5.0, 3.0
    verifier.expert = True
    verifier.save(tmp_path / "v.pt")
    loaded = Verifier.load(tmp_path / "v.pt")
    np.testing.assert_array_equal(loaded.pick(x), [2, 1])
    assert loaded.expert


class InfoPolicy:
    """Holds the arm still and reports a per-row extra, the way the best-of-N server's `last_info` does."""

    def __init__(self) -> None:
        self.last_info: dict[str, np.ndarray] = {}
        self.calls = 0

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        self.calls += 1
        self.last_info = {"chosen": np.full(len(observations), self.calls)}
        return np.stack([np.tile(o.joints, (8, 1)) for o in observations])


@pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")
def test_rollout_records_one_decision_per_chunk() -> None:
    layout = TwinLayout()
    sampler = SceneSampler(layout, seed=1)
    twin = Twin.from_objects(sampler.sample(), layout, start_pose=sampler.start_pose())
    policy = InfoPolicy()
    try:
        result = BatchRollout([twin], policy, "stack", record_decisions=True).run(StackCheck(), max_steps=24)[0]
    finally:
        twin.close()
    assert [d["chosen"] for d in result.decisions] == [1, 2, 3]  # 24 steps / 8 per chunk
    assert all(d["joints"].shape == (6,) for d in result.decisions)
