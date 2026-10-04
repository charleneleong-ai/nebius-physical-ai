import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import MsgCodec, Observation
from nebius_physical_ai.twin.rehearsal import VARIANT_COLUMNS, Perturbation, Rehearsal, RehearsalResult, Variant
from nebius_physical_ai.twin.rollout import BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout

A3 = TwinLayout()
BLOCKS = [DetectedObject("blue", 130.0, 150.0, 900.0), DetectedObject("red", 220.0, 150.0, 900.0)]
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


class HoldPolicy:
    """Stub policy: commands the same LeRobot-unit target for every chunk."""

    def __init__(self, target: np.ndarray, horizon: int = 16) -> None:
        self.chunk = np.tile(target, (horizon, 1))
        self.batch_sizes: list[int] = []

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        self.batch_sizes.append(len(observations))
        return np.stack([self.chunk] * len(observations))


class SlipperyFails:
    """Success check that passes exactly the variants whose blocks are grippy (friction > 0.8)."""

    def __call__(self, twin: Twin) -> bool:
        return bool(twin.model.geom_friction[-1, 0] > 0.8)


def test_codec_round_trips_nested_arrays() -> None:
    payload = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [["pick"]], "c": {"d": np.zeros(2)}}
    out = MsgCodec.unpack(MsgCodec.pack(payload))
    np.testing.assert_array_equal(out["a"], payload["a"])
    assert out["a"].dtype == np.float32 and out["b"] == [["pick"]]


def test_stack_batches_observations() -> None:
    img = np.zeros((4, 5, 3), np.uint8)
    inp = Observation.stack([Observation(img, img, np.arange(6.0), "a"), Observation(img, img, np.ones(6), "b")])
    assert inp["video"]["front"].shape == (2, 1, 4, 5, 3)
    assert inp["state"]["single_arm"].shape == (2, 1, 5) and inp["state"]["gripper"].shape == (2, 1, 1)
    assert inp["language"]["annotation.human.task_description"] == [["a"], ["b"]]


@pytest.mark.parametrize(
    ("successes", "n", "lo", "hi"), [(0, 20, 0.0, 0.161), (10, 20, 0.299, 0.701), (16, 20, 0.584, 0.919)]
)
def test_wilson_interval(successes: int, n: int, lo: float, hi: float) -> None:
    assert RehearsalResult(successes, n, [], []).wilson() == pytest.approx((lo, hi), abs=0.002)


def test_variant_rows_are_relative_to_perceived_pick() -> None:
    variant = {
        "index": 0,
        "success": True,
        "steps": 160,
        "physics": {"friction": 0.61234, "mass_kg": 0.04567},
        "objects": [{"x_mm": BLOCKS[0].x_mm + 3.0, "y_mm": BLOCKS[0].y_mm - 1.5}],
    }
    rows = RehearsalResult(1, 1, [160], [variant]).variant_rows(7, BLOCKS[0])
    assert dict(zip(VARIANT_COLUMNS, rows[0])) == {
        "scene_id": 7,
        "index": 0,
        "success": True,
        "steps": 160,
        "friction": 0.612,
        "mass_kg": 0.0457,
        "pick_dx_mm": 3.0,
        "pick_dy_mm": -1.5,
    }


def test_variant_sampling() -> None:
    rng = np.random.default_rng(0)
    exact = Variant.sample(BLOCKS, Perturbation.none(), rng)
    assert exact.objects == BLOCKS and exact.physics.friction == 1.0
    p = Perturbation()
    variants = [Variant.sample(BLOCKS, p, rng) for _ in range(200)]
    dx = np.array([v.objects[0].x_mm - BLOCKS[0].x_mm for v in variants])
    assert abs(dx.mean()) < 1.5 and 3.5 < dx.std() < 6.5
    frictions = [v.physics.friction for v in variants]
    assert p.friction[0] <= min(frictions) and max(frictions) <= p.friction[1]


@needs_model
class TestBatchRollout:
    def test_lerobot_units_round_trip(self) -> None:
        twin = Twin.from_objects(BLOCKS, A3)
        joints = np.array([-50.0, 0.0, 25.0, 90.0, -10.0, 40.0])
        assert twin.to_lerobot(twin.from_lerobot(joints)) == pytest.approx(joints)

    def test_twins_track_target_in_one_batch_per_chunk(self) -> None:
        twins = [Twin.from_objects(BLOCKS, A3) for _ in range(3)]
        target = np.array([30.0, -20.0, 20.0, 40.0, 0.0, 50.0])
        policy = HoldPolicy(target)
        results = BatchRollout(twins, policy, "hold", image_size=(64, 80)).run(StackCheck(), 48)
        assert [r.steps for r in results] == [48] * 3 and not any(r.success for r in results)
        assert policy.batch_sizes == [3, 3, 3]
        for twin in twins:
            assert twin.to_lerobot(twin.arm_qpos()) == pytest.approx(target, abs=3)

    def test_stack_check_detects_blue_on_red(self) -> None:
        twin = Twin.from_objects(BLOCKS, A3)
        check = StackCheck()
        twin.settle(0.2)
        assert not check(twin)
        twin.data.joint("blue").qpos[:3] = twin.block_pos("red") + [0, 0, 0.031]
        twin.settle(0.3)
        assert check(twin)

    def test_rehearsal_aggregates_per_variant_outcomes(self) -> None:
        policy = HoldPolicy(np.zeros(6))
        rehearsal = Rehearsal(policy, "t", SlipperyFails(), A3, batch_size=4, max_steps=16, image_size=(32, 40))
        result = rehearsal.run(BLOCKS, n=10, seed=3)
        assert result.n == 10 and [v["index"] for v in result.variants] == list(range(10))
        assert [v["success"] for v in result.variants] == [v["physics"]["friction"] > 0.8 for v in result.variants]
        assert policy.batch_sizes == [4, 4, 2]
