import numpy as np
import pytest

from nebius_physical_ai import benchmark
from nebius_physical_ai.benchmark import EYE_DROPS_MM, held_out_task_suites
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.twin.expert import start_pose
from nebius_physical_ai.twin.rollout import FailureWatch
from nebius_physical_ai.twin.scene import SO101_XML, BlockPhysics, Twin, TwinLayout
from nebius_physical_ai.twin.tasks import Basket, BasketCheck, Peg, PegCheck, Ring, TaskCheck, TaskExpert

pytestmark = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")
LAYOUT = TwinLayout()
MID_X, REACH_Y = LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm + LAYOUT.base_offset_mm - 190


def drops(x_mm: float) -> DetectedObject:
    dia, height = EYE_DROPS_MM
    return DetectedObject("eye_drops", x_mm, REACH_Y, dia * dia, length_mm=dia, width_mm=dia, height_mm=height,
                          shape="cylinder")  # fmt: skip


def solve(objects: list[DetectedObject], fixtures: tuple, check: TaskCheck) -> tuple[bool, bool, str]:
    """The check before and after the expert ran, and FailureWatch's verdict over the run (it reads fixtures too)."""
    with Twin.from_objects(objects, LAYOUT, start_pose=start_pose(np.random.default_rng(0)), fixtures=fixtures) as twin:
        before, watch = check(twin), FailureWatch(twin, check)
        for _ in TaskExpert(twin, "task", check, (24, 32)).run():
            watch.update(twin)
        twin.settle(0.8)
        return before, check(twin), watch.verdict(twin)


class TestTasks:
    def test_expert_puts_the_eye_drops_in_the_basket(self) -> None:
        basket = Basket(MID_X + 55, REACH_Y)
        before, after, _ = solve([drops(MID_X - 50)], (basket,), BasketCheck("eye_drops", basket))
        assert (before, after) == (False, True)

    @pytest.mark.parametrize("angle", [0.0, np.pi])
    def test_expert_puts_the_ring_over_the_peg(self, angle: float) -> None:
        ring, peg = Ring(MID_X - 45, REACH_Y, radius_mm=36.0, tube_mm=12.0), Peg(MID_X + 55, REACH_Y)
        before, after, _ = solve([], (ring, peg), PegCheck(ring, peg, angle))
        assert (before, after) == (False, True)

    def test_fixtures_stay_put_across_variants(self) -> None:
        # block_pos raises for a name the twin does not hold, so this also covers the fixture surviving at all.
        basket = Basket(MID_X + 55, REACH_Y)
        with Twin.from_objects([drops(MID_X - 50)], LAYOUT, fixtures=(basket,)) as twin:
            copy = twin.variant(
                list(twin.blocks.values()), LAYOUT, BlockPhysics(), start_pose(np.random.default_rng(1))
            )
            np.testing.assert_allclose(copy.block_pos("basket"), twin.block_pos("basket"))


def test_held_out_task_suites_keep_only_solvable_scenes() -> None:
    suites = held_out_task_suites(LAYOUT)
    assert [s.name for s in suites] == ["held_out_task_basket", "held_out_task_ring"]
    start = start_pose(np.random.default_rng(benchmark.SEED))
    for suite in suites:
        assert len(suite.cases) == 12
        # the point of the suite: a policy miss means the policy, because the scripted expert can do every scene kept
        assert benchmark.expert_solves(suite.cases[0], LAYOUT, start)
