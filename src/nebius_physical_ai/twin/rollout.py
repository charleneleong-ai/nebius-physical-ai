from dataclasses import dataclass

import numpy as np

from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.scene import Twin

CONTROL_HZ = 30  # LeRobot SO-101 recording rate: one action per control tick
STACK_TASK = "Pick blue cube and place on red cube"  # verbatim prompt the checkpoint was fine-tuned on


@dataclass
class RolloutResult:
    success: bool
    steps: int


class StackCheck:
    """Success for 'place <top> on <bottom>': top block resting on bottom block's upper face."""

    def __init__(self, top: str = "blue", bottom: str = "red") -> None:
        self.top, self.bottom = top, bottom

    def __call__(self, twin: Twin) -> bool:
        top, bottom = twin.block_pos(self.top), twin.block_pos(self.bottom)
        bottom_side = Twin.block_side_m(twin.blocks[self.bottom])
        on_top = top[2] > bottom[2] + 0.4 * bottom_side
        overlapping = np.linalg.norm(top[:2] - bottom[:2]) < 0.5 * bottom_side
        return bool(on_top and overlapping)


class BatchRollout:
    """Steps several twins in lockstep: one batched policy call per action chunk, each target at CONTROL_HZ.

    A twin stops at success or after `max_steps`.
    """

    def __init__(self, twins: list[Twin], policy: Policy, task: str, image_size: tuple[int, int] = (480, 640)) -> None:
        self.twins, self.policy, self.task, self.image_size = twins, policy, task, image_size

    def execute(self, twin: Twin, chunk: np.ndarray, success: StackCheck, steps_left: int) -> tuple[bool, int]:
        for n, target in enumerate(chunk[:steps_left], start=1):
            twin.step_towards(twin.from_lerobot(target), 1 / CONTROL_HZ)
            if success(twin):
                return True, n
        return False, min(len(chunk), steps_left)

    def run(self, success: StackCheck, max_steps: int = 480) -> list[RolloutResult]:
        results = [RolloutResult(False, 0) for _ in self.twins]
        active = list(range(len(self.twins)))
        while active:
            observations = [self.twins[i].observe(self.task, self.image_size) for i in active]
            for i, chunk in zip(active, self.policy.act_batch(observations)):
                done, steps = self.execute(self.twins[i], chunk, success, max_steps - results[i].steps)
                results[i].success, results[i].steps = done, results[i].steps + steps
            active = [i for i in active if not results[i].success and results[i].steps < max_steps]
        return results
