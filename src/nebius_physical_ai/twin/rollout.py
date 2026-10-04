from dataclasses import dataclass, field

import numpy as np

from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.scene import Twin, TwinRenderer

CONTROL_HZ = 30  # LeRobot SO-101 recording rate: one action per control tick
STACK_TASK = "Pick blue cube and place on red cube"  # verbatim prompt the checkpoint was fine-tuned on


@dataclass
class RolloutResult:
    success: bool
    steps: int
    frames: list[np.ndarray] = field(default_factory=list, repr=False)  # top camera, one per action chunk


class StackCheck:
    """Success for 'place <top> on <bottom>': top block resting on bottom block's upper face."""

    def __init__(self, top: str = "blue", bottom: str = "red") -> None:
        self.top, self.bottom = top, bottom

    def __call__(self, twin: Twin) -> bool:
        geometry = twin.geometry(self.bottom)
        hx, hy, hz = geometry.half
        # Offset in the bottom block's live frame, so long boxes and tipped blocks are judged by their real extent.
        x, y, z = (twin.block_pos(self.top) - twin.block_pos(self.bottom)) @ twin.data.body(self.bottom).xmat.reshape(
            3, 3
        )
        inside = np.hypot(x, y) < hx if geometry.shape == "cylinder" else abs(x) < hx and abs(y) < hy
        return bool(z > 0.8 * hz and inside)


class BatchRollout:
    """Steps several twins in lockstep: one batched policy call per action chunk, each target at CONTROL_HZ.

    A twin stops at success or after `max_steps`. With `record`, each result keeps the top-camera frame the policy
    saw at every chunk plus the final state, at no extra rendering cost.
    """

    def __init__(
        self,
        twins: list[Twin],
        policy: Policy,
        task: str,
        image_size: tuple[int, int] = (480, 640),
        renderer: TwinRenderer | None = None,
        record: bool = False,
    ) -> None:
        self.twins, self.policy, self.task, self.image_size, self.renderer = twins, policy, task, image_size, renderer
        self.record = record

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
            observations = [self.twins[i].observe(self.task, self.image_size, self.renderer) for i in active]
            for i, obs, chunk in zip(active, observations, self.policy.act_batch(observations)):
                done, steps = self.execute(self.twins[i], chunk, success, max_steps - results[i].steps)
                results[i].success, results[i].steps = done, results[i].steps + steps
                if self.record:
                    results[i].frames.append(obs.front)
            still_active = []
            for i in active:
                if not results[i].success and results[i].steps < max_steps:
                    still_active.append(i)
                elif self.record:  # finished: add the final state
                    results[i].frames.append(self.twins[i].render("front_cam", self.image_size, self.renderer))
            active = still_active
        return results
