from dataclasses import dataclass, field

import numpy as np

from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.scene import Twin, TwinRenderer

CONTROL_HZ = 30  # LeRobot SO-101 recording rate: one action per control tick
STACK_TASK = "Pick blue cube and place on red cube"  # verbatim prompt the checkpoint was fine-tuned on


LIFTED_M = 0.02  # a block this far above where it started is in the gripper
MOVED_M = 0.015  # ...and this far sideways has been disturbed
OVER_PLACE_M = 0.04  # carried to within this of the place block


@dataclass
class RolloutResult:
    success: bool
    steps: int
    frames: list[np.ndarray] = field(default_factory=list, repr=False)  # top camera, one per action chunk
    failure: str = ""  # why it failed (FailureWatch.FAILURES), empty on success


class FailureWatch:
    """Follows the pick and place blocks through one rollout and names how it failed.

    `no_contact`: the pick block never moved. `pushed`: it moved but was never lifted. `dropped`: lifted, then let go
    away from the place block. `fell_off`: carried over the place block but did not end on it. `holding`: still lifted
    at the end (ran out of time). `knocked_place`: the place block was disturbed before any of the above.
    """

    FAILURES = ("no_contact", "pushed", "dropped", "fell_off", "holding", "knocked_place")

    def __init__(self, twin: Twin, check: "StackCheck") -> None:
        self.pick, self.place = check.top, check.bottom
        self.pick_start, self.place_start = twin.block_pos(self.pick), twin.block_pos(self.place)
        self.lifted = self.over_place = False

    def update(self, twin: Twin) -> None:
        pick = twin.block_pos(self.pick)
        if pick[2] - self.pick_start[2] > LIFTED_M:
            self.lifted = True
            if np.linalg.norm(pick[:2] - twin.block_pos(self.place)[:2]) < OVER_PLACE_M:
                self.over_place = True

    def verdict(self, twin: Twin) -> str:
        pick, place = twin.block_pos(self.pick), twin.block_pos(self.place)
        if pick[2] - self.pick_start[2] > LIFTED_M:
            return "holding"
        if self.over_place:
            return "fell_off"
        if self.lifted:
            return "dropped"
        if np.linalg.norm(place[:2] - self.place_start[:2]) > MOVED_M:
            return "knocked_place"
        return "pushed" if np.linalg.norm(pick[:2] - self.pick_start[:2]) > MOVED_M else "no_contact"


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

    def execute(
        self, twin: Twin, chunk: np.ndarray, success: StackCheck, steps_left: int, watch: FailureWatch
    ) -> tuple[bool, int]:
        for n, target in enumerate(chunk[:steps_left], start=1):
            twin.step_towards(twin.from_lerobot(target), 1 / CONTROL_HZ)
            if success(twin):
                return True, n
            watch.update(twin)
        return False, min(len(chunk), steps_left)

    def run(self, success: StackCheck, max_steps: int = 480) -> list[RolloutResult]:
        results = [RolloutResult(False, 0) for _ in self.twins]
        watches = [FailureWatch(twin, success) for twin in self.twins]
        active = list(range(len(self.twins)))
        while active:
            observations = [self.twins[i].observe(self.task, self.image_size, self.renderer) for i in active]
            for i, obs, chunk in zip(active, observations, self.policy.act_batch(observations)):
                done, steps = self.execute(self.twins[i], chunk, success, max_steps - results[i].steps, watches[i])
                results[i].success, results[i].steps = done, results[i].steps + steps
                if self.record:
                    results[i].frames.append(obs.front)
            still_active = []
            for i in active:
                if not results[i].success and results[i].steps < max_steps:
                    still_active.append(i)
                    continue
                if not results[i].success:
                    results[i].failure = watches[i].verdict(self.twins[i])
                if self.record:  # finished: add the final state
                    results[i].frames.append(self.twins[i].render("front_cam", self.image_size, self.renderer))
            active = still_active
        return results
