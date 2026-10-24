from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.scene import FULL_IMAGE_SIZE, Twin, TwinRenderer

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
    decisions: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)  # policy extras, one per chunk


class FailureWatch:
    """Follows the pick and place blocks through one rollout and names how it failed.

    `no_contact`: the pick block never moved. `pushed`: it moved but was never lifted. `dropped`: lifted, then let go
    away from the place block. `fell_off`: carried over the place block but did not end on it. `holding`: still lifted
    at the end (ran out of time). `knocked_place`: the place block was disturbed before any of the above.
    """

    FAILURES = ("no_contact", "pushed", "dropped", "fell_off", "holding", "knocked_place")

    def __init__(self, twin: Twin, check: "Check") -> None:
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


class Check(Protocol):
    """A task's success test; `top` and `bottom` name the moved object and its target, for FailureWatch."""

    top: str
    bottom: str

    def __call__(self, twin: Twin) -> bool: ...


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
    saw at every chunk plus the final state, at no extra rendering cost. With `record_decisions`, it also keeps the
    policy's per-row extras for every chunk (the server's best-of-N features and candidates, read off `last_info`). `task` and `run`'s `success` are shared by
    every twin, or given one per twin when each runs its own scene.
    """

    def __init__(
        self,
        twins: list[Twin],
        policy: Policy,
        task: str | list[str],
        image_size: tuple[int, int] = FULL_IMAGE_SIZE,
        renderer: TwinRenderer | None = None,
        record: bool = False,
        record_decisions: bool = False,
    ) -> None:
        self.twins, self.policy, self.image_size, self.renderer = twins, policy, image_size, renderer
        self.tasks = [task] * len(twins) if isinstance(task, str) else task
        self.record, self.record_decisions = record, record_decisions

    def execute(
        self, twin: Twin, chunk: np.ndarray, success: Check, steps_left: int, watch: FailureWatch
    ) -> tuple[bool, int]:
        for n, target in enumerate(chunk[:steps_left], start=1):
            twin.step_towards(twin.from_lerobot(target), 1 / CONTROL_HZ)
            if success(twin):
                return True, n
            watch.update(twin)
        return False, min(len(chunk), steps_left)

    def run(self, success: Check | list[Check], max_steps: int = 480) -> list[RolloutResult]:
        checks = success if isinstance(success, list) else [success] * len(self.twins)
        results = [RolloutResult(False, 0) for _ in self.twins]
        watches = [FailureWatch(twin, check) for twin, check in zip(self.twins, checks)]
        active = list(range(len(self.twins)))
        while active:
            observations = [self.twins[i].observe(self.tasks[i], self.image_size, self.renderer) for i in active]
            chunks = self.policy.act_batch(observations)
            info = getattr(self.policy, "last_info", {}) if self.record_decisions else {}
            for row, (i, obs, chunk) in enumerate(zip(active, observations, chunks)):
                if info:
                    results[i].decisions.append({k: v[row] for k, v in info.items()} | {"joints": obs.joints})
                done, steps = self.execute(self.twins[i], chunk, checks[i], max_steps - results[i].steps, watches[i])
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
