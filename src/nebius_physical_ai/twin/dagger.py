"""DAgger in the twin (Ross, Gordon & Bagnell 2011, arXiv:1011.0686): roll the current policy out on fresh training
scenes, keep the exact state every failed rollout ended in, and let the scripted expert record a recovery from there.
The recoveries join the training set, so the data follows the states this policy actually reaches instead of the
expert's clean approaches. The expert takes over only where the learner failed, as in HG-DAgger (arXiv:1810.02890).
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import mujoco

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import Frame, PickPlaceExpert, SceneSampler, UnreachableTarget
from nebius_physical_ai.twin.randomize import SceneRandomizer
from nebius_physical_ai.twin.rehearsal import REHEARSAL_IMAGE_SIZE, REHEARSAL_MAX_STEPS, Perturbation, Variant
from nebius_physical_ai.twin.rollout import STACK_TASK, BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import ARM_JOINTS, BlockPhysics, SceneLook, Twin, TwinLayout

# Still in the jaws at the end: the open-loop expert opens on its way to the pick and would drop it where it hangs.
UNRECOVERABLE = ("holding",)


@dataclass
class FailureSnapshot:
    """Everything needed to rebuild a failed rollout's last moment: the scene as rehearsed and the simulator state."""

    objects: list[DetectedObject]
    pick: str
    place: str
    prompt: str
    look: SceneLook
    physics: BlockPhysics
    qpos: list[float]
    qvel: list[float]
    failure: str

    def record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "FailureSnapshot":
        look = row["look"]
        return cls(
            **{
                **row,
                "objects": [DetectedObject(**o) for o in row["objects"]],
                "look": SceneLook(**{**look, "light_dir": tuple(look["light_dir"])}),
                "physics": BlockPhysics(**row["physics"]),
            }
        )

    @classmethod
    def capture(cls, twin: Twin, scene: "Rehearsed", failure: str) -> "FailureSnapshot":
        state = {"qpos": twin.data.qpos.round(6).tolist(), "qvel": twin.data.qvel.round(6).tolist()}
        v = scene.variant
        return cls(v.objects, scene.check.top, scene.check.bottom, scene.prompt, scene.look, v.physics, **state,
                   failure=failure)  # fmt: skip

    def restore(self, layout: TwinLayout) -> Twin:
        """The same scene recompiled, put back in the saved state, with the arm holding where it stopped."""
        twin = Twin.from_objects(self.objects, layout, self.physics, look=self.look)
        twin.data.qpos[:] = self.qpos
        twin.data.qvel[:] = self.qvel
        twin.data.ctrl[: len(ARM_JOINTS)] = twin.arm_qpos()
        mujoco.mj_forward(twin.model, twin.data)
        return twin


@dataclass
class Rehearsed:
    """One harvested scene: the prompt, its success check, its look, and the perturbed variant that was rolled out."""

    prompt: str
    check: StackCheck
    look: SceneLook
    variant: Variant


class Harvester:
    """Rolls the policy out on training-distribution scenes (never the benchmark's) and snapshots every failure."""

    def __init__(
        self,
        policy: Policy,
        randomizer: SceneRandomizer,
        sampler: SceneSampler,
        task: str = STACK_TASK,
        perturbation: Perturbation | None = None,
        batch_size: int = 16,
        max_steps: int = REHEARSAL_MAX_STEPS,
        image_size: tuple[int, int] = REHEARSAL_IMAGE_SIZE,
    ) -> None:
        self.policy, self.randomizer, self.sampler, self.task = policy, randomizer, sampler, task
        self.perturbation = perturbation or Perturbation()
        self.batch_size, self.max_steps, self.image_size = batch_size, max_steps, image_size

    def harvest(self, scenes: int) -> tuple[list[FailureSnapshot], Counter[str]]:
        """Snapshots of the recoverable failures, and every outcome counted (`success` or the failure's name)."""
        snapshots: list[FailureSnapshot] = []
        outcomes: Counter[str] = Counter()
        for start in range(0, scenes, self.batch_size):
            batch = [self.draw() for _ in range(min(self.batch_size, scenes - start))]
            twins: list[Twin] = []
            try:
                twins.extend(
                    self.build(scene) for scene in batch
                )  # inside the try: a failed build still closes the rest
                rollout = BatchRollout(twins, self.policy, [scene.prompt for scene in batch], self.image_size)
                results = rollout.run([scene.check for scene in batch], self.max_steps)
                for twin, scene, r in zip(twins, batch, results):
                    outcomes[r.failure or "success"] += 1
                    if not r.success and r.failure not in UNRECOVERABLE:
                        snapshots.append(FailureSnapshot.capture(twin, scene, r.failure))
            finally:
                for twin in twins:
                    twin.close()
        return snapshots, outcomes

    def draw(self) -> Rehearsed:
        """A scene as the demos draw them, perturbed the way rehearsals are (positions, sizes, physics, start pose)."""
        episode = self.randomizer.episode(self.sampler, self.task)
        look = self.randomizer.look()
        variant = Variant.sample(episode.objects, self.perturbation, self.sampler.rng)
        return Rehearsed(episode.prompt, StackCheck(episode.pick, episode.place), look, variant)

    def build(self, scene: Rehearsed) -> Twin:
        v = scene.variant
        return Twin.from_objects(v.objects, self.sampler.layout, v.physics, v.start_pose, look=scene.look)


def record_recovery(
    snapshot: FailureSnapshot, layout: TwinLayout, image_size: tuple[int, int] = (480, 640)
) -> list[Frame] | None:
    """The expert's demo from the snapshot's state, or None when it can't reach or stack from there."""
    with snapshot.restore(layout) as twin:
        expert = PickPlaceExpert(twin, snapshot.prompt, snapshot.pick, snapshot.place, image_size)
        try:
            frames = list(expert.run())
        except UnreachableTarget:
            return None
        twin.settle(0.5)
        return frames if StackCheck(snapshot.pick, snapshot.place)(twin) else None
