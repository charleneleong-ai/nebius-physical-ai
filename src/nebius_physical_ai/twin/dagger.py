"""DAgger in the twin (Ross, Gordon & Bagnell 2011, arXiv:1011.0686): roll the current policy out on fresh training
scenes, keep the exact state every failed rollout ended in, and let the scripted expert record a recovery from there.
The recoveries join the training set, so the data follows the states this policy actually reaches instead of the
expert's clean approaches. The expert takes over only where the learner failed, as in HG-DAgger (arXiv:1810.02890).

Two refinements decide which failures are worth recording:
- Uncertainty: each scene is rehearsed several times and only scenes the policy sometimes solves and sometimes fails
  are kept (its success rate within a band, ranked by p(1-p)). Always-solved scenes have nothing to teach and
  always-failed ones are often beyond the expert or the task; the frontier in between is where recoveries pay off.
  The success rate is per scene, which keeps scenes comparable (raw rollout scores bunch into a narrow band).
- Balance: the recording budget is shared round-robin across failure types (a class-balanced buffer as in GDumb,
  arXiv:2007.14097), so one round's flood of `no_contact` cannot crowd out the rarer drops and pushes.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mujoco

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import Frame, PickPlaceExpert, SceneSampler, UnreachableTarget
from nebius_physical_ai.twin.randomize import SceneRandomizer
from nebius_physical_ai.twin.rehearsal import REHEARSAL_IMAGE_SIZE, REHEARSAL_MAX_STEPS, Perturbation, Variant
from nebius_physical_ai.twin.rollout import STACK_TASK, BatchRollout, RolloutResult, StackCheck
from nebius_physical_ai.twin.scene import ARM_JOINTS, FULL_IMAGE_SIZE, BlockPhysics, SceneLook, Twin, TwinLayout

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
    scene: int = 0  # which harvested scene; its rehearsals differ only by perturbation
    scene_p: float = 0.0  # the policy's success rate over this scene's rehearsals

    @property
    def uncertainty(self) -> float:
        return self.scene_p * (1 - self.scene_p)

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
    def load(cls, path: Path) -> list["FailureSnapshot"]:
        """Every snapshot in a `harvest` JSONL file."""
        return [cls.from_record(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]

    @classmethod
    def capture(cls, twin: Twin, scene: "Rehearsed", failure: str) -> "FailureSnapshot":
        state = {"qpos": twin.data.qpos.round(6).tolist(), "qvel": twin.data.qvel.round(6).tolist()}
        v = scene.variant
        return cls(v.objects, scene.check.top, scene.check.bottom, scene.prompt, scene.look, v.physics, **state,
                   failure=failure, scene=scene.scene)  # fmt: skip

    def restore(self, layout: TwinLayout) -> Twin:
        """The same scene recompiled, put back in the saved state, with the arm holding where it stopped."""
        return self.apply(Twin.from_objects(self.objects, layout, self.physics, look=self.look))

    def apply(self, twin: Twin) -> Twin:
        """Put this snapshot's saved state into an already compiled twin of the same scene."""
        twin.data.qpos[:] = self.qpos
        twin.data.qvel[:] = self.qvel
        twin.data.ctrl[: len(ARM_JOINTS)] = twin.arm_qpos()
        mujoco.mj_forward(twin.model, twin.data)
        return twin


@dataclass
class Rehearsed:
    """One harvested rollout: its scene, the prompt, its success check, its look, and the perturbed variant."""

    scene: int
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
        variants: int = 1,
    ) -> None:
        """Each scene is rehearsed `variants` times (perturbed), which is what gives every failure its `scene_p`."""
        self.policy, self.randomizer, self.sampler, self.task = policy, randomizer, sampler, task
        self.perturbation = perturbation or Perturbation()
        self.batch_size, self.max_steps, self.image_size = batch_size, max_steps, image_size
        self.variants = variants

    def harvest(self, scenes: int) -> tuple[list[FailureSnapshot], Counter[str]]:
        """Snapshots of the recoverable failures (with their scene's success rate), and every outcome counted
        (`success` or the failure's name)."""
        rollouts = [r for scene in range(scenes) for r in self.draw(scene)]
        failed: list[FailureSnapshot] = []
        outcomes: Counter[str] = Counter()
        wins: Counter[int] = Counter()
        for start in range(0, len(rollouts), self.batch_size):
            for rehearsed, failure, snapshot in self.roll(rollouts[start : start + self.batch_size]):
                outcomes[failure or "success"] += 1
                wins[rehearsed.scene] += not failure
                if snapshot is not None:
                    failed.append(snapshot)
        return [replace(s, scene_p=wins[s.scene] / self.variants) for s in failed], outcomes

    def roll(self, batch: list[Rehearsed]) -> list[tuple[Rehearsed, str, FailureSnapshot | None]]:
        """(rollout, failure or "", snapshot of a recoverable failure) for one batch."""
        return [
            (r, res.failure, FailureSnapshot.capture(twin, r, res.failure) if self.keep(res.failure) else None)
            for r, res, twin in self.rollouts(batch)
        ]

    def rollouts(
        self, batch: list[Rehearsed], record_decisions: bool = False
    ) -> Iterator[tuple[Rehearsed, RolloutResult, Twin]]:
        """Each rollout of one batch with its result and its twin, still open (closed once the batch is consumed)."""
        twins: list[Twin] = []
        try:
            twins.extend(self.build(r) for r in batch)  # inside the try: a failed build still closes the rest
            runner = BatchRollout(
                twins, self.policy, [r.prompt for r in batch], self.image_size, record_decisions=record_decisions
            )
            yield from zip(batch, runner.run([r.check for r in batch], self.max_steps), twins)
        finally:
            for twin in twins:
                twin.close()

    @staticmethod
    def keep(failure: str) -> bool:
        return bool(failure) and failure not in UNRECOVERABLE

    def draw(self, scene: int) -> list[Rehearsed]:
        """A scene as the demos draw them, perturbed `variants` times the way rehearsals are (positions, sizes,
        physics, start pose)."""
        episode = self.randomizer.episode(self.sampler, self.task)
        look, check = self.randomizer.look(), StackCheck(episode.pick, episode.place)
        return [
            Rehearsed(
                scene, episode.prompt, check, look, Variant.sample(episode.objects, self.perturbation, self.sampler.rng)
            )
            for _ in range(self.variants)
        ]

    def build(self, scene: Rehearsed) -> Twin:
        v = scene.variant
        return Twin.from_objects(v.objects, self.sampler.layout, v.physics, v.start_pose, look=scene.look)


def select(
    snapshots: list[FailureSnapshot], budget: int, band: tuple[float, float] = (0.0, 1.0), per_scene: int = 2
) -> list[FailureSnapshot]:
    """Up to `budget` snapshots worth recording: from scenes whose success rate lies in `band` (inclusive), most
    uncertain scenes first, at most `per_scene` from any one scene (its rarer failure types first), with the budget
    shared round-robin across failure types."""
    lo, hi = band
    inside = [s for s in snapshots if lo <= s.scene_p <= hi]
    frequency = Counter(s.failure for s in inside)
    taken: Counter[int] = Counter()
    queues: dict[str, list[FailureSnapshot]] = defaultdict(list)
    for s in sorted(inside, key=lambda s: (-s.uncertainty, frequency[s.failure])):
        if taken[s.scene] < per_scene:
            taken[s.scene] += 1
            queues[s.failure].append(s)
    # Round-robin: the i-th snapshot of every type before any type's (i+1)-th, longer queues first within a pass.
    order = sorted(queues, key=lambda f: -len(queues[f]))
    ranked = sorted((i, k, s) for k, f in enumerate(order) for i, s in enumerate(queues[f]))
    return [s for *_, s in ranked[:budget]]


def record_recovery(
    snapshot: FailureSnapshot, layout: TwinLayout, image_size: tuple[int, int] = FULL_IMAGE_SIZE
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
