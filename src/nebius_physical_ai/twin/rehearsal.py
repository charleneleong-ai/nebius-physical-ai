from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import start_pose
from nebius_physical_ai.twin.rollout import BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import BlockPhysics, Twin, TwinLayout


@dataclass(frozen=True)
class Perturbation:
    """Domain randomisation over what the twin can't know exactly: perception error and physics."""

    pos_sigma_mm: float = 5.0
    size_jitter_mm: float = 3.0
    friction: tuple[float, float] = (0.4, 1.2)
    mass_kg: tuple[float, float] = (0.03, 0.10)
    start_pose_sigma_rad: float = 0.05

    @classmethod
    def none(cls) -> "Perturbation":
        return cls(0.0, 0.0, (1.0, 1.0), (0.05, 0.05), 0.0)


@dataclass
class Variant:
    objects: list[DetectedObject]
    physics: BlockPhysics
    start_pose: np.ndarray

    @classmethod
    def sample(cls, objects: list[DetectedObject], p: Perturbation, rng: np.random.Generator) -> "Variant":
        jittered = []
        for obj in objects:
            dx, dy = rng.normal(0, p.pos_sigma_mm, 2)
            side = np.sqrt(obj.area_mm2) + rng.uniform(-p.size_jitter_mm, p.size_jitter_mm)
            jittered.append(replace(obj, x_mm=obj.x_mm + dx, y_mm=obj.y_mm + dy, area_mm2=side**2))
        physics = BlockPhysics(mass_kg=rng.uniform(*p.mass_kg), friction=rng.uniform(*p.friction))
        return cls(jittered, physics, start_pose(rng, p.start_pose_sigma_rad))


@dataclass
class RehearsalResult:
    successes: int
    n: int
    steps: list[int]
    variants: list[dict[str, Any]] = field(repr=False)

    @property
    def p(self) -> float:
        return self.successes / self.n

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        n, p = self.n, self.p
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return float(max(0.0, centre - half)), float(min(1.0, centre + half))

    def summary(self) -> dict[str, Any]:
        lo, hi = self.wilson()
        return {"successes": self.successes, "n": self.n, "p": round(self.p, 3), "ci95": [round(lo, 3), round(hi, 3)]}


class Rehearsal:
    """Rehearse one perceived scene N times across randomised variants, in policy-sized batches."""

    def __init__(
        self,
        policy: Policy,
        task: str,
        success: StackCheck,
        layout: TwinLayout,
        perturbation: Perturbation | None = None,
        batch_size: int = 16,
        max_steps: int = 480,
        image_size: tuple[int, int] = (480, 640),
    ) -> None:
        self.policy, self.task, self.success, self.layout = policy, task, success, layout
        self.perturbation = perturbation or Perturbation()
        self.batch_size, self.max_steps, self.image_size = batch_size, max_steps, image_size

    def build(self, variant: Variant) -> Twin:
        return Twin.from_objects(variant.objects, self.layout, variant.physics, variant.start_pose)

    def run(self, objects: list[DetectedObject], n: int, seed: int = 0) -> RehearsalResult:
        rng = np.random.default_rng(seed)
        variants = [Variant.sample(objects, self.perturbation, rng) for _ in range(n)]
        outcomes = []
        for start in range(0, n, self.batch_size):
            twins = [self.build(v) for v in variants[start : start + self.batch_size]]
            try:
                rollout = BatchRollout(twins, self.policy, self.task, self.image_size)
                outcomes += rollout.run(self.success, self.max_steps)
            finally:
                for twin in twins:
                    twin.close()
        records = [
            {
                "index": i,
                "start_pose": v.start_pose.round(4).tolist(),
                "physics": asdict(v.physics),
                "objects": [asdict(o) for o in v.objects],
                "success": r.success,
                "steps": r.steps,
            }
            for i, (v, r) in enumerate(zip(variants, outcomes))
        ]
        return RehearsalResult(sum(r.success for r in outcomes), n, [r.steps for r in outcomes], records)
