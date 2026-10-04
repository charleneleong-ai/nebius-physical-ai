import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Policy
from nebius_physical_ai.twin.expert import start_pose
from nebius_physical_ai.twin.rollout import BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import BlockPhysics, Twin, TwinLayout, TwinRenderer

# Successes after ~10.7 s of robot time are rare slow recoveries (3.5% in the first sweep); capping there cuts a
# batch's wall time by a third because every batch waits for its slowest twin.
REHEARSAL_MAX_STEPS = 320
# GR00T resizes on the server's (contended) CPU anyway; half-res renders change actions by ~0.2 LeRobot units.
REHEARSAL_IMAGE_SIZE = (240, 320)
VARIANT_COLUMNS = ["scene_id", "index", "success", "steps", "friction", "mass_kg", "pick_dx_mm", "pick_dy_mm"]


@dataclass(frozen=True)
class Perturbation:
    """Domain randomisation over what the twin can't know exactly: perception error and physics."""

    pos_sigma_mm: float = 5.0
    size_jitter_mm: float = 3.0
    friction: tuple[float, float] = (0.4, 1.2)
    mass_kg: tuple[float, float] = (0.03, 0.10)
    start_pose_sigma_rad: float = 0.05
    height_jitter_mm: float = 8.0  # worst MoGe-3 height error against a ruler (3 objects, 2026-10-04)

    @classmethod
    def none(cls) -> "Perturbation":
        return cls(
            pos_sigma_mm=0.0,
            size_jitter_mm=0.0,
            friction=(1.0, 1.0),
            mass_kg=(0.05, 0.05),
            start_pose_sigma_rad=0.0,
            height_jitter_mm=0.0,
        )


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
            side = np.sqrt(obj.area_mm2)
            scale = (side + rng.uniform(-p.size_jitter_mm, p.size_jitter_mm)) / side
            jittered.append(
                replace(
                    obj,
                    x_mm=obj.x_mm + dx,
                    y_mm=obj.y_mm + dy,
                    area_mm2=obj.area_mm2 * scale**2,
                    length_mm=obj.length_mm * scale,
                    width_mm=obj.width_mm * scale,
                    height_mm=max(obj.height_mm * scale + rng.uniform(-p.height_jitter_mm, p.height_jitter_mm), 5.0)
                    if obj.height_mm
                    else 0.0,
                )
            )
        physics = BlockPhysics(mass_kg=rng.uniform(*p.mass_kg), friction=rng.uniform(*p.friction))
        return cls(jittered, physics, start_pose(rng, p.start_pose_sigma_rad))


@dataclass
class RehearsalResult:
    successes: int
    n: int
    steps: list[int]
    variants: list[dict[str, Any]] = field(repr=False)
    # First successful and first failed variant: (variant index, top-camera frames)
    examples: dict[str, tuple[int, list[np.ndarray]]] = field(default_factory=dict, repr=False)

    @property
    def p(self) -> float:
        return self.successes / self.n

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        n, p = self.n, self.p
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return float(max(0.0, centre - half)), float(min(1.0, centre + half))

    def variant_rows(self, scene_id: int, origin: DetectedObject) -> list[list[Any]]:
        """One row per variant (VARIANT_COLUMNS); pick offsets are relative to the perceived pick object."""
        return [
            [
                scene_id,
                v["index"],
                v["success"],
                v["steps"],
                round(v["physics"]["friction"], 3),
                round(v["physics"]["mass_kg"], 4),
                round(v["objects"][0]["x_mm"] - origin.x_mm, 1),
                round(v["objects"][0]["y_mm"] - origin.y_mm, 1),
            ]
            for v in self.variants
        ]

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
        max_steps: int = REHEARSAL_MAX_STEPS,
        image_size: tuple[int, int] = REHEARSAL_IMAGE_SIZE,
        record_examples: bool = True,
    ) -> None:
        self.policy, self.task, self.success, self.layout = policy, task, success, layout
        self.perturbation = perturbation or Perturbation()
        self.batch_size, self.max_steps, self.image_size = batch_size, max_steps, image_size
        self.record_examples = record_examples

    def run(
        self,
        objects: list[DetectedObject],
        n: int,
        seed: int = 0,
        perturbation: Perturbation | None = None,
        record_examples: bool | None = None,
    ) -> RehearsalResult:
        """`perturbation` / `record_examples` override the instance defaults for this run only."""
        rng = np.random.default_rng(seed)
        perturbation = perturbation or self.perturbation
        record_examples = self.record_examples if record_examples is None else record_examples
        variants = [Variant.sample(objects, perturbation, rng) for _ in range(n)]
        template = Twin.from_objects(objects, self.layout)  # compiled once; variants copy and edit it
        renderer = TwinRenderer(template.model, self.image_size)  # variants share its meshes
        outcomes, examples = [], {}
        try:
            for start in range(0, n, self.batch_size):
                batch = variants[start : start + self.batch_size]
                twins = [template.variant(v.objects, self.layout, v.physics, v.start_pose) for v in batch]
                record = record_examples and len(examples) < 2  # stop recording once both are found
                rollout = BatchRollout(twins, self.policy, self.task, self.image_size, renderer, record)
                for i, r in enumerate(rollout.run(self.success, self.max_steps), start=start):
                    if r.frames:
                        examples.setdefault("success" if r.success else "failure", (i, r.frames))
                    outcomes.append(replace(r, frames=[]))  # keep only the two examples in memory
        finally:
            renderer.close()
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
        return RehearsalResult(sum(r.success for r in outcomes), n, [r.steps for r in outcomes], records, examples)


@dataclass
class SceneRun:
    scene_id: int
    scene: list[DetectedObject]
    result: RehearsalResult
    seconds: float
    tags: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def rehearse(
        cls, rehearsal: Rehearsal, scene_id: int, scene: list[DetectedObject], n: int, seed: int, **tags: Any
    ) -> "SceneRun":
        start = time.time()
        result = rehearsal.run(scene, n, seed)
        return cls(scene_id, scene, result, round(time.time() - start, 1), tags)

    def record(self) -> dict[str, Any]:
        """JSON-ready row: summary, timing, tags, the perceived scene and every variant."""
        return {
            "scene_id": self.scene_id,
            **self.tags,
            **self.result.summary(),
            "seconds": self.seconds,
            "scene": [asdict(o) for o in self.scene],
            "variants": self.result.variants,
        }
