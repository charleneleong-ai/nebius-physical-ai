from dataclasses import asdict, dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.trust.calibration import CalibrationReport, Calibrator
from nebius_physical_ai.trust.verdict import Assessment, Verdict
from nebius_physical_ai.twin.rehearsal import Perturbation, Rehearsal, RehearsalResult, SceneRun


@dataclass(frozen=True)
class HiddenWorld:
    """What the twin can't see: where the cubes really are and their real physics.

    The perceived scene is the true one plus perception error; Preflight rehearses the perceived scene, then the
    true world is rolled out to get the outcomes its predictions are scored against.
    """

    true_objects: list[DetectedObject]
    friction: float
    mass_kg: float

    @classmethod
    def sample(
        cls, perceived: list[DetectedObject], rng: np.random.Generator, perception_sigma_mm: float = 6.0
    ) -> "HiddenWorld":
        true = [
            replace(o, x_mm=o.x_mm + dx, y_mm=o.y_mm + dy)
            for o, (dx, dy) in zip(perceived, rng.normal(0, perception_sigma_mm, (len(perceived), 2)))
        ]
        # Slightly wider than the rehearsal's randomisation, so the twin is genuinely wrong sometimes.
        return cls(true, friction=float(rng.uniform(0.3, 1.3)), mass_kg=float(rng.uniform(0.03, 0.12)))

    def perturbation(self) -> Perturbation:
        """Only the arm's start pose varies between real attempts."""
        return Perturbation(0.0, 0.0, (self.friction, self.friction), (self.mass_kg, self.mass_kg))

    def record(self) -> dict[str, Any]:
        return {
            "friction": self.friction,
            "mass_kg": self.mass_kg,
            "true_objects": [asdict(o) for o in self.true_objects],
        }


@dataclass
class StressCase:
    prediction: SceneRun
    world: HiddenWorld
    outcomes: list[bool]

    @classmethod
    def run(
        cls,
        rehearsal: Rehearsal,
        scene_id: int,
        perceived: list[DetectedObject],
        n: int,
        k: int,
        seed: int,
        **tags: Any,
    ) -> "StressCase":
        world = HiddenWorld.sample(perceived, np.random.default_rng(10_000 + seed))
        prediction = SceneRun.rehearse(rehearsal, scene_id, perceived, n, seed, **tags)
        attempts = rehearsal.run(world.true_objects, k, seed + 1, world.perturbation(), record_examples=False)
        return cls(prediction, world, [v["success"] for v in attempts.variants])

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "StressCase":
        """Rebuild a case from `record()` output, so reports can be regenerated offline from the JSON."""
        scene = [DetectedObject(**o) for o in rec["scene"]]
        result = RehearsalResult(rec["successes"], rec["n"], [v["steps"] for v in rec["variants"]], rec["variants"])
        prediction = SceneRun(rec["scene_id"], scene, result, rec["seconds"], {"hard": rec.get("hard", False)})
        w = rec["world"]
        world = HiddenWorld([DetectedObject(**o) for o in w["true_objects"]], w["friction"], w["mass_kg"])
        return cls(prediction, world, rec["outcomes"])

    @property
    def true_rate(self) -> float:
        return float(np.mean(self.outcomes))

    def record(self) -> dict[str, Any]:
        return self.prediction.record() | {
            "true_rate": self.true_rate,
            "outcomes": self.outcomes,
            "world": self.world.record(),
        }


@dataclass
class StressReport:
    """Scores Preflight's predictions against hidden-world outcomes across many scenes.

    Calibration is cross-fitted (fit on even scenes, score odd, and vice versa) so the calibrated numbers are
    held-out rather than fitted to themselves.
    """

    cases: list[StressCase]

    @staticmethod
    def pairs(cases: list[StressCase]) -> tuple[np.ndarray, np.ndarray]:
        """(predicted p, outcome) per real attempt."""
        predicted = np.concatenate([np.full(len(c.outcomes), c.prediction.result.p) for c in cases])
        return predicted, np.concatenate([np.array(c.outcomes, dtype=float) for c in cases])

    @cached_property
    def calibrators(self) -> dict[int, Calibrator]:
        """Calibrator to apply to each scene, fitted on the other fold."""
        folds = [self.cases[0::2], self.cases[1::2]]
        fitted = [Calibrator().fit(*self.pairs(fold)) for fold in folds]
        return {c.prediction.scene_id: fitted[1 - f] for f, fold in enumerate(folds) for c in fold}

    def calibration(self) -> dict[str, CalibrationReport]:
        predicted, outcomes = self.pairs(self.cases)
        cal = self.calibrators
        held_out = np.concatenate(
            [cal[c.prediction.scene_id](np.full(len(c.outcomes), c.prediction.result.p)) for c in self.cases]
        )
        return {"raw": CalibrationReport(predicted, outcomes), "calibrated": CalibrationReport(held_out, outcomes)}

    @cached_property
    def assessments(self) -> list[Assessment]:
        return self.assess(calibrated=True)

    def assess(self, calibrated: bool) -> list[Assessment]:
        cal = self.calibrators if calibrated else dict.fromkeys(self.calibrators, Calibrator())
        return [
            Assessment.of(c.prediction.result, c.prediction.scene[0], calibrator=cal[c.prediction.scene_id])
            for c in self.cases
        ]

    def decision_value(self, calibrated: bool = True) -> dict[str, float | int | None]:
        """Executing only GO scenes vs executing everything, scored on the hidden worlds' outcomes."""
        verdicts = [a.verdict for a in (self.assessments if calibrated else self.assess(calibrated=False))]
        rates = np.array([c.true_rate for c in self.cases])
        go = np.array([v is Verdict.GO for v in verdicts])
        skipped = [o for c, is_go in zip(self.cases, go) if not is_go for o in c.outcomes]
        return {
            "always_execute_success": round(float(rates.mean()), 3),
            "preflight_go_success": round(float(rates[go].mean()), 3) if go.any() else None,
            "go_scenes": int(go.sum()),
            "ask_human_scenes": sum(v is Verdict.ASK_HUMAN for v in verdicts),
            "failures_avoided": skipped.count(False),
            "successes_forgone": skipped.count(True),
        }

    def summary(self) -> dict[str, float | int | None]:
        calibration = {f"{kind}_{k}": v for kind, r in self.calibration().items() for k, v in r.summary().items()}
        raw = {f"raw_{k}": v for k, v in self.decision_value(calibrated=False).items() if k != "always_execute_success"}
        return calibration | self.decision_value() | raw

    def tables(self) -> dict[str, tuple[list[str], list[list[Any]]]]:
        """W&B-ready (columns, rows): reliability bins and one verdict row per scene."""
        bins = [[kind, *b] for kind, r in self.calibration().items() for b in r.bins()]
        records = [
            {"scene_id": c.prediction.scene_id, "true_rate": c.true_rate} | a.record()
            for c, a in zip(self.cases, self.assessments)
        ]
        return {
            "reliability": (["kind", "mean_predicted", "observed", "count"], bins),
            "verdicts": (list(records[0]), [list(r.values()) for r in records]),
        }
