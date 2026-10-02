from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from nebius_physical_ai.capture.mat import DetectedObject, SceneObservation
from nebius_physical_ai.trust.calibration import Calibrator
from nebius_physical_ai.twin.rehearsal import RehearsalResult

MAX_MARKERS = 4


class Verdict(StrEnum):
    GO = "GO"
    RETRY = "RETRY"  # the twin is trustworthy and says this will likely fail: rearrange and re-rehearse
    ASK_HUMAN = "ASK_HUMAN"  # the twin isn't trustworthy for this scene: don't guess


@dataclass(frozen=True)
class Fidelity:
    """How much to trust the twin for this scene, in [0, 1]; the weakest signal wins."""

    perception: float
    robustness: float

    @property
    def score(self) -> float:
        return min(self.perception, self.robustness)

    @staticmethod
    def perception_from(obs: SceneObservation) -> float:
        """Visible corner markers (homography quality) times the least confident detection."""
        if not obs.ok or not obs.objects:
            return 0.0
        return len(obs.markers_found) / MAX_MARKERS * min(o.confidence for o in obs.objects)

    @staticmethod
    def robustness_from(result: RehearsalResult, perceived_pick: DetectedObject) -> float:
        """1 minus the success-rate gap between variants with small and large perception error.

        A large gap means the outcome hinges on exactly where the cubes really are, so a confident p is fragile.
        """
        rows = np.array([r[2:] for r in result.variant_rows(0, perceived_pick)], dtype=float)  # success..dy
        success, offsets = rows[:, 0], np.hypot(rows[:, -2], rows[:, -1])
        near = offsets <= np.median(offsets)
        if near.all() or not near.any():
            return 1.0
        return float(1.0 - abs(success[near].mean() - success[~near].mean()))


@dataclass(frozen=True)
class TrustPolicy:
    """Spec thresholds; tune on validation data rather than by hand."""

    go_lower: float = 0.7
    min_fidelity: float = 0.6

    def decide(self, p_interval: tuple[float, float], fidelity: Fidelity) -> Verdict:
        if fidelity.score < self.min_fidelity:
            return Verdict.ASK_HUMAN
        return Verdict.GO if p_interval[0] >= self.go_lower else Verdict.RETRY


@dataclass(frozen=True)
class Assessment:
    """Preflight's answer for one scene: calibrated success probability, twin fidelity, and the verdict."""

    p: float
    interval: tuple[float, float]
    fidelity: Fidelity
    verdict: Verdict

    @classmethod
    def of(
        cls,
        result: RehearsalResult,
        perceived_pick: DetectedObject,
        perception: float = 1.0,
        calibrator: Calibrator | None = None,
        policy: TrustPolicy | None = None,
    ) -> "Assessment":
        calibrate = calibrator or Calibrator()
        p, lo, hi = calibrate(np.array([result.p, *result.wilson()]))
        fidelity = Fidelity(perception, Fidelity.robustness_from(result, perceived_pick))
        verdict = (policy or TrustPolicy()).decide((float(lo), float(hi)), fidelity)
        return cls(float(p), (float(lo), float(hi)), fidelity, verdict)

    def record(self) -> dict[str, float | str]:
        return {
            "p_cal": round(self.p, 3),
            "ci95_cal": [round(v, 3) for v in self.interval],
            "perception": round(self.fidelity.perception, 3),
            "robustness": round(self.fidelity.robustness, 3),
            "verdict": self.verdict.value,
        }
