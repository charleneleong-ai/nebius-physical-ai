import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject, SceneObservation
from nebius_physical_ai.trust.calibration import CalibrationReport, Calibrator
from nebius_physical_ai.trust.stress import HiddenWorld, StressCase, StressReport
from nebius_physical_ai.trust.verdict import Assessment, Fidelity, TrustPolicy, Verdict
from nebius_physical_ai.twin.rehearsal import RehearsalResult, SceneRun

PICK = DetectedObject("blue", 100.0, 50.0, 900.0)


def rehearsal(outcomes: list[bool], offsets_mm: list[float] | None = None) -> RehearsalResult:
    """Fake result: variant i succeeds per `outcomes` and has its pick displaced by offsets_mm[i] along x."""
    offsets = offsets_mm or [0.0] * len(outcomes)
    variants = [
        {
            "index": i,
            "success": ok,
            "steps": 160,
            "physics": {"friction": 1.0, "mass_kg": 0.05},
            "objects": [{"x_mm": PICK.x_mm + d, "y_mm": PICK.y_mm}],
        }
        for i, (ok, d) in enumerate(zip(outcomes, offsets))
    ]
    return RehearsalResult(sum(outcomes), len(outcomes), [160] * len(outcomes), variants)


class TestVerdict:
    @pytest.mark.parametrize(
        ("interval", "fidelity", "verdict"),
        [
            ((0.75, 0.95), Fidelity(1.0, 0.9), Verdict.GO),
            ((0.55, 0.85), Fidelity(1.0, 0.9), Verdict.RETRY),  # likely fine, but not confidently
            ((0.0, 0.2), Fidelity(1.0, 0.9), Verdict.RETRY),
            ((0.8, 0.99), Fidelity(0.5, 0.9), Verdict.ASK_HUMAN),  # high p, but perception is poor
            ((0.8, 0.99), Fidelity(1.0, 0.4), Verdict.ASK_HUMAN),  # high p, but fragile to perception error
        ],
    )
    def test_policy(self, interval: tuple[float, float], fidelity: Fidelity, verdict: Verdict) -> None:
        assert TrustPolicy().decide(interval, fidelity) is verdict

    @pytest.mark.parametrize(
        ("markers", "confidence", "expected"), [([0, 1, 2, 3], 1.0, 1.0), ([0, 2], 1.0, 0.5), ([0, 1, 2, 3], 0.6, 0.6)]
    )
    def test_perception(self, markers: list[int], confidence: float, expected: float) -> None:
        obs = SceneObservation(markers, [DetectedObject("blue", 0, 0, 900, confidence)])
        assert Fidelity.perception_from(obs) == pytest.approx(expected)

    def test_failed_observation_has_no_perception_fidelity(self) -> None:
        assert Fidelity.perception_from(SceneObservation([0], error="need 2 corner markers, saw 1")) == 0.0

    def test_robustness_drops_when_outcome_hinges_on_perception_error(self) -> None:
        offsets = [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]
        fragile = rehearsal([True] * 4 + [False] * 4, offsets)  # fails exactly when the cube is far off
        steady = rehearsal([True, False] * 4, offsets)
        assert Fidelity.robustness_from(fragile, PICK) == pytest.approx(0.0)
        assert Fidelity.robustness_from(steady, PICK) == pytest.approx(1.0)

    def test_assessment_applies_calibration_to_point_and_interval(self) -> None:
        shrink = Calibrator().fit(np.array([0.0, 1.0]), np.array([0.0, 0.5]))  # halves every probability
        a = Assessment.of(rehearsal([True] * 16), PICK, calibrator=shrink)
        assert a.p == pytest.approx(0.5) and a.interval[1] == pytest.approx(0.5)
        assert a.verdict is Verdict.RETRY  # 16/16 in the twin, but the calibrator says the twin is optimistic


class TestCalibration:
    def test_perfectly_calibrated_predictions(self) -> None:
        rng = np.random.default_rng(0)
        predicted = rng.uniform(0, 1, 20_000)
        report = CalibrationReport(predicted, (rng.uniform(0, 1, 20_000) < predicted).astype(float))
        assert report.ece < 0.02 and report.brier == pytest.approx(1 / 6, abs=0.01)

    def test_isotonic_fixes_overconfidence(self) -> None:
        rng = np.random.default_rng(1)
        predicted = rng.uniform(0, 1, 5_000)
        outcomes = (rng.uniform(0, 1, 5_000) < 0.5 * predicted).astype(float)  # twin twice as optimistic
        raw = CalibrationReport(predicted, outcomes)
        fixed = CalibrationReport(Calibrator().fit(predicted, outcomes)(predicted), outcomes)
        assert raw.ece > 0.2 and fixed.ece < 0.03 and fixed.brier < raw.brier

    def test_isotonic_pools_violators_into_a_monotone_step(self) -> None:
        cal = Calibrator().fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 0.0, 1.0]))
        np.testing.assert_allclose(cal(np.array([0.1, 0.25, 0.4])), [0.0, 0.5, 1.0])


class TestStressReport:
    def case(self, scene_id: int, predicted: list[bool], real: list[bool]) -> StressCase:
        world = HiddenWorld([PICK], friction=1.0, mass_kg=0.05)
        return StressCase(SceneRun(scene_id, [PICK], rehearsal(predicted), 1.0), world, real)

    def test_decision_value_counts_avoided_failures(self) -> None:
        cases = [self.case(i, [True] * 16, [True] * 8) for i in range(0, 8, 2)]  # confident and right
        cases += [self.case(i, [False] * 16, [False] * 7 + [True]) for i in range(1, 9, 2)]  # predicted to fail
        value = StressReport(cases).decision_value()
        assert value["go_scenes"] == 4 and value["preflight_go_success"] == 1.0
        assert value["always_execute_success"] == pytest.approx(0.5625, abs=1e-3)
        assert (value["failures_avoided"], value["successes_forgone"]) == (28, 4)

    def test_record_round_trip_rebuilds_the_same_report(self) -> None:
        cases = [self.case(i, [True] * 12 + [False] * 4, [True] * 6 + [False] * 2) for i in range(4)]
        rebuilt = [StressCase.from_record(c.record()) for c in cases]
        assert StressReport(rebuilt).summary() == StressReport(cases).summary()

    def test_hidden_world_perturbation_only_varies_start_pose(self) -> None:
        world = HiddenWorld.sample([PICK], np.random.default_rng(0))
        p = world.perturbation()
        assert p.pos_sigma_mm == 0 and p.friction == (world.friction, world.friction)
        assert world.true_objects[0] != PICK
