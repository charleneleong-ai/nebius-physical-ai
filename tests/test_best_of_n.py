import numpy as np
import pytest

from nebius_physical_ai.policy.best_of_n import candidate_features, criticality, decision_weights, medoid_index

RNG = np.random.default_rng(0)


class TestMedoid:
    def test_picks_the_consensus_not_the_outlier_per_row(self) -> None:
        chunks = RNG.normal(0, 0.01, (5, 2, 4, 6))  # five candidates near zero for two batch rows
        chunks[0, 0] += 10  # an outlier in row 0
        chunks[2, 1] = 0  # the exact centre in row 1
        best = medoid_index(chunks)
        assert best[0] != 0
        assert best[1] == 2

    def test_single_candidate_is_chosen(self) -> None:
        np.testing.assert_array_equal(medoid_index(np.zeros((1, 3, 4, 6))), [0, 0, 0])


class TestVerifierFeatures:
    def test_candidates_share_the_observation_and_differ_by_their_offsets(self) -> None:
        features, joints = RNG.normal(size=(2, 5)), RNG.normal(size=(2, 6))
        chunks = joints[:, None, None, :] + RNG.normal(size=(2, 3, 4, 6))
        x = candidate_features(features, joints, chunks)
        assert x.shape == (2, 3, 5 + 6 + 4 * 6)
        np.testing.assert_allclose(x[:, 0, :11], x[:, 2, :11])
        np.testing.assert_allclose(x[1, 1, 11:], (chunks[1, 1] - joints[1]).ravel(), rtol=1e-6)

    def test_criticality_grows_with_candidate_disagreement(self) -> None:
        same = np.repeat(RNG.normal(size=(1, 1, 4, 6)), 8, axis=1)
        spread = same + RNG.normal(scale=2.0, size=same.shape)
        assert criticality(same)[0] == pytest.approx(0.0)
        assert criticality(spread)[0] > 1.0

    @pytest.mark.parametrize(("scores", "flat"), [(np.array([1.0, 2.0, 5.0]), False), (np.full(3, 2.0), True)])
    def test_decision_weights_favour_above_average_disagreement(self, scores: np.ndarray, flat: bool) -> None:
        w = decision_weights(scores, beta=1.0)
        assert (w > 0).all()
        assert np.allclose(w, 1.0) if flat else w.argmax() == scores.argmax()
