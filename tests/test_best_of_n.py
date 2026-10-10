import numpy as np

from nebius_physical_ai.policy.best_of_n import medoid


class TestMedoid:
    def test_picks_the_consensus_not_the_outlier_per_row(self):
        rng = np.random.default_rng(0)
        chunks = rng.normal(0, 0.01, (5, 2, 4, 6))  # five candidates near zero for two batch rows
        chunks[0, 0] += 10  # an outlier in row 0
        chunks[2, 1] = 0  # the exact centre in row 1
        out = medoid(chunks)
        assert out.shape == (2, 4, 6)
        assert np.abs(out[0]).max() < 1  # row 0 never returns the outlier
        np.testing.assert_array_equal(out[1], chunks[2, 1])

    def test_single_candidate_is_returned_unchanged(self):
        chunks = np.arange(48, dtype=float).reshape(1, 2, 4, 6)
        np.testing.assert_array_equal(medoid(chunks), chunks[0])
