from dataclasses import dataclass, field

import numpy as np


@dataclass
class Calibrator:
    """Maps rehearsal success rates to real success probabilities, fitted on (predicted p, real outcome) pairs.

    Isotonic (pool-adjacent-violators): monotone, so rankings and interval bounds survive the mapping.
    Identity until fitted.
    """

    knots_x: np.ndarray | None = field(default=None, repr=False)
    knots_y: np.ndarray | None = field(default=None, repr=False)

    def fit(self, predicted: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        xs, inverse = np.unique(np.asarray(predicted, float), return_inverse=True)
        weights = np.bincount(inverse).astype(float)
        means = np.bincount(inverse, weights=np.asarray(outcomes, float)) / weights
        blocks: list[list[float]] = []  # [weighted mean, weight, number of unique xs pooled]
        for mean, weight in zip(means, weights):
            blocks.append([mean, weight, 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                m2, w2, k2 = blocks.pop()
                m1, w1, k1 = blocks[-1]
                blocks[-1] = [(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, k1 + k2]
        self.knots_x = xs
        self.knots_y = np.repeat([b[0] for b in blocks], [b[2] for b in blocks])
        return self

    def __call__(self, p: np.ndarray | float) -> np.ndarray:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        if self.knots_x is None:
            return p
        return np.interp(p, self.knots_x, self.knots_y)  # clips outside the fitted range


@dataclass(frozen=True)
class CalibrationReport:
    """How well predicted probabilities match outcomes: Brier, expected calibration error, reliability bins."""

    predicted: np.ndarray
    outcomes: np.ndarray
    n_bins: int = 5

    @property
    def brier(self) -> float:
        return float(np.mean((self.predicted - self.outcomes) ** 2))

    def bins(self) -> list[tuple[float, float, int]]:
        """(mean predicted, observed rate, count) for each non-empty equal-width bin."""
        idx = np.minimum((self.predicted * self.n_bins).astype(int), self.n_bins - 1)
        return [
            (float(self.predicted[idx == b].mean()), float(self.outcomes[idx == b].mean()), int((idx == b).sum()))
            for b in range(self.n_bins)
            if (idx == b).any()
        ]

    @property
    def ece(self) -> float:
        n = len(self.predicted)
        return float(sum(count / n * abs(pred - obs) for pred, obs, count in self.bins()))

    def summary(self) -> dict[str, float | int]:
        return {"brier": round(self.brier, 4), "ece": round(self.ece, 4), "n": len(self.predicted)}
