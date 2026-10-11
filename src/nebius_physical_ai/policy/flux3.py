"""Joint units for FLUX 3 Action (black-forest-labs/flux-3-action-so101), which speaks LeRobot degrees in the
community datasets' calibrations while our twin speaks LeRobot units mapped onto the menagerie ranges."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 1st/99th percentiles of observation.state over the v6 twin demos (619,200 frames), LeRobot units.
OUR_Q01 = [-31.948, -39.912, 11.027, 61.839, -23.376, 14.308]
OUR_Q99 = [32.014, -1.077, 57.813, 95.29, 19.937, 71.596]


@dataclass(frozen=True)
class QuantileBridge:
    """Per-joint affine map between our units and the checkpoint's, matching 1st/99th percentiles."""

    ours: tuple[np.ndarray, np.ndarray]
    theirs: tuple[np.ndarray, np.ndarray]

    @classmethod
    def for_checkpoint(cls, path: Path) -> "QuantileBridge":
        stats = json.loads(path.read_text())["state"]
        return cls((np.array(OUR_Q01), np.array(OUR_Q99)), (np.array(stats["q01"]), np.array(stats["q99"])))

    def to_theirs(self, x: np.ndarray) -> np.ndarray:
        (a, b), (c, d) = self.ours, self.theirs
        return c + (x - a) * (d - c) / (b - a)

    def to_ours(self, y: np.ndarray) -> np.ndarray:
        (a, b), (c, d) = self.ours, self.theirs
        return a + (y - c) * (b - a) / (d - c)
