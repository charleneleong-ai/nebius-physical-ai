import json
from pathlib import Path

import numpy as np

from nebius_physical_ai.policy.flux3 import OUR_Q01, OUR_Q99, QuantileBridge


def test_the_bridge_lands_our_percentiles_on_the_checkpoints_and_round_trips(tmp_path: Path) -> None:
    theirs = {"q01": [-92.0, 84.8, 61.4, -24.0, -124.5, 0.5], "q99": [54.1, 184.2, 178.9, 80.6, 60.2, 53.6]}
    (tmp_path / "stats.json").write_text(json.dumps({"state": theirs}))
    bridge = QuantileBridge.for_checkpoint(tmp_path / "stats.json")
    np.testing.assert_allclose(bridge.to_theirs(np.array(OUR_Q01)), theirs["q01"])
    np.testing.assert_allclose(bridge.to_theirs(np.array(OUR_Q99)), theirs["q99"])
    x = np.random.default_rng(0).uniform(-50, 90, (5, 6))
    np.testing.assert_allclose(bridge.to_ours(bridge.to_theirs(x)), x)
