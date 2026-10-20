"""The experiment ledger (docs/experiments/so101/verdicts.yaml): one row per experiment, read by the W&B report, the
artifact and the autopilot."""

from pathlib import Path
from typing import Any

import yaml

VERDICTS = Path("docs/experiments/so101/verdicts.yaml")


def load_verdicts(path: Path = VERDICTS) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def closed_families(path: Path = VERDICTS) -> dict[str, str]:
    """Families every recorded attempt of which was discarded, with the last result that closed them. Autopilot reopens
    one only for failures it hasn't been tried on, so the reason travels with the family."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in load_verdicts(path)["experiments"]:
        rows.setdefault(row["family"], []).append(row)
    return {f: str(r[-1]["result"]) for f, r in rows.items() if {row["verdict"] for row in r} == {"DISCARD"}}
