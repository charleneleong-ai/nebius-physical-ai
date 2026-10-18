"""The experiment ledger (docs/experiments/so101/verdicts.yaml): one row per experiment, read by the W&B report, the
artifact and the autopilot."""

from pathlib import Path
from typing import Any

import yaml

VERDICTS = Path("docs/experiments/so101/verdicts.yaml")


def load_verdicts(path: Path = VERDICTS) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def closed_families(path: Path = VERDICTS) -> set[str]:
    """Families every recorded attempt of which was discarded: proposing them again would rediscover a known loss."""
    by_family: dict[str, set[str]] = {}
    for row in load_verdicts(path)["experiments"]:
        by_family.setdefault(row["family"], set()).add(row["verdict"])
    return {family for family, seen in by_family.items() if seen == {"DISCARD"}}
