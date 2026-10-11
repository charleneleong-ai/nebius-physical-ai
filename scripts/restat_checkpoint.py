"""Copy a LeRobot GR00T checkpoint with its state/action normalisation recomputed from one of our datasets.

`lerobot-train --policy.path=...` keeps the checkpoint's saved statistics. hqfang/groot-so100_101 pools 19M frames
in degrees (shoulder lift 24 to 196), while our twin records LeRobot units (shoulder lift about -42 to 5), so without
this every frame would normalise far outside the range the model learned. The weights are hard-linked, not copied.
Runs in the LeRobot environment:

    ~/lerobot-n17/.venv/bin/python scripts/restat_checkpoint.py ~/checkpoints/hqfang-groot-so100_101 \\
        ~/data/preflight_demos_v4c_v30 ~/checkpoints/hqfang-restat-v4c
"""

import os
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from safetensors.torch import load_file, save_file

STAT_FILES = ("*pack_inputs*.safetensors", "*unpack_unnormalize*.safetensors")
QUANTILES = {"q01": 1, "q10": 10, "q90": 90, "q99": 99}
app = typer.Typer(add_completion=False)


def column_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Every statistic the GR00T processors store, over all frames of one column (N, D)."""
    stats = {
        "min": values.min(0),
        "max": values.max(0),
        "mean": values.mean(0),
        "std": values.std(0),
        "count": np.array([len(values)]),
    }
    return stats | {name: np.percentile(values, q, axis=0) for name, q in QUANTILES.items()}


def dataset_stats(root: Path) -> dict[str, dict[str, np.ndarray]]:
    frames = pd.concat(
        pd.read_parquet(p, columns=["observation.state", "action"]) for p in root.glob("data/**/*.parquet")
    )
    return {key: column_stats(np.stack(frames[key].to_numpy()).astype(np.float64)) for key in frames.columns}


def restat(path: Path, stats: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Rewrite one saved stats file in place: same keys, dtypes and shapes, values from `stats`."""
    saved, changed = load_file(path), []
    for key, tensor in saved.items():
        feature, _, name = key.rpartition(".")
        if feature in stats and name in stats[feature]:
            value = np.asarray(stats[feature][name], dtype=np.float64).reshape(tensor.shape)
            saved[key] = torch.as_tensor(value, dtype=tensor.dtype)
            changed.append(key)
    save_file(saved, path)
    return changed


def link_or_copy(src: str, dst: str) -> None:
    """Hard-link the model weights (12 GB), copy everything else."""
    if Path(src).name.startswith("model") and src.endswith(".safetensors"):
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Argument(help="LeRobot GR00T checkpoint to start from")],
    dataset: Annotated[Path, typer.Argument(help="LeRobot v3.0 dataset whose statistics to use")],
    out: Annotated[Path, typer.Argument(help="New checkpoint directory (must not exist)")],
) -> None:
    if out.exists():
        raise typer.BadParameter(f"{out} exists")
    shutil.copytree(checkpoint, out, copy_function=link_or_copy)
    stats = dataset_stats(dataset)
    for pattern in STAT_FILES:
        for path in out.glob(pattern):
            changed = restat(path, stats)
            print(f"{path.name}: {len(changed)} statistics from {dataset.name}")
    for feature in stats:
        print(feature, "q01", np.round(stats[feature]["q01"], 1), "q99", np.round(stats[feature]["q99"], 1))


if __name__ == "__main__":
    app()
