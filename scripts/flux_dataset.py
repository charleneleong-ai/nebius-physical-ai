"""Copy a LeRobot v3.0 twin dataset into FLUX 3 Action's joint units, for its LoRA recipe.

The checkpoint's normalisation must stay fixed (its docs: a different statistics file needs a separately prepared
base), so the data move instead: `observation.state` and `action` go through `QuantileBridge`, and `meta/stats.json` is
rescaled to match. Videos are hard-linked, not copied. Cameras keep their names; train with
`--rename_map='{"observation.images.front": "observation.images.scene"}'`.

    ~/lerobot-flux/.venv/bin/python scripts/flux_dataset.py ~/data/preflight_demos_v6_v30 ~/data/preflight_demos_v6_flux
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.policy.flux3 import QuantileBridge

CHECKPOINT = "black-forest-labs/flux-3-action-so101"
JOINTS = ("observation.state", "action")


def bridged_column(column: pa.ChunkedArray, bridge: QuantileBridge) -> pa.Array:
    values = np.stack(column.to_numpy(zero_copy_only=False)).astype(np.float64)
    return pa.array(list(bridge.to_theirs(values).astype(np.float32)), type=column.type)


def bridged_stats(stats: dict[str, Any], bridge: QuantileBridge) -> dict[str, Any]:
    """An affine map moves min, max, mean and quantiles through it and scales std by its slope."""
    (a, b), (c, d) = bridge.ours, bridge.theirs
    slope = (d - c) / (b - a)
    moved = {}
    for key, value in stats.items():
        arr = np.asarray(value, dtype=np.float64)
        if key == "std":
            moved[key] = (arr * np.abs(slope)).tolist()
        elif key == "count":
            moved[key] = value
        else:
            moved[key] = bridge.to_theirs(arr).tolist()
    return moved


def link_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def main(
    source: Annotated[Path, typer.Argument(help="LeRobot v3.0 dataset recorded by the twin")],
    out: Annotated[Path, typer.Argument(help="Bridged copy")],
) -> None:
    bridge = QuantileBridge.for_checkpoint(Path(hf_hub_download(CHECKPOINT, "dataset_statistics.json")))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source / "meta", out / "meta")
    link_tree(source / "videos", out / "videos")
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for name in JOINTS:
            table = table.set_column(table.schema.get_field_index(name), name, bridged_column(table[name], bridge))
        (out / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / path.relative_to(source))
    stats_path = out / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats.update({name: bridged_stats(stats[name], bridge) for name in JOINTS})
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}: joints in FLUX units, videos hard-linked")


if __name__ == "__main__":
    typer.run(main)
