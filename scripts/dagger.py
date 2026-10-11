"""One DAgger round's data in the twin: harvest the served policy's failures, then record the expert's recoveries.

    uv run python scripts/dagger.py harvest logs/dagger/r1/snapshots.jsonl --port 5564 --scenes 400 --seed 900
    uv run python scripts/dagger.py recover logs/dagger/r1/snapshots.jsonl ~/data/dagger_r1 --part 0 --parts 8

Scenes are drawn like the v6 demos (role-agnostic, lookalike decoys, the front-oblique camera mix) from seeds the
benchmark never uses, so a round never trains on what it is scored on. `scripts/dagger_round.sh` runs a whole round:
harvest, recover, aggregate, fine-tune from the current policy, benchmark, and promote only past the gate.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import track

from nebius_physical_ai.cli import build_randomizer
from nebius_physical_ai.policy.gr00t import Gr00tClient
from nebius_physical_ai.twin.dagger import FailureSnapshot, Harvester, record_recovery, select
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import SceneSampler
from nebius_physical_ai.twin.scene import TwinLayout

REPO = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def harvest(
    out: Annotated[Path, typer.Argument(help="Snapshots, one JSON object per line")],
    port: int = 5564,
    scenes: int = 400,
    variants: Annotated[
        int, typer.Option(help="Rehearsals per scene, which give each failure its scene's success rate")
    ] = 1,
    budget: Annotated[int, typer.Option(help="Snapshots kept by uncertainty selection (0 = keep every failure)")] = 0,
    band: Annotated[str, typer.Option(help="With --budget, keep scenes whose success rate is within LO,HI")] = "0,1",
    per_scene: int = 2,
    seed: int = 900,
    lookalikes: int = 2,
    original_share: float = 0.35,
    real_objects: Path = REPO / "data/capture/real_objects.json",
    camera: Path = REPO / "configs/front_oblique_camera.json",
    camera_mix: str = "0.3,0.6,0.1",
) -> None:
    """Roll the served policy out on fresh training scenes and save every recoverable failure's final state."""
    sampler = SceneSampler(TwinLayout(), seed=seed)
    randomizer = build_randomizer(sampler, seed, real_objects, camera, camera_mix, True, original_share, lookalikes)
    lo, hi = (float(v) for v in band.split(","))
    with Gr00tClient(port=port) as policy:
        harvested, outcomes = Harvester(policy, randomizer, sampler, variants=variants).harvest(scenes)
    chosen = select(harvested, budget, (lo, hi), per_scene) if budget else harvested
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(s.record()) + "\n" for s in chosen))
    rollouts = scenes * variants
    print(f"{outcomes['success']}/{rollouts} rollouts succeeded; outcomes {dict(outcomes.most_common())}")
    rates = Counter(round(s.scene_p, 2) for s in harvested)
    print(f"failures by their scene's success rate: {dict(sorted(rates.items()))}")
    print(f"kept {len(chosen)} of {len(harvested)} snapshots: {dict(Counter(s.failure for s in chosen))} -> {out}")


@app.command()
def recover(
    snapshots: Annotated[Path, typer.Argument(help="From `harvest`")],
    out: Annotated[Path, typer.Argument(help="LeRobot v2.1 dataset of the recoveries")],
    part: Annotated[int, typer.Option(help="Record every `parts`-th snapshot starting here (parallel shards)")] = 0,
    parts: int = 1,
) -> None:
    """The expert's demo from every snapshot it can recover; the rest are counted and left out."""
    rows = snapshots.read_text().splitlines()[part::parts]
    writer, kept = LeRobotWriter(out), Counter()
    for line in track(rows, description="recording recoveries"):
        snapshot = FailureSnapshot.from_record(json.loads(line))
        frames = record_recovery(snapshot, TwinLayout())
        kept[(snapshot.failure, frames is not None)] += 1
        if frames is not None:
            writer.add_episode(frames, snapshot.prompt)
    writer.finalize()
    by_failure = {f: f"{kept[(f, True)]}/{kept[(f, True)] + kept[(f, False)]}" for f, _ in sorted(kept)}
    print(f"recovered {len(writer.episodes)}/{len(rows)} into {out}; by failure {by_failure}")


if __name__ == "__main__":
    app()
