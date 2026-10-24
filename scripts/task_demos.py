"""Scripted demos of the twin's non-stacking tasks, for multi-task fine-tuning with one task held out.

The held_out_tasks suite scores the eye-drops-into-basket and ring-over-peg tasks (`benchmark.TaskScenes`). Training on
one of them and testing on the other measures task transfer: a gain on the held-out task, not only the trained one. Each
episode is a fresh scene from its own seed stream (never the benchmark's), seen from an oblique camera jittered around
the benchmark's, with the instruction drawn from several phrasings so the policy keys on what is asked rather than one
string. Only episodes the expert completes are kept.

    uv run python scripts/task_demos.py ~/data/basket_demos --task basket --episodes 400
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.progress import track

from nebius_physical_ai.benchmark import (
    OBLIQUE_ELEVATION_DEG,
    Case,
    TaskScenes,
    oblique_camera,
    run_expert,
)
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import Frame, start_pose
from nebius_physical_ai.twin.scene import TwinLayout

PHRASINGS = {
    "basket": (
        "Put the eye drops into the basket",
        "Place the eye drops in the basket",
        "Pick up the eye drops and drop them in the basket",
        "Move the eye drops into the basket",
        "Put the bottle of eye drops in the basket",
    ),
    "ring": (
        "Insert the colourful ring into the central wooden peg",
        "Put the ring on the peg",
        "Place the ring over the wooden peg",
        "Slide the ring down onto the peg",
        "Drop the ring over the peg",
    ),
}
STREAMS = {"basket": 130, "ring": 131}  # away from the benchmark's held_out_tasks streams (30, 31)
DEMO_IMAGE_SIZE = (480, 640)  # the expert's own default: demonstrations are kept, so they render full size
app = typer.Typer(add_completion=False)


def record(case: Case, layout: TwinLayout, rng: np.random.Generator) -> list[Frame] | None:
    """The expert's frames for one case at demonstration resolution, or None when it cannot complete the task."""
    return run_expert(case, layout, start_pose(rng), image_size=DEMO_IMAGE_SIZE)


@app.command()
def main(
    out: Annotated[Path, typer.Argument(help="LeRobot v2.1 dataset of the demos")],
    task: Annotated[str, typer.Option(help="basket or ring")] = "basket",
    episodes: int = 400,
    azimuth_deg: Annotated[float, typer.Option(help="Camera azimuth jitter around the benchmark's, either way")] = 20.0,
    shard: Annotated[int, typer.Option(help="Seed offset, so parallel recordings draw different scenes")] = 0,
) -> None:
    """Record `episodes` successful demos of one task."""
    if task not in PHRASINGS:
        raise typer.BadParameter(f"--task must be one of {sorted(PHRASINGS)}")
    layout, rng = TwinLayout(), np.random.default_rng([STREAMS[task], shard])
    writer, attempts, prompts = LeRobotWriter(out), 0, Counter()
    for _ in track(range(episodes), description=f"recording {task} demos"):
        while True:
            attempts += 1
            if attempts > 10 * episodes:
                raise RuntimeError(f"the expert solved only {len(writer.episodes)} of {attempts} scenes")
            elevation = OBLIQUE_ELEVATION_DEG + rng.uniform(-10, 10)
            camera = oblique_camera(layout, elevation, rng.uniform(-azimuth_deg, azimuth_deg))
            prompt = str(rng.choice(PHRASINGS[task]))
            case = getattr(TaskScenes(layout, rng, camera), task)(prompt)
            if (frames := record(case, layout, rng)) is not None:
                writer.add_episode(frames, prompt)
                prompts[prompt] += 1
                break
    writer.finalize()
    print(
        f"wrote {episodes} {task} demos ({writer.total_frames} frames) to {out}; expert success {episodes}/{attempts}"
    )
    print(f"prompts: {dict(prompts)}")


if __name__ == "__main__":
    app()
