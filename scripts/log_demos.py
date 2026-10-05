"""Log a recorded demo dataset to W&B: sample videos, the prompts, and what the dataset contains.

    uv run python scripts/log_demos.py ~/data/preflight_demos_v4a --name v4a
    uv run python scripts/log_demos.py ~/data/preflight_demos_v4a --name v4a --episodes 16 --dry-run

`scripts/finetune.py run` calls this after recording, so every fine-tune has its demos next to its training run (same
W&B group). The run shows an evenly spaced sample of front-camera videos with their prompts, a contact sheet of first
frames, the prompt counts, and the dataset totals.
"""

from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
import wandb
from rich import print

from nebius_physical_ai.demo_report import DemoReport
from nebius_physical_ai.tracking import Tracker

app = typer.Typer(add_completion=False, no_args_is_help=True)
SHEET_TILE = (240, 320)  # height, width of each contact-sheet tile
SHEET_COLUMNS = 4


def contact_sheet(report: DemoReport, episodes: list[dict]) -> np.ndarray:
    """First front-camera frames of the sampled episodes, each labelled with its prompt."""
    tiles = []
    for episode in episodes:
        frame = cv2.resize(report.first_frame(episode), SHEET_TILE[::-1])
        cv2.putText(frame, report.prompt(episode)[:46], (6, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 40, 40), 1)
        tiles.append(frame)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % SHEET_COLUMNS)
    return np.vstack([np.hstack(tiles[i : i + SHEET_COLUMNS]) for i in range(0, len(tiles), SHEET_COLUMNS)])


@app.command()
def log(
    dataset: Annotated[Path, typer.Argument(help="Dataset folder written by `npai demos`")],
    name: Annotated[
        str | None, typer.Option(help="Run name and W&B group, e.g. v4a (default: the folder name)")
    ] = None,
    episodes: Annotated[int, typer.Option(min=1, help="Sample videos to upload")] = 12,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    dry_run: Annotated[bool, typer.Option(help="Print the summary and the sample without uploading")] = False,
) -> None:
    """Upload the dataset's summary, prompts, a contact sheet and sample videos."""
    try:
        report = DemoReport.load(dataset)
    except FileNotFoundError as err:
        raise typer.BadParameter(str(err)) from err
    if not report.episodes:
        raise typer.BadParameter(f"{dataset} holds no episodes")
    label, sample = name or dataset.name, report.sample(episodes)
    summary = report.summary()
    print(summary)
    print(f"sample: {[e['episode_index'] for e in sample]}")
    if dry_run:
        return
    with Tracker("demos", name=f"{label}-demos", config={"dataset": str(dataset)}, enabled=wandb_log, group=label) as t:
        rows = [
            [e["episode_index"], report.prompt(e), e["length"], wandb.Video(str(report.video_path(e)))] for e in sample
        ]
        t.run.log({"sheet": wandb.Image(contact_sheet(report, sample), caption=f"{label}: first frames")})
        t.log_summary(
            summary,
            {
                "sample_episodes": (["episode", "prompt", "frames", "front video"], rows),
                "prompts": (["prompt", "episodes"], [[p, n] for p, n in report.prompts().most_common()]),
            },
        )
    print(f"logged {len(sample)} of {summary['episodes']} episodes for {label}")


if __name__ == "__main__":
    app()
