"""Score a served checkpoint on the fixed benchmark, and compare checkpoints.

    uv run python scripts/benchmark.py run v4a --port 5559 --out logs/bench_v4a.json
    uv run python scripts/benchmark.py run v4a --dry-run                 # build and list the suites, no policy needed
    uv run python scripts/benchmark.py run v3 --group legacy --only 'real_*'
    uv run python scripts/benchmark.py compare logs/bench_v3.json logs/bench_v4a.json

Runs on the A100 from the repo root against a GR00T server (see scripts/finetune.py serve). Every cell is 96 randomised
rehearsals at seed 42, the same scenes for every checkpoint, so `compare` pairs them (McNemar, p < 0.05); older result
files without per-scene outcomes fall back to non-overlapping 95% intervals.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nebius_physical_ai.benchmark import Cell, Suite, build_suites, mcnemar, run_suite, verdict
from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import DEFAULT_PORT, Gr00tClient
from nebius_physical_ai.tracking import Tracker
from nebius_physical_ai.twin.rollout import FailureWatch
from nebius_physical_ai.twin.scene import TwinLayout

REPO = Path(__file__).resolve().parent.parent
GROUPS = ("all", "legacy", "held_out", "diagnostic")
console = Console(width=140)
app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_real(path: Path) -> list[DetectedObject]:
    return [DetectedObject(**o) for o in json.loads(path.read_text())] if path.exists() else []


def show_suites(suites: list[Suite]) -> None:
    table = Table("suite", "group", "cases", "variants", "n", "left out", "note")
    for s in suites:
        table.add_row(s.name, s.group, str(len(s.cases)), str(s.variants), str(s.n), str(s.rejected), s.note)
    console.print(table)


def show_cells(label: str, cells: list[Cell]) -> None:
    table = Table("suite", "successes", "n", "p", "95% CI", title=label)
    for c in cells:
        lo, hi = c.wilson()
        table.add_row(c.suite, str(c.successes), str(c.n), f"{c.p:.0%}", f"{lo:.0%} to {hi:.0%}")
    console.print(table)


@app.command()
def run(
    label: Annotated[str, typer.Argument(help="Name of the checkpoint being scored, e.g. v4a")],
    port: int = DEFAULT_PORT,
    group: Annotated[
        str, typer.Option(help="legacy (the v2 and v3 cells), held_out, diagnostic (clean_many), appearance, or all")
    ] = "all",
    only: Annotated[list[str] | None, typer.Option(help="Only suites matching this glob; repeatable")] = None,
    real_objects: Annotated[Path, typer.Option(help="Real objects (npai lift output)")] = REPO
    / "data/capture/real_objects.json",
    camera: Annotated[Path | None, typer.Option(help="Calibrated camera: adds the clean_calibrated suite")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the cells to this JSON file")] = None,
    batch_size: int = 32,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    videos: Annotated[
        bool | None,
        typer.Option(help="Record a success and a failure rollout per suite (default: whenever W&B logging is on)"),
    ] = None,
    wandb_group: Annotated[str | None, typer.Option(help="W&B group (default: the label)")] = None,
    dry_run: Annotated[bool, typer.Option(help="Build and list the suites without scoring them")] = False,
) -> None:
    """Score the checkpoint served on --port on every suite.

    With --wandb every suite is logged as soon as it finishes, with its rollout videos, so a run that dies midway
    still shows what it had."""
    if group not in GROUPS:
        raise typer.BadParameter(f"--group must be one of {GROUPS}, got {group!r}")
    layout = TwinLayout()
    calibrated = CameraPose.load(camera) if camera else None
    suites = build_suites(load_real(real_objects), layout, group, only or [], calibrated)
    if not suites:
        raise typer.BadParameter("no suite matches --group and --only")
    show_suites(suites)
    if dry_run:
        return
    empty = [s.name for s in suites if not s.cases]
    if empty:
        console.print(f"skipping suites with no scene the expert can solve: {', '.join(empty)}")
    record_videos = wandb_log if videos is None else videos
    config = {
        "label": label,
        "port": port,
        "group": group,
        "only": only or [],
        "headline_n": 96,
        "videos": record_videos,
    }
    cells: list[Cell] = []
    with (
        Tracker("benchmark", name=label, config=config, enabled=wandb_log, group=wandb_group or label) as tracker,
        Gr00tClient(port=port) as policy,
    ):
        for suite in (s for s in suites if s.cases):
            cell = run_suite(suite, policy, layout, batch_size, record_videos)
            cells.append(cell)
            tracker.log_cell(cell, step=len(cells))
            lo, hi = cell.wilson()
            console.print(
                f"{label} | {cell.suite}: {cell.successes}/{cell.n} ({cell.p:.0%}, 95% CI {lo:.0%} to {hi:.0%})"
            )
            if out:
                out.write_text(json.dumps({"label": label, "cells": [c.record() for c in cells]}, indent=2))
        rows = [[c.suite, c.group, c.successes, c.n, round(c.p, 3), *(round(v, 3) for v in c.wilson())] for c in cells]
        tracker.log_summary(
            {f"p/{c.suite}": c.p for c in cells},
            {"cells": (["suite", "group", "successes", "n", "p", "ci_lo", "ci_hi"], rows)},
        )
    show_cells(label, cells)


def load_cells(path: Path) -> tuple[str, dict[str, Cell]]:
    data = json.loads(path.read_text())
    keep = Cell.__dataclass_fields__
    return data["label"], {c["suite"]: Cell(**{k: v for k, v in c.items() if k in keep}) for c in data["cells"]}


@app.command()
def upload(
    result: Annotated[Path, typer.Argument(help="Result file from `run`")],
    wandb_group: Annotated[str | None, typer.Option(help="W&B group (default: the label)")] = None,
) -> None:
    """Log the cells of a finished run's JSON to W&B (scores only: the rollouts are not saved in the file)."""
    label, cells = load_cells(result)
    with Tracker(
        "benchmark", name=label, config={"label": label, "source": str(result)}, group=wandb_group or label
    ) as t:
        for step, cell in enumerate(cells.values(), start=1):
            t.log_cell(cell, step=step)
        rows = [
            [c.suite, c.group, c.successes, c.n, round(c.p, 3), *(round(v, 3) for v in c.wilson())]
            for c in cells.values()
        ]
        t.log_summary(
            {f"p/{c.suite}": c.p for c in cells.values()},
            {"cells": (["suite", "group", "successes", "n", "p", "ci_lo", "ci_hi"], rows)},
        )
    print(f"logged {len(cells)} cells for {label}")


def describe(cell: Cell, base: Cell) -> str:
    """Score and verdict against the reference, with the paired evidence when there is any."""
    text = f"{cell.successes}/{cell.n} ({cell.p:.0%}) {verdict(cell, base)}"
    if (paired := mcnemar(cell, base)) is None:
        return text + " (unpaired)"
    wins, losses, p = paired
    return text + f" (+{wins}/-{losses}, p={p:.3f})"


@app.command()
def compare(
    results: Annotated[list[Path], typer.Argument(help="Result files from `run`; the first is the reference")],
) -> None:
    """Side by side: each later checkpoint against the first. Scene-by-scene McNemar when both runs saved per-scene
    outcomes (+wins/-losses on the scenes where they disagree), else the interval-overlap check."""
    if len(results) < 2:
        raise typer.BadParameter("give at least two result files")
    loaded = [load_cells(p) for p in results]
    (ref_label, ref), others = loaded[0], loaded[1:]
    table = Table("suite", ref_label, *[f"{label} (vs {ref_label})" for label, _ in others])
    for suite, base in ref.items():
        row = [suite, f"{base.successes}/{base.n} ({base.p:.0%})"]
        for _, cells in others:
            cell = cells.get(suite)
            row.append("-" if cell is None else describe(cell, base))
        table.add_row(*row)
    console.print(table)


@app.command()
def failures(result: Annotated[Path, typer.Argument(help="A result file from `run`")]) -> None:
    """How each suite's failed rehearsals failed (`FailureWatch`), with the total over every suite."""
    label, cells = load_cells(result)
    table = Table("suite", "failed", *FailureWatch.FAILURES, title=label)
    total = dict.fromkeys(FailureWatch.FAILURES, 0)
    for suite, cell in cells.items():
        for kind in total:
            total[kind] += cell.failures.get(kind, 0)
        table.add_row(suite, str(cell.n - cell.successes), *(str(cell.failures.get(k, 0)) for k in total))
    table.add_row("total", str(sum(c.n - c.successes for c in cells.values())), *map(str, total.values()))
    console.print(table)


if __name__ == "__main__":
    app()
