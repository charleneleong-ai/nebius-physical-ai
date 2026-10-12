"""Keep the W&B project readable on its own: tag every run from the verdict file, and rebuild the project report.

`tidy` writes each experiment's family, `verdict:<V>`, notes and headline summary onto its runs, and tags killed or
relaunched runs `superseded`. `build` rewrites the report in a fixed order: the hand-written sections from
`docs/experiments/so101/wandb_narrative.md` (Highlights, Task, Approach, What we learned, Next) around generated
Results (verdict table, benchmark panels bound to the runs) and a collapsed full run log. Panel keys are read back off
the runs, so a key renamed at log time shows up as a missing panel rather than an empty one.

    uv run --extra report python scripts/wandb_report.py tidy
    uv run --extra report python scripts/wandb_report.py build
"""

import re
from pathlib import Path
from typing import Any

import typer
import wandb
import wandb_workspaces.reports.v2 as wr
import yaml
from autoresearch.wandb_report import results_table, verdict_table

from nebius_physical_ai.benchmark import HEADLINE_SUITES

ENTITY, PROJECT = "chaleong", "preflight"
REPORT_ID = "VmlldzoxODA5MDQ3Mg=="
VERDICTS = Path("docs/experiments/so101/verdicts.yaml")
NARRATIVE = Path("docs/experiments/so101/wandb_narrative.md")
BEFORE_RESULTS, AFTER_RESULTS = ("Highlights", "Task", "Approach"), ("What we learned", "Next")

app = typer.Typer(add_completion=False)


def load_verdicts() -> dict[str, Any]:
    return yaml.safe_load(VERDICTS.read_text())


def narrative_sections() -> dict[str, str]:
    """`## Heading` -> body, from the hand-written narrative."""
    parts = re.split(r"^## (.+)$", NARRATIVE.read_text(), flags=re.MULTILINE)
    return {heading.strip(): body.strip() for heading, body in zip(parts[1::2], parts[2::2], strict=True)}


def fetch_runs(api: wandb.Api, ids: list[str]) -> dict[str, Any]:
    """One query for all `ids`, keyed by run id."""
    return {r.id: r for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"name": {"$in": ids}}, per_page=len(ids) or 1)}


def tag_run(run: Any, experiment: dict[str, Any], role: str, pr_base: str) -> None:
    verdict = experiment["verdict"]
    kept = {t for t in run.tags if not t.startswith("verdict:")} - {"superseded"}  # one verdict tag, the current one
    run.tags = sorted({*kept, experiment["family"], f"verdict:{verdict}", role})
    pr = f" PR: {pr_base}/{experiment['pr']}." if experiment.get("pr") else ""
    run.notes = f"{experiment['label']} ({role}). Verdict {verdict}: {experiment['result']}{pr}"
    run.summary["verdict"] = verdict
    run.summary["baseline"] = str(experiment["baseline"])
    run.summary["gate"] = experiment["gate"]
    run.update()


@app.command()
def tidy() -> None:
    """Write verdict tags, notes and summary onto every run named in the verdict file."""
    verdicts = load_verdicts()
    roles = [(e, run_id, role) for e in verdicts["experiments"] for run_id, role in (e.get("runs") or {}).items()]
    superseded = verdicts.get("superseded", [])
    runs = fetch_runs(wandb.Api(), [run_id for _, run_id, _ in roles] + superseded)
    for experiment, run_id, role in roles:
        tag_run(runs[run_id], experiment, role, verdicts["pr_base"])
        typer.echo(f"{run_id}: {experiment['family']} verdict:{experiment['verdict']} {role}")
    for run_id in superseded:
        runs[run_id].tags = sorted({*runs[run_id].tags, "superseded"})
        runs[run_id].update()
    typer.echo(f"superseded: {len(superseded)} runs")


def bench_runs(verdicts: dict[str, Any]) -> list[Any]:
    ids = [i for e in verdicts["experiments"] for i, role in (e.get("runs") or {}).items() if role == "bench"]
    runs = fetch_runs(wandb.Api(), ids)
    return [runs[i] for i in ids]


def run_log(runs: list[Any]) -> str:
    rows = []
    for r in runs:
        scores = {suite: r.summary.get(f"successes/{suite}") for suite in HEADLINE_SUITES}
        complete = all(isinstance(v, int) for v in scores.values())
        pooled = sum(scores.values()) if complete else ""
        rows.append(
            {"tag": r.name, "wandb_url": r.url, "Verdict": r.summary.get("verdict", ""), "Pooled": pooled, **scores}
        )
    return results_table(rows, ["Verdict", "Pooled", *HEADLINE_SUITES], fmt="{}")


def results_panels(runs: list[Any]) -> list[Any]:
    """Panels for the keys these runs actually logged."""
    keys = set().union(*(r.summary.keys() for r in runs))
    present = [s for s in HEADLINE_SUITES if f"successes/{s}" in keys]
    missing = [s for s in HEADLINE_SUITES if s not in present]
    if missing:
        typer.echo(f"warning: no successes/<suite> logged for {missing}")
    videos = sorted(k for k in keys if k.startswith("video/"))
    panels: list[Any] = [wr.BarPlot(title=f"{s} (of 96)", metrics=[f"successes/{s}"]) for s in present]
    if videos:
        panels.append(wr.MediaBrowser(title="Rollouts: one success and one failure per suite", media_keys=videos))
    if "cells" in keys:
        panels.append(wr.WeavePanelSummaryTable(table_name="cells"))
    return panels


def report_blocks(verdicts: dict[str, Any], runs: list[Any]) -> list[Any]:
    text = narrative_sections()

    def sections(headings: tuple[str, ...]) -> list[Any]:
        return [block for h in headings for block in (wr.H2(text=h), wr.MarkdownBlock(text=text[h]))]

    runset = wr.Runset(entity=ENTITY, project=PROJECT, name="Benchmark runs", filters=f"ID in {[r.id for r in runs]}")
    return [
        wr.TableOfContents(),
        *sections(BEFORE_RESULTS),
        wr.H2(text="Results"),
        wr.MarkdownBlock(
            text="### Experiment verdicts\n\n" + verdict_table(verdicts["experiments"], verdicts["pr_base"])
        ),
        wr.PanelGrid(runsets=[runset], panels=results_panels(runs)),
        wr.H2(text="Full run log", collapsed_blocks=[wr.MarkdownBlock(text=run_log(runs))]),
        *sections(AFTER_RESULTS),
    ]


@app.command()
def build() -> None:
    """Rewrite the project report from the narrative and the verdict file."""
    verdicts = load_verdicts()
    runs = bench_runs(verdicts)
    report = wr.Report.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}/reports/--{REPORT_ID}")
    report.title = "Preflight: SO-101 generalisation experiments"
    report.description = "Experiment log, highlights first: what moved the gate, every verdict, and what is next."
    report.blocks = report_blocks(verdicts, runs)
    report.save()
    typer.echo(report.url)


if __name__ == "__main__":
    app()
