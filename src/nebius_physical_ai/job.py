"""One Preflight run as a batch job: a phone frame in, the verdict and its evidence out.

This is what the public demo launches on a Nebius Serverless Job per uploaded photo (deploy/job.Dockerfile): the same
`PreflightPipeline` as `npai preflight`, against a GR00T policy server the job started beside it, writing everything the
demo page shows into one directory: `result.json` (verdict, explanation, rehearsal numbers, perceived scene) and an mp4
per example rollout. A run that fails writes `result.json` with an `error` instead, so the page always has an answer.

    npai preflight-job frame.jpg out/ --n 32
"""

import asyncio
import base64
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from nebius_physical_ai.pipeline import PipelineResult, PreflightPipeline
from nebius_physical_ai.policy.gr00t import DEFAULT_PORT
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec
from nebius_physical_ai.twin.dataset import LeRobotWriter

ROLLOUT_FPS = 4  # one recorded frame per action chunk


def result_record(result: PipelineResult, seconds: float, task: TaskSpec) -> dict[str, Any]:
    """Everything the demo page shows, JSON-ready.

    `task` is carried so a verdict is never shown without the instruction it judged: the same scene is a GO for the
    task the checkpoint was fine-tuned on and an ASK_HUMAN for one naming an object it has never seen."""
    scene = result.scene
    return {
        "task": {
            "instruction": task.instruction, "pick": task.pick, "place": task.place,
            "in_distribution": task.in_distribution,
            "note": None if task.in_distribution else task.out_of_distribution_note(),
        },
        "verdict": result.verdict,
        "explanation": result.explanation,
        "rehearsal": result.rehearsal_summary,
        "assessment": result.assessment.record() if result.assessment else None,
        "scene": None if scene is None else {
            "objects": [asdict(o) for o in scene.objects], "markers_found": scene.markers_found,
            "residual_mm": round(scene.residual_mm, 2), "perception": round(scene.perception, 3),
            "warnings": scene.warnings,
        },
        "seconds": seconds,
    }  # fmt: skip


async def run(frame: Path, out: Path, n: int, port: int, markers_only: bool, task: TaskSpec) -> dict[str, Any]:
    pipeline = PreflightPipeline(n_rollouts=n, gr00t_port=port, markers_only=markers_only, task=task)
    started = time.time()
    try:
        result = await pipeline.run(base64.b64encode(frame.read_bytes()).decode())
    finally:
        await pipeline.close()
    record = result_record(result, round(time.time() - started, 1), task)
    examples = result.rehearsal.examples if result.rehearsal else {}
    record["videos"] = sorted(f"{kind}.mp4" for kind, (_, frames) in examples.items() if frames)
    for kind, (_, frames) in examples.items():
        if frames:
            LeRobotWriter.write_video(frames, out / f"{kind}.mp4", fps=ROLLOUT_FPS)
    return record


def preflight_job(
    frame: Annotated[Path, typer.Argument(help="Phone frame with the mat's corner markers in view")],
    out: Annotated[Path, typer.Argument(help="Directory for result.json and the rollout videos")],
    n: Annotated[int, typer.Option(help="Rehearsals of the scene")] = 32,
    port: int = DEFAULT_PORT,
    markers_only: Annotated[bool, typer.Option(help="Skip the vision model (no Token Factory VL call)")] = False,
    pick: str = DEFAULT_TASK.pick,
    place: str = DEFAULT_TASK.place,
) -> None:
    """Run Preflight on one frame and write its result for the demo page; failures are written too, then re-raised."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        record = asyncio.run(run(frame, out, n, port, markers_only, TaskSpec(pick, place)))
    except Exception as err:
        (out / "result.json").write_text(json.dumps({"error": f"{type(err).__name__}: {err}"}, indent=1))
        raise typer.Exit(1) from err
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"{record['verdict']} in {record['seconds']} s -> {out / 'result.json'}", flush=True)
