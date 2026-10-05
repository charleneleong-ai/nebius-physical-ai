import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import wandb

from nebius_physical_ai.twin.rehearsal import VARIANT_COLUMNS, SceneRun

if TYPE_CHECKING:
    from nebius_physical_ai.pipeline import PipelineResult

OBJECT_COLUMNS = ["label", "x_mm", "y_mm", "length_mm", "width_mm", "yaw_deg", "shape", "area_mm2", "confidence"]
VISION_COLUMNS = ["label", "bbox_2d", "confidence"]


class Tracker:
    """One W&B run per job (finetune, eval, rehearsal, sweep) in the Preflight project.

    Project/entity come from WANDB_PROJECT / WANDB_ENTITY; `enabled=False` (or WANDB_MODE=disabled) logs nothing.
    """

    def __init__(
        self, job_type: str, name: str | None = None, config: dict[str, Any] | None = None, enabled: bool = True
    ) -> None:
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "preflight"),
            entity=os.environ.get("WANDB_ENTITY"),
            job_type=job_type,
            name=name,
            config=config or {},
            mode=None if enabled else "disabled",
        )
        self.variant_rows: list[list[Any]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self.variant_rows:
            self.run.log({"variants": wandb.Table(columns=VARIANT_COLUMNS, data=self.variant_rows)})
        self.run.finish(exit_code=1 if exc else 0)

    def log_scene(self, run: SceneRun, **extra: Any) -> None:
        r = run.result
        lo, hi = r.wilson()
        metrics = {"p": r.p, "ci_lo": lo, "ci_hi": hi, "successes": r.successes, "n": r.n, "seconds": run.seconds}
        self.run.log({"scene_id": run.scene_id, **metrics, **run.tags, **extra, **self.media(run)}, step=run.scene_id)
        self.variant_rows += r.variant_rows(run.scene_id, run.scene[0])

    @staticmethod
    def media(run: SceneRun, fps: int = 4) -> dict[str, Any]:
        """Start image plus one success and one failure rollout video per scene, captioned with the variant's physics."""
        logged: dict[str, Any] = {}
        for kind, (index, frames) in run.result.examples.items():
            v = run.result.variants[index]
            physics = v["physics"]
            caption = (
                f"scene {run.scene_id} · {kind} · variant {index} · friction {physics['friction']:.2f} · "
                f"mass {physics['mass_kg'] * 1000:.0f} g · {v['steps']} steps"
            )
            video = np.stack(frames).transpose(0, 3, 1, 2)  # W&B wants (T, C, H, W)
            logged[f"video/scene_{run.scene_id}_{kind}"] = wandb.Video(video, fps=fps, caption=caption, format="mp4")
            logged.setdefault(
                f"media/scene_{run.scene_id}_start", wandb.Image(frames[0], caption=f"scene {run.scene_id}")
            )
        return logged

    def log_preflight(self, frame_bgr: np.ndarray, result: "PipelineResult", seconds: float) -> None:
        """One Preflight answer: the frame, what was perceived, the verdict and, when it was rehearsed, the rollouts."""
        summary: dict[str, Any] = {"verdict": result.verdict, "explanation": result.explanation, "seconds": seconds}
        tables: dict[str, tuple[list[str], list[list[Any]]]] = {}
        if scene := result.scene:
            summary |= {
                "markers": len(scene.markers_found),
                "marker_fit_mm": round(scene.residual_mm, 2),
                "perception": round(scene.perception, 3),
            }
            tables["objects"] = (
                OBJECT_COLUMNS,
                [[getattr(o, c) for c in OBJECT_COLUMNS] for o in scene.objects],
            )
            tables["warnings"] = (["warning"], [[w] for w in scene.warnings])
        if graph := result.scene_graph:
            summary["vision_issues"] = graph.issues
            tables["vision_model"] = (
                VISION_COLUMNS,
                [[o.get("label"), str(o.get("bbox_2d")), o.get("confidence")] for o in graph.objects],
            )
        self.run.log({"frame": wandb.Image(frame_bgr[..., ::-1], caption=f"verdict {result.verdict}")})
        if scene and result.assessment and result.rehearsal:
            summary |= result.rehearsal.summary() | result.assessment.record()  # raw counts and the calibrated answer
            self.log_scene(SceneRun(0, scene.objects, result.rehearsal, seconds))
        self.log_summary(summary, tables)

    def log_summary(self, metrics: dict[str, Any], tables: dict[str, tuple[list[str], list[list[Any]]]]) -> None:
        """Run-level results (written to the summary, not the per-scene history) plus named tables."""
        self.run.summary.update(metrics)
        self.run.log({name: wandb.Table(columns=cols, data=rows) for name, (cols, rows) in tables.items()})
