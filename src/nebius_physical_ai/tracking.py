import os
from types import TracebackType
from typing import Any, Self

import wandb

from nebius_physical_ai.twin.rehearsal import VARIANT_COLUMNS, SceneRun


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

    def log_scene(self, run: SceneRun) -> None:
        r = run.result
        lo, hi = r.wilson()
        metrics = {"p": r.p, "ci_lo": lo, "ci_hi": hi, "successes": r.successes, "n": r.n, "seconds": run.seconds}
        self.run.log({"scene_id": run.scene_id, **metrics, **run.tags}, step=run.scene_id)
        self.variant_rows += r.variant_rows(run.scene_id, run.scene[0])
