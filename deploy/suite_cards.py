"""The demo's "across environments, objects and tasks" cards: one per benchmark suite, from the served policy's own
benchmark runs. The score comes from the run's result file and the success and failure rollouts from the videos that
run logged to W&B, so nothing is re-run and every number on the page is the benchmark's.

    uv run python deploy/suite_cards.py GALLERY_DIR --bench logs/dagger/r3/bench.json:gxnpzs9q \
        --bench ~/npai-tasks/logs/held-out-tasks/bench.json:p3co1z59
"""

import json
import re
import shutil
from pathlib import Path
from typing import Annotated

import typer
import wandb

WHAT = {  # what each suite varies, in a visitor's words
    "clean_many": "Seen objects in 12 random scenes, straight-down camera",
    "seen_vocabulary": "Objects and colours named in new words",
    "clean_top_down_many": "A different camera angle",
    "held_out_objects": "Colours and sizes never seen in training",
    "real_1_pomade_tin_many": "A real object's measured shape (a pomade tin) as a distractor",
    "in_path_pomade_tin_many": "A real object standing in the arm's path",
    "held_out_task_basket": "New task: eye drops into a basket (ArmnetBench)",
    "held_out_task_ring": "New task: a ring over a peg (ArmnetBench)",
}
app = typer.Typer(add_completion=False)


def videos(run: "wandb.apis.public.Run", out: Path) -> dict[str, list[str]]:
    """Download the run's per-suite rollout videos to out/<suite>/<kind>.mp4."""
    found: dict[str, list[str]] = {}
    for f in run.files():
        if m := re.fullmatch(r"media/videos/video/(.+)_(success|failure)_\d+_[0-9a-f]+\.mp4", f.name):
            suite, kind = m.groups()
            (out / suite).mkdir(parents=True, exist_ok=True)
            shutil.move(f.download(root=str(out / ".dl"), replace=True).name, out / suite / f"{kind}.mp4")
            found.setdefault(suite, []).append(f"{kind}.mp4")
    shutil.rmtree(out / ".dl", ignore_errors=True)
    return found


@app.command()
def main(
    gallery: Annotated[Path, typer.Argument(help="The demo's gallery directory; cards go to gallery/suites")],
    bench: Annotated[list[str], typer.Option(help="RESULT_JSON:WANDB_RUN_ID of a benchmark run (repeatable)")],
    entity_project: str = "chaleong/preflight",
) -> None:
    api, out, cards = wandb.Api(), gallery / "suites", []
    for spec in bench:
        result, run_id = spec.rsplit(":", 1)
        found = videos(api.run(f"{entity_project}/{run_id}"), out)
        for cell in json.loads(Path(result).expanduser().read_text())["cells"]:
            suite = cell["suite"]
            cards.append({"suite": suite, "what": WHAT.get(suite, suite), "successes": cell["successes"],
                          "n": cell["n"], "videos": sorted(found.get(suite, []), reverse=True),
                          "failures": cell.get("failures", {})})  # fmt: skip
    (gallery / "suites.json").write_text(json.dumps(cards, indent=1))
    print(f"{len(cards)} suite cards -> {gallery / 'suites.json'}")


if __name__ == "__main__":
    app()
