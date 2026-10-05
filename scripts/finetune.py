"""Fine-tune GR00T on twin demos recorded around the real camera.

The default recipe is v4: 800 episodes, cameras drawn 60% around the calibrated pose, 30% from a wide arc over the
mat and 10% straight down, starting from the latest checkpoint.

    uv run python scripts/finetune.py run v4              # record demos, build stats, fine-tune
    uv run python scripts/finetune.py run v4 --dry-run    # print the steps without running them
    uv run python scripts/finetune.py preview             # sample frames of what the policy will see
    uv run python scripts/finetune.py serve CHECKPOINT --port 5558   # serve a checkpoint to the rehearsal

Runs on the A100 from the repo root. Needs ~/Isaac-GR00T on branch n1d6 (>= 9b37aa1) with its .venv. A checkpoint is
saved every --save-steps and the last --save-total-limit are kept; each is about 22 GB.
"""

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from nebius_physical_ai.capture.camera import DEFAULT_CAMERA_MIX, parse_camera_mix

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home()
LATEST = HOME / "checkpoints/gr00t-preflight-v3-restart3/checkpoint-2000"  # the latest fine-tune
BASE_MULTITASK = HOME / "checkpoints/gr00t-n1.6-so101-multitask"  # the community SO-101 checkpoint v1 to v3 began from
MIX_TEXT = ",".join(str(share) for share in DEFAULT_CAMERA_MIX)
# Run in GR00T's own environment: relative-action statistics need the SO-100 modality config registered first.
REL_STATS = """
import sys
sys.path.insert(0, "examples/SO100")
import so100_config  # noqa: F401 - registers the NEW_EMBODIMENT modality config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats
generate_rel_stats(sys.argv[1], EmbodimentTag.NEW_EMBODIMENT)
"""

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True)
class Step:
    """One command of the pipeline: what to run, where, and with which extra environment."""

    title: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    allow_failure: bool = False

    @property
    def line(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class Recipe:
    name: str
    episodes: int = 800
    max_steps: int = 8000
    base: Path = LATEST
    camera: Path = REPO / "configs/front_camera.json"
    camera_mix: tuple[float, float, float] = DEFAULT_CAMERA_MIX
    real_objects: Path = REPO / "data/capture/real_objects.json"
    seed: int = 1
    save_steps: int = 2000
    save_total_limit: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-4
    gr00t_dir: Path = HOME / "Isaac-GR00T"
    data_root: Path = HOME / "data"
    checkpoint_root: Path = HOME / "checkpoints"
    wandb_project: str = "preflight"

    @property
    def data_dir(self) -> Path:
        return self.data_root / f"preflight_demos_{self.name}"

    @property
    def out_dir(self) -> Path:
        return self.checkpoint_root / f"gr00t-preflight-{self.name}"

    @property
    def python(self) -> str:
        return str(self.gr00t_dir / ".venv/bin/python")

    @property
    def uses_pose(self) -> bool:
        """Whether any episode takes its camera from the calibrated pose (the first share of the mix)."""
        return self.camera_mix[0] > 0

    def problems(self, record: bool = True, force: bool = False) -> list[str]:
        """Everything that would stop the run, so it fails before recording for an hour, not after."""
        found = []
        if record and self.episodes < 1:
            found.append(f"--episodes must be at least 1, got {self.episodes}")
        if self.max_steps < 1 or self.save_steps < 1 or self.save_steps > self.max_steps:
            found.append(f"--save-steps ({self.save_steps}) must be between 1 and --max-steps ({self.max_steps})")
        for path, what in ((self.base, "base checkpoint"), (self.gr00t_dir, "Isaac-GR00T checkout")):
            if not path.exists():
                found.append(f"{what} not found: {path}")
        if self.uses_pose and not self.camera.exists():
            found.append(
                f"calibrated camera not found: {self.camera} (npai calibrate-camera), or use --camera-mix 0,0.7,0.3"
            )
        if record and not self.real_objects.exists():
            found.append(f"real objects file not found: {self.real_objects}")
        if not force and any(self.out_dir.glob("checkpoint-*")):
            found.append(f"{self.out_dir} already holds checkpoints: choose another name or pass --force")
        return found

    def record_step(self) -> Step:
        camera = ("--camera", str(self.camera), "--camera-mix", ",".join(str(s) for s in self.camera_mix))
        argv = (
            ("uv", "run", "npai", "demos", str(self.data_dir), "--episodes", str(self.episodes))
            + ("--seed", str(self.seed), "--randomize", "--real-objects", str(self.real_objects))
            + (camera if self.uses_pose else ())
        )
        return Step("record demos", argv, REPO, {"MUJOCO_GL": "egl"})

    def stats_steps(self) -> list[Step]:
        data = str(self.data_dir)
        stats = (self.python, "gr00t/data/stats.py", "--dataset-path", data, "--embodiment-tag", "NEW_EMBODIMENT")
        # The first call can fail until the modality config is registered; the relative stats step fixes that.
        return [
            Step("dataset stats", stats, self.gr00t_dir, allow_failure=True),
            Step("relative stats", (self.python, "-c", REL_STATS, data), self.gr00t_dir),
        ]

    def train_step(self) -> Step:
        argv = (
            (self.python, "gr00t/experiment/launch_finetune.py")
            + ("--base-model-path", str(self.base), "--dataset-path", str(self.data_dir))
            + ("--modality-config-path", "examples/SO100/so100_config.py", "--embodiment-tag", "NEW_EMBODIMENT")
            + ("--num-gpus", "1", "--output-dir", str(self.out_dir), "--max-steps", str(self.max_steps))
            + ("--save-steps", str(self.save_steps), "--save-total-limit", str(self.save_total_limit))
            + ("--global-batch-size", str(self.batch_size), "--dataloader-num-workers", "8")
            + ("--learning-rate", str(self.learning_rate), "--use-wandb", "--wandb-project", self.wandb_project)
        )
        return Step("fine-tune", argv, self.gr00t_dir, {"WANDB_RUN_GROUP": self.name})

    def steps(self, record: bool = True) -> list[Step]:
        return [*([self.record_step()] if record else []), *self.stats_steps(), self.train_step()]


def camera_mix_option(text: str) -> tuple[float, float, float]:
    try:
        return parse_camera_mix(text)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err


MixOpt = Annotated[str, typer.Option(help="Shares of (pose-centred, wide-arc, straight-down) cameras, summing to 1")]
CameraOpt = Annotated[Path, typer.Option(help="Calibrated camera the episodes are centred on (npai calibrate-camera)")]
RealObjectsOpt = Annotated[Path, typer.Option(help="Real objects placed as distractors (npai lift output)")]


@app.command()
def run(
    name: Annotated[str, typer.Argument(help="Run name, e.g. v4: names the dataset and the checkpoint folder")],
    episodes: int = 800,
    max_steps: int = 8000,
    base: Annotated[Path, typer.Option(help="Checkpoint to start from (default: the latest fine-tune)")] = LATEST,
    camera: CameraOpt = REPO / "configs/front_camera.json",
    camera_mix: MixOpt = MIX_TEXT,
    real_objects: RealObjectsOpt = REPO / "data/capture/real_objects.json",
    seed: int = 1,
    save_steps: int = 2000,
    save_total_limit: int = 2,
    gr00t_dir: Annotated[Path, typer.Option(help="Isaac-GR00T checkout (branch n1d6)")] = HOME / "Isaac-GR00T",
    data_root: Path = HOME / "data",
    checkpoint_root: Path = HOME / "checkpoints",
    skip_demos: Annotated[bool, typer.Option(help="Reuse the dataset already recorded under this name")] = False,
    force: Annotated[bool, typer.Option(help="Train even if the output folder already holds checkpoints")] = False,
    dry_run: Annotated[bool, typer.Option(help="Print the steps without running them")] = False,
) -> None:
    """Record demos around the real camera, build GR00T's statistics, and fine-tune."""
    recipe = Recipe(
        name=name,
        episodes=episodes,
        max_steps=max_steps,
        base=base,
        camera=camera,
        camera_mix=camera_mix_option(camera_mix),
        real_objects=real_objects,
        seed=seed,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        gr00t_dir=gr00t_dir,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
    )
    record = not skip_demos
    problems = recipe.problems(record, force)
    if problems and not dry_run:
        raise typer.BadParameter("; ".join(problems))
    steps = recipe.steps(record)
    if record and not dry_run:
        shutil.rmtree(recipe.data_dir, ignore_errors=True)  # a half-written dataset from an earlier try is no use
    for i, step in enumerate(steps, start=1):
        print(f"[{i}/{len(steps)}] {step.title}: {step.line}")
        if dry_run:
            continue
        try:
            subprocess.run(step.argv, cwd=step.cwd, env={**os.environ, **step.env}, check=not step.allow_failure)
        except subprocess.CalledProcessError as err:
            print(f"{step.title} failed with exit code {err.returncode}")
            raise typer.Exit(err.returncode) from err
    for problem in problems:
        print(f"warning: {problem}")
    if not dry_run:
        print(f"FINETUNE_DONE {recipe.out_dir}/checkpoint-{recipe.max_steps}")


@app.command()
def preview(
    n: Annotated[int, typer.Option(min=1, help="Frames to render")] = 12,
    seed: int = 1,
    camera: CameraOpt = REPO / "configs/front_camera.json",
    camera_mix: MixOpt = MIX_TEXT,
    real_objects: RealObjectsOpt = REPO / "data/capture/real_objects.json",
    out: Path = Path("/tmp/finetune_preview.jpg"),
) -> None:
    """Render what the policy will see (twin frames from the camera mix) to check, say, that the arm does not hide
    the blocks from the calibrated view. Each tile is labelled with its camera source."""
    import cv2
    import numpy as np

    from nebius_physical_ai.cli import build_randomizer
    from nebius_physical_ai.twin.expert import SceneSampler
    from nebius_physical_ai.twin.scene import Twin, TwinLayout

    sampler = SceneSampler(TwinLayout(), seed=seed)
    randomizer = build_randomizer(sampler, seed, real_objects, camera, camera_mix)
    tiles = []
    for _ in range(n):
        objects, look = randomizer.objects(sampler), randomizer.look()
        with Twin.from_objects(objects, sampler.layout, start_pose=sampler.start_pose(), look=look) as twin:
            tile = np.ascontiguousarray(twin.render("front_cam", (240, 320)))
        cv2.putText(tile, randomizer.last_camera_source, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 40, 40), 2)
        tiles.append(tile)
    columns = min(4, n)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"wrote {n} frames to {out}; cameras {dict(randomizer.camera_counts)}")


@app.command()
def serve(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint folder to serve")] = LATEST,
    port: int = 5558,
    gr00t_dir: Annotated[Path, typer.Option(help="Isaac-GR00T checkout (branch n1d6)")] = HOME / "Isaac-GR00T",
    dry_run: Annotated[bool, typer.Option(help="Print the command without running it")] = False,
) -> None:
    """Serve a checkpoint to the rehearsal over ZeroMQ, on localhost only. It runs in the foreground, so detach it
    with scripts/detach.sh."""
    step = Step(
        f"serve {checkpoint.name} on port {port}",
        (str(gr00t_dir / ".venv/bin/python"), "gr00t/eval/run_gr00t_server.py")
        + ("--model-path", str(checkpoint), "--embodiment-tag", "NEW_EMBODIMENT")
        + ("--host", "127.0.0.1", "--port", str(port)),
        gr00t_dir,
        {"MUJOCO_GL": "egl"},
    )
    print(f"{step.title}: {step.line}")
    if dry_run:
        return
    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint not found: {checkpoint}")
    subprocess.run(step.argv, cwd=step.cwd, env={**os.environ, **step.env}, check=False)


if __name__ == "__main__":
    app()
