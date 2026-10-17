"""Score FLUX 3 Action (black-forest-labs/flux-3-action-so101) on our benchmark suites and on so101-benchmark.

FLUX keeps its own observation and command history and expects one observation per 30 Hz tick, so it cannot sit behind
the batched, chunk-at-a-time GR00T server: this runs each rollout itself, one tick at a time, on exactly the scenes
`scripts/benchmark.py` rehearses (same variants from the same seed), and writes the same result file, so
`benchmark.py compare` pairs it scene by scene against any GR00T checkpoint.

The checkpoint speaks LeRobot degrees in the community datasets' calibrations; our twin speaks LeRobot units mapped
onto the menagerie ranges. `QuantileBridge` maps each joint affinely so our training data's 1st/99th percentiles land
on the checkpoint's: the same inputs the model would see had its statistics been recomputed from our data, which is
what rescued hqfang's checkpoint zero-shot (scripts/restat_checkpoint.py).

Runs in the LeRobot-main environment (`lerobot.policies.flux3` is not in 0.6.1), from the repo root:

    ~/lerobot-flux/.venv/bin/python scripts/flux_eval.py suites flux3-so101 --only clean_many --only held_out_objects
    ~/lerobot-flux/.venv/bin/python scripts/flux_eval.py external --episodes 50 --out logs/so101bench_flux3.json
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
from huggingface_hub import hf_hub_download
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.flux3.processor_flux3 import ObservationHistoryNormalizerProcessorStep
from peft import PeftConfig, PeftModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from nebius_physical_ai.benchmark import SEED, Cell, Suite, build_suites
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.flux3 import QuantileBridge
from nebius_physical_ai.tracking import Tracker
from nebius_physical_ai.twin.rehearsal import (
    REHEARSAL_IMAGE_SIZE,
    REHEARSAL_MAX_STEPS,
    Perturbation,
    Variant,
)
from nebius_physical_ai.twin.rollout import CONTROL_HZ, Check, FailureWatch
from nebius_physical_ai.twin.scene import Twin, TwinLayout, TwinRenderer

CHECKPOINT = "black-forest-labs/flux-3-action-so101"
SO101_BENCHMARK = Path.home() / "so101-benchmark"
FRAME_EVERY = 8  # ticks between recorded frames: a 320-tick rollout becomes a 40-frame video
app = typer.Typer(add_completion=False, no_args_is_help=True)


class FluxDriver:
    """FLUX 3 Action one tick at a time: images and joints in, the next joint target out, in our units."""

    def __init__(
        self, checkpoint: str = CHECKPOINT, bridge: bool = True, device: str = "cuda", action_steps: int = 0
    ) -> None:
        """`action_steps` overrides how many of each predicted chunk run before replanning (0 = the checkpoint's)."""
        self.policy = self.load(checkpoint).to(device).eval()
        if action_steps:
            self.policy.config.n_action_steps = action_steps
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        # A LoRA keeps the base's statistics, which its saved folder may not copy: fall back to the base's on the Hub.
        local = Path(checkpoint) / "dataset_statistics.json"
        stats = local if local.exists() else Path(hf_hub_download(CHECKPOINT, "dataset_statistics.json"))
        self.bridge = QuantileBridge.for_checkpoint(stats) if bridge else None
        self.device = device

    @staticmethod
    def load(checkpoint: str) -> Flux3Policy:
        """A full checkpoint, or a LoRA adapter (BFL's recipe saves only the adapter) merged into its base."""
        if not (Path(checkpoint) / "adapter_config.json").exists():
            return Flux3Policy.from_pretrained(checkpoint)
        adapter = PeftConfig.from_pretrained(checkpoint)
        base = Flux3Policy.from_pretrained(adapter.base_model_name_or_path)
        return PeftModel.from_pretrained(base, checkpoint, config=adapter).merge_and_unload()

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @staticmethod
    def image(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255

    def batch(self, scene: np.ndarray, wrist: np.ndarray, joints: np.ndarray, task: str) -> dict[str, Any]:
        state = self.bridge.to_theirs(joints) if self.bridge else joints
        return {
            "observation.images.scene": self.image(scene),
            "observation.images.wrist": self.image(wrist),
            "observation.state": torch.tensor(state, dtype=torch.float32)[None],
            "task": [task],
        }

    def observe(self, scene: np.ndarray, wrist: np.ndarray, joints: np.ndarray, task: str, command: np.ndarray) -> None:
        """Push one past frame into FLUX's observation history (it conditions on the last `n_obs_steps`) without
        predicting, with `command` as the target sent at that frame (our units)."""
        command = self.bridge.to_theirs(command) if self.bridge else command
        self.history.last_command = torch.tensor(command, dtype=torch.float32, device=self.device)[None]
        with torch.inference_mode():
            self.pre(self.batch(scene, wrist, joints, task))

    @property
    def history(self) -> ObservationHistoryNormalizerProcessorStep:
        return next(s for s in self.pre.steps if isinstance(s, ObservationHistoryNormalizerProcessorStep))

    def act(self, scene: np.ndarray, wrist: np.ndarray, joints: np.ndarray, task: str) -> np.ndarray:
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(self.batch(scene, wrist, joints, task))))
        target = action[0].float().cpu().numpy()
        return self.bridge.to_ours(target) if self.bridge else target


def rollout(
    driver: FluxDriver, twin: Twin, check: Check, prompt: str, renderer: TwinRenderer, record: bool = False
) -> tuple[str, list[np.ndarray]]:
    """(empty string on success, else how it failed (FailureWatch); top-camera frames every FRAME_EVERY ticks)."""
    driver.reset()
    watch, frames = FailureWatch(twin, check), []
    for tick in range(REHEARSAL_MAX_STEPS):
        obs = twin.observe(prompt, REHEARSAL_IMAGE_SIZE, renderer)
        if record and tick % FRAME_EVERY == 0:
            frames.append(obs.front)
        twin.step_towards(twin.from_lerobot(driver.act(obs.front, obs.wrist, obs.joints, prompt)), 1 / CONTROL_HZ)
        if check(twin):
            return "", frames + ([twin.render("front_cam", REHEARSAL_IMAGE_SIZE, renderer)] if record else [])
        watch.update(twin)
    return watch.verdict(twin), frames


def score_suite(driver: FluxDriver, suite: Suite, layout: TwinLayout) -> Cell:
    """Every case's variants drawn exactly as `Rehearsal.run` draws them, so outcomes pair with GR00T's."""
    start, per_case, outcomes, failures = time.time(), [], [], Counter()
    examples: dict[str, tuple[str, list[np.ndarray]]] = {}  # first success and first failure, for W&B
    for case in suite.cases:
        case_layout, rng = case.layout(layout), np.random.default_rng(SEED)
        variants = [Variant.sample(case.objects, Perturbation(), rng) for _ in range(suite.variants)]
        template = Twin.from_objects(case.objects, case_layout, fixtures=case.fixtures)
        renderer = TwinRenderer(template.model, REHEARSAL_IMAGE_SIZE)
        try:
            for i, v in enumerate(variants):
                twin = template.variant(v.objects, case_layout, v.physics, v.start_pose)
                record = len(examples) < 2
                check = case.success()
                failure, frames = rollout(driver, twin, check, case.prompt, renderer, record)
                kind = "failure" if failure else "success"
                if frames:
                    caption = f"{suite.name} | {case.name} | {kind} {failure} | variant {i} | {case.prompt}"
                    examples.setdefault(kind, (caption, frames))
                outcomes.append(not failure)
                failures.update([failure] if failure else [])
        finally:
            renderer.close()
        per_case.append(sum(outcomes[-len(variants) :]))
        print(f"  {suite.name} {case.name}: {per_case[-1]}/{len(variants)}", flush=True)
    seconds = round(time.time() - start, 1)
    cell = Cell(suite.name, suite.group, sum(per_case), suite.n, len(suite.cases), seconds, per_case, outcomes,
                dict(failures.most_common()))  # fmt: skip
    cell.examples = examples
    return cell


@app.command()
def suites(
    label: Annotated[str, typer.Argument(help="Name for the result file, e.g. flux3-so101")],
    only: Annotated[list[str] | None, typer.Option(help="Only suites matching this glob; repeatable")] = None,
    group: str = "all",
    native: Annotated[bool, typer.Option(help="Feed our units unmapped (no QuantileBridge)")] = False,
    checkpoint: Annotated[str, typer.Option(help="Hub id or a fine-tuned pretrained_model folder")] = CHECKPOINT,
    action_steps: Annotated[int, typer.Option(help="Actions run per chunk before replanning (0 = checkpoint's)")] = 0,
    out: Path = Path("logs/bench_flux3.json"),
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Scores and rollout videos to W&B")] = True,
    wandb_group: str = "flux3",
) -> None:
    layout = TwinLayout()
    real = [DetectedObject(**o) for o in json.loads((REPO / "data/capture/real_objects.json").read_text())]
    driver, cells = FluxDriver(checkpoint, bridge=not native, action_steps=action_steps), []
    config = {
        "checkpoint": checkpoint,
        "bridge": not native,
        "group": group,
        "only": only,
        "action_steps": action_steps,
    }
    with Tracker("benchmark", name=label, config=config, enabled=wandb_log, group=wandb_group) as tracker:
        for suite in build_suites(real, layout, group, only or ()):
            cells.append(score_suite(driver, suite, layout))
            c = cells[-1]
            tracker.log_cell(c, step=len(cells) - 1)
            print(f"{label} | {c.suite}: {c.successes}/{c.n} failures {c.failures}", flush=True)
            out.write_text(json.dumps({"label": label, "cells": [cell.record() for cell in cells]}, indent=1))


def external_episode(
    driver: FluxDriver, env: Any, seed: int, prompt: str, record: bool = False
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """One so101-benchmark episode, one FLUX tick per environment step, with its front frames when recorded."""
    # Imported here: the GR00T runner imports so101-benchmark's package, which only `external` needs installed.
    from so101_bench_eval import from_lerobot, to_lerobot

    obs, _ = env.reset(seed=seed)
    driver.reset()
    info: dict[str, Any] = {"is_success": False}
    done, frames, step = False, [], 0
    while not done:
        if record and step % (4 * FRAME_EVERY) == 0:  # episodes run 1,800 steps
            frames.append(obs["pixels"]["front"])
        step += 1
        joints = to_lerobot(env.to_sim(obs["agent_pos"]))
        target = driver.act(obs["pixels"]["front"], obs["pixels"]["wrist"], joints, prompt)
        obs, _, terminated, truncated, info = env.step(env.from_sim(from_lerobot(target)))
        done = terminated or truncated
    return {"seed": seed, "success": bool(info.get("is_success")), "reason": info.get("failure_reason")}, frames


@app.command()
def external(
    task: str = "stack_blue_on_red",
    episodes: int = 50,
    benchmark: Path = SO101_BENCHMARK,
    native: bool = False,
    checkpoint: str = CHECKPOINT,
    out: Path = Path("logs/so101bench_flux3.json"),
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    wandb_group: str = "flux3",
    label: str = "flux3-so101bench",
) -> None:
    """so101-benchmark's own scene, from its checkout's config (needs its package importable)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from lerobot_env_so101.config import load_config  # so101-benchmark's package: only this command needs it
    from lerobot_env_so101.env import SO101Env

    cfg = load_config(benchmark / "configs/benchmark.yaml", {"sim": {"render_backend": "egl"}})
    env, driver, results = SO101Env(task, cfg.sim, cfg.task_paths), FluxDriver(checkpoint, bridge=not native), []
    examples: dict[str, list[np.ndarray]] = {}
    config = {"checkpoint": checkpoint, "task": task, "episodes": episodes}
    with Tracker("so101-benchmark", name=label, config=config, enabled=wandb_log, group=wandb_group) as tracker:
        for seed in range(episodes):
            result, frames = external_episode(driver, env, seed, env.task_description, record=len(examples) < 2)
            results.append(result)
            if frames:
                examples.setdefault("success" if result["success"] else "failure", frames)
            wins = sum(r["success"] for r in results)
            print(f"{task} {seed + 1}/{episodes}: {result} | {wins}", flush=True)
            out.write_text(json.dumps({"task": task, "policy": checkpoint, "episodes": results}, indent=1))
            tracker.run.log({f"so101bench/{task}_successes": wins, "episodes": seed + 1}, step=seed)
        tracker.log_rollouts(examples, f"so101-benchmark {task}")
    env.close()


if __name__ == "__main__":
    app()
