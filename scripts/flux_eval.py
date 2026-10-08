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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from nebius_physical_ai.benchmark import SEED, Cell, Suite, build_suites
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.flux3 import QuantileBridge
from nebius_physical_ai.twin.rehearsal import (
    REHEARSAL_IMAGE_SIZE,
    REHEARSAL_MAX_STEPS,
    Perturbation,
    Variant,
)
from nebius_physical_ai.twin.rollout import CONTROL_HZ, FailureWatch, StackCheck
from nebius_physical_ai.twin.scene import Twin, TwinLayout, TwinRenderer

CHECKPOINT = "black-forest-labs/flux-3-action-so101"
SO101_BENCHMARK = Path.home() / "so101-benchmark"
app = typer.Typer(add_completion=False, no_args_is_help=True)


class FluxDriver:
    """FLUX 3 Action one tick at a time: images and joints in, the next joint target out, in our units."""

    def __init__(self, checkpoint: str = CHECKPOINT, bridge: bool = True, device: str = "cuda") -> None:
        self.policy = Flux3Policy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        stats = Path(hf_hub_download(checkpoint, "dataset_statistics.json"))
        self.bridge = QuantileBridge.for_checkpoint(stats) if bridge else None
        self.device = device

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @staticmethod
    def image(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255

    def act(self, scene: np.ndarray, wrist: np.ndarray, joints: np.ndarray, task: str) -> np.ndarray:
        state = self.bridge.to_theirs(joints) if self.bridge else joints
        batch = {
            "observation.images.scene": self.image(scene),
            "observation.images.wrist": self.image(wrist),
            "observation.state": torch.tensor(state, dtype=torch.float32)[None],
            "task": [task],
        }
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        target = action[0].float().cpu().numpy()
        return self.bridge.to_ours(target) if self.bridge else target


def rollout(driver: FluxDriver, twin: Twin, check: StackCheck, prompt: str, renderer: TwinRenderer) -> str:
    """Empty string on success, else how it failed (FailureWatch)."""
    driver.reset()
    watch = FailureWatch(twin, check)
    for _ in range(REHEARSAL_MAX_STEPS):
        obs = twin.observe(prompt, REHEARSAL_IMAGE_SIZE, renderer)
        twin.step_towards(twin.from_lerobot(driver.act(obs.front, obs.wrist, obs.joints, prompt)), 1 / CONTROL_HZ)
        if check(twin):
            return ""
        watch.update(twin)
    return watch.verdict(twin)


def score_suite(driver: FluxDriver, suite: Suite, layout: TwinLayout) -> Cell:
    """Every case's variants drawn exactly as `Rehearsal.run` draws them, so outcomes pair with GR00T's."""
    start, per_case, outcomes, failures = time.time(), [], [], Counter()
    for case in suite.cases:
        case_layout, rng = case.layout(layout), np.random.default_rng(SEED)
        variants = [Variant.sample(case.objects, Perturbation(), rng) for _ in range(suite.variants)]
        template = Twin.from_objects(case.objects, case_layout)
        renderer = TwinRenderer(template.model, REHEARSAL_IMAGE_SIZE)
        try:
            for v in variants:
                twin = template.variant(v.objects, case_layout, v.physics, v.start_pose)
                failure = rollout(driver, twin, StackCheck(case.pick, case.place), case.prompt, renderer)
                outcomes.append(not failure)
                failures.update([failure] if failure else [])
        finally:
            renderer.close()
        per_case.append(sum(outcomes[-len(variants) :]))
        print(f"  {suite.name} {case.name}: {per_case[-1]}/{len(variants)}", flush=True)
    seconds = round(time.time() - start, 1)
    return Cell(suite.name, suite.group, sum(per_case), suite.n, len(suite.cases), seconds, per_case, outcomes,
                dict(failures.most_common()))  # fmt: skip


@app.command()
def suites(
    label: Annotated[str, typer.Argument(help="Name for the result file, e.g. flux3-so101")],
    only: Annotated[list[str] | None, typer.Option(help="Only suites matching this glob; repeatable")] = None,
    group: str = "all",
    native: Annotated[bool, typer.Option(help="Feed our units unmapped (no QuantileBridge)")] = False,
    out: Path = Path("logs/bench_flux3.json"),
) -> None:
    layout = TwinLayout()
    real = [DetectedObject(**o) for o in json.loads((REPO / "data/capture/real_objects.json").read_text())]
    driver, cells = FluxDriver(bridge=not native), []
    for suite in build_suites(real, layout, group, only or ()):
        cells.append(score_suite(driver, suite, layout))
        c = cells[-1]
        print(f"{label} | {c.suite}: {c.successes}/{c.n} failures {c.failures}", flush=True)
        out.write_text(json.dumps({"label": label, "cells": [cell.record() for cell in cells]}, indent=1))


def external_episode(driver: FluxDriver, env: Any, seed: int, prompt: str) -> dict[str, Any]:
    """One so101-benchmark episode, one FLUX tick per environment step."""
    # Imported here: the GR00T runner imports so101-benchmark's package, which only `external` needs installed.
    from so101_bench_eval import from_lerobot, to_lerobot

    obs, _ = env.reset(seed=seed)
    driver.reset()
    info: dict[str, Any] = {"is_success": False}
    done = False
    while not done:
        joints = to_lerobot(env.to_sim(obs["agent_pos"]))
        target = driver.act(obs["pixels"]["front"], obs["pixels"]["wrist"], joints, prompt)
        obs, _, terminated, truncated, info = env.step(env.from_sim(from_lerobot(target)))
        done = terminated or truncated
    return {"seed": seed, "success": bool(info.get("is_success")), "reason": info.get("failure_reason")}


@app.command()
def external(
    task: str = "stack_blue_on_red",
    episodes: int = 50,
    benchmark: Path = SO101_BENCHMARK,
    native: bool = False,
    out: Path = Path("logs/so101bench_flux3.json"),
) -> None:
    """so101-benchmark's own scene, from its checkout's config (needs its package importable)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from lerobot_env_so101.config import load_config  # so101-benchmark's package: only this command needs it
    from lerobot_env_so101.env import SO101Env

    cfg = load_config(benchmark / "configs/benchmark.yaml", {"sim": {"render_backend": "egl"}})
    env, driver, results = SO101Env(task, cfg.sim, cfg.task_paths), FluxDriver(bridge=not native), []
    for seed in range(episodes):
        results.append(external_episode(driver, env, seed, env.task_description))
        print(f"{task} {seed + 1}/{episodes}: {results[-1]} | {sum(r['success'] for r in results)}", flush=True)
        out.write_text(json.dumps({"task": task, "policy": CHECKPOINT, "episodes": results}, indent=1))
    env.close()


if __name__ == "__main__":
    app()
