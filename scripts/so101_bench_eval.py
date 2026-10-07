"""Score a served Preflight policy on the external so101-benchmark (github.com/Fatsharkovo/so101-benchmark).

Its scene is the same MuJoCo SO-101 as our twin, with joints in degrees; our policies speak LeRobot units, mapped
linearly onto the menagerie joint ranges. This drives its gym environment with our `Gr00tClient` and converts both
ways, so any server `scripts/finetune.py serve` or `scripts/lerobot_policy_server.py` runs can be scored unchanged.
Runs in the benchmark's own environment, from its checkout:

    cd ~/so101-benchmark && .venv/bin/python ~/nebius-physical-ai/scripts/so101_bench_eval.py \\
        --port 5558 --task stack_blue_on_red --episodes 50 --out results_v5.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from lerobot_env_so101.config import load_config
from lerobot_env_so101.env import SO101Env

# The wire format and client are kept in one place: the project's own module, which needs only msgpack/zmq/numpy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.policy.gr00t import Gr00tClient, Observation

# Menagerie SO-101 joint ranges (radians) the twin maps LeRobot units onto: arm joints -> [-100, 100], gripper -> [0, 100].
OUR_LIMITS = np.array(
    [
        [-1.91986, 1.91986],
        [-1.7453293, 1.7453293],
        [-1.69, 1.69],
        [-1.658063, 1.658063],
        [-2.7438473, 2.8412063],
        [-0.174533, 1.7453292],
    ]
)
app = typer.Typer(add_completion=False)


def to_lerobot(q: np.ndarray) -> np.ndarray:
    lo, hi = OUR_LIMITS.T
    unit = (q - lo) / (hi - lo)
    return np.concatenate([unit[:5] * 200 - 100, unit[5:] * 100]).astype(np.float32)


def from_lerobot(joints: np.ndarray) -> np.ndarray:
    lo, hi = OUR_LIMITS.T
    unit = np.concatenate([(joints[:5] + 100) / 200, joints[5:] / 100])
    return lo + np.clip(unit, 0, 1) * (hi - lo)


def run_episode(env: SO101Env, policy: Gr00tClient, seed: int, horizon: int, prompt: str) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    info: dict[str, Any] = {"is_success": False}
    done, started = False, time.time()
    while not done:
        joints = to_lerobot(env.to_sim(obs["agent_pos"]))
        observation = Observation(obs["pixels"]["front"], obs["pixels"]["wrist"], joints, prompt)
        for target in policy.act_batch([observation])[0][:horizon]:
            obs, _, terminated, truncated, info = env.step(env.from_sim(from_lerobot(target)))
            done = terminated or truncated
            if done:
                break
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "reason": info.get("failure_reason"),
        "steps": env.step_index,
        "seconds": round(time.time() - started, 1),
    }


@app.command()
def main(
    port: int = 5558,
    task: str = "stack_blue_on_red",
    episodes: int = 50,
    seed: int = 0,
    horizon: Annotated[int, typer.Option(help="Steps of each action chunk executed before asking again")] = 16,
    config: Annotated[Path, typer.Option(help="The benchmark's run config (scene randomisation)")] = Path(
        "configs/benchmark.yaml"
    ),
    prompt: Annotated[str | None, typer.Option(help="Override the task's own instruction")] = None,
    out: Path = Path("results.json"),
) -> None:
    cfg = load_config(config, {"sim": {"render_backend": "egl"}})
    env = SO101Env(task, cfg.sim, cfg.task_paths)
    instruction = prompt or env.task_description
    results = []
    with Gr00tClient(port=port) as policy:
        for i in range(episodes):
            results.append(run_episode(env, policy, seed + i, horizon, instruction))
            wins = sum(r["success"] for r in results)
            print(f"{task} {i + 1}/{episodes}: {results[-1]} | {wins}/{len(results)}", flush=True)
            out.write_text(json.dumps({"task": task, "prompt": instruction, "episodes": results}, indent=1))
    env.close()
    wins = sum(r["success"] for r in results)
    print(f"{task}: {wins}/{episodes} ({wins / episodes:.0%}) with prompt {instruction!r}")


if __name__ == "__main__":
    app()
