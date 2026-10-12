"""Train an RL expert on the twin to replace the scripted one (as in RLDG, arXiv 2412.09858).

DAgger and the demo pipeline are capped by the scripted expert: it always grasps top-down at the block centre and
cannot recover a dropped or held block, which is where the policy's remaining failures are. A PPO policy on privileged
state (`twin.rl_env.PickPlaceEnv`) can learn varied grasps and recoveries, trains on the CPU, and can then stand in for
the scripted expert when recording demos and recoveries.

On fresh scenes the scripted expert already stacks 100% of the time; from DAgger failure snapshots it recovers only
about 40% (85/207, 72/189, 59/146 over rounds 1-3). That is the bar. Training resets into rounds 1-2 snapshots and fresh
scenes; evaluation uses round 3's snapshots, never seen in training.

    baseline  the scripted expert's recovery rate on the evaluation snapshots
    train     PPO over parallel environments, evaluated on held-out snapshots and fresh scenes, to W&B
    evaluate  a saved policy on the same

Fail fast: if held-out recovery is not clearly above the scripted expert's within a day of CPU, stop.

    DAGGER_LOGS=~/nebius-physical-ai/logs/dagger uv run --extra rl python scripts/rl_expert.py baseline
    OMP_NUM_THREADS=1 uv run --extra rl python scripts/rl_expert.py train --residual --envs 12 --out ~/checkpoints/rl-expert-residual
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.tracking import Tracker
from nebius_physical_ai.twin.dagger import FailureSnapshot, record_recovery
from nebius_physical_ai.twin.rl_env import GAMMA, PickPlaceEnv
from nebius_physical_ai.twin.scene import TwinLayout

app = typer.Typer(add_completion=False, no_args_is_help=True)
EVAL_SEED = 10_000  # fresh evaluation scenes never seen in training (training envs use seeds 0..envs-1)
DAGGER = Path(os.environ.get("DAGGER_LOGS", "logs/dagger"))  # where scripts/dagger_round.sh writes rN/snapshots.jsonl


def snapshots(rounds: list[int]) -> list[FailureSnapshot]:
    return [s for r in rounds for s in FailureSnapshot.load(DAGGER / f"r{r}/snapshots.jsonl")]


def make_env(
    seed: int, hard: bool, pool: list[FailureSnapshot], shards: int = 1, residual: bool = False
) -> Callable[[], PickPlaceEnv]:
    """Env `seed` gets every `shards`-th snapshot: each compiles and keeps only its own share of the scenes."""
    share = pool[seed % shards :: shards]
    return lambda: PickPlaceEnv(seed=seed, hard=hard, snapshots=share, residual=residual)


@app.command()
def baseline(eval_round: int = 3) -> None:
    """The scripted expert's recovery rate on one round's failure snapshots."""
    held = snapshots([eval_round])
    wins = sum(record_recovery(s, TwinLayout(), image_size=(64, 64)) is not None for s in held)
    print(f"scripted expert recovers {wins}/{len(held)} ({wins / len(held):.0%}) of round {eval_round}'s failures")


def success_rate(
    model: PPO | None,
    stats: VecNormalize | None,
    episodes: int,
    hard: bool,
    eval_round: int | None = None,
    residual: bool = False,
) -> float:
    """Fresh held-out scenes, or with `eval_round` every snapshot of that round once. `model=None` acts with zeros:
    in residual mode that is the scripted expert alone, the baseline the residual must beat."""
    held = snapshots([eval_round]) if eval_round else []
    env = PickPlaceEnv(seed=EVAL_SEED, hard=hard, snapshots=held, snapshot_share=0.0, residual=residual)
    runs = range(len(held)) if held else range(episodes)
    wins = 0
    for i in runs:
        obs, _ = env.reset(options={"snapshot": i} if held else None)
        done = False
        while not done:
            if stats is not None:
                obs = stats.normalize_obs(obs)
            action = model.predict(obs, deterministic=True)[0] if model else np.zeros(env.action_space.shape)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        wins += info["is_success"]
    env.close()
    return wins / len(runs)


class EvalSuccess(BaseCallback):
    """Every `every` steps, the deterministic policy's success rate on held-out scenes, printed and logged."""

    def __init__(
        self, tracker: Tracker, every: int, episodes: int, hard: bool, out: Path, eval_round: int, residual: bool
    ) -> None:
        super().__init__()
        self.tracker, self.every, self.episodes, self.hard, self.out = tracker, every, episodes, hard, out
        self.eval_round, self.residual, self.best = eval_round, residual, -1.0
        self.next = every

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next:
            return True
        self.next += self.every
        stats = self.model.get_vec_normalize_env()
        fresh = success_rate(self.model, stats, self.episodes, self.hard, residual=self.residual)
        recover = success_rate(self.model, stats, 0, self.hard, self.eval_round, self.residual)
        print(f"step {self.num_timesteps}: fresh {fresh:.0%}, held-out recovery {recover:.0%}", flush=True)
        self.tracker.run.log({"eval/fresh_success": fresh, "eval/recovery": recover}, step=self.num_timesteps)
        if recover > self.best:  # keep the best held-out recovery, not the last
            self.best = recover
            self.model.save(self.out / "policy")
            stats.save(str(self.out / "vecnormalize.pkl"))
        return True

    def _on_rollout_end(self) -> None:
        recent = list(self.model.ep_info_buffer)
        if recent:
            self.tracker.run.log(
                {
                    "train/ep_return": float(np.mean([e["r"] for e in recent])),
                    "train/success": float(np.mean([e["is_success"] for e in recent])),
                },
                step=self.num_timesteps,
            )


@app.command()
def train(
    out: Annotated[Path, typer.Option(help="Policy and normalisation statistics land here")],
    steps: int = 20_000_000,
    envs: int = 12,
    hard: bool = False,
    eval_every: int = 500_000,
    eval_episodes: int = 50,
    train_rounds: str = "1,2",
    eval_round: int = 3,
    residual: Annotated[bool, typer.Option(help="Learn corrections on top of the scripted expert's plan")] = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rounds = [int(r) for r in train_rounds.split(",")]
    pool = snapshots(rounds)
    workers = SubprocVecEnv([make_env(i, hard, pool, envs, residual) for i in range(envs)])
    # VecMonitor gives PPO's log (and W&B) episode return, length and training success rate
    vec = VecNormalize(VecMonitor(workers, info_keywords=("is_success",)), norm_obs=True, norm_reward=True)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=1024,
        batch_size=4096,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [256, 256, 256], "log_std_init": -2.0 if residual else 0.0},
        verbose=1,
    )
    # a residual policy must start near zero: unit exploration noise would bury the expert's plan under corrections
    config = {"steps": steps, "envs": envs, "hard": hard, "algo": "PPO", "obs": "privileged state", "residual": residual,
              "train_rounds": rounds, "eval_round": eval_round}  # fmt: skip
    with Tracker(
        "rl-expert",
        name=f"rl-expert-{'residual' if residual else 'ppo'}{'-hard' if hard else ''}",
        config=config,
        group="rl-expert",
    ) as t:
        model.learn(steps, callback=EvalSuccess(t, eval_every, eval_episodes, hard, out, eval_round, residual))


@app.command()
def evaluate(policy: Path, episodes: int = 200, hard: bool = False, eval_round: int = 3) -> None:
    model = PPO.load(policy / "policy")
    stats = VecNormalize.load(
        str(policy / "vecnormalize.pkl"), SubprocVecEnv([make_env(0, hard, snapshots([eval_round]))])
    )
    stats.training = False
    fresh, recover = success_rate(model, stats, episodes, hard), success_rate(model, stats, 0, hard, eval_round)
    print(f"RL expert: fresh {fresh:.0%} on {episodes} scenes; recovers {recover:.0%} of round {eval_round}'s failures")


if __name__ == "__main__":
    app()
