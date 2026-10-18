"""DiVeR-style verifier for best-of-N action chunks (arXiv 2610.04933; `policy/verifier.py` has the method).

`collect` rolls the served policy out on fresh training scenes (never the benchmark's) with the server drawing N chunks
per step but executing the first (`--samples 8 --select first`), so the rollouts are the policy's own, and saves each
episode's pooled features, joints, candidates and outcome. `train` fits the verifier on them: every executed step is
labelled with its episode's outcome and weighted by how much that step's candidates disagreed.

    # policy server: lerobot_policy_server.py <ckpt> --port 5566 --samples 8 --select first
    uv run python scripts/diver.py collect ~/data/diver_r3 --port 5566 --scenes 120 --variants 2
    ~/lerobot-n17/.venv/bin/python scripts/diver.py train ~/data/diver_r3 ~/checkpoints/diver-r3/verifier.pt
    # then serve with --samples 8 --select verifier --verifier ~/checkpoints/diver-r3/verifier.pt and benchmark
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.policy.best_of_n import candidate_features, criticality, decision_weights

REPO = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def collect(
    out: Annotated[Path, typer.Argument(help="One .npz per episode lands here")],
    port: int = 5566,
    scenes: int = 120,
    variants: int = 2,
    batch_size: int = 8,
    seed: int = 7000,  # away from the DAgger harvests (900 + round) and the benchmark (42)
) -> None:
    """Roll the policy out and save every episode's decisions with its outcome."""
    from nebius_physical_ai.cli import build_randomizer  # the twin stack, only needed here (not in `train`'s venv)
    from nebius_physical_ai.policy.gr00t import Gr00tClient
    from nebius_physical_ai.twin.dagger import Harvester
    from nebius_physical_ai.twin.expert import SceneSampler
    from nebius_physical_ai.twin.scene import TwinLayout

    sampler = SceneSampler(TwinLayout(), seed=seed)
    randomizer = build_randomizer(
        sampler, seed, REPO / "data/capture/real_objects.json", REPO / "configs/front_oblique_camera.json",
        "0.3,0.6,0.1", True, 0.35, 2,
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    with Gr00tClient(port=port, info=True) as policy:
        harvester = Harvester(policy, randomizer, sampler, batch_size=batch_size, variants=variants)
        rollouts = [r for scene in range(scenes) for r in harvester.draw(scene)]
        wins = 0
        for start in range(0, len(rollouts), batch_size):
            batch = rollouts[start : start + batch_size]
            for k, (_, res, _) in enumerate(harvester.rollouts(batch, record_decisions=True)):
                d = res.decisions
                np.savez_compressed(
                    out / f"ep{start + k:05d}.npz", features=np.stack([x["features"] for x in d]),
                    joints=np.stack([x["joints"] for x in d]), candidates=np.stack([x["candidates"] for x in d]),
                    chosen=np.array([x["chosen"] for x in d]), expert=np.stack([x["expert"] for x in d]),
                    success=res.success, failure=res.failure,
                )  # fmt: skip
                wins += res.success
            print(f"{start + len(batch)}/{len(rollouts)} episodes, {wins} succeeded", flush=True)


@app.command("collect-real")
def collect_real(
    real: Annotated[Path, typer.Argument(help="Converted ArmnetBench episodes with labels.json (real_mix.py convert)")],
    out: Annotated[Path, typer.Argument(help="One .npz per episode lands here")],
    port: int = 5568,
    stride: Annotated[int, typer.Option(help="Frames between decisions: the server's --horizon, as in the twin")] = 16,
    batch_size: int = 8,
) -> None:
    """Real outcomes for the verifier: every `stride` frames of each real episode the served policy draws its candidates
    for that real observation, and the chunk the robot actually executed next is stored as `executed`, labelled with
    the episode's human-judged outcome. Real episodes carry no action-expert state for what they executed, so they train
    the observation-and-chunk verifier only. Runs in the LeRobot venv, which reads the dataset's videos."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # LeRobot venv only

    from nebius_physical_ai.policy.gr00t import Gr00tClient, Observation

    labels = json.loads((real / "labels.json").read_text())
    dataset = LeRobotDataset(f"local/{real.name}", root=real)
    out.mkdir(parents=True, exist_ok=True)
    with Gr00tClient(port=port, info=True) as policy:
        for e, label in enumerate(labels):
            first, last = (int(dataset.meta.episodes[e][k]) for k in ("dataset_from_index", "dataset_to_index"))
            actions = np.stack([dataset.hf_dataset[i]["action"] for i in range(first, last)]).astype(np.float32)
            rows = []
            for start in range(first, last, stride * batch_size):
                frames = [dataset[i] for i in range(start, min(start + stride * batch_size, last), stride)]
                policy.act_batch([Observation(
                    front=(f["observation.images.front"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                    wrist=(f["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                    joints=f["observation.state"].numpy(), task=f["task"],
                ) for f in frames])  # fmt: skip
                info, horizon = policy.last_info, policy.last_info["candidates"].shape[2]
                for k, f in enumerate(frames):
                    t = int(f["index"]) - first
                    chunk = actions[np.minimum(np.arange(t, t + horizon), len(actions) - 1)]
                    rows.append((info["features"][k], f["observation.state"].numpy(), info["candidates"][k], chunk))
            np.savez_compressed(
                out / f"real{e:05d}.npz", features=np.stack([r[0] for r in rows]), joints=np.stack([r[1] for r in rows]),
                candidates=np.stack([r[2] for r in rows]), executed=np.stack([r[3] for r in rows]),
                success=label["success"], policy=label["policy"],
            )  # fmt: skip
            print(f"{e + 1}/{len(labels)} real episodes ({label['policy']}, success {label['success']})", flush=True)


def load(root: Path, beta: float, expert: bool = False) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """Per episode: executed-candidate features (T, F), outcome, DiVeR weights (T,). With `expert`, each step's
    features end with the executed candidate's action-expert state. Real episodes (`collect-real`) store the chunk the
    robot executed rather than an index into the candidates, and have no expert state."""
    episodes = []
    for path in sorted(root.glob("*.npz")):
        ep = np.load(path)
        candidates = ep["candidates"]
        if "executed" in ep:
            if expert:
                continue
            executed, state = ep["executed"][:, None], None
        else:
            steps, chosen = np.arange(len(ep["chosen"])), ep["chosen"]
            executed = candidates[steps, chosen][:, None]  # (T, 1, H, 6)
            state = ep["expert"][steps, chosen][:, None].astype(np.float32) if expert else None  # (T, 1, E)
        x = candidate_features(ep["features"].astype(np.float32), ep["joints"], executed, state)[:, 0]
        episodes.append((x, int(ep["success"]), decision_weights(criticality(candidates), beta)))
    return episodes


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random success step outscores a random failure step (Mann-Whitney U from ranks)."""
    from scipy.stats import rankdata  # in the LeRobot venv that `train` runs in

    ranks, n_pos = rankdata(scores), int(labels.sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos)))


Episode = tuple[np.ndarray, int, np.ndarray]


def split_episodes(episodes: list[Episode], val_share: float, seed: int) -> tuple[list[Episode], list[Episode]]:
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_val = int(len(episodes) * val_share)
    return [episodes[i] for i in order[n_val:]], [episodes[i] for i in order[:n_val]]


def to_arrays(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps (x, y, weight), the weights rescaled so successes and failures carry equal total weight."""
    x = np.concatenate([e[0] for e in episodes])
    y = np.concatenate([np.full(len(e[0]), e[1], dtype=np.float32) for e in episodes])
    w = np.concatenate([e[2] for e in episodes]).astype(np.float32)
    return x, y, w * np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))


@app.command()
def train(
    data: Annotated[Path, typer.Argument(help="From `collect`")],
    out: Annotated[Path, typer.Argument(help="Verifier checkpoint (.pt)")],
    beta: Annotated[float, typer.Option(help="DiVeR weight temperature; 0 = uniform outcome classification")] = 1.0,
    expert: Annotated[bool, typer.Option(help="Add each candidate's action-expert state (needs it collected)")] = False,
    extra: Annotated[
        list[Path] | None, typer.Option(help="More episodes for training only, e.g. collect-real's")
    ] = None,
    epochs: int = 40,
    val_share: float = 0.2,
    seed: int = 0,
) -> None:
    """Fit the verifier with decision-weighted, class-balanced BCE; report held-out episodes' step AUC."""
    import torch

    from nebius_physical_ai.policy.verifier import Verifier

    episodes = load(data, beta, expert)
    train_eps, val_eps = split_episodes(episodes, val_share, seed)  # held-out AUC stays on `data`'s own episodes
    train_eps += [e for root in extra or [] for e in load(root, beta, expert)]
    x, y, w = (torch.as_tensor(a) for a in to_arrays(train_eps))
    vx, vy, _ = to_arrays(val_eps)
    verifier = Verifier(x.shape[1], expert=expert).fit_normaliser(x)
    optimiser = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(epochs):
        verifier.train()
        for batch in torch.randperm(len(x)).split(512):
            optimiser.zero_grad()
            loss = (loss_fn(verifier(x[batch]), y[batch]) * w[batch]).sum() / w[batch].sum()
            loss.backward()
            optimiser.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_auc = auc(verifier.eval()(torch.as_tensor(vx)).numpy(), vy)
            print(f"epoch {epoch + 1}: train loss {loss.item():.3f}, held-out step AUC {val_auc:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    verifier.eval().save(out)
    summary = {
        "episodes": {"train": len(train_eps), "val": len(val_eps)}, "extra": [str(p) for p in extra or []],
        "successes": int(sum(e[1] for e in episodes)), "beta": beta, "expert": expert, "val_step_auc": val_auc,
    }  # fmt: skip
    (out.parent / "train.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    app()
