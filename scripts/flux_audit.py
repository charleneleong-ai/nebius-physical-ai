"""Audit FLUX 3 Action without rolling it out: the demos it trains on, and the future it imagines.

LeRobot training is offline imitation, so nothing acts during it and W&B gets only loss curves. Two things can still be
looked at:

    demos    an even sample of a dataset's episodes as front and wrist videos with their prompts, to check what the
             model is fed (the v8 set's last 1,200 episodes carry the Seeed wrist mount and textured floors)
    imagine  FLUX denoises the next 42 video frames together with its actions; decoding them beside the real next
             frames of held-out demos shows whether it understands the scene before any rollout
    bridge   whether the data bridge loses anything: our demo actions mapped into FLUX units, turned into the deltas
             its processor normalises (and clips at 6), and mapped back, against the originals
    openloop the model's first action on held-out demo frames against the demo's own, beside holding still, so a
             closed-loop failure can be told apart from a model that never learned the mapping

    ~/lerobot-flux/.venv/bin/python scripts/flux_audit.py demos ~/data/preflight_demos_v8_v30 --name v8
    ~/lerobot-flux/.venv/bin/python scripts/flux_audit.py imagine ~/data/preflight_demos_v8_v30 \\
        --checkpoint ~/checkpoints/flux3-lora-v8/checkpoints/last/pretrained_model --name flux3-lora-v8
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import wandb
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import modeling_flux3

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from nebius_physical_ai.policy.flux3 import QuantileBridge
from nebius_physical_ai.tracking import Tracker

CAMERAS = ("observation.images.front", "observation.images.wrist")
app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_flux_eval() -> Any:
    """scripts/flux_eval.py as a module, for its FluxDriver (one implementation of the FLUX tick loop)."""
    spec = importlib.util.spec_from_file_location("flux_eval", REPO / "scripts/flux_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rgb(tensor: torch.Tensor) -> np.ndarray:
    """A (3, H, W) float image in [0, 1] as (H, W, 3) uint8."""
    return (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def episode_range(dataset: LeRobotDataset, episode: int) -> tuple[int, int]:
    meta = dataset.meta.episodes[episode]
    return int(meta["dataset_from_index"]), int(meta["dataset_to_index"])


def evenly(n: int, total: int) -> list[int]:
    return sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})


@app.command()
def demos(
    root: Annotated[Path, typer.Argument(help="A LeRobot v3.0 dataset")],
    name: str = "demos",
    samples: int = 12,
    every: Annotated[int, typer.Option(help="Keep one frame in this many")] = 4,
    group: str = "flux3",
) -> None:
    """Front and wrist videos of evenly spaced episodes, captioned with episode index and prompt."""
    dataset = LeRobotDataset(f"local/{root.name}", root=root)
    total = dataset.meta.total_episodes
    with Tracker("demos", name=f"{name}-demos", config={"root": str(root), "episodes": total}, group=group) as t:
        logged = {}
        for episode in evenly(samples, total):
            start, stop = episode_range(dataset, episode)
            frames = [dataset[i] for i in range(start, stop, every)]
            task = frames[0]["task"]
            for camera in CAMERAS:
                video = [rgb(f[camera]) for f in frames]
                caption = f"episode {episode}/{total} | {camera.split('.')[-1]} | {task}"
                logged[f"demo/{episode:05d}_{camera.split('.')[-1]}"] = Tracker.video(video, caption, fps=30 // every)
            print(f"episode {episode}: {stop - start} frames, {task!r}", flush=True)
        t.run.log(logged)


class Imagination:
    """Keeps the video latents FLUX denoises alongside its actions, which its sampler otherwise discards. The SO-101
    checkpoint samples with Euler, so that is the sampler wrapped."""

    def __init__(self) -> None:
        self.ids: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None
        self.wrap_ids, self.wrap_sampler = modeling_flux3.batched_prc_vid, modeling_flux3.sampling.euler

        def ids(*args: Any, **kwargs: Any) -> Any:
            tokens, ids_ = self.wrap_ids(*args, **kwargs)
            self.ids = ids_
            return tokens, ids_

        def sampler(*args: Any, **kwargs: Any) -> Any:
            out = self.wrap_sampler(*args, **kwargs)
            self.tokens = out["x_video"]
            return out

        modeling_flux3.batched_prc_vid, modeling_flux3.sampling.euler = ids, sampler

    def decode(self, vae: Any) -> np.ndarray:
        """The imagined clip as (T, H, W, 3) uint8: the scene and wrist cameras side by side, as FLUX composes them."""
        inner = getattr(vae, "module", vae)  # LeRobot's VideoVAE wraps the model that knows token ids
        video = inner.decode_from_latent_with_ids(
            self.tokens[0].to(torch.bfloat16), self.ids[0].to(self.tokens.device)
        )  # (3, T, H, W)
        return ((video.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()


def canvas(f: dict[str, Any], hw: tuple[int, int]) -> np.ndarray:
    """The real frame composed like FLUX's canvas: scene left, wrist right, each resized to the half-canvas."""
    halves = [torch.nn.functional.interpolate(f[c][None], size=hw, mode="bilinear")[0] for c in CAMERAS]
    return rgb(torch.cat(halves, dim=2))


@app.command()
def imagine(
    root: Annotated[Path, typer.Argument(help="A LeRobot v3.0 dataset in our units (not the bridged copy)")],
    checkpoint: str = "black-forest-labs/flux-3-action-so101",
    name: str = "flux3",
    episodes: int = 6,
    at: Annotated[float, typer.Option(help="Where in each episode to predict from, as a fraction")] = 0.25,
    group: str = "flux3",
) -> None:
    """Real next frames above FLUX's imagined ones, for evenly spaced episodes (later ones hold the Seeed mount)."""
    driver = load_flux_eval().FluxDriver(checkpoint)
    imagination, vae = Imagination(), driver.policy.frozen.video_vae
    dataset = LeRobotDataset(f"local/{root.name}", root=root)
    total, chunk = dataset.meta.total_episodes, driver.policy.config.chunk_size
    config = {"checkpoint": checkpoint, "root": str(root), "at": at}
    with Tracker("imagination", name=f"{name}-imagine", config=config, group=group) as t:
        logged = {}
        for episode in evenly(episodes, total):
            start, stop = episode_range(dataset, episode)
            now = start + int(at * (stop - start - chunk))
            f = dataset[now]
            driver.reset()
            driver.act(rgb(f[CAMERAS[0]]), rgb(f[CAMERAS[1]]), f["observation.state"].numpy(), f["task"])
            imagined = imagination.decode(vae)
            hw = (imagined.shape[1], imagined.shape[2] // 2)
            step = max(1, chunk // len(imagined))  # the VAE compresses time: one decoded frame per few ticks
            real = [canvas(dataset[min(now + i * step, stop - 1)], hw) for i in range(len(imagined))]
            clip = [np.concatenate([r, i], axis=0) for r, i in zip(real, imagined)]
            caption = f"episode {episode} from frame {now - start} | top real, bottom imagined | {f['task']}"
            logged[f"imagine/{episode:05d}"] = Tracker.video(clip, caption, fps=max(1, 30 // step))
            error = float(np.mean([np.abs(r.astype(int) - i.astype(int)).mean() for r, i in zip(real, imagined)]))
            logged[f"imagine_l1/{episode:05d}"] = error
            print(f"episode {episode}: {len(imagined)} frames, mean L1 {error:.1f}/255", flush=True)
        t.run.log(logged)
        t.run.summary["imagine_l1_mean"] = float(np.mean([v for k, v in logged.items() if k.startswith("imagine_l1")]))


def checkpoint_stats_path() -> Path:
    return Path(hf_hub_download(load_flux_eval().CHECKPOINT, "dataset_statistics.json"))


def episode_actions(dataset: LeRobotDataset, episode: int) -> np.ndarray:
    start, stop = episode_range(dataset, episode)
    return np.stack(dataset.hf_dataset[start:stop]["action"]).astype(np.float64)


@app.command()
def bridge(
    root: Annotated[Path, typer.Argument(help="A LeRobot v3.0 dataset in our units")],
    episodes: int = 200,
) -> None:
    """Round-trip the demo actions through the bridge and FLUX's delta normalisation; report coverage and loss."""
    path = checkpoint_stats_path()
    action = json.loads(path.read_text())["action"]
    lo, span, clip = np.array(action["q01"]), np.array(action["q99"]) - np.array(action["q01"]), action["clip"]
    unit = QuantileBridge.for_checkpoint(path)
    dataset = LeRobotDataset(f"local/{root.name}", root=root)
    outside, clipped, errors = [], [], []
    for episode in evenly(episodes, dataset.meta.total_episodes):
        ours = episode_actions(dataset, episode)
        theirs = unit.to_theirs(ours)
        delta = np.diff(theirs, axis=0)
        delta[:, -1] = theirs[1:, -1]  # the gripper stays absolute
        z = 2 * (delta - lo) / span - 1
        outside.append(np.abs(z) > 1)
        clipped.append(np.abs(z) > clip)
        rebuilt_delta = (np.clip(z, -clip, clip) + 1) * span / 2 + lo
        rebuilt = np.vstack([theirs[:1], theirs[:1] + np.cumsum(rebuilt_delta, axis=0)])
        rebuilt[1:, -1] = rebuilt_delta[:, -1]
        errors.append(np.abs(unit.to_ours(rebuilt) - ours).max(axis=0))
    outside, clipped = np.concatenate(outside), np.concatenate(clipped)
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
    print("per joint: share of steps outside the checkpoint's 1-99% action range | clipped at 6 | max round-trip error")
    for j, n in enumerate(names):
        print(f"  {n:11s} {outside[:, j].mean():6.1%} | {clipped[:, j].mean():6.2%} | {np.max(errors, axis=0)[j]:.4f}")


JOINT_NAMES = ("pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper")


def chunk_errors(predicted: np.ndarray, demo: np.ndarray, state: np.ndarray, horizons: list[int]) -> np.ndarray:
    """(2, len(horizons), 6): |error| of the model's h-th action and of holding `state`, against the demo's h-th."""
    return np.stack(
        [
            np.abs(np.stack([x[h - 1] for h in horizons]) - demo[np.array(horizons) - 1])
            for x in (predicted, np.broadcast_to(state, demo.shape))
        ]
    )


def latex_table(horizons: list[int], model: np.ndarray, hold: np.ndarray, caption: str) -> str:
    """A booktabs table: mean |action error| per joint at each horizon, model above holding still."""
    head = " & ".join(["Horizon", "Policy", *JOINT_NAMES, "Mean"])
    rows = []
    for i, h in enumerate(horizons):
        for label, e in (("FLUX LoRA", model[i]), ("Hold pose", hold[i])):
            cells = " & ".join(f"{v:.2f}" for v in (*e, e.mean()))
            rows.append(f"{h if label == 'FLUX LoRA' else ''} & {label} & {cells} \\\\")
        rows.append("\\midrule" if i < len(horizons) - 1 else "")
    body = "\n".join(r for r in rows if r)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{ll"
        + "r" * 7
        + "}\n\\toprule\n"
        + head
        + " \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n"
        + f"\\caption{{{caption}}}\n\\label{{tab:flux-openloop}}\n\\end{{table}}\n"
    )


def trajectory_figure(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> Any:
    """One row per sample, one column per joint: the demo's next actions, FLUX's chunk, and the held pose."""
    fig, axes = plt.subplots(len(samples), 6, figsize=(15, 2.2 * len(samples)), squeeze=False, sharex=True)
    for row, (predicted, demo, state, label) in zip(axes, samples):
        steps = np.arange(1, len(demo) + 1)
        for j, ax in enumerate(row):
            ax.plot(steps, demo[:, j], color="black", lw=1.5, label="demo")
            ax.plot(steps, predicted[:, j], color="tab:blue", lw=1.2, label="FLUX LoRA")
            ax.axhline(state[j], color="tab:gray", ls="--", lw=1, label="hold")
            ax.set_title(f"{JOINT_NAMES[j]}" if row is axes[0] else "", fontsize=9)
        row[0].set_ylabel(label, fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("steps ahead (30 Hz)", fontsize=8)
    fig.tight_layout()
    return fig


def horizon_figure(horizons: list[int], model: np.ndarray, hold: np.ndarray) -> Any:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(horizons, model.mean(axis=1), marker="o", label="FLUX LoRA")
    ax.plot(horizons, hold.mean(axis=1), marker="s", ls="--", color="tab:gray", label="hold pose")
    ax.set_xlabel("steps ahead (30 Hz)")
    ax.set_ylabel("mean |action error| (LeRobot units)")
    ax.legend()
    fig.tight_layout()
    return fig


@app.command()
def openloop(
    root: Annotated[Path, typer.Argument(help="A LeRobot v3.0 dataset in our units")],
    checkpoint: str = "black-forest-labs/flux-3-action-so101",
    first_episode: Annotated[int, typer.Option(help="Held-out episodes start here (not in the training set)")] = 800,
    samples: int = 60,
    horizons: Annotated[str, typer.Option(help="Steps ahead to score, within one executed chunk")] = "1,8,16,32",
    name: str = "flux3",
    group: str = "flux3",
    out: Annotated[Path, typer.Option(help="PDF figures and the LaTeX table land here")] = REPO / "docs/figures",
) -> None:
    """FLUX's predicted chunk on held-out demo frames against the demo's next actions, beside holding the pose, at
    several horizons. The first action is a tick ahead, where holding still is nearly exact; later ones separate a
    model that learned the task from one that copies its input. FLUX conditions on its last `n_obs_steps` frames, so
    those are fed from the demo first (without them it sees an arm at rest and starts late). Logged to W&B with trajectory plots; a PDF figure and
    a booktabs table are written for the report."""
    steps = sorted(int(h) for h in horizons.split(","))
    driver = load_flux_eval().FluxDriver(checkpoint)
    history = driver.policy.config.n_obs_steps
    dataset = LeRobotDataset(f"local/{root.name}", root=root)
    held = list(range(first_episode, dataset.meta.total_episodes))
    rng = np.random.default_rng(0)
    errors, shown = [], []
    for episode in rng.choice(held, size=min(samples, len(held)), replace=False):
        start, stop = episode_range(dataset, int(episode))
        if stop - start <= steps[-1] + 1:
            continue
        t = int(rng.integers(min(start + history, stop - steps[-1] - 1), stop - steps[-1]))
        f = dataset[t]
        state = f["observation.state"].numpy()
        demo = np.stack(dataset.hf_dataset[t : t + steps[-1]]["action"]).astype(np.float64)
        driver.reset()
        for i in range(max(start, t - history + 1), t):  # the frames FLUX conditions on, with the targets sent then
            g = dataset[i]
            driver.observe(
                rgb(g[CAMERAS[0]]), rgb(g[CAMERAS[1]]), g["observation.state"].numpy(), g["task"], g["action"].numpy()
            )
        # the same frame each tick: every action comes from the one chunk predicted at t
        predicted = np.stack(
            [driver.act(rgb(f[CAMERAS[0]]), rgb(f[CAMERAS[1]]), state, f["task"]) for _ in range(steps[-1])]
        )
        errors.append(chunk_errors(predicted, demo, state, steps))
        if len(shown) < 4:
            shown.append((predicted, demo, state, f"ep {episode}, t={t - start}"))
    model, hold = np.mean(errors, axis=0)  # (horizons, 6) each
    print(f"{checkpoint}: mean |action error| per joint (LeRobot units) on {len(errors)} held-out frames")
    for i, h in enumerate(steps):
        print(
            f"  {h:>2} ahead  model {np.round(model[i], 2)} mean {model[i].mean():.2f} | hold {np.round(hold[i], 2)} mean {hold[i].mean():.2f}"
        )
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "openloop_trajectories": trajectory_figure(shown),
        "openloop_horizons": horizon_figure(steps, model, hold),
    }
    for stem, fig in figures.items():
        fig.savefig(out / f"{stem}_{name}.pdf")
        fig.savefig(out / f"{stem}_{name}.png", dpi=150)
    caption = (
        f"Open-loop action error of {name} on {len(errors)} held-out demo frames: mean absolute error per joint "
        "(LeRobot units) of the action $h$ steps ahead in one predicted chunk, against holding the current pose."
    )
    (out / f"openloop_{name}.tex").write_text(latex_table(steps, model, hold, caption))
    config = {"checkpoint": checkpoint, "root": str(root), "samples": len(errors), "horizons": steps}
    with Tracker("openloop", name=f"{name}-openloop", config=config, group=group) as t:
        t.run.log({stem: wandb.Image(str(out / f"{stem}_{name}.png")) for stem in figures})
        cols = ["horizon", "policy", *JOINT_NAMES, "mean"]
        rows = [
            [h, p, *map(float, e), float(e.mean())]
            for i, h in enumerate(steps)
            for p, e in (("flux", model[i]), ("hold", hold[i]))
        ]
        metrics = {
            f"openloop/{p}_h{h}": float(e[i].mean())
            for i, h in enumerate(steps)
            for p, e in (("model", model), ("hold", hold))
        }
        t.log_summary(metrics, {"openloop": (cols, rows)})
    print(f"wrote {out}/openloop_*_{name}.pdf and openloop_{name}.tex")


if __name__ == "__main__":
    app()
