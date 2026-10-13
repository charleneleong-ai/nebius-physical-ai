"""Serve a LeRobot GR00T N1.7 checkpoint (e.g. hqfang/groot-so100_101) behind the Isaac-GR00T ZMQ protocol.

The rehearsal and benchmark talk to `Gr00tClient`; this answers it with a LeRobot policy instead, so a LeRobot-format
checkpoint is scored exactly like our own. Runs in a LeRobot >= 0.6 environment (not the project's), from any directory:

    ~/lerobot-n17/.venv/bin/python scripts/lerobot_policy_server.py ~/checkpoints/hqfang-groot-so100_101 --port 5562

Our front and wrist frames go to camera_0 and camera_1; joints stay in LeRobot units (arm in [-100, 100], gripper in
[0, 100]), the SO-101 defaults the checkpoint's datasets were recorded in. Each reply is the first `--horizon` steps of
the 40-step chunk, the length our rollouts execute. `--samples N` draws N chunks per observation and returns one, chosen
by `--select`: their consensus (`best_of_n.medoid_index`), the first sample (the policy as is, for collecting verifier data),
or the highest-scoring under a learned `--verifier` (`policy.verifier`). With N > 1 the reply's info carries each row's
pooled vision-language features, its candidates and the chosen index, which `scripts/diver.py collect` records.
"""

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
import zmq
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy

# The wire format is kept in one place: the project's own client module, importable without its heavy deps.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.policy.best_of_n import candidate_features, medoid_index
from nebius_physical_ai.policy.gr00t import MsgCodec
from nebius_physical_ai.policy.verifier import Verifier

ACTION_KEYS = {"single_arm": (0, 5), "gripper": (5, 6)}
Selector = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None  # (N,B,T,6), (B,D), (B,6) -> (B,) index


def first(drawn: np.ndarray, _features: np.ndarray, _joints: np.ndarray) -> np.ndarray:
    """The policy as is: the first sample (how verifier training data is collected)."""
    return np.zeros(drawn.shape[1], dtype=int)


def consensus(drawn: np.ndarray, _features: np.ndarray, _joints: np.ndarray) -> np.ndarray:
    return medoid_index(drawn)


def scored_by(verifier: Verifier) -> Selector:
    def pick(drawn: np.ndarray, features: np.ndarray, joints: np.ndarray) -> np.ndarray:
        return verifier.pick(candidate_features(features, joints, drawn.transpose(1, 0, 2, 3)))

    return pick


app = typer.Typer(add_completion=False)


class LeRobotGr00t:
    """One LeRobot GR00T policy with its saved processors, answering batches in Isaac-GR00T's layout."""

    def __init__(self, checkpoint: Path, horizon: int, cameras: int, samples: int = 1, select: Selector = None) -> None:
        os.chdir(checkpoint)  # the saved processor names its VLM processor by a path relative to the checkpoint
        self.policy = GrootPolicy.from_pretrained(str(checkpoint)).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint))
        self.horizon, self.cameras, self.samples = horizon, cameras, samples
        self.select = select or first
        self.rows, self.pooled = 0, torch.empty(0)
        self.policy._groot_model.backbone.register_forward_hook(self.pool)

    def pool(self, _module: torch.nn.Module, _inputs: object, output: Any) -> None:
        """Keep the vision-language features of one copy per observation, mean-pooled over the attended tokens."""
        features, mask = output["backbone_features"][: self.rows].float(), output.get("backbone_attention_mask")
        weights = torch.ones(features.shape[:2], device=features.device) if mask is None else mask[: self.rows].float()
        self.pooled = (features * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)

    def sample(self, front: np.ndarray, wrist: np.ndarray, joints: np.ndarray, task: str) -> dict[str, Any]:
        views = [front, wrist, front, wrist][: self.cameras]
        sample: dict[str, Any] = {"observation.state": torch.as_tensor(joints, dtype=torch.float32), "task": task}
        for i, view in enumerate(views):
            sample[f"observation.images.camera_{i}"] = torch.as_tensor(view).permute(2, 0, 1).float() / 255
        return sample

    def batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Samples stacked into one batch (B, ...), tasks as a list, as LeRobot's dataloader would hand them over."""
        stacked = {k: torch.stack([s[k] for s in samples]) for k in samples[0] if k != "task"}
        return stacked | {"task": [s["task"] for s in samples]}

    @torch.no_grad()
    def act(
        self, observation: dict[str, Any], info: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Actions in Isaac-GR00T's layout; with `info`, each row's pooled features, candidates and chosen index too."""
        video, state = observation["video"], observation["state"]
        joints = np.concatenate([state[k] for k in ACTION_KEYS], axis=-1)[:, -1]  # (B, 6), last time step
        tasks = observation["language"]["annotation.human.task_description"]
        samples = [
            self.sample(video["front"][b, -1], video["wrist"][b, -1], joints[b], tasks[b][0])
            for b in range(len(joints))
        ]
        # every sample repeated `self.samples` times: each copy draws its own flow-matching noise, so its own chunk
        self.rows = len(samples)
        chunk = self.post(self.policy.predict_action_chunk(self.pre(self.batch(samples * self.samples))))
        drawn = np.asarray(chunk.float().cpu()).reshape(self.samples, len(samples), -1, 6)[:, :, : self.horizon]
        features = self.pooled.cpu().numpy()  # copies of one observation share their features
        chosen = self.select(drawn, features, joints) if self.samples > 1 else np.zeros(len(samples), dtype=int)
        actions = drawn[chosen, np.arange(len(samples))].astype(np.float32)
        extras = {
            "features": features.astype(np.float16), "candidates": drawn.transpose(1, 0, 2, 3).astype(np.float32),
            "chosen": chosen.astype(np.int64),
        } if info else {}  # fmt: skip
        return {key: actions[..., lo:hi] for key, (lo, hi) in ACTION_KEYS.items()}, extras


def reply(model: LeRobotGr00t, request: dict[str, Any]) -> Any:
    endpoint = request.get("endpoint")
    if endpoint == "ping":
        return {"status": "ok"}
    if endpoint == "get_modality_config":
        return {"action": {"modality_keys": list(ACTION_KEYS)}}
    if endpoint == "get_action":
        data = request["data"]
        return list(model.act(data["observation"], bool((data.get("options") or {}).get("info"))))
    return {"error": f"unknown endpoint {endpoint!r}"}


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Argument(help="Local LeRobot GR00T checkpoint directory")],
    port: int = 5562,
    horizon: Annotated[int, typer.Option(help="Steps of each 40-step chunk to return")] = 16,
    cameras: Annotated[int, typer.Option(help="Views fed: front and wrist, repeated to fill 3 or 4 slots")] = 4,
    samples: Annotated[int, typer.Option(help="Chunks drawn per observation; `--select` picks the one returned")] = 1,
    select: Annotated[str, typer.Option(help="medoid | first | verifier")] = "medoid",
    verifier: Annotated[
        Path | None, typer.Option(help="Trained verifier (scripts/diver.py train) for --select verifier")
    ] = None,
) -> None:
    if select == "verifier" and verifier is None:
        raise typer.BadParameter("--select verifier needs --verifier")
    selectors = {"first": lambda: first, "medoid": lambda: consensus,
                 "verifier": lambda: scored_by(Verifier.load(verifier.expanduser().resolve(), "cuda"))}  # fmt: skip
    model = LeRobotGr00t(checkpoint.expanduser().resolve(), horizon, cameras, samples, selectors[select]())
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")
    print(
        f"serving {checkpoint} on port {port} (horizon {horizon}, {cameras} camera slots, best of {samples} by {select})",
        flush=True,
    )
    while True:
        raw, request, start = socket.recv(), {}, time.time()
        try:  # a request that won't decode is answered too: an unanswered REQ client hangs until its timeout
            request = MsgCodec.unpack(raw)
            answer = reply(model, request)
        except Exception as error:  # noqa: BLE001 - report any failure to the client, keep serving
            answer = {"error": f"{type(error).__name__}: {error}"}
        socket.send(MsgCodec.pack(answer))
        if request.get("endpoint") == "get_action":
            print(f"get_action in {time.time() - start:.2f}s", flush=True)


if __name__ == "__main__":
    app()
