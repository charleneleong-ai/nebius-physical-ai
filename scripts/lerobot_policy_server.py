"""Serve a LeRobot GR00T N1.7 checkpoint (e.g. hqfang/groot-so100_101) behind the Isaac-GR00T ZMQ protocol.

The rehearsal and benchmark talk to `Gr00tClient`; this answers it with a LeRobot policy instead, so a LeRobot-format
checkpoint is scored exactly like our own. Runs in a LeRobot >= 0.6 environment (not the project's), from any directory:

    ~/lerobot-n17/.venv/bin/python scripts/lerobot_policy_server.py ~/checkpoints/hqfang-groot-so100_101 --port 5562

Our front and wrist frames go to camera_0 and camera_1; joints stay in LeRobot units (arm in [-100, 100], gripper in
[0, 100]), the SO-101 defaults the checkpoint's datasets were recorded in. Each reply is the first `--horizon` steps of
the 40-step chunk, the length our rollouts execute.
"""

import os
import sys
import time
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
from nebius_physical_ai.policy.gr00t import MsgCodec

ACTION_KEYS = {"single_arm": (0, 5), "gripper": (5, 6)}
app = typer.Typer(add_completion=False)


class LeRobotGr00t:
    """One LeRobot GR00T policy with its saved processors, answering batches in Isaac-GR00T's layout."""

    def __init__(self, checkpoint: Path, horizon: int, cameras: int) -> None:
        os.chdir(checkpoint)  # the saved processor names its VLM processor by a path relative to the checkpoint
        self.policy = GrootPolicy.from_pretrained(str(checkpoint)).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint))
        self.horizon, self.cameras = horizon, cameras

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
    def act(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        video, state = observation["video"], observation["state"]
        joints = np.concatenate([state[k] for k in ACTION_KEYS], axis=-1)[:, -1]  # (B, 6), last time step
        tasks = observation["language"]["annotation.human.task_description"]
        samples = [
            self.sample(video["front"][b, -1], video["wrist"][b, -1], joints[b], tasks[b][0])
            for b in range(len(joints))
        ]
        chunk = self.post(self.policy.predict_action_chunk(self.pre(self.batch(samples))))
        actions = np.asarray(chunk.float().cpu()).reshape(len(samples), -1, 6)[:, : self.horizon].astype(np.float32)
        return {key: actions[..., lo:hi] for key, (lo, hi) in ACTION_KEYS.items()}


def reply(model: LeRobotGr00t, request: dict[str, Any]) -> Any:
    endpoint = request.get("endpoint")
    if endpoint == "ping":
        return {"status": "ok"}
    if endpoint == "get_modality_config":
        return {"action": {"modality_keys": list(ACTION_KEYS)}}
    if endpoint == "get_action":
        return [model.act(request["data"]["observation"]), {}]
    return {"error": f"unknown endpoint {endpoint!r}"}


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Argument(help="Local LeRobot GR00T checkpoint directory")],
    port: int = 5562,
    horizon: Annotated[int, typer.Option(help="Steps of each 40-step chunk to return")] = 16,
    cameras: Annotated[int, typer.Option(help="Views fed: front and wrist, repeated to fill 3 or 4 slots")] = 4,
) -> None:
    model = LeRobotGr00t(checkpoint.expanduser().resolve(), horizon, cameras)
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")
    print(f"serving {checkpoint} on port {port} (horizon {horizon}, {cameras} camera slots)", flush=True)
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
