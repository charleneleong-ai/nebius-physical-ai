"""What a recorded demo dataset contains, in a form a run log can show: counts, prompts, sample videos and frames."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from nebius_physical_ai.twin.dataset import CAMERAS, LeRobotWriter
from nebius_physical_ai.twin.rollout import STACK_TASK


@dataclass(frozen=True)
class DemoReport:
    """A LeRobot v2.1 dataset written by `npai demos`."""

    root: Path
    episodes: list[dict[str, Any]]  # episodes.jsonl rows: episode_index, tasks, length

    @classmethod
    def load(cls, root: Path) -> "DemoReport":
        path = root / "meta/episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: is {root} a recorded demo dataset?")
        return cls(root, [json.loads(line) for line in path.read_text().splitlines() if line.strip()])

    def prompt(self, episode: dict[str, Any]) -> str:
        return str(episode["tasks"][0])

    def prompts(self) -> Counter[str]:
        return Counter(self.prompt(e) for e in self.episodes)

    def summary(self) -> dict[str, Any]:
        """Episode and frame counts, how many distinct prompts, and the share still on the original prompt."""
        lengths = [int(e["length"]) for e in self.episodes]
        prompts = self.prompts()
        return {
            "episodes": len(self.episodes),
            "frames": sum(lengths),
            "mean_episode_frames": round(float(np.mean(lengths)), 1) if lengths else 0.0,
            "distinct_prompts": len(prompts),
            "original_prompt_share": round(prompts[STACK_TASK] / len(self.episodes), 3) if self.episodes else 0.0,
        }

    def sample(self, n: int) -> list[dict[str, Any]]:
        """`n` episodes evenly spread over the dataset (all of them when it has fewer), first and last included."""
        if n >= len(self.episodes):
            return list(self.episodes)
        picks = np.linspace(0, len(self.episodes) - 1, max(n, 1)).round().astype(int)
        return [self.episodes[i] for i in dict.fromkeys(picks.tolist())]

    def video_path(self, episode: dict[str, Any], camera: str = "front") -> Path:
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; the dataset has {sorted(CAMERAS)}")
        index = int(episode["episode_index"])
        return self.root / LeRobotWriter.VIDEO_PATH.format(
            episode_chunk=index // 1000, video_key=CAMERAS[camera], episode_index=index
        )

    def first_frame(self, episode: dict[str, Any], camera: str = "front") -> np.ndarray:
        """The episode's first video frame as an RGB array."""
        with av.open(str(self.video_path(episode, camera))) as container:
            return next(container.decode(video=0)).to_ndarray(format="rgb24")
