import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nebius_physical_ai.policy.gr00t import CAMERA_KEYS, LAYOUTS
from nebius_physical_ai.twin.expert import Frame
from nebius_physical_ai.twin.rollout import CONTROL_HZ
from nebius_physical_ai.twin.scene import ARM_JOINTS, FULL_IMAGE_SIZE

JOINT_NAMES = [f"{j}.pos" for j in ARM_JOINTS]
CAMERAS = {name: f"observation.images.{name}" for name in CAMERA_KEYS}


def modality(layout: str = "stock") -> dict[str, Any]:
    """The dataset's meta/modality.json for a joint layout: `stock` matches Isaac-GR00T examples/SO100/modality.json,
    `split` gives the wrist roll a block of its own, as the community SO-101 checkpoints expect."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
    blocks = {key: {"start": start, "end": end} for key, start, end in LAYOUTS[layout].blocks}
    return {
        "state": blocks,
        "action": dict(blocks),
        "video": {name: {"original_key": key} for name, key in CAMERAS.items()},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


MODALITY = modality()


@dataclass
class LeRobotWriter:
    """Writes a LeRobot v2.1 dataset (parquet per episode, H.264 video per camera) as Isaac-GR00T n1d6 expects.

    Run `python gr00t/data/stats.py <root> NEW_EMBODIMENT` in the GR00T env afterwards to add stats.
    """

    root: Path
    robot_type: str = "so101_follower"
    layout: str = "stock"  # joint layout written to modality.json (see `modality`)
    image_size: tuple[int, int] = FULL_IMAGE_SIZE
    tasks: list[str] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    total_frames: int = 0

    DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    def episode_path(self, template: str, index: int, **kw: str) -> Path:
        path = self.root / template.format(episode_chunk=index // 1000, episode_index=index, **kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks.append(task)
        return self.tasks.index(task)

    def add_episode(self, frames: list[Frame], task: str) -> None:
        index, n = len(self.episodes), len(frames)
        table = pa.table(
            {
                "observation.state": [f.obs.joints.astype(np.float32) for f in frames],
                "action": [f.action.astype(np.float32) for f in frames],
                "timestamp": np.arange(n, dtype=np.float32) / CONTROL_HZ,
                "frame_index": np.arange(n),
                "episode_index": np.full(n, index),
                "index": np.arange(self.total_frames, self.total_frames + n),
                "task_index": np.full(n, self.task_index(task)),
            }
        )
        pq.write_table(table, self.episode_path(self.DATA_PATH, index))
        for name, key in CAMERAS.items():
            self.write_video(
                [getattr(f.obs, name) for f in frames], self.episode_path(self.VIDEO_PATH, index, video_key=key)
            )
        self.episodes.append({"episode_index": index, "tasks": [task], "length": n})
        self.total_frames += n

    @staticmethod
    def write_video(images: list[np.ndarray], path: Path, fps: int = CONTROL_HZ) -> None:
        """RGB (H, W, 3) frames as an H.264 mp4."""
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = images[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            for image in images:
                container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
            container.mux(stream.encode())

    def info(self) -> dict[str, Any]:
        h, w = self.image_size
        joints = {"dtype": "float32", "names": JOINT_NAMES, "shape": [len(JOINT_NAMES)]}
        video = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": CONTROL_HZ,
                "video.channels": 3,
                "has_audio": False,
            },
        }
        scalar = {"shape": [1], "names": None}
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": len(self.episodes) * len(CAMERAS),
            "total_chunks": len(self.episodes) // 1000 + 1,
            "chunks_size": 1000,
            "fps": CONTROL_HZ,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": self.DATA_PATH,
            "video_path": self.VIDEO_PATH,
            "features": {
                "action": joints,
                "observation.state": joints,
                **{key: video for key in CAMERAS.values()},
                "timestamp": {"dtype": "float32", **scalar},
                "frame_index": {"dtype": "int64", **scalar},
                "episode_index": {"dtype": "int64", **scalar},
                "index": {"dtype": "int64", **scalar},
                "task_index": {"dtype": "int64", **scalar},
            },
        }

    def finalize(self) -> None:
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(json.dumps(self.info(), indent=4))
        (meta / "modality.json").write_text(json.dumps(modality(self.layout), indent=4))
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.episodes))
        (meta / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(self.tasks))
        )
