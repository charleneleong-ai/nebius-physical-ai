"""Write meta/episodes_stats.jsonl for a LeRobot v2.1 dataset recorded by `npai demos`, which LeRobot's v2.1 -> v3.0
converter needs and our writer never produced.

    uv run python scripts/episode_stats.py ~/data/preflight_demos_v4c_v30

Numeric columns get exact per-episode stats; each video gets per-channel stats in [0, 1] from a few evenly spaced
frames (enough for image normalisation, and GR00T's processor normalises images itself).
"""

import json
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import pandas as pd
import typer
from rich.progress import track

app = typer.Typer(add_completion=False)


def numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = values.reshape(len(values), -1).astype(np.float64)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def video_stats(path: Path, samples: int) -> dict[str, Any]:
    """Per-channel stats over `samples` evenly spaced frames, RGB in [0, 1], shaped (3, 1, 1) as LeRobot stores them."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0)
    cap.release()
    if not frames:
        raise ValueError(f"no frame decoded from {path}")
    pixels = np.concatenate(frames)

    def channel(values: np.ndarray) -> list[list[list[float]]]:
        return [[[float(v)]] for v in values]

    return {
        "min": channel(pixels.min(0)),
        "max": channel(pixels.max(0)),
        "mean": channel(pixels.mean(0)),
        "std": channel(pixels.std(0)),
        "count": [len(frames)],
    }


@app.command()
def main(
    root: Annotated[Path, typer.Argument(help="LeRobot v2.1 dataset root")],
    samples: Annotated[int, typer.Option(help="Frames sampled per video")] = 4,
) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    videos = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    numeric = [k for k, f in info["features"].items() if f["dtype"] not in ("video", "image", "string")]
    chunk_size = info["chunks_size"]
    lines = []
    for episode in track(range(info["total_episodes"]), description="episode stats"):
        chunk = episode // chunk_size
        frame = pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=episode))
        stats = {k: numeric_stats(np.stack(frame[k].to_numpy())) for k in numeric if k in frame}
        for key in videos:
            path = root / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode)
            stats[key] = video_stats(path, samples)
        lines.append(json.dumps({"episode_index": episode, "stats": stats}))
    (root / "meta/episodes_stats.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote stats for {len(lines)} episodes to {root / 'meta/episodes_stats.jsonl'}")


if __name__ == "__main__":
    app()
