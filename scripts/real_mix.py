"""Co-training replay: mix real SO-101 teleoperation back into a twin fine-tune.

Every fine-tune so far starts from hqfang/groot-so100_101 (pretrained on 1,209 real SO-100/101 datasets) and trains only
on twin demos with a narrow object set; held-out objects have not moved in any of them. Mixing real episodes back in
(co-training, as in Octo, pi0 and GR00T's own recipes) should keep the base's object generality while it learns the
twin. ArmnetBench (released after the base was trained) is the real set: 8 tasks, new objects (eye drops, tools, rings,
cables, blocks), 50 teleoperated demos each, front and wrist cameras like ours.

    convert  ArmnetBench episodes into the twin's dataset shape: front and wrist cameras centre-cropped to 640x480,
             20 fps resampled to 30, joints already in our units (`to_twin_units`); one video file is downloaded at a
             time and dropped when no later episode needs it
    check    where the gripper is when it starts closing, real against twin: a wrong unit mapping would put real
             grasps in the air or under the table
    mix      the twin demos repeated against the real set so real frames are `--real-share` of the training samples

    ~/lerobot-n17/.venv/bin/python scripts/real_mix.py convert ~/data/armnet_twin --like ~/data/preflight_demos_v4c_v30
"""

import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import av
import mujoco
import numpy as np
import pandas as pd
import typer
from huggingface_hub import hf_hub_download, snapshot_download
from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nebius_physical_ai.real_data import (
    ARMNET,
    armnet_episodes,
    closing_frames,
    fit_frame,
    interpolate,
    resample_positions,
    to_twin_units,
)
from nebius_physical_ai.twin.scene import ARM_JOINTS, SO101_XML, lerobot_to_qpos

ARMNET_CACHE = Path.home() / "data/armnet_raw"  # metadata (50 MB) plus the one video file in flight
CAMERAS = ("observation.images.front", "observation.images.wrist")
JOINTS = ("observation.state", "action")
app = typer.Typer(add_completion=False, no_args_is_help=True)


def video_name(episode: pd.Series, camera: str) -> str:
    chunk, file = episode[f"videos/{camera}/chunk_index"], episode[f"videos/{camera}/file_index"]
    return f"videos/{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def decode(path: Path, start: float, stop: float) -> Iterator[np.ndarray]:
    """RGB frames of one episode's segment of a packed video file."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(start / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t >= stop - 1e-3:
                break
            if t >= start - 1e-3:
                yield frame.to_ndarray(format="rgb24")


def episode_frames(
    episode: pd.Series, rows: pd.DataFrame, cache: Path, src_fps: int, fps: int
) -> Iterator[dict[str, object]]:
    """One ArmnetBench episode as twin frames, resampled to `fps`: joints interpolated, images nearest."""
    rows = rows.sort_values("frame_index")
    images = {}
    for camera in CAMERAS:
        start, stop = episode[f"videos/{camera}/from_timestamp"], episode[f"videos/{camera}/to_timestamp"]
        images[camera] = [fit_frame(f) for f in decode(cache / video_name(episode, camera), start, stop)]
    n = min(len(rows), *(len(v) for v in images.values()))
    positions = resample_positions(n, src_fps, fps)
    joints = {k: interpolate(to_twin_units(np.stack(rows[k].to_numpy()[:n])), positions) for k in JOINTS}
    nearest = np.minimum(np.round(positions).astype(int), n - 1)
    for i, j in enumerate(nearest):
        yield {k: joints[k][i].astype(np.float32) for k in JOINTS} | {c: images[c][j] for c in CAMERAS}


@app.command()
def convert(
    out: Annotated[Path, typer.Argument(help="Converted LeRobot v3.0 dataset")],
    like: Annotated[Path, typer.Option(help="A twin dataset whose features and fps to copy")],
    cache: Path = ARMNET_CACHE,
    rollouts: Annotated[bool, typer.Option(help="The policies' rollouts labelled successful")] = False,
    teleop: Annotated[bool, typer.Option(help="The 400 teleoperated demos")] = True,
    limit: Annotated[int | None, typer.Option(help="Only the first N selected episodes (a smoke test)")] = None,
) -> None:
    """ArmnetBench in the twin's dataset shape, one packed video file on disk at a time."""
    info = json.loads((like / "meta/info.json").read_text())
    features = {k: v for k, v in info["features"].items() if k in (*JOINTS, *CAMERAS)}
    snapshot_download(ARMNET, repo_type="dataset", local_dir=cache, allow_patterns=["meta/*", "data/*"])  # 50 MB
    src_fps = json.loads((cache / "meta/info.json").read_text())["fps"]
    episodes = pd.read_parquet(cache / "meta/episodes/chunk-000/file-000.parquet")
    chosen = episodes.set_index("episode_index").loc[armnet_episodes(episodes, rollouts, teleop)[:limit]].reset_index()
    data = pd.concat(pd.read_parquet(p) for p in (cache / "data").rglob("*.parquet"))
    rows = dict(iter(data.groupby("episode_index")))
    drop_after = defaultdict(list)  # episode position -> video files no later episode needs
    for name, last in {video_name(e, c): i for i, e in chosen.iterrows() for c in CAMERAS}.items():
        drop_after[last].append(name)
    if out.exists():
        shutil.rmtree(out)
    target = LeRobotDataset.create(
        f"local/{out.name}",
        fps=info["fps"],
        features=features,
        root=out,
        robot_type="so101_follower",
        image_writer_threads=8,
    )
    for i, episode in chosen.iterrows():
        for camera in CAMERAS:
            hf_hub_download(ARMNET, video_name(episode, camera), repo_type="dataset", local_dir=cache)
        task = str(episode.tasks[0])
        for frame in episode_frames(episode, rows[episode.episode_index], cache, src_fps, info["fps"]):
            target.add_frame(frame | {"task": task})
        target.save_episode()
        for name in drop_after[i]:
            (cache / name).unlink(missing_ok=True)
        print(f"{i + 1}/{len(chosen)} episode {episode.episode_index}: {task}", flush=True)
    target.finalize()
    # Encoding adds per-feature keys the twin's info lacks, and merging insists the feature dicts match exactly.
    info_path = out / "meta/info.json"
    written = json.loads(info_path.read_text())
    assert {k: v["shape"] for k, v in written["features"].items()} == {
        k: v["shape"] for k, v in info["features"].items()
    }
    info_path.write_text(json.dumps({**written, "features": info["features"]}, indent=4))
    print(f"wrote {out}: {target.meta.total_episodes} episodes, {target.meta.total_frames} frames")


def episode_grasps(root: Path, episodes: int | None = None) -> dict[int, np.ndarray]:
    """Per episode, the gripper-frame positions (base frame, metres) where its gripper starts closing, through the twin's
    kinematics: where the recorded joints would put the gripper on the twin."""
    model = mujoco.MjModel.from_xml_path(str(SO101_XML))
    data, site = mujoco.MjData(model), model.site("gripperframe").id
    limits = np.array([model.joint(j).range for j in ARM_JOINTS])
    adr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    frames = pd.concat(
        pd.read_parquet(p, columns=["episode_index", "observation.state"]) for p in root.rglob("data/**/*.parquet")
    )
    grasps = {}
    for index, rows in list(frames.groupby("episode_index"))[:episodes]:
        state, points = np.stack(rows["observation.state"].to_numpy()), []
        for i in closing_frames(state[:, 5]):
            data.qpos[adr] = lerobot_to_qpos(state[i], limits)
            mujoco.mj_kinematics(model, data)
            points.append(data.site_xpos[site].copy())
        grasps[int(index)] = np.array(points).reshape(-1, 3)
    return grasps


def grasp_points(root: Path, episodes: int | None = None) -> np.ndarray:
    return np.concatenate(list(episode_grasps(root, episodes).values()))


@app.command()
def check(
    real: Annotated[Path, typer.Argument(help="Converted real dataset")],
    twin: Annotated[Path, typer.Argument(help="Twin demos")],
    episodes: int = 100,
) -> None:
    """Grasp-onset positions of real and twin demos side by side; fails when their median heights differ by over 5 cm."""
    heights = {}
    for name, root in (("real", real), ("twin", twin)):
        p = grasp_points(root, episodes)
        q = np.percentile(p, [10, 50, 90], axis=0).round(3)
        print(f"{name}: {len(p)} grasp onsets; x, y, z (m) at 10/50/90%: {q[:, 0]}, {q[:, 1]}, {q[:, 2]}")
        heights[name] = q[1, 2]
    if abs(heights["real"] - heights["twin"]) > 0.05:
        print(f"median grasp height real {heights['real']} m vs twin {heights['twin']} m: the unit mapping looks wrong")
        raise typer.Exit(1)


@app.command("filter")
def keep_in_workspace(
    real: Annotated[Path, typer.Argument(help="Converted real dataset")],
    twin: Annotated[Path, typer.Argument(help="Twin demos whose grasp band the real demos must fall in")],
    out: Annotated[Path, typer.Argument(help="The real demos that pass")],
    margin_xy: Annotated[float, typer.Option(help="Metres the band is widened by in reach and sideways")] = 0.03,
    margin_z: Annotated[float, typer.Option(help="Metres the band is widened by in height")] = 0.02,
) -> None:
    """Keep the real demos whose every grasp, through the twin's kinematics, lands inside the twin's own grasp band.

    ArmnetBench does not record which farm arm ran an episode, so calibration can't be corrected per arm. Instead the
    twin decides: a grasp under the table or out of reach is a calibration or workspace mismatch, and its demo is
    dropped rather than teaching the policy joint targets the twin never needs. The band is the twin demos' 2-98%
    grasp range, widened by the margins; demos that never close the gripper are dropped too."""
    lo, hi = np.percentile(grasp_points(twin, 400), [2, 98], axis=0)
    margin = np.array([margin_xy, margin_xy, margin_z])
    lo, hi = lo - margin, hi + margin
    grasps = episode_grasps(real)
    keep = [i for i, g in grasps.items() if len(g) and ((g >= lo) & (g <= hi)).all()]
    print(f"band x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}, z {lo[2]:.3f}..{hi[2]:.3f} m")
    source = LeRobotDataset(f"local/{real.name}", root=real)
    tasks = pd.Series([source.meta.episodes[i]["tasks"][0] for i in keep]).value_counts()
    print(f"kept {len(keep)}/{len(grasps)} demos; by task: {tasks.to_dict()}")
    if out.exists():
        shutil.rmtree(out)
    drop = sorted(set(grasps) - set(keep))
    delete_episodes(source, drop, output_dir=out, repo_id=f"local/{out.name}")


@app.command()
def mix(
    twin: Annotated[Path, typer.Argument(help="Twin demos")],
    real: Annotated[Path, typer.Argument(help="Converted real dataset")],
    out: Annotated[Path, typer.Argument(help="Merged training set")],
    real_share: Annotated[float, typer.Option(help="Real frames' share of the training samples")] = 0.25,
    extra: Annotated[list[Path] | None, typer.Option(help="More converted real datasets (e.g. the rollouts)")] = None,
) -> None:
    """Twin demos repeated against the real sets so real frames are about `real_share` of the merged set."""
    t = LeRobotDataset("local/twin", root=twin)
    reals = [LeRobotDataset(f"local/real_{i}", root=root) for i, root in enumerate([real, *(extra or [])])]
    real_frames = sum(r.meta.total_frames for r in reals)
    copies = max(1, round(real_frames * (1 - real_share) / (real_share * t.meta.total_frames)))
    merged = merge_datasets([*[t] * copies, *reals], f"local/{out.name}", output_dir=out, concatenate_videos=False)
    episodes = sum(r.meta.total_episodes for r in reals)
    share = real_frames / merged.meta.total_frames
    print(f"{out}: twin x{copies} + {episodes} real episodes = {merged.meta.total_frames} frames, real {share:.0%}")


if __name__ == "__main__":
    app()
