import itertools
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import GRIP_OPEN, REST_POSE, Frame, PickPlaceExpert, SceneSampler, TopDownIK
from nebius_physical_ai.twin.rollout import STACK_TASK, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout

A3 = TwinLayout()
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


@pytest.fixture
def sampler() -> SceneSampler:
    return SceneSampler(A3, seed=0)


def test_sampler_keeps_cubes_in_workspace_and_apart(sampler: SceneSampler) -> None:
    for _ in range(50):
        blue, red = sampler.sample()
        for obj in (blue, red):
            x, _ = A3.mat_to_world(obj.x_mm, obj.y_mm)
            assert 0.16 <= x <= 0.22
        assert np.hypot(blue.x_mm - red.x_mm, blue.y_mm - red.y_mm) >= 60


@needs_model
class TestExpert:
    def test_ik_reaches_top_down_targets(self, sampler: SceneSampler) -> None:
        twin = Twin.from_objects(sampler.sample(), A3)
        ik = TopDownIK(twin.model)
        for target in ([0.16, 0.09, 0.015], [0.22, -0.09, 0.075]):
            _, err = ik.solve(np.array(target), REST_POSE)
            assert err < 0.002

    def test_expert_stacks_most_sampled_scenes(self, sampler: SceneSampler) -> None:
        wins = 0
        for _ in range(8):
            with Twin.from_objects(sampler.sample(), A3, start_pose=sampler.start_pose()) as twin:
                frames = list(PickPlaceExpert(twin, STACK_TASK, image_size=(24, 32)).run())
                twin.settle(0.5)
                wins += StackCheck()(twin)
        assert frames[0].obs.front.shape == (24, 32, 3) and frames[0].obs.task == STACK_TASK
        assert wins >= 7

    @pytest.mark.parametrize(("kind", "seed"), [("miss", 0), ("drop", 1)])
    def test_a_staged_mistake_is_left_out_and_the_demo_recovers_from_it(self, kind: str, seed: int) -> None:
        sampler = SceneSampler(A3, seed=seed)
        with Twin.from_objects(sampler.sample(), A3, start_pose=sampler.start_pose()) as twin:
            expert = PickPlaceExpert(twin, STACK_TASK, image_size=(24, 32))
            start = twin.block_pos("blue").copy()
            expert.stage(kind, sampler.rng)
            staged, moved_mm = twin.arm_qpos(), 1000 * np.linalg.norm(twin.block_pos("blue")[:2] - start[:2])
            frames = list(expert.run())
            twin.settle(0.5)
            assert StackCheck()(twin)
            # The recording opens where the mistake left the arm, not at rest: only the recovery is recorded.
            np.testing.assert_allclose(frames[0].obs.joints, twin.to_lerobot(staged), atol=1e-4)
        if kind == "miss":
            assert staged[5] < GRIP_OPEN / 2  # jaws closed on nothing
        else:
            assert moved_mm > 10  # let go on the way, away from where it started

    def test_an_unknown_mistake_is_rejected(self, sampler: SceneSampler) -> None:
        with Twin.from_objects(sampler.sample(), A3) as twin, pytest.raises(ValueError, match="unknown mistake"):
            PickPlaceExpert(twin, STACK_TASK).mistake("wave", np.random.default_rng(0))


def test_writer_produces_readable_lerobot_v21(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, (48, 64, 3), np.uint8)
    frames = [Frame(Observation(img(), img(), rng.normal(size=6), "task a"), rng.normal(size=6)) for _ in range(12)]
    writer = LeRobotWriter(tmp_path, image_size=(48, 64))
    writer.add_episode(frames, "task a")
    writer.add_episode(frames[:5], "task a")
    writer.finalize()

    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (2, 17, 1)
    table = pq.read_table(tmp_path / "data/chunk-000/episode_000001.parquet").to_pydict()
    assert table["index"] == list(range(12, 17)) and set(table["task_index"]) == {0}
    np.testing.assert_allclose(table["action"][0], frames[0].action, rtol=1e-6)
    with av.open(str(tmp_path / "videos/chunk-000/observation.images.front/episode_000000.mp4")) as c:
        decoded = list(c.decode(video=0))
    assert len(decoded) == 12 and decoded[0].height == 48
    assert json.loads((tmp_path / "meta/modality.json").read_text())["state"]["gripper"] == {"start": 5, "end": 6}


@pytest.mark.parametrize(
    ("layout", "blocks"),
    [
        ("stock", {"single_arm": (0, 5), "gripper": (5, 6)}),
        ("split", {"single_arm": (0, 4), "wrist_roll": (4, 5), "gripper": (5, 6)}),
    ],
)
def test_modality_json_covers_all_six_joints_once_in_either_layout(layout: str, blocks: dict, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (48, 64, 3), np.uint8)
    frames = [Frame(Observation(img, img, rng.normal(size=6), "t"), rng.normal(size=6)) for _ in range(3)]
    writer = LeRobotWriter(tmp_path, image_size=(48, 64), layout=layout)
    writer.add_episode(frames, "t")
    writer.finalize()
    written = json.loads((tmp_path / "meta/modality.json").read_text())
    for part in ("state", "action"):
        assert {k: (v["start"], v["end"]) for k, v in written[part].items()} == blocks
        spans = sorted((v["start"], v["end"]) for v in written[part].values())
        assert spans[0][0] == 0 and spans[-1][1] == 6 and all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert set(written["video"]) == {"front", "wrist"} and written["annotation"] == {
        "human.task_description": {"original_key": "task_index"}
    }


def test_an_unknown_layout_is_rejected_by_the_writer_and_the_demos_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from nebius_physical_ai.cli import app

    writer = LeRobotWriter(tmp_path / "a", layout="wide")
    with pytest.raises(ValueError, match="unknown layout"):
        writer.finalize()
    result = CliRunner().invoke(app, ["demos", str(tmp_path / "b"), "--layout", "wide"])
    assert result.exit_code == 2 and "--layout must be one of" in " ".join(result.output.replace("│", " ").split())
