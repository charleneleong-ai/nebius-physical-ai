import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from typer.testing import CliRunner

from nebius_physical_ai.demo_report import DemoReport
from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import Frame
from nebius_physical_ai.twin.rollout import STACK_TASK

SCRIPT = Path(__file__).parent.parent / "scripts" / "log_demos.py"
PROMPTS = [STACK_TASK, "Pick green can and place on red box", STACK_TASK, "Pick black block and place on pomade tin"]
COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]


def episode_frames(colour: tuple[int, int, int], n: int = 8) -> list[Frame]:
    """A solid-colour front camera, so a decoded frame can be told apart from every other episode's."""
    front = np.full((48, 64, 3), colour, np.uint8)
    wrist = np.zeros((48, 64, 3), np.uint8)
    return [Frame(Observation(front, wrist, np.zeros(6), "t"), np.zeros(6)) for _ in range(n)]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    writer = LeRobotWriter(tmp_path / "demos", image_size=(48, 64))
    for colour, prompt in zip(COLOURS, PROMPTS):
        writer.add_episode(episode_frames(colour), prompt)
    writer.finalize()
    return tmp_path / "demos"


class TestDemoReport:
    def test_the_summary_counts_episodes_frames_prompts_and_the_original_share(self, dataset: Path) -> None:
        summary = DemoReport.load(dataset).summary()
        assert summary == {
            "episodes": 4,
            "frames": 32,
            "mean_episode_frames": 8.0,
            "distinct_prompts": 3,
            "original_prompt_share": 0.5,
        }

    def test_prompts_are_counted_per_episode(self, dataset: Path) -> None:
        assert DemoReport.load(dataset).prompts() == {STACK_TASK: 2, PROMPTS[1]: 1, PROMPTS[3]: 1}

    def test_a_sample_is_spread_over_the_dataset_with_first_and_last_included(self, dataset: Path) -> None:
        report = DemoReport.load(dataset)
        assert [e["episode_index"] for e in report.sample(2)] == [0, 3]
        assert [e["episode_index"] for e in report.sample(3)] == [0, 2, 3]
        assert [e["episode_index"] for e in report.sample(10)] == [0, 1, 2, 3]  # more than there are: all, once

    def test_the_first_frame_is_that_episodes_own_picture(self, dataset: Path) -> None:
        report = DemoReport.load(dataset)
        for episode, colour in zip(report.episodes, COLOURS):
            frame = report.first_frame(episode).reshape(-1, 3).mean(axis=0)
            assert np.abs(frame - colour).max() < 14  # H.264 shifts a flat colour by a few levels at most

    def test_each_camera_has_its_own_video(self, dataset: Path) -> None:
        report = DemoReport.load(dataset)
        front, wrist = (report.first_frame(report.episodes[0], c).mean() for c in ("front", "wrist"))
        assert front > 50 and wrist < 10  # the front camera was coloured, the wrist camera black
        with pytest.raises(ValueError, match="unknown camera"):
            report.video_path(report.episodes[0], "overhead")

    def test_a_folder_that_is_not_a_dataset_is_named_in_the_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="recorded demo dataset"):
            DemoReport.load(tmp_path)


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("log_demos_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["log_demos_script"] = module
    spec.loader.exec_module(module)
    return module


class TestLogDemosScript:
    def test_a_dry_run_prints_the_summary_and_the_sample_and_uploads_nothing(
        self, script: ModuleType, dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_upload(*_a: object, **_k: object) -> None:
            raise AssertionError("a dry run must not start a W&B run")

        monkeypatch.setattr(script, "Tracker", no_upload)
        result = CliRunner().invoke(script.app, [str(dataset), "--episodes", "2", "--dry-run"])
        out = " ".join(result.output.split())
        assert result.exit_code == 0 and "'episodes': 4" in out and "sample: [0, 3]" in out

    def test_the_contact_sheet_has_a_labelled_tile_per_sampled_episode(self, script: ModuleType, dataset: Path) -> None:
        report = DemoReport.load(dataset)
        sheet = script.contact_sheet(report, report.episodes)
        h, w = script.SHEET_TILE
        assert sheet.shape == (h, 4 * w, 3)  # four episodes, one row of four tiles
        assert np.abs(sheet[10 : h - 40, 10:30].reshape(-1, 3).mean(axis=0) - COLOURS[0]).max() < 20

    def test_an_empty_or_missing_dataset_is_a_usage_error(self, script: ModuleType, tmp_path: Path) -> None:
        assert CliRunner().invoke(script.app, [str(tmp_path)]).exit_code == 2
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/episodes.jsonl").write_text("")
        assert CliRunner().invoke(script.app, [str(tmp_path)]).exit_code == 2
