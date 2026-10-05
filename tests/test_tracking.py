from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nebius_physical_ai import tracking
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.perception import CosmosSceneGraph, MarkerScene
from nebius_physical_ai.pipeline import PipelineResult
from nebius_physical_ai.trust.verdict import Assessment
from nebius_physical_ai.twin.rehearsal import RehearsalResult


class FakeRun:
    def __init__(self, **init: Any) -> None:
        self.init, self.logged, self.summary, self.exit_code = init, [], {}, None
        self.url = "https://wandb.example/run/abc"

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.logged.append(data)

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def keys(self) -> set[str]:
        return {k for entry in self.logged for k in entry}

    def find(self, key: str) -> Any:
        return next(entry[key] for entry in self.logged if key in entry)


class Table:
    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns, self.data = columns, data


class Image:
    def __init__(self, data: np.ndarray, caption: str = "") -> None:
        self.data, self.caption = data, caption


class Video:
    def __init__(self, data: np.ndarray, fps: int, caption: str, format: str) -> None:
        self.data, self.caption = data, caption


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The `wandb` module replaced by a recorder; `.created` holds every run that was started."""
    created: list[FakeRun] = []

    def init(**kwargs: Any) -> FakeRun:
        created.append(FakeRun(**kwargs))
        return created[-1]

    monkeypatch.setattr(tracking, "wandb", SimpleNamespace(init=init, Table=Table, Image=Image, Video=Video))
    return SimpleNamespace(created=created)


def rehearsed_result() -> PipelineResult:
    frames = [np.full((24, 32, 3), 50 * i, np.uint8) for i in range(3)]
    variants = [
        {
            "index": i,
            "success": i % 2
            == 0,  # independent of the pick offset, so the twin counts as robust and the verdict is RETRY
            "steps": 160,
            "physics": {"friction": 0.8, "mass_kg": 0.05},
            "objects": [{"x_mm": 130.0 + i, "y_mm": 150.0}],
        }
        for i in range(4)
    ]
    rehearsal = RehearsalResult(2, 4, [160] * 4, variants, {"success": (0, frames), "failure": (1, frames)})
    objects = [
        DetectedObject("blue", 130, 150, 900, length_mm=30, width_mm=30),
        DetectedObject("red", 220, 150, 1600, length_mm=40, width_mm=40),
    ]
    scene = MarkerScene(objects, [0, 1, 2, 3], 0.4, 0.85, ["vision model did not confirm the red block"])
    graph = CosmosSceneGraph(
        [{"label": "blue", "bbox_2d": [349, 500, 417, 603], "confidence": 0.9}], [], [], ["blue is on red"]
    )
    assessment = Assessment.of(rehearsal, perceived_pick=objects[0], perception=scene.perception)
    summary = rehearsal.summary()
    return PipelineResult(
        scene, graph, summary, assessment, "Retry with the cubes closer.", assessment.verdict.value, rehearsal
    )


def tracker_run(created: list[FakeRun]) -> FakeRun:
    assert len(created) == 1
    return created[0]


def test_the_wandb_config_records_the_task_and_whether_it_is_in_distribution(run: Any) -> None:
    from pathlib import Path

    import cv2

    from nebius_physical_ai.pipeline import log_to_wandb
    from nebius_physical_ai.task import TaskSpec

    ok, png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    assert ok
    pipe = SimpleNamespace(
        perception=SimpleNamespace(model="nvidia/Cosmos3-Nano"),
        nemotron=SimpleNamespace(model="nvidia/Nemotron"),
        n_rollouts=32,
        seed=42,
        gr00t=SimpleNamespace(address="tcp://x:5558"),
        task=TaskSpec("pomade tin", "red"),
    )
    import base64

    url = log_to_wandb(
        Path("frame.png"), base64.b64encode(png.tobytes()).decode(), pipe, PipelineResult.ask_human("x"), 1.0, None
    )
    config = tracker_run(run.created).init["config"]
    assert url == "https://wandb.example/run/abc"
    assert (config["pick"], config["place"]) == ("pomade tin", "red")
    assert config["instruction"] == "Pick pomade tin and place on red" and config["in_distribution"] is False
    assert config["vision_model"] == "Cosmos3-Nano" and tracker_run(run.created).init["name"] == "frame-Cosmos3-Nano"


class TestLogPreflight:
    def test_a_rehearsed_answer_logs_the_verdict_the_scene_the_boxes_and_the_rollouts(self, run: Any) -> None:
        result = rehearsed_result()
        frame = np.zeros((10, 20, 3), np.uint8)
        frame[..., 2] = 200  # red in BGR
        with tracking.Tracker("preflight", name="clean", config={"n": 4}) as tracker:
            tracker.log_preflight(frame, result, seconds=12.5)
        fake = tracker_run(run.created)

        assert fake.init["job_type"] == "preflight" and fake.init["name"] == "clean" and fake.init["config"] == {"n": 4}
        assert fake.summary["verdict"] == result.verdict == "RETRY"
        assert fake.summary["explanation"] == "Retry with the cubes closer."
        assert fake.summary["markers"] == 4 and fake.summary["marker_fit_mm"] == 0.4
        assert fake.summary["p_cal"] == round(result.assessment.p, 3)
        assert (fake.summary["successes"], fake.summary["n"], fake.summary["p"]) == (2, 4, 0.5)  # raw, uncalibrated
        assert fake.summary["vision_issues"] == ["blue is on red"]

        objects = fake.find("objects")
        assert [row[0] for row in objects.data] == ["blue", "red"] and objects.data[1][1:3] == [220, 150]
        assert fake.find("warnings").data == [["vision model did not confirm the red block"]]
        assert fake.find("vision_model").data == [["blue", "[349, 500, 417, 603]", 0.9]]

        assert fake.find("frame").data[0, 0].tolist() == [200, 0, 0]  # BGR frame shown to W&B as RGB
        assert {"video/scene_0_success", "video/scene_0_failure"} <= fake.keys()
        assert fake.find("video/scene_0_success").data.shape == (3, 3, 24, 32)  # (T, C, H, W)
        assert len(fake.find("variants").data) == 4 and fake.exit_code == 0

    def test_an_ask_human_answer_logs_the_reason_and_no_rehearsal(self, run: Any) -> None:
        result = PipelineResult.ask_human("no blue block on the mat; the marker view shows ['yellow']")
        with tracking.Tracker("preflight") as tracker:
            tracker.log_preflight(np.zeros((10, 20, 3), np.uint8), result, seconds=0.4)
        fake = tracker_run(run.created)

        assert fake.summary["verdict"] == "ASK_HUMAN" and "no blue block" in fake.summary["explanation"]
        assert "p_cal" not in fake.summary and "markers" not in fake.summary
        assert fake.keys() == {"frame"}  # no tables, videos or variants for a frame that was never rehearsed

    def test_a_failed_run_is_marked_failed(self, run: Any) -> None:
        with pytest.raises(RuntimeError), tracking.Tracker("preflight"):
            raise RuntimeError("GR00T did not answer")
        assert tracker_run(run.created).exit_code == 1

    def test_a_run_can_be_put_in_a_group(self, run: Any) -> None:
        with tracking.Tracker("benchmark", name="v4a", group="v4a"):
            pass
        assert tracker_run(run.created).init["group"] == "v4a"

    def test_a_benchmark_cell_logs_its_score_and_its_rollout_videos(self, run: Any) -> None:
        from nebius_physical_ai.benchmark import Cell

        frames = [np.full((24, 32, 3), 40 * i, np.uint8) for i in range(3)]
        cell = Cell("clean", "legacy", 29, 96, 1, 5.0, [29], {"failure": ("clean | failure", frames)})
        with tracking.Tracker("benchmark") as tracker:
            tracker.log_cell(cell, step=1)
        fake = tracker_run(run.created)
        assert fake.find("p/clean") == pytest.approx(29 / 96) and fake.find("successes/clean") == 29
        assert fake.find("ci_lo/clean") < fake.find("p/clean") < fake.find("ci_hi/clean")
        assert fake.find("video/clean_failure").data.shape == (3, 3, 24, 32)  # (T, C, H, W)
        assert fake.find("video/clean_failure").caption == "clean | failure"
        assert fake.find("media/clean_start").caption == "clean | failure"  # the first frame, as a still

    def test_a_cell_without_rollouts_logs_scores_only(self, run: Any) -> None:
        from nebius_physical_ai.benchmark import Cell

        with tracking.Tracker("benchmark") as tracker:
            tracker.log_cell(Cell("clean", "legacy", 29, 96, 1, 5.0), step=2)
        assert tracker_run(run.created).keys() == {
            "suite_index",
            "p/clean",
            "ci_lo/clean",
            "ci_hi/clean",
            "successes/clean",
        }

    def test_rollouts_of_a_job_without_per_scene_logs_become_named_videos(self, run: Any) -> None:
        frames = [np.zeros((24, 32, 3), np.uint8)] * 2
        with tracking.Tracker("eval") as tracker:
            tracker.log_rollouts({"success": frames, "failure": frames}, "v4a")
        fake = tracker_run(run.created)
        assert (
            fake.keys() == {"video/success", "video/failure"} and fake.find("video/failure").caption == "v4a | failure"
        )

    def test_disabled_tracking_asks_wandb_for_a_disabled_run(self, run: Any) -> None:
        with tracking.Tracker("preflight", enabled=False):
            pass
        assert tracker_run(run.created).init["mode"] == "disabled"
