import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nebius_physical_ai.demo import HourlyCap, create_demo
from nebius_physical_ai.job import result_record
from nebius_physical_ai.pipeline import PipelineResult
from nebius_physical_ai.task import TaskSpec


def test_hourly_cap_refuses_past_the_limit_and_frees_after_an_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr("nebius_physical_ai.demo.time.monotonic", lambda: clock[0])
    cap = HourlyCap(2)
    assert cap.take() is None and cap.take() is None
    assert cap.take() == pytest.approx(3600)
    clock[0] += 3601
    assert cap.take() is None


@pytest.fixture
def jpeg() -> bytes:
    _, data = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    return data.tobytes()


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    (tmp_path / "centre").mkdir()
    (tmp_path / "centre/result.json").write_text(json.dumps({"verdict": "GO", "videos": ["success.mp4"]}))
    (tmp_path / "centre.jpg").write_bytes(b"jpg")
    return tmp_path


class TestDemo:
    def test_gallery_lists_rehearsed_scenes_with_their_frames(self, gallery: Path) -> None:
        scenes = TestClient(create_demo(gallery)).get("/api/gallery").json()
        assert scenes == [{"name": "centre", "frame": "centre.jpg", "verdict": "GO", "videos": ["success.mp4"]}]

    def test_suites_are_empty_until_rendered_then_listed(self, gallery: Path) -> None:
        client = TestClient(create_demo(gallery))
        assert client.get("/api/suites").json() == []
        (gallery / "suites.json").write_text(json.dumps([{"suite": "clean_many", "successes": 88, "n": 96}]))
        assert client.get("/api/suites").json()[0]["successes"] == 88

    def test_check_refuses_when_the_hourly_cap_is_spent(self, gallery: Path, jpeg: bytes) -> None:
        assert TestClient(create_demo(gallery, cap_per_hour=0)).post("/api/check", content=jpeg).status_code == 429

    @pytest.mark.parametrize(("body", "status"), [(b"", 400), (b"not an image", 400), (b"x" * 8_000_001, 413)])
    def test_bad_uploads_are_refused_before_spending_a_slot(self, gallery: Path, body: bytes, status: int) -> None:
        app = create_demo(gallery, cap_per_hour=1)
        assert TestClient(app).post("/api/check", content=body).status_code == status
        assert app.state.admission.cap.take() is None  # the slot is still there

    def test_an_impossible_instruction_is_refused_before_spending_a_slot(self, gallery: Path, jpeg: bytes) -> None:
        app = create_demo(gallery, cap_per_hour=1)
        answer = TestClient(app).post("/api/check?pick=red cube&place=red cube", content=jpeg)
        assert answer.status_code == 400 and "same object" in answer.json()["detail"]
        assert app.state.admission.cap.take() is None


def test_the_record_carries_the_instruction_it_judged() -> None:
    """What a verdict was asked to do travels with it; the wording itself is test_task.py's."""
    task = TaskSpec(pick="pomade tin", place="red cube")
    assert result_record(PipelineResult.ask_human("no markers"), 1.0, task)["task"] == {
        "instruction": task.instruction,
        "pick": task.pick,
        "place": task.place,
        "in_distribution": False,
        "note": task.out_of_distribution_note(),
    }
