import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import pytest
from synthetic_scene import CLEAN, SyntheticScene, scene
from typer.testing import CliRunner

from nebius_physical_ai.perception import CosmosSceneGraph

SCRIPT = Path(__file__).parent.parent / "scripts" / "verify_bbox.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_bbox", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_bbox"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return load_script()


def write(tmp_path: Path, s: SyntheticScene, name: str = "frame.png") -> Path:
    path = tmp_path / name
    assert cv2.imwrite(str(path), s.image)
    return path


def answer(script: ModuleType, monkeypatch: pytest.MonkeyPatch, objects: list[dict]) -> None:
    async def fake(use_token_factory: bool, image) -> CosmosSceneGraph:
        return CosmosSceneGraph(objects, [], [])

    monkeypatch.setattr(script, "raw_scene", fake)


def run(script: ModuleType, *args: str):
    result = CliRunner().invoke(script.app, list(args))
    return result.exit_code, " ".join(result.output.split())


class TestVerdict:
    def test_boxes_in_the_expected_convention_are_confirmed(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        answer(script, monkeypatch, [s.vl_object(b) for b in CLEAN])
        code, out = run(script, str(write(tmp_path, s)))
        assert code == 0 and "convention confirmed: 0-1000" in out
        assert "* 0-1000 (x1,y1,x2,y2) median" in out  # the winning reading is the starred one

    def test_boxes_in_pixels_are_not_mistaken_for_the_expected_convention(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        height, width = s.image.shape[:2]
        pixel = [
            {**o, "bbox_2d": [o["bbox_2d"][0] * width / 1000, o["bbox_2d"][1] * height / 1000,
                              o["bbox_2d"][2] * width / 1000, o["bbox_2d"][3] * height / 1000]}
            for o in (s.vl_object(b) for b in CLEAN)
        ]  # fmt: skip
        answer(script, monkeypatch, pixel)
        code, out = run(script, str(write(tmp_path, s)))
        assert code == 1 and "convention NOT confirmed" in out and "* pixels of the image sent" in out

    def test_swapped_axes_are_detected(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        swapped = [
            {**o, "bbox_2d": [o["bbox_2d"][1], o["bbox_2d"][0], o["bbox_2d"][3], o["bbox_2d"][2]]}
            for o in (s.vl_object(b) for b in CLEAN)
        ]
        answer(script, monkeypatch, swapped)
        code, out = run(script, str(write(tmp_path, s)))
        assert code == 1 and "* 0-1000 (y1,x1,y2,x2)" in out

    def test_rig_boxes_do_not_count_and_no_usable_boxes_fails(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        answer(script, monkeypatch, [{**s.vl_object(CLEAN[0]), "label": "corner marker"}, {"label": "blue"}])
        code, out = run(script, str(write(tmp_path, s)))
        assert code == 1 and "0 boxes from the model" in out and "no usable boxes" in out

    def test_a_frame_without_markers_is_skipped_and_cannot_confirm_anything(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN, markers=(1,))
        answer(script, monkeypatch, [s.vl_object(b) for b in CLEAN])
        code, out = run(script, str(write(tmp_path, s)))
        assert code == 1 and "markers not found, skipped" in out and "convention NOT confirmed" in out

    def test_one_bad_frame_among_good_ones_fails_the_run(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good, bad = scene(CLEAN), scene(CLEAN, skew=0.2)
        replies = iter([[good.vl_object(b) for b in CLEAN], []])  # the second frame gets no boxes

        async def fake(use_token_factory: bool, image) -> CosmosSceneGraph:
            return CosmosSceneGraph(next(replies), [], [])

        monkeypatch.setattr(script, "raw_scene", fake)
        code, _ = run(script, str(write(tmp_path, good, "a.png")), str(write(tmp_path, bad, "b.png")))
        assert code == 1


class TestCommandLine:
    def test_it_is_a_typer_app_with_the_documented_options(self, script: ModuleType) -> None:
        code, out = run(script, "--help")
        assert code == 0 and "--tf-vl" in out and "frames" in out.lower() and "marker ground truth" in out

    def test_a_missing_frame_is_a_usage_error_before_any_model_call(self, script: ModuleType, tmp_path: Path) -> None:
        code, out = run(script, str(tmp_path / "nope.png"))
        assert code == 2 and "does not exist" in out

    def test_tf_vl_selects_the_token_factory_client(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool] = []
        s = scene(CLEAN)

        async def fake(use_token_factory: bool, image) -> CosmosSceneGraph:
            seen.append(use_token_factory)
            return CosmosSceneGraph([s.vl_object(b) for b in CLEAN], [], [])

        monkeypatch.setattr(script, "raw_scene", fake)
        path = str(write(tmp_path, s))
        run(script, path)
        run(script, path, "--tf-vl")
        assert seen == [False, True]
