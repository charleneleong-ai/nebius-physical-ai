import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Self

import numpy as np
import pytest
from typer.testing import CliRunner

from nebius_physical_ai import benchmark
from nebius_physical_ai.benchmark import HEADLINE_N, Cell, FixedCameraLayout, build_suites, legacy_suites, verdict
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin import roles
from nebius_physical_ai.twin.randomize import outline, radius
from nebius_physical_ai.twin.rollout import STACK_TASK
from nebius_physical_ai.twin.scene import SO101_XML, TwinLayout

ROOT = Path(__file__).parent.parent
REAL = [DetectedObject(**o) for o in json.loads((ROOT / "data/capture/real_objects.json").read_text())]
LAYOUT = TwinLayout()
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


def by_name(suites: list[benchmark.Suite]) -> dict[str, benchmark.Suite]:
    return {s.name: s for s in suites}


def cell(successes: int, n: int = 96) -> Cell:
    return Cell("x", "legacy", successes, n, 1, 0.0)


class TestLegacySuites:
    def test_every_cell_is_the_headline_size_and_the_names_match_the_old_experiments(self) -> None:
        suites = by_name(legacy_suites(REAL, LAYOUT))
        assert {"clean", "clean_top_down", "azimuth_90", "azimuth_130", "azimuth_160", "azimuth_180"} <= set(suites)
        assert {"real_1_pomade_tin", "real_1_glasses_case", "real_1_ball_cluster", "real_3"} <= set(suites)
        assert {"real_2_pomade_tin+glasses_case", "real_2_glasses_case+ball_cluster", "in_path_pomade_tin"} <= set(
            suites
        )
        assert all(s.n == HEADLINE_N and len(s.cases) == 1 for s in suites.values())

    def test_the_clean_scene_is_the_pair_the_v2_and_v3_numbers_used(self) -> None:
        (case,) = by_name(legacy_suites(REAL, LAYOUT))["clean"].cases
        blue, red = case.objects
        assert (blue.label, blue.x_mm, blue.y_mm, blue.area_mm2) == ("blue", 131.0, 151.0, 900.0)
        assert (red.label, red.x_mm, red.y_mm, red.area_mm2) == ("red", 221.0, 151.0, 1600.0)
        assert (case.pick, case.place, case.prompt) == ("blue", "red", STACK_TASK)

    def test_the_azimuth_cells_look_from_the_stated_side_at_65_degrees(self) -> None:
        suites = by_name(legacy_suites(REAL, LAYOUT))
        centre = np.array(LAYOUT.mat_to_world(LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm / 2))
        for azimuth in (90, 130, 160, 180):
            (case,) = suites[f"azimuth_{azimuth}"].cases
            x, y, z = case.camera["pos"]
            offset = np.array([x, y]) - centre
            assert np.degrees(np.arctan2(offset[1], offset[0])) % 360 == pytest.approx(azimuth, abs=0.5)
            assert np.degrees(np.arctan2(z, np.hypot(*offset))) == pytest.approx(65.0, abs=0.5)
        assert suites["clean_top_down"].cases[0].camera is None

    def test_real_objects_are_clear_of_the_pair_and_of_each_other(self) -> None:
        (case,) = by_name(legacy_suites(REAL, LAYOUT))["real_3"].cases
        assert [o.label for o in case.objects[2:]] == [o.label for o in REAL]
        for i, earlier in enumerate(case.objects):
            for later in case.objects[i + 1 :]:  # each was placed so that its outline clears everything before it
                edge = outline(later, np.array([later.x_mm, later.y_mm]))
                gap = np.hypot(*(edge - [earlier.x_mm, earlier.y_mm]).T).min()
                assert gap > radius(earlier), (earlier.label, later.label)

    def test_the_in_path_object_sits_on_the_arms_way_to_the_blue_cube(self) -> None:
        (case,) = by_name(legacy_suites(REAL, LAYOUT))["in_path_pomade_tin"].cases
        tin = case.objects[2]
        base = np.array([LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm + LAYOUT.base_offset_mm])
        blue = np.array([131.0, 151.0])
        along = (np.array([tin.x_mm, tin.y_mm]) - base) @ (blue - base) / np.linalg.norm(blue - base) ** 2
        off_line = np.linalg.norm(np.array([tin.x_mm, tin.y_mm]) - (base + along * (blue - base)))
        assert 0.3 < along < 0.7 and off_line < 2.0

    def test_building_twice_gives_the_same_scenes(self) -> None:
        a, b = legacy_suites(REAL, LAYOUT), legacy_suites(REAL, LAYOUT)
        assert [(s.name, s.cases) for s in a] == [(s.name, s.cases) for s in b]

    def test_a_calibrated_camera_adds_its_own_cell(self) -> None:
        from nebius_physical_ai.capture.camera import CameraPose

        pose = CameraPose(
            (176.2, 235.2, 418.6), (0.00585, -0.23954, -0.97087), (0.00438, -0.97087, 0.23956), 69.0, (720, 1280), 1.3
        )
        assert "clean_calibrated" not in by_name(legacy_suites(REAL, LAYOUT))
        suite = by_name(legacy_suites(REAL, LAYOUT, pose))["clean_calibrated"]
        assert suite.cases[0].camera == LAYOUT.camera_spec(pose)

    def test_without_real_objects_only_the_clean_cells_remain(self) -> None:
        names = {s.name for s in legacy_suites([], LAYOUT)}
        assert "clean" in names and not {n for n in names if n.startswith(("real_", "in_path"))}


class TestSelection:
    def test_group_and_glob_filters(self) -> None:
        legacy = build_suites(REAL, LAYOUT, group="legacy")
        assert {s.group for s in legacy} == {"legacy"}
        assert {s.name for s in build_suites(REAL, LAYOUT, "legacy", ["real_1_*"])} == {
            "real_1_pomade_tin",
            "real_1_glasses_case",
            "real_1_ball_cluster",
        }
        assert build_suites(REAL, LAYOUT, "legacy", ["nothing_like_this"]) == []


class TestScoring:
    def test_a_cell_reports_a_wilson_interval_around_its_rate(self) -> None:
        lo, hi = cell(29, 48).wilson()
        assert lo < 29 / 48 < hi and (lo, hi) == pytest.approx((0.466, 0.729), abs=0.005)
        assert cell(0).wilson()[0] == 0.0 and cell(96).wilson()[1] == 1.0
        assert cell(0, 0).p == 0.0 and cell(0, 0).wilson() == (0.0, 1.0)

    def test_verdicts_need_non_overlapping_intervals(self) -> None:
        assert verdict(cell(60), cell(20)) == "better"
        assert verdict(cell(20), cell(60)) == "worse"
        assert verdict(cell(30), cell(26)) == "within noise"  # the run-to-run wobble seen on identical scenes
        assert verdict(cell(30), cell(30)) == "within noise"

    def test_the_record_round_trips_through_json(self) -> None:
        record = json.loads(json.dumps(Cell("clean", "legacy", 29, 96, 1, 12.5, [29]).record()))
        assert record["ci95"][0] < record["p"] < record["ci95"][1] and record["per_case"] == [29]

    def test_rollout_frames_are_never_written_to_the_result_file(self) -> None:
        frames = [np.zeros((24, 32, 3), np.uint8)] * 3
        cell = Cell("clean", "legacy", 29, 96, 1, 12.5, [29], {"failure": ("caption", frames)})
        assert "examples" not in cell.record() and len(json.dumps(cell.record())) < 400


def test_a_fixed_camera_layout_overrides_only_the_camera() -> None:
    spec = {"pos": [0.1, 0.2, 0.3], "xyaxes": [1, 0, 0, 0, 1, 0], "fovy": 40.0}
    fixed = FixedCameraLayout(camera=spec)
    assert fixed.front_camera_spec() == spec and fixed.mat == LAYOUT.mat
    assert FixedCameraLayout().front_camera_spec() == LAYOUT.front_camera_spec()


@pytest.fixture(scope="module")
def suites() -> dict[str, benchmark.Suite]:
    """The held-out suites, built once: the expert has to solve every candidate scene, which takes minutes."""
    return by_name(build_suites(REAL, LAYOUT, group="held_out"))


@needs_model
class TestHeldOutSuites:
    def test_the_expected_suites_exist_and_are_headline_sized(self, suites: dict[str, benchmark.Suite]) -> None:
        assert {"held_out_objects", "seen_vocabulary", "place_on_pomade_tin"} <= set(suites)
        for name in ("held_out_objects", "seen_vocabulary"):
            assert len(suites[name].cases) == benchmark.GENERATED_CASES and suites[name].n >= HEADLINE_N

    def test_held_out_objects_use_only_colours_and_sizes_training_never_draws(
        self, suites: dict[str, benchmark.Suite]
    ) -> None:
        for case in suites["held_out_objects"].cases:
            pick, place = case.objects[:2]
            assert (
                pick.label.split("_")[0] in roles.HELD_OUT_COLOURS
                and place.label.split("_")[0] in roles.HELD_OUT_COLOURS
            )
            assert pick.height_mm > 40 and place.height_mm > 32  # taller than any training pick or place target

    def test_the_seen_suite_uses_trained_colours_only(self, suites: dict[str, benchmark.Suite]) -> None:
        for case in suites["seen_vocabulary"].cases:
            assert {o.label.split("_")[0] for o in case.objects[:2]} <= set(roles.TRAIN_COLOURS) | {"pomade"}

    def test_place_on_cells_end_on_that_real_object(self, suites: dict[str, benchmark.Suite]) -> None:
        for label in (*roles.HELD_OUT_REAL, "pomade_tin"):
            for case in suites.get(f"place_on_{label}", benchmark.Suite("", "", [])).cases:
                assert case.place == label and case.prompt.endswith(f"place on {label.replace('_', ' ')}")

    def test_every_kept_case_is_one_the_expert_can_solve(self, suites: dict[str, benchmark.Suite]) -> None:
        from nebius_physical_ai.twin.expert import start_pose

        start = start_pose(np.random.default_rng(benchmark.SEED))
        for case in suites["held_out_objects"].cases[:4]:
            assert benchmark.expert_solves(case, LAYOUT, start)

    def test_scenes_do_not_repeat_the_training_streams(self, suites: dict[str, benchmark.Suite]) -> None:
        from nebius_physical_ai.twin.expert import SceneSampler

        demo_positions = {(round(o.x_mm, 1), round(o.y_mm, 1)) for o in SceneSampler(LAYOUT, seed=42).sample()}
        first = suites["seen_vocabulary"].cases[0].objects[0]
        assert (first.x_mm, first.y_mm) not in demo_positions


@needs_model
def test_run_suite_sends_the_cases_own_prompt_to_the_policy_and_pools_the_rehearsals() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.tasks: set[str] = set()

        def act_batch(self, observations: list[Observation]) -> np.ndarray:
            self.tasks |= {o.task for o in observations}
            return np.zeros((len(observations), 4, 6))

    case = benchmark.Case(
        "t", [DetectedObject("green_can", 131.0, 151.0, 900.0, shape="cylinder"), DetectedObject("red_box", 221.0, 151.0, 1600.0)],
        "green_can", "red_box", "Pick green can and place on red box",
    )  # fmt: skip
    suite = benchmark.Suite("t", "held_out", [case, case])
    assert suite.variants == 48 and suite.n == 96
    policy = Recorder()
    result = benchmark.run_suite(suite, policy, LAYOUT, batch_size=48)
    assert policy.tasks == {"Pick green can and place on red box"}
    assert (result.n, result.cases, len(result.per_case)) == (96, 2, 2) and result.successes == 0  # it never moves
    assert result.examples == {}  # no videos unless asked for


@needs_model
def test_run_suite_keeps_one_rollout_per_outcome_when_videos_are_asked_for() -> None:
    class Idle:
        def act_batch(self, observations: list[Observation]) -> np.ndarray:
            return np.zeros((len(observations), 4, 6))

    case = benchmark.Case(
        "t", [DetectedObject("blue", 131.0, 151.0, 900.0), DetectedObject("red", 221.0, 151.0, 1600.0)],
        "blue", "red", STACK_TASK,
    )  # fmt: skip
    cell = benchmark.run_suite(
        benchmark.Suite("t", "legacy", [case]), Idle(), LAYOUT, batch_size=32, record_videos=True
    )
    assert set(cell.examples) == {"failure"}  # an arm that never moves never stacks anything
    caption, frames = cell.examples["failure"]
    assert caption.startswith("t | t | failure | variant ") and STACK_TASK in caption
    assert len(frames) >= 2 and frames[0].shape[2] == 3 and frames[0].dtype == np.uint8


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts/benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_script"] = module
    spec.loader.exec_module(module)
    return module


class TestScript:
    def write(self, path: Path, label: str, *scores: tuple[str, int]) -> Path:
        cells = [Cell(name, "legacy", s, 96, 1, 1.0).record() for name, s in scores]
        path.write_text(json.dumps({"label": label, "cells": cells}))
        return path

    def test_compare_marks_a_difference_only_beyond_the_noise(self, script: ModuleType, tmp_path: Path) -> None:
        ref = self.write(tmp_path / "a.json", "v3", ("clean", 55), ("real_3", 1), ("azimuth_180", 20))
        new = self.write(tmp_path / "b.json", "v4a", ("clean", 52), ("real_3", 40), ("azimuth_180", 3))
        result = CliRunner().invoke(script.app, ["compare", str(ref), str(new)])
        out = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "52/96 (54%) within noise" in out  # 55 against 52: the wobble of identical scenes
        assert "40/96 (42%) better" in out and "3/96 (3%) worse" in out

    def test_compare_needs_two_files(self, script: ModuleType, tmp_path: Path) -> None:
        one = self.write(tmp_path / "a.json", "v3", ("clean", 55))
        assert CliRunner().invoke(script.app, ["compare", str(one)]).exit_code == 2

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [(["--wandb"], True), ([], False), (["--wandb", "--no-videos"], False), (["--videos"], True)],
    )
    def test_rollout_videos_follow_wandb_logging_unless_overridden(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str], expected: bool
    ) -> None:
        asked: list[bool] = []
        logged: list[dict] = []

        class FakeTracker:
            def __init__(self, *_a: object, **kw: object) -> None:
                logged.append(dict(kw))

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_a: object) -> None:
                pass

            def log_cell(self, cell: Cell, step: int) -> None:
                logged.append({"cell": cell.suite, "step": step})

            def log_summary(self, *_a: object) -> None:
                pass

        class FakePolicy:
            def __init__(self, **_k: object) -> None:
                pass

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_a: object) -> None:
                pass

        def fake_run_suite(suite: benchmark.Suite, policy: object, layout: object, batch: int, videos: bool) -> Cell:
            asked.append(videos)
            return Cell(suite.name, suite.group, 5, suite.n, 1, 0.1)

        monkeypatch.setattr(script, "Tracker", FakeTracker)
        monkeypatch.setattr(script, "Gr00tClient", FakePolicy)
        monkeypatch.setattr(script, "run_suite", fake_run_suite)
        args = ["run", "v4a", "--group", "legacy", "--only", "clean", *flags]
        result = CliRunner().invoke(script.app, args)
        assert result.exit_code == 0, result.output
        assert asked == [expected]
        assert logged[0].get("group") == "v4a"  # every run of a checkpoint shares its W&B group
        assert {"cell": "clean", "step": 1} in logged  # each suite is logged as soon as it is scored

    def test_a_bad_group_is_a_usage_error(self, script: ModuleType) -> None:
        result = CliRunner().invoke(script.app, ["run", "x", "--group", "nope", "--dry-run"])
        assert result.exit_code == 2 and "--group" in " ".join(result.output.split())

    def test_a_dry_run_lists_the_suites_without_a_policy(self, script: ModuleType) -> None:
        result = CliRunner().invoke(script.app, ["run", "x", "--group", "legacy", "--only", "clean*", "--dry-run"])
        out = " ".join(result.output.split())
        assert result.exit_code == 0 and "clean_top_down" in out and "azimuth_90" not in out
