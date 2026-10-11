import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Self

import numpy as np
import pytest
from typer.testing import CliRunner

from nebius_physical_ai import benchmark
from nebius_physical_ai.benchmark import (
    HEADLINE_N,
    Cell,
    FixedCameraLayout,
    build_suites,
    legacy_suites,
    mcnemar,
    verdict,
)
from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import Observation
from nebius_physical_ai.twin import roles
from nebius_physical_ai.twin.randomize import outline, radius, segment_distance
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


class FakeTracker:
    """Records what the benchmark script sends to W&B into `logged`, a list the test sets."""

    logged: ClassVar[list[dict]] = []

    def __init__(self, *_a: object, **kw: object) -> None:
        self.logged.append(dict(kw))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_a: object) -> None:
        pass

    def log_cell(self, cell: Cell, step: int) -> None:
        self.logged.append({"cell": cell.suite, "step": step})

    def log_progress(self, suite: str, successes: int, rehearsals: int) -> None:
        self.logged.append({"progress": suite, "successes": successes, "rehearsals": rehearsals})

    def log_summary(self, *_a: object) -> None:
        pass


class FakePolicy:
    def __init__(self, **_k: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_a: object) -> None:
        pass


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

    @pytest.mark.parametrize(
        ("wins", "losses", "expected"),
        [(12, 2, "better"), (2, 12, "worse"), (8, 4, "within noise"), (0, 0, "within noise")],
    )
    def test_paired_verdicts_count_only_the_scenes_the_checkpoints_disagree_on(
        self, wins: int, losses: int, expected: str
    ) -> None:
        both, neither = 30, 96 - 30 - wins - losses
        a = [True] * both + [True] * wins + [False] * losses + [False] * neither
        b = [True] * both + [False] * wins + [True] * losses + [False] * neither
        ca, cb = (Cell("s", "legacy", sum(o), 96, 1, 0.0, [sum(o)], o) for o in (a, b))
        assert verdict(ca, cb) == expected
        if wins + losses:
            assert mcnemar(ca, cb)[:2] == (wins, losses)

    def test_mcnemar_matches_the_exact_binomial_and_sees_what_intervals_miss(self) -> None:
        a = [True] * 40 + [True] * 9 + [False] + [False] * 46  # 49/96: 9 wins and 1 loss against b
        b = [True] * 40 + [False] * 9 + [True] + [False] * 46  # 41/96
        ca, cb = (Cell("s", "legacy", sum(o), 96, 1, 0.0, [sum(o)], o) for o in (a, b))
        assert mcnemar(ca, cb) == (9, 1, pytest.approx(2 * (1 + 10) / 2**10))  # exact binomial, p = 0.021
        assert verdict(cell(49), cell(41)) == "within noise" and verdict(ca, cb) == "better"
        assert mcnemar(ca, cell(41)) is None  # no per-scene outcomes: unpaired

    def test_the_record_round_trips_through_json(self) -> None:
        record = json.loads(json.dumps(Cell("clean", "legacy", 29, 96, 1, 12.5, [29]).record()))
        assert record["ci95"][0] < record["p"] < record["ci95"][1] and record["per_case"] == [29]

    def test_rollout_frames_are_never_written_to_the_result_file(self) -> None:
        frames = [np.zeros((24, 32, 3), np.uint8)] * 3
        cell = Cell("clean", "legacy", 29, 96, 1, 12.5, [29], examples={"failure": ("caption", frames)})
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
class TestDiagnosticSuite:
    @pytest.fixture(scope="class")
    def suites(self) -> dict[str, benchmark.Suite]:
        return by_name(benchmark.diagnostic_suites(LAYOUT, REAL))

    @pytest.fixture(scope="class")
    def clean_many(self, suites: dict[str, benchmark.Suite]) -> benchmark.Suite:
        return suites["clean_many"]

    def test_it_is_the_original_task_in_many_different_scenes(self, suites: dict[str, benchmark.Suite]) -> None:
        for suite in suites.values():
            assert suite.group == "diagnostic"
            assert len(suite.cases) == benchmark.GENERATED_CASES and suite.n >= HEADLINE_N
            for case in suite.cases:
                assert (case.pick, case.place, case.prompt) == ("blue", "red", STACK_TASK)
                assert [o.label for o in case.objects[:2]] == ["blue", "red"]
            blue_positions = np.array([[c.objects[0].x_mm, c.objects[0].y_mm] for c in suite.cases])
            assert blue_positions.std(axis=0).min() > 5.0  # many layouts, not one repeated

    def test_it_uses_the_same_camera_as_the_clean_cell(self, clean_many: benchmark.Suite) -> None:
        (clean,) = by_name(legacy_suites(REAL, LAYOUT))["clean"].cases
        assert all(case.camera == clean.camera for case in clean_many.cases)

    def test_tin_suites_add_exactly_the_tin_to_the_pair_clear_of_or_in_the_arms_path(
        self, suites: dict[str, benchmark.Suite]
    ) -> None:
        base = np.array([LAYOUT.mat.width_mm / 2, LAYOUT.mat.height_mm + LAYOUT.base_offset_mm])
        for name, in_path in (("real_1_pomade_tin_many", False), ("in_path_pomade_tin_many", True)):
            for case in suites[name].cases:
                blue, red, tin = case.objects  # the pair plus the tin, nothing else
                assert tin.label == "pomade_tin" and {blue.label, red.label} == {"blue", "red"}
                path_gap = segment_distance(np.array([[tin.x_mm, tin.y_mm]]), base, np.array([blue.x_mm, blue.y_mm]))[0]
                assert (path_gap < 1.0) if in_path else (path_gap > radius(tin))
        assert all(c.camera is None for c in suites["clean_top_down_many"].cases)  # the layout's top-down camera

    def test_it_is_part_of_all_and_selectable_alone(self) -> None:
        assert {s.name for s in build_suites([], LAYOUT, "diagnostic")} == {"clean_many", "clean_top_down_many"}
        assert "clean_many" in {s.name for s in build_suites([], LAYOUT, "all", ["clean_many"])}


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
    assert result.outcomes == [False] * 96  # one outcome per rehearsal, for paired comparisons


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


class TestPromotion:
    """The gate a DAgger round must pass before its checkpoint replaces the served one."""

    @staticmethod
    def cell(suite: str, wins: int, n: int = 96) -> Cell:
        return Cell(suite, "g", wins, n, 1, 1.0, outcomes=[True] * wins + [False] * (n - wins))

    @pytest.mark.parametrize(
        ("new", "promote", "why"),
        [
            ({"a": 90, "b": 80}, True, "+30 rehearsals over 2 suites, pooled better"),
            ({"a": 71, "b": 71}, False, "pooled within noise"),  # +2 on paired scenes
            ({"a": 96, "b": 50}, False, "worse on b"),  # better pooled, but one suite regressed
        ],
    )
    def test_promotes_only_a_pooled_gain_with_no_suite_worse(self, new: dict, promote: bool, why: str) -> None:
        ref = {"a": self.cell("a", 70), "b": self.cell("b", 70)}
        ok, reason = benchmark.promotion({k: self.cell(k, v) for k, v in new.items()}, ref)
        assert ok is promote and why in reason

    def test_nothing_shared_is_not_promoted(self) -> None:
        assert benchmark.promotion({"x": self.cell("x", 96)}, {"a": self.cell("a", 10)}) == (
            False,
            "no suites in common",
        )


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

    def test_failures_tabulates_each_suite_and_the_total(self, script: ModuleType, tmp_path: Path) -> None:
        cells = [
            Cell("clean", "legacy", 90, 96, 1, 1.0, failures={"no_contact": 4, "dropped": 2}).record(),
            Cell("held_out", "held_out", 86, 96, 1, 1.0, failures={"no_contact": 7, "pushed": 3}).record(),
        ]
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"label": "v5", "cells": cells}))
        result = CliRunner().invoke(script.app, ["failures", str(path)], terminal_width=200)
        cells_by_row = [[c.strip() for c in line.split("│")[1:-1]] for line in result.output.splitlines()]
        rows = {cells[0]: cells[1:] for cells in cells_by_row if len(cells) == 8}
        assert result.exit_code == 0
        # failed, then no_contact pushed dropped knocked_place fell_off holding
        assert rows["total"] == ["16", "11", "3", "2", "0", "0", "0"]
        assert rows["clean"][:4] == ["6", "4", "0", "2"]

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

        def fake_run_suite(
            suite: benchmark.Suite, policy: object, layout: object, batch: int, videos: bool, on_case: Callable
        ) -> Cell:
            asked.append(videos)
            on_case("clean_0", 5, suite.n)
            return Cell(suite.name, suite.group, 5, suite.n, 1, 0.1)

        FakeTracker.logged = logged
        monkeypatch.setattr(script, "Tracker", FakeTracker)
        monkeypatch.setattr(script, "Gr00tClient", FakePolicy)
        monkeypatch.setattr(script, "run_suite", fake_run_suite)
        args = ["run", "v4a", "--group", "legacy", "--only", "clean", *flags]
        result = CliRunner().invoke(script.app, args)
        assert result.exit_code == 0, result.output
        assert asked == [expected]
        assert logged[0].get("group") == "v4a"  # every run of a checkpoint shares its W&B group
        assert {"cell": "clean", "step": 1} in logged  # each suite is logged as soon as it is scored
        assert any(e.get("progress") == "clean" and e["successes"] == 5 for e in logged)  # and its cases while it runs

    def test_a_bad_group_is_a_usage_error(self, script: ModuleType) -> None:
        result = CliRunner().invoke(script.app, ["run", "x", "--group", "nope", "--dry-run"])
        assert result.exit_code == 2 and "--group" in " ".join(result.output.split())

    def test_a_dry_run_lists_the_suites_without_a_policy(self, script: ModuleType) -> None:
        result = CliRunner().invoke(script.app, ["run", "x", "--group", "legacy", "--only", "clean*", "--dry-run"])
        out = " ".join(result.output.split())
        assert result.exit_code == 0 and "clean_top_down" in out and "azimuth_90" not in out


def test_the_front_oblique_camera_config_is_the_clean_many_benchmark_view() -> None:
    """configs/front_oblique_camera.json centres training's pose share on the view clean_many scores."""
    layout = TwinLayout()
    spec = layout.camera_spec(CameraPose.load(Path(__file__).parent.parent / "configs/front_oblique_camera.json"))
    expected = benchmark.oblique_camera(layout, benchmark.OBLIQUE_ELEVATION_DEG, 0.0)
    np.testing.assert_allclose(spec["pos"], expected["pos"], atol=1e-9)
    np.testing.assert_allclose(spec["xyaxes"], expected["xyaxes"], atol=1e-9)
    assert spec["fovy"] == pytest.approx(expected["fovy"])
