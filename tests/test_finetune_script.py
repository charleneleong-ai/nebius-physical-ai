import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import pytest
from typer.testing import CliRunner

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.twin.scene import SO101_XML

SCRIPT = Path(__file__).parent.parent / "scripts" / "finetune.py"
POSE = CameraPose(
    (176.2, 235.2, 418.6), (0.00585, -0.23954, -0.97087), (0.00438, -0.97087, 0.23956), 69.0, (720, 1280), 1.3
)
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finetune_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["finetune_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def home(tmp_path: Path) -> SimpleNamespace:
    """Everything a real run needs, under a temporary directory."""
    paths = SimpleNamespace(
        base=tmp_path / "checkpoints/base",
        gr00t=tmp_path / "Isaac-GR00T",
        camera=tmp_path / "front_camera.json",
        real=tmp_path / "real_objects.json",
        data=tmp_path / "data",
        checkpoints=tmp_path / "checkpoints",
    )
    paths.base.mkdir(parents=True)
    paths.gr00t.mkdir()
    POSE.save(paths.camera)
    paths.real.write_text("[]")
    return paths


def recipe(script: ModuleType, home: SimpleNamespace, **overrides: Any):
    fields = {
        "base": home.base,
        "gr00t_dir": home.gr00t,
        "camera": home.camera,
        "real_objects": home.real,
        "data_root": home.data,
        "checkpoint_root": home.checkpoints,
    }
    return script.Recipe("v4", **(fields | overrides))


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


class TestRecipe:
    def test_the_default_is_the_v4_recipe(self, script: ModuleType) -> None:
        r = script.Recipe("v4")
        assert (r.episodes, r.max_steps, r.camera_mix) == (800, 4000, (0.3, 0.6, 0.1))
        assert r.role_agnostic and r.seed == 42
        assert r.base == script.LATEST and (r.save_steps, r.save_total_limit) == (2000, 2)
        assert r.data_dir.name == "preflight_demos_v4" and r.out_dir.name == "gr00t-preflight-v4"

    def test_the_recording_step_passes_the_episodes_the_camera_and_the_mix(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        step = recipe(script, home).record_step()
        assert step.argv[:4] == ("uv", "run", "npai", "demos") and step.argv[4] == str(home.data / "preflight_demos_v4")
        assert flag(step.argv, "--episodes") == "800" and flag(step.argv, "--seed") == "42"
        assert flag(step.argv, "--camera") == str(home.camera) and flag(step.argv, "--camera-mix") == "0.3,0.6,0.1"
        assert "--randomize" in step.argv and flag(step.argv, "--real-objects") == str(home.real)
        assert step.env == {"MUJOCO_GL": "egl"} and step.cwd == script.REPO

    def test_the_original_share_reaches_the_demo_recorder_and_the_learning_rate_the_trainer(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        default = recipe(script, home)
        assert flag(default.record_step().argv, "--original-share") == "0.25"
        assert flag(default.train_step().argv, "--learning-rate") == "0.0001"
        replay = recipe(script, home, original_share=0.5, learning_rate=5e-5)
        assert flag(replay.record_step().argv, "--original-share") == "0.5"
        assert flag(replay.train_step().argv, "--learning-rate") == "5e-05"
        assert "--original-share" not in recipe(script, home, role_agnostic=False).record_step().argv

    def test_role_agnostic_demos_are_recorded_unless_switched_off(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        assert "--role-agnostic" in recipe(script, home).record_step().argv
        assert "--role-agnostic" not in recipe(script, home, role_agnostic=False).record_step().argv

    def test_with_no_pose_share_the_camera_is_not_passed_at_all(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        argv = recipe(script, home, camera_mix=(0.0, 0.7, 0.3)).record_step().argv
        assert "--camera" not in argv and "--camera-mix" not in argv

    def test_the_training_step_starts_from_the_base_and_keeps_two_checkpoints(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        r = recipe(script, home)
        step = r.train_step()
        assert step.argv[:2] == (str(home.gr00t / ".venv/bin/python"), "gr00t/experiment/launch_finetune.py")
        assert flag(step.argv, "--base-model-path") == str(home.base)
        assert flag(step.argv, "--dataset-path") == str(r.data_dir) and flag(step.argv, "--output-dir") == str(
            r.out_dir
        )
        assert flag(step.argv, "--max-steps") == "4000" and flag(step.argv, "--save-steps") == "2000"
        assert flag(step.argv, "--save-total-limit") == "2" and "--use-wandb" in step.argv
        assert step.env == {"WANDB_RUN_GROUP": "v4"} and step.cwd == home.gr00t

    def test_relative_stats_register_the_modality_config_in_groots_environment(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        first, second = recipe(script, home).stats_steps()
        assert first.allow_failure and not second.allow_failure  # the first can fail until the config is registered
        assert second.argv[1] == "-c" and "register" in second.argv[2] and "generate_rel_stats" in second.argv[2]
        assert second.argv[3] == str(home.data / "preflight_demos_v4") and second.cwd == home.gr00t
        assert second.argv[4] == "examples/SO100/so100_config.py"  # the config the stats are computed under

    def test_the_stock_layout_trains_under_nvidias_so100_config(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        step = recipe(script, home).train_step()
        assert flag(step.argv, "--modality-config-path") == "examples/SO100/so100_config.py"
        assert "--save-only-model" not in step.argv

    def test_the_split_layout_trains_and_computes_statistics_under_our_split_config(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        r = recipe(script, home, layout="split")
        config = str(script.REPO / "scripts/modality_configs/so101_split.py")
        assert Path(config).exists()  # the file the steps point at is really there
        assert flag(r.train_step().argv, "--modality-config-path") == config
        assert r.stats_steps()[1].argv[4] == config
        assert flag(r.record_step().argv, "--layout") == "split"
        assert "--layout" not in recipe(script, home).record_step().argv

    @pytest.mark.parametrize("layout", ["stock", "split"])
    def test_wandb_logging_is_on_for_every_layout_in_the_runs_group(
        self, script: ModuleType, home: SimpleNamespace, layout: str
    ) -> None:
        step = recipe(script, home, layout=layout).train_step()
        assert "--use-wandb" in step.argv and flag(step.argv, "--wandb-project") == "preflight"
        assert step.env["WANDB_RUN_GROUP"] == "v4"

    def test_checkpoints_can_be_saved_without_optimizer_state(self, script: ModuleType, home: SimpleNamespace) -> None:
        assert "--save-only-model" in recipe(script, home, save_only_model=True).train_step().argv

    def test_a_dataset_grouped_for_the_other_layout_is_a_problem_before_anything_runs(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        stock = recipe(script, home)
        meta = stock.data_dir / "meta"
        meta.mkdir(parents=True)
        (meta / "modality.json").write_text(json.dumps({"action": {"single_arm": {}, "gripper": {}}}))
        assert stock.problems(record=False) == []
        found = " ".join(recipe(script, home, layout="split").problems(record=False))
        assert "['single_arm', 'gripper']" in found and "--layout split" in found and "relink" in found
        (meta / "modality.json").write_text(json.dumps({"action": {"single_arm": {}, "wrist_roll": {}, "gripper": {}}}))
        assert recipe(script, home, layout="split").problems(record=False) == []
        assert "--layout" in " ".join(stock.problems(record=False))

    def test_an_unknown_layout_is_a_problem(self, script: ModuleType, home: SimpleNamespace) -> None:
        assert "--layout must be one of" in " ".join(recipe(script, home, layout="wide").problems())

    def test_the_steps_run_in_order_and_recording_can_be_skipped(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        r = recipe(script, home)
        titles = [s.title for s in r.steps()]
        assert titles == ["record demos", "log demos", "dataset stats", "relative stats", "fine-tune"]
        # A reused dataset is not recorded again, but its demos still go to W&B: the trainer logs no media itself.
        assert [s.title for s in r.steps(record=False)] == ["log demos", "dataset stats", "relative stats", "fine-tune"]

    def test_the_demos_are_logged_to_wandb_under_the_runs_name_and_can_never_stop_training(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        step = recipe(script, home).log_demos_step()
        assert step.argv[:4] == ("uv", "run", "python", "scripts/log_demos.py")
        assert step.argv[4] == str(home.data / "preflight_demos_v4") and flag(step.argv, "--name") == "v4"
        assert step.allow_failure and step.cwd == script.REPO

    def test_a_valid_recipe_has_no_problems(self, script: ModuleType, home: SimpleNamespace) -> None:
        assert recipe(script, home).problems() == []

    def test_problems_are_named_before_anything_runs(self, script: ModuleType, home: SimpleNamespace) -> None:
        home.camera.unlink()
        found = " | ".join(recipe(script, home).problems())
        assert "calibrated camera not found" in found
        assert recipe(script, home, camera_mix=(0.0, 0.7, 0.3)).problems() == []  # not needed without a pose share
        assert "base checkpoint not found" in " ".join(recipe(script, home, base=home.base / "gone").problems())
        assert "--episodes must be at least 1" in " ".join(recipe(script, home, episodes=0).problems())
        assert "--save-steps" in " ".join(recipe(script, home, save_steps=9000).problems())

    def test_existing_checkpoints_are_never_trained_over_without_force(
        self, script: ModuleType, home: SimpleNamespace
    ) -> None:
        r = recipe(script, home)
        (r.out_dir / "checkpoint-2000").mkdir(parents=True)
        assert "already holds checkpoints" in " ".join(r.problems())
        assert r.problems(force=True) == []


class FakeRun:
    """subprocess.run that records every call and fails the commands named in `exit_codes`."""

    def __init__(self, exit_codes: dict[str, int] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.exit_codes = exit_codes or {}
        self.seen_dataset: list[bool] = []

    def __call__(self, argv: tuple[str, ...], cwd: Path, env: dict[str, str], check: bool):
        self.calls.append({"argv": argv, "cwd": cwd, "env": env, "check": check})
        code = next((c for fragment, c in self.exit_codes.items() if fragment in " ".join(argv)), 0)
        if check and code:
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, code)


def args(home: SimpleNamespace, *extra: str) -> list[str]:
    return [
        "run", "v4", "--base", str(home.base), "--camera", str(home.camera), "--real-objects", str(home.real),
        "--gr00t-dir", str(home.gr00t), "--data-root", str(home.data), "--checkpoint-root", str(home.checkpoints),
        *extra,
    ]  # fmt: skip


def invoke(script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, *extra: str, fake=None):
    fake = fake or FakeRun()
    monkeypatch.setattr(script.subprocess, "run", fake)
    result = CliRunner().invoke(script.app, args(home, *extra))
    return result.exit_code, " ".join(result.output.split()), fake


class TestRunCommand:
    def test_a_run_executes_every_step_in_order_and_names_the_final_checkpoint(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = invoke(script, home, monkeypatch)
        assert code == 0 and [c["argv"][1] for c in fake.calls] == [
            "run",
            "run",
            "gr00t/data/stats.py",
            "-c",
            "gr00t/experiment/launch_finetune.py",
        ]
        assert fake.calls[1]["argv"][3] == "scripts/log_demos.py"
        # only the demo logging and the first stats step may fail without stopping the run
        assert [c["check"] for c in fake.calls] == [True, False, False, True, True]
        assert fake.calls[0]["env"]["MUJOCO_GL"] == "egl" and fake.calls[4]["env"]["WANDB_RUN_GROUP"] == "v4"
        assert out.endswith(f"FINETUNE_DONE {home.checkpoints}/gr00t-preflight-v4/checkpoint-4000")

    def test_a_stale_dataset_is_cleared_before_recording_and_kept_when_recording_is_skipped(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = home.data / "preflight_demos_v4" / "old.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("half a dataset")
        invoke(script, home, monkeypatch, "--skip-demos")
        assert stale.exists()
        _, _, fake = invoke(script, home, monkeypatch)
        assert not stale.exists() and fake.calls[0]["argv"][3] == "demos"

    def test_skipping_the_recording_still_logs_the_demos_then_runs_the_stats_and_the_training(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, fake = invoke(script, home, monkeypatch, "--skip-demos")
        assert len(fake.calls) == 4 and "npai" not in fake.calls[0]["argv"]  # nothing is recorded
        assert fake.calls[0]["argv"][3] == "scripts/log_demos.py" and fake.calls[0]["check"] is False

    def test_a_dry_run_prints_the_steps_and_runs_nothing_even_with_problems(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home.camera.unlink()
        code, out, fake = invoke(script, home, monkeypatch, "--dry-run")
        assert code == 0 and fake.calls == []
        assert "[1/5] record demos: uv run npai demos" in out and "[2/5] log demos:" in out
        assert "[5/5] fine-tune:" in out and "--episodes 800" in out
        assert "warning: calibrated camera not found" in out and "FINETUNE_DONE" not in out

    def test_a_run_with_problems_refuses_to_start(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home.camera.unlink()
        code, out, fake = invoke(script, home, monkeypatch)
        assert code == 2 and fake.calls == [] and "calibrated camera not found" in out

    def test_force_trains_over_existing_checkpoints(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (home.checkpoints / "gr00t-preflight-v4/checkpoint-2000").mkdir(parents=True)
        assert invoke(script, home, monkeypatch)[0] == 2
        code, _, fake = invoke(script, home, monkeypatch, "--force")
        assert code == 0 and len(fake.calls) == 5

    def test_a_failing_step_stops_the_run_with_its_exit_code(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = invoke(script, home, monkeypatch, fake=FakeRun({"demos": 5}))
        assert code == 5 and len(fake.calls) == 1 and "record demos failed with exit code 5" in out
        assert "FINETUNE_DONE" not in out

    def test_the_stats_step_may_fail_without_stopping_the_run(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, _, fake = invoke(script, home, monkeypatch, fake=FakeRun({"stats.py": 1}))
        assert code == 0 and len(fake.calls) == 5

    def test_a_failed_wandb_upload_does_not_stop_the_training(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = invoke(script, home, monkeypatch, fake=FakeRun({"log_demos.py": 1}))
        assert code == 0 and len(fake.calls) == 5 and "FINETUNE_DONE" in out

    @pytest.mark.parametrize("mix", ["0.5,0.5", "0.6,0.3,0.2", "x,y,z"])
    def test_a_bad_camera_mix_is_a_usage_error(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mix: str
    ) -> None:
        code, out, fake = invoke(script, home, monkeypatch, "--camera-mix", mix)
        assert code == 2 and fake.calls == [] and "camera mix" in out

    def test_the_documented_options_exist(self, script: ModuleType) -> None:
        out = CliRunner().invoke(script.app, ["run", "--help"]).output
        for option in ("--episodes", "--camera-mix", "--base", "--skip-demos", "--force", "--dry-run", "--save-steps"):
            assert option in out
        for option in ("--layout", "--save-only-model", "--original-share", "--learning-rate"):
            assert option in out

    def test_a_dry_run_for_the_split_layout_on_a_community_checkpoint_shows_every_flag(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = invoke(
            script, home, monkeypatch, "--skip-demos", "--layout", "split", "--save-only-model", "--dry-run"
        )
        assert code == 0 and fake.calls == []
        assert "scripts/modality_configs/so101_split.py" in out and "--save-only-model" in out and "--use-wandb" in out


class TestLinkDataset:
    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        root = tmp_path / "recorded"
        (root / "meta").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        (root / "videos/chunk-000").mkdir(parents=True)
        (root / "meta/episodes.jsonl").write_text('{"episode_index": 0}\n')
        (root / "meta/stats.json").write_text('{"made_by": "n1.6"}')
        (root / "data/chunk-000/episode_000000.parquet").write_text("rows")
        (root / "videos/chunk-000/episode_000000.mp4").write_text("video")
        return root

    def link(self, script: ModuleType, *args: str):
        result = CliRunner().invoke(script.app, ["link-dataset", *args])
        return result.exit_code, " ".join(result.output.replace("│", " ").split())  # rich boxes and wraps errors

    def test_episodes_and_videos_are_linked_and_meta_is_a_private_copy(
        self, script: ModuleType, source: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "linked"
        code, _ = self.link(script, str(source), str(dest))
        assert code == 0
        assert (dest / "data").is_symlink() and (dest / "data/chunk-000/episode_000000.parquet").read_text() == "rows"
        assert (dest / "videos/chunk-000/episode_000000.mp4").read_text() == "video"
        assert not (dest / "meta").is_symlink() and (dest / "meta/stats.json").read_text() == '{"made_by": "n1.6"}'
        (dest / "meta/stats.json").write_text('{"made_by": "n1.7"}')  # what the other GR00T version does
        assert (source / "meta/stats.json").read_text() == '{"made_by": "n1.6"}'  # the original is untouched

    def test_an_existing_copy_is_kept_unless_forced_and_forcing_never_touches_the_source(
        self, script: ModuleType, source: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "linked"
        assert self.link(script, str(source), str(dest))[0] == 0
        code, out = self.link(script, str(source), str(dest))
        assert code == 2 and "already exists" in out
        assert self.link(script, str(source), str(dest), "--force")[0] == 0
        assert (source / "data/chunk-000/episode_000000.parquet").exists()  # the links went, not their targets
        assert (source / "videos/chunk-000/episode_000000.mp4").exists()

    def test_a_layout_regroups_the_joints_in_the_copy_only(
        self, script: ModuleType, source: Path, tmp_path: Path
    ) -> None:
        (source / "meta/modality.json").write_text(json.dumps({"state": {"single_arm": {"start": 0, "end": 5}}}))
        dest = tmp_path / "linked"
        code, out = self.link(script, str(source), str(dest), "--layout", "split")
        copied = json.loads((dest / "meta/modality.json").read_text())
        assert code == 0 and "joints grouped as ['single_arm', 'wrist_roll', 'gripper']" in out
        assert copied["action"]["wrist_roll"] == {"start": 4, "end": 5} and copied["state"]["single_arm"]["end"] == 4
        assert set(copied["video"]) == {"front", "wrist"}
        assert json.loads((source / "meta/modality.json").read_text()) == {
            "state": {"single_arm": {"start": 0, "end": 5}}
        }

    def test_an_unknown_layout_is_refused(self, script: ModuleType, source: Path, tmp_path: Path) -> None:
        code, out = self.link(script, str(source), str(tmp_path / "linked"), "--layout", "wide")
        assert code == 2 and "--layout must be one of" in out and not (tmp_path / "linked").exists()

    def test_forcing_onto_the_source_itself_is_refused(self, script: ModuleType, source: Path) -> None:
        code, out = self.link(script, str(source), str(source), "--force")
        assert code == 2 and "source dataset itself" in out and (source / "meta/episodes.jsonl").exists()

    def test_something_that_is_not_a_recorded_dataset_is_refused(self, script: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        code, out = self.link(script, str(tmp_path / "empty"), str(tmp_path / "linked"))
        assert code == 2 and "not a recorded dataset" in out and not (tmp_path / "linked").exists()


class TestServe:
    def serve(self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, *extra: str):
        fake = FakeRun()
        monkeypatch.setattr(script.subprocess, "run", fake)
        result = CliRunner().invoke(script.app, ["serve", *extra, "--gr00t-dir", str(home.gr00t)])
        return result.exit_code, " ".join(result.output.split()), fake

    def test_it_serves_the_checkpoint_on_localhost_from_groots_environment(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = self.serve(script, home, monkeypatch, str(home.base), "--port", "5560")
        (call,) = fake.calls
        assert code == 0 and call["cwd"] == home.gr00t and call["env"]["MUJOCO_GL"] == "egl"
        assert call["argv"][:2] == (str(home.gr00t / ".venv/bin/python"), "gr00t/eval/run_gr00t_server.py")
        assert flag(call["argv"], "--model-path") == str(home.base) and flag(call["argv"], "--port") == "5560"
        assert (
            flag(call["argv"], "--host") == "127.0.0.1" and flag(call["argv"], "--embodiment-tag") == "NEW_EMBODIMENT"
        )
        assert "serve base on port 5560" in out

    def test_the_default_checkpoint_is_the_latest_fine_tune_on_the_default_port(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = self.serve(script, home, monkeypatch, "--dry-run")
        assert code == 0 and fake.calls == [] and str(script.LATEST) in out and "--port 5558" in out

    def test_a_missing_checkpoint_is_a_usage_error_and_starts_nothing(
        self, script: ModuleType, home: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, out, fake = self.serve(script, home, monkeypatch, str(home.base / "gone"))
        assert code == 2 and fake.calls == [] and "checkpoint not found" in out


@needs_model
class TestPreview:
    def test_it_renders_labelled_tiles_from_the_camera_mix(
        self, script: ModuleType, home: SimpleNamespace, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "preview.jpg"
        real = script.REPO / "data/capture/real_objects.json"
        result = CliRunner().invoke(
            script.app,
            ["preview", "--n", "4", "--camera", str(home.camera), "--real-objects", str(real), "--out", str(out_path)],
        )
        assert result.exit_code == 0, result.output
        image = cv2.imread(str(out_path))
        assert image.shape == (240, 4 * 320, 3)  # four 240x320 tiles in one row
        assert "wrote 4 frames" in result.output
