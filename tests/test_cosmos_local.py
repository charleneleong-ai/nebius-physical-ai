import os
import shlex
import subprocess
from pathlib import Path

import pytest

from nebius_physical_ai.cosmos import COSMOS3_IMAGE, COSMOS3_NANO, SERVED_MODEL_NAME

SCRIPT = Path(__file__).parent.parent / "scripts" / "cosmos_local.sh"


def run(*args: str, **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("COSMOS_")}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=clean | env, check=False)


def docker_command(**env: str) -> list[str]:
    done = run("print", **env)
    assert done.returncode == 0, done.stderr
    return shlex.split(done.stdout)


def value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class TestPrintedCommand:
    def test_serves_the_default_model_on_the_cosmos3_image_with_a_capped_gpu_share(self) -> None:
        command = docker_command()
        assert command[:3] == ["docker", "run", "-d"]
        assert value(command, "--runtime") == "nvidia" and value(command, "--gpus") == "all"
        image_at = command.index(COSMOS3_IMAGE)
        assert command[image_at + 1] == COSMOS3_NANO  # the model is the entrypoint's first argument
        assert value(command, "--gpu-memory-utilization") == "0.30"  # leaves room for the GR00T servers
        assert value(command, "--max-model-len") == "16384"
        assert "--async-scheduling" in command

    def test_the_unauthenticated_server_is_published_on_localhost_only(self) -> None:
        command = docker_command()
        published = value(command, "-p")
        assert published == "127.0.0.1:8001:8000"
        assert not any(arg.startswith(("0.0.0.0", ":")) for arg in command if arg.count(":") == 2)

    def test_served_name_and_image_stay_in_sync_with_the_package(self) -> None:
        command = docker_command()
        assert value(command, "--served-model-name") == SERVED_MODEL_NAME
        assert COSMOS3_IMAGE in command

    def test_needs_no_hugging_face_token(self) -> None:
        command = docker_command()
        assert not any("HF_TOKEN" in arg for arg in command)

    def test_weights_are_cached_outside_the_existing_hugging_face_cache(self) -> None:
        mount = value(docker_command(COSMOS_CACHE="/data/cosmos3"), "-v")
        assert mount == "/data/cosmos3:/hf"
        assert ".cache/huggingface" not in value(docker_command(), "-v")

    def test_settings_can_be_overridden_from_the_environment(self) -> None:
        command = docker_command(
            COSMOS_PORT="9001", COSMOS_GPU_FRACTION="0.5", COSMOS_NAME="other", COSMOS_MODEL_ID="nvidia/Cosmos3-Edge"
        )
        assert value(command, "-p") == "127.0.0.1:9001:8000"
        assert value(command, "--gpu-memory-utilization") == "0.5"
        assert value(command, "--name") == "other" and "nvidia/Cosmos3-Edge" in command


def fake_docker(tmp_path: Path, ps_output: str) -> dict[str, str]:
    """A docker stub that logs its arguments; `ps` answers with `ps_output` and `inspect` mimics a missing container."""
    log = tmp_path / "docker.log"
    stub = tmp_path / "docker"
    stub.write_text(
        '#!/usr/bin/env bash\necho "$*" >> "$DOCKER_LOG"\n'
        f'case "$1" in ps) printf "%s" "{ps_output}" ;; inspect) echo; exit 1 ;; esac\n'
    )
    stub.chmod(0o755)
    return {"PATH": f"{tmp_path}:{os.environ['PATH']}", "DOCKER_LOG": str(log), "COSMOS_CACHE": str(tmp_path / "cache")}


class TestUp:
    def test_an_absent_container_is_created_not_started(self, tmp_path: Path) -> None:
        env = fake_docker(tmp_path, "")
        done = run("up", **env)
        calls = Path(env["DOCKER_LOG"]).read_text().splitlines()
        assert done.returncode == 0, done.stderr
        assert any(call.startswith("run -d --name preflight-cosmos-local") for call in calls)
        assert not any(call.startswith("start ") for call in calls)

    def test_a_stopped_container_is_restarted_not_recreated(self, tmp_path: Path) -> None:
        env = fake_docker(tmp_path, "exited")
        done = run("up", **env)
        calls = Path(env["DOCKER_LOG"]).read_text().splitlines()
        assert done.returncode == 0 and "restarted" in done.stdout
        assert any(call.startswith("start preflight-cosmos-local") for call in calls)
        assert not any(call.startswith("run ") for call in calls)

    def test_a_running_container_is_left_alone(self, tmp_path: Path) -> None:
        env = fake_docker(tmp_path, "running")
        done = run("up", **env)
        calls = Path(env["DOCKER_LOG"]).read_text().splitlines()
        assert "already running" in done.stdout
        assert all(call.startswith("ps ") for call in calls)


class TestUsage:
    @pytest.mark.parametrize("args", [(), ("nonsense",)])
    def test_an_unknown_subcommand_prints_usage_and_fails(self, args: tuple[str, ...]) -> None:
        done = run(*args)
        assert done.returncode == 2 and "usage:" in done.stderr

    def test_the_script_is_valid_bash(self) -> None:
        assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
