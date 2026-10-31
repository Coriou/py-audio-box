"""Exercise the real Bash launcher without Docker, sudo or elevated privileges."""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, *, sudo=None, gpu=False, network=None, exitcode=0):
    commands = tmp_path / "commands.jsonl"
    bin_dir = ROOT / "tests/fixtures/ci/launcher-bin"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               LAUNCH_COMMANDS=str(commands), LAUNCH_EXITCODE=str(exitcode))
    env.pop("TOOLBOX_DOCKER_SUDO", None)
    env.pop("TOOLBOX_VARIANT", None)
    env.pop("TOOLBOX_NETWORK", None)
    if sudo is not None:
        env["TOOLBOX_DOCKER_SUDO"] = sudo
    if gpu:
        env["TOOLBOX_VARIANT"] = "gpu"
    if network is not None:
        env["TOOLBOX_NETWORK"] = network
    result = subprocess.run(["bash", str(ROOT / "run"), "voice-synth", "speak",
                             "--text", "Words with spaces; $(not-a-command)", "--seed", "7"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
    return result, calls


@pytest.mark.parametrize("sudo", [None, "0", "1"])
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("network", [None, "default", "host"])
def test_explicit_docker_privileges_preserve_arguments(tmp_path, sudo, gpu, network):
    result, calls = launch(tmp_path, sudo=sudo, gpu=gpu, network=network)
    assert result.returncode == 0, result.stderr
    args = ["compose", "-f", "docker-compose.yml"]
    if gpu:
        args += ["-f", "docker-compose.gpu.yml"]
    if network == "host":
        args += ["-f", "docker-compose.host.yml"]
    args += ["run", "--rm", "pab", "python", "apps/voice-synth/voice-synth.py", "speak",
             "--text", "Words with spaces; $(not-a-command)", "--seed", "7"]
    expected = [["docker", *args]]
    if sudo == "1":
        expected.insert(0, ["sudo", "--", "docker", *args])
    assert calls == expected


@pytest.mark.parametrize("network", [None, "host"])
def test_failed_docker_is_never_retried_with_sudo(tmp_path, network):
    result, calls = launch(tmp_path, network=network, exitcode=13)
    assert result.returncode == 13
    assert [call[0] for call in calls] == ["docker"]


def test_invalid_privilege_option_launches_nothing(tmp_path):
    result, calls = launch(tmp_path, sudo="yes")
    assert result.returncode == 2
    assert "must be 0 or 1" in result.stderr
    assert calls == []


@pytest.mark.parametrize("network", ["bridge", "host; echo injected"])
def test_invalid_network_option_launches_nothing(tmp_path, network):
    result, calls = launch(tmp_path, network=network, sudo="1")
    assert result.returncode == 2
    assert "TOOLBOX_NETWORK must be default or host" in result.stderr
    assert calls == []


def test_failed_explicit_sudo_docker_is_never_retried(tmp_path):
    result, calls = launch(tmp_path, sudo="1", exitcode=13)
    assert result.returncode == 13
    assert [call[0] for call in calls] == ["sudo", "docker"]
    assert calls[0] == ["sudo", "--", *calls[1]]
