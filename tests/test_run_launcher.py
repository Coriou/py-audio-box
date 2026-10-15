"""Exercise the real Bash launcher without Docker, sudo or elevated privileges."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, *, sudo=None, gpu=False, exitcode=0):
    commands = tmp_path / "commands.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = f'''#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["LAUNCH_COMMANDS"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "sudo":
    assert sys.argv[1] == "--"
    os.execvp(sys.argv[2], sys.argv[2:])
sys.exit(int(os.environ["LAUNCH_EXITCODE"]))
'''
    for name in ("docker", "sudo"):
        file = bin_dir / name
        file.write_text(stub)
        file.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               LAUNCH_COMMANDS=str(commands), LAUNCH_EXITCODE=str(exitcode))
    env.pop("TOOLBOX_DOCKER_SUDO", None)
    env.pop("TOOLBOX_VARIANT", None)
    if sudo is not None:
        env["TOOLBOX_DOCKER_SUDO"] = sudo
    if gpu:
        env["TOOLBOX_VARIANT"] = "gpu"
    result = subprocess.run(["bash", str(ROOT / "run"), "voice-synth", "speak",
                             "--text", "Words with spaces; $(not-a-command)", "--seed", "7"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
    return result, calls


@pytest.mark.parametrize("sudo,gpu", [(None, False), ("0", False), ("1", False), ("1", True)])
def test_explicit_docker_privileges_preserve_arguments(tmp_path, sudo, gpu):
    result, calls = launch(tmp_path, sudo=sudo, gpu=gpu)
    assert result.returncode == 0, result.stderr
    args = ["compose", "-f", "docker-compose.yml"]
    if gpu:
        args += ["-f", "docker-compose.gpu.yml"]
    args += ["run", "--rm", "pab", "python", "apps/voice-synth/voice-synth.py", "speak",
             "--text", "Words with spaces; $(not-a-command)", "--seed", "7"]
    expected = [["docker", *args]]
    if sudo == "1":
        expected.insert(0, ["sudo", "--", "docker", *args])
    assert calls == expected


def test_failed_docker_is_never_retried_with_sudo(tmp_path):
    result, calls = launch(tmp_path, exitcode=13)
    assert result.returncode == 13
    assert [call[0] for call in calls] == ["docker"]


def test_invalid_privilege_option_launches_nothing(tmp_path):
    result, calls = launch(tmp_path, sudo="yes")
    assert result.returncode == 2
    assert "must be 0 or 1" in result.stderr
    assert calls == []
