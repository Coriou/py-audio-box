"""Focused lifecycle tests for the Phase 2C experiment control plane."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase2c_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_phase2c_experiment", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "locked-holdout.bin").write_bytes(b"secret holdout bytes")
    (root / "split.json").write_text('{"development": ["source-a"]}\n', encoding="utf-8")
    (root / "config.json").write_text('{"threshold": 0.5}\n', encoding="utf-8")
    (root / "model.bin").write_bytes(b"model weights")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def _command_args(repo: Path, code: str, *, experiment_id: str = "phase2c-test") -> list[str]:
    return [
        "--repo-root",
        str(repo),
        "--manifest-dir",
        str(repo / "manifests"),
        "--experiment-id",
        experiment_id,
        "--input",
        f"holdout={repo / 'locked-holdout.bin'}",
        "--split",
        str(repo / "split.json"),
        "--config",
        str(repo / "config.json"),
        "--model",
        f"contact-model={repo / 'model.bin'}",
        "--license",
        "source=operator-approved",
        "--license",
        "contact-model=Apache-2.0",
        "--thread-budget",
        "2",
        "--artifact",
        f"events={repo / 'events.json'}",
        "--metrics",
        str(repo / "metrics.json"),
        "--public-metric",
        "precision",
        "--",
        sys.executable,
        "-c",
        code,
    ]


def _run_cli(repo: Path, subcommand: str, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), subcommand, *_command_args(repo, code)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _manifest(repo: Path) -> tuple[Path, dict]:
    paths = list((repo / "manifests").glob("*.manifest.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def test_success_records_provenance_exact_execution_and_validated_outputs(repo: Path) -> None:
    events = repo / "events.json"
    metrics = repo / "metrics.json"
    code = (
        "import json,os,pathlib; "
        "assert os.environ['OMP_NUM_THREADS']=='2'; "
        f"pathlib.Path({str(events)!r}).write_text('[]'); "
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps("
        "{'precision':0.8,'private_holdout_note':'do not report'}))"
    )
    completed = _run_cli(repo, "run", code)
    assert completed.returncode == 0, completed.stderr

    _, manifest = _manifest(repo)
    assert manifest["status"] == "complete"
    assert manifest["timestamps"]["queued_at"] <= manifest["timestamps"]["started_at"]
    assert manifest["timestamps"]["started_at"] <= manifest["timestamps"]["finished_at"]
    assert manifest["repository"]["dirty"] is False
    assert manifest["repository"]["dirty_declaration"] == "clean"
    assert "locked-holdout" not in json.dumps(manifest["repository"])
    assert len(manifest["repository"]["head"]) == 40
    assert manifest["provenance"]["inputs"][0]["sha256"]
    assert manifest["provenance"]["split"]["sha256"]
    assert manifest["provenance"]["config"]["sha256"]
    assert manifest["provenance"]["models"][0]["sha256"]
    assert manifest["provenance"]["licenses"]["source"] == "operator-approved"
    assert manifest["execution"]["shell"] is False
    assert manifest["execution"]["argv"] == [sys.executable, "-c", code]
    assert manifest["execution"]["environment"]["OMP_NUM_THREADS"] == "2"
    assert manifest["execution"]["thread_budget"]["logical_cpus"] == 2
    assert manifest["execution"]["exit_code"] == 0
    assert Path(manifest["execution"]["stdout_path"]).is_file()
    assert Path(manifest["execution"]["stderr_path"]).is_file()
    assert manifest["resources"]["wall_seconds"] >= 0
    assert manifest["artifacts"][0]["sha256"]
    assert manifest["metrics"]["values"]["precision"] == 0.8


def test_nonzero_attempt_fails_and_cannot_be_retried(repo: Path) -> None:
    completed = _run_cli(repo, "run", "raise SystemExit(7)")
    assert completed.returncode == 1
    path, manifest = _manifest(repo)
    assert manifest["status"] == "failed"
    assert manifest["execution"]["exit_code"] == 7
    assert manifest["failure"]["kind"] == "command_exit"

    retried = subprocess.run(
        [sys.executable, str(SCRIPT), "execute", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert retried.returncode == 2
    assert "only queued attempts execute" in retried.stderr
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"


def test_exit_zero_is_failed_when_explicit_finalization_finds_missing_outputs(repo: Path) -> None:
    completed = _run_cli(repo, "run", "pass")
    assert completed.returncode == 1
    _, manifest = _manifest(repo)
    assert manifest["status"] == "failed"
    assert manifest["execution"]["exit_code"] == 0
    assert manifest["failure"]["kind"] == "finalization"


def test_preflight_provenance_change_fails_without_spawning(repo: Path) -> None:
    sentinel = repo / "spawned.txt"
    code = f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('spawned')"
    queued = _run_cli(repo, "queue", code)
    assert queued.returncode == 0, queued.stderr
    path, _ = _manifest(repo)
    (repo / "config.json").write_text('{"threshold": 0.9}\n', encoding="utf-8")

    executed = subprocess.run(
        [sys.executable, str(SCRIPT), "execute", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert executed.returncode == 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["kind"] == "provenance_preflight"
    assert manifest["timestamps"]["started_at"] is None
    assert manifest["execution"]["pid"] is None
    assert manifest["provenance_checks"]["preflight"]["ok"] is False
    assert manifest["provenance_checks"]["postflight"] is None
    assert not sentinel.exists()
    assert not Path(manifest["execution"]["stdout_path"]).exists()


def test_postflight_provenance_mutation_blocks_completion(repo: Path) -> None:
    events = repo / "events.json"
    metrics = repo / "metrics.json"
    model = repo / "model.bin"
    code = (
        "import json,pathlib; "
        f"pathlib.Path({str(events)!r}).write_text('[]'); "
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps({{'precision':0.8}})); "
        f"pathlib.Path({str(model)!r}).write_bytes(b'mutated weights')"
    )
    completed = _run_cli(repo, "run", code)
    assert completed.returncode == 1
    _, manifest = _manifest(repo)
    assert manifest["status"] == "failed"
    assert manifest["execution"]["exit_code"] == 0
    assert manifest["failure"]["kind"] == "provenance_postflight"
    assert manifest["provenance_checks"]["preflight"]["ok"] is True
    assert manifest["provenance_checks"]["postflight"]["ok"] is False
    assert manifest["failure"]["mismatches"][0]["kind"] == "model"


def test_running_is_atomic_before_spawn_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.make_parser().parse_args(
        ["queue", *_command_args(repo, "raise AssertionError('must not run')")]
    )
    path = runner.queue_manifest(args)
    observed: dict[str, str] = {}

    def fail_spawn(*_args, **_kwargs):
        observed["status"] = json.loads(path.read_text(encoding="utf-8"))["status"]
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(runner.subprocess, "Popen", fail_spawn)
    result = runner.execute_manifest(path)
    assert observed == {"status": "running"}
    assert result["status"] == "failed"
    assert result["failure"]["kind"] == "runner_error"


def test_runner_crash_leaves_running_manifest_and_lock(repo: Path) -> None:
    queued = _run_cli(repo, "queue", "import os,signal; os.kill(os.getppid(), signal.SIGKILL)")
    assert queued.returncode == 0, queued.stderr
    path, manifest = _manifest(repo)
    assert manifest["status"] == "queued"

    crashed = subprocess.run(
        [sys.executable, str(SCRIPT), "execute", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert crashed.returncode == -signal.SIGKILL
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["timestamps"]["finished_at"] is None
    assert path.with_name(path.name + ".lock").is_file()

    reconciled = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "reconcile",
            str(path),
            "--reason",
            "runner was killed during rotation",
            "--confirm-no-unrecorded-process",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert reconciled.returncode == 0, reconciled.stderr
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["failure"]["kind"] == "stale_running_reconciled"
    assert not path.with_name(path.name + ".lock").exists()


def test_sigint_cancels_owned_process_group(repo: Path) -> None:
    code = (
        "import os,signal,time; time.sleep(.2); "
        "os.kill(os.getppid(), signal.SIGINT); time.sleep(30)"
    )
    completed = _run_cli(repo, "run", code)
    assert completed.returncode == 1
    _, manifest = _manifest(repo)
    assert manifest["status"] == "cancelled"
    assert manifest["timestamps"]["finished_at"] is not None
    assert manifest["failure"]["kind"] == "signal"
    assert not list((repo / "manifests").glob("*.lock"))


def test_report_is_deterministic_completed_only_and_holdout_safe(repo: Path) -> None:
    events = repo / "events.json"
    metrics = repo / "metrics.json"
    code = (
        "import json,pathlib; "
        f"pathlib.Path({str(events)!r}).write_text('[]'); "
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps("
        f"{{'precision':0.75,'private_input':{str(repo / 'locked-holdout.bin')!r}}}))"
    )
    completed = _run_cli(repo, "run", code)
    assert completed.returncode == 0, completed.stderr

    first = runner.build_report(repo / "manifests")
    second = runner.build_report(repo / "manifests")
    assert runner._canonical_json(first) == runner._canonical_json(second)
    encoded = runner._canonical_json(first).decode("utf-8")
    assert len(first["completed_attempts"]) == 1
    assert first["completed_attempts"][0]["metrics"] == {"precision": 0.75}
    assert "locked-holdout" not in encoded
    assert "private_input" not in encoded
    assert "argv" not in encoded
    assert "environment" not in encoded
    assert "stdout_path" not in encoded


def test_atomic_update_preserves_old_json_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "manifest.json"
    runner.atomic_write_json(target, {"version": 1}, overwrite=False)

    def fail_replace(_source, _target):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        runner.atomic_write_json(target, {"version": 2}, overwrite=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_model_free_queue_and_bidirectional_supersession(repo: Path) -> None:
    raw_args = _command_args(repo, "pass")
    model_index = raw_args.index("--model")
    del raw_args[model_index : model_index + 2]
    args = runner.make_parser().parse_args(["queue", *raw_args])
    path = runner.queue_manifest(args)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["provenance"]["models"] == []

    runner.cancel_queued(path, "not selected")
    older = str(uuid.uuid4())
    newer = str(uuid.uuid4())
    decided = runner.decide_manifest(path, "rejected", "superseded", [older], [newer])
    assert decided["decision"] == {
        "value": "rejected",
        "reason": "superseded",
        "supersedes": [older],
        "superseded_by": [newer],
    }


def test_every_declared_model_requires_a_license(repo: Path) -> None:
    raw_args = _command_args(repo, "pass")
    license_index = raw_args.index("contact-model=Apache-2.0")
    del raw_args[license_index - 1 : license_index + 1]
    args = runner.make_parser().parse_args(["queue", *raw_args])
    with pytest.raises(runner.RunnerError, match="missing --license for model"):
        runner.queue_manifest(args)


def test_dirty_repository_requires_declaration_but_never_reads_dirty_contents(repo: Path) -> None:
    dirty_path = repo / "unrelated-operator-note.txt"
    dirty_path.write_text("private operator content", encoding="utf-8")
    raw_args = _command_args(repo, "pass")
    args = runner.make_parser().parse_args(["queue", *raw_args])
    with pytest.raises(runner.RunnerError, match="--dirty-declaration is required"):
        runner.queue_manifest(args)

    delimiter = raw_args.index("--")
    raw_args[delimiter:delimiter] = [
        "--dirty-declaration",
        "unrelated local operator work present",
    ]
    declared = runner.make_parser().parse_args(["queue", *raw_args])
    path = runner.queue_manifest(declared)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    repository = manifest["repository"]
    assert repository["dirty"] is True
    assert repository["porcelain_record_count"] == 1
    assert repository["dirty_declaration"] == "unrelated local operator work present"
    encoded = json.dumps(repository)
    assert dirty_path.name not in encoded
    assert "private operator content" not in encoded
