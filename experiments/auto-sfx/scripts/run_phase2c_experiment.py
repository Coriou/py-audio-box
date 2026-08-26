#!/usr/bin/env python3
"""Atomic, provenance-preserving runner for Phase 2C experiments.

The manifest is the control plane.  ``queue`` records immutable inputs and an
exact argv/environment, and ``execute`` makes a single attempt.  A process
loss leaves a manifest in ``running``; it is never interpreted as success and
is never retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "auto-sfx-experiment-manifest/v1"
REPORT_SCHEMA = "auto-sfx-experiment-report/v1"
STATUSES = {"queued", "running", "complete", "failed", "cancelled"}
TERMINAL_STATUSES = {"complete", "failed", "cancelled"}
DECISIONS = {"pending", "accepted", "rejected", "inconclusive"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class RunnerError(RuntimeError):
    """Expected command-line or manifest error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    """Durably install JSON without exposing a partial manifest or report."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(tmp, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_canonical_json(value))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns fd once constructed.
            raise
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise RunnerError(f"refusing to overwrite existing file: {path}") from exc
            finally:
                tmp.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"expected a JSON object: {path}")
    return value


def _validate_name(value: str, label: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise RunnerError(f"{label} must match {SAFE_NAME.pattern!r}: {value!r}")
    return value


def _parse_named_path(raw: str, label: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not path:
        raise RunnerError(f"{label} must be NAME=PATH: {raw!r}")
    return _validate_name(name, f"{label} name"), Path(path).expanduser().resolve()


def _parse_pair(raw: str, label: str) -> tuple[str, str]:
    name, separator, value = raw.partition("=")
    if not separator:
        raise RunnerError(f"{label} must be NAME=VALUE: {raw!r}")
    _validate_name(name, f"{label} name")
    if not value or "\x00" in value:
        raise RunnerError(f"{label} value may not be empty or contain NUL")
    return name, value


def _hash_regular_file(
    path: Path, digest: "hashlib._Hash", *, follow_symlinks: bool = False
) -> tuple[int, int]:
    before = path.stat(follow_symlinks=follow_symlinks)
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    after = path.stat(follow_symlinks=follow_symlinks)
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    )
    if not stable or size != after.st_size:
        raise RunnerError(f"file changed while hashing: {path}")
    return size, 1


def hash_path(path: Path) -> dict[str, Any]:
    """Hash a file or a deterministic directory tree without following links."""

    path = path.expanduser().resolve(strict=False)
    if not path.exists() and not path.is_symlink():
        raise RunnerError(f"provenance path does not exist: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    entries = 0
    if path.is_symlink():
        target = os.readlink(path)
        digest.update(b"symlink\0" + os.fsencode(target))
        kind = "symlink"
        entries = 1
    elif path.is_file():
        digest.update(b"file\0")
        total_bytes, entries = _hash_regular_file(path, digest)
        kind = "file"
    elif path.is_dir():
        kind = "directory"
        initial = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
        for item in initial:
            relative = item.relative_to(path).as_posix().encode("utf-8", errors="surrogateescape")
            if item.is_symlink():
                link_target = os.readlink(item)
                digest.update(b"link\0" + relative + b"\0" + os.fsencode(link_target) + b"\0")
                try:
                    resolved_target = item.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise RunnerError(f"broken or cyclic provenance symlink: {item}") from exc
                if not resolved_target.is_file():
                    raise RunnerError(
                        "provenance directory symlinks are unsupported; "
                        f"declare target directly: {item}"
                    )
                size, count = _hash_regular_file(item, digest, follow_symlinks=True)
                total_bytes += size
                entries += count
            elif item.is_dir():
                digest.update(b"dir\0" + relative + b"\0")
                entries += 1
            elif item.is_file():
                digest.update(b"file\0" + relative + b"\0")
                size, count = _hash_regular_file(item, digest)
                total_bytes += size
                entries += count
            else:
                raise RunnerError(f"unsupported provenance entry: {item}")
        final = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
        initial_names = [item.relative_to(path) for item in initial]
        final_names = [item.relative_to(path) for item in final]
        if initial_names != final_names:
            raise RunnerError(f"directory changed while hashing: {path}")
    else:
        raise RunnerError(f"unsupported provenance path: {path}")
    return {
        "path": str(path),
        "kind": kind,
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "entries": entries,
        "hash_algorithm": "sha256-tree-v1",
    }


def _git_metadata(repo_root: Path, dirty_declaration: str | None) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_command = [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ]
        status = subprocess.run(
            status_command,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"cannot inspect git repository {repo_root}: {exc}") from exc
    dirty = bool(status)
    if dirty and not dirty_declaration:
        raise RunnerError("repository is dirty; --dirty-declaration is required")
    if not dirty and dirty_declaration:
        raise RunnerError("repository is clean; omit --dirty-declaration")
    return {
        "root": str(repo_root),
        "head": head,
        "dirty": dirty,
        "dirty_declaration": dirty_declaration if dirty else "clean",
        # Capture the complete dirty-state identity, not file contents or paths.
        "porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
        "porcelain_record_count": status.count(b"\0"),
    }


def _resolve_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RunnerError(f"{label} must be inside repository {root}: {resolved}") from exc
    return resolved


def _named_asset(raw: str, label: str) -> dict[str, Any]:
    name, path = _parse_named_path(raw, label)
    return {"name": name, **hash_path(path)}


def _unique_named(items: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    names = [item["name"] for item in items]
    if len(names) != len(set(names)):
        raise RunnerError(f"duplicate {label} name")
    return items


def _exact_environment(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    environment: dict[str, str] = {}
    passed: list[str] = []
    for name in args.pass_env:
        _validate_name(name, "passed environment variable")
        if name not in os.environ:
            raise RunnerError(f"cannot pass unset environment variable: {name}")
        environment[name] = os.environ[name]
        passed.append(name)
    for raw in args.env:
        name, value = _parse_pair(raw, "--env")
        environment[name] = value
    if "PATH" not in environment:
        environment["PATH"] = os.environ.get("PATH", os.defpath)
        passed.append("PATH")
    thread_value = str(args.thread_budget)
    for name in THREAD_ENV_KEYS:
        environment[name] = thread_value
    environment["AUTO_SFX_THREAD_BUDGET"] = thread_value
    return dict(sorted(environment.items())), {
        "logical_cpus": args.thread_budget,
        "enforced_environment_keys": list(THREAD_ENV_KEYS) + ["AUTO_SFX_THREAD_BUDGET"],
        "passed_from_parent": sorted(set(passed)),
    }


def _command_from_remainder(command: Sequence[str]) -> list[str]:
    if not command or command[0] != "--" or len(command) == 1:
        raise RunnerError("command must follow a literal -- delimiter")
    argv = list(command[1:])
    if any("\x00" in item for item in argv):
        raise RunnerError("command arguments may not contain NUL")
    return argv


def _attempt_references(values: Sequence[str], label: str) -> list[str]:
    references = sorted(set(values))
    for value in references:
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise RunnerError(f"{label} must contain attempt UUIDs: {value!r}") from exc
    return references


def build_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        raise RunnerError(f"repository root is not a directory: {repo_root}")
    manifest_dir = _resolve_inside(Path(args.manifest_dir), repo_root, "manifest directory")
    cwd = _resolve_inside(Path(args.cwd) if args.cwd else repo_root, repo_root, "command cwd")
    if not cwd.is_dir():
        raise RunnerError(f"command cwd is not a directory: {cwd}")
    experiment_id = _validate_name(args.experiment_id, "experiment id")
    attempt_id = str(uuid.uuid4())
    argv = _command_from_remainder(args.command)
    inputs = _unique_named([_named_asset(raw, "--input") for raw in args.input], "input")
    models = _unique_named([_named_asset(raw, "--model") for raw in args.model], "model")
    licenses = dict(_parse_pair(raw, "--license") for raw in args.license)
    if len(licenses) != len(args.license):
        raise RunnerError("duplicate --license name")
    unlicensed_models = [model["name"] for model in models if model["name"] not in licenses]
    if unlicensed_models:
        raise RunnerError("missing --license for model: " + ", ".join(unlicensed_models))
    split = hash_path(Path(args.split))
    config = hash_path(Path(args.config))
    environment, thread_budget = _exact_environment(args)

    log_dir = manifest_dir / "logs"
    stdout_path = log_dir / f"{experiment_id}.{attempt_id}.stdout.log"
    stderr_path = log_dir / f"{experiment_id}.{attempt_id}.stderr.log"
    artifacts = []
    for raw in args.artifact:
        name, path = _parse_named_path(raw, "--artifact")
        path = _resolve_inside(path, repo_root, "artifact path")
        artifacts.append({"name": name, "path": str(path), "exists": False})
    _unique_named(artifacts, "artifact")
    output_paths = [Path(item["path"]) for item in artifacts]
    metrics_path = _resolve_inside(Path(args.metrics), repo_root, "metrics path")
    output_paths.append(metrics_path)
    if len(output_paths) != len(set(output_paths)):
        raise RunnerError("artifact and metrics output paths must be distinct")
    occupied = [path for path in output_paths if path.exists() or path.is_symlink()]
    if occupied:
        raise RunnerError(f"refusing pre-existing output path: {occupied[0]}")
    public_metrics = sorted(set(args.public_metric))
    for name in public_metrics:
        _validate_name(name, "public metric name")

    now = utc_now()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "status": "queued",
        "timestamps": {
            "created_at": now,
            "queued_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
        },
        "repository": _git_metadata(repo_root, args.dirty_declaration),
        "provenance": {
            "inputs": inputs,
            "split": split,
            "config": config,
            "models": models,
            "licenses": dict(sorted(licenses.items())),
        },
        "provenance_checks": {"preflight": None, "postflight": None},
        "execution": {
            "argv": argv,
            "shell": False,
            "cwd": str(cwd),
            "environment": environment,
            "thread_budget": thread_budget,
            "pid": None,
            "exit_code": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
        "resources": None,
        "artifacts": artifacts,
        "metrics": {
            "path": str(metrics_path),
            "exists": False,
            "sha256": None,
            "values": None,
            "public_names": public_metrics,
        },
        "decision": {
            "value": args.decision,
            "reason": args.reason,
            "supersedes": _attempt_references(args.supersedes, "--supersedes"),
            "superseded_by": _attempt_references(args.superseded_by, "--superseded-by"),
        },
        "failure": None,
    }
    manifest_path = manifest_dir / f"{experiment_id}.{attempt_id}.manifest.json"
    return manifest_path, manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise RunnerError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if manifest.get("status") not in STATUSES:
        raise RunnerError(f"invalid manifest status: {manifest.get('status')!r}")
    _validate_name(str(manifest.get("experiment_id", "")), "experiment id")
    try:
        uuid.UUID(str(manifest["attempt_id"]))
    except (KeyError, ValueError) as exc:
        raise RunnerError("invalid attempt id") from exc


def _verify_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    provenance = manifest["provenance"]
    assets: list[tuple[str, str, dict[str, Any]]] = []
    assets.extend(("input", item["name"], item) for item in provenance["inputs"])
    assets.append(("split", "split", provenance["split"]))
    assets.append(("config", "config", provenance["config"]))
    assets.extend(("model", item["name"], item) for item in provenance["models"])
    mismatches: list[dict[str, Any]] = []
    for kind, name, expected in assets:
        try:
            actual = hash_path(Path(expected["path"]))
        except (OSError, RunnerError) as exc:
            mismatches.append({"kind": kind, "name": name, "error": str(exc)})
            continue
        if actual["sha256"] != expected["sha256"]:
            mismatches.append(
                {
                    "kind": kind,
                    "name": name,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual["sha256"],
                }
            )
    return {"checked_at": utc_now(), "ok": not mismatches, "mismatches": mismatches}


def queue_manifest(args: argparse.Namespace) -> Path:
    path, manifest = build_manifest(args)
    atomic_write_json(path, manifest, overwrite=False)
    return path


class _ManifestLock:
    def __init__(self, manifest_path: Path, *, suffix: str = ".lock") -> None:
        self.path = manifest_path.with_name(manifest_path.name + suffix)
        self.fd: int | None = None

    def __enter__(self) -> "_ManifestLock":
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(self.fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(self.fd)
        except FileExistsError as exc:
            raise RunnerError(f"manifest is locked (possibly interrupted): {self.path}") from exc
        return self

    def __exit__(self, *_args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _update(path: Path, manifest: dict[str, Any]) -> None:
    manifest["timestamps"]["updated_at"] = utc_now()
    atomic_write_json(path, manifest, overwrite=True)


def _usage_snapshot() -> tuple[float, float, int, int, int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        usage.ru_utime,
        usage.ru_stime,
        usage.ru_maxrss,
        usage.ru_inblock,
        usage.ru_oublock,
    )


def _resource_result(
    before: tuple[float, float, int, int, int],
    after: tuple[float, float, int, int, int],
    wall_seconds: float,
) -> dict[str, Any]:
    max_rss = after[2]
    if platform.system() == "Darwin":
        max_rss //= 1024
    return {
        "wall_seconds": round(wall_seconds, 6),
        "user_cpu_seconds": round(after[0] - before[0], 6),
        "system_cpu_seconds": round(after[1] - before[1], 6),
        "max_rss_kib": max_rss,
        "block_inputs": after[3] - before[3],
        "block_outputs": after[4] - before[4],
    }


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _request_process_group_termination(process: subprocess.Popen[bytes]) -> None:
    """Request termination without waiting inside a Python signal handler.

    ``Popen.wait`` owns an internal non-reentrant wait lock.  Calling the full
    termination helper from a signal handler that interrupted ``wait`` can
    deadlock on that lock, so handlers may only deliver the signal.  The normal
    execution path below remains responsible for waiting and SIGKILL escalation.
    """

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _collect_outputs(manifest: dict[str, Any]) -> str | None:
    missing: list[str] = []
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if path.exists() or path.is_symlink():
            artifact.update(hash_path(path))
            artifact["exists"] = True
        else:
            artifact["exists"] = False
            missing.append(artifact["name"])
    metrics = manifest["metrics"]
    metrics_path = Path(metrics["path"])
    if not metrics_path.is_file():
        metrics["exists"] = False
        missing.append("metrics")
    else:
        metrics_asset = hash_path(metrics_path)
        try:
            values = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"metrics are not valid JSON: {exc}"
        if not isinstance(values, dict):
            return "metrics JSON must be an object"
        for name in metrics["public_names"]:
            if name not in values:
                return f"public metric is absent from metrics: {name}"
            if not isinstance(values[name], (str, int, float, bool)) or values[name] is None:
                return f"public metric must be a non-null scalar: {name}"
        metrics.update(metrics_asset)
        metrics["exists"] = True
        metrics["values"] = values
    if missing:
        return "missing declared outputs: " + ", ".join(missing)
    return None


def execute_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with _ManifestLock(path):
        manifest = read_json(path)
        _validate_manifest(manifest)
        if manifest["status"] != "queued":
            raise RunnerError(
                f"attempt {manifest['attempt_id']} is {manifest['status']}; "
                "only queued attempts execute"
            )
        outputs = [Path(item["path"]) for item in manifest["artifacts"]]
        outputs.append(Path(manifest["metrics"]["path"]))
        occupied = [output for output in outputs if output.exists() or output.is_symlink()]
        if occupied:
            raise RunnerError(f"refusing pre-existing output path: {occupied[0]}")
        stdout_path = Path(manifest["execution"]["stdout_path"])
        stderr_path = Path(manifest["execution"]["stderr_path"])
        if stdout_path.exists() or stderr_path.exists():
            raise RunnerError("refusing to overwrite an existing stdout/stderr log")

        preflight = _verify_provenance(manifest)
        manifest["provenance_checks"]["preflight"] = preflight
        if not preflight["ok"]:
            manifest["status"] = "failed"
            manifest["timestamps"]["finished_at"] = utc_now()
            manifest["failure"] = {
                "kind": "provenance_preflight",
                "message": "queued provenance changed before execution; command was not launched",
                "mismatches": preflight["mismatches"],
            }
            _update(path, manifest)
            return manifest

        now = utc_now()
        manifest["status"] = "running"
        manifest["timestamps"]["started_at"] = now
        _update(path, manifest)  # Persist running before logs are opened or a child is spawned.

        process: subprocess.Popen[bytes] | None = None
        cancelled = False
        cancellation_signal: int | None = None
        before = _usage_snapshot()
        started = time.monotonic()
        previous_handlers: dict[int, Any] = {}

        def cancel(signum: int, _frame: Any) -> None:
            nonlocal cancelled, cancellation_signal
            cancelled = True
            cancellation_signal = signum
            if process is not None:
                _request_process_group_termination(process)

        try:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, cancel)
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(
                    manifest["execution"]["argv"],
                    cwd=manifest["execution"]["cwd"],
                    env=manifest["execution"]["environment"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    start_new_session=True,
                )
                manifest["execution"]["pid"] = process.pid
                _update(path, manifest)
                if cancelled:
                    _terminate_process_group(process)
                    exit_code = process.returncode
                else:
                    while True:
                        try:
                            exit_code = process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            if cancelled:
                                _terminate_process_group(process)
                                exit_code = process.returncode
                                break
                        except KeyboardInterrupt:
                            cancelled = True
                            cancellation_signal = signal.SIGINT
                            _terminate_process_group(process)
                            exit_code = process.returncode
                            break
        except BaseException as exc:
            if process is not None:
                _terminate_process_group(process)
            exit_code = process.returncode if process is not None else None
            manifest["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["failure"] = {
                "kind": "interrupted" if isinstance(exc, KeyboardInterrupt) else "runner_error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        elapsed = time.monotonic() - started
        after = _usage_snapshot()
        manifest["execution"]["exit_code"] = exit_code
        manifest["resources"] = _resource_result(before, after, elapsed)
        try:
            output_error = _collect_outputs(manifest)
        except (OSError, RunnerError) as exc:
            output_error = f"cannot validate declared outputs: {exc}"
        postflight = _verify_provenance(manifest)
        manifest["provenance_checks"]["postflight"] = postflight

        if cancelled:
            manifest["status"] = "cancelled"
            manifest["failure"] = {
                "kind": "signal",
                "message": f"runner cancelled by signal {cancellation_signal}",
            }
        elif manifest["status"] == "failed":
            pass
        elif exit_code != 0:
            manifest["status"] = "failed"
            manifest["failure"] = {
                "kind": "command_exit",
                "message": f"command exited with status {exit_code}",
            }
        elif not postflight["ok"]:
            manifest["status"] = "failed"
            manifest["failure"] = {
                "kind": "provenance_postflight",
                "message": "command mutated declared provenance; completion refused",
                "mismatches": postflight["mismatches"],
            }
        elif output_error:
            manifest["status"] = "failed"
            manifest["failure"] = {"kind": "finalization", "message": output_error}
        else:
            # Completion is the last atomic transition, after exit=0 and output validation.
            manifest["status"] = "complete"
            manifest["failure"] = None
        manifest["timestamps"]["finished_at"] = utc_now()
        _update(path, manifest)
        return manifest


def cancel_queued(path: Path, reason: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with _ManifestLock(path):
        manifest = read_json(path)
        _validate_manifest(manifest)
        if manifest["status"] != "queued":
            raise RunnerError("only a queued attempt can be cancelled externally")
        manifest["status"] = "cancelled"
        manifest["failure"] = {"kind": "operator_cancel", "message": reason}
        manifest["timestamps"]["finished_at"] = utc_now()
        _update(path, manifest)
        return manifest


def decide_manifest(
    path: Path,
    decision: str,
    reason: str,
    supersedes: Sequence[str],
    superseded_by: Sequence[str],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with _ManifestLock(path):
        manifest = read_json(path)
        _validate_manifest(manifest)
        if manifest["status"] not in TERMINAL_STATUSES:
            raise RunnerError("decision requires a terminal attempt")
        manifest["decision"] = {
            "value": decision,
            "reason": reason,
            "supersedes": _attempt_references(supersedes, "--supersedes"),
            "superseded_by": _attempt_references(superseded_by, "--superseded-by"),
        }
        _update(path, manifest)
        return manifest


def _pid_or_group_alive(pid: int, *, process_group: bool = False) -> bool:
    try:
        if process_group:
            os.killpg(pid, 0)
        else:
            os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile_stale_running(
    path: Path, reason: str, *, confirm_no_unrecorded_process: bool
) -> dict[str, Any]:
    """Cancel, but never retry, a running attempt whose owner and child are gone."""

    path = path.expanduser().resolve()
    primary_lock = path.with_name(path.name + ".lock")
    with _ManifestLock(path, suffix=".reconcile.lock"):
        manifest = read_json(path)
        _validate_manifest(manifest)
        if manifest["status"] != "running":
            raise RunnerError("only a running attempt can be reconciled")

        owner_pid: int | None = None
        if primary_lock.is_file():
            try:
                lock_text = primary_lock.read_text(encoding="ascii").strip()
                prefix, raw_pid = lock_text.split("=", 1)
                if prefix != "pid":
                    raise ValueError(lock_text)
                owner_pid = int(raw_pid)
            except (OSError, UnicodeError, ValueError) as exc:
                raise RunnerError(f"cannot verify malformed attempt lock: {primary_lock}") from exc
            if _pid_or_group_alive(owner_pid):
                raise RunnerError(f"runner pid {owner_pid} is still alive; refusing reconciliation")

        child_pid = manifest["execution"].get("pid")
        if child_pid is None:
            if not confirm_no_unrecorded_process:
                raise RunnerError(
                    "attempt has no recorded child pid; inspect local processes, then pass "
                    "--confirm-no-unrecorded-process"
                )
        elif not isinstance(child_pid, int) or child_pid < 1:
            raise RunnerError(f"invalid recorded child pid: {child_pid!r}")
        elif _pid_or_group_alive(child_pid, process_group=True):
            raise RunnerError(
                f"child process group {child_pid} is still alive; refusing reconciliation"
            )

        now = utc_now()
        manifest["status"] = "cancelled"
        manifest["timestamps"]["finished_at"] = now
        manifest["failure"] = {
            "kind": "stale_running_reconciled",
            "message": reason,
            "runner_pid_checked_dead": owner_pid,
            "child_process_group_checked_dead": child_pid,
            "confirmed_no_unrecorded_process": bool(confirm_no_unrecorded_process),
        }
        _update(path, manifest)
        primary_lock.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        return manifest


def _public_report_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a review row with no input, split, argv, env, or path disclosure."""

    public_names = manifest["metrics"].get("public_names", [])
    values = manifest["metrics"].get("values") or {}
    return {
        "experiment_id": manifest["experiment_id"],
        "attempt_id": manifest["attempt_id"],
        "completed_at": manifest["timestamps"]["finished_at"],
        "repository": {
            "head": manifest["repository"]["head"],
            "dirty": manifest["repository"]["dirty"],
            "dirty_status_sha256": manifest["repository"]["porcelain_v1_z_sha256"],
        },
        "exit_code": manifest["execution"]["exit_code"],
        "resources": manifest["resources"],
        "metrics": {name: values[name] for name in sorted(public_names)},
        "decision": manifest["decision"],
        "artifact_hashes": {
            item["name"]: item["sha256"] for item in manifest["artifacts"] if item.get("sha256")
        },
    }


def build_report(manifest_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(manifest_dir.expanduser().resolve().glob("*.manifest.json")):
        manifest = read_json(path)
        _validate_manifest(manifest)
        if manifest["status"] == "complete":
            rows.append(_public_report_row(manifest))
    rows.sort(key=lambda row: (row["experiment_id"], row["attempt_id"]))
    return {"schema": REPORT_SCHEMA, "completed_attempts": rows}


def _add_queue_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--manifest-dir", default="experiments/auto-sfx/work/phase2c/manifests"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dirty-declaration")
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--license", action="append", required=True, metavar="NAME=VALUE")
    parser.add_argument("--cwd")
    parser.add_argument("--thread-budget", type=int, required=True)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--pass-env", action="append", default=[], metavar="NAME")
    parser.add_argument("--artifact", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--public-metric", action="append", default=[])
    parser.add_argument("--decision", choices=sorted(DECISIONS), default="pending")
    parser.add_argument("--reason", default="awaiting review")
    parser.add_argument("--supersedes", action="append", default=[])
    parser.add_argument("--superseded-by", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subcommand", required=True)
    queue = commands.add_parser("queue", help="create one immutable queued attempt")
    _add_queue_arguments(queue)
    run = commands.add_parser("run", help="queue and execute one attempt")
    _add_queue_arguments(run)
    execute = commands.add_parser("execute", help="execute one queued manifest")
    execute.add_argument("manifest")
    cancel = commands.add_parser("cancel", help="cancel one queued manifest")
    cancel.add_argument("manifest")
    cancel.add_argument("--reason", required=True)
    reconcile = commands.add_parser(
        "reconcile", help="cancel a stale running attempt after verifying its processes are gone"
    )
    reconcile.add_argument("manifest")
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument("--confirm-no-unrecorded-process", action="store_true")
    decide = commands.add_parser("decide", help="record a terminal attempt decision")
    decide.add_argument("manifest")
    decide.add_argument("--decision", required=True, choices=sorted(DECISIONS - {"pending"}))
    decide.add_argument("--reason", required=True)
    decide.add_argument("--supersedes", action="append", default=[])
    decide.add_argument("--superseded-by", action="append", default=[])
    report = commands.add_parser("report", help="build a deterministic completed-only report")
    report.add_argument("--manifest-dir", required=True)
    report.add_argument("--output", default="-")
    report.add_argument(
        "--replace", action="store_true", help="explicitly replace an existing report atomically"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand in {"queue", "run"}:
            if args.thread_budget < 1:
                raise RunnerError("--thread-budget must be positive")
            path = queue_manifest(args)
            if args.subcommand == "queue":
                print(path)
                return 0
            manifest = execute_manifest(path)
            print(path)
            return 0 if manifest["status"] == "complete" else 1
        if args.subcommand == "execute":
            manifest = execute_manifest(Path(args.manifest))
            return 0 if manifest["status"] == "complete" else 1
        if args.subcommand == "cancel":
            cancel_queued(Path(args.manifest), args.reason)
            return 0
        if args.subcommand == "reconcile":
            reconcile_stale_running(
                Path(args.manifest),
                args.reason,
                confirm_no_unrecorded_process=args.confirm_no_unrecorded_process,
            )
            return 0
        if args.subcommand == "decide":
            decide_manifest(
                Path(args.manifest),
                args.decision,
                args.reason,
                args.supersedes,
                args.superseded_by,
            )
            return 0
        if args.subcommand == "report":
            report = build_report(Path(args.manifest_dir))
            if args.output == "-":
                sys.stdout.buffer.write(_canonical_json(report))
            else:
                output = Path(args.output).expanduser().resolve()
                if output.name.endswith(".manifest.json"):
                    raise RunnerError("refusing to overwrite a manifest with a report")
                atomic_write_json(output, report, overwrite=args.replace)
            return 0
        raise AssertionError(args.subcommand)
    except RunnerError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
