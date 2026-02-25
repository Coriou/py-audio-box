#!/usr/bin/env python3
"""
job-watcher.py — Redis stream consumer daemon for remote synthesis execution.

This daemon:
  - leases jobs from Redis Streams consumer groups
  - provisions a Vast instance under a renewable distributed lock
  - executes deterministic manifests remotely through job-runner
  - uploads completed artifacts via rclone
  - writes result/failure records and ACKs stream entries
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import redis
except Exception as _redis_exc:  # pragma: no cover - dependency guard
    redis = None
    REDIS_IMPORT_ERROR = _redis_exc
else:
    REDIS_IMPORT_ERROR = None

_LIB = str(Path(__file__).resolve().parent.parent.parent / "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from jobqueue import JobQueue, JobSpec, StreamLease, build_speak_argv, normalize_job_spec  # noqa: E402


RETRY_BACKOFF_SECONDS = (30, 120, 600)
DEFAULT_VAST_IMAGE = os.getenv("VAST_IMAGE", "ghcr.io/coriou/voice-tools:cuda")
DEFAULT_VAST_DISK_GB = int(os.getenv("VAST_DISK", "60"))
DEFAULT_VAST_REPO = os.getenv("VAST_REPO", "https://github.com/Coriou/py-audio-box")
DEFAULT_VAST_QUERY = os.getenv(
    "VAST_QUERY",
    (
        "reliability > 0.98 gpu_ram >= 20 compute_cap >= 800 compute_cap < 1200 "
        "inet_down >= 500 disk_space >= 50 rented=False"
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_epoch() -> float:
    return time.time()


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def retry_delay_seconds(next_attempt: int) -> int:
    if next_attempt <= 0:
        return RETRY_BACKOFF_SECONDS[0]
    idx = min(next_attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
    return RETRY_BACKOFF_SECONDS[idx]


def classify_failure(error: str) -> str:
    text = error.lower()

    permanent_markers = (
        "invalid voice",
        "voice '",
        "speaker",
        "must be",
        "unsupported",
        "schema",
        "no text provided",
    )
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "network",
        "ssh",
        "rsync",
        "rclone",
        "upload",
        "provision",
        "remote",
    )

    if any(marker in text for marker in permanent_markers):
        return "permanent"
    if any(marker in text for marker in transient_markers):
        return "transient"
    return "transient"


def should_start_run(*, outstanding: int, oldest_age_sec: float | None, batch_min: int, batch_max_wait: int) -> bool:
    if outstanding <= 0:
        return False
    if outstanding >= batch_min:
        return True
    if oldest_age_sec is not None and oldest_age_sec >= float(batch_max_wait):
        return True
    return False


@dataclass(slots=True, frozen=True)
class WatcherConfig:
    redis_url: str
    vast_api_key: str
    vast_ssh_key: Path | None
    s3_bucket: str
    rclone_remote: str

    # Execution and sink backends
    executor_mode: str          # "vast" | "local"
    sink_mode: str              # "local" | "rclone"
    output_root: Path           # local sink: base dir for jobs_out/<topic_id>/
    stream_retention_days: int  # stream XTRIM horizon in days (0 = disabled)

    s3_prefix: str
    poll_interval: int
    batch_min: int
    batch_max_wait: int
    batch_max_jobs: int
    idle_grace: int
    max_runtime: int
    max_attempts: int
    lock_ttl: int
    lock_renew_every: int
    lease_idle_ms: int
    voices_dir: Path
    work_dir: Path
    synth_concurrency: int

    vast_query: str
    vast_image: str
    vast_disk_gb: int
    vast_repo: str
    vast_max_duration: str | None
    ghcr_user: str
    ghcr_token: str | None

    rsync_timeout: int

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        executor_mode = str(os.getenv("WATCHER_EXECUTOR_MODE", "vast")).lower()
        sink_mode = str(os.getenv("WATCHER_SINK_MODE", "local")).lower()

        if executor_mode not in ("vast", "local"):
            raise ValueError(f"WATCHER_EXECUTOR_MODE must be 'vast' or 'local', got: {executor_mode!r}")
        if sink_mode not in ("local", "rclone"):
            raise ValueError(f"WATCHER_SINK_MODE must be 'local' or 'rclone', got: {sink_mode!r}")

        # Required vars differ by mode.
        always_required = ["REDIS_URL"]
        vast_required = ["VAST_API_KEY", "VAST_SSH_KEY"] if executor_mode == "vast" else []
        rclone_required = ["S3_BUCKET", "RCLONE_REMOTE"] if sink_mode == "rclone" else []
        required_names = always_required + vast_required + rclone_required
        missing = [name for name in required_names if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

        vast_ssh_key: Path | None = None
        if os.getenv("VAST_SSH_KEY"):
            vast_ssh_key = Path(str(os.getenv("VAST_SSH_KEY"))).expanduser()

        cfg = cls(
            redis_url=str(os.getenv("REDIS_URL")),
            vast_api_key=str(os.getenv("VAST_API_KEY", "")),
            vast_ssh_key=vast_ssh_key,
            s3_bucket=str(os.getenv("S3_BUCKET", "")),
            rclone_remote=str(os.getenv("RCLONE_REMOTE", "")),
            executor_mode=executor_mode,
            sink_mode=sink_mode,
            output_root=Path(str(os.getenv("WATCHER_OUTPUT_ROOT", "/work_remote/jobs_out"))),
            stream_retention_days=_env_int("WATCHER_STREAM_RETENTION_DAYS", 7),
            s3_prefix=str(os.getenv("S3_PREFIX", "voice-results")),
            poll_interval=_env_int("WATCHER_POLL_INTERVAL", 10),
            batch_min=_env_int("WATCHER_BATCH_MIN", 1),
            batch_max_wait=_env_int("WATCHER_BATCH_MAX_WAIT", 300),
            batch_max_jobs=_env_int("WATCHER_BATCH_MAX_JOBS", 128),
            idle_grace=_env_int("WATCHER_IDLE_GRACE", 75),
            max_runtime=_env_int("WATCHER_MAX_RUNTIME", 7200),
            max_attempts=_env_int("WATCHER_MAX_ATTEMPTS", 3),
            lock_ttl=_env_int("WATCHER_LOCK_TTL", 120),
            lock_renew_every=_env_int("WATCHER_LOCK_RENEW_EVERY", 40),
            lease_idle_ms=_env_int("WATCHER_LEASE_IDLE_MS", 180_000),
            voices_dir=Path(str(os.getenv("WATCHER_VOICES_DIR", "/cache/voices"))),
            work_dir=Path(str(os.getenv("WATCHER_WORK_DIR", "/work_remote"))),
            synth_concurrency=_env_int("WATCHER_SYNTH_CONCURRENCY", 1),
            vast_query=str(os.getenv("VAST_QUERY", DEFAULT_VAST_QUERY)),
            vast_image=str(os.getenv("VAST_IMAGE", DEFAULT_VAST_IMAGE)),
            vast_disk_gb=_env_int("VAST_DISK", DEFAULT_VAST_DISK_GB),
            vast_repo=str(os.getenv("VAST_REPO", DEFAULT_VAST_REPO)),
            vast_max_duration=os.getenv("VAST_MAX_DURATION"),
            ghcr_user=str(os.getenv("GHCR_USER", "Coriou")),
            ghcr_token=os.getenv("GHCR_TOKEN"),
            rsync_timeout=_env_int("WATCHER_RSYNC_TIMEOUT", 300),
        )

        if cfg.poll_interval <= 0:
            raise ValueError("WATCHER_POLL_INTERVAL must be > 0")
        if cfg.batch_min <= 0:
            raise ValueError("WATCHER_BATCH_MIN must be > 0")
        if cfg.batch_max_jobs <= 0:
            raise ValueError("WATCHER_BATCH_MAX_JOBS must be > 0")
        if cfg.max_attempts <= 0:
            raise ValueError("WATCHER_MAX_ATTEMPTS must be > 0")
        if cfg.lock_ttl <= 0:
            raise ValueError("WATCHER_LOCK_TTL must be > 0")
        if cfg.lock_renew_every <= 0:
            raise ValueError("WATCHER_LOCK_RENEW_EVERY must be > 0")
        if cfg.lock_renew_every >= cfg.lock_ttl:
            raise ValueError("WATCHER_LOCK_RENEW_EVERY must be smaller than WATCHER_LOCK_TTL")
        if cfg.synth_concurrency < 1:
            raise ValueError("WATCHER_SYNTH_CONCURRENCY must be >= 1")
        if cfg.synth_concurrency > 1:
            raise ValueError(
                "WATCHER_SYNTH_CONCURRENCY > 1 is not enabled in v1 (OOM safety policy)."
            )
        if executor_mode == "vast":
            if cfg.vast_ssh_key is None or not cfg.vast_ssh_key.exists():
                raise ValueError(f"VAST_SSH_KEY file does not exist: {cfg.vast_ssh_key}")

        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        return cfg


def _log(level: str, event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "level": level,
        "event": event,
    }
    payload.update(fields)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


def _short_error(text: str, limit: int = 400) -> str:
    trimmed = " ".join(text.strip().split())
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[: limit - 3]}..."


def _run_subprocess(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(cmd),
        text=True,
        capture_output=True,
        env=dict(env) if env is not None else None,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stderr: {_short_error(proc.stderr or '')}"
        )
    return proc


class LockRenewer(threading.Thread):
    def __init__(
        self,
        *,
        queue: JobQueue,
        run_id: str,
        ttl_seconds: int,
        renew_every_seconds: int,
        stop_event: threading.Event,
        lock_lost_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.queue = queue
        self.run_id = run_id
        self.ttl_seconds = ttl_seconds
        self.renew_every_seconds = renew_every_seconds
        self.stop_event = stop_event
        self.lock_lost_event = lock_lost_event

    def run(self) -> None:
        while not self.stop_event.wait(self.renew_every_seconds):
            try:
                ok = self.queue.renew_lock(self.run_id, ttl_seconds=self.ttl_seconds)
            except Exception as exc:
                _log("error", "lock_renew_error", run_id=self.run_id, error=str(exc))
                self.lock_lost_event.set()
                return
            if not ok:
                _log("error", "lock_lost", run_id=self.run_id)
                self.lock_lost_event.set()
                return


@dataclass(slots=True)
class VastInstance:
    instance_id: str
    host: str
    port: int
    known_hosts_file: Path


class VastController:
    def __init__(self, config: WatcherConfig) -> None:
        self.config = config

    def _vast_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["VAST_API_KEY"] = self.config.vast_api_key
        return env

    def _vast(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return _run_subprocess(["vastai", *args], env=self._vast_env(), check=check)

    def _wait_for_status(self, instance_id: str, target: str, *, timeout_sec: int = 600) -> None:
        deadline = _now_epoch() + float(timeout_sec)
        last_status = "unknown"
        while _now_epoch() < deadline:
            raw = self._vast("show", "instance", instance_id, "--raw", check=False)
            if raw.returncode == 0 and raw.stdout.strip():
                try:
                    data = json.loads(raw.stdout)
                except json.JSONDecodeError:
                    data = {}
                status = str(data.get("actual_status") or "unknown")
                if status != last_status:
                    _log("info", "instance_status", instance_id=instance_id, status=status, target=target)
                    last_status = status
                if status == target:
                    return
                if status in {"error", "deleted", "exited"}:
                    raise RuntimeError(f"instance {instance_id} entered terminal status: {status}")
            time.sleep(5)
        raise RuntimeError(f"instance {instance_id} did not reach status '{target}' in time")

    def _resolve_ssh(self, instance_id: str) -> tuple[str, int]:
        raw = self._vast("ssh-url", instance_id)
        url = raw.stdout.strip()
        m = re.match(r"^ssh://[^@]+@([^:]+):(\d+)$", url)
        if not m:
            raise RuntimeError(f"unexpected ssh-url format: {url!r}")
        return m.group(1), int(m.group(2))

    def _best_offer_id(self) -> str:
        raw = self._vast(
            "search",
            "offers",
            self.config.vast_query,
            "--order",
            "dlperf_per_dphtotal-",
            "--raw",
        )
        try:
            offers = json.loads(raw.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("failed to decode vast offers JSON") from exc
        if not isinstance(offers, list) or not offers:
            raise RuntimeError("no vast offers matched query")
        top = offers[0]
        offer_id = top.get("id")
        if offer_id is None:
            raise RuntimeError("vast offer payload missing id")
        return str(offer_id)

    def _wait_for_ssh(
        self,
        host: str,
        port: int,
        *,
        ssh_key: Path,
        timeout_sec: int = 120,
        poll_sec: int = 5,
    ) -> None:
        """Poll until SSH accepts connections or timeout_sec elapses."""
        deadline = _now_epoch() + float(timeout_sec)
        last_err = ""
        while _now_epoch() < deadline:
            result = _run_subprocess(
                [
                    "ssh",
                    "-p", str(port),
                    "-i", str(ssh_key),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    f"root@{host}",
                    "true",
                ],
                check=False,
            )
            if result.returncode == 0:
                return
            last_err = (result.stderr or "").strip()
            time.sleep(poll_sec)
        raise RuntimeError(
            f"SSH on {host}:{port} did not become ready within {timeout_sec}s. "
            f"Last error: {last_err}"
        )

    def _wait_for_onstart(
        self,
        host: str,
        port: int,
        *,
        ssh_key: Path,
        marker: str = "/tmp/pab_onstart_done",
        timeout_sec: int = 180,
        poll_sec: int = 10,
    ) -> None:
        """Poll until the onstart-cmd has completed and written its marker file.

        The marker /tmp/pab_onstart_done is the last thing written by the
        onstart script, so its presence proves both code availability and
        git sync completion.  Falls back to checking the legacy sentinel path
        so older images still work.
        """
        legacy_sentinel = "/app/apps/job-runner/job-runner.py"
        deadline = _now_epoch() + float(timeout_sec)
        while _now_epoch() < deadline:
            result = _run_subprocess(
                [
                    "ssh",
                    "-p", str(port),
                    "-i", str(ssh_key),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    f"root@{host}",
                    f"test -f {marker} || test -f {legacy_sentinel}",
                ],
                check=False,
            )
            if result.returncode == 0:
                return
            time.sleep(poll_sec)
        raise RuntimeError(
            f"Onstart marker {marker!r} not found on {host}:{port} "
            f"within {timeout_sec}s — onstart script may have failed"
        )

    def provision(self, *, run_id: str) -> VastInstance:
        offer_id = self._best_offer_id()

        git_sync = (
            # The image already has all source files via COPY .  We initialise
            # git in-place so we can pull the latest main, but we never delete
            # /app — the pre-installed code is always the safe fallback.
            "git -C /app init -q 2>/dev/null; "
            "git -C /app remote add origin "
            f"{shlex.quote(self.config.vast_repo)} 2>/dev/null || "
            "git -C /app remote set-url origin "
            f"{shlex.quote(self.config.vast_repo)} 2>/dev/null; "
            "git -C /app fetch --depth=1 origin main 2>/dev/null && "
            "git -C /app reset --hard FETCH_HEAD 2>/dev/null || true"
        )
        onstart = (
            "chmod 700 /root/.ssh 2>/dev/null; "
            "chmod 600 /root/.ssh/authorized_keys 2>/dev/null; "
            f"{git_sync}; "
            "mkdir -p /work /cache; "
            "chmod +x /app/run-direct 2>/dev/null || true; "
            # Write marker file last so _wait_for_onstart() proves sync is done.
            "touch /tmp/pab_onstart_done"
        )

        args = [
            "create",
            "instance",
            offer_id,
            "--image",
            self.config.vast_image,
            "--disk",
            str(self.config.vast_disk_gb),
            "--ssh",
            "--direct",
            "--cancel-unavail",
            "--label",
            run_id,
            "--onstart-cmd",
            onstart,
            "--raw",
        ]
        if self.config.vast_max_duration:
            args.extend(["--max-dph-duration", self.config.vast_max_duration])
        if self.config.ghcr_token and self.config.vast_image.startswith("ghcr.io/"):
            args.extend(["--login", f"-u {self.config.ghcr_user} -p {self.config.ghcr_token} ghcr.io"])

        raw = self._vast(*args)
        try:
            create_payload = json.loads(raw.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("failed to decode vast create JSON") from exc

        instance_id = str(create_payload.get("new_contract") or create_payload.get("id") or "")
        if not instance_id:
            raise RuntimeError("failed to parse instance id from create response")

        try:
            self._wait_for_status(instance_id, "running")
        except Exception:
            # Instance was created but failed to reach 'running' — destroy it
            # immediately so we don't leave a billing ghost behind.
            self.destroy(instance_id)
            raise
        host, port = self._resolve_ssh(instance_id)

        known_hosts = Path(tempfile.gettempdir()) / f"watcher_known_hosts_{run_id}"
        known_hosts.touch(exist_ok=True)

        try:
            self._wait_for_ssh(host, port, ssh_key=self.config.vast_ssh_key)
            _log("info", "instance_ssh_ready", instance_id=instance_id, host=host, port=port)
            self._wait_for_onstart(host, port, ssh_key=self.config.vast_ssh_key)
            _log("info", "instance_onstart_done", instance_id=instance_id)
        except Exception:
            self.destroy(instance_id)
            raise

        return VastInstance(
            instance_id=instance_id,
            host=host,
            port=port,
            known_hosts_file=known_hosts,
        )

    def destroy(self, instance_id: str) -> None:
        self._vast("destroy", "instance", instance_id, check=False)


class RemoteExecutor:
    def __init__(self, instance: VastInstance, ssh_key: Path) -> None:
        self.instance = instance
        self.ssh_key = ssh_key

    def _ssh_cmd(self, command: str) -> list[str]:
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={self.instance.known_hosts_file}",
            "-o",
            "BatchMode=yes",
            "-i",
            str(self.ssh_key),
            "-p",
            str(self.instance.port),
            f"root@{self.instance.host}",
            command,
        ]

    def run(self, command: str, *, check: bool, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return _run_subprocess(self._ssh_cmd(command), check=check, timeout=timeout)

    # rsync exit codes that are clearly transient network failures — worth retrying.
    _RSYNC_TRANSIENT_CODES = {12, 23, 255}

    def _rsync(self, cmd: list[str], *, timeout: int | None, max_attempts: int = 3, retry_delay: int = 15) -> None:
        for attempt in range(1, max_attempts + 1):
            result = _run_subprocess(cmd, check=False, timeout=timeout)
            if result.returncode == 0:
                return
            if attempt < max_attempts and result.returncode in self._RSYNC_TRANSIENT_CODES:
                _log(
                    "warning",
                    "rsync_retry",
                    attempt=attempt,
                    returncode=result.returncode,
                    stderr=_short_error(result.stderr or ""),
                    delay_sec=retry_delay,
                )
                time.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(cmd)}\nstderr: {result.stderr or ''}"
            )

    def rsync_to_dir(self, local_dir: Path, remote_dir: str, *, timeout: int | None = None) -> None:
        local = Path(local_dir)
        if not local.is_dir():
            raise RuntimeError(f"local rsync source is not a directory: {local}")
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "--mkpath",
            "-e",
            (
                f"ssh -p {self.instance.port} -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile={self.instance.known_hosts_file} "
                f"-o ServerAliveInterval=10 -o ServerAliveCountMax=3"
            ),
            f"{local}/",
            f"root@{self.instance.host}:{remote_dir}/",
        ]
        self._rsync(cmd, timeout=timeout)

    def rsync_from_dir(self, remote_dir: str, local_dir: Path, *, timeout: int | None = None) -> None:
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            "-e",
            (
                f"ssh -p {self.instance.port} -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile={self.instance.known_hosts_file} "
                f"-o ServerAliveInterval=10 -o ServerAliveCountMax=3"
            ),
            f"root@{self.instance.host}:{remote_dir}/",
            f"{local}/",
        ]
        self._rsync(cmd, timeout=timeout)


def _estimate_outstanding(redis_client: Any, queue: JobQueue) -> int:
    # Counts jobs that are neither done nor terminally failed.
    # req:index grows monotonically (set on enqueue, cleared only by retry).
    # result and failed grow monotonically too, so the difference correctly
    # reflects jobs still in-flight or queued — including PEL entries.
    requested = int(redis_client.hlen(queue.keys.request_index))
    done = int(redis_client.hlen(queue.keys.result))
    failed = int(redis_client.hlen(queue.keys.failed))
    return max(0, requested - done - failed)


def _first_queued_age_sec(redis_client: Any, queue: JobQueue) -> float | None:
    raw = redis_client.get(queue.keys.first_queued)
    if raw is None:
        return None
    dt = _parse_iso8601(_to_text(raw))
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def _required_output_files(spec: JobSpec, out_dir: Path) -> tuple[list[Path], Path | None, Path]:
    takes = sorted(out_dir.glob("take_*.wav"))
    if not takes:
        raise RuntimeError(f"missing take_*.wav outputs in {out_dir}")
    for take in takes:
        if take.stat().st_size <= 0:
            raise RuntimeError(f"empty take file: {take}")

    best = out_dir / "best.wav"
    best_path = best if best.exists() else None
    if spec.select_best and best_path is None:
        raise RuntimeError(f"select_best job missing best.wav in {out_dir}")
    if best_path is not None and best_path.stat().st_size <= 0:
        raise RuntimeError(f"empty best.wav: {best_path}")

    meta = out_dir / "takes.meta.json"
    if not meta.exists() or meta.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty takes.meta.json in {out_dir}")

    return takes, best_path, meta


def _s3_uri(bucket: str, prefix: str, output_name: str, filename: str) -> str:
    clean_prefix = prefix.strip().strip("/")
    if clean_prefix:
        return f"s3://{bucket}/{clean_prefix}/{output_name}/{filename}"
    return f"s3://{bucket}/{output_name}/{filename}"


class JobWatcher:
    def __init__(self, config: WatcherConfig, *, consumer_name: str, once: bool = False) -> None:
        if redis is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "redis package is required for job-watcher. Rebuild dependencies first."
            ) from REDIS_IMPORT_ERROR

        self.config = config
        self.consumer_name = consumer_name
        self.once = once

        self.redis = redis.Redis.from_url(
            self.config.redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            decode_responses=False,
        )
        self.redis.ping()

        self.queue = JobQueue(self.redis)
        self.queue.ensure_group()

        self.stop_event = threading.Event()
        self.lock_lost_event = threading.Event()
        self._claim_cursor = "0-0"

    def _metric_incr(self, name: str, count: int = 1) -> None:
        key = f"{self.queue.keys.metrics_prefix}{name}"
        self.redis.incrby(key, int(count))

    def _metric_timing(self, name: str, value: float) -> None:
        key = f"{self.queue.keys.metrics_prefix}{name}:sum"
        self.redis.incrbyfloat(key, float(value))

    def _set_heartbeat(self, run_id: str | None) -> None:
        payload = {
            "run_id": run_id,
            "consumer": self.consumer_name,
            "ts": _utc_now_iso(),
            "pid": os.getpid(),
        }
        self.redis.set(self.queue.keys.watcher_heartbeat, json.dumps(payload), ex=max(10, self.config.poll_interval * 3))

    def _claim_stale(self, *, max_count: int) -> list[StreamLease]:
        try:
            next_cursor, leases = self.queue.claim_stale_jobs(
                consumer_name=self.consumer_name,
                min_idle_ms=self.config.lease_idle_ms,
                start_id=self._claim_cursor,
                count=max_count,
            )
            self._claim_cursor = next_cursor
            if leases:
                _log(
                    "info",
                    "stale_claimed",
                    consumer=self.consumer_name,
                    count=len(leases),
                    next_cursor=next_cursor,
                )
            return leases
        except Exception as exc:
            _log("error", "stale_claim_error", error=str(exc))
            return []

    def _move_due_retries(self, *, limit: int = 100) -> int:
        now_ts = _now_epoch()
        due_members = self.redis.zrangebyscore(self.queue.keys.retry_zset, "-inf", now_ts, start=0, num=limit)
        moved = 0

        for member_raw in due_members:
            member_text = _to_text(member_raw)
            try:
                payload = json.loads(member_text)
            except json.JSONDecodeError:
                self.redis.zrem(self.queue.keys.retry_zset, member_raw)
                _log("error", "retry_payload_invalid_json", payload=member_text)
                continue

            job_payload = payload.get("job")
            if not isinstance(job_payload, Mapping):
                self.redis.zrem(self.queue.keys.retry_zset, member_raw)
                _log("error", "retry_payload_missing_job", payload=payload)
                continue

            request_id = str(payload.get("request_id") or job_payload.get("request_id") or "")
            if not request_id:
                self.redis.zrem(self.queue.keys.retry_zset, member_raw)
                _log("error", "retry_payload_missing_request_id", payload=payload)
                continue

            try:
                spec = normalize_job_spec(dict(job_payload))
            except Exception as exc:
                self.redis.zrem(self.queue.keys.retry_zset, member_raw)
                self._write_failed_terminal(
                    request_id=request_id,
                    job=dict(job_payload),
                    attempt=int(job_payload.get("attempt") or 0),
                    error_type="permanent",
                    error=f"invalid retry payload: {exc}",
                    run_id=None,
                )
                _log("error", "retry_payload_invalid_job", request_id=request_id, error=str(exc))
                continue

            self.redis.hdel(self.queue.keys.request_index, request_id)
            result = self.queue.enqueue(spec)
            if result.status in {"queued", "already_done"}:
                self.redis.zrem(self.queue.keys.retry_zset, member_raw)
                if result.status == "queued":
                    moved += 1
                    self._metric_incr("jobs_retried")
                _log("info", "retry_reenqueued", request_id=request_id, status=result.status)
                continue

            # Duplicate can happen transiently under races; push slightly forward.
            self.redis.zadd(self.queue.keys.retry_zset, {member_raw: now_ts + 5})
            _log("info", "retry_deferred_duplicate", request_id=request_id)

        return moved

    def _validate_voice_or_fail(self, lease: StreamLease, *, run_id: str | None) -> bool:
        spec = lease.job
        if not spec.voice:
            return True

        voice_dir = self.config.voices_dir / spec.voice
        voice_json = voice_dir / "voice.json"
        if voice_dir.exists() and voice_json.exists():
            return True

        self._write_failed_terminal(
            request_id=spec.request_id,
            job=spec.to_dict(),
            attempt=spec.attempt,
            error_type="permanent",
            error=f"invalid voice: '{spec.voice}' not found in {self.config.voices_dir}",
            run_id=run_id,
            stage="validate",
        )
        self.queue.ack(lease.stream_id)
        self._metric_incr("jobs_failed")
        _log(
            "warning",
            "job_failed_invalid_voice",
            request_id=spec.request_id,
            stream_id=lease.stream_id,
            voice=spec.voice,
        )
        return False

    def _schedule_retry(self, lease: StreamLease, *, error: str, run_id: str | None, stage: str | None = None) -> None:
        next_attempt = lease.job.attempt + 1
        delay_sec = retry_delay_seconds(next_attempt)
        due_ts = _now_epoch() + float(delay_sec)

        payload = lease.job.to_dict()
        payload["attempt"] = next_attempt
        payload["queued_at"] = _utc_now_iso()

        member_payload: dict[str, Any] = {
            "schema_version": 1,
            "request_id": lease.job.request_id,
            "stream_id": lease.stream_id,
            "scheduled_at": _utc_now_iso(),
            "due_at": due_ts,
            "run_id": run_id,
            "job": payload,
            "error": _short_error(error),
        }
        if stage:
            member_payload["stage"] = stage
        member = json.dumps(member_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

        pipe = self.redis.pipeline(transaction=True)
        pipe.zadd(self.queue.keys.retry_zset, {member: due_ts})
        pipe.xack(self.queue.keys.stream, self.queue.consumer_group, lease.stream_id)
        pipe.execute()

        self._metric_incr("jobs_retried")
        _log(
            "info",
            "job_retry_scheduled",
            request_id=lease.job.request_id,
            stream_id=lease.stream_id,
            attempt=next_attempt,
            delay_sec=delay_sec,
            due_at=due_ts,
        )

    def _write_failed_terminal(
        self,
        *,
        request_id: str,
        job: Mapping[str, Any],
        attempt: int,
        error_type: str,
        error: str,
        run_id: str | None,
        stage: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "request_id": request_id,
            "status": "failed",
            "attempt": attempt,
            "error_type": error_type,
            "error": _short_error(error),
            "failed_at": _utc_now_iso(),
            "run_id": run_id,
            "job": dict(job),
        }
        if stage:
            record["stage"] = stage
        self.redis.hset(
            self.queue.keys.failed,
            request_id,
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
        self._record_package_failure(
            request_id=request_id,
            job=job,
            error_type=error_type,
            error=error,
            run_id=run_id,
            stage=stage,
        )

    def _upload_output_dir(self, out_dir: Path, output_name: str) -> dict[str, Any]:
        """Upload/copy beat output files; return a sink-mode-specific artifact descriptor."""
        if self.config.sink_mode == "local":
            return self._copy_to_local_package(out_dir, output_name)
        return self._upload_via_rclone(out_dir, output_name)

    def _copy_to_local_package(self, out_dir: Path, output_name: str) -> dict[str, Any]:
        """Copy validated outputs into the deterministic local package tree.

        Layout: <output_root>/<topic_id>/beats/<beat_tag>/<files>
        output_name is "<topic_id>/<beat_tag>" (e.g. "why-pineapple/beat-001").
        """
        parts = Path(output_name).parts  # ("why-pineapple", "beat-001")
        if len(parts) != 2:
            raise RuntimeError(f"unexpected output_name format for local sink: {output_name!r}")
        topic_id, beat_tag = parts
        pkg_beat_dir = self.config.output_root / topic_id / "beats" / beat_tag
        pkg_beat_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str] = {}
        for src in sorted(out_dir.iterdir()):
            dst = pkg_beat_dir / src.name
            shutil.copy2(src, dst)
            rel = f"beats/{beat_tag}/{src.name}"
            artifacts[src.name] = rel

        takes = sorted(k for k in artifacts if k.startswith("take_") and k.endswith(".wav"))
        meta_rel = artifacts.get("takes.meta.json")
        best_rel = artifacts.get("best.wav")
        result_rel = artifacts.get("result.json")

        return {
            "sink": "local",
            "package_root": str(self.config.output_root / topic_id),
            "takes": [artifacts[t] for t in takes],
            "meta": meta_rel,
            "best": best_rel,
            "result": result_rel,
        }

    def _upload_via_rclone(self, out_dir: Path, output_name: str) -> dict[str, Any]:
        """Upload to rclone remote and return s3:// artifact descriptors."""
        dst = f"{self.config.rclone_remote}:{self.config.s3_bucket}/{self.config.s3_prefix}/{output_name}/"
        cmd = [
            "rclone",
            "copy",
            "--checkers",
            "8",
            "--transfers",
            "4",
            "--fast-list",
            f"{out_dir}/",
            dst,
        ]
        _run_subprocess(cmd, check=True)

        takes = sorted(out_dir.glob("take_*.wav"))
        meta = out_dir / "takes.meta.json"
        best = out_dir / "best.wav"
        return {
            "sink": "rclone",
            "takes": [
                _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, t.name)
                for t in takes
            ],
            "meta": _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, meta.name),
            "best": (
                _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, best.name)
                if best.exists() else None
            ),
        }

    def _write_done_and_ack(
        self,
        *,
        lease: StreamLease,
        run_id: str,
        artifacts: dict[str, Any],
    ) -> None:
        record = {
            "request_id": lease.job.request_id,
            "content_hash": lease.job.content_hash,
            "status": "done",
            "attempt": lease.job.attempt,
            "run_id": run_id,
            "completed_at": _utc_now_iso(),
            "output_name": lease.job.output_name,
            "outputs": artifacts,
            "callback_data": lease.job.callback_data,
        }

        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(
            self.queue.keys.result,
            lease.job.request_id,
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
        pipe.hdel(self.queue.keys.failed, lease.job.request_id)
        pipe.xack(self.queue.keys.stream, self.queue.consumer_group, lease.stream_id)
        pipe.execute()

        self._metric_incr("jobs_done")

        if self.config.sink_mode == "local":
            self._assemble_package(lease=lease, artifacts=artifacts, run_id=run_id)

    def _assemble_package(
        self,
        *,
        lease: StreamLease,
        artifacts: dict[str, Any],
        run_id: str,
    ) -> None:
        """Upsert the topic-level job.json package manifest after each successful beat.

        The package manifest aggregates all beat results into a single machine-readable
        file at <output_root>/<topic_id>/job.json.  Paths are relative to the topic root
        for portability.
        """
        cb = lease.job.callback_data or {}
        topic_id = str(cb.get("topic_id") or lease.job.output_name.split("/")[0])
        beat_id = cb.get("beat_id")
        total_beats = int(cb.get("total_beats") or 0)
        request_id = lease.job.request_id
        output_name = lease.job.output_name
        beat_tag = Path(output_name).name  # "beat-001"

        pkg_dir = self.config.output_root / topic_id
        pkg_dir.mkdir(parents=True, exist_ok=True)
        job_json_path = pkg_dir / "job.json"

        # Serialize new beat entry.
        beat_entry: dict[str, Any] = {
            "beat_id": beat_id,
            "request_id": request_id,
            "output_name": output_name,
            "status": "done",
            "audio": {
                "takes": artifacts.get("takes", []),
                "best": artifacts.get("best"),
                "selected": (artifacts.get("best") or (artifacts.get("takes") or [None])[0]),
            },
            "meta": {
                "result_json": artifacts.get("result"),
                "takes_meta_json": artifacts.get("meta"),
            },
            "callback_data": dict(cb),
        }

        # Atomic read-modify-write under an advisory flock so concurrent watcher
        # processes (or future parallel sinks) do not clobber each other.
        lock_path = pkg_dir / "job.json.lock"
        with open(lock_path, "w") as lock_fh:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
            except Exception:
                pass  # Non-fatal if locking unsupported (e.g. Docker tmpfs)

            now = _utc_now_iso()
            if job_json_path.exists():
                try:
                    doc = json.loads(job_json_path.read_text(encoding="utf-8"))
                except Exception:
                    doc = {}
            else:
                doc = {}

            beats_list: list[dict[str, Any]] = doc.get("beats") or []
            # Replace existing entry for this beat or append.
            beats_list = [b for b in beats_list if b.get("request_id") != request_id]
            beats_list.append(beat_entry)
            beats_list.sort(key=lambda b: b.get("beat_id") or 0)

            failures: list[dict[str, Any]] = doc.get("failures") or []
            # Remove this request from failures if it previously failed.
            failures = [f for f in failures if f.get("request_id") != request_id]

            done_ids = {b["request_id"] for b in beats_list if b.get("status") == "done"}
            completed_beats = len(done_ids)
            status = "complete" if (total_beats > 0 and completed_beats >= total_beats) else "partial"

            updated_doc: dict[str, Any] = {
                "schema_version": 1,
                "topic_id": topic_id,
                "status": status,
                "created_at": doc.get("created_at") or now,
                "updated_at": now,
                "total_beats": total_beats,
                "completed_beats": completed_beats,
                "run_id": run_id,
                "beats": beats_list,
                "failures": failures,
            }
            # Preserve topic_title if present in any beat's callback_data.
            if not updated_doc.get("topic_title"):
                for b in beats_list:
                    title = (b.get("callback_data") or {}).get("topic_title")
                    if title:
                        updated_doc["topic_title"] = title
                        break

            job_json_path.write_text(
                json.dumps(updated_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            try:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
            except Exception:
                pass

        _log(
            "info",
            "package_updated",
            topic_id=topic_id,
            beat_id=beat_id,
            completed_beats=completed_beats,
            total_beats=total_beats,
            status=status,
        )

    def _record_package_failure(
        self,
        *,
        request_id: str,
        job: Mapping[str, Any],
        error_type: str,
        error: str,
        run_id: str | None,
        stage: str | None = None,
    ) -> None:
        """Upsert a failure entry in the topic-level job.json when sink_mode=local.

        Called from _write_failed_terminal so that terminal failures are visible
        in the package manifest alongside successful beats.
        """
        if self.config.sink_mode != "local":
            return
        cb = job.get("callback_data") or {}
        topic_id = str(cb.get("topic_id") or "")
        if not topic_id:
            return  # cannot locate the package without topic_id

        beat_id = cb.get("beat_id")
        total_beats = int(cb.get("total_beats") or 0)
        output_name = str(job.get("output_name") or "")

        failure_entry: dict[str, Any] = {
            "beat_id": beat_id,
            "request_id": request_id,
            "output_name": output_name,
            "status": "failed",
            "error_type": error_type,
            "error": _short_error(error),
            "failed_at": _utc_now_iso(),
            "run_id": run_id,
            "callback_data": dict(cb),
        }
        if stage:
            failure_entry["stage"] = stage

        pkg_dir = self.config.output_root / topic_id
        try:
            pkg_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log("warning", "package_failure_mkdir_error", topic_id=topic_id, error=str(exc))
            return

        job_json_path = pkg_dir / "job.json"
        lock_path = pkg_dir / "job.json.lock"
        with open(lock_path, "w") as lock_fh:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
            except Exception:
                pass  # Non-fatal if locking unsupported

            now = _utc_now_iso()
            if job_json_path.exists():
                try:
                    doc = json.loads(job_json_path.read_text(encoding="utf-8"))
                except Exception:
                    doc = {}
            else:
                doc = {}

            beats_list: list[dict[str, Any]] = doc.get("beats") or []
            failures: list[dict[str, Any]] = doc.get("failures") or []
            failures = [f for f in failures if f.get("request_id") != request_id]
            failures.append(failure_entry)

            done_ids = {b["request_id"] for b in beats_list if b.get("status") == "done"}
            completed_beats = len(done_ids)
            status = "complete" if (total_beats > 0 and completed_beats >= total_beats) else "partial"

            updated_doc: dict[str, Any] = {
                "schema_version": 1,
                "topic_id": topic_id,
                "status": status,
                "created_at": doc.get("created_at") or now,
                "updated_at": now,
                "total_beats": total_beats,
                "completed_beats": completed_beats,
                "run_id": run_id,
                "beats": beats_list,
                "failures": failures,
            }
            # Preserve topic_title from existing doc, beat callback data, or this beat.
            topic_title = doc.get("topic_title") or cb.get("topic_title")
            if not topic_title:
                for b in beats_list:
                    topic_title = (b.get("callback_data") or {}).get("topic_title")
                    if topic_title:
                        break
            if topic_title:
                updated_doc["topic_title"] = topic_title

            job_json_path.write_text(
                json.dumps(updated_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            try:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
            except Exception:
                pass

        _log(
            "info",
            "package_failure_recorded",
            topic_id=topic_id,
            beat_id=beat_id,
            request_id=request_id,
            error_type=error_type,
            stage=stage,
        )

    def _trim_stream(self) -> None:
        """Trim acknowledged (old) stream entries based on retention window.

        Uses XTRIM MINID to remove entries older than stream_retention_days.
        Only runs when stream_retention_days > 0.
        """
        if self.config.stream_retention_days <= 0:
            return
        horizon_ms = int(
            (time.time() - self.config.stream_retention_days * 86400) * 1000
        )
        if horizon_ms <= 0:
            return
        # MINID: entries with IDs less than <ts>-0 are trimmed.
        minid = f"{horizon_ms}-0"
        try:
            trimmed = self.redis.xtrim(self.queue.keys.stream, minid=minid, approximate=True)
            if trimmed:
                self._metric_incr("stream_trimmed", trimmed)
                _log("info", "stream_trimmed", count=trimmed, minid=minid)
        except Exception as exc:
            _log("warning", "stream_trim_error", error=str(exc))

    def _handle_execution_failure(self, lease: StreamLease, *, error: str, run_id: str, stage: str | None = None) -> None:
        classification = classify_failure(error)

        if classification == "transient" and (lease.job.attempt + 1) < self.config.max_attempts:
            self._schedule_retry(lease, error=error, run_id=run_id, stage=stage)
            return

        terminal_type = "transient_exhausted" if classification == "transient" else "permanent"
        self._write_failed_terminal(
            request_id=lease.job.request_id,
            job=lease.job.to_dict(),
            attempt=lease.job.attempt,
            error_type=terminal_type,
            error=error,
            run_id=run_id,
            stage=stage,
        )
        self.queue.ack(lease.stream_id)
        self._metric_incr("jobs_failed")

    def _manifest_for_batch(
        self,
        *,
        run_id: str,
        batch_dir: Path,
        remote_batch_dir: PurePosixPath,
        leases: Sequence[StreamLease],
    ) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        for lease in leases:
            spec = lease.job
            local_out = batch_dir / spec.output_name
            local_out.mkdir(parents=True, exist_ok=True)
            (local_out / "text.txt").write_text(spec.text, encoding="utf-8")

            remote_out = remote_batch_dir.joinpath(*Path(spec.output_name).parts)
            remote_text = remote_out / "text.txt"
            remote_result = remote_out / "result.json"

            argv = build_speak_argv(
                spec,
                text_file=str(remote_text),
                out_exact=str(remote_out),
                json_result=str(remote_result),
            )

            jobs.append(
                {
                    "request_id": spec.request_id,
                    "stream_id": lease.stream_id,
                    "attempt": spec.attempt,
                    "output_name": spec.output_name,
                    "out_exact": str(remote_out),
                    "argv": argv,
                }
            )

        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "jobs": jobs,
        }
        with open(batch_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        return manifest

    def _load_execution_map(self, batch_dir: Path) -> dict[str, Mapping[str, Any]]:
        execution_path = batch_dir / "execution.json"
        if not execution_path.exists():
            return {}
        try:
            with open(execution_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return {}
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            return {}
        mapping: dict[str, Mapping[str, Any]] = {}
        for item in jobs:
            if isinstance(item, Mapping) and item.get("request_id"):
                mapping[str(item["request_id"])] = item
        return mapping

    def _run_batch(
        self,
        *,
        run_id: str,
        batch_index: int,
        leases: Sequence[StreamLease],
        instance: VastInstance | None,
    ) -> None:
        """Execute a single batch of jobs.

        When instance is None the executor_mode is "local": the manifest runs
        in-process without any Vast or SSH involvement.  When instance is a
        VastInstance the manifest is rsynced + executed remotely over SSH.
        """
        batch_run_id = f"{run_id}-b{batch_index:04d}"
        batch_dir = self.config.work_dir / batch_run_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        if self.config.executor_mode == "local":
            # Local paths are used for both staging and execution.
            exec_batch_dir = PurePosixPath(str(batch_dir))
        else:
            exec_batch_dir = PurePosixPath("/work") / batch_run_id

        self._manifest_for_batch(
            run_id=batch_run_id,
            batch_dir=batch_dir,
            remote_batch_dir=exec_batch_dir,
            leases=leases,
        )

        if self.config.executor_mode == "local":
            # ── Local executor ──────────────────────────────────────────────
            local_manifest = batch_dir / "manifest.json"
            job_runner_py = Path(__file__).resolve().parent.parent / "job-runner" / "job-runner.py"
            exec_result = _run_subprocess(
                [
                    sys.executable,
                    str(job_runner_py),
                    "execute-manifest",
                    str(local_manifest),
                    "--fail-fast",
                    "--json",
                ],
                check=False,
            )
        else:
            # ── Vast / remote executor ──────────────────────────────────────
            assert instance is not None, "VastInstance required for vast executor mode"
            executor = RemoteExecutor(instance, self.config.vast_ssh_key)  # type: ignore[arg-type]

            required_voices = sorted({lease.job.voice for lease in leases if lease.job.voice})
            for slug in required_voices:
                assert slug is not None
                executor.rsync_to_dir(self.config.voices_dir / slug, f"/cache/voices/{slug}", timeout=self.config.rsync_timeout)

            t_sync_start = _now_epoch()
            executor.rsync_to_dir(batch_dir, str(exec_batch_dir), timeout=self.config.rsync_timeout)
            sync_sec = _now_epoch() - t_sync_start
            self._metric_timing("sync_duration_sec", sync_sec)

            remote_manifest = exec_batch_dir / "manifest.json"
            exec_cmd = (
                "cd /app && "
                "python apps/job-runner/job-runner.py "
                f"execute-manifest {shlex.quote(str(remote_manifest))} --fail-fast --json"
            )
            exec_result = executor.run(exec_cmd, check=False)

            t_pull_start = _now_epoch()
            executor.rsync_from_dir(str(exec_batch_dir), batch_dir, timeout=self.config.rsync_timeout)
            pull_sec = _now_epoch() - t_pull_start
            self._metric_timing("pull_duration_sec", pull_sec)

        _log(
            "info" if exec_result.returncode == 0 else "warning",
            "execution_result",
            run_id=run_id,
            executor=self.config.executor_mode,
            returncode=exec_result.returncode,
            stderr=_short_error(exec_result.stderr or ""),
            stdout=_short_error(exec_result.stdout or ""),
        )

        execution_map = self._load_execution_map(batch_dir)

        for lease in leases:
            req_id = lease.job.request_id
            exec_entry = execution_map.get(req_id)

            if exec_entry is None:
                err = (
                    "missing execution entry"
                    f" (exit={exec_result.returncode}, stderr={_short_error(exec_result.stderr or '')})"
                )
                self._handle_execution_failure(lease, error=err, run_id=run_id, stage="execute")
                continue

            status = str(exec_entry.get("status") or "failed")
            if status != "ok":
                stderr_log = exec_entry.get("stderr_log")
                stderr_text = ""
                if isinstance(stderr_log, str):
                    basename = Path(stderr_log).name
                    matches = list(batch_dir.rglob(basename))
                    if matches:
                        stderr_text = _short_error(
                            matches[0].read_text(encoding="utf-8", errors="ignore")
                        )
                err = (
                    f"execution failed (request_id={req_id}, return_code={exec_entry.get('return_code')}, "
                    f"stderr={stderr_text or _short_error(exec_result.stderr or '')})"
                )
                self._handle_execution_failure(lease, error=err, run_id=run_id, stage="execute")
                continue

            out_dir = batch_dir / lease.job.output_name
            try:
                _required_output_files(lease.job, out_dir)
                t_upload_start = _now_epoch()
                artifacts = self._upload_output_dir(out_dir, lease.job.output_name)
                upload_sec = _now_epoch() - t_upload_start
                self._metric_timing("upload_duration_sec", upload_sec)
                self._write_done_and_ack(
                    lease=lease,
                    run_id=run_id,
                    artifacts=artifacts,
                )
            except Exception as exc:
                self._metric_incr("upload_failures")
                self._handle_execution_failure(lease, error=str(exc), run_id=run_id, stage="upload")

    def _run_locked_session(self, *, run_id: str, reclaimed: Sequence[StreamLease]) -> None:
        start_ts = _now_epoch()
        self._set_heartbeat(run_id)
        _log(
            "info",
            "session_start",
            run_id=run_id,
            reclaimed=len(reclaimed),
            executor=self.config.executor_mode,
            sink=self.config.sink_mode,
        )

        pending: list[StreamLease] = list(reclaimed)
        instance: VastInstance | None = None
        vast = VastController(self.config) if self.config.executor_mode == "vast" else None

        idle_start: float | None = None
        batch_index = 0

        try:
            while not self.stop_event.is_set() and not self.lock_lost_event.is_set():
                if (_now_epoch() - start_ts) >= float(self.config.max_runtime):
                    _log("warning", "session_max_runtime", run_id=run_id)
                    break

                self._set_heartbeat(run_id)
                # Move due retries and claim stale entries inside the lock —
                # no other consumer should mutate stream ownership concurrently.
                self._move_due_retries(limit=max(10, self.config.batch_max_jobs))
                pending.extend(self._claim_stale(max_count=self.config.batch_max_jobs))

                if len(pending) < self.config.batch_max_jobs:
                    needed = max(1, self.config.batch_max_jobs - len(pending))
                    # Use non-blocking fetch when we already have work, or cap at
                    # 4 000 ms so we stay safely under socket_timeout (5 s).
                    block_ms = 0 if pending else min(4000, self.config.poll_interval * 1000)
                    leased = self.queue.lease_jobs(
                        consumer_name=self.consumer_name,
                        count=needed,
                        block_ms=block_ms,
                    )
                    pending.extend(leased)

                if not pending:
                    outstanding = _estimate_outstanding(self.redis, self.queue)
                    retry_count = int(self.redis.zcard(self.queue.keys.retry_zset))
                    if outstanding == 0 and retry_count == 0:
                        # Queue is fully drained — clear the age hint so a stale
                        # first_queued timestamp doesn't trigger batch_max_wait on
                        # the next poll cycle.
                        self.redis.delete(self.queue.keys.first_queued)

                    if self.config.executor_mode == "local" or instance is None:
                        break

                    if idle_start is None:
                        idle_start = _now_epoch()
                    if (_now_epoch() - idle_start) >= float(self.config.idle_grace):
                        _log("info", "session_idle_grace_reached", run_id=run_id)
                        break
                    continue

                idle_start = None
                batch = pending[: self.config.batch_max_jobs]
                pending = pending[self.config.batch_max_jobs :]

                valid: list[StreamLease] = []
                for lease in batch:
                    if self._validate_voice_or_fail(lease, run_id=run_id):
                        valid.append(lease)

                if not valid:
                    continue

                if self.config.executor_mode == "vast" and instance is None:
                    assert vast is not None
                    t0 = _now_epoch()
                    instance = vast.provision(run_id=run_id)
                    self.redis.set(self.queue.keys.watcher_instance, instance.instance_id)
                    self._metric_incr("provisioning_count")
                    self._metric_timing("provision_duration_sec", _now_epoch() - t0)
                    _log(
                        "info",
                        "instance_provisioned",
                        run_id=run_id,
                        instance_id=instance.instance_id,
                        host=instance.host,
                        port=instance.port,
                    )

                batch_index += 1
                t_batch = _now_epoch()
                self._run_batch(run_id=run_id, batch_index=batch_index, leases=valid, instance=instance)
                self._metric_timing("batch_duration_sec", _now_epoch() - t_batch)

            # Trim old stream entries once per session to keep Redis bounded.
            self._trim_stream()

        finally:
            if instance is not None:
                try:
                    assert vast is not None
                    vast.destroy(instance.instance_id)
                    _log("info", "instance_destroyed", run_id=run_id, instance_id=instance.instance_id)
                except Exception as exc:
                    _log(
                        "error",
                        "instance_destroy_error",
                        run_id=run_id,
                        instance_id=instance.instance_id,
                        error=str(exc),
                    )
                self.redis.delete(self.queue.keys.watcher_instance)
                try:
                    instance.known_hosts_file.unlink(missing_ok=True)
                except Exception:
                    pass

            self._set_heartbeat(None)
            _log("info", "session_end", run_id=run_id)

    def _has_stale_pel_entries(self) -> bool:
        """Read-only check: are there any PEL entries idle longer than lease_idle_ms?"""
        try:
            entries = self.redis.xpending_range(
                self.queue.keys.stream,
                self.queue.consumer_group,
                "-",
                "+",
                1,
                idle=max(1, self.config.lease_idle_ms),
            )
            return bool(entries)
        except Exception:
            return False

    def _acquire_and_run(self) -> bool:
        # Phase 1: read-only trigger evaluation — no stream mutations.
        outstanding = _estimate_outstanding(self.redis, self.queue)
        oldest_age = _first_queued_age_sec(self.redis, self.queue)
        has_stale = self._has_stale_pel_entries()

        trigger = should_start_run(
            outstanding=outstanding,
            oldest_age_sec=oldest_age,
            batch_min=self.config.batch_min,
            batch_max_wait=self.config.batch_max_wait,
        ) or has_stale

        if not trigger:
            return False

        # Phase 2: acquire distributed lock.
        run_id = f"watcher-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        if not self.queue.acquire_lock(run_id, ttl_seconds=self.config.lock_ttl):
            _log("info", "lock_busy", run_id=run_id)
            return False

        _log(
            "info",
            "lock_acquired",
            run_id=run_id,
            outstanding=outstanding,
            oldest_age_sec=oldest_age,
            has_stale=has_stale,
        )

        renew_stop = threading.Event()
        self.lock_lost_event.clear()
        renewer = LockRenewer(
            queue=self.queue,
            run_id=run_id,
            ttl_seconds=self.config.lock_ttl,
            renew_every_seconds=self.config.lock_renew_every,
            stop_event=renew_stop,
            lock_lost_event=self.lock_lost_event,
        )
        renewer.start()

        try:
            # Phase 3: inside lock — stale claim + retry moves happen inside
            # _run_locked_session on its first iteration.
            self._run_locked_session(run_id=run_id, reclaimed=[])
            return True
        finally:
            renew_stop.set()
            renewer.join(timeout=5)
            if self.queue.release_lock(run_id):
                _log("info", "lock_released", run_id=run_id)
            else:
                _log("warning", "lock_release_skipped", run_id=run_id)

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        self.stop_event.set()
        _log("warning", "signal_received", signal=signum)

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        _log("info", "watcher_start", consumer=self.consumer_name)

        while not self.stop_event.is_set():
            try:
                self._set_heartbeat(None)
                # _move_due_retries and _claim_stale are both called inside the
                # locked session, so we do not call them here to avoid the
                # lock/claim race (non-owner mutating stream state).
                ran = self._acquire_and_run()
                if self.once:
                    break
                if not ran:
                    time.sleep(self.config.poll_interval)
            except Exception as exc:
                self._metric_incr("watcher_errors")
                _log("error", "watcher_loop_error", error=str(exc))
                if self.once:
                    raise
                time.sleep(self.config.poll_interval)

        _log("info", "watcher_stop", consumer=self.consumer_name)


def _default_consumer_name() -> str:
    host = socket.gethostname().split(".")[0]
    return f"watcher-{host}-{os.getpid()}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="job-watcher — Redis stream daemon for Vast-backed synthesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--once", action="store_true", help="Run at most one scheduling cycle and exit.")
    ap.add_argument("--consumer-name", default=_default_consumer_name())
    ap.add_argument(
        "--ping",
        action="store_true",
        help="Print 'ok' and exit 0 without connecting to Redis. Used by healthchecks.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # Lightweight liveness probe: no Redis, no env validation required.
    if args.ping:
        print("ok")
        raise SystemExit(0)

    try:
        config = WatcherConfig.from_env()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    watcher = JobWatcher(config, consumer_name=str(args.consumer_name), once=bool(args.once))
    watcher.run()


if __name__ == "__main__":
    main()
