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
import json
import os
import re
import shlex
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
    vast_ssh_key: Path
    s3_bucket: str
    rclone_remote: str

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

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        required_names = ["REDIS_URL", "VAST_API_KEY", "VAST_SSH_KEY", "S3_BUCKET", "RCLONE_REMOTE"]
        missing = [name for name in required_names if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

        cfg = cls(
            redis_url=str(os.getenv("REDIS_URL")),
            vast_api_key=str(os.getenv("VAST_API_KEY")),
            vast_ssh_key=Path(str(os.getenv("VAST_SSH_KEY"))).expanduser(),
            s3_bucket=str(os.getenv("S3_BUCKET")),
            rclone_remote=str(os.getenv("RCLONE_REMOTE")),
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
        if not cfg.vast_ssh_key.exists():
            raise ValueError(f"VAST_SSH_KEY file does not exist: {cfg.vast_ssh_key}")

        cfg.work_dir.mkdir(parents=True, exist_ok=True)
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

    def _wait_for_status(self, instance_id: str, target: str, *, timeout_sec: int = 1500) -> None:
        deadline = _now_epoch() + float(timeout_sec)
        while _now_epoch() < deadline:
            raw = self._vast("show", "instance", instance_id, "--raw", check=False)
            if raw.returncode == 0 and raw.stdout.strip():
                try:
                    data = json.loads(raw.stdout)
                except json.JSONDecodeError:
                    data = {}
                status = str(data.get("actual_status") or "unknown")
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

    def provision(self, *, run_id: str) -> VastInstance:
        offer_id = self._best_offer_id()

        git_sync = (
            "git -C /app remote set-url origin "
            f"{shlex.quote(self.config.vast_repo)} 2>/dev/null; "
            "git -C /app fetch --depth=1 origin main 2>/dev/null && "
            "git -C /app reset --hard FETCH_HEAD 2>/dev/null || "
            f"(rm -rf /app 2>/dev/null; git clone --depth=1 {shlex.quote(self.config.vast_repo)} /app 2>/dev/null) || true"
        )
        onstart = (
            "chmod 700 /root/.ssh 2>/dev/null; "
            "chmod 600 /root/.ssh/authorized_keys 2>/dev/null; "
            f"{git_sync}; "
            "mkdir -p /work /cache; "
            "chmod +x /app/run-direct 2>/dev/null || true"
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

        self._wait_for_status(instance_id, "running")
        host, port = self._resolve_ssh(instance_id)

        known_hosts = Path(tempfile.gettempdir()) / f"watcher_known_hosts_{run_id}"
        known_hosts.touch(exist_ok=True)

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

    def rsync_to_dir(self, local_dir: Path, remote_dir: str) -> None:
        local = Path(local_dir)
        if not local.is_dir():
            raise RuntimeError(f"local rsync source is not a directory: {local}")
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "-e",
            (
                f"ssh -p {self.instance.port} -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile={self.instance.known_hosts_file}"
            ),
            f"{local}/",
            f"root@{self.instance.host}:{remote_dir}/",
        ]
        _run_subprocess(cmd, check=True)

    def rsync_from_dir(self, remote_dir: str, local_dir: Path) -> None:
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            "-e",
            (
                f"ssh -p {self.instance.port} -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile={self.instance.known_hosts_file}"
            ),
            f"root@{self.instance.host}:{remote_dir}/",
            f"{local}/",
        ]
        _run_subprocess(cmd, check=True)


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

    def _schedule_retry(self, lease: StreamLease, *, error: str, run_id: str | None) -> None:
        next_attempt = lease.job.attempt + 1
        delay_sec = retry_delay_seconds(next_attempt)
        due_ts = _now_epoch() + float(delay_sec)

        payload = lease.job.to_dict()
        payload["attempt"] = next_attempt
        payload["queued_at"] = _utc_now_iso()

        member_payload = {
            "schema_version": 1,
            "request_id": lease.job.request_id,
            "stream_id": lease.stream_id,
            "scheduled_at": _utc_now_iso(),
            "due_at": due_ts,
            "run_id": run_id,
            "job": payload,
            "error": _short_error(error),
        }
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
    ) -> None:
        record = {
            "request_id": request_id,
            "status": "failed",
            "attempt": attempt,
            "error_type": error_type,
            "error": _short_error(error),
            "failed_at": _utc_now_iso(),
            "run_id": run_id,
            "job": dict(job),
        }
        self.redis.hset(
            self.queue.keys.failed,
            request_id,
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )

    def _write_done_and_ack(
        self,
        *,
        lease: StreamLease,
        run_id: str,
        takes: list[Path],
        best: Path | None,
        meta: Path,
    ) -> None:
        output_name = lease.job.output_name
        outputs: dict[str, Any] = {
            "takes": [
                _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, take.name)
                for take in takes
            ],
            "meta": _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, meta.name),
        }
        if best is not None:
            outputs["best"] = _s3_uri(self.config.s3_bucket, self.config.s3_prefix, output_name, best.name)

        record = {
            "request_id": lease.job.request_id,
            "content_hash": lease.job.content_hash,
            "status": "done",
            "attempt": lease.job.attempt,
            "run_id": run_id,
            "completed_at": _utc_now_iso(),
            "output_name": output_name,
            "outputs": outputs,
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

    def _upload_output_dir(self, out_dir: Path, output_name: str) -> None:
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

    def _handle_execution_failure(self, lease: StreamLease, *, error: str, run_id: str) -> None:
        classification = classify_failure(error)

        if classification == "transient" and (lease.job.attempt + 1) < self.config.max_attempts:
            self._schedule_retry(lease, error=error, run_id=run_id)
            return

        terminal_type = "transient_exhausted" if classification == "transient" else "permanent"
        self._write_failed_terminal(
            request_id=lease.job.request_id,
            job=lease.job.to_dict(),
            attempt=lease.job.attempt,
            error_type=terminal_type,
            error=error,
            run_id=run_id,
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
        instance: VastInstance,
    ) -> None:
        batch_run_id = f"{run_id}-b{batch_index:04d}"
        batch_dir = self.config.work_dir / batch_run_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        remote_batch_dir = PurePosixPath("/work") / batch_run_id
        self._manifest_for_batch(
            run_id=batch_run_id,
            batch_dir=batch_dir,
            remote_batch_dir=remote_batch_dir,
            leases=leases,
        )

        executor = RemoteExecutor(instance, self.config.vast_ssh_key)

        required_voices = sorted({lease.job.voice for lease in leases if lease.job.voice})
        for slug in required_voices:
            assert slug is not None
            executor.rsync_to_dir(self.config.voices_dir / slug, f"/cache/voices/{slug}")

        t_sync_start = _now_epoch()
        executor.rsync_to_dir(batch_dir, str(remote_batch_dir))
        sync_sec = _now_epoch() - t_sync_start
        self._metric_timing("sync_duration_sec", sync_sec)

        remote_manifest = remote_batch_dir / "manifest.json"
        exec_cmd = (
            "cd /app && "
            "python apps/job-runner/job-runner.py "
            f"execute-manifest {shlex.quote(str(remote_manifest))} --fail-fast --json"
        )
        exec_result = executor.run(exec_cmd, check=False)

        t_pull_start = _now_epoch()
        executor.rsync_from_dir(str(remote_batch_dir), batch_dir)
        pull_sec = _now_epoch() - t_pull_start
        self._metric_timing("pull_duration_sec", pull_sec)

        execution_map = self._load_execution_map(batch_dir)

        for lease in leases:
            req_id = lease.job.request_id
            exec_entry = execution_map.get(req_id)

            if exec_entry is None:
                err = (
                    "missing execution entry"
                    f" (remote exit={exec_result.returncode}, stderr={_short_error(exec_result.stderr or '')})"
                )
                self._handle_execution_failure(lease, error=err, run_id=run_id)
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
                    f"remote execution failed (request_id={req_id}, return_code={exec_entry.get('return_code')}, "
                    f"stderr={stderr_text or _short_error(exec_result.stderr or '')})"
                )
                self._handle_execution_failure(lease, error=err, run_id=run_id)
                continue

            out_dir = batch_dir / lease.job.output_name
            try:
                takes, best, meta = _required_output_files(lease.job, out_dir)
                t_upload_start = _now_epoch()
                self._upload_output_dir(out_dir, lease.job.output_name)
                upload_sec = _now_epoch() - t_upload_start
                self._metric_timing("upload_duration_sec", upload_sec)
                self._write_done_and_ack(
                    lease=lease,
                    run_id=run_id,
                    takes=takes,
                    best=best,
                    meta=meta,
                )
            except Exception as exc:
                self._metric_incr("upload_failures")
                self._handle_execution_failure(lease, error=str(exc), run_id=run_id)

    def _run_locked_session(self, *, run_id: str, reclaimed: Sequence[StreamLease]) -> None:
        start_ts = _now_epoch()
        self._set_heartbeat(run_id)
        _log("info", "session_start", run_id=run_id, reclaimed=len(reclaimed))

        pending: list[StreamLease] = list(reclaimed)
        instance: VastInstance | None = None
        vast = VastController(self.config)

        idle_start: float | None = None
        batch_index = 0

        try:
            while not self.stop_event.is_set() and not self.lock_lost_event.is_set():
                if (_now_epoch() - start_ts) >= float(self.config.max_runtime):
                    _log("warning", "session_max_runtime", run_id=run_id)
                    break

                self._set_heartbeat(run_id)
                self._move_due_retries(limit=max(10, self.config.batch_max_jobs))
                pending.extend(self._claim_stale(max_count=self.config.batch_max_jobs))

                if len(pending) < self.config.batch_max_jobs:
                    needed = max(1, self.config.batch_max_jobs - len(pending))
                    leased = self.queue.lease_jobs(
                        consumer_name=self.consumer_name,
                        count=needed,
                        block_ms=self.config.poll_interval * 1000,
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

                    if instance is None:
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

                if instance is None:
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

        finally:
            if instance is not None:
                try:
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

    def _acquire_and_run(self) -> bool:
        reclaimed = self._claim_stale(max_count=self.config.batch_max_jobs)

        outstanding = _estimate_outstanding(self.redis, self.queue)
        oldest_age = _first_queued_age_sec(self.redis, self.queue)
        trigger = should_start_run(
            outstanding=outstanding,
            oldest_age_sec=oldest_age,
            batch_min=self.config.batch_min,
            batch_max_wait=self.config.batch_max_wait,
        ) or bool(reclaimed)

        if not trigger:
            return False

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
            reclaimed=len(reclaimed),
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
            self._run_locked_session(run_id=run_id, reclaimed=reclaimed)
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
                self._move_due_retries(limit=max(10, self.config.batch_max_jobs))
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
