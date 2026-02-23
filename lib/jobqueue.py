"""
Redis stream-backed job queue helpers for watcher/runner automation.

Phase 1 scope:
  - JobSpec schema + validation/defaulting
  - deterministic content hashing for audio-defining fields
  - idempotent enqueue to Redis stream
  - distributed watcher lock with compare-and-renew/release Lua scripts
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping


# Redis keys from docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md (section 3.4).
REDIS_STREAM_KEY = "pab:jobs:stream"
REDIS_GROUP_KEY = "pab:jobs:group"
REDIS_RESULT_KEY = "pab:jobs:result"
REDIS_FAILED_KEY = "pab:jobs:failed"
REDIS_REQUEST_INDEX_KEY = "pab:jobs:req:index"
REDIS_FIRST_QUEUED_KEY = "pab:jobs:first_queued"
REDIS_RETRY_ZSET_KEY = "pab:jobs:retry:zset"
REDIS_WATCHER_LOCK_KEY = "pab:watcher:lock"
REDIS_WATCHER_HEARTBEAT_KEY = "pab:watcher:heartbeat"
REDIS_WATCHER_INSTANCE_KEY = "pab:watcher:instance"
REDIS_METRICS_PREFIX = "pab:metrics:"

WATCHER_CONSUMER_GROUP = "watchers"
STREAM_JOB_FIELD = "job"

# Hash only synthesis-defining fields (section 3.3).
CONTENT_HASH_FIELDS = (
    "voice",
    "speaker",
    "text",
    "language",
    "tone",
    "instruct",
    "instruct_style",
    "profile",
    "variants",
    "select_best",
    "chunk",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
)

# Lua scripts for lock safety (section 7.3).
LOCK_RENEW_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  redis.call("PEXPIRE", KEYS[1], ARGV[2])
  return 1
end
return 0
""".strip()

LOCK_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
""".strip()

# Atomic enqueue:
# - reject duplicates by request_id index
# - optionally surface already_done if result already exists
# - append to stream
# - set first_queued hint once
ENQUEUE_JOB_LUA = """
if redis.call("HEXISTS", KEYS[2], ARGV[1]) == 1 then
  return {0, "already_done"}
end
if redis.call("HEXISTS", KEYS[1], ARGV[1]) == 1 then
  return {0, "duplicate_request"}
end
local stream_id = redis.call("XADD", KEYS[3], "*", ARGV[4], ARGV[2])
redis.call("HSET", KEYS[1], ARGV[1], "queued")
redis.call("SETNX", KEYS[4], ARGV[3])
return {1, stream_id}
""".strip()


EnqueueStatus = Literal["queued", "duplicate_request", "already_done"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_text(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = str(value).strip()
    return text or None


def _required_text(field_name: str, value: Any) -> str:
    text = _optional_text(field_name, value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _required_nonblank_string(field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} is required")
    return value


def _normalize_queued_at(value: Any) -> str:
    text = _required_text("queued_at", value)
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("queued_at must be ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("queued_at must include timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_bool(field_name: str, value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_int(field_name: str, value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_optional_int(field_name: str, value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_optional_float(field_name: str, value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def make_content_hash(job: JobSpec | Mapping[str, Any]) -> str:
    """
    Return a deterministic sha256 hash of synthesis-defining fields only.
    """
    if isinstance(job, JobSpec):
        payload = job.to_dict()
    else:
        payload = dict(job)
    canonical = {field: payload.get(field) for field in CONTENT_HASH_FIELDS}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(slots=True)
class JobSpec:
    request_id: str
    queued_at: str

    # synthesis source (exactly one required)
    voice: str | None
    speaker: str | None

    text: str
    language: str | None
    tone: str | None
    instruct: str | None
    instruct_style: str | None
    profile: str | None
    variants: int
    select_best: bool
    chunk: bool

    temperature: float | None
    top_p: float | None
    repetition_penalty: float | None
    max_new_tokens: int | None

    output_name: str
    callback_data: dict[str, Any] = field(default_factory=dict)

    # system
    content_hash: str = ""
    attempt: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_job_spec(job: JobSpec | Mapping[str, Any]) -> JobSpec:
    """
    Validate and normalize a job payload according to the JobSpec contract.
    """
    data = job.to_dict() if isinstance(job, JobSpec) else dict(job)

    request_id = _required_text("request_id", data.get("request_id"))
    queued_at = _normalize_queued_at(data.get("queued_at") or _utc_now_iso())

    voice = _optional_text("voice", data.get("voice"))
    speaker = _optional_text("speaker", data.get("speaker"))
    if bool(voice) == bool(speaker):
        raise ValueError("Exactly one of voice or speaker must be provided")

    text = _required_nonblank_string("text", data.get("text"))

    output_name = _required_text("output_name", data.get("output_name"))
    output_path = PurePosixPath(output_name)
    if output_path.is_absolute():
        raise ValueError("output_name must be a relative path")
    if ".." in output_path.parts:
        raise ValueError("output_name must not contain '..'")
    if output_name.strip().endswith("/"):
        raise ValueError("output_name must not end with '/'")

    variants = _coerce_int("variants", data.get("variants"), default=1)
    if variants < 1:
        raise ValueError("variants must be >= 1")

    attempt = _coerce_int("attempt", data.get("attempt"), default=0)
    if attempt < 0:
        raise ValueError("attempt must be >= 0")

    max_new_tokens = _coerce_optional_int("max_new_tokens", data.get("max_new_tokens"))
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    temperature = _coerce_optional_float("temperature", data.get("temperature"))
    if temperature is not None and temperature < 0:
        raise ValueError("temperature must be >= 0")

    top_p = _coerce_optional_float("top_p", data.get("top_p"))
    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")

    repetition_penalty = _coerce_optional_float("repetition_penalty", data.get("repetition_penalty"))
    if repetition_penalty is not None and repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")

    callback_data = data.get("callback_data")
    if callback_data is None:
        callback_data = {}
    if not isinstance(callback_data, dict):
        raise ValueError("callback_data must be an object")

    spec = JobSpec(
        request_id=request_id,
        queued_at=queued_at,
        voice=voice,
        speaker=speaker,
        text=text,
        language=_optional_text("language", data.get("language")),
        tone=_optional_text("tone", data.get("tone")),
        instruct=_optional_text("instruct", data.get("instruct")),
        instruct_style=_optional_text("instruct_style", data.get("instruct_style")),
        profile=_optional_text("profile", data.get("profile")),
        variants=variants,
        select_best=_coerce_bool("select_best", data.get("select_best"), default=False),
        chunk=_coerce_bool("chunk", data.get("chunk"), default=False),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        output_name=output_name,
        callback_data=dict(callback_data),
        content_hash="",
        attempt=attempt,
    )
    spec.content_hash = make_content_hash(spec)
    return spec


@dataclass(slots=True, frozen=True)
class QueueKeys:
    stream: str = REDIS_STREAM_KEY
    group: str = REDIS_GROUP_KEY
    result: str = REDIS_RESULT_KEY
    failed: str = REDIS_FAILED_KEY
    request_index: str = REDIS_REQUEST_INDEX_KEY
    first_queued: str = REDIS_FIRST_QUEUED_KEY
    retry_zset: str = REDIS_RETRY_ZSET_KEY
    watcher_lock: str = REDIS_WATCHER_LOCK_KEY
    watcher_heartbeat: str = REDIS_WATCHER_HEARTBEAT_KEY
    watcher_instance: str = REDIS_WATCHER_INSTANCE_KEY
    metrics_prefix: str = REDIS_METRICS_PREFIX


@dataclass(slots=True, frozen=True)
class EnqueueResult:
    status: EnqueueStatus
    request_id: str
    stream_id: str | None
    job: JobSpec


@dataclass(slots=True, frozen=True)
class StreamLease:
    stream_id: str
    job: JobSpec


class JobQueue:
    """
    Thin Redis stream client with idempotent enqueue and watcher lock helpers.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        keys: QueueKeys | None = None,
        consumer_group: str = WATCHER_CONSUMER_GROUP,
        stream_job_field: str = STREAM_JOB_FIELD,
    ) -> None:
        self.redis = redis_client
        self.keys = keys or QueueKeys()
        self.consumer_group = consumer_group
        self.stream_job_field = stream_job_field

    def ensure_group(self, *, start_id: str = "0") -> None:
        try:
            self.redis.xgroup_create(
                name=self.keys.stream,
                groupname=self.consumer_group,
                id=start_id,
                mkstream=True,
            )
        except Exception as exc:  # pragma: no cover - exercised with real Redis integration.
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, job: JobSpec | Mapping[str, Any]) -> EnqueueResult:
        spec = normalize_job_spec(job)
        response = self.redis.eval(
            ENQUEUE_JOB_LUA,
            4,
            self.keys.request_index,
            self.keys.result,
            self.keys.stream,
            self.keys.first_queued,
            spec.request_id,
            spec.to_json(),
            spec.queued_at,
            self.stream_job_field,
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError(f"Unexpected enqueue script response: {response!r}")

        code = int(response[0])
        detail = _to_text(response[1])
        if code == 1:
            return EnqueueResult("queued", spec.request_id, detail, spec)
        if detail == "already_done":
            return EnqueueResult("already_done", spec.request_id, None, spec)
        if detail == "duplicate_request":
            return EnqueueResult("duplicate_request", spec.request_id, None, spec)
        raise RuntimeError(f"Unexpected enqueue script detail: {detail!r}")

    def lease_jobs(
        self,
        *,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 0,
    ) -> list[StreamLease]:
        records = self.redis.xreadgroup(
            groupname=self.consumer_group,
            consumername=consumer_name,
            streams={self.keys.stream: ">"},
            count=count,
            block=block_ms if block_ms > 0 else None,
        )
        return self._parse_stream_records(records)

    def claim_stale_jobs(
        self,
        *,
        consumer_name: str,
        min_idle_ms: int,
        start_id: str = "0-0",
        count: int = 100,
    ) -> tuple[str, list[StreamLease]]:
        result = self.redis.xautoclaim(
            name=self.keys.stream,
            groupname=self.consumer_group,
            consumername=consumer_name,
            min_idle_time=min_idle_ms,
            start_id=start_id,
            count=count,
        )

        next_start: str
        messages: Any
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            next_start = _to_text(result[0])
            messages = result[1]
        else:
            raise RuntimeError(f"Unexpected XAUTOCLAIM response: {result!r}")
        return next_start, self._parse_claim_messages(messages)

    def ack(self, *stream_ids: str) -> int:
        if not stream_ids:
            return 0
        return int(self.redis.xack(self.keys.stream, self.consumer_group, *stream_ids))

    def stream_length(self) -> int:
        return int(self.redis.xlen(self.keys.stream))

    def pending_count(self) -> int:
        info = self.redis.xpending(self.keys.stream, self.consumer_group)
        if isinstance(info, dict):
            return int(info.get("pending", 0))
        if isinstance(info, (list, tuple)) and info:
            return int(info[0])
        return 0

    def acquire_lock(self, run_id: str, *, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        ok = self.redis.set(self.keys.watcher_lock, run_id, nx=True, ex=int(ttl_seconds))
        return bool(ok)

    def renew_lock(self, run_id: str, *, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        ttl_ms = int(ttl_seconds * 1000)
        refreshed = self.redis.eval(LOCK_RENEW_LUA, 1, self.keys.watcher_lock, run_id, ttl_ms)
        return int(refreshed) == 1

    def release_lock(self, run_id: str) -> bool:
        released = self.redis.eval(LOCK_RELEASE_LUA, 1, self.keys.watcher_lock, run_id)
        return int(released) == 1

    def lock_owner(self) -> str | None:
        raw = self.redis.get(self.keys.watcher_lock)
        if raw is None:
            return None
        return _to_text(raw)

    def result_exists(self, request_id: str) -> bool:
        return bool(self.redis.hexists(self.keys.result, request_id))

    def _parse_stream_records(self, records: Any) -> list[StreamLease]:
        leases: list[StreamLease] = []
        if not records:
            return leases
        for stream_name, entries in records:
            _ = stream_name  # only one stream key in this client
            for stream_id, raw_fields in entries:
                leases.append(StreamLease(stream_id=_to_text(stream_id), job=self._parse_job(raw_fields)))
        return leases

    def _parse_claim_messages(self, messages: Any) -> list[StreamLease]:
        leases: list[StreamLease] = []
        if not messages:
            return leases
        for stream_id, raw_fields in messages:
            leases.append(StreamLease(stream_id=_to_text(stream_id), job=self._parse_job(raw_fields)))
        return leases

    def _parse_job(self, raw_fields: Mapping[Any, Any]) -> JobSpec:
        decoded = {_to_text(key): _to_text(value) for key, value in raw_fields.items()}
        payload = decoded.get(self.stream_job_field)
        if payload is None:
            raise ValueError(f"Stream entry missing field '{self.stream_job_field}'")
        data = json.loads(payload)
        return normalize_job_spec(data)


__all__ = [
    "CONTENT_HASH_FIELDS",
    "ENQUEUE_JOB_LUA",
    "EnqueueResult",
    "JobQueue",
    "JobSpec",
    "LOCK_RELEASE_LUA",
    "LOCK_RENEW_LUA",
    "QueueKeys",
    "REDIS_FAILED_KEY",
    "REDIS_FIRST_QUEUED_KEY",
    "REDIS_GROUP_KEY",
    "REDIS_METRICS_PREFIX",
    "REDIS_REQUEST_INDEX_KEY",
    "REDIS_RESULT_KEY",
    "REDIS_RETRY_ZSET_KEY",
    "REDIS_STREAM_KEY",
    "REDIS_WATCHER_HEARTBEAT_KEY",
    "REDIS_WATCHER_INSTANCE_KEY",
    "REDIS_WATCHER_LOCK_KEY",
    "STREAM_JOB_FIELD",
    "WATCHER_CONSUMER_GROUP",
    "make_content_hash",
    "normalize_job_spec",
]
