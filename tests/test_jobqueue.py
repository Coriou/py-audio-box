from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from jobqueue import (  # noqa: E402
    ENQUEUE_JOB_LUA,
    LOCK_RELEASE_LUA,
    LOCK_RENEW_LUA,
    JobQueue,
    make_content_hash,
    normalize_job_spec,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.ttls_ms: dict[str, int] = {}

    def set(self, name: str, value: str, nx: bool = False, ex: int | None = None) -> bool | None:
        if nx and name in self.values:
            return None
        self.values[name] = value
        if ex is not None:
            self.ttls_ms[name] = int(ex) * 1000
        return True

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def hexists(self, key: str, field: str) -> int:
        return 1 if field in self.hashes.get(key, {}) else 0

    def hset(self, key: str, field: str, value: str) -> int:
        target = self.hashes.setdefault(key, {})
        is_new = 1 if field not in target else 0
        target[field] = value
        return is_new

    def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        if script == ENQUEUE_JOB_LUA:
            req_index_key, result_key, stream_key, first_queued_key = keys
            request_id, payload, queued_at, stream_field = argv

            if self.hexists(result_key, request_id):
                return [0, "already_done"]
            if self.hexists(req_index_key, request_id):
                return [0, "duplicate_request"]

            stream = self.streams.setdefault(stream_key, [])
            stream_id = f"{len(stream) + 1}-0"
            stream.append((stream_id, {str(stream_field): str(payload)}))

            self.hset(req_index_key, str(request_id), "queued")
            self.values.setdefault(str(first_queued_key), str(queued_at))
            return [1, stream_id]

        if script == LOCK_RENEW_LUA:
            lock_key = str(keys[0])
            token = str(argv[0])
            ttl_ms = int(argv[1])
            if self.values.get(lock_key) == token:
                self.ttls_ms[lock_key] = ttl_ms
                return 1
            return 0

        if script == LOCK_RELEASE_LUA:
            lock_key = str(keys[0])
            token = str(argv[0])
            if self.values.get(lock_key) == token:
                del self.values[lock_key]
                self.ttls_ms.pop(lock_key, None)
                return 1
            return 0

        raise AssertionError(f"Unexpected script: {script}")


class _StubRedisForGroup:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def xgroup_create(self, **_kwargs: Any) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


class _StubRedisForClaim:
    def xautoclaim(self, **_kwargs: Any) -> tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]], list]:
        payload = {
            b"job": (
                b'{"request_id":"topic:beat-500","queued_at":"2026-02-22T20:00:00Z",'
                b'"voice":"rascar-capac","speaker":null,"text":"hello","language":"English",'
                b'"tone":"neutral","instruct":null,"instruct_style":null,"profile":"balanced",'
                b'"variants":1,"select_best":false,"chunk":false,"temperature":0.7,'
                b'"top_p":0.9,"repetition_penalty":1.05,"max_new_tokens":120,'
                b'"output_name":"topic/beat-500","callback_data":{},"content_hash":"",'
                b'"attempt":0}'
            )
        }
        return b"1-0", [(b"1-0", payload)], []


def _base_job_payload() -> dict[str, Any]:
    return {
        "request_id": "topic:beat-001",
        "voice": "rascar-capac",
        "speaker": None,
        "text": "Pineapple can irritate your mouth because of bromelain enzymes.",
        "language": "English",
        "tone": "neutral",
        "instruct": None,
        "instruct_style": None,
        "profile": "balanced",
        "variants": 1,
        "select_best": True,
        "chunk": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
        "output_name": "topic/beat-001",
        "callback_data": {"topic_id": "topic", "beat_id": 1},
    }


def test_make_content_hash_ignores_request_and_output_identity_fields() -> None:
    payload_a = _base_job_payload()
    payload_a["request_id"] = "topic:beat-001"
    payload_a["queued_at"] = "2026-02-22T20:10:00Z"
    payload_a["output_name"] = "topic/beat-001"
    payload_a["callback_data"] = {"row": 1}
    payload_a["attempt"] = 0

    payload_b = _base_job_payload()
    payload_b["request_id"] = "topic:beat-999"
    payload_b["queued_at"] = "2026-02-22T20:30:00Z"
    payload_b["output_name"] = "topic/beat-999"
    payload_b["callback_data"] = {"row": 999}
    payload_b["attempt"] = 9

    assert make_content_hash(payload_a) == make_content_hash(payload_b)


def test_normalize_job_spec_requires_exactly_one_voice_source() -> None:
    payload = _base_job_payload()
    payload["speaker"] = "Ryan"
    with pytest.raises(ValueError, match="Exactly one of voice or speaker"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["voice"] = None
    payload["speaker"] = None
    with pytest.raises(ValueError, match="Exactly one of voice or speaker"):
        normalize_job_spec(payload)


def test_normalize_job_spec_rejects_non_string_text() -> None:
    payload = _base_job_payload()
    payload["text"] = None
    with pytest.raises(ValueError, match="text must be a string"):
        normalize_job_spec(payload)


def test_normalize_job_spec_rejects_invalid_generation_bounds() -> None:
    payload = _base_job_payload()
    payload["top_p"] = 1.4
    with pytest.raises(ValueError, match="top_p must be in"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["temperature"] = -0.1
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["repetition_penalty"] = 0
    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["max_new_tokens"] = 0
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        normalize_job_spec(payload)


def test_normalize_job_spec_rejects_unsafe_output_name_paths() -> None:
    payload = _base_job_payload()
    payload["output_name"] = "/abs/path"
    with pytest.raises(ValueError, match="output_name must be a relative path"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["output_name"] = "topic/../beat-001"
    with pytest.raises(ValueError, match="must not contain"):
        normalize_job_spec(payload)

    payload = _base_job_payload()
    payload["output_name"] = "topic/beat-001/"
    with pytest.raises(ValueError, match="must not end"):
        normalize_job_spec(payload)


def test_normalize_job_spec_defaults_and_hash() -> None:
    payload = _base_job_payload()
    payload.pop("variants")
    payload.pop("select_best")
    payload.pop("chunk")
    payload.pop("queued_at", None)
    payload.pop("attempt", None)

    spec = normalize_job_spec(payload)
    assert spec.variants == 1
    assert spec.select_best is False
    assert spec.chunk is False
    assert spec.attempt == 0
    assert spec.queued_at.endswith("Z")
    assert spec.content_hash.startswith("sha256:")


def test_normalize_job_spec_normalizes_queued_at_to_utc() -> None:
    payload = _base_job_payload()
    payload["queued_at"] = "2026-02-22T15:34:10-05:00"

    spec = normalize_job_spec(payload)
    assert spec.queued_at == "2026-02-22T20:34:10Z"


def test_enqueue_idempotency_and_first_queued_hint() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis)

    first = queue.enqueue(_base_job_payload())
    assert first.status == "queued"
    assert first.stream_id == "1-0"
    assert redis.values[queue.keys.first_queued] == first.job.queued_at

    duplicate = queue.enqueue(_base_job_payload())
    assert duplicate.status == "duplicate_request"
    assert duplicate.stream_id is None
    assert len(redis.streams[queue.keys.stream]) == 1


def test_enqueue_returns_already_done_when_result_exists() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis)
    redis.hset(queue.keys.result, "topic:beat-001", '{"status":"done"}')

    result = queue.enqueue(_base_job_payload())
    assert result.status == "already_done"
    assert result.stream_id is None
    assert queue.keys.stream not in redis.streams


def test_lock_compare_and_renew_release() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis)

    assert queue.acquire_lock("run-1", ttl_seconds=120) is True
    assert queue.acquire_lock("run-2", ttl_seconds=120) is False
    assert queue.lock_owner() == "run-1"

    assert queue.renew_lock("run-2", ttl_seconds=30) is False
    assert queue.renew_lock("run-1", ttl_seconds=30) is True
    assert redis.ttls_ms[queue.keys.watcher_lock] == 30_000

    assert queue.release_lock("run-2") is False
    assert queue.lock_owner() == "run-1"

    assert queue.release_lock("run-1") is True
    assert queue.lock_owner() is None


def test_ensure_group_ignores_busygroup_error() -> None:
    redis = _StubRedisForGroup(Exception("BUSYGROUP Consumer Group name already exists"))
    queue = JobQueue(redis)
    queue.ensure_group()
    assert redis.calls == 1


def test_ensure_group_raises_non_busygroup_error() -> None:
    redis = _StubRedisForGroup(RuntimeError("wrongtype"))
    queue = JobQueue(redis)
    with pytest.raises(RuntimeError, match="wrongtype"):
        queue.ensure_group()


def test_claim_stale_jobs_parses_three_part_response() -> None:
    queue = JobQueue(_StubRedisForClaim())
    next_start, reclaimed = queue.claim_stale_jobs(
        consumer_name="watcher-2",
        min_idle_ms=1,
        start_id="0-0",
        count=10,
    )
    assert next_start == "1-0"
    assert len(reclaimed) == 1
    assert reclaimed[0].stream_id == "1-0"
    assert reclaimed[0].job.request_id == "topic:beat-500"
