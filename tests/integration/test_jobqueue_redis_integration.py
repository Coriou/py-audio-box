import os
from pathlib import Path
import sys
from typing import Any

import pytest


if os.getenv("PAB_RUN_REDIS_INTEGRATION") != "1":
    pytest.skip(
        "Redis integration tests are opt-in. Set PAB_RUN_REDIS_INTEGRATION=1 to run.",
        allow_module_level=True,
    )

try:
    import redis
except Exception:  # pragma: no cover - environment/dependency guard
    pytest.fail(
        "redis package is not available in the toolbox image. "
        "Rebuild dependencies (make build) before running Redis integration tests.",
        pytrace=False,
    )

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from jobqueue import JobQueue  # noqa: E402


pytestmark = [pytest.mark.integration, pytest.mark.redis_integration]


def _job_payload(request_id: str) -> dict[str, Any]:
    beat = request_id.split("-")[-1]
    return {
        "request_id": request_id,
        "queued_at": "2026-02-22T20:10:00Z",
        "voice": "rascar-capac",
        "speaker": None,
        "text": f"Redis integration test for beat {beat}.",
        "language": "English",
        "tone": "neutral",
        "instruct": None,
        "instruct_style": None,
        "profile": "balanced",
        "variants": 1,
        "select_best": False,
        "chunk": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "output_name": f"topic/{beat}",
        "callback_data": {"topic_id": "topic", "beat_id": beat},
    }


@pytest.fixture()
def redis_client():
    redis_url = os.getenv("PAB_TEST_REDIS_URL", "redis://127.0.0.1:6380/15")
    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to connect to Redis integration test instance at {redis_url}: {exc}"
        ) from exc

    client.flushdb()
    yield client
    client.flushdb()


def test_enqueue_lease_ack_roundtrip(redis_client) -> None:
    queue = JobQueue(redis_client)
    queue.ensure_group()

    payload = _job_payload("topic:beat-001")
    queued = queue.enqueue(payload)
    assert queued.status == "queued"
    assert queued.stream_id is not None

    duplicate = queue.enqueue(payload)
    assert duplicate.status == "duplicate_request"

    leases = queue.lease_jobs(consumer_name="watcher-a", count=1, block_ms=100)
    assert len(leases) == 1
    assert leases[0].job.request_id == payload["request_id"]
    assert queue.pending_count() == 1
    assert queue.ack(leases[0].stream_id) == 1
    assert queue.pending_count() == 0


def test_enqueue_returns_already_done_when_result_exists(redis_client) -> None:
    queue = JobQueue(redis_client)
    queue.ensure_group()

    payload = _job_payload("topic:beat-002")
    redis_client.hset(queue.keys.result, payload["request_id"], '{"status":"done"}')
    result = queue.enqueue(payload)

    assert result.status == "already_done"
    assert result.stream_id is None
    assert queue.stream_length() == 0


def test_xautoclaim_reclaims_pending_entry(redis_client) -> None:
    queue = JobQueue(redis_client)
    queue.ensure_group()

    payload = _job_payload("topic:beat-003")
    queue.enqueue(payload)

    leased = queue.lease_jobs(consumer_name="watcher-a", count=1, block_ms=100)
    assert len(leased) == 1

    next_start, reclaimed = queue.claim_stale_jobs(
        consumer_name="watcher-b",
        min_idle_ms=0,
        start_id="0-0",
        count=10,
    )
    assert next_start
    assert len(reclaimed) == 1
    assert reclaimed[0].stream_id == leased[0].stream_id
    assert reclaimed[0].job.request_id == payload["request_id"]
    assert queue.ack(reclaimed[0].stream_id) == 1


def test_lock_acquire_renew_release_with_token_safety(redis_client) -> None:
    queue = JobQueue(redis_client)

    assert queue.acquire_lock("run-a", ttl_seconds=60) is True
    assert queue.acquire_lock("run-b", ttl_seconds=60) is False
    assert queue.lock_owner() == "run-a"

    assert queue.renew_lock("run-b", ttl_seconds=20) is False
    assert queue.renew_lock("run-a", ttl_seconds=20) is True

    ttl_ms = int(redis_client.pttl(queue.keys.watcher_lock))
    assert ttl_ms > 0
    assert ttl_ms <= 20_000

    assert queue.release_lock("run-b") is False
    assert queue.lock_owner() == "run-a"
    assert queue.release_lock("run-a") is True
    assert queue.lock_owner() is None
