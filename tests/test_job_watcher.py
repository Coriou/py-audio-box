from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WATCHER_PATH = ROOT / "apps" / "job-watcher" / "job-watcher.py"

spec = importlib.util.spec_from_file_location("job_watcher_module", WATCHER_PATH)
assert spec is not None and spec.loader is not None
job_watcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = job_watcher
spec.loader.exec_module(job_watcher)


def _job_spec(**overrides):
    payload = {
        "request_id": "topic:beat-001",
        "queued_at": "2026-02-22T20:10:00Z",
        "voice": "newsroom",
        "speaker": None,
        "text": "Hello",
        "language": "English",
        "tone": None,
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
        "output_name": "topic/beat-001",
        "callback_data": {},
        "attempt": 0,
    }
    payload.update(overrides)
    return job_watcher.normalize_job_spec(payload)


def test_retry_delay_seconds_schedule() -> None:
    assert job_watcher.retry_delay_seconds(1) == 30
    assert job_watcher.retry_delay_seconds(2) == 120
    assert job_watcher.retry_delay_seconds(3) == 600
    assert job_watcher.retry_delay_seconds(9) == 600


def test_classify_failure_permanent_vs_transient() -> None:
    assert job_watcher.classify_failure("invalid voice: not found") == "permanent"
    assert job_watcher.classify_failure("ssh timeout while executing") == "transient"
    assert job_watcher.classify_failure("unknown failure") == "transient"


def test_should_start_run_thresholds() -> None:
    assert job_watcher.should_start_run(
        outstanding=1,
        oldest_age_sec=10,
        batch_min=1,
        batch_max_wait=300,
    )
    assert job_watcher.should_start_run(
        outstanding=1,
        oldest_age_sec=305,
        batch_min=5,
        batch_max_wait=300,
    )
    assert not job_watcher.should_start_run(
        outstanding=0,
        oldest_age_sec=999,
        batch_min=1,
        batch_max_wait=300,
    )
    assert not job_watcher.should_start_run(
        outstanding=1,
        oldest_age_sec=20,
        batch_min=5,
        batch_max_wait=300,
    )


def test_build_speak_argv_includes_out_exact_and_json_result() -> None:
    spec = _job_spec()
    argv = job_watcher.build_speak_argv(
        spec,
        text_file="/work/topic/beat-001/text.txt",
        out_exact="/work/topic/beat-001",
        json_result="/work/topic/beat-001/result.json",
    )

    assert argv[0:2] == ["voice-synth", "speak"]
    assert "--out-exact" in argv
    assert argv[argv.index("--out-exact") + 1] == "/work/topic/beat-001"
    assert "--json-result" in argv
    assert argv[argv.index("--json-result") + 1] == "/work/topic/beat-001/result.json"


def test_required_output_files_happy_path(tmp_path: Path) -> None:
    spec = _job_spec(select_best=True)
    out_dir = tmp_path / "topic" / "beat-001"
    out_dir.mkdir(parents=True, exist_ok=True)

    take1 = out_dir / "take_01.wav"
    take2 = out_dir / "take_02.wav"
    best = out_dir / "best.wav"
    meta = out_dir / "takes.meta.json"

    take1.write_bytes(b"RIFFdata")
    take2.write_bytes(b"RIFFmore")
    best.write_bytes(b"RIFFbest")
    meta.write_text("{}", encoding="utf-8")

    takes, best_path, meta_path = job_watcher._required_output_files(spec, out_dir)

    assert [p.name for p in takes] == ["take_01.wav", "take_02.wav"]
    assert best_path == best
    assert meta_path == meta


def test_required_output_files_requires_best_when_select_best(tmp_path: Path) -> None:
    spec = _job_spec(select_best=True)
    out_dir = tmp_path / "topic" / "beat-001"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "take_01.wav").write_bytes(b"RIFFdata")
    (out_dir / "takes.meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="best.wav"):
        job_watcher._required_output_files(spec, out_dir)


def test_s3_uri_builder() -> None:
    uri = job_watcher._s3_uri("bucket", "voice-results", "topic/beat-001", "take_01.wav")
    assert uri == "s3://bucket/voice-results/topic/beat-001/take_01.wav"

    uri2 = job_watcher._s3_uri("bucket", "", "topic/beat-001", "meta.json")
    assert uri2 == "s3://bucket/topic/beat-001/meta.json"
