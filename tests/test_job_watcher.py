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


def _make_watcher_config(tmp_path: Path, **overrides) -> object:
    """Build a minimal WatcherConfig-like object for testing sink/package logic."""
    from dataclasses import dataclass
    from pathlib import Path as _Path

    @dataclass
    class MinimalConfig:
        sink_mode: str = "local"
        executor_mode: str = "local"
        output_root: _Path = None  # type: ignore[assignment]
        s3_bucket: str = ""
        rclone_remote: str = ""
        s3_prefix: str = "voice-results"
        stream_retention_days: int = 7

    cfg = MinimalConfig(output_root=tmp_path / "jobs_out")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_local_sink_copies_files_and_returns_relative_paths(tmp_path: Path) -> None:
    """_copy_to_local_package should copy files and return relative paths."""
    # Simulate batch output directory for a beat
    out_dir = tmp_path / "batch" / "my-topic" / "beat-001"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "take_01.wav").write_bytes(b"RIFF")
    (out_dir / "takes.meta.json").write_text("{}", encoding="utf-8")
    (out_dir / "result.json").write_text("{}", encoding="utf-8")

    cfg = _make_watcher_config(tmp_path)

    # Patch config into a minimal JobWatcher-like object
    class FakeWatcher:
        config = cfg

    fw = FakeWatcher()
    fw._copy_to_local_package = job_watcher.JobWatcher._copy_to_local_package.__get__(fw, type(fw))

    artifacts = fw._copy_to_local_package(out_dir, "my-topic/beat-001")

    pkg_beat_dir = tmp_path / "jobs_out" / "my-topic" / "beats" / "beat-001"
    assert (pkg_beat_dir / "take_01.wav").exists()
    assert (pkg_beat_dir / "takes.meta.json").exists()

    assert artifacts["sink"] == "local"
    assert "beats/beat-001/take_01.wav" in artifacts["takes"]
    assert artifacts["meta"] == "beats/beat-001/takes.meta.json"


def test_assemble_package_creates_job_json(tmp_path: Path) -> None:
    """_assemble_package should upsert job.json with correct schema."""
    import json

    cfg = _make_watcher_config(tmp_path)

    class FakeQueue:
        keys = type("k", (), {"stream": "s", "result": "r", "failed": "f"})()
        consumer_group = "g"

    class FakeWatcher:
        config = cfg
        queue = FakeQueue()
        redis = None

    fw = FakeWatcher()
    fw._assemble_package = job_watcher.JobWatcher._assemble_package.__get__(fw, type(fw))

    spec = _job_spec(
        output_name="my-topic/beat-001",
        callback_data={
            "topic_id": "my-topic",
            "beat_id": 1,
            "total_beats": 3,
            "topic_title": "My Topic",
        },
    )

    class FakeLease:
        job = spec
        stream_id = "1-0"

    artifacts = {
        "sink": "local",
        "takes": ["beats/beat-001/take_01.wav"],
        "best": None,
        "meta": "beats/beat-001/takes.meta.json",
        "result": "beats/beat-001/result.json",
    }

    pkg_dir = tmp_path / "jobs_out" / "my-topic"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    fw._assemble_package(lease=FakeLease(), artifacts=artifacts, run_id="run-test")

    job_json_path = pkg_dir / "job.json"
    assert job_json_path.exists()

    doc = json.loads(job_json_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["topic_id"] == "my-topic"
    assert doc["topic_title"] == "My Topic"
    assert doc["total_beats"] == 3
    assert doc["completed_beats"] == 1
    assert doc["status"] == "partial"
    assert len(doc["beats"]) == 1
    assert doc["beats"][0]["beat_id"] == 1
    assert doc["beats"][0]["status"] == "done"


def test_assemble_package_marks_complete_when_all_done(tmp_path: Path) -> None:
    """Package status should become 'complete' when completed_beats == total_beats."""
    import json

    cfg = _make_watcher_config(tmp_path)

    class FakeWatcher:
        config = cfg

    fw = FakeWatcher()
    fw._assemble_package = job_watcher.JobWatcher._assemble_package.__get__(fw, type(fw))

    pkg_dir = tmp_path / "jobs_out" / "one-beat-topic"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    spec = _job_spec(
        output_name="one-beat-topic/beat-001",
        callback_data={"topic_id": "one-beat-topic", "beat_id": 1, "total_beats": 1},
    )

    class FakeLease:
        job = spec
        stream_id = "1-0"

    artifacts = {"sink": "local", "takes": ["beats/beat-001/take_01.wav"], "best": None, "meta": "beats/beat-001/meta.json", "result": None}
    fw._assemble_package(lease=FakeLease(), artifacts=artifacts, run_id="run-complete")

    doc = json.loads((pkg_dir / "job.json").read_text())
    assert doc["status"] == "complete"
    assert doc["completed_beats"] == 1


# ── _record_package_failure ───────────────────────────────────────────────────


def _make_fake_watcher(tmp_path: Path, sink_mode: str = "local"):
    """Construct a minimal fake JobWatcher with _record_package_failure unbound."""
    cfg = _make_watcher_config(tmp_path, sink_mode=sink_mode)

    class FakeWatcher:
        config = cfg

    fw = FakeWatcher()
    fw._record_package_failure = job_watcher.JobWatcher._record_package_failure.__get__(fw, type(fw))
    return fw


def test_record_package_failure_creates_job_json(tmp_path: Path) -> None:
    """_record_package_failure should create job.json with a failures entry."""
    import json

    fw = _make_fake_watcher(tmp_path)
    job_dict = {
        "output_name": "my-topic/beat-002",
        "callback_data": {"topic_id": "my-topic", "beat_id": 2, "total_beats": 5},
    }
    fw._record_package_failure(
        request_id="my-topic:beat-002",
        job=job_dict,
        error_type="permanent",
        error="invalid voice: 'ghost' not found",
        run_id="run-xyz",
        stage="validate",
    )

    job_json = tmp_path / "jobs_out" / "my-topic" / "job.json"
    assert job_json.exists()
    doc = json.loads(job_json.read_text())

    assert doc["topic_id"] == "my-topic"
    assert doc["status"] == "partial"
    assert doc["total_beats"] == 5
    assert doc["completed_beats"] == 0
    assert len(doc["failures"]) == 1

    f = doc["failures"][0]
    assert f["request_id"] == "my-topic:beat-002"
    assert f["beat_id"] == 2
    assert f["status"] == "failed"
    assert f["error_type"] == "permanent"
    assert "invalid voice" in f["error"]
    assert f["stage"] == "validate"
    assert f["run_id"] == "run-xyz"


def test_record_package_failure_no_stage_omits_field(tmp_path: Path) -> None:
    """When stage is None the failures entry should not include a 'stage' key."""
    import json

    fw = _make_fake_watcher(tmp_path)
    fw._record_package_failure(
        request_id="my-topic:beat-001",
        job={"output_name": "my-topic/beat-001", "callback_data": {"topic_id": "my-topic", "beat_id": 1, "total_beats": 3}},
        error_type="transient_exhausted",
        error="ssh timeout",
        run_id=None,
        stage=None,
    )

    doc = json.loads((tmp_path / "jobs_out" / "my-topic" / "job.json").read_text())
    assert "stage" not in doc["failures"][0]


def test_record_package_failure_skips_rclone_sink(tmp_path: Path) -> None:
    """When sink_mode=rclone the method is a no-op (no local file created)."""
    fw = _make_fake_watcher(tmp_path, sink_mode="rclone")
    fw._record_package_failure(
        request_id="topic:beat-001",
        job={"output_name": "topic/beat-001", "callback_data": {"topic_id": "topic", "beat_id": 1, "total_beats": 2}},
        error_type="permanent",
        error="bad voice",
        run_id=None,
        stage=None,
    )
    package_dir = tmp_path / "jobs_out" / "topic"
    assert not package_dir.exists()


def test_record_package_failure_skips_when_no_topic_id(tmp_path: Path) -> None:
    """When callback_data has no topic_id the method is a no-op."""
    fw = _make_fake_watcher(tmp_path)
    fw._record_package_failure(
        request_id="orphan:beat-001",
        job={"output_name": "orphan/beat-001", "callback_data": {}},
        error_type="permanent",
        error="bad voice",
        run_id=None,
        stage=None,
    )
    assert not (tmp_path / "jobs_out").exists()


def test_record_package_failure_preserves_existing_done_beats(tmp_path: Path) -> None:
    """A failure entry must not overwrite previously assembled successful beats."""
    import json

    fw_done = _make_fake_watcher(tmp_path)
    fw_done._assemble_package = job_watcher.JobWatcher._assemble_package.__get__(fw_done, type(fw_done))

    pkg_dir = tmp_path / "jobs_out" / "mixed-topic"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    done_spec = _job_spec(
        output_name="mixed-topic/beat-001",
        callback_data={"topic_id": "mixed-topic", "beat_id": 1, "total_beats": 2},
    )

    class FakeLease:
        job = done_spec
        stream_id = "1-0"

    fw_done._assemble_package(
        lease=FakeLease(),
        artifacts={"sink": "local", "takes": ["beats/beat-001/take_01.wav"], "best": None, "meta": None, "result": None},
        run_id="run-1",
    )

    # Now record a failure for beat-002
    fw_done._record_package_failure(
        request_id="mixed-topic:beat-002",
        job={"output_name": "mixed-topic/beat-002", "callback_data": {"topic_id": "mixed-topic", "beat_id": 2, "total_beats": 2}},
        error_type="permanent",
        error="voice gone",
        run_id="run-1",
        stage="validate",
    )

    doc = json.loads((pkg_dir / "job.json").read_text())
    assert doc["completed_beats"] == 1
    assert len(doc["beats"]) == 1
    assert doc["beats"][0]["beat_id"] == 1
    assert len(doc["failures"]) == 1
    assert doc["failures"][0]["beat_id"] == 2
    # One done + one failed out of 2 total — still partial (failure != completion)
    assert doc["status"] == "partial"


def test_record_package_failure_replaces_previous_failure_for_same_request(tmp_path: Path) -> None:
    """Re-recording a failure for the same request_id replaces the old entry."""
    import json

    fw = _make_fake_watcher(tmp_path)
    job_dict = {"output_name": "t/beat-001", "callback_data": {"topic_id": "t", "beat_id": 1, "total_beats": 1}}

    fw._record_package_failure(
        request_id="t:beat-001",
        job=job_dict,
        error_type="transient_exhausted",
        error="first error",
        run_id="run-a",
        stage="execute",
    )
    fw._record_package_failure(
        request_id="t:beat-001",
        job=job_dict,
        error_type="permanent",
        error="final error",
        run_id="run-b",
        stage="upload",
    )

    doc = json.loads((tmp_path / "jobs_out" / "t" / "job.json").read_text())
    assert len(doc["failures"]) == 1
    assert doc["failures"][0]["error_type"] == "permanent"
    assert doc["failures"][0]["stage"] == "upload"
