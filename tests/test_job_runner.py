from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
JOB_RUNNER_PATH = ROOT / "apps" / "job-runner" / "job-runner.py"

spec = importlib.util.spec_from_file_location("job_runner_module", JOB_RUNNER_PATH)
assert spec is not None and spec.loader is not None
job_runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = job_runner
spec.loader.exec_module(job_runner)


def _create_named_voice(cache_dir: Path, slug: str) -> None:
    voice_dir = cache_dir / "voices" / slug
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "voice.json").write_text("{}", encoding="utf-8")


def test_render_jobs_derives_request_id_from_output_name() -> None:
    rows = [
        {
            "voice": "newsroom",
            "speaker": None,
            "text": "Hello world",
            "output_name": "topic/beat-001",
        }
    ]

    rendered = job_runner.render_jobs(rows)

    assert len(rendered) == 1
    assert rendered[0].spec.request_id == "topic:beat-001"
    assert rendered[0].spec.output_name == "topic/beat-001"
    assert rendered[0].source_key == "voice:newsroom"


def test_render_jobs_merges_defaults_and_callback_data() -> None:
    defaults = {
        "voice": "newsroom",
        "speaker": None,
        "language": "English",
        "variants": 2,
        "callback_data": {"topic_id": "topic"},
    }
    rows = [
        {
            "text": "Beat text",
            "output_name": "topic/beat-005",
            "callback_data": {"beat_id": 5},
        }
    ]

    rendered = job_runner.render_jobs(rows, defaults=defaults)

    spec = rendered[0].spec
    assert spec.voice == "newsroom"
    assert spec.language == "English"
    assert spec.variants == 2
    assert spec.callback_data == {"topic_id": "topic", "beat_id": 5}


def test_build_manifest_writes_text_files_and_out_exact_contract(tmp_path: Path) -> None:
    jobs = job_runner.render_jobs(
        [
            {
                "voice": "newsroom",
                "speaker": None,
                "text": "A-2",
                "output_name": "topic/beat-002",
            },
            {
                "speaker": "Ryan",
                "voice": None,
                "text": "S-1",
                "output_name": "topic/beat-001",
            },
            {
                "voice": "newsroom",
                "speaker": None,
                "text": "A-3",
                "output_name": "topic/beat-003",
            },
        ]
    )

    manifest = job_runner.build_manifest(jobs, out_dir=tmp_path, run_id="run-test")

    assert manifest["run_id"] == "run-test"
    assert len(manifest["jobs"]) == 3

    sources = [str(entry["source_key"]) for entry in manifest["jobs"]]
    assert sources == sorted(sources)

    for entry in manifest["jobs"]:
        out_exact = Path(str(entry["out_exact"]))
        text_file = Path(str(entry["text_file"]))
        json_result = Path(str(entry["json_result"]))
        argv = list(entry["argv"])

        assert out_exact.is_dir()
        assert text_file.exists()
        assert text_file.parent == out_exact

        assert "--out-exact" in argv
        assert argv[argv.index("--out-exact") + 1] == str(out_exact)

        assert "--json-result" in argv
        assert argv[argv.index("--json-result") + 1] == str(json_result)


def test_beatsheet_mapping_to_request_and_output_names() -> None:
    beatsheet: dict[str, Any] = {
        "topicId": "pineapple",
        "beats": [
            {"id": 1, "narration": "One"},
            {"id": 7, "narration": "Seven"},
        ],
    }

    jobs = job_runner._build_beatsheet_jobs(
        beatsheet,
        voice="newsroom",
        speaker=None,
        language="English",
        tone=None,
        instruct=None,
        instruct_style=None,
        profile="balanced",
        variants=1,
        select_best=False,
        chunk=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        max_new_tokens=300,
    )

    assert [j.spec.request_id for j in jobs] == ["pineapple:beat-001", "pineapple:beat-007"]
    assert [j.spec.output_name for j in jobs] == ["pineapple/beat-001", "pineapple/beat-007"]
    assert jobs[0].spec.callback_data == {"topic_id": "pineapple", "beat_id": 1}


def test_voice_presence_errors_detect_missing_voice(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _create_named_voice(cache_dir, "present-voice")

    jobs = job_runner.render_jobs(
        [
            {
                "voice": "present-voice",
                "speaker": None,
                "text": "ok",
                "output_name": "x/1",
            },
            {
                "voice": "missing-voice",
                "speaker": None,
                "text": "missing",
                "output_name": "x/2",
            },
        ]
    )

    errors = job_runner._voice_presence_errors(jobs, cache_dir=cache_dir)

    assert len(errors) == 1
    assert "missing-voice" in errors[0]


def test_execute_manifest_data_dry_run_marks_success(tmp_path: Path) -> None:
    manifest = {
        "run_id": "run-001",
        "created_at": "2026-02-22T21:00:00Z",
        "jobs": [
            {
                "request_id": "topic:beat-001",
                "output_name": "topic/beat-001",
                "argv": ["voice-synth", "speak", "--voice", "newsroom", "--text", "Hello"],
            }
        ],
    }

    execution = job_runner.execute_manifest_data(
        manifest,
        manifest_dir=tmp_path,
        fail_fast=True,
        dry_run=True,
    )

    assert execution["success"] is True
    assert execution["failed_count"] == 0
    assert execution["executed_count"] == 1
    assert execution["jobs"][0]["status"] == "ok"

    stdout_log = Path(execution["jobs"][0]["stdout_log"])
    stderr_log = Path(execution["jobs"][0]["stderr_log"])
    assert stdout_log.exists()
    assert stderr_log.exists()


def test_render_jobs_rejects_missing_output_name_for_derived_request_id() -> None:
    rows = [
        {
            "voice": "newsroom",
            "speaker": None,
            "text": "missing output name",
        }
    ]

    with pytest.raises(ValueError, match="output_name"):
        job_runner.render_jobs(rows)
