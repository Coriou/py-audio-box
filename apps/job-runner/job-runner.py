#!/usr/bin/env python3
"""
job-runner.py — producer/validation tooling for Redis stream-backed TTS jobs.

Implements the Phase 2 runner contract from docs/JOB_RUNNER_IMPLEMENTATION_PLAN.md:
  - plan / enqueue / enqueue-beatsheet
  - compile / run / execute-manifest
  - status / result / report / retry / flush
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

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

from jobqueue import JobQueue, JobSpec, normalize_job_spec  # noqa: E402
from voices import VoiceRegistry  # noqa: E402


DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
MANIFEST_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1
VOICE_SYNTH_SCRIPT = Path(__file__).resolve().parent.parent / "voice-synth" / "voice-synth.py"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _require_redis() -> None:
    if redis is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "redis package is required for this command. "
            "Rebuild dependencies if needed (make build)."
        ) from REDIS_IMPORT_ERROR


def _parse_iso8601(value: str) -> datetime | None:
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


def _json_load_path(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _yaml_load_path(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


_JOB_SPEC_KEYS = {
    "request_id",
    "queued_at",
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
    "output_name",
    "callback_data",
    "attempt",
}


def _looks_like_job_mapping(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in _JOB_SPEC_KEYS)


def _extract_job_rows(doc: Any, *, source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if doc is None:
        return {}, []

    if isinstance(doc, list):
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(doc, start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}: jobs[{i}] must be an object")
            rows.append(dict(row))
        return {}, rows

    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: YAML root must be a list or mapping with 'jobs'")

    mapping = dict(doc)
    defaults_raw = mapping.get("defaults", {})
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, Mapping):
        raise ValueError(f"{source}: 'defaults' must be an object")
    defaults = dict(defaults_raw)

    if "jobs" in mapping:
        jobs_raw = mapping.get("jobs")
        if jobs_raw is None:
            jobs_raw = []
        if not isinstance(jobs_raw, list):
            raise ValueError(f"{source}: 'jobs' must be a list")
        rows = []
        for i, row in enumerate(jobs_raw, start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}: jobs[{i}] must be an object")
            rows.append(dict(row))
        return defaults, rows

    if _looks_like_job_mapping(mapping):
        return defaults, [mapping]

    raise ValueError(f"{source}: expected a 'jobs' list or a single job object")


def _merge_job_row(defaults: Mapping[str, Any], row: Mapping[str, Any], *, row_index: int) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    merged.update(dict(row))

    defaults_cb = defaults.get("callback_data")
    row_cb = row.get("callback_data")
    if defaults_cb is not None or row_cb is not None:
        callback_data: dict[str, Any] = {}
        if defaults_cb is not None:
            if not isinstance(defaults_cb, Mapping):
                raise ValueError(f"defaults.callback_data must be an object (row {row_index})")
            callback_data.update(dict(defaults_cb))
        if row_cb is not None:
            if not isinstance(row_cb, Mapping):
                raise ValueError(f"jobs[{row_index}].callback_data must be an object")
            callback_data.update(dict(row_cb))
        merged["callback_data"] = callback_data

    return merged


def _default_request_id_from_output_name(output_name: str) -> str:
    cleaned = output_name.strip().strip("/")
    if not cleaned:
        raise ValueError("output_name is required to derive request_id")
    return cleaned.replace("/", ":")


@dataclass(slots=True, frozen=True)
class RenderedJob:
    index: int
    spec: JobSpec
    source_key: str


def _source_key(spec: JobSpec) -> str:
    if spec.voice:
        return f"voice:{spec.voice}"
    assert spec.speaker is not None
    return f"speaker:{spec.speaker}"


def render_jobs(rows: Sequence[Mapping[str, Any]], *, defaults: Mapping[str, Any] | None = None) -> list[RenderedJob]:
    merged_defaults = dict(defaults or {})
    rendered: list[RenderedJob] = []

    for i, row in enumerate(rows, start=1):
        merged = _merge_job_row(merged_defaults, row, row_index=i)
        request_id = merged.get("request_id")
        if request_id is None:
            output_name = merged.get("output_name")
            if not isinstance(output_name, str):
                raise ValueError(f"jobs[{i}].output_name must be provided to derive request_id")
            merged["request_id"] = _default_request_id_from_output_name(output_name)

        try:
            spec = normalize_job_spec(merged)
        except ValueError as exc:
            raise ValueError(f"jobs[{i}]: {exc}") from exc

        rendered.append(RenderedJob(index=i, spec=spec, source_key=_source_key(spec)))

    return rendered


def render_jobs_from_yaml(yaml_file: Path) -> list[RenderedJob]:
    doc = _yaml_load_path(yaml_file)
    defaults, rows = _extract_job_rows(doc, source=yaml_file)
    return render_jobs(rows, defaults=defaults)


def _voice_presence_errors(jobs: Sequence[RenderedJob], *, cache_dir: Path) -> list[str]:
    registry = VoiceRegistry(cache_dir)
    missing = sorted(
        {
            job.spec.voice
            for job in jobs
            if job.spec.voice and not registry.exists(job.spec.voice)
        }
    )
    return [f"voice '{slug}' does not exist in {registry.voice_dir(slug).parent}" for slug in missing]


def _speak_argv_for_job(spec: JobSpec, *, text_file: Path, out_exact: Path, json_result: Path) -> list[str]:
    argv = ["voice-synth", "speak"]
    if spec.voice:
        argv.extend(["--voice", spec.voice])
    elif spec.speaker:
        argv.extend(["--speaker", spec.speaker])

    argv.extend(["--text-file", str(text_file)])

    if spec.language:
        argv.extend(["--language", spec.language])
    if spec.tone:
        argv.extend(["--tone", spec.tone])
    if spec.instruct:
        argv.extend(["--instruct", spec.instruct])
    if spec.instruct_style:
        argv.extend(["--instruct-style", spec.instruct_style])
    if spec.profile:
        argv.extend(["--profile", spec.profile])

    argv.extend(["--variants", str(spec.variants)])
    if spec.select_best:
        argv.append("--select-best")
    if spec.chunk:
        argv.append("--chunk")

    if spec.temperature is not None:
        argv.extend(["--temperature", str(spec.temperature)])
    if spec.top_p is not None:
        argv.extend(["--top-p", str(spec.top_p)])
    if spec.repetition_penalty is not None:
        argv.extend(["--repetition-penalty", str(spec.repetition_penalty)])
    if spec.max_new_tokens is not None:
        argv.extend(["--max-new-tokens", str(spec.max_new_tokens)])

    argv.extend(["--out-exact", str(out_exact), "--json-result", str(json_result)])
    return argv


def build_manifest(
    jobs: Sequence[RenderedJob],
    *,
    out_dir: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_jobs = sorted(
        jobs,
        key=lambda j: (
            j.source_key,
            j.spec.output_name,
            j.spec.request_id,
        ),
    )

    manifest_jobs: list[dict[str, Any]] = []
    for item in ordered_jobs:
        spec = item.spec
        out_exact = out_dir / spec.output_name
        out_exact.mkdir(parents=True, exist_ok=True)
        text_file = out_exact / "text.txt"
        json_result = out_exact / "result.json"
        text_file.write_text(spec.text, encoding="utf-8")

        argv = _speak_argv_for_job(
            spec,
            text_file=text_file,
            out_exact=out_exact,
            json_result=json_result,
        )

        manifest_jobs.append(
            {
                "request_id": spec.request_id,
                "content_hash": spec.content_hash,
                "attempt": spec.attempt,
                "source_key": item.source_key,
                "output_name": spec.output_name,
                "out_exact": str(out_exact),
                "text_file": str(text_file),
                "json_result": str(json_result),
                "argv": argv,
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "run_id": run_id or f"job-runner-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "jobs": manifest_jobs,
    }


def write_manifest(manifest: Mapping[str, Any], *, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(manifest), fh, indent=2, sort_keys=True)
    return path


def _slugify_filename(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "job"


def _command_for_manifest_argv(argv: Sequence[str]) -> list[str]:
    if not argv:
        raise ValueError("manifest job argv cannot be empty")
    if argv[0] == "voice-synth":
        return [sys.executable, str(VOICE_SYNTH_SCRIPT), *argv[1:]]
    return list(argv)


def execute_manifest_data(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: Path,
    fail_fast: bool,
    dry_run: bool,
) -> dict[str, Any]:
    jobs_raw = manifest.get("jobs")
    if not isinstance(jobs_raw, list):
        raise ValueError("Manifest must contain a jobs list")

    manifest_dir = Path(manifest_dir)
    logs_dir = manifest_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(manifest.get("run_id") or "manifest-run")
    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    execution_jobs: list[dict[str, Any]] = []
    aborted = False

    for idx, raw_job in enumerate(jobs_raw, start=1):
        if not isinstance(raw_job, Mapping):
            raise ValueError(f"manifest jobs[{idx}] must be an object")

        request_id = str(raw_job.get("request_id") or f"job-{idx:03d}")
        output_name = str(raw_job.get("output_name") or "")
        argv_raw = raw_job.get("argv")
        if not isinstance(argv_raw, list) or not all(isinstance(a, str) for a in argv_raw):
            raise ValueError(f"manifest jobs[{idx}].argv must be a string list")

        exec_cmd = _command_for_manifest_argv(argv_raw)

        safe_name = _slugify_filename(request_id)
        stdout_log = logs_dir / f"{idx:03d}-{safe_name}.stdout.log"
        stderr_log = logs_dir / f"{idx:03d}-{safe_name}.stderr.log"

        job_started = _utc_now_iso()
        job_t0 = time.perf_counter()

        if dry_run:
            return_code = 0
            stdout_text = ""
            stderr_text = ""
        else:
            proc = subprocess.run(exec_cmd, capture_output=True, text=True)
            return_code = int(proc.returncode)
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""

        stdout_log.write_text(stdout_text, encoding="utf-8")
        stderr_log.write_text(stderr_text, encoding="utf-8")

        duration_sec = round(time.perf_counter() - job_t0, 3)
        status = "ok" if return_code == 0 else "failed"

        execution_jobs.append(
            {
                "request_id": request_id,
                "output_name": output_name,
                "argv": list(argv_raw),
                "exec_cmd": exec_cmd,
                "status": status,
                "return_code": return_code,
                "started_at": job_started,
                "completed_at": _utc_now_iso(),
                "duration_sec": duration_sec,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )

        if return_code != 0 and fail_fast:
            aborted = True
            break

    total_sec = round(time.perf_counter() - t0, 3)
    failed_count = sum(1 for item in execution_jobs if item["status"] != "ok")

    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_created_at": manifest.get("created_at"),
        "started_at": started_at,
        "completed_at": _utc_now_iso(),
        "duration_sec": total_sec,
        "job_count": len(jobs_raw),
        "executed_count": len(execution_jobs),
        "failed_count": failed_count,
        "success": failed_count == 0,
        "aborted": aborted,
        "dry_run": dry_run,
        "jobs": execution_jobs,
    }


def _redis_client(redis_url: str | None) -> Any:
    _require_redis()
    url = redis_url or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
    client = redis.Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Unable to connect to Redis at {url}: {exc}") from exc
    return client


def _parse_json_hash_value(value: Any) -> Any:
    if value is None:
        return None
    text = _to_text(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _summary_rows_from_enqueue_results(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _print_enqueue_table(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        print("No jobs.")
        return

    print("IDX  STATUS             REQUEST_ID                         STREAM_ID")
    for row in rows:
        idx = int(row.get("index", 0))
        status = str(row.get("status", ""))
        request_id = str(row.get("request_id", ""))
        stream_id = str(row.get("stream_id", "") or "-")
        print(f"{idx:>3}  {status:<17}  {request_id:<33}  {stream_id}")


def _print_plan(jobs: Sequence[RenderedJob]) -> None:
    print("IDX  SOURCE                  REQUEST_ID                         OUTPUT_NAME                CHARS")
    by_source: dict[str, int] = {}
    for item in jobs:
        spec = item.spec
        chars = len(spec.text)
        by_source[item.source_key] = by_source.get(item.source_key, 0) + chars
        print(
            f"{item.index:>3}  {item.source_key:<22}  {spec.request_id:<33}  "
            f"{spec.output_name:<24}  {chars:>5}"
        )

    print("\nCharacter totals by source:")
    for source_key in sorted(by_source):
        print(f"  {source_key:<22}  {by_source[source_key]:>7}")


def _build_beatsheet_jobs(
    beatsheet: Mapping[str, Any],
    *,
    voice: str | None,
    speaker: str | None,
    language: str,
    tone: str | None,
    instruct: str | None,
    instruct_style: str | None,
    profile: str | None,
    variants: int,
    select_best: bool,
    chunk: bool,
    temperature: float | None,
    top_p: float | None,
    repetition_penalty: float | None,
    max_new_tokens: int | None,
) -> list[RenderedJob]:
    topic_id = beatsheet.get("topicId")
    if not isinstance(topic_id, str) or not topic_id.strip():
        raise ValueError("beatsheet.topicId is required")
    topic_id = topic_id.strip()

    beats = beatsheet.get("beats")
    if not isinstance(beats, list):
        raise ValueError("beatsheet.beats must be a list")

    rows: list[dict[str, Any]] = []
    for i, beat in enumerate(beats, start=1):
        if not isinstance(beat, Mapping):
            raise ValueError(f"beats[{i}] must be an object")

        beat_id_raw = beat.get("id")
        try:
            beat_id = int(beat_id_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"beats[{i}].id must be an integer") from exc

        narration = beat.get("narration")
        if not isinstance(narration, str) or not narration.strip():
            raise ValueError(f"beats[{i}].narration must be a non-empty string")

        beat_tag = f"beat-{beat_id:03d}"
        row: dict[str, Any] = {
            "request_id": f"{topic_id}:{beat_tag}",
            "output_name": f"{topic_id}/{beat_tag}",
            "voice": voice,
            "speaker": speaker,
            "text": narration.strip(),
            "language": language,
            "tone": tone,
            "instruct": instruct,
            "instruct_style": instruct_style,
            "profile": profile,
            "variants": variants,
            "select_best": select_best,
            "chunk": chunk,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "callback_data": {
                "topic_id": topic_id,
                "beat_id": beat_id,
            },
        }
        rows.append(row)

    return render_jobs(rows)


def _get_result_payload(redis_client: Any, queue: JobQueue, request_id: str) -> dict[str, Any]:
    done = _parse_json_hash_value(redis_client.hget(queue.keys.result, request_id))
    failed = _parse_json_hash_value(redis_client.hget(queue.keys.failed, request_id))

    if done is not None:
        return {
            "request_id": request_id,
            "status": "done",
            "record": done,
        }

    if failed is not None:
        return {
            "request_id": request_id,
            "status": "failed",
            "record": failed,
        }

    return {
        "request_id": request_id,
        "status": "not_found",
        "record": None,
    }


def _collect_status(redis_client: Any, queue: JobQueue, *, lease_idle_ms: int) -> dict[str, Any]:
    try:
        stream_length = queue.stream_length()
    except Exception:
        stream_length = 0

    group_exists = True
    try:
        pending_count = queue.pending_count()
    except Exception as exc:
        if "NOGROUP" in str(exc):
            pending_count = 0
            group_exists = False
        else:
            raise

    stale_pel_count = 0
    if group_exists:
        try:
            stale_entries = redis_client.xpending_range(
                queue.keys.stream,
                queue.consumer_group,
                "-",
                "+",
                10_000,
                idle=max(1, int(lease_idle_ms)),
            )
            stale_pel_count = len(stale_entries)
        except Exception:
            stale_pel_count = 0

    retry_queue_size = int(redis_client.zcard(queue.keys.retry_zset))
    lock_owner = queue.lock_owner()
    lock_ttl_ms = int(redis_client.pttl(queue.keys.watcher_lock)) if lock_owner else -2
    active_instance = redis_client.get(queue.keys.watcher_instance)
    first_queued_raw = redis_client.get(queue.keys.first_queued)

    first_queued_iso = _to_text(first_queued_raw) if first_queued_raw is not None else None
    first_queued_age_sec: float | None = None
    if first_queued_iso is not None:
        dt = _parse_iso8601(first_queued_iso)
        if dt is not None:
            first_queued_age_sec = round((datetime.now(timezone.utc) - dt).total_seconds(), 3)

    return {
        "timestamp": _utc_now_iso(),
        "stream": {
            "key": queue.keys.stream,
            "length": stream_length,
            "group": queue.consumer_group,
            "group_exists": group_exists,
            "pending": pending_count,
            "stale_pel": stale_pel_count,
            "first_queued": first_queued_iso,
            "first_queued_age_sec": first_queued_age_sec,
        },
        "retry_queue_size": retry_queue_size,
        "lock": {
            "owner": lock_owner,
            "ttl_ms": lock_ttl_ms,
        },
        "active_instance": _to_text(active_instance) if active_instance is not None else None,
    }


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _handle_plan(args: argparse.Namespace) -> int:
    jobs = render_jobs_from_yaml(Path(args.yaml_file))
    errors = _voice_presence_errors(jobs, cache_dir=Path(args.cache))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "job_count": len(jobs),
            "jobs": [
                {
                    "index": item.index,
                    "source": item.source_key,
                    "request_id": item.spec.request_id,
                    "output_name": item.spec.output_name,
                    "characters": len(item.spec.text),
                }
                for item in jobs
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(jobs)
    return 0


def _enqueue_jobs(
    *,
    queue: JobQueue,
    jobs: Sequence[RenderedJob],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in jobs:
        result = queue.enqueue(item.spec)
        rows.append(
            {
                "index": item.index,
                "request_id": result.request_id,
                "output_name": item.spec.output_name,
                "status": result.status,
                "stream_id": result.stream_id,
                "attempt": item.spec.attempt,
            }
        )
    return rows


def _handle_enqueue(args: argparse.Namespace) -> int:
    jobs = render_jobs_from_yaml(Path(args.yaml_file))
    errors = _voice_presence_errors(jobs, cache_dir=Path(args.cache))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    queue.ensure_group()
    rows = _enqueue_jobs(queue=queue, jobs=jobs)
    summary = _summary_rows_from_enqueue_results(rows)

    if args.json:
        print(json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True))
    else:
        _print_enqueue_table(rows)
        print("\nSummary:")
        for key in sorted(summary):
            print(f"  {key:<17} {summary[key]}")
    return 0


def _handle_enqueue_beatsheet(args: argparse.Namespace) -> int:
    beatsheet = _json_load_path(Path(args.beatsheet_json))
    if not isinstance(beatsheet, Mapping):
        print("ERROR: beatsheet JSON root must be an object", file=sys.stderr)
        return 1

    try:
        jobs = _build_beatsheet_jobs(
            beatsheet,
            voice=args.voice,
            speaker=args.speaker,
            language=args.language,
            tone=args.tone,
            instruct=args.instruct,
            instruct_style=args.instruct_style,
            profile=args.profile,
            variants=args.variants,
            select_best=args.select_best,
            chunk=args.chunk,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors = _voice_presence_errors(jobs, cache_dir=Path(args.cache))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    queue.ensure_group()
    rows = _enqueue_jobs(queue=queue, jobs=jobs)
    summary = _summary_rows_from_enqueue_results(rows)

    if args.json:
        print(json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True))
    else:
        _print_enqueue_table(rows)
        print("\nSummary:")
        for key in sorted(summary):
            print(f"  {key:<17} {summary[key]}")
    return 0


def _handle_compile(args: argparse.Namespace) -> int:
    jobs = render_jobs_from_yaml(Path(args.yaml_file))
    errors = _voice_presence_errors(jobs, cache_dir=Path(args.cache))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    manifest = build_manifest(jobs, out_dir=out_dir, run_id=args.run_id)
    manifest_path = write_manifest(manifest, out_dir=out_dir)

    payload = {
        "run_id": manifest["run_id"],
        "job_count": len(manifest["jobs"]),
        "manifest": str(manifest_path),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Compiled {payload['job_count']} job(s)")
        print(f"  run_id:    {payload['run_id']}")
        print(f"  manifest:  {payload['manifest']}")
    return 0


def _run_manifest(path: Path, *, fail_fast: bool, dry_run: bool) -> dict[str, Any]:
    manifest_doc = _json_load_path(path)
    if not isinstance(manifest_doc, Mapping):
        raise ValueError(f"Manifest root must be an object: {path}")

    execution = execute_manifest_data(
        manifest_doc,
        manifest_dir=path.parent,
        fail_fast=fail_fast,
        dry_run=dry_run,
    )

    execution_path = path.parent / "execution.json"
    with open(execution_path, "w", encoding="utf-8") as fh:
        json.dump(execution, fh, indent=2, sort_keys=True)

    execution["execution_path"] = str(execution_path)
    return execution


def _handle_run(args: argparse.Namespace) -> int:
    jobs = render_jobs_from_yaml(Path(args.yaml_file))
    errors = _voice_presence_errors(jobs, cache_dir=Path(args.cache))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="job-runner-run-"))

    manifest = build_manifest(jobs, out_dir=out_dir, run_id=args.run_id)
    manifest_path = write_manifest(manifest, out_dir=out_dir)

    try:
        execution = _run_manifest(
            manifest_path,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(execution, indent=2, sort_keys=True))
    else:
        print(f"Run complete: {'success' if execution['success'] else 'failed'}")
        print(f"  manifest:   {manifest_path}")
        print(f"  execution:  {execution['execution_path']}")
        print(f"  executed:   {execution['executed_count']} / {execution['job_count']}")
        print(f"  failed:     {execution['failed_count']}")

    return 0 if execution["success"] else 1


def _handle_execute_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_json)
    try:
        execution = _run_manifest(
            manifest_path,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(execution, indent=2, sort_keys=True))
    else:
        print(f"Execution complete: {'success' if execution['success'] else 'failed'}")
        print(f"  execution:  {execution['execution_path']}")
        print(f"  executed:   {execution['executed_count']} / {execution['job_count']}")
        print(f"  failed:     {execution['failed_count']}")

    return 0 if execution["success"] else 1


def _handle_status(args: argparse.Namespace) -> int:
    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    status_payload = _collect_status(
        redis_client,
        queue,
        lease_idle_ms=args.lease_idle_ms,
    )

    if args.json:
        print(json.dumps(status_payload, indent=2, sort_keys=True))
    else:
        stream = status_payload["stream"]
        lock = status_payload["lock"]
        print(f"stream_length:        {stream['length']}")
        print(f"pending:              {stream['pending']}")
        print(f"stale_pel:            {stream['stale_pel']}")
        print(f"retry_queue_size:     {status_payload['retry_queue_size']}")
        print(f"lock_owner:           {lock['owner'] or '-'}")
        print(f"lock_ttl_ms:          {lock['ttl_ms']}")
        print(f"active_instance:      {status_payload['active_instance'] or '-'}")
        print(f"first_queued:         {stream['first_queued'] or '-'}")
        print(f"first_queued_age_sec: {stream['first_queued_age_sec']}")
    return 0


def _handle_result(args: argparse.Namespace) -> int:
    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    payload = _get_result_payload(redis_client, queue, args.request_id)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"request_id: {payload['request_id']}")
        print(f"status:     {payload['status']}")
        if payload["record"] is not None:
            print(json.dumps(payload["record"], indent=2, sort_keys=True))
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    jobs = render_jobs_from_yaml(Path(args.yaml_file))

    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)

    rows: list[dict[str, Any]] = []
    summary: dict[str, int] = {}

    for item in jobs:
        request_id = item.spec.request_id
        result_payload = _get_result_payload(redis_client, queue, request_id)
        status = str(result_payload["status"])

        if status == "not_found" and redis_client.hexists(queue.keys.request_index, request_id):
            status = "queued"

        rows.append(
            {
                "request_id": request_id,
                "output_name": item.spec.output_name,
                "status": status,
                "record": result_payload["record"],
            }
        )
        summary[status] = summary.get(status, 0) + 1

    payload = {
        "job_count": len(rows),
        "summary": summary,
        "rows": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("STATUS   REQUEST_ID                         OUTPUT_NAME")
        for row in rows:
            print(f"{row['status']:<7}  {row['request_id']:<33}  {row['output_name']}")
        print("\nSummary:")
        for key in sorted(summary):
            print(f"  {key:<7} {summary[key]}")

    return 0


def _handle_retry(args: argparse.Namespace) -> int:
    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    failed_raw = redis_client.hget(queue.keys.failed, args.request_id)
    if failed_raw is None:
        print(f"ERROR: request_id '{args.request_id}' has no terminal failure record", file=sys.stderr)
        return 1

    failed = _parse_json_hash_value(failed_raw)
    if not isinstance(failed, Mapping):
        print(
            f"ERROR: failed record for '{args.request_id}' is not JSON object; cannot retry safely",
            file=sys.stderr,
        )
        return 1

    original_job = failed.get("job")
    if not isinstance(original_job, Mapping):
        print(
            f"ERROR: failed record for '{args.request_id}' has no embedded job payload",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        if not _confirm(f"Requeue failed job '{args.request_id}' with attempt=0?"):
            print("Aborted.")
            return 0

    retry_job = dict(original_job)
    retry_job["request_id"] = args.request_id
    retry_job["attempt"] = 0
    retry_job["queued_at"] = _utc_now_iso()

    failed_record_json = _to_text(failed_raw)

    pipe = redis_client.pipeline(transaction=True)
    pipe.hdel(queue.keys.failed, args.request_id)
    pipe.hdel(queue.keys.request_index, args.request_id)
    pipe.execute()

    try:
        result = queue.enqueue(retry_job)
    except Exception:
        redis_client.hset(queue.keys.failed, args.request_id, failed_record_json)
        raise

    if result.status != "queued":
        # Preserve historical failure context if requeue did not succeed.
        redis_client.hset(queue.keys.failed, args.request_id, failed_record_json)

    payload = {
        "request_id": args.request_id,
        "status": result.status,
        "stream_id": result.stream_id,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"request_id: {payload['request_id']}")
        print(f"status:     {payload['status']}")
        print(f"stream_id:  {payload['stream_id'] or '-'}")

    return 0 if result.status == "queued" else 1


def _collect_flush_keys(redis_client: Any, queue: JobQueue, *, hard: bool) -> list[str]:
    keys = [
        queue.keys.stream,
        queue.keys.request_index,
        queue.keys.first_queued,
        queue.keys.retry_zset,
        queue.keys.watcher_lock,
        queue.keys.watcher_heartbeat,
        queue.keys.watcher_instance,
    ]

    metric_pattern = f"{queue.keys.metrics_prefix}*"
    metric_keys = [_to_text(k) for k in redis_client.scan_iter(metric_pattern)]
    keys.extend(metric_keys)

    if hard:
        keys.extend([queue.keys.result, queue.keys.failed])

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _handle_flush(args: argparse.Namespace) -> int:
    try:
        redis_client = _redis_client(args.redis_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    queue = JobQueue(redis_client)
    keys = _collect_flush_keys(redis_client, queue, hard=args.hard)

    if not args.yes:
        mode = "HARD" if args.hard else "SOFT"
        if not _confirm(f"{mode} flush will delete {len(keys)} key(s). Continue?"):
            print("Aborted.")
            return 0

    deleted = int(redis_client.delete(*keys)) if keys else 0
    payload = {
        "deleted": deleted,
        "key_count": len(keys),
        "hard": bool(args.hard),
        "keys": keys,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Deleted {deleted} key(s).")
        print(f"  hard: {payload['hard']}")
        print(f"  considered: {payload['key_count']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="job-runner — Redis stream producer/validation tooling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    plan.add_argument("yaml_file")
    plan.add_argument("--cache", default="/cache")
    plan.add_argument("--json", action="store_true")

    enqueue = sub.add_parser("enqueue", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    enqueue.add_argument("yaml_file")
    enqueue.add_argument("--cache", default="/cache")
    enqueue.add_argument("--redis-url", default=None)
    enqueue.add_argument("--json", action="store_true")

    eb = sub.add_parser("enqueue-beatsheet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eb.add_argument("beatsheet_json")
    eb_mode = eb.add_mutually_exclusive_group(required=True)
    eb_mode.add_argument("--voice", default=None)
    eb_mode.add_argument("--speaker", default=None)
    eb.add_argument("--language", default="English")
    eb.add_argument("--tone", default=None)
    eb.add_argument("--instruct", default=None)
    eb.add_argument("--instruct-style", default=None, dest="instruct_style")
    eb.add_argument("--profile", default=None)
    eb.add_argument("--variants", type=int, default=1)
    eb.add_argument("--select-best", action="store_true")
    eb.add_argument("--chunk", action="store_true")
    eb.add_argument("--temperature", type=float, default=None)
    eb.add_argument("--top-p", type=float, default=None, dest="top_p")
    eb.add_argument("--repetition-penalty", type=float, default=None, dest="repetition_penalty")
    eb.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    eb.add_argument("--cache", default="/cache")
    eb.add_argument("--redis-url", default=None)
    eb.add_argument("--json", action="store_true")

    compile_cmd = sub.add_parser("compile", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    compile_cmd.add_argument("yaml_file")
    compile_cmd.add_argument("--out", required=True)
    compile_cmd.add_argument("--run-id", default=None)
    compile_cmd.add_argument("--cache", default="/cache")
    compile_cmd.add_argument("--json", action="store_true")

    run_cmd = sub.add_parser("run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_cmd.add_argument("yaml_file")
    run_cmd.add_argument("--out", default=None)
    run_cmd.add_argument("--run-id", default=None)
    run_cmd.add_argument("--cache", default="/cache")
    run_cmd.add_argument("--dry-run", action="store_true")
    run_cmd.add_argument("--fail-fast", action="store_true")
    run_cmd.add_argument("--json", action="store_true")

    em = sub.add_parser(
        "execute-manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    em.add_argument("manifest_json")
    em.add_argument("--dry-run", action="store_true")
    em.add_argument("--fail-fast", action="store_true")
    em.add_argument("--json", action="store_true")

    status = sub.add_parser("status", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    status.add_argument("--redis-url", default=None)
    status.add_argument(
        "--lease-idle-ms",
        type=int,
        default=int(os.getenv("WATCHER_LEASE_IDLE_MS", "180000")),
    )
    status.add_argument("--json", action="store_true")

    result = sub.add_parser("result", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("request_id")
    result.add_argument("--redis-url", default=None)
    result.add_argument("--json", action="store_true")

    report = sub.add_parser("report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    report.add_argument("yaml_file")
    report.add_argument("--redis-url", default=None)
    report.add_argument("--json", action="store_true")

    retry = sub.add_parser("retry", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    retry.add_argument("request_id")
    retry.add_argument("--redis-url", default=None)
    retry.add_argument("--yes", action="store_true")
    retry.add_argument("--json", action="store_true")

    flush = sub.add_parser("flush", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    flush.add_argument("--redis-url", default=None)
    flush.add_argument("--hard", action="store_true")
    flush.add_argument("--yes", action="store_true")
    flush.add_argument("--json", action="store_true")

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "plan":
            raise SystemExit(_handle_plan(args))
        if args.command == "enqueue":
            raise SystemExit(_handle_enqueue(args))
        if args.command == "enqueue-beatsheet":
            raise SystemExit(_handle_enqueue_beatsheet(args))
        if args.command == "compile":
            raise SystemExit(_handle_compile(args))
        if args.command == "run":
            raise SystemExit(_handle_run(args))
        if args.command == "execute-manifest":
            raise SystemExit(_handle_execute_manifest(args))
        if args.command == "status":
            raise SystemExit(_handle_status(args))
        if args.command == "result":
            raise SystemExit(_handle_result(args))
        if args.command == "report":
            raise SystemExit(_handle_report(args))
        if args.command == "retry":
            raise SystemExit(_handle_retry(args))
        if args.command == "flush":
            raise SystemExit(_handle_flush(args))

        parser.print_help()
        raise SystemExit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
