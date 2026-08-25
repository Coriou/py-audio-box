#!/usr/bin/env python3
"""Run the fal-sfx-bench benchmark over a prepared run directory.

Submits every selected manifest case to its fal.ai endpoint concurrently,
records status transitions, downloads every returned artifact into
runs/<run_id>/raw/<case>/ (original remote filenames preserved), probes and
hashes each artifact, and appends one fsynced ledger.jsonl line per submission
attempt so interrupted runs resume cleanly. Failures are recorded per case
(record + ledger) and never abort the run.

FAL_KEY is read from the environment, falling back to a naive KEY=VALUE parse
of the repo-root .env. It is never logged or persisted into any record.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fal_client

from models import MODELS, OutputArtifact, ModelSpec, estimate_cost, get_model


REPO_ROOT = Path(__file__).resolve().parents[2]

CONTENT_TYPE_SUFFIXES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
}
KIND_DEFAULT_SUFFIXES = {"video": ".mp4", "audio": ".bin"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,"
        "width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def status_payload(status: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(status)
    payload["state"] = type(status).__name__
    return payload


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "py-audio-box-eval/1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


@dataclass
class Ledger:
    """Append-only JSONL log; every line is flushed and fsynced."""

    path: Path
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def append(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def load_fal_key() -> str | None:
    """FAL_KEY from the environment, else a naive repo-root .env parse."""
    key = os.environ.get("FAL_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != "FAL_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            return value
    return None


def completed_ledger_cases(ledger_path: Path) -> set[str]:
    """Case names whose latest submission attempt completed successfully."""
    completed: set[str] = set()
    if not ledger_path.is_file():
        return completed
    for raw_line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("event") == "completed" and entry.get("request_id"):
            name = entry.get("case")
            if name:
                completed.add(name)
    return completed


def _url_suffix(url: str) -> str:
    return Path(url.split("?", 1)[0]).suffix.lower()


def artifact_filename(artifact: OutputArtifact, case_name: str, index: int) -> str:
    """Remote file_name when present, else a deterministic local name."""
    remote = artifact.get("file_name") or ""
    base = remote.replace("\\", "/").split("/")[-1].strip()
    if base and base not in {".", ".."}:
        return base
    suffix = (
        CONTENT_TYPE_SUFFIXES.get(artifact.get("content_type") or "")
        or _url_suffix(artifact["url"])
        or KIND_DEFAULT_SUFFIXES[artifact["kind"]]
    )
    return f"{case_name}_{index}{suffix}"


def unique_filename(candidate: str, taken: set[str]) -> str:
    if candidate not in taken:
        return candidate
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    counter = 1
    while f"{stem}_{counter}{suffix}" in taken:
        counter += 1
    return f"{stem}_{counter}{suffix}"


def has_audio_stream(probe: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def probe_duration(probe: dict[str, Any]) -> float | None:
    try:
        return float((probe.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        return None


def choose_primary_output(outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Video-with-audio artifact if any, else the first audio file."""
    videos_with_audio = [
        output
        for output in outputs
        if output["kind"] == "video" and has_audio_stream(output["probe"])
    ]
    if videos_with_audio:
        return videos_with_audio[0]
    audios = [output for output in outputs if output["kind"] == "audio"]
    if audios:
        return audios[0]
    return outputs[0] if outputs else None


def execute_case(
    case: dict[str, Any],
    spec: ModelSpec,
    runs_dir: Path,
    ledger: Ledger,
) -> dict[str, Any]:
    """Drive one case end to end; always returns a record, never raises."""
    name = case["name"]
    records_dir = runs_dir / "records"
    raw_dir = runs_dir / "raw" / name
    records_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "name": name,
        "model": spec.key,
        "endpoint_id": spec.endpoint_id,
        "input": None,
        "input_sha256": None,
        "input_probe": None,
        "prompt": case.get("prompt"),
        "arguments": None,
        "started_at": utc_now(),
        "finished_at": None,
        "upload_s": None,
        "submit_s": None,
        "queue_to_processing_s": None,
        "processing_observed_s": None,
        "result_fetch_s": None,
        "download_s": None,
        "total_wall_s": None,
        "request_id": None,
        "statuses": [],
        "fal_metrics": {},
        "result": None,
        "outputs": [],
        "primary_output": None,
        "output_duration_s": None,
        "estimated_cost_usd": case.get("estimated_cost_usd"),
        "status": None,
        "error": None,
    }
    started = time.perf_counter()
    print(f"{name}: starting ({spec.endpoint_id})", flush=True)
    try:
        input_path = Path(case["input"]).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        record["input"] = str(input_path)
        record["input_sha256"] = sha256(input_path)
        record["input_probe"] = ffprobe(input_path)

        # Manifest cases carry the final payload minus the media URL; rebuild
        # from the registry when absent.
        arguments = dict(case["arguments"]) if case.get("arguments") else spec.build_arguments(case)
        if spec.media_key:
            upload_started = time.perf_counter()
            media_url = fal_client.upload_file(input_path)
            record["upload_s"] = time.perf_counter() - upload_started
            arguments[spec.media_key] = media_url
        record["arguments"] = arguments

        submit_started = time.perf_counter()
        handle = fal_client.submit(spec.endpoint_id, arguments=arguments)
        submit_finished = time.perf_counter()
        record["request_id"] = handle.request_id
        record["submit_s"] = submit_finished - submit_started
        ledger.append(
            {
                "ts": utc_now(),
                "case": name,
                "model": spec.key,
                "endpoint_id": spec.endpoint_id,
                "request_id": handle.request_id,
                "event": "submitted",
                "estimated_cost_usd": case.get("estimated_cost_usd"),
            }
        )

        statuses: list[dict[str, Any]] = []
        last_state: str | None = None
        inference_started: float | None = None
        completed_at: float | None = None
        final_status: Any = None
        while True:
            status = handle.status(with_logs=True)
            state = type(status).__name__
            now = time.perf_counter()
            if state != last_state:
                statuses.append(
                    {
                        "elapsed_s": now - started,
                        "observed_at": utc_now(),
                        **status_payload(status),
                    }
                )
                print(f"{name}: {state} at {now - started:.2f}s", flush=True)
                last_state = state
            if state == "InProgress" and inference_started is None:
                inference_started = now
            if state == "Completed":
                completed_at = now
                final_status = status
                break
            time.sleep(0.5)

        record["statuses"] = statuses
        if getattr(final_status, "error", None):
            raise RuntimeError(f"fal request failed: {final_status.error}")

        result_started = time.perf_counter()
        result = handle.get()
        result_finished = time.perf_counter()
        record["result_fetch_s"] = result_finished - result_started
        record["result"] = result
        record["fal_metrics"] = getattr(final_status, "metrics", {})
        record["queue_to_processing_s"] = (
            inference_started - submit_finished if inference_started is not None else None
        )
        record["processing_observed_s"] = (
            completed_at - inference_started
            if completed_at is not None and inference_started is not None
            else None
        )

        artifacts = spec.extract_outputs(result)
        if not artifacts:
            raise RuntimeError("response contained no downloadable artifacts")

        raw_dir.mkdir(parents=True, exist_ok=True)
        download_started = time.perf_counter()
        outputs: list[dict[str, Any]] = []
        taken: set[str] = set()
        for index, artifact in enumerate(artifacts):
            local_name = unique_filename(artifact_filename(artifact, name, index), taken)
            taken.add(local_name)
            destination = raw_dir / local_name
            download(artifact["url"], destination)
            outputs.append(
                {
                    "kind": artifact["kind"],
                    "url": artifact["url"],
                    "content_type": artifact.get("content_type"),
                    "file_name": artifact.get("file_name"),
                    "local_path": str(destination.resolve()),
                    "sha256": sha256(destination),
                    "probe": ffprobe(destination),
                }
            )
        download_finished = time.perf_counter()
        record["download_s"] = download_finished - download_started
        record["outputs"] = outputs

        primary = choose_primary_output(outputs)
        record["primary_output"] = primary["local_path"] if primary else None
        record["output_duration_s"] = probe_duration(primary["probe"]) if primary else None
        record["estimated_cost_usd"] = estimate_cost(
            spec, arguments, record["output_duration_s"]
        )
        record["total_wall_s"] = download_finished - started
        record["finished_at"] = utc_now()
        record["status"] = "completed"
        record["error"] = None
        ledger.append(
            {
                "ts": utc_now(),
                "case": name,
                "model": spec.key,
                "endpoint_id": spec.endpoint_id,
                "request_id": handle.request_id,
                "event": "completed",
                "estimated_cost_usd": record["estimated_cost_usd"],
            }
        )
    except Exception as exc:  # noqa: BLE001 — failures must not abort the run
        record["total_wall_s"] = time.perf_counter() - started
        record["finished_at"] = utc_now()
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        ledger.append(
            {
                "ts": utc_now(),
                "case": name,
                "model": spec.key,
                "endpoint_id": spec.endpoint_id,
                "request_id": record["request_id"],
                "event": "failed",
                "estimated_cost_usd": record["estimated_cost_usd"],
            }
        )
        print(f"{name}: FAILED — {record['error']}", flush=True)
    atomic_json(records_dir / f"{name}.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="work/sfx-bench/<run_id> directory holding manifest.json")
    parser.add_argument("--models", default=None, help="comma-separated model keys to run (default: all)")
    parser.add_argument("--cases", default=None, help="substring filter on case names")
    parser.add_argument("--max-workers", type=int, default=4, help="concurrent submissions (default: 4)")
    parser.add_argument("--force", action="store_true", help="re-run cases already completed in the ledger")
    return parser.parse_args()


def aggregate_run(
    run_id: str,
    started_at: str,
    manifest_path: Path,
    model_keys: list[str],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = per_model.setdefault(
            record["model"],
            {"cases": 0, "completed": 0, "failed": 0, "estimated_cost_usd": 0.0, "wall_times_s": []},
        )
        bucket["cases"] += 1
        bucket[record["status"]] += 1
        bucket["estimated_cost_usd"] += record.get("estimated_cost_usd") or 0.0
        if record.get("total_wall_s") is not None:
            bucket["wall_times_s"].append(record["total_wall_s"])
    for bucket in per_model.values():
        walls = bucket.pop("wall_times_s")
        bucket["mean_total_wall_s"] = round(sum(walls) / len(walls), 3) if walls else None
        bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"], 6)
    case_index = [
        {
            "name": record["name"],
            "model": record["model"],
            "status": record["status"],
            "request_id": record["request_id"],
            "error": record["error"],
            "total_wall_s": record["total_wall_s"],
            "estimated_cost_usd": record["estimated_cost_usd"],
        }
        for record in sorted(records, key=lambda item: item["name"])
    ]
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "manifest": str(manifest_path.resolve()),
        "models": model_keys,
        "per_model": dict(sorted(per_model.items())),
        "cases": case_index,
    }


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        print("--max-workers must be >= 1", file=sys.stderr)
        return 1

    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    selected_models = [key.strip() for key in args.models.split(",")] if args.models else None
    if selected_models:
        unknown = [key for key in selected_models if key not in MODELS]
        if unknown:
            print(
                f"unknown model key(s): {', '.join(unknown)}; known keys: {', '.join(sorted(MODELS))}",
                file=sys.stderr,
            )
            return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_cases = manifest.get("cases", [])

    def matches(case: dict[str, Any]) -> bool:
        if selected_models and case["model"] not in selected_models:
            return False
        return bool(not args.cases or args.cases in case["name"])

    chosen = [case for case in manifest_cases if matches(case)]
    if manifest_cases and not chosen:
        print(
            "no manifest cases match --models/--cases filters "
            f"({len(manifest_cases)} cases in manifest)",
            file=sys.stderr,
        )
        return 1

    fal_key = load_fal_key()
    if not fal_key:
        print("FAL_KEY must be set in the environment or repo-root .env", file=sys.stderr)
        return 2
    os.environ["FAL_KEY"] = fal_key

    run_id = run_dir.name
    runs_dir = run_dir / "runs" / run_id
    ledger_path = runs_dir / "ledger.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)

    done = set() if args.force else completed_ledger_cases(ledger_path)
    pending = [case for case in chosen if case["name"] not in done]
    for case in chosen:
        if case["name"] in done:
            print(f"{case['name']}: already complete (ledger)", flush=True)

    started_at = utc_now()
    records: list[dict[str, Any]] = []
    ledger = Ledger(ledger_path)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(execute_case, case, get_model(case["model"]), runs_dir, ledger)
            for case in pending
        ]
        for future in as_completed(futures):
            records.append(future.result())

    run = aggregate_run(run_id, started_at, manifest_path, sorted({case["model"] for case in chosen}), records)
    run_path = runs_dir / "run.json"
    atomic_json(run_path, run)

    completed_count = sum(1 for record in records if record["status"] == "completed")
    failed_count = sum(1 for record in records if record["status"] == "failed")
    total_cost = round(sum(record.get("estimated_cost_usd") or 0.0 for record in records), 6)
    print(
        f"{run_id}: {completed_count} completed, {failed_count} failed, "
        f"est ${total_cost:.4f} — {run_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
