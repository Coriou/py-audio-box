#!/usr/bin/env python3
"""Ledgered fal caller for the Stable Audio 3 SFX audio→audio variation tier.

Mirrors ``experiments/fal-sfx-bench/run_benchmark.py`` discipline: fsynced
append-only ledger line per submission, full per-request record JSON, sha256
of every artifact, published-price cost estimate ($0.024/generation,
owner-cited — confirm against the billing CSV during reconciliation).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import fal_client

from common import Ledger, atomic_json, ffprobe, sha256, utc_now

SA3_ENDPOINT = "fal-ai/stable-audio-3/small/sfx/audio-to-audio"
SA3_UNIT_COST_USD = 0.024     # per generation (owner-cited 2026-08-25)
DEFAULT_SEED = 20260825


def load_fal_key() -> str | None:
    """FAL_KEY from the environment, else a naive repo-root .env parse."""
    key = os.environ.get("FAL_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("FAL_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def download(url: str, destination: Path) -> None:
    import httpx

    temporary = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with open(temporary, "wb") as stream:
            for chunk in response.iter_bytes(1 << 16):
                stream.write(chunk)
    temporary.replace(destination)


def run_sa3_a2a(
    run_dir: Path,
    ledger: Ledger,
    case_name: str,
    audio_path: Path,
    prompt: str,
    duration_s: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute one SA3 a2a variation; returns the full request record."""
    key = load_fal_key()
    if not key:
        raise RuntimeError("FAL_KEY not found (env or repo .env)")
    os.environ.setdefault("FAL_KEY", key)

    records_dir = run_dir / "records"
    raw_dir = run_dir / "raw" / case_name
    records_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    arguments = {
        "prompt": prompt,
        "negative_prompt": "speech, vocals, music, singing",
        # Schema default duration is 30 s — always set per event.
        "duration": int(duration_s),
        "num_inference_steps": 8,
        "seed": int(seed),
        # Schema default is mp3; request lossless for placement DSP.
        "output_format": "wav",
    }
    audio_url = fal_client.upload_file(audio_path)
    arguments["audio_url"] = audio_url

    handle = fal_client.submit(SA3_ENDPOINT, arguments=arguments)
    request_id = handle.request_id
    ledger.append({
        "ts": utc_now(),
        "case": case_name,
        "model": "sa3-a2a",
        "endpoint_id": SA3_ENDPOINT,
        "request_id": request_id,
        "event": "submitted",
        "estimated_cost_usd": SA3_UNIT_COST_USD,
    })

    statuses: list[dict[str, Any]] = []
    last_state = None
    while True:
        status = handle.status(with_logs=False)
        state = type(status).__name__
        elapsed = time.perf_counter() - started
        if state != last_state:
            statuses.append({"elapsed_s": round(elapsed, 2),
                             "observed_at": utc_now(),
                             "state": state})
            print(f"{case_name}: {state} at {elapsed:.1f}s", flush=True)
            last_state = state
        if state == "Completed":
            break
        if state == "Failed" or getattr(status, "error", None):
            raise RuntimeError(f"fal request failed: {getattr(status, 'error', state)}")
        time.sleep(0.5)

    result = handle.get()
    outputs = []
    entry = result.get("audio_file") if isinstance(result, dict) else None
    if isinstance(entry, dict) and entry.get("url"):
        outputs.append({"kind": "audio",
                        **{k: entry.get(k) for k in ("url", "content_type", "file_name")}})
    if not outputs and isinstance(result, dict):
        for value in result.values():
            if isinstance(value, dict) and value.get("url"):
                outputs.append({"kind": "audio",
                                **{k: value.get(k) for k in ("url", "content_type", "file_name")}})
                break
    if not outputs:
        raise RuntimeError(f"no audio output in SA3 result keys: {list(result) if isinstance(result, dict) else type(result)}")

    file_name = outputs[0].get("file_name") or Path(outputs[0]["url"].split("?")[0]).name
    destination = raw_dir / file_name
    download(outputs[0]["url"], destination)

    record = {
        "name": case_name,
        "model": "sa3-a2a",
        "endpoint_id": SA3_ENDPOINT,
        "input": str(audio_path),
        "input_sha256": sha256(audio_path),
        "arguments": arguments,
        "started_at": None,
        "finished_at": utc_now(),
        "total_wall_s": round(time.perf_counter() - started, 2),
        "request_id": request_id,
        "statuses": statuses,
        "result_keys": list(result) if isinstance(result, dict) else [],
        "outputs": outputs,
        "primary_output": str(destination),
        "primary_output_sha256": sha256(destination),
        "probe": ffprobe(destination),
        "estimated_cost_usd": SA3_UNIT_COST_USD,
        "status": "completed",
        "error": None,
    }
    atomic_json(records_dir / f"{case_name}.json", record)
    ledger.append({
        "ts": utc_now(),
        "case": case_name,
        "model": "sa3-a2a",
        "endpoint_id": SA3_ENDPOINT,
        "request_id": request_id,
        "event": "completed",
        "estimated_cost_usd": SA3_UNIT_COST_USD,
        "output": str(destination),
        "output_sha256": record["primary_output_sha256"],
    })
    return record
