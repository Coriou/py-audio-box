#!/usr/bin/env python3
"""Shared helpers for the sfx-bank-pipeline POC.

Conventions follow ``experiments/fal-sfx-bench/``: standalone scripts, atomic
JSON writes, fsynced append-only ledgers, and run directories under ``work/``.

Timeline mapping matters: the standardized bench clips are trimmed windows of
longer sources (see ``experiments/fal-sfx-bench/prepare_inputs.py::CLIPS``).
Auto-sfx contact timestamps are on the SOURCE timeline; subtracting the trim
start yields clip-relative seconds used everywhere in this pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Prior benchmark run whose raw outputs feed this POC.
BENCH_RUN_DIR = REPO_ROOT / "work" / "sfx-bench" / "20260825-165347"
BENCH_RAW_DIR = BENCH_RUN_DIR / "runs" / "20260825-165347" / "raw"

#: Owner-provided template bank (confirmed 2026-08-26).
SFX_BANK_DIR = REPO_ROOT / "work" / "sfx-bank" / "template"

#: V2A models from the benchmark whose raw outputs are timing-oracle
#: candidates (validated against contacts; best per clip wins).
ORACLE_MODELS: tuple[str, ...] = ("mirelo16", "kling-v2a", "sonilo11", "controlfoley")

#: Auto-sfx runs supplying foot-contact ground truth, per clip key.
#: None = no contact ground truth for this clip (non-footstep content).
CONTACT_RUNS: dict[str, str | None] = {
    "walk1": "real-walk1",
    "walk2": "real-walk2",
    "man": None,
    "ocean-long": None,
    "ocean-short": None,
}

#: Trim windows, mirrored verbatim from prepare_inputs.py::CLIPS.
CLIP_TRIMS: dict[str, tuple[float, float] | None] = {
    "walk1": (10.0, 19.4),
    "walk2": (18.0, 27.4),
    "man": None,
    "ocean-long": (10.0, 20.0),
    "ocean-short": None,
}

#: Canonical prompts reused from the benchmark manifest (for SA3 a2a calls).
CANONICAL_PROMPTS: dict[str, str] = {
    "walk1": (
        "Synchronized footsteps on a forest hiking trail: one crisp footstep "
        "at every visible foot contact, boots on dirt and leaves, no speech, "
        "no music"
    ),
    "walk2": (
        "Synchronized barefoot footsteps on grass, one soft step at every "
        "visible foot contact, outdoor ambience kept subtle, no speech, no music"
    ),
    "man": (
        "Synchronized footsteps on paved stone, one clean step at every "
        "visible foot contact, light city air, no speech, no music"
    ),
    "ocean-long": (
        "Ocean waves rolling onto shore seen from above, natural surf rhythm, "
        "no speech, no music"
    ),
    "ocean-short": (
        "Gentle ocean waves washing ashore, natural surf rhythm, no speech, "
        "no music"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(command[:6])}…): {process.stderr[-400:]}")
    return process


def ffprobe(path: Path) -> dict[str, Any]:
    import json as _json

    command = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    return _json.loads(subprocess.check_output(command, text=True))


def model_raw_source(bench_run_dir: Path, model: str, clip: str) -> Path:
    """Locate the untouched benchmark output of one model for one clip.

    Prefers Mirelo-style ``*_var0`` videos, then any video, then audio files.
    """
    directory = bench_run_dir / "runs" / "20260825-165347" / "raw" / f"{model}__{clip}"
    candidates = sorted(directory.glob("*"))
    preferred = [p for p in candidates if "_var0" in p.name and p.suffix == ".mp4"]
    videos = [p for p in candidates if p.suffix == ".mp4"]
    audios = [p for p in candidates if p.suffix in {".wav", ".mp3", ".flac"}]
    chosen = preferred or videos or audios
    if not chosen:
        raise FileNotFoundError(f"no raw output for {model}__{clip} under {directory}")
    return chosen[0]


def clip_relative_seconds(source_seconds: float, clip: str) -> float | None:
    """Map a source-timeline timestamp onto the standardized clip window."""
    trim = CLIP_TRIMS.get(clip)
    if trim is None:
        return source_seconds
    start, end = trim
    if not (start <= source_seconds < end):
        return None
    return source_seconds - start


def extract_mono_wav(source: Path, destination: Path, sample_rate: int) -> None:
    run_ffmpeg([
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        str(destination),
    ])


class Ledger:
    """Append-only JSONL log; every line is flushed and fsynced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(path, "a", encoding="utf-8")

    def append(self, entry: dict[str, Any]) -> None:
        self._stream.write(json.dumps(entry, sort_keys=True) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def load_contacts(clip: str, auto_sfx_runs: Path) -> list[dict[str, Any]]:
    """Foot-contact ground truth for one clip, collapsed to impacts.

    Left/right contacts within 40 ms describe one audible impact; they are
    merged (earliest timestamp wins) so matching does not double-count.
    Returned times are CLIP-RELATIVE seconds.
    """
    run_name = CONTACT_RUNS.get(clip)
    if run_name is None:
        return []
    path = auto_sfx_runs / run_name / "events.json"
    payload = json.loads(path.read_text())
    raw_times = sorted(
        float(event["time_s"])
        for event in payload.get("events", [])
        if event.get("family") == "locomotion.foot_contact"
    )
    impacts: list[float] = []
    for time_s in raw_times:
        relative = clip_relative_seconds(time_s, clip)
        if relative is None:
            continue
        if impacts and relative - impacts[-1] < 0.040:
            continue
        impacts.append(relative)
    return [
        {"time_s": round(time_s, 4), "id": f"gt_{index:03d}"}
        for index, time_s in enumerate(impacts, start=1)
    ]
