#!/usr/bin/env python3
"""Stage 3 — index the owner's SFX bank for retrieval.

For every audio file under the bank root:

- normalized working copy (48 kHz mono float WAV) used later for placement
- duration, overall RMS, onset list (same SuperFlux detector as the oracle)
- CLAP audio embedding + zero-shot category labels (top-k with scores)

Outputs ``bank/index.json`` (metadata; embeddings referenced by row id) and
``bank/embeddings.npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from clap_embedder import CLAP_SR, embed_audio, label_embeddings
from common import (
    SFX_BANK_DIR,
    atomic_json,
    extract_mono_wav,
    ffprobe,
    sha256,
    utc_now,
)
from oracle_events import FRAME_S, JUMP_PRE_S, detect_onsets, superflux_envelope

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".ogg", ".opus"}

#: Zero-shot label set. Deliberately broad; retrieval uses audio-audio
#: similarity first and treats these labels as an interpretable prior.
DEFAULT_LABELS: tuple[str, ...] = (
    "footsteps on grass",
    "footsteps on gravel",
    "footsteps on dirt",
    "footsteps on wood",
    "footsteps on pavement",
    "a single footstep",
    "a door slam",
    "a door creak",
    "a water splash",
    "ocean waves surf",
    "a stream or river flowing",
    "rain falling",
    "a wind gust",
    "birds chirping",
    "cloth rustling",
    "leaves crunching",
    "a punch impact thud",
    "glass breaking",
    "a metal clang",
    "a car engine",
    "paper crumpling",
    "a camera shutter click",
    "a sword swing whoosh",
    "an explosion boom",
)

TOP_LABELS = 3


def probe_duration(path: Path) -> float:
    probe = ffprobe(path)
    duration = probe.get("format", {}).get("duration")
    if duration is None:
        for stream in probe.get("streams", []):
            if stream.get("duration"):
                duration = stream["duration"]
                break
    return float(duration)


def classify(embedding: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    matrix = label_embeddings(labels)
    scores = matrix @ embedding
    order = np.argsort(-scores)[:TOP_LABELS]
    return [
        {"label": labels[int(index)], "score": round(float(scores[index]), 4)}
        for index in order
    ]


def index_file(path: Path, bank_root: Path, work_dir: Path,
               labels: tuple[str, ...]) -> dict[str, Any]:
    digest = sha256(path)
    entry_id = f"{path.stem}_{digest[:8]}"
    wav_path = work_dir / "audio" / f"{entry_id}.wav"
    extract_mono_wav(path, wav_path, CLAP_SR)
    sample_rate, audio = wavfile.read(wav_path)
    assert sample_rate == CLAP_SR
    audio = audio.astype(np.float32)
    # The SuperFlux detector's geometry is defined at ANALYSIS_SR; analyse a
    # downsampled view while keeping the 48 kHz WAV for placement.
    analysis_audio = resample_poly(audio, 147, 320)
    envelope, _ = superflux_envelope(analysis_audio.astype(np.float64))
    onset_frames, effective_delta = detect_onsets(
        envelope, analysis_audio.astype(np.float64),
        relax_gate_before_s=JUMP_PRE_S + 0.05)

    rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(audio)))), 1e-9))

    embedding = embed_audio(audio)
    entry = {
        "id": entry_id,
        "rel_path": str(path.relative_to(bank_root)),
        "sha256": digest,
        "working_wav": str(wav_path),
        "duration_s": round(probe_duration(path), 3),
        "rms_db": round(rms_db, 2),
        "onsets_s": [round(frame * FRAME_S, 4) for frame in onset_frames],
        "detector_delta": effective_delta,
        "labels": classify(embedding, labels),
    }
    return entry, embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=SFX_BANK_DIR)
    parser.add_argument("--labels-file", type=Path, default=None,
                        help="one zero-shot label per line (overrides defaults)")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = DEFAULT_LABELS
    if args.labels_file:
        labels = tuple(line.strip() for line in args.labels_file.read_text().splitlines()
                       if line.strip())

    files = sorted(
        path for path in args.bank_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise SystemExit(f"no audio files under {args.bank_dir} — drop the template bank first")

    work_dir = args.run_dir / "bank"
    (work_dir / "audio").mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    index_path = work_dir / "index.json"
    if index_path.exists() and not args.force:
        existing = {
            entry["rel_path"]: entry
            for entry in json.loads(index_path.read_text())["entries"]
            if Path(entry.get("working_wav", "")).exists()
        }

    embeddings: dict[str, np.ndarray] = {}
    if not args.force and (work_dir / "embeddings.npz").exists():
        archive = np.load(work_dir / "embeddings.npz")
        embeddings = {key: archive[key] for key in archive.files}

    entries: list[dict[str, Any]] = []
    for path in files:
        rel = str(path.relative_to(args.bank_dir))
        if rel in existing:
            entries.append(existing[rel])
            print(f"{rel}: cached")
            continue
        entry, embedding = index_file(path, args.bank_dir, work_dir, labels)
        entries.append(entry)
        embeddings[entry["id"]] = embedding
        print(f"{rel}: indexed ({len(entry['onsets_s'])} onsets, top label "
              f"'{entry['labels'][0]['label']}')")

    # Deterministic row order; ids travel inside the npz.
    np.savez_compressed(
        work_dir / "embeddings.npz",
        **{entry["id"]: embeddings[entry["id"]] for entry in entries if entry["id"] in embeddings},
    )
    atomic_json(index_path, {
        "schema_version": 1,
        "generated_at": utc_now(),
        "bank_dir": str(args.bank_dir),
        "file_count": len(entries),
        "model": "laion/clap-htsat-fused",
        "labels": list(labels),
        "entries": entries,
    })
    print(f"indexed {len(entries)} files → {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
