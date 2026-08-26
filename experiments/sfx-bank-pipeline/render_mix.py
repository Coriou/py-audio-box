#!/usr/bin/env python3
"""Stage 5 — execute placement plans and render frame-locked mixes.

Per clip:

- materialize each planned event (bank WAV / SA3 variation / oracle crop),
  aligning the candidate's transient sample-exactly onto the event time
- local DSP: gain, tail trim, ≤1.35× tempo compression of the POST-transient
  region only (the attack itself is never stretched — that is the core rule
  of the tiered strategy)
- ``sa3`` intents are executed through the ledgered fal caller (capped)
- overlap-add into a 48 kHz bed, peak-guard, two-pass loudnorm (−16 LUFS /
  TP −1.5 / LRA 11 — identical constants to fal-sfx-bench) and lossless mux
  onto the ORIGINAL input frames

Outputs ``mix/<clip>.bed.wav``, ``review/<clip>.bank.mp4`` +
``review/<clip>.bank.loudness.json``. The untouched Mirelo review renders are
copied alongside as the A side of the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

BENCH_DIR = Path(__file__).resolve().parents[1] / "fal-sfx-bench"
sys.path.insert(0, str(BENCH_DIR))
from build_review import (  # noqa: E402  (bench conventions, reused verbatim)
    measure_loudness,
    mux_review,
    normalize_audio,
)

from common import (  # noqa: E402
    BENCH_RUN_DIR,
    CANONICAL_PROMPTS,
    Ledger,
    atomic_json,
    extract_mono_wav,
    run_ffmpeg,
    sha256,
    utc_now,
)
from fal_calls import run_sa3_a2a  # noqa: E402
from oracle_events import ANALYSIS_SR, detect_onsets, superflux_envelope  # noqa: E402

SR = 48_000
FADE_IN_S = 0.003
FADE_OUT_S = 0.008
BED_EDGE_FADE_S = 0.05
PEAK_GUARD = 10.0 ** (-1.0 / 20.0)     # −1 dBFS before loudnorm
TEMPO_MAX = 1.35
SNIPPET_PRE_S = 0.02                   # fallback crop lead-in
MAX_SA3_INTENTS = 4


def load_mono_48k(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 at 48 kHz."""
    probe_ext = path.suffix.lower()
    if probe_ext == ".wav":
        sample_rate, audio = wavfile.read(path)
        if sample_rate == SR and audio.ndim == 1 and audio.dtype == np.float32:
            return audio
    temporary = path.with_suffix(".render48.wav")
    extract_mono_wav(path, temporary, SR)
    sample_rate, audio = wavfile.read(temporary)
    temporary.unlink(missing_ok=True)
    assert sample_rate == SR
    return audio.astype(np.float32)


def first_onset_of(audio: np.ndarray) -> int:
    envelope, _ = superflux_envelope(audio.astype(np.float64))
    frames, _ = detect_onsets(envelope, audio.astype(np.float64))
    if not frames:
        return 0
    return int(frames[0] * 128 * SR // ANALYSIS_SR)


def apply_fades(segment: np.ndarray, fade_out_s: float = FADE_OUT_S) -> np.ndarray:
    result = segment.copy()
    fade_in = min(result.size, int(FADE_IN_S * SR))
    fade_out = min(result.size, int(fade_out_s * SR))
    if fade_in:
        result[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        result[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return result


def tempo_compress(segment: np.ndarray, factor: float, work_dir: Path,
                   tag: str) -> np.ndarray:
    """Shorten a segment by `factor` (≥1) using rubberband, pitch preserved."""
    if factor <= 1.001:
        return segment
    source = work_dir / f"{tag}.src.wav"
    target = work_dir / f"{tag}.tmp.wav"
    wavfile.write(source, SR, segment)
    run_ffmpeg([
        "-i", str(source),
        "-af", f"rubberband=tempo={factor:.6f}:pitch=1:transients=crisp",
        "-c:a", "pcm_f32le", "-ar", str(SR), "-ac", "1",
        str(target),
    ])
    _, compressed = wavfile.read(target)
    source.unlink()
    target.unlink()
    return compressed.astype(np.float32)


def materialize(event: dict[str, Any], plan: dict[str, Any], run_dir: Path,
                ledger: Ledger,
                allow_sa3: bool, sa3_state: dict[str, int]) -> tuple[np.ndarray, int]:
    """Return (audio at 48 kHz mono, transient onset index in samples)."""
    clip = plan["clip"]
    work_dir = run_dir / "mix" / "parts"
    work_dir.mkdir(parents=True, exist_ok=True)
    tier = event["tier"]
    gap_samples = int(event["available_gap_s"] * SR)
    tag = f"{clip}_{event['id']}"

    if tier == "mirelo-fallback":
        oracle_wav = run_dir / "oracle" / f"{clip}.{plan['oracle_model']}.wav"
        sample_rate, oracle = wavfile.read(oracle_wav)
        start_s = max(0.0, event["time_s"] - SNIPPET_PRE_S)
        start = int(start_s * sample_rate)
        stop = min(oracle.size, start + int((event["available_gap_s"] + SNIPPET_PRE_S) * sample_rate))
        segment = oracle[max(0, start):stop].astype(np.float32)
        segment = resample_poly(segment, SR, sample_rate).astype(np.float32)
        return apply_fades(segment), int(SNIPPET_PRE_S * SR)

    if tier == "sa3":
        if not allow_sa3 or sa3_state["used"] >= MAX_SA3_INTENTS:
            raise RuntimeError("sa3 tier requested but budget exhausted")
        meta = event["_candidate_meta"]
        prompt = f"{CANONICAL_PROMPTS[clip]}; {event['label']}"
        duration = int(min(30, max(2, np.ceil(event["available_gap_s"] + 0.25))))
        case_name = f"sa3__{clip}__{event['id']}"
        record = run_sa3_a2a(
            run_dir, ledger, case_name, Path(meta["working_wav"]),
            prompt, duration,
        )
        sa3_state["used"] += 1
        generated = load_mono_48k(Path(record["primary_output"]))
        onset_idx = first_onset_of(generated)
        tail = generated[onset_idx:onset_idx + gap_samples]
        return apply_fades(np.concatenate([generated[:onset_idx], tail])), onset_idx

    # direct / dsp tiers: the bank's normalized 48 kHz copy
    candidate = load_mono_48k(Path(event["_candidate_meta"]["working_wav"]))
    onset_idx = int(round(float(event["first_onset_s"]) * SR))
    onset_idx = min(onset_idx, candidate.size)
    head = candidate[:onset_idx]
    tail = candidate[onset_idx:]

    ops = event.get("ops", [])
    if "tempo-compress" in ops and tail.size > gap_samples:
        factor = min(TEMPO_MAX, tail.size / max(gap_samples, 1))
        tail = tempo_compress(tail, factor, work_dir, tag)
    if len(tail) > gap_samples:
        tail = tail[:gap_samples]

    placed = np.concatenate([head, tail])
    if "gain" in ops:
        linear = 10.0 ** (float(event["gain_db"]) / 20.0)
        placed = placed * linear
    return apply_fades(placed).astype(np.float32), onset_idx


def render_bed(plan: dict[str, Any], run_dir: Path, ledger: Ledger,
               allow_sa3: bool, sa3_state: dict[str, int],
               stems_dir: Path | None = None) -> Path:
    clip_duration = int(plan["clip_duration_s"] * SR)
    bed = np.zeros(clip_duration + int(1.0 * SR), dtype=np.float32)

    def stem(name: str, audio: np.ndarray) -> None:
        if stems_dir is None:
            return
        target = stems_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(target, SR, audio.astype(np.float32))

    for event in plan["events"]:
        if event.get("kind") == "ambience-bed":
            meta = event["_candidate_meta"]
            candidate = load_mono_48k(Path(meta["working_wav"]))
            repeats = int(np.ceil(bed.size / max(candidate.size, 1)))
            filled = np.tile(candidate, repeats)[:bed.size]
            edge = int(BED_EDGE_FADE_S * SR)
            filled *= np.concatenate([
                np.linspace(0.0, 1.0, edge, dtype=np.float32),
                np.ones(bed.size - 2 * edge, dtype=np.float32),
                np.linspace(1.0, 0.0, edge, dtype=np.float32),
            ])
            linear = 10.0 ** (float(event.get("gain_db") or 0.0) / 20.0)
            contribution = (filled * linear).astype(np.float32)
            stem(f"{plan['clip']}/ambience.wav", contribution)
            bed += contribution
            continue

        placed, onset_idx = materialize(event, plan, run_dir, ledger,
                                        allow_sa3, sa3_state)
        anchor = int(round(float(event["time_s"]) * SR)) - onset_idx
        start = max(0, anchor)
        stop = min(bed.size, anchor + placed.size)
        if stop > start:
            bed[start:stop] += placed[start - anchor:stop - anchor]
        stem(f"{plan['clip']}/evt_{event['id']}.wav", placed)

    peak = float(np.max(np.abs(bed))) if bed.size else 0.0
    if peak > PEAK_GUARD:
        bed *= PEAK_GUARD / peak
    out_path = run_dir / "mix" / f"{plan['clip']}.bed.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_path, SR, bed)
    return out_path


def attach_candidate_meta(run_dir: Path, plan: dict[str, Any]) -> None:
    bank_entries = {
        entry["id"]: entry
        for entry in json.loads((run_dir / "bank" / "index.json").read_text())["entries"]
    }
    for event in plan["events"]:
        chosen = event.get("candidate")
        if chosen and chosen["entry_id"] in bank_entries:
            event["_candidate_meta"] = bank_entries[chosen["entry_id"]]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clips", default="walk1,walk2,man,ocean-long,ocean-short")
    parser.add_argument("--no-sa3", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = [part.strip() for part in args.clips.split(",") if part.strip()]
    inputs_index = {
        entry["key"]: entry
        for entry in json.loads(
            (BENCH_RUN_DIR / "inputs" / "inputs-index.json").read_text())["clips"]
    }
    review_dir = args.run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.run_dir / "runs" / "ledger.jsonl")
    sa3_state = {"used": 0}
    failures: list[str] = []

    try:
        for clip in clips:
            plan_path = args.run_dir / "plan" / f"{clip}.placement.json"
            if not plan_path.exists():
                print(f"{clip}: no placement plan — skipping", flush=True)
                continue
            try:
                plan = json.loads(plan_path.read_text())
                attach_candidate_meta(args.run_dir, plan)
                bed_path = render_bed(plan, args.run_dir, ledger, not args.no_sa3,
                                      sa3_state, stems_dir=args.run_dir / "stems")
                measured = measure_loudness(bed_path)
                norm_path = review_dir / f"{clip}.bank.norm.m4a"
                linear = normalize_audio(bed_path, norm_path, measured)
                # Pass-1 has no "out" fields; measure the normalized artifact
                # itself so the page can show achieved loudness.
                final = measure_loudness(norm_path)
                output_mp4 = review_dir / f"{clip}.bank.mp4"
                mux_review(inputs_index[clip]["output_path"], norm_path, output_mp4)
                norm_path.unlink(missing_ok=True)

                # B-side reference: the benchmark's Mirelo review render (A side
                # of every comparison page), copied untouched.
                bench_review = (BENCH_RUN_DIR / "runs" / "20260825-165347" / "review"
                                / f"mirelo16__{clip}.mp4")
                if bench_review.exists():
                    ab_copy = review_dir / f"{clip}.mirelo16.mp4"
                    ab_copy.write_bytes(bench_review.read_bytes())

                atomic_json(review_dir / f"{clip}.bank.loudness.json", {
                    "clip": clip,
                    "generated_at": utc_now(),
                    "bed_sha256": sha256(bed_path),
                    "linear": linear,
                    "input_i": measured.get("input_i"),
                    "input_tp": measured.get("input_tp"),
                    "input_lra": measured.get("input_lra"),
                    "target_i": measured.get("target_i"),
                    "output_i": final.get("input_i"),
                    "output_tp": final.get("input_tp"),
                    "output_lra": final.get("input_lra"),
                })
                print(f"{clip}: rendered → {output_mp4.name} "
                      f"(SA3 calls so far: {sa3_state['used']})", flush=True)
            except Exception as error:  # noqa: BLE001 — per-clip isolation
                failures.append(f"{clip}: {error}")
                print(f"{clip}: FAILED — {error}", flush=True)
    finally:
        ledger.close()

    if failures:
        for line in failures:
            print(f"FAILURE {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
