"""
lib/audio.py — shared audio file helpers.

Provides:
  VOICE_AF           — ffmpeg filter chain for voice-export clips (voice-split)
  sha256_file        — short SHA-256 prefix for a file
  get_duration       — audio duration via ffprobe
  normalize_audio    — ffmpeg loudnorm, convert to mono WAV at sample_rate
  trim_audio_encode  — trim + re-encode to a target sample_rate (voice-clone)
  trim_audio_copy    — trim via stream-copy (no re-encode, for large files)
  extract_chunk      — extract a window and decode to 44.1 kHz stereo WAV
  load_audio_mono    — decode audio to mono float32 samples
  analyse_acoustics  — non-ML acoustic heuristics for quality scoring
  score_take_selection / rank_take_selection
                     — deterministic best-take policy helpers
"""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# FFmpeg voice filter used for clip export (voice-split).
VOICE_AF = (
    "highpass=f=80,lowpass=f=8000,"
    "afftdn=nf=-25,"
    "compand=attacks=0:decays=0.25:points=-90/-90|-35/-18|-10/-8|0/-6"
)

# Shared deterministic ranking weights for --select-best flows.
SELECTION_WEIGHTS = {
    "intelligibility": 0.55,
    "pacing_sanity": 0.25,
    "duration_fit": 0.20,
}


def sha256_file(path: Path, n_bytes: int = 16) -> str:
    """Return a short (n_bytes*2 hex chars) SHA-256 prefix of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()[: n_bytes * 2]


def get_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Decode *path* with soundfile and return ``(mono_float32, sample_rate)``.
    """
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _frame_rms_dbfs(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """
    Return per-frame RMS in dBFS for a mono sample vector.
    """
    if samples.size == 0:
        return np.array([-120.0], dtype=np.float32)
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))

    if samples.size < frame_len:
        pad = frame_len - samples.size
        samples = np.pad(samples, (0, pad), mode="constant")

    rms_vals: list[float] = []
    for start in range(0, samples.size - frame_len + 1, hop):
        frame = samples[start:start + frame_len]
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        rms_vals.append(rms)

    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    return np.asarray(20.0 * np.log10(rms_arr + 1e-12), dtype=np.float32)


def analyse_acoustics(path: Path) -> dict[str, float]:
    """
    Compute deterministic non-ML acoustic heuristics from an audio file.

    Returns normalized component scores (0..1) and raw measurements used by
    reference-candidate and best-take ranking logic.
    """
    samples, sr = load_audio_mono(path)
    if samples.size == 0:
        return {
            "peak": 0.0,
            "clipping_ratio": 1.0,
            "rms_dbfs": -120.0,
            "noise_floor_dbfs": -120.0,
            "dynamic_range_db": 0.0,
            "speech_ratio": 0.0,
            "speech_continuity": 0.0,
            "clipping_score": 0.0,
            "noise_score": 0.0,
            "speech_continuity_score": 0.0,
            "acoustic_quality_score": 0.0,
        }

    abs_samples = np.abs(samples)
    peak = float(abs_samples.max())
    clipping_ratio = float(np.mean(abs_samples >= 0.985))

    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))
    rms_dbfs = float(20.0 * math.log10(rms + 1e-12))

    frame_db = _frame_rms_dbfs(samples, sr)
    noise_floor_db = float(np.percentile(frame_db, 10))
    dynamic_range_db = float(np.percentile(frame_db, 90) - noise_floor_db)
    speech_threshold_db = noise_floor_db + 10.0
    speech_mask = frame_db > speech_threshold_db
    speech_ratio = float(np.mean(speech_mask)) if speech_mask.size else 0.0
    transitions = (
        float(np.mean(speech_mask[1:] != speech_mask[:-1]))
        if speech_mask.size > 1 else 0.0
    )
    speech_continuity = 1.0 - transitions

    clipping_score = _clamp01(1.0 - (clipping_ratio / 0.01))
    rms_score = _clamp01(1.0 - abs(rms_dbfs + 20.0) / 20.0)
    range_score = _clamp01((dynamic_range_db - 8.0) / 22.0)
    noise_score = 0.55 * rms_score + 0.45 * range_score

    speech_ratio_score = _clamp01(1.0 - abs(speech_ratio - 0.55) / 0.45)
    continuity_score = _clamp01((speech_continuity - 0.35) / 0.65)
    speech_continuity_score = 0.6 * speech_ratio_score + 0.4 * continuity_score

    acoustic_quality_score = (
        0.40 * clipping_score +
        0.30 * noise_score +
        0.30 * speech_continuity_score
    )

    return {
        "peak": round(peak, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "rms_dbfs": round(rms_dbfs, 3),
        "noise_floor_dbfs": round(noise_floor_db, 3),
        "dynamic_range_db": round(dynamic_range_db, 3),
        "speech_ratio": round(speech_ratio, 4),
        "speech_continuity": round(speech_continuity, 4),
        "clipping_score": round(clipping_score, 4),
        "noise_score": round(noise_score, 4),
        "speech_continuity_score": round(speech_continuity_score, 4),
        "acoustic_quality_score": round(acoustic_quality_score, 4),
    }


def estimate_text_duration(text: str, chars_per_sec: float = 13.0) -> float:
    """
    Estimate natural speech duration from text length.
    """
    return max(1.0, len(text.strip()) / chars_per_sec)


def score_duration_fit(duration_sec: float, target_sec: float) -> float:
    """
    Score how close *duration_sec* is to a target duration (0..1).
    """
    if duration_sec <= 0.0 or target_sec <= 0.0:
        return 0.0
    ratio = duration_sec / target_sec
    return _clamp01(1.0 - abs(ratio - 1.0) / 0.70)


def score_pacing_sanity(
    text: str,
    duration_sec: float,
    *,
    speech_ratio: float,
    speech_continuity: float,
) -> dict[str, float]:
    """
    Heuristic pacing score combining speaking rate and continuity.
    """
    words = re.findall(r"\w+", text)
    words_per_sec = (len(words) / duration_sec) if duration_sec > 0 else 0.0
    words_per_sec_score = _clamp01(1.0 - abs(words_per_sec - 2.6) / 1.8)
    speech_ratio_score = _clamp01(1.0 - abs(speech_ratio - 0.55) / 0.45)
    continuity_score = _clamp01((speech_continuity - 0.35) / 0.65)

    score = (
        0.55 * words_per_sec_score +
        0.25 * speech_ratio_score +
        0.20 * continuity_score
    )
    return {
        "words_per_sec": round(words_per_sec, 4),
        "words_per_sec_score": round(words_per_sec_score, 4),
        "speech_ratio_score": round(speech_ratio_score, 4),
        "continuity_score": round(continuity_score, 4),
        "score": round(score, 4),
    }


def score_take_selection(
    *,
    text: str,
    duration_sec: float,
    intelligibility: float,
    acoustics: dict[str, float],
) -> dict[str, float]:
    """
    Compute weighted ranking metrics for a generated take.
    """
    intelligibility = _clamp01(intelligibility)
    target_sec = estimate_text_duration(text)
    duration_fit = score_duration_fit(duration_sec, target_sec)
    pacing = score_pacing_sanity(
        text,
        duration_sec,
        speech_ratio=float(acoustics.get("speech_ratio", 0.0)),
        speech_continuity=float(acoustics.get("speech_continuity", 0.0)),
    )
    final_score = (
        SELECTION_WEIGHTS["intelligibility"] * intelligibility +
        SELECTION_WEIGHTS["pacing_sanity"] * pacing["score"] +
        SELECTION_WEIGHTS["duration_fit"] * duration_fit
    )
    return {
        "intelligibility": round(intelligibility, 4),
        "pacing_sanity": round(pacing["score"], 4),
        "duration_fit": round(duration_fit, 4),
        "expected_duration_sec": round(target_sec, 3),
        "duration_sec": round(duration_sec, 3),
        "final_score": round(final_score, 4),
    }


def rank_take_selection(takes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deterministically rank takes by the selection metrics.
    """
    def _key(t: dict[str, Any]) -> tuple[float, float, float, float, str]:
        metrics = t.get("selection_metrics") or {}
        return (
            -float(metrics.get("final_score", 0.0)),
            -float(metrics.get("intelligibility", 0.0)),
            -float(metrics.get("pacing_sanity", 0.0)),
            -float(metrics.get("duration_fit", 0.0)),
            str(t.get("take", "")),
        )

    ranked = sorted(takes, key=_key)
    for i, take in enumerate(ranked, start=1):
        take["selection_rank"] = i
    return ranked


def normalize_audio(src: Path, dest: Path, sample_rate: int = 16_000) -> None:
    """
    Convert *src* to mono WAV at *sample_rate*, applying loudness normalisation
    (EBU R128 via ffmpeg loudnorm filter).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-ac", "1", "-ar", str(sample_rate),
            "-af", "loudnorm",
            str(dest),
        ],
        check=True,
    )


def trim_audio_encode(
    src: Path,
    start: float,
    duration: float,
    dest: Path,
    sample_rate: int = 24_000,
) -> None:
    """
    Extract [start, start+duration] from *src*, re-encode to *sample_rate* mono WAV.

    24 kHz (the default) matches Qwen3-TTS internal sample rate for best quality.
    Use this for reference-segment clips fed to the TTS model.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
            str(dest),
        ],
        check=True,
    )


def trim_audio_copy(
    audio: Path,
    start_sec: float,
    end_sec: float,
    dest: Path,
) -> Path:
    """
    Trim audio to [start_sec, end_sec] via ffmpeg stream copy (no re-encode).

    Result is cached — re-runs are instant.
    Intended for large source files where re-encoding is wasteful (voice-split).
    """
    if dest.exists():
        print(f"  [cache hit] trimmed audio → {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = end_sec - start_sec
    print(f"  trimming {start_sec:.1f}s – {end_sec:.1f}s  ({dur:.1f}s)  → {dest.name}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(audio),
            "-t", f"{dur:.3f}",
            "-c", "copy",
            str(dest),
        ],
        check=True,
    )
    return dest


def extract_chunk(audio: Path, start: float, length: float, dest: Path) -> None:
    """
    Extract [start, start+length] from *audio*, decode to 44.1 kHz stereo WAV.

    Used by voice-split to produce windows for Demucs processing.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio),
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(dest),
        ],
        check=True,
    )
