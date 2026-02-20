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
"""

import hashlib
import json
import subprocess
from pathlib import Path


# FFmpeg voice filter used for clip export (voice-split).
VOICE_AF = (
    "highpass=f=80,lowpass=f=8000,"
    "afftdn=nf=-25,"
    "compand=attacks=0:decays=0.25:points=-90/-90|-35/-18|-10/-8|0/-6"
)


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
