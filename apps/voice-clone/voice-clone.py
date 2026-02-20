#!/usr/bin/env python3
"""
voice-clone.py — Clone a voice from a reference WAV and synthesise new speech.

Pipeline (all stages cached by content hash):
  1. Normalise reference audio     (ffmpeg → 16 kHz mono, loudnorm)
  2. Select best clean segment     (Silero VAD → top-3 candidates → whisper-scored)
  3. Transcribe reference segment  (faster-whisper, int8/CPU)
  4. Build voice-clone prompt      (Qwen3-TTS, pickled for reuse)
  5. Synthesise speech             (Qwen3-TTS generate_voice_clone)
  6. Write output WAV + meta.json

Sub-commands:
  prepare-ref   — Run stages 1–3 only; print cached clip + transcript path
  synth         — Full pipeline; requires --ref-audio and --text / --text-file
  self-test     — End-to-end smoke test using a bundled public-domain clip

Usage:
  ./run voice-clone synth --ref-audio /work/myvoice.wav --text "Hello, world"
  ./run voice-clone synth --ref-audio /work/myvoice.wav \\
        --text "Hello" --language French --style serious_doc
  ./run voice-clone prepare-ref --ref-audio /work/myvoice.wav
  ./run voice-clone self-test
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_TTS_MODEL    = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_WHISPER      = "small"
WHISPER_COMPUTE      = "int8"
MIN_REF_SECONDS      = 3.0
MAX_REF_SECONDS      = 12.0
SWEET_SPOT_SECONDS   = 8.5     # ideal duration for ref clip
CANDIDATE_COUNT      = 3       # top-N VAD candidates to whisper-score

# Transcripts below this avg_logprob may hurt cloning quality
QUALITY_GATE_LOGPROB = -1.2

# Bump to invalidate all cached prompt pickles (e.g. after model-format change)
PROMPT_SCHEMA_VERSION = 1

# All languages the Qwen3-TTS Base model supports
QWEN3_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# ISO 639-1 code → Qwen3 language name (for langid-based auto-detection)
_LANGID_TO_QWEN: dict[str, str] = {
    "zh": "Chinese",  "en": "English",    "ja": "Japanese",
    "ko": "Korean",   "de": "German",     "fr": "French",
    "ru": "Russian",  "pt": "Portuguese", "es": "Spanish",
    "it": "Italian",
}

# Public-domain demo clip used by the Qwen3-TTS team — reused for self-test
SELF_TEST_REF_URL  = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
SELF_TEST_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
SELF_TEST_SYNTH_TEXT = "The quick brown fox jumps over the lazy dog."


# ── tiny timing helper ─────────────────────────────────────────────────────────

class _Timer:
    """Context manager that prints elapsed time on exit."""

    def __init__(self, label: str) -> None:
        self._label = label
        self.elapsed = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
        print(f"    ↳ {self._label}: {self.elapsed:.2f}s")


# ── audio helpers ──────────────────────────────────────────────────────────────

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


def trim_audio(src: Path, start: float, duration: float, dest: Path,
               sample_rate: int = 24_000) -> None:
    """
    Extract [start, start+duration] from *src*, re-encode to 24 kHz mono WAV.
    24 kHz matches Qwen3-TTS internal sample rate for best quality.
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


# ── Silero VAD ─────────────────────────────────────────────────────────────────

def _load_silero(hub_dir: Path):
    torch.hub.set_dir(str(hub_dir))
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    (get_speech_timestamps, *_) = utils
    return model, get_speech_timestamps


def _run_silero(wav_path: Path, model, get_ts) -> list[tuple[float, float]]:
    """Return merged speech segments as (start_sec, end_sec) pairs."""
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    TARGET_SR = 16_000
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR

    timestamps = get_ts(waveform.squeeze(0), model, sampling_rate=sr)

    MERGE_GAP = 0.20   # merge segments separated by < 200 ms
    MIN_LEN   = 0.30   # drop segments shorter than 300 ms

    segs: list[tuple[float, float]] = []
    for ts in timestamps:
        s = ts["start"] / sr
        e = ts["end"]   / sr
        if e - s < MIN_LEN:
            continue
        if segs and s - segs[-1][1] <= MERGE_GAP:
            segs[-1] = (segs[-1][0], max(segs[-1][1], e))
        else:
            segs.append((s, e))
    return segs


def _pick_top_candidates(
    wav_path: Path,
    model,
    get_ts,
    min_len: float = MIN_REF_SECONDS,
    max_len: float = MAX_REF_SECONDS,
    n: int = CANDIDATE_COUNT,
) -> list[tuple[float, float]]:
    """
    Run Silero VAD and return up to *n* candidate segments in [min_len, max_len],
    ordered by proximity to SWEET_SPOT_SECONDS then descending duration.
    Falls back to a file-start clip if nothing qualifies.
    """
    segs = _run_silero(wav_path, model, get_ts)
    file_dur = get_duration(wav_path)

    candidates: list[tuple[float, float, float]] = []  # (start, end, dur_score)
    for seg_s, seg_e in segs:
        clip_end = min(seg_s + max_len, seg_e, file_dur)
        clip_len = clip_end - seg_s
        if clip_len < min_len:
            continue
        dur_score = 1.0 - abs(clip_len - SWEET_SPOT_SECONDS) / max_len
        candidates.append((seg_s, clip_end, dur_score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    result = [(s, e) for s, e, _ in candidates[:n]]
    if not result and file_dur >= min_len:
        result = [(0.0, min(file_dur, max_len))]
    return result


def _score_segment(duration: float, avg_logprob: float) -> float:
    """
    Combined quality score for a candidate segment.

    Weights:
      60%  transcription confidence  (avg_logprob normalised from [-2, 0] → [0, 1])
      40%  duration proximity        (1.0 at SWEET_SPOT_SECONDS, 0.0 at MAX)
    """
    lp_norm  = max(0.0, min(1.0, (avg_logprob + 2.0) / 2.0))
    dur_norm = max(0.0, 1.0 - abs(duration - SWEET_SPOT_SECONDS) / MAX_REF_SECONDS)
    return 0.60 * lp_norm + 0.40 * dur_norm


# ── faster-whisper transcription ───────────────────────────────────────────────

def _build_whisper_model(whisper_model: str, num_threads: int):
    from faster_whisper import WhisperModel
    return WhisperModel(
        whisper_model,
        device="cpu",
        compute_type=WHISPER_COMPUTE,
        cpu_threads=num_threads,
    )


def transcribe_segment(
    wav_path: Path,
    wm,
    beam_size: int = 5,
) -> tuple[str, float, str, float]:
    """
    Transcribe *wav_path* with the pre-loaded faster-whisper model.
    Returns (transcript, avg_logprob, detected_language_qwen, language_probability).
    """
    segments, info = wm.transcribe(
        str(wav_path),
        beam_size=beam_size,
        vad_filter=True,
    )

    texts: list[str] = []
    logprobs: list[float] = []
    for seg in segments:
        texts.append(seg.text.strip())
        logprobs.append(seg.avg_logprob)

    transcript  = " ".join(texts).strip()
    avg_logprob = float(np.mean(logprobs)) if logprobs else -2.0

    detected_iso  = info.language or "en"
    detected_qwen = _LANGID_TO_QWEN.get(detected_iso, "English")
    lang_prob     = float(info.language_probability)

    return transcript, avg_logprob, detected_qwen, lang_prob


def transcribe_ref(
    wav_path: Path,
    whisper_model: str,
    num_threads: int,
) -> tuple[str, float, str, float]:
    """
    Load faster-whisper and transcribe the reference clip at full quality.
    Returns (transcript, avg_logprob, detected_language_qwen, language_probability).
    """
    print(
        f"  loading faster-whisper/{whisper_model} "
        f"(int8, cpu, {num_threads} threads)..."
    )
    wm = _build_whisper_model(whisper_model, num_threads)
    transcript, avg_logprob, detected_qwen, lang_prob = transcribe_segment(
        wav_path, wm, beam_size=5
    )
    print(
        f"  [{detected_qwen} p={lang_prob:.2f}] "
        f"conf={avg_logprob:.2f}  {transcript!r}"
    )
    return transcript, avg_logprob, detected_qwen, lang_prob


# ── language helpers ───────────────────────────────────────────────────────────

def detect_language_from_text(text: str) -> str:
    """
    Auto-detect the language of *text* using langid.
    Returns a Qwen3 language name, or "English" if detection fails.
    """
    try:
        import langid  # type: ignore
        iso, _conf = langid.classify(text)
        return _LANGID_TO_QWEN.get(iso, "English")
    except Exception:
        return "English"


def resolve_language(
    language_flag: str,
    ref_language: str,
    text: str,
) -> str:
    """
    Resolve the synthesis language.

    Rules (in order):
      1. If --language is not "Auto", use it directly.
      2. If text has ≥3 words, run langid detection; use that if supported.
      3. Fall back to the ref language detected by whisper.
      4. Default to "English".
    """
    if language_flag != "Auto":
        return language_flag

    if len(text.split()) >= 3:
        detected = detect_language_from_text(text)
        if detected in QWEN3_LANGUAGES and detected != "Auto":
            return detected

    if ref_language and ref_language not in ("Auto", ""):
        return ref_language

    return "English"


def validate_language(language: str, model) -> None:
    """
    Verify *language* against the model's supported language list.
    Raises SystemExit with a helpful message if unsupported.
    """
    if language == "Auto":
        return
    try:
        supported = model.get_supported_languages()
        if supported and language not in supported:
            print(
                f"ERROR: language '{language}' is not supported by the loaded model.\n"
                f"  Supported: {', '.join(sorted(supported))}",
                file=sys.stderr,
            )
            sys.exit(1)
    except AttributeError:
        pass  # older qwen-tts versions may not expose this method


# ── style presets ──────────────────────────────────────────────────────────────

def load_style_presets(styles_path: Path) -> dict[str, dict[str, str]]:
    """Load YAML style presets from *styles_path*; return {} if missing/broken."""
    if not styles_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(styles_path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"  WARNING: could not load styles.yaml: {exc}", file=sys.stderr)
        return {}


def apply_style(
    text: str,
    style_name: str | None,
    styles_path: Path,
    prompt_prefix: str | None,
    prompt_suffix: str | None,
) -> str:
    """
    Compose the final synthesis text:
      style_prefix + prompt_prefix + text + prompt_suffix + style_suffix

    Both style and manual prefix/suffix are optional.
    """
    style_prefix = style_suffix = ""
    if style_name:
        presets = load_style_presets(styles_path)
        if style_name not in presets:
            available = ", ".join(sorted(presets)) or "(none)"
            print(
                f"  WARNING: style '{style_name}' not found. "
                f"Available: {available}",
                file=sys.stderr,
            )
        else:
            style_prefix = presets[style_name].get("prefix", "")
            style_suffix = presets[style_name].get("suffix", "")

    parts = []
    if style_prefix:
        parts.append(style_prefix.rstrip())
    if prompt_prefix:
        parts.append(prompt_prefix.strip())
    parts.append(text)
    if prompt_suffix:
        parts.append(prompt_suffix.strip())
    if style_suffix:
        parts.append(style_suffix.strip())

    return " ".join(p for p in parts if p)


# ── pipeline state ─────────────────────────────────────────────────────────────

@dataclass
class RefResult:
    ref_hash:          str
    ref_dir:           Path
    normalized:        Path       # 16 kHz mono loudnorm WAV
    segment:           Path       # 24 kHz mono trimmed clip (ready for Qwen3)
    seg_start:         float
    seg_end:           float
    transcript:        str
    transcript_conf:   float
    whisper_model:     str
    ref_language:      str        # Qwen3-format language name detected by whisper
    ref_language_prob: float      # whisper language detection probability


# ── candidate scoring (stage 2 helper) ────────────────────────────────────────

def _score_and_select_best_candidate(
    candidates: list[tuple[float, float]],
    normalized: Path,
    ref_dir: Path,
    whisper_model: str,
    num_threads: int,
    force: bool,
) -> tuple[tuple[float, float], str, float, str, float]:
    """
    Score each candidate segment via fast whisper (beam_size=2), pick the best.

    Returns (best_seg, transcript, avg_logprob, detected_language, lang_prob).
    Caches individual candidate clips under ref_dir/candidates/.
    Writes ref_dir/best_segment.json for future cache hits.
    """
    cand_dir = ref_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    best_json = ref_dir / "best_segment.json"
    if best_json.exists() and not force:
        with open(best_json) as fh:
            d = json.load(fh)
        print(
            f"  [cache hit] best segment: "
            f"{d['seg_start']:.2f}s – {d['seg_end']:.2f}s  "
            f"(score={d.get('score', '?'):.3f})"
        )
        return (
            (d["seg_start"], d["seg_end"]),
            d.get("transcript", ""),
            d.get("avg_logprob", 0.0),
            d.get("ref_language", "English"),
            d.get("ref_language_prob", 1.0),
        )

    if len(candidates) == 1:
        # Single candidate — skip scoring; full transcription handled in stage 3
        s, e = candidates[0]
        return (s, e), "", 0.0, "English", 1.0

    print(f"  scoring {len(candidates)} candidate segment(s) with whisper …")
    wm = _build_whisper_model(whisper_model, num_threads)

    scored: list[dict] = []
    for i, (seg_s, seg_e) in enumerate(candidates):
        wav_path = cand_dir / f"candidate_{i:02d}.wav"
        if not wav_path.exists() or force:
            trim_audio(normalized, seg_s, seg_e - seg_s, wav_path)

        t, lp, lang, lp_prob = transcribe_segment(wav_path, wm, beam_size=2)
        dur   = seg_e - seg_s
        score = _score_segment(dur, lp)
        scored.append({
            "index":             i,
            "seg_start":         seg_s,
            "seg_end":           seg_e,
            "duration":          round(dur, 3),
            "transcript":        t,
            "avg_logprob":       round(lp, 4),
            "ref_language":      lang,
            "ref_language_prob": round(lp_prob, 4),
            "score":             round(score, 4),
        })
        print(
            f"    candidate {i}: {seg_s:.1f}s–{seg_e:.1f}s  "
            f"[{lang} p={lp_prob:.2f}]  lp={lp:.2f}  score={score:.3f}"
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    print(
        f"  ✓ best candidate: {best['seg_start']:.2f}s – {best['seg_end']:.2f}s  "
        f"score={best['score']:.3f}"
    )

    # Persist all scores for debugging
    with open(cand_dir / "scores.json", "w") as fh:
        json.dump({"candidates": scored, "selected_index": best["index"]}, fh, indent=2)

    with open(best_json, "w") as fh:
        json.dump(
            {
                "seg_start":         best["seg_start"],
                "seg_end":           best["seg_end"],
                "score":             best["score"],
                "transcript":        best["transcript"],
                "avg_logprob":       best["avg_logprob"],
                "ref_language":      best["ref_language"],
                "ref_language_prob": best["ref_language_prob"],
                "scored_at":         datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )

    return (
        (best["seg_start"], best["seg_end"]),
        best["transcript"],
        best["avg_logprob"],
        best["ref_language"],
        best["ref_language_prob"],
    )


# ── Stage 1–3: prepare-ref ─────────────────────────────────────────────────────

def prepare_ref(
    ref_audio:     Path,
    cache:         Path,
    whisper_model: str,
    num_threads:   int,
    ref_start:     float | None,
    ref_end:       float | None,
    ref_language:  str,
    x_vector_only: bool,
    force:         bool,
    force_bad_ref: bool = False,
) -> RefResult:
    """
    Run stages 1–3 of the pipeline (normalise → VAD trim → transcribe).
    All outputs are cached under /cache/voice-clone/refs/<hash>/.
    """

    # ── 1. Normalise ───────────────────────────────────────────────────────────
    print("\n==> [1/3] normalise reference audio")
    ref_hash = sha256_file(ref_audio)
    ref_dir  = cache / "voice-clone" / "refs" / ref_hash
    ref_dir.mkdir(parents=True, exist_ok=True)

    normalized = ref_dir / "ref_normalized.wav"
    if normalized.exists() and not force:
        print(f"  [cache hit] normalized → {normalized}")
    else:
        print(f"  {ref_audio.name}  →  16 kHz mono loudnorm …")
        with _Timer("ffmpeg normalize"):
            normalize_audio(ref_audio, normalized)

    # ── 2. Select best segment ─────────────────────────────────────────────────
    print("\n==> [2/3] select reference segment")

    # These are set below via one of three paths
    seg_start = seg_end = 0.0
    segment_transcript   = ""
    segment_logprob      = 0.0
    segment_lang         = "English"
    segment_lang_prob    = 1.0
    used_scoring_pass    = False

    if ref_start is not None and ref_end is not None:
        seg_start, seg_end = ref_start, ref_end
        print(f"  using manual bounds: {seg_start:.2f}s – {seg_end:.2f}s")

    else:
        # Honour either the new best_segment.json or the old vad_best_segment.json
        legacy_json = ref_dir / "vad_best_segment.json"
        best_json   = ref_dir / "best_segment.json"

        if (best_json.exists() or legacy_json.exists()) and not force:
            src = best_json if best_json.exists() else legacy_json
            with open(src) as fh:
                d = json.load(fh)
            seg_start, seg_end      = d["seg_start"], d["seg_end"]
            segment_transcript      = d.get("transcript", "")
            segment_logprob         = d.get("avg_logprob", 0.0)
            segment_lang            = d.get("ref_language", "English")
            segment_lang_prob       = d.get("ref_language_prob", 1.0)
            print(
                f"  [cache hit] segment: {seg_start:.2f}s – {seg_end:.2f}s  "
                f"({seg_end - seg_start:.1f}s)"
            )

        else:
            print("  loading Silero VAD …")
            hub_dir = cache / "torch" / "hub"
            vad_model, get_ts = _load_silero(hub_dir)
            with _Timer("Silero VAD"):
                candidates = _pick_top_candidates(normalized, vad_model, get_ts)

            if not candidates:
                dur = get_duration(normalized)
                candidates = [(0.0, min(dur, MAX_REF_SECONDS))]
                print(
                    f"  WARNING: no clean segment found — "
                    f"using first {candidates[0][1]:.1f}s"
                )
            else:
                print(
                    f"  {len(candidates)} candidate(s): "
                    + ", ".join(f"{s:.1f}s–{e:.1f}s" for s, e in candidates)
                )

            with _Timer("candidate scoring"):
                (seg_start, seg_end), segment_transcript, segment_logprob, \
                    segment_lang, segment_lang_prob = \
                    _score_and_select_best_candidate(
                        candidates, normalized, ref_dir,
                        whisper_model, num_threads, force,
                    )

            # Note: if >1 candidate, scoring already trimmed the wav; reuse it
            used_scoring_pass = len(candidates) > 1
            print(
                f"  best segment: {seg_start:.2f}s – {seg_end:.2f}s  "
                f"({seg_end - seg_start:.1f}s)"
            )

    segment_wav = ref_dir / "ref_segment.wav"
    if segment_wav.exists() and not force:
        print(f"  [cache hit] segment wav → {segment_wav}")
    else:
        print("  trimming segment …")
        with _Timer("ffmpeg trim"):
            trim_audio(normalized, seg_start, seg_end - seg_start, segment_wav)

    # ── 3. Transcribe ──────────────────────────────────────────────────────────
    print("\n==> [3/3] transcribe reference segment")

    transcript_json = ref_dir / "ref_transcript.json"

    if x_vector_only:
        print("  --x-vector-only: skipping transcription (quality may be lower)")
        transcript    = ""
        conf          = 0.0
        detected_lang = "English"
        lang_prob     = 1.0

    elif transcript_json.exists() and not force:
        with open(transcript_json) as fh:
            td = json.load(fh)
        transcript    = td["transcript"]
        conf          = td["avg_logprob"]
        whisper_model = td.get("whisper_model", whisper_model)
        detected_lang = td.get("ref_language_detected", "English")
        lang_prob     = td.get("ref_language_probability", 1.0)
        print(
            f"  [cache hit] ({whisper_model}) "
            f"[{detected_lang} p={lang_prob:.2f}]  {transcript!r}"
        )

    else:
        # If candidate scoring already gave us a transcript (from fast pass),
        # upgrade to a full beam=5 pass for the final file.
        with _Timer("faster-whisper"):
            transcript, conf, detected_lang, lang_prob = transcribe_ref(
                segment_wav, whisper_model, num_threads,
            )

        # ── quality gate ───────────────────────────────────────────────────────
        if conf < QUALITY_GATE_LOGPROB and not force_bad_ref:
            print(
                f"\n  ⚠  LOW TRANSCRIPT CONFIDENCE: avg_logprob={conf:.2f} "
                f"(threshold {QUALITY_GATE_LOGPROB}).\n"
                f"  This may result in poor cloning quality. Try:\n"
                f"    • --ref-start / --ref-end to select a cleaner segment\n"
                f"    • a higher-quality reference recording\n"
                f"    • --force-bad-ref to proceed anyway",
                file=sys.stderr,
            )
            sys.exit(1)
        elif conf < QUALITY_GATE_LOGPROB:
            print(
                f"  WARNING: low transcript confidence ({conf:.2f}) — "
                f"proceeding due to --force-bad-ref",
                file=sys.stderr,
            )

        if not transcript:
            print(
                "  WARNING: empty transcript — cloning quality may be reduced.",
                file=sys.stderr,
            )

        with open(transcript_json, "w") as fh:
            json.dump(
                {
                    "transcript":               transcript,
                    "avg_logprob":              conf,
                    "whisper_model":            whisper_model,
                    "ref_language_detected":    detected_lang,
                    "ref_language_probability": lang_prob,
                    "timestamp":                datetime.now(timezone.utc).isoformat(),
                },
                fh, indent=2,
            )

    # Persist lightweight pipeline state
    state_path = ref_dir / "pipeline_state.json"
    with open(state_path, "w") as fh:
        json.dump(
            {
                "ref_hash":                ref_hash,
                "ref_audio":               str(ref_audio),
                "seg_start":               seg_start,
                "seg_end":                 seg_end,
                "transcript":              transcript,
                "ref_language_detected":   detected_lang,
                "ref_language_probability": lang_prob,
                "x_vector_only":           x_vector_only,
                "updated_at":              datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )

    return RefResult(
        ref_hash=ref_hash,
        ref_dir=ref_dir,
        normalized=normalized,
        segment=segment_wav,
        seg_start=seg_start,
        seg_end=seg_end,
        transcript=transcript,
        transcript_conf=conf,
        whisper_model=whisper_model,
        ref_language=detected_lang,
        ref_language_prob=lang_prob,
    )


# ── Stage 4: build / load voice-clone prompt ───────────────────────────────────

def build_voice_clone_prompt(
    model,
    ref: RefResult,
    cache: Path,
    x_vector_only: bool,
    force: bool,
):
    """
    Call model.create_voice_clone_prompt once and pickle the result.
    A companion .meta.json is written so voice-synth list-voices can
    display info without unpickling.

    The filename encodes PROMPT_SCHEMA_VERSION so any format change
    automatically invalidates stale pickles.
    """
    mode      = "xvec" if x_vector_only else "full"
    model_tag = getattr(model, "name_or_path", "qwen3tts").replace("/", "_")
    stem      = f"{ref.ref_hash}_{model_tag}_{mode}_v{PROMPT_SCHEMA_VERSION}"

    prompts_dir = cache / "voice-clone" / "prompts"
    prompt_path = prompts_dir / f"{stem}.pkl"
    meta_path   = prompts_dir / f"{stem}.meta.json"

    if prompt_path.exists() and not force:
        print(f"  [cache hit] voice prompt → {prompt_path.name}")
        with open(prompt_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  building voice-clone prompt (mode={mode}) …")
    with _Timer("create_voice_clone_prompt"):
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref.segment),
            ref_text=ref.transcript if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )

    prompts_dir.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "wb") as fh:
        pickle.dump(prompt, fh)

    with open(meta_path, "w") as fh:
        json.dump(
            {
                "prompt_id":               stem,
                "schema_version":          PROMPT_SCHEMA_VERSION,
                "model":                   model_tag,
                "mode":                    mode,
                "ref_hash":                ref.ref_hash,
                "transcript":              ref.transcript,
                "ref_language_detected":   ref.ref_language,
                "ref_language_probability": round(ref.ref_language_prob, 4),
                "seg_start":               ref.seg_start,
                "seg_end":                 ref.seg_end,
                "segment_duration_sec":    round(ref.seg_end - ref.seg_start, 3),
                "created_at":              datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )

    print(f"  cached → {prompt_path.name}")
    return prompt


# ── Stage 5: synthesis ─────────────────────────────────────────────────────────

def load_tts_model(model_name: str, num_threads: int):
    """Load Qwen3-TTS-Base for CPU inference (float32)."""
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(num_threads)
    print(f"  loading {model_name} (cpu, float32, threads={num_threads}) …")
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cpu",
        dtype=torch.float32,
    )
    return model


def synthesise(
    text: str,
    language: str,
    model,
    prompt,
    seed: int | None,
    gen_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Generate speech.

    Extra Transformers model.generate kwargs (temperature, top_p,
    repetition_penalty, max_new_tokens, …) can be passed via gen_kwargs.
    """
    if seed is not None:
        torch.manual_seed(seed)
    kw = gen_kwargs or {}
    with _Timer("generate_voice_clone"):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            **kw,
        )
    return wavs[0], sr


# ── Stage 6: output ────────────────────────────────────────────────────────────

def write_output(
    wav: np.ndarray,
    sr: int,
    text: str,
    ref: RefResult,
    model_name: str,
    language: str,
    ref_language: str,
    x_vector_only: bool,
    seed: int | None,
    style: str | None,
    gen_kwargs: dict[str, Any] | None,
    out_dir: Path,
    timings: dict[str, float],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path  = out_dir / f"voice_clone_{ts}.wav"
    meta_path = out_dir / f"voice_clone_{ts}.meta.json"

    sf.write(str(wav_path), wav, sr)

    duration  = len(wav) / sr
    synth_sec = timings.get("synth_sec", 0.0)
    rtf       = synth_sec / duration if duration > 0 else 0.0

    meta: dict = {
        "app":                      "voice-clone",
        "created_at":               datetime.now(timezone.utc).isoformat(),
        "model":                    model_name,
        "language":                 language,
        "ref_language_detected":    ref_language,
        "ref_language_probability": round(ref.ref_language_prob, 4),
        "x_vector_only":            x_vector_only,
        "seed":                     seed,
        "style":                    style,
        "generation_kwargs":        gen_kwargs or {},
        "text":                     text,
        "output": {
            "wav":          str(wav_path),
            "duration_sec": round(duration, 3),
            "sample_rate":  sr,
        },
        "reference": {
            "hash":               ref.ref_hash,
            "segment_start":      ref.seg_start,
            "segment_end":        ref.seg_end,
            "segment_duration":   round(ref.seg_end - ref.seg_start, 3),
            "transcript":         ref.transcript,
            "transcript_conf":    round(ref.transcript_conf, 4),
            "whisper_model":      ref.whisper_model,
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "rtf":     round(rtf, 3),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return wav_path, meta_path


# ── self-test ──────────────────────────────────────────────────────────────────

def cmd_self_test(args) -> None:
    cache   = Path(args.cache)
    out_dir = Path(args.out)

    test_dir = cache / "voice-clone" / "self-test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ref_wav = test_dir / "self_test_ref.wav"
    if not ref_wav.exists():
        print("  downloading self-test reference clip …")
        urllib.request.urlretrieve(SELF_TEST_REF_URL, ref_wav)
        print(f"  saved → {ref_wav}")
    else:
        print(f"  [cache hit] self-test ref → {ref_wav}")

    print("\n--- prepare-ref ---")
    ref = prepare_ref(
        ref_audio=ref_wav,
        cache=cache,
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=None, ref_end=None,
        ref_language=args.ref_language,
        x_vector_only=args.x_vector_only,
        force=False,
        force_bad_ref=False,
    )

    if not args.x_vector_only:
        assert ref.transcript, "Transcript is empty — faster-whisper may have failed"
        expected = SELF_TEST_REF_TEXT.lower().split()
        got      = ref.transcript.lower().split()
        overlap  = sum(w in got for w in expected[:5])
        if overlap < 2:
            print(
                f"  WARNING: transcript does not match expected first words "
                f"(got {ref.transcript!r})"
            )

    print("\n--- synth ---")
    t0    = time.perf_counter()
    model = load_tts_model(args.model, args.threads)
    t_model_done = time.perf_counter()

    language = resolve_language(args.language, ref.ref_language, SELF_TEST_SYNTH_TEXT)
    validate_language(language, model)

    prompt  = build_voice_clone_prompt(model, ref, cache, args.x_vector_only, force=False)

    t_synth   = time.perf_counter()
    wav, sr   = synthesise(SELF_TEST_SYNTH_TEXT, language, model, prompt, args.seed)
    synth_sec = time.perf_counter() - t_synth

    total_sec = time.perf_counter() - t0
    duration  = len(wav) / sr

    assert not np.any(np.isnan(wav)),  "NaN values found in output audio"
    assert 1.0 < duration < 30.0,      f"Output duration {duration:.1f}s out of range"
    peak = float(np.abs(wav).max())
    assert peak > 5e-4,                f"Peak amplitude {peak:.5f} suspiciously low"

    timings = {
        "model_load_sec": round(t_model_done - t0, 2),
        "synth_sec":      synth_sec,
        "total_sec":      total_sec,
    }
    wav_path, meta_path = write_output(
        wav, sr, SELF_TEST_SYNTH_TEXT, ref,
        args.model, language, ref.ref_language,
        args.x_vector_only, args.seed, None, {},
        out_dir, timings,
    )

    rtf = synth_sec / duration
    print(f"\n  self-test PASSED ✓")
    print(f"  output:   {wav_path}")
    print(f"  duration: {duration:.1f}s  RTF: {rtf:.2f}x  total: {total_sec:.1f}s")
    print(f"  meta:     {meta_path}")


# ── CLI commands ───────────────────────────────────────────────────────────────

def cmd_prepare_ref(args) -> None:
    ref = prepare_ref(
        ref_audio=Path(args.ref_audio),
        cache=Path(args.cache),
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        ref_language=args.ref_language,
        x_vector_only=args.x_vector_only,
        force=args.force,
        force_bad_ref=getattr(args, "force_bad_ref", False),
    )
    print(f"\n  ref_dir:      {ref.ref_dir}")
    print(f"  segment:      {ref.seg_start:.2f}s – {ref.seg_end:.2f}s"
          f"  ({ref.seg_end - ref.seg_start:.1f}s)")
    print(f"  segment wav:  {ref.segment}")
    print(f"  transcript:   {ref.transcript!r}")
    print(f"  ref language: {ref.ref_language} (p={ref.ref_language_prob:.2f})")


def cmd_synth(args) -> None:
    cache   = Path(args.cache)
    out_dir = Path(args.out)

    t0 = time.perf_counter()

    # Stages 1–3
    ref = prepare_ref(
        ref_audio=Path(args.ref_audio),
        cache=cache,
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        ref_language=args.ref_language,
        x_vector_only=args.x_vector_only,
        force=args.force,
        force_bad_ref=args.force_bad_ref,
    )
    prep_sec = time.perf_counter() - t0

    # --ref-text overrides auto-transcription
    if args.ref_text:
        ref.transcript = args.ref_text.strip()
        print(f"  --ref-text override: {ref.transcript!r}")

    # Resolve synthesis text
    if args.text_file:
        raw_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        raw_text = args.text or ""

    if not raw_text:
        print("ERROR: no synthesis text provided (use --text or --text-file)",
              file=sys.stderr)
        sys.exit(1)

    # Apply style + prefix/suffix
    styles_path = Path(__file__).parent / "styles.yaml"
    text = apply_style(
        raw_text,
        style_name=args.style,
        styles_path=styles_path,
        prompt_prefix=args.prompt_prefix,
        prompt_suffix=args.prompt_suffix,
    )

    # Resolve + validate synthesis language
    language = resolve_language(args.language, ref.ref_language, raw_text)
    print(f"\n  ref language detected: {ref.ref_language} (p={ref.ref_language_prob:.2f})")
    print(f"  synthesis language:    {language}")
    print(f"\n==> synthesising ({len(text)} chars): {text!r}")

    # Stage 4: load model + build prompt
    t_model = time.perf_counter()
    model   = load_tts_model(args.model, args.threads)
    validate_language(language, model)
    prompt  = build_voice_clone_prompt(model, ref, cache, args.x_vector_only, args.force)
    model_sec = time.perf_counter() - t_model

    # Collect generation kwargs
    gen_kwargs: dict[str, Any] = {}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    # Stage 5: generate
    t_synth   = time.perf_counter()
    wav, sr   = synthesise(text, language, model, prompt, args.seed, gen_kwargs)
    synth_sec = time.perf_counter() - t_synth

    total_sec = time.perf_counter() - t0
    duration  = len(wav) / sr
    rtf       = synth_sec / duration

    timings = {
        "prep_sec":       prep_sec,
        "model_load_sec": model_sec,
        "synth_sec":      synth_sec,
        "total_sec":      total_sec,
    }

    # Stage 6: write
    wav_path, meta_path = write_output(
        wav, sr, text, ref,
        args.model, language, ref.ref_language,
        args.x_vector_only, args.seed, args.style, gen_kwargs,
        out_dir, timings,
    )

    print(f"\nDone")
    print(f"  WAV:        {wav_path}")
    print(f"  meta:       {meta_path}")
    print(f"  duration:   {duration:.1f}s")
    print(f"  RTF:        {rtf:.2f}x  (synth {synth_sec:.1f}s / audio {duration:.1f}s)")
    print(f"  model load: {model_sec:.1f}s  (cached prompt skips this)")
    print(f"  total:      {total_sec:.1f}s")


# ── argument parser ────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    """Shared flags across all sub-commands."""
    p.add_argument(
        "--model", default=DEFAULT_TTS_MODEL,
        help="Qwen3-TTS model — HF repo ID or local path",
    )
    p.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER,
        help="faster-whisper model size for ref transcription",
    )
    p.add_argument(
        "--ref-language",
        default="Auto",
        choices=QWEN3_LANGUAGES,
        help=(
            "Language of the reference audio. 'Auto' = use whisper auto-detection. "
            "Set explicitly if detection is unreliable for your accent/language."
        ),
    )
    p.add_argument(
        "--language",
        default="Auto",
        choices=QWEN3_LANGUAGES,
        help=(
            "Target synthesis language. 'Auto' detects from the target text "
            "(langid, ≥3 words) then falls back to the detected ref language."
        ),
    )
    p.add_argument(
        "--x-vector-only", action="store_true",
        help=(
            "Build voice prompt from speaker embedding only — no ref_text needed. "
            "Faster but cloning quality may be reduced."
        ),
    )
    p.add_argument(
        "--threads", type=int, default=8,
        help="Number of CPU threads for torch + faster-whisper (default: 8)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible synthesis",
    )
    p.add_argument("--cache", default="/cache", help="Persistent cache directory")
    p.add_argument(
        "--force", action="store_true",
        help="Ignore all cached results and recompute from scratch",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Voice cloning with Qwen3-TTS-Base + faster-whisper "
            "(CPU-first, all stages cached)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ── synth ──────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "synth",
        help="Clone a voice and synthesise new text (full pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument("--ref-audio",  required=True,
                    help="Reference voice recording (WAV / MP3 / etc.)")
    sp.add_argument("--text",       default=None,
                    help="Text to synthesise")
    sp.add_argument("--text-file",  default=None, metavar="FILE",
                    help="Read synthesis text from a file")
    sp.add_argument("--ref-text",   default=None,
                    help="Transcript of the reference audio (skips auto-transcription)")
    sp.add_argument("--ref-start",  type=float, default=None, metavar="SEC",
                    help="Manual reference segment start (seconds)")
    sp.add_argument("--ref-end",    type=float, default=None, metavar="SEC",
                    help="Manual reference segment end (seconds)")
    sp.add_argument(
        "--style", default=None, metavar="PRESET",
        help=(
            "Style preset from styles.yaml "
            "(e.g. serious_doc, excited, whisper, energetic)"
        ),
    )
    sp.add_argument("--prompt-prefix", default=None,
                    help="Text prepended to synthesis text (after style expansion)")
    sp.add_argument("--prompt-suffix", default=None,
                    help="Text appended to synthesis text (after style expansion)")
    # Generation tuning knobs
    sp.add_argument("--temperature",        type=float, default=None,
                    help="Sampling temperature (Transformers generate kwarg)")
    sp.add_argument("--top-p",              type=float, default=None, dest="top_p",
                    help="Top-p nucleus sampling")
    sp.add_argument("--repetition-penalty", type=float, default=None,
                    dest="repetition_penalty",
                    help="Repetition penalty")
    sp.add_argument("--max-new-tokens",     type=int,   default=None,
                    dest="max_new_tokens",
                    help="Maximum generated tokens")
    sp.add_argument(
        "--force-bad-ref", action="store_true",
        help="Bypass the transcript quality gate (low avg_logprob warning)",
    )
    sp.add_argument("--out", default="/work",
                    help="Output directory for WAV + meta.json")
    _add_common(sp)

    # ── prepare-ref ────────────────────────────────────────────────────────────
    pr = sub.add_parser(
        "prepare-ref",
        help="Run stages 1–3 only (normalise, VAD score, transcribe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pr.add_argument("--ref-audio",  required=True,
                    help="Reference voice recording")
    pr.add_argument("--ref-start",  type=float, default=None, metavar="SEC")
    pr.add_argument("--ref-end",    type=float, default=None, metavar="SEC")
    pr.add_argument("--force-bad-ref", action="store_true",
                    help="Bypass transcript quality gate")
    _add_common(pr)

    # ── self-test ──────────────────────────────────────────────────────────────
    st = sub.add_parser(
        "self-test",
        help="End-to-end smoke test using the Qwen3-TTS demo reference clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    st.add_argument("--out", default="/work",
                    help="Output directory for test WAV + meta.json")
    _add_common(st)

    return ap


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    cache   = Path(args.cache)
    hub_dir = cache / "torch" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))

    match args.command:
        case "synth":
            if not args.text and not args.text_file:
                ap.error("synth requires --text or --text-file")
            cmd_synth(args)
        case "prepare-ref":
            cmd_prepare_ref(args)
        case "self-test":
            cmd_self_test(args)
        case _:
            ap.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
