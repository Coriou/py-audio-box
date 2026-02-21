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
        --text "Hello" --language French
  ./run voice-clone prepare-ref --ref-audio /work/myvoice.wav
  ./run voice-clone self-test
"""

import argparse
import json
import os
import pickle
import re
import shutil
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

# ── shared lib ─────────────────────────────────────────────────────────────────
# All helpers common to voice-clone and voice-synth live in lib/.
_LIB = str(Path(__file__).resolve().parent.parent.parent / "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from tts import (  # noqa: E402  (import after path setup)
    DEFAULT_TTS_MODEL, DEFAULT_WHISPER, PROMPT_SCHEMA_VERSION,
    QWEN3_LANGUAGES,
    estimate_max_new_tokens,
    GENERATION_PROFILE_CHOICES,
    resolve_generation_profile, build_generation_kwargs,
    resolve_language, validate_language,
    build_whisper_model, transcribe_segment, transcribe_ref,
    load_tts_model, synthesise,
    Timer,
)
from audio import (  # noqa: E402
    SELECTION_WEIGHTS,
    sha256_file, get_duration, normalize_audio, trim_audio_encode,
    analyse_acoustics, score_take_selection, rank_take_selection,
)
from vad import load_silero, run_silero  # noqa: E402
from voices import VoiceRegistry  # noqa: E402

# ── voice-clone-specific constants ─────────────────────────────────────────────

MIN_REF_SECONDS      = 3.0
MAX_REF_SECONDS      = 12.0
SWEET_SPOT_SECONDS   = 8.5     # ideal reference clip duration
CANDIDATE_COUNT      = 3       # top-N VAD candidates to whisper-score

# Transcripts below this avg_logprob may hurt cloning quality.
QUALITY_GATE_LOGPROB = -1.2

# Bump whenever candidate scoring policy/weights change so old cached
# best_segment.json entries are re-evaluated automatically.
CANDIDATE_SCORING_VERSION = 2
TAKES_META_SCHEMA_VERSION = 1

# Public-domain demo clips used for self-test.
# First URL is the original Qwen-hosted sample; remaining entries are fallbacks.
SELF_TEST_REF_SOURCES: tuple[dict[str, str], ...] = (
    {
        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        "expected_text": (
            "Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you."
        ),
    },
    {
        "url": (
            "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/"
            "data/smoke_test/new-home-in-the-stars-16k.wav"
        ),
        "expected_text": "We must find a new home in the stars.",
    },
)
SELF_TEST_SYNTH_TEXT = "The quick brown fox jumps over the lazy dog."


def transcript_confidence_passes(
    avg_logprob: float,
    threshold: float = QUALITY_GATE_LOGPROB,
) -> bool:
    """
    Return True when the transcript confidence passes the quality gate.
    """
    return float(avg_logprob) >= float(threshold)


def candidate_scoring_cache_is_current(payload: dict[str, Any]) -> bool:
    """
    Return True when a cached best-segment entry matches the current policy.
    """
    try:
        version = int(payload.get("candidate_scoring_version", -1))
    except (TypeError, ValueError):
        return False
    return version == CANDIDATE_SCORING_VERSION


def _resolve_ref_audio(args, cache: Path) -> Path:
    """
    Return the reference audio path from either ``--ref-audio`` or ``--voice``.

    When ``--voice SLUG`` is used the registry's **source_clip.wav** is
    preferred as pipeline input because it produces a deterministic hash —
    ``ref.wav`` changes every time the segment selection changes, which would
    rebuild the prompt needlessly.  Falls back to ``ref.wav`` if no
    source_clip exists yet (e.g. the voice was registered manually).
    """
    if getattr(args, "voice", None):
        reg = VoiceRegistry(cache)
        if not reg.exists(args.voice):
            print(
                f"ERROR: voice '{args.voice}' not found in registry.\n"
                f"  Run: ./run voice-synth list-voices",
                file=sys.stderr,
            )
            sys.exit(1)
        # Prefer source_clip for stable hash; fallback to ref.wav
        src = reg.source_clip(args.voice)
        if src.exists():
            print(f"  [registry] using source_clip from voice '{args.voice}': {src}")
            return src
        ref = reg.ref_wav(args.voice)
        if ref.exists():
            print(f"  [registry] using ref.wav from voice '{args.voice}': {ref}")
            return ref
        print(
            f"ERROR: voice '{args.voice}' has no reference audio.\n"
            f"  Run voice-split first, or supply --ref-audio directly.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(args.ref_audio)


# ── Silero VAD helpers (voice-clone specific) ──────────────────────────────────

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
    segs = run_silero(wav_path, model, get_ts)
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


# ── candidate scoring (stage 2 helper) ────────────────────────────────────────


def _interactive_pick_candidate(
    scored: list[dict],
    cand_dir: Path,
) -> dict:
    """
    Present a numbered menu of scored VAD candidates and return the one the
    user picks.  Falls back to the auto-selected best on EOF / Ctrl-C.

    ``scored`` must be sorted best-first (as produced by
    ``_score_and_select_best_candidate``).
    """
    # Nothing to choose — skip the prompt.
    if len(scored) == 1:
        best = scored[0]
        print(
            f"  (only one candidate — auto-selecting candidate_{best['index']:02d}.wav "
            f"{best['seg_start']:.1f}s\u2013{best['seg_end']:.1f}s)"
        )
        return best

    n = len(scored)
    line = "─" * 60
    print(f"\n{line}")
    print("  Interactive segment selection")
    print(f"{line}")
    print(f"  Candidate WAV{'s are' if n > 1 else ' is'} in: {cand_dir}")
    print()
    for entry in scored:
        i = entry["index"]
        marker = "  ← auto-selected (best score)" if entry is scored[0] else ""
        print(
            f"  [{i}] candidate_{i:02d}.wav  "
            f"{entry['seg_start']:.1f}s–{entry['seg_end']:.1f}s  "
            f"({entry['duration']:.1f}s)  "
            f"[{entry['ref_language']} p={entry['ref_language_prob']:.2f}]  "
            f"score={entry['score']:.3f}"
            f"{marker}"
        )
        if entry["transcript"]:
            print(f"       transcript: {entry['transcript']!r}")
    print()
    print("  Listen to each candidate, then enter the number of the segment")
    print("  you want to use as the voice reference.")
    default_idx = scored[0]["index"]
    idx_map = {e["index"]: e for e in scored}
    while True:
        try:
            raw = input(f"  Your choice [default {default_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  (no tty / interrupted — keeping auto-selected candidate)")
            return scored[0]
        if raw == "":
            print(f"  ✓ keeping auto-selected: candidate_{default_idx:02d}.wav")
            return scored[0]
        try:
            idx = int(raw)
        except ValueError:
            valid = sorted(idx_map)
            print(f"  Please enter one of: {valid}")
            continue
        if idx not in idx_map:
            valid = sorted(idx_map)
            print(f"  Please enter one of: {valid}")
            continue
        chosen = idx_map[idx]
        if chosen is scored[0]:
            print(f"  ✓ keeping auto-selected: candidate_{idx:02d}.wav")
        else:
            print(f"  ✓ selected: candidate_{idx:02d}.wav")
        return chosen


def _score_segment(
    duration: float,
    avg_logprob: float,
    acoustics: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Combined quality score for a candidate segment.

    Weights:
      45%  transcription confidence (avg_logprob normalized from [-2, 0] → [0, 1])
      20%  duration proximity       (1.0 at SWEET_SPOT_SECONDS, 0.0 at MAX)
      35%  acoustic heuristics      (clipping, noise/rms, speech continuity)
    """
    lp_norm = max(0.0, min(1.0, (avg_logprob + 2.0) / 2.0))
    dur_norm = max(0.0, 1.0 - abs(duration - SWEET_SPOT_SECONDS) / MAX_REF_SECONDS)
    acoustic_score = (
        0.40 * float(acoustics.get("clipping_score", 0.0)) +
        0.30 * float(acoustics.get("noise_score", 0.0)) +
        0.30 * float(acoustics.get("speech_continuity_score", 0.0))
    )
    total = (0.45 * lp_norm) + (0.20 * dur_norm) + (0.35 * acoustic_score)
    return total, {
        "logprob_score": round(lp_norm, 4),
        "duration_score": round(dur_norm, 4),
        "acoustic_score": round(acoustic_score, 4),
        "clipping_score": round(float(acoustics.get("clipping_score", 0.0)), 4),
        "noise_score": round(float(acoustics.get("noise_score", 0.0)), 4),
        "speech_continuity_score": round(
            float(acoustics.get("speech_continuity_score", 0.0)),
            4,
        ),
    }


def _word_overlap_ratio(ref_words: list[str], hyp_words: list[str]) -> float:
    if not ref_words:
        return 0.0
    hyp_set = set(hyp_words)
    matched = sum(1 for w in ref_words if w in hyp_set)
    return matched / len(ref_words)


def _qa_transcribe_take(wav_path: Path, wm, target_text: str) -> dict[str, Any]:
    """
    Fast QA transcription for take ranking.
    """
    transcript, _lp, _lang, _prob = transcribe_segment(wav_path, wm, beam_size=2)
    ref_words = re.findall(r"\w+", target_text.lower())
    hyp_words = re.findall(r"\w+", transcript.lower())
    overlap = _word_overlap_ratio(ref_words, hyp_words)
    return {
        "transcript": transcript,
        "intelligibility": round(overlap, 4),
        "duration_sec": round(get_duration(wav_path), 3),
    }


_INSTRUCTION_HACK_PATTERNS = [
    re.compile(r"^\s*\[[^\]]{2,120}\]"),
    re.compile(r"^\s*\([^)]{2,120}\)"),
    re.compile(r"^\s*(style|tone|emotion|mood)\s*:", re.IGNORECASE),
    re.compile(r"\bin the style of\b", re.IGNORECASE),
    re.compile(r"\b(speak|read|say)\s+(this|it)\s+(in|with)\b", re.IGNORECASE),
]


def instruction_like_text_hack(text: str) -> str | None:
    """
    Return the matched cue when *text* looks like an instruction hack.

    Clone mode does not support runtime instruction control; these cues are
    likely to be spoken literally.
    """
    for pat in _INSTRUCTION_HACK_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


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
    interactive: bool = False,
) -> tuple[tuple[float, float], str, float, str, float]:
    """
    Score each candidate segment via fast whisper (beam_size=2), pick the best.

    Returns (best_seg, transcript, avg_logprob, detected_language, lang_prob).
    Caches individual candidate clips under ref_dir/candidates/.
    Writes ref_dir/best_segment.json for future cache hits.

    When *interactive* is True the cache is bypassed so the user can always
    choose even if a previous automatic selection was cached.
    """
    cand_dir = ref_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    best_json = ref_dir / "best_segment.json"
    if best_json.exists() and not force and not interactive:
        with open(best_json) as fh:
            d = json.load(fh)
        if candidate_scoring_cache_is_current(d):
            score_val = d.get("score")
            score_txt = f"{float(score_val):.3f}" if isinstance(score_val, (int, float)) else "?"
            print(
                f"  [cache hit] best segment: "
                f"{d['seg_start']:.2f}s – {d['seg_end']:.2f}s  "
                f"(score={score_txt})"
            )
            return (
                (d["seg_start"], d["seg_end"]),
                d.get("transcript", ""),
                d.get("avg_logprob", 0.0),
                d.get("ref_language", "English"),
                d.get("ref_language_prob", 1.0),
            )
        print("  [cache stale] candidate scoring policy updated; rescoring best segment …")

    if len(candidates) == 1 and not interactive:
        # Single candidate — skip scoring; full transcription handled in stage 3
        s, e = candidates[0]
        return (s, e), "", 0.0, "English", 1.0

    print(f"  scoring {len(candidates)} candidate segment(s) with whisper …")
    wm = build_whisper_model(whisper_model, num_threads)

    scored: list[dict] = []
    for i, (seg_s, seg_e) in enumerate(candidates):
        wav_path = cand_dir / f"candidate_{i:02d}.wav"
        if not wav_path.exists() or force:
            trim_audio_encode(normalized, seg_s, seg_e - seg_s, wav_path)

        t, lp, lang, lp_prob = transcribe_segment(wav_path, wm, beam_size=2)
        dur = seg_e - seg_s
        acoustics = analyse_acoustics(wav_path)
        score, breakdown = _score_segment(dur, lp, acoustics)
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
            "score_breakdown":   breakdown,
            "acoustics":         acoustics,
        })
        print(
            f"    candidate {i}: {seg_s:.1f}s–{seg_e:.1f}s  "
            f"[{lang} p={lp_prob:.2f}]  lp={lp:.2f}  score={score:.3f}  "
            f"clip={acoustics['clipping_ratio']:.3%}  "
            f"speech={acoustics['speech_ratio']:.2f}"
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    if interactive:
        best = _interactive_pick_candidate(scored, cand_dir)
    else:
        best = scored[0]

    print(
        f"  ✓ best candidate: {best['seg_start']:.2f}s – {best['seg_end']:.2f}s  "
        f"score={best['score']:.3f}"
    )

    # Persist all scores for debugging
    with open(cand_dir / "scores.json", "w") as fh:
        json.dump(
            {
                "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
                "candidates": scored,
                "selected_index": best["index"],
            },
            fh,
            indent=2,
        )

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
                "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
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
    interactive:   bool = False,
    ref_text:      str | None = None,
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
        with Timer("ffmpeg normalize"):
            normalize_audio(ref_audio, normalized)

    # ── 2. Select best segment ─────────────────────────────────────────────────
    print("\n==> [2/3] select reference segment")

    # These are set below via one of three paths
    seg_start = seg_end = 0.0
    segment_transcript   = ""
    segment_logprob      = 0.0
    segment_lang         = "English"
    segment_lang_prob    = 1.0

    if ref_start is not None and ref_end is not None:
        seg_start, seg_end = ref_start, ref_end
        print(f"  using manual bounds: {seg_start:.2f}s – {seg_end:.2f}s")

    else:
        # Honour either the new best_segment.json or the old vad_best_segment.json
        legacy_json = ref_dir / "vad_best_segment.json"
        best_json   = ref_dir / "best_segment.json"

        cached_segment_loaded = False
        if best_json.exists() and not force and not interactive:
            with open(best_json) as fh:
                d = json.load(fh)
            if candidate_scoring_cache_is_current(d):
                seg_start, seg_end = d["seg_start"], d["seg_end"]
                segment_transcript = d.get("transcript", "")
                segment_logprob = d.get("avg_logprob", 0.0)
                segment_lang = d.get("ref_language", "English")
                segment_lang_prob = d.get("ref_language_prob", 1.0)
                cached_segment_loaded = True
                print(
                    f"  [cache hit] segment: {seg_start:.2f}s – {seg_end:.2f}s  "
                    f"({seg_end - seg_start:.1f}s)"
                )
            else:
                print("  [cache stale] candidate scoring policy updated; rescoring …")
        elif legacy_json.exists() and not force and not interactive:
            print("  [cache stale] legacy segment cache detected; rescoring …")

        if not cached_segment_loaded:
            print("  loading Silero VAD …")
            hub_dir = cache / "torch" / "hub"
            vad_model, get_ts = load_silero(hub_dir=hub_dir)
            with Timer("Silero VAD"):
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

            with Timer("candidate scoring"):
                (seg_start, seg_end), segment_transcript, segment_logprob, \
                    segment_lang, segment_lang_prob = \
                    _score_and_select_best_candidate(
                        candidates, normalized, ref_dir,
                        whisper_model, num_threads, force,
                        interactive=interactive,
                    )

            print(
                f"  best segment: {seg_start:.2f}s – {seg_end:.2f}s  "
                f"({seg_end - seg_start:.1f}s)"
            )

    segment_wav = ref_dir / "ref_segment.wav"
    if segment_wav.exists() and not force and not interactive:
        print(f"  [cache hit] segment wav → {segment_wav}")
    else:
        print("  trimming segment …")
        with Timer("ffmpeg trim"):
            trim_audio_encode(normalized, seg_start, seg_end - seg_start, segment_wav)

    # ── 3. Transcribe ──────────────────────────────────────────────────────────
    print("\n==> [3/3] transcribe reference segment")

    transcript_json = ref_dir / "ref_transcript.json"
    provided_ref_text = ref_text.strip() if ref_text is not None else None

    if x_vector_only:
        print("  --x-vector-only: skipping transcription (quality may be lower)")
        transcript    = ""
        conf          = 0.0
        detected_lang = "English"
        lang_prob     = 1.0

    elif provided_ref_text is not None:
        if not provided_ref_text:
            print("ERROR: --ref-text cannot be empty.", file=sys.stderr)
            sys.exit(1)
        transcript = provided_ref_text
        conf = 0.0
        if ref_language and ref_language != "Auto":
            detected_lang = ref_language
            lang_prob = 1.0
        else:
            detected_lang = segment_lang or "English"
            lang_prob = float(segment_lang_prob) if segment_lang_prob else 1.0
        print("  using provided --ref-text; skipping final whisper transcription + quality gate")
        with open(transcript_json, "w") as fh:
            json.dump(
                {
                    "transcript": transcript,
                    "avg_logprob": conf,
                    "whisper_model": "provided_ref_text",
                    "ref_language_detected": detected_lang,
                    "ref_language_probability": lang_prob,
                    "transcript_source": "provided_ref_text",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                fh,
                indent=2,
            )

    elif transcript_json.exists() and not force and not interactive:
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
        with Timer("faster-whisper"):
            transcript, conf, detected_lang, lang_prob = transcribe_ref(
                segment_wav, whisper_model, num_threads,
            )

        # ── quality gate ───────────────────────────────────────────────────────
        if not transcript_confidence_passes(conf) and not force_bad_ref:
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
        elif not transcript_confidence_passes(conf):
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


def _model_tag(model) -> str:
    """
    Return a short, filesystem-safe model tag used in prompt filenames.
    Prefers ``model.name_or_path`` (HuggingFace convention); falls back to
    ``"qwen3tts"`` so that the stem is stable regardless of whether the model
    was loaded by repo ID, local path, or another name.
    """
    raw = getattr(model, "name_or_path", None) or "qwen3tts"
    # Use only the final component of a repo path (e.g. "Qwen/Foo" → "foo")
    return raw.split("/")[-1].lower().replace(" ", "-")


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
    mtag      = _model_tag(model)
    stem      = f"{ref.ref_hash}_{mtag}_{mode}_v{PROMPT_SCHEMA_VERSION}"

    prompts_dir = cache / "voice-clone" / "prompts"
    prompt_path = prompts_dir / f"{stem}.pkl"
    meta_path   = prompts_dir / f"{stem}.meta.json"

    if prompt_path.exists() and not force:
        prompt_cache_valid = True
        if not x_vector_only:
            try:
                if meta_path.exists() and meta_path.stat().st_size > 0:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    if str(meta.get("transcript", "")) != ref.transcript:
                        prompt_cache_valid = False
                        print("  [cache stale] prompt transcript changed; rebuilding cached prompt …")
                else:
                    prompt_cache_valid = False
                    print("  [cache stale] prompt meta missing/empty; rebuilding cached prompt …")
            except Exception:
                prompt_cache_valid = False
                print("  [cache stale] prompt meta unreadable; rebuilding cached prompt …")

        if prompt_cache_valid:
            print(f"  [cache hit] voice prompt → {prompt_path.name}")
            # Repair missing or empty meta.json (can happen after an interrupted write)
            if not meta_path.exists() or meta_path.stat().st_size == 0:
                prompts_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w") as fh:
                    json.dump(
                        {
                            "prompt_id":               stem,
                            "schema_version":          PROMPT_SCHEMA_VERSION,
                            "model":                   mtag,
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
                print(f"  [repaired] meta.json → {meta_path.name}")
            with open(prompt_path, "rb") as fh:
                return pickle.load(fh), prompt_path

    print(f"  building voice-clone prompt (mode={mode}) …")
    with Timer("create_voice_clone_prompt"):
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
                "model":                   mtag,
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
    return prompt, prompt_path


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
    tone: str | None,
    gen_kwargs: dict[str, Any] | None,
    out_dir: Path,
    timings: dict[str, float],
    extra_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path  = out_dir / f"voice_clone_{ts}.wav"
    meta_path = out_dir / f"voice_clone_{ts}.meta.json"

    sf.write(str(wav_path), wav, sr)
    (out_dir / "text.txt").write_text(text, encoding="utf-8")

    duration  = len(wav) / sr
    synth_sec = timings.get("synth_sec", 0.0)
    rtf       = synth_sec / duration if duration > 0 else 0.0

    meta: dict = {
        "schema_version":           TAKES_META_SCHEMA_VERSION,
        "app":                      "voice-clone",
        "created_at":               datetime.now(timezone.utc).isoformat(),
        "model":                    model_name,
        "engine_mode":              "clone_prompt",
        "source_type":              "clone_prompt",
        "speaker":                  None,
        "instruct":                 None,
        "instruct_source":          "none",
        "language":                 language,
        "ref_language_detected":    ref_language,
        "ref_language_probability": round(ref.ref_language_prob, 4),
        "x_vector_only":            x_vector_only,
        "seed":                     seed,
        "tone":                     tone,
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
    if extra_meta:
        meta.update(extra_meta)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return wav_path, meta_path


# ── self-test ──────────────────────────────────────────────────────────────────

def _download_self_test_reference(
    dest: Path,
    sources: tuple[dict[str, str], ...] = SELF_TEST_REF_SOURCES,
) -> dict[str, str]:
    """
    Download the self-test reference clip with fallback URLs.

    Returns the source descriptor used for the successful download.
    Raises RuntimeError if all URLs fail.
    """
    errors: list[str] = []

    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "py-audio-box/self-test"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return {"url": url, "expected_text": str(source.get("expected_text") or "")}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
            dest.unlink(missing_ok=True)

    details = "\n    ".join(errors) if errors else "(no URLs configured)"
    raise RuntimeError(
        "Failed to download self-test reference clip from all sources:\n"
        f"    {details}"
    )


def cmd_self_test(args) -> None:
    cache   = Path(args.cache)
    out_dir = Path(args.out)

    test_dir = cache / "voice-clone" / "self-test"
    test_dir.mkdir(parents=True, exist_ok=True)

    ref_wav = test_dir / "self_test_ref.wav"
    source_meta_path = test_dir / "self_test_ref.source.json"
    source_info: dict[str, str] = {}
    if not ref_wav.exists():
        print("  downloading self-test reference clip …")
        try:
            source_info = _download_self_test_reference(ref_wav)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        with open(source_meta_path, "w") as fh:
            json.dump(source_info, fh, indent=2)
        print(f"  saved → {ref_wav}")
        print(f"  source: {source_info.get('url')}")
    else:
        print(f"  [cache hit] self-test ref → {ref_wav}")
        if source_meta_path.exists():
            try:
                with open(source_meta_path) as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict):
                    source_info = {k: str(v) for k, v in payload.items()}
            except Exception:
                source_info = {}

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

    expected_ref_text = str(source_info.get("expected_text") or "").strip()
    if not args.x_vector_only and expected_ref_text:
        assert ref.transcript, "Transcript is empty — faster-whisper may have failed"
        expected = expected_ref_text.lower().split()
        got      = ref.transcript.lower().split()
        overlap  = sum(w in got for w in expected[:5])
        if overlap < 2:
            print(
                f"  WARNING: transcript does not match expected first words "
                f"(got {ref.transcript!r})"
            )

    print("\n--- synth ---")
    t0    = time.perf_counter()
    model = load_tts_model(args.model, args.threads, args.dtype)
    t_model_done = time.perf_counter()

    language = resolve_language(args.language, ref.ref_language, SELF_TEST_SYNTH_TEXT)
    language = validate_language(language, model)

    prompt, _pkl = build_voice_clone_prompt(model, ref, cache, args.x_vector_only, force=False)

    t_synth   = time.perf_counter()
    wav, sr   = synthesise(
        SELF_TEST_SYNTH_TEXT, language, model, prompt, args.seed,
        gen_kwargs={"max_new_tokens": estimate_max_new_tokens(SELF_TEST_SYNTH_TEXT)},
    )
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
    print("\n  self-test PASSED ✓")
    print(f"  output:   {wav_path}")
    print(f"  duration: {duration:.1f}s  RTF: {rtf:.2f}x  total: {total_sec:.1f}s")
    print(f"  meta:     {meta_path}")


# ── CLI commands ───────────────────────────────────────────────────────────────

def cmd_prepare_ref(args) -> None:
    cache     = Path(args.cache)
    ref_audio = _resolve_ref_audio(args, cache)

    ref = prepare_ref(
        ref_audio=ref_audio,
        cache=cache,
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        ref_language=args.ref_language,
        x_vector_only=args.x_vector_only,
        force=args.force,
        force_bad_ref=getattr(args, "force_bad_ref", False),
        interactive=getattr(args, "interactive", False),
        ref_text=getattr(args, "ref_text", None),
    )

    print(f"\n  ref_dir:      {ref.ref_dir}")
    print(f"  segment:      {ref.seg_start:.2f}s – {ref.seg_end:.2f}s"
          f"  ({ref.seg_end - ref.seg_start:.1f}s)")
    print(f"  segment wav:  {ref.segment}")
    print(f"  transcript:   {ref.transcript!r}")
    print(f"  ref language: {ref.ref_language} (p={ref.ref_language_prob:.2f})")

    # Register ref into named voice if requested.
    # Also auto-registers when --voice was used (not just explicit --voice-name).
    voice_name = getattr(args, "voice_name", None) or getattr(args, "voice", None)
    if voice_name:
        from voices import validate_slug  # type: ignore (lib already on path)
        slug = validate_slug(voice_name)
        reg = VoiceRegistry(cache)
        if not reg.exists(slug):
            reg.create(slug, display_name=slug,
                       source={"type": "file", "path": str(ref_audio)})
        reg.update_ref(slug, ref.segment, {
            "hash":            ref.ref_hash,
            "segment_start":   ref.seg_start,
            "segment_end":     ref.seg_end,
            "duration_sec":    round(ref.seg_end - ref.seg_start, 3),
            "transcript":      ref.transcript,
            "transcript_conf": round(ref.transcript_conf, 4),
            "language":        ref.ref_language,
            "language_prob":   round(ref.ref_language_prob, 4),
        })
        print(f"\n  [registry] voice '{slug}' updated")
        print(f"  Next: ./run voice-clone synth --voice {slug} --text 'Hello'")


def build_clone_takes_metadata(
    *,
    model: str,
    language: str,
    ref: RefResult,
    x_vector_only: bool,
    tone: str | None,
    gen_kwargs: dict[str, Any],
    profile_name: str,
    profile_source: str,
    generation_meta: dict[str, Any],
    voice_generation_defaults: dict[str, Any],
    saved_generation_defaults: dict[str, Any] | None,
    selection_policy: dict[str, Any],
    selection_breakdown: list[dict[str, Any]],
    selected_take: str | None,
    selected_wav: str | None,
    text: str,
    prep_sec: float,
    model_sec: float,
    total_sec: float,
    n_variants: int,
    takes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": TAKES_META_SCHEMA_VERSION,
        "app": "voice-clone",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "engine_mode": "clone_prompt",
        "source_type": "clone_prompt",
        "speaker": None,
        "instruct": None,
        "instruct_source": "none",
        "language": language,
        "ref_language_detected": ref.ref_language,
        "ref_language_probability": round(ref.ref_language_prob, 4),
        "x_vector_only": x_vector_only,
        "tone": tone,
        "generation_kwargs": gen_kwargs,
        "generation_profile": profile_name,
        "generation_profile_source": profile_source,
        "generation_profile_meta": generation_meta,
        "voice_generation_defaults": voice_generation_defaults,
        "saved_generation_defaults": saved_generation_defaults,
        "selection_policy": selection_policy,
        "selection_metrics": selection_breakdown,
        "selected_take": selected_take,
        "selected_wav": selected_wav,
        "text": text,
        "timings": {
            "prep_sec": round(prep_sec, 2),
            "model_load_sec": round(model_sec, 2),
            "total_sec": round(total_sec, 2),
        },
        "reference": {
            "hash": ref.ref_hash,
            "segment_start": ref.seg_start,
            "segment_end": ref.seg_end,
            "segment_duration": round(ref.seg_end - ref.seg_start, 3),
            "transcript": ref.transcript,
            "transcript_conf": round(ref.transcript_conf, 4),
            "whisper_model": ref.whisper_model,
        },
        "variants": n_variants,
        "takes": takes,
    }


def cmd_synth(args) -> None:
    cache     = Path(args.cache)
    out_dir   = Path(args.out)
    ref_audio = _resolve_ref_audio(args, cache)

    # Validate + ensure named voice exists in registry before doing any work.
    # If --voice SLUG was used without an explicit --voice-name, auto-register
    # back to the same slug so each synth run keeps the prompt up-to-date.
    voice_name = getattr(args, "voice_name", None) or getattr(args, "voice", None)
    reg = VoiceRegistry(cache)
    if voice_name:
        _lib = str(Path(__file__).resolve().parent.parent.parent / "lib")
        if _lib not in sys.path:
            sys.path.insert(0, _lib)
        from voices import validate_slug  # type: ignore
        voice_name = validate_slug(voice_name)
        if not reg.exists(voice_name):
            reg.create(
                voice_name,
                display_name=voice_name,
                source={"type": "file", "path": str(ref_audio)},
            )
    elif getattr(args, "save_profile_default", False):
        print(
            "ERROR: --save-profile-default requires --voice or --voice-name.",
            file=sys.stderr,
        )
        sys.exit(1)

    t0 = time.perf_counter()

    # Stages 1–3
    ref = prepare_ref(
        ref_audio=ref_audio,
        cache=cache,
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        ref_language=args.ref_language,
        x_vector_only=args.x_vector_only,
        force=args.force,
        force_bad_ref=args.force_bad_ref,
        interactive=getattr(args, "interactive", False),
        ref_text=getattr(args, "ref_text", None),
    )
    prep_sec = time.perf_counter() - t0

    # Resolve synthesis text
    if args.text_file:
        raw_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        raw_text = args.text or ""

    if not raw_text:
        print("ERROR: no synthesis text provided (use --text or --text-file)",
              file=sys.stderr)
        sys.exit(1)

    # Build synthesis text: prompt-prefix/suffix are explicit text insertions
    # (they will be spoken) — no style-prefix injection for voice clone.
    # Tone/style comes from the reference audio, not from text instructions.
    text_parts: list[str] = []
    if args.prompt_prefix:
        text_parts.append(args.prompt_prefix.strip())
    text_parts.append(raw_text)
    if args.prompt_suffix:
        text_parts.append(args.prompt_suffix.strip())
    text = " ".join(text_parts)

    hack_hint = instruction_like_text_hack(raw_text)
    if hack_hint:
        print(
            "  WARNING: clone mode does not support runtime style/instruction control.\n"
            f"  Detected instruction-like cue in text ({hack_hint!r}); it will likely be spoken literally.\n"
            "  For controllable style use: (a) a tone-matched reference clip, or "
            "(b) voice-synth --speaker/--instruct (CustomVoice mode).",
            file=sys.stderr,
        )

    # Resolve + validate synthesis language
    language = resolve_language(args.language, ref.ref_language, raw_text)
    print(f"\n  ref language detected: {ref.ref_language} (p={ref.ref_language_prob:.2f})")
    print(f"  synthesis language:    {language}")
    print(f"\n==> synthesising ({len(text)} chars): {text!r}")

    voice_generation_defaults = reg.generation_defaults(voice_name) if voice_name else {}
    profile_name, profile_source = resolve_generation_profile(
        args.profile,
        voice_generation_defaults if voice_name else None,
    )
    gen_kwargs, generation_meta = build_generation_kwargs(
        text=text,
        profile=profile_name,
        voice_defaults=voice_generation_defaults if voice_name else None,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  generation profile:    {profile_name} ({profile_source})")

    saved_generation_defaults: dict[str, Any] | None = None
    if getattr(args, "save_profile_default", False) and voice_name:
        policy = str(generation_meta.get("max_new_tokens_policy") or profile_name)
        if policy not in ("stable", "balanced", "expressive"):
            policy = profile_name
        saved_generation_defaults = reg.update_generation_defaults(
            voice_name,
            profile=profile_name,
            temperature=float(gen_kwargs["temperature"]),
            top_p=float(gen_kwargs["top_p"]),
            repetition_penalty=float(gen_kwargs["repetition_penalty"]),
            max_new_tokens_policy=policy,
        )
        print(f"  saved generation defaults to voice '{voice_name}'")

    # Stage 4: load model + build prompt
    t_model = time.perf_counter()
    model   = load_tts_model(args.model, args.threads, args.dtype)
    language = validate_language(language, model)
    prompt, pkl_path = build_voice_clone_prompt(
        model, ref, cache, args.x_vector_only, args.force
    )
    model_sec = time.perf_counter() - t_model

    # Register ref + prompt to named voice if requested
    if voice_name:
        meta_path = pkl_path.parent / (pkl_path.stem + ".meta.json")
        reg.update_ref(voice_name, ref.segment, {
            "hash":            ref.ref_hash,
            "segment_start":   ref.seg_start,
            "segment_end":     ref.seg_end,
            "duration_sec":    round(ref.seg_end - ref.seg_start, 3),
            "transcript":      ref.transcript,
            "transcript_conf": round(ref.transcript_conf, 4),
            "language":        ref.ref_language,
            "language_prob":   round(ref.ref_language_prob, 4),
        })
        if meta_path.exists() and meta_path.stat().st_size > 0:
            with open(meta_path) as _fh:
                _meta = json.load(_fh)
            reg.register_prompt(voice_name, pkl_path.stem, pkl_path, _meta, tone=args.tone)
            tone_note = f"  (tone={args.tone!r})" if args.tone else ""
            print(f"  [registry] voice '{voice_name}' \u2192 prompt registered{tone_note}")
        else:
            print(
                f"  WARNING: meta.json not found or empty — prompt not registered\n"
                f"    expected: {meta_path}",
                file=sys.stderr,
            )
    n_variants = max(1, args.variants)
    if args.select_best and args.seed is None:
        print("  --select-best enabled without --seed; using deterministic base seed 0")
    base_seed = args.seed if args.seed is not None else (0 if args.select_best else None)

    # Single-take legacy output path remains unchanged for compatibility.
    if n_variants == 1 and not args.select_best:
        t_synth = time.perf_counter()
        wav, sr = synthesise(
            text, language, model, prompt, base_seed, gen_kwargs,
            timeout_s=args.timeout,
        )
        synth_sec = time.perf_counter() - t_synth

        total_sec = time.perf_counter() - t0
        duration = len(wav) / sr
        rtf = synth_sec / duration if duration > 0 else 0.0

        timings = {
            "prep_sec": prep_sec,
            "model_load_sec": model_sec,
            "synth_sec": synth_sec,
            "total_sec": total_sec,
        }

        eff_out_dir = out_dir / voice_name if voice_name else out_dir
        wav_path, meta_path = write_output(
            wav, sr, text, ref,
            args.model, language, ref.ref_language,
            args.x_vector_only, base_seed, args.tone, gen_kwargs,
            eff_out_dir, timings,
            extra_meta={
                "generation_profile": profile_name,
                "generation_profile_source": profile_source,
                "generation_profile_meta": generation_meta,
                "voice_generation_defaults": voice_generation_defaults,
                "saved_generation_defaults": saved_generation_defaults,
                "selection_policy": {"enabled": False},
            },
        )

        print("\nDone")
        print(f"  WAV:        {wav_path}")
        print(f"  meta:       {meta_path}")
        print(f"  duration:   {duration:.1f}s")
        print(f"  RTF:        {rtf:.2f}x  (synth {synth_sec:.1f}s / audio {duration:.1f}s)")
        print(f"  model load: {model_sec:.1f}s  (cached prompt skips this)")
        print(f"  total:      {total_sec:.1f}s")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_dir / voice_name / ts) if voice_name else (out_dir / f"voice_clone_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    wm = build_whisper_model(args.whisper_model, args.threads) if args.select_best else None

    takes: list[dict[str, Any]] = []
    for variant in range(n_variants):
        seed = (base_seed + variant) if base_seed is not None else None
        take_label = f"take_{variant + 1:02d}"
        t_synth = time.perf_counter()
        wav, sr = synthesise(
            text, language, model, prompt, seed, gen_kwargs,
            timeout_s=args.timeout,
        )
        synth_sec = time.perf_counter() - t_synth
        wav_path = run_dir / f"{take_label}.wav"
        sf.write(str(wav_path), wav, sr)
        duration = len(wav) / sr
        rtf = synth_sec / duration if duration > 0 else 0.0

        take: dict[str, Any] = {
            "take": take_label,
            "wav": str(wav_path),
            "seed": seed,
            "duration_sec": round(duration, 3),
            "synth_sec": round(synth_sec, 2),
            "rtf": round(rtf, 3),
        }

        if args.select_best:
            assert wm is not None
            qa = _qa_transcribe_take(wav_path, wm, text)
            acoustics = analyse_acoustics(wav_path)
            selection_metrics = score_take_selection(
                text=text,
                duration_sec=duration,
                intelligibility=float(qa.get("intelligibility", 0.0)),
                acoustics=acoustics,
            )
            take["qa"] = qa
            take["acoustics"] = acoustics
            take["selection_metrics"] = selection_metrics

        takes.append(take)
        print(
            f"  {take_label}: {duration:.1f}s  RTF: {rtf:.2f}x  "
            f"seed={seed}  → {wav_path.name}"
        )

    selected_take: str | None = None
    selected_wav: str | None = None
    selection_breakdown: list[dict[str, Any]] = []
    selection_policy: dict[str, Any] = {"enabled": False}

    if args.select_best:
        ranked = rank_take_selection(takes)
        best = ranked[0]
        selected_take = str(best["take"])
        best_src = Path(str(best["wav"]))
        best_dest = run_dir / "best.wav"
        if best_src.resolve() != best_dest.resolve():
            shutil.copyfile(best_src, best_dest)
        selected_wav = str(best_dest)

        print("\n  Best-take scoreboard:")
        for t in ranked:
            sm = t.get("selection_metrics") or {}
            print(
                f"    #{t.get('selection_rank')} {t['take']}  "
                f"score={float(sm.get('final_score', 0.0)):.3f}  "
                f"intel={float(sm.get('intelligibility', 0.0)):.0%}  "
                f"pace={float(sm.get('pacing_sanity', 0.0)):.3f}  "
                f"durfit={float(sm.get('duration_fit', 0.0)):.3f}"
            )
            selection_breakdown.append({
                "take": t["take"],
                "rank": t.get("selection_rank"),
                **sm,
            })

        selection_policy = {
            "enabled": True,
            "method": "weighted_intelligibility_pacing_duration_fit",
            "weights": dict(SELECTION_WEIGHTS),
            "selected_take": selected_take,
            "selected_wav": selected_wav,
            "ranked": selection_breakdown,
        }
        print(f"\n  selected best take: {selected_take}  →  {best_dest.name}")

    total_sec = time.perf_counter() - t0
    takes_meta = build_clone_takes_metadata(
        model=args.model,
        language=language,
        ref=ref,
        x_vector_only=args.x_vector_only,
        tone=args.tone,
        gen_kwargs=gen_kwargs,
        profile_name=profile_name,
        profile_source=profile_source,
        generation_meta=generation_meta,
        voice_generation_defaults=voice_generation_defaults,
        saved_generation_defaults=saved_generation_defaults,
        selection_policy=selection_policy,
        selection_breakdown=selection_breakdown,
        selected_take=selected_take,
        selected_wav=selected_wav,
        text=text,
        prep_sec=prep_sec,
        model_sec=model_sec,
        total_sec=total_sec,
        n_variants=n_variants,
        takes=takes,
    )
    takes_meta_path = run_dir / "takes.meta.json"
    with open(takes_meta_path, "w") as fh:
        json.dump(takes_meta, fh, indent=2)

    (run_dir / "text.txt").write_text(text, encoding="utf-8")

    print("\nDone")
    print(f"  dir:        {run_dir}")
    print(f"  takes:      {n_variants}")
    if selected_take:
        print(f"  best take:  {selected_take}")
    print(f"  meta:       {takes_meta_path}")
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
        "--threads", type=int, default=os.cpu_count() or 8,
        help="CPU threads for torch + faster-whisper (default: all logical cores)",
    )
    p.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype.  auto (default): float32 on CPU; bfloat16 on CUDA Ampere+; "
            "float16 on older CUDA GPUs (Maxwell / Pascal / Volta / Turing)."
        ),
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
    p.add_argument(
        "--interactive", "-i",
        action="store_true",
        help=(
            "After scoring VAD candidates, show a numbered menu so you can listen"
            " to each candidate WAV and pick the best reference segment manually."
            " Bypasses the cached segment selection."
        ),
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
    _ref_grp = sp.add_mutually_exclusive_group(required=True)
    _ref_grp.add_argument("--ref-audio", metavar="PATH",
                          help="Reference voice recording (WAV / MP3 / etc.)")
    _ref_grp.add_argument("--voice", metavar="SLUG",
                          help="Named voice from the registry (see voice-synth list-voices)")
    sp.add_argument("--voice-name", default=None, metavar="SLUG",
                    help="Register (or update) this result as a named voice in /cache/voices/")
    sp.add_argument("--text",       default=None,
                    help="Text to synthesise")
    sp.add_argument("--text-file",  default=None, metavar="FILE",
                    help="Read synthesis text from a file")
    sp.add_argument(
        "--ref-text",
        default=None,
        help=(
            "Transcript of the reference audio. Uses this transcript for prompt build "
            "and skips final whisper transcription + quality gate."
        ),
    )
    sp.add_argument("--ref-start",  type=float, default=None, metavar="SEC",
                    help="Manual reference segment start (seconds)")
    sp.add_argument("--ref-end",    type=float, default=None, metavar="SEC",
                    help="Manual reference segment end (seconds)")
    sp.add_argument(
        "--tone", default=None, metavar="NAME",
        help=(
            "Tone label for this reference clip, e.g. 'neutral', 'sad', 'excited'. "
            "Stored in the prompt meta and indexed in the voice registry so "
            "'voice-synth speak --tone NAME' can select it. "
            "Tone/style comes from the reference audio itself — record or extract "
            "a clip that already sounds the way you want."
        ),
    )
    sp.add_argument("--prompt-prefix", default=None,
                    help="Text prepended verbatim to the synthesis text (will be spoken)")
    sp.add_argument("--prompt-suffix", default=None,
                    help="Text appended verbatim to the synthesis text (will be spoken)")
    sp.add_argument("--variants", type=int, default=1,
                    help="Generate N takes with seeds base_seed + i")
    sp.add_argument(
        "--select-best",
        action="store_true",
        help=(
            "Deterministically rank variants and copy the best to best.wav "
            "(weighted intelligibility + pacing sanity + duration fit)."
        ),
    )
    sp.add_argument(
        "--profile",
        default=None,
        choices=GENERATION_PROFILE_CHOICES,
        help=(
            "Generation profile. If omitted, uses voice default when available, "
            "else balanced."
        ),
    )
    sp.add_argument(
        "--save-profile-default",
        action="store_true",
        help=(
            "Persist resolved profile + sampling defaults to voice.json "
            "generation_defaults (requires --voice or --voice-name)."
        ),
    )
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
    sp.add_argument("--timeout",             type=float, default=None,
                    dest="timeout",
                    help=(
                        "Wall-clock timeout in seconds for the synthesis step. "
                        "Defaults to 4× the CPU ETA estimate. "
                        "Pass 0 to disable entirely."
                    ))
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
    _pr_grp = pr.add_mutually_exclusive_group(required=True)
    _pr_grp.add_argument("--ref-audio", metavar="PATH",
                         help="Reference voice recording (WAV / MP3 / etc.)")
    _pr_grp.add_argument("--voice", metavar="SLUG",
                         help="Named voice from the registry")
    pr.add_argument("--voice-name", default=None, metavar="SLUG",
                    help="Register (or update) result as a named voice in /cache/voices/")
    pr.add_argument(
        "--ref-text",
        default=None,
        help=(
            "Exact transcript of the reference audio. Uses this transcript for prompt build "
            "and skips final whisper transcription + quality gate."
        ),
    )
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
