#!/usr/bin/env python3
"""
voice-clone.py — Clone a voice from a reference WAV and synthesise new speech.

Pipeline (all stages cached by content hash):
  1. Normalise reference audio     (ffmpeg → 16 kHz mono, loudnorm)
  2. Select best clean segment     (Silero VAD, 3–12 s)
  3. Transcribe reference segment  (faster-whisper small, int8/CPU)
  4. Build voice clone prompt      (Qwen3-TTS, pickled for reuse)
  5. Synthesise speech             (Qwen3-TTS generate_voice_clone)
  6. Write output WAV + meta.json

Sub-commands:
  prepare-ref   — Run stages 1–3 only; print cached clip + transcript path
  synth         — Full pipeline; requires --ref-audio and --text[|-file]
  self-test     — End-to-end smoke test using a bundled public-domain clip

Usage:
  ./run voice-clone synth --ref-audio /work/myvoice.wav --text "Hello, world"
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

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_TTS_MODEL   = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_WHISPER     = "small"
WHISPER_COMPUTE     = "int8"
MIN_REF_SECONDS     = 3.0
MAX_REF_SECONDS     = 12.0

# Public-domain demo clip used by the Qwen3-TTS team — reused for self-test
SELF_TEST_REF_URL  = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
SELF_TEST_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
SELF_TEST_SYNTH_TEXT = "The quick brown fox jumps over the lazy dog."


# ── tiny timing helper ────────────────────────────────────────────────────────

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


# ── audio helpers ─────────────────────────────────────────────────────────────

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


# ── Silero VAD ────────────────────────────────────────────────────────────────

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


def pick_best_ref_segment(
    wav_path: Path,
    model,
    get_ts,
    min_len: float = MIN_REF_SECONDS,
    max_len: float = MAX_REF_SECONDS,
) -> tuple[float, float] | None:
    """
    Pick the longest clean continuous speech segment in [min_len, max_len].
    Falls back to (0.0, min(file_dur, max_len)) when nothing qualifies.
    """
    segs = _run_silero(wav_path, model, get_ts)
    file_dur = get_duration(wav_path)

    best: tuple[float, float] | None = None
    best_len = 0.0

    for seg_s, seg_e in segs:
        # Clamp to max_len
        clip_end = min(seg_s + max_len, seg_e, file_dur)
        clip_len = clip_end - seg_s
        if clip_len < min_len:
            continue
        if clip_len > best_len:
            best_len = clip_len
            best = (seg_s, clip_end)

    if best is None and file_dur >= min_len:
        best = (0.0, min(file_dur, max_len))

    return best


# ── faster-whisper transcription ──────────────────────────────────────────────

def transcribe_ref(
    wav_path: Path,
    whisper_model: str,
    num_threads: int,
) -> tuple[str, float]:
    """
    Transcribe *wav_path* with faster-whisper on CPU/int8.
    Returns (transcript_text, avg_log_probability).
    """
    from faster_whisper import WhisperModel  # deferred: not needed for xvec modes

    print(
        f"  loading faster-whisper/{whisper_model} "
        f"(int8, cpu, {num_threads} threads)..."
    )
    wm = WhisperModel(
        whisper_model,
        device="cpu",
        compute_type=WHISPER_COMPUTE,
        cpu_threads=num_threads,
        # download_root is None → uses $HF_HOME or $XDG_CACHE_HOME/huggingface
    )

    segments, info = wm.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
    )

    texts: list[str] = []
    logprobs: list[float] = []
    for seg in segments:  # generator — must iterate to run transcription
        texts.append(seg.text.strip())
        logprobs.append(seg.avg_logprob)

    transcript = " ".join(texts).strip()
    avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0

    print(
        f"  [{info.language} p={info.language_probability:.2f}] "
        f"conf={avg_logprob:.2f}  {transcript!r}"
    )
    return transcript, avg_logprob


# ── pipeline state ────────────────────────────────────────────────────────────

@dataclass
class RefResult:
    ref_hash:        str
    ref_dir:         Path
    normalized:      Path       # 16 kHz mono loudnorm WAV
    segment:         Path       # 24 kHz mono trimmed clip (ready for Qwen3)
    seg_start:       float
    seg_end:         float
    transcript:      str
    transcript_conf: float
    whisper_model:   str


# ── Stage 1–3: prepare-ref ────────────────────────────────────────────────────

def prepare_ref(
    ref_audio:    Path,
    cache:        Path,
    whisper_model: str,
    num_threads:  int,
    ref_start:    float | None,
    ref_end:      float | None,
    x_vector_only: bool,
    force:        bool,
) -> RefResult:
    """
    Run stages 1–3 of the pipeline (normalise → VAD trim → transcribe).
    All outputs are cached under /cache/voice-clone/refs/<hash>/.
    """

    # ── 1. Normalise ──────────────────────────────────────────────────────────
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

    # ── 2. Select best reference segment ─────────────────────────────────────
    print("\n==> [2/3] select reference segment")

    if ref_start is not None and ref_end is not None:
        seg_start, seg_end = ref_start, ref_end
        print(f"  using manual bounds: {seg_start:.2f}s – {seg_end:.2f}s")
    else:
        vad_json = ref_dir / "vad_best_segment.json"
        if vad_json.exists() and not force:
            with open(vad_json) as fh:
                d = json.load(fh)
            seg_start, seg_end = d["seg_start"], d["seg_end"]
            print(f"  [cache hit] segment: {seg_start:.2f}s – {seg_end:.2f}s")
        else:
            print("  loading Silero VAD …")
            hub_dir  = cache / "torch" / "hub"
            vad_model, get_ts = _load_silero(hub_dir)
            with _Timer("Silero VAD"):
                seg = pick_best_ref_segment(normalized, vad_model, get_ts)
            if seg is None:
                dur = get_duration(normalized)
                seg = (0.0, min(dur, MAX_REF_SECONDS))
                print(f"  WARNING: no clean segment found — using first {seg[1]:.1f}s")
            seg_start, seg_end = seg
            with open(vad_json, "w") as fh:
                json.dump({"seg_start": seg_start, "seg_end": seg_end}, fh, indent=2)
            print(f"  best segment: {seg_start:.2f}s – {seg_end:.2f}s  "
                  f"({seg_end - seg_start:.1f}s)")

    segment_wav = ref_dir / "ref_segment.wav"
    if segment_wav.exists() and not force:
        print(f"  [cache hit] segment wav → {segment_wav}")
    else:
        print("  trimming segment …")
        with _Timer("ffmpeg trim"):
            trim_audio(normalized, seg_start, seg_end - seg_start, segment_wav)

    # ── 3. Transcribe ─────────────────────────────────────────────────────────
    print("\n==> [3/3] transcribe reference segment")

    transcript_json = ref_dir / "ref_transcript.json"

    if x_vector_only:
        print("  --x-vector-only: skipping transcription (quality may be lower)")
        transcript, conf = "", 0.0
    elif transcript_json.exists() and not force:
        with open(transcript_json) as fh:
            td = json.load(fh)
        transcript    = td["transcript"]
        conf          = td["avg_logprob"]
        whisper_model = td.get("whisper_model", whisper_model)
        print(f"  [cache hit] ({whisper_model}) {transcript!r}")
    else:
        with _Timer("faster-whisper"):
            transcript, conf = transcribe_ref(segment_wav, whisper_model, num_threads)
        with open(transcript_json, "w") as fh:
            json.dump(
                {
                    "transcript":   transcript,
                    "avg_logprob":  conf,
                    "whisper_model": whisper_model,
                    "timestamp":    datetime.now(timezone.utc).isoformat(),
                },
                fh, indent=2,
            )

    # Persist lightweight pipeline state (useful for debugging / inspection)
    state_path = ref_dir / "pipeline_state.json"
    with open(state_path, "w") as fh:
        json.dump(
            {
                "ref_hash":     ref_hash,
                "ref_audio":    str(ref_audio),
                "seg_start":    seg_start,
                "seg_end":      seg_end,
                "transcript":   transcript,
                "x_vector_only": x_vector_only,
                "updated_at":   datetime.now(timezone.utc).isoformat(),
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
    )


# ── Stage 4: build / load voice clone prompt ─────────────────────────────────

def build_voice_clone_prompt(
    model,
    ref: RefResult,
    cache: Path,
    x_vector_only: bool,
    force: bool,
):
    """
    Call model.create_voice_clone_prompt once and pickle the result.
    Subsequent calls with the same ref + model + mode are instant.
    """
    mode = "xvec" if x_vector_only else "full"
    # Derive a stable tag from the model's HF id (e.g. "Qwen_Qwen3-TTS-12Hz-0.6B-Base")
    model_tag = getattr(model, "name_or_path", "qwen3tts").replace("/", "_")
    prompt_path = (
        cache / "voice-clone" / "prompts"
        / f"{ref.ref_hash}_{model_tag}_{mode}.pkl"
    )

    if prompt_path.exists() and not force:
        print(f"  [cache hit] voice prompt → {prompt_path.name}")
        with open(prompt_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  building voice clone prompt (mode={mode}) …")
    with _Timer("create_voice_clone_prompt"):
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref.segment),
            ref_text=ref.transcript if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "wb") as fh:
        pickle.dump(prompt, fh)
    print(f"  cached → {prompt_path.name}")
    return prompt


# ── Stage 5: synthesis ────────────────────────────────────────────────────────

def load_tts_model(model_name: str, num_threads: int):
    """
    Load Qwen3-TTS-Base for CPU inference.
    Uses float32 (bfloat16/float16 have limited CPU support on most machines).
    """
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
) -> tuple[np.ndarray, int]:
    if seed is not None:
        torch.manual_seed(seed)
    with _Timer("generate_voice_clone"):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
        )
    return wavs[0], sr


# ── Stage 6: output ───────────────────────────────────────────────────────────

def write_output(
    wav: np.ndarray,
    sr: int,
    text: str,
    ref: RefResult,
    model_name: str,
    language: str,
    x_vector_only: bool,
    seed: int | None,
    out_dir: Path,
    timings: dict[str, float],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path  = out_dir / f"voice_clone_{ts}.wav"
    meta_path = out_dir / f"voice_clone_{ts}.meta.json"

    sf.write(str(wav_path), wav, sr)

    duration = len(wav) / sr
    synth_sec = timings.get("synth_sec", 0.0)
    rtf = synth_sec / duration if duration > 0 else 0.0

    meta: dict = {
        "app":        "voice-clone",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model":      model_name,
        "language":   language,
        "x_vector_only": x_vector_only,
        "seed":       seed,
        "text":       text,
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
        "rtf":        round(rtf, 3),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return wav_path, meta_path


# ── self-test ─────────────────────────────────────────────────────────────────

def cmd_self_test(args) -> None:
    cache   = Path(args.cache)
    out_dir = Path(args.out)

    test_dir = cache / "voice-clone" / "self-test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Download once
    ref_wav = test_dir / "self_test_ref.wav"
    if not ref_wav.exists():
        print(f"  downloading self-test reference clip …")
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
        x_vector_only=args.x_vector_only,
        force=False,
    )

    if not args.x_vector_only:
        assert ref.transcript, "Transcript is empty — faster-whisper may have failed"
        ref_text_check = SELF_TEST_REF_TEXT.lower().split()
        got_text_check = ref.transcript.lower().split()
        overlap = sum(w in got_text_check for w in ref_text_check[:5])
        if overlap < 2:
            print(
                f"  WARNING: transcript does not match expected first words "
                f"(got {ref.transcript!r})"
            )

    print("\n--- synth ---")
    t0    = time.perf_counter()
    model = load_tts_model(args.model, args.threads)
    t_model_done = time.perf_counter()

    prompt  = build_voice_clone_prompt(
        model, ref, cache, args.x_vector_only, force=False
    )

    t_synth = time.perf_counter()
    wav, sr = synthesise(SELF_TEST_SYNTH_TEXT, args.language, model, prompt, args.seed)
    synth_sec = time.perf_counter() - t_synth

    total_sec = time.perf_counter() - t0
    duration  = len(wav) / sr

    # Sanity assertions
    assert not np.any(np.isnan(wav)),    "NaN values found in output audio"
    assert 1.0 < duration < 30.0,        f"Output duration {duration:.1f}s is out of range"
    peak = float(np.abs(wav).max())
    assert peak > 5e-4,                  f"Peak amplitude {peak:.5f} suspiciously low (silence?)"

    timings  = {
        "model_load_sec": round(t_model_done - t0, 2),
        "synth_sec":      synth_sec,
        "total_sec":      total_sec,
    }
    wav_path, meta_path = write_output(
        wav, sr, SELF_TEST_SYNTH_TEXT, ref,
        args.model, args.language,
        args.x_vector_only, args.seed,
        out_dir, timings,
    )

    rtf = synth_sec / duration
    print(f"\n  self-test PASSED ✓")
    print(f"  output:   {wav_path}")
    print(f"  duration: {duration:.1f}s  RTF: {rtf:.2f}x  total: {total_sec:.1f}s")
    print(f"  meta:     {meta_path}")


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_prepare_ref(args) -> None:
    ref = prepare_ref(
        ref_audio=Path(args.ref_audio),
        cache=Path(args.cache),
        whisper_model=args.whisper_model,
        num_threads=args.threads,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        x_vector_only=args.x_vector_only,
        force=args.force,
    )
    print(f"\n  ref_dir:    {ref.ref_dir}")
    print(f"  segment:    {ref.seg_start:.2f}s – {ref.seg_end:.2f}s"
          f"  ({ref.seg_end - ref.seg_start:.1f}s)")
    print(f"  segment wav: {ref.segment}")
    print(f"  transcript: {ref.transcript!r}")


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
        x_vector_only=args.x_vector_only,
        force=args.force,
    )
    prep_sec = time.perf_counter() - t0

    # --ref-text overrides auto-transcription (e.g. you already have it)
    if args.ref_text:
        ref.transcript = args.ref_text.strip()
        print(f"  --ref-text override: {ref.transcript!r}")

    # Resolve synthesis text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        text = args.text or ""

    # Style shaping via optional prefix/suffix
    if args.prompt_prefix:
        text = args.prompt_prefix.strip() + " " + text
    if args.prompt_suffix:
        text = text + " " + args.prompt_suffix.strip()

    if not text:
        print("ERROR: no synthesis text provided (use --text or --text-file)", file=sys.stderr)
        sys.exit(1)

    print(f"\n==> synthesising ({len(text)} chars): {text!r}")

    # Stage 4: load model + build prompt
    t_model = time.perf_counter()
    model   = load_tts_model(args.model, args.threads)
    prompt  = build_voice_clone_prompt(model, ref, cache, args.x_vector_only, args.force)
    model_sec = time.perf_counter() - t_model

    # Stage 5: generate
    t_synth = time.perf_counter()
    wav, sr = synthesise(text, args.language, model, prompt, args.seed)
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
        args.model, args.language,
        args.x_vector_only, args.seed,
        out_dir, timings,
    )

    print(f"\nDone")
    print(f"  WAV:       {wav_path}")
    print(f"  meta:      {meta_path}")
    print(f"  duration:  {duration:.1f}s")
    print(f"  RTF:       {rtf:.2f}x  (synth {synth_sec:.1f}s / audio {duration:.1f}s)")
    print(f"  model load: {model_sec:.1f}s  (cached prompt skips this)")
    print(f"  total:     {total_sec:.1f}s")


# ── argument parser ───────────────────────────────────────────────────────────

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
        "--language", default="English",
        help="Synthesis language (e.g. English, Chinese, French …)",
    )
    p.add_argument(
        "--x-vector-only", action="store_true",
        help=(
            "Build the voice prompt from speaker embedding only — no ref_text needed. "
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
            "Voice cloning with Qwen3-TTS-Base + faster-whisper (CPU-first, "
            "all stages cached)"
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
    sp.add_argument("--ref-audio", required=True,
                    help="Reference voice recording (WAV / MP3 / etc.)")
    sp.add_argument("--text",      default=None,
                    help="Text to synthesise")
    sp.add_argument("--text-file", default=None, metavar="FILE",
                    help="Read synthesis text from a file")
    sp.add_argument("--ref-text",  default=None,
                    help="Transcript of the reference audio (skips auto-transcription)")
    sp.add_argument("--ref-start", type=float, default=None, metavar="SEC",
                    help="Manual reference segment start (seconds)")
    sp.add_argument("--ref-end",   type=float, default=None, metavar="SEC",
                    help="Manual reference segment end (seconds)")
    sp.add_argument("--prompt-prefix", default=None,
                    help="Prepend to synthesis text for style steering")
    sp.add_argument("--prompt-suffix", default=None,
                    help="Append to synthesis text for style steering")
    sp.add_argument("--out", default="/work",
                    help="Output directory for WAV + meta.json")
    _add_common(sp)

    # ── prepare-ref ────────────────────────────────────────────────────────────
    pr = sub.add_parser(
        "prepare-ref",
        help="Run stages 1–3 only (normalise, VAD-trim, transcribe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pr.add_argument("--ref-audio", required=True,
                    help="Reference voice recording")
    pr.add_argument("--ref-start", type=float, default=None, metavar="SEC")
    pr.add_argument("--ref-end",   type=float, default=None, metavar="SEC")
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


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    # Propagate thread count to OpenMP-based libraries
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    # Point Torch Hub at our shared cache mount
    cache    = Path(args.cache)
    hub_dir  = cache / "torch" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    # XDG_CACHE_HOME=/cache is set in the container, so HF downloads land there.

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
