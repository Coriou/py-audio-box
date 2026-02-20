#!/usr/bin/env python3
"""
voice-synth.py — DevX-first synthesis rig built on cached voice-clone prompts.

Sub-commands:
  list-voices    List cached voice prompts available for synthesis
  speak          Synthesise text from a cached voice prompt (with variants, QA, chunking)
  design-voice   Create a reusable voice from a natural-language description
                 (VoiceDesign → Clone workflow; needs the 1.7B-VoiceDesign model)

Usage:
  ./run voice-synth list-voices
  ./run voice-synth speak --voice <id> --text "Hello, world"
  ./run voice-synth speak --voice <id> --text-file /work/script.txt --variants 4
  ./run voice-synth design-voice \\
        --instruct "Calm male narrator, mid-40s, warm and unhurried" \\
        --ref-text "The forest was silent except for the distant call of birds."
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL    = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_DESIGN_MODEL  = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_WHISPER       = "small"
PROMPT_SCHEMA_VERSION = 1   # must match voice-clone.py

# Hard ceiling on generated tokens.  At 12 Hz this allows ~341 s of audio,
# preventing runaway generation when EOS is unreliable on CPU.
MAX_NEW_TOKENS_DEFAULT = 4096

QWEN3_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

_LANGID_TO_QWEN: dict[str, str] = {
    "zh": "Chinese",  "en": "English",    "ja": "Japanese",
    "ko": "Korean",   "de": "German",     "fr": "French",
    "ru": "Russian",  "pt": "Portuguese", "es": "Spanish",
    "it": "Italian",
}

# Silence (ms) inserted between auto-chunked sentences when concatenating
CHUNK_SILENCE_MS = 300


# ── tiny timing helper ─────────────────────────────────────────────────────────

class _Timer:
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

def get_duration_sf(path: Path) -> float:
    info = sf.info(str(path))
    return info.duration



def concat_wavs_simple(wav_paths: list[Path], dest: Path,
                        silence_ms: int = CHUNK_SILENCE_MS,
                        sample_rate: int = 24_000) -> None:
    """
    Pure-numpy concatenation of WAV files with silence between them.
    More reliable across ffmpeg versions for simple concat.
    """
    import soundfile as sf
    chunks = []
    silence = np.zeros(int(sample_rate * silence_ms / 1000.0), dtype=np.float32)
    for i, p in enumerate(wav_paths):
        data, sr = sf.read(str(p), dtype="float32")
        if i > 0:
            chunks.append(silence)
        chunks.append(data)
    combined = np.concatenate(chunks)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), combined, sample_rate)


# ── text chunking ──────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """
    Split text into sentence-level chunks for chunked synthesis.
    Sentences are split at ". ", "! ", "? " boundaries.
    Chunks longer than max_chars are further split at commas.
    """
    # Split on sentence-ending punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            chunks.append(sent)
        else:
            # Split long sentences at commas
            sub = re.split(r',\s+', sent)
            buf = ""
            for part in sub:
                part = part.strip()
                if buf and len(buf) + len(part) + 2 > max_chars:
                    chunks.append(buf)
                    buf = part
                else:
                    buf = (buf + ", " + part) if buf else part
            if buf:
                chunks.append(buf)
    return chunks or [text]


# ── language detection ─────────────────────────────────────────────────────────

def detect_language_from_text(text: str) -> str | None:
    try:
        import langid  # type: ignore
        iso, _ = langid.classify(text)
        return _LANGID_TO_QWEN.get(iso)  # None when unsupported
    except Exception:
        return None


def resolve_language(flag: str, ref_language: str, text: str) -> str:
    if flag != "Auto":
        return flag
    if len(text.split()) >= 3:
        detected = detect_language_from_text(text)
        if detected and detected in QWEN3_LANGUAGES and detected != "Auto":
            return detected
    if ref_language and ref_language not in ("Auto", ""):
        return ref_language
    return "English"


# ── style presets ──────────────────────────────────────────────────────────────

def load_style_presets(styles_path: Path) -> dict[str, dict[str, str]]:
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
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
) -> str:
    style_prefix = style_suffix = ""
    if style_name:
        presets = load_style_presets(styles_path)
        if style_name not in presets:
            available = ", ".join(sorted(presets)) or "(none)"
            print(f"  WARNING: style '{style_name}' not found. Available: {available}",
                  file=sys.stderr)
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


# ── dtype helper ──────────────────────────────────────────────────────────────

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
    "float16":  torch.float16,
}


# ── device & dtype helpers ─────────────────────────────────────────────────────

def _get_device() -> str:
    """
    Select the best available compute device.

    Resolution order:
      1. ``TORCH_DEVICE`` env var  — explicit override (e.g. ``"cpu"``, ``"cuda:0"``).
                                     Set automatically by docker-compose.gpu.yml.
      2. CUDA                      — when a GPU is present and torch was built with
                                     CUDA support (i.e. the CUDA image variant).
      3. CPU                       — universal fallback.
    """
    override = os.getenv("TORCH_DEVICE", "").strip()
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _best_dtype(device: str, dtype_str: str) -> torch.dtype:
    """
    Resolve ``dtype_str`` to a concrete ``torch.dtype``.

    When ``dtype_str == "auto"`` (the default):
      - CPU              → ``float32``   (native; bfloat16 on CPU upcasts every
                                          op to float32 internally — same compute
                                          cost, extra cast overhead, no benefit)
      - CUDA SM < 8.0    → ``float16``   (Maxwell / Pascal / Volta / Turing)
      - CUDA SM ≥ 8.0    → ``bfloat16``  (Ampere, Ada Lovelace, Hopper …)

    Explicit dtype strings ("bfloat16", "float32", "float16") are returned as-is.
    """
    if dtype_str != "auto":
        return _DTYPE_MAP[dtype_str]
    if not device.startswith("cuda"):
        return torch.float32                    # CPU: float32 is fastest (no cast overhead)
    idx = int(device.split(":")[-1]) if ":" in device else 0
    major, _ = torch.cuda.get_device_capability(idx)
    return torch.bfloat16 if major >= 8 else torch.float16  # Ampere+ vs older


# ── model loading ──────────────────────────────────────────────────────────────

def load_tts_model(model_name: str, num_threads: int, dtype_str: str = "auto"):
    """
    Load a Qwen3-TTS model, targeting the best available device.

    dtype_str:
      "auto" (default)  — float32 on CPU (native, no cast overhead); bfloat16 on
                          CUDA Ampere+ (SM ≥ 8.0); float16 on older CUDA GPUs
                          (Maxwell / Pascal / Volta / Turing).
      "bfloat16"        — for CUDA SM ≥ 8.0 (Ampere+). On CPU this adds cast
                          overhead without benefit; avoid unless testing.
      "float32"         — safest and fastest on CPU. Also a useful debug fallback
                          if you see NaN/quality issues on CUDA.
      "float16"         — recommended for CUDA GPUs with SM < 8.0.

    attn_implementation="sdpa" uses PyTorch's fused scaled-dot-product attention,
    which is faster than the default "eager" path on both CPU and CUDA.
    """
    from qwen_tts import Qwen3TTSModel
    device = _get_device()
    dtype  = _best_dtype(device, dtype_str)
    torch.set_num_threads(num_threads)
    print(f"  loading {model_name} ({device}, {dtype}, threads={num_threads}) …")
    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )


def synthesise(
    text: str,
    language: str,
    model,
    prompt,
    seed: int | None,
    gen_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    if seed is not None:
        torch.manual_seed(seed)
    with _Timer("generate_voice_clone"):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            **(gen_kwargs or {}),
        )
    return wavs[0], sr


# ── QA: whisper-based intelligibility check ────────────────────────────────────

def _word_overlap_ratio(ref_words: list[str], hyp_words: list[str]) -> float:
    """Fraction of ref words found in hyp (order-insensitive, lower-cased)."""
    if not ref_words:
        return 0.0
    hyp_set = set(hyp_words)
    matched = sum(1 for w in ref_words if w in hyp_set)
    return matched / len(ref_words)


def _cuda_ctranslate2_compute_type() -> str:
    """
    Return the most efficient CTranslate2 compute_type for the current CUDA GPU.

    CTranslate2 capability requirements:
      float16  → SM 7.0+ (Volta and newer)
      int8     → SM 6.1+ (Pascal and newer)
      float32  → any device
    """
    cc = torch.cuda.get_device_capability()
    if cc >= (7, 0):
        return "float16"
    if cc >= (6, 1):
        return "int8"
    return "float32"


def _ctranslate2_device() -> str:
    """
    Return the device to use for CTranslate2 / faster-whisper.

    CTranslate2 compiled with CUDA 12.x (cu124/cu126) only ships CUDA kernels
    for Volta SM 7.0+.  Older GPUs will hit ``cudaErrorNoKernelImageForDevice``
    at runtime.  Fall back to CPU for pre-Volta hardware.
    """
    device = _get_device()
    if device.startswith("cuda") and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc < (7, 0):
            return "cpu"
    return device


def qa_transcribe(wav_path: Path, whisper_model: str, num_threads: int,
                  target_text: str) -> dict[str, Any]:
    """
    Run faster-whisper on *wav_path* and return a QA dict:
      transcript, intelligibility (word overlap), duration_sec.
    Uses the best compute type for the current device.
    """
    try:
        from faster_whisper import WhisperModel
        device       = _ctranslate2_device()
        compute_type = _cuda_ctranslate2_compute_type() if device.startswith("cuda") else "int8"
        wm = WhisperModel(whisper_model, device=device,
                          compute_type=compute_type, cpu_threads=num_threads)
        segs, _info = wm.transcribe(str(wav_path), beam_size=2, vad_filter=True)
        transcript = " ".join(s.text.strip() for s in segs).strip()
    except Exception as exc:
        return {"transcript": "", "intelligibility": 0.0,
                "duration_sec": 0.0, "error": str(exc)}

    ref_words = re.findall(r'\w+', target_text.lower())
    hyp_words = re.findall(r'\w+', transcript.lower())
    overlap   = _word_overlap_ratio(ref_words, hyp_words)
    duration  = get_duration_sf(wav_path)

    return {
        "transcript":     transcript,
        "intelligibility": round(overlap, 4),
        "duration_sec":   round(duration, 3),
    }


# ── voice registry ─────────────────────────────────────────────────────────────

def _voice_registry(cache: Path):
    """Lazily import VoiceRegistry from the shared lib."""
    _lib = str(Path(__file__).resolve().parent.parent.parent / "lib")
    if _lib not in sys.path:
        sys.path.insert(0, _lib)
    from voices import VoiceRegistry  # type: ignore
    return VoiceRegistry(cache)


# ── prompts directory helpers ──────────────────────────────────────────────────

def list_all_voices(cache: Path) -> list[dict[str, Any]]:
    """
    Return all available voice prompts from two sources:
      1.  Named voices  (``/cache/voices/<slug>/``) — kind="named"
      2.  Legacy prompts (``/cache/voice-clone/prompts/*.meta.json``) — kind="legacy"

    Named voices that have at least one prompt appear with kind="named".
    Named voices that exist but have no prompts are still returned so
    ``list-voices`` can show their status and suggest the next step.
    """
    results: list[dict[str, Any]] = []
    # Deduplicate by PKL *stem* (filename without .pkl) so that prompts registered
    # under a named voice (in /cache/voices/<slug>/prompts/) are not also shown in
    # the legacy section (in /cache/voice-clone/prompts/), even though they live in
    # different directories but share the same filename.
    seen_stems: set[str] = set()

    # 1. Named voices
    reg = _voice_registry(cache)
    for v in reg.list_voices():
        slug    = v["slug"]
        ref     = v.get("ref") or {}
        best    = reg.best_prompt(slug)
        # Mark all registered prompt stems so legacy scan skips them
        for pkey in (v.get("prompts") or {}).values():
            seen_stems.add(Path(pkey).stem)
        results.append({
            "kind":        "named",
            "id":          slug,
            "slug":        slug,
            "display_name": v.get("display_name", slug),
            "description": v.get("description", ""),
            "pkl":         str(best) if best else None,
            "pkl_exists":  best is not None and best.exists(),
            "model":       v.get("prompts") and next(iter(v["prompts"]), "") or "?",
            "ref_language": ref.get("language", "?"),
            "duration_sec": ref.get("duration_sec", 0.0),
            "transcript":  ref.get("transcript", ""),
            "created_at":  v.get("created_at", ""),
            "_ready":      v.get("_ready", False),
            "_prompt_count": v.get("_prompt_count", 0),
            "_has_ref":    v.get("_has_ref", False),
        })

    # 2. Legacy prompts not already covered by named voices
    prompts_dir = cache / "voice-clone" / "prompts"
    if prompts_dir.exists():
        # Load with meta.json when available, fall back to pkl-only entries.
        # Note: files are named "<stem>.meta.json" so we must strip ".meta.json"
        # to get the stem — not use Path.stem which would only strip ".json".
        all_pkl_stems = {p.stem for p in prompts_dir.glob("*.pkl")}
        for meta_file in sorted(prompts_dir.glob("*.meta.json")):
            stem = meta_file.name.removesuffix(".meta.json")
            pkl  = meta_file.parent / f"{stem}.pkl"
            if stem in seen_stems:
                continue
            all_pkl_stems.discard(stem)  # covered by meta
            try:
                with open(meta_file) as fh:
                    m = json.load(fh)
            except Exception:
                continue
            results.append({
                "kind":        "legacy",
                "id":          stem,
                "slug":        None,
                "display_name": stem,
                "pkl":         str(pkl) if pkl.exists() else None,
                "pkl_exists":  pkl.exists(),
                "model":       m.get("model", "?"),
                "ref_language": m.get("ref_language_detected", "?"),
                "duration_sec": m.get("segment_duration_sec", 0.0),
                "transcript":  m.get("transcript", ""),
                "created_at":  m.get("created_at", ""),
                "_ready":      pkl.exists(),
                "_prompt_count": 1 if pkl.exists() else 0,
                "_has_ref":    True,
            })
            if pkl.exists():
                seen_stems.add(stem)

        # pkl-only legacy prompts (no meta.json)
        for stem in sorted(all_pkl_stems):
            pkl = prompts_dir / f"{stem}.pkl"
            if stem in seen_stems or not pkl.exists():
                continue
            results.append({
                "kind":        "legacy",
                "id":          stem,
                "slug":        None,
                "display_name": stem,
                "pkl":         str(pkl),
                "pkl_exists":  True,
                "model":       "?",
                "ref_language": "?",
                "duration_sec": 0.0,
                "transcript":  "",
                "created_at":  "",
                "_ready":      True,
                "_prompt_count": 1,
                "_has_ref":    True,
            })

    results.sort(key=lambda v: v["created_at"], reverse=True)
    return results


# keep old name as an alias for any callers
def list_prompts(cache: Path) -> list[dict[str, Any]]:
    return list_all_voices(cache)


def resolve_voice(voice_arg: str, cache: Path, tone: str | None = None) -> tuple[str, str]:
    """
    Given --voice <arg>, return (pkl_path_str, voice_id).
    Resolution order:
      1. Absolute path ending in ``.pkl`` — used directly (tone ignored)
      2. Exact named-voice slug in ``/cache/voices/<slug>/``
         - If *tone* given: looks up the tone-labelled prompt via voice.json["tones"]
         - Otherwise: returns the most recently written prompt (best_prompt)
      3. Unambiguous prefix match against named voices
      4. Exact stem or unambiguous prefix in legacy ``/cache/voice-clone/prompts/``
    """
    # 1. Direct path
    if voice_arg.endswith(".pkl"):
        p = Path(voice_arg)
        if p.exists():
            return str(p), p.stem
        print(f"ERROR: pkl not found: {voice_arg}", file=sys.stderr)
        sys.exit(1)

    reg = _voice_registry(cache)

    # 2. Exact named-voice slug
    if reg.exists(voice_arg):
        if tone:
            pkl = reg.prompt_for_tone(voice_arg, tone)
            if pkl is None:
                available = reg.list_tones(voice_arg)
                if available:
                    print(
                        f"ERROR: voice '{voice_arg}' has no prompt for tone '{tone}'.\n"
                        f"  Available tones: {', '.join(sorted(available))}\n"
                        f"  Build one with:\n"
                        f"    ./run voice-clone synth --voice {voice_arg} "
                        f"--tone {tone} --text '...' "
                        f"  (use a reference clip that already sounds {tone!r})",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"ERROR: voice '{voice_arg}' has no tone-labelled prompts.\n"
                        f"  Build one with:\n"
                        f"    ./run voice-clone synth --voice {voice_arg} "
                        f"--tone {tone} --text '...'\n"
                        f"  Note: tone comes from the reference audio, not from text.\n"
                        f"  Record/extract a clip that already sounds {tone!r}.",
                        file=sys.stderr,
                    )
                sys.exit(1)
            return str(pkl), voice_arg
        best = reg.best_prompt(voice_arg)
        if best is None:
            print(
                f"ERROR: voice '{voice_arg}' exists but has no prompts yet.\n"
                f"  Build one with:\n"
                f"    ./run voice-clone synth --voice {voice_arg} --text 'Hello'",
                file=sys.stderr,
            )
            sys.exit(1)
        return str(best), voice_arg

    # 3. Prefix match against named voices
    named = [v["slug"] for v in reg.list_voices()]
    name_matches = [s for s in named if s.startswith(voice_arg)]
    if len(name_matches) == 1:
        return resolve_voice(name_matches[0], cache, tone=tone)  # tail-recurse with full slug
    if len(name_matches) > 1:
        print(
            f"ERROR: ambiguous voice prefix '{voice_arg}' — matches named voices:\n"
            + "\n".join(f"  {m}" for m in sorted(name_matches)),
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Legacy: full or prefix match against prompt stems (*.meta.json OR *.pkl)
    prompts_dir = cache / "voice-clone" / "prompts"
    if prompts_dir.exists():
        all_ids = sorted({p.stem for p in prompts_dir.glob("*.pkl")})
    else:
        all_ids = []
    matches = [i for i in all_ids if i == voice_arg or i.startswith(voice_arg)]
    if len(matches) == 1:
        pkl = prompts_dir / f"{matches[0]}.pkl"
        if not pkl.exists():
            print(f"ERROR: pkl missing for voice '{matches[0]}'", file=sys.stderr)
            sys.exit(1)
        return str(pkl), matches[0]
    if len(matches) == 0:
        print(
            f"ERROR: no voice found matching '{voice_arg}'.\n"
            f"  Run `./run voice-synth list-voices` to see available voices.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Ambiguous legacy
    print(
        f"ERROR: ambiguous voice prefix '{voice_arg}' — legacy matches:\n"
        + "\n".join(f"  {m}" for m in sorted(matches)),
        file=sys.stderr,
    )
    sys.exit(1)


# ── sub-command implementations ────────────────────────────────────────────────

def cmd_list_voices(args) -> None:
    cache  = Path(args.cache)
    voices = list_all_voices(cache)

    named  = [v for v in voices if v["kind"] == "named"]
    legacy = [v for v in voices if v["kind"] == "legacy"]

    if not voices:
        print("No cached voice prompts found.")
        print("  Extract clips:   ./run voice-split --url '...' --voice-name my-voice")
        print("  Register a file: ./run voice-clone prepare-ref --ref-audio /work/clip.wav"
              " --voice-name my-voice")
        return

    if named:
        print(f"\n\033[1mNAMED VOICES\033[0m  ({len(named)})")
        print(f"  {'NAME':<28}  {'LANG':<10}  {'DUR':>5}  {'PROMPTS':>7}  STATUS")
        print("  " + "-" * 72)
        list_reg = _voice_registry(cache)
        for v in named:
            status = (
                "\033[32mready\033[0m" if v["_ready"]
                else ("\033[33mno prompts\033[0m  run: voice-clone synth --voice " + v["slug"])
                if v["_has_ref"]
                else "\033[31mno ref\033[0m  run: voice-clone prepare-ref --voice " + v["slug"]
            )
            print(
                f"  {v['slug']:<28}  {v['ref_language']:<10}"
                f"  {v['duration_sec']:>4.1f}s  {v['_prompt_count']:>7}  {status}"
            )
            if v["transcript"]:
                preview = textwrap.shorten(v["transcript"], width=68, placeholder="…")
                print(f"    \033[2m{preview}\033[0m")
            # Show registered tones if any
            tones = list_reg.list_tones(v["slug"])
            if tones:
                tone_list = "  ".join(f"{t}" for t in sorted(tones))
                print(f"    \033[2mtones: {tone_list}\033[0m")

    if legacy:
        print(f"\n\033[1mLEGACY PROMPTS\033[0m  ({len(legacy)})  "
              "(no name — tip: run voice-clone synth --voice-name <slug> to register)")
        print(f"  {'ID (truncated)':<50}  {'LANG':<10}  {'DUR':>5}")
        print("  " + "-" * 72)
        for v in legacy:
            short_id = v["id"]
            if len(short_id) > 48:
                short_id = short_id[:22] + "…" + short_id[-24:]
            print(
                f"  {short_id:<50}  {v['ref_language']:<10}  {v['duration_sec']:>4.1f}s"
            )
            if v["transcript"]:
                preview = textwrap.shorten(v["transcript"], width=68, placeholder="…")
                print(f"    \033[2m{preview}\033[0m")

    total = len(named) + len(legacy)
    print(f"\n{total} voice(s): {len(named)} named, {len(legacy)} legacy")
    print(f"Registry: {cache}/voices/    Legacy: {cache}/voice-clone/prompts/")


def cmd_speak(args) -> None:
    cache   = Path(args.cache)
    out_dir = Path(args.out)

    # Resolve prompt
    tone = getattr(args, "tone", None)
    pkl_path, voice_id = resolve_voice(args.voice, cache, tone=tone)
    print(f"\n  voice: {voice_id}" + (f"  tone: {tone}" if tone else ""))

    # Load meta (for ref_language, etc.)
    meta_path = Path(pkl_path).with_suffix(".meta.json")
    voice_meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            voice_meta = json.load(fh)

    ref_language = voice_meta.get("ref_language_detected", "English")

    # Resolve text
    if args.text_file:
        raw_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        raw_text = args.text or ""

    if not raw_text:
        print("ERROR: no text provided (use --text or --text-file)", file=sys.stderr)
        sys.exit(1)

    # Resolve language; normalise to lowercase so the model accepts it
    language = resolve_language(args.language, ref_language, raw_text).lower()
    print(f"  language: {language}  (ref detected: {ref_language})")

    # Synthesis text — no style-prefix injection for voice clone.
    # Tone/delivery comes from the reference audio used to build the prompt.
    # Select the right prompt via --tone; use --prompt-prefix for explicit text inserts.
    text = raw_text

    # Chunk if requested
    if args.chunk:
        text_chunks = chunk_text(text)
        print(f"  auto-chunked into {len(text_chunks)} piece(s)")
    else:
        text_chunks = [text]

    # Generation kwargs
    gen_kwargs: dict[str, Any] = {}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    # Always set a ceiling: without it the model may run indefinitely on CPU
    # when EOS is unreliable.  User can raise or lower with --max-new-tokens.
    gen_kwargs["max_new_tokens"] = (
        args.max_new_tokens if args.max_new_tokens is not None
        else MAX_NEW_TOKENS_DEFAULT
    )

    # Load model + prompt
    t0    = time.perf_counter()
    model = load_tts_model(args.model, args.threads, args.dtype)
    with open(pkl_path, "rb") as fh:
        prompt = pickle.load(fh)
    load_sec = time.perf_counter() - t0

    # Output directory — one subdir per voice, then a timestamped run dir
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / voice_id / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    n_variants = max(1, args.variants)
    all_takes: list[dict] = []

    for variant in range(n_variants):
        seed = (args.seed + variant) if args.seed is not None else None
        take_label = f"take_{variant + 1:02d}"

        if len(text_chunks) == 1:
            # Single chunk — no concat needed
            t_synth = time.perf_counter()
            wav, sr = synthesise(text_chunks[0], language, model, prompt, seed, gen_kwargs)
            synth_sec = time.perf_counter() - t_synth

            wav_path = run_dir / f"{take_label}.wav"
            sf.write(str(wav_path), wav, sr)
            duration = len(wav) / sr

        else:
            # Multi-chunk: synthesise each, concat
            chunk_wavs: list[Path] = []
            synth_sec = 0.0
            sr = 24_000
            for ci, chunk in enumerate(text_chunks):
                t_synth = time.perf_counter()
                w, sr   = synthesise(chunk, language, model, prompt, seed, gen_kwargs)
                synth_sec += time.perf_counter() - t_synth
                chunk_path = run_dir / f"{take_label}_chunk{ci:02d}.wav"
                sf.write(str(chunk_path), w, sr)
                chunk_wavs.append(chunk_path)

            wav_path = run_dir / f"{take_label}.wav"
            concat_wavs_simple(chunk_wavs, wav_path, sample_rate=sr)

            # Clean up chunk files
            for cp in chunk_wavs:
                cp.unlink(missing_ok=True)

            wav_data, _sr = sf.read(str(wav_path), dtype="float32")
            duration = len(wav_data) / sr

        rtf = synth_sec / duration if duration > 0 else 0.0
        take_info: dict[str, Any] = {
            "take":        take_label,
            "wav":         str(wav_path),
            "seed":        seed,
            "duration_sec": round(duration, 3),
            "synth_sec":   round(synth_sec, 2),
            "rtf":         round(rtf, 3),
        }

        # QA pass
        if args.qa:
            print(f"  QA transcribing {take_label} …")
            qa = qa_transcribe(wav_path, args.whisper_model,
                               args.threads, raw_text)
            take_info["qa"] = qa
            intel = qa.get("intelligibility", 0.0)
            print(f"    intelligibility: {intel:.0%}  transcript: {qa.get('transcript', '')!r}")

        all_takes.append(take_info)
        print(
            f"  {take_label}: {duration:.1f}s  RTF: {rtf:.2f}x  seed={seed}  "
            f"→ {wav_path.name}"
        )

    # If QA was run, print ranked scoreboard
    if args.qa and len(all_takes) > 1:
        ranked = sorted(all_takes, key=lambda t: t.get("qa", {}).get("intelligibility", 0.0),
                        reverse=True)
        print("\n  QA scoreboard (by intelligibility):")
        for rank, t in enumerate(ranked, 1):
            intel = t.get("qa", {}).get("intelligibility", 0.0)
            print(f"    #{rank} {t['take']}  intelligibility={intel:.0%}  "
                  f"dur={t['duration_sec']:.1f}s")

    # Write takes meta
    total_sec = time.perf_counter() - t0
    takes_meta = {
        "app":            "voice-synth",
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "voice_id":       voice_id,
        "model":          args.model,
        "language":       language,
        "ref_language":   ref_language,
        "original_text":  raw_text,
        "text":           text,
        "tone":           tone,
        "generation_kwargs": gen_kwargs,
        "chunked":        args.chunk,
        "n_chunks":       len(text_chunks),
        "variants":       n_variants,
        "load_sec":       round(load_sec, 2),
        "total_sec":      round(total_sec, 2),
        "takes":          all_takes,
    }
    takes_meta_path = run_dir / "takes.meta.json"
    with open(takes_meta_path, "w") as fh:
        json.dump(takes_meta, fh, indent=2)

    (run_dir / "text.txt").write_text(raw_text, encoding="utf-8")

    print(f"\nDone  →  {run_dir}")
    print(f"  {n_variants} take(s) written   total: {total_sec:.1f}s")
    print(f"  meta: {takes_meta_path}")


def cmd_design_voice(args) -> None:
    """
    VoiceDesign → Clone workflow.

    1. Load the VoiceDesign model.
    2. Synthesise a short ref clip using instruct + ref_text.
    3. Save the clip to /cache/voice-clone/designed_refs/<hash>/.
    4. Load the Base clone model and build a reusable voice_clone_prompt.
    5. Write prompt .pkl + .meta.json alongside existing prompts so
       voice-synth speak can use it immediately.
    """
    cache   = Path(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_text    = args.ref_text.strip()
    instruct    = args.instruct.strip()
    language    = args.language if args.language != "Auto" else "English"

    # ── 1. Generate design reference ──────────────────────────────────────────
    print(f"\n==> [1/3] generate VoiceDesign reference clip")
    print(f"  design model: {args.design_model}")
    print(f"  instruct:     {instruct!r}")
    print(f"  ref_text:     {ref_text!r}")
    print(f"  language:     {language}")
    print(f"  NOTE: VoiceDesign model is 1.7B — this may be slow on CPU.")

    t0           = time.perf_counter()
    design_model = load_tts_model(args.design_model, args.threads, args.dtype)

    with _Timer("generate_voice_design"):
        wavs, sr = design_model.generate_voice_design(
            text=ref_text,
            language=language,
            instruct=instruct,
        )
    design_wav = wavs[0]
    design_sec = time.perf_counter() - t0
    duration   = len(design_wav) / sr
    print(f"  designed ref: {duration:.1f}s  ({design_sec:.1f}s elapsed)")

    # ── 2. Save designed ref to cache ─────────────────────────────────────────
    print(f"\n==> [2/3] cache designed reference audio")
    content_hash = hashlib.sha256(
        (instruct + "|" + ref_text + "|" + language).encode()
    ).hexdigest()[:16]

    design_dir = cache / "voice-clone" / "designed_refs" / content_hash
    design_dir.mkdir(parents=True, exist_ok=True)

    design_wav_path = design_dir / "design_ref.wav"
    sf.write(str(design_wav_path), design_wav, sr)
    with open(design_dir / "design_meta.json", "w") as fh:
        json.dump(
            {
                "content_hash": content_hash,
                "instruct":     instruct,
                "ref_text":     ref_text,
                "language":     language,
                "design_model": args.design_model,
                "duration_sec": round(duration, 3),
                "created_at":   datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )
    print(f"  saved → {design_wav_path}")
    del design_model  # free memory before loading clone model

    # ── 3. Build clone prompt ─────────────────────────────────────────────────
    print(f"\n==> [3/3] build voice-clone prompt from designed ref")
    print(f"  clone model: {args.clone_model}")

    clone_model = load_tts_model(args.clone_model, args.threads, args.dtype)
    model_tag   = getattr(clone_model, "name_or_path", args.clone_model).replace("/", "_")
    stem        = f"{content_hash}_{model_tag}_designed_full_v{PROMPT_SCHEMA_VERSION}"
    prompts_dir = cache / "voice-clone" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{stem}.pkl"
    meta_path   = prompts_dir / f"{stem}.meta.json"

    with _Timer("create_voice_clone_prompt"):
        prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(design_wav, sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    with open(prompt_path, "wb") as fh:
        pickle.dump(prompt, fh)

    with open(meta_path, "w") as fh:
        json.dump(
            {
                "prompt_id":               stem,
                "schema_version":          PROMPT_SCHEMA_VERSION,
                "model":                   model_tag,
                "mode":                    "designed_full",
                "ref_hash":                content_hash,
                "transcript":              ref_text,
                "ref_language_detected":   language,
                "ref_language_probability": 1.0,
                "instruct":                instruct,
                "design_model":            args.design_model,
                "segment_duration_sec":    round(duration, 3),
                "created_at":              datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )

    total_sec = time.perf_counter() - t0

    # Register to named voice if requested
    voice_name = getattr(args, "voice_name", None)
    if voice_name:
        _lib = str(Path(__file__).resolve().parent.parent.parent / "lib")
        if _lib not in sys.path:
            sys.path.insert(0, _lib)
        from voices import VoiceRegistry, validate_slug  # type: ignore
        slug = validate_slug(voice_name)
        reg  = VoiceRegistry(Path(args.cache))
        if not reg.exists(slug):
            reg.create(slug, display_name=slug, source={
                "type":     "designed",
                "instruct": instruct,
                "ref_text": ref_text,
            })
        ref_meta = {
            "hash":            content_hash,
            "segment_start":   0.0,
            "segment_end":     round(duration, 3),
            "duration_sec":    round(duration, 3),
            "transcript":      ref_text,
            "transcript_conf": 0.0,
            "language":        language,
            "language_prob":   1.0,
        }
        reg.update_ref(slug, design_wav_path, ref_meta)
        if prompt_path.exists():
            with open(meta_path) as _fh:
                _meta = json.load(_fh)
            reg.register_prompt(slug, stem, prompt_path, _meta)
        voice_label = slug
    else:
        voice_label = stem[:22] + "..."

    print(f"\nDone  ({total_sec:.1f}s)")
    print(f"  voice ID:  {voice_name or stem}")
    print(f"  prompt:    {prompt_path}")
    print(f"  use with:  ./run voice-synth speak --voice {voice_label} --text '...'")


# ── voice management commands ──────────────────────────────────────────────────

def _voice_reg(args):
    """Return a VoiceRegistry for the current cache."""
    return _voice_registry(Path(args.cache))


def cmd_rename_voice(args) -> None:
    reg = _voice_reg(args)
    try:
        reg.rename(args.old_name, args.new_name)
        print(f"  renamed '{args.old_name}'  →  '{args.new_name}'")
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_delete_voice(args) -> None:
    reg = _voice_reg(args)
    if not reg.exists(args.voice):
        print(f"ERROR: voice '{args.voice}' not found.", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        ans = input(f"Delete voice '{args.voice}' and ALL its files? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("  Aborted.")
            return
    reg.delete(args.voice)
    print(f"  deleted '{args.voice}'.")


def cmd_export_voice(args) -> None:
    reg = _voice_reg(args)
    dest = Path(args.out) / f"{args.voice}.zip" if args.dest is None else Path(args.dest)
    try:
        reg.export_zip(args.voice, dest)
        print(f"  exported '{args.voice}'  →  {dest}")
        print(f"  Share: copy the zip to another machine, then run:")
        print(f"    ./run voice-synth import-voice --zip /work/{args.voice}.zip")
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_import_voice(args) -> None:
    reg = _voice_reg(args)
    try:
        slug = reg.import_zip(Path(args.zip), force=args.force)
        print(f"  imported voice '{slug}'  from  {args.zip}")
        print(f"  use with:  ./run voice-synth speak --voice {slug} --text '...'")
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ── argument parser ────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_BASE_MODEL,
                   help="Qwen3-TTS Base model (HF repo ID or local path)")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER,
                   help="faster-whisper model for QA transcription")
    p.add_argument("--language", default="Auto", choices=QWEN3_LANGUAGES,
                   help="Synthesis language; Auto = detect from text then ref")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                   help="CPU threads for torch + whisper (default: all logical cores)")
    p.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype.  auto (default): float32 on CPU; bfloat16 on CUDA Ampere+; "
            "float16 on older CUDA GPUs (Maxwell / Pascal / Volta / Turing)."
        ),
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Base random seed (variant i uses seed+i)")
    p.add_argument("--cache", default="/cache", help="Persistent cache directory")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="voice-synth — synthesise from cached voice-clone prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ── list-voices ────────────────────────────────────────────────────────────
    lv = sub.add_parser("list-voices",
                        help="List cached voice prompts",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    lv.add_argument("--cache", default="/cache")

    # ── speak ──────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "speak",
        help="Synthesise text from a cached voice prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument("--voice", required=True,
                    help="Voice ID (or path to .pkl) from list-voices")
    sp.add_argument("--text",      default=None,
                    help="Text to synthesise")
    sp.add_argument("--text-file", default=None, metavar="FILE",
                    help="Read synthesis text from a file")
    sp.add_argument(
        "--tone", default=None, metavar="NAME",
        help=(
            "Tone label to select (e.g. 'neutral', 'sad', 'excited'). "
            "Picks the prompt built from a reference clip labelled with that tone. "
            "Register tones with: voice-clone synth --tone NAME. "
            "If omitted, uses the most recently built prompt."
        ),
    )
    sp.add_argument("--variants",  type=int, default=1,
                    help="Number of takes to generate with different seeds")
    sp.add_argument("--chunk",     action="store_true",
                    help="Auto-chunk long text into sentences and concatenate output")
    sp.add_argument("--qa",        action="store_true",
                    help="Run whisper QA on each take and print intelligibility score")
    # Generation knobs
    sp.add_argument("--temperature",        type=float, default=None)
    sp.add_argument("--top-p",              type=float, default=None, dest="top_p")
    sp.add_argument("--repetition-penalty", type=float, default=None,
                    dest="repetition_penalty")
    sp.add_argument("--max-new-tokens",     type=int,   default=None,
                    dest="max_new_tokens")
    sp.add_argument("--out", default="/work",
                    help="Output directory")
    _add_common(sp)

    # ── design-voice ───────────────────────────────────────────────────────────
    dv = sub.add_parser(
        "design-voice",
        help="Create a voice from natural language (VoiceDesign → Clone; slow on CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dv.add_argument("--instruct",     required=True,
                    help="Natural language voice description (persona, style, timbre)")
    dv.add_argument("--ref-text",     required=True,
                    help="Script text spoken in the designed voice (short, ~1–2 sentences)")
    dv.add_argument("--design-model", default=DEFAULT_DESIGN_MODEL,
                    help="Qwen3-TTS VoiceDesign model")
    dv.add_argument("--clone-model",  default=DEFAULT_BASE_MODEL,
                    help="Qwen3-TTS Base model for building the clone prompt")
    dv.add_argument("--out", default="/work",
                    help="Output directory for the designed ref clip")
    dv.add_argument("--voice-name", default=None, metavar="SLUG",
                    help="Register the designed voice as a named voice in /cache/voices/")
    _add_common(dv)

    # ── rename-voice ───────────────────────────────────────────────────────────
    rv = sub.add_parser("rename-voice",
                        help="Rename a voice in the registry",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    rv.add_argument("old_name", help="Current voice slug")
    rv.add_argument("new_name", help="New voice slug")
    rv.add_argument("--cache", default="/cache")

    # ── delete-voice ───────────────────────────────────────────────────────────
    delv = sub.add_parser("delete-voice",
                          help="Permanently delete a voice and all its files",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    delv.add_argument("voice", help="Voice slug to delete")
    delv.add_argument("--yes", action="store_true",
                      help="Skip confirmation prompt")
    delv.add_argument("--cache", default="/cache")

    # ── export-voice ───────────────────────────────────────────────────────────
    ev = sub.add_parser("export-voice",
                        help="Export a voice to a shareable zip archive",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ev.add_argument("voice", help="Voice slug to export")
    ev.add_argument("--dest", default=None, metavar="FILE",
                    help="Output zip path (default: /work/<voice>.zip)")
    ev.add_argument("--out", default="/work", help="Output directory (used when --dest omitted)")
    ev.add_argument("--cache", default="/cache")

    # ── import-voice ───────────────────────────────────────────────────────────
    iv = sub.add_parser("import-voice",
                        help="Import a voice from a zip archive (exported by export-voice)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    iv.add_argument("--zip", required=True, metavar="FILE",
                    help="Path to the .zip file to import")
    iv.add_argument("--force", action="store_true",
                    help="Overwrite if a voice with the same slug already exists")
    iv.add_argument("--cache", default="/cache")

    return ap


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(getattr(args, "threads", 8)))

    cache = Path(getattr(args, "cache", "/cache"))
    hub_dir = cache / "torch" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))

    match args.command:
        case "list-voices":
            cmd_list_voices(args)
        case "speak":
            if not args.text and not args.text_file:
                ap.error("speak requires --text or --text-file")
            cmd_speak(args)
        case "design-voice":
            cmd_design_voice(args)
        case "rename-voice":
            cmd_rename_voice(args)
        case "delete-voice":
            cmd_delete_voice(args)
        case "export-voice":
            cmd_export_voice(args)
        case "import-voice":
            cmd_import_voice(args)
        case _:
            ap.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
