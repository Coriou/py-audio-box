#!/usr/bin/env python3
"""
voice-synth.py — DevX-first synthesis rig for clone prompts and built-in CustomVoice.

Sub-commands:
  list-voices    List cached voice prompts available for synthesis
  list-speakers  List supported built-in speakers for Qwen3 CustomVoice models
  capabilities   Emit machine-readable runtime/model capability metadata
  register-builtin
                 Register/update a named built-in CustomVoice profile
  speak          Synthesise text from a cached voice prompt (with variants, QA, chunking)
  design-voice   Create a reusable voice from a natural-language description
                 (VoiceDesign → Clone workflow; needs the 1.7B-VoiceDesign model)

Usage:
  ./run voice-synth list-voices
  ./run voice-synth list-speakers
  ./run voice-synth capabilities --json
  ./run voice-synth register-builtin --voice-name announcer --speaker Ryan
  ./run voice-synth speak --voice <id> --text "Hello, world"
  ./run voice-synth speak --speaker Ryan --text "Hello, world" --instruct "Warm, calm delivery"
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
import shutil
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

# ── shared lib ─────────────────────────────────────────────────────────────────
# All helpers common to voice-synth and voice-clone live in lib/.
_LIB = str(Path(__file__).resolve().parent.parent.parent / "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from tts import (  # noqa: E402  (import after path setup)
    DEFAULT_TTS_MODEL, DEFAULT_CUSTOM_MODEL, DEFAULT_DESIGN_MODEL, DEFAULT_WHISPER,
    PROMPT_SCHEMA_VERSION, QWEN3_LANGUAGES, QWEN3_CUSTOM_SPEAKERS,
    QWEN3_REQUIRED_MODEL_METHODS, QWEN3_CUSTOMVOICE_FLAG_ENV,
    GENERATION_PROFILE_CHOICES,
    load_instruction_templates, resolve_instruction_template,
    resolve_generation_profile, build_generation_kwargs,
    resolve_language, validate_language, supported_speakers, validate_speaker,
    ctranslate2_device, cuda_ctranslate2_compute_type, get_device, best_dtype,
    qwen_tts_package_version, probe_qwen_tts_api,
    custom_voice_feature_enabled, custom_voice_feature_env_value,
    load_tts_model, synthesise, synthesise_custom_voice,
    Timer,
)
from audio import (  # noqa: E402
    SELECTION_WEIGHTS,
    get_duration,
    analyse_acoustics,
    score_take_selection,
    rank_take_selection,
)
from voices import VoiceRegistry, validate_slug  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────

# Silence (ms) inserted between auto-chunked sentences when concatenating
CHUNK_SILENCE_MS = 300
TAKES_META_SCHEMA_VERSION = 1
CAPABILITIES_SCHEMA_VERSION = 1
SPEAK_JSON_RESULT_SCHEMA_VERSION = 1

KNOWN_QWEN3_MODELS: dict[str, list[str]] = {
    "clone": [
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ],
    "custom_voice": [
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ],
    "voice_design": [
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ],
}


def custom_voice_feature_state() -> dict[str, Any]:
    """
    Return machine-readable rollout flag state for CustomVoice functionality.
    """
    return {
        "enabled": custom_voice_feature_enabled(),
        "env_var": QWEN3_CUSTOMVOICE_FLAG_ENV,
        "raw_value": custom_voice_feature_env_value(),
        "default_enabled": True,
    }


def require_custom_voice_enabled(context: str) -> None:
    """
    Exit with a clear rollout-flag message when CustomVoice is disabled.
    """
    state = custom_voice_feature_state()
    if state["enabled"]:
        return
    raw = state["raw_value"] if state["raw_value"] is not None else "0"
    print(
        f"ERROR: {context} requires CustomVoice, but "
        f"{state['env_var']}={raw!r} disables it.\n"
        f"  Re-enable with: export {state['env_var']}=1",
        file=sys.stderr,
    )
    sys.exit(1)


def _slugify_id(raw: str) -> str:
    """Return a filesystem-safe slug from arbitrary user/model labels."""
    slug = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return slug or "item"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def concat_wavs_simple(wav_paths: list[Path], dest: Path,
                        silence_ms: int = CHUNK_SILENCE_MS,
                        sample_rate: int = 24_000) -> None:
    """
    Pure-numpy concatenation of WAV files with silence between them.
    More reliable across ffmpeg versions for simple concat.
    """
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


# ── QA: whisper-based intelligibility check ────────────────────────────────────

def _word_overlap_ratio(ref_words: list[str], hyp_words: list[str]) -> float:
    """Fraction of ref words found in hyp (order-insensitive, lower-cased)."""
    if not ref_words:
        return 0.0
    hyp_set = set(hyp_words)
    matched = sum(1 for w in ref_words if w in hyp_set)
    return matched / len(ref_words)


def qa_transcribe(wav_path: Path, whisper_model: str, num_threads: int,
                  target_text: str) -> dict[str, Any]:
    """
    Run faster-whisper on *wav_path* and return a QA dict:
      transcript, intelligibility (word overlap), duration_sec.
    Uses the best compute type for the current device.
    """
    try:
        from faster_whisper import WhisperModel
        device       = ctranslate2_device()
        compute_type = cuda_ctranslate2_compute_type() if device.startswith("cuda") else "int8"
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
    duration  = get_duration(wav_path)

    return {
        "transcript":     transcript,
        "intelligibility": round(overlap, 4),
        "duration_sec":   round(duration, 3),
    }


def assert_selection_qa_ok(qa: dict[str, Any], take_label: str) -> None:
    """
    Raise when selection-time QA failed.

    --select-best relies on intelligibility, so ranking must not proceed if
    Whisper QA crashed for a take.
    """
    err = str(qa.get("error") or "").strip()
    if err:
        raise RuntimeError(
            f"QA failed for {take_label}: {err}"
        )


def resolve_instruct_text(
    instruct: str | None,
    instruct_style: str | None,
    *,
    required: bool,
    context: str,
) -> tuple[str | None, str]:
    """
    Resolve explicit instruction text vs named instruction template.
    """
    style = (instruct_style or "").strip() or None
    text = (instruct or "").strip() or None

    if text and style:
        raise ValueError("cannot use both explicit instruction text and --instruct-style.")

    if style:
        resolved = resolve_instruction_template(style)
        if not resolved:
            templates = load_instruction_templates()
            available = ", ".join(sorted(templates)) or "(none)"
            raise ValueError(
                f"unknown instruction template '{style}' for {context}. Available: {available}"
            )
        return resolved, f"style_template:{style}"

    if text:
        return text, "explicit"

    if required:
        raise ValueError("requires --instruct or --instruct-style.")

    return None, "none"


# ── prompts directory helpers ──────────────────────────────────────────────────

def _prompt_model_from_meta(pkl_path: Path | None) -> str:
    """Best-effort model tag from sibling ``.meta.json``."""
    if pkl_path is None:
        return "?"
    meta_path = pkl_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return "?"
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        return str(meta.get("model") or "?")
    except Exception:
        return "?"


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
    reg = VoiceRegistry(cache)
    for v in reg.list_voices():
        slug    = v["slug"]
        ref     = v.get("ref") or {}
        best    = reg.best_prompt(slug)
        engine  = str(v.get("_engine") or v.get("engine") or "clone_prompt")
        profile = v.get("custom_voice") if isinstance(v.get("custom_voice"), dict) else {}
        generation_defaults = (
            v.get("generation_defaults")
            if isinstance(v.get("generation_defaults"), dict)
            else {}
        )
        custom_tones = sorted((profile.get("tones") or {}).keys()) if profile else []
        # Mark all registered prompt stems so legacy scan skips them
        for pkey in (v.get("prompts") or {}).values():
            seen_stems.add(Path(pkey).stem)

        if engine == "custom_voice":
            model = str(profile.get("model") or "?")
            ref_language = str(profile.get("language_default") or "English")
            duration_sec = 0.0
            transcript = ""
            pkl_path = None
        else:
            model = _prompt_model_from_meta(best)
            ref_language = str(ref.get("language", "?"))
            duration_sec = float(ref.get("duration_sec", 0.0))
            transcript = str(ref.get("transcript", ""))
            pkl_path = str(best) if best else None

        results.append({
            "kind":        "named",
            "engine":      engine,
            "id":          slug,
            "slug":        slug,
            "display_name": v.get("display_name", slug),
            "description": v.get("description", ""),
            "pkl":         pkl_path,
            "pkl_exists":  best is not None and best.exists(),
            "model":       model,
            "speaker":     profile.get("speaker") if profile else None,
            "instruct_default": profile.get("instruct_default") if profile else None,
            "custom_tones": custom_tones,
            "generation_defaults": generation_defaults,
            "generation_profile_default": generation_defaults.get("profile"),
            "ref_language": ref_language,
            "duration_sec": duration_sec,
            "transcript":  transcript,
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
                "engine":      "clone_prompt",
                "id":          stem,
                "slug":        None,
                "display_name": stem,
                "pkl":         str(pkl) if pkl.exists() else None,
                "pkl_exists":  pkl.exists(),
                "model":       m.get("model", "?"),
                "speaker":     None,
                "instruct_default": None,
                "custom_tones": [],
                "generation_defaults": {},
                "generation_profile_default": None,
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
                "engine":      "clone_prompt",
                "id":          stem,
                "slug":        None,
                "display_name": stem,
                "pkl":         str(pkl),
                "pkl_exists":  True,
                "model":       "?",
                "speaker":     None,
                "instruct_default": None,
                "custom_tones": [],
                "generation_defaults": {},
                "generation_profile_default": None,
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


def resolve_voice(voice_arg: str, cache: Path, tone: str | None = None) -> dict[str, Any]:
    """
    Resolve ``--voice`` to a synthesis source.

    Return shape (mode-dependent):
      {
        "voice_id": str,
        "engine_mode": "clone_prompt" | "custom_voice" | "designed_clone",
        "prompt_path": str | None,
        "speaker": str | None,
        "instruct": str | None,
        "instruct_source": str | None,
        "model": str | None,
        "language_default": str | None,
        "generation_defaults": dict,
      }

    Resolution order:
      1. Absolute path ending in ``.pkl`` (clone prompt path; tone ignored)
      2. Exact named-voice slug in ``/cache/voices/<slug>/``
      3. Unambiguous prefix match against named voices
      4. Exact stem or unambiguous prefix in legacy ``/cache/voice-clone/prompts/``
    """
    # 1. Direct path
    if voice_arg.endswith(".pkl"):
        p = Path(voice_arg)
        if p.exists():
            return {
                "voice_id": p.stem,
                "engine_mode": "clone_prompt",
                "prompt_path": str(p),
                "speaker": None,
                "instruct": None,
                "instruct_source": None,
                "model": None,
                "language_default": None,
                "generation_defaults": {},
            }
        print(f"ERROR: pkl not found: {voice_arg}", file=sys.stderr)
        sys.exit(1)

    reg = VoiceRegistry(cache)

    # 2. Exact named-voice slug
    if reg.exists(voice_arg):
        data = reg.load(voice_arg)
        engine = str(data.get("engine") or "clone_prompt")

        if engine == "custom_voice":
            profile = reg.custom_voice_profile(voice_arg)
            if not profile:
                print(
                    f"ERROR: voice '{voice_arg}' is marked custom_voice but missing speaker.",
                    file=sys.stderr,
                )
                sys.exit(1)

            tone_presets = profile.get("tones") or {}
            instruct: str | None
            instruct_source: str
            if tone is not None:
                if tone not in tone_presets:
                    available = sorted(tone_presets.keys())
                    if available:
                        print(
                            f"ERROR: voice '{voice_arg}' has no CustomVoice tone '{tone}'.\n"
                            f"  Available tones: {', '.join(available)}\n"
                            f"  Add one with:\n"
                            f"    ./run voice-synth register-builtin --voice-name {voice_arg} "
                            f"--speaker {profile['speaker']} --tone {tone} "
                            f"--tone-instruct '...'",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"ERROR: voice '{voice_arg}' has no CustomVoice tone presets yet.\n"
                            f"  Add one with:\n"
                            f"    ./run voice-synth register-builtin --voice-name {voice_arg} "
                            f"--speaker {profile['speaker']} --tone {tone} "
                            f"--tone-instruct '...'",
                            file=sys.stderr,
                        )
                    sys.exit(1)
                instruct = str(tone_presets[tone]).strip() or None
                instruct_source = "tone_preset"
            else:
                default_instruct = str(profile.get("instruct_default") or "").strip()
                instruct = default_instruct or None
                instruct_source = "default" if instruct else "none"

            return {
                "voice_id": voice_arg,
                "engine_mode": "custom_voice",
                "prompt_path": None,
                "speaker": profile["speaker"],
                "instruct": instruct,
                "instruct_source": instruct_source,
                "model": profile.get("model"),
                "language_default": profile.get("language_default"),
                "generation_defaults": reg.generation_defaults(voice_arg),
            }

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
            return {
                "voice_id": voice_arg,
                "engine_mode": engine,
                "prompt_path": str(pkl),
                "speaker": None,
                "instruct": None,
                "instruct_source": None,
                "model": None,
                "language_default": None,
                "generation_defaults": reg.generation_defaults(voice_arg),
            }

        best = reg.best_prompt(voice_arg)
        if best is None:
            print(
                f"ERROR: voice '{voice_arg}' exists but has no prompts yet.\n"
                f"  Build one with:\n"
                f"    ./run voice-clone synth --voice {voice_arg} --text 'Hello'",
                file=sys.stderr,
            )
            sys.exit(1)
        return {
            "voice_id": voice_arg,
            "engine_mode": engine,
            "prompt_path": str(best),
            "speaker": None,
            "instruct": None,
            "instruct_source": None,
            "model": None,
            "language_default": None,
            "generation_defaults": reg.generation_defaults(voice_arg),
        }

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
        return {
            "voice_id": matches[0],
            "engine_mode": "clone_prompt",
            "prompt_path": str(pkl),
            "speaker": None,
            "instruct": None,
            "instruct_source": None,
            "model": None,
            "language_default": None,
            "generation_defaults": {},
        }
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


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def build_capabilities_payload(args) -> tuple[dict[str, Any], list[str]]:
    """
    Build a machine-readable capability report for agent/CI introspection.
    """
    failures: list[str] = []
    probe_model = args.model or DEFAULT_CUSTOM_MODEL
    feature_state = custom_voice_feature_state()
    custom_voice_enabled = bool(feature_state["enabled"])
    torch_device = get_device()
    resolved_dtype = _dtype_name(best_dtype(torch_device, args.dtype))
    ctranslate_device = torch_device
    if torch_device.startswith("cuda") and torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability()
            if cc < (7, 0):
                ctranslate_device = "cpu"
        except Exception:  # noqa: BLE001
            ctranslate_device = "cpu"

    cuda_available = bool(torch.cuda.is_available())
    cuda_name = None
    cuda_capability = None
    if cuda_available:
        try:
            cuda_index = int(torch_device.split(":")[-1]) if ":" in torch_device else 0
        except ValueError:
            cuda_index = 0
        try:
            cuda_name = torch.cuda.get_device_name(cuda_index)
            major, minor = torch.cuda.get_device_capability(cuda_index)
            cuda_capability = f"{major}.{minor}"
        except Exception:  # noqa: BLE001
            cuda_name = "unknown"
            cuda_capability = "unknown"

    required_methods = dict(QWEN3_REQUIRED_MODEL_METHODS)
    if not custom_voice_enabled:
        required_methods.pop("custom_voice", None)

    api_probe = probe_qwen_tts_api(required_methods=required_methods)
    if not bool(api_probe.get("compatible")):
        missing = api_probe.get("missing") or {}
        failures.append(f"qwen-tts API compatibility failed: missing methods={missing}")

    speakers: list[str] = []
    speaker_probe_mode = "runtime"
    speaker_probe_error: str | None = None

    if not custom_voice_enabled:
        speaker_probe_mode = "feature_disabled"
    else:
        if getattr(args, "skip_speaker_probe", False):
            speakers = list(QWEN3_CUSTOM_SPEAKERS)
            speaker_probe_mode = "static_fallback"
        else:
            try:
                model = load_tts_model(probe_model, args.threads, args.dtype)
                speakers = supported_speakers(model)
                if not speakers:
                    speaker_probe_mode = "runtime_empty_static_fallback"
                    speakers = list(QWEN3_CUSTOM_SPEAKERS)
            except Exception as exc:  # noqa: BLE001
                speaker_probe_mode = "error_static_fallback"
                speaker_probe_error = str(exc)
                speakers = list(QWEN3_CUSTOM_SPEAKERS)

    if custom_voice_enabled and not speakers:
        failures.append("CustomVoice speaker probe returned an empty list.")
    if getattr(args, "require_runtime_speakers", False):
        if not custom_voice_enabled:
            failures.append(
                "Runtime speaker probe required but CustomVoice is disabled via feature flag."
            )
        elif speaker_probe_mode != "runtime":
            failures.append(
                "Runtime speaker probe required but unavailable "
                f"(mode={speaker_probe_mode})."
            )

    payload = {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_flags": {
            "custom_voice": feature_state,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_device": torch_device,
            "cuda_available": cuda_available,
            "cuda_name": cuda_name,
            "cuda_capability": cuda_capability,
            "ctranslate2_device": ctranslate_device,
            "ctranslate2_compute_type": (
                cuda_ctranslate2_compute_type() if ctranslate_device.startswith("cuda")
                else "int8"
            ),
            "dtype_requested": args.dtype,
            "dtype_resolved": resolved_dtype,
            "threads": args.threads,
        },
        "packages": {
            "qwen_tts_version": qwen_tts_package_version(),
        },
        "available_models": KNOWN_QWEN3_MODELS,
        "default_models": {
            "clone": DEFAULT_TTS_MODEL,
            "custom_voice": DEFAULT_CUSTOM_MODEL,
            "voice_design": DEFAULT_DESIGN_MODEL,
        },
        "api_compatibility": api_probe,
        "custom_voice": {
            "enabled": custom_voice_enabled,
            "probe_model": probe_model,
            "speaker_probe_mode": speaker_probe_mode,
            "speaker_probe_error": speaker_probe_error,
            "speaker_count": len(speakers),
            "speakers": sorted(set(speakers), key=str.lower),
        },
    }
    return payload, failures


def cmd_capabilities(args) -> None:
    payload, failures = build_capabilities_payload(args)

    if getattr(args, "json", False):
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        runtime = payload["runtime"]
        custom = payload["custom_voice"]
        custom_flag = payload["feature_flags"]["custom_voice"]
        api = payload["api_compatibility"]
        print("Qwen3 capability probe")
        print(
            f"  device: {runtime['torch_device']}  "
            f"(cuda={runtime['cuda_available']} cc={runtime['cuda_capability'] or 'n/a'})"
        )
        if runtime.get("cuda_name"):
            print(f"  gpu: {runtime['cuda_name']}")
        print(
            f"  dtype: requested={runtime['dtype_requested']} "
            f"resolved={runtime['dtype_resolved']}"
        )
        print(f"  qwen-tts: {payload['packages']['qwen_tts_version']}")
        print(
            "  custom voice feature: "
            + ("enabled" if custom_flag["enabled"] else "disabled")
            + f" ({custom_flag['env_var']}="
            + (repr(custom_flag["raw_value"]) if custom_flag["raw_value"] is not None else "'<unset>'")
            + ")"
        )
        print(
            "  api compatibility: "
            + ("ok" if api.get("compatible") else f"FAILED (missing={api.get('missing')})")
        )
        print(
            f"  custom speakers: {custom['speaker_count']} "
            f"(probe={custom['speaker_probe_mode']}, model={custom['probe_model']})"
        )
        if custom.get("speaker_probe_error"):
            print(f"  speaker probe error: {custom['speaker_probe_error']}")

    if failures and not args.strict:
        for msg in failures:
            print(f"  WARNING: {msg}", file=sys.stderr)

    if failures and args.strict:
        for msg in failures:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)


# ── sub-command implementations ────────────────────────────────────────────────

def cmd_list_voices(args) -> None:
    cache  = Path(args.cache)
    voices = list_all_voices(cache)

    named  = [v for v in voices if v["kind"] == "named"]
    legacy = [v for v in voices if v["kind"] == "legacy"]

    if getattr(args, "json", False):
        payload = {
            "cache": str(cache),
            "registry_dir": str(cache / "voices"),
            "legacy_dir": str(cache / "voice-clone" / "prompts"),
            "total": len(voices),
            "named_count": len(named),
            "legacy_count": len(legacy),
            "named": named,
            "legacy": legacy,
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
        return

    if not voices:
        print("No named voices or cached prompts found.")
        print("  Extract clips:   ./run voice-split --url '...' --voice-name my-voice")
        print("  Register a file: ./run voice-clone prepare-ref --ref-audio /work/clip.wav"
              " --voice-name my-voice")
        print("  Register builtin: ./run voice-synth register-builtin --voice-name announcer "
              "--speaker Ryan")
        return

    if named:
        print(f"\n\033[1mNAMED VOICES\033[0m  ({len(named)})")
        print(f"  {'NAME':<28}  {'ENGINE':<14}  {'LANG':<10}  {'DUR':>5}  {'PROMPTS':>7}  STATUS")
        print("  " + "-" * 90)
        list_reg = VoiceRegistry(cache)
        for v in named:
            if v["engine"] == "custom_voice":
                status = (
                    "\033[32mready\033[0m" if v["_ready"]
                    else "\033[31mmisconfigured\033[0m  run: voice-synth register-builtin --voice-name "
                    + v["slug"] + " --speaker <name>"
                )
            else:
                status = (
                    "\033[32mready\033[0m" if v["_ready"]
                    else ("\033[33mno prompts\033[0m  run: voice-clone synth --voice " + v["slug"])
                    if v["_has_ref"]
                    else "\033[31mno ref\033[0m  run: voice-clone prepare-ref --voice " + v["slug"]
                )
            print(
                f"  {v['slug']:<28}  {v['engine']:<14}  {v['ref_language']:<10}"
                f"  {v['duration_sec']:>4.1f}s  {v['_prompt_count']:>7}  {status}"
            )
            if v["engine"] == "custom_voice":
                model = textwrap.shorten(str(v.get("model") or "?"), width=54, placeholder="…")
                speaker = v.get("speaker") or "?"
                print(f"    \033[2mspeaker: {speaker}  model: {model}\033[0m")
                if v.get("instruct_default"):
                    preview = textwrap.shorten(str(v["instruct_default"]), width=80, placeholder="…")
                    print(f"    \033[2mdefault instruct: {preview}\033[0m")
                if v.get("custom_tones"):
                    print(
                        "    \033[2mcustom tones: "
                        + ", ".join(sorted(v["custom_tones"]))
                        + "\033[0m"
                    )
            if v.get("generation_profile_default"):
                print(
                    "    \033[2mgeneration profile default: "
                    + str(v["generation_profile_default"])
                    + "\033[0m"
                )
            if v["transcript"]:
                preview = textwrap.shorten(v["transcript"], width=68, placeholder="…")
                print(f"    \033[2m{preview}\033[0m")
            # Show registered tones if any
            if v["engine"] != "custom_voice":
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


def cmd_list_speakers(args) -> None:
    require_custom_voice_enabled("list-speakers")
    model_name = args.model or DEFAULT_CUSTOM_MODEL

    model = load_tts_model(model_name, args.threads, args.dtype)
    speakers = supported_speakers(model)
    if not speakers:
        print(
            "ERROR: no supported speakers found for this model.\n"
            "  Use a Qwen3 CustomVoice model, e.g.\n"
            f"    {DEFAULT_CUSTOM_MODEL}",
            file=sys.stderr,
        )
        sys.exit(1)

    if getattr(args, "json", False):
        payload = {
            "model": model_name,
            "count": len(speakers),
            "speakers": speakers,
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
        return

    print(f"  model: {model_name}")
    print(f"\n{len(speakers)} built-in speaker(s):")
    for spk in speakers:
        print(f"  - {spk}")


def cmd_register_builtin(args) -> None:
    require_custom_voice_enabled("register-builtin")
    cache = Path(args.cache)
    reg = VoiceRegistry(cache)

    if args.tone and not (
        (args.tone_instruct and args.tone_instruct.strip())
        or (args.tone_instruct_style and args.tone_instruct_style.strip())
    ):
        print("ERROR: --tone requires --tone-instruct or --tone-instruct-style.", file=sys.stderr)
        sys.exit(1)
    if (args.tone_instruct or args.tone_instruct_style) and not args.tone:
        print("ERROR: --tone-instruct/--tone-instruct-style requires --tone.", file=sys.stderr)
        sys.exit(1)

    try:
        instruct_default, instruct_default_source = resolve_instruct_text(
            args.instruct_default,
            args.instruct_default_style,
            required=False,
            context="register-builtin --instruct-default",
        )
        tone_instruct, tone_instruct_source = resolve_instruct_text(
            args.tone_instruct,
            args.tone_instruct_style,
            required=bool(args.tone),
            context="register-builtin --tone",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    slug = validate_slug(args.voice_name)
    model_name = args.model or DEFAULT_CUSTOM_MODEL

    model = load_tts_model(model_name, args.threads, args.dtype)
    speaker = validate_speaker(args.speaker, model)

    try:
        reg.register_custom_voice(
            slug,
            model=model_name,
            speaker=speaker,
            instruct_default=instruct_default,
            language_default=args.language_default,
            tone=args.tone,
            tone_instruct=tone_instruct,
            display_name=args.display_name,
            description=args.description,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    profile = reg.custom_voice_profile(slug)
    assert profile is not None

    print(f"  registered built-in voice '{slug}'")
    print(f"  speaker: {profile['speaker']}")
    print(f"  model:   {profile.get('model') or model_name}")
    print(f"  language default: {profile.get('language_default') or 'English'}")
    if profile.get("instruct_default"):
        source = f" ({instruct_default_source})" if instruct_default_source != "none" else ""
        print(f"  default instruct{source}: {profile['instruct_default']!r}")
    tones = profile.get("tones") or {}
    if tones:
        tone_msg = f"  tone presets: {', '.join(sorted(tones))}"
        if args.tone and tone_instruct_source != "none":
            tone_msg += f"  (updated {args.tone!r} via {tone_instruct_source})"
        print(tone_msg)
    print(f"  use with:  ./run voice-synth speak --voice {slug} --text '...'")


def build_speak_takes_metadata(
    *,
    voice_id: str,
    model_name: str,
    engine_mode: str,
    source_type: str,
    speaker: str | None,
    instruct: str | None,
    instruct_source: str | None,
    instruct_style_applied: str | None,
    language: str,
    ref_language: str,
    raw_text: str,
    text: str,
    tone: str | None,
    pkl_path: str | None,
    gen_kwargs: dict[str, Any],
    profile_name: str,
    profile_source: str,
    generation_meta: dict[str, Any],
    voice_generation_defaults: dict[str, Any],
    saved_generation_defaults: dict[str, Any] | None,
    chunked: bool,
    text_chunks: list[str],
    n_variants: int,
    selection_policy: dict[str, Any],
    selection_breakdown: list[dict[str, Any]],
    selected_take: str | None,
    selected_wav: str | None,
    load_sec: float,
    total_sec: float,
    takes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": TAKES_META_SCHEMA_VERSION,
        "app":            "voice-synth",
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "voice_id":       voice_id,
        "model":          model_name,
        "engine_mode":    engine_mode,
        "source_type":    source_type,
        "speaker":        speaker,
        "instruct":       instruct,
        "instruct_source": instruct_source,
        "instruct_style": instruct_style_applied,
        "language":       language,
        "ref_language":   ref_language,
        "original_text":  raw_text,
        "text":           text,
        "tone":           tone,
        "prompt_path":    pkl_path,
        "generation_kwargs": gen_kwargs,
        "generation_profile": profile_name,
        "generation_profile_source": profile_source,
        "generation_profile_meta": generation_meta,
        "voice_generation_defaults": voice_generation_defaults,
        "saved_generation_defaults": saved_generation_defaults,
        "chunked":        chunked,
        "n_chunks":       len(text_chunks),
        "variants":       n_variants,
        "selection_policy": selection_policy,
        "selection_metrics": selection_breakdown,
        "selected_take": selected_take,
        "selected_wav": selected_wav,
        "load_sec":       round(load_sec, 2),
        "total_sec":      round(total_sec, 2),
        "takes":          takes,
    }


def _resolve_speak_run_dir(
    *,
    out_dir: Path,
    out_exact: str | None,
    use_direct_speaker_mode: bool,
    speaker: str | None,
    voice_id: str,
) -> Path:
    """
    Resolve destination directory for speak output.

    `--out-exact` writes artifacts directly in the provided directory.
    Otherwise keep legacy timestamped layout under --out.
    """
    if out_exact is not None:
        target = Path(out_exact)
        if target.exists() and not target.is_dir():
            raise ValueError(f"--out-exact must be a directory path: {target}")
        target.mkdir(parents=True, exist_ok=True)
        return target

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_direct_speaker_mode:
        run_dir = out_dir / "customvoice" / _slugify_id(speaker or "") / ts
    else:
        run_dir = out_dir / voice_id / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_speak_json_result(
    *,
    run_dir: Path,
    takes_meta_path: Path,
    voice_id: str,
    model_name: str,
    engine_mode: str,
    source_type: str,
    speaker: str | None,
    language: str,
    raw_text: str,
    n_variants: int,
    selected_take: str | None,
    selected_wav: str | None,
    load_sec: float,
    total_sec: float,
    takes: list[dict[str, Any]],
) -> dict[str, Any]:
    takes_files = [str(take.get("wav")) for take in takes if isinstance(take.get("wav"), str)]
    return {
        "schema_version": SPEAK_JSON_RESULT_SCHEMA_VERSION,
        "app": "voice-synth",
        "created_at": _utc_now_iso(),
        "run_dir": str(run_dir),
        "request": {
            "voice_id": voice_id,
            "engine_mode": engine_mode,
            "source_type": source_type,
            "speaker": speaker,
            "language": language,
            "text_chars": len(raw_text),
            "variants": n_variants,
        },
        "model": {
            "name": model_name,
        },
        "selection": {
            "selected_take": selected_take,
            "selected_wav": selected_wav,
            "enabled": selected_take is not None,
        },
        "timings": {
            "load_sec": round(load_sec, 2),
            "total_sec": round(total_sec, 2),
        },
        "files": {
            "text": str(run_dir / "text.txt"),
            "meta": str(takes_meta_path),
            "takes": takes_files,
            "best": selected_wav,
        },
    }


def cmd_speak(args) -> None:
    cache = Path(args.cache)
    out_dir = Path(args.out)
    reg = VoiceRegistry(cache)

    use_direct_speaker_mode = getattr(args, "speaker", None) is not None
    tone = getattr(args, "tone", None)
    instruct_style_name = (getattr(args, "instruct_style", None) or "").strip() or None
    if use_direct_speaker_mode and tone is not None:
        print("ERROR: --tone can only be used with --voice mode.", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "instruct", None) and instruct_style_name:
        print("ERROR: --instruct cannot be combined with --instruct-style.", file=sys.stderr)
        sys.exit(1)
    if not use_direct_speaker_mode and getattr(args, "instruct", None) is not None:
        print("ERROR: --instruct can only be used with --speaker mode.", file=sys.stderr)
        sys.exit(1)
    if use_direct_speaker_mode and getattr(args, "save_profile_default", False):
        print(
            "ERROR: --save-profile-default requires --voice mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    if use_direct_speaker_mode:
        require_custom_voice_enabled("speak --speaker")

    voice_meta: dict[str, Any] = {}
    prompt = None
    pkl_path: str | None = None
    ref_language = "English"
    speaker = None
    instruct = None
    instruct_source: str | None = None
    instruct_style_applied: str | None = None
    source_type = "clone_prompt"
    engine_mode = "clone_prompt"
    model_name: str | None = args.model
    voice_generation_defaults: dict[str, Any] = {}

    if use_direct_speaker_mode:
        speaker = args.speaker.strip()
        try:
            instruct, instruct_source = resolve_instruct_text(
                args.instruct,
                instruct_style_name,
                required=False,
                context="speak --speaker",
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        if instruct_source and instruct_source.startswith("style_template:"):
            instruct_style_applied = instruct_source.split(":", 1)[1] or None
        source_type = "custom_voice_builtin"
        engine_mode = "custom_voice"
        voice_id = f"speaker-{_slugify_id(speaker)}"
        model_name = model_name or DEFAULT_CUSTOM_MODEL
        print(f"\n  speaker: {speaker}" + (f"  instruct: {instruct!r}" if instruct else ""))
    else:
        resolved = resolve_voice(args.voice, cache, tone=tone)
        voice_id = str(resolved["voice_id"])
        engine_mode = str(resolved["engine_mode"])
        voice_generation_defaults = dict(resolved.get("generation_defaults") or {})

        if engine_mode == "custom_voice":
            require_custom_voice_enabled(
                f"speak --voice {voice_id}"
            )
            speaker = str(resolved.get("speaker") or "").strip()
            instruct = str(resolved.get("instruct") or "").strip() or None
            instruct_source = str(resolved.get("instruct_source") or "none")
            ref_language = str(resolved.get("language_default") or "English")
            source_type = "custom_voice_named"
            model_name = model_name or str(resolved.get("model") or DEFAULT_CUSTOM_MODEL)

            print(f"\n  voice: {voice_id}" + (f"  tone: {tone}" if tone else ""))
            print(
                f"  speaker: {speaker}"
                + (f"  instruct({instruct_source}): {instruct!r}" if instruct else "")
            )
        else:
            pkl_path = str(resolved.get("prompt_path") or "")
            if not pkl_path:
                print(f"ERROR: resolved voice '{voice_id}' has no prompt path.", file=sys.stderr)
                sys.exit(1)
            model_name = model_name or DEFAULT_TTS_MODEL
            source_type = "designed_clone_prompt" if engine_mode == "designed_clone" else "clone_prompt"
            print(f"\n  voice: {voice_id}" + (f"  tone: {tone}" if tone else ""))

            # Load meta (for ref_language, etc.)
            meta_path = Path(pkl_path).with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path) as fh:
                    voice_meta = json.load(fh)
            ref_language = voice_meta.get("ref_language_detected", "English")

    if instruct_style_name and not use_direct_speaker_mode:
        if engine_mode != "custom_voice":
            print(
                "ERROR: --instruct-style is only supported for CustomVoice synthesis "
                "(--speaker mode or --voice entries with engine=custom_voice).",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            instruct, instruct_source = resolve_instruct_text(
                None,
                instruct_style_name,
                required=True,
                context="speak --voice custom_voice",
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        if instruct_source and instruct_source.startswith("style_template:"):
            instruct_style_applied = instruct_source.split(":", 1)[1] or None
        print(f"  instruct template override ({instruct_source}): {instruct!r}")

    # Resolve text
    if args.text_file:
        raw_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        raw_text = args.text or ""

    if not raw_text:
        print("ERROR: no text provided (use --text or --text-file)", file=sys.stderr)
        sys.exit(1)

    # Synthesis text is passed through verbatim.
    # Clone mode delivery comes from the reference prompt (--tone selects prompt variant).
    # Speaker mode delivery is controlled via --speaker + optional --instruct.
    text = raw_text

    profile_name, profile_source = resolve_generation_profile(
        args.profile,
        voice_generation_defaults if not use_direct_speaker_mode else None,
    )
    gen_kwargs, generation_meta = build_generation_kwargs(
        text=text,
        profile=profile_name,
        voice_defaults=voice_generation_defaults if not use_direct_speaker_mode else None,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  generation profile: {profile_name} ({profile_source})")

    saved_generation_defaults: dict[str, Any] | None = None
    if (
        getattr(args, "save_profile_default", False)
        and not use_direct_speaker_mode
    ):
        if reg.exists(voice_id):
            policy = str(generation_meta.get("max_new_tokens_policy") or profile_name)
            if policy not in ("stable", "balanced", "expressive"):
                policy = profile_name
            saved_generation_defaults = reg.update_generation_defaults(
                voice_id,
                profile=profile_name,
                temperature=float(gen_kwargs["temperature"]),
                top_p=float(gen_kwargs["top_p"]),
                repetition_penalty=float(gen_kwargs["repetition_penalty"]),
                max_new_tokens_policy=policy,
            )
            print(f"  saved generation defaults to voice '{voice_id}'")
        else:
            print(
                "  WARNING: --save-profile-default ignored because this voice is not "
                "a named registry entry.",
                file=sys.stderr,
            )

    # Chunk if requested
    if args.chunk:
        text_chunks = chunk_text(text)
        print(f"  auto-chunked into {len(text_chunks)} piece(s)")
    else:
        text_chunks = [text]

    # Load model
    assert model_name is not None
    t0 = time.perf_counter()
    model = load_tts_model(model_name, args.threads, args.dtype)
    language = validate_language(resolve_language(args.language, ref_language, raw_text), model)
    print(f"  language: {language}  (ref detected: {ref_language})")

    use_custom_voice_mode = engine_mode == "custom_voice"
    if use_custom_voice_mode:
        speaker = validate_speaker(speaker or "", model)
        print(f"  speaker: {speaker}")
    else:
        assert pkl_path is not None
        with open(pkl_path, "rb") as fh:
            prompt = pickle.load(fh)
    load_sec = time.perf_counter() - t0

    try:
        run_dir = _resolve_speak_run_dir(
            out_dir=out_dir,
            out_exact=getattr(args, "out_exact", None),
            use_direct_speaker_mode=use_direct_speaker_mode,
            speaker=speaker,
            voice_id=voice_id,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "out_exact", None):
        print(f"  out-exact: {run_dir}")

    n_variants = max(1, args.variants)
    if args.select_best and args.seed is None:
        print("  --select-best enabled without --seed; using deterministic base seed 0")
    base_seed = args.seed if args.seed is not None else (0 if args.select_best else None)
    qa_enabled = bool(args.qa or args.select_best)

    all_takes: list[dict] = []

    for variant in range(n_variants):
        seed = (base_seed + variant) if base_seed is not None else None
        take_label = f"take_{variant + 1:02d}"

        if len(text_chunks) == 1:
            # Single chunk — no concat needed
            t_synth = time.perf_counter()
            if use_custom_voice_mode:
                wav, sr = synthesise_custom_voice(
                    text_chunks[0], language, model,
                    speaker=speaker or "",
                    instruct=instruct,
                    seed=seed,
                    gen_kwargs=gen_kwargs,
                )
            else:
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
                if use_custom_voice_mode:
                    w, sr = synthesise_custom_voice(
                        chunk, language, model,
                        speaker=speaker or "",
                        instruct=instruct,
                        seed=seed,
                        gen_kwargs=gen_kwargs,
                    )
                else:
                    w, sr = synthesise(chunk, language, model, prompt, seed, gen_kwargs)
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

        # QA pass (always enabled for --select-best because intelligibility is
        # part of the ranking policy).
        qa: dict[str, Any] | None = None
        if qa_enabled:
            print(f"  QA transcribing {take_label} …")
            qa = qa_transcribe(wav_path, args.whisper_model, args.threads, raw_text)
            if args.select_best:
                try:
                    assert_selection_qa_ok(qa, take_label)
                except RuntimeError as exc:
                    print(
                        "ERROR: --select-best requires successful whisper QA for each take.\n"
                        f"  {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            take_info["qa"] = qa
            intel = qa.get("intelligibility", 0.0)
            if args.qa:
                print(
                    f"    intelligibility: {intel:.0%}  "
                    f"transcript: {qa.get('transcript', '')!r}"
                )

        if args.select_best:
            acoustics = analyse_acoustics(wav_path)
            selection_metrics = score_take_selection(
                text=raw_text,
                duration_sec=duration,
                intelligibility=float((qa or {}).get("intelligibility", 0.0)),
                acoustics=acoustics,
            )
            take_info["acoustics"] = acoustics
            take_info["selection_metrics"] = selection_metrics

        all_takes.append(take_info)
        print(
            f"  {take_label}: {duration:.1f}s  RTF: {rtf:.2f}x  seed={seed}  "
            f"→ {wav_path.name}"
        )

    selection_policy: dict[str, Any] = {"enabled": False}
    selected_wav: str | None = None
    selected_take: str | None = None
    selection_breakdown: list[dict[str, Any]] = []

    if args.select_best:
        ranked = rank_take_selection(all_takes)
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
            selection_breakdown.append(
                {
                    "take": t["take"],
                    "rank": t.get("selection_rank"),
                    **sm,
                }
            )

        selection_policy = {
            "enabled": True,
            "method": "weighted_intelligibility_pacing_duration_fit",
            "weights": dict(SELECTION_WEIGHTS),
            "selected_take": selected_take,
            "selected_wav": selected_wav,
            "ranked": selection_breakdown,
        }
        print(f"\n  selected best take: {selected_take}  →  {best_dest.name}")

    # If QA was run, print ranked scoreboard
    if args.qa and not args.select_best and len(all_takes) > 1:
        ranked = sorted(all_takes, key=lambda t: t.get("qa", {}).get("intelligibility", 0.0),
                        reverse=True)
        print("\n  QA scoreboard (by intelligibility):")
        for rank, t in enumerate(ranked, 1):
            intel = t.get("qa", {}).get("intelligibility", 0.0)
            print(f"    #{rank} {t['take']}  intelligibility={intel:.0%}  "
                  f"dur={t['duration_sec']:.1f}s")

    # Write takes meta
    total_sec = time.perf_counter() - t0
    takes_meta = build_speak_takes_metadata(
        voice_id=voice_id,
        model_name=model_name,
        engine_mode=engine_mode,
        source_type=source_type,
        speaker=speaker,
        instruct=instruct,
        instruct_source=instruct_source,
        instruct_style_applied=instruct_style_applied,
        language=language,
        ref_language=ref_language,
        raw_text=raw_text,
        text=text,
        tone=tone,
        pkl_path=pkl_path,
        gen_kwargs=gen_kwargs,
        profile_name=profile_name,
        profile_source=profile_source,
        generation_meta=generation_meta,
        voice_generation_defaults=voice_generation_defaults,
        saved_generation_defaults=saved_generation_defaults,
        chunked=args.chunk,
        text_chunks=text_chunks,
        n_variants=n_variants,
        selection_policy=selection_policy,
        selection_breakdown=selection_breakdown,
        selected_take=selected_take,
        selected_wav=selected_wav,
        load_sec=load_sec,
        total_sec=total_sec,
        takes=all_takes,
    )
    takes_meta_path = run_dir / "takes.meta.json"
    with open(takes_meta_path, "w") as fh:
        json.dump(takes_meta, fh, indent=2)

    (run_dir / "text.txt").write_text(raw_text, encoding="utf-8")

    json_result_path_raw = getattr(args, "json_result", None)
    if json_result_path_raw:
        json_result_path = Path(json_result_path_raw)
        json_result_path.parent.mkdir(parents=True, exist_ok=True)
        json_result = build_speak_json_result(
            run_dir=run_dir,
            takes_meta_path=takes_meta_path,
            voice_id=voice_id,
            model_name=model_name,
            engine_mode=engine_mode,
            source_type=source_type,
            speaker=speaker,
            language=language,
            raw_text=raw_text,
            n_variants=n_variants,
            selected_take=selected_take,
            selected_wav=selected_wav,
            load_sec=load_sec,
            total_sec=total_sec,
            takes=all_takes,
        )
        with open(json_result_path, "w", encoding="utf-8") as fh:
            json.dump(json_result, fh, indent=2)
        print(f"  json result: {json_result_path}")

    print(f"\nDone  →  {run_dir}")
    print(f"  {n_variants} take(s) written   total: {total_sec:.1f}s")
    if selected_take:
        print(f"  best take: {selected_take}  ({Path(selected_wav).name if selected_wav else '?'})")
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
    instruct_style_name = (getattr(args, "instruct_style", None) or "").strip() or None
    try:
        instruct, instruct_source = resolve_instruct_text(
            getattr(args, "instruct", None),
            instruct_style_name,
            required=True,
            context="design-voice",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    assert instruct is not None

    language    = args.language if args.language != "Auto" else "English"

    # ── 1. Generate design reference ──────────────────────────────────────────
    print("\n==> [1/3] generate VoiceDesign reference clip")
    print(f"  design model: {args.design_model}")
    print(f"  instruct:     {instruct!r} ({instruct_source})")
    print(f"  ref_text:     {ref_text!r}")
    print(f"  language:     {language}")
    print("  NOTE: VoiceDesign model is 1.7B — this may be slow on CPU.")

    t0           = time.perf_counter()
    design_model = load_tts_model(args.design_model, args.threads, args.dtype)

    with Timer("generate_voice_design"):
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
    print("\n==> [2/3] cache designed reference audio")
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
                "instruct_source": instruct_source,
                "instruct_style": instruct_style_name,
                "design_model": args.design_model,
                "duration_sec": round(duration, 3),
                "created_at":   datetime.now(timezone.utc).isoformat(),
            },
            fh, indent=2,
        )
    print(f"  saved → {design_wav_path}")
    del design_model  # free memory before loading clone model

    # ── 3. Build clone prompt ─────────────────────────────────────────────────
    print("\n==> [3/3] build voice-clone prompt from designed ref")
    print(f"  clone model: {args.clone_model}")

    clone_model = load_tts_model(args.clone_model, args.threads, args.dtype)
    model_tag   = getattr(clone_model, "name_or_path", args.clone_model).replace("/", "_")
    stem        = f"{content_hash}_{model_tag}_designed_full_v{PROMPT_SCHEMA_VERSION}"
    prompts_dir = cache / "voice-clone" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{stem}.pkl"
    meta_path   = prompts_dir / f"{stem}.meta.json"

    with Timer("create_voice_clone_prompt"):
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
                "instruct_source":         instruct_source,
                "instruct_style":          instruct_style_name,
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
                "instruct_source": instruct_source,
                "instruct_style": instruct_style_name,
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
    return VoiceRegistry(Path(args.cache))


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
        print("  Share: copy the zip to another machine, then run:")
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

def _add_common(
    p: argparse.ArgumentParser,
    *,
    model_default: str | None = DEFAULT_TTS_MODEL,
    model_help: str | None = None,
) -> None:
    p.add_argument(
        "--model",
        default=model_default,
        help=(
            model_help or
            "Qwen3-TTS model (HF repo ID or local path)"
        ),
    )
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
        description="voice-synth — synthesise from clone prompts or built-in CustomVoice speakers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ── list-voices ────────────────────────────────────────────────────────────
    lv = sub.add_parser("list-voices",
                        help="List named voices (clone + built-in) and legacy prompts",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    lv.add_argument("--cache", default="/cache")
    lv.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    # ── list-speakers ─────────────────────────────────────────────────────────
    ls = sub.add_parser(
        "list-speakers",
        help="List built-in speakers for a Qwen3 CustomVoice model (requires QWEN3_ENABLE_CUSTOMVOICE=1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ls.add_argument(
        "--model",
        default=DEFAULT_CUSTOM_MODEL,
        help="Qwen3 CustomVoice model (HF repo ID or local path)",
    )
    ls.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                    help="CPU threads for torch (default: all logical cores)")
    ls.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype.  auto (default): float32 on CPU; bfloat16 on CUDA Ampere+; "
            "float16 on older CUDA GPUs (Maxwell / Pascal / Volta / Turing)."
        ),
    )
    ls.add_argument("--cache", default="/cache", help="Persistent cache directory")
    ls.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    # ── capabilities ──────────────────────────────────────────────────────────
    cp = sub.add_parser(
        "capabilities",
        help="Probe runtime/device/model capabilities for CI/agent usage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cp.add_argument(
        "--model",
        default=DEFAULT_CUSTOM_MODEL,
        help="CustomVoice model used for runtime speaker probing",
    )
    cp.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                    help="CPU threads used when model probing is enabled")
    cp.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype used for runtime speaker probing. "
            "Ignored when --skip-speaker-probe is set."
        ),
    )
    cp.add_argument(
        "--skip-speaker-probe",
        action="store_true",
        help="Skip model load and use the static built-in speaker fallback list",
    )
    cp.add_argument(
        "--require-runtime-speakers",
        action="store_true",
        help=(
            "Fail strict mode if speakers are not obtained from a live model load "
            "(fallback/static lists are treated as errors)."
        ),
    )
    cp.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when API compatibility or speaker probing fails",
    )
    cp.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    # ── register-builtin ─────────────────────────────────────────────────────
    rb = sub.add_parser(
        "register-builtin",
        help="Register/update a named built-in CustomVoice profile (requires QWEN3_ENABLE_CUSTOMVOICE=1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rb.add_argument("--voice-name", required=True, metavar="SLUG",
                    help="Named voice slug in /cache/voices/<slug>/")
    rb.add_argument("--speaker", required=True,
                    help="Built-in Qwen3 CustomVoice speaker")
    rb.add_argument(
        "--instruct-default", default=None, metavar="TEXT",
        help="Default instruction used when speaking this named voice without --tone",
    )
    rb.add_argument(
        "--instruct-default-style", default=None, metavar="NAME",
        help=(
            "Named instruction template for default delivery (mutually exclusive with "
            "--instruct-default)."
        ),
    )
    rb.add_argument(
        "--tone", default=None, metavar="NAME",
        help="Optional tone label to store for this built-in voice (e.g. neutral, promo)",
    )
    rb.add_argument(
        "--tone-instruct", default=None, metavar="TEXT",
        help="Instruction preset for --tone (requires --tone)",
    )
    rb.add_argument(
        "--tone-instruct-style", default=None, metavar="NAME",
        help=(
            "Named instruction template for --tone preset (requires --tone; mutually "
            "exclusive with --tone-instruct)."
        ),
    )
    rb.add_argument(
        "--language-default", default="English", choices=QWEN3_LANGUAGES,
        help="Default language used when --language Auto is selected",
    )
    rb.add_argument("--display-name", default=None,
                    help="Optional display label stored in voice.json")
    rb.add_argument("--description", default=None,
                    help="Optional description stored in voice.json")
    rb.add_argument(
        "--model",
        default=DEFAULT_CUSTOM_MODEL,
        help="Qwen3 CustomVoice model (HF repo ID or local path)",
    )
    rb.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                    help="CPU threads for torch (default: all logical cores)")
    rb.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype.  auto (default): float32 on CPU; bfloat16 on CUDA Ampere+; "
            "float16 on older CUDA GPUs (Maxwell / Pascal / Volta / Turing)."
        ),
    )
    rb.add_argument("--cache", default="/cache", help="Persistent cache directory")

    # ── speak ──────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "speak",
        help="Synthesise text from a named voice (--voice) or built-in speaker (--speaker, requires QWEN3_ENABLE_CUSTOMVOICE=1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    speak_mode = sp.add_mutually_exclusive_group(required=True)
    speak_mode.add_argument("--voice",
                            help="Voice ID (or path to .pkl) from list-voices")
    speak_mode.add_argument("--speaker",
                            help="Built-in Qwen3 CustomVoice speaker (see list-speakers)")
    sp.add_argument(
        "--instruct", default=None,
        help=(
            "Optional instruction string for --speaker mode "
            "(e.g. style, delivery, emotion)."
        ),
    )
    sp.add_argument(
        "--instruct-style", default=None, metavar="NAME",
        help=(
            "Named instruction template. Supported in --speaker mode and for named "
            "CustomVoice profiles in --voice mode."
        ),
    )
    sp.add_argument("--text",      default=None,
                    help="Text to synthesise")
    sp.add_argument("--text-file", default=None, metavar="FILE",
                    help="Read synthesis text from a file")
    sp.add_argument(
        "--tone", default=None, metavar="NAME",
        help=(
            "--voice mode only. Clone/designed voices: select a tone-labelled clone prompt. "
            "Named built-in voices: select a stored tone instruction preset. "
            "Register clone tones with: voice-clone synth --tone NAME. "
            "Register built-in tones with: voice-synth register-builtin --tone NAME "
            "--tone-instruct '...' (or --tone-instruct-style NAME)."
        ),
    )
    sp.add_argument("--variants",  type=int, default=1,
                    help="Number of takes to generate with different seeds")
    sp.add_argument(
        "--select-best",
        action="store_true",
        help=(
            "Deterministically rank generated takes and copy the best to best.wav "
            "(weighted intelligibility + pacing sanity + duration fit)."
        ),
    )
    sp.add_argument("--chunk",     action="store_true",
                    help="Auto-chunk long text into sentences and concatenate output")
    sp.add_argument("--qa",        action="store_true",
                    help="Run whisper QA on each take and print intelligibility score")
    sp.add_argument(
        "--profile",
        default=None,
        choices=GENERATION_PROFILE_CHOICES,
        help=(
            "Generation profile. If omitted, uses the voice default profile when "
            "available; otherwise falls back to balanced."
        ),
    )
    sp.add_argument(
        "--save-profile-default",
        action="store_true",
        help=(
            "--voice mode only. Persist the resolved profile + sampling defaults to "
            "voice.json generation_defaults."
        ),
    )
    # Generation knobs
    sp.add_argument("--temperature",        type=float, default=None)
    sp.add_argument("--top-p",              type=float, default=None, dest="top_p")
    sp.add_argument("--repetition-penalty", type=float, default=None,
                    dest="repetition_penalty")
    sp.add_argument("--max-new-tokens",     type=int,   default=None,
                    dest="max_new_tokens")
    sp.add_argument(
        "--out",
        default="/work",
        help="Output root directory (used when --out-exact is not set)",
    )
    sp.add_argument(
        "--out-exact",
        default=None,
        metavar="DIR",
        help=(
            "Write take_XX.wav, best.wav, takes.meta.json, and text.txt directly "
            "inside DIR (no timestamped subfolder)."
        ),
    )
    sp.add_argument(
        "--json-result",
        default=None,
        metavar="FILE",
        help="Write compact machine-readable speak result summary to FILE.",
    )
    _add_common(
        sp,
        model_default=None,
        model_help=(
            "Qwen3-TTS model override. Defaults: clone/designed --voice uses Base; "
            "named built-in --voice uses the profile model; --speaker uses CustomVoice."
        ),
    )

    # ── design-voice ───────────────────────────────────────────────────────────
    dv = sub.add_parser(
        "design-voice",
        help="Create a voice from natural language (VoiceDesign → Clone; slow on CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dv_instruct = dv.add_mutually_exclusive_group(required=True)
    dv_instruct.add_argument(
        "--instruct",
        help="Natural language voice description (persona, style, timbre)",
    )
    dv_instruct.add_argument(
        "--instruct-style",
        default=None,
        metavar="NAME",
        help="Named instruction template from lib/styles.yaml",
    )
    dv.add_argument("--ref-text",     required=True,
                    help="Script text spoken in the designed voice (short, ~1–2 sentences)")
    dv.add_argument("--design-model", default=DEFAULT_DESIGN_MODEL,
                    help="Qwen3-TTS VoiceDesign model")
    dv.add_argument("--clone-model",  default=DEFAULT_TTS_MODEL,
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
        case "list-speakers":
            cmd_list_speakers(args)
        case "capabilities":
            cmd_capabilities(args)
        case "register-builtin":
            cmd_register_builtin(args)
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
