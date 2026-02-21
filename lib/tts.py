"""
lib/tts.py — shared Qwen3-TTS helpers used by voice-clone and voice-synth.

Provides
--------
Constants
    PROMPT_SCHEMA_VERSION, DEFAULT_TTS_MODEL, DEFAULT_CUSTOM_MODEL, DEFAULT_DESIGN_MODEL,
    DEFAULT_WHISPER, WHISPER_COMPUTE, MAX_NEW_TOKENS_DEFAULT,
    AUDIO_TOKENS_PER_SEC, CHARS_PER_SECOND, TOKEN_MARGIN, CPU_TOKENS_PER_SEC,
    QWEN3_LANGUAGES, QWEN3_CUSTOM_SPEAKERS, _LANGID_TO_QWEN,
    QWEN3_CUSTOMVOICE_FLAG_ENV,
    GENERATION_PROFILE_DEFAULT, GENERATION_PROFILE_PRESETS,
    GENERATION_PROFILE_CHOICES

Token budget
    estimate_max_new_tokens(text) → int
    resolve_generation_profile(profile, voice_defaults) -> (profile, source)
    build_generation_kwargs(...) -> (kwargs, metadata)

Language helpers
    detect_language_from_text(text) → str | None
    resolve_language(flag, ref_language, text) → str
    validate_language(language, model) → str
    supported_speakers(model) → list[str]
    validate_speaker(speaker, model) → str

Device / dtype
    get_device() → str
    best_dtype(device, dtype_str) → torch.dtype

Whisper (faster-whisper) helpers
    ctranslate2_device() → str
    cuda_ctranslate2_compute_type() → str
    build_whisper_model(model_name, num_threads) → WhisperModel
    transcribe_segment(wav_path, wm, beam_size) → (transcript, avg_logprob, lang, lang_prob)
    transcribe_ref(wav_path, whisper_model, num_threads) → same

Model / synthesis
    load_tts_model(model_name, num_threads, dtype_str) → Qwen3TTSModel
    synthesise_clone(text, language, model, prompt, seed, gen_kwargs, timeout_s) → (wav, sr)
    synthesise_custom_voice(text, language, model, speaker, instruct, seed, gen_kwargs, timeout_s)
        → (wav, sr)
    synthesise(text, language, model, prompt, seed, gen_kwargs, timeout_s) → (wav, sr)

Instruction templates
    load_instruction_templates(styles_path) → dict
    resolve_instruction_template(name, styles_path) → str | None
    load_style_presets(...) / apply_style(...) remain compatibility helpers

Utilities
    env_flag_enabled(name, default=True) -> bool
    custom_voice_feature_enabled() -> bool
    custom_voice_feature_env_value() -> str | None
    Timer — context-manager that prints elapsed time
"""

import os
import sys
import time
import threading
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── constants ──────────────────────────────────────────────────────────────────

# Bump to invalidate all cached prompt pickles (e.g. after format change).
# Must match across voice-clone and voice-synth — the single source is here.
PROMPT_SCHEMA_VERSION = 1

DEFAULT_TTS_MODEL    = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_WHISPER      = "small"
WHISPER_COMPUTE      = "int8"   # CTranslate2 compute type for CPU whisper

# Hard ceiling on generated tokens.  At 12 Hz this allows ~341 s of audio —
# preventing runaway generation when EOS is unreliable on CPU.
MAX_NEW_TOKENS_DEFAULT = 4096

# Audio synthesis codec rate — used to derive a tight per-request token budget.
AUDIO_TOKENS_PER_SEC = 12      # Qwen3-TTS codec tokens per second of output
CHARS_PER_SECOND     = 13.0    # typical speaking rate (characters / second)
TOKEN_MARGIN         = 3.5     # safety headroom multiplier over the naive estimate

# Empirical CPU-only throughput for ETA display / auto-timeout.
# Apple M-series Mac Mini observed: ~0.18 t/s.  Faster CPUs: 0.3–0.6 t/s.
CPU_TOKENS_PER_SEC   = 0.18

# Deterministic generation profiles for clone/custom-voice synthesis.
# Profiles define stochasticity, while max_new_tokens_policy controls
# budget scaling derived from text length.
GENERATION_PROFILE_DEFAULT = "balanced"
GENERATION_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "temperature": 0.45,
        "top_p": 0.78,
        "repetition_penalty": 1.12,
        "max_new_tokens_policy": "stable",
    },
    "balanced": {
        "temperature": 0.70,
        "top_p": 0.90,
        "repetition_penalty": 1.05,
        "max_new_tokens_policy": "balanced",
    },
    "expressive": {
        "temperature": 0.92,
        "top_p": 0.96,
        "repetition_penalty": 1.00,
        "max_new_tokens_policy": "expressive",
    },
}
GENERATION_PROFILE_CHOICES = tuple(sorted(GENERATION_PROFILE_PRESETS.keys()))

_MAX_NEW_TOKENS_POLICY_SCALE = {
    "stable": 0.90,
    "balanced": 1.00,
    "expressive": 1.20,
}

# All languages supported by the Qwen3-TTS Base model.
QWEN3_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# ISO 639-1 code → Qwen3 language name (for langid-based auto-detection).
_LANGID_TO_QWEN: dict[str, str] = {
    "zh": "Chinese",  "en": "English",    "ja": "Japanese",
    "ko": "Korean",   "de": "German",     "fr": "French",
    "ru": "Russian",  "pt": "Portuguese", "es": "Spanish",
    "it": "Italian",
}

# Built-in speakers published for Qwen3 CustomVoice.
# Used as a fallback if the runtime API does not expose get_supported_speakers().
QWEN3_CUSTOM_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

# Minimum method surface required by this project across Qwen3 modes.
QWEN3_REQUIRED_MODEL_METHODS: dict[str, tuple[str, ...]] = {
    "clone": ("create_voice_clone_prompt", "generate_voice_clone"),
    "custom_voice": ("generate_custom_voice", "get_supported_speakers"),
    "voice_design": ("generate_voice_design",),
}
QWEN3_CUSTOMVOICE_FLAG_ENV = "QWEN3_ENABLE_CUSTOMVOICE"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
    "float16":  torch.float16,
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}


def env_flag_enabled(name: str, default: bool = True) -> bool:
    """
    Parse a boolean feature flag from the environment.

    Accepted truthy values:  1, true, yes, on, enabled
    Accepted falsy values:   0, false, no, off, disabled
    Empty/unset or invalid values fall back to *default*.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if not val:
        return default
    if val in _TRUE_ENV_VALUES:
        return True
    if val in _FALSE_ENV_VALUES:
        return False
    print(
        f"  WARNING: invalid boolean env value {name}={raw!r}; "
        f"using default {int(default)}.",
        file=sys.stderr,
    )
    return default


def custom_voice_feature_enabled() -> bool:
    """
    Return whether CustomVoice features are enabled for rollout control.
    """
    return env_flag_enabled(QWEN3_CUSTOMVOICE_FLAG_ENV, default=True)


def custom_voice_feature_env_value() -> str | None:
    """
    Return the raw feature-flag env value (trimmed), or None when unset.
    """
    raw = os.getenv(QWEN3_CUSTOMVOICE_FLAG_ENV)
    if raw is None:
        return None
    trimmed = raw.strip()
    return trimmed or None


# ── token budget ───────────────────────────────────────────────────────────────

def estimate_max_new_tokens(text: str) -> int:
    """
    Return a tight token budget derived from text length.

    For short sentences this produces a ceiling 5-10× smaller than
    MAX_NEW_TOKENS_DEFAULT, cutting worst-case CPU generation time proportionally.

    Examples at 13 chars/s, 12 Hz, 3.5× margin:
        100 chars  →  ~323 tokens  (~27 s ceiling)
        158 chars  →  ~508 tokens  (~42 s ceiling)
        400 chars  →  ~1292 tokens (~108 s ceiling)

    The hard ceiling MAX_NEW_TOKENS_DEFAULT still applies as a backstop.
    """
    est_seconds = max(3.0, len(text) / CHARS_PER_SECOND)
    est_tokens  = int(est_seconds * AUDIO_TOKENS_PER_SEC * TOKEN_MARGIN)
    return min(est_tokens, MAX_NEW_TOKENS_DEFAULT)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_policy(value: Any) -> str | None:
    if value is None:
        return None
    policy = str(value).strip().lower()
    if policy in _MAX_NEW_TOKENS_POLICY_SCALE:
        return policy
    return None


def _max_new_tokens_for_policy(text: str, policy: str) -> int:
    base = estimate_max_new_tokens(text)
    scale = _MAX_NEW_TOKENS_POLICY_SCALE.get(policy, 1.0)
    proposed = max(64, int(round(base * scale)))
    return min(proposed, MAX_NEW_TOKENS_DEFAULT)


def resolve_generation_profile(
    requested_profile: str | None,
    voice_defaults: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Resolve generation profile and source.

    Source values:
      explicit      --profile flag
      voice_default voice.json generation_defaults.profile
      default       fallback (balanced)
    """
    if requested_profile is not None:
        profile = requested_profile.strip().lower()
        if profile not in GENERATION_PROFILE_PRESETS:
            options = ", ".join(GENERATION_PROFILE_CHOICES)
            print(
                f"ERROR: unknown generation profile '{requested_profile}'. "
                f"Choose one of: {options}",
                file=sys.stderr,
            )
            sys.exit(1)
        return profile, "explicit"

    if isinstance(voice_defaults, dict):
        profile = str(voice_defaults.get("profile", "")).strip().lower()
        if profile in GENERATION_PROFILE_PRESETS:
            return profile, "voice_default"

    return GENERATION_PROFILE_DEFAULT, "default"


def build_generation_kwargs(
    *,
    text: str,
    profile: str,
    voice_defaults: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_new_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Merge generation controls with precedence:
      1. profile preset
      2. optional voice defaults from registry
      3. explicit CLI overrides
    """
    if profile not in GENERATION_PROFILE_PRESETS:
        options = ", ".join(GENERATION_PROFILE_CHOICES)
        raise ValueError(
            f"Unknown generation profile '{profile}'. Valid: {options}"
        )

    preset = GENERATION_PROFILE_PRESETS[profile]
    kwargs: dict[str, Any] = {
        "temperature": float(preset["temperature"]),
        "top_p": float(preset["top_p"]),
        "repetition_penalty": float(preset["repetition_penalty"]),
    }
    max_tokens_policy = str(preset["max_new_tokens_policy"])

    voice_overrides: list[str] = []
    explicit_overrides: list[str] = []

    if isinstance(voice_defaults, dict):
        for key in ("temperature", "top_p", "repetition_penalty"):
            v = _coerce_float(voice_defaults.get(key))
            if v is not None:
                kwargs[key] = v
                voice_overrides.append(key)
        voice_policy = _coerce_policy(voice_defaults.get("max_new_tokens_policy"))
        if voice_policy is not None:
            max_tokens_policy = voice_policy
            voice_overrides.append("max_new_tokens_policy")

    if temperature is not None:
        kwargs["temperature"] = float(temperature)
        explicit_overrides.append("temperature")
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
        explicit_overrides.append("top_p")
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
        explicit_overrides.append("repetition_penalty")

    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
        explicit_overrides.append("max_new_tokens")
        max_tokens_policy = "explicit"
    else:
        kwargs["max_new_tokens"] = _max_new_tokens_for_policy(text, max_tokens_policy)

    metadata = {
        "profile": profile,
        "max_new_tokens_policy": max_tokens_policy,
        "voice_overrides": voice_overrides,
        "explicit_overrides": explicit_overrides,
    }
    return kwargs, metadata


# ── language helpers ───────────────────────────────────────────────────────────

def detect_language_from_text(text: str) -> str | None:
    """
    Auto-detect the language of *text* using langid.
    Returns a Qwen3 language name, or None on failure / unsupported ISO code.
    """
    try:
        import langid  # type: ignore
        iso, _conf = langid.classify(text)
        return _LANGID_TO_QWEN.get(iso)   # None when not in the supported map
    except Exception:
        return None


def resolve_language(language_flag: str, ref_language: str, text: str) -> str:
    """
    Resolve the synthesis language.

    Resolution order:
      1. ``--language`` not "Auto"  → use it directly.
      2. text ≥ 3 words             → langid detection (when supported).
      3. whisper-detected ref lang  → fall through from ref.
      4. Default                    → "English".
    """
    if language_flag != "Auto":
        return language_flag
    if len(text.split()) >= 3:
        detected = detect_language_from_text(text)
        if detected and detected in QWEN3_LANGUAGES and detected != "Auto":
            return detected
    if ref_language and ref_language not in ("Auto", ""):
        return ref_language
    return "English"


def validate_language(language: str, model) -> str:
    """
    Verify *language* against the model's supported language list.
    Returns the normalised lowercase string accepted by the model.
    Exits with a helpful message if the language is unsupported.
    """
    if language in ("Auto", "auto"):
        return "auto"
    try:
        supported = model.get_supported_languages()
        if supported:
            lang_lower = language.lower()
            supported_lower = {s.lower() for s in supported}
            if lang_lower not in supported_lower:
                print(
                    f"ERROR: language '{language}' is not supported by the loaded model.\n"
                    f"  Supported: {', '.join(sorted(supported))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return lang_lower
    except AttributeError:
        pass   # older qwen-tts versions may not expose get_supported_languages
    return language.lower()


def supported_speakers(model) -> list[str]:
    """
    Return the model's supported speaker list (CustomVoice), if available.

    Resolution order:
      1. Runtime API ``model.get_supported_speakers()``
      2. Known Qwen3 CustomVoice built-in speakers (fallback)
      3. Empty list when speakers are not supported by this model
    """
    try:
        speakers = model.get_supported_speakers()
        if speakers:
            known_by_lower = {s.lower(): s for s in QWEN3_CUSTOM_SPEAKERS}
            normalized = [known_by_lower.get(str(s).strip().lower(), str(s).strip())
                          for s in speakers]
            return sorted(set(normalized), key=str.lower)
    except AttributeError:
        pass

    name = str(getattr(model, "name_or_path", "")).lower()
    if "customvoice" in name:
        return list(QWEN3_CUSTOM_SPEAKERS)
    return []


def validate_speaker(speaker: str, model) -> str:
    """
    Validate *speaker* against this model's supported CustomVoice speakers.
    Returns the canonical speaker label accepted by the model.
    Exits with a helpful message when speakers are unsupported / unknown.
    """
    supported = supported_speakers(model)
    if not supported:
        print(
            "ERROR: this model does not expose CustomVoice speakers.\n"
            "  Use a Qwen3 CustomVoice model, e.g.\n"
            f"    {DEFAULT_CUSTOM_MODEL}",
            file=sys.stderr,
        )
        sys.exit(1)

    normalized = speaker.strip().lower()
    by_lower = {s.lower(): s for s in supported}
    if normalized not in by_lower:
        print(
            f"ERROR: speaker '{speaker}' is not supported by the loaded model.\n"
            f"  Supported: {', '.join(supported)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return by_lower[normalized]


# ── device & dtype ─────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Select the best available compute device.

    Resolution order:
      1. ``TORCH_DEVICE`` env var  — explicit override (set by docker-compose.gpu.yml).
      2. CUDA                      — when available.
      3. CPU                       — universal fallback.
    """
    override = os.getenv("TORCH_DEVICE", "").strip()
    if override:
        if override.startswith("cuda") and not torch.cuda.is_available():
            print(
                "  WARNING: TORCH_DEVICE is set to CUDA but this torch build "
                "has no CUDA support; falling back to CPU.",
                file=sys.stderr,
            )
            return "cpu"
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def best_dtype(device: str, dtype_str: str) -> torch.dtype:
    """
    Resolve *dtype_str* to a concrete ``torch.dtype``.

    When ``dtype_str == "auto"`` (the default):
      CPU                → float32  (native; bfloat16 would add cast overhead)
      CUDA SM < 7.0      → float32  (Maxwell / Pascal lack native FP16 arithmetic;
                                     ops overflow to NaN during LLM sampling)
      CUDA SM 7.0–7.x    → float16  (Volta / Turing — first desktop FP16 that works)
      CUDA SM ≥ 8.0      → bfloat16 (Ampere, Ada Lovelace, Hopper)

    Note: Maxwell SM 5.x has no native FP16 compute units. Even though
    float16 halves memory bandwidth (attractive for memory-bound autoregressive
    inference), the forward pass produces NaN/inf logits on these chips.
    float32 is required for numerical stability.

    Explicit strings ("bfloat16", "float32", "float16") are returned as-is.
    """
    if dtype_str != "auto":
        return _DTYPE_MAP[dtype_str]
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    idx = int(device.split(":")[-1]) if ":" in device else 0
    major, minor = torch.cuda.get_device_capability(idx)
    if major >= 8:
        return torch.bfloat16
    if (major, minor) >= (7, 0):
        return torch.float16
    return torch.float32


# ── whisper helpers ────────────────────────────────────────────────────────────

def ctranslate2_device() -> str:
    """
    Return the device string for CTranslate2 / faster-whisper.

    CTranslate2 (cu124/cu126 build) only ships CUDA kernels for SM 7.0+.
    Pre-Volta GPUs fall back to CPU so transcription still works, while
    PyTorch-based models (Demucs, Qwen) continue to use the GPU.
    """
    device = get_device()
    if device.startswith("cuda") and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc < (7, 0):
            return "cpu"
    return device


def cuda_ctranslate2_compute_type() -> str:
    """
    Return the most efficient CTranslate2 ``compute_type`` for the current GPU.

      SM ≥ 7.0  → "float16"
      SM ≥ 6.1  → "int8"
      otherwise → "float32"
    """
    cc = torch.cuda.get_device_capability()
    if cc >= (7, 0):
        return "float16"
    if cc >= (6, 1):
        return "int8"
    return "float32"


def build_whisper_model(model_name: str, num_threads: int):
    """Load and return a ``faster_whisper.WhisperModel`` for the current device."""
    from faster_whisper import WhisperModel  # type: ignore
    device       = ctranslate2_device()
    compute_type = cuda_ctranslate2_compute_type() if device.startswith("cuda") else WHISPER_COMPUTE
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=num_threads,
    )


def transcribe_segment(
    wav_path: Path,
    wm,
    beam_size: int = 5,
) -> tuple[str, float, str, float]:
    """
    Transcribe *wav_path* with a pre-loaded faster-whisper model.

    Returns ``(transcript, avg_logprob, detected_language_qwen, language_probability)``.
    """
    segments, info = wm.transcribe(
        str(wav_path),
        beam_size=beam_size,
        vad_filter=True,
    )
    texts:    list[str]   = []
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
    Load faster-whisper and transcribe at full quality (beam_size=5).

    Returns ``(transcript, avg_logprob, detected_language_qwen, language_probability)``.
    """
    device       = ctranslate2_device()
    compute_type = cuda_ctranslate2_compute_type() if device.startswith("cuda") else WHISPER_COMPUTE
    print(
        f"  loading faster-whisper/{whisper_model} "
        f"({compute_type}, {device}, {num_threads} threads)..."
    )
    wm = build_whisper_model(whisper_model, num_threads)
    transcript, avg_logprob, detected_qwen, lang_prob = transcribe_segment(
        wav_path, wm, beam_size=5
    )
    print(
        f"  [{detected_qwen} p={lang_prob:.2f}] "
        f"conf={avg_logprob:.2f}  {transcript!r}"
    )
    return transcript, avg_logprob, detected_qwen, lang_prob


# ── model loading ──────────────────────────────────────────────────────────────

def _has_flash_attn() -> bool:
    """Return True when the flash-attn package is importable (i.e. installed)."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def load_tts_model(model_name: str, num_threads: int, dtype_str: str = "auto"):
    """
    Load a Qwen3-TTS model targeting the best available device.

    dtype_str:
      "auto"     — float32 on CPU; bfloat16 on CUDA Ampere+ (SM ≥ 8.0);
                   float16 on older CUDA GPUs (Volta / Turing SM 7.x).
      "bfloat16" — CUDA SM ≥ 8.0 only.
      "float32"  — safest/fastest on CPU; debug fallback.
      "float16"  — for CUDA SM < 8.0.

    attn_implementation selection:
      CUDA + flash-attn installed → "flash_attention_2"
        qwen_tts's custom attention layers require flash-attn to dispatch GPU
        kernels.  Without it they fall back to a Python loop that runs on CPU
        regardless of where the model weights are — producing ~1× RTF instead
        of the expected 30–100× on modern GPUs.
      Otherwise → "sdpa"  (PyTorch fused SDPA, faster than "eager" on both CPU
        and CUDA, no extra dependencies).
    """
    from qwen_tts import Qwen3TTSModel  # type: ignore
    device = get_device()
    dtype  = best_dtype(device, dtype_str)
    torch.set_num_threads(num_threads)

    on_cuda = device.startswith("cuda")
    if on_cuda and _has_flash_attn():
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"

    # qwen_tts demo always uses "cuda:0" explicitly for single-GPU; match that.
    device_map = "cuda:0" if on_cuda else device

    print(f"  loading {model_name} ({device_map}, {dtype}, attn={attn_impl}, threads={num_threads}) …")
    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_impl,
    )


def qwen_tts_package_version() -> str:
    """
    Return the installed qwen-tts package version.
    """
    try:
        return importlib_metadata.version("qwen-tts")
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def probe_qwen_tts_api(
    required_methods: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """
    Inspect the qwen-tts model class for required method compatibility.

    Returns a machine-friendly payload with a per-method availability matrix.
    """
    required = required_methods or QWEN3_REQUIRED_MODEL_METHODS
    payload: dict[str, Any] = {
        "package_version": qwen_tts_package_version(),
        "class_name": "qwen_tts.Qwen3TTSModel",
        "required_methods": {k: list(v) for k, v in required.items()},
        "present": {},
        "missing": {},
        "compatible": False,
        "error": None,
    }

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
        for mode, methods in required.items():
            payload["present"][mode] = []
            payload["missing"][mode] = list(methods)
        return payload

    all_missing: list[str] = []
    for mode, methods in required.items():
        present: list[str] = []
        missing: list[str] = []
        for name in methods:
            attr = getattr(Qwen3TTSModel, name, None)
            if callable(attr):
                present.append(name)
            else:
                missing.append(name)
                all_missing.append(f"{mode}:{name}")
        payload["present"][mode] = present
        payload["missing"][mode] = missing

    payload["compatible"] = not all_missing
    return payload


# ── synthesis ──────────────────────────────────────────────────────────────────

def _run_generation(
    generate_fn,
    *,
    op_name: str,
    gen_kwargs: dict[str, Any],
    timeout_s: float | None,
) -> tuple[np.ndarray, int]:
    """
    Run a TTS generation callable with heartbeat logging and optional timeout.

    Runs generation on a background thread so a progress line is printed every
    10 s.  Without this the call is silent for 30-60 min on CPU-only inference.

    Parameters
    ----------
    timeout_s:
        Raise ``TimeoutError`` after this many wall-clock seconds.
        ``None`` → auto-set to 4× the CPU ETA estimate.
        ``0``    → no timeout.

    Returns ``(wav_array, sample_rate)``.
    """
    HEARTBEAT_INTERVAL = 10   # seconds between progress lines

    budget = gen_kwargs.get("max_new_tokens")
    if isinstance(budget, int):
        ceiling_s  = budget / AUDIO_TOKENS_PER_SEC
        est_wall_s = budget / CPU_TOKENS_PER_SEC
        print(
            f"  token budget : {budget} tokens  "
            f"(audio ceiling ≈ {ceiling_s:.0f}s)",
            flush=True,
        )
        print(
            f"  CPU ETA      : ~{est_wall_s / 60:.0f} min  "
            f"(use GPU for fast inference — see README)",
            flush=True,
        )
        if timeout_s is None:
            timeout_s = est_wall_s * 4.0   # 4× CPU ETA — generous but bounded

    # timeout_s == 0 means "no timeout"
    effective_timeout = None if (timeout_s is not None and timeout_s <= 0) else timeout_s

    result: list[Any]                  = [None]
    exc:    list[BaseException | None] = [None]

    def _worker() -> None:
        try:
            result[0] = generate_fn()
        except BaseException as e:  # noqa: BLE001
            exc[0] = e

    t_start = time.perf_counter()
    thread  = threading.Thread(target=_worker, daemon=True, name="tts-generate")
    thread.start()

    while thread.is_alive():
        thread.join(timeout=HEARTBEAT_INTERVAL)
        if thread.is_alive():
            elapsed = time.perf_counter() - t_start
            print(f"  … still generating ({elapsed:.0f}s elapsed) …", flush=True)
            if effective_timeout is not None and elapsed >= effective_timeout:
                print(
                    f"\n  ✗  Timeout: generation exceeded {effective_timeout:.0f}s "
                    f"(4× CPU ETA).  Kill the container to free resources.",
                    flush=True,
                )
                print(
                    "     To run longer pass --timeout <seconds>, or 0 to disable.",
                    flush=True,
                )
                print(
                    "     For practical speeds, run on a CUDA GPU — see README.",
                    flush=True,
                )
                raise TimeoutError(
                    f"{op_name} exceeded {effective_timeout:.0f}s "
                    "wall-clock timeout."
                )

    elapsed = time.perf_counter() - t_start
    print(f"    ↳ {op_name}: {elapsed:.2f}s")

    if exc[0] is not None:
        raise exc[0]   # re-raise worker exception in the main thread

    wavs, sr = result[0]
    return wavs[0], sr


def synthesise_clone(
    text: str,
    language: str,
    model,
    prompt,
    seed: int | None,
    gen_kwargs: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """Generate speech from a voice-clone prompt."""
    if seed is not None:
        torch.manual_seed(seed)
    kw = gen_kwargs or {}
    return _run_generation(
        lambda: model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            **kw,
        ),
        op_name="generate_voice_clone",
        gen_kwargs=kw,
        timeout_s=timeout_s,
    )


def synthesise_custom_voice(
    text: str,
    language: str,
    model,
    speaker: str,
    instruct: str | None,
    seed: int | None,
    gen_kwargs: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """Generate speech using Qwen3 CustomVoice built-in speakers."""
    if seed is not None:
        torch.manual_seed(seed)
    kw = gen_kwargs or {}

    params: dict[str, Any] = {
        "text": text,
        "language": language,
        "speaker": speaker,
        **kw,
    }
    if instruct and instruct.strip():
        params["instruct"] = instruct.strip()

    return _run_generation(
        lambda: model.generate_custom_voice(**params),
        op_name="generate_custom_voice",
        gen_kwargs=kw,
        timeout_s=timeout_s,
    )


def synthesise(
    text: str,
    language: str,
    model,
    prompt,
    seed: int | None,
    gen_kwargs: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    Backward-compatible alias for clone-prompt synthesis.

    New code should prefer ``synthesise_clone`` / ``synthesise_custom_voice``
    to make the generation mode explicit.
    """
    return synthesise_clone(text, language, model, prompt, seed, gen_kwargs, timeout_s)


# ── instruction templates ──────────────────────────────────────────────────────

# Canonical styles.yaml location inside the project.
# Individual apps may still pass a different path if needed.
STYLES_YAML = Path(__file__).resolve().parent / "styles.yaml"


def load_instruction_templates(styles_path: Path | None = None) -> dict[str, str]:
    """
    Load YAML instruction templates.

    Defaults to ``lib/styles.yaml`` (the single canonical source). Supports both:
      - ``name: {instruct: "..."} `` (preferred)
      - legacy ``name: {prefix: "...", suffix: "..."} `` (fallback)

    Returns ``{}`` if the file is missing or cannot be parsed.
    """
    path = styles_path or STYLES_YAML
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"  WARNING: could not load styles.yaml ({path}): {exc}", file=sys.stderr)
        return {}

    if not isinstance(raw, dict):
        return {}

    templates: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, str):
            instruct = value.strip()
        elif isinstance(value, dict):
            instruct = str(value.get("instruct") or value.get("instruction") or "").strip()
            if not instruct:
                # Backward-compat fallback for older prefix/suffix entries.
                prefix = str(value.get("prefix") or "").strip()
                suffix = str(value.get("suffix") or "").strip()
                instruct = " ".join(x for x in (prefix, suffix) if x)
        else:
            continue
        if instruct:
            templates[name] = instruct
    return templates


def resolve_instruction_template(
    name: str | None,
    styles_path: Path | None = None,
) -> str | None:
    """
    Resolve a named instruction template to a concrete instruction string.
    """
    if not name:
        return None
    templates = load_instruction_templates(styles_path)
    if name not in templates:
        available = ", ".join(sorted(templates)) or "(none)"
        print(
            f"  WARNING: instruction template '{name}' not found. Available: {available}",
            file=sys.stderr,
        )
        return None
    return templates[name]


def load_style_presets(styles_path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Backward-compatible alias for older imports.
    """
    templates = load_instruction_templates(styles_path)
    return {name: {"instruct": instruct} for name, instruct in templates.items()}


def apply_style(
    text: str,
    style_name: str | None,
    styles_path: Path | None = None,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
) -> str:
    """
    Backward-compatible text compositor.

    Deprecated behavior: style_name is no longer injected into clone text.
    Clone mode style should come from reference audio; instruction templates are
    intended for CustomVoice/VoiceDesign instruct fields.
    """
    if style_name:
        _ = styles_path  # keep arg for compatibility
        print(
            "  WARNING: style text injection is deprecated and ignored; "
            "use CustomVoice/VoiceDesign instruction templates instead.",
            file=sys.stderr,
        )

    parts = []
    if prompt_prefix:
        parts.append(prompt_prefix.strip())
    parts.append(text)
    if prompt_suffix:
        parts.append(prompt_suffix.strip())
    return " ".join(p for p in parts if p)


# ── utility ────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that prints elapsed time on exit."""

    def __init__(self, label: str) -> None:
        self._label  = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
        print(f"    ↳ {self._label}: {self.elapsed:.2f}s")
