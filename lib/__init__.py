"""
lib — shared helpers for all py-audio-box apps.

Modules
-------
voices  — named voice registry (VoiceRegistry, validate_slug)
tts     — TTS constants, device/dtype selection, model loading, synthesis,
          language helpers, whisper helpers, Timer, style presets
audio   — audio file operations: normalise, trim, sha256, get_duration
vad     — Silero VAD: load, run, segment helpers

Import pattern (works from any app script):
    import sys
    from pathlib import Path
    _LIB = str(Path(__file__).resolve().parent.parent.parent / "lib")
    if _LIB not in sys.path:
        sys.path.insert(0, _LIB)

    from tts import load_tts_model, synthesise, QWEN3_LANGUAGES
    from audio import get_duration, normalize_audio
    from vad import load_silero, run_silero
    from voices import VoiceRegistry
"""
