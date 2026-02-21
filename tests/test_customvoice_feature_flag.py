import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from tts import (  # noqa: E402
    QWEN3_CUSTOMVOICE_FLAG_ENV,
    custom_voice_feature_enabled,
)


VOICE_SYNTH_PATH = ROOT / "apps" / "voice-synth" / "voice-synth.py"


def _load_voice_synth_module():
    spec = importlib.util.spec_from_file_location("voice_synth_feature_flag", VOICE_SYNTH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_voice_feature_flag_default_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(QWEN3_CUSTOMVOICE_FLAG_ENV, raising=False)
    assert custom_voice_feature_enabled() is True


def test_custom_voice_feature_flag_false_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "false", "no", "off", "disabled"):
        monkeypatch.setenv(QWEN3_CUSTOMVOICE_FLAG_ENV, value)
        assert custom_voice_feature_enabled() is False


def test_list_speakers_requires_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["list-speakers"])

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    with pytest.raises(SystemExit) as exc:
        vs.cmd_list_speakers(args)
    assert exc.value.code == 1


def test_register_builtin_requires_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(
        ["register-builtin", "--voice-name", "newsroom", "--speaker", "Ryan"]
    )

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    with pytest.raises(SystemExit) as exc:
        vs.cmd_register_builtin(args)
    assert exc.value.code == 1


def test_speak_speaker_mode_requires_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["speak", "--speaker", "Ryan", "--text", "hello"])

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    with pytest.raises(SystemExit) as exc:
        vs.cmd_speak(args)
    assert exc.value.code == 1


def test_speak_named_custom_voice_requires_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["speak", "--voice", "newsroom", "--text", "hello"])

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    monkeypatch.setattr(
        vs,
        "resolve_voice",
        lambda *_a, **_kw: {
            "voice_id": "newsroom",
            "engine_mode": "custom_voice",
            "prompt_path": None,
            "speaker": "Ryan",
            "instruct": "Calm delivery",
            "instruct_source": "default",
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "language_default": "English",
            "generation_defaults": {},
        },
    )
    with pytest.raises(SystemExit) as exc:
        vs.cmd_speak(args)
    assert exc.value.code == 1
