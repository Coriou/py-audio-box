import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

VOICE_SYNTH_PATH = ROOT / "apps" / "voice-synth" / "voice-synth.py"


def _load_voice_synth_module():
    spec = importlib.util.spec_from_file_location("voice_synth_app", VOICE_SYNTH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assert_selection_qa_ok_accepts_good_payload() -> None:
    vs = _load_voice_synth_module()
    vs.assert_selection_qa_ok(
        {"transcript": "hello", "intelligibility": 0.8, "duration_sec": 1.2},
        "take_01",
    )


def test_assert_selection_qa_ok_raises_on_error_payload() -> None:
    vs = _load_voice_synth_module()
    with pytest.raises(RuntimeError):
        vs.assert_selection_qa_ok(
            {"transcript": "", "intelligibility": 0.0, "duration_sec": 0.0, "error": "boom"},
            "take_02",
        )


def test_resolve_instruct_text_prefers_explicit_text() -> None:
    vs = _load_voice_synth_module()
    instruct, source = vs.resolve_instruct_text(
        "Warm, calm delivery",
        None,
        required=False,
        context="test",
    )
    assert instruct == "Warm, calm delivery"
    assert source == "explicit"


def test_resolve_instruct_text_resolves_named_template() -> None:
    vs = _load_voice_synth_module()
    instruct, source = vs.resolve_instruct_text(
        None,
        "serious_doc",
        required=True,
        context="test",
    )
    assert isinstance(instruct, str) and len(instruct) > 0
    assert source == "style_template:serious_doc"


def test_resolve_instruct_text_rejects_mixed_inputs() -> None:
    vs = _load_voice_synth_module()
    with pytest.raises(ValueError):
        vs.resolve_instruct_text(
            "explicit text",
            "serious_doc",
            required=False,
            context="test",
        )


def test_resolve_instruct_text_requires_one_when_required() -> None:
    vs = _load_voice_synth_module()
    with pytest.raises(ValueError):
        vs.resolve_instruct_text(
            None,
            None,
            required=True,
            context="test",
        )


def test_resolve_instruct_text_rejects_unknown_template() -> None:
    vs = _load_voice_synth_module()
    with pytest.raises(ValueError):
        vs.resolve_instruct_text(
            None,
            "not_a_template",
            required=True,
            context="test",
        )


def test_parser_accepts_new_instruction_template_flags() -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()

    speak_args = ap.parse_args(
        ["speak", "--speaker", "Ryan", "--text", "hello", "--instruct-style", "serious_doc"]
    )
    assert speak_args.instruct_style == "serious_doc"

    register_args = ap.parse_args(
        [
            "register-builtin",
            "--voice-name", "newsroom",
            "--speaker", "Ryan",
            "--instruct-default-style", "serious_doc",
            "--tone", "promo",
            "--tone-instruct-style", "energetic",
        ]
    )
    assert register_args.instruct_default_style == "serious_doc"
    assert register_args.tone_instruct_style == "energetic"

    design_args = ap.parse_args(
        ["design-voice", "--instruct-style", "warm", "--ref-text", "hello world"]
    )
    assert design_args.instruct_style == "warm"
