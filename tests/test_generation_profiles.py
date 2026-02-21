from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from tts import (  # noqa: E402
    build_generation_kwargs,
    load_instruction_templates,
    resolve_generation_profile,
    resolve_instruction_template,
)


def test_resolve_generation_profile_precedence() -> None:
    profile, source = resolve_generation_profile("stable", {"profile": "expressive"})
    assert profile == "stable"
    assert source == "explicit"


def test_resolve_generation_profile_uses_voice_default() -> None:
    profile, source = resolve_generation_profile(None, {"profile": "expressive"})
    assert profile == "expressive"
    assert source == "voice_default"


def test_build_generation_kwargs_precedence() -> None:
    kwargs, meta = build_generation_kwargs(
        text="The quick brown fox jumps over the lazy dog.",
        profile="balanced",
        voice_defaults={
            "temperature": 0.81,
            "max_new_tokens_policy": "stable",
        },
        top_p=0.88,
    )

    assert kwargs["temperature"] == 0.81
    assert kwargs["top_p"] == 0.88
    assert "temperature" in meta["voice_overrides"]
    assert "top_p" in meta["explicit_overrides"]
    assert kwargs["max_new_tokens"] > 0


def test_max_new_tokens_policy_scales_with_profile() -> None:
    text = " ".join(["longer"] * 120)
    stable_kwargs, _ = build_generation_kwargs(text=text, profile="stable")
    expressive_kwargs, _ = build_generation_kwargs(text=text, profile="expressive")
    assert stable_kwargs["max_new_tokens"] < expressive_kwargs["max_new_tokens"]


def test_instruction_templates_load_and_resolve() -> None:
    templates = load_instruction_templates()
    assert "serious_doc" in templates
    assert resolve_instruction_template("serious_doc")
