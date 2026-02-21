import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

VOICE_SYNTH_PATH = ROOT / "apps" / "voice-synth" / "voice-synth.py"
VOICE_CLONE_PATH = ROOT / "apps" / "voice-clone" / "voice-clone.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_voice_synth_module():
    return _load_module(VOICE_SYNTH_PATH, "voice_synth_cap_meta")


def _load_voice_clone_module():
    return _load_module(VOICE_CLONE_PATH, "voice_clone_cap_meta")


def test_capabilities_parser_flags() -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(
        [
            "capabilities",
            "--json",
            "--strict",
            "--skip-speaker-probe",
            "--require-runtime-speakers",
        ]
    )
    assert args.command == "capabilities"
    assert args.json is True
    assert args.strict is True
    assert args.skip_speaker_probe is True
    assert args.require_runtime_speakers is True


def test_capabilities_payload_static_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["capabilities", "--skip-speaker-probe"])

    monkeypatch.setattr(vs, "get_device", lambda: "cpu")
    monkeypatch.setattr(vs, "best_dtype", lambda _device, _dtype: vs.torch.float32)
    monkeypatch.setattr(vs, "ctranslate2_device", lambda: "cpu")
    monkeypatch.setattr(
        vs,
        "probe_qwen_tts_api",
        lambda required_methods=None: {
            "compatible": True,
            "missing": {},
            "required_methods": required_methods or {},
        },
    )
    monkeypatch.setattr(vs, "qwen_tts_package_version", lambda: "0.1.1")

    payload, failures = vs.build_capabilities_payload(args)

    assert payload["schema_version"] == vs.CAPABILITIES_SCHEMA_VERSION
    assert payload["custom_voice"]["speaker_probe_mode"] == "static_fallback"
    assert payload["custom_voice"]["speaker_count"] == len(vs.QWEN3_CUSTOM_SPEAKERS)
    assert failures == []


def test_capabilities_payload_customvoice_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["capabilities", "--skip-speaker-probe"])

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    monkeypatch.setattr(vs, "get_device", lambda: "cpu")
    monkeypatch.setattr(vs, "best_dtype", lambda _device, _dtype: vs.torch.float32)
    monkeypatch.setattr(vs, "ctranslate2_device", lambda: "cpu")

    observed: dict[str, dict[str, list[str]]] = {}

    def _probe(required_methods=None):
        observed["required_methods"] = required_methods or {}
        return {
            "compatible": True,
            "missing": {},
            "required_methods": required_methods or {},
        }

    monkeypatch.setattr(vs, "probe_qwen_tts_api", _probe)
    monkeypatch.setattr(vs, "qwen_tts_package_version", lambda: "0.1.1")

    payload, failures = vs.build_capabilities_payload(args)

    assert payload["feature_flags"]["custom_voice"]["enabled"] is False
    assert payload["custom_voice"]["enabled"] is False
    assert payload["custom_voice"]["speaker_probe_mode"] == "feature_disabled"
    assert payload["custom_voice"]["speaker_count"] == 0
    assert "custom_voice" not in observed["required_methods"]
    assert failures == []


def test_capabilities_payload_customvoice_disabled_runtime_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(
        [
            "capabilities",
            "--skip-speaker-probe",
            "--require-runtime-speakers",
        ]
    )

    monkeypatch.setenv(vs.QWEN3_CUSTOMVOICE_FLAG_ENV, "0")
    monkeypatch.setattr(vs, "get_device", lambda: "cpu")
    monkeypatch.setattr(vs, "best_dtype", lambda _device, _dtype: vs.torch.float32)
    monkeypatch.setattr(vs, "ctranslate2_device", lambda: "cpu")
    monkeypatch.setattr(
        vs,
        "probe_qwen_tts_api",
        lambda required_methods=None: {
            "compatible": True,
            "missing": {},
            "required_methods": required_methods or {},
        },
    )

    _payload, failures = vs.build_capabilities_payload(args)
    assert any("CustomVoice is disabled" in msg for msg in failures)


def test_capabilities_payload_reports_api_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    vs = _load_voice_synth_module()
    ap = vs.build_parser()
    args = ap.parse_args(["capabilities", "--skip-speaker-probe"])

    monkeypatch.setattr(vs, "get_device", lambda: "cpu")
    monkeypatch.setattr(vs, "best_dtype", lambda _device, _dtype: vs.torch.float32)
    monkeypatch.setattr(vs, "ctranslate2_device", lambda: "cpu")
    monkeypatch.setattr(
        vs,
        "probe_qwen_tts_api",
        lambda required_methods=None: {
            "compatible": False,
            "missing": {"custom_voice": ["generate_custom_voice"]},
            "required_methods": required_methods or {},
        },
    )

    _payload, failures = vs.build_capabilities_payload(args)
    assert any("API compatibility failed" in msg for msg in failures)


def test_voice_synth_takes_meta_schema_golden() -> None:
    vs = _load_voice_synth_module()
    meta = vs.build_speak_takes_metadata(
        voice_id="newsroom",
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        engine_mode="custom_voice",
        source_type="custom_voice_named",
        speaker="Ryan",
        instruct="Warm, calm delivery",
        instruct_source="default",
        instruct_style_applied=None,
        language="english",
        ref_language="English",
        raw_text="Hello world",
        text="Hello world",
        tone=None,
        pkl_path=None,
        gen_kwargs={"temperature": 0.7},
        profile_name="balanced",
        profile_source="default",
        generation_meta={"profile": "balanced"},
        voice_generation_defaults={},
        saved_generation_defaults=None,
        chunked=False,
        text_chunks=["Hello world"],
        n_variants=1,
        selection_policy={"enabled": False},
        selection_breakdown=[],
        selected_take=None,
        selected_wav=None,
        load_sec=1.0,
        total_sec=2.0,
        takes=[{"take": "take_01"}],
    )

    assert meta["schema_version"] == vs.TAKES_META_SCHEMA_VERSION
    assert list(meta.keys()) == [
        "schema_version",
        "app",
        "created_at",
        "voice_id",
        "model",
        "engine_mode",
        "source_type",
        "speaker",
        "instruct",
        "instruct_source",
        "instruct_style",
        "language",
        "ref_language",
        "original_text",
        "text",
        "tone",
        "prompt_path",
        "generation_kwargs",
        "generation_profile",
        "generation_profile_source",
        "generation_profile_meta",
        "voice_generation_defaults",
        "saved_generation_defaults",
        "chunked",
        "n_chunks",
        "variants",
        "selection_policy",
        "selection_metrics",
        "selected_take",
        "selected_wav",
        "load_sec",
        "total_sec",
        "takes",
    ]


def test_voice_clone_takes_meta_schema_golden() -> None:
    vc = _load_voice_clone_module()
    ref = vc.RefResult(
        ref_hash="abc123",
        ref_dir=Path("/tmp/ref"),
        normalized=Path("/tmp/ref/normalized.wav"),
        segment=Path("/tmp/ref/segment.wav"),
        seg_start=1.0,
        seg_end=9.5,
        transcript="hello",
        transcript_conf=-0.3,
        whisper_model="small",
        ref_language="English",
        ref_language_prob=0.98,
    )

    meta = vc.build_clone_takes_metadata(
        model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language="english",
        ref=ref,
        x_vector_only=False,
        tone="neutral",
        gen_kwargs={"temperature": 0.7},
        profile_name="balanced",
        profile_source="default",
        generation_meta={"profile": "balanced"},
        voice_generation_defaults={},
        saved_generation_defaults=None,
        selection_policy={"enabled": False},
        selection_breakdown=[],
        selected_take=None,
        selected_wav=None,
        text="Hello world",
        prep_sec=1.0,
        model_sec=2.0,
        total_sec=3.0,
        n_variants=1,
        takes=[{"take": "take_01"}],
    )

    assert meta["schema_version"] == vc.TAKES_META_SCHEMA_VERSION
    assert list(meta.keys()) == [
        "schema_version",
        "app",
        "created_at",
        "model",
        "engine_mode",
        "source_type",
        "speaker",
        "instruct",
        "instruct_source",
        "language",
        "ref_language_detected",
        "ref_language_probability",
        "x_vector_only",
        "tone",
        "generation_kwargs",
        "generation_profile",
        "generation_profile_source",
        "generation_profile_meta",
        "voice_generation_defaults",
        "saved_generation_defaults",
        "selection_policy",
        "selection_metrics",
        "selected_take",
        "selected_wav",
        "text",
        "timings",
        "reference",
        "variants",
        "takes",
    ]
    assert list(meta["timings"].keys()) == ["prep_sec", "model_load_sec", "total_sec"]
    assert list(meta["reference"].keys()) == [
        "hash",
        "segment_start",
        "segment_end",
        "segment_duration",
        "transcript",
        "transcript_conf",
        "whisper_model",
    ]
