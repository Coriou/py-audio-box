import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

VOICE_CLONE_PATH = ROOT / "apps" / "voice-clone" / "voice-clone.py"


def _load_voice_clone_module():
    spec = importlib.util.spec_from_file_location("voice_clone_app", VOICE_CLONE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transcript_confidence_gate_threshold() -> None:
    vc = _load_voice_clone_module()
    threshold = vc.QUALITY_GATE_LOGPROB
    assert vc.transcript_confidence_passes(threshold)
    assert vc.transcript_confidence_passes(threshold + 0.01)
    assert not vc.transcript_confidence_passes(threshold - 0.01)


def test_segment_scoring_penalizes_bad_acoustics() -> None:
    vc = _load_voice_clone_module()
    clean_score, _ = vc._score_segment(
        duration=vc.SWEET_SPOT_SECONDS,
        avg_logprob=-0.3,
        acoustics={
            "clipping_score": 1.0,
            "noise_score": 1.0,
            "speech_continuity_score": 1.0,
        },
    )
    bad_score, _ = vc._score_segment(
        duration=vc.SWEET_SPOT_SECONDS,
        avg_logprob=-0.3,
        acoustics={
            "clipping_score": 0.0,
            "noise_score": 0.0,
            "speech_continuity_score": 0.0,
        },
    )
    assert clean_score > bad_score


def test_candidate_scoring_cache_version_guard() -> None:
    vc = _load_voice_clone_module()
    assert vc.candidate_scoring_cache_is_current(
        {"candidate_scoring_version": vc.CANDIDATE_SCORING_VERSION}
    )
    assert not vc.candidate_scoring_cache_is_current({})
    assert not vc.candidate_scoring_cache_is_current({"candidate_scoring_version": 1})


def test_instruction_like_text_hack_detection() -> None:
    vc = _load_voice_clone_module()
    assert vc.instruction_like_text_hack("[Calm documentary tone] Hello.") is not None
    assert vc.instruction_like_text_hack("Style: calm narrator") is not None
    assert vc.instruction_like_text_hack("Hello world.") is None


def test_prepare_ref_uses_provided_ref_text_without_whisper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vc = _load_voice_clone_module()
    ref_audio = tmp_path / "ref.raw"
    ref_audio.write_bytes(b"raw")

    def _fake_normalize(_src: Path, dest: Path, sample_rate: int = 16_000) -> None:
        _ = sample_rate
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")

    def _fake_trim(
        _src: Path,
        _start: float,
        _duration: float,
        dest: Path,
        sample_rate: int = 24_000,
    ) -> None:
        _ = sample_rate
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"seg")

    def _should_not_transcribe(*_args, **_kwargs):
        raise AssertionError("transcribe_ref should not run when ref_text is provided")

    monkeypatch.setattr(vc, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(vc, "trim_audio_encode", _fake_trim)
    monkeypatch.setattr(vc, "transcribe_ref", _should_not_transcribe)

    res = vc.prepare_ref(
        ref_audio=ref_audio,
        cache=tmp_path / "cache",
        whisper_model="small",
        num_threads=1,
        ref_start=0.0,
        ref_end=4.0,
        ref_language="English",
        x_vector_only=False,
        force=False,
        force_bad_ref=False,
        interactive=False,
        ref_text="Hello from provided transcript",
    )

    assert res.transcript == "Hello from provided transcript"
    transcript_json = res.ref_dir / "ref_transcript.json"
    payload = json.loads(transcript_json.read_text(encoding="utf-8"))
    assert payload["transcript_source"] == "provided_ref_text"
