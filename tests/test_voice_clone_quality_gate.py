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


@pytest.fixture
def prepared_reference(tmp_path, monkeypatch):
    vc = _load_voice_clone_module()
    raw = tmp_path / 'reference.raw'; raw.write_bytes(b'synthetic reference')
    monkeypatch.setattr(vc, 'normalize_audio', lambda src, dest: dest.write_bytes(b'normalized'))
    monkeypatch.setattr(vc, 'trim_audio_encode', lambda src, start, duration, dest: dest.write_bytes(b'segment'))
    kwargs = dict(ref_audio=raw, cache=tmp_path / 'cache', whisper_model='small', num_threads=1,
                  ref_start=0.0, ref_end=4.0, ref_language='Auto', x_vector_only=False, force=False)
    directory = kwargs['cache'] / 'voice-clone/refs' / vc.sha256_file(raw)
    return vc, kwargs, directory


@pytest.mark.parametrize('transcript,lp,prob,passed', [
    ('nonsense transcript', -0.78, 0.63, False),
    ('Éléonore Nguyễn', -1.2, 0.8, True), ('你好世界', -0.5, 0.99, True),
    ('Renée', -1.2001, 0.99, False), ('Hello', -0.5, 0.7999, False),
    ('Hello', float('nan'), 0.99, False), ('Hello', float('inf'), 0.99, False),
    ('Hello', -0.5, float('nan'), False), ('Hello', -0.5, 1.01, False),
    ('Hello', None, 0.99, False), ('Hello', -0.5, None, False),
    ('   ', -0.5, 0.99, False), ('...!?123', -0.5, 0.99, False),
])
def test_asr_quality_checks_are_unicode_and_finite(transcript, lp, prob, passed):
    vc = _load_voice_clone_module()
    report = vc.reference_quality_report(transcript, lp, prob)
    assert report['accepted'] is passed
    assert report['override'] is False


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('force', [False, True])
def test_reported_low_language_confidence_rejected_fresh_and_cached(prepared_reference, monkeypatch, cached, force):
    vc, kwargs, directory = prepared_reference
    calls = []
    def transcribe(*a):
        calls.append(1)
        return 'nonsense transcript', -0.78, 'English', 0.63
    monkeypatch.setattr(vc, 'transcribe_ref', transcribe)
    if cached:
        directory.mkdir(parents=True)
        (directory / 'ref_transcript.json').write_text(json.dumps(dict(transcript='nonsense transcript',
            avg_logprob=-0.78, ref_language_probability=0.63, whisper_model='small')))
    with pytest.raises(SystemExit):
        vc.prepare_ref(**{**kwargs, 'force': force})
    assert len(calls) == (0 if cached and not force else 1)
    for filename in ('ref_transcript.json', 'pipeline_state.json'):
        report = json.loads((directory / filename).read_text())['quality_report']
        assert report['status'] == 'failed'
        assert report['reasons'] == ['language_probability']
        assert report['accepted'] is False
        assert report['policy_version'] == vc.REFERENCE_QUALITY_POLICY_VERSION


def test_override_warns_and_cached_run_rechecks(prepared_reference, monkeypatch, capsys):
    vc, kwargs, directory = prepared_reference
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: ('nonsense', -0.78, 'English', 0.63))
    vc.prepare_ref(**kwargs, force_bad_ref=True)
    assert '--force-bad-ref' in capsys.readouterr().err
    report = json.loads((directory / 'ref_transcript.json').read_text())['quality_report']
    assert report['override'] is True and report['accepted'] is True
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: pytest.fail('cached ASR should be rechecked'))
    with pytest.raises(SystemExit):
        vc.prepare_ref(**kwargs)
    assert json.loads((directory / 'ref_transcript.json').read_text())['quality_report']['override'] is False


@pytest.mark.parametrize('mode', ['provided_ref_text', 'x_vector_only'])
def test_explicit_bypass_is_unvalidated_and_not_later_asr_evidence(prepared_reference, monkeypatch, mode):
    vc, kwargs, directory = prepared_reference
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: pytest.fail('explicit bypass must not transcribe'))
    bypass = {'ref_text': 'Éléonore'} if mode == 'provided_ref_text' else {'x_vector_only': True}
    vc.prepare_ref(**{**kwargs, **bypass})
    for filename in ('ref_transcript.json', 'pipeline_state.json'):
        report = json.loads((directory / filename).read_text())['quality_report']
        assert report['source'] == mode and report['status'] == 'unvalidated'
        assert report['checks'] == {} and not report['override']
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: ('fresh ASR', -0.4, 'French', 0.98))
    result = vc.prepare_ref(**kwargs)
    assert result.transcript == 'fresh ASR'
    assert json.loads((directory / 'ref_transcript.json').read_text())['quality_report']['source'] == 'asr'


@pytest.mark.parametrize('metric,value', [('avg_logprob', None), ('avg_logprob', float('inf')),
    ('ref_language_probability', None), ('ref_language_probability', float('nan'))])
def test_cached_invalid_metrics_fail_and_persist_strict_json(prepared_reference, monkeypatch, metric, value):
    vc, kwargs, directory = prepared_reference
    directory.mkdir(parents=True)
    record = dict(transcript='Hello', avg_logprob=-0.4, ref_language_probability=0.99, whisper_model='small')
    record[metric] = value
    (directory / 'ref_transcript.json').write_text(json.dumps(record))
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: pytest.fail('recheck existing ASR evidence'))
    with pytest.raises(SystemExit):
        vc.prepare_ref(**kwargs)
    for filename in ('ref_transcript.json', 'pipeline_state.json'):
        text = (directory / filename).read_text()
        json.loads(text, parse_constant=lambda value: pytest.fail(f'nonstandard JSON {value}'))
        assert json.loads(text)['quality_report']['status'] == 'failed'


def test_quality_failure_precedes_model_load_and_named_registration(prepared_reference, monkeypatch):
    vc, kwargs, directory = prepared_reference
    monkeypatch.setattr(vc, 'transcribe_ref', lambda *a: ('nonsense', -0.78, 'English', 0.63))
    monkeypatch.setattr(vc, 'load_tts_model', lambda *a: pytest.fail('rejection must precede model loading'))
    args = vc.build_parser().parse_args(['synth', '--ref-audio', str(kwargs['ref_audio']),
        '--ref-start', '0', '--ref-end', '4', '--voice-name', 'rejected-test', '--text', 'Hello',
        '--cache', str(kwargs['cache'])])
    with pytest.raises(SystemExit):
        vc.cmd_synth(args)
    assert not vc.VoiceRegistry(kwargs['cache']).exists('rejected-test')
