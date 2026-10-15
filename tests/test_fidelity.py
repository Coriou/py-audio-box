import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from fidelity import plan_chunks, transcript_metrics, concatenate_wavs


def test_sequence_qa_rejects_audited_false_perfect_score():
    result = transcript_metrics('je pars demain', 'demain je pars pars')
    assert result['intelligibility'] < 1
    assert result['wer'] > 0
    assert 'unexpected_repetition' in result['alerts']
    assert result['raw_transcript'] == 'demain je pars pars'


def test_edit_counts_and_empty_output():
    result = transcript_metrics('un deux trois', 'un quatre trois cinq')
    assert result['word_edits']['substitutions'] == 1
    assert result['word_edits']['insertions'] == 1
    assert result['word_edits']['deletions'] == 0
    result = transcript_metrics('un deux trois', '')
    assert result['word_edits']['deletions'] == 3
    assert result['wer'] == 1
    assert result['alerts'] == ['empty_transcript']


def test_normalization_preserves_semantic_numbers_and_accents():
    assert transcript_metrics('Été, 1250!', 'été 1250')['wer'] == 0
    assert transcript_metrics('1250', '125')['wer'] == 1
    assert transcript_metrics('été', 'ete')['cer'] > 0


@pytest.mark.parametrize('text', ['mot ' * 100, 'a' * 400, ' Un texte.\n\nDeuxième paragraphe!  ', ''])
def test_chunk_hard_bound_and_exact_original_conservation(text):
    plan = plan_chunks(text, 31)
    assert ''.join(c['text'] for c in plan) == text
    assert all(0 < len(c['text']) <= 31 for c in plan)
    assert all(text[c['start']:c['end']] == c['text'] for c in plan)
    assert all(a['end'] == b['start'] for a, b in zip(plan, plan[1:]))


def test_paragraph_and_continuation_pauses():
    plan = plan_chunks('Bonjour.\n\n' + 'mot ' * 15, 20, 310, 620, 110)
    assert plan[0]['pause_after_ms'] == 620
    assert plan[1]['pause_after_ms'] == 110
    assert plan[-1]['pause_after_ms'] == 0


def test_wav_contract_duration_stereo_and_mixed_rate(tmp_path):
    np = pytest.importorskip('numpy')
    sf = pytest.importorskip('soundfile')
    paths = [tmp_path / 'a.wav', tmp_path / 'b.wav']
    for p in paths:
        sf.write(p, np.zeros((24000, 2)), 24000)
    dest = tmp_path / 'out.wav'
    concatenate_wavs(paths, dest, pauses_ms=[300])
    data, sr = sf.read(dest)
    assert sr == 24000 and data.shape == (55200, 2)
    sf.write(paths[1], np.zeros(48000), 48000)
    with pytest.raises(ValueError, match='sample rate'):
        concatenate_wavs(paths, dest)
    sf.write(paths[1], np.zeros(24000), 24000)
    with pytest.raises(ValueError, match='channel'):
        concatenate_wavs(paths, dest)


def test_asr_reuses_model_transcript_and_known_language(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    app = Path(__file__).resolve().parents[1] / 'apps/voice-synth/voice-synth.py'
    spec = importlib.util.spec_from_file_location('fidelity_voice_synth', app)
    vs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vs)
    calls = []
    model = SimpleNamespace(transcribe=lambda path, **kwargs: (
        calls.append(kwargs) or iter([SimpleNamespace(text=' Bonjour', words=[
            SimpleNamespace(word=' Bonjour', start=0.0, end=1.0)])]), None))
    monkeypatch.setattr(vs, '_asr_model', lambda name, threads: model)
    monkeypatch.setattr(vs, 'get_duration', lambda path: 1.0)
    wav = tmp_path / 'fake.wav'
    wav.write_bytes(b'not-read-by-fake-asr')
    qa = vs.qa_transcribe(wav, 'tiny', 2, 'Bonjour', language='french')
    alignment = vs.align_words(wav, 'tiny', 2, language='french')
    assert qa['wer'] == 0 and qa['raw_transcript'] == ' Bonjour'
    assert alignment['kind'] == 'transcription_timestamps'
    assert len(calls) == 1 and calls[0]['language'] == 'fr'
    wav.write_bytes(b'changed-input')
    vs.qa_transcribe(wav, 'tiny', 2, 'Bonjour', language='french')
    assert len(calls) == 2


def _voice_synth_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'apps/voice-synth/voice-synth.py'
    spec = importlib.util.spec_from_file_location('voice_synth_owner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', [None, ValueError, SystemExit])
def test_owned_models_close_on_success_and_error(monkeypatch, failure):
    from types import SimpleNamespace
    vs = _voice_synth_module()
    closed = []
    monkeypatch.setattr(vs, 'load_tts_model', lambda name: SimpleNamespace(close=lambda: closed.append(name)))
    @vs._owns_models
    def operation():
        vs._load_owned_model('design')
        vs._load_owned_model('clone')
        if failure:
            raise failure('stop')
        return 'ok'
    if failure:
        with pytest.raises(failure):
            operation()
    else:
        assert operation() == 'ok'
    assert closed == ['clone', 'design']


def test_owned_models_close_remaining_owners_if_one_close_fails(monkeypatch):
    from types import SimpleNamespace
    vs = _voice_synth_module()
    closed = []
    def close(name):
        closed.append(name)
        if name == 'clone':
            raise RuntimeError('failed reap')
    monkeypatch.setattr(vs, 'load_tts_model', lambda name: SimpleNamespace(close=lambda: close(name)))
    @vs._owns_models
    def operation():
        vs._load_owned_model('design')
        vs._load_owned_model('clone')
    with pytest.raises(RuntimeError, match='failed reap'):
        operation()
    assert closed == ['clone', 'design']


def test_voice_synth_validates_selected_tone_own_reference(tmp_path):
    import hashlib
    from types import SimpleNamespace
    vs = _voice_synth_module()
    reg = vs.VoiceRegistry(tmp_path / 'cache')
    reg.create('narrator', 'Narrator', {})
    model = SimpleNamespace(_audio_box_revision='pinned')
    selected = []
    for tone, content in [('neutral', b'neutral ref'), ('excited', b'excited ref')]:
        reg.ref_wav('narrator').write_bytes(content)
        prompt = tmp_path / f'{tone}.pkl'
        prompt.write_bytes(tone.encode())
        meta = dict(engine='qwen3', checkpoint=vs.DEFAULT_TTS_MODEL, revision='pinned',
            cache_identity_version=1, schema_version=vs.PROMPT_SCHEMA_VERSION, mode='full',
            reference_sha256=hashlib.sha256(content).hexdigest(),
            pickle_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest())
        selected.append(reg.register_prompt('narrator', tone, prompt, meta, tone=tone))
    for prompt in selected:
        vs._validate_prompt_cache(reg, prompt, model, vs.DEFAULT_TTS_MODEL)
    reg.prompt_reference(selected[0]).write_bytes(b'tampered own ref')
    with pytest.raises(ValueError, match='identity'):
        vs._validate_prompt_cache(reg, selected[0], model, vs.DEFAULT_TTS_MODEL)


@pytest.mark.parametrize('fail_generation', [False, True])
def test_voice_design_closes_before_clone_load_and_on_generation_error(tmp_path, monkeypatch, fail_generation):
    import numpy as np
    vs = _voice_synth_module()
    events = []
    class Design:
        def generate_voice_design(self, **kwargs):
            events.append('generate_design')
            if fail_generation:
                raise RuntimeError('design error')
            return [np.zeros(2400)], 24000
        def close(self):
            events.append('close_design')
    class Clone:
        name_or_path = vs.DEFAULT_TTS_MODEL
        _audio_box_checkpoint = vs.DEFAULT_TTS_MODEL
        _audio_box_revision = 'resolved-snapshot'
        def create_voice_clone_prompt(self, **kwargs):
            events.append('create_clone')
            return {'local': 'cache'}
        def close(self):
            events.append('close_clone')
    def load(name, *_args):
        if name == vs.DEFAULT_DESIGN_MODEL:
            events.append('load_design')
            return Design()
        assert events[-1] == 'close_design'
        events.append('load_clone')
        return Clone()
    monkeypatch.setattr(vs, 'load_tts_model', load)
    args = vs.build_parser().parse_args(['design-voice', '--instruct', 'Calm delivery',
        '--ref-text', 'Bonjour.', '--cache', str(tmp_path / 'cache'), '--out', str(tmp_path / 'out')])
    if fail_generation:
        with pytest.raises(RuntimeError, match='design error'):
            vs.cmd_design_voice(args)
        assert events == ['load_design', 'generate_design', 'close_design']
    else:
        vs.cmd_design_voice(args)
        assert events == ['load_design', 'generate_design', 'close_design', 'load_clone', 'create_clone', 'close_clone']


@pytest.mark.parametrize('identifier_kind', ['full_stem', 'hex_prefix', 'pkl_path'])
def test_fresh_legacy_prompt_build_resolve_gate_and_speak(tmp_path, monkeypatch, identifier_kind):
    import importlib.util
    import json
    import numpy as np
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('legacy_prompt_builder', root / 'apps/voice-clone/voice-clone.py')
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    vs = _voice_synth_module()
    cache = tmp_path / 'cache'
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'trusted local placeholder')
    ref = SimpleNamespace(ref_hash='abcdef1234567890', segment=reference, transcript='Bonjour.',
        ref_language='French', ref_language_prob=1.0, seg_start=0.0, seg_end=1.0)
    class Model:
        name_or_path = vs.DEFAULT_TTS_MODEL
        _audio_box_checkpoint = vs.DEFAULT_TTS_MODEL
        _audio_box_revision = 'pinned-snapshot'
        closed = False
        def create_voice_clone_prompt(self, **kwargs):
            return {'approved': 'local clone'}
        def get_supported_languages(self):
            return ['French']
        def close(self):
            self.closed = True
    model = Model()
    prompt, path = vc.build_voice_clone_prompt(model, ref, cache, False, False)
    identifier = path.stem if identifier_kind == 'full_stem' else ref.ref_hash[:8] if identifier_kind == 'hex_prefix' else str(path)
    resolved = vs.resolve_voice(identifier, cache)
    assert Path(resolved['prompt_path']) == path
    vs._validate_prompt_cache(vs.VoiceRegistry(cache), resolved['prompt_path'], model, vs.DEFAULT_TTS_MODEL)
    monkeypatch.setattr(vs, 'load_tts_model', lambda *_args: model)
    def synthesise(text, language, _model, loaded_prompt, seed, gen_kwargs):
        assert loaded_prompt == prompt
        assert text == 'Bonjour.' and language == 'french'
        return np.zeros(2400), 24000
    monkeypatch.setattr(vs, 'synthesise', synthesise)
    out = tmp_path / 'out'
    args = vs.build_parser().parse_args(['speak', '--voice', identifier, '--text', 'Bonjour.',
        '--language', 'French', '--cache', str(cache), '--out-exact', str(out), '--save-profile-default'])
    vs.cmd_speak(args)
    assert (out / 'take_01.wav').exists()
    assert json.loads((out / 'takes.meta.json').read_text())['voice_id'] == path.stem
    assert model.closed
