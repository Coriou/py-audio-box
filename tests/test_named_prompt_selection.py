import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from voices import VoiceRegistry


@pytest.fixture
def app():
    spec = importlib.util.spec_from_file_location('named_selection_app', ROOT / 'apps/voice-synth/voice-synth.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_prompt(reg, tmp_path, stem, model, revision='revision', tone=None):
    reference = tmp_path / f'{stem}.wav'
    reference.write_bytes(stem.encode())
    pkl = tmp_path / f'{stem}.pkl'
    pkl.write_bytes(pickle.dumps({'stem': stem}))
    reg.finalize_clone('speaker', segment_wav=reference, ref_meta={}, stem=stem, pkl_src=pkl,
        source={'type': 'file'}, tone=tone, meta={'engine': 'qwen3', 'checkpoint': model,
        'revision': revision, 'schema_version': 1, 'mode': 'full', 'cache_identity_version': 1,
        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
        'pickle_sha256': hashlib.sha256(pkl.read_bytes()).hexdigest(), 'ref_language_detected': stem})
    return reg.prompts_dir('speaker') / f'{stem}.pkl'


def select(app, reg, cache, model, tone=None):
    resolved = app.resolve_voice('speaker', cache, tone=tone)
    worker = SimpleNamespace(_audio_box_revision='revision')
    return Path(app._select_named_prompt_variant(reg, resolved, worker, model, tone=tone))


def test_named_explicit_model_selects_variant_and_default_is_newest(tmp_path, app):
    reg = VoiceRegistry(tmp_path)
    a = add_prompt(reg, tmp_path, 'small', 'model-0.6')
    b = add_prompt(reg, tmp_path, 'large', 'model-1.7')
    os.utime(a, (1, 1)); os.utime(b, (2, 2))
    assert Path(app.resolve_voice('speaker', tmp_path)['prompt_path']) == b
    assert select(app, reg, tmp_path, 'model-0.6') == a
    assert select(app, reg, tmp_path, 'model-1.7') == b
    assert select(app, reg, tmp_path, 'model-0.6') == a
    assert reg.prompt_reference(a).read_bytes() == b'small'


@pytest.mark.parametrize('damage', ['revision', 'pickle', 'missing_ref', 'ref_hash', 'schema'])
def test_selection_rejects_damaged_target_without_using_other_model(tmp_path, app, damage):
    reg = VoiceRegistry(tmp_path)
    target = add_prompt(reg, tmp_path, 'small', 'model-0.6')
    add_prompt(reg, tmp_path, 'large', 'model-1.7')
    meta_path = target.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())
    if damage == 'revision': meta['revision'] = 'stale'
    elif damage == 'schema': meta['schema_version'] = -1
    elif damage == 'pickle': target.write_bytes(b'changed')
    elif damage == 'missing_ref': target.with_suffix('.ref.wav').unlink()
    else: target.with_suffix('.ref.wav').write_bytes(b'changed reference')
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='no compatible cached prompt'):
        select(app, reg, tmp_path, 'model-0.6')


def test_tone_history_selects_same_tone_only(tmp_path, app):
    reg = VoiceRegistry(tmp_path)
    a = add_prompt(reg, tmp_path, 'small-neutral', 'model-0.6', tone='neutral')
    add_prompt(reg, tmp_path, 'large-neutral', 'model-1.7', tone='neutral')
    add_prompt(reg, tmp_path, 'small-happy', 'model-0.6', tone='happy')
    assert select(app, reg, tmp_path, 'model-0.6', tone='neutral') == a
    with pytest.raises(ValueError, match='no compatible cached prompt'):
        select(app, reg, tmp_path, 'model-1.7', tone='happy')


def test_legacy_exact_id_and_direct_path_never_switch(tmp_path, app):
    reg = VoiceRegistry(tmp_path)
    add_prompt(reg, tmp_path, 'small', 'model-0.6')
    legacy = tmp_path / 'voice-clone/prompts'
    legacy.mkdir(parents=True)
    chosen = legacy / 'model1.7_exact.pkl'
    chosen.write_bytes(b'exact')
    worker = SimpleNamespace(_audio_box_revision='revision')
    for voice in ('model1.7_exact', str(chosen)):
        resolved = app.resolve_voice(voice, tmp_path)
        assert Path(app._select_named_prompt_variant(reg, resolved, worker, 'model-0.6')) == chosen


def test_symbolic_revision_resolves_before_selection(tmp_path, app):
    reg = VoiceRegistry(tmp_path)
    target = add_prompt(reg, tmp_path, 'small', 'model-0.6', revision='actual-sha')
    resolved = app.resolve_voice('speaker', tmp_path)
    class Worker:
        _audio_box_revision = 'main'
        def get_supported_languages(self):
            self._audio_box_revision = 'actual-sha'
            return ['English']
    assert Path(app._select_named_prompt_variant(reg, resolved, Worker(), 'model-0.6')) == target


def test_old_unlabelled_history_cannot_be_assumed_to_have_current_tone(tmp_path, app):
    reg = VoiceRegistry(tmp_path)
    old = add_prompt(reg, tmp_path, 'small-neutral', 'model-0.6', tone='neutral')
    meta_path = old.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text()); meta.pop('tone')
    meta_path.write_text(json.dumps(meta))
    add_prompt(reg, tmp_path, 'large-neutral', 'model-1.7', tone='neutral')
    with pytest.raises(ValueError, match='no compatible cached prompt'):
        select(app, reg, tmp_path, 'model-0.6', tone='neutral')


@pytest.mark.parametrize('explicit_model', [False, True])
def test_speak_uses_matching_payload_language_and_output_metadata(tmp_path, app, monkeypatch, explicit_model):
    import numpy as np
    reg = VoiceRegistry(tmp_path / 'cache')
    small_model = app.DEFAULT_TTS_MODEL
    large_model = small_model.replace('0.6B', '1.7B')
    small = add_prompt(reg, tmp_path, 'small', small_model)
    large = add_prompt(reg, tmp_path, 'large', large_model)
    for path, language in ((small, 'French'), (large, 'German')):
        sidecar = path.with_suffix('.meta.json')
        meta = json.loads(sidecar.read_text()); meta['ref_language_detected'] = language
        sidecar.write_text(json.dumps(meta))
    os.utime(small, (1, 1)); os.utime(large, (2, 2))
    class Worker:
        _audio_box_checkpoint = small_model
        _audio_box_revision = 'main'
        def get_supported_languages(self):
            self._audio_box_revision = 'revision'
            return ['French', 'German']
        def close(self):
            pass
    monkeypatch.setattr(app, 'load_tts_model', lambda *a: Worker())
    monkeypatch.setattr(app, 'validate_language', lambda language, model: language)
    calls = []
    def generate(text, language, model, prompt, *a):
        calls.append((language, prompt.payload, prompt.checkpoint))
        return np.zeros(2400, dtype=np.float32), 24000
    monkeypatch.setattr(app, 'synthesise', generate)
    command = ['speak', '--voice', 'speaker', '--text', 'Bonjour', '--language', 'Auto',
               '--cache', str(tmp_path / 'cache'), '--out', str(tmp_path / 'out')]
    if explicit_model:
        command += ['--model', small_model]
    args = app.build_parser().parse_args(command)
    app.cmd_speak(args)
    assert calls == [('French', {'stem': 'small'}, small_model)]
    result = json.loads(next((tmp_path / 'out').rglob('takes.meta.json')).read_text())
    assert result['prompt_path'] == str(small)
    assert result['ref_language'] == 'French'
    assert result['language'] == 'French'
    assert result['model'] == small_model
