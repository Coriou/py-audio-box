"""Clone publication must not expose partial named voices or damage earlier takes."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import voices
from voices import VoiceRegistry


def assets(tmp_path):
    ref = tmp_path / 'segment.wav'
    ref.write_bytes(b'local reference')
    prompt = tmp_path / 'prompt.pkl'
    prompt.write_bytes(b'trusted local prompt')
    return dict(segment_wav=ref, ref_meta={'language': 'English'}, stem='take',
                pkl_src=prompt, meta={'reference_sha256': hashlib.sha256(ref.read_bytes()).hexdigest()},
                source={'type': 'file'}, tone='neutral')


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('failure', ['copy', 'metadata', 'promotion'])
def test_failed_registration_preserves_existing_voice(tmp_path, monkeypatch, existing, failure):
    reg = VoiceRegistry(tmp_path / 'cache')
    args = assets(tmp_path)
    if existing:
        reg.finalize_clone('speaker', **args)
        reg.update_generation_defaults('speaker', temperature=0.7)
    before = snapshot(reg.voice_dir('speaker')) if existing else {}
    if failure == 'copy':
        original = voices.shutil.copyfile
        def fail(src, dest, *a, **kw):
            if Path(src) == args['pkl_src']:
                raise OSError('injected prompt copy failure')
            return original(src, dest, *a, **kw)
        monkeypatch.setattr(voices.shutil, 'copyfile', fail)
    elif failure == 'metadata':
        args['meta'] = {}
    else:
        original = voices.os.replace
        def fail(src, dest):
            if Path(dest) == reg.voice_dir('speaker') and Path(src).parent.name == 'voices':
                raise OSError('injected promotion failure')
            return original(src, dest)
        monkeypatch.setattr(voices.os, 'replace', fail)
    with pytest.raises((OSError, ValueError)):
        reg.finalize_clone('speaker', **args)
    assert reg.exists('speaker') == existing
    assert snapshot(reg.voice_dir('speaker')) == before
    assert not list(reg.root.glob('.clone-*'))


def test_registration_keeps_other_tones_and_settings(tmp_path):
    reg = VoiceRegistry(tmp_path / 'cache')
    args = assets(tmp_path)
    reg.finalize_clone('speaker', **args)
    reg.update_generation_defaults('speaker', temperature=0.7)
    args.update(stem='second', tone='happy')
    reg.finalize_clone('speaker', **args)
    record = reg.load('speaker')
    assert record['tones'] == {'neutral': 'take', 'happy': 'second'}
    assert record['generation_defaults'] == {'temperature': 0.7}
    assert reg.prompt_reference(reg.prompt_for_tone('speaker', 'neutral')).read_bytes() == b'local reference'


def test_concurrent_registration_keeps_both_tone_updates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cache = tmp_path / 'cache'
    args = assets(tmp_path)
    def register(index):
        VoiceRegistry(cache).finalize_clone('speaker', **{**args, 'stem': f'take{index}', 'tone': f'tone{index}'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(register, range(2)))
    record = VoiceRegistry(cache).load('speaker')
    assert record['tones'] == {'tone0': 'take0', 'tone1': 'take1'}
    assert set(record['prompts']) == {'take0', 'take1'}


@pytest.mark.parametrize('failure', ['model', 'prompt', 'metadata'])
@pytest.mark.parametrize('existing', [False, True])
def test_clone_command_failure_never_changes_registry(tmp_path, monkeypatch, failure, existing):
    import importlib.util
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'apps/voice-clone/voice-clone.py'
    spec = importlib.util.spec_from_file_location('clone_registration_test', path)
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
    cache = tmp_path / 'cache'
    reg = VoiceRegistry(cache)
    data = assets(tmp_path)
    if existing:
        reg.finalize_clone('speaker', **data)
    before = snapshot(reg.voice_dir('speaker')) if existing else {}
    args = app.build_parser().parse_args(['synth', '--ref-audio', str(data['segment_wav']),
        '--voice-name', 'speaker', '--text', 'Hello', '--cache', str(cache),
        '--save-profile-default'])
    ref = SimpleNamespace(segment=data['segment_wav'], ref_language='English', ref_language_prob=1.0)
    monkeypatch.setattr(app, 'prepare_ref', lambda **kw: ref)
    monkeypatch.setattr(app, 'validate_language', lambda language, model: language)
    def fail(*a, **kw):
        raise RuntimeError('injected construction failure')
    monkeypatch.setattr(app, 'load_tts_model', fail if failure == 'model' else lambda *a: object())
    monkeypatch.setattr(app, 'build_voice_clone_prompt', fail if failure == 'prompt' else lambda *a: (object(), data['pkl_src']))
    with pytest.raises((RuntimeError, ValueError)):
        app.cmd_synth(args)
    assert reg.exists('speaker') == existing
    assert snapshot(reg.voice_dir('speaker')) == before
