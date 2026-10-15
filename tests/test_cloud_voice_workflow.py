"""Cloud-facing command contracts; no model downloads or private references."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from tts import ClonePrompt
from tts_backends.qwen import QwenWorkerError
from voices import VoiceRegistry


def app(name):
    spec = importlib.util.spec_from_file_location('cloud_' + name.replace('-', '_'), ROOT / 'apps' / name / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('chunk', [False, True])
def test_speak_full_text_context_is_explicit_and_verbatim(tmp_path, monkeypatch, chunk):
    module = app('voice-synth')
    text = '  Bonjour à tous.\n\nVoici une deuxième phrase, avec sa ponctuation.  '
    args = module.build_parser().parse_args(['speak', '--speaker', 'Ryan', '--text', text,
        '--cache', str(tmp_path/'cache'), '--out', str(tmp_path/'out'),
        '--chunk' if chunk else '--no-chunk', '--chunk-max-chars', '25'])
    calls = []
    monkeypatch.setattr(module, 'load_tts_model', lambda *a: object())
    monkeypatch.setattr(module, 'validate_language', lambda language, model: language)
    monkeypatch.setattr(module, 'validate_speaker', lambda speaker, model: speaker)
    def generate(text, *args, **kwargs):
        calls.append(text)
        return np.zeros(2400, dtype=np.float32), 24000
    monkeypatch.setattr(module, 'synthesise_custom_voice', generate)
    module.cmd_speak(args)
    if chunk:
        assert len(calls) > 1 and ''.join(calls) == text
    else:
        assert calls == [text]
    metadata = json.loads(next((tmp_path/'out').rglob('takes.meta.json')).read_text())
    assert metadata['original_text'] == text
    assert metadata['chunked'] is chunk and metadata['n_chunks'] == len(calls)
    defaults = module.build_parser().parse_args(['speak', '--speaker', 'Ryan', '--text', 'Hello'])
    assert defaults.chunk is False
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(['speak', '--speaker', 'Ryan', '--text', 'Hello', '--chunk', '--no-chunk'])


@pytest.mark.parametrize('name', ['voice-synth', 'voice-clone'])
def test_main_reports_worker_death_without_raw_eof_traceback(tmp_path, monkeypatch, capsys, name):
    module = app(name)
    def fail(args):
        try:
            raise EOFError('raw transport detail')
        except EOFError as cause:
            raise QwenWorkerError(dict(stage='response', method='generate_custom_voice',
                checkpoint='Qwen3-TTS-12Hz-1.7B-CustomVoice', dtype='float32', exitcode=-9, signal='SIGKILL')) from cause
    monkeypatch.setattr(module, 'cmd_speak' if name == 'voice-synth' else 'cmd_synth', fail)
    command = ['speak', '--speaker', 'Ryan'] if name == 'voice-synth' else ['synth', '--ref-audio', str(tmp_path/'ref.wav')]
    # Speak's --json-result requests a result file on success and structured stderr on failure.
    extra = ['--json-result', str(tmp_path/'result.json')] if name == 'voice-synth' else []
    monkeypatch.setattr(sys, 'argv', [name, *command, '--text', 'Hello', '--cache', str(tmp_path/'cache'), *extra])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert 'EOFError' not in captured.err and 'raw transport detail' not in captured.err
    assert 'SIGKILL' in captured.err and 'float32' in captured.err
    if name == 'voice-synth':
        payload = json.loads(captured.err)
        assert payload['error']['code'] == 'qwen_worker_failed'
        assert payload['error']['details']['exitcode'] == -9


def test_design_registration_failure_does_not_publish_partial_voice(tmp_path, monkeypatch):
    module = app('voice-synth')
    owners = []
    class Model:
        _audio_box_revision = 'immutable-test-revision'
        def __init__(self, name):
            self.name_or_path = self._audio_box_checkpoint = name
            self.closed = False
            owners.append(self)
        def generate_voice_design(self, **kwargs):
            return [np.zeros(2400, dtype=np.float32)], 24000
        def create_voice_clone_prompt(self, **kwargs):
            return [{'synthetic': True}]
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, 'load_tts_model', lambda name, *a: Model(name))
    def fail_finalize(self, *args, **kwargs):
        assert not self.exists('designed-test')
        raise RuntimeError('injected publication failure')
    monkeypatch.setattr(VoiceRegistry, 'finalize_clone', fail_finalize)
    args = module.build_parser().parse_args(['design-voice', '--instruct', 'Calm narrator',
        '--ref-text', 'Hello world.', '--voice-name', 'designed-test',
        '--cache', str(tmp_path/'cache'), '--out', str(tmp_path/'out')])
    with pytest.raises(RuntimeError, match='publication failure'):
        module.cmd_design_voice(args)
    assert not VoiceRegistry(tmp_path/'cache').exists('designed-test')
    assert len(owners) == 2 and all(model.closed for model in owners)
    prompt = next((tmp_path/'cache'/'voice-clone'/'prompts').glob('*.pkl'))
    with prompt.open('rb') as handle:
        stored = pickle.load(handle)
    assert isinstance(stored, ClonePrompt)
    assert stored.checkpoint == args.clone_model
