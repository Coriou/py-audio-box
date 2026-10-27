import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from tts import _run_generation, load_tts_model
from tts_backends.contract import AudioResult, BackendError
from tts_backends.qwen import checkpoint_capabilities, validate_checkpoint_request


def sleeping_generation():
    time.sleep(30)
    return [np.ones(10)], 24000


def healthy_generation():
    return [np.ones(24)], 24000


def test_qwen_worker_releases_sent_audio_before_next_request(monkeypatch):
    import weakref
    import tts
    import tts_backends.qwen as qwen
    references, outputs = [], []
    expected = np.arange(24, dtype=np.float32)

    class Model:
        def generate(self):
            assert all(reference() is None for reference in references)
            samples = expected.copy()
            references.append(weakref.ref(samples))
            return [samples], 24000

    class Connection:
        calls = 0
        def recv(self):
            assert all(reference() is None for reference in references)
            self.calls += 1
            if self.calls > 2:
                raise EOFError
            return 'generate', (), {}
        def send(self, message):
            if message[0] == 'ok':
                # A real Pipe serializes; retain only a copy in this fake.
                wavs, rate = message[1]
                outputs.append((wavs[0].copy(), rate))
        def close(self):
            pass

    monkeypatch.setattr(tts, '_load_tts_model_local', lambda **_settings: Model())
    qwen._worker_impl(Connection(), {})
    assert len(outputs) == 2
    for samples, rate in outputs:
        assert rate == 24000
        np.testing.assert_array_equal(samples, expected)


def failing_generation():
    raise MemoryError('synthetic OOM')


def test_checkpoint_matrix():
    assert not checkpoint_capabilities('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice').text_direction
    assert checkpoint_capabilities('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice').text_direction
    with pytest.raises(BackendError, match='does not support text direction'):
        validate_checkpoint_request('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', 'custom_voice', 'warm')
    with pytest.raises(BackendError):
        validate_checkpoint_request('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'clone', 'warm')
    with pytest.raises(BackendError):
        validate_checkpoint_request('Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'custom_voice')


def test_timeout_reaped_then_next_job_healthy():
    before = {p.pid for p in mp.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _run_generation(sleeping_generation, op_name='fake', gen_kwargs={}, timeout_s=.15)
    assert time.monotonic() - started < 5
    assert {p.pid for p in mp.active_children()} == before
    audio, rate = _run_generation(healthy_generation, op_name='fake', gen_kwargs={}, timeout_s=30)
    assert rate == 24000 and len(audio) == 24


def test_failure_reaped_then_next_job_healthy():
    with pytest.raises(RuntimeError, match='MemoryError'):
        _run_generation(failing_generation, op_name='fake', gen_kwargs={}, timeout_s=30)
    assert not mp.active_children()
    assert _run_generation(healthy_generation, op_name='fake', gen_kwargs={}, timeout_s=30)[1] == 24000


def test_loading_is_lazy_and_records_explicit_identity():
    model = load_tts_model('Qwen/Qwen3-TTS-12Hz-0.6B-Base', 1, revision='abc123', offline=True)
    assert model._process is None
    assert model._audio_box_revision == 'abc123'
    assert model.settings['offline'] is True
    model.close()


def test_audio_contract_invalid_layout():
    with pytest.raises(BackendError):
        AudioResult(np.ones((2, 10)), 24000, 2)
    with pytest.raises(BackendError):
        AudioResult(np.array([float('nan')]), 24000)


def fake_backend_worker(connection, settings, runtime_dir):
    connection.send(('ready', None))
    try:
        while True:
            method, args, kwargs = connection.recv()
            if method == 'sleep':
                time.sleep(30)
            elif method == 'fail':
                connection.send(('error', ('MemoryError', 'synthetic OOM')))
            else:
                connection.send(('ok', ([np.ones(24)], 24000)))
    except EOFError:
        pass
    finally:
        connection.close()


def test_resident_backend_timeout_restart_and_failure(monkeypatch):
    import tts_backends.qwen as qwen
    monkeypatch.setattr(qwen, '_worker', fake_backend_worker)
    model = load_tts_model('Qwen/Qwen3-TTS-12Hz-0.6B-Base', 1)
    try:
        with pytest.raises(TimeoutError):
            model.invoke('sleep', timeout_s=3)
        assert model._process is None
        assert model.invoke('healthy', timeout_s=30)[1] == 24000
        healthy_pid = model._process.pid
        with pytest.raises(RuntimeError, match='MemoryError'):
            model.invoke('fail', timeout_s=30)
        assert model._process is None
        assert model.invoke('healthy', timeout_s=30)[1] == 24000
        assert model._process.pid != healthy_pid
    finally:
        model.close()
    assert not mp.active_children()


def test_local_checkpoint_capabilities(tmp_path):
    import json
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_tts', 'tts_model_size': '0b6', 'tts_model_type': 'custom_voice'}))
    assert checkpoint_capabilities(str(tmp_path)).modes == ('custom_voice',)
    assert checkpoint_capabilities(str(tmp_path)).text_direction is False


def test_audio_contract_rejects_non_integer_rates_and_complex():
    with pytest.raises(BackendError):
        AudioResult(np.ones(10), 24000.5)
    with pytest.raises(BackendError):
        AudioResult(np.ones(10, dtype=complex), 24000)


def test_backend_atexit_does_not_retain_owner():
    import gc
    import weakref
    model = load_tts_model('Qwen/Qwen3-TTS-12Hz-0.6B-Base', 1)
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None


def test_worker_redirects_load_and_method_stdout(monkeypatch, capsys, tmp_path):
    import tts
    from tts_backends.qwen import _worker
    class Model:
        def get_supported_languages(self):
            print('method diagnostics')
            return ['english']
    def load(**kwargs):
        print('model loading diagnostics')
        return Model()
    class Connection:
        received = False
        def send(self, value):
            pass
        def recv(self):
            if self.received:
                raise EOFError()
            self.received = True
            return 'get_supported_languages', (), {}
        def close(self):
            pass
    monkeypatch.setattr(tts, '_load_tts_model_local', load)
    _worker(Connection(), {}, str(tmp_path))
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'model loading diagnostics' in captured.err
    assert 'method diagnostics' in captured.err


def test_deleting_loaded_owner_reaps_before_next_model(monkeypatch):
    import gc
    import tts_backends.qwen as qwen
    monkeypatch.setattr(qwen, '_worker', fake_backend_worker)
    first = load_tts_model('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', 1)
    first.invoke('healthy', timeout_s=30)
    previous_pid = first._process.pid
    del first
    gc.collect()
    assert previous_pid not in {p.pid for p in mp.active_children()}
    second = load_tts_model('Qwen/Qwen3-TTS-12Hz-0.6B-Base', 1)
    try:
        assert second.invoke('healthy', timeout_s=30)[1] == 24000
    finally:
        second.close()


def test_portable_prompt_preserves_transcript_without_qwen_import(tmp_path):
    import subprocess
    from tts_backends.qwen import ClonePromptItem
    import pickle
    prompt = [ClonePromptItem(None, np.ones(5), False, True, 'approved reference transcript')]
    path = tmp_path / 'local-test-prompt.pkl'
    path.write_bytes(pickle.dumps(prompt))
    lib = str(Path(__file__).resolve().parents[1] / 'lib')
    code = f'import sys,pickle; sys.path.insert(0,{lib!r}); prompt=pickle.load(open({str(path)!r},"rb")); assert prompt[0].ref_text=="approved reference transcript"; assert prompt[0].icl_mode; assert "qwen_tts" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_auto_device_environment_resolves_real_device(monkeypatch):
    import tts
    monkeypatch.setenv('TORCH_DEVICE', 'auto')
    monkeypatch.setattr(tts.torch.cuda, 'is_available', lambda: False)
    assert tts.get_device() == 'cpu'
    monkeypatch.setattr(tts.torch.cuda, 'is_available', lambda: True)
    assert tts.get_device() == 'cuda'


def test_probe_redirects_upstream_import_diagnostics(monkeypatch, capsys):
    import builtins
    import types
    import tts
    original_import = builtins.__import__
    def noisy_import(name, *args, **kwargs):
        if name == 'qwen_tts':
            print('upstream flash-attn import warning')
            return types.SimpleNamespace(Qwen3TTSModel=type('Model', (), {'generate_custom_voice': lambda self: None}))
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', noisy_import)
    result = tts.probe_qwen_tts_api({'custom_voice': ('generate_custom_voice',)})
    assert result['compatible'] is True
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'upstream flash-attn import warning' in captured.err


def test_actual_capabilities_stdout_is_parseable_json():
    import subprocess
    import json
    cli = Path(__file__).resolve().parents[1] / 'apps/voice-synth/voice-synth.py'
    code = f'import runpy; app=runpy.run_path({str(cli)!r}); args=app["build_parser"]().parse_args(["capabilities","--json","--skip-speaker-probe"]); app["cmd_capabilities"](args)'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert 'schema_version' in payload
    assert 'custom_voice' in payload
