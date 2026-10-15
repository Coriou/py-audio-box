import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from tts_backends import get_backend
from tts_backends.assets import sha256, verified_bundle
from tts_backends.contract import BackendError


def bundle(tmp_path, backend, files, **extra):
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = {'schema_version': 1, 'backend': backend, 'checkpoint': 'test-pinned-sha', 'language': 'fr', 'files': [{'path': name, 'sha256': sha256(tmp_path / name)} for name in files], **extra}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_lazy_discovery_does_not_import_torch():
    lib = str(Path(__file__).resolve().parents[1] / 'lib')
    proc = subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {lib!r}); from tts_backends import get_backend; get_backend("kokoro"); get_backend("pocket-tts"); assert "torch" not in sys.modules; assert "qwen_tts" not in sys.modules'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert get_backend('qwen') is get_backend('qwen3')
    with pytest.raises(BackendError):
        get_backend('fish')


def test_bundle_tamper_and_escape_rejected(tmp_path):
    manifest = bundle(tmp_path, 'kokoro', {'model.onnx': b'fake'})
    assert verified_bundle(tmp_path, 'kokoro')[1]['checkpoint'] == 'test-pinned-sha'
    (tmp_path / 'model.onnx').write_bytes(b'tamper')
    with pytest.raises(BackendError):
        verified_bundle(tmp_path, 'kokoro')
    manifest['files'][0]['path'] = '../outside'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(BackendError):
        verified_bundle(tmp_path, 'kokoro')


def test_kokoro_mock_contract(tmp_path, monkeypatch):
    from tts_backends.kokoro import KokoroBackend
    bundle(tmp_path, 'kokoro', {'model.onnx': b'model', 'voices.bin': b'voices'})
    class Session:
        provider_reads = 0
        def __init__(self, path, sess_options, providers):
            assert sess_options.intra_op_num_threads == 2
            assert sess_options.inter_op_num_threads == 1
            assert sess_options.enable_cpu_mem_arena is False
            assert Path(path).is_file()
            assert providers == ['CPUExecutionProvider']
            self._model_path = path
        def get_inputs(self):
            return ['tokens']
        def get_outputs(self):
            return ['audio']
        def get_modelmeta(self):
            return 'model metadata'
        def run(self, output_names, input_feed, run_options=None):
            assert output_names is None
            assert input_feed == {'text': Kokoro.generated_texts[-1]}
            assert run_options is None
            return [np.arange(24, dtype=np.float32)]
        def get_providers(self):
            Session.provider_reads += 1
            return ['CPUExecutionProvider']
    class Kokoro:
        catalog_reads = 0
        generated_texts = []
        @classmethod
        def from_session(cls, session, voices):
            assert isinstance(session, Session)
            assert session._model_path == str(tmp_path / 'model.onnx')
            assert session.get_inputs() == ['tokens']
            assert session.get_outputs() == ['audio']
            assert session.get_modelmeta() == 'model metadata'
            obj = cls()
            obj.sess = session
            return obj
        def get_voices(self):
            Kokoro.catalog_reads += 1
            return ['ff_siwis', 'af_heart']
        def create(self, text, voice, lang):
            Kokoro.generated_texts.append(text)
            assert voice == 'ff_siwis' and lang == 'fr-fr'
            return self.sess.run(None, {'text': text})[0], 24000
    monkeypatch.setitem(sys.modules, 'onnxruntime', types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Session))
    monkeypatch.setitem(sys.modules, 'kokoro_onnx', types.SimpleNamespace(Kokoro=Kokoro))
    model = KokoroBackend(tmp_path)
    result = model.synthesise('Bonjour.', preset='ff_siwis')
    assert result.sample_rate == 24000
    assert result.metadata['text_submitted'] == 'Bonjour.'
    assert result.metadata['checkpoint'] == 'test-pinned-sha'
    assert result.metadata['model_load_seconds'] == result.metadata['load_seconds']
    assert result.metadata['model_reused'] is False
    assert result.metadata['request_index'] == 1
    assert result.metadata['cpu_mem_arena'] is False
    for kwargs in ({'reference': 'x.wav'}, {'instruct': 'warm'}, {'seed': 1}):
        with pytest.raises(BackendError):
            model.synthesise('Bonjour.', preset='ff_siwis', **kwargs)
    with pytest.raises(BackendError):
        model.synthesise('Bonjour.', preset='af_heart')
    repeated = model.synthesise('Bonjour.', preset='ff_siwis')
    other = model.synthesise('Une nouvelle phrase.', preset='ff_siwis')
    assert repeated.metadata['load_seconds'] == 0.0
    assert repeated.metadata['model_load_seconds'] == result.metadata['model_load_seconds']
    assert repeated.metadata['model_reused'] is True
    assert repeated.metadata['request_index'] == 2
    assert other.metadata['request_index'] == 3
    np.testing.assert_array_equal(result.samples, np.arange(24, dtype=np.float32))
    assert Kokoro.generated_texts == ['Bonjour.', 'Bonjour.', 'Une nouvelle phrase.']
    assert Kokoro.catalog_reads == 1
    assert Session.provider_reads == 1
    model.close()


def test_pocket_remote_config_rejected_before_model_load(tmp_path):
    from tts_backends.pocket_tts import PocketBackend
    bundle(tmp_path, 'pocket-tts', {'config.yaml': b'flow_lm:\n  weights_path: hf://untrusted/model.safetensors\n'}, language='fr')
    with pytest.raises(BackendError, match='local bundle'):
        PocketBackend(tmp_path)


def test_pocket_mock_reference_contract(tmp_path, monkeypatch):
    from tts_backends.pocket_tts import PocketBackend
    bundle(tmp_path, 'pocket-tts', {'config.yaml': b'flow_lm: {}\n'}, language='fr')
    class Tensor:
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.ones(24)
    class Model:
        sample_rate = 24000
        conditioning_calls = 0
        generation_calls = 0
        @classmethod
        def load_model(cls, config):
            assert Path(config).is_file()
            return cls()
        def get_state_for_audio_prompt(self, path):
            Model.conditioning_calls += 1
            return {'reference': path}
        def generate_audio(self, state, text, copy_state):
            Model.generation_calls += 1
            assert copy_state
            return Tensor()
    monkeypatch.setitem(sys.modules, 'pocket_tts', types.SimpleNamespace(TTSModel=Model))
    reference = tmp_path / 'approved.wav'
    reference.write_bytes(b'fake explicit test reference')
    model = PocketBackend(tmp_path)
    result = model.synthesise('Bonjour.', reference=reference, seed=1)
    assert result.metadata['reference_sha256'] == sha256(reference)
    assert result.metadata['text_submitted'] == 'Bonjour.'
    assert result.metadata['model_reused'] is False
    assert result.metadata['reference_state_reused'] is False
    repeated = model.synthesise('Une nouvelle phrase.', reference=reference, seed=1)
    assert repeated.metadata['model_reused'] is True
    assert repeated.metadata['reference_state_reused'] is True
    assert repeated.metadata['load_seconds'] == 0.0
    assert repeated.metadata['model_load_seconds'] == result.metadata['model_load_seconds']
    assert repeated.metadata['request_index'] == 2
    assert Model.conditioning_calls == 1
    assert Model.generation_calls == 2
    with pytest.raises(BackendError):
        model.synthesise('Bonjour.', reference=reference, instruct='warm')
    with pytest.raises(BackendError):
        model.synthesise('Bonjour.', preset='ff_siwis')
    model.close()


def test_pocket_relative_bundle_relocation(tmp_path, monkeypatch):
    import shutil
    import yaml
    from tts_backends.pocket_tts import PocketBackend
    original = tmp_path / 'original'
    original.mkdir()
    bundle(original, 'pocket-tts', {'config.yaml': b'weights_path: model.safetensors\n', 'model.safetensors': b'approved weights'})
    relocated = tmp_path / 'relocated'
    shutil.copytree(original, relocated)
    class Model:
        @classmethod
        def load_model(cls, config):
            value = yaml.safe_load(Path(config).read_text())
            weights = Path(value['weights_path'])
            assert weights == relocated / 'model.safetensors'
            assert weights.read_bytes() == b'approved weights'
            return cls()
    monkeypatch.setitem(sys.modules, 'pocket_tts', types.SimpleNamespace(TTSModel=Model))
    model = PocketBackend(relocated)
    model.close()


def test_cpu_cli_help_does_not_import_inference():
    cli = str(Path(__file__).resolve().parents[1] / 'apps/voice-cpu/voice-cpu.py')
    source = f'import sys,runpy; sys.argv=[{cli!r}, "--help"];\ntry: runpy.run_path({cli!r},run_name="__main__")\nexcept SystemExit as e: assert e.code == 0\nassert "torch" not in sys.modules\nassert "qwen_tts" not in sys.modules\nassert "onnxruntime" not in sys.modules'
    proc = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_cpu_cli_timeout_is_bounded(tmp_path):
    cli = Path(__file__).resolve().parents[1] / 'apps/voice-cpu/voice-cpu.py'
    proc = subprocess.run([sys.executable, str(cli), 'speak', '--backend', 'kokoro', '--assets', str(tmp_path), '--language', 'fr', '--preset', 'ff_siwis', '--text', 'Bonjour.', '--output', str(tmp_path / 'result.wav'), '--timeout', '.0001', '--json'], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 1
    assert json.loads(proc.stderr.splitlines()[-1])['error']['code'] == 'timeout'
    assert not (tmp_path / 'result.wav').exists()


@pytest.mark.parametrize('backend_name', ['kokoro', 'pocket-tts'])
@pytest.mark.parametrize('threads', [0, -1, True, 1.5, '2'])
def test_backends_reject_invalid_threads_before_assets(backend_name, threads, tmp_path):
    with pytest.raises(BackendError) as error:
        get_backend(backend_name)(tmp_path / 'missing', threads=threads)
    assert error.value.code == 'invalid_threads'


def test_kokoro_rejects_non_cpu_effective_provider(tmp_path, monkeypatch):
    bundle(tmp_path, 'kokoro', {'model.onnx': b'model', 'voices.bin': b'voices'})
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def get_providers(self):
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    class Kokoro:
        @classmethod
        def from_session(cls, session, voices):
            return cls()
    monkeypatch.setitem(sys.modules, 'onnxruntime', types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Session))
    monkeypatch.setitem(sys.modules, 'kokoro_onnx', types.SimpleNamespace(Kokoro=Kokoro))
    with pytest.raises(BackendError) as error:
        get_backend('kokoro')(tmp_path)
    assert error.value.code == 'unsupported_provider'


def test_kokoro_failed_inference_is_accounted_in_model_reuse(tmp_path, monkeypatch):
    bundle(tmp_path, 'kokoro', {'model.onnx': b'model', 'voices.bin': b'voices'})
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def get_providers(self):
            return ['CPUExecutionProvider']
    class Kokoro:
        calls = 0
        @classmethod
        def from_session(cls, session, voices):
            return cls()
        def get_voices(self):
            return ['ff_siwis']
        def create(self, *args, **kwargs):
            self.calls += 1
            return np.array([float('nan')]) if self.calls == 1 else np.ones(24), 24000
    monkeypatch.setitem(sys.modules, 'onnxruntime', types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Session))
    monkeypatch.setitem(sys.modules, 'kokoro_onnx', types.SimpleNamespace(Kokoro=Kokoro))
    model = get_backend('kokoro')(tmp_path)
    model.load_seconds = 2.5
    with pytest.raises(BackendError):
        model.synthesise('Bonjour.', preset='ff_siwis')
    result = model.synthesise('Bonjour.', preset='ff_siwis')
    assert result.metadata['request_index'] == 2
    assert result.metadata['load_seconds'] == 0.0
    assert result.metadata['model_load_seconds'] == 2.5
    assert result.metadata['model_reused'] is True
    assert model.model.calls == 2
    model.close()


@pytest.mark.parametrize('encoder_fails', [False, True])
def test_pocket_conditions_exact_private_snapshot_and_cleans_it(tmp_path, monkeypatch, encoder_fails):
    import hashlib
    import stat
    import torch
    bundle(tmp_path, 'pocket-tts', {'config.yaml': b'flow_lm: {}\n'}, language='fr')
    reference = tmp_path / 'approved.wav'
    approved_bytes = b'approved synthetic test reference A'
    reference.write_bytes(approved_bytes)
    snapshot_paths = []
    class Model:
        sample_rate = 24000
        @classmethod
        def load_model(cls, config):
            return cls()
        def get_state_for_audio_prompt(self, path):
            snapshot = Path(path)
            snapshot_paths.append(snapshot)
            assert snapshot != reference
            assert snapshot.suffix == '.wav'
            assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
            # Mutate the source at the exact hash/encoder race boundary.
            reference.write_bytes(b'changed synthetic test reference B')
            assert snapshot.read_bytes() == approved_bytes
            if encoder_fails:
                raise RuntimeError('synthetic encoder failure')
            return {'approved_bytes': snapshot.read_bytes()}
        def generate_audio(self, state, text, copy_state):
            assert state['approved_bytes'] == approved_bytes
            return torch.ones(24)
    monkeypatch.setitem(sys.modules, 'pocket_tts', types.SimpleNamespace(TTSModel=Model))
    model = get_backend('pocket-tts')(tmp_path)
    try:
        if encoder_fails:
            with pytest.raises(RuntimeError, match='synthetic encoder failure'):
                model.synthesise('Bonjour.', reference=reference)
            assert not model._states
        else:
            result = model.synthesise('Bonjour.', reference=reference)
            assert result.metadata['reference_sha256'] == hashlib.sha256(approved_bytes).hexdigest()
            assert result.metadata['reference_sha256'] != sha256(reference)
            assert result.metadata['reference_snapshot_seconds'] >= 0
        assert len(snapshot_paths) == 1
        assert not snapshot_paths[0].exists()
    finally:
        model.close()


def test_pocket_uses_owned_snapshot_namespace_without_deleting_directory(tmp_path, monkeypatch):
    import torch
    bundle(tmp_path, 'pocket-tts', {'config.yaml': b'flow_lm: {}\n'}, language='fr')
    namespace = tmp_path / 'worker-private'
    namespace.mkdir(mode=0o700)
    reference = tmp_path / 'approved.wav'
    reference.write_bytes(b'synthetic approved test reference')
    snapshots = []
    class Model:
        sample_rate = 24000
        @classmethod
        def load_model(cls, config):
            assert Path(config).parent == namespace
            return cls()
        def get_state_for_audio_prompt(self, path):
            snapshot = Path(path)
            assert snapshot.parent == namespace
            snapshots.append(snapshot)
            return {'reference_bytes': snapshot.read_bytes()}
        def generate_audio(self, *args, **kwargs):
            return torch.ones(24)
    monkeypatch.setitem(sys.modules, 'pocket_tts', types.SimpleNamespace(TTSModel=Model))
    model = get_backend('pocket-tts')(tmp_path, reference_temp_dir=namespace)
    model.synthesise('Bonjour.', reference=reference)
    assert len(snapshots) == 1
    assert not snapshots[0].exists()
    model.close()
    assert namespace.is_dir()
    assert list(namespace.iterdir()) == []


def test_pocket_rejects_non_private_snapshot_namespace_before_loading(tmp_path):
    namespace = tmp_path / 'public'
    namespace.mkdir(mode=0o777)
    namespace.chmod(0o777)
    with pytest.raises(BackendError) as error:
        get_backend('pocket-tts')(tmp_path / 'missing-assets', reference_temp_dir=namespace)
    assert error.value.code == 'invalid_reference_temp_dir'
