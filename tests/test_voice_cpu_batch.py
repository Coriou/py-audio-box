import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_app():
    path = ROOT / 'apps/voice-cpu/voice-cpu.py'
    spec = importlib.util.spec_from_file_location('voice_cpu_batch_test_app', path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def fake_backend_worker(connection, args, source):
    """Spawn-safe fake engine exercises the real staging/IPC worker implementation."""
    import numpy as np
    app = load_app()
    class FakeBackend:
        manifest = {'checkpoint': 'synthetic-test', 'files': []}
        def __init__(self, **kwargs):
            with (args.assets / 'loads.txt').open('a') as handle:
                handle.write('load\n')
        def synthesise(self, text, **kwargs):
            if text == 'slow':
                time.sleep(2)
                (args.assets / 'leaked.txt').write_text('generation survived timeout')
            if text == 'error':
                raise RuntimeError('mock generation error')
            return SimpleNamespace(samples=np.zeros(2400), sample_rate=24000, channels=1,
                                   metadata={'load_seconds': 999, 'text_submitted': text})
        def close(self):
            with (args.assets / 'closes.txt').open('a') as handle:
                handle.write('close\n')
    app.get_backend = lambda _name: FakeBackend
    app._batch_worker(connection, args, source)


def args_for(app, tmp_path, jobs, extra=()):
    jobs_path = tmp_path / 'jobs.json'
    jobs_path.write_text(json.dumps(jobs))
    return app.build_parser().parse_args(['batch', '--backend', 'kokoro', '--assets', str(tmp_path),
        '--language', 'fr', '--preset', 'ff_siwis', '--threads', '2', '--timeout', '8',
        '--jobs', str(jobs_path), '--report', str(tmp_path / 'report.json'), '--json', *extra])


def test_batch_releases_previous_audio_before_waiting_and_next_inference(tmp_path, monkeypatch):
    import weakref
    import numpy as np
    import soundfile as sf
    app = load_app()
    prior_audio = []
    records = []
    jobs = [dict(id=str(i), text='Bonjour.', output=str(tmp_path / f'{i}.wav'),
                 _audio_stage=str(tmp_path / f'{i}.stage.wav'),
                 _meta_stage=str(tmp_path / f'{i}.stage.json')) for i in range(2)]
    expected = (np.arange(2400, dtype=np.float64) - 1200) / 4096

    class Backend:
        manifest = {'checkpoint': 'synthetic', 'files': []}
        def synthesise(self, text, **kwargs):
            assert all(reference() is None for reference in prior_audio)
            samples = expected.copy()
            prior_audio.append(weakref.ref(samples))
            return SimpleNamespace(samples=samples, sample_rate=24000, channels=1, metadata={})
        def close(self):
            assert all(reference() is None for reference in prior_audio)

    pending = iter([*jobs, None])
    class Connection:
        def recv(self):
            # Includes the idle wait after every completed take.
            assert all(reference() is None for reference in prior_audio)
            return next(pending)
        def send(self, payload):
            records.append(payload)
        def close(self):
            pass

    args = args_for(app, tmp_path, [])
    monkeypatch.setattr(app, '_make_backend', lambda _args: Backend())
    monkeypatch.setattr(app, '_runtime_versions', lambda _backend: {})
    app._batch_worker(Connection(), args, {})
    assert [record['status'] for record in records] == ['ok', 'ok']
    # Release happens after export; both takes retain the same PCM24 samples.
    expected_pcm = np.floor(expected * 2**23) / 2**23
    for job in jobs:
        actual, rate = sf.read(job['_audio_stage'])
        assert rate == 24000
        np.testing.assert_array_equal(actual, expected_pcm)


def test_real_spawn_reuses_one_model_and_preserves_job_identity(tmp_path, monkeypatch):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_backend_worker)
    jobs = [dict(id='first', text=' Bonjour.\n', output='first.wav'),
            dict(id='second', text='Le budget est de 1 250,50 euros.', output='second.wav')]
    args = args_for(app, tmp_path, jobs)
    assert app.run_batch(args) == 0
    payload = json.loads(args.report.read_text())
    first, second = payload['attempts']
    assert first['text_original'] == jobs[0]['text']
    assert first['timings']['cache_state'] == 'cold_model'
    assert first['timings']['load_seconds'] > 0
    assert second['timings']['cache_state'] == 'warm_model'
    assert second['timings']['load_seconds'] == 0
    assert second['metadata']['load_seconds'] == 0
    assert second['metadata']['model_reused']
    assert first['worker_generation'] == second['worker_generation'] == 1
    assert (tmp_path / 'loads.txt').read_text() == 'load\n'
    assert (tmp_path / 'closes.txt').read_text() == 'close\n'
    assert json.loads((tmp_path / 'first.meta.json').read_text())['text_original'] == jobs[0]['text']
    assert not list(tmp_path.glob('.batch-take-*'))


@pytest.mark.parametrize('failure_text', ['slow', 'error'])
def test_timeout_or_error_reaps_then_starts_fresh_model(tmp_path, monkeypatch, failure_text):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_backend_worker)
    jobs = [dict(id='ok', text='ok', output='ok.wav'),
            dict(id='failed', text=failure_text, output='failed.wav', timeout=0.2 if failure_text == 'slow' else 8),
            dict(id='after', text='after', output='after.wav')]
    args = args_for(app, tmp_path, jobs, ['--continue-on-error'])
    assert app.run_batch(args) == 1
    records = json.loads(args.report.read_text())['attempts']
    assert [r['status'] for r in records] == ['ok', 'error', 'ok']
    assert records[2]['worker_generation'] == 2
    assert records[2]['timings']['cache_state'] == 'cold_model'
    assert (tmp_path / 'loads.txt').read_text() == 'load\nload\n'
    assert not (tmp_path / 'failed.wav').exists()
    assert not (tmp_path / 'failed.meta.json').exists()
    assert not list(tmp_path.glob('.batch-take-*'))
    for pid in {record['worker_pid'] for record in records}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not (tmp_path / 'leaked.txt').exists()


def test_first_job_timeout_includes_startup_and_failfast_skips(tmp_path, monkeypatch):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_backend_worker)
    args = args_for(app, tmp_path, [dict(id='first', text='slow', output='first.wav', timeout=0.01),
                                   dict(id='second', text='ok', output='second.wav')])
    assert app.run_batch(args) == 1
    records = json.loads(args.report.read_text())['attempts']
    assert [r['status'] for r in records] == ['error', 'skipped']
    assert records[0]['error']['code'] == 'timeout'
    assert not (tmp_path / 'second.wav').exists()


@pytest.mark.parametrize('change', ['duplicate_id', 'duplicate_output', 'existing_audio', 'existing_meta', 'invalid_last_text', 'report_collision', 'invalid_timeout'])
def test_whole_batch_preflight_before_model_load(tmp_path, monkeypatch, change):
    app = load_app()
    jobs = [dict(id='first', text='ok', output='first.wav'), dict(id='last', text='ok', output='last.wav')]
    if change == 'duplicate_id': jobs[1]['id'] = 'first'
    if change == 'duplicate_output': jobs[1]['output'] = 'first.wav'
    if change == 'existing_audio': (tmp_path / 'last.wav').write_bytes(b'preserve')
    if change == 'existing_meta': (tmp_path / 'last.meta.json').write_bytes(b'preserve')
    if change == 'invalid_last_text': jobs[1]['text'] = ' '
    if change == 'report_collision': (tmp_path / 'report.json').write_bytes(b'preserve')
    if change == 'invalid_timeout': jobs[1]['timeout'] = float('nan')
    args = args_for(app, tmp_path, jobs)
    monkeypatch.setattr(app.mp, 'get_context', lambda *_args: pytest.fail('worker spawning is forbidden'))
    with pytest.raises(Exception):
        app.run_batch(args)
    assert not (tmp_path / 'first.wav').exists()
    if change == 'existing_audio': assert (tmp_path / 'last.wav').read_bytes() == b'preserve'
    if change == 'existing_meta': assert (tmp_path / 'last.meta.json').read_bytes() == b'preserve'


@pytest.mark.parametrize('timeout', ['nan', 'inf', '-inf', '0'])
def test_speak_nonfinite_timeout_never_spawns(tmp_path, monkeypatch, timeout):
    app = load_app()
    monkeypatch.setattr(sys, 'argv', ['voice-cpu', 'speak', '--backend', 'kokoro', '--assets', str(tmp_path),
        '--language', 'fr', '--preset', 'ff_siwis', '--text', 'Bonjour.', '--output', str(tmp_path / 'out.wav'),
        '--timeout=' + timeout])
    monkeypatch.setattr(app.mp, 'get_context', lambda *_args: pytest.fail('worker must not start'))
    assert app.main() == 1


def test_batch_rejects_dangling_output_symlink_before_spawn(tmp_path, monkeypatch):
    app = load_app()
    (tmp_path / 'output.wav').symlink_to(tmp_path / 'missing.wav')
    args = args_for(app, tmp_path, [dict(id='one', text='ok', output='output.wav')])
    monkeypatch.setattr(app.mp, 'get_context', lambda *_args: pytest.fail('worker must not start'))
    with pytest.raises(Exception, match='fresh'):
        app.run_batch(args)
    assert (tmp_path / 'output.wav').is_symlink()
    assert not (tmp_path / 'missing.wav').exists()


def test_publication_collision_preserves_other_sidecar_and_rolls_back_owned_audio(tmp_path):
    app = load_app()
    audio, meta, output = tmp_path / 'audio.stage', tmp_path / 'meta.stage', tmp_path / 'output.wav'
    audio.write_bytes(b'new audio')
    meta.write_text('new metadata')
    other = output.with_suffix('.meta.json')
    other.write_text('existing metadata')
    with pytest.raises(FileExistsError):
        app._publish_staged(output, audio, meta)
    assert not output.exists()
    assert other.read_text() == 'existing metadata'


@pytest.mark.parametrize('continue_after_error', [False, True])
def test_output_parent_failure_is_recorded_with_batch_policy_and_reap(tmp_path, monkeypatch, continue_after_error):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_backend_worker)
    blocker = tmp_path / 'blocked'
    blocker.write_text('preserve file')
    jobs = [dict(id='first', text='ok', output='first.wav'),
            dict(id='blocked', text='ok', output='blocked/second.wav'),
            dict(id='third', text='ok', output='third.wav')]
    args = args_for(app, tmp_path, jobs, ['--continue-on-error'] if continue_after_error else [])
    assert app.run_batch(args) == 1
    records = json.loads(args.report.read_text())['attempts']
    assert [r['status'] for r in records] == ['ok', 'error', 'ok' if continue_after_error else 'skipped']
    assert records[1]['error']['code'] == 'FileExistsError'
    assert blocker.read_text() == 'preserve file'
    if continue_after_error:
        assert records[2]['worker_generation'] == 2
        assert records[2]['timings']['cache_state'] == 'cold_model'
        assert (tmp_path / 'third.wav').exists()
    else:
        assert not (tmp_path / 'third.wav').exists()
    for pid in {record.get('worker_pid') for record in records} - {None}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not list(tmp_path.glob('.batch-take-*'))


def test_partial_stage_creation_failure_cleans_and_continues(tmp_path, monkeypatch):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_backend_worker)
    mkstemp = app.tempfile.mkstemp
    calls = 0
    def fail_second_job_metadata_stage(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise PermissionError('synthetic stage error')
        return mkstemp(*args, **kwargs)
    monkeypatch.setattr(app.tempfile, 'mkstemp', fail_second_job_metadata_stage)
    args = args_for(app, tmp_path, [dict(id=str(i), text='ok', output=f'{i}.wav') for i in range(3)], ['--continue-on-error'])
    assert app.run_batch(args) == 1
    records = json.loads(args.report.read_text())['attempts']
    assert [r['status'] for r in records] == ['ok', 'error', 'ok']
    assert records[1]['error']['code'] == 'PermissionError'
    assert not (tmp_path / '1.wav').exists()
    assert not list(tmp_path.glob('.batch-take-*'))


def fake_reference_identity_worker(connection, args, source):
    import numpy as np
    app = load_app()
    mode = (args.assets / 'test-mode').read_text()
    class FakePocket:
        manifest = {'checkpoint': 'synthetic'}
        def __init__(self, **kwargs):
            if mode == 'constructor_mutates':
                args.reference.write_bytes(b'changed during model initialization')
        def synthesise(self, text, **kwargs):
            (args.assets / 'synthesized').write_text('called')
            return SimpleNamespace(samples=np.zeros(2400), sample_rate=24000, channels=1,
                                   metadata={'reference_sha256': 'wrong-conditioning-identity'})
        def close(self):
            pass
    app.get_backend = lambda _name: FakePocket
    app._batch_worker(connection, args, source)


@pytest.mark.parametrize('mode', ['constructor_mutates', 'metadata_diverges'])
def test_pocket_reference_mutation_or_divergent_conditioning_never_published(tmp_path, monkeypatch, mode):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', fake_reference_identity_worker)
    (tmp_path / 'test-mode').write_text(mode)
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'original approved synthetic reference')
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps([dict(id='clone', text='Bonjour.', output='clone.wav')]))
    args = app.build_parser().parse_args(['batch', '--backend', 'pocket-tts', '--assets', str(tmp_path),
        '--language', 'fr', '--reference', str(reference), '--jobs', str(jobs),
        '--report', str(tmp_path / 'report.json'), '--timeout', '8', '--json'])
    assert app.run_batch(args) == 1
    record = json.loads(args.report.read_text())['attempts'][0]
    assert record['status'] == 'error'
    assert record['error']['code'] == ('reference_changed' if mode == 'constructor_mutates' else 'reference_identity_mismatch')
    if mode == 'constructor_mutates':
        assert not (tmp_path / 'synthesized').exists()
    assert not (tmp_path / 'clone.wav').exists()
    assert not (tmp_path / 'clone.meta.json').exists()
    assert not list(tmp_path.glob('.batch-take-*'))
    with pytest.raises(ProcessLookupError):
        os.kill(record['worker_pid'], 0)


def _configure_actual_pocket_fake_encoder(args):
    """Run the real snapshot/conditioning adapter without models or torch imports."""
    import numpy as np
    import stat
    from tts_backends.pocket_tts import PocketBackend
    sys.modules['torch'] = SimpleNamespace(manual_seed=lambda _seed: None)
    app = load_app()
    mode = (args.assets / 'snapshot-mode').read_text()
    events = args.assets / 'snapshot-events.json'
    class Samples:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.zeros(2400)
    class Encoder:
        sample_rate = 24000
        def get_state_for_audio_prompt(self, path):
            snapshot = Path(path)
            assert snapshot.parent in (Path('/proc/self/fd'), Path('/dev/fd'))
            assert list(Path(args._reference_temp_dir).iterdir()) == []
            assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
            assert snapshot.read_bytes() == args.reference.read_bytes()
            prior = json.loads(events.read_text()) if events.exists() else []
            prior.append({'snapshot': str(snapshot), 'namespace': str(args._reference_temp_dir)})
            events.write_text(json.dumps(prior))
            if mode == 'encoder_error':
                raise RuntimeError('synthetic encoder error')
            if mode == 'timeout' or (mode == 'timeout_then_success' and len(prior) == 1):
                time.sleep(10)
            return {'approved': True}
        def generate_audio(self, state, text, **kwargs):
            return Samples()
    class FakeModelPocket(PocketBackend):
        def __init__(self, **kwargs):
            self.reference_temp_dir = Path(kwargs['reference_temp_dir'])
            assert self.reference_temp_dir == Path(args._reference_temp_dir)
            assert stat.S_IMODE(self.reference_temp_dir.stat().st_mode) == 0o700
            # This runs in the NEXT worker: parent must already have reaped and
            # removed the previous worker namespace, not just its final WAV.
            if events.exists():
                for old in json.loads(events.read_text()):
                    assert not Path(old['namespace']).exists()
            self.manifest = {'checkpoint': 'actual-adapter-fake-model'}
            self.model = Encoder()
            self.load_seconds = 0.0
            self._states = {}
            self._requests_started = 0
    app.get_backend = lambda _name: FakeModelPocket
    return app


def actual_pocket_snapshot_batch_worker(connection, args, source):
    app = _configure_actual_pocket_fake_encoder(args)
    app._batch_worker(connection, args, source)


def actual_pocket_snapshot_speak_worker(args):
    app = _configure_actual_pocket_fake_encoder(args)
    raise SystemExit(app._speak(args))


@pytest.mark.parametrize('mode', ['timeout_then_success', 'success', 'encoder_error'])
def test_supervisor_removes_actual_pocket_snapshots_after_reap(tmp_path, monkeypatch, mode):
    app = load_app()
    monkeypatch.setattr(app, '_batch_worker', actual_pocket_snapshot_batch_worker)
    (tmp_path / 'snapshot-mode').write_text(mode)
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'approved synthetic reference bytes')
    jobs = tmp_path / 'jobs.json'
    data = [dict(id='first', text='Bonjour.', output='first.wav', timeout=2)]
    if mode == 'timeout_then_success':
        data.append(dict(id='after', text='Bonjour ensuite.', output='after.wav', timeout=8))
    jobs.write_text(json.dumps(data))
    args = app.build_parser().parse_args(['batch', '--backend', 'pocket-tts', '--assets', str(tmp_path),
        '--language', 'fr', '--reference', str(reference), '--jobs', str(jobs),
        '--report', str(tmp_path / 'report.json'), '--timeout', '8', '--continue-on-error', '--json'])
    assert app.run_batch(args) == (0 if mode == 'success' else 1)
    records = json.loads(args.report.read_text())['attempts']
    if mode == 'timeout_then_success':
        assert [r['status'] for r in records] == ['error', 'ok']
        assert records[0]['error']['code'] == 'timeout'
        assert records[1]['worker_generation'] == 2
        assert not (tmp_path / 'first.wav').exists()
        assert (tmp_path / 'after.wav').exists()
    snapshots = json.loads((tmp_path / 'snapshot-events.json').read_text())
    assert snapshots
    for snapshot in snapshots:
        assert not Path(snapshot['namespace']).exists()
    for pid in {record['worker_pid'] for record in records}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize('mode', ['timeout', 'success', 'encoder_error'])
def test_speak_supervisor_removes_actual_pocket_snapshot_namespace(tmp_path, monkeypatch, mode):
    app = load_app()
    monkeypatch.setattr(app, '_worker', actual_pocket_snapshot_speak_worker)
    (tmp_path / 'snapshot-mode').write_text(mode)
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'approved synthetic reference bytes')
    monkeypatch.setattr(sys, 'argv', ['voice-cpu', 'speak', '--backend', 'pocket-tts',
        '--assets', str(tmp_path), '--language', 'fr', '--reference', str(reference),
        '--text', 'Bonjour.', '--output', str(tmp_path / 'take.wav'), '--timeout', '2', '--json'])
    assert app.main() == (0 if mode == 'success' else 1)
    for snapshot in json.loads((tmp_path / 'snapshot-events.json').read_text()):
        assert not Path(snapshot['namespace']).exists()
    assert (tmp_path / 'take.wav').exists() == (mode == 'success')


def test_unreaped_worker_preserves_reference_namespace(tmp_path):
    app = load_app()
    namespace = tmp_path / 'owned-reference'
    namespace.mkdir(mode=0o700)
    (namespace / 'snapshot.wav').write_bytes(b'private synthetic bytes')
    class Unreapable:
        pid = 12345
        def is_alive(self): return True
        def terminate(self): pass
        def kill(self): pass
        def join(self, *args): pass
    with pytest.raises(RuntimeError, match='could not be reaped'):
        app._stop_worker(Unreapable(), None, reference_temp_dir=str(namespace))
    assert (namespace / 'snapshot.wav').read_bytes() == b'private synthetic bytes'


def test_pocket_spawn_construction_failure_cleans_parent_namespace(tmp_path, monkeypatch):
    app = load_app()
    args = args_for(app, tmp_path, [dict(id='one', text='ok', output='one.wav')])
    args.backend, args.preset = 'pocket-tts', None
    args.reference = tmp_path / 'reference.wav'
    args.reference.write_bytes(b'approved synthetic reference')
    namespace = tmp_path / 'private-namespace'
    def create_namespace():
        namespace.mkdir(mode=0o700)
        return str(namespace)
    monkeypatch.setattr(app, '_private_reference_directory', create_namespace)
    real_context = app.mp.get_context('spawn')
    def fail_process(**kwargs):
        raise RuntimeError('synthetic process construction failure')
    context = SimpleNamespace(Pipe=real_context.Pipe, Process=fail_process)
    monkeypatch.setattr(app.mp, 'get_context', lambda _kind: context)
    assert app.run_batch(args) == 1
    record = json.loads(args.report.read_text())['attempts'][0]
    assert record['error']['code'] == 'RuntimeError'
    assert record['worker_pid'] is None
    assert not namespace.exists()
    assert not list(tmp_path.glob('.batch-take-*'))


def test_pocket_speak_spawn_start_failure_cleans_parent_namespace(tmp_path, monkeypatch):
    app = load_app()
    namespace = tmp_path / 'private-namespace'
    def create_namespace():
        namespace.mkdir(mode=0o700)
        return str(namespace)
    monkeypatch.setattr(app, '_private_reference_directory', create_namespace)
    class CannotStart:
        pid = None
        def start(self): raise RuntimeError('synthetic spawn failure')
        def close(self): pass
    monkeypatch.setattr(app.mp, 'get_context', lambda _kind: SimpleNamespace(Process=lambda **_kwargs: CannotStart()))
    monkeypatch.setattr(sys, 'argv', ['voice-cpu', 'speak', '--backend', 'pocket-tts', '--assets', str(tmp_path),
        '--language', 'fr', '--reference', str(tmp_path / 'reference.wav'), '--text', 'Bonjour.',
        '--output', str(tmp_path / 'out.wav')])
    with pytest.raises(RuntimeError, match='spawn failure'):
        app.main()
    assert not namespace.exists()
