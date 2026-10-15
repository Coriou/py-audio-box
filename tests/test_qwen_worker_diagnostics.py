"""No model imports: exercise real spawned worker death and restart."""
import os
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from tts_backends import qwen


def diagnostic_worker(connection, settings):
    if settings.get('early_exit'):
        os._exit(23)
    connection.send(('ready', {'revision': 'test-revision'}))
    method, args, kwargs = connection.recv()
    if method == 'die':
        os._exit(24)
    if method == 'kill':
        os.kill(os.getpid(), signal.SIGKILL)
    connection.send(('ok', 'healthy'))
    connection.close()


def backend(monkeypatch, **settings):
    monkeypatch.setattr(qwen, '_worker', diagnostic_worker)
    return qwen.QwenBackend(model_name='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', dtype_str='float32', **settings)


def test_startup_exit_is_diagnostic_and_restartable(monkeypatch):
    model = backend(monkeypatch, early_exit=True)
    with pytest.raises(qwen.QwenWorkerError) as caught:
        model.invoke('generate_custom_voice', kwargs={'text': 'private text'}, timeout_s=10)
    assert caught.value.details['stage'] == 'startup'
    assert caught.value.details['exitcode'] == 23
    assert caught.value.details['dtype'] == 'float32'
    assert 'private text' not in str(caught.value)
    assert model._process is None
    model.settings['early_exit'] = False
    try:
        assert model.invoke('healthy', timeout_s=10) == 'healthy'
    finally:
        model.close()


@pytest.mark.parametrize('method,exitcode', [('die', 24), ('kill', -signal.SIGKILL)])
def test_midrequest_exit_has_status_without_false_oom(monkeypatch, method, exitcode):
    monkeypatch.setattr(qwen, '_memory_snapshot', lambda: {'oom_kill': 7, 'memory_limit_bytes': 9000})
    model = backend(monkeypatch)
    with pytest.raises(qwen.QwenWorkerError) as caught:
        model.invoke(method, timeout_s=10)
    assert caught.value.details['stage'] == 'response'
    assert caught.value.details['exitcode'] == exitcode
    assert caught.value.details['cgroup_oom_kill_delta'] == 0
    assert 'OOM observed' not in str(caught.value)
    if method == 'kill':
        assert caught.value.details['signal'] == 'SIGKILL'
    try:
        assert model.invoke('healthy', timeout_s=10) == 'healthy'
    finally:
        model.close()


def test_shared_cgroup_oom_evidence_is_bounded(monkeypatch):
    snapshots = iter([{'oom_kill': 2}, {'oom_kill': 3, 'memory_limit_bytes': 1024}])
    monkeypatch.setattr(qwen, '_memory_snapshot', lambda: next(snapshots))
    model = backend(monkeypatch)
    with pytest.raises(qwen.QwenWorkerError) as caught:
        model.invoke('die', timeout_s=10)
    assert caught.value.details['cgroup_oom_kill_delta'] == 1
    assert 'attribution unproven' in str(caught.value)
    assert model._process is None


@pytest.mark.parametrize('stage', ['send', 'poll', 'recv'])
def test_transport_races_are_diagnostic_and_reaped(monkeypatch, stage):
    class Process:
        pid = 123
        exitcode = 31
        def is_alive(self):
            return False
        def join(self, timeout):
            pass
        def close(self):
            self.closed = True

    class Connection:
        def send(self, payload):
            if stage == 'send':
                raise BrokenPipeError('synthetic')
        def poll(self, timeout):
            if stage == 'poll':
                raise OSError('synthetic')
            return True
        def recv(self):
            raise EOFError('synthetic')
        def close(self):
            self.closed = True

    monkeypatch.setattr(qwen, '_memory_snapshot', lambda: {})
    model = backend(monkeypatch)
    process, connection = Process(), Connection()
    model._process, model._connection = process, connection
    with pytest.raises(qwen.QwenWorkerError) as caught:
        model.invoke('generate_custom_voice', timeout_s=10)
    assert caught.value.details['exitcode'] == 31
    assert caught.value.details['stage'] == ('send' if stage == 'send' else 'response')
    assert process.closed and connection.closed
    assert model._process is None


@pytest.mark.parametrize('failure_stage', ['construct', 'start'])
def test_spawn_failures_close_both_pipe_ends(monkeypatch, failure_stage):
    class Connection:
        closed = False
        def close(self):
            self.closed = True

    class Process:
        pid = None
        exitcode = None
        closed = False
        def start(self):
            raise OSError('synthetic spawn failure')
        def join(self, timeout):
            raise AssertionError('cannot join unstarted process')
        def close(self):
            self.closed = True

    parent, child, process = Connection(), Connection(), Process()

    class Context:
        def Pipe(self):
            return parent, child
        def Process(self, **kwargs):
            if failure_stage == 'construct':
                raise OSError('synthetic constructor failure')
            return process

    monkeypatch.setattr(qwen.mp, 'get_context', lambda method: Context())
    monkeypatch.setattr(qwen, '_memory_snapshot', lambda: {})
    model = backend(monkeypatch)
    with pytest.raises(qwen.QwenWorkerError):
        model.invoke('healthy', timeout_s=10)
    assert parent.closed and child.closed
    if failure_stage == 'start':
        assert process.closed
    assert model._process is None
