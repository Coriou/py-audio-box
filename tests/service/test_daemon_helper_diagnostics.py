import pytest
import os
import sys
import time

from service import daemon_helper, daemon_client


def test_refusal_location_does_not_disclose_exception_payload(monkeypatch, capsys):
    monkeypatch.setattr(daemon_helper.sys, 'platform', 'linux')
    def denied():
        raise ValueError('private-credential-or-manifest-content')
    monkeypatch.setattr(daemon_helper.os, 'getresuid', denied, raising=False)
    with pytest.raises(SystemExit) as stopped:
        daemon_helper.main()
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert 'daemon_helper_refused:ValueError:main:' in output.err
    assert 'private-credential-or-manifest-content' not in output.err
    assert len(output.err) < 256


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_original_diagnostic_precedes_cleanup_without_suppressing_it(capsys, cleanup_fails):
    from contextlib import contextmanager
    events=[]
    primary=ValueError('SECRET primary config')
    secondary=OSError('SECRET cleanup key')
    helper=object.__new__(daemon_helper.Helper)
    helper.phase='up'
    @contextmanager
    def held():
        try:
            yield
        finally:
            events.append('cleanup')
            if cleanup_fails:raise secondary
    helper.held=held
    def runtime():
        events.append('runtime')
        raise primary
    helper.runtime=runtime
    with pytest.raises((ValueError,OSError)) as caught:
        helper.run()
    assert caught.value is (secondary if cleanup_fails else primary)
    if cleanup_fails:assert secondary.__context__ is primary
    assert events==['runtime','cleanup']
    daemon_helper.diagnostic(caught.value,'up','final_refusal')
    output=capsys.readouterr()
    assert output.out==''
    lines=output.err.splitlines()
    assert lines[0].startswith('daemon_helper_diagnostic:up:operation_failed:ValueError:run:')
    assert ('final_refusal:OSError:' if cleanup_fails else 'final_refusal:ValueError:') in lines[1]
    assert len(lines)==3
    assert 'SECRET' not in output.err


def test_diagnostic_uses_fixed_vocabulary(capsys):
    secret_type=type('SECRET_CLASS',(Exception,),{})
    daemon_helper.diagnostic(secret_type('SECRET_PAYLOAD'),'SECRET_PHASE','SECRET_CODE')
    output=capsys.readouterr()
    assert output.out==''
    assert 'SECRET' not in output.err
    assert output.err.splitlines()==[
        'daemon_helper_diagnostic:unknown:final_refusal:Exception:unknown:0',
        'daemon_helper_refused:Exception:unknown:0']


def test_nested_cleanup_chain_retains_both_failures_and_runs_all_callbacks(capsys):
    from contextlib import ExitStack
    events=[]
    def cleanup_failure():
        events.append('failing_cleanup')
        raise OSError('SECRET cleanup')
    try:
        with ExitStack() as stack:
            stack.callback(events.append,'last_cleanup')
            stack.callback(cleanup_failure)
            raise ValueError('SECRET startup')
    except OSError as exc:
        daemon_helper.refusal(exc,'up')
    assert events==['failing_cleanup','last_cleanup']
    output=capsys.readouterr()
    assert output.out=='' and 'SECRET' not in output.err
    assert ':prior_failure:ValueError:' in output.err
    assert ':final_refusal:OSError:' in output.err
    assert len(output.err.splitlines())==3


def test_known_helper_cause_crosses_host_boundary_without_secret(monkeypatch, capsys):
    daemon_helper.diagnostic(ValueError('daemon_helper_mountinfo'),'up','final_refusal')
    records=capsys.readouterr().err.encode()
    assert b'daemon_helper_cause:up:final_refusal:daemon_helper_mountinfo' in records
    def attached(self, identity, phase, nonce, config, binding, source, stream):
        stream.feed(b'Docker daemon says SECRET\n'+records)
        raise ValueError('daemon_phase_eof_retained')
    monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
    client=object.__new__(daemon_client.Docker)
    with pytest.raises(ValueError,match=r'^daemon_helper_refusal:up:final_refusal:daemon_helper_mountinfo:ValueError:unknown:0:host=daemon_phase_eof_retained$') as caught:
        client.exchange('id','up','nonce',{}, {}, [])
    assert str(caught.value.__cause__)=='daemon_phase_eof_retained'
    assert 'SECRET' not in str(caught.value)


def test_unknown_helper_error_never_projects_payload(capsys):
    daemon_helper.diagnostic(ValueError('SECRET credential'),'up','final_refusal')
    records=capsys.readouterr().err.encode()
    assert b'daemon_helper_cause' not in records
    diagnostics=daemon_client.HelperDiagnostics('up')
    diagnostics.feed(records)
    assert diagnostics.refusal()=='up:final_refusal:unclassified:ValueError:unknown:0'


def test_early_cause_survives_large_noisy_stderr_and_bounded_tail(monkeypatch):
    record=(b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:12\n'
            b'daemon_helper_cause:up:operation_failed:daemon_helper_mountinfo\n')
    script=("import sys; e=sys.stderr.buffer; e.write("+repr(record)+"); e.flush(); "
            "e.write(b'PRIVATE_NOISE\\n'*200000); e.flush(); sys.stdin.buffer.readline()")
    client=object.__new__(daemon_client.Docker)
    client.argv=lambda _: [sys.executable,'-c',script]
    client.container=lambda _: {'State':{'Running':True}}
    client.env=os.environ.copy(); client.end=time.monotonic()+5
    client.retained=[]
    monkeypatch.setattr(daemon_client.dv,'inspect_container',lambda *a,**k: None)
    spawned=[]; popen=daemon_client.subprocess.Popen
    def capture_process(*args,**kwargs):
        process=popen(*args,**kwargs); spawned.append(process); return process
    monkeypatch.setattr(daemon_client.subprocess,'Popen',capture_process)
    diagnostics=daemon_client.HelperDiagnostics('up')
    monkeypatch.setattr(daemon_client,'HelperDiagnostics',lambda phase: diagnostics)
    with pytest.raises(ValueError,match=r'^daemon_helper_refusal:up:operation_failed:daemon_helper_mountinfo:ValueError:run:12:host=daemon_phase_eof_retained$'):
        client.exchange('id','up','nonce',{}, {}, [])
    assert len(diagnostics.tail)<=4096
    assert len(diagnostics.line)<=512
    assert diagnostics.thread is not None and not diagnostics.thread.is_alive()
    assert diagnostics.stream.closed and client.retained==[]
    assert len(spawned)==1 and spawned[0].poll() is not None
    assert spawned[0].stdin.closed and spawned[0].stdout.closed


def test_partial_overlong_and_invalid_records_are_ignored():
    diagnostics=daemon_client.HelperDiagnostics('up')
    diagnostics.feed(b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:1')
    assert diagnostics.refusal() is None
    diagnostics.feed(b'PRIVATE'*100+b'\n')
    diagnostics.feed(b'X'*1024+b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:2\n')
    diagnostics.feed(b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:3\xff\n')
    diagnostics.feed(b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:4\n')
    diagnostics.feed(b'daemon_helper_diagnostic:up:operation_failed:ValueError:run:5\n')
    assert diagnostics.refusal()=='up:operation_failed:unclassified:ValueError:run:4'


def test_unrelated_host_exception_remains_primary_with_safe_diagnostic(monkeypatch):
    primary=OSError('PRIVATE host path')
    def attached(self, identity, phase, nonce, config, binding, source, diagnostics):
        diagnostics.feed(b'daemon_helper_diagnostic:up:final_refusal:ValueError:run:4\n')
        raise primary
    monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
    client=object.__new__(daemon_client.Docker)
    with pytest.raises(OSError) as caught:
        client.exchange('id','up','nonce',{}, {}, [])
    assert caught.value is primary
    assert caught.value.__notes__==['daemon_helper_diagnostic:up:final_refusal:unclassified:ValueError:run:4']


def test_deadline_retains_process_until_reaped_and_closes_reader(monkeypatch):
    script='import sys,time; sys.stderr.write("noise\\n"); time.sleep(.2)'
    client=object.__new__(daemon_client.Docker)
    client.argv=lambda _: [sys.executable,'-c',script]
    client.container=lambda _: {'State':{'Running':True}}
    client.env=os.environ.copy(); client.end=time.monotonic()+.05
    client.retained=[]
    monkeypatch.setattr(daemon_client.dv,'inspect_container',lambda *a,**k: None)
    spawned=[]; popen=daemon_client.subprocess.Popen
    def capture_process(*args,**kwargs):
        process=popen(*args,**kwargs); spawned.append(process); return process
    monkeypatch.setattr(daemon_client.subprocess,'Popen',capture_process)
    diagnostics=daemon_client.HelperDiagnostics('up')
    monkeypatch.setattr(daemon_client,'HelperDiagnostics',lambda phase: diagnostics)
    with pytest.raises(ValueError,match='daemon_phase_timeout'):
        client.exchange('id','up','nonce',{}, {}, [])
    assert client.retained==[]
    assert diagnostics.thread is not None and not diagnostics.thread.is_alive()
    assert diagnostics.stream.closed
    assert len(spawned)==1 and spawned[0].poll() is not None
    assert spawned[0].stdin.closed and spawned[0].stdout.closed
