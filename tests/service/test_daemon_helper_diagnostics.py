import pytest
import os
import errno
import sys
import time

from service import daemon_helper, daemon_client, install_policy


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


@pytest.mark.parametrize('reason,field,label', (
    ('daemon_observation_pair_changed','pid','control.state.pid'),
    ('daemon_observation_pair_changed','started','worker.state.started_at'),
    ('daemon_observation_startup_changed','boot','startup.hello.boot'),
))
def test_observation_change_crosses_actual_helper_client_parser_without_payload(
        reason, field, label, monkeypatch, capsys):
    from copy import deepcopy
    before={'control':{'State':{'Pid':1}},'worker':{'State':{'StartedAt':'old'}}}
    after=deepcopy(before)
    if field=='pid': after['control']['State']['Pid']=2
    elif field=='started': after['worker']['State']['StartedAt']='PRIVATE started'
    else:
        before={'hello':{'boot':'old'},'local':{}}
        after={'hello':{'boot':'PRIVATE boot'},'local':{}}
    kind='startup' if field=='boot' else 'pair'
    error=daemon_helper.observation_refusal(reason,before,after,kind)
    daemon_helper.diagnostic(error,'up','operation_failed')
    records=capsys.readouterr().err.encode()
    assert b'PRIVATE' not in records
    def attached(self, identity, phase, nonce, config, binding, source, diagnostics):
        diagnostics.feed(b'Docker PRIVATE payload\n'+records)
        raise ValueError('daemon_phase_eof_retained')
    monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
    client=object.__new__(daemon_client.Docker)
    with pytest.raises(ValueError) as caught:
        client.exchange('id','up','nonce',{}, {}, [])
    assert ':'.join((reason,'ValueError','unknown','0','changed='+label,
                     'host=daemon_phase_eof_retained')) in str(caught.value)
    assert str(caught.value.__cause__)=='daemon_phase_eof_retained'
    assert 'PRIVATE' not in str(caught.value)


def test_observation_change_labels_reject_private_and_unknown_fields():
    before={'control':{'State':{}},'worker':{}}
    after={'control':{'State':{},'PRIVATE-KEY':'PRIVATE-VALUE'},'worker':{}}
    assert daemon_helper.observation_changes(before,after,'pair')==('control.other',)
    diagnostics=daemon_client.HelperDiagnostics('up')
    diagnostics.feed(b'daemon_helper_diagnostic:up:operation_failed:ValueError:observe:511\n'
                     b'daemon_helper_cause:up:operation_failed:daemon_observation_pair_changed\n'
                     b'daemon_helper_change:up:operation_failed:control.PRIVATE-KEY\n')
    assert diagnostics.refusal()=='up:operation_failed:daemon_observation_pair_changed:ValueError:observe:511'


def test_observe_keeps_callee_failure_and_prior_bracket_cause(capsys):
    from types import SimpleNamespace
    from copy import deepcopy
    first={role:{'State':{'Pid':1},
                 'Mounts':[dict(Type='volume',Name=f'pab-{role}',
                                Source=f'/var/lib/docker/volumes/pab-{role}/_data',
                                Destination=f'/{role}',Driver='local',Mode='',RW=True,
                                Propagation='')],
                 'HostConfig':{'Mounts':[dict(Type='volume',Source=f'pab-{role}',
                                               Target=f'/{role}',ReadOnly=False,
                                               VolumeOptions={'NoCopy':True})]}}
           for role in ('control','worker')}
    second=deepcopy(first);second['worker']['State']['Pid']=2
    startup={'hello':{'boot':'b'*32},'local':{'roots':{}}}
    h=SimpleNamespace(evidence='fixture',b={},export=False,pocket=False,
        rpc=lambda *_:{'roots':{}},read=lambda *_:{'identities':{}},
        startup=lambda:deepcopy(startup))
    pairs=iter((first,second));h.pair=lambda _:next(pairs)
    with pytest.raises(ValueError,match='daemon_observation_pair_changed') as caught:
        daemon_helper.Helper.observe(h,{'control':'id','worker':'id'}, {})
    cleanup=OSError('PRIVATE cleanup')
    cleanup.__context__=caught.value
    daemon_helper.refusal(cleanup,'up')
    records=capsys.readouterr().err.encode()
    assert b'daemon_helper_cause:up:prior_failure:daemon_observation_pair_changed' in records
    assert b'daemon_helper_change:up:prior_failure:worker.state.pid' in records
    assert b'PRIVATE' not in records
    diagnostics=daemon_client.HelperDiagnostics('up');diagnostics.feed(records)
    assert diagnostics.refusal().startswith('up:prior_failure:daemon_observation_pair_changed:')
    h.pair=lambda _:deepcopy(first)
    h.startup=lambda: (_ for _ in ()).throw(ValueError('PRIVATE child failure'))
    with pytest.raises(ValueError,match='PRIVATE child failure'):
        daemon_helper.Helper.observe(h,{'control':'id','worker':'id'}, {})


def test_unknown_helper_error_never_projects_payload(capsys):
    daemon_helper.diagnostic(ValueError('SECRET credential'),'up','final_refusal')
    records=capsys.readouterr().err.encode()
    assert b'daemon_helper_cause' not in records
    diagnostics=daemon_client.HelperDiagnostics('up')
    diagnostics.feed(records)
    assert diagnostics.refusal()=='up:final_refusal:unclassified:ValueError:unknown:0'


def test_oserror_errno_and_internal_frame_cross_actual_parser_without_path(tmp_path, capsys, monkeypatch):
    secret=tmp_path/'PRIVATE-CREDENTIAL-AND-PATH'/'leaf'
    with pytest.raises(FileNotFoundError) as caught:
        with install_policy.parent_fd(secret):
            pass
    daemon_helper.diagnostic(caught.value,'init','operation_failed')
    records=capsys.readouterr().err.encode()
    assert b'PRIVATE-CREDENTIAL-AND-PATH' not in records
    assert b':install_policy.py:2\n' in records
    def attached(self, identity, phase, nonce, config, binding, source, diagnostics):
        diagnostics.feed(b'Docker PRIVATE-CREDENTIAL-AND-PATH\n'+records)
        raise ValueError('daemon_phase_eof_retained')
    monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
    client=object.__new__(daemon_client.Docker)
    with pytest.raises(ValueError) as failure:
        client.exchange('id','init','nonce',{}, {}, [])
    assert str(failure.value).startswith(
        'daemon_helper_refusal:init:operation_failed:unclassified:FileNotFoundError:parent_fd:')
    assert str(failure.value).endswith(
        ':file=install_policy.py:errno=2:host=daemon_phase_eof_retained')
    assert 'PRIVATE-CREDENTIAL-AND-PATH' not in str(failure.value)


@pytest.mark.parametrize('primary,cleanup,expected_class', (
    (TimeoutError('PRIVATE timeout'), None, 'TimeoutError'),
    (TimeoutError('PRIVATE timeout'), OSError(errno.EIO,'PRIVATE cleanup'), 'TimeoutError'),
    (BrokenPipeError(errno.EPIPE,'PRIVATE pipe'), None, 'OSError'),
    (type('PrivateOSError',(OSError,),{})(errno.EIO,'PRIVATE subclass'), None, 'OSError'),
    (type('TimeoutError',(OSError,),{})(errno.EIO,'PRIVATE disguised subclass'), None, 'OSError'),
))
def test_oserror_diagnostic_survives_helper_run_and_host_exchange(
        primary, cleanup, expected_class, capsys, monkeypatch):
    from contextlib import contextmanager
    helper=object.__new__(daemon_helper.Helper)
    helper.phase='up'
    @contextmanager
    def held():
        try:
            yield
        finally:
            if cleanup is not None: raise cleanup
    helper.held=held
    def runtime(): raise primary
    helper.runtime=runtime
    with pytest.raises(OSError) as caught:
        helper.run()
    assert caught.value is (cleanup or primary)
    if cleanup is not None: assert cleanup.__context__ is primary
    daemon_helper.refusal(caught.value,'up')
    records=capsys.readouterr().err.encode()
    assert b'PRIVATE' not in records
    assert b':operation_failed:'+expected_class.encode()+b':' in records
    def attached(self, identity, phase, nonce, config, binding, source, diagnostics):
        diagnostics.feed(records)
        raise ValueError('daemon_phase_eof_retained')
    monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
    client=object.__new__(daemon_client.Docker)
    with pytest.raises(ValueError) as host:
        client.exchange('id','up','nonce',{}, {}, [])
    assert str(host.value).startswith('daemon_helper_refusal:up:operation_failed:unclassified:'+
                                      expected_class+':run:')
    assert str(host.value).endswith(':host=daemon_phase_eof_retained')
    assert host.value.__cause__.args==('daemon_phase_eof_retained',)
    assert 'PRIVATE' not in str(host.value)


@pytest.mark.parametrize('location', (
    'private.py:2', 'persistent_init.py:999',
    'persistent_init.py:2:/private/SECRET',
))
def test_parser_rejects_untrusted_oserror_provenance(location):
    diagnostics=daemon_client.HelperDiagnostics('init')
    diagnostics.feed(('daemon_helper_diagnostic:init:operation_failed:OSError:'
                      'initialize_split:403:'+location+'\n').encode())
    assert diagnostics.refusal() is None


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
