import pytest

from service import daemon_helper as dh


def helper(states):
    h=object.__new__(dh.Helper)
    h.end=10**30
    calls=[]
    def inspect(role, ids):
        calls.append(role)
        value=next(states)
        if isinstance(value, Exception): raise value
        return {'State': value}
    h.inspect=inspect
    return h,calls


@pytest.mark.parametrize('code', [2, 143, -15])
def test_terminal_nonzero_refuses_without_another_poll(code, capsys):
    h,calls=helper(iter([{'Running': False, 'ExitCode': code}]))
    with pytest.raises(ValueError, match='^daemon_control_terminal_nonzero_retained$') as caught:
        h.wait_control_stop({'control': 'id'})
    assert caught.value.control_exit_code==code
    assert calls==['control']
    dh.diagnostic(caught.value,'stop','operation_failed')
    output=capsys.readouterr().err
    assert 'daemon_helper_cause:stop:operation_failed:daemon_control_terminal_nonzero_retained' in output
    assert f'daemon_helper_control_exit:stop:operation_failed:{code}' in output


def test_live_then_zero_succeeds_in_order(monkeypatch):
    h,calls=helper(iter([{'Running': True, 'ExitCode': 0}, {'Running': False, 'ExitCode': 0}]))
    monkeypatch.setattr(dh.time,'sleep',lambda _:None)
    assert h.wait_control_stop({'control':'id'})=={'State':{'Running':False,'ExitCode':0}}
    assert calls==['control','control']


def test_live_deadline_remains_retained(monkeypatch):
    h,calls=helper(iter([{'Running': True, 'ExitCode': 0}]))
    monkeypatch.setattr(dh.time,'monotonic',iter([0,1]).__next__)
    h.end=1
    with pytest.raises(ValueError,match='^daemon_runtime_wait_timeout_retained$'):
        h.wait_control_stop({'control':'id'})
    assert calls==['control']


def test_invalid_inspect_fails_closed():
    h,calls=helper(iter([ValueError('daemon_container_state')]))
    with pytest.raises(ValueError,match='^daemon_container_state$'):
        h.wait_control_stop({'control':'id'})
    assert calls==['control']

# Runtime integration uses real Helper.runtime/run and real inspection validation.
# Only the surrounding durable enrollment and Docker wire are inert fixtures.
@pytest.mark.parametrize('marker_phase', ['ready', 'not_ready'])
@pytest.mark.parametrize('code', [0, 2, 143, -15])
def test_runtime_stop_refuses_before_worker_or_receipts(monkeypatch, capsys, marker_phase, code):
    from contextlib import nullcontext
    from copy import deepcopy
    from tests.service.test_split_init import SplitInit
    from tests.service.daemon_volume_fixtures import binding, container
    from service import daemon_volumes as dv, daemon_client

    fixture=SplitInit(); fixture.setUp()
    try:
        config=fixture.config; b=binding(config)
        ids={role:container(role,config,b)['Id'] for role in ('control','worker')}
        pair={role:container(role,config,b) for role in ids}
        for c in pair.values():
            c['State'].update(Running=True, Status='running', StartedAt='2026-10-25T00:00:01Z', Pid=1)
        record={'fixture':'registration','evidence':dv.EVIDENCE}; rid=dv.digest(record)
        public={'fixture':'enrollment'}
        marker=dict(phase=marker_phase,registration_id=rid)
        events=[]; inspect_count=0; control_terminated=False; live_seen=False
        def rpc(action,payload):
            nonlocal inspect_count, control_terminated, live_seen
            events.append((action,payload.get('role')))
            if action=='marker': return deepcopy(marker)
            c=pair[payload['role']]
            if action=='term':
                if payload['role']=='control': control_terminated=True
                if payload['role']=='worker':
                    assert live_seen and not pair['control']['State']['Running']
                if payload['role']!='control' or code!=0:
                    c['State'].update(Running=False,Status='exited',ExitCode=code if payload['role']=='control' else 0,Pid=0)
            if action=='inspect' and payload['role']=='control' and control_terminated and code==0:
                if not live_seen:
                    live_seen=True
                    events.append(('observed_live_control',None))
                else:
                    c['State'].update(Running=False,Status='exited',ExitCode=0,Pid=0)
                    events.append(('observed_zero_control',None))
            if action=='inspect' and payload['role']=='control': inspect_count+=1
            return deepcopy(c)
        h=object.__new__(dh.Helper)
        h.phase='stop'; h.nonce='f'*32; h.config=config; h.b=b; h.family=None
        h.evidence=dv.EVIDENCE; h.end=10**30; h.own={'Id':'a'*64,'State':{'StartedAt':'2026-10-25T00:00:00Z'}}
        h.held=nullcontext; h.public=lambda:public; h.exists=lambda name:name=='containers.json'
        h.read=lambda name: ids if name=='containers.json' else (dict(binding=dv.digest(b)) if name=='container-creation-intent.json' else
            dict(id=ids[name.split('-')[1].split('.')[0]]) if name.startswith('created-') else
            dict(registration=record,enrollment_digest=dv.digest(public)))
        h.rpc=rpc; h.observe=lambda *_: {'fixture':'observation'}
        h.clean=lambda **_: events.append(('clean',None)) or {}
        h.snapshot=lambda: events.append(('snapshot',None))
        h.quarantined_reap=lambda *_: events.append(('reap',None))
        h.write=lambda name,_:events.append(('write',name))
        monkeypatch.setattr(dh.time,'sleep',lambda _:None)
        monkeypatch.setattr(dv,'observation',lambda *_ ,**__:None)
        if code==0:
            result=h.run()
            assert result['phase']==('stopped' if marker_phase=='ready' else 'stopped_quarantined')
            assert events.index(('term','control')) < events.index(('observed_live_control',None)) < events.index(('observed_zero_control',None)) < events.index(('term','worker'))
            assert events.index(('inspect','control'),events.index(('term','control'))) < events.index(('term','worker'))
            assert ('reap',None) in events if marker_phase=='not_ready' else ('write','stopped-'+h.nonce+'.json') in events
            return
        with pytest.raises(ValueError,match='^daemon_control_terminal_nonzero_retained$'):
            h.run()
        diagnostic=capsys.readouterr().err.encode()
        assert inspect_count==2  # initial pair, then terminal control inspection
        assert events[-2:]==[('term','control'),('inspect','control')]
        assert not any(action in ('clean','reap','write') or role=='worker' and action=='term' for action,role in events)
        assert b'daemon_helper_control_exit:stop:operation_failed:' in diagnostic
        def attached(self,identity,phase,nonce,config,binding,source,stream):
            stream.feed(b'private stderr SECRET\n'+diagnostic)
            raise ValueError('daemon_phase_eof_retained')
        monkeypatch.setattr(daemon_client.Docker,'_exchange_attached',attached)
        with pytest.raises(ValueError,match='^daemon_helper_refusal:stop:operation_failed:daemon_control_terminal_nonzero_retained:') as caught:
            object.__new__(daemon_client.Docker).exchange('id','stop','nonce',{}, {}, [])
        assert f'control_exit={code}' in str(caught.value)
        assert 'SECRET' not in str(caught.value)
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('value', ['0','-0','256','-256','01','+2','True','2x','9'*500,'SECRET'])
def test_control_exit_parser_rejects_untrusted_values(value):
    from service.daemon_client import HelperDiagnostics
    d=HelperDiagnostics('stop')
    d.feed(('daemon_helper_diagnostic:stop:operation_failed:ValueError:run:1\n'
            'daemon_helper_cause:stop:operation_failed:daemon_control_terminal_nonzero_retained\n'
            f'daemon_helper_control_exit:stop:operation_failed:{value}\n').encode())
    assert 'control_exit=' not in d.refusal()
    assert value not in d.refusal()

@pytest.mark.parametrize('state_change', [
    {'Running':'false'}, {'Running':1}, {'ExitCode':'2'}, {'ExitCode':True},
])
def test_real_inspection_rejects_malformed_control_state(state_change):
    from tests.service.test_split_init import SplitInit
    from tests.service.daemon_volume_fixtures import binding, container
    fixture=SplitInit(); fixture.setUp()
    try:
        b=binding(fixture.config)
        c=container('control',fixture.config,b)
        c['State'].update(Running=False,Status='exited',ExitCode=2,StartedAt='2026-10-25T00:00:01Z',Pid=0)
        c['State'].update(state_change)
        h=object.__new__(dh.Helper)
        h.config=fixture.config; h.b=b; h.family=None
        h.rpc=lambda action,payload:c
        with pytest.raises(ValueError):
            h.inspect('control',{'control':c['Id']})
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('values', [('2','143'), ('-0',), ('0',), ('2','143','2')])
def test_exchange_suppresses_noncanonical_and_conflicting_exit(monkeypatch, values):
    from service.daemon_client import Docker
    def attached(self, identity, phase, nonce, config, binding, source, stream):
        stream.feed(b'daemon_helper_diagnostic:stop:operation_failed:ValueError:run:1\n'
                    b'daemon_helper_cause:stop:operation_failed:daemon_control_terminal_nonzero_retained\n')
        for value in values:
            stream.feed(('daemon_helper_control_exit:stop:operation_failed:'+value+'\n').encode())
        raise ValueError('daemon_phase_eof_retained')
    monkeypatch.setattr(Docker,'_exchange_attached',attached)
    with pytest.raises(ValueError,match='daemon_control_terminal_nonzero_retained') as caught:
        object.__new__(Docker).exchange('id','stop','nonce',{}, {}, [])
    assert 'control_exit=' not in str(caught.value)

@pytest.mark.parametrize('code', [256,-256])
def test_out_of_diagnostic_range_exit_still_refuses_with_valid_inspection(code, capsys):
    from tests.service.test_split_init import SplitInit
    from tests.service.daemon_volume_fixtures import binding, container
    fixture=SplitInit(); fixture.setUp()
    try:
        b=binding(fixture.config)
        c=container('control',fixture.config,b)
        c['State'].update(Running=False,Status='exited',ExitCode=code,StartedAt='2026-10-25T00:00:01Z',Pid=0)
        h=object.__new__(dh.Helper)
        h.config=fixture.config; h.b=b; h.family=None; h.end=10**30
        h.rpc=lambda action,payload:c
        assert h.inspect('control',{'control':c['Id']})['State']['ExitCode']==code
        with pytest.raises(ValueError,match='^daemon_control_terminal_nonzero_retained$') as caught:
            h.wait_control_stop({'control':c['Id']})
        dh.diagnostic(caught.value,'stop','operation_failed')
        assert 'daemon_helper_control_exit:' not in capsys.readouterr().err
    finally:
        fixture.doCleanups()
