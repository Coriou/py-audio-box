"""E1 durable journal and real control ordering; no invented reaper evidence."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from worker.runtime_journal import Journal
from worker.runtime_protocol import request, session
from worker.transport import Refused, Session, canonical
from service.lifecycle_cli import fixture_authority
from service.owner_store import OwnerStore
from service.worker_supervisor import Supervisor
from worker.lifecycle import Observation, State


def write_journal(path, value):
    (path/'journal.json').write_text(json.dumps(dict(payload=value,sha256=hashlib.sha256(canonical(value)).hexdigest())))


def test_domain_and_closed_frames():
    key = 'ab'*32
    assert session(key, 1, 2).secret != key
    for value in ({}, {'version':'lifecycle-v1','op':'execute','body':{}},
                  {'version':'lifecycle-v1','op':'hello','body':{},'argv':[]}):
        with pytest.raises(Refused): request(value)
    assert request(dict(version='lifecycle-v1',op='hello',body={})) == ('hello',{})


def test_missing_corrupt_and_exclusive_journal(tmp_path):
    with pytest.raises(FileNotFoundError): Journal(tmp_path)
    (tmp_path/'journal.json').write_text('{}')
    with pytest.raises(ValueError): Journal(tmp_path)
    write_journal(tmp_path,dict(version=1,lane='test',records=[]))
    journal = Journal(tmp_path)
    with pytest.raises(BlockingIOError): Journal(tmp_path)
    value = copy.deepcopy(journal.value)
    value['lane']='changed'
    journal.save(value)
    assert json.loads((tmp_path/'journal.json').read_text())['payload'] == value
    assert (tmp_path/'journal.json').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('point',['prespawn','preregistration','prestart','postspawn'])
def test_intent_bind_and_reconnect_never_respawn(tmp_path, point):
    authority = fixture_authority()
    store = OwnerStore(tmp_path/'owner.db')
    calls=[]
    class Runtime:
        def observe(self, identity, timeout):
            return Observation(identity,State.UNKNOWN)
    supervisor=Supervisor(store,Runtime())
    def prepare():
        assert store.current()['runtime_id'] is None
        calls.append('prepare');return 'lane/boot/allocation'
    def start(identity):
        assert store.current()['runtime_id'] == identity
        calls.append('start')
    def fault(where):
        if where == point: raise RuntimeError('control_death')
    with pytest.raises(RuntimeError): supervisor.dispatch(authority,prepare,start,fault=fault)
    before=list(calls)
    supervisor.dispatch(authority,prepare,start)
    assert calls == before
    store.close()


def test_enospc_start_fence_never_launches(tmp_path, monkeypatch):
    """E1 injected ENOSPC at the actual fence persistence call, not a callback count."""
    import errno
    import queue
    import threading
    import time
    from worker.runtime_supervisor import RuntimeSupervisor
    authority=fixture_authority()
    allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='silence')
    row=dict(id='lane/boot/nonce',allocation=allocation,phase='prepared',acked=False,uid=20000,boot='boot')
    write_journal(tmp_path,dict(version=1,lane='lane',records=[row]))
    runtime=RuntimeSupervisor.__new__(RuntimeSupervisor)
    runtime.journal=Journal(tmp_path);runtime.failed=False;runtime.launches=0
    runtime.active=dict(id=row['id'],cutoff=time.monotonic()+5,started=False)
    def no_space(*args,**kwargs):raise OSError(errno.ENOSPC,'injected bounded ENOSPC')
    monkeypatch.setattr('worker.runtime_journal.os.open',no_space)
    launched=[]
    monkeypatch.setattr('worker.runtime_supervisor.subprocess.Popen',lambda *a,**k:launched.append(a))
    with pytest.raises(Refused):runtime.operation('start',dict(runtime_id=row['id']))
    assert not launched and runtime.failed and runtime.state(row)=='unknown'
    assert runtime.journal.value['records'][0]['phase']=='prepared'


def test_reservation_replay_and_new_boot_quarantine(tmp_path):
    from worker.runtime_supervisor import RuntimeSupervisor
    authority=fixture_authority()
    write_journal(tmp_path,dict(version=1,lane='lane',records=[]))
    def runtime():
        value=RuntimeSupervisor.__new__(RuntimeSupervisor)
        import threading
        value.tree_lock=threading.Lock()
        value.journal=Journal(tmp_path);value.cutoffs={};value.boot='boot';value.failed=False;value.active=None
        return value
    first=runtime()
    body=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='silence')
    reservation=first.operation('prepare',body)
    cutoff=first.active['cutoff']
    assert first.operation('prepare',body)==reservation
    assert first.active['cutoff']==cutoff
    with pytest.raises(Refused):first.operation('ack',dict(runtime_id=reservation['runtime_id']))
    first.journal.lock.close()
    second=runtime();second.boot='newboot'
    assert second.operation('prepare',body)['state']=='unknown'
    assert second.operation('start',dict(runtime_id=reservation['runtime_id']))['state']=='unknown'
    fresh=fixture_authority()
    with pytest.raises(Refused):second.operation('prepare',dict(snapshot=json.loads(fresh._snapshot),binding=fresh.binding,fixture='silence'))


def test_corrupt_terminal_cannot_claim_stopped(tmp_path):
    authority=fixture_authority()
    row=dict(id='lane/boot/nonce',allocation=dict(snapshot=json.loads(authority._snapshot),
        binding=authority.binding,fixture='silence'),phase='terminal',acked=False,uid=20000,boot='boot')
    write_journal(tmp_path,dict(version=1,lane='lane',records=[row]))
    with pytest.raises(ValueError):Journal(tmp_path)


def test_receipt_capacity_refuses_instead_of_evicting(tmp_path):
    from worker.runtime_supervisor import RuntimeSupervisor
    authority=fixture_authority()
    allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='silence')
    records=[]
    for index in range(8):
        identity='lane/boot/'+str(index)
        records.append(dict(id=identity,allocation=allocation,phase='terminal',acked=False,uid=20000+index,boot='boot',
            proposal=None,receipt=dict(runtime_id=identity,binding=authority.binding,reaped=True,timely=False,
            original_deadline=authority.job['deadline'],engine_uid=20000+index,reap_method='pid1-waitpid-echild',reaped_children=1)))
    write_journal(tmp_path,dict(version=1,lane='lane',records=records))
    runtime=RuntimeSupervisor.__new__(RuntimeSupervisor)
    import threading
    runtime.tree_lock=threading.Lock()
    runtime.journal=Journal(tmp_path);runtime.boot='boot';runtime.failed=False;runtime.active=None;runtime.cutoffs={}
    fresh=fixture_authority()
    body=dict(snapshot=json.loads(fresh._snapshot),binding=fresh.binding,fixture='silence')
    with pytest.raises(Refused):runtime.operation('prepare',body)
    assert runtime.journal.value['records']==records
    runtime.operation('ack',dict(runtime_id=records[0]['id']))
    assert runtime.journal.value['records'][0]['id']==records[0]['id']
    assert runtime.operation('start',dict(runtime_id=records[0]['id']))['state']=='stopped-and-reaped'
    runtime.operation('prepare',body)
    assert len(runtime.journal.value['records'])==9


def test_unreaped_stop_becomes_unknown_without_release():
    import time
    from worker.runtime_supervisor import RuntimeSupervisor
    runtime=RuntimeSupervisor.__new__(RuntimeSupervisor)
    runtime.failed=False;runtime.active=dict(id='exact',stopping=time.monotonic()-2)
    assert runtime.state(dict(id='exact',phase='started'))=='unknown'
    assert runtime.active is not None


def test_watchdog_has_original_cutoff_and_no_request_io(monkeypatch):
    import threading
    import signal
    from worker.runtime_supervisor import RuntimeSupervisor
    runtime=RuntimeSupervisor.__new__(RuntimeSupervisor)
    runtime.tree_lock=threading.Lock();runtime.halt=False
    runtime.active=dict(id='exact',cutoff=10,stopping=None)
    signals=[];runtime.signal_namespace=signals.append
    clock=[10.0];monkeypatch.setattr('worker.runtime_supervisor.time.monotonic',lambda:clock[0])
    runtime.watchdog_tick();assert signals==[signal.SIGTERM]
    clock[0]=10.21;runtime.watchdog_tick();assert signals[-1]==signal.SIGKILL
    assert runtime.active['cutoff']==10
    with runtime.tree_lock:runtime.active=None
    before=list(signals);runtime.watchdog_tick();assert signals==before


def test_journal_semantic_bit_change_and_oversize_fail_closed(tmp_path):
    from worker.runtime_journal import MAX_BYTES
    write_journal(tmp_path,dict(version=1,lane='lane',records=[]))
    envelope=json.loads((tmp_path/'journal.json').read_bytes())
    envelope['payload']['lane']='changed'
    (tmp_path/'journal.json').write_text(json.dumps(envelope))
    with pytest.raises(ValueError):Journal(tmp_path)
    (tmp_path/'journal.json').write_bytes(b'x'*(MAX_BYTES+1))
    with pytest.raises(ValueError):Journal(tmp_path)
