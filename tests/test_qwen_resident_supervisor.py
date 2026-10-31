"""Resident attempt custody and receipt behavior; installed gates remain separate."""
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from worker.runtime_supervisor import RuntimeSupervisor
from worker.runtime_protocol import managed_qwen_cpu, resident_managed_qwen_cpu, resident_qwen_model
import worker.runtime_supervisor as module


@pytest.fixture(autouse=True)
def owned_diagnostic_consumer(monkeypatch):
    pending=queue.Queue(32)
    monkeypatch.setattr(module,'_diagnostic_queue',pending)
    monkeypatch.setattr(module,'_diagnostic_started',False)
    monkeypatch.setattr(module,'_diagnostic_thread',None)
    monkeypatch.setattr(module,'_diagnostic_events',0)
    yield
    consumer=module._diagnostic_thread
    if consumer is not None:
        pending.put(None,timeout=1)
        consumer.join(1)
        assert not consumer.is_alive()


def test_polled_resident_exit_survives_terminal_cleanup(monkeypatch):
    from worker.transport import Refused
    supervisor,row,active=_lane()
    process=supervisor.resident['process']
    def poll():
        process.returncode=-9
        return -9
    process.poll=poll
    supervisor.signal_owned=lambda *args:None
    with pytest.raises(Refused,match='resident_lost'):
        supervisor.ensure_resident(active['cutoff'])
    assert active['stop_reason']=='resident_exit'
    assert active['resident_exit_status']==-9
    active['leader_reaped']=True
    supervisor.owned_processes=lambda uid:{}
    monkeypatch.setattr(module.os,'waitpid',lambda *args:(0,0))
    observed=[]
    monkeypatch.setattr(module,'_resident_diagnostic',lambda *args:observed.append(args))
    supervisor.tick()
    assert observed==[('resident_exit',-9)]
    assert supervisor.resident is None


def test_managed_cpu_icl_requires_transcript_and_exact_clone_task():
    spec=SimpleNamespace(engine='qwen',device='cpu',task='clone',x_vector_only=False,
                         reference_text='trusted transcript')
    assert resident_managed_qwen_cpu(spec) and not managed_qwen_cpu(spec)
    spec.reference_text=None
    assert not resident_managed_qwen_cpu(spec)
    spec.x_vector_only=True
    assert resident_managed_qwen_cpu(spec) and managed_qwen_cpu(spec)
    spec.task='custom_voice'
    assert not resident_managed_qwen_cpu(spec)


def test_exact_service_identity_is_required(monkeypatch):
    from service import installed_qwen_daemon as qd
    from worker.transport import Refused
    identities={'clone':('Base','base-revision','base-digest'),
                'Ryan':('CustomVoice','custom-revision','custom-digest')}
    monkeypatch.setattr(qd,'service_model',lambda name:identities[name],raising=False)
    spec=SimpleNamespace(engine='qwen',device='cpu',task='custom_voice',preset='Ryan',
                         checkpoint='CustomVoice',revision='custom-revision',manifest_sha256='custom-digest')
    assert resident_qwen_model(spec)=='Ryan'
    spec.manifest_sha256='wrong'
    with pytest.raises(Refused,match='resident_model_identity'):resident_qwen_model(spec)


def _lane():
    supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.resident_qwen=True
    control=SimpleNamespace(close=lambda:None)
    supervisor.resident={'process':SimpleNamespace(pid=9000,returncode=None,poll=lambda:None),
                         'invalidating':False,'control':control}
    supervisor.failed=False
    supervisor.halt=False
    supervisor.tree_lock=threading.Lock()
    supervisor.received=queue.Queue()
    supervisor.conditioning=lambda row:None
    row={'id':'lane/boot/attempt','allocation':{'fixture':'qwen','binding':'binding',
         'snapshot':[{'deadline':'2026-10-29T12:00:00+00:00'},None,None,None]},
         'uid':20000,'phase':'started'}
    supervisor.lookup=lambda identity:row
    def change(original,**changes):
        original.update(changes)
        return original
    supervisor.change=change
    active=dict(id=row['id'],cutoff=time.monotonic()+10,started=True,stopping=None,
        proposal=None,proposed_at=None,leader=4242,process=SimpleNamespace(returncode=None),
        reaped_children=0,receiver_done=threading.Event())
    active['receiver_done'].set()
    supervisor.active=active
    return supervisor,row,active


def test_terminal_attempt_reaped_while_model_owner_remains(monkeypatch):
    supervisor,row,active=_lane()
    supervisor.received.put((row['id'],{'message':'fake-result'},time.monotonic()))
    calls=[]
    def waitpid(pid,options):
        calls.append(pid)
        return (4242,0) if len(calls)==1 else (0,0)
    monkeypatch.setattr(module.os,'waitpid',waitpid)
    supervisor.owned_processes=lambda uid: {} if uid==20000 else {9000:1}
    supervisor.signal_owned=lambda *args: (_ for _ in ()).throw(AssertionError('healthy_model_killed'))
    controls=[]
    supervisor.resident_control=lambda op, authority=None:controls.append(op)
    supervisor.tick()
    assert supervisor.active is None
    assert row['receipt']['timely'] is True and row['receipt']['reaped_children']==1
    assert row['proposal']=={'message':'fake-result'}
    assert supervisor.resident is not None
    assert controls==['revoke']


def test_detached_attempt_blocks_terminal_and_stops_only_owned_uids(monkeypatch):
    supervisor,row,active=_lane()
    supervisor.received.put((row['id'],{'message':'fake-result'},time.monotonic()))
    monkeypatch.setattr(module.os,'waitpid',lambda pid,flags:(4242,0) if not active.get('leader_reaped') else (0,0))
    # A /proc enumeration gap still cannot hide a populated attempt cgroup.
    group=SimpleNamespace(rmdir=lambda:None)
    active['cgroup']=group
    populated=[True]
    supervisor.cgroup_empty=lambda group:not populated[0]
    owned={20000:{},19999:{9000:456}}
    supervisor.owned_processes=lambda uid:owned[uid]
    signals=[]
    supervisor.signal_owned=lambda uid,sig:signals.append((uid,sig))
    supervisor.tick()
    assert row['phase']=='started' and supervisor.active is active
    assert active['stopping'] is not None
    assert signals==[(20000,signal.SIGTERM),(19999,signal.SIGTERM)]
    assert supervisor.resident['invalidating'] is True
    # The detached attempt and model owner are now both confirmed gone/reaped.
    populated[0]=False; owned[19999]={}
    monkeypatch.setattr(module.os,'waitpid',lambda pid,flags:(9000,0) if supervisor.resident['process'].returncode is None else (0,0))
    supervisor.tick()
    assert supervisor.active is None and supervisor.resident is None
    assert supervisor.failed is False
    assert row['phase']=='terminal' and row['proposal'] is None
    assert row['receipt']['timely'] is False


def test_resident_terminal_diagnostic_preserves_first_cause_and_exit(monkeypatch):
    supervisor,row,active=_lane()
    lines=[]
    monkeypatch.setattr(module,'_resident_diagnostic',lambda reason,status:lines.append((reason,status)))
    supervisor.signal_owned=lambda *args:None
    supervisor.stop_tree('no_proposal')
    supervisor.stop_tree('deadline')
    assert active['stop_reason']=='no_proposal'
    active['leader_reaped']=True
    active['resident_exit_status']=-9
    supervisor.resident=None
    supervisor.owned_processes=lambda uid:{}
    monkeypatch.setattr(module.os,'waitpid',lambda *args:(0,0))
    supervisor.tick()
    assert row['receipt']['timely'] is False
    assert lines==[('no_proposal',-9)]


def test_resident_diagnostic_fixed_vocabulary_and_sink_failure(monkeypatch):
    pending=queue.Queue(2)
    monkeypatch.setattr(module,'_diagnostic_queue',pending)
    monkeypatch.setattr(module,'_diagnostic_started',True)
    monkeypatch.setattr(module,'_diagnostic_events',0)
    module._resident_diagnostic('private request text',-9)
    assert pending.get_nowait()==b'lifecycle_resident_terminal:stop:signal_9\n'
    monkeypatch.setattr(pending,'put_nowait',lambda line:(_ for _ in ()).throw(OSError('sink failed')))
    module._resident_diagnostic('deadline',0)


@pytest.mark.parametrize('exit_code,expected',[(0,'no_proposal'),(2,'attempt_exit')])
def test_missing_proposal_identifies_observed_attempt_exit(monkeypatch,exit_code,expected):
    supervisor,row,active=_lane()
    lines=[]
    monkeypatch.setattr(module,'_resident_diagnostic',lambda reason,status:lines.append(reason))
    monkeypatch.setattr(module.os,'waitpid',lambda *args:(4242,exit_code << 8) if not active.get('leader_reaped') else (0,0))
    supervisor.owned_processes=lambda uid:{} if uid==20000 else {9000:1}
    supervisor.signal_owned=lambda *args:None
    supervisor.tick()
    assert active['stop_reason']==expected
    assert row['phase']=='started'
    supervisor.resident=None
    supervisor.tick()
    assert row['receipt']['timely'] is False
    assert lines==[expected]


def test_model_descendant_blocks_terminal_even_after_owner_reap(monkeypatch):
    supervisor,row,active=_lane()
    active.update(stopping=time.monotonic(),leader_reaped=True)
    supervisor.resident['invalidating']=True
    supervisor.resident['process'].returncode=0
    populated=[True]
    supervisor.resident['cgroup']=SimpleNamespace(rmdir=lambda:None)
    supervisor.cgroup_empty=lambda group: not populated[0] if group is supervisor.resident['cgroup'] else True
    supervisor.owned_processes=lambda uid:{}
    monkeypatch.setattr(module.os,'waitpid',lambda pid,flags:(0,0))
    supervisor.tick()
    assert supervisor.active is active and supervisor.resident is not None
    populated[0]=False
    supervisor.tick()
    assert supervisor.active is None and supervisor.resident is None


def test_prepared_cancellation_reaps_expected_owner_then_allows_new_attempt(monkeypatch):
    supervisor,row,active=_lane()
    row['phase']='prepared'; active['started']=False
    supervisor.owned_processes=lambda uid: {9000:1} if uid==19999 and supervisor.resident is not None else {}
    supervisor.signal_owned=lambda uid,sig:None
    supervisor.stop_tree()
    assert supervisor.resident['invalidating'] is True
    monkeypatch.setattr(module.os,'waitpid',lambda pid,flags:(9000,0) if supervisor.resident['process'].returncode is None else (0,0))
    supervisor.owned_processes=lambda uid:{}
    supervisor.tick()
    assert supervisor.active is None and supervisor.resident is None and not supervisor.failed
    assert row['phase']=='terminal' and row['receipt']['reaped'] is True


def test_real_setsid_descendant_outside_leader_group():
    # Same UID, separate session and process group. No attempt to signal other processes.
    script='''import os,sys,time
pid=os.fork()
if pid:
    print(pid,flush=True); os._exit(0)
os.setsid()
time.sleep(10)
'''
    proc=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True)
    child=int(proc.stdout.readline().strip()); proc.wait(timeout=2)
    try:
        assert os.getpgid(child)!=os.getpgid(os.getpid())
        if sys.platform=='linux' and os.path.exists('/proc'):
            end=time.monotonic()+2
            while time.monotonic()<end:
                found=RuntimeSupervisor.owned_processes(os.getuid())
                if child in found:break
                time.sleep(.01)
            assert child in found
    finally:
        os.kill(child,signal.SIGKILL)


def test_cgroup_populated_is_required_for_terminal(tmp_path):
    from worker.transport import Refused
    events=tmp_path/'cgroup.events'
    events.write_text('populated 1\nfrozen 0\n')
    assert RuntimeSupervisor.cgroup_empty(tmp_path) is False
    events.write_text('populated 0\nfrozen 0\n')
    assert RuntimeSupervisor.cgroup_empty(tmp_path) is True
    events.write_text('frozen 0\n')
    with pytest.raises(Refused,match='resident_cgroup_unknown'):
        RuntimeSupervisor.cgroup_empty(tmp_path)


def test_aggregate_parent_limits_cover_worker_resident_and_attempt(tmp_path):
    from worker.transport import Refused
    system=tmp_path/'sys';system.mkdir()
    (system/'cgroup.controllers').write_text('cpu memory pids\n')
    parent=system/'speech-box-qwen'/('a'*32);parent.mkdir(parents=True)
    worker=parent/('b'*64);worker.mkdir()
    (worker/'cpu.max').write_text('200000 100000\n')
    for name,value in (('memory.max',str(12*1024**3)),('memory.swap.max','0'),
                       ('pids.max','64'),('cpu.max','200000 100000')):
        (parent/name).write_text(value)
    member=tmp_path/'membership';member.write_text('0::/speech-box-qwen/'+('a'*32)+'/'+('b'*64)+'\n')
    supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.resident_budget={'ram_bytes':12*1024**3,'pids':64}
    def verify(root=parent):
        supervisor.verify_aggregate_custody(root=root,system=system,membership_file=member)
    verify()
    # Attempt and resident are siblings of the worker, but descendants of
    # this finite parent. Docker's worker-only limit cannot stand in for it.
    for name,value in (('memory.max','max'),('memory.swap.max','max'),('pids.max','max'),
                       ('cpu.max','max 100000'),('memory.max',str(13*1024**3)),
                       ('pids.max','65'),('cpu.max','300000 100000')):
        original=(parent/name).read_text()
        (parent/name).write_text(value)
        with pytest.raises(Refused,match='resident_aggregate_cgroup_limits_required'):verify()
        (parent/name).write_text(original)
    # A lookalike bind from another installation must not pass.
    impostor=tmp_path/'impostor';impostor.mkdir()
    with pytest.raises(Refused,match='resident_aggregate_cgroup_limits_required'):verify(impostor)
    member.write_text('0::/speech-box-qwen/'+('c'*32)+'/'+('b'*64))
    with pytest.raises(Refused,match='resident_aggregate_cgroup_limits_required'):verify()


def test_pipe_gate_requires_migrated_child_membership(tmp_path):
    from worker.transport import Refused
    proc=tmp_path/'proc';(proc/'self').mkdir(parents=True);(proc/'123').mkdir()
    parent='0::/speech-box-qwen/'+('a'*32)
    (proc/'self'/'cgroup').write_text(parent+'/'+('b'*64))
    child=proc/'123'/'cgroup'
    group=tmp_path/'attempt-123'
    child.write_text(parent+'/'+group.name)
    RuntimeSupervisor.verify_child_cgroup(123,group,proc=proc)
    for escaped in (parent+'/'+('b'*64), '0::/other/'+group.name, parent+'/other'):
        child.write_text(escaped)
        with pytest.raises(Refused,match='resident_cgroup_assignment_required'):
            RuntimeSupervisor.verify_child_cgroup(123,group,proc=proc)
