"""Receipt ownership tests for the opt-in model process; no installed claim."""
import queue
import threading
import time
from types import SimpleNamespace

from worker.runtime_supervisor import RuntimeSupervisor
from worker.runtime_protocol import managed_qwen_cpu, resident_managed_qwen_cpu, resident_qwen_model
import worker.runtime_supervisor as module
import pytest


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
    supervisor.resident={'process':SimpleNamespace(pid=9000,returncode=None,poll=lambda:None)}
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
        if pid == -1 and calls.count(-1)==1:return (4242,0)
        if pid == -4242:raise ChildProcessError()
        return (0,0)
    monkeypatch.setattr(module.os,'waitpid',waitpid)
    supervisor.signal_namespace=lambda sig: (_ for _ in ()).throw(AssertionError('healthy_model_killed'))
    supervisor.tick()
    assert supervisor.active is None
    assert row['receipt']['timely'] is True and row['receipt']['reaped_children']==1
    assert row['proposal']=={'message':'fake-result'}
    assert supervisor.resident is not None
    assert -4242 in calls


def test_failed_attempt_never_publishes_and_invalidates_model(monkeypatch):
    supervisor,row,active=_lane()
    calls=[]
    signals=[]
    def waitpid(pid,options):
        calls.append(pid)
        if pid == -1 and calls.count(-1)==1:return (4242,0)
        if pid == -4242:raise ChildProcessError()
        if pid == -1:raise ChildProcessError()
        return (0,0)
    monkeypatch.setattr(module.os,'waitpid',waitpid)
    supervisor.signal_namespace=lambda sig:signals.append(sig)
    supervisor.tick()
    assert signals and active['stopping'] is not None
    assert row['phase']=='started' and 'proposal' not in row
    supervisor.tick()
    assert row['receipt']['timely'] is False and row['proposal'] is None
    assert supervisor.active is None
