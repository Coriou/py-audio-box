"""E1 deterministic policy/SQLite proofs; fake receipts are not installed E2."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import wave

import pytest
from service.dispatcher import Dispatcher
from service.worker_transport import TransportObservation
from worker.lifecycle import Observation, State


def wait(d, predicate, timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError(('timeout',d.call('status')))


def ready(d):
    return wait(d,lambda:d.call('status')['reconciled'])


class Runtime:
    def __init__(self,staging):
        self.staging=staging;self.prepares=0;self.starts=0;self.acks=0
        self.state=State.ALIVE;self.authority=None;self.block=threading.Event();self.ack_fail=False
        self.timely=True;self.corrupt=False
    def call(self,op,body):
        if op=='hello':return dict(quarantined=False)
        assert op=='ack';self.acks+=1
        if self.ack_fail:self.ack_fail=False;raise OSError('lost_ack_response')
        return dict(acked=True)
    def prepare(self,a,fixture):
        self.prepares+=1;self.authority=a;return 'lane/boot/allocation-'+str(self.prepares)
    def start(self,identity):
        self.starts+=1;return dict(state='alive')
    def observe(self,identity,timeout):
        self.block.wait(.2)
        return Observation(identity,self.state)
    def terminate(self,identity,timeout):self.state=State.STOPPED
    def collect(self,a,identity):
        receipt=dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=0,timely=self.timely,original_deadline=a.job['deadline'])
        if not self.timely:return receipt,None
        stage=self.staging/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770,exist_ok=True);stage.chmod(0o770)
        buf=io.BytesIO()
        with wave.open(buf,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(b'\0'*9600)
        data=buf.getvalue();(stage/'output.wav').write_bytes(data)
        digest=hashlib.sha256(data).hexdigest()
        output=dict(id='asset-test-'+str(self.prepares),sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[]))
        message=a.message();message.update(kind='result',stage='completed',result_asset_ids=[output['id']])
        manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation='speech_generate',artifact_id='artifact-test-'+str(self.prepares),outputs=[output],candidates=[])
        if self.corrupt:message['generation']+=1
        return receipt,TransportObservation('proposal','E1',message=message,manifest=manifest)


@pytest.fixture
def pair(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=runtime);ready(d)
    yield d,runtime
    d.close()


def running(d,r,key='a'):
    job=d.call('submit','p',key)
    wait(d,lambda:r.starts)
    return job


@pytest.mark.parametrize('stop_obligation',[False,True])
def test_unknown_prepare_retains_custody_without_busy_poll(tmp_path, monkeypatch, capsys, stop_obligation):
    stage=tmp_path/'stage';stage.mkdir()
    class FailedPrepare(Runtime):
        def prepare(self, authority, fixture):
            self.prepares+=1
            raise RuntimeError('private-media-token')
    runtime=FailedPrepare(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=runtime)
    try:
        ready(d)
        monkeypatch.setattr(d,'_qwen_state_poll',lambda:True)
        job=d.call('submit','p','failed-prepare')
        wait(d,lambda:d.call('status')['phase']=='unknown')
        deadline=d.call('job',job['id'])['deadline']
        d.stop_obligation=stop_obligation
        count=0
        original=d.store.current
        def counted():
            nonlocal count
            count+=1
            return original()
        monkeypatch.setattr(d.store,'current',counted)
        start=time.monotonic()
        for _ in range(5):
            current=d.call('job',job['id'])
            assert current['state']=='running' and current['deadline']==deadline
            assert d.call('status')['quarantined']
        assert time.monotonic()-start<.5
        time.sleep(.08)
        assert count<8
        status=d.call('status')
        assert status['phase']=='unknown' and status['runtime_id'] is None
        assert status['reservations'][0]['state']=='executing'
        assert status['stop_obligation'] is stop_obligation
        assert runtime.prepares==1 and runtime.starts==0
        trace=capsys.readouterr().err
        assert 'qwen_dispatch_refused:' in trace
        assert 'private-media-token' not in trace
    finally:
        d.close()


def test_qwen_diagnostic_write_failure_cannot_skip_unknown_quarantine(tmp_path, monkeypatch):
    from service import dispatcher as module
    stage=tmp_path/'stage';stage.mkdir()
    class FailedPrepare(Runtime):
        def prepare(self, authority, fixture):
            raise RuntimeError('prepare_failed')
    runtime=FailedPrepare(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=runtime)
    try:
        ready(d)
        monkeypatch.setattr(d,'_qwen_state_poll',lambda:True)
        def failed_print(*args,**kwargs):
            raise OSError('diagnostic_sink_failed')
        monkeypatch.setattr(module,'print',failed_print,raising=False)
        job=d.call('submit','p','diagnostic-failure')
        wait(d,lambda:d.call('status')['phase']=='unknown')
        status=d.call('status')
        assert status['quarantined'] and status['runtime_id'] is None
        assert status['reservations'][0]['state']=='executing'
        assert d.call('job',job['id'])['state']=='running'
        assert runtime.starts==0
    finally:
        d.close()


def test_serialized_responsive_cancel_unknown_and_retry(pair):
    d,r=pair;job=running(d,r)
    t=time.monotonic();assert d.call('job',job['id'])['state']=='running'
    # Configure the stop reply before cancellation can launch its pending RPC.
    # Default terminate overwrites UNKNOWN with STOPPED; that forced fixture
    # race says nothing about an ambiguous supervisor observation.
    r.terminate=lambda identity,timeout:setattr(r,'state',State.UNKNOWN)
    assert d.call('cancel',job['id'])['state']=='cancel_requested'
    assert time.monotonic()-t<.5
    wait(d,lambda:d.call('status')['quarantined'])
    assert d.call('job',job['id'])['state']=='cancel_requested'
    assert d.call('submit','p','a')['id']==job['id']
    with pytest.raises(ValueError):d.call('replace',job['id'])
    assert r.prepares==r.starts==1 and r.acks==0
    r.timely=False;r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='cancelled' else None))
    assert result['attempts'][-1]['reaped']
    wait(d,lambda:d.call('status')['phase']=='idle')
    retry=d.call('submit','p','retry',retry_of=job['id'])
    assert retry['id']!=job['id'] and retry['input_sha256']==job['input_sha256']


def test_timely_completion_wins_cancel_and_lost_ack(pair):
    d,r=pair;job=running(d,r);r.ack_fail=True
    d.call('cancel',job['id']);r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='succeeded' else None))
    wait(d,lambda:d.call('status')['phase']=='idle')
    assert r.acks==2 and r.starts==1
    assert d.call('artifact',result['artifact_ids'][0])['input_sha256']==job['input_sha256']
    assert d.call('submit','p','a')['id']==job['id']


@pytest.mark.parametrize('timely,corrupt',[(False,False),(True,True)])
def test_late_or_stale_proposal_cannot_publish(pair,timely,corrupt):
    d,r=pair;job=running(d,r);r.timely=timely;r.corrupt=corrupt;r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='failed' else None))
    assert result['artifact_ids']==[] and result['attempts'][-1]['reaped']


def test_replacement_only_confirmed_terminal_increments_generation(pair):
    d,r=pair;job=running(d,r);r.timely=False;r.state=State.STOPPED
    wait(d,lambda:d.call('job',job['id'])['state']=='failed')
    wait(d,lambda:d.call('status')['phase']=='idle')
    old=d.call('job',job['id']);r.state=State.ALIVE
    new=d.call('replace',job['id'])
    assert new['generation']==2 and new['input_sha256']==old['input_sha256']
    for key in ('id','owner_id','staging_id'):assert new['attempts'][0][key]!=new['attempts'][1][key]
    assert new['attempts'][0]['reaped'] and not new['attempts'][1]['reaped']


@pytest.mark.parametrize('point',['prespawn','before_prepare','after_prepare','preregistration','prestart','before_start','after_start','postspawn','observation_reap','preverification','before_ack','after_ack','durable_outcome'])
def test_interruption_reconnect_no_duplicate_execution(tmp_path,point):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage);hit=threading.Event();release=threading.Event()
    def fault(where):
        if where==point and not hit.is_set():
            hit.set();release.wait(3);raise RuntimeError('E1_control_interruption')
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    job=d.call('submit','p','a')
    if point in ('observation_reap','preverification','before_ack','after_ack','durable_outcome'):
        wait(d,lambda:r.starts);r.state=State.STOPPED
    assert hit.wait(4);release.set();d.close()
    before=r.starts
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r)
    try:
        ready(d);time.sleep(.3)
        assert r.starts==before
        stored=d.call('job',job['id'])
        assert len(stored['attempts'])==1 and stored['input_sha256']==job['input_sha256']
        if point in ('prespawn','before_prepare','after_prepare','preregistration'):
            assert d.call('status')['quarantined'] and not stored['artifact_ids']
    finally:d.close()


def test_unjournaled_foundation_attempt_cannot_admit(tmp_path):
    from service.core import Core
    root=tmp_path/'core';core=Core(root)
    fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
    job=core.submit_synthetic('p','a',fixture['context'])
    with core.transaction():
        job['state']='cancel_requested';core._save(job)
    core.close()
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage)
    d=Dispatcher(root,staging=stage,runtime=r)
    try:
        ready(d);assert d.call('status')['quarantined']
        with pytest.raises(ValueError):d.call('submit','p','new')
        assert r.starts==0
    finally:d.close()


def test_admission_settings_and_job_are_one_transaction(tmp_path):
    from service.core import Core
    from service.dispatch_store import DispatchStore
    root=tmp_path/'core';core=Core(root);DispatchStore(core)
    context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    # A real SQLite failure after job INSERT rolls admission back as a whole.
    core.db.execute("CREATE TRIGGER fail_settings BEFORE INSERT ON dispatch_settings BEGIN SELECT RAISE(ABORT,'settings_failure'); END")
    import sqlite3
    try:
        with pytest.raises(sqlite3.IntegrityError):core.submit_synthetic('p','a',context,_dispatch_settings=('silence',8))
        assert core.jobs('p')==[]
        assert core.db.execute('SELECT * FROM events').fetchall()==[]
        core.db.execute('DROP TRIGGER fail_settings')
        job=core.submit_synthetic('p','a',context,_dispatch_settings=('busy',8))
        assert core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(job['id'],)).fetchone()==('busy',8)
    finally:core.close()


def test_queued_cancel_terminal_and_request_are_atomic(tmp_path):
    from service.core import Core
    import sqlite3
    core=Core(tmp_path/'core')
    context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    job=core.submit_synthetic('p','a',context)
    core.db.execute("CREATE TRIGGER fail_cancel BEFORE INSERT ON events WHEN NEW.state='cancelled' BEGIN SELECT RAISE(ABORT,'cancel_commit_failure'); END")
    try:
        with pytest.raises(sqlite3.IntegrityError):core.request_cancel(job['id'],_cancel_unstarted=True)
        assert core.job(job['id'])['state']=='queued'
        assert [e['state'] for e in core.events(job['id'])]==['queued']
        core.db.execute('DROP TRIGGER fail_cancel')
        assert core.request_cancel(job['id'],_cancel_unstarted=True)['state']=='cancelled'
        assert [e['state'] for e in core.events(job['id'])]==['queued','cancel_requested','cancelled']
    finally:core.close()


@pytest.mark.parametrize('damage', ['missing','corrupt'])
def test_startup_integrity_preserves_success_and_strict_reads(tmp_path, damage):
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage);root=tmp_path/'core'
    d=Dispatcher(root,staging=stage,runtime=runtime);ready(d)
    job=running(d,runtime);runtime.state=State.STOPPED
    wait(d,lambda:d.call('status')['phase']=='idle')
    terminal=d.call('job',job['id']);events=d.call('events',job['id'])
    artifact=d.call('artifact',terminal['artifact_ids'][0]);asset_id=artifact['asset_ids'][0]
    d.close()
    media=root/'assets'/asset_id
    if damage=='missing':media.unlink()
    else:media.write_bytes(b'changed')
    for _ in range(2):
        d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            ready(d);status=d.call('status')
            assert status['integrity_fault'] and status['integrity']['complete']
            assert status['integrity']['counts']['missing_hash_mismatched_terminal']==1
            assert d.call('job',job['id'])==terminal and d.call('events',job['id'])==events
            assert d.call('submit','p','a')==terminal
            with pytest.raises(ValueError):d.call('submit','p','new')
            with pytest.raises(Exception):d.call('asset',asset_id)
            with pytest.raises(Exception):d.call('artifact',artifact['id'])
            assert runtime.starts==1 and runtime.acks==1
        finally:d.close()


@pytest.mark.parametrize('damage', ['receipt','reservation','unbound','attempt','terminal_no_receipt','demand','state'])
def test_startup_join_refuses_ack_release_and_launch(tmp_path, damage):
    import sqlite3
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage);root=tmp_path/'core'
    def fault(point):
        if point=='durable_outcome':raise OSError('retain outcome before ack')
    d=Dispatcher(root,staging=stage,runtime=runtime,fault=fault);ready(d)
    job=running(d,runtime);runtime.state=State.STOPPED
    wait(d,lambda:d.call('job',job['id'])['state']=='succeeded')
    d.close()
    with sqlite3.connect(root/'core.sqlite') as db:
        if damage=='receipt':
            original_receipt=db.execute('SELECT receipt FROM dispatches').fetchone()[0]
            receipt=json.loads(original_receipt);receipt['binding']='wrong'
            db.execute('UPDATE dispatches SET receipt=?',(json.dumps(receipt),))
        elif damage=='reservation':db.execute('DELETE FROM resource_reservations')
        elif damage=='state':db.execute("UPDATE resource_reservations SET state='executing'")
        elif damage=='demand':
            demand=json.loads(db.execute('SELECT demand FROM resource_reservations').fetchone()[0]);demand['configuration']='wrong'
            db.execute('UPDATE resource_reservations SET demand=?',(json.dumps(demand),))
        elif damage=='unbound':db.execute('UPDATE dispatches SET runtime=NULL')
        elif damage=='terminal_no_receipt':db.execute('UPDATE dispatches SET receipt=NULL')
        else:
            stored=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0]);stored['attempts'][-1]['owner_id']='owner-wrong'
            db.execute('UPDATE jobs SET job=?',(json.dumps(stored),))
    for _ in range(2):
        d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            status=d.call('status');assert status['integrity_fault']
            if damage == 'attempt':
                assert 'public_artifact_mismatch' in status['integrity']['reasons']
                assert status['integrity']['defect_latched']
            else:
                assert status['integrity']['counts'].get('unresolved_owner_receipt_reservation',0)>=1, status['integrity']
            assert runtime.acks==0 and runtime.starts==1
            assert d.call('job',job['id'])['state']=='succeeded'
            with pytest.raises(ValueError):d.call('submit','p','fresh')
        finally:d.close()
    if damage == 'receipt':
        # Restoring bytes cannot silently acknowledge an already observed defect.
        with sqlite3.connect(root/'core.sqlite') as db:
            db.execute('UPDATE dispatches SET receipt=?',(original_receipt,))
        d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            status=d.call('status')
            assert status['integrity_fault'] and status['integrity']['defect_latched']
            assert runtime.acks==0 and runtime.starts==1
        finally:d.close()


def test_startup_normal_exact_receipt_two_restarts(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage);root=tmp_path/'core'
    def fault(point):
        if point=='durable_outcome':raise OSError('crash before ack')
    d=Dispatcher(root,staging=stage,runtime=runtime,fault=fault);ready(d)
    job=running(d,runtime);runtime.state=State.STOPPED
    wait(d,lambda:d.call('job',job['id'])['state']=='succeeded');d.close()
    for _ in range(2):
        d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            wait(d,lambda:d.call('status')['phase']=='idle')
            assert not d.call('status')['integrity_fault']
            assert d.call('status')['reservations'][0]['state']=='retained'
            assert len(d.call('events',job['id']))==3
            assert runtime.starts==runtime.acks==1
        finally:d.close()


def test_live_inventory_failure_preserves_responsive_stop(tmp_path, monkeypatch):
    from service.reconciliation import Reconciliation, Incomplete
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage)
    stops=[]
    terminate=runtime.terminate
    def stop(*args):stops.append(args);return terminate(*args)
    runtime.terminate=stop
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=runtime);ready(d)
    try:
        job=running(d,runtime)
        scan=d.call('status')['integrity']['scan_id']
        def denied(self):raise Incomplete('inventory_budget')
        monkeypatch.setattr(Reconciliation,'_collect',denied)
        with pytest.raises(ValueError,match='stale'):d.call('inventory_page',scan)
        assert d.call('job',job['id'])['id']==job['id']
        wait(d,lambda:stops)
        assert len(stops)==1 and runtime.acks==0
        with pytest.raises(ValueError):d.call('submit','p','fresh')
    finally:d.close()


def test_startup_conflicting_original_receipt_retained(tmp_path):
    import sqlite3
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage);root=tmp_path/'core'
    def fault(point):
        if point=='preverification':raise OSError('stop before publication')
    d=Dispatcher(root,staging=stage,runtime=runtime,fault=fault);ready(d)
    job=running(d,runtime);runtime.state=State.STOPPED
    wait(d,lambda:d.call('status')['phase']=='receipt');d.close()
    with sqlite3.connect(root/'core.sqlite') as db:
        bad=json.loads(db.execute('SELECT receipt FROM dispatches').fetchone()[0]);bad['binding']='conflicting-original'
        original=json.dumps(bad);db.execute('UPDATE dispatches SET receipt=?',(original,))
    collects=[];collect=runtime.collect
    def collected(*args):collects.append(True);return collect(*args)
    runtime.collect=collected
    for _ in range(2):
        collects.clear();d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            wait(d,lambda:collects)
            assert d.call('status')['integrity_fault'] and runtime.acks==0
            assert d.call('job',job['id'])['state']=='running'
            with sqlite3.connect(root/'core.sqlite') as db:
                assert db.execute('SELECT receipt FROM dispatches').fetchone()[0]==original
                assert db.execute('SELECT state FROM resource_reservations').fetchone()[0]=='executing'
        finally:d.close()


@pytest.mark.parametrize('point', ['fault_denial', 'completion_denial', 'healthy_death'])
def test_integrity_open_epoch_valid_receipt_two_restarts(tmp_path, point):
    import subprocess
    import sys
    import sqlite3
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage);root=tmp_path/'core'
    d=Dispatcher(root,staging=stage,runtime=runtime);ready(d)
    job=running(d,runtime);d.close()
    script=r'''
import os,sys,sqlite3
from service.core import Core
from service.reconciliation import Reconciliation
c=Core(sys.argv[1]);i=Reconciliation(c,sys.argv[2]);point=sys.argv[3]
if point=='fault_denial': (c.root/'assets'/'asset-owned-test-orphan').write_bytes(b'x')
original=i._collect
def barrier():
    with sqlite3.connect(c.root/'core.sqlite') as reader:
        assert reader.execute('SELECT state FROM p1_integrity').fetchone()[0]=='checking'
    if point=='healthy_death': os._exit(23)
    result=original()
    if point=='fault_denial': assert 'no_publication_provenance' in i.reasons
    c.db.execute('PRAGMA query_only=ON')
    return result
i._collect=barrier
assert i.reconcile_startup()['blocked']
os._exit(23)
'''
    child=subprocess.run([sys.executable,'-c',script,str(root),str(stage),point], timeout=10,
                         env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==23
    (root/'assets'/'asset-owned-test-orphan').unlink(missing_ok=True)
    runtime.state=State.STOPPED
    for _ in range(2):
        d=Dispatcher(root,staging=stage,runtime=runtime)
        try:
            assert d.call('status')['integrity_fault']
            wait(d,lambda:d.call('status')['phase']=='receipt')
            assert d.call('job',job['id'])['state'] != 'succeeded'
            assert runtime.acks==0 and runtime.starts==1
            assert d.call('status')['reservations'][0]['state']=='executing'
            with pytest.raises(ValueError):d.call('submit','p','fresh')
            with sqlite3.connect(root/'core.sqlite') as db:
                receipt=json.loads(db.execute('SELECT receipt FROM dispatches').fetchone()[0])
                assert receipt['reaped'] and receipt['binding']==runtime.authority.binding
                assert db.execute('SELECT count(*) FROM publications').fetchone()[0]==0
        finally:d.close()


# Coherent enrolled B2 fixtures are shared test data, never runtime entrypoints.
from tests.service.test_recovery import installation, b2_intent, b2_receipt, registration, logical


@pytest.mark.parametrize('changed',['none','boot','certificate','unavailable'])
def test_b2_explicit_startup_rechecks_hello_before_admission(installation,monkeypatch,changed):
    from service import recovery
    from service.admission import DEFAULT_BUDGET
    import sqlite3
    roots,scope,profiles,_=installation
    intent=b2_intent(installation,complete=True)
    fence=b2_receipt(roots,intent,complete=True)
    record=registration(installation,'actual-host-inspection')
    rid=recovery.register(roots,scope,record)
    hello=dict(lane=record['lane'],boot=record['boot'],quarantined=False,maintenance=fence['certificate'])
    class Hello:
        calls=0
        def call(self,op,body):
            assert op=='hello';self.calls+=1
            if self.calls==2:
                if changed=='unavailable':raise OSError('E1 unavailable hello')
                if changed=='boot':return {**hello,'boot':'f'*32}
                if changed=='certificate':return {**hello,'maintenance':{**hello['maintenance'],'certificate_digest':'f'*64}}
            return hello
    runtime=Hello();before=logical(roots)
    if changed!='none':
        monkeypatch.setattr('service.dispatcher.Admission',lambda *a,**k:pytest.fail('Admission before repeated hello'))
        with pytest.raises((ValueError,OSError)):
            Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=runtime,
                persistent_registration=rid,profiles=profiles,budget=DEFAULT_BUDGET)
        assert logical(roots)==before
    else:
        dispatcher=Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=runtime,
            persistent_registration=rid,profiles=profiles,budget=DEFAULT_BUDGET)
        try:
            assert runtime.calls==2
            assert dispatcher.call('status')['persistent_activation'] is False
            with pytest.raises(ValueError,match='persistent_activation_closed'):dispatcher.call('submit','p','closed')
        finally:dispatcher.close()


@pytest.mark.parametrize('cleanup', ['core', 'pool'])
def test_lifetime_failed_startup_publication_waits_for_cleanup(tmp_path, monkeypatch, cleanup):
    from concurrent.futures import Future
    from service import dispatcher as module
    from service import recovery
    entered = threading.Event(); release = threading.Event(); events = []
    original = ValueError('startup refusal')
    class DB:
        def execute(self, *args): return self
        def fetchone(self): return None
    class Core:
        db = DB()
        def __init__(self, root, **kwargs): events.append('open')
        def close(self):
            if cleanup == 'core': entered.set(); assert release.wait(2)
            events.append('closed')
    class Pool:
        def shutdown(self, **kwargs): entered.set(); assert release.wait(2); events.append('shutdown')
    def admission(*args, **kwargs): raise original
    monkeypatch.setattr(recovery, 'startup_read', lambda _:None)
    monkeypatch.setattr(module, 'Core', Core)
    monkeypatch.setattr(module, 'Admission', admission)
    value = Dispatcher.__new__(Dispatcher)
    value.installed_activation = None; value.installed_checked = False
    value.runtime = object(); value.production_contexts = None
    value.root = tmp_path; value.persistent_registration = None
    value.staging = tmp_path; value.budget = None; value.initialized = Future()
    if cleanup == 'pool': value.pool = Pool()
    thread = threading.Thread(target=value._run)
    thread.start()
    try:
        assert entered.wait(2)
        assert not value.initialized.done(), 'failure observable while owned cleanup pending'
    finally:
        release.set(); thread.join(2)
    assert not thread.is_alive()
    with pytest.raises(ValueError) as caught: value.initialized.result(timeout=.1)
    assert caught.value is original and events[-1] == 'closed'
    assert events == (['open', 'shutdown', 'closed'] if cleanup == 'pool' else ['open', 'closed'])


def test_lifetime_real_core_refusal_has_no_next_startup_overlap(tmp_path, monkeypatch):
    from service import dispatcher as module
    from service.core import Core
    entered = threading.Event(); release = threading.Event(); closed = threading.Event()
    primary = ValueError('admission refusal')
    close = Core.close
    def paused(core):
        entered.set()
        assert release.wait(2)
        close(core); closed.set()
    def refuse(*args, **kwargs): raise primary
    monkeypatch.setattr(module.Core, 'close', paused)
    monkeypatch.setattr(module, 'Admission', refuse)
    failures = []; done = threading.Event()
    def caller():
        try: Dispatcher(tmp_path/'core')
        except BaseException as exc:
            failures.append(exc)
            assert closed.is_set()
        finally: done.set()
    thread = threading.Thread(target=caller); thread.start()
    try:
        assert entered.wait(2)
        assert not done.wait(.05)
    finally:
        release.set(); thread.join(2)
    assert not thread.is_alive() and failures == [primary]
    # The caller's next owner can acquire the actual SQLite/control-lock pair.
    monkeypatch.setattr(module.Core, 'close', close)
    core = Core(tmp_path/'core'); core.close()


@pytest.mark.parametrize('secondary', [False, True])
def test_lifetime_actual_timeout_excludes_next_reader(tmp_path, monkeypatch, secondary):
    import subprocess
    import sqlite3
    import sys
    import time
    from service import dispatcher as module, recovery
    from service.core import Core
    entered = threading.Event(); release = threading.Event()
    owned = []; cores = []; reads = []; closed_checks = []
    original_close = Core.close
    original_read = recovery.startup_read
    primary = ValueError('admission refusal')
    def paused(core):
        owned.append(threading.current_thread()); cores.append(core)
        entered.set()
        assert release.wait(10), 'test cleanup barrier deadline'
        original_close(core)
        try: core.db.execute('SELECT 1')
        except sqlite3.ProgrammingError as exc: closed_checks.append('closed database' in str(exc))
        if secondary: raise RuntimeError('secondary after real close')
    def read(root):
        reads.append(root)
        return original_read(root)
    def refuse(*args, **kwargs): raise primary
    monkeypatch.setattr(Core, 'close', paused)
    monkeypatch.setattr(module, 'Admission', refuse)
    monkeypatch.setattr(recovery, 'startup_read', read)
    root = tmp_path/'core'
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError): Dispatcher(root)
        assert 4.8 <= time.monotonic()-start < 8
        assert entered.is_set()
        before = len(reads)
        with pytest.raises(ValueError, match='^control_busy$'): Dispatcher(root)
        assert len(reads) == before
        # A fresh harmless interpreter uses the identical read-side lock, with
        # no SQLite open, inheritance of handles, model or maintenance operation.
        code = """import sys
from service.recovery import _startup_read_lock
try:
    with _startup_read_lock(sys.argv[1]): raise AssertionError('owner bypassed')
except ValueError as exc:
    assert str(exc) == 'control_busy'
"""
        subprocess.run([sys.executable, '-c', code, str(root)], check=True, timeout=3)
    finally:
        release.set()
        for thread in owned: thread.join(3)
    assert owned and all(not t.is_alive() for t in owned)
    assert closed_checks == [True]
    assert cores[0].owner.closed
    if secondary: assert any('RuntimeError' in n for n in primary.__notes__)
    monkeypatch.setattr(Core, 'close', original_close)
    core = Core(root)
    core.close()


def test_lifetime_reader_core_gap_competing_owner_refuses_before_admission(tmp_path, monkeypatch):
    from service import dispatcher as module
    from service.core import Core
    root = tmp_path/'core'
    core = Core(root); core.close()
    owned = []
    def competed(path, **kwargs):
        owner = Core(path)
        owned.append(owner)
        try: return Core(path, **kwargs)
        finally: owner.close()
    monkeypatch.setattr(module, 'Core', competed)
    monkeypatch.setattr(module, 'Admission', lambda *a:pytest.fail('Admission after lock refusal'))
    with pytest.raises(ValueError, match='^control_busy$'): Dispatcher(root)
    assert owned and owned[0].owner.closed
    core = Core(root); core.close()


def test_lifetime_reader_core_gap_new_enrollment_refuses_before_admission(installation, monkeypatch):
    from service import recovery
    roots, scope, profiles, _ = installation
    before = logical(roots)
    # Model an unenrolled observation followed by a completed competing owner
    # before Core acquisition. No registration/hello from that owner is borrowed.
    monkeypatch.setattr(recovery, 'startup_read', lambda _:None)
    monkeypatch.setattr('service.dispatcher.Admission', lambda *a:pytest.fail('Admission on changed enrollment'))
    with pytest.raises(ValueError, match='startup_deployment_changed'):
        Dispatcher(roots['control']/'core', staging=roots['staging'])
    assert logical(roots) == before


@pytest.mark.parametrize('startup', [False, True])
@pytest.mark.parametrize('faults', ['pool', 'core', 'both'])
def test_bounded_core_dispatcher_cleanup_outcome(tmp_path, monkeypatch, startup, faults):
    from concurrent.futures import Future
    from service import dispatcher as module
    from service import recovery
    from contextlib import nullcontext
    events = []; primary = ValueError('original startup')
    class DB:
        def execute(self, *a): return self
        def fetchone(self): return None
    class Core:
        db = DB()
        def __init__(self, root, **kwargs): pass
        def close(self):
            events.append('core')
            if faults in ('core', 'both'): raise RuntimeError('owned Core fault')
    class Pool:
        def shutdown(self, **kw):
            events.append('pool')
            if faults in ('pool', 'both'): raise RuntimeError('owned pool fault')
    def admission(*a, **kwargs): raise primary
    monkeypatch.setattr(module, 'Core', Core)
    monkeypatch.setattr(recovery, '_startup_read_lock', lambda root: nullcontext(False))
    monkeypatch.setattr(module, 'Admission', admission)
    value = Dispatcher.__new__(Dispatcher)
    value.installed_activation = None; value.installed_checked = False
    value.runtime = object(); value.production_contexts = None
    value.root = tmp_path; value.staging = tmp_path; value.budget = None
    value.persistent_registration = None; value.pool = Pool(); value._cleanup_failed = False
    value.initialized = Future()
    if not startup: value.initialized.set_result(True)
    value.thread = threading.Thread(target=value._run)
    value.thread.start(); value.thread.join(3)
    assert not value.thread.is_alive() and events == ['pool', 'core']
    assert value._cleanup_failed is True
    # An already-dead writer must refuse before trying any queue/Future wait.
    value.call = lambda *a: (_ for _ in ()).throw(AssertionError('dead writer enqueue'))
    with pytest.raises(RuntimeError, match='control_cleanup_failed'): value.close()
    if startup:
        with pytest.raises(ValueError) as caught: value.initialized.result(.1)
        assert caught.value is primary
        assert len(primary.__notes__) == (2 if faults == 'both' else 1)


def test_bounded_core_dispatcher_normal_stop_join_outcome(tmp_path, monkeypatch):
    from service import dispatcher as module
    close = module.Core.close
    events = []
    def failed(core):
        close(core); events.append('closed')
        raise RuntimeError('after actual successful Core close')
    monkeypatch.setattr(module.Core, 'close', failed)
    d = Dispatcher(tmp_path/'core', staging=tmp_path, runtime=Runtime(tmp_path))
    with pytest.raises(RuntimeError, match='control_cleanup_failed'): d.close()
    assert not d.thread.is_alive() and events == ['closed']
    with pytest.raises(RuntimeError, match='control_cleanup_failed'): d.close()


def test_bounded_core_dead_writer_isolated_process(tmp_path):
    import subprocess, sys
    script = r'''
import sys, threading, sqlite3
from pathlib import Path
from service import core as module, recovery, dispatcher
root = Path(sys.argv[1])
connect = sqlite3.connect
class DB(sqlite3.Connection):
    def close(self): raise RuntimeError('owned before-close fault')
module.sqlite3.connect = lambda *a, **k: connect(*a, **k) if a and a[0] == ':memory:' else connect(*a, factory=DB, **k)
# The real Dispatcher constructs a real Core, then fails Admission; finally's
# DB close fails and the actual owning writer terminates.
primary = ValueError('original admission failure')
def fail(*a, **kwargs): raise primary
dispatcher.Admission = fail
try: dispatcher.Dispatcher(root)
except ValueError as e:
    assert e is primary
    e.__traceback__ = None
else: raise AssertionError('startup should fail')
records = [r for r in module._core_registry if r is not None]
assert len(records) == 1
record = records[0]
record.thread.join(2)
assert not record.thread.is_alive() and record.state == 'uncertain'
primary = None
try: record.close()
except RuntimeError as e: assert 'owning_thread' in str(e)
else: raise AssertionError('wrong thread cleanup')
try: module.Core(root)
except ValueError as e: assert str(e) == 'control_busy'
else: raise AssertionError('abandoned exclusion lost')
# Other roots retain the remaining 63 slots; no raw handles required.
owned = [module._reserve_core(root.parent/str(i)) for i in range(63)]
try:
    try: module._reserve_core(root.parent/'overflow')
    except ValueError as e: assert 'capacity' in str(e)
    else: raise AssertionError('capacity overshot')
finally:
    for r in owned: r.close()
sys.platform = 'linux'
# This isolated ownership model has no enrolled roots; admission is tested above.
recovery._require_legacy_writer_scope = lambda roots: None
try: recovery._run_bounded_maintenance({}, {}, lambda s: True)
except ValueError as e: assert 'core_ownership_busy' in str(e)
else: raise AssertionError('abandoned maintenance admitted')
assert [r for r in module._core_registry if r is not None] == records
print('one abandoned record; 63 remaining slots; maintenance refused; normal interpreter exit')
'''
    child = subprocess.run([sys.executable, '-c', script, str(tmp_path/'dead')], capture_output=True, text=True, timeout=8)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == 'one abandoned record; 63 remaining slots; maintenance refused; normal interpreter exit'


@pytest.mark.parametrize('family,polls', [('pocket',5),('legacy',None),('export',None)])
def test_alive_poll_cadence_bounds_real_command_journal_and_keeps_stop_immediate(monkeypatch,family,polls):
    """Drive owner ticks with a controlled clock and real SQLite command rows."""
    from concurrent.futures import Future
    from contextlib import nullcontext
    import sqlite3
    from types import SimpleNamespace
    from unittest.mock import Mock
    from service.dispatch_store import DispatchStore
    from service import installed_pocket_daemon as pd, installed_export as ex
    from service import dispatcher as module
    clock=[100.0]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    db=sqlite3.connect(':memory:')
    try:
        db.execute('CREATE TABLE dispatch_commands(seq INTEGER PRIMARY KEY,dispatch_id INTEGER,op TEXT,state TEXT)')
        core=SimpleNamespace(db=db,transaction=lambda:nullcontext(),job=Mock(return_value={'state':'running'}))
        row=dict(seq=1,job_id='job',runtime='runtime',phase='observing',outcome=None,receipt=None)
        store=DispatchStore.__new__(DispatchStore);store.core=core;store.db=db
        store.current=lambda:row
        store.update=lambda unused,**fields:row.update(fields)
        def submit(fn):
            future=Future();future.set_result(fn());return future
        runtime=SimpleNamespace(family={'pocket':pd.FAMILY,'legacy':None,'export':ex.FAMILY}[family],
            observe=Mock(return_value=Observation('runtime',State.ALIVE)),terminate=Mock(return_value=None))
        owner=Dispatcher.__new__(Dispatcher)
        owner.core=core;owner.store=store;owner.runtime=runtime;owner.pool=SimpleNamespace(submit=submit)
        owner.enrolled=False;owner.installed_checked=False;owner.pending=None;owner.next_poll=0
        owner.resource_fault=False;owner.persistence_fault=False;owner.stop_obligation=False
        owner.resource_stop_sent=set();owner.quarantined=False;owner.reconciled=True
        owner.fault=lambda _:None;owner._resource_check=lambda:None
        # Launch + settle every clock point. Actual Dispatcher._launch writes the
        # SQL journal; suppressed observations create neither sent nor returned rows.
        for tick in range(100):
            clock[0]=100+tick*.01
            owner._tick();owner._tick()
        count=db.execute("SELECT count(*) FROM dispatch_commands WHERE op='observe'").fetchone()[0]
        assert count==runtime.observe.call_count
        assert db.execute("SELECT count(*) FROM dispatch_commands WHERE state!='returned'").fetchone()[0]==0
        if polls is not None:assert count==polls
        else:assert count>20
        # A cancel discovered in an in-flight ALIVE response launches stop at
        # this same instant, without first applying the longer observation delay.
        core.job.return_value={'state':'cancel_requested'}
        future=Future();future.set_result(Observation('runtime',State.ALIVE))
        owner.pending=(future,row,'observe',store.command(row,'observe'))
        owner._tick()
        assert owner.pending[2]=='stop'
        runtime.terminate.assert_called_once_with('runtime',1.5)
        # Resource stop obligations likewise run before the next-poll guard.
        owner.pending=None;owner.stop_obligation=True;owner.next_poll=clock[0]+100
        owner._tick()
        assert owner.pending is None
        assert runtime.terminate.call_count==1
    finally:db.close()


@pytest.fixture
def qwen_poll_owner(monkeypatch):
    """Real command SQL with immediate or explicitly held single-executor futures."""
    from concurrent.futures import Future
    from contextlib import nullcontext
    import sqlite3
    from types import SimpleNamespace
    from unittest.mock import Mock
    from service.dispatch_store import DispatchStore
    from service.installed_qwen_daemon import FAMILY
    from service import dispatcher as module
    clock=[0.0];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE dispatch_commands(seq INTEGER PRIMARY KEY,dispatch_id INTEGER,op TEXT,state TEXT)')
    core=SimpleNamespace(db=db,transaction=lambda:nullcontext(),job=Mock(return_value={'state':'running'}))
    row=dict(seq=1,job_id='job',runtime='runtime',phase='observing',outcome=None,receipt=None,authority='authority')
    store=DispatchStore.__new__(DispatchStore);store.core=core;store.db=db
    store.current=lambda:None if row.get('acked') else row
    store.update=lambda unused,**fields:row.update(fields)
    runtime=SimpleNamespace(family=FAMILY,observe=Mock(return_value=Observation('runtime',State.ALIVE)),
        terminate=Mock(),start=Mock(),collect=Mock(return_value=('receipt',None)),call=Mock(return_value={'acked':True}))
    def submit(fn):
        future=Future()
        try:future.set_result(fn())
        except Exception as exc:future.set_exception(exc)
        return future
    owner=Dispatcher.__new__(Dispatcher)
    owner.core=core;owner.store=store;owner.runtime=runtime;owner.pool=SimpleNamespace(submit=submit)
    owner.enrolled=False;owner.installed_checked=True;owner.installed_activation=SimpleNamespace(family=FAMILY)
    owner.pending=None;owner.next_poll=0;owner.resource_fault=False;owner.persistence_fault=False
    owner.stop_obligation=False;owner.resource_stop_sent=set();owner.quarantined=False;owner.reconciled=True
    owner.integrity_fault=False;owner.closing=False;owner.fault=lambda _:None;owner._resource_check=Mock()
    def terminal(unused,receipt,observation):
        assert receipt=='receipt'
        assert not runtime.call.called  # ACK follows receipt/reap processing.
        row.update(outcome='terminal',receipt=receipt)
    owner._terminal=terminal
    yield owner,row,runtime,clock,db,submit
    db.close()


@pytest.mark.parametrize('held',[False,True])
def test_qwen_five_1800_second_attempts_have_25_commands(qwen_poll_owner,held):
    from concurrent.futures import Future
    owner,row,runtime,clock,db,submit=qwen_poll_owner
    for attempt in range(5):
        row.update(seq=attempt+1,phase='bound',outcome=None,receipt=None,acked=0)
        owner.next_poll=0;runtime.call.reset_mock()
        owner._launch(row,'prepare',lambda:'runtime')
        owner._tick();owner._tick()  # prepare -> start -> observing
        for second in range(1800):
            clock[0]+=1
            if held:
                future=Future();owner.pool.submit=lambda fn:future
                owner._tick()
                for _ in range(3):owner._tick()
                assert owner.pending[0] is future
                future.set_result(Observation('runtime',State.ALIVE))
                owner.pool.submit=submit
            else:owner._tick()
            owner._tick()
        runtime.observe.return_value=Observation('runtime',State.STOPPED)
        clock[0]+=1
        owner._tick();owner._tick()
        assert owner.pending[2]=='confirm'
        owner._tick();assert owner.pending[2]=='collect'
        owner._tick();clock[0]+=1;owner._tick();owner._tick()
        assert row['acked']==1
        runtime.observe.return_value=Observation('runtime',State.ALIVE)
    assert db.execute('SELECT count(*) FROM dispatch_commands').fetchone()[0]==25
    assert db.execute("SELECT count(*) FROM dispatch_commands WHERE state!='returned'").fetchone()[0]==0


@pytest.mark.parametrize('pending_op',['prepare','start','poll','observe'])
def test_cancel_pending_io_and_replays_share_resource_stop_latch(qwen_poll_owner,pending_op):
    from concurrent.futures import Future
    owner,row,runtime,clock,db,submit=qwen_poll_owner
    future=Future()
    command=None if pending_op=='poll' else owner.store.command(row,pending_op)
    owner.pending=(future,row,pending_op,command)
    owner.core.job.return_value={'state':'cancel_requested'}
    owner.next_poll=999
    owner._tick();assert not runtime.terminate.called
    future.set_result('runtime' if pending_op=='prepare' else
                      None if pending_op=='start' else Observation('runtime',State.ALIVE))
    owner._tick();assert owner.pending[2]=='stop'
    owner._tick()
    for _ in range(10):
        owner.next_poll=0;owner.stop_obligation=True;owner._tick();owner._tick()
    assert runtime.terminate.call_count==1
    assert db.execute("SELECT count(*) FROM dispatch_commands WHERE op='stop'").fetchone()[0]==1


@pytest.mark.parametrize('result',['error','unknown','contradiction','identity','stopped_error','stopped_unknown'])
def test_qwen_poll_failure_and_terminal_confirmation_evidence(qwen_poll_owner,result):
    owner,row,runtime,clock,db,submit=qwen_poll_owner
    if result=='error':runtime.observe.side_effect=OSError('uncertain')
    elif result=='identity':runtime.observe.return_value=Observation('other',State.ALIVE)
    elif result=='unknown':runtime.observe.return_value=Observation('runtime',State.UNKNOWN)
    elif result=='stopped_error':runtime.observe.side_effect=[Observation('runtime',State.STOPPED),OSError('uncertain confirmation')]
    elif result=='stopped_unknown':runtime.observe.side_effect=[Observation('runtime',State.STOPPED),Observation('runtime',State.UNKNOWN)]
    else:runtime.observe.side_effect=[Observation('runtime',State.STOPPED),Observation('runtime',State.ALIVE)]
    owner._tick();owner._tick()
    if result in ('unknown','contradiction','stopped_error','stopped_unknown'):owner._tick()
    assert owner.quarantined
    assert row['phase']=='unknown'
    assert not runtime.collect.called and not runtime.call.called
    assert row['outcome'] is None and row['receipt'] is None
    assert db.execute('SELECT op,state FROM dispatch_commands').fetchall()==[
        ('observe','uncertain' if result in ('error','identity','stopped_error') else 'returned')]


def test_poll_requires_both_checked_installed_qwen_tokens(qwen_poll_owner):
    owner,row,runtime,clock,db,submit=qwen_poll_owner
    assert owner._qwen_state_poll()
    owner.installed_checked=False;assert not owner._qwen_state_poll()
    owner.installed_checked=True;runtime.family=object();assert not owner._qwen_state_poll()
