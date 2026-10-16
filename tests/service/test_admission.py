"""E1 resource policy/SQLite tests; no hardware qualification."""
import copy
import json
from pathlib import Path
import sqlite3

import pytest
from service.admission import Admission, DEFAULT_BUDGET, finite_budget
from service.core import Core
from service.dispatch_store import DispatchStore
from tests.service.test_dispatcher import Runtime, ready, wait
from service.dispatcher import Dispatcher
from worker.lifecycle import State


def fixture():
    f=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
    for p in f['profiles']:
        p.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='E1 synthetic only')
    return f


@pytest.fixture
def allocator(tmp_path):
    core=Core(tmp_path/'core');stage=tmp_path/'stage';stage.mkdir()
    a=Admission(core,stage);DispatchStore(core,a)
    yield core,a,stage
    core.close()


def demand():
    return dict(configuration='synthetic-config',slots=1,ram_bytes=192*1024**2,vram_bytes=0,pids=48,disk_bytes=2*1024**2,attempt_disk_bytes=2*1024**2)


@pytest.mark.parametrize('category',['inference','decode','render','provision'])
def test_shared_category_contention_and_receipt_release(allocator,category):
    c,a,_=allocator;s=a.store
    with c.transaction():s.reserve_tx('a','inference',demand());s.acquire_tx('a')
    with c.transaction():s.reserve_tx('b',category,demand())
    with pytest.raises(ValueError,match='resource_'):
        with c.transaction():s.acquire_tx('b')
    with pytest.raises(ValueError,match='reap'):
        with c.transaction():s.release_tx('a')
    with c.transaction():s.release_tx('a',confirmed_reap=True);s.acquire_tx('b')
    assert s.rows()[0]['state']=='retained' and s.rows()[0]['demand']['disk_bytes']>0


@pytest.mark.parametrize('key,value',[('slots',float('inf')),('pids',True),('queue',-1),('ram_bytes',0)])
def test_finite_numeric_budget(key,value):
    b=dict(DEFAULT_BUDGET);b[key]=value
    with pytest.raises(ValueError):finite_budget(b)


@pytest.mark.parametrize('field,value',[('readiness','busy'),('readiness','disconnected'),('readiness','quarantined'),('support','unqualified'),('installation','absent'),('config_revision','stale'),('device','metal')])
def test_trusted_current_separate_frozen(allocator,field,value):
    _,a,_=allocator;f=fixture();original=copy.deepcopy(f['context']);f['profiles'][0][field]=value
    with pytest.raises(Exception):a.demand(f['context'],f['profiles'])
    assert f['context']==original


def test_precision_and_absent_profile(allocator):
    _,a,_=allocator;f=fixture()
    with pytest.raises(ValueError):a.demand(f['context'],[])
    f['profiles'][0]['compatibility'][0]['identity']['precision']='float16'
    with pytest.raises(ValueError):a.demand(f['context'],f['profiles'])


def test_atomic_reserve_failure_rolls_job_settings_events(allocator):
    c,a,_=allocator;f=fixture()
    c.db.execute("CREATE TRIGGER fail_resource BEFORE INSERT ON resource_reservations BEGIN SELECT RAISE(ABORT,'reserve_failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    for table in ('jobs','events','dispatch_settings','resource_reservations'):assert c.db.execute('SELECT * FROM '+table).fetchall()==[]


def test_queue_quota_free_space_and_growth(allocator,monkeypatch):
    c,a,stage=allocator;s=a.store;s.budget['queue']=1
    with c.transaction():s.reserve_tx('a','decode',demand())
    with pytest.raises(ValueError,match='queue_full'):
        with c.transaction():s.reserve_tx('b','render',demand())
    s.budget['disk_bytes']=1
    with pytest.raises(ValueError,match='disk_quota'):s.check_disk()
    s.budget['disk_bytes']=DEFAULT_BUDGET['disk_bytes'];s.budget['free_bytes']=2**53-1
    with pytest.raises(ValueError,match='disk_free'):s.check_disk()
    s.budget['free_bytes']=0
    (stage/'growth').write_bytes(b'x'*100000)
    measured=s.check_disk();(stage/'growth').unlink()
    assert s.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]>=measured
    stage.rmdir()
    with pytest.raises(ValueError,match='root_missing'):s.check_disk()


def test_independent_stronger_budget_no_cpu_ceiling(allocator):
    c,a,_=allocator;s=a.store
    s.budget.update(slots=3,ram_bytes=3*192*1024**2,pids=144,resident_models=2,vram_bytes=1024)
    for i,category in enumerate(('inference','decode','render')):
        d=demand();d['configuration']='cuda' if i else 'metal';d['vram_bytes']=100
        with c.transaction():s.reserve_tx(str(i),category,d);s.acquire_tx(str(i))
    assert len([r for r in s.rows() if r['state']=='executing'])==3


def test_dispatch_restart_unknown_idempotency_and_retained_disk(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r);ready(d)
    j=d.call('submit','p','a');wait(d,lambda:r.starts)
    assert len(d.call('status')['reservations'])==1
    d.close();r.state=State.UNKNOWN
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,profiles=[])
    try:
        ready(d);assert d.call('status')['quarantined']
        assert d.call('submit','p','a')['id']==j['id']
        assert d.call('status')['reservations'][0]['state']=='executing'
        with pytest.raises(ValueError):d.call('submit','p','b')
        r.timely=False;r.state=State.STOPPED
        wait(d,lambda:d.call('status')['phase']=='idle')
        row=d.call('status')['reservations'][0]
        assert row['state']=='retained' and row['demand']['disk_bytes']==2*1024**2
    finally:d.close()


def test_actual_byte_growth_stops_and_refuses_publication(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r);ready(d)
    try:
        j=d.call('submit','p','a');wait(d,lambda:r.starts)
        path=stage/r.authority.job['attempts'][-1]['staging_id'];path.mkdir()
        (path/'output.wav').write_bytes(b'x'*10000)
        wait(d,lambda:d.call('job',j['id'])['state']=='failed')
        assert not d.call('job',j['id'])['artifact_ids']
        wait(d,lambda:d.call('status')['phase']=='idle')
        assert d.call('status')['quarantined']
    finally:d.close()


@pytest.mark.parametrize('key',['slots','ram_bytes','vram_bytes','pids','resident_models'])
def test_independent_capacity_dimensions(allocator,key):
    c,a,_=allocator;s=a.store
    s.budget.update(slots=10,ram_bytes=10000,pids=1000,resident_models=10,vram_bytes=100)
    d=demand();d.update(ram_bytes=100,pids=2,vram_bytes=3)
    s.budget[key]=1 if key=='resident_models' else d[key]
    with c.transaction():s.reserve_tx('a','decode',d);s.acquire_tx('a')
    other=dict(d,configuration='different')
    with c.transaction():s.reserve_tx('b','provision',other)
    with pytest.raises(ValueError,match='resource_'+key):
        with c.transaction():s.acquire_tx('b')


def test_reservation_retry_before_mutable_capacity_and_fifo(allocator):
    c,a,_=allocator;s=a.store;d=demand()
    with c.transaction():s.reserve_tx('a','decode',d);s.reserve_tx('b','render',d)
    s.budget['ram_bytes']=1
    with c.transaction():assert s.reserve_tx('a','decode',d)['id']=='a'
    s.budget['ram_bytes']=DEFAULT_BUDGET['ram_bytes']
    with pytest.raises(ValueError,match='fifo'):
        with c.transaction():s.acquire_tx('b')


@pytest.mark.parametrize('field,value',[('text','x'*10001),('variants',5)])
def test_frozen_text_variant_limits(allocator,field,value):
    _,a,_=allocator;f=fixture();f['context']['plan']['request'][field]=value
    with pytest.raises(ValueError):a.demand(f['context'],f['profiles'])


def test_concurrent_idempotent_calls_one_budget(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r);ready(d)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs=list(pool.map(lambda _:d.call('submit','p','same'),range(16)))
        assert len({j['id'] for j in jobs})==1
        assert len(d.call('status')['reservations'])==1
        wait(d,lambda:r.starts);assert r.starts==1
    finally:d.close()


def test_cancel_release_failure_is_atomic(allocator):
    c,a,_=allocator;f=fixture()
    j=c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    c.db.execute("CREATE TRIGGER fail_release BEFORE UPDATE ON resource_reservations BEGIN SELECT RAISE(ABORT,'release_failure'); END")
    with pytest.raises(sqlite3.IntegrityError):c.request_cancel(j['id'],_cancel_unstarted=True,_admission=a)
    assert c.job(j['id'])['state']=='queued' and a.store.rows()[0]['state']=='queued'


def test_decoded_header_bound_before_publication(allocator):
    import io
    import wave
    c,a,stage=allocator;f=fixture()
    j=c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    row=DispatchStore(c,a).allocate(f['profiles'],'silence',8)
    directory=stage/row['authority'].job['attempts'][-1]['staging_id'];directory.mkdir()
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(b'\0'*9600)
    data=bytearray(buffer.getvalue());data[40:44]=(960000).to_bytes(4,'little')
    (directory/'output.wav').write_bytes(data)
    with pytest.raises(ValueError,match='decoded_limit'):a.check(publication=True)


def test_reconstructed_disk_charge_and_missing_stage_never_frees(allocator):
    from service.resource_store import ResourceStore
    c,a,stage=allocator;s=a.store
    with c.transaction():s.reserve_tx('a','provision',demand());s.acquire_tx('a');s.release_tx('a',confirmed_reap=True)
    (stage/'retained').mkdir();(stage/'retained'/'bytes').write_bytes(b'x'*1000);s.check_disk()
    before=s.rows();(stage/'retained'/'bytes').unlink();(stage/'retained').rmdir()
    reconstructed=ResourceStore(c,dict(DEFAULT_BUDGET),(c.root,stage))
    assert reconstructed.rows()==before
    assert reconstructed.rows()[0]['demand']['disk_bytes']==2*1024**2


class PendingRuntime(Runtime):
    """Deterministically park one real Dispatcher I/O future until fault observed."""
    def __init__(self, stage, pending='observe', stop_unknown=False):
        import threading
        super().__init__(stage)
        self.pending_op=pending;self.entered=threading.Event();self.release=threading.Event()
        self.stops=0;self.stop_unknown=stop_unknown

    def park(self, op):
        if op==self.pending_op and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(4)

    def prepare(self, a, fixture):
        self.park('prepare');return super().prepare(a,fixture)

    def start(self, identity):
        self.park('start');return super().start(identity)

    def observe(self, identity, timeout):
        self.park('observe');return super().observe(identity,timeout)

    def terminate(self, identity, timeout):
        self.stops+=1
        if self.stop_unknown:
            self.state=State.UNKNOWN
            raise OSError('stop_response_unknown')
        return super().terminate(identity,timeout)


@pytest.mark.parametrize('cancel',[False,True])
@pytest.mark.parametrize('pending',['prepare','start','observe'])
@pytest.mark.parametrize('stop_unknown',[False,True])
def test_transient_pressure_pending_io_latches_stop(tmp_path,pending,stop_unknown,cancel):
    stage=tmp_path/'stage';stage.mkdir();r=PendingRuntime(stage,pending,stop_unknown)
    fired=[]
    def fault(point):
        if point=='prespawn':
            original=d.admission.check
            def check(**kwargs):
                if r.entered.is_set() and not fired:
                    fired.append(True);raise ValueError('resource_stage_growth')
                return original(**kwargs)
            d.admission.check=check
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a')
        assert r.entered.wait(2)
        wait(d,lambda:getattr(d,'resource_fault',False))
        assert not r.stops
        assert d.call('status')['reservations'][0]['state']=='executing'
        if cancel:d.call('cancel',j['id'])  # cancellation cannot discard resource stop
        r.release.set();wait(d,lambda:r.stops)
        if stop_unknown:
            wait(d,lambda:d.call('status')['phase']=='unknown')
            # Multiple healthy observations must not erase the obligation/latch.
            import time
            time.sleep(.5)
            assert d.call('status')['stop_obligation']
            assert d.call('status')['reservations'][0]['state']=='executing'
            assert d.call('job',j['id'])['state']==('cancel_requested' if cancel else 'running')
            r.state=State.STOPPED
        wait(d,lambda:d.call('status')['phase']=='idle')
        result=d.call('job',j['id'])
        assert result['state']==('cancelled' if cancel else 'failed') and not result['artifact_ids']
        assert result['attempts'][-1]['reaped']
        assert d.call('status')['reservations'][0]['state']=='retained'
        import time
        time.sleep(.3)  # ACK and later healthy scans cannot unlatch the fault.
        assert d.call('status')['quarantined'] and not d.call('status')['stop_obligation']
        assert r.stops==1 and r.prepares==1
        assert r.starts==(0 if pending=='prepare' else 1)
        with pytest.raises(ValueError):d.call('submit','p','fresh')
        assert d.call('submit','p','a')['id']==j['id']
    finally:r.release.set();d.close()


def test_fault_insert_failure_transient_output_never_recovers(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=PendingRuntime(stage,stop_unknown=True)
    def fault(point):
        if point=='prespawn':
            d.core.db.execute("CREATE TRIGGER fail_fault BEFORE INSERT ON resource_faults BEGIN SELECT RAISE(ABORT,'fault_write_failed'); END")
    root=tmp_path/'core'
    d=Dispatcher(root,staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a');assert r.entered.wait(2)
        path=stage/r.authority.job['attempts'][-1]['staging_id'];path.mkdir()
        output=path/'output.wav';output.write_bytes(b'x'*10000)
        wait(d,lambda:d.call('status')['error']=='IntegrityError')
        output.unlink();r.release.set();wait(d,lambda:r.stops)
        import time
        r.state=State.ALIVE;time.sleep(.5)
        assert d.call('status')['quarantined'] and d.call('status')['persistence_fault']
        assert d.call('status')['reservations'][0]['state']=='executing'
        with pytest.raises(ValueError):d.call('submit','p','b')
        r.state=State.STOPPED
        wait(d,lambda:d.call('job',j['id'])['state']=='failed')
        time.sleep(.3)
        assert not d.call('job',j['id'])['artifact_ids']
        assert d.call('status')['reservations'][0]['state']=='executing'
        assert r.acks==0 and r.stops==1
    finally:r.release.set();d.close()
    # No resource_faults INSERT succeeded; unresolved monitor state is authority.
    c=Core(root)
    assert c.db.execute('SELECT * FROM resource_faults').fetchall()==[]
    assert c.db.execute('SELECT state FROM resource_monitor').fetchone()[0]=='checking'
    c.close()
    d=Dispatcher(root,staging=stage,runtime=r)
    try:
        ready(d);time.sleep(.4)
        assert d.call('status')['persistence_fault'] and d.call('status')['quarantined']
        assert d.call('status')['reservations'][0]['state']=='executing'
        assert d.call('submit','p','a')['id']==j['id']
        with pytest.raises(ValueError):d.call('submit','p','new')
        assert not d.call('job',j['id'])['artifact_ids'] and r.acks==0 and r.starts==1
    finally:d.close()


def test_unresolved_monitor_blocks_store_acquire_reserve_and_release(allocator):
    c,a,_=allocator;s=a.store;d=demand()
    with c.transaction():s.reserve_tx('a','inference',d);s.acquire_tx('a')
    c.db.execute("UPDATE resource_monitor SET state='checking'")
    with c.transaction():assert s.reserve_tx('a','inference',d)['id']=='a'
    with pytest.raises(ValueError,match='unresolved'):
        with c.transaction():s.reserve_tx('b','decode',d)
    with pytest.raises(ValueError,match='unresolved'):
        with c.transaction():s.acquire_tx('a')
    with pytest.raises(ValueError,match='unresolved'):
        with c.transaction():s.release_tx('a',confirmed_reap=True)
    assert s.rows()[0]['state']=='executing'


@pytest.mark.parametrize('pending',['prepare','start','observe'])
def test_pressure_pending_error_keeps_stop_or_unknown_ownership(tmp_path,pending):
    stage=tmp_path/'stage';stage.mkdir()
    class FailedRuntime(PendingRuntime):
        def park(self, op):
            super().park(op)
            if op==self.pending_op:raise OSError('pending_io_failed')
    r=FailedRuntime(stage,pending);fired=[]
    def fault(point):
        if point=='prespawn':
            original=d.admission.check
            def check(**kwargs):
                if r.entered.is_set() and not fired:
                    fired.append(True);raise ValueError('resource_stage_growth')
                return original(**kwargs)
            d.admission.check=check
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a');assert r.entered.wait(2)
        wait(d,lambda:d.call('status')['resource_fault']);r.release.set()
        if pending=='prepare':
            wait(d,lambda:d.call('status')['phase']=='unknown')
            import time
            time.sleep(.3)
            assert d.call('status')['stop_obligation']
            assert d.call('status')['reservations'][0]['state']=='executing'
            assert r.stops==r.starts==r.prepares==0
        elif pending=='observe':
            wait(d,lambda:r.stops==1)
            # Failed observe replies never prove death, even after successful stop.
            import time
            time.sleep(.3)
            assert d.call('status')['reservations'][0]['state']=='executing'
            assert d.call('status')['stop_obligation']
        else:
            wait(d,lambda:d.call('status')['phase']=='idle')
            assert r.stops==1 and r.starts==0
            assert d.call('status')['reservations'][0]['state']=='retained'
        assert not d.call('job',j['id'])['artifact_ids']
        with pytest.raises(ValueError):d.call('submit','p','fresh')
    finally:r.release.set();d.close()


def test_repeated_faults_do_not_storm_stop_or_release_unknown(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=PendingRuntime(stage,stop_unknown=True);fired=[]
    def fault(point):
        if point=='prespawn':
            original=d.admission.check
            def check(**kwargs):
                if r.entered.is_set() and len(fired)<8:
                    fired.append(True);raise ValueError('resource_stage_growth')
                return original(**kwargs)
            d.admission.check=check
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a');assert r.entered.wait(2)
        wait(d,lambda:len(fired)>=3);r.release.set()
        wait(d,lambda:r.stops==1 and len(fired)==8)
        import time
        time.sleep(.5)
        assert r.stops==1 and r.starts==1 and r.prepares==1
        assert d.call('status')['stop_obligation'] and d.call('status')['quarantined']
        assert d.call('status')['reservations'][0]['state']=='executing'
        assert not d.call('job',j['id'])['artifact_ids']
        r.state=State.STOPPED;wait(d,lambda:d.call('status')['phase']=='idle')
        time.sleep(.3)
        assert d.call('status')['quarantined'] and r.acks==1
        with pytest.raises(ValueError):d.call('submit','p','fresh')
    finally:r.release.set();d.close()
