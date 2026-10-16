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
    with s._preflight(), c.transaction():s.reserve_tx('a','inference',demand());s.acquire_tx('a')
    with s._preflight(), c.transaction():s.reserve_tx('b',category,demand())
    with pytest.raises(ValueError,match='resource_'):
        with s._preflight(), c.transaction():s.acquire_tx('b')
    with pytest.raises(ValueError,match='reap'):
        with c.transaction():s.release_tx('a')
    with s._preflight(), c.transaction():s.release_tx('a',confirmed_reap=True);s.acquire_tx('b')
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


def test_queue_quota_free_space_and_growth(allocator):
    c,a,stage=allocator;s=a.store;s.budget['queue']=1
    with s._preflight(), c.transaction():s.reserve_tx('a','decode',demand())
    with pytest.raises(ValueError,match='queue_full'):
        with s._preflight(), c.transaction():s.reserve_tx('b','render',demand())
    assert s.state()=='healthy'
    (stage/'growth').write_bytes(b'x'*100000)
    measured=s.check_disk();(stage/'growth').unlink()
    assert s.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]>=measured
    stage.rmdir()
    with pytest.raises(ValueError,match='root_missing'):s.check_disk()
    assert s.state()=='fault'


@pytest.mark.parametrize('kind',['quota','free'])
def test_observed_disk_fault_latches(allocator,kind):
    _,a,_=allocator;s=a.store
    if kind=='quota':s.budget['disk_bytes']=1
    else:s.budget['free_bytes']=2**53-1
    with pytest.raises(ValueError,match='disk_'):s.check_disk()
    assert s.state()=='fault'
    with pytest.raises(ValueError,match='unresolved'):s.check_disk()


def test_independent_stronger_budget_no_cpu_ceiling(allocator):
    c,a,_=allocator;s=a.store
    s.budget.update(slots=3,ram_bytes=3*192*1024**2,pids=144,resident_models=2,vram_bytes=1024)
    for i,category in enumerate(('inference','decode','render')):
        d=demand();d['configuration']='cuda' if i else 'metal';d['vram_bytes']=100
        with s._preflight(), c.transaction():s.reserve_tx(str(i),category,d);s.acquire_tx(str(i))
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
    with s._preflight(), c.transaction():s.reserve_tx('a','decode',d);s.acquire_tx('a')
    other=dict(d,configuration='different')
    with s._preflight(), c.transaction():s.reserve_tx('b','provision',other)
    with pytest.raises(ValueError,match='resource_'+key):
        with s._preflight(), c.transaction():s.acquire_tx('b')


def test_reservation_retry_before_mutable_capacity_and_fifo(allocator):
    c,a,_=allocator;s=a.store;d=demand()
    with s._preflight(), c.transaction():s.reserve_tx('a','decode',d);s.reserve_tx('b','render',d)
    s.budget['ram_bytes']=1
    with c.transaction():assert s.reserve_tx('a','decode',d)['id']=='a'
    s.budget['ram_bytes']=DEFAULT_BUDGET['ram_bytes']
    with pytest.raises(ValueError,match='fifo'):
        with s._preflight(), c.transaction():s.acquire_tx('b')


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
    with s._preflight(), c.transaction():s.reserve_tx('a','provision',demand());s.acquire_tx('a');s.release_tx('a',confirmed_reap=True)
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


@pytest.mark.parametrize('cancel',[False,True])
def test_fault_insert_failure_transient_output_never_recovers(tmp_path,cancel):
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
        output.unlink()
        if cancel:d.call('cancel',j['id'])
        r.release.set();wait(d,lambda:r.stops)
        import time
        r.state=State.ALIVE;time.sleep(.5)
        assert d.call('status')['quarantined'] and d.call('status')['persistence_fault']
        assert d.call('status')['reservations'][0]['state']=='executing'
        with pytest.raises(ValueError):d.call('submit','p','b')
        r.state=State.STOPPED
        wait(d,lambda:d.call('job',j['id'])['state']==('cancelled' if cancel else 'failed'))
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
    with s._preflight(), c.transaction():s.reserve_tx('a','inference',d);s.acquire_tx('a')
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
        wait(d,lambda:len(fired)==1);r.release.set()
        wait(d,lambda:r.stops==1 and len(fired)==1)
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


def authority_crash_child(root, stage, mode):
    """Disposable process: no close/finally/retry between denied UPDATE and exit."""
    import os
    r=Runtime(Path(stage));r.block.set()
    armed=[]
    def fault(point):
        if point!='prespawn':return
        original=d.admission.check
        def check(**kwargs):
            if not r.starts:return original(**kwargs)
            assert d.admission.store.state()=='checking'
            independent=sqlite3.connect(Path(root)/'core.sqlite')
            assert independent.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
            independent.close()
            directory=Path(stage)/r.authority.job['attempts'][-1]['staging_id'];directory.mkdir(exist_ok=True)
            output=directory/'output.wav';output.write_bytes(b'x'*10000)
            try:original(**kwargs)
            except ValueError as exc:
                assert str(exc)=='resource_output_bytes'
                armed.append(True)
                d.core.db.execute('PRAGMA query_only=ON')
                d.core.db.set_trace_callback(lambda sql: armed.append(sql) if sql.startswith("UPDATE resource_monitor SET state='fault'") else None)
                raise
            raise AssertionError('real_violation_missing')
        d.admission.check=check
        observe=d.admission.store.observe
        def denied(fn):
            try:return observe(fn)
            except sqlite3.OperationalError:
                assert len(armed)==2 and armed[1].startswith("UPDATE resource_monitor SET state='fault'") and d.admission.store.state()=='checking'
                if mode=='interval':
                    d.core.db.execute('PRAGMA query_only=OFF')
                    (Path(stage)/r.authority.job['attempts'][-1]['staging_id']/'output.wav').unlink()
                    d._sync_resource_fault()
                    for _ in range(5):d._resource_check()
                    assert len(armed)==2 and d.persistence_fault
                os._exit(29)
        d.admission.store.observe=denied
    d=Dispatcher(root,staging=stage,runtime=r,fault=fault);ready(d)
    d.call('submit','p','a')
    d.thread.join(8)
    os._exit(99)


@pytest.mark.parametrize('mode',['immediate','interval'])
def test_query_only_after_real_violation_immediate_death_twice_restart(tmp_path,mode):
    import os
    import subprocess
    import sys
    root=tmp_path/'core';stage=tmp_path/'stage';stage.mkdir()
    script='from tests.service.test_admission import authority_crash_child; authority_crash_child('+repr(str(root))+','+repr(str(stage))+','+repr(mode)+')'
    result=subprocess.run([sys.executable,'-c',script],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=15,capture_output=True,text=True)
    assert result.returncode==29,result.stderr
    db=sqlite3.connect(root/'core.sqlite')
    assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
    assert db.execute('SELECT * FROM resource_faults').fetchall()==[]
    job=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0]);db.close()
    for output in stage.rglob('output.wav'):output.unlink()
    r=Runtime(stage);r.state=State.STOPPED;r.block.set()
    for _ in range(2):
        d=Dispatcher(root,staging=stage,runtime=r)
        try:
            ready(d);wait(d,lambda:d.call('job',job['id'])['state']=='failed')
            status=d.call('status')
            assert status['persistence_fault'] and status['quarantined']
            assert status['reservations'][0]['state']=='executing'
            assert d.call('submit','p','a')['id']==job['id']
            for op,args in [('submit',('p','new')),('replace',(job['id'],))]:
                with pytest.raises(ValueError):d.call(op,*args)
            assert not d.call('job',job['id'])['artifact_ids'] and r.acks==r.starts==0
        finally:d.close()
    db=sqlite3.connect(root/'core.sqlite')
    assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
    assert db.execute('SELECT * FROM artifacts').fetchall()==[]
    db.close()


@pytest.mark.parametrize('point',['before','mid','after'])
def test_healthy_check_death_retains_committed_fence(tmp_path,point):
    import os
    import subprocess
    import sys
    stage=tmp_path/'stage';stage.mkdir();root=tmp_path/'core'
    script='''
import os,sqlite3,sys
from pathlib import Path
from service.core import Core
from service.admission import Admission
c=Core(Path(sys.argv[1]));a=Admission(c,Path(sys.argv[2]));s=a.store
original=s.measure
def observation():
    independent=sqlite3.connect(c.root/'core.sqlite')
    assert independent.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
    independent.close()
    if sys.argv[3]=='before':os._exit(31)
    def measure():
        result=original()
        if sys.argv[3]=='mid':os._exit(31)
        return result
    s.measure=measure
    a.check()
    os._exit(31)
s.observe(observation)
'''
    result=subprocess.run([sys.executable,'-c',script,str(root),str(stage),point],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=8,capture_output=True,text=True)
    assert result.returncode==31,result.stderr
    c=Core(root);a=Admission(c,stage)
    try:
        assert a.store.state()=='checking'
        with pytest.raises(ValueError,match='unresolved'):a.check()
    finally:c.close()


@pytest.mark.parametrize('transition',['checking','healthy','fault'])
def test_fence_write_failure_and_zero_observation_on_begin(allocator,transition):
    c,a,_=allocator;s=a.store;calls=[]
    c.db.execute("CREATE TRIGGER deny_monitor BEFORE UPDATE ON resource_monitor WHEN NEW.state='"+transition+"' BEGIN SELECT RAISE(ABORT,'denied'); END")
    def observe():
        calls.append(1)
        assert not c.db.in_transaction
        assert s.state()=='checking'
        s.check_disk()
        if transition=='fault':raise ValueError('actual_policy_fault')
    with pytest.raises(sqlite3.IntegrityError):s.observe(observe)
    assert len(calls)==(0 if transition=='checking' else 1)
    assert s.state()==('healthy' if transition=='checking' else 'checking')
    with pytest.raises(ValueError,match='unresolved'):s.check_disk()
    assert c.db.execute('SELECT * FROM resource_faults').fetchall()==[]


def test_snapshot_is_private_immediate_and_not_rollbackable(allocator):
    c,a,_=allocator;s=a.store
    with pytest.raises(ValueError,match='preflight'):
        with c.transaction():s.reserve_tx('none','decode',demand())
    with s._preflight():
        with pytest.raises(RuntimeError):
            with c.transaction():
                s.reserve_tx('rolled','decode',demand())
                raise RuntimeError('rollback')
        with pytest.raises(ValueError,match='stale'):
            with c.transaction():s.reserve_tx('stale','decode',demand())
    assert s.state()=='healthy' and s.rows()==[]
    with pytest.raises(ValueError,match='reentrant'):
        with c.transaction():a.check()
    with pytest.raises(ValueError,match='unfenced'):s.measure()


@pytest.mark.parametrize('damage',['revision','missing','old','invalid','empty-old'])
def test_populated_roots_never_invent_healthy_authority(allocator,damage):
    c,a,stage=allocator;s=a.store
    with s._preflight(), c.transaction():s.reserve_tx('a','decode',demand())
    if damage=='revision':c.db.execute('DELETE FROM resource_protocol')
    elif damage=='missing':c.db.execute('DELETE FROM resource_monitor')
    elif damage=='old':c.db.execute('UPDATE resource_protocol SET revision=1')
    else:
        c.db.execute('DROP TABLE resource_monitor')
        if damage=='empty-old':c.db.execute('DELETE FROM resource_reservations')
    from service.resource_store import ResourceStore
    rebuilt=ResourceStore(c,dict(DEFAULT_BUDGET),(c.root,stage))
    assert rebuilt.state()=='invalid'
    with pytest.raises(ValueError,match='unresolved'):rebuilt.check_disk()


@pytest.mark.parametrize('phase',['begin','complete'])
def test_fence_commit_failure_rolls_back_only_its_own_transaction(allocator,phase,monkeypatch):
    from contextlib import contextmanager
    c,a,_=allocator;s=a.store;transaction=c.transaction;calls=[]
    @contextmanager
    def denied_commit():
        before=s.state()
        with transaction():
            yield
            if (phase=='begin' and before=='healthy' and s.state()=='checking') or (phase=='complete' and before=='checking' and s.state()=='healthy'):
                raise sqlite3.OperationalError('commit_denied')
    monkeypatch.setattr(c,'transaction',denied_commit)
    def observation():calls.append(True);return s.check_disk()
    with pytest.raises(sqlite3.OperationalError):s.observe(observation)
    assert len(calls)==(0 if phase=='begin' else 1)
    assert s.state()==('healthy' if phase=='begin' else 'checking')
    assert s.db.execute('SELECT highwater FROM resource_accounting').fetchone()==(0,)


def test_no_constructor_observation_and_allocation_reads_outside_transaction(tmp_path,monkeypatch):
    from service.resource_store import ResourceStore
    calls=[];measure=ResourceStore.measure
    def measured(s):
        assert not s.db.in_transaction and s.state()=='checking'
        calls.append(True)
        return measure(s)
    monkeypatch.setattr(ResourceStore,'measure',measured)
    c=Core(tmp_path/'core');stage=tmp_path/'stage';stage.mkdir();a=Admission(c,stage)
    try:
        assert calls==[]
        store=DispatchStore(c,a);f=fixture()
        j=c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
        assert len(calls)==1
        store.allocate(f['profiles'],'silence',8)
        assert len(calls)==2
        a.check(publication=True);assert len(calls)==3
        c.submit_synthetic('p','a',f['context'],_admission=a,_profiles=[])
        assert len(calls)==3  # dedup is a metadata-only read
        assert c.job(j['id'])['generation']==1
    finally:c.close()


def test_stored_terminal_outcome_cannot_ack_under_inherited_fence(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage);r.block.set();r.state=State.STOPPED
    hit=[]
    def fault(point):
        if point=='durable_outcome':
            d.core.db.execute("UPDATE resource_monitor SET state='checking'")
            d._sync_resource_fault();hit.append(True)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a');wait(d,lambda:hit)
        result=d.call('job',j['id']);assert result['state']=='succeeded'
        assert d.call('artifact',result['artifact_ids'][0])['job_id']==j['id']
        assert d.call('submit','p','a')['id']==j['id']
        assert r.acks==0
        retained=d.call('status')['reservations'];assert retained[0]['state']=='retained'
    finally:d.close()
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r)
    try:
        wait(d,lambda:d.call('status')['persistence_fault'])
        assert d.call('job',j['id'])==result and d.call('status')['reservations']==retained
        assert r.acks==0
        with pytest.raises(ValueError):d.call('submit','p','new')
    finally:d.close()


def test_unresolved_queued_cancel_has_no_partial_state(allocator):
    c,a,_=allocator;f=fixture()
    j=c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    c.db.execute("UPDATE resource_monitor SET state='checking'")
    events=c.events(j['id'])
    with pytest.raises(ValueError,match='unresolved'):c.request_cancel(j['id'],_cancel_unstarted=True,_admission=a)
    assert c.job(j['id'])==j and c.events(j['id'])==events and a.store.rows()[0]['state']=='queued'


def test_failed_begin_unchanged_healthy_restarts_without_invented_fault(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();root=tmp_path/'core';c=Core(root);a=Admission(c,stage)
    c.db.execute('PRAGMA query_only=ON')
    with pytest.raises(sqlite3.OperationalError):a.check()
    assert a.store.state()=='healthy' and a.store.failed
    c.close()
    c=Core(root);a=Admission(c,stage)
    try:a.check();assert a.store.state()=='healthy' and not a.store.failed
    finally:c.close()


@pytest.mark.parametrize('state',['checking','healthy'])
def test_failed_preflight_cannot_insert_job_or_attempt(allocator,state):
    c,a,_=allocator;f=fixture()
    c.db.execute("CREATE TRIGGER deny_fence BEFORE UPDATE ON resource_monitor WHEN NEW.state='"+state+"' BEGIN SELECT RAISE(ABORT,'denied'); END")
    with pytest.raises(sqlite3.IntegrityError):
        c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    for table in ('jobs','events','resource_reservations','dispatch_settings','dispatches'):
        assert c.db.execute('SELECT * FROM '+table).fetchall()==[]
    assert a.store.state()==('healthy' if state=='checking' else 'checking')


def test_acquisition_rollback_preserves_job_reservation_and_attestation(allocator):
    c,a,_=allocator;f=fixture();store=DispatchStore(c,a)
    j=c.submit_synthetic('p','a',f['context'],_dispatch_settings=('silence',8),_admission=a,_profiles=f['profiles'])
    rows=a.store.rows();events=c.events(j['id'])
    c.db.execute("CREATE TRIGGER deny_dispatch BEFORE INSERT ON dispatches BEGIN SELECT RAISE(ABORT,'dispatch_denied'); END")
    with pytest.raises(sqlite3.IntegrityError):store.allocate(f['profiles'],'silence',8)
    assert c.job(j['id'])==j and a.store.rows()==rows and c.events(j['id'])==events
    assert a.store.state()=='healthy' and not a.store.failed


def test_publication_intent_is_not_recovered_under_unresolved_authority(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage);r.block.set();r.state=State.STOPPED;hit=[]
    def fault(point):
        if point=='before_adoption':
            d.core.db.execute("UPDATE resource_monitor SET state='checking'")
            d._sync_resource_fault();hit.append(True)
            raise RuntimeError('control_interruption_after_intent')
    root=tmp_path/'core';d=Dispatcher(root,staging=stage,runtime=r,fault=fault);ready(d)
    try:
        j=d.call('submit','p','a');wait(d,lambda:hit)
    finally:d.close()
    d=Dispatcher(root,staging=stage,runtime=r)
    try:
        ready(d);wait(d,lambda:d.call('job',j['id'])['state']=='failed')
        assert not d.call('job',j['id'])['artifact_ids'] and r.acks==0
        assert d.call('status')['reservations'][0]['state']=='executing'
    finally:d.close()
    c=Core(root)
    try:
        assert c.db.execute('SELECT COUNT(*) FROM publications').fetchone()==(1,)
        assert c.db.execute('SELECT COUNT(*) FROM artifacts').fetchone()==(0,)
        assert list((root/'assets').iterdir())==[]
    finally:c.close()
