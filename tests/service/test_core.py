"""Real SQLite/process tests for the bounded synthetic core; no ML/network."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from contracts.validate import ContractError
from service.core import Core as ActiveCore

def Core(root):
    return ActiveCore(root, historical_local=True)

FIXTURES=Path(__file__).resolve().parents[2]/'contracts/fixtures/v1'

@pytest.fixture
def context():
    return json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['context']

@pytest.fixture
def core(tmp_path):
    value=Core(tmp_path/'data')
    yield value
    if value.active: value.finish(cancel=True)
    value.close()


def test_persisted_idempotency_conflict_scope(core,context):
    first=core.submit_synthetic('p','key',context)
    changed=copy.deepcopy(context);changed['plan']['request']['text']='changed'
    with pytest.raises(ContractError,match='idempotency_conflict'):
        core.submit_synthetic('p','key',changed)
    assert core.submit_synthetic('p','key',context)==first
    assert core.submit_synthetic('other','key',context)['id']!=first['id']
    root=core.root
    core.close(); reopened=Core(root)
    assert reopened.submit_synthetic('p','key',context)==first
    reopened.close()
    # fixture still owns a valid connection for teardown
    core.__init__(root)


def test_one_owner_and_lane(core,context):
    with pytest.raises(ValueError,match='control_busy'): Core(core.root)
    job=core.submit_synthetic('p','a',context)
    core.start(delay=.2)
    assert core.job(job['id'])['state']=='running'
    with pytest.raises(ValueError,match='lane_busy'): core.start()
    assert core.finish()['state']=='succeeded'
    assert [e['state'] for e in core.events(job['id'])]==['queued','running','succeeded']
    assert len(core.jobs('p',limit=1))==1


def test_cas_and_history_no_inference(core):
    project=json.loads((FIXTURES/'production-30.json').read_text())
    core.create_project(project)
    block=project['blocks'][0]['id']
    revised=core.edit_text(project['id'],1,block,'  Exact\n é ')
    assert revised['blocks'][0]['text']=='  Exact\n é '
    with pytest.raises(ContractError,match='revision_conflict'):
        core.edit_text(project['id'],1,block,'lost update')
    assert core.project(project['id'],1)==project
    assert core.project(project['id'])==revised
    assert core.jobs('p')==[]


def test_cancel_reap_and_new_retry_lineage(core,context):
    job=core.submit_synthetic('p','a',context)
    core.start(delay=10)
    assert core.request_cancel(job['id'])['state']=='cancel_requested'
    result=core.finish(cancel=True)
    assert result['state']=='cancelled' and result['attempts'][-1]['reaped']
    retry=core.submit_synthetic('p','b',context,retry_of=job['id'])
    assert retry['id']!=job['id'] and retry['input_sha256']==job['input_sha256']
    assert core.db.execute('SELECT retry_of FROM jobs WHERE id=?',(retry['id'],)).fetchone()[0]==job['id']
    core.start(); assert core.finish()['state']=='succeeded'


def test_verified_completion_wins_cancel(core,context):
    job=core.submit_synthetic('p','a',context);core.start()
    core.active[1].wait(timeout=5)
    core.request_cancel(job['id'])
    assert core.finish()['state']=='succeeded'


def proposal(core,context):
    job=core.submit_synthetic('p','a',context);core.start()
    _,process,connection,lock=core.active
    process.wait(timeout=5)
    raw=connection.recv(65536)
    connection.close();lock.close();core.active=None
    return job,json.loads(raw)

@pytest.mark.parametrize('field,value',[('owner_id','foreign'),('generation',99),('staging_id','foreign'),('input_sha256','0'*64)])
def test_foreign_result_fences(core,context,field,value):
    job,result=proposal(core,context)
    result['message'][field]=value
    with pytest.raises(ContractError): core._publish(job['id'],result)
    assert core.job(job['id'])['state']=='running'
    core.reconcile(); assert core.job(job['id'])['state']=='failed'

@pytest.mark.parametrize('corruption',['missing','symlink','provenance','truncated','hash'])
def test_actual_file_and_provenance_verification(core,context,corruption):
    job,result=proposal(core,context)
    path=core.root/'staging'/core.job(job['id'])['attempts'][-1]['staging_id']/'output.wav'
    if corruption=='missing': path.unlink()
    if corruption=='symlink':
        path.unlink();path.symlink_to('/etc/hosts')
    if corruption=='provenance': result['manifest']['outputs'][0]['provenance']['source_id']='foreign'
    if corruption=='truncated': path.write_bytes(path.read_bytes()[:48])
    if corruption=='hash': result['manifest']['outputs'][0]['sha256']='0'*64
    with pytest.raises((OSError,ValueError,ContractError)): core._publish(job['id'],result)
    assert not core.db.execute('SELECT * FROM artifacts').fetchall()
    core.reconcile()


def test_disk_full_no_false_success(core,context,monkeypatch):
    job,result=proposal(core,context)
    import builtins
    real=builtins.open
    def failing(path,*args,**kwargs):
        if Path(path).parent==core.root/'assets': raise OSError(28,'synthetic disk full')
        return real(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'open',failing)
        with pytest.raises(OSError): core._publish(job['id'],result)
    assert core.reconcile()  # Storage ambiguity requires a reopened session.
    assert core.job(job['id'])['state']=='running'
    root=core.root;core.close();core.__init__(root,historical_local=True)
    assert not core.reconcile()
    assert core.job(job['id'])['state']=='failed'
    assert not core.db.execute('SELECT * FROM artifacts').fetchall()


@pytest.mark.parametrize('point,state',[('before_adoption','failed'),('after_adoption','succeeded'),('after_commit','succeeded')])
def test_control_process_death_and_restart(tmp_path,point,state):
    root=tmp_path/'data'
    script='''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]), historical_local=True);context=json.loads(Path(sys.argv[2]).read_text())['context']
c.submit_synthetic('p','a',context);c.start()
def die(point):
    if point==sys.argv[3]: os._exit(73)
c.finish(fault=die)
'''
    process=subprocess.run([sys.executable,'-c',script,str(root),str(FIXTURES/'transcript-speech_generate.json'),point],timeout=10)
    assert process.returncode==73
    core=Core(root)
    try:
        assert not core.reconcile()
        job=core.jobs('p')[0]['job']
        assert job['state']==state
        assert not core.reconcile() # repeat recovery is idempotent
        if state=='succeeded':
            asset=json.loads(core.db.execute('SELECT metadata FROM assets').fetchone()[0])
            assert (root/'assets'/asset['id']).is_file()
    finally: core.close()


def test_control_death_live_worker_quarantine_then_recover(tmp_path):
    root=tmp_path/'data'
    script='''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]), historical_local=True);x=json.loads(Path(sys.argv[2]).read_text())['context']
c.submit_synthetic('p','a',x);c.start(delay=1);os._exit(74)
'''
    assert subprocess.run([sys.executable,'-c',script,str(root),str(FIXTURES/'transcript-speech_generate.json')],timeout=5).returncode==74
    core=Core(root)
    try:
        assert core.reconcile()
        with pytest.raises(ValueError,match='lane_quarantined'): core.start()
        deadline=time.monotonic()+5
        while core.reconcile() and time.monotonic()<deadline: time.sleep(.05)
        assert not core.reconcile()
        assert core.jobs('p')[0]['job']['state']=='failed'
    finally: core.close()


def test_terminal_stale_result_cannot_publish(core,context):
    job,result=proposal(core,context)
    core.reconcile()
    with pytest.raises(ContractError): core._publish(job['id'],result)


def test_queue_cancel_without_worker(core,context):
    job=core.submit_synthetic('p','a',context)
    core.request_cancel(job['id']);core.reconcile()
    assert core.job(job['id'])['state']=='cancelled'
    assert core.start() is None


def test_frozen_input_and_missing_published_media(core,context):
    job=core.submit_synthetic('p','a',context)
    original=copy.deepcopy(context)
    context['plan']['request']['text']='draft changed after admission'
    assert core._load(job['id'])[1]==original
    core.start();result=core.finish()
    artifact=core.artifact(result['artifact_ids'][0])
    asset=core.asset(artifact['asset_ids'][0])
    (core.root/'assets'/asset['id']).unlink()
    with pytest.raises(OSError): core.artifact(artifact['id'])
    # Corruption is detected on retrieval, never replaced by invented bytes.


def test_foreign_stage_refused(core,context):
    job,result=proposal(core,context)
    stage=core.root/'staging'/core.job(job['id'])['attempts'][-1]['staging_id']
    stage.chmod(0o755)
    with pytest.raises(ContractError,match='foreign_stage'): core._publish(job['id'],result)
    core.reconcile()


def test_explicit_unimplemented_render_and_reuse(core,context):
    render=json.loads((FIXTURES/'transcript-project_render.json').read_text())['context']
    with pytest.raises(NotImplementedError): core.submit_synthetic('p','render',render)
    context['plan']['request']['reuse']='exact'
    with pytest.raises(NotImplementedError): core.submit_synthetic('p','reuse',context)
    assert core.jobs('p')==[]


def test_deadline_kills_and_reaps(core,context):
    context['plan']['resolved']['profile']['limits']['deadline_seconds']=1
    core.submit_synthetic('p','a',context);core.start(delay=10)
    result=core.finish()
    assert result['state']=='failed' and result['stage']=='deadline'
    assert result['attempts'][-1]['reaped']


def test_late_completion_delayed_collection(core,context):
    context['plan']['resolved']['profile']['limits']['deadline_seconds']=1
    core.submit_synthetic('p','late',context)
    core.start(delay=2)
    process=core.active[1]
    # No finish call: the control observer must enforce and reap the deadline.
    process.completion.thread.join(timeout=5)
    assert not process.completion.thread.is_alive()
    assert process.poll() is not None and process.completion.reaped
    time.sleep(.1)
    result=core.finish()
    assert result['state']=='failed' and result['stage']=='deadline'
    assert result['attempts'][-1]['reaped'] and not result['artifact_ids']


def test_timely_completion_collected_after_deadline(core,context):
    context['plan']['resolved']['profile']['limits']['deadline_seconds']=2
    job=core.submit_synthetic('p','timely',context)
    core.start()
    process=core.active[1]
    process.completion.thread.join(timeout=5)
    assert process.completion.reaped and not process.completion.expired
    from datetime import datetime, timezone
    deadline=datetime.fromisoformat(core.job(job['id'])['deadline'])
    time.sleep(max(0,(deadline-datetime.now(timezone.utc)).total_seconds())+.1)
    result=core.finish()
    assert result['state']=='succeeded' and result['input_sha256']==job['input_sha256']
    assert result['attempts'][-1]['reaped']


@pytest.mark.parametrize('offset,expired', [(-.001,False),(0,True),(.001,True)])
def test_control_observation_deadline_edge(monkeypatch,offset,expired):
    # Deterministic boundary of control observation, independent of child clocks.
    from service.core import CompletionObserver
    class Exited:
        def wait(self,timeout): return 0
        def poll(self): return 0
    monkeypatch.setattr('service.core.time.monotonic',lambda:100+offset)
    observer=CompletionObserver(Exited(),100)
    assert observer.collect() is expired
    assert observer.reaped


def test_retry_canonical_numeric_context_rejected_unchanged(core,context):
    from contracts import validate as oracle
    job=core.submit_synthetic('p','old',context)
    core.request_cancel(job['id']);core.reconcile()
    old=core.job(job['id'])
    changed=copy.deepcopy(context)
    limits=changed['plan']['resolved']['profile']['limits']
    limits['queue']=float(limits['queue'])
    assert changed==context
    oracle.frozen_plan(changed['plan'],changed['bundles'])
    assert oracle.plan_hash(changed['plan'])!=old['input_sha256']
    with pytest.raises(ContractError,match='invalid_retry'):
        core.submit_synthetic('p','new',changed,retry_of=job['id'])
    assert core.job(job['id'])==old and len(core.jobs('p'))==1
    retry=core.submit_synthetic('p','exact',copy.deepcopy(context),retry_of=job['id'])
    assert retry['id']!=old['id'] and retry['input_sha256']==old['input_sha256']
    assert core.db.execute('SELECT retry_of FROM jobs WHERE id=?',(retry['id'],)).fetchone()==(old['id'],)


def test_retry_checks_prior_stored_plan_hash(core,context):
    job=core.submit_synthetic('p','old',context)
    core.request_cancel(job['id']);core.reconcile()
    old=core.job(job['id']);old['input_sha256']='0'*64
    # Local synthetic inconsistent-store fixture, not an external probe.
    from service.core import encode
    core.db.execute('UPDATE jobs SET job=? WHERE id=?',(encode(old),old['id']))
    with pytest.raises(ContractError,match='invalid_retry'):
        core.submit_synthetic('p','retry',context,retry_of=old['id'])
    assert core.job(old['id'])==old and len(core.jobs('p'))==1


def test_deadline_observation_lost_on_restart_fails_closed(tmp_path):
    root=tmp_path/'data'
    script='''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]), historical_local=True);x=json.loads(Path(sys.argv[2]).read_text())['context']
x['plan']['resolved']['profile']['limits']['deadline_seconds']=1
c.submit_synthetic('p','a',x);c.start()
c.active[1].completion.thread.join(timeout=5)
assert c.active[1].completion.reaped and not c.active[1].completion.expired
os._exit(75)
'''
    assert subprocess.run([sys.executable,'-c',script,str(root),str(FIXTURES/'transcript-speech_generate.json')],timeout=8).returncode==75
    reopened=Core(root)
    try:
        assert not reopened.reconcile()
        job=reopened.jobs('p')[0]['job']
        assert job['state']=='failed' and job['stage']=='interrupted'
        assert job['attempts'][-1]['reaped'] and not job['artifact_ids']
        assert not reopened.reconcile()
    finally: reopened.close()


def test_already_exited_late_worker_rejected(core,context,monkeypatch):
    import threading
    from service.core import CompletionObserver
    release=threading.Event()
    observe=CompletionObserver._observe
    def delayed_observer(self):
        assert release.wait(timeout=6)
        observe(self)
    monkeypatch.setattr(CompletionObserver,'_observe',delayed_observer)
    context['plan']['resolved']['profile']['limits']['deadline_seconds']=1
    core.submit_synthetic('p','late-exit',context);core.start(delay=1.2)
    try:
        assert core.active[1].wait(timeout=5)==0
    finally: release.set()
    result=core.finish()
    assert result['state']=='failed' and result['stage']=='deadline'
    assert result['attempts'][-1]['reaped'] and not result['artifact_ids']


def test_default_core_cannot_launch_historical_worker(tmp_path,context):
    active=ActiveCore(tmp_path/'active')
    try:
        active.submit_synthetic('p','a',context)
        with pytest.raises(ValueError,match='dispatcher_required'): active.start()
        with pytest.raises(ValueError,match='dispatcher_required'): active.finish()
        with pytest.raises(ValueError,match='dispatcher_required'): active.reconcile()
        assert active.active is None
    finally: active.close()


@pytest.mark.parametrize('phase', ['connect', 'pragma', 'version', 'schema', 'persistence', 'namespace', 'double-failure'])
def test_lifetime_constructor_failure_releases_owned_resources(tmp_path, monkeypatch, phase):
    import sqlite3
    import fcntl
    from service import core as module
    root = tmp_path/'partial'
    opened = []
    connect = sqlite3.connect
    original = RuntimeError('constructor phase '+phase)
    fault = True
    before = set(module._failed_constructor_cleanup)
    class Connection(sqlite3.Connection):
        def close(self):
            super().close()
            if phase == 'double-failure' and fault: raise RuntimeError('secondary close failure')
        def executescript(self, sql):
            if phase == 'schema': raise original
            return super().executescript(sql)
        def execute(self, sql, *args, **kwargs):
            if phase == 'pragma' and sql == 'PRAGMA synchronous=FULL': raise original
            if phase == 'version' and sql == 'PRAGMA user_version': raise original
            return super().execute(sql, *args, **kwargs)
    def tracked(*args, **kwargs):
        if phase == 'connect': raise original
        db = connect(*args, factory=Connection, **kwargs); opened.append(db); return db
    monkeypatch.setattr(module.sqlite3, 'connect', tracked)
    if phase in ('persistence', 'double-failure'):
        monkeypatch.setattr(module.persistence, 'initialize', lambda *a, **k: (_ for _ in ()).throw(original))
    if phase == 'namespace':
        mkdir = module.persistence.mkdir_durable
        def failed(path, *args):
            if path.name == 'staging': raise original
            return mkdir(path, *args)
        monkeypatch.setattr(module.persistence, 'mkdir_durable', failed)
    try:
        with pytest.raises(RuntimeError) as caught: ActiveCore(root)
        assert caught.value is original and caught.tb is not None
        for db in opened:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'): db.execute('SELECT 1')
        if phase == 'double-failure': assert 'RuntimeError' in caught.value.__notes__[0]
        if phase == 'double-failure':
            cleanup, = module._failed_constructor_cleanup - before
            with pytest.raises(ValueError, match='control_busy'): ActiveCore(root)
            with (root/'control.lock').open('a+b') as owner:
                with pytest.raises(BlockingIOError): fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            with (root/'control.lock').open('a+b') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fault = False
        for cleanup in module._failed_constructor_cleanup - before:
            cleanup.close()
            assert cleanup.owner.closed
    assert module._failed_constructor_cleanup == before
    with (root/'control.lock').open('a+b') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize('retain_traceback', [True, False])
def test_lifetime_f2_uncertain_close_retains_exclusion(tmp_path, monkeypatch, when, retain_traceback):
    import sqlite3
    import threading
    from service import core as module, recovery
    from service.dispatcher import Dispatcher

    root = tmp_path/'f2'
    initial = ActiveCore(root)
    initial.close()
    original_connect = sqlite3.connect
    primary = ValueError('synthetic persistence failure')
    opened, reads, close_order = [], [], []
    before = set(module._failed_constructor_cleanup)
    fault = True

    class Connection(sqlite3.Connection):
        def close(self):
            if fault and when == 'before':
                raise RuntimeError('before actual SQLite release')
            # Exclusion survives both sides of the actual SQLite close.
            with pytest.raises(ValueError, match='control_busy'):
                with recovery._startup_read_lock(root): pass
            super().close()
            close_order.append('db')
            with pytest.raises(ValueError, match='control_busy'):
                with recovery._startup_read_lock(root): pass
            if fault:
                raise RuntimeError('after actual SQLite release')

    def connect(*args, **kwargs):
        db = original_connect(*args, factory=Connection, **kwargs)
        opened.append(db)
        return db

    def fail(*args, **kwargs): raise primary
    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    monkeypatch.setattr(module.persistence, 'initialize', fail)
    try:
        with pytest.raises(ValueError) as caught: ActiveCore(root)
        assert caught.value is primary and caught.tb is not None
        assert primary.__notes__ == ['constructor cleanup failed: RuntimeError']
        cleanup, = module._failed_constructor_cleanup - before
        assert cleanup.db is opened[0] and not cleanup.owner.closed
        if not retain_traceback:
            primary.__traceback__ = None
            del caught
        if when == 'before':
            assert opened[0].execute('SELECT 1').fetchone() == (1,)
        else:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                opened[0].execute('SELECT 1')
        monkeypatch.setattr(module.sqlite3, 'connect', original_connect)
        def witness(*args):
            reads.append(True)
            raise AssertionError('startup reader entered while release uncertain')
        monkeypatch.setattr(recovery, 'startup_read', witness)
        with pytest.raises(ValueError, match='control_busy'): Dispatcher(root)
        with pytest.raises(ValueError, match='control_busy'):
            with recovery._startup_read_lock(root): pass
        script = '''
import sys
from pathlib import Path
from service import recovery
from service.dispatcher import Dispatcher
root = Path(sys.argv[1])
def witness(*args): raise AssertionError('reader entered')
recovery.startup_read = witness
for attempt in (lambda: Dispatcher(root), lambda: recovery._startup_read_lock(root).__enter__()):
    try: attempt()
    except ValueError as exc: assert str(exc) == 'control_busy'
    else: raise AssertionError('exclusion lost')
print('both refused before startup read')
'''
        child = subprocess.run([sys.executable, '-c', script, str(root)],
                               capture_output=True, text=True, timeout=8)
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == 'both refused before startup read'
        assert reads == []
        if when == 'before': assert opened[0].execute('SELECT 1').fetchone() == (1,)
        errors = []
        def wrong_thread():
            try: cleanup.close()
            except RuntimeError as exc: errors.append(str(exc))
        thread = threading.Thread(target=wrong_thread)
        thread.start(); thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == ['constructor_cleanup_requires_owning_thread']
        assert not cleanup.owner.closed
    finally:
        # Explicit recovery only on this connection's owning thread, after the
        # synthetic fault is removed; no automatic retry or cross-thread close.
        fault = False
        for cleanup in module._failed_constructor_cleanup - before:
            cleanup.close()
            close_order.append('owner')
            assert cleanup.owner.closed and cleanup.db is None
        for db in opened:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'): db.execute('SELECT 1')
    assert close_order[-2:] == ['db', 'owner']
    assert module._failed_constructor_cleanup == before
    with recovery._startup_read_lock(root) as readable: assert readable


def test_lifetime_f2_owner_close_failure_preserves_primary(tmp_path, monkeypatch):
    import sqlite3
    from service import core as module
    before = set(module._failed_constructor_cleanup)
    original_open = open
    primary = ValueError('synthetic persistence failure')
    fault = True
    class Owner:
        def __init__(self, actual): self.actual = actual
        def fileno(self): return self.actual.fileno()
        def close(self):
            if fault: raise RuntimeError('owner close failure')
            self.actual.close()
    def owned_open(*args, **kwargs): return Owner(original_open(*args, **kwargs))
    monkeypatch.setattr(module, 'open', owned_open, raising=False)
    def fail(*args, **kwargs): raise primary
    monkeypatch.setattr(module.persistence, 'initialize', fail)
    try:
        with pytest.raises(ValueError) as caught: ActiveCore(tmp_path/'owner-failure')
        assert caught.value is primary and caught.tb
        assert primary.__notes__ == ['constructor cleanup failed: RuntimeError']
        cleanup, = module._failed_constructor_cleanup - before
        assert cleanup.db is None and not cleanup.owner.actual.closed
    finally:
        fault = False
        for cleanup in module._failed_constructor_cleanup - before: cleanup.close()
    assert module._failed_constructor_cleanup == before


def test_lifetime_f2_db_failure_does_not_attempt_failing_owner(tmp_path, monkeypatch):
    import sqlite3
    from service import core as module
    before = set(module._failed_constructor_cleanup)
    original_open, original_connect = open, sqlite3.connect
    primary = ValueError('synthetic primary with two cleanup faults')
    fault, owner_attempts = True, []
    class Owner:
        def __init__(self, actual): self.actual = actual
        def fileno(self): return self.actual.fileno()
        def close(self):
            owner_attempts.append(True)
            if fault: raise RuntimeError('owner close failure')
            self.actual.close()
    class Connection(sqlite3.Connection):
        def close(self):
            if fault: raise RuntimeError('DB close before release')
            super().close()
    monkeypatch.setattr(module, 'open', lambda *a, **k: Owner(original_open(*a, **k)), raising=False)
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *a, **k: original_connect(*a, factory=Connection, **k))
    def fail(*args, **kwargs): raise primary
    monkeypatch.setattr(module.persistence, 'initialize', fail)
    try:
        with pytest.raises(ValueError) as caught: ActiveCore(tmp_path/'two-faults')
        assert caught.value is primary and caught.tb
        assert primary.__notes__ == ['constructor cleanup failed: RuntimeError']
        cleanup, = module._failed_constructor_cleanup - before
        assert cleanup.db.execute('SELECT 1').fetchone() == (1,)
        assert owner_attempts == [] and not cleanup.owner.actual.closed
    finally:
        fault = False
        for cleanup in module._failed_constructor_cleanup - before: cleanup.close()
    assert owner_attempts == [True]
    assert module._failed_constructor_cleanup == before


# Selected bounded full-lifetime contract. These tests release only their own
# records, on their creating threads; abandoned writer coverage uses a child.
def _bounded_records():
    from service import core as module
    return tuple(r for r in module._core_registry if r is not None)


def test_bounded_core_capacity_before_effects(tmp_path, monkeypatch):
    from service import core as module
    baseline = _bounded_records()
    owned = []
    effects = []
    assert module.CORE_OWNERSHIP_LIMIT == 64
    monkeypatch.setattr(module.persistence, 'mkdir_durable', lambda *a: effects.append('mkdir'))
    monkeypatch.setattr(module, 'open', lambda *a: effects.append('owner'), raising=False)
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *a, **k: effects.append('db'))
    try:
        for i in range(63): owned.append(module._reserve_core(tmp_path/str(i)))
        assert len(_bounded_records()) == 63
        owned.append(module._reserve_core(tmp_path/'last'))
        assert len(_bounded_records()) == 64
        with pytest.raises(ValueError, match='core_ownership_capacity'):
            ActiveCore(tmp_path/'overflow')
        assert effects == [] and not (tmp_path/'overflow').exists()
        owned.pop().close()
        replacement = module._reserve_core(tmp_path/'replacement'); owned.append(replacement)
        assert len(_bounded_records()) == 64
    finally:
        for record in owned: record.close()
    assert _bounded_records() == baseline


def test_bounded_core_atomic_last_slot(tmp_path):
    import threading
    from service import core as module
    baseline = _bounded_records()
    owned = [module._reserve_core(tmp_path/str(i)) for i in range(63)]
    start, held, release = threading.Barrier(3), threading.Barrier(3), threading.Event()
    results = []
    def contend(i):
        record = None
        try:
            start.wait(timeout=3)
            try:
                record = module._reserve_core(tmp_path/('race'+str(i)))
                results.append('admitted')
            except ValueError: results.append('refused')
            held.wait(timeout=3)
            assert release.wait(3)
        finally:
            if record is not None: record.close()
    threads = [threading.Thread(target=contend, args=(i,)) for i in range(2)]
    try:
        for t in threads: t.start()
        start.wait(timeout=3); held.wait(timeout=3)
        assert sorted(results) == ['admitted', 'refused']
        assert len(_bounded_records()) == 64
    finally:
        release.set()
        for t in threads: t.join(timeout=4); assert not t.is_alive()
        for r in owned: r.close()
    assert _bounded_records() == baseline


def test_bounded_core_nested_roots_aliases_and_sequential(tmp_path, monkeypatch):
    from service import core as module
    baseline = _bounded_records()
    original = module.persistence.mkdir_durable
    nested = []
    outer_root = tmp_path/'outer'
    def mkdir(path, sync):
        if path == outer_root and not nested:
            nested.append(ActiveCore(tmp_path/'inner'))
            assert len(_bounded_records()) == len(baseline)+2
            with pytest.raises(ValueError, match='control_busy'): ActiveCore(outer_root/'..'/'outer')
        return original(path, sync)
    monkeypatch.setattr(module.persistence, 'mkdir_durable', mkdir)
    outer = None
    try:
        outer = ActiveCore(outer_root)
        alias = tmp_path/'alias'; alias.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match='control_busy'): ActiveCore(alias/'inner')
        assert len(_bounded_records()) == len(baseline)+2
        with pytest.raises(ValueError, match='control_busy'): ActiveCore(outer_root)
    finally:
        if outer: outer.close()
        for c in nested: c.close()
    monkeypatch.setattr(module.persistence, 'mkdir_durable', original)
    for i in range(66):
        c = ActiveCore(tmp_path/'sequential'); c.close()
        assert _bounded_records() == baseline


@pytest.mark.parametrize('phase', ['root', 'owner', 'connect', 'namespace', 'pragma', 'schema', 'persistence', 'sync'])
def test_bounded_core_early_failure_accounting(tmp_path, monkeypatch, phase):
    from service import core as module
    import sqlite3
    baseline = _bounded_records(); opened = []
    primary = KeyboardInterrupt('owned early fault')
    root = tmp_path/'early'
    mkdir, connect, sync = module.persistence.mkdir_durable, sqlite3.connect, module.sync_dir
    def fail(): raise primary
    def make(path, callback):
        if phase == 'root' and path == root or phase == 'namespace' and path.name == 'staging': fail()
        return mkdir(path, callback)
    class DB(sqlite3.Connection):
        def execute(self, sql, *args):
            if phase == 'pragma' and sql == 'PRAGMA synchronous=FULL': fail()
            return super().execute(sql, *args)
        def executescript(self, sql):
            if phase == 'schema': fail()
            return super().executescript(sql)
    def connection(*a, **kw):
        if phase == 'connect': fail()
        db = connect(*a, factory=DB, **kw); opened.append(db); return db
    monkeypatch.setattr(module.persistence, 'mkdir_durable', make)
    monkeypatch.setattr(module.sqlite3, 'connect', connection)
    if phase == 'owner': monkeypatch.setattr(module, 'open', lambda *a: fail(), raising=False)
    if phase == 'persistence': monkeypatch.setattr(module.persistence, 'initialize', lambda *a, **k: fail())
    if phase == 'sync':
        def syncing(path):
            if opened: fail()
            return sync(path)
        monkeypatch.setattr(module, 'sync_dir', syncing)
    with pytest.raises(KeyboardInterrupt) as caught: ActiveCore(root)
    assert caught.value is primary and caught.tb
    assert _bounded_records() == baseline
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError): db.execute('SELECT 1')


@pytest.mark.parametrize('when', ['before', 'after', 'owner'])
@pytest.mark.parametrize('drop', [False, True])
def test_bounded_core_ordinary_close_faults(tmp_path, monkeypatch, when, drop):
    import sqlite3
    from service import core as module
    baseline = _bounded_records(); fault = True; calls = []
    connect, opening = sqlite3.connect, open
    class DB(sqlite3.Connection):
        def close(self):
            calls.append('db')
            if fault and when == 'before': raise RuntimeError('DB before')
            super().close()
            if fault and when == 'after': raise RuntimeError('DB after')
    class Owner:
        def __init__(self, file): self.file = file
        def fileno(self): return self.file.fileno()
        def close(self):
            calls.append('owner')
            if fault and when == 'owner': raise RuntimeError('owner')
            self.file.close()
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *a, **k: connect(*a, factory=DB, **k))
    monkeypatch.setattr(module, 'open', lambda *a: Owner(opening(*a)), raising=False)
    c = ActiveCore(tmp_path/'ordinary'); r = c._ownership
    assert r in _bounded_records() and r.state == 'live'
    try:
        with pytest.raises(RuntimeError) as caught: c.close()
        assert r in _bounded_records()
        assert calls == (['db', 'owner'] if when == 'owner' else ['db'])
        with pytest.raises(sqlite3.ProgrammingError, match='closed'): c.jobs('p')
        if when == 'before': assert r.db.execute('SELECT 1').fetchone() == (1,)
        if drop:
            caught.value.__traceback__ = None
            del caught, c
        with pytest.raises(ValueError, match='control_busy'): ActiveCore(tmp_path/'ordinary')
    finally:
        fault = False
        r.close()
    assert calls.count('db') == (1 if when == 'owner' else 2)
    assert r.db is None and r.owner.file.closed
    assert _bounded_records() == baseline


def test_bounded_core_identity_active_reentrant_and_stale(tmp_path, monkeypatch):
    import threading
    from service import core as module
    baseline = _bounded_records(); c = ActiveCore(tmp_path/'identity'); r = c._ownership
    try:
        c.active = object()
        with pytest.raises(ValueError, match='live_owner_close'): c.close()
        with pytest.raises(ValueError, match='live_owner_close'): r.close()
        assert r.state == 'live' and r.db.execute('SELECT 1').fetchone() == (1,)
        c.active = None
        errors = []
        def wrong():
            # Matching the reusable numeric identity supplies no authority.
            current = threading.current_thread(); original_ident = current._ident
            current._ident = r.thread.ident
            try: c.close()
            except RuntimeError as e: errors.append(str(e))
            finally: current._ident = original_ident
        t = threading.Thread(target=wrong); t.start(); t.join(3)
        assert not t.is_alive() and errors == ['constructor_cleanup_requires_owning_thread']
        with monkeypatch.context() as m:
            m.setattr(module.os, 'getpid', lambda: r.pid+1)
            class Trap:
                def acquire(self, **k): raise AssertionError('inherited mutex touched')
            m.setattr(module, '_core_registry_mutex', Trap())
            with pytest.raises(ValueError, match='inherited_process'): c.close()
            with pytest.raises(ValueError, match='inherited_process'): module._reserve_core(tmp_path/'pid')
            with pytest.raises(ValueError, match='inherited_process'):
                with module._core_maintenance_lease(): pass
        real = r.db
        class Reentrant:
            def close(self):
                with pytest.raises(ValueError, match='reentrant'): c.close()
                real.close()
        r.db = Reentrant()
        c.close()
        newer = ActiveCore(tmp_path/'newer')
        try:
            assert newer._ownership.slot == r.slot
            c.close(); r.close()
            assert newer._ownership in _bounded_records()
        finally: newer.close()
    finally:
        c.active = None; c.close()
    assert _bounded_records() == baseline


def test_bounded_core_nested_failure_separate_accounting(tmp_path, monkeypatch):
    from service import core as module
    baseline = _bounded_records(); mkdir = module.persistence.mkdir_durable
    outer = tmp_path/'outer-failure'; inner = tmp_path/'inner-failure'; seen = []
    def make(path, sync):
        if path == outer and not seen:
            with pytest.raises(KeyboardInterrupt, match='inner fault'): ActiveCore(inner)
            seen.append(len(_bounded_records()))
            assert _bounded_records()[0].root_key == str(outer)
        if path == inner: raise KeyboardInterrupt('inner fault')
        return mkdir(path, sync)
    monkeypatch.setattr(module.persistence, 'mkdir_durable', make)
    c = ActiveCore(outer)
    try:
        assert seen == [len(baseline)+1]
        assert len(_bounded_records()) == len(baseline)+1
    finally: c.close()
    assert _bounded_records() == baseline


@pytest.mark.parametrize('phase', ['wal', 'foreign', 'fetch', 'query', 'sync-parent'])
def test_bounded_core_remaining_early_phases(tmp_path, monkeypatch, phase):
    import sqlite3
    from service import core as module
    baseline = _bounded_records(); opened = []; primary = KeyboardInterrupt(phase)
    connect, sync = sqlite3.connect, module.sync_dir
    root = tmp_path/'remaining'
    def fail(): raise primary
    class DB(sqlite3.Connection):
        def execute(self, sql, *a):
            if (phase == 'wal' and sql == 'PRAGMA journal_mode=WAL' or
                phase == 'foreign' and sql == 'PRAGMA foreign_keys=ON' or
                phase == 'query' and sql == 'PRAGMA query_only=ON'): fail()
            if phase == 'fetch' and sql == 'PRAGMA user_version':
                class Cursor:
                    def fetchone(self): fail()
                return Cursor()
            return super().execute(sql, *a)
    def connection(*a, **k):
        db = connect(*a, factory=DB, **k); opened.append(db); return db
    monkeypatch.setattr(module.sqlite3, 'connect', connection)
    if phase == 'query': monkeypatch.setattr(module.persistence, 'initialize', lambda *a, **k: False)
    def syncing(path):
        if phase == 'sync-parent' and opened and path == root.parent: fail()
        return sync(path)
    monkeypatch.setattr(module, 'sync_dir', syncing)
    with pytest.raises(KeyboardInterrupt) as caught: ActiveCore(root)
    assert caught.value is primary and caught.tb
    assert _bounded_records() == baseline
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError): db.execute('SELECT 1')


def test_bounded_core_no_mutex_during_resource_hooks(tmp_path, monkeypatch):
    from service import core as module
    import sqlite3
    baseline = _bounded_records(); connect = sqlite3.connect; opening = open; witnessed = []
    def check():
        assert module._core_registry_mutex.acquire(blocking=False)
        module._core_registry_mutex.release(); witnessed.append(True)
    class DB(sqlite3.Connection):
        def execute(self, *a, **k): check(); return super().execute(*a, **k)
        def close(self): check(); super().close()
    class Owner:
        def __init__(self, file): self.file = file
        def fileno(self): return self.file.fileno()
        def close(self): check(); self.file.close()
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *a, **k: connect(*a, factory=DB, **k))
    monkeypatch.setattr(module, 'open', lambda *a: Owner(opening(*a)), raising=False)
    c = ActiveCore(tmp_path/'hooks')
    try:
        c.active = object()
        with pytest.raises(ValueError, match='live_owner_close'): c._ownership.close()
        # There is no private caller-supplied ordinary-close authorization flag.
        with pytest.raises(TypeError): c._ownership.close(_ordinary=True)
        assert c._ownership.state == 'live'
    finally:
        c.active = None; c.close()
    assert witnessed and _bounded_records() == baseline
