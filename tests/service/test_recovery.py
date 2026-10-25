import sys
from pathlib import Path
# Fresh `python -c` tests import this module without pytest's pythonpath hook.
_tests = str(Path(__file__).resolve().parents[1])
if _tests not in sys.path:
    sys.path.insert(0, _tests)
from _maintenance_wait_guard import fresh_run as _guarded_fresh_run
import os
"""B1 E1 only: real disposable roots/SQLite/locks, mocked host projection."""
import copy
from contextlib import closing
import fcntl
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from service import recovery
from service.admission import DEFAULT_BUDGET
from service.core import Core, encode
from service.persistent_init import initialize, digest, read_file
from service.reconciliation import Reconciliation


def synthetic_profiles():
    return json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['profiles']


@pytest.fixture
def installation(tmp_path):
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values(): p.mkdir()
    scope=dict(daemon='owned-daemon',context='owned-context',volumes={k:dict(name='owned-'+k,driver='local',created='2026-10-16T00:00:00Z') for k in roots})
    profiles=synthetic_profiles()
    result=initialize(roots,scope,profiles,DEFAULT_BUDGET)
    return roots,scope,profiles,result


def registration(installation,evidence='E1-mock'):
    roots,scope,profiles,result=installation
    state=recovery.inspect(roots,scope)
    return dict(deployment_id=result['deployment_id'],scope=scope,evidence=evidence,lane=state['lane'],boot='b'*32,containers=[
        dict(id='c'*64,image='sha256:'+'1'*64,role='control',started='start-control',volumes=['control','ipc','staging']),
        dict(id='d'*64,image='sha256:'+'2'*64,role='worker',started='start-worker',volumes=['ipc','receipts','staging'])])


def enrolled_plan(installation):
    roots,scope,_,_=installation
    r=registration(installation); recovery.register(roots,scope,r)
    observation={k:r[k] for k in ('scope','containers','evidence')}
    observation['states']=[dict(id=c['id'],running=False,restarting=False,pid=0,started=c['started'],finished='finished') for c in r['containers']]
    return observation,recovery.plan(roots,scope,observation,now=100)


def logical(roots):
    from contextlib import closing
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        return list(db.iterdump())


def test_create_versions_and_read_only_inspection(installation):
    roots,scope,_,result=installation
    before=logical(roots); first=recovery.inspect(roots,scope); second=recovery.inspect(roots,scope)
    assert first==second and before==logical(roots)
    assert first['monitor']==[(1,'healthy')] and first['integrity']==[(1,1,'healthy',None)]
    assert first['versions']==dict(contract='1.3.0',core=1,private=1,resource=2)
    assert first['deployment_id']==result['deployment_id'] and first['apply_available'] is False
    with pytest.raises(ValueError,match='new_empty'): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)


@pytest.mark.parametrize('point',['init_record','lifecycle.key','execution.key','journal.json','initialized','lane.lock','database'])
def test_explicit_matching_partial_completion(tmp_path,point):
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    def fault(where):
        if where==point: raise OSError('crash-boundary')
    with pytest.raises(OSError): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,fault=fault)
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    with pytest.raises(ValueError,match='new_empty'): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    with pytest.raises(ValueError,match='matching_initialization'): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete='0'*64)
    result=initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete=digest(record))
    assert result['deployment_id']==record['deployment_id']


@pytest.mark.parametrize('role,name',[('control','old.db'),('ipc','execution.key'),('receipts','journal.json'),('receipts','initialized'),('staging','unknown')])
def test_populated_old_preserved(tmp_path,role,name):
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    old=roots[role]/name;old.write_bytes(b'old-protected')
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    with pytest.raises(ValueError,match='new_empty'): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    assert old.read_bytes()==b'old-protected' and not (roots['receipts']/'p1-init.json').exists()


def test_partial_missing_key_never_reminted(tmp_path):
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    def fault(where):
        if where=='execution.key': raise OSError('crash')
    with pytest.raises(OSError):initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,fault=fault)
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    (roots['ipc']/'lifecycle.key').unlink()
    with pytest.raises(ValueError,match='missing_initialized'): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete=digest(record))
    assert not (roots['ipc']/'lifecycle.key').exists()


def test_plan_immutable_bound_and_unavailable(installation):
    roots,scope,_,_=installation
    observation,planned=enrolled_plan(installation)
    value=planned['plan']
    assert recovery.verify_plan(roots,scope,observation,value['id'],planned['digest'],now=101)==value
    assert value['count']==0 and value['bytes']<=recovery.MAX_BYTES and value['expires']==400
    for digest_value,now,error in [('0'*64,101,'digest'),(planned['digest'],401,'expired')]:
        with pytest.raises(ValueError,match=error): recovery.verify_plan(roots,scope,observation,value['id'],digest_value,now=now)
    before=logical(roots)
    with pytest.raises(NotImplementedError,match='P1B2'): recovery.apply(roots,scope,policy='abandon-uncertain-synthetic-v1')
    assert before==logical(roots)


@pytest.mark.parametrize('mutation',['monitor','journal','registration','scope','consumer','key','volume'])
def test_changed_state_scope_and_consumers_refuse(installation,mutation):
    roots,scope,_,_=installation
    observation,planned=enrolled_plan(installation)
    if mutation=='monitor':
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:db.execute("UPDATE resource_monitor SET state='checking'")
    elif mutation=='journal':
        p=roots['receipts']/'journal.json';v=json.loads(p.read_text());v['payload']['lane']='a'*32;v['sha256']=digest(v['payload']);p.write_text(encode(v))
    elif mutation=='registration': observation['containers'][0]['started']='another-start'
    elif mutation=='scope': scope['daemon']='another-daemon'
    elif mutation=='consumer': observation['containers'].append({**observation['containers'][0],'id':'e'*64})
    elif mutation=='key': (roots['ipc']/'execution.key').unlink()
    else:
        new=roots['staging'].parent/'other';new.mkdir();roots['staging']=new
    with pytest.raises((ValueError,OSError)): recovery.verify_plan(roots,scope,observation,planned['plan']['id'],planned['digest'],now=101)


@pytest.mark.parametrize('lock',['control','journal'])
def test_locks_fail_closed(installation,lock):
    roots,scope,_,_=installation
    p=roots['control']/'core'/'control.lock' if lock=='control' else roots['receipts']/'lane.lock'
    with p.open('rb') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='maintenance_lock_busy'):recovery.inspect(roots,scope)


def test_a2_recognizes_b1_healthy_and_unknown_columns_refuse(installation):
    roots,scope,_,_=installation
    enrolled_plan(installation)
    core=Core(roots['control']/'core')
    try:
        scan=Reconciliation(core,roots['staging']); result=scan.reconcile_startup()
        assert result['complete'] and not result['blocked'],result
        core.db.execute('ALTER TABLE p1_recovery_plans ADD COLUMN surprise TEXT')
        assert Reconciliation(core,roots['staging']).reconcile_startup()['blocked']
    finally:core.close()


def test_bad_typed_reference_refused(installation):
    roots,scope,_,_=installation
    observation,planned=enrolled_plan(installation)
    p=planned['plan'];p['affected']['jobs']=['unknown-job'];p['count']=1
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        db.execute('UPDATE p1_recovery_plans SET record=?,digest=?',(encode(p),digest(p)))
    with pytest.raises(ValueError,match='unknown_b1_job_reference'):recovery.inspect(roots,scope)


def test_gate_requires_actual_host_boot_configuration_and_enrollment(installation):
    roots,scope,profiles,_=installation
    r=registration(installation)
    rid=recovery.register(roots,scope,r)
    hello=dict(lane=r['lane'],boot=r['boot'],quarantined=False)
    with pytest.raises(ValueError,match='registration_mismatch'):recovery.enrollment_gate(roots['control']/'core',rid,hello,profiles,DEFAULT_BUDGET)
    r['evidence']='actual-host-inspection';rid=recovery.register(roots,scope,r)
    assert recovery.enrollment_gate(roots['control']/'core',rid,hello,profiles,DEFAULT_BUDGET)==r['deployment_id']
    with pytest.raises(ValueError):recovery.enrollment_gate(roots['control']/'core',rid,{**hello,'boot':'f'*32},profiles,DEFAULT_BUDGET)
    with pytest.raises(ValueError):recovery.enrollment_gate(roots['control']/'core',rid,hello,[{}],DEFAULT_BUDGET)


def test_docker_host_projection_fixed_and_unknown_consumer(installation):
    from service import host_install as module
    roots,scope,_,_=installation;r=registration(installation,'actual-host-inspection');calls=[]
    def read(argv):
        calls.append(argv)
        if argv[0]=='info':return scope['daemon']
        if argv[0]=='volume':
            v=next(v for v in scope['volumes'].values() if v['name']==argv[2]);return ' '.join(json.dumps(v[k]) for k in ('name','driver','created'))
        if argv[1]=='ls':return '\n'.join(c['id'] for c in r['containers'])
        c=next(c for c in r['containers'] if c['id']==argv[2])
        return encode(dict(id=c['id'],image=c['image'],started=c['started'],running=False,restarting=False,pid=0,finished='finished',restart='no',pidmode='',mounts=[dict(type='volume',name=scope['volumes'][role]['name'],destination='/'+role,rw=not(c['role']=='control' and role in ('ipc','staging'))) for role in c['volumes']]))
    docker=module.Docker(scope['context']);docker._read=read
    assert docker.observe(scope,r)['evidence']=='actual-host-inspection'
    assert any(c[:2]==['container','ls'] for c in calls)
    original=read
    def extra(argv):
        if argv[:2]==['container','ls']:return original(argv)+'\n'+'e'*64
        if argv[:2]==['container','inspect'] and argv[2]=='e'*64:
            return encode(dict(id='e'*64,image='sha256:'+'3'*64,started='x',running=False,restarting=False,pid=0,finished='finished',restart='no',pidmode='',mounts=[dict(type='volume',name=scope['volumes']['ipc']['name'],destination='/ipc',rw=True)]))
        return original(argv)
    docker._read=extra
    with pytest.raises(ValueError,match='unknown_volume_consumer'):docker.observe(scope,r)


@pytest.mark.parametrize('boundary',[1,2,3,4,8,12,20,30,50])
def test_real_fsync_failure_retains_identity_and_never_reseeds(tmp_path,monkeypatch,boundary):
    import os
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    original=os.fsync;calls=0
    def fail(fd):
        nonlocal calls
        calls+=1
        if calls==boundary:raise OSError('fsync-injection')
        return original(fd)
    monkeypatch.setattr(os,'fsync',fail)
    try: initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    except OSError: pass
    monkeypatch.setattr(os,'fsync',original)
    record=read_file(roots['receipts']/'p1-init.json')
    old={p:p.read_bytes() for k in ('ipc','receipts') for p in roots[k].iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='new_empty'):initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    assert read_file(roots['receipts']/'p1-init.json')==record
    assert all(p.read_bytes()==data for p,data in old.items())


def test_partial_core_schema_quarantined_without_health_creation(tmp_path,monkeypatch):
    from service import persistent_init
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    real=persistent_init.Admission
    def crash(*args,**kwargs):raise OSError('before-R1')
    monkeypatch.setattr(persistent_init,'Admission',crash)
    with pytest.raises(OSError):initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    monkeypatch.setattr(persistent_init,'Admission',real)
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    before=logical(roots)
    with pytest.raises(ValueError,match='partial_database'):initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete=digest(record))
    assert before==logical(roots)
    assert not any('resource_monitor' in row for row in before)


def test_typed_plan_retains_publication_job_and_dispatch_assets(installation):
    from tests.service.test_core import proposal
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation
    core=Core(roots['control']/'core',historical_local=True)
    try:
        ctx=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
        job,result=proposal(core,ctx)
        core._publish(job['id'],result)
        # This local historical fixture has no R1 reservation and must stay
        # unresolved, but its real publication/asset references must be retained.
        row=core.db.execute('SELECT id,metadata FROM assets').fetchone()
        state=dict(id='a'*32,deployment_id=installation[3]['deployment_id'],revision=1,
            source_digest='1'*64,proof_digest='2'*64,affected=dict(jobs=[job['id']],dispatches=[],publications=[job['id']]),
            count=2,bytes=100,created=100,expires=400,policy='abandon-uncertain-synthetic-v1',apply_available=False)
        core.db.execute('INSERT INTO p1_recovery_plans VALUES(?,?,?)',(state['id'],encode(state),digest(state)))
        scanner=Reconciliation(core,roots['staging']);summary=scanner.reconcile_startup()
        assert summary['complete'] and 'unknown_reference_schema' not in summary['reasons']
        media=next(r for r in scanner.rows.values() if r['namespace']=='assets' and r['name']==row[0])
        assert media['classification']=='committed_referenced'
        assert 'p1_recovery_plans:'+state['id'] in media['references']
        assert core.job(job['id'])['state']=='succeeded'
    finally:core.close()


def test_host_state_change_invalidates_proposal(installation):
    roots,scope,_,_=installation
    observation,planned=enrolled_plan(installation)
    observation['states'][0].update(running=True,pid=123,finished='')
    with pytest.raises(ValueError,match='live_volume_consumer'):
        recovery.verify_plan(roots,scope,observation,planned['plan']['id'],planned['digest'],now=101)


def test_enrolled_missing_journal_never_reinitializes(installation):
    roots,scope,profiles,_=installation
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    (roots['receipts']/'journal.json').unlink()
    with pytest.raises(ValueError,match='already_enrolled'): initialize(roots,scope,profiles,DEFAULT_BUDGET,complete=digest(record))
    assert not (roots['receipts']/'journal.json').exists()


def test_finite_inspection_refuses_oversized_records(installation):
    roots,scope,_,_=installation
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        db.execute('INSERT INTO p1_registrations VALUES(?,?,?)',('x','x'*(recovery.MAX_BYTES+1),'x'))
    with pytest.raises(ValueError,match='inspection_budget|snapshot_byte_budget|unsafe_private_file'):recovery.inspect(roots,scope)


def test_nonempty_plan_remains_verifiable_and_dispatch_reference_is_retained(installation):
    from tests.service.test_admission import fixture
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture();a=Admission(core,roots['staging']);store=DispatchStore(core,a)
        job=core.submit_synthetic('p','one',f['context'],_admission=a,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        row=store.allocate(f['profiles'],'silence',8)
        dispatch_id=row['seq']
    finally:core.close()
    observation,planned=enrolled_plan(installation)
    assert planned['plan']['affected']==dict(jobs=[job['id']],dispatches=[dispatch_id],publications=[])
    assert recovery.verify_plan(roots,scope,observation,planned['plan']['id'],planned['digest'],now=101)==planned['plan']
    core=Core(roots['control']/'core')
    try:
        scanner=Reconciliation(core,roots['staging']);summary=scanner.reconcile_startup()
        assert summary['complete'] and summary['blocked']
        assert 'unknown_reference_schema' not in summary['reasons']
        assert 'unbound_dispatch' in summary['reasons']
    finally:core.close()


def test_missing_enrollment_refuses_persistent_dispatcher_before_core_creation(tmp_path):
    from service.dispatcher import Dispatcher
    class Hello:
        def call(self,*args):return dict(lane='a'*32,boot='b'*32,quarantined=False)
    root=tmp_path/'missing'
    with pytest.raises(sqlite3.OperationalError):Dispatcher(root,runtime=Hello(),persistent_registration='a'*64)
    assert not root.exists()


def test_partial_file_write_refuses_resume_without_replacement(tmp_path,monkeypatch):
    from service import persistent_init
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values():p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    real=persistent_init.exclusive
    def short(path,data):
        if path.name=='execution.key':
            real(path,data[:3]);raise OSError('partial-write')
        return real(path,data)
    monkeypatch.setattr(persistent_init,'exclusive',short)
    with pytest.raises(OSError):initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET)
    monkeypatch.setattr(persistent_init,'exclusive',real)
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    before=(roots['ipc']/'execution.key').read_bytes()
    with pytest.raises(ValueError,match='partial_initialization_mismatch'):
        initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete=digest(record))
    assert (roots['ipc']/'execution.key').read_bytes()==before


def test_enrolled_root_cannot_bypass_closed_activation_via_diagnostic_default(installation):
    from service.dispatcher import Dispatcher
    roots,_,_,_=installation
    class Hello:
        def call(self,op,body):
            assert op=='hello'
            return dict(lane='a'*32,boot='b'*32,quarantined=False)
    before=logical(roots)
    with pytest.raises(ValueError,match='persistent_registration_required'):
        Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=Hello())
    assert before==logical(roots)


def test_unknown_inventory_refuses_plan_but_is_visible_in_inspect(installation):
    roots,scope,_,_=installation
    observation,_=enrolled_plan(installation)
    (roots['staging']/'unexplained').write_bytes(b'unknown')
    state=recovery.inspect(roots,scope)
    assert state['inventory']['counts']['unknown_foreign']==1
    with pytest.raises(ValueError,match='unknown_recovery_scope'):recovery.plan(roots,scope,observation)


@pytest.mark.parametrize('damage',['lost','empty','intent'])
def test_core_creation_intent_never_remints(tmp_path,damage):
    import shutil
    roots={k:tmp_path/k for k in ('control','staging','receipts','ipc')}
    for p in roots.values(): p.mkdir()
    scope=dict(daemon='d',context='c',volumes={k:dict(name=k,driver='local',created='now') for k in roots})
    def fault(point):
        if point==('core_intent' if damage=='intent' else 'database'): raise OSError('crash')
    with pytest.raises(OSError): initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,fault=fault)
    record=json.loads(read_file(roots['receipts']/'p1-init.json'))
    core=roots['control']/'core'
    if damage!='intent': shutil.rmtree(core)
    if damage=='empty': core.mkdir()
    with pytest.raises(ValueError,match='core.*quarantine'):
        initialize(roots,scope,synthetic_profiles(),DEFAULT_BUDGET,complete=digest(record))
    assert not (core/'core.sqlite').exists()
    assert not (roots['receipts']/'p1-enrolled.json').exists()


@pytest.mark.parametrize('mutation',['delete','content','append'])
def test_plan_binds_other_plan_evidence(installation,mutation):
    roots,scope,_,_=installation
    observation,first=enrolled_plan(installation)
    second=recovery.plan(roots,scope,observation,now=101)
    assert recovery.verify_plan(roots,scope,observation,second['plan']['id'],second['digest'],now=102)==second['plan']
    if mutation=='append': recovery.plan(roots,scope,observation,now=102)
    else:
        with sqlite3.connect(roots['control']/'core/core.sqlite') as db:
            if mutation=='delete': db.execute('DELETE FROM p1_recovery_plans WHERE id=?',(first['plan']['id'],))
            else:
                value=first['plan']; value['expires']-=1
                db.execute('UPDATE p1_recovery_plans SET record=?,digest=? WHERE id=?',(encode(value),digest(value),value['id']))
    with pytest.raises(ValueError,match='stale_plan'):
        recovery.verify_plan(roots,scope,observation,second['plan']['id'],second['digest'],now=103)


@pytest.mark.parametrize('missing_shm,readonly',[(False,False),(True,False),(False,True),(True,True)])
def test_crash_wal_inspection_preserves_source(installation,missing_shm,readonly):
    import subprocess,sys,os,hashlib
    roots,scope,_,_=installation
    directory=roots['control']/'core'; path=directory/'core.sqlite'
    subprocess.run([sys.executable,'-c',"import sqlite3,os,sys; db=sqlite3.connect(sys.argv[1]); db.execute('PRAGMA journal_mode=WAL'); db.execute(\"UPDATE resource_monitor SET state='checking'\"); db.commit(); os._exit(0)",str(path)],check=True)
    if missing_shm: Path(str(path)+'-shm').unlink()
    def files(): return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    before=files()
    if readonly: directory.chmod(0o500)
    try:
        assert recovery.inspect(roots,scope)['monitor']==[(1,'checking')]
        assert files()==before
    finally: directory.chmod(0o700)


def test_private_snapshot_source_change_refuses_and_cleans(installation,monkeypatch):
    roots,scope,_,_=installation
    original=recovery.snapshot
    private_dirs=[]
    def changed(roots,db,scope,**kwargs):
        private_dirs.append(Path(db.execute('PRAGMA database_list').fetchone()[2]).parent)
        result=original(roots,db,scope,**kwargs)
        (roots['control']/'core'/'unexpected').write_bytes(b'changed')
        return result
    monkeypatch.setattr(recovery,'snapshot',changed)
    with pytest.raises(ValueError,match='snapshot_source_changed'): recovery.inspect(roots,scope)
    assert private_dirs and all(not p.exists() for p in private_dirs)
    assert (roots['control']/'core'/'unexpected').read_bytes()==b'changed'


def test_private_snapshot_copy_error_cleans_only_private(installation,monkeypatch):
    roots,scope,_,_=installation
    original=recovery.sqlite3.connect
    private_dirs=[]
    def failed(path,*args,**kwargs):
        from urllib.parse import urlparse,unquote
        private_dirs.append(Path(unquote(urlparse(path).path)).parent)
        raise sqlite3.OperationalError('private-open-injected')
    monkeypatch.setattr(recovery.sqlite3,'connect',failed)
    with pytest.raises(sqlite3.OperationalError,match='private-open-injected'): recovery.inspect(roots,scope)
    monkeypatch.setattr(recovery.sqlite3,'connect',original)
    assert private_dirs and all(not p.exists() for p in private_dirs)
    assert recovery.inspect(roots,scope)['monitor']==[(1,'healthy')]


def test_b2_optional_schema_and_plan_self_binding(installation):
    roots,scope,_,_=installation
    observation,_=enrolled_plan(installation)
    planned=recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=installation[2],budget=DEFAULT_BUDGET,now=100)
    with recovery.offline(roots) as (checked,db):
        assert recovery.verify_abandon_plan(checked,db,scope,observation,planned['plan']['id'],planned['digest'],proof_digest='3'*64,now=101)==planned['plan']
    assert planned['plan']['new_lane']!=planned['plan']['old_lane']
    assert planned['plan']['new_epoch']!=planned['plan']['prior_epoch']
    with pytest.raises(NotImplementedError): recovery.apply(plan_id=planned['plan']['id'],plan_digest=planned['digest'],proof_digest='3'*64)


def test_b2_partial_schema_core_refuses(installation):
    roots,_,_,_=installation
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db, db:
        db.execute('CREATE TABLE p1_b2_format(id INTEGER PRIMARY KEY, revision INTEGER NOT NULL)')
    for historical in (False,True):
        with pytest.raises(ValueError,match='b2'):
            Core(roots['control']/'core',historical_local=historical)


def b2_plan(installation):
    roots,scope,profiles,_=installation
    observation,_=enrolled_plan(installation)
    result=recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    return observation,result


def b2_intent(installation, *, complete=False,recovery_id='9'*32,archive_id='8'*32,prepare_only=False,diagnostic_decisions=None):
    """Test-only evidence construction; no production transition/apply writer."""
    import hashlib
    roots,scope,_,_=installation
    observation,planned=b2_plan(installation);p=planned['plan'];root=roots['control']/'core'
    proof=dict(kind='E1-mock',observation=observation,prior_epoch=p['prior_epoch'],old_lane=p['old_lane'])
    p['proof_digest']=digest(proof)
    if diagnostic_decisions is not None:p['diagnostic_decisions']=diagnostic_decisions
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_abandon_plans SET record=?,digest=? WHERE id=?',(encode(p),digest(p),p['id']))
    with recovery.offline(roots) as (checked,db):
        source=recovery._b2_state(checked,db,scope,exclude_plan=p['id'])['evidence_projection']
        # JSON round-trip canonicalizes SQL tuples into the frozen scalar row lists.
        source=json.loads(encode(source))
        before=dict(source=source,prior_certificate=recovery.completion_binding(db,root),journal=dict(state='valid',bytes_hex=(roots['receipts']/'journal.json').read_bytes().hex()))
        import tempfile
        with tempfile.TemporaryDirectory(prefix='b21-fixture-') as temporary:
            temporary_path=Path(temporary)/'snapshot.sqlite'
            with closing(sqlite3.connect(temporary_path)) as archive:db.backup(archive)
            snapshot_bytes=temporary_path.read_bytes()
    snapshot_path=root/('p1-recovery-'+archive_id+'.sqlite')
    if not prepare_only: snapshot_path.write_bytes(snapshot_bytes);snapshot_path.chmod(0o600)
    jobs=[json.loads(row[6]) for row in source['data']['jobs']]
    actions=[dict(job_id=j['id'],action=('retain-terminal' if j['state'] in ('succeeded','failed','cancelled') else 'cancel-never-started' if not j['attempts'] else 'cancel-requested' if j['state']=='cancel_requested' else 'fail'),before_digest=digest(j)) for j in jobs]
    r=dict(id=recovery_id,deployment_id=p['deployment_id'],plan_id=p['id'],plan_digest=digest(p),
        **{k:p[k] for k in ('source_digest','observation_digest','proof_digest','prior_epoch','new_epoch','old_lane','new_lane','affected','budget_before','budget_after','diagnostic_decisions')},
        proof_kind='E1-mock',before=before,proof=proof,before_digest=digest(before),
        monitor_digest=digest(source['data']['resource_monitor']),faults_digest=digest(source['data']['resource_faults']),
        highwater=source['data']['resource_accounting'][0][1],actions=actions,archive_id=archive_id,expected=[],created=101)
    core_rows={k:copy.deepcopy(source['data'][k]) for k in recovery.CORE_OUTPUT_TABLES}
    for row,action in zip(core_rows['jobs'],actions):
        if action['action']=='retain-terminal':continue
        job=json.loads(row[6]);job['state']='failed' if action['action']=='fail' else 'cancelled'
        if job['attempts']:job['attempts'][-1]['reaped']=True
        row[6]=encode(job)
    for row in core_rows['resource_reservations']:row[3]='retained'
    core_rows['resource_monitor']=[[1,'healthy']];core_rows['resource_faults']=[]
    outputs=[dict(recovery_id=r['id'],plan_digest=r['plan_digest'],proof_digest=r['proof_digest'],before_digest=r['before_digest']),
        dict(archive_id=r['archive_id'],snapshot_digest=hashlib.sha256(snapshot_bytes).hexdigest(),manifest_digest='0'*64,journal_state=p['journal_state'],journal_digest=p['journal_digest']),
        dict(old_lane=r['old_lane'],new_lane=r['new_lane'],journal=dict(version=1,lane=r['new_lane'],records=[])),
        dict(before_digest=r['before_digest'],prior_epoch=r['prior_epoch'],new_epoch=r['new_epoch'],actions=actions,diagnostic_decisions=r['diagnostic_decisions'],highwater=r['highwater'],budget_after=r['budget_after'],monitor_state='healthy',rows=core_rows)]
    # Manifest does not hash itself or the completion certificate.
    r['expected']=[dict(output=output) for output in outputs]
    outputs[1]['manifest_digest']=digest(recovery.archive_manifest(r))
    previous=r['before_digest'];r['expected']=[]
    for name,output in zip(recovery.STEPS,outputs):
        output_digest=digest(dict(step=name,input_digest=previous,output=output))
        r['expected'].append(dict(step=name,input_digest=previous,output_digest=output_digest,output=output));previous=output_digest
    output=dict(recovery_id=r['id'],lane=r['new_lane'],epoch=r['new_epoch'],core_output_digest=previous)
    r['expected'].append(dict(step='receipt-complete',input_digest=previous,output_digest=digest(dict(step='receipt-complete',input_digest=previous,output=output)),output=output))
    if prepare_only: return r
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('INSERT INTO p1_recoveries VALUES(?,?,?,?)',(r['id'],r['plan_id'],encode(r),digest(r)))
        if complete:
            for expected in r['expected'][:4]:
                step=dict(recovery_id=r['id'],**expected,lane=r['new_lane'],epoch=r['new_epoch'])
                db.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',(r['id'],step['step'],encode(step),digest(step)))
    if complete:
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
            for table,records in recovery._resolution_outputs(r).items():
                for record in records:
                    key=record.get('dispatch_seq',record.get('job_id',record.get('object_id')))
                    db.execute('INSERT INTO '+table+' VALUES(?,?,?,?)',(key,r['id'],encode(record),digest(record)))
    for name,kind in recovery.archive_names(r).items():
        if kind=='snapshot':continue
        value=recovery.archive_manifest(r) if kind=='manifest' else recovery._certificate(r,step) if complete else None
        if value is not None:
            path=root/name;path.write_text(encode(value));path.chmod(0o600)
    return r


def b2_receipt(roots,r, *, complete=False):
    cert=None
    if complete:
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db, db: cert=recovery.completion_binding(db)
    marker=dict(revision=1,deployment_id=r['deployment_id'])
    fence=dict(revision=1,deployment_id=r['deployment_id'],recovery_id=r['id'],plan_id=r['plan_id'],
        plan_digest=r['plan_digest'],proof_digest=r['proof_digest'],old_lane=r['old_lane'],new_lane=r['new_lane'],
        step='complete' if complete else 'pending',certificate=cert)
    for name,value in (('p1-b2-required.json',marker),('p1-maintenance.json',fence)):
        path=roots['receipts']/name;path.write_text(encode(value));path.chmod(0o600)
    if complete:
        prefix='p1-recovery-'+r['id']
        steps=[dict(recovery_id=r['id'],**e,lane=r['new_lane'],epoch=r['new_epoch']) for e in r['expected'][:4]]
        path=roots['receipts']/(prefix+'.certificate.json')
        path.write_text(encode(dict(certificate=cert,steps=steps)));path.chmod(0o600)
        path=roots['receipts']/(prefix+'.journal')
        path.write_bytes(bytes.fromhex(r['before']['journal']['bytes_hex']));path.chmod(0o600)
        value=dict(version=1,lane=r['new_lane'],records=[],maintenance=cert)
        (roots['receipts']/'journal.json').write_text(encode(dict(payload=value,sha256=digest(value))))
    return fence


@pytest.mark.parametrize('mutation',['delete','content','append'])
def test_b2_prior_history_is_bound(installation,mutation):
    roots,scope,profiles,_=installation
    observation,first=b2_plan(installation)
    second=recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    with recovery.offline(roots) as (checked,db):
        recovery.verify_abandon_plan(checked,db,scope,observation,second['plan']['id'],second['digest'],proof_digest='3'*64,now=101)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db, db:
        if mutation=='delete': db.execute('DELETE FROM p1_abandon_plans WHERE id=?',(first['plan']['id'],))
        elif mutation=='content':
            p=first['plan'];p['created']=99;p['expires']=399
            db.execute('UPDATE p1_abandon_plans SET record=?,digest=? WHERE id=?',(encode(p),digest(p),p['id']))
    if mutation=='append': recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    with recovery.offline(roots) as (checked,db), pytest.raises(ValueError,match='stale_b2_plan'):
        recovery.verify_abandon_plan(checked,db,scope,observation,second['plan']['id'],second['digest'],proof_digest='3'*64,now=101)


@pytest.mark.parametrize('historical',[False,True])
def test_b2_pending_direct_core_refusal_twice(installation,historical):
    roots,_,_,_=installation;b2_intent(installation)
    before=logical(roots)
    for _ in range(2):
        with pytest.raises(ValueError,match='pending'): Core(roots['control']/'core',historical_local=historical)
        assert logical(roots)==before


@pytest.mark.parametrize('core_state,receipt_state',[(False,'pending'),(False,'absent'),(True,'pending'),(True,'absent'),(True,'complete')])
def test_b2_cross_volume_startup_before_mutation(installation,monkeypatch,core_state,receipt_state):
    from service.dispatcher import Dispatcher
    from worker.runtime_journal import Journal
    roots,_,_,_=installation
    r=b2_intent(installation,complete=core_state)
    if receipt_state!='absent':b2_receipt(roots,r,complete=receipt_state=='complete')
    before=logical(roots)
    class Runtime:
        def call(self,op,body):
            assert op=='hello'
            journal=Journal(roots['receipts'])
            try:return dict(lane=journal.value['lane'],boot='b'*32,quarantined=False,maintenance=journal.maintenance)
            finally:journal.lock.close()
    monkeypatch.setattr('service.dispatcher.Core',lambda *a,**k:pytest.fail('must refuse before mutable Core'))
    for _ in range(2):
        with pytest.raises(ValueError): Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=Runtime())
        assert logical(roots)==before


def test_b2_nonempty_plan_pure_self_verify(installation):
    from tests.service.test_admission import fixture
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture();a=Admission(core,roots['staging']);DispatchStore(core,a)
        job=core.submit_synthetic('p','b2',f['context'],_admission=a,_profiles=f['profiles'],_dispatch_settings=('silence',8))
    finally:core.close()
    observation,result=b2_plan(installation)
    p=result['plan'];assert p['affected']['jobs']==[job['id']] and p['affected']['reservations']==[job['id']]
    with recovery.offline(roots) as (checked,db):
        db.execute('BEGIN')
        before=db.total_changes
        assert recovery.verify_abandon_plan(checked,db,scope,observation,p['id'],result['digest'],proof_digest='3'*64,now=101)==p
        assert db.in_transaction and db.total_changes==before
        db.execute('ROLLBACK')


@pytest.mark.parametrize('mutation',['unknown-field','bool','nan','duplicate','reference','oversize','columns'])
def test_b2_closed_records_refuse(installation,mutation):
    roots,scope,_,_=installation
    _,result=b2_plan(installation);p=result['plan']
    if mutation=='unknown-field':p['patch']={'sql':'no'}
    elif mutation=='bool':p['revision']=True
    elif mutation=='nan':p['created']=float('nan')
    elif mutation=='duplicate':p['affected']['jobs']=['x','x'];p['count']=2
    elif mutation=='reference':p['affected']['jobs']=['unknown'];p['count']=1
    elif mutation=='oversize':p['id']='x'*(recovery.MAX_BYTES+1)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db, db:
        if mutation=='columns':db.execute('ALTER TABLE p1_abandon_plans ADD COLUMN extra TEXT')
        else:
            raw=json.dumps(p) if mutation=='nan' else encode(p)
            h='0'*64 if mutation=='nan' else digest(p)
            db.execute('UPDATE p1_abandon_plans SET record=?,digest=?',(raw,h))
    with pytest.raises((ValueError,OSError)):
        recovery.inspect(roots,scope)


def terminal_enrolled(installation):
    from tests.service.test_admission import fixture
    from tests.service.test_dispatcher import Runtime
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation
    lane=recovery.inspect(roots,scope)['lane']
    core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        job=core.submit_synthetic('p','terminal-b2',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        row=store.allocate(f['profiles'],'silence',8)
        runtime=Runtime(roots['staging']);runtime.prepares=1
        identity=lane+'/'+'b'*32+'/one'
        receipt,observation=runtime.collect(row['authority'],identity)
        store.update(row,runtime=identity,receipt=encode(receipt),phase='receipt')
        core._publish(job['id'],dict(message=observation.message,manifest=observation.manifest),stage_root=roots['staging'],engine_uid=receipt['engine_uid'])
        with admission._preflight(),core.transaction():
            admission.store.release_tx(job['id'],confirmed_reap=True)
            core.db.execute("UPDATE dispatches SET outcome=?,phase='done',acked=1 WHERE seq=?",(encode(core.job(job['id'])),row['seq']))
        return job['id']
    finally:core.close()


@pytest.mark.parametrize('method',['_publish','_commit_publication'])
def test_b2_retirement_precedes_both_publication_bodies(installation,monkeypatch,method):
    from service import persistence
    roots,_,_,_=installation
    job_id=terminal_enrolled(installation)
    r=b2_intent(installation,complete=True)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        publication=db.execute('SELECT * FROM publications WHERE job_id=?',(job_id,)).fetchone()
        work=db.execute('SELECT * FROM publication_work WHERE job_id=?',(job_id,)).fetchone()
        job=json.loads(db.execute('SELECT job FROM jobs WHERE id=?',(job_id,)).fetchone()[0])
        value=dict(job_id=job_id,recovery_id=r['id'],attempt_id=job['attempts'][-1]['id'],binding=digest(json.loads(work[1])),
            publication_digest=digest(publication),work_digest=digest(work),decision='retired')
        db.execute('INSERT OR REPLACE INTO p1_retired_publications VALUES(?,?,?,?)',(job_id,r['id'],encode(value),digest(value)))
    before=logical(roots)
    files={p.name:p.read_bytes() for p in (roots['control']/'core'/'assets').iterdir()}
    core=Core(roots['control']/'core',historical_local=True)
    try:
        monkeypatch.setattr(core,'_verified',lambda *a,**k:pytest.fail('verification must not run'))
        monkeypatch.setattr(core,'_load',lambda *a,**k:pytest.fail('even terminal fast return must not run'))
        with pytest.raises(persistence.PublicationQuarantine,match='retired'):
            getattr(core,method)(job_id, {}) if method=='_publish' else core._commit_publication(job_id)
        assert before==logical(roots)
        assert files=={p.name:p.read_bytes() for p in (roots['control']/'core'/'assets').iterdir()}
    finally:core.close()


@pytest.mark.parametrize('damage',['missing','unknown','digest','lane','pending','journal-binding','all-fences-missing','archive-missing','archive-journal','archive-chain','archive-certificate'])
def test_b2_worker_refuses_before_boot_or_listen(installation,monkeypatch,damage):
    import worker.runtime_supervisor as module
    roots,_,_,_=installation
    r=b2_intent(installation,complete=True);fence=b2_receipt(roots,r,complete=True)
    if damage.startswith('archive-'):
        prefix=roots['receipts']/('p1-recovery-'+r['id'])
        path=Path(str(prefix)+'.certificate.json')
        if damage=='archive-missing':path.unlink()
        elif damage=='archive-journal':Path(str(prefix)+'.journal').write_bytes(b'changed')
        else:
            archived=json.loads(path.read_text())
            if damage=='archive-chain':archived['steps'][1]['input_digest']='a'*64
            else:archived['certificate']['certificate_digest']='a'*64
            path.write_text(encode(archived))
    elif damage=='missing':(roots['receipts']/'p1-maintenance.json').unlink()
    elif damage=='all-fences-missing':
        (roots['receipts']/'p1-maintenance.json').unlink();(roots['receipts']/'p1-b2-required.json').unlink()
    elif damage=='journal-binding':
        value=dict(version=1,lane=r['new_lane'],records=[])
        (roots['receipts']/'journal.json').write_text(encode(dict(payload=value,sha256=digest(value))))
    else:
        if damage=='unknown':fence['argv']=[]
        elif damage=='digest':fence['certificate']['certificate_digest']='x'
        elif damage=='lane':fence['certificate']['lane']='f'*32
        elif damage=='pending':fence.update(step='pending',certificate=None)
        (roots['receipts']/'p1-maintenance.json').write_text(encode(fence))
    before=(roots['receipts']/'journal.json').read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module.os,'getpid',lambda:1)
        patch.setattr(module.secrets,'token_hex',lambda *a:pytest.fail('boot minted before fence'))
        patch.setattr(module.socket,'socket',lambda *a:pytest.fail('listen before fence'))
        for _ in range(2):
            with pytest.raises(ValueError):module.RuntimeSupervisor(roots['receipts'],roots['ipc'])
            assert (roots['receipts']/'journal.json').read_bytes()==before


def test_b2_worker_completed_hello_matches_core(installation):
    from worker.runtime_journal import Journal
    from worker.runtime_supervisor import RuntimeSupervisor
    roots,_,_,_=installation
    r=b2_intent(installation,complete=True);fence=b2_receipt(roots,r,complete=True)
    runtime=RuntimeSupervisor.__new__(RuntimeSupervisor)
    runtime.journal=Journal(roots['receipts']);runtime.boot='b'*32;runtime.failed=False;runtime.active=None;runtime.launches=0
    try:
        hello=runtime.operation('hello',{})
        assert hello['maintenance']==fence['certificate'] and hello['lane']==r['new_lane']
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
            assert recovery.check_completion_hello(db,hello)==fence['certificate']
    finally:runtime.journal.lock.close()


def test_b2_nonempty_publication_self_reference_and_namespace(installation):
    from service.reconciliation import Reconciliation
    from types import SimpleNamespace
    roots,scope,_,_=installation
    job_id=terminal_enrolled(installation)
    observation,result=b2_plan(installation)
    with recovery.offline(roots) as (checked,db):
        p=recovery.verify_abandon_plan(checked,db,scope,observation,result['plan']['id'],result['digest'],proof_digest='3'*64,now=101)
        assert p['affected']['publications']==[job_id]
        scanner=Reconciliation.__new__(Reconciliation)
        scanner.core=SimpleNamespace(root=roots['control']/'core');scanner.db=db;scanner.staging=roots['staging']
        scanner.entry_limit=2048;scanner.byte_limit=16*1024**2;scanner.seconds=1;scanner._collect()
        media=next(r for r in scanner.rows.values() if r['namespace']=='assets')
        assert 'p1_abandon_plans:'+p['id'] in media['references']


@pytest.mark.parametrize('core_state',['absent','pending','complete'])
def test_b2_receipt_pending_always_blocks_two_restarts(installation,core_state):
    from service.dispatcher import Dispatcher
    from worker.runtime_journal import Journal
    roots,_,_,_=installation
    r=b2_intent(installation,complete=core_state=='complete')
    if core_state=='absent':
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute('DELETE FROM p1_recoveries')
    b2_receipt(roots,r)
    before=logical(roots)
    class Worker:
        def call(self,op,body):
            assert op=='hello'
            journal=Journal(roots['receipts'])
            journal.lock.close()
            pytest.fail('pending worker cannot hello')
    for _ in range(2):
        with pytest.raises(ValueError):Dispatcher(roots['control']/'core',runtime=Worker())
        assert before==logical(roots)


def test_b2_schema_install_is_atomic_and_configuration_bound(installation,monkeypatch):
    from service import persistence
    roots,scope,profiles,_=installation
    observation,_=enrolled_plan(installation);before=logical(roots)
    with pytest.raises(ValueError,match='configuration'):
        recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget={**DEFAULT_BUDGET,'disk_bytes':1},now=100)
    assert before==logical(roots)
    monkeypatch.setattr(persistence,'B2_DDL',(*persistence.B2_DDL[:3],'INVALID DDL'))
    with pytest.raises(sqlite3.Error):
        recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    assert before==logical(roots)


def test_b2_archive_namespace_is_closed_retained_evidence(installation):
    from types import SimpleNamespace
    roots,_,_,_=installation
    intent=b2_intent(installation,complete=True)
    root=roots['control']/'core'
    manifest=recovery.archive_manifest(intent)
    name=next(n for n,k in recovery.archive_names(intent).items() if k=='manifest')
    path=root/name;path.write_text(encode(manifest));path.chmod(0o600)
    with closing(sqlite3.connect(root/'core.sqlite')) as db:
        scanner=Reconciliation.__new__(Reconciliation);scanner.core=SimpleNamespace(root=root);scanner.db=db
        scanner.staging=roots['staging'];scanner.entry_limit=2048;scanner.byte_limit=16*1024**2;scanner.seconds=1
        scanner._collect()
        evidence=next(row for row in scanner.rows.values() if row['namespace']=='recovery')
        assert evidence['classification']=='recovery_evidence'
        assert evidence['references']==['p1_recoveries:'+intent['id']]
        manifest['unknown']=True;path.write_text(encode(manifest))
        with pytest.raises(ValueError,match='archive_manifest'):scanner._collect()


@pytest.mark.parametrize('installed',[False,True])
def test_b2_unexplained_execution_refuses_without_mutation(installation,installed):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,scope,profiles,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture(); admission=Admission(core,roots['staging']);DispatchStore(core,admission)
        job=core.submit_synthetic('p','unexplained',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
    finally:core.close()
    observation,_=enrolled_plan(installation)
    planned=None
    if installed:
        planned=recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute("UPDATE resource_reservations SET state='executing' WHERE id=?",(job['id'],))
    before=logical(roots)
    with pytest.raises(ValueError,match='incoherent_b2'):
        recovery.plan_abandon(roots,scope,observation,proof_digest='3'*64,profiles=profiles,budget=DEFAULT_BUDGET,now=100)
    assert logical(roots)==before
    if installed:
        with recovery.offline(roots) as (checked,db),pytest.raises(ValueError,match='incoherent_b2'):
            recovery.verify_abandon_plan(checked,db,scope,observation,planned['plan']['id'],planned['digest'],proof_digest='3'*64,now=101)
        assert logical(roots)==before


@pytest.mark.parametrize('field',['before_digest','monitor_digest','faults_digest','action','output','archive-missing','archive-bytes','certificate'])
def test_b2_completion_requires_grounded_evidence(installation,field):
    roots,_,_,_=installation
    terminal_enrolled(installation)
    r=b2_intent(installation,complete=True)
    root=roots['control']/'core'
    if field in ('before_digest','monitor_digest','faults_digest'):r[field]='a'*64
    elif field=='action':r['actions'][0]['before_digest']='a'*64
    elif field=='output':
        r['expected'][2]['output_digest']='a'*64;r['expected'][3]['input_digest']='a'*64
    else:
        names=recovery.archive_names(r)
        kind='certificate' if field=='certificate' else 'snapshot'
        path=root/next(n for n,k in names.items() if k==kind)
        if field=='archive-missing':path.unlink(missing_ok=True)
        else:path.write_bytes(b'changed');path.chmod(0o600)
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_recoveries SET record=?,digest=?',(encode(r),digest(r)))
    with pytest.raises((ValueError,OSError)):
        core=Core(root);core.close()


@pytest.mark.parametrize('runtime',[None,'unknown'])
def test_b2_coherent_executing_unbound_and_unknown_plan(installation,runtime):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,scope,_,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture();a=Admission(core,roots['staging']);store=DispatchStore(core,a)
        core.submit_synthetic('p','allocated',f['context'],_admission=a,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        row=store.allocate(f['profiles'],'silence',8)
        if runtime:
            lane=json.loads((roots['receipts']/'journal.json').read_text())['payload']['lane']
            store.update(row,runtime=lane+'/'+'b'*32+'/unknown',phase='prepared')
    finally:core.close()
    observation,planned=b2_plan(installation)
    with recovery.offline(roots) as (checked,db):
        assert recovery.verify_abandon_plan(checked,db,scope,observation,planned['plan']['id'],planned['digest'],proof_digest='3'*64,now=101)==planned['plan']


@pytest.mark.parametrize('field',[
    'source-monitor','source-faults','source-highwater','source-job','source-history','before-journal',
    'proof-observation','proof-kind','prior-certificate',
    'pending-output','evidence-output','journal-output','core-output','receipt-output',
    'manifest-missing','certificate-missing','manifest-source','manifest-snapshot','manifest-journal',
    'certificate-digest','certificate-epoch','snapshot-logical'])
def test_b2_bound_documents_refuse_valid_row_digest_tampering(installation,field):
    roots,_,_,_=installation;terminal_enrolled(installation)
    r=b2_intent(installation,complete=True);root=roots['control']/'core'
    names={kind:root/name for name,kind in recovery.archive_names(r).items()}
    source=r['before']['source']['data']
    if field=='before-journal':r['before']['journal']['bytes_hex']='00'
    elif field=='source-monitor':source['resource_monitor'][0][1]='checking'
    elif field=='source-faults':source['resource_faults'].append(['changed'])
    elif field=='source-highwater':source['resource_accounting'][0][1]+=1
    elif field=='source-job':
        job=json.loads(source['jobs'][0][6]);job['generation']+=1;source['jobs'][0][6]=encode(job)
    elif field=='source-history':source['p1_recovery_plans']=[]
    elif field=='proof-observation':r['proof']['observation']['states'][0]['finished']='changed'
    elif field=='proof-kind':r['proof']['kind']='host-admin-container-teardown-v1'
    elif field=='prior-certificate':r['before']['prior_certificate']={}
    elif field.endswith('-output'):
        index=['pending','evidence','journal','core','receipt'].index(field[:-7]);output=r['expected'][index]['output']
        key=['before_digest','snapshot_digest','new_lane','highwater','core_output_digest'][index]
        output[key]=output[key]+1 if type(output[key])==int else 'a'*len(output[key])
    elif field.endswith('-missing'):names[field[:-8]].unlink()
    elif field.startswith('manifest-'):
        value=json.loads(names['manifest'].read_text());value[{'manifest-source':'source_digest','manifest-snapshot':'snapshot_digest','manifest-journal':'journal_digest'}[field]]='a'*64
        names['manifest'].write_text(encode(value))
    elif field.startswith('certificate-'):
        value=json.loads(names['certificate'].read_text());key='epoch' if field.endswith('epoch') else 'certificate_digest';value[key]='a'*len(value[key]);names['certificate'].write_text(encode(value))
    elif field=='snapshot-logical':
        with closing(sqlite3.connect(names['snapshot'])) as db,db:db.execute("UPDATE resource_monitor SET state='checking'")
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_recoveries SET record=?,digest=?',(encode(r),digest(r)))
    for reader in ('completion','startup','core'):
        with pytest.raises((ValueError,OSError,sqlite3.Error)):
            if reader=='completion':
                with closing(sqlite3.connect(root/'core.sqlite')) as db:recovery.completion_binding(db)
            elif reader=='startup':recovery.startup_read(root)
            else:
                core=Core(root,historical_local=True);core.close()


@pytest.mark.parametrize('field',['snapshot','core-output'])
def test_b2_rehashed_documents_still_require_grounding(installation,field):
    import hashlib
    roots,_,_,_=installation;terminal_enrolled(installation)
    r=b2_intent(installation,complete=True);root=roots['control']/'core'
    names={kind:root/name for name,kind in recovery.archive_names(r).items()}
    if field=='snapshot':
        with closing(sqlite3.connect(names['snapshot'])) as db,db:db.execute("UPDATE resource_monitor SET state='checking'")
        r['expected'][1]['output']['snapshot_digest']=hashlib.sha256(names['snapshot'].read_bytes()).hexdigest()
        r['expected'][1]['output']['manifest_digest']=digest(recovery.archive_manifest(r))
    else:
        rows=r['expected'][3]['output']['rows']
        job=json.loads(rows['jobs'][0][6]);job['generation']+=1;rows['jobs'][0][6]=encode(job)
    previous=r['before_digest']
    for descriptor in r['expected']:
        if descriptor['step']=='receipt-complete':descriptor['output']['core_output_digest']=previous
        descriptor['input_digest']=previous
        descriptor['output_digest']=digest(dict(step=descriptor['step'],input_digest=previous,output=descriptor['output']))
        previous=descriptor['output_digest']
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_recoveries SET record=?,digest=?',(encode(r),digest(r)))
        for descriptor in r['expected'][:4]:
            step=dict(recovery_id=r['id'],**descriptor,lane=r['new_lane'],epoch=r['new_epoch'])
            db.execute('UPDATE p1_recovery_steps SET record=?,digest=? WHERE step=?',(encode(step),digest(step),step['step']))
    names['manifest'].write_text(encode(recovery.archive_manifest(r)))
    names['certificate'].write_text(encode(recovery._certificate(r,step)))
    with closing(sqlite3.connect(root/'core.sqlite')) as db,pytest.raises(ValueError):recovery.completion_binding(db)


@pytest.mark.parametrize('damage',['none','old-archive','old-history'])
def test_b2_second_completion_retains_prior_evidence(installation,damage):
    roots,_,_,_=installation;root=roots['control']/'core'
    first=b2_intent(installation,complete=True);b2_receipt(roots,first,complete=True)
    second=b2_intent(installation,complete=True,recovery_id='7'*32,archive_id='6'*32)
    b2_receipt(roots,second,complete=True)
    with closing(sqlite3.connect(root/'core.sqlite')) as db:
        assert recovery.completion_binding(db)['recovery_id']==second['id']
    assert second['before']['prior_certificate']['recovery_id']==first['id']
    if damage=='none':
        core=Core(root);core.close();return
    if damage=='old-archive':
        name=next(n for n,k in recovery.archive_names(first).items() if k=='manifest');(root/name).unlink()
    else:
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
            db.execute('DELETE FROM p1_recovery_plans')
    with pytest.raises((ValueError,OSError)):Core(root)



def test_b2_cancel_requested_action_uses_frozen_job_state(installation):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,_,_,_=installation;root=roots['control']/'core'
    core=Core(root)
    try:
        f=fixture();a=Admission(core,roots['staging']);store=DispatchStore(core,a)
        job=core.submit_synthetic('p','cancel-contract',f['context'],_admission=a,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        store.allocate(f['profiles'],'silence',8)
        core.request_cancel(job['id'])
    finally:core.close()
    intent=b2_intent(installation)
    assert intent['actions'][0]['action']=='cancel-requested'
    with recovery.offline(roots) as (checked,db):
        recovery.validate_b2_database(db,root)
    # Only the record is varied; no recovery transitions run in this fixture.
    intent['actions'][0]['action']='fail'
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_recoveries SET record=?,digest=?',(encode(intent),digest(intent)))
    with closing(sqlite3.connect(root/'core.sqlite')) as db,pytest.raises(ValueError,match='action_before_binding'):
        recovery.validate_b2_database(db,root)


@pytest.mark.parametrize('cancel',[False,True])
@pytest.mark.parametrize('field',['none','deleted-attempts','attempt-id','attempt-generation','owner','staging','unknown-job','unknown-attempt','stage','artifacts','input','principal','context','unreaped','state','generation','deadline'])
def test_b2_rehashed_nonterminal_output_preserves_contract(installation,cancel,field):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,_,_,_=installation;root=roots['control']/'core'
    core=Core(root)
    try:
        f=fixture();a=Admission(core,roots['staging']);store=DispatchStore(core,a)
        job=core.submit_synthetic('p','output-contract',f['context'],_admission=a,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        store.allocate(f['profiles'],'silence',8)
        if cancel:core.request_cancel(job['id'])
    finally:core.close()
    r=b2_intent(installation,complete=True)
    row=r['expected'][3]['output']['rows']['jobs'][0];job=json.loads(row[6])
    if field=='deleted-attempts':job['attempts']=[]
    elif field in ('attempt-id','attempt-generation','owner','staging'):
        key={'attempt-id':'id','attempt-generation':'generation','owner':'owner_id','staging':'staging_id'}[field]
        job['attempts'][0][key]=job['attempts'][0][key]+1 if key=='generation' else 'changed'
    elif field=='unknown-job':job['invented_field']={'arbitrary':['unvalidated']}
    elif field=='unknown-attempt':job['attempts'][0]['invented_field']={'arbitrary':['unvalidated']}
    elif field=='stage':job['stage']='invented-outcome'
    elif field=='artifacts':job['artifact_ids']=['invented-artifact']
    elif field=='input':job['input_sha256']='a'*64
    elif field=='principal':job['principal_id']='changed'
    elif field=='context':
        context=json.loads(row[7]);context['invented_field']={'arbitrary':['unvalidated']};row[7]=encode(context)
    elif field=='unreaped':job['attempts'][-1]['reaped']=False
    elif field=='state':job['state']='failed' if cancel else 'cancelled'
    elif field=='generation':job['generation']+=1
    elif field=='deadline':job['deadline']='changed'
    row[6]=encode(job)
    previous=r['before_digest']
    for descriptor in r['expected']:
        if descriptor['step']=='receipt-complete':descriptor['output']['core_output_digest']=previous
        descriptor['input_digest']=previous
        descriptor['output_digest']=digest(dict(step=descriptor['step'],input_digest=previous,output=descriptor['output']))
        previous=descriptor['output_digest']
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE p1_recoveries SET record=?,digest=?',(encode(r),digest(r)))
        for descriptor in r['expected'][:4]:
            step=dict(recovery_id=r['id'],**descriptor,lane=r['new_lane'],epoch=r['new_epoch'])
            db.execute('UPDATE p1_recovery_steps SET record=?,digest=? WHERE step=?',(encode(step),digest(step),step['step']))
    name=next(n for n,k in recovery.archive_names(r).items() if k=='certificate')
    (root/name).write_text(encode(recovery._certificate(r,step)))
    for reader in ('completion','startup','historical'):
        def read():
            if reader=='completion':
                with closing(sqlite3.connect(root/'core.sqlite')) as db:return recovery.completion_binding(db)
            if reader=='startup':return recovery.startup_read(root)
            core=Core(root,historical_local=True);core.close();return True
        if field=='none':assert read() is not None
        else:
            with pytest.raises(ValueError):read()


def prefix_fixture(installation):
    intent=b2_intent(installation,prepare_only=True)
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'snapshot.sqlite'
        with recovery.offline(installation[0]) as (_,db), closing(sqlite3.connect(path)) as target:db.backup(target)
        raw=path.read_bytes()
    return intent,intent['proof']['observation'],raw


def test_b22_prefix_pending_and_exact_expired_resume(installation):
    roots,scope,_,_=installation
    intent,observation,evidence=prefix_fixture(installation)
    result=recovery._pending_through_evidence(roots,scope,intent,observation,now=102)
    assert result['status']=='pending-through-evidence'
    root=roots['control']/'core'
    with sqlite3.connect(root/'core.sqlite') as db:
        for table,rows in intent['before']['source']['data'].items():
            if table in ('p1_recoveries','p1_recovery_steps','p1_abandon_plans'):continue
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(rows)
        assert db.execute('SELECT step FROM p1_recovery_steps ORDER BY rowid').fetchall()==[('pending',),('evidence',)]
    original={p:p.read_bytes() for p in root.glob('p1-recovery-*')}
    journal=(roots['receipts']/'journal.json').read_bytes()
    for _ in range(2):
        with pytest.raises(ValueError,match='pending'):Core(root,historical_local=True)
        from worker.runtime_journal import Journal
        with pytest.raises(ValueError,match='pending'):Journal(roots['receipts'])
        assert recovery._pending_through_evidence(roots,scope,intent,observation,now=1000)==result
    assert all(p.read_bytes()==raw for p,raw in original.items())
    assert (roots['receipts']/'journal.json').read_bytes()==journal
    with pytest.raises(NotImplementedError):recovery.apply(intent=intent)


@pytest.mark.parametrize('point',[
    'accepted-write','accepted-fsync','accepted-parent','fence-write','fence-fsync','fence-parent','core-intent-begin','core-intent-write','core-intent-commit','core-intent-parent',
    'pending-receipt-begin','pending-receipt-write','pending-receipt-commit',
    'snapshot-write','snapshot-fsync','snapshot-parent','manifest-parent','original-journal-parent','evidence-receipt-begin',
    'evidence-receipt-write','evidence-receipt-commit'])
def test_b22_prefix_crash_resume(installation,point):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    journal=(roots['receipts']/'journal.json').read_bytes()
    def crash(at):
        if at==point:raise OSError('injected-'+point)
    with pytest.raises(OSError,match='injected'):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence,fault=crash)
    assert recovery._pending_through_evidence(roots,scope,intent,observation,now=1000)['status']=='pending-through-evidence'
    assert (roots['receipts']/'journal.json').read_bytes()==journal


@pytest.mark.parametrize('point',['fence-create','marker-create','marker-write','marker-fsync','marker-parent','accepted-create','snapshot-intent-recorded','snapshot-create','manifest-intent-recorded','original-journal-intent-recorded'])
def test_b22_prefix_partial_evidence_never_remints(installation,point):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    def crash(at):
        if at==point:raise OSError('injected')
    with pytest.raises(OSError):recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence,fault=crash)
    files={p:p.read_bytes() for directory in (roots['control']/'core',roots['receipts']) for p in directory.glob('p1-*') if p.is_file()}
    with pytest.raises((ValueError,FileNotFoundError)):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)
    assert all(p.read_bytes()==raw for p,raw in files.items())


@pytest.mark.parametrize('change',['job','history','media','journal','proof','archive','extra','intent','lost-fence','lost-creation'])
def test_b22_prefix_changed_resume_refuses(installation,change):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
    root=roots['control']/'core'
    if change=='job':
        with sqlite3.connect(root/'core.sqlite') as db:db.execute("UPDATE resource_monitor SET state='checking'")
    elif change=='history':
        with sqlite3.connect(root/'core.sqlite') as db:db.execute('DELETE FROM p1_recovery_plans')
    elif change=='media':(roots['staging']/'unknown').write_bytes(b'x')
    elif change=='journal':(roots['receipts']/'journal.json').write_bytes(b'changed')
    elif change=='proof':observation=copy.deepcopy(observation);observation['states'][0]['finished']='new'
    elif change=='archive':next(root.glob('*.manifest.json')).write_bytes(b'{}')
    elif change=='extra':(root/'p1-recovery-unexpected.sqlite').write_bytes(b'extra')
    elif change=='lost-fence':(roots['receipts']/'p1-maintenance.json').unlink()
    elif change=='lost-creation':(roots['receipts']/('p1-recovery-'+intent['id']+'.snapshot.intent.json')).unlink()
    else:intent=copy.deepcopy(intent);intent['created']+=1
    with pytest.raises((ValueError,KeyError,json.JSONDecodeError)):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)


def test_b22_prefix_unaccepted_expiry_and_locks(installation):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    with pytest.raises(ValueError,match='expired'):recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)
    for path in (roots['control']/'core'/'control.lock',roots['receipts']/'lane.lock'):
        with path.open() as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with pytest.raises(ValueError,match='lock_busy'):recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
    assert not (roots['receipts']/'p1-b2-required.json').exists()


@pytest.mark.parametrize('journal_tranche',[False,True,'core'])
@pytest.mark.parametrize('journal_state',['missing','unreadable'])
@pytest.mark.parametrize('coverage',['exact','missing-owner','live'])
def test_b22_prefix_damaged_journal_preserves_status_and_coverage(installation,journal_state,coverage,journal_tranche):
    import hashlib,tempfile
    roots,scope,_,_=installation
    if journal_tranche=='core':
        # SQLite's transaction context does not close the source handle. Close it
        # before private-copy entry; rebinding db inside that context could
        # otherwise checkpoint/unlink the original WAL/SHM during inspection.
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    intent,observation,_=prefix_fixture(installation)
    journal=roots['receipts']/'journal.json'
    raw=b'' if journal_state=='missing' else b'original corrupt journal bytes'
    if journal_state=='missing':journal.unlink()
    else:journal.write_bytes(raw)
    intent['before']['journal']=dict(state=journal_state,bytes_hex=raw.hex())
    output=intent['expected'][1]['output'];output.update(journal_state=journal_state,journal_digest=hashlib.sha256(raw).hexdigest())
    with recovery.offline(roots) as (checked,db):
        state=recovery.snapshot(checked,db,scope,exclude_abandon=intent['plan_id'],_prefix=intent)
        assert state['journal_records'] is None
        source=json.loads(encode(state['evidence_projection']))
    intent['before']['source']=source;intent['source_digest']=digest(source);intent['before_digest']=digest(intent['before'])
    root=roots['control']/'core'
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        p=json.loads(db.execute('SELECT record FROM p1_abandon_plans WHERE id=?',(intent['plan_id'],)).fetchone()[0])
        p.update(source_digest=intent['source_digest'],journal_state=journal_state,journal_digest=output['journal_digest'])
        db.execute('UPDATE p1_abandon_plans SET record=?,digest=? WHERE id=?',(encode(p),digest(p),p['id']))
    intent['plan_digest']=digest(p)
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'snapshot.sqlite'
        with recovery.offline(roots) as (_,db),closing(sqlite3.connect(path)) as target:db.backup(target)
        evidence=path.read_bytes()
    output['snapshot_digest']=hashlib.sha256(evidence).hexdigest()
    output['manifest_digest']=digest(recovery.archive_manifest(intent))
    intent['expected'][0]['output'].update(plan_digest=intent['plan_digest'],before_digest=intent['before_digest'])
    intent['expected'][3]['output']['before_digest']=intent['before_digest']
    previous=intent['before_digest']
    for descriptor in intent['expected']:
        if descriptor['step']=='receipt-complete':descriptor['output']['core_output_digest']=previous
        descriptor['input_digest']=previous
        descriptor['output_digest']=digest(dict(step=descriptor['step'],input_digest=previous,output=descriptor['output']))
        previous=descriptor['output_digest']
    if coverage!='exact':
        observation=copy.deepcopy(observation)
        if coverage=='missing-owner':observation['containers'].pop()
        else:observation['states'][0]['running']=True
        with pytest.raises(ValueError):
            if journal_tranche=='core':recovery._pending_through_core(roots,scope,intent,observation,now=102)
            elif journal_tranche:recovery._pending_through_journal(roots,scope,intent,observation,now=102)
            else:recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
        assert not (roots['receipts']/'p1-maintenance.json').exists()
        return
    for _ in range(2):
        if journal_tranche=='core':
            assert recovery._pending_through_core(roots,scope,intent,observation,now=102)['status']=='core-complete-receipt-pending'
        elif journal_tranche:
            assert recovery._pending_through_journal(roots,scope,intent,observation,now=102)['status']=='pending-through-journal'
            assert json.loads(journal.read_bytes())['payload']==intent['expected'][2]['output']['journal']
        else:
            assert recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)['status']=='pending-through-evidence'
            assert journal.exists()==(journal_state!='missing')
    archive=roots['receipts']/('p1-recovery-'+intent['id']+'.journal')
    assert archive.read_bytes()==raw
    manifest=json.loads(next(root.glob('*.manifest.json')).read_bytes())
    assert manifest['journal_state']==journal_state and manifest['journal']['bytes_hex']==raw.hex()


@pytest.mark.parametrize('change',['later-plan','changed-plan','schema','receipt-extra','affected','mock-to-host','missing-created-archive'])
def test_b22_prefix_exact_inputs_and_archive_loss(installation,change):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    root=roots['control']/'core'
    if change=='missing-created-archive':
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
        (root/next(n for n,k in recovery.archive_names(intent).items() if k=='snapshot')).unlink()
    elif change=='later-plan':recovery.plan_abandon(roots,scope,observation,proof_digest=intent['proof_digest'],profiles=installation[2],budget=DEFAULT_BUDGET,now=101)
    elif change=='changed-plan':
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
            p=json.loads(db.execute('SELECT record FROM p1_abandon_plans').fetchone()[0]);p['expires']-=1
            db.execute('UPDATE p1_abandon_plans SET record=?,digest=?',(encode(p),digest(p)))
    elif change=='schema':
        with sqlite3.connect(root/'core.sqlite') as db:db.execute('CREATE TABLE unexpected(id)')
    elif change=='receipt-extra':(roots['receipts']/'unexpected').write_bytes(b'foreign')
    elif change=='affected':intent['affected']['jobs']=['invented']
    else:intent['proof_kind']='host-admin-container-teardown-v1'
    with pytest.raises((ValueError,FileNotFoundError)):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)


@pytest.mark.parametrize('journal_tranche',[False,True])
def test_b22_prefix_real_allocated_work_is_not_transitioned(installation,journal_tranche):
    # Same real allocation seam as the B2.1 coherence/output regressions.
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots=installation[0]
    core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        job=core.submit_synthetic('p','prefix-cancel',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        store.allocate(f['profiles'],'silence',8);core.request_cancel(job['id'])
    finally:core.close()
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    assert any(a['action']=='cancel-requested' for a in intent['actions'])
    (recovery._pending_through_journal(roots,scope,intent,observation,now=102) if journal_tranche else recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence))
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        for table in ('jobs','dispatches','resource_reservations','resource_accounting','resource_monitor','resource_faults','events','publications'):
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])


@pytest.mark.parametrize('barrier',['file','parent'])
@pytest.mark.parametrize('kind',['snapshot','bundle'])
def test_b22_prefix_actual_barrier_failure_preserves_output(installation,monkeypatch,barrier,kind):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    original_fsync=recovery.os.fsync;original_sync=recovery.sync_dir
    def failed(*args):raise OSError('actual-barrier-failure')
    def arm(point):
        if point==(kind+'-write' if barrier=='file' else kind+'-fsync'):
            if barrier=='file':monkeypatch.setattr(recovery.os,'fsync',failed)
            else:monkeypatch.setattr(recovery,'sync_dir',failed)
    with pytest.raises(OSError,match='actual-barrier-failure'):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence,fault=arm)
    monkeypatch.setattr(recovery.os,'fsync',original_fsync);monkeypatch.setattr(recovery,'sync_dir',original_sync)
    path=next(roots['control'].glob('core/p1-recovery-*.sqlite')) if kind=='snapshot' else next(roots['receipts'].glob('*.snapshot.bundle'))
    identity=path.stat().st_ino;raw=path.read_bytes()
    recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)
    assert path.stat().st_ino==identity and path.read_bytes()==raw


@pytest.mark.parametrize('damage',['none','prior-history','prior-archive'])
def test_b22_prefix_retains_prior_completed_history(installation,damage):
    import tempfile
    roots,scope,_,_=installation;root=roots['control']/'core'
    first=b2_intent(installation,complete=True);b2_receipt(roots,first,complete=True)
    intent=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    observation=intent['proof']['observation']
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'snapshot.sqlite'
        with recovery.offline(roots) as (_,db),closing(sqlite3.connect(path)) as target:db.backup(target)
        evidence=path.read_bytes()
    originals={root/n:(root/n).read_bytes() for n in recovery.archive_names(first)}
    if damage=='prior-history':
        with sqlite3.connect(root/'core.sqlite') as db:db.execute('DELETE FROM p1_recovery_plans')
    elif damage=='prior-archive':next(p for p in originals if p.suffix=='.json').unlink()
    if damage!='none':
        with pytest.raises((ValueError,FileNotFoundError)):
            recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
        return
    result=recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence)
    assert result['status']=='pending-through-evidence'
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)==result


@pytest.mark.parametrize('journal_tranche',[False,True])
def test_b22_prefix_terminal_media_and_history_unchanged(installation,journal_tranche):
    terminal_enrolled(installation)
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    root=roots['control']/'core'
    media={p:p.read_bytes() for p in (root/'assets').iterdir() if p.is_file()}
    assert media
    (recovery._pending_through_journal(roots,scope,intent,observation,now=102) if journal_tranche else recovery._pending_through_evidence(roots,scope,intent,observation,now=102,snapshot_bytes=evidence))
    assert all(p.read_bytes()==raw for p,raw in media.items())
    with sqlite3.connect(root/'core.sqlite') as db:
        for table in ('jobs','events','attempts','publications','publication_work','dispatches','artifacts'):
            if table not in intent['before']['source']['data']:continue
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])


@pytest.mark.parametrize('point',['fence-parent','core-intent-begin'])
def test_b22_prefix_receipt_only_resume_without_new_identity(installation,point):
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    def crash(at):
        if at==point:raise OSError('receipt-only')
    with pytest.raises(OSError):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,fault=crash)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
    result=recovery._pending_through_evidence(roots,scope,intent,observation,now=1000)
    assert result['recovery_id']==intent['id'] and result['lane']==intent['new_lane']


def _b22_complete_intent_path(candidates):
    import re
    complete=[p for p in candidates if re.fullmatch(r'p1-recovery-[0-9a-f]{32}\.intent\.json',p.name)]
    assert len(complete)==1, 'expected exactly one canonical complete intent'
    return complete[0]


@pytest.mark.parametrize('case',['bundle-first','zero','multiple'])
def test_b22_complete_intent_selection(case):
    bundle=Path('p1-recovery-'+('9'*32)+'.bundle.intent.json')
    full=Path('p1-recovery-'+('9'*32)+'.intent.json')
    candidates=[bundle,Path('p1-recovery-'+('A'*32)+'.intent.json'),
        Path('p1-recovery-'+('a'*31)+'.intent.json'),Path(full.name+'.extra')]
    if case!='zero':candidates.append(full)
    if case=='multiple':candidates.append(Path('p1-recovery-'+('a'*32)+'.intent.json'))
    if case=='bundle-first':assert _b22_complete_intent_path(candidates)==full
    else:
        with pytest.raises(AssertionError,match='exactly one canonical complete intent'):
            _b22_complete_intent_path(candidates)


@pytest.mark.parametrize('point',['core-intent-commit','pending-receipt-commit'])
def test_b22_correction_process_death_owned_restart(installation,point):
    import os
    roots,scope,_,_=installation
    intent,observation,snapshot_raw=prefix_fixture(installation)
    del snapshot_raw  # Neither child nor parent retains the caller snapshot bytes.
    pid=os.fork()
    if pid==0:
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,
            fault=lambda at:os._exit(73) if at==point else None)
        os._exit(74)
    _,status=os.waitpid(pid,0)
    assert os.waitstatus_to_exitcode(status)==73
    del intent,observation
    candidates=list(roots['receipts'].glob('p1-recovery-*.intent.json'))
    path=_b22_complete_intent_path(candidates)
    assert path.name=='p1-recovery-'+('9'*32)+'.intent.json'
    assert roots['receipts']/('p1-recovery-'+('9'*32)+'.bundle.intent.json') in candidates
    owned=json.loads(path.read_bytes())
    from worker.runtime_journal import Journal
    for _ in range(2):
        with pytest.raises(ValueError,match='pending'):Core(roots['control']/'core',historical_local=True)
        with pytest.raises(ValueError,match='pending'):Journal(roots['receipts'])
        result=recovery._pending_through_evidence(roots,scope,owned,owned['proof']['observation'],now=1000)
        assert result==dict(status='pending-through-evidence',recovery_id=owned['id'],lane=owned['new_lane'],epoch=owned['new_epoch'])
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(2,)
    archives=list((roots['control']/'core').glob('p1-recovery-*.sqlite'))
    assert len(archives)==1
    assert archives[0].read_bytes()==next(roots['receipts'].glob('*.snapshot.bundle')).read_bytes()


@pytest.mark.parametrize('journal_tranche',[False,True])
@pytest.mark.parametrize('suffix',['.journal','.certificate.json'])
@pytest.mark.parametrize('damage',['changed','missing','none'])
def test_b22_correction_prior_receipts_every_resume(installation,suffix,damage,journal_tranche):
    roots,scope,_,_=installation
    first=b2_intent(installation,complete=True);b2_receipt(roots,first,complete=True)
    second=b2_intent(installation,complete=True,recovery_id='7'*32,archive_id='6'*32)
    b2_receipt(roots,second,complete=True)
    intent=b2_intent(installation,prepare_only=True,recovery_id='5'*32,archive_id='4'*32)
    helper=recovery._pending_through_journal if journal_tranche else recovery._pending_through_evidence
    result=helper(roots,scope,intent,intent['proof']['observation'],now=102)
    path=roots['receipts']/('p1-recovery-'+first['id']+suffix)
    if damage=='changed':path.write_bytes(b'X'+path.read_bytes()[1:])
    elif damage=='missing':path.unlink()
    for _ in range(2):
        if damage=='none':
            assert helper(roots,scope,intent,intent['proof']['observation'],now=1000)==result
        else:
            with pytest.raises((ValueError,FileNotFoundError)):
                helper(roots,scope,intent,intent['proof']['observation'],now=1000)


@pytest.mark.parametrize('point',['bundle-intent-create','bundle-intent-write','bundle-intent-fsync','bundle-intent-parent','bundle-intent-recorded','bundle-create','bundle-write','bundle-fsync','bundle-parent'])
def test_b22_correction_bundle_boundaries(installation,point):
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    def crash(at):
        if at==point:raise OSError('bundle-boundary')
    with pytest.raises(OSError,match='bundle-boundary'):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=102,fault=crash)
    originals={p:(p.stat().st_ino,p.read_bytes()) for p in roots['receipts'].glob('p1-recovery-*')}
    if point in ('bundle-write','bundle-fsync','bundle-parent'):
        recovery._pending_through_evidence(roots,scope,intent,observation,now=1000)
    else:
        with pytest.raises((ValueError,FileNotFoundError)):
            recovery._pending_through_evidence(roots,scope,intent,observation,now=1000)
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
            assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
    assert all((p.stat().st_ino,p.read_bytes())==value for p,value in originals.items())


@pytest.mark.parametrize('damage',['missing','changed','descriptor','mode','symlink','hardlink','oversize'])
def test_b22_correction_bundle_damage_no_repair(installation,damage):
    roots,scope,_,_=installation;intent,observation,evidence=prefix_fixture(installation)
    def crash(at):
        if at=='core-intent-commit':raise OSError('accepted')
    with pytest.raises(OSError):recovery._pending_through_evidence(roots,scope,intent,observation,now=102,fault=crash)
    bundle=next(roots['receipts'].glob('*.snapshot.bundle'))
    if damage=='missing':bundle.unlink()
    elif damage=='changed':bundle.write_bytes(b'X'+bundle.read_bytes()[1:])
    elif damage=='descriptor':next(roots['receipts'].glob('*.bundle.intent.json')).write_bytes(b'{}')
    elif damage=='mode':bundle.chmod(0o644)
    elif damage=='symlink':
        bundle.unlink();bundle.symlink_to(roots['receipts']/'journal.json')
    elif damage=='hardlink':
        import os
        os.link(bundle,roots['receipts']/'extra-link')
    else:
        with bundle.open('r+b') as stream:stream.truncate(recovery.MAX_BYTES+1)
    for _ in range(2):
        with pytest.raises((ValueError,FileNotFoundError,OSError)):
            recovery._pending_through_evidence(roots,scope,intent,observation,now=1000,snapshot_bytes=evidence)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(0,)


def test_b22b_rotation_pending_resume(installation):
    roots,scope,_,_=installation
    intent,observation,_=prefix_fixture(installation)
    result=recovery._pending_through_journal(roots,scope,intent,observation,now=102)
    assert result['status']=='pending-through-journal'
    journal=roots['receipts']/'journal.json'
    inode=journal.stat().st_ino
    assert json.loads(journal.read_bytes())['payload']==intent['expected'][2]['output']['journal']
    for _ in range(2):
        from worker.runtime_journal import Journal
        with pytest.raises(ValueError,match='pending'):Journal(roots['receipts'])
        with pytest.raises(ValueError,match='pending'):Core(roots['control']/'core',historical_local=True)
        assert recovery._pending_through_journal(roots,scope,intent,observation,now=1000)==result
        assert journal.stat().st_ino==inode
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT step FROM p1_recovery_steps ORDER BY rowid').fetchall()==[('pending',),('evidence',),('journal',)]
        for table,rows in intent['before']['source']['data'].items():
            if table in ('p1_recoveries','p1_recovery_steps','p1_abandon_plans'):continue
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(rows)


@pytest.mark.parametrize('point',[
    'journal-intent-create','journal-intent-write','journal-intent-fsync','journal-intent-parent','journal-intent-recorded',
    'journal-write','journal-fsync','journal-replace','journal-dirsync',
    'journal-receipt-begin','journal-receipt-write','journal-receipt-commit'])
def test_b22b_faults(installation,point):
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    def fault(at):
        if at==point:raise OSError(at)
    with pytest.raises(OSError,match=point):
        recovery._pending_through_journal(roots,scope,intent,observation,now=102,fault=fault)
    journal=roots['receipts']/'journal.json';raw=journal.read_bytes();inode=journal.stat().st_ino
    for _ in range(2):
        if point in ('journal-replace','journal-dirsync','journal-receipt-begin','journal-receipt-write','journal-receipt-commit'):
            assert recovery._pending_through_journal(roots,scope,intent,observation,now=1000)['status']=='pending-through-journal'
        else:
            with pytest.raises((ValueError,OSError)):
                recovery._pending_through_journal(roots,scope,intent,observation,now=1000)
        assert journal.read_bytes()==raw and journal.stat().st_ino==inode


@pytest.mark.parametrize('damage',['missing','wrong-lane','wrong-hash','symlink','hardlink','archive','manifest','media','history','proof','extra'])
def test_b22b_changed_output_refuses(installation,damage):
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    recovery._pending_through_journal(roots,scope,intent,observation,now=102)
    journal=roots['receipts']/'journal.json'
    if damage=='missing':journal.unlink()
    elif damage in ('wrong-lane','wrong-hash'):
        doc=json.loads(journal.read_bytes())
        if damage=='wrong-lane':
            doc['payload']['lane']='f'*32;doc['sha256']=digest(doc['payload'])
        else:doc['sha256']='f'*64
        journal.write_text(encode(doc))
    elif damage in ('symlink','hardlink'):
        target=roots['receipts'].parent/'journal-retained-test';journal.rename(target)
        if damage=='symlink':journal.symlink_to(target)
        else:os.link(target,journal)
    elif damage=='archive':(roots['receipts']/('p1-recovery-'+intent['id']+'.journal')).write_bytes(b'changed')
    elif damage=='manifest':next((roots['control']/'core').glob('*.manifest.json')).write_bytes(b'changed')
    elif damage=='media':(roots['staging']/'foreign').write_bytes(b'changed')
    elif damage=='history':
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:db.execute('UPDATE resource_monitor SET state=?',('fault',))
    elif damage=='proof':observation=copy.deepcopy(observation);observation['states'][0]['running']=True
    else:(roots['receipts']/'foreign').write_bytes(b'changed')
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):
            recovery._pending_through_journal(roots,scope,intent,observation,now=1000)


def test_b22b_process_death_fresh_interpreter(installation):
    import subprocess,sys
    roots,scope,_,_=installation;intent,observation,raw=prefix_fixture(installation)
    del raw
    child=os.fork()
    if child==0:
        recovery._pending_through_journal(roots,scope,intent,observation,now=102,
            fault=lambda point:os._exit(73) if point=='journal-replace' else None)
        os._exit(74)
    assert os.waitpid(child,0)[1]==73<<8
    identity=intent['id'];del intent,observation
    script='''
import json,sys,sqlite3
from pathlib import Path
from service import recovery
from service.core import Core
from worker.runtime_journal import Journal
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()}
intent=json.loads((roots['receipts']/('p1-recovery-'+v['id']+'.intent.json')).read_bytes())
journal=roots['receipts']/'journal.json';raw=journal.read_bytes();inode=journal.stat().st_ino
for _ in range(2):
    for constructor in (lambda:Core(roots['control']/'core',historical_local=True),lambda:Journal(roots['receipts'])):
        try:constructor()
        except ValueError as e:assert 'pending' in str(e)
        else:raise AssertionError('startup opened')
    result=recovery._pending_through_journal(roots,v['scope'],intent,intent['proof']['observation'],now=1000)
    assert result['lane']==intent['new_lane'] and result['status']=='pending-through-journal'
    assert journal.read_bytes()==raw and journal.stat().st_ino==inode
with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
    assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
    assert db.execute('SELECT step FROM p1_recovery_steps ORDER BY rowid').fetchall()==[('pending',),('evidence',),('journal',)]
assert len(list((roots['control']/'core').glob('*.manifest.json')))==1
'''
    result=_guarded_fresh_run([sys.executable,'-c',script],input=json.dumps(dict(roots={k:str(v) for k,v in roots.items()},scope=scope,id=identity)),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('barrier',['file','replace','directory'])
def test_b22b_actual_io_denial(installation,monkeypatch,barrier):
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    recovery._pending_through_evidence(roots,scope,intent,observation,now=102)
    import worker.runtime_journal as runtime
    original_fsync=os.fsync;original_replace=os.replace
    def fsync(fd):
        import stat
        info=os.fstat(fd)
        target=roots['receipts']/('journal.tmp' if barrier=='file' else 'journal.json')
        if barrier=='file' and target.exists() and info.st_ino==target.stat().st_ino:raise OSError('actual journal IO denial')
        if barrier=='directory' and info.st_ino==roots['receipts'].stat().st_ino and (roots['receipts']/'journal.json').read_bytes()==runtime.journal_bytes(intent['expected'][2]['output']['journal']):raise OSError('actual journal IO denial')
        return original_fsync(fd)
    def replace(source,target):
        if Path(source).name=='journal.tmp':raise OSError('actual journal IO denial')
        return original_replace(source,target)
    with monkeypatch.context() as patch:
        patch.setattr(os,'replace',replace) if barrier=='replace' else patch.setattr(os,'fsync',fsync)
        with pytest.raises(OSError,match='actual journal IO denial'):
            recovery._pending_through_journal(roots,scope,intent,observation,now=1000)
    for _ in range(2):
        if barrier=='directory':assert recovery._pending_through_journal(roots,scope,intent,observation,now=1000)['status']=='pending-through-journal'
        else:
            with pytest.raises(ValueError):recovery._pending_through_journal(roots,scope,intent,observation,now=1000)


@pytest.mark.parametrize('coverage',['exact','omitted','live','unknown'])
def test_b22b_unknown_dispatch_coverage(installation,coverage):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,scope,_,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        core.submit_synthetic('p','rotation-unknown',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        row=store.allocate(f['profiles'],'silence',8)
        lane=json.loads((roots['receipts']/'journal.json').read_bytes())['payload']['lane']
        store.update(row,runtime=lane+'/'+'b'*32+'/unknown',phase='prepared')
    finally:core.close()
    intent,observation,_=prefix_fixture(installation)
    if coverage!='exact':
        observation=copy.deepcopy(observation)
        if coverage=='omitted':observation['containers'].pop()
        elif coverage=='live':observation['states'][0]['running']=True
        else:observation['states'].pop()
        with pytest.raises(ValueError):recovery._pending_through_journal(roots,scope,intent,observation,now=102)
        return
    for now in (102,1000):
        recovery._pending_through_journal(roots,scope,intent,observation,now=now)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        for table in ('jobs','dispatches','resource_reservations','resource_accounting','resource_monitor','resource_faults','events'):
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])


def test_b22b_expiry_and_both_locks(installation):
    import fcntl
    roots,scope,_,_=installation;intent,observation,_=prefix_fixture(installation)
    with pytest.raises(ValueError,match='expired'):recovery._pending_through_journal(roots,scope,intent,observation,now=1000)
    for path in (roots['control']/'core'/'control.lock',roots['receipts']/'lane.lock'):
        with path.open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with pytest.raises(ValueError,match='lock_busy'):recovery._pending_through_journal(roots,scope,intent,observation,now=102)
    assert not (roots['receipts']/'p1-maintenance.json').exists()


def test_b22c_atomic_core_completion(installation):
    roots,scope,_,_=installation
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    intent,observation,_=prefix_fixture(installation)
    result=recovery._pending_through_core(roots,scope,intent,observation,now=102)
    assert result['status']=='core-complete-receipt-pending'
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(4,)
        assert recovery.completion_binding(db)['epoch']==intent['new_epoch']
    before=logical(roots)
    assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)==result
    assert logical(roots)==before
    from worker.runtime_journal import Journal
    with pytest.raises(ValueError):Journal(roots['receipts'])


def core_fixture(installation,kind='queued',*,highwater=20*1024**2):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots=installation[0];core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        job=core.submit_synthetic('p','atomic-core',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        if kind!='queued':
            allocated=store.allocate(f['profiles'],'silence',8)
            if kind=='unknown':
                lane=json.loads((roots['receipts']/'journal.json').read_bytes())['payload']['lane']
                store.update(allocated,runtime=lane+'/'+'b'*32+'/unknown',phase='prepared')
        if kind=='cancel':core.request_cancel(job['id'])
        with core.transaction():
            core.db.execute('UPDATE resource_accounting SET highwater=?',(highwater,))
            core.db.execute("UPDATE resource_monitor SET state='checking'")
            core.db.execute("UPDATE p1_integrity SET state='checking',token='inherited-A2'")
    finally:core.close()
    intent,observation,_=prefix_fixture(installation)
    return intent,observation,job['id']


@pytest.mark.parametrize('kind',['queued','running','cancel'])
def test_b22c_real_actions_history_and_resolution_guards(installation,kind):
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation;intent,observation,job_id=core_fixture(installation,kind)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        expected=intent['expected'][3]['output']['rows']
        for table,rows in expected.items():assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(rows)
        for table in ('dispatches','dispatch_commands','events','assets','artifacts','publications','publication_work','p1_inventory_history'):
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])
        assert db.execute('SELECT * FROM p1_integrity').fetchall()==[(1,1,'healthy',None)]
        assert db.execute('SELECT count(*) FROM dispatch_resolutions').fetchone()==(int(kind!='queued'),)
    core=Core(roots['control']/'core')
    try:
        store=DispatchStore(core);assert store.current() is None
        if kind!='queued':
            with pytest.raises(ValueError,match='operator_resolved_job'):store.allocate([], 'silence',8,replacement=job_id)
    finally:core.close()
    assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)['epoch']==intent['new_epoch']


@pytest.mark.parametrize('point',['core-scan','core-complete-begin','core-complete-write','core-complete-commit','core-complete-parent','core-certificate-create','core-certificate-write','core-certificate-fsync','core-certificate-parent'])
def test_b22c_atomic_faults(installation,point):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    def fault(where):
        if where==point:raise OSError(point)
    with pytest.raises(OSError,match=point):recovery._pending_through_core(roots,scope,intent,observation,now=102,fault=fault)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        committed=bool(db.execute("SELECT 1 FROM p1_recovery_steps WHERE step='core-complete'").fetchone())
        for table in recovery.CORE_OUTPUT_TABLES:
            expected=intent['expected'][3]['output']['rows'][table] if committed else intent['before']['source']['data'][table]
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(expected)
    if point=='core-certificate-create':
        for _ in range(2):
            with pytest.raises(ValueError):recovery._pending_through_core(roots,scope,intent,observation,now=1000)
    else:
        for _ in range(2):assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)['epoch']==intent['new_epoch']


@pytest.mark.parametrize('mode',['quota','free','changed-highwater'])
def test_b22c_scan_refusal_retains_pending_authority(installation,monkeypatch,mode):
    roots,scope,_,_=installation
    if mode=='changed-highwater':intent,observation,_=prefix_fixture(installation)
    else:intent,observation,_=core_fixture(installation,'running',highwater=(64 if mode=='quota' else 20)*1024**2)
    if mode=='free':monkeypatch.setattr(os,'statvfs',lambda root:type('V',(),dict(f_bavail=0,f_frsize=4096))())
    for _ in range(2):
        with pytest.raises(ValueError,match='disk_|b2_scan_output_changed'):
            recovery._pending_through_core(roots,scope,intent,observation,now=102 if _==0 else 1000)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(3,)
        assert not db.execute('SELECT 1 FROM dispatch_resolutions').fetchone()
        assert db.execute('SELECT * FROM resource_monitor').fetchall()==[tuple(v) for v in intent['before']['source']['data']['resource_monitor']]


@pytest.mark.parametrize('point',['core-complete-write','core-complete-commit'])
def test_b22c_process_death_fresh_interpreter_twice(installation,point):
    import subprocess,sys
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'cancel')
    pid=os.fork()
    if pid==0:
        recovery._pending_through_core(roots,scope,intent,observation,now=102,fault=lambda where:os._exit(73) if where==point else None)
        os._exit(74)
    assert os.waitpid(pid,0)[1]==73<<8
    rid=intent['id'];epoch=intent['new_epoch'];del intent,observation
    code='''import json,sys,sqlite3
from pathlib import Path
from service import recovery
from worker.runtime_journal import Journal
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()}
intent=json.loads((roots['receipts']/('p1-recovery-'+v['rid']+'.intent.json')).read_bytes())
for _ in range(2):
 try:Journal(roots['receipts'])
 except ValueError:pass
 else:raise AssertionError('worker started')
 result=recovery._pending_through_core(roots,v['scope'],intent,intent['proof']['observation'],now=1000)
 assert result['epoch']==v['epoch']
with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
 assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(4,)
 assert db.execute('SELECT count(*) FROM dispatch_resolutions').fetchone()==(1,)
 assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
'''
    child=_guarded_fresh_run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid,epoch=epoch)),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==0,child.stderr


def test_b22c_completed_no_scan_and_pending_resolution_has_no_effect(installation,monkeypatch):
    from service.dispatch_store import DispatchStore
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    recovery._pending_through_journal(roots,scope,intent,observation,now=102)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        record=recovery._resolution_outputs(intent)['dispatch_resolutions'][0]
        db.execute('INSERT INTO dispatch_resolutions VALUES(?,?,?,?)',(record['dispatch_seq'],intent['id'],encode(record),digest(record)))
        store=DispatchStore.__new__(DispatchStore);store.db=db
        assert store.current()['seq']==record['dispatch_seq']
        db.execute('DELETE FROM dispatch_resolutions')
    result=recovery._pending_through_core(roots,scope,intent,observation,now=1000)
    def forbidden(*args,**kwargs):raise AssertionError('extra scan')
    monkeypatch.setattr(Reconciliation,'_collect',forbidden)
    from service import resource_store
    monkeypatch.setattr(resource_store,'_recovery_scan',forbidden)
    for _ in range(2):assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)==result


@pytest.mark.parametrize('changed',['none','new-defect','changed-old','unknown-reference'])
def test_b22c_exact_degraded_resolution_and_strict_unavailability(installation,changed):
    roots,scope,_,_=installation;job_id=terminal_enrolled(installation);root=roots['control']/'core'
    asset=next(p for p in (root/'assets').iterdir() if p.is_file());asset_id=asset.name;asset.unlink()
    core=Core(root)
    try:
        scanner=Reconciliation(core,roots['staging']);assert scanner.reconcile_startup()['blocked']
        row=next(json.loads(raw) for (raw,) in core.db.execute('SELECT record FROM p1_inventory_objects') if json.loads(raw)['classification']=='missing_hash_mismatched_terminal')
        with core.transaction():core.db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    finally:core.close()
    decision=dict(object_id=row['id'],reason=row['reason'],content_digest=recovery.diagnostic_digest(row),decision='accept-degraded',availability='unavailable')
    intent=b2_intent(installation,prepare_only=True,diagnostic_decisions=[decision]);observation=intent['proof']['observation']
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    if changed=='new-defect':(root/'assets'/'new-orphan').write_bytes(b'new')
    elif changed=='changed-old':asset.write_bytes(b'wrong-media')
    elif changed=='unknown-reference':
        with sqlite3.connect(root/'core.sqlite') as db:db.execute('CREATE TABLE unknown_refs(id TEXT)')
    if changed=='unknown-reference':
        with pytest.raises(ValueError):Core(root)
        return
    core=Core(root)
    try:
        scanner=Reconciliation(core,roots['staging']);result=scanner.reconcile_startup()
        assert result['complete']
        assert result['blocked']==(changed!='none')
        assert core.job(job_id)['state']=='succeeded'
        with pytest.raises((ValueError,OSError)):core.asset(asset_id)
        assert any(r['classification']=='missing_hash_mismatched_terminal' for r in scanner.rows.values())
    finally:core.close()


@pytest.mark.parametrize('method',['_publish','_commit_publication'])
def test_b22c_terminal_history_and_permanent_retirement(installation,monkeypatch,method):
    from service.persistence import PublicationQuarantine
    roots,scope,_,_=installation;job_id=terminal_enrolled(installation);root=roots['control']/'core'
    with sqlite3.connect(root/'core.sqlite') as db:db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    intent,observation,_=prefix_fixture(installation)
    media={p:p.read_bytes() for p in (root/'assets').iterdir()}
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    with sqlite3.connect(root/'core.sqlite') as db:
        for table in ('jobs','dispatches','dispatch_commands','events','assets','artifacts','publications','publication_work'):
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])
    assert all(p.read_bytes()==raw for p,raw in media.items())
    core=Core(root,historical_local=True)
    try:
        monkeypatch.setattr(core,'_load',lambda *args:pytest.fail('retired body entered'))
        with pytest.raises(PublicationQuarantine,match='retired'):
            if method=='_publish':core._publish(job_id,{})
            else:core._commit_publication(job_id)
    finally:core.close()


@pytest.mark.parametrize('point',['core-certificate-intent-create','core-certificate-intent-write','core-certificate-intent-fsync','core-certificate-intent-parent','core-certificate-intent-recorded'])
def test_b22c_certificate_creation_intent_never_remints(installation,point):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    with pytest.raises(OSError):
        recovery._pending_through_core(roots,scope,intent,observation,now=102,fault=lambda at:(_ for _ in ()).throw(OSError(at)) if at==point else None)
    before=logical(roots)
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):recovery._pending_through_core(roots,scope,intent,observation,now=1000)
        assert logical(roots)==before


@pytest.mark.parametrize('barrier',['file','parent'])
def test_b22c_actual_certificate_barrier_denial(installation,monkeypatch,barrier):
    import stat
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    original=os.fsync;armed=False;denied=False
    def fault(at):
        nonlocal armed
        if at==('core-certificate-write' if barrier=='file' else 'core-certificate-fsync'):armed=True
    def sync(fd):
        nonlocal denied
        if armed and not denied and (stat.S_ISREG(os.fstat(fd).st_mode) if barrier=='file' else stat.S_ISDIR(os.fstat(fd).st_mode)):
            denied=True;raise OSError('actual-certificate-barrier')
        return original(fd)
    monkeypatch.setattr(os,'fsync',sync)
    with pytest.raises(OSError,match='actual-certificate-barrier'):recovery._pending_through_core(roots,scope,intent,observation,now=102,fault=fault)
    assert denied
    for _ in range(2):assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)['epoch']==intent['new_epoch']


def test_b22c_sql_write_denial_rolls_back_all_core_authority(installation,monkeypatch):
    from contextlib import contextmanager
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    original=recovery.offline
    @contextmanager
    def denied(*args,**kwargs):
        with original(*args,**kwargs) as (checked,db):
            db.set_authorizer(lambda action,table,*rest:sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_UPDATE and table=='resource_monitor' else sqlite3.SQLITE_OK)
            yield checked,db
    monkeypatch.setattr(recovery,'offline',denied)
    with pytest.raises(sqlite3.DatabaseError):recovery._pending_through_core(roots,scope,intent,observation,now=102)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        for table in recovery.CORE_OUTPUT_TABLES:
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(intent['before']['source']['data'][table])
        assert not db.execute('SELECT 1 FROM dispatch_resolutions').fetchone()
    monkeypatch.setattr(recovery,'offline',original)
    recovery._pending_through_core(roots,scope,intent,observation,now=1000)


@pytest.mark.parametrize('coverage',['exact','omitted','live','unknown'])
def test_b22c_unknown_dispatch_requires_exact_death(installation,coverage):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'unknown')
    if coverage!='exact':
        observation=copy.deepcopy(observation)
        if coverage=='omitted':observation['containers'].pop()
        elif coverage=='live':observation['states'][0]['running']=True
        else:observation['states'].pop()
        for _ in range(2):
            with pytest.raises(ValueError):recovery._pending_through_core(roots,scope,intent,observation,now=102)
        return
    for now in (102,1000):recovery._pending_through_core(roots,scope,intent,observation,now=now)
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert encode(db.execute('SELECT * FROM dispatches').fetchall())==encode(intent['before']['source']['data']['dispatches'])
        assert db.execute('SELECT count(*) FROM dispatch_resolutions').fetchone()==(1,)


@pytest.mark.parametrize('registration',[None,'recorded'])
def test_b22c_core_complete_receipt_pending_blocks_startup(installation,registration):
    from service.dispatcher import Dispatcher
    from worker.runtime_journal import Journal
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    class Runtime:
        def call(self,op,body):
            assert op=='hello'
            Journal(roots['receipts'])
            pytest.fail('pending worker supplied hello')
    for _ in range(2):
        before=logical(roots)
        with pytest.raises(ValueError):Dispatcher(roots['control']/'core',runtime=Runtime(),**({} if registration is None else {'persistent_registration':dict(registration_id='recorded')}))
        assert before==logical(roots)


@pytest.mark.parametrize('damage',['history','media','scope','proof','bundle','resolution','certificate','receipt-archive'])
def test_b22c_completed_immutable_inputs_and_history_refuse(installation,damage):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    if damage=='history':
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:db.execute('DELETE FROM p1_recovery_plans')
    elif damage=='scope':scope={**scope,'daemon':'wrong'}
    elif damage=='proof':observation={**observation,'evidence':'actual-host'}
    elif damage=='bundle':next(roots['receipts'].glob('*.snapshot.bundle')).write_bytes(b'wrong')
    elif damage=='resolution':
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:db.execute('DELETE FROM dispatch_resolutions')
    elif damage=='certificate':
        path=next(roots['control'].glob('core/*.certificate.json'));path.write_bytes(b'{}')
    elif damage=='receipt-archive':next(roots['receipts'].glob('*.journal')).unlink()
    else:
        # Current media eligibility is separate from immutable completed history.
        (roots['staging']/'new-activity').write_bytes(b'new')
        assert recovery._pending_through_core(roots,scope,intent,observation,now=1000)['status']=='core-complete-receipt-pending'
        return
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):recovery._pending_through_core(roots,scope,intent,observation,now=1000)


def test_b22c_denied_fault_update_inherited_checking_new_epoch(installation):
    from service.resource_store import ResourceStore
    roots,scope,_,_=installation;root=roots['control']/'core'
    import sys
    # Finite test-owned JSON input; fresh exec imports its own empty registry.
    payload=json.dumps(dict(root=str(root),staging=str(roots['staging']),budget=DEFAULT_BUDGET),separators=(',',':'))
    assert len(payload.encode())<=16384
    script="""
import json,os,sqlite3,sys
from pathlib import Path
from service.core import Core
from service.resource_store import ResourceStore
raw=sys.stdin.buffer.read(16385)
assert len(raw)<=16384
inputs=json.loads(raw)
assert set(inputs)=={'root','staging','budget'}
root=Path(inputs['root']);roots={'staging':Path(inputs['staging'])};DEFAULT_BUDGET=inputs['budget']
core=Core(root);store=ResourceStore(core,DEFAULT_BUDGET,(root,roots['staging']))
def observe():
    assert store.state()=='checking'
    core.db.execute('PRAGMA query_only=ON')
    raise ValueError('transient-output-growth')
try:store.observe(observe)
except sqlite3.OperationalError:
    # No product close/finally/later loop writes a witness before exit.
    # Normal interpreter exit permits only the existing bootstrap done receipt.
    sys.exit(73)
sys.exit(74)
"""
    result=_guarded_fresh_run([sys.executable,'-c',script],input=payload,text=True,capture_output=True,timeout=30)
    assert result.returncode==73,(result.stdout,result.stderr)
    for _ in range(2):
        core=Core(root)
        try:
            store=ResourceStore(core,DEFAULT_BUDGET,(root,roots['staging']))
            with pytest.raises(ValueError,match='unresolved'):store.check_disk()
        finally:core.close()
    with sqlite3.connect(root/'core.sqlite') as db:db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    intent,observation,_=prefix_fixture(installation)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    assert intent['before']['source']['data']['resource_monitor']==[[1,'checking']]
    assert intent['prior_epoch']!=intent['new_epoch']
    with sqlite3.connect(root/'core.sqlite') as db:assert db.execute('SELECT * FROM resource_monitor').fetchall()==[(1,'healthy')]


def test_b22c_dispatch_diagnostic_history_is_exactly_accounted(installation):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,scope,_,_=installation;root=roots['control']/'core';core=Core(root)
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        core.submit_synthetic('p','diagnostic-dispatch',f['context'],_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        store.allocate(f['profiles'],'silence',8)
        scanner=Reconciliation(core,roots['staging']);assert scanner.reconcile_startup()['blocked']
        with core.transaction():
            core.db.execute("UPDATE resource_monitor SET state='fault'")
            core.db.execute("INSERT INTO resource_faults VALUES('old-observed-output-fault')")
            core.db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    finally:core.close()
    intent,observation,_=prefix_fixture(installation)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    assert intent['before']['source']['data']['resource_faults']==[['old-observed-output-fault']]
    core=Core(root)
    try:
        scanner=Reconciliation(core,roots['staging']);result=scanner.reconcile_startup()
        assert result['complete'] and not result['blocked'],result
        # A different historical defect cannot borrow the completed dispatch ID.
        with core.transaction():
            raw=core.db.execute("SELECT id,record FROM p1_inventory_objects WHERE record LIKE '%unbound_dispatch%'").fetchone()
            value=json.loads(raw[1]);value['reason']='new-owner-defect'
            core.db.execute('UPDATE p1_inventory_objects SET record=? WHERE id=?',(encode(value),raw[0]))
        assert scanner.reconcile_startup()['blocked']
    finally:core.close()


@pytest.mark.parametrize('damage',['bundle-missing','bundle-changed','descriptor','journal-archive'])
def test_b22c_uncertified_core_revalidates_owned_restart_evidence(installation,damage):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    with pytest.raises(OSError):
        recovery._pending_through_core(roots,scope,intent,observation,now=102,fault=lambda at:(_ for _ in ()).throw(OSError(at)) if at=='core-complete-commit' else None)
    if damage=='bundle-missing':next(roots['receipts'].glob('*.snapshot.bundle')).unlink()
    elif damage=='bundle-changed':next(roots['receipts'].glob('*.snapshot.bundle')).write_bytes(b'changed')
    elif damage=='descriptor':next(roots['receipts'].glob('*.bundle.intent.json')).write_bytes(b'{}')
    else:next(roots['receipts'].glob('*.journal')).write_bytes(b'changed')
    certificate=next(roots['control']/'core'/n for n,k in recovery.archive_names(intent).items() if k=='certificate')
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):recovery._pending_through_core(roots,scope,intent,observation,now=1000)
        assert not certificate.exists()


@pytest.mark.parametrize('change', ['unchanged', 'context-text', 'context-bundle', 'job-input', 'job-owner', 'job-generation', 'row-principal'])
def test_b22c_completed_resolution_current_joins_two_startups(installation, change):
    from contextlib import closing
    roots, scope, _, _ = installation
    intent, observation, jid = core_fixture(installation, 'running')
    recovery._pending_through_core(roots, scope, intent, observation, now=102)
    root = roots['control'] / 'core'
    with closing(sqlite3.connect(root / 'core.sqlite')) as db, db:
        original = db.execute('SELECT acked,receipt,outcome FROM dispatches').fetchall()
        job, context = map(json.loads, db.execute('SELECT job,context FROM jobs WHERE id=?', (jid,)).fetchone())
        if change == 'context-text': context['plan']['request']['text'] = 'changed after recovery'
        elif change == 'context-bundle': context['bundles'] = []
        elif change == 'job-input': job['input_sha256'] = 'f' * 64
        elif change == 'job-owner': job['attempts'][-1]['owner_id'] = 'f' * 32
        elif change == 'job-generation': job['generation'] += 1
        elif change == 'row-principal': db.execute("UPDATE jobs SET principal='changed' WHERE id=?", (jid,))
        db.execute('UPDATE jobs SET job=?,context=? WHERE id=?', (encode(job), encode(context), jid))
    results = []
    for _ in range(2):
        core = Core(root)
        try:
            results.append(Reconciliation(core, roots['staging']).reconcile_startup())
            assert core.db.execute('SELECT acked,receipt,outcome FROM dispatches').fetchall() == original
        finally: core.close()
    assert all(result['complete'] for result in results), results
    assert all(result['blocked'] == (change != 'unchanged') for result in results), results


def test_b22d_complete_production_writer_and_noop(installation,monkeypatch):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    result=recovery._complete_receipts(roots,scope,intent,observation,now=102)
    assert result['status']=='complete'
    from worker.runtime_journal import Journal
    journal=Journal(roots['receipts'])
    assert journal.maintenance==result['certificate']
    journal.lock.close()
    before=logical(roots)
    monkeypatch.setattr(recovery,'snapshot',lambda *a,**k:(_ for _ in ()).throw(AssertionError('scan')))
    for _ in range(2):assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)==result
    assert logical(roots)==before
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
        assert db.execute('SELECT count(*) FROM dispatch_resolutions').fetchone()==(1,)


B22D_POINTS=(
    [(kind+'-intent-'+boundary,False) for kind in ('receipt-certificate','completion-journal','completion-fence') for boundary in ('create','write','fsync','parent')]+
    [(kind+'-intent-recorded',False) for kind in ('receipt-certificate','completion-journal','completion-fence')]+
    [('receipt-certificate-'+boundary,boundary!='create') for boundary in ('create','write','fsync','parent')]+
    [(kind+'-'+boundary,boundary in ('replace','dirsync')) for kind in ('completion-journal','completion-fence') for boundary in ('create','write','fsync','parent','before-replace','replace','dirsync')]+
    [('receipt-complete-'+boundary,True) for boundary in ('begin','write','commit','parent')])


@pytest.mark.parametrize('point,resumable',B22D_POINTS)
def test_b22d_fault_matrix(installation,point,resumable):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    before=logical(roots)
    def fault(where):
        if where==point:raise OSError(point)
    with pytest.raises(OSError,match=point):recovery._complete_receipts(roots,scope,intent,observation,now=1000,fault=fault)
    for _ in range(2):
        if resumable:assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)['status']=='complete'
        else:
            with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,intent,observation,now=1000)
    after=logical(roots)
    assert [v for v in after if not v.startswith('INSERT INTO \"p1_recovery_steps\"')]==[v for v in before if not v.startswith('INSERT INTO \"p1_recovery_steps\"')]
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5 if resumable else 4,)


@pytest.mark.parametrize('point',['completion-fence-before-replace','completion-fence-replace','receipt-complete-write','receipt-complete-commit'])
def test_b22d_process_death_owned_fresh_resume_twice(installation,point):
    import subprocess,sys
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation,'running')
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    pid=os.fork()
    if pid==0:
        recovery._complete_receipts(roots,scope,intent,observation,now=1000,fault=lambda at:os._exit(73) if at==point else None)
        os._exit(74)
    assert os.waitpid(pid,0)[1]==73<<8
    rid=intent['id'];epoch=intent['new_epoch'];lane=intent['new_lane'];del intent,observation
    code='''import json,sys,sqlite3
from pathlib import Path
from service import recovery
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()}
intent=json.loads((roots['receipts']/('p1-recovery-'+v['rid']+'.intent.json')).read_bytes())
for _ in range(2):
 if v['quarantine']:
  try:recovery._complete_receipts(roots,v['scope'],intent,intent['proof']['observation'],now=1000)
  except (ValueError,OSError):pass
  else:raise AssertionError('reminted attempted fence')
 else:
  result=recovery._complete_receipts(roots,v['scope'],intent,intent['proof']['observation'],now=1000)
  assert result['epoch']==v['epoch'] and result['lane']==v['lane'] and result['status']=='complete'
with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
 assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(4 if v['quarantine'] else 5,)
 assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
 assert db.execute('SELECT count(*) FROM dispatch_resolutions').fetchone()==(1,)
'''
    child=_guarded_fresh_run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid,epoch=epoch,lane=lane,quarantine=point.endswith('before-replace'))),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==0,child.stderr


@pytest.mark.parametrize('damage',['receipt-missing','receipt-malformed','receipt-mode','receipt-symlink','receipt-hardlink','core','bundle','original','history','fence-missing','fence-malformed','fence-mode','lane','epoch','proof','extra'])
def test_b22d_wrong_evidence_refuses_without_rewrite(installation,damage):
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    def fault(at):
        if at=='receipt-certificate-parent':raise OSError('stop')
    with pytest.raises(OSError,match='stop'):recovery._complete_receipts(roots,scope,intent,observation,now=102,fault=fault)
    receipts=roots['receipts'];path=receipts/('p1-recovery-'+intent['id']+'.certificate.json')
    if damage=='receipt-missing':path.unlink()
    elif damage=='receipt-malformed':path.write_bytes(b'{}')
    elif damage=='receipt-mode':path.chmod(0o644)
    elif damage=='receipt-symlink':
        other=receipts/'other';path.rename(other);path.symlink_to(other)
    elif damage=='receipt-hardlink':os.link(path,receipts/'other')
    elif damage=='core':next(roots['control'].glob('core/*.certificate.json')).write_bytes(b'{}')
    elif damage=='bundle':next(receipts.glob('*.snapshot.bundle')).write_bytes(b'changed')
    elif damage=='original':next(receipts.glob('*.journal')).write_bytes(b'changed')
    elif damage=='history':
        with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:db.execute('DELETE FROM p1_recovery_plans')
    elif damage=='fence-missing':(receipts/'p1-maintenance.json').unlink()
    elif damage=='fence-malformed':(receipts/'p1-maintenance.json').write_bytes(b'{}')
    elif damage=='fence-mode':(receipts/'p1-maintenance.json').chmod(0o644)
    elif damage=='extra':(receipts/'foreign').write_bytes(b'x')
    else:
        value=json.loads(path.read_bytes());value['certificate'][damage if damage!='proof' else 'proof_digest']='f'*(64 if damage=='proof' else 32);path.write_text(encode(value))
    before={p.name:(p.lstat().st_ino,p.read_bytes()) for p in receipts.iterdir() if p.is_file()}
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,intent,observation,now=1000)
        assert before=={p.name:(p.lstat().st_ino,p.read_bytes()) for p in receipts.iterdir() if p.is_file()}


@pytest.mark.parametrize('operation',['file-fsync','parent-fsync','journal-replace','fence-replace','sql-write','sql-commit'])
def test_b22d_actual_io_denials(installation,monkeypatch,operation):
    import stat
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    recovery._pending_through_core(roots,scope,intent,observation,now=102)
    original_fsync=os.fsync;original_replace=os.replace;original_transaction=recovery._prefix_transaction
    triggered=[];armed=[]
    def fault(at):
        if at=='receipt-certificate-intent-recorded':armed.append(True)
    def fsync(fd):
        info=os.fstat(fd)
        if armed and not triggered and ((operation=='file-fsync' and stat.S_ISREG(info.st_mode) and info.st_size>0) or (operation=='parent-fsync' and stat.S_ISDIR(info.st_mode))):
            triggered.append(True);raise PermissionError('actual-io-denial')
        return original_fsync(fd)
    def replace(src,dst):
        if (operation=='journal-replace' and Path(src).name=='p1-maintenance.journal.tmp') or (operation=='fence-replace' and Path(src).name=='p1-maintenance.complete.tmp'):
            triggered.append(True);raise PermissionError('actual-io-denial')
        return original_replace(src,dst)
    def transaction(db,write,fault,point):
        if point=='receipt-complete':
            def authorize(action,a,b,c,d):
                if ((operation=='sql-write' and action==sqlite3.SQLITE_INSERT and a=='p1_recovery_steps') or
                    (operation=='sql-commit' and action==sqlite3.SQLITE_TRANSACTION and a=='COMMIT')):
                    triggered.append(True);return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
        return original_transaction(db,write,fault,point)
    with monkeypatch.context() as patch:
        patch.setattr(os,'fsync',fsync);patch.setattr(os,'replace',replace);patch.setattr(recovery,'_prefix_transaction',transaction)
        with pytest.raises((OSError,sqlite3.DatabaseError)):recovery._complete_receipts(roots,scope,intent,observation,now=1000,fault=fault)
    assert triggered
    for _ in range(2):
        if operation.endswith('replace'):
            with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,intent,observation,now=1000)
        else:assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)['status']=='complete'


def test_b22d_real_journal_hello_fresh_registration_dispatcher_stays_closed(installation):
    from worker.runtime_journal import Journal
    from worker.runtime_supervisor import RuntimeSupervisor
    from service.dispatcher import Dispatcher
    roots,scope,profiles,_=installation;intent,observation,jid=core_fixture(installation,'running')
    result=recovery._complete_receipts(roots,scope,intent,observation,now=102)
    supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.journal=Journal(roots['receipts']);supervisor.boot='e'*32
    supervisor.failed=False;supervisor.active=None;supervisor.launches=0
    try:hello=supervisor.operation('hello',{})
    finally:supervisor.journal.lock.close()
    assert hello['maintenance']==result['certificate']
    class Runtime:
        def __init__(self):self.calls=[]
        def call(self,op,body):self.calls.append(op);assert op=='hello';return hello
    runtime=Runtime()
    old_id=digest(json.loads(intent['before']['source']['data']['p1_registrations'][0][1]))
    for registration_id in (None,old_id):
        with pytest.raises(ValueError):Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=runtime,profiles=profiles,persistent_registration=registration_id)
    r=json.loads(intent['before']['source']['data']['p1_registrations'][0][1])
    r.update(lane=hello['lane'],boot=hello['boot'],evidence='actual-host-inspection')
    # E1 document only; this does not authenticate B2.3 host transport.
    rid=recovery.register(roots,scope,r)
    dispatcher=Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=runtime,profiles=profiles,persistent_registration=rid)
    try:
        status=dispatcher.call('status');assert not status['persistent_activation'] and not status['resource_activation']
        for op,args in (('submit',('new','key')),('replace',(jid,)),('cancel',(jid,))):
            with pytest.raises(ValueError,match='activation_closed'):dispatcher.call(op,*args)
        assert supervisor.launches==0 and all(op=='hello' for op in runtime.calls)
    finally:dispatcher.close()
    before=logical(roots)
    for _ in range(2):assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)==result
    assert logical(roots)==before


@pytest.mark.parametrize('half',['core-complete-receipt-pending','core-pending-receipt-complete'])
def test_b22d_half_startup_twice(installation,half):
    from worker.runtime_journal import Journal
    from service.dispatcher import Dispatcher
    roots,scope,profiles,_=installation;intent,observation,_=core_fixture(installation,'running')
    if half=='core-complete-receipt-pending':recovery._pending_through_core(roots,scope,intent,observation,now=102)
    else:
        recovery._complete_receipts(roots,scope,intent,observation,now=102)
        # Deliberately unsupported cross-volume state; retain all receipt files.
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute("DELETE FROM p1_recovery_steps WHERE step IN ('core-complete','receipt-complete')")
    class Runtime:
        def call(self,op,body):
            assert op=='hello';journal=Journal(roots['receipts'])
            try:return dict(lane=journal.value['lane'],boot='e'*32,quarantined=False,maintenance=journal.maintenance)
            finally:journal.lock.close()
    before=logical(roots)
    for _ in range(2):
        with pytest.raises(ValueError):Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=Runtime(),profiles=profiles,persistent_registration='f'*64)
        if half=='core-complete-receipt-pending':
            with pytest.raises(ValueError):Journal(roots['receipts'])
        else:
            with pytest.raises(ValueError):Core(roots['control']/'core')
        assert logical(roots)==before


def test_b22d_completed_later_ordinary_activity_no_scans_or_mutations(installation,monkeypatch):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from service import resource_store
    from tests.service.test_admission import fixture
    roots,scope,_,_=installation;intent,observation,jid=core_fixture(installation,'running')
    result=recovery._complete_receipts(roots,scope,intent,observation,now=102)
    core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        core.submit_synthetic('p','later-retry',f['context'],retry_of=jid,_admission=admission,_profiles=f['profiles'],_dispatch_settings=('silence',8))
        assert store.allocate(f['profiles'],'silence',8)
    finally:core.close()
    before=logical(roots)
    def forbidden(*args,**kwargs):raise AssertionError('completed replay scan')
    monkeypatch.setattr(Reconciliation,'_collect',forbidden);monkeypatch.setattr(resource_store,'_recovery_scan',forbidden)
    for _ in range(2):assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)==result
    assert logical(roots)==before


@pytest.mark.parametrize('damage',['none','bundle','rotation','receipt-descriptor','fence-descriptor','journal-descriptor'])
def test_b22d_prior_completed_creation_evidence_retained(installation,damage):
    roots,scope,_,_=installation;first,observation,_=core_fixture(installation)
    recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=102)
    if damage!='none':
        suffix={'bundle':'.snapshot.bundle','rotation':'.rotation.intent.json','receipt-descriptor':'.receipt-certificate.intent.json','fence-descriptor':'.completion-fence.intent.json','journal-descriptor':'.completion-journal.intent.json'}[damage]
        (roots['receipts']/('p1-recovery-'+first['id']+suffix)).write_bytes(b'changed')
    for _ in range(2):
        if damage=='none':assert recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=1000)['status']=='complete'
        else:
            with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=1000)


def retained_bytes(roots):
    return {str(p.relative_to(root))+':'+role:p.read_bytes()
            for role,root in roots.items() for p in root.rglob('*')
            if p.is_file()}


@pytest.mark.parametrize('other',['predecessor','successor'])
@pytest.mark.parametrize('damage',['none','missing','wrong','mode'])
def test_b22d_r4_other_completed_core_descriptor(installation,monkeypatch,other,damage):
    from service import resource_store
    roots,scope,_,_=installation
    first,observation,_=core_fixture(installation)
    first_result=recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    second_result=recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=102)
    damaged,requested,expected=(first,second,second_result) if other=='predecessor' else (second,first,first_result)
    path=roots['receipts']/('p1-recovery-'+damaged['id']+'.core-certificate.intent.json')
    if damage=='missing':path.unlink()
    elif damage=='wrong':path.write_bytes(b'wrong')
    elif damage=='mode':path.chmod(0o644)
    # Only the descriptor changes: actual certificates and all five SQL steps
    # remain intact, so this cannot pass merely by detecting a broken output.
    with recovery.offline(roots) as (_,db):
        data=recovery.validate_b2_database(db,roots['control']/'core')
    assert all(sum(s['recovery_id']==i['id'] for s in data['p1_recovery_steps'])==5 for i in (first,second))
    for i in (first,second):
        chain=[dict(recovery_id=i['id'],**e,lane=i['new_lane'],epoch=i['new_epoch']) for e in i['expected'][:4]]
        certificate=recovery._certificate(i,chain[-1])
        name=next(n for n,k in recovery.archive_names(i).items() if k=='certificate')
        assert (roots['control']/'core'/name).read_bytes()==encode(certificate).encode()
        assert (roots['receipts']/('p1-recovery-'+i['id']+'.certificate.json')).read_bytes()==encode(dict(certificate=certificate,steps=chain)).encode()
    before=logical(roots);files=retained_bytes(roots)
    modes={str(p):p.stat().st_mode for root in roots.values() for p in root.rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    monkeypatch.setattr(recovery,'snapshot',forbidden)
    monkeypatch.setattr(Reconciliation,'_collect',forbidden)
    monkeypatch.setattr(resource_store,'_recovery_scan',forbidden)
    monkeypatch.setattr(recovery,'_prefix_file',forbidden)
    monkeypatch.setattr(recovery,'_prefix_transaction',forbidden)
    for _ in range(2):
        if damage=='none':
            assert recovery._complete_receipts(roots,scope,requested,requested['proof']['observation'],now=1000)==expected
        else:
            with pytest.raises((ValueError,OSError)):
                recovery._complete_receipts(roots,scope,requested,requested['proof']['observation'],now=1000)
        assert logical(roots)==before
        assert retained_bytes(roots)==files
        assert {str(p):p.stat().st_mode for root in roots.values() for p in root.rglob('*') if p.is_file()}==modes


@pytest.mark.parametrize('restart',['unchanged','same-container','different-container'])
def test_b22d_correction_unfinished_current_registration(installation,restart,monkeypatch):
    from worker.runtime_journal import Journal
    from worker.runtime_supervisor import RuntimeSupervisor
    roots,scope,_,_=installation;intent,observation,_=core_fixture(installation)
    def fault(point):
        if point=='completion-fence-dirsync':raise OSError('stop-before-fifth')
    with pytest.raises(OSError,match='stop-before-fifth'):
        recovery._complete_receipts(roots,scope,intent,observation,now=102,fault=fault)
    if restart!='unchanged':
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
        supervisor.journal=Journal(roots['receipts']);supervisor.boot='e'*32
        supervisor.failed=False;supervisor.active=None;supervisor.launches=0
        try:hello=supervisor.operation('hello',{})
        finally:supervisor.journal.lock.close()
        assert hello['maintenance']['recovery_id']==intent['id']
        r=json.loads(intent['before']['source']['data']['p1_registrations'][0][1])
        r.update(lane=hello['lane'],boot=hello['boot'])
        if restart=='different-container':r['containers'][1].update(id='e'*64,started='restarted-worker')
        recovery.register(roots,scope,r)
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('unfinished resume wrote before registration validation')
    if restart!='unchanged':
        monkeypatch.setattr(recovery,'_prefix_file',forbidden)
        monkeypatch.setattr(recovery,'_prefix_transaction',forbidden)
    for _ in range(2):
        if restart=='unchanged':assert recovery._complete_receipts(roots,scope,intent,observation,now=1000)['status']=='complete'
        else:
            with pytest.raises(ValueError,match='b2_prefix_registrations_changed'):
                recovery._complete_receipts(roots,scope,intent,observation,now=1000)
            assert logical(roots)==before and retained_bytes(roots)==files
            with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
                assert db.execute("SELECT count(*) FROM p1_recovery_steps WHERE step='receipt-complete'").fetchone()==(0,)


@pytest.mark.parametrize('stage',['evidence','journal','core'])
@pytest.mark.parametrize('damage',['none','bundle','original-journal','stage-descriptor','current-journal','fence'])
def test_b22d_correction_completed_replay_pending_successor(installation,monkeypatch,stage,damage):
    from service import resource_store
    roots,scope,_,_=installation;first,observation,_=core_fixture(installation)
    expected=recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    writer={'evidence':recovery._pending_through_evidence,'journal':recovery._pending_through_journal,'core':recovery._pending_through_core}[stage]
    writer(roots,scope,second,second['proof']['observation'],now=102)
    if damage in ('current-journal','fence'):
        (roots['receipts']/('journal.json' if damage=='current-journal' else 'p1-maintenance.json')).write_bytes(b'changed')
    elif damage!='none':
        suffix={'bundle':'.snapshot.bundle','original-journal':'.journal','stage-descriptor':
            {'evidence':'.manifest.intent.json','journal':'.rotation.intent.json','core':'.core-certificate.intent.json'}[stage]}[damage]
        (roots['receipts']/('p1-recovery-'+second['id']+suffix)).write_bytes(b'changed')
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    monkeypatch.setattr(recovery,'snapshot',forbidden)
    monkeypatch.setattr(Reconciliation,'_collect',forbidden)
    monkeypatch.setattr(resource_store,'_recovery_scan',forbidden)
    monkeypatch.setattr(recovery,'_prefix_file',forbidden)
    monkeypatch.setattr(recovery,'_prefix_transaction',forbidden)
    for _ in range(2):
        if damage=='none':assert recovery._complete_receipts(roots,scope,first,observation,now=1000)==expected
        else:
            with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,first,observation,now=1000)
        assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('point',['journal-replace','completion-fence-dirsync'])
def test_b22d_correction_completed_replay_pending_exact_attempt(installation,monkeypatch,point):
    roots,scope,_,_=installation;first,observation,_=core_fixture(installation)
    expected=recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    def fault(where):
        if where==point:raise OSError(point)
    with pytest.raises(OSError,match=point):
        recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=102,fault=fault)
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    monkeypatch.setattr(recovery,'snapshot',forbidden)
    monkeypatch.setattr(recovery,'_prefix_file',forbidden)
    monkeypatch.setattr(recovery,'_prefix_transaction',forbidden)
    for _ in range(2):
        assert recovery._complete_receipts(roots,scope,first,observation,now=1000)==expected
        assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('point,deleted',[
    ('core-complete-commit',None),('core-complete-parent',None),
    ('journal-replace',None),('journal-replace','evidence'),
    ('snapshot-parent','pending'),
    ('core-intent-parent',None),('pending-receipt-commit',None),
    ('snapshot-parent',None),('manifest-parent',None),('original-journal-parent',None),
    ('evidence-receipt-commit',None),('journal-receipt-commit',None),
    ('core-certificate-parent',None),('receipt-certificate-parent',None),
    ('completion-journal-replace',None),('completion-fence-replace',None),
    ('manifest-parent','snapshot-attempt'),
    ('original-journal-parent','manifest-attempt'),
    ('receipt-certificate-parent','core-certificate-attempt'),
    ('completion-journal-replace','receipt-certificate-attempt'),
    ('completion-fence-replace','completion-journal-attempt'),
    ('core-certificate-parent','core-certificate-output'),
    ('core-certificate-parent','core-certificate-partial'),
    ('core-certificate-parent','core-certificate-wrong'),
    ('core-certificate-intent-recorded',None),
])
def test_b22d_r2_retained_successor_prefix(installation,monkeypatch,point,deleted):
    from service import resource_store
    roots,scope,_,_=installation;first,observation,_=core_fixture(installation)
    expected=recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    fired=[]
    def fault(where):
        if where==point:
            fired.append(where)
            raise OSError('r2-stop')
    with pytest.raises(OSError,match='r2-stop'):
        recovery._complete_receipts(roots,scope,second,second['proof']['observation'],now=102,fault=fault)
    assert fired==[point]
    if point.startswith('core-complete'):
        assert not (roots['receipts']/('p1-recovery-'+second['id']+'.core-certificate.intent.json')).exists()
        assert not (roots['control']/'core'/('p1-recovery-'+second['archive_id']+'.certificate.json')).exists()
    quarantine=deleted is not None or point=='core-certificate-intent-recorded'
    if deleted and deleted.endswith('-attempt'):
        (roots['receipts']/('p1-recovery-'+second['id']+'.'+deleted[:-8]+'.intent.json')).unlink()
    elif deleted and deleted.startswith('core-certificate-'):
        path=roots['control']/'core'/('p1-recovery-'+second['archive_id']+'.certificate.json')
        if deleted.endswith('output'):path.unlink()
        else:path.write_bytes(b'{' if deleted.endswith('partial') else b'{}')
    elif deleted:
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            assert db.execute('DELETE FROM p1_recovery_steps WHERE recovery_id=? AND step=?',(second['id'],deleted)).rowcount==1
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    monkeypatch.setattr(recovery,'snapshot',forbidden)
    monkeypatch.setattr(Reconciliation,'_collect',forbidden)
    monkeypatch.setattr(resource_store,'_recovery_scan',forbidden)
    monkeypatch.setattr(recovery,'_prefix_file',forbidden)
    monkeypatch.setattr(recovery,'_prefix_transaction',forbidden)
    for _ in range(2):
        if quarantine:
            with pytest.raises((ValueError,OSError)):recovery._complete_receipts(roots,scope,first,observation,now=1000)
        else:assert recovery._complete_receipts(roots,scope,first,observation,now=1000)==expected
        assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('point',['core-complete-commit','core-complete-parent'])
def test_b22d_r2_completed_replay_crash_successor_fresh_interpreter(installation,point):
    import subprocess,sys
    roots,scope,_,_=installation;first,observation,_=core_fixture(installation)
    recovery._complete_receipts(roots,scope,first,observation,now=102)
    second=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    pid=os.fork()
    if pid==0:
        recovery._pending_through_core(roots,scope,second,second['proof']['observation'],now=102,
            fault=lambda where:os._exit(73) if where==point else None)
        os._exit(74)
    assert os.waitpid(pid,0)[1]==73<<8
    rid=first['id'];del first,second,observation
    code='''import json,sys
from pathlib import Path
from tests.service.test_recovery import logical,retained_bytes
from service import recovery,resource_store
from service.reconciliation import Reconciliation
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()}
a=json.loads((roots['receipts']/('p1-recovery-'+v['rid']+'.intent.json')).read_bytes())
b=json.loads((roots['receipts']/('p1-recovery-'+'7'*32+'.intent.json')).read_bytes())
assert not (roots['receipts']/('p1-recovery-'+b['id']+'.core-certificate.intent.json')).exists()
expected=dict(status='complete',recovery_id=a['id'],lane=a['new_lane'],epoch=a['new_epoch'],certificate=recovery._core_result(a)['certificate'])
# Capture logical state first: opening crash-left SQLite can recover/checkpoint
# WAL. Physical preservation is asserted only after this explicit capture.
before=logical(roots);files=retained_bytes(roots)
def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
recovery.snapshot=Reconciliation._collect=resource_store._recovery_scan=forbidden
recovery._prefix_file=recovery._prefix_transaction=forbidden
for _ in range(2):
 assert recovery._complete_receipts(roots,v['scope'],a,a['proof']['observation'],now=1000)==expected
 assert logical(roots)==before and retained_bytes(roots)==files
'''
    child=_guarded_fresh_run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid)),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==0,child.stderr


@pytest.mark.parametrize('point,quarantine',[
    ('accepted-create',True),('accepted-write',False),('accepted-fsync',False),('accepted-parent',False),
    ('fence-create',True),('fence-write',True),('fence-fsync',True),('fence-parent',True),
    ('fence-replace-parent',False),
    ('bundle-intent-create',True),('bundle-intent-write',True),('bundle-intent-fsync',True),
    ('bundle-intent-parent',True),('bundle-intent-recorded',True),('bundle-create',True),
    ('bundle-write',False),('bundle-fsync',False),('bundle-parent',False),
])
def test_b22d_r3_receipt_only_successor(installation,monkeypatch,point,quarantine):
    from service import resource_store
    roots,scope,_,_=installation;a,observation,_=core_fixture(installation)
    expected=recovery._complete_receipts(roots,scope,a,observation,now=102)
    b=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    fired=[]
    def stop(where):
        if where==point:
            fired.append(where);raise OSError('r3-stop')
    with pytest.raises(OSError,match='r3-stop'):
        recovery._complete_receipts(roots,scope,b,b['proof']['observation'],now=102,fault=stop)
    assert fired==[point]
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    with monkeypatch.context() as m:
        for obj,name in ((recovery,'snapshot'),(Reconciliation,'_collect'),(resource_store,'_recovery_scan'),(recovery,'_prefix_file'),(recovery,'_prefix_transaction')):
            m.setattr(obj,name,forbidden)
        for _ in range(2):
            if quarantine:
                with pytest.raises((ValueError,OSError)):
                    recovery._complete_receipts(roots,scope,a,observation,now=1000)
            else:assert recovery._complete_receipts(roots,scope,a,observation,now=1000)==expected
            assert logical(roots)==before and retained_bytes(roots)==files
    if quarantine:return
    assert recovery._complete_receipts(roots,scope,b,b['proof']['observation'],now=1000)['status']=='complete'
    assert recovery._complete_receipts(roots,scope,a,observation,now=1000)==expected


@pytest.mark.parametrize('damage',[
    'accepted-missing','accepted-partial','accepted-unknown','accepted-noncanonical','accepted-mode',
    'accepted-symlink','accepted-hardlink','filename','ambiguous','plan','prior-binding','scope','proof',
    'descriptor','descriptor-missing','descriptor-partial','bundle-missing','bundle-partial','bundle-wrong',
    'bundle-mode','bundle-symlink','bundle-hardlink','fence','fence-missing','marker','marker-missing',
    'journal','unknown-extra','future-output','future-attempt','old-core-attempt','later-plan',
])
def test_b22d_r3_receipt_only_tamper(installation,monkeypatch,damage):
    from service import resource_store
    roots,scope,_,_=installation;a,observation,_=core_fixture(installation)
    recovery._complete_receipts(roots,scope,a,observation,now=102)
    b=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    def stop(where):
        if where=='bundle-parent':raise OSError('r3-stop')
    with pytest.raises(OSError,match='r3-stop'):
        recovery._complete_receipts(roots,scope,b,b['proof']['observation'],now=102,fault=stop)
    receipts=roots['receipts'];prefix='p1-recovery-'+b['id']
    accepted=receipts/(prefix+'.intent.json');bundle=receipts/(prefix+'.snapshot.bundle')
    descriptor=receipts/(prefix+'.bundle.intent.json')
    if damage in ('plan','later-plan'):
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            plan=json.loads(db.execute('SELECT record FROM p1_abandon_plans WHERE id=?',(b['plan_id'],)).fetchone()[0])
            if damage=='plan':
                plan['created']+=1
                db.execute('UPDATE p1_abandon_plans SET record=?,digest=? WHERE id=?',(encode(plan),digest(plan),plan['id']))
            else:
                plan['id']='8'*32
                db.execute('INSERT INTO p1_abandon_plans VALUES(?,?,?)',(plan['id'],encode(plan),digest(plan)))
    elif damage in ('prior-binding','scope','proof','accepted-unknown'):
        changed=json.loads(accepted.read_bytes())
        if damage=='prior-binding':changed['before']['prior_certificate']['certificate_digest']='f'*64
        elif damage=='scope':changed['proof']['observation']['scope']['context']='changed'
        elif damage=='proof':changed['proof']['kind']='host-admin-container-teardown-v1'
        else:changed['unknown']=True
        accepted.write_text(encode(changed))
    elif damage=='accepted-noncanonical':accepted.write_text(json.dumps(b,indent=2))
    elif damage=='filename':accepted.rename(receipts/('p1-recovery-'+'8'*32+'.intent.json'))
    elif damage=='ambiguous':
        extra=receipts/('p1-recovery-'+'8'*32+'.intent.json');extra.write_bytes(accepted.read_bytes());extra.chmod(0o600)
    elif damage=='descriptor':
        value=json.loads(descriptor.read_bytes());value['intent_digest']='f'*64;descriptor.write_text(encode(value))
    elif damage in ('fence','marker'):
        path=receipts/('p1-maintenance.json' if damage=='fence' else 'p1-b2-required.json')
        value=json.loads(path.read_bytes());value['deployment_id']='f'*32;path.write_text(encode(value))
    elif damage=='journal':(receipts/'journal.json').write_bytes(b'changed')
    elif damage=='unknown-extra':(receipts/'unknown.json').write_bytes(b'{}')
    elif damage=='future-output':(receipts/(prefix+'.journal')).write_bytes(bytes.fromhex(b['before']['journal']['bytes_hex']))
    elif damage=='future-attempt':(receipts/(prefix+'.snapshot.intent.json')).write_bytes(b'{}')
    elif damage=='old-core-attempt':(receipts/('p1-recovery-'+a['id']+'.core-certificate.intent.json')).unlink()
    else:
        kind,change=damage.rsplit('-',1)
        path={'accepted':accepted,'bundle':bundle,'descriptor':descriptor,'fence':receipts/'p1-maintenance.json','marker':receipts/'p1-b2-required.json'}[kind]
        if change=='missing':path.unlink()
        elif change=='partial':path.write_bytes(b'{')
        elif change=='wrong':path.write_bytes(b'wrong')
        elif change=='mode':path.chmod(0o644)
        else:
            target=receipts/'tamper-target';path.rename(target)
            if change=='symlink':path.symlink_to(target)
            else:os.link(target,path)
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    for obj,name in ((recovery,'snapshot'),(Reconciliation,'_collect'),(resource_store,'_recovery_scan'),(recovery,'_prefix_file'),(recovery,'_prefix_transaction')):
        monkeypatch.setattr(obj,name,forbidden)
    for _ in range(2):
        with pytest.raises((ValueError,OSError)):
            recovery._complete_receipts(roots,scope,a,observation,now=1000)
        assert logical(roots)==before and retained_bytes(roots)==files


def test_b22d_r3_receipt_only_crash_fresh_interpreter(installation):
    import subprocess,sys
    roots,scope,_,_=installation;a,observation,snapshot_bytes=core_fixture(installation)
    recovery._complete_receipts(roots,scope,a,observation,now=102)
    b=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    del snapshot_bytes
    pid=os.fork()
    if pid==0:
        recovery._complete_receipts(roots,scope,b,b['proof']['observation'],now=102,
            fault=lambda where:os._exit(73) if where=='bundle-parent' else None)
        os._exit(74)
    assert os.waitpid(pid,0)[1]==73<<8
    rid=a['id'];del a,b,observation
    code='''import json,sys,re
from pathlib import Path
from unittest.mock import patch
from tests.service.test_recovery import logical,retained_bytes
from service import recovery,resource_store
from service.reconciliation import Reconciliation
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()}
a=json.loads((roots['receipts']/('p1-recovery-'+v['rid']+'.intent.json')).read_bytes())
before=logical(roots);files=retained_bytes(roots)
expected=dict(recovery._core_result(a),status='complete')
def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
with patch.object(recovery,'snapshot',forbidden),patch.object(Reconciliation,'_collect',forbidden),patch.object(resource_store,'_recovery_scan',forbidden),patch.object(recovery,'_prefix_file',forbidden),patch.object(recovery,'_prefix_transaction',forbidden):
 for _ in range(2):
  assert recovery._complete_receipts(roots,v['scope'],a,a['proof']['observation'],now=1000)==expected
  assert logical(roots)==before and retained_bytes(roots)==files
candidates=[p for p in roots['receipts'].iterdir() if re.fullmatch(r'p1-recovery-[0-9a-f]{32}\\.intent\\.json',p.name) and p.name!='p1-recovery-'+v['rid']+'.intent.json']
assert len(candidates)==1
b=json.loads(candidates[0].read_bytes())
assert recovery._complete_receipts(roots,v['scope'],b,b['proof']['observation'],now=1000)['status']=='complete'
assert recovery._complete_receipts(roots,v['scope'],a,a['proof']['observation'],now=1000)==expected
'''
    child=_guarded_fresh_run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid)),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==0,child.stderr


def test_b22d_r3_receipt_only_marker_precedes_acceptance(installation,monkeypatch):
    roots,scope,_,_=installation;a,observation,_=core_fixture(installation)
    expected=recovery._complete_receipts(roots,scope,a,observation,now=102)
    b=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    writer=recovery._prefix_file
    def stop(path,raw,fault,point):
        if point=='accepted':raise OSError('before-accepted-attempt')
        return writer(path,raw,fault,point)
    with monkeypatch.context() as m:
        m.setattr(recovery,'_prefix_file',stop)
        with pytest.raises(OSError,match='before-accepted-attempt'):
            recovery._complete_receipts(roots,scope,b,b['proof']['observation'],now=102)
    assert not (roots['receipts']/('p1-recovery-'+b['id']+'.intent.json')).exists()
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('completed replay scan or write')
    from service import resource_store
    for obj,name in ((recovery,'snapshot'),(Reconciliation,'_collect'),(resource_store,'_recovery_scan'),(recovery,'_prefix_file'),(recovery,'_prefix_transaction')):
        monkeypatch.setattr(obj,name,forbidden)
    for _ in range(2):
        assert recovery._complete_receipts(roots,scope,a,observation,now=1000)==expected
        assert logical(roots)==before and retained_bytes(roots)==files


# B2.3a E1: the Docker, /proc and mount transport boundaries below are explicitly
# fake. No production option/flag turns a mock document into host authority.
@pytest.fixture
def b23_host(installation, tmp_path, monkeypatch):
    import time
    from service import host_install as module
    roots,scope,_,_=installation
    def forbidden(*a,**kw):pytest.fail('host authority must not call private kernel')
    for name in ('_complete_receipts','_pending_through_core','_pending_through_journal','_pending_through_evidence'):
        monkeypatch.setattr(recovery,name,forbidden)
    r=registration(installation,'actual-host-inspection')
    for index,c in enumerate(r['containers']): c['started']=f'2026-10-17T01:00:0{index}Z'
    recovery.register(roots,scope,r)
    state=module.host_state(roots,scope)
    base=tmp_path/'fake-host-transport'; base.mkdir(mode=0o700)
    monkeypatch.setattr(recovery,'HOST_AUTHORITY_DIRECTORY',str(base))
    # Fake the host directory ancestor boundary (/tmp is deliberately not a
    # production trusted ancestor). Real leaf/file ownership/mode checks remain.
    def directory(path,*,create=False):
        if create:
            try: Path(path).mkdir(mode=0o700)
            except FileExistsError: pass
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        import stat
        st=os.fstat(fd)
        if st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700:
            os.close(fd); raise ValueError('host_unsafe_directory')
        return fd
    monkeypatch.setattr(module,'_protected_directory',directory)
    monkeypatch.setattr(module.sys,'platform','linux')
    identities={f'/fake/docker/volumes/{role}/_data':(7,index+1) for index,role in enumerate(sorted(scope['volumes']))}
    monkeypatch.setattr(module,'_volume_identity',lambda source:identities[source])
    consumers=[]; proc=[]
    for index,c in enumerate(r['containers']):
        consumers.append(dict(id=c['id'],image=c['image'],started=c['started'],finished='0001-01-01T00:00:00Z',
            running=True,restarting=False,pid=100+index,restart='no',pidmode='',privileged=False,capadd=None,cgroupns='private',mounts=[
                dict(type='volume',name=scope['volumes'][role]['name'],source=f'/fake/docker/volumes/{role}/_data',
                    destination='/'+role,rw=c['role']!='control' or role not in ('ipc','staging')) for role in c['volumes']]))
        proc.append(dict(id=c['id'],image=c['image'],started=c['started'],pid=100+index,start_ticks=1000+index,
            pid_namespace=10000+index,cgroup='/docker/'+c['id']))
    def process(pid,identity,ipc):
        p=next(p for p in proc if p['id']==identity)
        assert pid==p['pid'] and Path(ipc)==roots['ipc']
        return {k:p[k] for k in ('pid','start_ticks','pid_namespace','cgroup')}
    monkeypatch.setattr(module,'_proc_observation',process)
    ipc=roots['ipc'].stat()
    provenance=dict(revision=1,kind='local-linux-proc-v1',registration_digest=digest(r),
        ipc_identity=[ipc.st_dev,ipc.st_ino],hello={k:r[k] for k in ('lane','boot')},owners=proc)
    module._host_write(base/('registration-'+digest(r)+'.json'),provenance)
    calls=[]
    observer=module.Docker(scope['context'])
    def read(argv):
        calls.append(argv)
        if argv[:2]==['context','inspect']:
            assert argv==['context','inspect',scope['context'],'--format','{{json .Endpoints.docker.Host}}']
            return json.dumps('unix:///var/run/docker.sock')
        if argv[0]=='info':
            assert argv==['info','--format','{"id":{{json .ID}},"os":{{json .OSType}}}']
            return encode(dict(id=scope['daemon'],os='linux'))
        if argv[:2]==['volume','inspect']:
            assert argv[3:]==['--format',module.VOLUME_FORMAT]
            role=next(k for k,v in scope['volumes'].items() if v['name']==argv[2])
            return encode(dict(**scope['volumes'][role],source=f'/fake/docker/volumes/{role}/_data',options=None))
        if argv[:2]==['container','ls']:
            assert argv==['container','ls','--all','--quiet','--no-trunc']
            return '\n'.join(c['id'] for c in consumers)
        c=next(c for c in consumers if c['id']==argv[-1 if argv[1] in ('stop','wait') else 2])
        if argv[1]=='inspect':
            assert argv==['container','inspect',c['id'],'--format',module.CONTAINER_FORMAT]
            return encode(c)
        if argv[1]=='stop':
            assert argv==['container','stop','--time','2',c['id']]
            if c['running']: c.update(running=False,pid=0,finished='2026-10-17T02:00:00Z')
            return c['id']
        assert argv==['container','wait',c['id']]
        return '137'  # docker wait succeeds even when the container exited nonzero.
    observer._read=read
    request=dict(deployment_id=state['deployment_id'],scope=scope,
        registration_digest=recovery.host_registration_digest(state['registrations']),
        purpose=dict(kind='pre-plan',plan_id=None,plan_digest=None),original_invocation=None)
    return dict(module=module,roots=roots,scope=scope,state=state,registration=r,observer=observer,request=request,
        consumers=consumers,provenance=provenance,proc=proc,base=base,calls=calls,read=read)


def test_b23_closed_capsule_freeze_and_apply_closed(b23_host):
    h=b23_host; m=h['module']; before=logical(h['roots'])
    _,_,preflight,_=m.host_preflight(h['observer'],h['state'],h['request'])
    assert all(c['running'] for c in preflight['consumers'])
    assert not any(c[1] in ('stop','wait') for c in h['calls'])
    capsule=m.freeze_host(h['observer'],h['state'],h['request'])
    assert recovery.validate_host_capsule(capsule,now=capsule['issued'])==capsule
    assert capsule['kernel_proof_digest']==digest(capsule['kernel_proof'])
    assert set(capsule['kernel_proof'])=={'kind','observation','prior_epoch','old_lane'}
    assert capsule['waits']==[dict(id=c['id'],command_status=0,container_status=137) for c in h['registration']['containers']]
    actions=[c for c in h['calls'] if c[1] in ('stop','wait')]
    control,worker=h['registration']['containers']
    assert actions==[['container','stop','--time','2',control['id']],['container','wait',control['id']],
                     ['container','stop','--time','2',worker['id']],['container','wait',worker['id']]]
    path=h['base']/'invocations'/capsule['invocation_id']/'capsule.json'
    assert recovery.host_json(path.read_bytes())==capsule and path.stat().st_mode&0o777==0o600
    assert logical(h['roots'])==before
    with pytest.raises(NotImplementedError,match='P1B2_apply_unavailable'): recovery.apply(capsule)
    assert len(h['consumers'])==2 and all(not c['running'] for c in h['consumers'])


@pytest.mark.parametrize('fault',['context','daemon','volume','image','start','missing','unknown-live','unknown-stopped',
    'duplicate-id','restart','pidmode','pid','privileged','capadd','cgroupns','extra-mount','mount-mode','mount-source','alias-volume','enum-drift',
    'old-registration','ambiguous-registration','E1','scope-request','registration-digest','purpose','proc','cgroup','provenance-missing'])
def test_b23_preflight_refusals_before_stop_or_mutation(b23_host,monkeypatch,fault):
    h=b23_host; m=h['module']; r=h['registration']; cs=h['consumers']; read=h['read']; before=logical(h['roots'])
    if fault=='context': h['observer'].context='other'
    elif fault=='image': cs[0]['image']='sha256:'+'a'*64
    elif fault=='start': cs[0]['started']='2026-10-17T01:30:00Z'
    elif fault=='missing': cs.pop()
    elif fault.startswith('unknown'):
        c=copy.deepcopy(cs[1]);c['id']='e'*64
        if fault=='unknown-stopped':c.update(running=False,pid=0,finished='2026-10-17T02:00:00Z')
        cs.append(c)
    elif fault=='restart':cs[0]['restart']='always'
    elif fault=='pidmode':cs[0]['pidmode']='host'
    elif fault=='pid':cs[0]['pid']=0
    elif fault=='privileged':cs[0]['privileged']=True
    elif fault=='capadd':cs[0]['capadd']=['SYS_ADMIN']
    elif fault=='cgroupns':cs[0]['cgroupns']='host'
    elif fault=='extra-mount':cs[0]['mounts'].append(dict(type='bind',name='',source='/other',destination='/other',rw=True))
    elif fault=='mount-mode':cs[0]['mounts'][0]['rw']=False
    elif fault=='mount-source':cs[0]['mounts'][0]['source']='/other'
    elif fault in ('old-registration','ambiguous-registration'):
        old=copy.deepcopy(r);old['boot']='f'*32
        if fault=='old-registration':old['containers'][1]['id']='e'*64
        else:old['containers'][1]['started']='2026-10-17T00:00:00Z'
        h['state']['registrations'].append(old)
        h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    elif fault=='E1':h['state']['registrations'][0]['evidence']='E1-mock';h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    elif fault=='scope-request':h['request']['scope']={**h['scope'],'daemon':'other'}
    elif fault=='registration-digest':h['request']['registration_digest']='0'*64
    elif fault=='purpose':h['request']['purpose']['argv']=['rm']
    elif fault=='proc':monkeypatch.setattr(m,'_proc_observation',lambda *a:dict(pid=999,start_ticks=1,pid_namespace=2,cgroup='/docker/'+'c'*64))
    elif fault=='cgroup':
        h['provenance']['owners'][0]['cgroup']='/guessed/parent'
        (h['base']/('registration-'+digest(r)+'.json')).write_text(encode(h['provenance']))
    elif fault=='provenance-missing':(h['base']/('registration-'+digest(r)+'.json')).unlink()
    counter=0
    def changed(argv):
        nonlocal counter
        result=read(argv)
        if fault=='daemon' and argv[0]=='info':return encode(dict(id='wrong',os='linux'))
        if fault in ('volume','alias-volume') and argv[0]=='volume':
            value=json.loads(result)
            if fault=='volume':value['created']='wrong'
            else:value['source']='/fake/docker/volumes/control/_data'
            return encode(value)
        if fault=='duplicate-id' and argv[1]=='ls':return result+'\n'+cs[0]['id']
        if fault=='enum-drift' and argv[1]=='ls':
            counter+=1
            if counter%2==0:return result+'\n'+'f'*64
        return result
    h['observer']._read=changed
    with pytest.raises((ValueError,OSError)):m.freeze_host(h['observer'],h['state'],h['request'])
    assert not any(c[1] in ('stop','wait') for c in h['calls'])
    assert not (h['base']/'invocations').exists() and logical(h['roots'])==before


@pytest.mark.parametrize('fault',['wait-error','wait-timeout','wait-output','finish','restart-after-stop','final-drift','registration-drift','deadline'])
def test_b23_freeze_failure_retains_containers_closed_authority(b23_host,monkeypatch,fault):
    h=b23_host; m=h['module']; read=h['read']; before=logical(h['roots']); stopped=False; inspections=0
    def fail(argv):
        nonlocal stopped,inspections
        if argv[1]=='wait' and fault in ('wait-error','wait-timeout'):raise ValueError(fault)
        result=read(argv)
        if argv[1]=='stop':stopped=True
        if argv[1]=='wait' and fault=='wait-output':return 'not-status'
        if stopped and argv[:2]==['container','inspect']:
            inspections+=1;v=json.loads(result)
            if fault=='finish':v['finished']=''
            if fault=='restart-after-stop':v['started']='2026-10-17T01:50:00Z'
            if fault=='final-drift' and inspections>=3:v['image']='sha256:'+'f'*64
            return encode(v)
        return result
    h['observer']._read=fail
    if fault=='registration-drift':
        monkeypatch.setattr(m,'host_state',lambda *a:{**h['state'],'lane':'e'*32})
    if fault=='deadline':h['observer'].deadline=0
    with pytest.raises((ValueError,OSError)):m.freeze_host(h['observer'],h['state'],h['request'])
    assert len(h['consumers'])==2 and logical(h['roots'])==before
    assert not (h['base']/'invocations').exists()
    assert all(c[1] not in ('rm','prune','create','restart','start') for c in h['calls'])


def test_b23_fresh_resume_retains_original_proof_and_rejects_restarted_owner(b23_host):
    h=b23_host;m=h['module']; original=m.freeze_host(h['observer'],h['state'],h['request'])
    proof=copy.deepcopy(original['kernel_proof']); h['calls'].clear()
    request={**h['request'],'original_invocation':original['invocation_id'],
        'purpose':dict(kind='resume',plan_id='a'*32,plan_digest='b'*64)}
    fresh=m.freeze_host(h['observer'],h['state'],request)
    assert fresh['invocation_id']!=original['invocation_id'] and fresh['kernel_proof']==proof
    assert fresh['kernel_proof_digest']==original['kernel_proof_digest']
    assert fresh['issued']>=original['issued']
    assert not any(c[1]=='stop' for c in h['calls']) and sum(c[1]=='wait' for c in h['calls'])==2
    assert (h['base']/'invocations'/original['invocation_id']/'capsule.json').exists()
    h['calls'].clear();h['consumers'][1].update(started='2026-10-17T03:00:00Z',finished='2026-10-17T04:00:00Z')
    with pytest.raises(ValueError):m.freeze_host(h['observer'],h['state'],request)
    assert not any(c[1] in ('stop','wait') for c in h['calls'])


@pytest.mark.parametrize('fault',['unknown','nested','bool','nonfinite','duplicate-registration','missing-wait','duplicate-wait',
    'wait-failed','E1','forged-hash','forged-label','omitted-provenance','bad-cgroup','same-pidns','expiry','oversize','duration','finished'])
def test_b23_closed_capsule_negative_matrix(b23_host,fault):
    h=b23_host;c=copy.deepcopy(h['module'].freeze_host(h['observer'],h['state'],h['request']))
    if fault=='unknown':c['authorization']=True
    elif fault=='nested':c['topology']['consumers'][0]['trusted_label']='yes'
    elif fault=='bool':c['revision']=True
    elif fault=='nonfinite':c['issued']=float('nan')
    elif fault=='duplicate-registration':c['registrations']*=2
    elif fault=='missing-wait':c['waits'].pop()
    elif fault=='duplicate-wait':c['waits'][1]=c['waits'][0]
    elif fault=='wait-failed':c['waits'][0]['command_status']=1
    elif fault=='E1':c['kernel_proof']['kind']='E1-mock';c['kernel_proof_digest']=digest(c['kernel_proof'])
    elif fault=='forged-hash':c['kernel_proof_digest']='0'*64
    elif fault=='forged-label':c['kind']='actual-host-inspection'
    elif fault=='omitted-provenance':c['provenance']=[]
    elif fault=='bad-cgroup':c['provenance'][0]['owners'][0]['cgroup']='/parent'
    elif fault=='same-pidns':c['provenance'][0]['owners'][1]['pid_namespace']=c['provenance'][0]['owners'][0]['pid_namespace']
    elif fault=='expiry':c['expires']=c['issued']+301
    elif fault=='oversize':c['kind']='x'*(recovery.HOST_CAPSULE_BYTES+1)
    elif fault=='duration':c['collection_seconds']=121
    elif fault=='finished':c['topology']['consumers'][0]['finished']=''
    with pytest.raises((ValueError,TypeError)):recovery.validate_host_capsule(c,now=c['issued'])


def _b23_fake_mount_transport(monkeypatch,h,capsule):
    """Fake only Linux mount table/open-at-fixed-mount; real protected files."""
    directory=h['base']/'invocations'/capsule['invocation_id']
    device=directory.stat().st_dev
    mounts=[dict(id='10',device=f'{os.major(device)}:{os.minor(device)}',
        root=str(directory),target=recovery.HOST_CAPSULE_MOUNT,options=['ro'])]
    volumes={v['role']:v for v in capsule['topology']['volumes']}
    mounts += [dict(id=str(20+i),device=f"{os.major(volumes[role]['device'])}:{os.minor(volumes[role]['device'])}",
        root=volumes[role]['source'],target='/'+role,options=['rw'])
        for i,role in enumerate(('control','staging','receipts','ipc'))]
    monkeypatch.setattr(recovery,'_host_role_identity',lambda role:(volumes[role]['device'],volumes[role]['inode']))
    monkeypatch.setattr(recovery,'_host_mount_topology',lambda:copy.deepcopy(mounts))
    # Explicit test OS mapping; no production owner option is exposed.
    monkeypatch.setattr(recovery,'HOST_MAPPED_UID',os.getuid())
    real_open=os.open
    def mapped(path,*args,**kwargs):
        if path==recovery.HOST_CAPSULE_MOUNT:path=directory
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(recovery.os,'open',mapped)
    return directory,mounts


def test_b23_reader_exact_fixed_ro_transport_retains_evidence(b23_host,monkeypatch):
    h=b23_host;c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    directory,_=_b23_fake_mount_transport(monkeypatch,h,c)
    before=(directory/'capsule.json').read_bytes()
    assert recovery.read_host_capsule()==c and recovery.read_host_capsule()==c
    assert (directory/'capsule.json').read_bytes()==before
    with pytest.raises(NotImplementedError):recovery.apply(c)


@pytest.mark.parametrize('fault',['absent','rw','ordinary-path','alias','nested-mount','wrong-device','missing-role',
    'owner','file-owner','directory-mode','mode','hardlink','symlink','oversize','duplicate-key','E1','mount-drift','file-replacement'])
def test_b23_reader_refuses_untrusted_transport(b23_host,monkeypatch,fault):
    h=b23_host;c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    directory,mounts=_b23_fake_mount_transport(monkeypatch,h,c); path=directory/'capsule.json'
    if fault=='absent':mounts.clear()
    elif fault=='rw':mounts[0]['options']=['rw']
    elif fault=='ordinary-path':mounts[0]['target']='/control/proof'
    elif fault=='alias':mounts.append({**mounts[0],'id':'99','target':'/control/proof','options':['rw']})
    elif fault=='nested-mount':mounts.append({**mounts[0],'id':'99','target':recovery.HOST_CAPSULE_MOUNT+'/capsule.json'})
    elif fault=='wrong-device':mounts[0]['device']='999:999'
    elif fault=='missing-role':mounts.pop()
    elif fault=='owner':monkeypatch.setattr(recovery,'HOST_MAPPED_UID',os.getuid()+1)
    elif fault=='file-owner':
        original_fstat=os.fstat;inode=path.stat().st_ino
        def foreign(fd):
            st=original_fstat(fd)
            if st.st_ino==inode:
                values=list(st);values[4]=os.getuid()+1;return os.stat_result(values)
            return st
        monkeypatch.setattr(recovery.os,'fstat',foreign)
    elif fault=='directory-mode':directory.chmod(0o755)
    elif fault=='mode':path.chmod(0o644)
    elif fault=='hardlink':os.link(path,directory.parent/'extra-hardlink')
    elif fault=='symlink':path.rename(directory.parent/'saved');path.symlink_to(directory.parent/'saved')
    elif fault=='oversize':path.write_bytes(b'x'*(recovery.HOST_CAPSULE_BYTES+1))
    elif fault=='duplicate-key':path.write_bytes(b'{"revision":1,"revision":1}')
    elif fault=='E1':c['kernel_proof']['kind']='E1-mock';path.write_text(encode(c))
    elif fault=='mount-drift':
        count=0
        def changed():
            nonlocal count
            count+=1
            return copy.deepcopy(mounts) if count==1 else []
        monkeypatch.setattr(recovery,'_host_mount_topology',changed)
    elif fault=='file-replacement':
        original=recovery.host_json
        def replace(raw):
            value=original(raw);path.rename(directory.parent/'replaced');path.write_bytes(raw);path.chmod(0o600)
            return value
        monkeypatch.setattr(recovery,'host_json',replace)
    with pytest.raises((ValueError,OSError)):recovery.read_host_capsule()
    assert len(h['consumers'])==2


def test_b23_shipped_reader_absent_mount_never_falls_back(monkeypatch):
    monkeypatch.setattr(recovery,'_host_mount_topology',lambda:[])
    monkeypatch.setenv('P1_HOST_AUTHORITY','actual-host-inspection')
    monkeypatch.setenv('P1_FORCE','1')
    with pytest.raises(ValueError,match='trusted_mount_absent'):recovery.read_host_capsule()
    with pytest.raises(ValueError):recovery.host_json(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):recovery.host_json(b'{"a":NaN}')
    with pytest.raises(ValueError):recovery.host_json(b'{"a":1e999}')


@pytest.mark.parametrize('fault',['none','stdout','stderr','exit','process-timeout','global-timeout','spawn'])
def test_b23_fixed_subprocess_bounds_and_sanitized_errors(monkeypatch,fault):
    import time
    from service import host_install as m
    observer=m.Docker('pinned-context');calls=[];processes=[]
    class Process:
        def __init__(self,argv,**kwargs):
            calls.append((argv,kwargs))
            assert argv==['docker','--context','pinned-context','container','wait','c'*64]
            assert kwargs.get('shell',False) is False and kwargs['stdin']==m.subprocess.DEVNULL
            self.returncode=1 if fault=='exit' else 0;self.killed=False;self.waited=False
            kwargs['stdout'].write(b'x'*(1024*1024+1) if fault=='stdout' else b'137\n')
            kwargs['stderr'].write(b'secret-'*10000 if fault=='stderr' else b'')
            kwargs['stdout'].flush();kwargs['stderr'].flush();processes.append(self)
        def poll(self):return None if fault=='process-timeout' and not self.killed else self.returncode
        def kill(self):self.killed=True
        def wait(self,timeout):assert timeout==2;self.waited=True;return self.returncode
    def spawn(*a,**kw):
        if fault=='spawn':raise OSError('secret-host-information')
        return Process(*a,**kw)
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    if fault=='global-timeout':observer.deadline=0
    if fault=='process-timeout':
        clock=[observer.started]
        def monotonic():clock[0]+=2;return clock[0]
        monkeypatch.setattr(m.time,'monotonic',monotonic)
        monkeypatch.setattr(m.time,'sleep',lambda _:None)
    if fault=='none':assert observer._read(['container','wait','c'*64])=='137'
    else:
        with pytest.raises(ValueError) as error:observer._read(['container','wait','c'*64])
        assert 'secret' not in str(error.value)
    if fault=='global-timeout':assert calls==[]
    if processes:assert processes[0].waited
    if fault=='process-timeout':assert processes[0].killed


def test_b23_registration_brackets_authenticated_hello(b23_host,monkeypatch):
    from service.supervisor_runtime import SupervisorRuntime
    h=b23_host;m=h['module'];r=h['registration'];events=[]
    path=h['base']/('registration-'+digest(r)+'.json');path.unlink()
    # Fake only the authenticated runtime transport; production register_host
    # still performs both inventory/proc brackets and writes the protected file.
    def hello(self,operation,args):
        events.append('hello');assert operation=='hello' and args=={}
        return dict(lane=r['lane'],boot=r['boot'],quarantined=False)
    monkeypatch.setattr(SupervisorRuntime,'call',hello)
    monkeypatch.setattr(recovery,'register',lambda *a:events.append('register') or digest(r))
    assert m.register_host(h['observer'],h['roots'],h['scope'],r)==digest(r)
    assert events==['hello','register']
    assert recovery.host_json(path.read_bytes())==h['provenance']


@pytest.mark.parametrize('fault',['hello-boot','hello-lane','hello-quarantine','proc-drift','host-drift','remote','desktop','shared-pidns'])
def test_b23_registration_provenance_refuses_before_registration(b23_host,monkeypatch,fault):
    from service.supervisor_runtime import SupervisorRuntime
    h=b23_host;m=h['module'];r=h['registration'];path=h['base']/('registration-'+digest(r)+'.json');path.unlink()
    def hello(self,*args):
        value=dict(lane=r['lane'],boot=r['boot'],quarantined=False)
        if fault=='hello-boot':value['boot']='a'*32
        elif fault=='hello-lane':value['lane']='a'*32
        elif fault=='hello-quarantine':value['quarantined']=True
        elif fault=='host-drift':h['consumers'][1]['started']='2026-10-17T03:00:00Z'
        return value
    monkeypatch.setattr(SupervisorRuntime,'call',hello)
    monkeypatch.setattr(recovery,'register',lambda *a:pytest.fail('unauthorized registration mutation'))
    if fault=='proc-drift':
        real=m._proc_observation;count=0
        def changed(*args):
            nonlocal count
            count+=1;value=real(*args)
            if count>2:value['start_ticks']+=1
            return value
        monkeypatch.setattr(m,'_proc_observation',changed)
    if fault=='shared-pidns':h['proc'][1]['pid_namespace']=h['proc'][0]['pid_namespace']
    if fault=='desktop':monkeypatch.setattr(m.sys,'platform','darwin')
    if fault=='remote':
        read=h['read']
        def remote(argv):return json.dumps('ssh://remote') if argv[0]=='context' else read(argv)
        h['observer']._read=remote
    with pytest.raises((ValueError,OSError)):m.register_host(h['observer'],h['roots'],h['scope'],r)
    assert not path.exists() and not any(c[1] in ('stop','wait') for c in h['calls'])


def test_b23_closed_authority_entrypoints_exist():
    from service import host_install as m
    assert callable(getattr(m,'freeze_host',None))
    assert callable(getattr(m,'host_preflight',None))
    assert callable(getattr(recovery,'read_host_capsule',None))


def test_b23_retained_inventory_works_with_both_live_owner_locks(b23_host):
    h=b23_host
    with (h['roots']['control']/'core/control.lock').open('rb') as control, (h['roots']['receipts']/'lane.lock').open('rb') as worker:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=logical(h['roots'])
        assert h['module'].host_state(h['roots'],h['scope'])==h['state']
        assert logical(h['roots'])==before


def test_b23_resume_does_not_replace_original_lane_epoch_after_rotation(b23_host,monkeypatch):
    h=b23_host;m=h['module'];c=m.freeze_host(h['observer'],h['state'],h['request'])
    state={**h['state'],'lane':'e'*32,'prior_epoch':'f'*32,'unfinished':True}
    monkeypatch.setattr(m,'host_state',lambda *a:state)
    request={**h['request'],'original_invocation':c['invocation_id'],'purpose':dict(kind='resume',plan_id='a'*32,plan_digest='b'*64)}
    fresh=m.freeze_host(h['observer'],state,request)
    assert fresh['kernel_proof']==c['kernel_proof'] and fresh['kernel_proof']['old_lane']!=state['lane']


@pytest.mark.parametrize('fault',['wrong-root','wrong-inode','wrong-device','role-alias'])
def test_b23_reader_checks_actual_role_mounts(b23_host,monkeypatch,fault):
    h=b23_host;c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    _,mounts=_b23_fake_mount_transport(monkeypatch,h,c)
    if fault=='wrong-root':mounts[1]['root']='/ordinary/service'
    elif fault=='wrong-device':mounts[1]['device']='999:999'
    elif fault=='role-alias':mounts[1]['root']=mounts[0]['root'];mounts[1]['device']=mounts[0]['device']
    else:monkeypatch.setattr(recovery,'_host_role_identity',lambda role:(999,999))
    with pytest.raises(ValueError):recovery.read_host_capsule()


@pytest.mark.parametrize('fault',['none','foreign-cgroup','shared-namespace','child-namespace','child-escape','pid-reuse',
    'ipc-alias','cgroup-symlink','oversize','deadline','no-cgroup-v2'])
def test_b23_local_proc_tree_os_boundary(tmp_path,monkeypatch,fault):
    """Synthetic procfs/cgroupfs only; no host process or daemon is inspected."""
    import builtins
    from service import host_install as m
    real_path=Path;fake=tmp_path/'fake-linux';identity='d'*64
    def mapped(value):
        value=str(value)
        if value.startswith('/proc') or value.startswith('/sys/fs/cgroup'):return fake/value.lstrip('/')
        return real_path(value)
    ipc=tmp_path/'ipc';ipc.mkdir()
    for pid in ('100','101','self'):
        root=mapped('/proc')/pid; (root/'ns').mkdir(parents=True)
        (root/'ns/pid').write_text('namespace')
        (root/'cgroup').write_text('0::/\n' if pid=='self' else '0::/docker/'+identity+'\n')
        (root/'stat').write_text(pid+' (fake worker) '+' '.join(['S']+['0']*18+['1000']))
    (mapped('/proc')/'101/ns/pid').unlink()
    os.link(mapped('/proc')/'100/ns/pid',mapped('/proc')/'101/ns/pid')
    (mapped('/proc')/'100/root').mkdir();(mapped('/proc')/'100/root/ipc').symlink_to(ipc,target_is_directory=True)
    cgroup=mapped('/sys/fs/cgroup')/'docker'/identity;cgroup.mkdir(parents=True)
    (cgroup/'cgroup.procs').write_text('100\n101\n')
    mapped('/sys/fs/cgroup/cgroup.controllers').write_text('cpu memory')
    monkeypatch.setattr(m,'Path',mapped);monkeypatch.setattr(m.sys,'platform','linux')
    real_open=builtins.open;real_stat=os.stat
    reads=0
    def opened(path,*args,**kwargs):
        nonlocal reads
        path=mapped(path)
        if fault=='pid-reuse' and path==mapped('/proc')/'100/stat':
            reads+=1
            if reads==2:path.write_text('100 (fake worker) '+' '.join(['S']+['0']*18+['2000']))
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',opened)
    monkeypatch.setattr(m.os,'stat',lambda path,*a,**kw:real_stat(mapped(path),*a,**kw))
    if fault=='foreign-cgroup':(mapped('/proc')/'100/cgroup').write_text('0::/guessed/parent')
    elif fault=='shared-namespace':
        p=mapped('/proc')/'self/ns/pid';p.unlink();os.link(mapped('/proc')/'100/ns/pid',p)
    elif fault=='child-namespace':
        p=mapped('/proc')/'101/ns/pid';p.unlink();p.write_text('different-namespace')
    elif fault=='child-escape':(mapped('/proc')/'101/cgroup').write_text('0::/another-container')
    elif fault=='ipc-alias':
        (mapped('/proc')/'100/root/ipc').unlink();(mapped('/proc')/'100/root/ipc').mkdir()
    elif fault=='cgroup-symlink':(cgroup/'unknown').symlink_to(ipc,target_is_directory=True)
    elif fault=='oversize':(mapped('/proc')/'100/cgroup').write_text('x'*65537)
    elif fault=='deadline':
        clock=[0]
        def monotonic():clock[0]+=3;return clock[0]
        monkeypatch.setattr(m.time,'monotonic',monotonic)
    elif fault=='no-cgroup-v2':mapped('/sys/fs/cgroup/cgroup.controllers').unlink()
    if fault=='none':
        result=m._proc_observation(100,identity,ipc)
        assert result==dict(pid=100,start_ticks=1000,pid_namespace=real_stat(mapped('/proc')/'100/ns/pid').st_ino,cgroup='/docker/'+identity)
    else:
        with pytest.raises((ValueError,OSError)):m._proc_observation(100,identity,ipc)


def _b23_add_retained_registration(h,*,consumers=True,same_worker=False):
    old=copy.deepcopy(h['registration']);old['boot']='f'*32
    provenance=copy.deepcopy(h['provenance'])
    if not same_worker:
        for i,(c,p) in enumerate(zip(old['containers'],provenance['owners'])):
            c.update(id=('e' if i==0 else 'f')*64,started='2026-10-17T00:00:00Z')
            p.update(id=c['id'],started=c['started'],pid=200+i,pid_namespace=20000+i,cgroup='/docker/'+c['id'])
            if consumers:
                value=copy.deepcopy(h['consumers'][i]);value.update(id=c['id'],started=c['started'],
                    running=False,pid=0,finished='2026-10-17T00:30:00Z')
                h['consumers'].append(value)
    provenance['registration_digest']=digest(old);provenance['hello']['boot']=old['boot']
    h['module']._host_write(h['base']/('registration-'+digest(old)+'.json'),provenance)
    recovery.register(h['roots'],h['scope'],old)
    h['state']=h['module'].host_state(h['roots'],h['scope'])
    h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    return old


def test_b23_complete_retained_union_including_old_stopped_owners(b23_host):
    h=b23_host;_b23_add_retained_registration(h)
    old_finished={c['id']:c['finished'] for c in h['consumers'] if not c['running']}
    capsule=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    assert len(capsule['registrations'])==2 and len(capsule['waits'])==4
    assert {c['id'] for c in capsule['kernel_proof']['observation']['containers']}=={c['id'] for c in h['consumers']}
    assert all(c['finished']==old_finished[c['id']] for c in capsule['topology']['consumers'] if c['id'] in old_finished)
    stops=[c[-1] for c in h['calls'] if c[1]=='stop']
    assert stops==['c'*64,'e'*64,'d'*64,'f'*64]


@pytest.mark.parametrize('fault',['missing-old','same-worker-new-boot','duplicate-registration'])
def test_b23_retained_owner_omission_and_boot_ambiguity(b23_host,fault):
    h=b23_host
    if fault=='duplicate-registration':
        h['state']['registrations']*=2
        h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    else:_b23_add_retained_registration(h,consumers=False,same_worker=fault=='same-worker-new-boot')
    before=logical(h['roots'])
    with pytest.raises(ValueError):h['module'].freeze_host(h['observer'],h['state'],h['request'])
    assert not any(c[1] in ('stop','wait') for c in h['calls'])
    assert logical(h['roots'])==before and not (h['base']/'invocations').exists()


def test_b23_incarnation_times_preserve_nanosecond_order():
    assert recovery.host_timestamp('2026-10-17T00:00:00.000000001Z')<recovery.host_timestamp('2026-10-17T00:00:00.000000002Z')
    with pytest.raises(ValueError):recovery.host_timestamp('0001-01-01T00:00:00Z')


def test_b23_expired_trusted_transport_cannot_override_os_clock(b23_host,monkeypatch):
    h=b23_host;c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    directory,_=_b23_fake_mount_transport(monkeypatch,h,c)
    before=(directory/'capsule.json').read_bytes()
    monkeypatch.setattr(recovery.time,'time',lambda:c['expires']+1)
    with pytest.raises(ValueError,match='host_capsule_expired'):recovery.read_host_capsule()
    # A supplied old clock must not make an expired, correctly mounted capsule
    # acceptable. Tests fake the OS clock, never a production authorization flag.
    with pytest.raises((ValueError,TypeError)):recovery.read_host_capsule(now=c['issued'])
    assert (directory/'capsule.json').read_bytes()==before


@pytest.mark.parametrize('inventory',[1,2])
@pytest.mark.parametrize('live',[True,False])
@pytest.mark.parametrize('source,device,inode',[
    ('/alias/staging/_data/retained-subdir',7,999),
    ('/unproved/foreign-storage',7,998),
    ('/unproved/other-filesystem',88,997),
    ('/direct-root-alias',7,4),
])
def test_b23a_r1_unknown_storage_refuses_before_stop(b23_host,monkeypatch,live,source,device,inode,inventory):
    # E1 exact R1 ancestor-bind-alias descendant plus deliberately unsupported
    # foreign storage. Different stat identities alone never prove disjointness.
    from types import SimpleNamespace
    h=b23_host;m=h['module'];before=logical(h['roots'])
    u=copy.deepcopy(h['consumers'][1])
    u.update(id='e'*64,running=live,pid=199 if live else 0,
        finished='0001-01-01T00:00:00Z' if live else '2026-10-17T02:00:00Z')
    u['mounts']=[dict(type='bind',name='',source=source,destination='/data',rw=True)]
    read=h['read'];enumerations=0
    def observed(argv):
        nonlocal enumerations
        if argv[:2]==['container','ls']:
            enumerations+=1
            if enumerations==2*inventory-1:h['consumers'].append(u)
        return read(argv)
    h['observer']._read=observed
    real_stat=m.os.stat
    def fake_stat(path,*a,**kw):
        if str(path)==source:return SimpleNamespace(st_dev=device,st_ino=inode)
        return real_stat(path,*a,**kw)
    monkeypatch.setattr(m.os,'stat',fake_stat)
    with pytest.raises(ValueError):m.freeze_host(h['observer'],h['state'],h['request'])
    assert not any(a[1] in ('stop','wait') for a in h['calls'])
    assert not (h['base']/'invocations').exists()
    assert u['running']==live and len(h['consumers'])==3
    assert logical(h['roots'])==before


@pytest.mark.parametrize('role,managed',[
    ('control','core'),('receipts','journal.json'),('staging','foreign'),('ipc','lifecycle.sock')])
@pytest.mark.parametrize('destination',['directory','managed'])
@pytest.mark.parametrize('observation',[1,2,3])
def test_b23a_r1_reader_nested_role_mount_refuses(b23_host,monkeypatch,role,managed,destination,observation):
    h=b23_host;c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    directory,mounts=_b23_fake_mount_transport(monkeypatch,h,c)
    before=(directory/'capsule.json').read_bytes();sql=logical(h['roots'])
    assert recovery.read_host_capsule()==c
    extra=dict(id='99',device='99:99',root='/foreign',
        target='/'+role+'/'+('nested-directory' if destination=='directory' else managed),options=['rw'])
    calls=0
    def observed():
        nonlocal calls
        calls+=1
        return copy.deepcopy(mounts+([extra] if calls>=observation else []))
    monkeypatch.setattr(recovery,'_host_mount_topology',observed)
    with pytest.raises(ValueError,match='host_role_nested_mount'):recovery.read_host_capsule()
    assert calls==observation
    assert (directory/'capsule.json').read_bytes()==before and logical(h['roots'])==sql


def test_lotk_exact_supplied_snapshot_is_not_recaptured(installation, monkeypatch):
    """A frozen actual snapshot must survive acceptance without a new backup."""
    intent, observation, raw = prefix_fixture(installation)
    def forbidden_recapture(*args):
        raise AssertionError('frozen snapshot was recaptured')
    # Read-only validation may still use a private database. Only making a new
    # original snapshot would replace the supplied immutable physical evidence.
    monkeypatch.setattr(recovery, '_capture_original_snapshot', forbidden_recapture)
    roots, scope, _, _ = installation
    result = recovery._pending_through_evidence(roots, scope, intent, observation,
        now=102, snapshot_bytes=raw)
    assert result['status'] == 'pending-through-evidence'
    assert (roots['receipts']/('p1-recovery-'+intent['id']+'.snapshot.bundle')).read_bytes() == raw


def _lotk_model(monkeypatch, roots, scope, intent):
    """Explicit E1 OS/host model; NEVER calls setrlimit in the pytest process."""
    import resource, sys
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(recovery, '_maintenance_filesystems', lambda roots: None)
    monkeypatch.setattr(recovery, '_maintenance_task_count', lambda: 1)
    monkeypatch.setattr(resource, 'setrlimit', lambda kind, limits: None)
    monkeypatch.setattr(resource, 'getrlimit', lambda kind: (4_194_304,)*2)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        regs=[json.loads(r[0]) for r in db.execute('SELECT record FROM p1_registrations ORDER BY rowid')]
    capsule=dict(scope=scope, registrations=regs, registration_digest=recovery.host_registration_digest(regs),
        kernel_proof=intent['proof'],kernel_proof_digest=intent['proof_digest'],
        purpose=dict(kind='pre-plan',plan_id=None,plan_digest=None))
    monkeypatch.setattr(recovery, 'read_host_capsule', lambda: capsule)
    monkeypatch.setattr(recovery.time, 'time', lambda: 102)


def test_lotk_same_lock_owned_snapshot_completes_without_recapture(installation, monkeypatch):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        import hashlib
        from service.resource_store import _maintenance_footprint
        bound=_maintenance_footprint((roots['control']/'core',roots['staging']),intent['highwater'],
            sum(json.loads(r[2])['disk_bytes'] for r in intent['before']['source']['data']['resource_reservations']),intent['budget_before'])
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        assert owned.sha256==intent['expected'][1]['output']['snapshot_digest']
        raw=owned._raw
        stage_before={str(p.relative_to(roots['staging'])):(p.stat().st_mode,p.stat().st_size,p.stat().st_mtime_ns)
            for p in roots['staging'].rglob('*')}
        descendant_before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in (roots['control']/'core').rglob('*') if p.is_dir()}
        writes=[];writer=recovery._prefix_file
        def trace(path,*args):
            writes.append(path)
            assert path.parent in (roots['control']/'core',roots['receipts'])
            return writer(path,*args)
        recovery._prefix_file=trace
        def forbidden(*args):raise AssertionError('owned frozen snapshot recaptured')
        recovery._capture_original_snapshot=forbidden
        result=session.continue_kernel(intent,prepared=owned)
        assert result['status']=='complete'
        assert writes
        assert descendant_before=={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in descendant_before}
        assert stage_before=={str(p.relative_to(roots['staging'])):(p.stat().st_mode,p.stat().st_size,p.stat().st_mtime_ns)
            for p in roots['staging'].rglob('*')}
        assert {p.name for p in writes if p.parent==roots['control']/'core'}==set(recovery.archive_names(intent))
        assert (roots['receipts']/('p1-recovery-'+intent['id']+'.snapshot.bundle')).read_bytes()==raw
        assert (roots['control']/'core'/('p1-recovery-'+intent['archive_id']+'.sqlite')).read_bytes()==raw
        assert hashlib.sha256(raw).hexdigest()==owned.sha256
        with pytest.raises(ValueError,match='session_closed'):session.continue_kernel(intent,prepared=owned)
        return dict(status=result['status'],upper=bound['upper'])
    result=recovery._run_bounded_maintenance(roots,scope,operation)
    assert result['status']=='complete' and result['upper']<=intent['highwater']


@pytest.mark.parametrize('failure',['platform','readback','setlimit','host','control-lock','lane-lock','oversize','sidecar'])
def test_lotk_refuses_before_original_writable_open(installation, monkeypatch, failure):
    import resource, sys
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    path=roots['control']/'core'/'core.sqlite'
    if failure=='platform':monkeypatch.setattr(sys,'platform','darwin')
    if failure=='readback':monkeypatch.setattr(resource,'getrlimit',lambda kind:(4_194_305,)*2)
    if failure=='setlimit':
        def fail(*args):raise OSError('limit refused')
        monkeypatch.setattr(resource,'setrlimit',fail)
    if failure=='host':
        def fail():raise ValueError('host_trusted_mount_absent')
        monkeypatch.setattr(recovery,'read_host_capsule',fail)
    if failure=='oversize':
        with (path.parent/'core.sqlite-wal').open('wb') as stream:stream.truncate(4_194_305)
    if failure=='sidecar':(path.parent/'core.sqlite-journal').write_bytes(b'preserve')
    before={p:p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    original=recovery.sqlite3.connect
    opened=path.parent.parent.parent/'unexpected-original-open'
    def guarded(database,*args,**kwargs):
        if str(database).startswith(path.as_uri()) and 'mode=rw' in str(database):
            opened.write_text('attempted')
            raise AssertionError('original writable open before verified envelope')
        return original(database,*args,**kwargs)
    monkeypatch.setattr(recovery.sqlite3,'connect',guarded)
    fd=None
    if failure.endswith('-lock'):
        target=path.parent/'control.lock' if failure=='control-lock' else roots['receipts']/'lane.lock'
        fd=os.open(target,os.O_RDONLY);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError,match='maintenance'):
            recovery._run_bounded_maintenance(roots,scope,lambda session:True)
    finally:
        if fd is not None:os.close(fd)
    assert not opened.exists()
    assert before=={p:p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    assert not list(roots['receipts'].glob('p1-recovery-*'))


@pytest.mark.parametrize('tamper',['byte','plan','source','caller-bytes','caller-object'])
def test_lotk_owned_handoff_tamper_refuses_before_acceptance(installation, monkeypatch, tamper):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        if tamper=='byte':owned._raw=owned._raw[:-1]+bytes([owned._raw[-1]^1])
        if tamper=='plan':session.db.execute("UPDATE p1_abandon_plans SET digest=? WHERE id=?",('0'*64,intent['plan_id']))
        if tamper=='source':session.db.execute("UPDATE resource_monitor SET state='fault'")
        supplied=owned._raw if tamper=='caller-bytes' else object() if tamper=='caller-object' else owned
        with pytest.raises(ValueError):session.continue_kernel(intent,prepared=supplied)
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation) is True


def test_lotk_caller_capability_cannot_select_process_or_limit(installation):
    roots,scope,_,_=installation
    with pytest.raises(TypeError):recovery._run_bounded_maintenance(roots,scope,lambda s:True,limit=2**40)
    with pytest.raises(TypeError):recovery._run_bounded_maintenance(roots,scope,lambda s:True,capability=True)
    with pytest.raises(ValueError,match='maintenance_process'):
        with recovery._maintenance_session(roots,scope,object()):pass


# Test-local physical-cap witness protocol; fabricated controls below are not SQLite positives.
def _lotk_cap_validate(record, mode, returncode):
    assert returncode == 0, ('helper exit', returncode)
    assert record['cleanup'] == 'closed-and-observed'
    assert record['errors'] == []
    assert record['failure']['operation'] in (('INSERT', 'COMMIT') if mode == 'append' else ('checkpoint',))
    assert record['failure']['code'] == 778
    assert record['failure']['name'] == 'SQLITE_IOERR_WRITE'
    assert record['committed'] > 0
    assert record['checkpoint'] == ([0, 0, 0] if mode == 'append' else None)
    assert record['rollback_count'] == record['close_count'] == 1
    assert record['observations']
    for observation in record['observations']:
        assert observation['limits'] == [4_194_304, 4_194_304]
        assert observation['token'] is True
        files = observation['files']
        assert 'database.sqlite' in files
        assert set(files) <= {'database.sqlite', 'database.sqlite-wal', 'database.sqlite-shm'}
        assert all(type(size) is int and 0 <= size <= 4_194_304 for size in files.values())
    assert record['lengths'] == [65_536] * record['committed']
    if mode == 'checkpoint':
        assert record['committed'] == 68
        assert record['seed_checkpoint'] == [0, 0, 0]
        assert record['seed_logical'] < 4_194_304 < record['logical']
        assert record['failure_db_size'] == 4_194_304


_LOTK_CAP_SCRIPT = r'''
import errno, json, os, resource, sqlite3, sys
from pathlib import Path
from service import recovery
recovery._maintenance_child_pid=os.getpid()
p=recovery._install_maintenance_limit(); L=4_194_304
root=Path(sys.argv[1]); mode=sys.argv[2]
record=dict(cleanup=None, errors=[], failure=None, committed=0, checkpoint=None,
    rollback_count=0, close_count=0, observations=[], lengths=[])
db=None; primary=None; cleanup_errors=[]
def observe(operation):
    p.verify()
    limits=resource.getrlimit(resource.RLIMIT_FSIZE)
    assert limits==(L,L)
    files={f.name:f.stat().st_size for f in root.iterdir() if f.name.startswith('database.sqlite')}
    record['observations'].append(dict(operation=operation,limits=list(limits),token=True,files=files))
    assert 'database.sqlite' in files
    assert set(files)<={'database.sqlite','database.sqlite-wal','database.sqlite-shm'}
    assert all(size<=L for size in files.values())
def sql(statement, parameters=()):
    try:
        return db.execute(statement, parameters).fetchall()
    finally:
        observe(statement)
def commit():
    try: db.commit()
    finally: observe('COMMIT')
def rollback():
    record['rollback_count']+=1
    try:
        db.rollback()
        assert not db.in_transaction
    finally: observe('ROLLBACK')
def failure(exc, operation):
    record['failure']=dict(operation=operation,code=exc.sqlite_errorcode,name=exc.sqlite_errorname)
    assert exc.sqlite_errorcode==778 and exc.sqlite_errorname=='SQLITE_IOERR_WRITE'
def insert():
    sql('INSERT INTO bounded VALUES(?)',(b'x'*65536,))
    commit()
    record['committed']+=1
try:
    if mode=='append':
        for operation in ('write','pwrite','truncate','ftruncate'):
            path=root/operation
            fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
            os.ftruncate(fd,L)
            os.lseek(fd,L-1,0);assert os.write(fd,b'x')==1
            assert os.pwrite(fd,b'x',L-1)==1
            try:
                if operation=='write':os.lseek(fd,L,0);os.write(fd,b'x')
                elif operation=='pwrite':os.pwrite(fd,b'x',L)
                elif operation=='truncate':os.truncate(path,L+1)
                else:os.ftruncate(fd,L+1)
            except OSError as exc:assert exc.errno==errno.EFBIG
            else:raise AssertionError('extension succeeded')
            assert path.stat().st_size==L
            os.close(fd); p.verify()
    db=sqlite3.connect((root/'database.sqlite').as_uri()+'?vfs=unix',uri=True)
    observe('OPEN')
    if mode=='checkpoint':
        sql('PRAGMA page_size=4096')
        assert sql('PRAGMA page_size')==[(4096,)]
    assert sql('PRAGMA journal_mode=WAL')==[('wal',)]
    sql('PRAGMA synchronous=FULL'); assert sql('PRAGMA synchronous')==[(2,)]
    sql('PRAGMA mmap_size=0'); assert sql('PRAGMA mmap_size')==[(0,)]
    sql('PRAGMA wal_autocheckpoint=0'); assert sql('PRAGMA wal_autocheckpoint')==[(0,)]
    record['page_size']=sql('PRAGMA page_size')[0][0]
    sql('CREATE TABLE bounded(data BLOB)');commit()
    if mode=='append':
        for i in range(100):
            operation='INSERT'
            try:
                sql('INSERT INTO bounded VALUES(?)',(b'x'*65536,))
                operation='COMMIT';commit()
            except sqlite3.Error as exc:
                failure(exc,operation)
                break
            record['committed']+=1
        assert record['failure'] is not None, 'missing append failure within 100 pairs'
        assert record['committed']>0
        rollback()
        record['checkpoint']=list(sql('PRAGMA wal_checkpoint(TRUNCATE)')[0])
        assert record['checkpoint']==[0,0,0]
        record['logical']=sql('PRAGMA page_count')[0][0]*record['page_size']
        record['lengths']=[r[0] for r in sql('SELECT length(data) FROM bounded')]
        assert record['lengths']==[65536]*record['committed']
    else:
        assert mode=='checkpoint'
        for i in range(48):insert()
        record['seed_logical']=sql('PRAGMA page_count')[0][0]*record['page_size']
        assert record['seed_logical']<L
        record['seed_checkpoint']=list(sql('PRAGMA wal_checkpoint(TRUNCATE)')[0])
        assert record['seed_checkpoint']==[0,0,0]
        for i in range(20):insert()
        record['lengths']=[r[0] for r in sql('SELECT length(data) FROM bounded')]
        assert record['lengths']==[65536]*68
        record['logical']=sql('PRAGMA page_count')[0][0]*record['page_size']
        assert record['logical']>L and not db.in_transaction
        try:
            row=sql('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.Error as exc:failure(exc,'checkpoint')
        else:raise AssertionError(('checkpoint did not raise',row))
        record['failure_db_size']=(root/'database.sqlite').stat().st_size
        assert record['failure_db_size']==L
        rollback()
except BaseException as exc:
    primary=exc
finally:
    if db is not None:
        # Exactly one rollback even if a prerequisite or primary assertion failed.
        if not record['rollback_count']:
            try:rollback()
            except BaseException as exc:cleanup_errors.append(exc)
        record['close_count']+=1
        try:db.close()
        except BaseException as exc:cleanup_errors.append(exc)
        try:observe('CLOSE/final')
        except BaseException as exc:cleanup_errors.append(exc)
        if not cleanup_errors:record['cleanup']='closed-and-observed'
    record['errors']=[repr(exc) for exc in ([primary] if primary is not None else [])+cleanup_errors]
    print(json.dumps(record),flush=True)
if primary is not None or cleanup_errors:
    raise BaseExceptionGroup('cap primary and cleanup failures',([primary] if primary is not None else [])+cleanup_errors)
'''


def _lotk_cap_run(tmp_path, mode):
    import resource, sys
    before = resource.getrlimit(resource.RLIMIT_FSIZE)
    failures = []
    try:
        result = _guarded_fresh_run([sys.executable, '-c', _LOTK_CAP_SCRIPT, str(tmp_path), mode],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        record = json.loads(result.stdout)
        _lotk_cap_validate(record, mode, result.returncode)
        print(json.dumps(dict(mode=mode, witness=record)))
    except BaseException as exc:
        failures.append(exc)
    finally:
        try:assert resource.getrlimit(resource.RLIMIT_FSIZE) == before
        except BaseException as exc:failures.append(exc)
    if failures:raise BaseExceptionGroup('cap helper and parent limits', failures)


@pytest.mark.skipif(__import__('sys').platform!='linux',reason='Actual Linux RLIMIT_FSIZE evidence PENDING; macOS E1 model is not proof')
def test_lotk_actual_linux_child_file_and_sqlite_cap(tmp_path):
    _lotk_cap_run(tmp_path, 'append')


@pytest.mark.skipif(__import__('sys').platform!='linux',reason='Actual Linux RLIMIT_FSIZE evidence PENDING; macOS E1 model is not proof')
def test_lotk_actual_linux_sqlite_checkpoint_cap(tmp_path):
    _lotk_cap_run(tmp_path, 'checkpoint')


@pytest.mark.parametrize('mode', ['append', 'checkpoint'])
@pytest.mark.parametrize('defect', ['missing-exception', 'wrong-operation', 'wrong-code', 'wrong-name',
    'busy-row', 'zero-prefix', 'oversize', 'missing-db', 'unexpected-sidecar', 'missing-cleanup',
    'changed-limits', 'nonzero-helper', 'signal-helper', 'cleanup-error'])
def test_lotk_cap_witness_rejection_controls(mode, defect):
    # Pure protocol records only: actual positive witnesses are the two guarded tests above.
    record = dict(cleanup='closed-and-observed', errors=[],
        failure=dict(operation='INSERT' if mode == 'append' else 'checkpoint', code=778, name='SQLITE_IOERR_WRITE'),
        committed=1 if mode == 'append' else 68, checkpoint=[0, 0, 0] if mode == 'append' else None,
        rollback_count=1, close_count=1, observations=[dict(limits=[4_194_304]*2, token=True,
            files={'database.sqlite':4_194_304})], lengths=[65_536]*(1 if mode == 'append' else 68),
        seed_checkpoint=[0, 0, 0], seed_logical=3_000_000, logical=4_500_000, failure_db_size=4_194_304)
    _lotk_cap_validate(record, mode, 0)
    status = 0
    if defect == 'missing-exception':record['failure'] = None
    elif defect == 'wrong-operation':record['failure']['operation'] = 'CLOSE'
    elif defect == 'wrong-code':record['failure']['code'] = 13
    elif defect == 'wrong-name':record['failure']['name'] = 'SQLITE_FULL'
    elif defect == 'busy-row':record['checkpoint'] = [1, 2, 1]
    elif defect == 'zero-prefix':record['committed'] = 0
    elif defect == 'oversize':record['observations'][0]['files']['database.sqlite'] += 1
    elif defect == 'missing-db':record['observations'][0]['files'] = {}
    elif defect == 'unexpected-sidecar':record['observations'][0]['files']['database.sqlite-journal'] = 0
    elif defect == 'missing-cleanup':record['cleanup'] = None
    elif defect == 'changed-limits':record['observations'][0]['limits'][1] += 1
    elif defect == 'nonzero-helper':status = 1
    elif defect == 'signal-helper':status = -11
    elif defect == 'cleanup-error':record['errors'] = ['close failed']
    with pytest.raises((AssertionError, TypeError)):_lotk_cap_validate(record, mode, status)


@pytest.mark.parametrize('helper_failure', ['exit', 'exception'])
def test_lotk_cap_parent_limits_checked_on_helper_failure(tmp_path, monkeypatch, helper_failure):
    import resource
    from types import SimpleNamespace
    calls = []
    def limits(kind):
        calls.append(kind)
        return (100, 100) if len(calls) == 1 else (99, 100)
    def helper(*args, **kwargs):
        if helper_failure == 'exception':raise RuntimeError('fabricated helper failure')
        return SimpleNamespace(returncode=1, stdout='', stderr='fabricated helper failure')
    monkeypatch.setattr(resource, 'getrlimit', limits)
    monkeypatch.setattr(__import__(__name__, fromlist=['_guarded_fresh_run']), '_guarded_fresh_run', helper)
    with pytest.raises(BaseExceptionGroup) as caught:_lotk_cap_run(tmp_path, 'append')
    assert len(calls) == 2
    assert len(caught.value.exceptions) == 2
    assert isinstance(caught.value.exceptions[-1], AssertionError)


@pytest.mark.parametrize('point',['bundle-parent','core-intent-commit'])
def test_lotk_actual_death_fresh_owned_resume(installation,monkeypatch,point):
    import subprocess,sys
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    identity=intent['id']
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        def die(where):
            if where==point:os._exit(73)
        session.continue_kernel(intent,prepared=owned,fault=die)
        raise AssertionError('death boundary not reached')
    with pytest.raises(ValueError,match='child_failed'):recovery._run_bounded_maintenance(roots,scope,operation)
    del intent
    script=r'''
import json,os,resource,sys
from pathlib import Path
from service import recovery
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()};scope=v['scope'];identity=v['id']
# Explicit macOS E1 OS/host model; no installed Linux enforcement claim.
sys.platform='linux';resource.setrlimit=lambda *a:None;resource.getrlimit=lambda *a:(4_194_304,)*2
recovery._maintenance_filesystems=lambda roots:None;recovery._maintenance_task_count=lambda:1
recovery.time.time=lambda:1000
accepted=json.loads(recovery.read_file(roots['receipts']/('p1-recovery-'+identity+'.intent.json')))
regs=[json.loads(r[1]) for r in accepted['before']['source']['data']['p1_registrations']]
capsule=dict(scope=scope,registrations=regs,registration_digest=recovery.host_registration_digest(regs),
    kernel_proof=accepted['proof'],kernel_proof_digest=accepted['proof_digest'],
    purpose=dict(kind='resume',plan_id=accepted['plan_id'],plan_digest=accepted['plan_digest']))
recovery.read_host_capsule=lambda:capsule
for _ in range(2):
    def resume(session):
        owned=json.loads(recovery.read_file(roots['receipts']/('p1-recovery-'+identity+'.intent.json')))
        assert session.continue_kernel(owned)['status']=='complete'
        return True
    assert recovery._run_bounded_maintenance(roots,scope,resume)
'''
    result=_guarded_fresh_run([sys.executable,'-c',script],input=json.dumps(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,id=identity)),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)


@pytest.mark.parametrize('point',['bundle-intent-recorded','core-complete-write','core-complete-commit'])
def test_lotk_io_failure_burns_session_no_partial_healthy_authority(installation,monkeypatch,point):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest']);fired=[]
        def fail(where):
            if where==point:fired.append(where);raise OSError('EFBIG-modeled')
        with pytest.raises(OSError):session.continue_kernel(intent,prepared=owned,fault=fail)
        assert fired==[point]
        with pytest.raises(ValueError,match='session_closed'):session.continue_kernel(intent,prepared=owned)
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        healthy=db.execute("SELECT state FROM resource_monitor").fetchone()==('healthy',)
        integrity=db.execute("SELECT state FROM p1_integrity").fetchone()==('healthy',)
        completed=db.execute("SELECT 1 FROM p1_recovery_steps WHERE step='core-complete'").fetchone() is not None
        assert healthy==integrity==completed
        assert completed==(point=='core-complete-commit')
    if point=='bundle-intent-recorded':
        def resume(session):
            with pytest.raises(FileNotFoundError):session.continue_kernel(intent)
            return True
        assert recovery._run_bounded_maintenance(roots,scope,resume)


def test_lotk_private_connection_rejects_escape_and_preserves_wal(installation,monkeypatch):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        for sql in ("ATTACH ':memory:' AS foreign_db",'PRAGMA mmap_size=10000000',
                'PRAGMA journal_mode=DELETE','PRAGMA max_page_count=10',"SELECT load_extension('foreign')"):
            with pytest.raises(sqlite3.Error):session.db.execute(sql)
        assert session.db.execute('PRAGMA journal_mode').fetchone()==('wal',)
        assert session.db.execute('PRAGMA synchronous').fetchone()==(2,)
        assert session.db.execute('PRAGMA temp_store').fetchone()==(2,)
        assert session.db.execute('PRAGMA mmap_size').fetchone()==(0,)
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation)


def test_lotk_old_archive_drift_refuses_owned_handoff(installation,monkeypatch):
    roots,scope,_,_=installation
    first,observation,_=core_fixture(installation)
    recovery._complete_receipts(roots,scope,first,observation,now=102)
    intent=b2_intent(installation,prepare_only=True,recovery_id='7'*32,archive_id='6'*32)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        old=roots['control']/'core'/next(n for n,k in recovery.archive_names(first).items() if k=='snapshot')
        raw=old.read_bytes();old.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with pytest.raises(ValueError):session.continue_kernel(intent,prepared=owned)
        assert not (roots['receipts']/('p1-recovery-'+intent['id']+'.intent.json')).exists()
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation)


def test_lotk_fresh_unfinished_resume_reinstalls_fixed_cap(installation,monkeypatch):
    import resource
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        def fail(point):
            if point=='core-intent-commit':raise OSError('interrupted')
        with pytest.raises(OSError):session.continue_kernel(intent,prepared=owned,fault=fail)
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation)
    before={p:p.read_bytes() for root in roots.values() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(resource,'getrlimit',lambda kind:(2**40,)*2)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._run_bounded_maintenance(roots,scope,lambda session:session.continue_kernel(intent))
    assert before=={p:p.read_bytes() for root in roots.values() for p in root.rglob('*') if p.is_file()}


def test_lotk_fixed_cap_setup_ignores_environment_and_parent_limit(installation,monkeypatch):
    import resource
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    parent=resource.getrlimit(resource.RLIMIT_FSIZE)
    _lotk_model(monkeypatch,roots,scope,intent)
    calls=[]
    monkeypatch.setattr(resource,'setrlimit',lambda kind,limits:calls.append((kind,limits)))
    monkeypatch.setenv('P1_MAINTENANCE_LIMIT',str(2**40))
    def operation(session):
        assert calls==[(resource.RLIMIT_FSIZE,(4_194_304,4_194_304))]
        return True
    assert recovery._run_bounded_maintenance(roots,scope,operation)
    assert calls==[]  # Child setup did not call the parent's resource API.
    monkeypatch.undo()
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==parent


@pytest.mark.parametrize('filesystem,nested',[('ext4',False),('overlay',False),('nfs',False),('fuse',False),('unknown',False),('ext4',True)])
def test_lotk_declared_ext4_namespace_only(monkeypatch,filesystem,nested):
    import builtins,io
    roots={role:Path('/'+role) for role in ('control','staging','receipts','ipc')}
    raw=''.join(f'{i+1} 0 1:{i} / /{role} rw - {filesystem} source rw\n' for i,role in enumerate(roots))
    if nested:raw+='5 1 1:5 / /control/core rw - ext4 foreign rw\n'
    original=builtins.open
    monkeypatch.setattr(builtins,'open',lambda name,*a,**k:io.BytesIO(raw.encode()) if name=='/proc/self/mountinfo' else original(name,*a,**k))
    if filesystem=='ext4' and not nested:recovery._maintenance_filesystems(roots)
    else:
        with pytest.raises(ValueError,match='maintenance_(filesystem|submount)'):recovery._maintenance_filesystems(roots)


def test_lotk_original_connection_pins_standard_unix_vfs(installation,monkeypatch):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    original=recovery.sqlite3.connect;path=roots['control']/'core'/'core.sqlite'
    def pinned(database,*args,**kwargs):
        if str(database).startswith(path.as_uri()) and 'mode=rw' in str(database):
            assert str(database).endswith('&vfs=unix'),'bounded original relied on an unknown default VFS'
        return original(database,*args,**kwargs)
    monkeypatch.setattr(recovery.sqlite3,'connect',pinned)
    assert recovery._run_bounded_maintenance(roots,scope,lambda session:True)


def test_lotk_f1_all_private_connections_at_actual_use(installation,monkeypatch):
    roots,scope,_,_=installation
    intent,_,_=core_fixture(installation)
    _lotk_model(monkeypatch,roots,scope,intent)
    original=sqlite3.connect
    opened=[]; first_inserts=[]
    class Audited(sqlite3.Connection):
        checked=False
        reader_checked=False
        explicitly_closed=False
        def audit(self):
            for sql in ("ATTACH ':memory:' AS extra_db",'DETACH extra_db',
                    'PRAGMA mmap_size=1048576','PRAGMA journal_mode=DELETE',
                    'PRAGMA max_page_count=10',"SELECT load_extension('foreign')"):
                with pytest.raises(sqlite3.DatabaseError):
                    super().execute(sql)
            assert self.execute('PRAGMA trusted_schema').fetchone()==(0,)
            assert self.execute('PRAGMA temp_store').fetchone()==(2,)
            assert self.execute('PRAGMA mmap_size').fetchone()==(0,)
        def execute(self,sql,*args,**kwargs):
            if not self.checked and sql.startswith('INSERT INTO p1_recoveries '):
                self.checked=True
                self.audit()  # FIRST actual INSERT, after configuration opportunity.
                first_inserts.append(self)
            if not self.reader_checked and (sql.lower().startswith('pragma table_info') or sql.startswith('SELECT record FROM p1_registrations')):
                self.reader_checked=True
                self.audit()
            return super().execute(sql,*args,**kwargs)
        def backup(self,target,*args,**kwargs):
            self.reader_checked=True; self.audit()
            target.reader_checked=True; target.audit()
            return super().backup(target,*args,**kwargs)
        def close(self):
            self.explicitly_closed=True
            return super().close()
    def connect(database,*args,**kwargs):
        text=str(database)
        if 'p1-evidence-' in text or 'p1-inspect-' in text:
            kwargs['factory']=Audited
            db=original(database,*args,**kwargs)
            opened.append((text,kwargs.get('uri'),db))
            return db
        return original(database,*args,**kwargs)
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest'])
        result=session.continue_kernel(intent,prepared=owned)
        assert result['status']=='complete'
        assert len(first_inserts)==1
        assert opened
        for uri,is_uri,db in opened:
            assert is_uri and 'vfs=unix' in uri,uri
            assert db.checked or db.reader_checked,uri
            assert db.explicitly_closed,uri
            with pytest.raises(sqlite3.ProgrammingError):db.execute('SELECT 1')
        return True
    monkeypatch.setattr(sqlite3,'connect',connect)
    assert recovery._run_bounded_maintenance(roots,scope,operation)


# P production construction: no supplied intent/expected rows or b2_intent calls.
def _lotp_population(installation,kind='queued',*,highwater=20*1024**2):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots=installation[0];core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        job=core.submit_synthetic('p','production-P',f['context'],_admission=admission,
            _profiles=f['profiles'],_dispatch_settings=('silence',8))
        if kind!='queued':store.allocate(f['profiles'],'silence',8)
        if kind=='cancel':core.request_cancel(job['id'])
        # Declared retained test PRESTATE, before preparation/host freeze.
        with core.transaction():
            core.db.execute('UPDATE resource_accounting SET highwater=?',(highwater,))
            core.db.execute("UPDATE resource_monitor SET state='checking'")
            core.db.execute("UPDATE p1_integrity SET state='checking',token='inherited-A2'")
    finally:core.close()
    return job['id']


def _lotp_transport(h,monkeypatch):
    import resource
    h['state']=h['module'].host_state(h['roots'],h['scope'])
    c=h['module'].freeze_host(h['observer'],h['state'],h['request'])
    _b23_fake_mount_transport(monkeypatch,h,c)
    monkeypatch.setattr(recovery,'_maintenance_filesystems',lambda roots:None)
    monkeypatch.setattr(recovery,'_maintenance_task_count',lambda:1)
    monkeypatch.setattr(resource,'setrlimit',lambda kind,limits:None)
    monkeypatch.setattr(resource,'getrlimit',lambda kind:(4_194_304,)*2)
    # read_host_capsule remains the ACTUAL production reader, OS clock included.
    assert recovery.read_host_capsule()==c
    return c


@pytest.mark.parametrize('kind',['queued','running','cancel'])
@pytest.mark.parametrize('journal_state',['valid','missing','unreadable'])
def test_lotp_production_nonempty_complete(b23_host,installation,monkeypatch,kind,journal_state,tmp_path):
    import hashlib
    from service.resource_store import _maintenance_footprint,_measure_recovery_roots
    h=b23_host;roots,scope,profiles,_=installation
    job_id=_lotp_population(installation,kind)
    c=_lotp_transport(h,monkeypatch)
    journal=roots['receipts']/'journal.json'
    if journal_state=='missing':journal.unlink()
    elif journal_state=='unreadable':journal.write_bytes(b'{bounded damaged journal')
    old_journal=journal.read_bytes() if journal.exists() else b''
    before=logical(roots)
    early_bound=_maintenance_footprint((roots['control']/'core',roots['staging']),20*1024**2,2*1024**2,DEFAULT_BUDGET)
    metrics=tmp_path/'production-metrics.json'
    prepare=getattr(recovery,'_prepare_abandon_locked',None)
    def finish(session,**request):
        assert session.capsule['kernel_proof_digest']==digest(c['kernel_proof'])
        result=prepare(session,**request)
        assert set(result)=={'plan','plan_digest','intent','intent_digest'}
        p,r=result['plan'],result['intent'];owned=result._snapshot
        assert result._session is session
        assert result['plan_digest']==digest(p) and result['intent_digest']==digest(r)
        assert p['affected']['jobs']==[job_id] and r['actions']
        assert p['proof_digest']==c['kernel_proof_digest']
        assert r['before']['journal']==dict(state=journal_state,bytes_hex=old_journal.hex())
        recovery.validate_b2_record('p1_recoveries',r)
        assert recovery.verify_abandon_plan(roots,session.db,scope,c['kernel_proof']['observation'],
            p['id'],digest(p),proof_digest=c['kernel_proof_digest'])==p
        with recovery._archive_database(owned._raw) as archive:
            assert archive.execute('SELECT record,digest FROM p1_abandon_plans WHERE id=?',(p['id'],)).fetchone()==(encode(p),digest(p))
            assert archive.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
            assert archive.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(0,)
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        assert session.db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
        charges=sum(json.loads(row[2])['disk_bytes'] for row in r['before']['source']['data']['resource_reservations'])
        bound=_maintenance_footprint((roots['control']/'core',roots['staging']),r['highwater'],charges,r['budget_before'])
        def forbidden(*args):raise AssertionError('production frozen snapshot recaptured')
        recovery._capture_original_snapshot=forbidden
        from service import resource_store
        scan=resource_store._recovery_scan;observed_scans=[]
        def observed(db,scan_roots,intent):
            if scan_roots[0]==roots['control']/'core':observed_scans.append(_measure_recovery_roots(scan_roots))
            return scan(db,scan_roots,intent)
        resource_store._recovery_scan=observed
        completed=session.continue_kernel(r,prepared=owned)
        assert completed['status']=='complete' and len(observed_scans)==1
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
            for table,rows in r['expected'][3]['output']['rows'].items():
                assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(rows)
            for table in ('dispatches','dispatch_commands','events','assets','artifacts','publications','publication_work','p1_inventory_history'):
                assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(r['before']['source']['data'][table])
            assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
            assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
            assert recovery.completion_binding(db,roots['control']/'core')['epoch']==p['new_epoch']
        actual=_measure_recovery_roots((roots['control']/'core',roots['staging']))
        assert max(actual,r['highwater'])==r['highwater']
        assert charges==sum(json.loads(row[2])['disk_bytes'] for row in r['expected'][3]['output']['rows']['resource_reservations'])
        assert r['budget_before']==r['budget_after']==DEFAULT_BUDGET
        names=recovery.archive_names(r)
        sizes={k:(roots['control']/'core'/n).stat().st_size for n,k in names.items()}
        assert (roots['receipts']/('p1-recovery-'+r['id']+'.snapshot.bundle')).read_bytes()==owned._raw
        assert hashlib.sha256(owned._raw).hexdigest()==r['expected'][1]['output']['snapshot_digest']
        metrics.write_text(encode(dict(kind=kind,journal=journal_state,**bound,B=DEFAULT_BUDGET['disk_bytes'],
            S=sizes['snapshot'],J=sizes['manifest'],K=sizes['certificate'],observed_all=actual,observed_scan=observed_scans[0],M_early=early_bound['measured'],U_early=early_bound['upper'])))
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish,raising=False)
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert result['intent']['actions'] and logical(roots)!=before
    print(metrics.read_text())


@pytest.mark.parametrize('failure',['low-H','quota','free','configuration','origin','foreign','schema','index','view','journal-symlink','journal-large','live','omitted','purpose','sidecar','sidecar-large'])
def test_lotp_early_refusals_precede_original_writer(b23_host,installation,monkeypatch,failure,tmp_path):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,highwater=(1024 if failure=='low-H' else 64*1024**2 if failure=='quota' else 20*1024**2))
    c=_lotp_transport(h,monkeypatch)
    root=roots['control']/'core'
    if failure=='origin':
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:db.execute("UPDATE resource_reservations SET state='executing'")
    elif failure=='foreign':(roots['staging']/'foreign').write_bytes(b'unknown')
    elif failure=='schema':
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:db.execute('CREATE TABLE unknown_refs(id TEXT)')
    elif failure in ('index','view'):
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
            db.execute('CREATE INDEX foreign_index ON jobs(id)' if failure=='index' else 'CREATE VIEW foreign_view AS SELECT * FROM jobs')
    elif failure=='journal-symlink':
        (roots['receipts']/'journal.json').unlink();(roots['receipts']/'journal.json').symlink_to(roots['receipts']/'p1-init.json')
    elif failure=='journal-large':(roots['receipts']/'journal.json').write_bytes(b'x'*(1024**2+1))
    elif failure=='sidecar':(root/'core.sqlite-unknown').write_bytes(b'unknown')
    elif failure=='sidecar-large':
        with (root/'core.sqlite-wal').open('wb') as stream:stream.truncate(4_194_305)
    elif failure=='free':
        monkeypatch.setattr(recovery.os,'statvfs',lambda path:type('Free',(),dict(f_bavail=0,f_frsize=4096))())
    elif failure in ('live','omitted','purpose'):
        c=copy.deepcopy(c)
        if failure=='live':c['kernel_proof']['observation']['states'][0]['running']=True
        elif failure=='omitted':c['kernel_proof']['observation']['containers'].pop()
        else:c['purpose']=dict(kind='initial',plan_id='a'*32,plan_digest='a'*64)
        c['kernel_proof_digest']=digest(c['kernel_proof'])
        directory=h['base']/'invocations'/c['invocation_id'];(directory/'capsule.json').write_text(encode(c))
    before=logical(roots) if failure!='sidecar-large' else (root/'core.sqlite').read_bytes()
    receipt_before={p.name:p.read_bytes() for p in roots['receipts'].iterdir() if not p.is_symlink()}
    original=sqlite3.connect;trap=tmp_path/'unexpected-original-open'
    def connect(database,*args,**kw):
        if str(database)==(root/'core.sqlite').as_uri()+'?mode=rw&vfs=unix':
            trap.write_text('original writable open before early refusal');raise AssertionError('original writer')
        return original(database,*args,**kw)
    monkeypatch.setattr(sqlite3,'connect',connect)
    with pytest.raises(ValueError):
        recovery._prepare_abandon(roots,scope,profiles=[] if failure=='configuration' else profiles,
            budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert not trap.exists()
    assert before==(logical(roots) if failure!='sidecar-large' else (root/'core.sqlite').read_bytes())
    assert receipt_before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir() if not p.is_symlink()}


@pytest.mark.parametrize('failure',['rehearsal','snapshot','manifest','certificate','history'])
def test_lotp_late_refusal_retains_unaccepted_plan(b23_host,installation,monkeypatch,failure):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation);_lotp_transport(h,monkeypatch)
    if failure=='manifest':(roots['receipts']/'journal.json').write_bytes(b'x'*40000)
    completion_sentinel=roots['control']/'original-completion-called'
    original_completion_code=recovery._write_core_completion.__code__
    import sys
    previous_profile=sys.getprofile()
    def completion_profile(frame,event,arg):
        if event=='call' and frame.f_code is original_completion_code:
            completion_sentinel.write_text('original completion implementation called')
        if previous_profile is not None:previous_profile(frame,event,arg)
    if failure=='rehearsal':
        def fail(*args):raise sqlite3.OperationalError('modeled private capacity failure')
        monkeypatch.setattr(recovery,'_write_core_completion',fail)
    elif failure=='snapshot':monkeypatch.setattr(recovery,'MAINTENANCE_SNAPSHOT_LIMIT',1024)
    elif failure=='certificate':
        certificate=recovery._certificate
        monkeypatch.setattr(recovery,'_certificate',lambda *args:{**certificate(*args),'oversize':'x'*65536})
    elif failure=='history':
        validate=recovery.validate_b2_database
        def fail(db,*args,**kw):
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='p1_recoveries'").fetchone() and db.execute('SELECT 1 FROM p1_recoveries').fetchone():
                raise ValueError('modeled prospective history ceiling')
            return validate(db,*args,**kw)
        monkeypatch.setattr(recovery,'validate_b2_database',fail)
    if failure=='rehearsal':sys.setprofile(completion_profile)
    try:
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    finally:
        if failure=='rehearsal':sys.setprofile(previous_profile)
    assert not completion_sentinel.exists()
    assert not list(roots['receipts'].glob('p1-recovery-*'))
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(0,)
        assert db.execute('SELECT highwater FROM resource_accounting').fetchone()==(20*1024**2,)
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)


@pytest.mark.parametrize('tamper',['expected','source','later-plan','new-defect','journal','plan','snapshot'])
def test_lotp_frozen_output_and_source_oracles(b23_host,installation,monkeypatch,tamper):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def inspect(session,**request):
        result=prepare(session,**request);r=result['intent'];p=result['plan']
        if tamper=='expected':
            row=r['expected'][3]['output']['rows']['jobs'][0];job=json.loads(row[6]);job['attempts']=[];row[6]=encode(job)
            previous=r['before_digest']
            for descriptor in r['expected']:
                if descriptor['step']=='receipt-complete':descriptor['output']['core_output_digest']=previous
                descriptor['input_digest']=previous
                descriptor['output_digest']=digest({k:v for k,v in descriptor.items() if k!='output_digest'});previous=descriptor['output_digest']
        elif tamper=='source':session.db.execute("UPDATE resource_monitor SET state='fault'")
        elif tamper=='later-plan':
            other={**p,'id':'f'*32};session.db.execute('INSERT INTO p1_abandon_plans VALUES(?,?,?)',(other['id'],encode(other),digest(other)))
        elif tamper=='new-defect':(roots['staging']/'new-foreign').write_bytes(b'new')
        elif tamper=='journal':(roots['receipts']/'journal.json').write_bytes(b'new defect')
        elif tamper=='plan':session.db.execute('DELETE FROM p1_abandon_plans')
        else:result._snapshot._raw=b'changed'+result._snapshot._raw[7:]
        with pytest.raises((ValueError,FileNotFoundError)):
            session.continue_kernel(r,prepared=result._snapshot)
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        assert session.db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',inspect)
    assert recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent']


@pytest.mark.parametrize('point',['bundle-parent','core-intent-commit'])
def test_lotp_production_intent_actual_death_fresh_resume(b23_host,installation,monkeypatch,point):
    import subprocess,sys
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def die(session,**request):
        result=prepare(session,**request)
        def fault(where):
            if where==point:os._exit(73)
        session.continue_kernel(result['intent'],prepared=result._snapshot,fault=fault)
        raise AssertionError('death boundary not reached')
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',die)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    # No generated document or snapshot is passed to the NEW interpreter.
    script=r'''
import json,os,resource,sys
from pathlib import Path
from service import recovery
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()};scope=v['scope']
sys.platform='linux';resource.setrlimit=lambda *a:None;resource.getrlimit=lambda *a:(4_194_304,)*2
recovery._maintenance_filesystems=lambda roots:None;recovery._maintenance_task_count=lambda:1
path=next(roots['receipts'].glob('*.intent.json'))
paths=list(roots['receipts'].glob('p1-recovery-*.intent.json'))
paths=[p for p in paths if p.name.count('.')==2]
assert len(paths)==1
accepted=json.loads(recovery.read_file(paths[0]));identity=accepted['id']
regs=[json.loads(r[1]) for r in accepted['before']['source']['data']['p1_registrations']]
capsule=dict(scope=scope,registrations=regs,registration_digest=recovery.host_registration_digest(regs),
    kernel_proof=accepted['proof'],kernel_proof_digest=accepted['proof_digest'],
    purpose=dict(kind='resume',plan_id=accepted['plan_id'],plan_digest=accepted['plan_digest']))
# Explicit E1 fresh transport model; durable authority itself is real owned data.
recovery.read_host_capsule=lambda:capsule
bundle=roots['receipts']/('p1-recovery-'+identity+'.snapshot.bundle');raw=bundle.read_bytes()
for _ in range(2):
    def resume(session):
        owned=json.loads(recovery.read_file(paths[0]))
        assert session.continue_kernel(owned)['status']=='complete'
        return True
    assert recovery._run_bounded_maintenance(roots,scope,resume)
    assert bundle.read_bytes()==raw and recovery.read_file(paths[0])==recovery.encode(accepted).encode()
'''
    child=_guarded_fresh_run([sys.executable,'-c',script],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope)),
        capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)


@pytest.mark.parametrize('decision_case',['exact','missing','extra','content','reason','availability'])
def test_lotp_degraded_terminal_exact_decisions(b23_host,installation,monkeypatch,decision_case):
    roots,scope,profiles,_=installation;h=b23_host
    job_id=terminal_enrolled(installation);root=roots['control']/'core'
    asset=next(p for p in (root/'assets').iterdir() if p.is_file());asset_id=asset.name;asset.unlink()
    core=Core(root)
    try:
        assert Reconciliation(core,roots['staging']).reconcile_startup()['blocked']
        row=next(json.loads(raw) for (raw,) in core.db.execute('SELECT record FROM p1_inventory_objects')
            if json.loads(raw)['classification']=='missing_hash_mismatched_terminal')
    finally:core.close()
    # Seed retained H before preparation; no expected outputs or intent supplied.
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
        terminal=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
    _lotp_transport(h,monkeypatch)
    decision=dict(object_id=row['id'],reason=row['reason'],content_digest=recovery.diagnostic_digest(row),
        decision='accept-degraded',availability='unavailable')
    decisions=[decision]
    if decision_case=='missing':decisions=[]
    elif decision_case=='extra':decisions.append({**decision,'object_id':'e'*64})
    elif decision_case=='content':decision['content_digest']='f'*64
    elif decision_case=='reason':decision['reason']='changed'
    elif decision_case=='availability':decision['availability']='retained'
    before=logical(roots)
    if decision_case!='exact':
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=decisions)
        assert logical(roots)==before
        return
    prepare=recovery._prepare_abandon_locked
    def finish(session,**request):
        result=prepare(session,**request)
        assert session.continue_kernel(result['intent'],prepared=result._snapshot)['status']=='complete'
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish)
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=decisions)
    assert result['intent']['actions']==[dict(job_id=job_id,action='retain-terminal',before_digest=digest(json.loads(terminal[6])))]
    for _ in range(2):
        core=Core(root)
        try:
            assert core.db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()==terminal
            scan=Reconciliation(core,roots['staging']).reconcile_startup()
            assert scan['complete'] and not scan['blocked']
            with pytest.raises((ValueError,FileNotFoundError)):core.asset(asset_id)
        finally:core.close()


def test_lotp_single_immutable_plan_proof_first_and_preaccept_exit(b23_host,installation,monkeypatch,tmp_path):
    import subprocess,sys
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');capsule=_lotp_transport(h,monkeypatch)
    original=sqlite3.connect;trace=tmp_path/'plan-trace';unaccepted=tmp_path/'unaccepted.json'
    class Audit(sqlite3.Connection):
        def execute(self,sql,*args,**kw):
            if 'p1_abandon_plans' in sql and sql.startswith(('INSERT','UPDATE','DELETE')):
                assert sql.startswith('INSERT')
                value=json.loads(args[0][1]);assert value['proof_digest']==capsule['kernel_proof_digest']
                with trace.open('a') as stream:stream.write('insert\n')
            if sql.startswith('UPDATE resource_accounting'):raise AssertionError('preparation H update')
            return super().execute(sql,*args,**kw)
    def connect(database,*args,**kw):
        if str(database)==(roots['control']/'core'/'core.sqlite').as_uri()+'?mode=rw&vfs=unix':kw['factory']=Audit
        return original(database,*args,**kw)
    monkeypatch.setattr(sqlite3,'connect',connect)
    prepare=recovery._prepare_abandon_locked
    def die(session,**request):
        result=prepare(session,**request)
        unaccepted.write_text(encode(result))
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        os._exit(73)
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',die)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert trace.read_text()=='insert\n'
    assert not list(roots['receipts'].glob('p1-recovery-*'))
    # Fresh process has no snapshot authority. A physical recapture with changed
    # bytes cannot remint the frozen intent hash, even if the SQL source is equal.
    script=r'''
import json,os,resource,sys
from pathlib import Path
from service import recovery
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()};scope=v['scope']
r=json.loads(Path(v['document']).read_text())['intent']
sys.platform='linux';resource.setrlimit=lambda *a:None;resource.getrlimit=lambda *a:(4_194_304,)*2
recovery._maintenance_filesystems=lambda roots:None;recovery._maintenance_task_count=lambda:1
regs=[json.loads(row[1]) for row in r['before']['source']['data']['p1_registrations']]
recovery.read_host_capsule=lambda:dict(scope=scope,registrations=regs,registration_digest=recovery.host_registration_digest(regs),
    kernel_proof=r['proof'],kernel_proof_digest=r['proof_digest'],purpose=dict(kind='initial',plan_id=r['plan_id'],plan_digest=r['plan_digest']))
def missing(session):
    try:session.continue_kernel(r)
    except FileNotFoundError:return True
    raise AssertionError('unaccepted RAM document became restart authority')
assert recovery._run_bounded_maintenance(roots,scope,missing)
capture=recovery._capture_original_snapshot
def changed(*args):
    raw=bytearray(capture(*args));raw[24]^=1;return bytes(raw)
recovery._capture_original_snapshot=changed
def mismatch(session):
    owned=session.capture(r['plan_id'],r['plan_digest'])
    assert owned.sha256!=r['expected'][1]['output']['snapshot_digest']
    try:session.continue_kernel(r,prepared=owned)
    except ValueError:return True
    raise AssertionError('hash mismatch accepted')
assert recovery._run_bounded_maintenance(roots,scope,mismatch)
assert not list(roots['receipts'].glob('p1-recovery-*'))
'''
    child=_guarded_fresh_run([sys.executable,'-c',script],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,document=str(unaccepted))),
        capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr


@pytest.mark.parametrize('point',['accepted-parent','bundle-intent-recorded','bundle-parent','core-intent-write',
    'core-intent-commit','pending-receipt-commit','evidence-receipt-commit','journal-receipt-commit',
    'core-complete-write','core-complete-commit','completion-fence-dirsync','receipt-complete-commit'])
def test_lotp_production_intent_faults_no_partial_authority(b23_host,installation,monkeypatch,point):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    fired=roots['control'].parent/'fault-fired'
    def interrupt(session,**request):
        result=prepare(session,**request)
        def fault(where):
            if where==point:
                fired.write_text(where);raise OSError('modeled I/O interruption')
        with pytest.raises(OSError):session.continue_kernel(result['intent'],prepared=result._snapshot,fault=fault)
        with pytest.raises(ValueError,match='session_closed'):session.continue_kernel(result['intent'],prepared=result._snapshot)
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',interrupt)
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert fired.read_text()==point
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        completed=db.execute("SELECT 1 FROM p1_recovery_steps WHERE step='core-complete'").fetchone() is not None
        assert completed==(point in ('core-complete-commit','completion-fence-dirsync','receipt-complete-commit'))
        rows=result['intent']['expected'][3]['output']['rows'] if completed else result['intent']['before']['source']['data']
        for table in recovery.CORE_OUTPUT_TABLES:
            assert encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())==encode(rows[table])


@pytest.mark.parametrize('change',['none','delete','content','append'])
def test_lotp_prior_history_exact_exclusion(b23_host,installation,monkeypatch,change):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation);c=_lotp_transport(h,monkeypatch)
    prior=recovery.plan(roots,scope,c['kernel_proof']['observation'])['plan']
    prepare=recovery._prepare_abandon_locked
    def inspect(session,**request):
        result=prepare(session,**request);p=result['plan'];r=result['intent']
        assert len(r['before']['source']['data']['p1_recovery_plans'])==1
        assert not r['before']['source']['data']['p1_abandon_plans']
        if change=='delete':session.db.execute('DELETE FROM p1_recovery_plans')
        elif change=='content':
            old={**prior,'created':prior['created']+1,'expires':prior['expires']+1}
            session.db.execute('UPDATE p1_recovery_plans SET record=?,digest=?',(encode(old),digest(old)))
        elif change=='append':
            old={**prior,'id':'a'*32};session.db.execute('INSERT INTO p1_recovery_plans VALUES(?,?,?)',(old['id'],encode(old),digest(old)))
        def verify():return recovery.verify_abandon_plan(roots,session.db,scope,c['kernel_proof']['observation'],p['id'],digest(p),proof_digest=c['kernel_proof_digest'])
        if change=='none':assert verify()==p
        else:
            with pytest.raises(ValueError):verify()
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',inspect)
    assert recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent']


@pytest.mark.parametrize('argument',['actions','before','expected','snapshot_bytes','proof','now','highwater','limit','capability','budget_after'])
def test_lotp_closed_request_cannot_supply_authority(installation,argument):
    roots,scope,profiles,_=installation
    before=logical(roots)
    with pytest.raises(TypeError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[],**{argument:True})
    assert logical(roots)==before


@pytest.mark.parametrize('journal_state',['valid','missing','unreadable'])
@pytest.mark.parametrize('damage',['none','old-archive','old-descriptor','result-budget'])
def test_lotp_completed_history_preparation(b23_host,installation,monkeypatch,damage,journal_state):
    h=b23_host;roots,scope,profiles,_=installation
    # The result-budget branch isolates the adopted whole 1MiB UTF-8 envelope.
    if damage=='result-budget':_lotp_population(installation)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    _lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def finish(session,**request):
        result=prepare(session,**request)
        assert session.continue_kernel(result['intent'],prepared=result._snapshot)['status']=='complete'
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish)
    first=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    # A distinct owner incarnation for the new lane; all old stopped owners and
    # their protected registration provenance are retained in the host union.
    reg=copy.deepcopy(h['registration']);reg['lane']=first['intent']['new_lane'];reg['boot']='e'*32
    for index,owner in enumerate(reg['containers']):
        owner['id']=('e' if index==0 else 'f')*64;owner['started']=f'2026-10-17T03:00:0{index}Z'
    recovery.register(roots,scope,reg)
    provenance=copy.deepcopy(h['provenance']);provenance['registration_digest']=digest(reg)
    provenance['hello']={k:reg[k] for k in ('lane','boot')}
    for owner,c in zip(provenance['owners'],reg['containers']):owner.update(id=c['id'],started=c['started'],cgroup='/docker/'+c['id'])
    h['module']._host_write(h['base']/('registration-'+digest(reg)+'.json'),provenance)
    for old,c in zip(copy.deepcopy(h['consumers']),reg['containers']):
        old.update(id=c['id'],started=c['started'],finished='2026-10-17T04:00:00Z')
        h['consumers'].append(old)
    h['state']=h['module'].host_state(roots,scope)
    h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    _lotp_transport(h,monkeypatch)
    if journal_state=='missing':(roots['receipts']/'journal.json').unlink()
    elif journal_state=='unreadable':(roots['receipts']/'journal.json').write_bytes(b'bounded damaged later journal')
    if damage=='old-archive':next((roots['control']/'core').glob('*.manifest.json')).write_bytes(b'changed')
    elif damage=='old-descriptor':next(roots['receipts'].glob('*.core-certificate.intent.json')).unlink()
    before=logical(roots)
    if damage=='result-budget':
        chunks=recovery._maintenance_result_chunks
        def over_budget(result):
            # Real B documents are validated first; only encoder budget is
            # isolated. No caller padding becomes production history.
            assert sum(map(len,chunks(result)))<recovery.MAINTENANCE_RESULT_LIMIT
            yield from chunks('é'+ 'a'*(recovery.MAINTENANCE_RESULT_LIMIT-14))
        monkeypatch.setattr(recovery,'_maintenance_result_chunks',over_budget)
        writer_hit=roots['control'].parent/'result-budget-writer'
        protocol_before=retained_bytes(roots)
        def forbidden(*a,**k):
            writer_hit.write_text('acceptance');raise AssertionError('oversize result reached acceptance')
        for name in ('_rehearse_preparation','_prefix_file','_pending_prefix_locked'):
            monkeypatch.setattr(recovery,name,forbidden)
        burned=roots['control'].parent/'result-budget-burned'
        def budget_refusal(session,**request):
            try:return finish(session,**request)
            except ValueError as exc:
                assert str(exc)=='b2_maintenance_result_budget' and not session._live
                with pytest.raises(ValueError,match='session_closed'):
                    session.capture(first['plan']['id'],first['plan_digest'])
                burned.write_text(str(exc));raise
        monkeypatch.setattr(recovery,'_prepare_abandon_locked',budget_refusal)
    if damage!='none':
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        if damage=='result-budget':
            assert not writer_hit.exists() and burned.exists()
            assert [r for r in logical(roots) if 'INSERT INTO "p1_abandon_plans"' not in r]==[r for r in before if 'INSERT INTO "p1_abandon_plans"' not in r]
            for name,raw in protocol_before.items():
                if Path(name.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):assert retained_bytes(roots)[name]==raw
            with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
                assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(2,)
                assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
                assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
        else:assert logical(roots)==before
        return
    second=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert second['intent']['prior_epoch']==first['intent']['new_epoch']
    assert second['intent']['old_lane']==first['intent']['new_lane']
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(2,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(10,)


def test_lotp_rehearsal_connections_hardened_at_actual_use(b23_host,installation,monkeypatch):
    roots,scope,profiles,_=installation;h=b23_host
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    connect=sqlite3.connect;opened=[];inserts=[]
    class Audit(sqlite3.Connection):
        checked=False;reader_checked=False;explicitly_closed=False
        def audit(self):
            for sql in ("ATTACH ':memory:' AS extra_db",'DETACH extra_db','PRAGMA mmap_size=1048576',
                'PRAGMA journal_mode=DELETE','PRAGMA max_page_count=10','PRAGMA query_only=OFF',"SELECT load_extension('foreign')"):
                with pytest.raises(sqlite3.DatabaseError):super().execute(sql)
            assert super().execute('PRAGMA trusted_schema').fetchone()==(0,)
            assert super().execute('PRAGMA temp_store').fetchone()==(2,)
            assert super().execute('PRAGMA mmap_size').fetchone()==(0,)
        def execute(self,sql,*args,**kw):
            if not self.checked and sql.startswith('INSERT INTO p1_recoveries '):
                self.checked=True;self.audit();inserts.append(self)
            if not self.reader_checked and (sql.lower().startswith('pragma table_info') or sql.startswith('SELECT record FROM p1_registrations')):
                self.reader_checked=True;self.audit()
            return super().execute(sql,*args,**kw)
        def backup(self,target,*args,**kw):
            self.reader_checked=True;self.audit();target.reader_checked=True;target.audit()
            return super().backup(target,*args,**kw)
        def close(self):
            self.explicitly_closed=True;return super().close()
    def audited(database,*args,**kw):
        text=str(database)
        if any(tag in text for tag in ('p1-evidence-','p1-inspect-','p1-rehearsal-')):
            assert kw.get('uri') and 'vfs=unix' in text
            kw['factory']=Audit;db=connect(database,*args,**kw);opened.append(db);return db
        return connect(database,*args,**kw)
    monkeypatch.setattr(sqlite3,'connect',audited)
    prepare=recovery._prepare_abandon_locked
    def finish(session,**request):
        result=prepare(session,**request)
        assert session.continue_kernel(result['intent'],prepared=result._snapshot)['status']=='complete'
        assert len(inserts)==2  # concrete rehearsal + original kernel's prospective writer
        assert opened and all(db.checked or db.reader_checked for db in opened)
        for db in opened:
            assert db.explicitly_closed
            with pytest.raises(sqlite3.ProgrammingError):db.execute('SELECT 1')
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish)
    assert recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent']


def test_lotp_preparation_alone_no_acceptance_or_ordinary_healing(b23_host,installation,monkeypatch):
    from service.admission import Admission
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');c=_lotp_transport(h,monkeypatch)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        before={table:db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
            for table in (*recovery.CORE_OUTPUT_TABLES,'p1_integrity','dispatches','dispatch_commands','events')}
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert set(result)=={'plan','plan_digest','intent','intent_digest'}
    assert result['plan_digest']==digest(result['plan']) and result['intent_digest']==digest(result['intent'])
    assert not list(roots['receipts'].glob('p1-recovery-*'))
    assert not (roots['receipts']/'p1-b2-required.json').exists()
    with recovery.offline(roots) as (_,db):
        assert recovery.verify_abandon_plan(roots,db,scope,c['kernel_proof']['observation'],result['plan']['id'],result['plan_digest'],proof_digest=c['kernel_proof_digest'])==result['plan']
        for table,rows in before.items():assert db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()==rows
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(0,)
    for _ in range(2):
        core=Core(roots['control']/'core')
        try:
            with pytest.raises(ValueError):Admission(core,roots['staging']).check()
            assert Reconciliation(core,roots['staging']).reconcile_startup()['blocked']
            assert core.db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
        finally:core.close()
    with pytest.raises(NotImplementedError,match='P1B2_apply_unavailable'):recovery.apply(result)


@pytest.mark.parametrize('failure',['expired','changed'])
def test_lotp_final_capsule_revalidation_before_return(b23_host,installation,monkeypatch,failure):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation);c=_lotp_transport(h,monkeypatch)
    rehearse=recovery._rehearse_preparation
    def change(*args):
        rehearse(*args)
        altered=copy.deepcopy(c)
        if failure=='expired':altered['issued']=0;altered['expires']=1
        else:altered['collection_seconds']+=1
        (h['base']/'invocations'/c['invocation_id']/'capsule.json').write_text(encode(altered))
    monkeypatch.setattr(recovery,'_rehearse_preparation',change)
    with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert not list(roots['receipts'].glob('p1-recovery-*'))
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(0,)
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)


def test_lotp_capacity_rehearsal_preserves_actual_wal_prestate(b23_host,installation,monkeypatch):
    h=b23_host;roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    rehearse=recovery._rehearse_preparation;insert=recovery._insert_recovery;original_files={};seen=[]
    def capture(session,*args):
        root=roots['control']/'core'
        original_files.update({name:recovery.read_file(root/name,4_194_304)
            for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm')})
        assert len(original_files['core.sqlite-wal'])>32
        return rehearse(session,*args)
    def check(db,intent):
        path=Path(db.execute('PRAGMA database_list').fetchone()[2])
        if path.parent.name.startswith('p1-rehearsal-'):
            assert path.read_bytes()==original_files['core.sqlite']
            assert (path.parent/'core.sqlite-wal').read_bytes()==original_files['core.sqlite-wal']
            assert (path.parent/'core.sqlite-shm').stat().st_size==len(original_files['core.sqlite-shm'])
            seen.append(True)
        return insert(db,intent)
    monkeypatch.setattr(recovery,'_rehearse_preparation',capture)
    monkeypatch.setattr(recovery,'_insert_recovery',check)
    prepare=recovery._prepare_abandon_locked
    def verify(session,**request):
        result=prepare(session,**request)
        assert seen==[True]
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',verify)
    assert recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent']


# K correction: production P acceptance, actual child death, owned fresh inputs.
def _kpre_accept(installation, h, monkeypatch, point='accepted-parent', *, population=True):
    roots,scope,profiles,_=installation
    if population:_lotp_population(installation,'running')
    _lotp_transport(h,monkeypatch)
    trace=roots['control'].parent/'kpre-death.txt'
    prepare=recovery._prepare_abandon_locked
    def die(session,**request):
        result=prepare(session,**request)
        def fault(where):
            if where==point:
                trace.write_text(where);os._exit(73)
        session.continue_kernel(result['intent'],prepared=result._snapshot,fault=fault)
        raise AssertionError('death hook not reached')
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',die)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert trace.read_text()==point
    paths=[p for p in roots['receipts'].glob('p1-recovery-*.intent.json') if p.name.count('.')==2]
    paths.sort(key=lambda p:json.loads(p.read_bytes())['created'])
    paths=[paths[-1]]
    intent=json.loads(paths[0].read_bytes())
    assert paths[0].read_bytes()==encode(intent).encode()
    assert (roots['receipts']/'p1-b2-required.json').read_bytes()==encode(dict(revision=1,deployment_id=intent['deployment_id'])).encode()
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries WHERE id=?',(intent['id'],)).fetchone()==(0,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps WHERE recovery_id=?',(intent['id'],)).fetchone()==(0,)
    assert not (roots['receipts']/('p1-recovery-'+intent['id']+'.bundle.intent.json')).exists()
    assert not (roots['receipts']/('p1-recovery-'+intent['id']+'.snapshot.bundle')).exists()
    return intent


_KPRE_FRESH = r'''
import hashlib,json,os,resource,sqlite3,sys
from pathlib import Path
from service import recovery
v=json.load(sys.stdin);roots={k:Path(p) for k,p in v['roots'].items()};scope=v['scope']
sys.platform='linux';resource.setrlimit=lambda *a:None;resource.getrlimit=lambda *a:(4_194_304,)*2
recovery._maintenance_filesystems=lambda roots:None;recovery._maintenance_task_count=lambda:1
paths=[p for p in roots['receipts'].glob('p1-recovery-*.intent.json') if p.name.count('.')==2]
latest=max((json.loads(p.read_bytes()) for p in paths),key=lambda r:r['created'])
if v.get('target'):paths=[p for p in paths if p.name=='p1-recovery-'+v['target']+'.intent.json']
assert len(paths)==1
accepted=json.loads(paths[0].read_bytes());identity=accepted['id']
regs=[json.loads(r[1]) for r in latest['before']['source']['data']['p1_registrations']]
# E1 OS transport model only; no caller snapshot/document is an input.
recovery.read_host_capsule=lambda:dict(scope=scope,registrations=regs,
    registration_digest=recovery.host_registration_digest(regs),kernel_proof=accepted['proof'],
    kernel_proof_digest=accepted['proof_digest'],purpose=dict(kind='resume',plan_id=accepted['plan_id'],plan_digest=accepted['plan_digest']))
exec(v.get('setup',''))
for attempt in range(v.get('attempts',2)):
    def resume(session):
        owned=json.loads(paths[0].read_bytes())
        try:
            def fault(p):
                if p==v.get('death'):
                    Path(v['trace']).write_text(p);os._exit(73)
            result=session.continue_kernel(owned,fault=fault)
        except BaseException as exc:
            if not isinstance(exc,(ValueError,OSError,sqlite3.Error)) or not v.get('refuse'):
                import traceback
                traceback.print_exc();raise
            assert not session._live
            try:session.continue_kernel(owned)
            except ValueError as e:assert str(e)=='b2_maintenance_session_closed'
            else:raise AssertionError('live retry')
            return True
        assert not v.get('refuse'),'unexpected success'
        assert result['status']=='complete'
        return True
    assert recovery._run_bounded_maintenance(roots,scope,resume)
exec(v.get('check',''))
if not v.get('refuse'):
    bundle=roots['receipts']/('p1-recovery-'+identity+'.snapshot.bundle')
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==accepted['expected'][1]['output']['snapshot_digest']
    archive=next(n for n,k in recovery.archive_names(accepted).items() if k=='snapshot')
    assert (roots['control']/'core'/archive).read_bytes()==bundle.read_bytes()
    assert recovery.read_file(paths[0])==recovery.encode(accepted).encode()
'''


def _kpre_fresh(roots,scope,**options):
    import subprocess,sys
    child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(
        roots={k:str(p) for k,p in roots.items()},scope=scope,**options)),capture_output=True,text=True,timeout=90)
    assert child.returncode==0,child.stderr


@pytest.mark.parametrize('point',['accepted-parent','fence-parent'])
def test_kpre_actual_death_owned_fresh_resume(b23_host,installation,monkeypatch,point):
    roots,scope,_,_=installation
    intent=_kpre_accept(installation,b23_host,monkeypatch,point)
    assert (roots['receipts']/'p1-maintenance.json').exists()==(point=='fence-parent')
    _kpre_fresh(roots,scope)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)


@pytest.mark.parametrize('point',['accepted-write','accepted-fsync','accepted-parent','fence-write','fence-fsync','fence-parent'])
@pytest.mark.parametrize('journal_state',['valid','missing','unreadable'])
def test_kpre_acceptance_barriers_journal(b23_host,installation,monkeypatch,point,journal_state):
    roots,scope,_,_=installation
    if journal_state=='missing':(roots['receipts']/'journal.json').unlink()
    elif journal_state=='unreadable':(roots['receipts']/'journal.json').write_bytes(b'bounded damaged original journal')
    intent=_kpre_accept(installation,b23_host,monkeypatch,point)
    _kpre_fresh(roots,scope,setup="""
clock=recovery.time.time
recovery.time.time=lambda:accepted['created']+1000 # accepted TTL is irrelevant; transport explicitly modeled
capture=recovery._capture_original_snapshot
calls=[]
def once(*args):
    calls.append(1);assert len(calls)==1
    assert args[2]==1_048_576
    return capture(*args)
recovery._capture_original_snapshot=once
""",check="assert len(calls)==0 # captures occur only in bounded children")
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
    assert (roots['receipts']/('p1-recovery-'+intent['id']+'.journal')).read_bytes()==bytes.fromhex(intent['before']['journal']['bytes_hex'])


@pytest.mark.parametrize('point',['recapture-before','recapture-after','recapture-hash','recapture-oracle','recapture-verified'])
def test_kpre_recapture_actual_death_fresh_retry(b23_host,installation,monkeypatch,point):
    import subprocess,sys
    roots,scope,_,_=installation
    _kpre_accept(installation,b23_host,monkeypatch)
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(
        roots={k:str(p) for k,p in roots.items()},scope=scope,death=point,trace=str(roots['control'].parent/'recapture-death.txt'),attempts=1)),capture_output=True,text=True,timeout=60)
    assert child.returncode!=0
    assert (roots['control'].parent/'recapture-death.txt').read_text()==point
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    _kpre_fresh(roots,scope)


@pytest.mark.parametrize('damage',['snapshot','manifest','original-journal','rotation','core-certificate',
    'receipt-certificate','completion-journal','completion-fence','bundle-descriptor','bundle-output',
    'dangling-descriptor','dangling-bundle','mode','hardlink','oversize','extra','temporary','marker',
    'accepted','fence','source','plan','later-plan','schema','media','journal','proof','physical-hash'])
def test_kpre_no_attempt_closed_refusal_twice(b23_host,installation,monkeypatch,damage):
    roots,scope,_,_=installation
    r=_kpre_accept(installation,b23_host,monkeypatch)
    receipts=roots['receipts'];prefix='p1-recovery-'+r['id'];root=roots['control']/'core'
    if damage in ('snapshot','manifest','original-journal','rotation','core-certificate','receipt-certificate','completion-journal','completion-fence'):
        (receipts/(prefix+'.'+damage+'.intent.json')).write_bytes(b'{}')
    elif damage=='bundle-descriptor':(receipts/(prefix+'.bundle.intent.json')).write_text(encode(dict(revision=1,recovery_id=r['id'],intent_digest=digest(r),source_digest=r['source_digest'],snapshot_digest=r['expected'][1]['output']['snapshot_digest'])))
    elif damage=='bundle-output':(receipts/(prefix+'.snapshot.bundle')).write_bytes(b'output without descriptor')
    elif damage.startswith('dangling-'):(receipts/(prefix+('.bundle.intent.json' if damage.endswith('descriptor') else '.snapshot.bundle'))).symlink_to(receipts/'absent')
    elif damage=='mode':(receipts/(prefix+'.intent.json')).chmod(0o644)
    elif damage=='hardlink':os.link(receipts/(prefix+'.intent.json'),receipts/'extra-link')
    elif damage=='oversize':(receipts/(prefix+'.snapshot.bundle')).write_bytes(b'x'*(1024**2+1))
    elif damage=='extra':(receipts/'foreign').write_bytes(b'x')
    elif damage=='temporary':(receipts/'p1-maintenance.pending.json').write_bytes(b'{}')
    elif damage=='marker':(receipts/'p1-b2-required.json').unlink()
    elif damage in ('accepted','proof'):
        if damage=='proof':r['proof']['kind']='changed'
        else:r['source_digest']='f'*64
        (receipts/(prefix+'.intent.json')).write_text(encode(r))
    elif damage=='fence':(receipts/'p1-maintenance.json').write_bytes(b'partial')
    elif damage=='media':(roots['staging']/'new-defect').write_bytes(b'new')
    elif damage=='journal':(receipts/'journal.json').write_bytes(b'changed')
    elif damage!='physical-hash':
        with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
            if damage=='source':db.execute("UPDATE resource_monitor SET state='fault'")
            elif damage=='plan':db.execute('DELETE FROM p1_abandon_plans')
            elif damage=='later-plan':
                raw=json.loads(db.execute('SELECT record FROM p1_abandon_plans').fetchone()[0]);raw['id']='f'*32
                db.execute('INSERT INTO p1_abandon_plans VALUES(?,?,?)',(raw['id'],encode(raw),digest(raw)))
            elif damage=='schema':db.execute('CREATE TABLE foreign_table(id)')
    # Measure after crash-WAL opening; no claim of zero SQLite physical churn.
    before={p.name:(p.lstat().st_mode,p.read_bytes() if not p.is_symlink() else os.readlink(p)) for p in receipts.iterdir()}
    setup="""
def forbid(*a):raise AssertionError('capture before closed predicate')
recovery._capture_original_snapshot=forbid
"""
    if damage=='physical-hash':setup="""
capture=recovery._capture_original_snapshot
def different(*args):
    raw=bytearray(capture(*args));raw[24]^=1
    return bytes(raw) # SQLite header change counter: logically equal, physically distinct
recovery._capture_original_snapshot=different
"""
    _kpre_fresh(roots,scope,refuse=True,setup=setup)
    assert before=={p.name:(p.lstat().st_mode,p.read_bytes() if not p.is_symlink() else os.readlink(p)) for p in receipts.iterdir()}


@pytest.mark.parametrize('failure',['copy','budget','cap','readback','host','ceiling','vfs','read-io','oracle','prospective'])
def test_kpre_capture_failures_burn_without_outputs(b23_host,installation,monkeypatch,failure):
    roots,scope,_,_=installation
    _kpre_accept(installation,b23_host,monkeypatch)
    before_sql=logical(roots)
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    setup="def fail(*a,**kw):raise ValueError('modeled bounded failure')\n"
    if failure=='copy':setup+="recovery._capture_original_snapshot=fail"
    elif failure=='budget':setup+="import service.resource_store as store\nstore._maintenance_footprint=fail"
    elif failure=='cap':setup+="recovery._MaintenanceProcess.verify=fail"
    elif failure=='readback':setup+="resource.getrlimit=lambda *a:(4_194_303,)*2"
    elif failure=='host':setup+="recovery.read_host_capsule=fail"
    elif failure=='ceiling':setup+="recovery.MAINTENANCE_SNAPSHOT_LIMIT=1024"
    elif failure=='vfs':setup+='''
import sqlite3
connect=sqlite3.connect
def unavailable(path,*a,**kw):
    if 'snapshot.sqlite?mode=rwc&vfs=unix' in str(path):raise sqlite3.OperationalError('modeled Unix VFS unavailable')
    return connect(path,*a,**kw)
sqlite3.connect=unavailable
'''
    elif failure=='read-io':setup+='''
reader=recovery.read_file
def denied(path,*a,**kw):
    if path.name=='snapshot.sqlite':raise OSError('modeled snapshot read denied')
    return reader(path,*a,**kw)
recovery.read_file=denied
'''
    elif failure=='oracle':setup+='''
validate=recovery._validate_archives
def oracle(root,data,*a,**kw):
    if not data['p1_recoveries']:return validate(root,data,*a,**kw)
    path=next(root.glob('p1-recovery-*.sqlite'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted['expected'][1]['output']['snapshot_digest']
    raise ValueError('exact hash still requires semantic oracle')
recovery._validate_archives=oracle
'''
    else:setup+="recovery._insert_recovery=fail"
    # Cap failure is before session creation and cannot be caught by operation.
    if failure in ('cap','readback','host'):
        import subprocess,sys
        for _ in range(2):
            child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,setup=setup)),capture_output=True,text=True,timeout=60)
            assert child.returncode!=0
    else:_kpre_fresh(roots,scope,refuse=True,setup=setup)
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}

    assert logical(roots)==before_sql


def _kpre_predecessor(installation,h,monkeypatch):
    roots,scope,profiles,_=installation
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
    _lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def finish(session,**request):
        result=prepare(session,**request)
        assert session.continue_kernel(result['intent'],prepared=result._snapshot)['status']=='complete'
        return result
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish)
    first=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',prepare)
    reg=copy.deepcopy(h['registration']);reg['lane']=first['intent']['new_lane'];reg['boot']='e'*32
    for index,owner in enumerate(reg['containers']):
        owner['id']=('e' if index==0 else 'f')*64;owner['started']=f'2026-10-17T03:00:0{index}Z'
    recovery.register(roots,scope,reg)
    provenance=copy.deepcopy(h['provenance']);provenance['registration_digest']=digest(reg)
    provenance['hello']={k:reg[k] for k in ('lane','boot')}
    for owner,c in zip(provenance['owners'],reg['containers']):owner.update(id=c['id'],started=c['started'],cgroup='/docker/'+c['id'])
    h['module']._host_write(h['base']/('registration-'+digest(reg)+'.json'),provenance)
    for old,c in zip(copy.deepcopy(h['consumers']),reg['containers']):
        old.update(id=c['id'],started=c['started'],finished='2026-10-17T04:00:00Z');h['consumers'].append(old)
    h['state']=h['module'].host_state(roots,scope)
    h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])
    return first['intent']


@pytest.mark.parametrize('point',['accepted-write','accepted-fsync','accepted-parent','fence-replace-parent'])
@pytest.mark.parametrize('journal_state',['valid','missing','unreadable'])
def test_kpre_history_and_observational_replay(b23_host,installation,monkeypatch,point,journal_state):
    roots,scope,_,_=installation
    first=_kpre_predecessor(installation,b23_host,monkeypatch)
    if journal_state=='missing':(roots['receipts']/'journal.json').unlink()
    elif journal_state=='unreadable':(roots['receipts']/'journal.json').write_bytes(b'bounded later damaged journal')
    second=_kpre_accept(installation,b23_host,monkeypatch,point,population=False)
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    _kpre_fresh(roots,scope,target=first['id'],setup="""
recovery.time.time=lambda:accepted['created']+1000
def forbidden(*a,**kw):raise AssertionError('historical replay mutated/scanned/captured')
recovery._capture_original_snapshot=forbidden
recovery.snapshot=forbidden
recovery._prefix_file=forbidden
recovery._prefix_transaction=forbidden
""")
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    _kpre_fresh(roots,scope,target=second['id'],setup="recovery.time.time=lambda:accepted['created']+1000")
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(2,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(10,)


@pytest.mark.parametrize('damage',['absent-fence','old-bundle','old-core-descriptor','old-completion-descriptor','old-archive'])
def test_kpre_predecessor_damage_no_capture(b23_host,installation,monkeypatch,damage):
    roots,scope,_,_=installation
    first=_kpre_predecessor(installation,b23_host,monkeypatch)
    second=_kpre_accept(installation,b23_host,monkeypatch,population=False)
    receipts=roots['receipts'];prefix='p1-recovery-'+first['id']
    if damage=='absent-fence':(receipts/'p1-maintenance.json').unlink()
    elif damage=='old-bundle':(receipts/(prefix+'.snapshot.bundle')).write_bytes(b'changed')
    elif damage=='old-core-descriptor':(receipts/(prefix+'.core-certificate.intent.json')).unlink()
    elif damage=='old-completion-descriptor':(receipts/(prefix+'.completion-fence.intent.json')).unlink()
    else:next((roots['control']/'core'/n for n,k in recovery.archive_names(first).items() if k=='manifest')).write_bytes(b'changed')
    before={p.name:p.read_bytes() for p in receipts.iterdir()}
    _kpre_fresh(roots,scope,target=second['id'],refuse=True,setup="def forbid(*a):raise AssertionError('damaged history captured')\nrecovery._capture_original_snapshot=forbid")
    assert before=={p.name:p.read_bytes() for p in receipts.iterdir()}


def _kpre_attempt(installation,h,monkeypatch,point):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(h,monkeypatch)
    trace=roots['control'].parent/'kpre-death.txt'
    prepare=recovery._prepare_abandon_locked
    def die(session,**request):
        result=prepare(session,**request)
        def fault(where):
            if where==point:
                trace.write_text(where);os._exit(73)
        session.continue_kernel(result['intent'],prepared=result._snapshot,fault=fault)
        raise AssertionError('hook not reached')
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',die)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert trace.read_text()==point
    paths=[p for p in roots['receipts'].glob('p1-recovery-*.intent.json') if p.name.count('.')==2]
    assert len(paths)==1
    r=json.loads(paths[0].read_bytes())
    assert (roots['receipts']/('p1-recovery-'+r['id']+'.bundle.intent.json')).exists()
    return r


@pytest.mark.parametrize('point',['bundle-intent-create','bundle-intent-write','bundle-intent-fsync',
    'bundle-intent-parent','bundle-intent-recorded','bundle-create','bundle-write','bundle-fsync','bundle-parent',
    'core-intent-begin','core-intent-write','core-intent-commit','core-intent-parent',
    'pending-receipt-commit','evidence-receipt-commit','journal-receipt-commit','core-complete-commit','receipt-complete-commit'])
def test_kpre_after_attempt_never_recaptures(b23_host,installation,monkeypatch,point):
    roots,scope,_,_=installation
    _kpre_attempt(installation,b23_host,monkeypatch,point)
    refuse=point in ('bundle-intent-create','bundle-intent-write','bundle-intent-fsync','bundle-intent-parent','bundle-intent-recorded','bundle-create')
    _kpre_fresh(roots,scope,refuse=refuse,setup="def forbidden(*a):raise AssertionError('recapture after creation attempt')\nrecovery._capture_original_snapshot=forbidden")


@pytest.mark.parametrize('point',['core-intent-commit','pending-receipt-commit','evidence-receipt-commit','receipt-complete-commit'])
def test_kpre_deleted_bundle_sql_is_not_no_attempt(b23_host,installation,monkeypatch,point):
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,point)
    for suffix in ('.bundle.intent.json','.snapshot.bundle'):(roots['receipts']/('p1-recovery-'+r['id']+suffix)).unlink()
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    _kpre_fresh(roots,scope,refuse=True,setup="def forbidden(*a):raise AssertionError('deleted attempt recaptured')\nrecovery._capture_original_snapshot=forbidden")
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}


def test_kpre_recapture_connections_at_actual_use_and_close(b23_host,installation,monkeypatch):
    roots,scope,_,_=installation
    _kpre_accept(installation,b23_host,monkeypatch)
    _kpre_fresh(roots,scope,attempts=1,setup=r'''
import sqlite3,urllib.parse
connect=sqlite3.connect;opened=[];backups=[];inserts=[]
class Audit(sqlite3.Connection):
    checked=False;reader_checked=False;explicitly_closed=False
    def audit(self):
        assert resource.getrlimit(resource.RLIMIT_FSIZE)==(4_194_304,)*2
        for sql in ("ATTACH ':memory:' AS extra_db",'DETACH extra_db','PRAGMA mmap_size=1048576',
            'PRAGMA journal_mode=DELETE','PRAGMA max_page_count=10','PRAGMA query_only=OFF',"SELECT load_extension('foreign')"):
            try:super().execute(sql)
            except sqlite3.DatabaseError:pass
            else:raise AssertionError(sql)
        for key,value in [('trusted_schema',0),('temp_store',2),('mmap_size',0)]:
            assert super().execute('PRAGMA '+key).fetchone()==(value,)
    def execute(self,sql,*args,**kw):
        if not self.checked and sql.startswith('INSERT INTO p1_recoveries '):
            self.checked=True;self.audit();inserts.append(self)
        if not self.reader_checked and (sql.lower().startswith('pragma table_info') or sql.startswith('SELECT record FROM p1_registrations')):
            self.reader_checked=True;self.audit()
        return super().execute(sql,*args,**kw)
    def backup(self,target,*args,**kw):
        self.reader_checked=True;self.audit();target.reader_checked=True;target.audit();backups.append(self)
        return super().backup(target,*args,**kw)
    def close(self):self.explicitly_closed=True;return super().close()
def audited(database,*args,**kw):
    text=str(database)
    if any(tag in text for tag in ('p1-evidence-','p1-inspect-')):
        assert kw.get('uri') and 'vfs=unix' in text
        kw['factory']=Audit;db=connect(database,*args,**kw);opened.append(db);return db
    return connect(database,*args,**kw)
sqlite3.connect=audited
continuation=recovery._MaintenanceSession.continue_kernel
def audit_continuation(session,*a,**kw):
    result=continuation(session,*a,**kw)
    assert len(backups)==1 and len(inserts)==1
    assert opened and all(db.checked or db.reader_checked for db in opened)
    assert all(db.explicitly_closed for db in opened)
    return result
recovery._MaintenanceSession.continue_kernel=audit_continuation
''')


@pytest.mark.parametrize('role',['control','receipts'])
def test_kpre_no_attempt_both_busy_locks(b23_host,installation,monkeypatch,role):
    import fcntl,subprocess,sys
    roots,scope,_,_=installation
    _kpre_accept(installation,b23_host,monkeypatch)
    lock=roots[role]/('core/control.lock' if role=='control' else 'lane.lock')
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    with lock.open('rb') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(2):
            child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope)),capture_output=True,text=True,timeout=60)
            assert child.returncode!=0 and 'child_failed' in child.stderr
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}


@pytest.mark.parametrize('damage',['multiple','foreign-fence','key','plan-content','control-output','journal-link','noncanonical','marker-partial'])
def test_kpre_additional_closed_inputs(b23_host,installation,monkeypatch,damage):
    roots,scope,_,_=installation
    r=_kpre_accept(installation,b23_host,monkeypatch)
    receipts=roots['receipts'];accepted=receipts/('p1-recovery-'+r['id']+'.intent.json')
    if damage=='multiple':
        other=copy.deepcopy(r);other['id']='f'*32
        p=receipts/('p1-recovery-'+other['id']+'.intent.json');p.write_text(encode(other));p.chmod(0o600)
    elif damage=='foreign-fence':
        fence=dict(revision=1,**{k:r[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id='f'*32,step='pending',certificate=None)
        p=receipts/'p1-maintenance.json';p.write_text(encode(fence));p.chmod(0o600)
    elif damage=='key':(roots['ipc']/'lifecycle.key').write_bytes(b'changed')
    elif damage=='plan-content':
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            plan=json.loads(db.execute('SELECT record FROM p1_abandon_plans').fetchone()[0]);plan['expires']-=1
            db.execute('UPDATE p1_abandon_plans SET record=?,digest=?',(encode(plan),digest(plan)))
    elif damage=='control-output':
        p=roots['control']/'core'/next(iter(recovery.archive_names(r)));p.write_bytes(b'premature');p.chmod(0o600)
    elif damage=='journal-link':
        (receipts/'journal.json').unlink();(receipts/'journal.json').symlink_to(receipts/'absent')
    elif damage=='noncanonical':accepted.write_text(json.dumps(r,indent=1))
    else:(receipts/'p1-b2-required.json').write_bytes(b'partial')
    before={p.name:p.read_bytes() if not p.is_symlink() else os.readlink(p) for p in receipts.iterdir()}
    _kpre_fresh(roots,scope,target=r['id'],refuse=True,setup="def forbidden(*a):raise AssertionError('invalid predicate captured')\nrecovery._capture_original_snapshot=forbidden")
    assert before=={p.name:p.read_bytes() if not p.is_symlink() else os.readlink(p) for p in receipts.iterdir()}


@pytest.mark.parametrize('point',['fence-create','fence-write','fence-fsync','fence-parent'])
def test_kpre_pending_replacement_temporary_quarantines(b23_host,installation,monkeypatch,point):
    roots,scope,_,_=installation
    first=_kpre_predecessor(installation,b23_host,monkeypatch)
    second=_kpre_accept(installation,b23_host,monkeypatch,point,population=False)
    assert (roots['receipts']/'p1-maintenance.pending.json').exists()
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    for r in (first,second):
        _kpre_fresh(roots,scope,target=r['id'],refuse=True,setup="def forbidden(*a):raise AssertionError('temporary captured')\nrecovery._capture_original_snapshot=forbidden")
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}


@pytest.mark.parametrize('history,point',[(False,p) for p in ('accepted-write','accepted-fsync','accepted-parent','fence-write','fence-fsync','fence-parent')]+[(True,p) for p in ('accepted-write','accepted-fsync','accepted-parent','fence-replace-parent')])
@pytest.mark.parametrize('journal_state',['valid','missing','unreadable'])
@pytest.mark.parametrize('failure',['source','capsule','control-lock','lane-lock'])
def test_kpre_each_no_attempt_window_rechecks(b23_host,installation,monkeypatch,history,point,journal_state,failure):
    import fcntl,subprocess,sys
    roots,scope,_,_=installation
    if history:_kpre_predecessor(installation,b23_host,monkeypatch)
    if journal_state=='missing':(roots['receipts']/'journal.json').unlink()
    elif journal_state=='unreadable':(roots['receipts']/'journal.json').write_bytes(b'bounded damaged journal')
    r=_kpre_accept(installation,b23_host,monkeypatch,point,population=not history)
    if failure=='source':
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute("UPDATE resource_monitor SET state='fault'")
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    setup="def forbidden(*a):raise AssertionError('unverified recapture')\nrecovery._capture_original_snapshot=forbidden\n"
    if failure=='source':_kpre_fresh(roots,scope,target=r['id'],refuse=True,setup=setup)
    else:
        if failure=='capsule':setup+="def absent():raise ValueError('fresh trusted capsule unavailable')\nrecovery.read_host_capsule=absent"
        lock=None
        try:
            if failure.endswith('lock'):
                lock=(roots['control']/'core'/'control.lock' if failure=='control-lock' else roots['receipts']/'lane.lock').open('rb')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(2):
                child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,target=r['id'],setup=setup)),capture_output=True,text=True,timeout=60)
                assert child.returncode!=0 and 'child_failed' in child.stderr
        finally:
            if lock is not None:lock.close()
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}


@pytest.mark.parametrize('field',['new_epoch','new_lane'])
def test_kpre_history_identity_collision_refuses_before_capture(b23_host,installation,monkeypatch,field):
    roots,scope,_,_=installation
    first=_kpre_predecessor(installation,b23_host,monkeypatch)
    r=_kpre_accept(installation,b23_host,monkeypatch,population=False)
    # A fully rehashed, closed plan/acceptance collision must fail before backup,
    # not merely at the later frozen-snapshot/SQL oracle. No repair is requested.
    r[field]=first['new_lane' if field=='new_epoch' else 'new_epoch']
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        plan=json.loads(db.execute('SELECT record FROM p1_abandon_plans WHERE id=?',(r['plan_id'],)).fetchone()[0])
        plan[field]=r[field];recovery.validate_abandon_plan(plan)
        r['plan_digest']=digest(plan)
        db.execute('UPDATE p1_abandon_plans SET record=?,digest=? WHERE id=?',(encode(plan),digest(plan),plan['id']))
    r['expected'][0]['output']['plan_digest']=r['plan_digest']
    r['expected'][3]['output']['new_epoch']=r['new_epoch']
    r['expected'][2]['output']['new_lane']=r['new_lane']
    r['expected'][2]['output']['journal']['lane']=r['new_lane']
    r['expected'][4]['output'].update(lane=r['new_lane'],epoch=r['new_epoch'])
    previous=r['before_digest']
    for descriptor in r['expected']:
        if descriptor['step']=='receipt-complete':descriptor['output']['core_output_digest']=previous
        descriptor['input_digest']=previous
        descriptor['output_digest']=digest({k:v for k,v in descriptor.items() if k!='output_digest'});previous=descriptor['output_digest']
    recovery.validate_b2_record('p1_recoveries',r)
    accepted=roots['receipts']/('p1-recovery-'+r['id']+'.intent.json');accepted.write_text(encode(r))
    before={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}
    _kpre_fresh(roots,scope,target=r['id'],refuse=True,setup="def forbidden(*a):raise AssertionError('identity collision captured')\nrecovery._capture_original_snapshot=forbidden")
    assert before=={p.name:p.read_bytes() for p in roots['receipts'].iterdir()}


# Certified unfinished K must validate current source before any remaining write.
_KFIFTH_FORBID = """
def forbidden(*args,**kwargs):raise AssertionError('unfinished drift reached writer')
for name in ('_prefix_file','_prefix_transaction','_core_certificate','_finish_receipts','_capture_original_snapshot','_write_core_completion'):
    setattr(recovery,name,forbidden)
"""


@pytest.mark.parametrize('repeat',range(2))
def test_kfifth_actual_death_monitor_refuses_before_writers(b23_host,installation,monkeypatch,repeat):
    roots,scope,_,_=installation
    intent=_kpre_attempt(installation,b23_host,monkeypatch,'completion-fence-dirsync')
    path=roots['control']/'core'/'core.sqlite'
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT count(*) FROM p1_recovery_steps WHERE step='receipt-complete'").fetchone()==(0,)
        db.execute("UPDATE resource_monitor SET state='fault' WHERE id=1");db.commit()
        with pytest.raises(ValueError,match='b2_core_output_changed'):recovery._verify_core_rows(db,intent)
        before=list(db.iterdump())
    receipts={p.name:p.read_bytes() for p in roots['receipts'].iterdir() if p.is_file()}
    _kfifth_fresh(roots,scope,refuse=True,setup=_KFIFTH_FORBID)
    with closing(sqlite3.connect(path)) as db:
        assert list(db.iterdump())==before
        assert db.execute("SELECT state FROM resource_monitor WHERE id=1").fetchone()==('fault',)
    assert {p.name:p.read_bytes() for p in roots['receipts'].iterdir() if p.is_file()}==receipts


_KFIFTH_STAGES=['core-complete-commit','core-complete-parent','core-certificate-intent-recorded',
    'core-certificate-parent','receipt-certificate-intent-recorded','receipt-certificate-parent',
    'completion-journal-intent-recorded','completion-journal-replace','completion-journal-dirsync',
    'completion-fence-intent-recorded','completion-fence-replace','completion-fence-dirsync']


@pytest.mark.parametrize('point',_KFIFTH_STAGES)
@pytest.mark.parametrize('damage',['none','monitor'])
def test_kfifth_reached_actual_death_prefix(b23_host,installation,monkeypatch,point,damage):
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,point)
    # Intent-only attempts with missing/old targets quarantine, including when
    # every earlier durable output is correct. No future-output prerequisites.
    quarantine=point.endswith('intent-recorded') or damage!='none'
    if damage=='monitor':
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute("UPDATE resource_monitor SET state='fault'")
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,refuse=quarantine,setup=_KFIFTH_FORBID if quarantine else
        "recovery.time.time=lambda:accepted['created']+1000")
    if quarantine:assert logical(roots)==before and retained_bytes(roots)==files
    else:
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
            assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
        assert (roots['receipts']/('p1-recovery-'+r['id']+'.intent.json')).read_bytes()==encode(r).encode()


@pytest.mark.parametrize('damage',['jobs','resource_reservations','resource_monitor','resource_faults',
    'resource_accounting','p1_integrity','dispatch_resolutions','p1_retired_publications',
    'p1_integrity_resolutions','unaffected-sql','typed-ref','history','media','key','installation','volume'])
@pytest.mark.parametrize('point',['core-certificate-parent','completion-fence-dirsync'])
def test_kfifth_full_current_source_before_writers(b23_host,installation,monkeypatch,damage,point):
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,point)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        if damage in ('jobs','resource_reservations','dispatch_resolutions'):
            assert db.execute('DELETE FROM '+damage).rowcount>0
        elif damage=='resource_monitor':db.execute("UPDATE resource_monitor SET state='fault'")
        elif damage=='resource_faults':db.execute("INSERT INTO resource_faults VALUES('new-fault')")
        elif damage=='resource_accounting':db.execute('UPDATE resource_accounting SET highwater=highwater+1')
        elif damage=='p1_integrity':db.execute("UPDATE p1_integrity SET state='checking'")
        elif damage in ('p1_retired_publications','p1_integrity_resolutions'):
            db.execute('INSERT INTO '+damage+' VALUES(?,?,?,?)',('unexpected',r['id'],'{}','0'*64))
        elif damage=='typed-ref':
            seq,raw=db.execute('SELECT seq,authority FROM dispatches LIMIT 1').fetchone()
            value=json.loads(raw);value[0]['artifact_ids']=['unexpected-artifact']
            db.execute('UPDATE dispatches SET authority=? WHERE seq=?',(encode(value),seq))
        elif damage=='history':
            assert db.execute("UPDATE events SET state='changed-history'").rowcount>0
        elif damage=='unaffected-sql':
            # SQLite's retained sequence is part of the exact source, too.
            assert db.execute('UPDATE sqlite_sequence SET seq=seq+1').rowcount>0
    if damage=='media':(roots['staging']/'new-defect').write_bytes(b'unexpected')
    elif damage=='key':(roots['ipc']/'execution.key').write_bytes(b'0'*64)
    elif damage=='installation':
        p=roots['receipts']/'p1-init.json';v=json.loads(p.read_bytes());v['seed']='0'*32;p.write_text(encode(v))
    elif damage=='volume':
        p=roots['receipts']/'p1-init.json';v=json.loads(p.read_bytes());v['roots']['staging'][1]+=1;p.write_text(encode(v))
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,refuse=True,setup=_KFIFTH_FORBID)
    assert logical(roots)==before and retained_bytes(roots)==files


def _kfifth_successor(installation,h,monkeypatch,point):
    """Production P on a later history; no fixture expected rows or C routing."""
    roots,scope,profiles,_=installation
    _lotp_transport(h,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    trace=roots['control'].parent/'kfifth-death.txt'
    def die(session,**request):
        result=prepare(session,**request)
        def fault(where):
            if where==point:trace.write_text(where);os._exit(73)
        session.continue_kernel(result['intent'],prepared=result._snapshot,fault=fault)
        raise AssertionError('death hook not reached')
    with monkeypatch.context() as m:
        m.setattr(recovery,'_prepare_abandon_locked',die)
        with pytest.raises(ValueError,match='child_failed'):
            recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert trace.read_text()==point
    return max((json.loads(p.read_bytes()) for p in roots['receipts'].glob('p1-recovery-*.intent.json')
        if p.name.count('.')==2),key=lambda r:r['created'])


_KFIFTH_OBSERVE = """
from service import resource_store
from service.reconciliation import Reconciliation
def forbidden(*args,**kwargs):raise AssertionError('completed history scanned or wrote')
for name in ('snapshot','_unfinished_source','_prefix_file','_prefix_transaction','_capture_original_snapshot','_write_core_completion','_finish_receipts'):
    setattr(recovery,name,forbidden)
resource_store._recovery_scan=forbidden
Reconciliation._collect=forbidden
"""


@pytest.mark.parametrize('point',['accepted-parent','fence-replace-parent','bundle-parent','core-intent-commit',
    'pending-receipt-commit','snapshot-parent','manifest-parent','original-journal-parent',
    'evidence-receipt-commit','journal-replace','journal-receipt-commit',*_KFIFTH_STAGES])
def test_kfifth_production_history_actual_prefix(b23_host,installation,monkeypatch,point):
    roots,scope,_,_=installation
    a=_kpre_predecessor(installation,b23_host,monkeypatch)
    b=_kfifth_successor(installation,b23_host,monkeypatch,point)
    quarantine=point.endswith('intent-recorded')
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,target=a['id'],refuse=quarantine,setup=_KFIFTH_OBSERVE)
    assert logical(roots)==before and retained_bytes(roots)==files
    _kfifth_fresh(roots,scope,target=b['id'],refuse=quarantine,
        setup=_KFIFTH_FORBID if quarantine else "recovery.time.time=lambda:accepted['created']+1000")
    if quarantine:assert logical(roots)==before and retained_bytes(roots)==files
    else:
        before=logical(roots);files=retained_bytes(roots)
        for r in (a,b):_kfifth_fresh(roots,scope,target=r['id'],setup=_KFIFTH_OBSERVE)
        assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('kind',['bundle','snapshot','manifest','original-journal','rotation',
    'core-certificate','receipt-certificate','completion-journal','completion-fence'])
@pytest.mark.parametrize('damage',['missing','wrong','mode'])
@pytest.mark.parametrize('direction',['older','newer'])
def test_kfifth_every_completed_descriptor_both_directions(b23_host,installation,monkeypatch,kind,damage,direction):
    roots,scope,_,_=installation
    a=_kpre_predecessor(installation,b23_host,monkeypatch)
    b=_kfifth_successor(installation,b23_host,monkeypatch,'completion-fence-dirsync')
    _kfifth_fresh(roots,scope,target=b['id'])
    victim,target=(a,b) if direction=='older' else (b,a)
    p=roots['receipts']/('p1-recovery-'+victim['id']+'.'+kind+'.intent.json')
    assert p.is_file()
    if damage=='missing':p.unlink()
    elif damage=='wrong':p.write_bytes(b'{}')
    else:p.chmod(0o644)
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,target=target['id'],refuse=True,setup=_KFIFTH_OBSERVE)
    assert logical(roots)==before and retained_bytes(roots)==files


def test_kfifth_completed_current_changes_observational(b23_host,installation,monkeypatch):
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,'completion-fence-dirsync')
    _kpre_fresh(roots,scope)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute("UPDATE resource_monitor SET state='checking'")
        db.execute('UPDATE resource_accounting SET highwater=highwater+1')
    (roots['staging']/'later-ordinary-file').write_bytes(b'later')
    (roots['receipts']/'journal.json').write_bytes(b'later ordinary journal')
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,target=r['id'],setup=_KFIFTH_OBSERVE)
    assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('kind,point',[
    ('core-certificate','core-certificate-parent'),('receipt-certificate','receipt-certificate-parent'),
    ('completion-journal','completion-journal-dirsync'),('completion-fence','completion-fence-dirsync')])
@pytest.mark.parametrize('damage',['missing','partial','wrong','descriptor','temporary'])
def test_kfifth_attempted_output_quarantines_before_writers(b23_host,installation,monkeypatch,kind,point,damage):
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,point)
    prefix='p1-recovery-'+r['id']
    if kind=='core-certificate':p=roots['control']/'core'/next(n for n,k in recovery.archive_names(r).items() if k=='certificate')
    else:p=roots['receipts']/({'receipt-certificate':prefix+'.certificate.json','completion-journal':'journal.json','completion-fence':'p1-maintenance.json'}[kind])
    if damage=='missing':p.unlink()
    elif damage in ('partial','wrong'):p.write_bytes(b'{' if damage=='partial' else b'{}')
    elif damage=='descriptor':(roots['receipts']/(prefix+'.'+kind+'.intent.json')).write_bytes(b'{}')
    else:
        p=roots['receipts']/'p1-maintenance.journal.tmp';p.write_bytes(b'leftover');p.chmod(0o600)
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,refuse=True,setup=_KFIFTH_FORBID)
    assert logical(roots)==before and retained_bytes(roots)==files


def _kfifth_fresh(roots,scope,**options):
    for _ in range(2):_kpre_fresh(roots,scope,attempts=1,**options)


def test_kfifth_fourth_query_cannot_bypass_unfinished_finish(b23_host,installation,monkeypatch):
    import subprocess,sys
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,'core-certificate-parent')
    (roots['staging']/'later-media').write_bytes(b'changed')
    before=logical(roots);files=retained_bytes(roots)
    # Existing private fourth-only query has no remaining output to create.
    # Execute it under an actual bounded owning session, with writes forbidden.
    script=_KPRE_FRESH.replace('result=session.continue_kernel(owned,fault=fault)',
        "result=recovery._pending_prefix_locked(session.roots,session.db,scope,owned,owned['proof']['observation'],now=accepted['created']+1000,through_journal=True,through_core=True,_session=session)")
    script=script.replace("assert result['status']=='complete'","assert result['status']=='core-complete-receipt-pending'")
    setup="""
from service import resource_store
from service.reconciliation import Reconciliation
def forbidden(*a,**k):raise AssertionError('fourth-only observation scanned or wrote')
for name in ('snapshot','_unfinished_source','_prefix_file','_prefix_transaction','_capture_original_snapshot','_core_certificate','_finish_receipts','_write_core_completion'):
    setattr(recovery,name,forbidden)
resource_store._recovery_scan=forbidden
Reconciliation._collect=forbidden
"""
    for _ in range(2):
        child=_guarded_fresh_run([sys.executable,'-c',script],input=encode(dict(
            roots={k:str(p) for k,p in roots.items()},scope=scope,attempts=1,setup=setup)),
            capture_output=True,text=True,timeout=90)
        assert child.returncode==0,child.stderr
        assert logical(roots)==before and retained_bytes(roots)==files
    # Explicit completion still reconstructs current source and refuses BEFORE
    # all writers. The observation minted neither fifth nor a new certificate.
    _kfifth_fresh(roots,scope,refuse=True,setup=_KFIFTH_FORBID)
    assert logical(roots)==before and retained_bytes(roots)==files
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute("SELECT count(*) FROM p1_recovery_steps WHERE step='receipt-complete'").fetchone()==(0,)


@pytest.mark.parametrize('restart,point',[
    ('same-container','completion-fence-dirsync'),
    ('new-registration','completion-fence-dirsync'),
    ('different-container','completion-fence-dirsync')])
def test_kfifth_production_owner_changes_before_writers(b23_host,installation,monkeypatch,restart,point):
    """Real P acceptance, then a retained new incarnation, then fresh direct K."""
    import subprocess,sys
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,point)
    reg=json.loads(r['before']['source']['data']['p1_registrations'][0][1])
    reg.update(boot='e'*32,lane=r['new_lane'])
    if restart=='same-container':
        from worker.runtime_journal import Journal
        from worker.runtime_supervisor import RuntimeSupervisor
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
        supervisor.journal=Journal(roots['receipts']);supervisor.boot=reg['boot']
        supervisor.failed=False;supervisor.active=None;supervisor.launches=0
        try:hello=supervisor.operation('hello',{})
        finally:supervisor.journal.lock.close()
        reg.update(lane=hello['lane'],boot=hello['boot'])
    elif restart=='different-container':
        reg['containers'][1].update(id='e'*64,started='restarted-worker')
    recovery.register(roots,scope,reg)
    before=logical(roots);files=retained_bytes(roots)
    # Original capsule coverage no longer equals retained registrations. The
    # owning session refuses before opening the original writable connection;
    # do not manufacture a replacement proof or repair the original capsule.
    for _ in range(2):
        child=_guarded_fresh_run([sys.executable,'-c',_KPRE_FRESH],input=encode(dict(
            roots={k:str(p) for k,p in roots.items()},scope=scope,attempts=1,
            setup=_KFIFTH_FORBID)),capture_output=True,text=True,timeout=90)
        assert child.returncode!=0 and 'b2_maintenance_child_failed' in child.stderr
        assert logical(roots)==before and retained_bytes(roots)==files
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute("SELECT count(*) FROM p1_recovery_steps WHERE step='receipt-complete'").fetchone()==(0,)


@pytest.mark.parametrize('missing',['descriptor','output','both'])
def test_kfifth_fourth_query_missing_certificate_cannot_write(b23_host,installation,monkeypatch,missing):
    import subprocess,sys
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,'core-certificate-parent')
    descriptor=roots['receipts']/('p1-recovery-'+r['id']+'.core-certificate.intent.json')
    output=roots['control']/'core'/next(n for n,k in recovery.archive_names(r).items() if k=='certificate')
    if missing in ('descriptor','both'):descriptor.unlink()
    if missing in ('output','both'):output.unlink()
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute("UPDATE resource_monitor SET state='fault'")
    before=logical(roots);files=retained_bytes(roots)
    script=_KPRE_FRESH.replace('result=session.continue_kernel(owned,fault=fault)',
        "result=recovery._pending_prefix_locked(session.roots,session.db,scope,owned,owned['proof']['observation'],now=accepted['created']+1000,through_journal=True,through_core=True,_session=session)")
    # This direct private query does not own continue_kernel's session-burning
    # wrapper; its refusal must still precede every writer.
    start=script.index('            assert not session._live')
    end=script.index('            return True',start)
    script=script[:start]+script[end:]
    for _ in range(2):
        child=_guarded_fresh_run([sys.executable,'-c',script],input=encode(dict(
            roots={k:str(p) for k,p in roots.items()},scope=scope,attempts=1,
            refuse=True,setup=_KFIFTH_FORBID)),capture_output=True,text=True,timeout=90)
        assert child.returncode==0,child.stderr
        assert logical(roots)==before and retained_bytes(roots)==files


def test_kfifth_production_completed_replay_after_valid_hello(b23_host,installation,monkeypatch):
    from worker.runtime_journal import Journal,read_maintenance
    from worker.runtime_supervisor import RuntimeSupervisor
    roots,scope,_,_=installation
    r=_kpre_attempt(installation,b23_host,monkeypatch,'completion-fence-dirsync')
    _kfifth_fresh(roots,scope)
    supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.journal=Journal(roots['receipts']);supervisor.boot='e'*32
    supervisor.failed=False;supervisor.active=None;supervisor.launches=0
    try:hello=supervisor.operation('hello',{})
    finally:supervisor.journal.lock.close()
    assert hello['maintenance']==read_maintenance(roots['receipts'])
    reg=json.loads(r['before']['source']['data']['p1_registrations'][0][1])
    reg.update(lane=hello['lane'],boot=hello['boot'])
    recovery.register(roots,scope,reg)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute("UPDATE resource_monitor SET state='checking'")
        db.execute('UPDATE resource_accounting SET highwater=highwater+1')
    before=logical(roots);files=retained_bytes(roots)
    # Model the fresh transport's actual current registration union. The
    # accepted intent/proof/hash remains byte-for-byte original. A completed
    # historical read needs no current consumer observation or reconstruction.
    setup=_KFIFTH_OBSERVE+"""
with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
    regs=[json.loads(row[0]) for row in db.execute('SELECT record FROM p1_registrations ORDER BY rowid')]
"""
    _kfifth_fresh(roots,scope,target=r['id'],setup=setup)
    assert logical(roots)==before and retained_bytes(roots)==files


# Owning successor coherence: real nonempty P/K histories, no C source imports.
@pytest.mark.parametrize('independent_setup',range(2))
def test_successor_nonempty_production_composition(b23_host,installation,monkeypatch,independent_setup):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running')
    a=_kpre_predecessor(installation,b23_host,monkeypatch)
    with recovery.offline(roots) as (_,db):
        recovery._verify_core_rows(db,a)
        state=recovery.snapshot(roots,db,scope)
        assert state['inventory']['reasons']==[]
        assert state['inventory']['counts']['recovery_evidence']==3
        original=db.execute('SELECT * FROM dispatches').fetchone()
        assert original[7:]==(None,None,0)
        assert db.execute('SELECT state FROM resource_reservations').fetchone()==('retained',)
        assert state['allocation_coherent']
    evidence=retained_bytes(roots)
    _lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def finish(session,**request):
        try:result=prepare(session,**request)
        except BaseException as exc:
            # Preserve the required positive RED and expose the exact next
            # production boundary, without changing its transport ceiling.
            trace=exc.__traceback__;sizes={}
            while trace:
                if trace.tb_frame.f_code.co_name=='_prepare_abandon_locked':
                    local=trace.tb_frame.f_locals
                    if 'result' in local:
                        sizes={k:len(encode(v).encode()) for k,v in local['result'].items()}
                        sizes['transport']=len(encode(dict(result=local['result'])).encode())
                trace=trace.tb_next
            (roots['control'].parent/'successor-error.txt').write_text(encode(dict(error=str(exc),bytes=sizes)))
            raise
        p,r=result['plan'],result['intent']
        assert recovery.verify_abandon_plan(roots,session.db,scope,r['proof']['observation'],
            p['id'],digest(p),proof_digest=p['proof_digest'])==p
        assert session.continue_kernel(r,prepared=result._snapshot)['status']=='complete'
        return result
    with monkeypatch.context() as m:
        m.setattr(recovery,'_prepare_abandon_locked',finish)
        try:b=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent']
        except ValueError:
            error=roots['control'].parent/'successor-error.txt'
            if error.exists():
                with recovery.offline(roots) as (_,db):
                    recovery._verify_core_rows(db,a)
                    assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
                    assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
                    assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(2,)
                    assert db.execute('SELECT * FROM dispatches').fetchone()==original
                raise AssertionError(error.read_text())
            raise
    with recovery.offline(roots) as (_,db):
        validated=recovery.validate_b2_database(db,roots['control']/'core')
        assert len(validated['p1_recoveries'])==2 and len(validated['p1_recovery_steps'])==10
        assert len(validated['dispatch_resolutions'])==2
        assert db.execute('SELECT * FROM dispatches').fetchone()==original
        assert recovery.snapshot(roots,db,scope)['allocation_coherent']
        for r in (a,b):
            recovery._completed_prefix_receipts(roots,r)
            recovery._verify_finished_receipt_evidence(roots,r)
    for name,raw in evidence.items():
        if 'p1-recovery-'+a['id'] in name:assert retained_bytes(roots)[name]==raw
    _kfifth_fresh(roots,scope,target=a['id'],setup=_KFIFTH_OBSERVE)
    _kfifth_fresh(roots,scope,target=b['id'],setup=_KFIFTH_OBSERVE)

def _t1_next_owner(h,roots,scope,previous,generation):
    reg=copy.deepcopy(h['registration']);reg.update(lane=previous['new_lane'],boot=str(generation)*32)
    for index,owner in enumerate(reg['containers']):
        owner.update(id=format(generation*2+index,'x')*64,started=f'2026-10-17T0{generation+3}:00:0{index}Z')
    recovery.register(roots,scope,reg)
    provenance=copy.deepcopy(h['provenance']);provenance['registration_digest']=digest(reg)
    provenance['hello']={k:reg[k] for k in ('lane','boot')}
    for owner,c in zip(provenance['owners'],reg['containers']):
        owner.update(id=c['id'],started=c['started'],cgroup='/docker/'+c['id'])
    h['module']._host_write(h['base']/('registration-'+digest(reg)+'.json'),provenance)
    for old,c in zip(copy.deepcopy(h['consumers'][:2]),reg['containers']):
        old.update(id=c['id'],started=c['started'],finished=f'2026-10-17T0{generation+4}:00:00Z');h['consumers'].append(old)
    h['state']=h['module'].host_state(roots,scope)
    h['request']['registration_digest']=recovery.host_registration_digest(h['state']['registrations'])


@pytest.mark.parametrize('independent_setup',range(2))
def test_t1_nonempty_third_production_generation(b23_host,installation,monkeypatch,independent_setup,tmp_path):
    """Three production histories, then ONE predeclared D capacity attempt."""
    import hashlib
    from service import resource_store
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running')
    prepare=recovery._prepare_abandon_locked
    capture=recovery._capture_original_snapshot
    scan=resource_store._recovery_scan
    histories=[];metrics=[];captures=[]
    def counted(*a,**k):
        raw=capture(*a,**k);captures.append(hashlib.sha256(raw).hexdigest());return raw
    monkeypatch.setattr(recovery,'_capture_original_snapshot',counted)
    for generation in range(3):
        if histories:_t1_next_owner(b23_host,roots,scope,histories[-1],generation)
        _lotp_transport(b23_host,monkeypatch)
        def finish(session,**request):
            from service.resource_store import _maintenance_footprint,_measure_recovery_roots
            prior_capture=len(captures)
            result=prepare(session,**request);r=result['intent'];owned=result._snapshot
            assert len(captures)==prior_capture+1
            assert captures[-1]==r['expected'][1]['output']['snapshot_digest']
            bound=_maintenance_footprint((roots['control']/'core',roots['staging']),r['highwater'],
                sum(json.loads(row[2])['disk_bytes'] for row in r['before']['source']['data']['resource_reservations']),r['budget_before'])
            maxima={name:0 for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm')}
            observed=[]
            def observe(db,scan_roots,intent):
                if scan_roots[0]==roots['control']/'core':observed.append(_measure_recovery_roots(scan_roots))
                return scan(db,scan_roots,intent)
            def fault(point):
                for name in maxima:
                    p=roots['control']/'core'/name
                    if p.exists():maxima[name]=max(maxima[name],p.stat().st_size)
            with monkeypatch.context() as m:
                m.setattr(resource_store,'_recovery_scan',observe)
                assert session.continue_kernel(r,prepared=owned,fault=fault)['status']=='complete'
            assert len(captures)==prior_capture+1 and len(observed)==1
            recovery._verify_core_rows(session.db,r)
            for old in histories+[r]:
                recovery._completed_prefix_receipts(roots,old)
                recovery._verify_finished_receipt_evidence(roots,old)
                bundle=roots['receipts']/('p1-recovery-'+old['id']+'.snapshot.bundle')
                archive=next(n for n,k in recovery.archive_names(old).items() if k=='snapshot')
                assert bundle.read_bytes()==(roots['control']/'core'/archive).read_bytes()
                assert hashlib.sha256(bundle.read_bytes()).hexdigest()==old['expected'][1]['output']['snapshot_digest']
            used,count=recovery._prefix_receipt_namespace(roots['receipts'],histories+[r])
            certificate=dict(certificate=recovery._certificate(r,dict(recovery_id=r['id'],**r['expected'][3],lane=r['new_lane'],epoch=r['new_epoch'])),
                steps=[dict(recovery_id=r['id'],**e,lane=r['new_lane'],epoch=r['new_epoch']) for e in r['expected'][:4]])
            metrics.append(dict(generation=generation+1,frame=len(recovery._maintenance_result_frame(result)),
                plan=len(encode(result['plan']).encode()),intent=len(encode(r).encode()),snapshot=len(owned._raw),
                manifest=len(encode(recovery.archive_manifest(r)).encode()),receipt_certificate=len(encode(certificate).encode()),
                sqlite_observed_maxima=maxima,receipts_bytes=used,receipts_count=count,bound=bound,
                H=r['highwater'],budget=r['budget_before'],scan=observed[0],actual=_measure_recovery_roots((roots['control']/'core',roots['staging']))))
            (tmp_path/f'generation-{generation}.json').write_text(encode(metrics[-1]))
            return result
        with monkeypatch.context() as m:
            m.setattr(recovery,'_prepare_abandon_locked',finish)
            result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        histories.append(result['intent'])
        metrics.append(json.loads((tmp_path/f'generation-{generation}.json').read_text()))
    with recovery.offline(roots) as (_,db):
        validated=recovery.validate_b2_database(db,roots['control']/'core')
        assert len(validated['p1_recoveries'])==3 and len(validated['p1_recovery_steps'])==15
        assert len(validated['dispatch_resolutions'])==3
        original=db.execute('SELECT * FROM dispatches').fetchone()
        assert original[7:]==(None,None,0)
        assert recovery.snapshot(roots,db,scope)['allocation_coherent']
    # Predeclared bound: at most three further preparations (D/E/F), 30s each.
    capacity=None
    for generation in range(3,6):
        _t1_next_owner(b23_host,roots,scope,histories[-1],generation)
        _lotp_transport(b23_host,monkeypatch)
        before=logical(roots);files=retained_bytes(roots)
        def attempt(session):
            try:result=prepare(session,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
            except (ValueError,OSError,sqlite3.Error) as exc:
                assert not session._live
                return dict(refused=str(exc))
            assert session.continue_kernel(result['intent'],prepared=result._snapshot)['status']=='complete'
            return dict(admitted=result['intent'])
        def prewrite(roots,db,capsule):
            state,_=recovery._preparation_prestate(roots,db,scope,capsule,profiles,DEFAULT_BUDGET,[])
            return encode(state['evidence_projection'])
        outcome=recovery._run_bounded_maintenance(roots,scope,attempt,_prewrite=prewrite)
        if 'admitted' in outcome:
            histories.append(outcome['admitted']);continue
        capacity=dict(generation=generation+1,error=outcome['refused'],metrics=metrics)
        (tmp_path/'capacity.json').write_text(encode(capacity))
        after=logical(roots)
        assert [line for line in after if 'INSERT INTO "p1_abandon_plans"' not in line]==[line for line in before if 'INSERT INTO "p1_abandon_plans"' not in line]
        for name,raw in files.items():
            if Path(name.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
                assert retained_bytes(roots)[name]==raw
        break
    assert capacity is not None,'predeclared capacity witness bound exhausted'
    (tmp_path/'capacity.json').write_text(encode(capacity))
    current_union="""
db=sqlite3.connect(roots['control']/'core'/'core.sqlite')
try:regs=[json.loads(row[0]) for row in db.execute('SELECT record FROM p1_registrations ORDER BY rowid')]
finally:db.close()
"""
    for r in histories:_kfifth_fresh(roots,scope,target=r['id'],setup=_KFIFTH_OBSERVE+current_union)
    print(encode(capacity))


@pytest.mark.parametrize('size',[65536,65537,recovery.MAINTENANCE_RESULT_LIMIT-1,recovery.MAINTENANCE_RESULT_LIMIT,recovery.MAINTENANCE_RESULT_LIMIT+1])
@pytest.mark.parametrize('seed',['é🙂','{"older":"\\\\\\\"é"}'])
def test_t1_exact_utf8_envelope(size,seed):
    # Framing fixtures only: these strings are not preparation authority.
    overhead=len(encode(dict(result=seed)).encode())
    value=seed+'a'*(size-overhead)
    assert len(encode(dict(result=value)).encode())==size
    if size>recovery.MAINTENANCE_RESULT_LIMIT:
        with pytest.raises(ValueError,match='result_budget'):recovery._maintenance_result_frame(value)
    else:
        raw=recovery._maintenance_result_frame(value)
        assert len(raw)==size and raw==encode(dict(result=value)).encode()
        assert recovery._decode_maintenance_result(raw)==value
        with pytest.raises(ValueError):recovery._decode_maintenance_result(raw,preparation=True)


@pytest.mark.parametrize('raw',[
    b'',b'{',b'[]',b'{}',b'{"result":true,"extra":0}',b'{"result":1,"result":1}',
    b'{"result":{"x":1,"x":1}}',b'{"result":NaN}',b'{"result":Infinity}',
    b'{"result":1e999}',b'{"result":"\xff"}',b'{"result": 1}',b'{"result":1}\n',
    b'{"result":1}{"result":1}',b'{"result":"\\u00e9"}',b'{"result":-0}',
    b'{"result":"\\ud800"}',b'{"result":{"z":1,"a":2}}'])
def test_t1_strict_closed_frame(raw):
    with pytest.raises((ValueError,UnicodeError)):recovery._decode_maintenance_result(raw)


def _t1_pipe_model(monkeypatch):
    """Actual fork/pipe, only storage/host session modeled; no limits changed."""
    import sys
    from contextlib import contextmanager
    monkeypatch.setattr(sys,'platform','linux')
    monkeypatch.setattr(recovery,'_install_maintenance_limit',lambda:None)
    @contextmanager
    def session(*a,**k):yield None
    monkeypatch.setattr(recovery,'_maintenance_session',session)
    fork=os.fork;owned=[]
    def tracked():
        pid=fork()
        if pid:owned.append(pid)
        return pid
    monkeypatch.setattr(os,'fork',tracked)
    return owned


@pytest.mark.parametrize('mode',['ordinary','fragment','short','write-EINTR','read-EINTR','EPIPE','zero-write',
    'empty','partial','extra','second','nonzero','signal','post-EOF-alive','never-EOF','slow','backpressure','exact-T-alive'])
def test_t1_actual_pipe_transport(monkeypatch,mode):
    import resource,signal,time,select
    from types import SimpleNamespace
    prior=resource.getrlimit(resource.RLIMIT_FSIZE)
    owned=_t1_pipe_model(monkeypatch)
    write=os.write;read=os.read;waitpid=os.waitpid
    original_time=time.monotonic;original_select=select.select
    # Deadline behavior uses the same absolute 30 units, accelerated 100x in
    # this small process test; select scales the remaining duration consistently.
    start=original_time()
    monkeypatch.setattr(recovery,'time',SimpleNamespace(monotonic=lambda:start+(original_time()-start)*100,time=time.time))
    def scaled_select(r,w,x,timeout):return original_select(r,w,x,timeout/100)
    monkeypatch.setattr(select,'select',scaled_select)
    state={'write':0,'read':0}
    def changed_write(fd,raw):
        state['write']+=1
        if mode=='write-EINTR' and state['write']==1:raise InterruptedError()
        if mode=='EPIPE':raise BrokenPipeError()
        if mode=='zero-write':return 0
        if mode in ('fragment','short'):return write(fd,raw[:7])
        if mode in ('nonzero','signal','post-EOF-alive','never-EOF','slow','backpressure','exact-T-alive'):
            if mode=='slow':
                write(fd,raw[:1]);time.sleep(1);os._exit(74)
            if mode=='backpressure':
                while True:write(fd,b'x'*65536)
            # Write a complete valid frame, then exercise EOF/exit separation.
            offset=0
            while offset<len(raw):
                try:offset+=write(fd,raw[offset:offset+4096])
                except BlockingIOError:original_select([],[fd],[],0.01)
            if mode=='nonzero':os._exit(74)
            if mode=='signal':os.kill(os.getpid(),signal.SIGKILL)
            if mode in ('post-EOF-alive','exact-T-alive'):os.close(fd)
            time.sleep(1);os._exit(74)
        if mode=='empty':os.close(fd);os._exit(0)
        if mode=='partial':write(fd,b'{');os.close(fd);os._exit(0)
        if mode in ('extra','second'):
            write(fd,raw);write(fd,b' ' if mode=='extra' else raw);os.close(fd);os._exit(0)
        return write(fd,raw)
    def changed_read(fd,count):
        assert count<=65536
        state['read']+=1
        if mode=='read-EINTR' and state['read']==1:raise InterruptedError()
        return read(fd,count)
    monkeypatch.setattr(os,'write',changed_write);monkeypatch.setattr(os,'read',changed_read)
    value='a'*(recovery.MAINTENANCE_RESULT_LIMIT-len(encode(dict(result='')).encode())) if mode=='exact-T-alive' else 'é🙂'
    began=original_time()
    if mode in ('ordinary','fragment','short','write-EINTR','read-EINTR'):
        assert recovery._run_bounded_maintenance({}, {},lambda session:value)==value
    else:
        with pytest.raises((ValueError,OSError)):recovery._run_bounded_maintenance({}, {},lambda session:value)
    assert original_time()-began<2
    assert resource.getrlimit(resource.RLIMIT_FSIZE)==prior
    assert len(owned)==1
    with pytest.raises(ChildProcessError):waitpid(owned[0],os.WNOHANG)


def test_t1_fork_failure_closes_descriptors(monkeypatch):
    _t1_pipe_model(monkeypatch)
    pipe=os.pipe;ends=[]
    def tracked():
        result=pipe();ends.extend(result);return result
    monkeypatch.setattr(os,'pipe',tracked)
    def failed():raise OSError('fork unavailable')
    monkeypatch.setattr(os,'fork',failed)
    with pytest.raises(OSError):recovery._run_bounded_maintenance({}, {},lambda s:True)
    for fd in ends:
        with pytest.raises(OSError):os.fstat(fd)


@pytest.mark.parametrize('damage',[
    'resolution-missing','resolution-wrong','resolution-extra','resolution-pending',
    'dispatch-authority','dispatch-binding','dispatch-runtime','dispatch-phase','dispatch-receipt','dispatch-outcome','dispatch-ACK',
    'job-principal','job-context','job-bundles','job-state','job-attempt','job-generation','job-deadline','job-reap',
    'reservation-category','reservation-configuration','reservation-demand','reservation-state',
    'history','archive','descriptor','media','typed-reference'])
def test_t1_successor_strict_refusal_twice(b23_host,installation,monkeypatch,damage):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running')
    a=_kpre_predecessor(installation,b23_host,monkeypatch)
    _lotp_transport(b23_host,monkeypatch)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        if damage=='resolution-missing':db.execute('DELETE FROM dispatch_resolutions')
        elif damage=='resolution-wrong':db.execute("UPDATE dispatch_resolutions SET digest=?",('0'*64,))
        elif damage=='resolution-extra':db.execute('INSERT INTO dispatch_resolutions VALUES(?,?,?,?)',(999,a['id'],'{}','0'*64))
        elif damage=='resolution-pending':db.execute("DELETE FROM p1_recovery_steps WHERE step='core-complete'")
        elif damage.startswith('dispatch-'):
            column={'authority':'authority','binding':'binding','runtime':'runtime','phase':'phase','receipt':'receipt','outcome':'outcome','ACK':'acked'}[damage[9:]]
            value={'authority':'[]','binding':'0'*64,'runtime':'changed','phase':'changed','receipt':'{}','outcome':'{}','acked':1}[column]
            db.execute('UPDATE dispatches SET '+column+'=?',(value,))
        elif damage.startswith('job-'):
            if damage=='job-principal':db.execute("UPDATE jobs SET principal='changed'")
            elif damage in ('job-context','job-bundles'):
                value=json.loads(db.execute('SELECT context FROM jobs').fetchone()[0])
                if damage=='job-context':value['plan']['changed']=True
                else:value['bundles']=[]
                db.execute('UPDATE jobs SET context=?',(encode(value),))
            else:
                value=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0])
                if damage=='job-state':value['state']='running'
                elif damage=='job-attempt':value['attempts'][-1]['id']='0'*32
                elif damage=='job-generation':value['generation']+=1
                elif damage=='job-deadline':value['deadline']='2099-01-01T00:00:00Z'
                elif damage=='job-reap':value['attempts'][-1]['reaped']=False
                db.execute('UPDATE jobs SET job=?',(encode(value),))
        elif damage.startswith('reservation-'):
            if damage=='reservation-category':db.execute("UPDATE resource_reservations SET category='changed'")
            elif damage=='reservation-state':db.execute("UPDATE resource_reservations SET state='executing'")
            else:
                value=json.loads(db.execute('SELECT demand FROM resource_reservations').fetchone()[0])
                if damage=='reservation-configuration':value['configuration']='0'*64
                else:value['disk_bytes']+=1
                db.execute('UPDATE resource_reservations SET demand=?',(encode(value),))
        elif damage=='history':db.execute("UPDATE events SET state='changed'")
        elif damage=='typed-reference':
            value=json.loads(db.execute('SELECT authority FROM dispatches').fetchone()[0]);value[0]['artifact_ids']=['new-reference']
            db.execute('UPDATE dispatches SET authority=?',(encode(value),))
    if damage=='archive':next((roots['control']/'core').glob('*.manifest.json')).write_bytes(b'{}')
    elif damage=='descriptor':next(roots['receipts'].glob('*.core-certificate.intent.json')).write_bytes(b'{}')
    elif damage=='media':(roots['staging']/'new-defect').write_bytes(b'new')
    before=logical(roots);files=retained_bytes(roots)
    def forbidden(*a,**k):raise AssertionError('strict refusal reached writer')
    with monkeypatch.context() as m:
        for name in ('_prefix_file','_prefix_transaction','_pending_prefix_locked','_rehearse_preparation'):
            m.setattr(recovery,name,forbidden)
        for _ in range(2):
            with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
            assert logical(roots)==before
            for name,raw in files.items():
                if Path(name.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
                    assert retained_bytes(roots)[name]==raw


@pytest.mark.parametrize('same_consumer',[False,True])
def test_t1_preparation_budget_before_rehearsal_or_acceptance(b23_host,installation,monkeypatch,same_consumer,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    chunks=recovery._maintenance_result_chunks
    # Isolated encoder-budget fault, no padded field admitted as real history.
    def overflow(result):
        yield from chunks(result)
        yield b'x'*recovery.MAINTENANCE_RESULT_LIMIT
    monkeypatch.setattr(recovery,'_maintenance_result_chunks',overflow)
    def forbidden(*a,**k):raise AssertionError('over-budget preparation reached rehearsal/acceptance')
    for name in ('_rehearse_preparation','_pending_prefix_locked','_prefix_file'):
        monkeypatch.setattr(recovery,name,forbidden)
    transaction=recovery._prefix_transaction
    def preparation_only(db,operation,fault,label):
        assert label in ('preparation-schema','preparation-plan')
        return transaction(db,operation,fault,label)
    monkeypatch.setattr(recovery,'_prefix_transaction',preparation_only)
    prepare=recovery._prepare_abandon_locked
    def consume(session,**request):
        try:
            prepared=prepare(session,**request)
            forbidden(prepared)
        except ValueError as exc:
            assert str(exc)=='b2_maintenance_result_budget' and not session._live
            (tmp_path/'burned').write_text(str(exc));raise
    if same_consumer:monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    before=logical(roots);files=retained_bytes(roots)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    if same_consumer:assert (tmp_path/'burned').read_text()=='b2_maintenance_result_budget'
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(1,)
        for table in ('p1_recoveries','p1_recovery_steps','dispatch_resolutions'):
            assert db.execute('SELECT count(*) FROM '+table).fetchone()==(0,)
        assert db.execute('SELECT * FROM resource_accounting').fetchall()==[(1,20*1024**2)]
    assert not list(roots['receipts'].glob('p1-recovery-*'))


@pytest.mark.parametrize('damage',['missing','extra','plan-type','intent-type','plan-digest','intent-digest','plan-binding'])
def test_t1_preparation_semantics_both_codec_ends(b23_host,installation,monkeypatch,damage):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    if damage=='missing':result.pop('plan')
    elif damage=='extra':result['padding']='no'
    elif damage=='plan-type':result['plan']=[]
    elif damage=='intent-type':result['intent']=[]
    elif damage=='plan-digest':result['plan_digest']='0'*64
    elif damage=='intent-digest':result['intent_digest']='0'*64
    elif damage=='plan-binding':
        result['plan']['new_epoch']='0'*32;result['plan_digest']=digest(result['plan'])
    with pytest.raises((ValueError,TypeError,KeyError)):recovery._validate_preparation_result(result)
    raw=recovery._maintenance_result_frame(result)
    with pytest.raises((ValueError,TypeError,KeyError)):recovery._decode_maintenance_result(raw,preparation=True)


@pytest.mark.parametrize('size',[65536,65537,1048575,1048576])
@pytest.mark.parametrize('seed',['é🙂','{"old":"\\\\\\\"é"}'])
def test_t1_successful_exact_size_pipe_waitpid_eintr(monkeypatch,size,seed):
    import resource
    owned=_t1_pipe_model(monkeypatch)
    prior=resource.getrlimit(resource.RLIMIT_FSIZE)
    waitpid=os.waitpid;calls=[]
    def interrupted(pid,flags):
        calls.append(flags)
        if len(calls)==1:raise InterruptedError()
        return waitpid(pid,flags)
    monkeypatch.setattr(os,'waitpid',interrupted)
    value=seed+'a'*(size-len(encode(dict(result=seed)).encode()))
    assert len(encode(dict(result=value)).encode())==size
    assert recovery._run_bounded_maintenance({}, {},lambda s:value)==value
    assert calls[0]==os.WNOHANG and resource.getrlimit(resource.RLIMIT_FSIZE)==prior
    with pytest.raises(ChildProcessError):waitpid(owned[0],os.WNOHANG)


def test_t1_real_blocked_pipe_backpressure_deadline(monkeypatch,tmp_path):
    import select,time,resource
    from types import SimpleNamespace
    prior=resource.getrlimit(resource.RLIMIT_FSIZE)
    owned=_t1_pipe_model(monkeypatch)
    clock=time.monotonic;start=clock();select0=select.select;write=os.write
    marker=tmp_path/'actual-EAGAIN'
    monkeypatch.setattr(recovery,'time',SimpleNamespace(monotonic=lambda:start+(clock()-start)*100,time=time.time))
    def select_blocked(r,w,x,timeout):
        if r:
            # Deliberately leave the real pipe unread until the child exhausts
            # kernel capacity. The existing absolute deadline must still win.
            time.sleep(timeout/100);return ([],[],[])
        return select0(r,w,x,timeout/100)
    def observed(fd,raw):
        try:return write(fd,raw)
        except BlockingIOError:
            marker.write_text('kernel pipe full');raise
    # Child select normally prevents EAGAIN; force readiness only to observe
    # actual nonblocking backpressure, then let its absolute deadline expire.
    def ready(r,w,x,timeout):
        if w:return ([],w,[])
        return select_blocked(r,w,x,timeout)
    monkeypatch.setattr(select,'select',ready);monkeypatch.setattr(os,'write',observed)
    with pytest.raises(ValueError,match='timeout'):
        recovery._run_bounded_maintenance({}, {},lambda s:'a'*900000)
    assert marker.read_text()=='kernel pipe full'
    assert clock()-start<2 and resource.getrlimit(resource.RLIMIT_FSIZE)==prior
    with pytest.raises(ChildProcessError):os.waitpid(owned[0],os.WNOHANG)


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_isolated_canonical_preparation_boundary(b23_host,installation,monkeypatch,same_consumer,delta,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    chunks=recovery._maintenance_result_chunks;prepare=recovery._prepare_abandon_locked
    size=recovery.MAINTENANCE_RESULT_LIMIT+delta
    def boundary(result):
        # Canonical UTF-8 framing fixture only, after real semantic validation.
        # It is never returned as preparation authority or stored in history.
        if type(result) is recovery._Preparation:
            assert sum(map(len,chunks(result)))<size
            yield from chunks('é'+'a'*(size-15))
        else:yield from chunks(result)
    # 13-byte outer string frame + 2-byte é.
    monkeypatch.setattr(recovery,'_maintenance_result_chunks',boundary)
    def consume(session,**request):
        try:r=prepare(session,**request)
        except ValueError:
            assert delta==1 and not session._live
            (tmp_path/'burned').write_text('yes');raise
        assert delta<=0
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return dict(r)
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    if delta==1:
        def forbidden(*a,**k):raise AssertionError('over-budget reached writer')
        for name in ('_rehearse_preparation','_prefix_file','_pending_prefix_locked'):
            monkeypatch.setattr(recovery,name,forbidden)
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert (tmp_path/'burned').exists()
    else:
        r=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        recovery._validate_preparation_result(r)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(int(same_consumer and delta<=0),)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5*int(same_consumer and delta<=0),)


def _t1_completed_nonempty(installation,h,monkeypatch,generations=2):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running')
    prepare=recovery._prepare_abandon_locked;histories=[]
    for generation in range(generations):
        if histories:_t1_next_owner(h,roots,scope,histories[-1],generation)
        _lotp_transport(h,monkeypatch)
        def finish(session,**request):
            r=prepare(session,**request)
            assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
            return r
        with monkeypatch.context() as m:
            m.setattr(recovery,'_prepare_abandon_locked',finish)
            histories.append(recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])['intent'])
    return histories


_T1_CURRENT_UNION="""
db=sqlite3.connect(roots['control']/'core'/'core.sqlite')
try:regs=[json.loads(row[0]) for row in db.execute('SELECT record FROM p1_registrations ORDER BY rowid')]
finally:db.close()
"""


@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('point',['accepted-parent','fence-replace-parent','bundle-parent','core-intent-commit',
    'pending-receipt-commit','snapshot-parent','manifest-parent','original-journal-parent',
    'evidence-receipt-commit','journal-replace','journal-receipt-commit',*_KFIFTH_STAGES])
def test_t1_nonempty_B_C_actual_prefix_history(b23_host,installation,monkeypatch,generation,point):
    roots,scope,_,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,generation)
    _t1_next_owner(b23_host,roots,scope,histories[-1],generation)
    later=_kfifth_successor(installation,b23_host,monkeypatch,point)
    quarantine=point.endswith('intent-recorded')
    before=logical(roots);files=retained_bytes(roots)
    for old in histories:
        _kfifth_fresh(roots,scope,target=old['id'],refuse=quarantine,setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
        assert logical(roots)==before and retained_bytes(roots)==files
    _kfifth_fresh(roots,scope,target=later['id'],refuse=quarantine,
        setup=(_KFIFTH_FORBID if quarantine else "recovery.time.time=lambda:accepted['created']+1000")+_T1_CURRENT_UNION)
    if quarantine:assert logical(roots)==before and retained_bytes(roots)==files
    else:
        before=logical(roots);files=retained_bytes(roots)
        for old in histories+[later]:_kfifth_fresh(roots,scope,target=old['id'],setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
        assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('kind',['bundle','snapshot','manifest','original-journal','rotation',
    'core-certificate','receipt-certificate','completion-journal','completion-fence'])
@pytest.mark.parametrize('damage',['missing','wrong','mode'])
def test_t1_every_ABC_descriptor_all_directions(b23_host,installation,monkeypatch,kind,damage):
    roots,scope,_,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,3)
    for victim in histories:
        p=roots['receipts']/('p1-recovery-'+victim['id']+'.'+kind+'.intent.json')
        raw=p.read_bytes();mode=p.stat().st_mode & 0o777
        if damage=='missing':p.unlink()
        elif damage=='wrong':p.write_bytes(b'{}')
        else:p.chmod(0o644)
        before=logical(roots);files=retained_bytes(roots)
        for target in histories:
            _kfifth_fresh(roots,scope,target=target['id'],refuse=True,setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
            assert logical(roots)==before and retained_bytes(roots)==files
        # Restore only the deliberate test corruption to probe another history;
        # no production recovery or authority repair is invoked.
        p.write_bytes(raw);p.chmod(mode)
    before=logical(roots);files=retained_bytes(roots)
    for r in histories:_kfifth_fresh(roots,scope,target=r['id'],setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
    assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('point',['core-complete-commit','core-certificate-parent','receipt-certificate-parent',
    'completion-journal-dirsync','completion-fence-dirsync'])
@pytest.mark.parametrize('damage',['after-image','unaffected-source'])
def test_t1_nonempty_C_current_source_before_any_writer(b23_host,installation,monkeypatch,point,damage):
    roots,scope,_,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,2)
    _t1_next_owner(b23_host,roots,scope,histories[-1],2)
    c=_kfifth_successor(installation,b23_host,monkeypatch,point)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        if damage=='after-image':db.execute("UPDATE resource_monitor SET state='fault'")
        else:db.execute('UPDATE sqlite_sequence SET seq=seq+1')
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,target=c['id'],refuse=True,setup=_KFIFTH_FORBID+_T1_CURRENT_UNION)
    assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('damage',['new-fault','new-integrity','incomplete','old-resolution-malformed-receipt'])
def test_t1_additional_successor_refusals(b23_host,installation,monkeypatch,damage):
    roots,scope,profiles,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,2)
    _t1_next_owner(b23_host,roots,scope,histories[-1],2);_lotp_transport(b23_host,monkeypatch)
    if damage=='new-fault':
        # Resource faults are intentionally recoverable by a FRESH explicit
        # source/plan. The strict negative is a fault added AFTER C acceptance:
        # old A/B resolutions cannot consume this current-source mutation.
        c=_kfifth_successor(installation,b23_host,monkeypatch,'core-certificate-parent')
        with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
            db.execute("INSERT INTO resource_faults VALUES('new-unbound-fault')")
        before=logical(roots);files=retained_bytes(roots)
        _kfifth_fresh(roots,scope,target=c['id'],refuse=True,setup=_KFIFTH_FORBID+_T1_CURRENT_UNION)
        assert logical(roots)==before and retained_bytes(roots)==files
        return
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        if damage=='new-integrity':db.execute("UPDATE p1_integrity SET revision=2")
        elif damage=='incomplete':db.execute("DELETE FROM p1_recovery_steps WHERE recovery_id=? AND step='receipt-complete'",(histories[-1]['id'],))
        else:db.execute("UPDATE dispatches SET receipt='{}'")
    before=logical(roots);files=retained_bytes(roots)
    trap=roots['control'].parent/'unexpected-writer'
    def forbidden(*a,**k):
        trap.write_text('new defect reached writer');raise AssertionError('new defect reached writer')
    with monkeypatch.context() as m:
        for name in ('_prefix_file','_prefix_transaction','_pending_prefix_locked','_rehearse_preparation'):m.setattr(recovery,name,forbidden)
        for _ in range(2):
            with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
            assert not trap.exists(),trap.read_text() if trap.exists() else ''
            assert logical(roots)==before
            assert {n:v for n,v in retained_bytes(roots).items() if Path(n.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm')}=={n:v for n,v in files.items() if Path(n.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm')}


@pytest.mark.parametrize('damage',['valid','missing','malformed'])
def test_t1_genuine_terminal_receipt_control(b23_host,installation,monkeypatch,damage):
    roots,scope,profiles,_=installation
    job=terminal_enrolled(installation)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
        if damage!='valid':db.execute('UPDATE dispatches SET receipt=?',(None if damage=='missing' else '{}',))
    _lotp_transport(b23_host,monkeypatch)
    if damage=='valid':
        r=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert r['intent']['actions'][0]['job_id']==job and r['intent']['actions'][0]['action']=='retain-terminal'
        return
    before=logical(roots);files=retained_bytes(roots)
    trap=roots['control'].parent/'unexpected-terminal-writer'
    def forbidden(*a,**k):trap.write_text('write');raise AssertionError('terminal mismatch reached writer')
    for name in ('_prefix_file','_prefix_transaction','_pending_prefix_locked','_rehearse_preparation'):monkeypatch.setattr(recovery,name,forbidden)
    for _ in range(2):
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert not trap.exists() and logical(roots)==before and retained_bytes(roots)==files


def _t1_assert_unaccepted(roots,plans):
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'p1_abandon_plans' in tables:
            assert db.execute('SELECT count(*) FROM p1_abandon_plans').fetchone()==(plans,)
            for t in ('p1_recoveries','p1_recovery_steps','dispatch_resolutions'):
                assert db.execute('SELECT count(*) FROM '+t).fetchone()==(0,)
        else:assert plans==0
        assert db.execute('SELECT highwater FROM resource_accounting').fetchone()==(20*1024**2,)
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
    assert not list(roots['receipts'].glob('p1-recovery-*'))


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('pages',[255,256,257])
def test_t1_actual_sqlite_snapshot_capacity(b23_host,installation,monkeypatch,same_consumer,pages,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    capture=recovery._capture_original_snapshot;prepare=recovery._prepare_abandon_locked
    target=pages*4096
    def grown(roots,db,maximum):
        assert maximum==1048576
        # Valid SQLite freelist retained in the real original DB. Insert/delete
        # leaves the complete logical population unchanged; backup is real.
        before=list(db.iterdump());size=4088
        while db.execute('PRAGMA page_count').fetchone()[0]<pages:
            db.execute('INSERT INTO p1_registrations VALUES(?,?,?)',('capacity-only','x'*size,'x'))
            db.execute("DELETE FROM p1_registrations WHERE id='capacity-only'")
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            size+=4088
        assert db.execute('PRAGMA page_count').fetchone()[0]==pages
        assert list(db.iterdump())==before
        try:raw=capture(roots,db,maximum)
        except ValueError:
            assert pages==257;(tmp_path/'capacity').write_text('snapshot');raise
        assert len(raw)==target
        (tmp_path/'capacity').write_text(str(len(raw)));return raw
    monkeypatch.setattr(recovery,'_capture_original_snapshot',grown)
    def consume(session,**request):
        try:r=prepare(session,**request)
        except ValueError:
            assert not session._live;raise
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    if pages==257:
        def forbidden(*a,**k):raise AssertionError('snapshot overflow reached acceptance')
        for name in ('_prefix_file','_pending_prefix_locked','_rehearse_preparation'):monkeypatch.setattr(recovery,name,forbidden)
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        _t1_assert_unaccepted(roots,1)
    else:
        r=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        recovery._validate_preparation_result(r)
    assert (tmp_path/'capacity').exists()


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('size',[32489,32490,32768])
def test_t1_actual_journal_document_capacity(b23_host,installation,monkeypatch,same_consumer,size,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    # Actual unreadable journal is retained byte-for-byte in the closed manifest;
    # it is permitted degraded evidence, not arbitrary caller history padding.
    (roots['receipts']/'journal.json').write_bytes(b'x'*size)
    prepare=recovery._prepare_abandon_locked
    def consume(session,**request):
        try:r=prepare(session,**request)
        except ValueError as e:
            assert not session._live;(tmp_path/'error').write_text(str(e));raise
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    # A closed manifest has 557 fixed UTF-8 bytes plus the even-length hex
    # journal. Thus 65535/65537 are the adjacent attainable byte lengths;
    # equality at 65536 is structurally unavailable for this document shape.
    if size==32489:
        r=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert len(encode(recovery.archive_manifest(r['intent'])).encode())==65535
        return
    trap=tmp_path/'unexpected-writer'
    def forbidden(*a,**k):trap.write_text('acceptance');raise AssertionError('large document reached acceptance')
    for name in ('_prefix_file','_pending_prefix_locked'):monkeypatch.setattr(recovery,name,forbidden)
    with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert not trap.exists() and (tmp_path/'error').read_text()=='b2_maintenance_document'
    _t1_assert_unaccepted(roots,1)


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('failure',['file-known','file-copy','rehearsal-sql','entries','receipt-unknown','free'])
def test_t1_actual_capacity_refusal_both_routes(b23_host,installation,monkeypatch,same_consumer,failure,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked;root=roots['control']/'core'
    if failure=='file-known':
        with (root/'core.sqlite-wal').open('wb') as stream:stream.truncate(4194305)
    elif failure=='entries':
        for i in range(2049):(roots['staging']/('entry-'+str(i))).touch()
    elif failure=='receipt-unknown':
        for i in range(2049):(roots['receipts']/('entry-'+str(i))).touch()
    elif failure=='free':
        monkeypatch.setattr(recovery.os,'statvfs',lambda p:type('Free',(),dict(f_bavail=16777216+2097152-1,f_frsize=1))())
    elif failure=='file-copy':
        capture=recovery._capture_original_snapshot
        def after_capture(*a):
            raw=capture(*a)
            # Real oversized physical sidecar, encountered independently by
            # physical-copy admission; normalized snapshot remains unchanged.
            with (root/'core.sqlite-shm').open('r+b') as stream:stream.truncate(4194305)
            return raw
        monkeypatch.setattr(recovery,'_capture_original_snapshot',after_capture)
    else:
        writer=recovery._write_core_completion
        def full(db,path,intent):
            # Actual SQLite max_page_count at the private prospective write.
            db.execute('PRAGMA max_page_count=1')
            db.execute('INSERT INTO p1_registrations VALUES(?,?,?)',('capacity','x'*4194304,'x'))
            return writer(db,path,intent)
        monkeypatch.setattr(recovery,'_write_core_completion',full)
    def consume(session,**request):
        try:r=prepare(session,**request)
        except (ValueError,OSError,sqlite3.Error):
            assert not session._live;(tmp_path/'burned').write_text('yes');raise
        raise AssertionError('capacity failure returned proposal')
    if same_consumer:monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    trap=tmp_path/'unexpected-capacity-writer'
    def forbidden(*a,**k):trap.write_text('write');raise AssertionError('capacity refusal reached original receipt/continuation')
    for name in ('_prefix_file','_pending_prefix_locked'):monkeypatch.setattr(recovery,name,forbidden)
    # Traps permit only preparation schema/immutable plan and private rehearsal
    # transactions; original accepted authority must never be inserted.
    with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    assert not trap.exists()
    if failure=='file-known':
        # Deliberately corrupt WAL cannot be opened for an SQL comparison.
        assert not list(roots['receipts'].glob('p1-recovery-*'))
    else:_t1_assert_unaccepted(roots,int(failure in ('file-copy','rehearsal-sql')))
    if same_consumer and failure in ('file-copy','rehearsal-sql'):assert (tmp_path/'burned').exists()


@pytest.mark.parametrize('limit',[65536,1048576,4194304])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_real_private_file_inclusive_bound(tmp_path,limit,delta):
    path=tmp_path/'bounded';path.write_bytes(b'x'*(limit+delta));path.chmod(0o600)
    if delta>0:
        with pytest.raises(ValueError):recovery.read_file(path,limit)
    else:assert len(recovery.read_file(path,limit))==limit+delta


@pytest.mark.parametrize('dimension',['entries','bytes'])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_actual_receipt_namespace_inclusive_bound(tmp_path,dimension,delta):
    # Namespace guard fixture only: no accepted history is fabricated. Every
    # supplied name is explicitly retained; production semantic checks remain.
    count=2048+delta if dimension=='entries' else 1
    names=[]
    for i in range(count):
        name='retained-'+str(i);names.append(name);p=tmp_path/name
        p.write_bytes(b'x'*(16777216+delta) if dimension=='bytes' else b'');p.chmod(0o600)
    if delta>0:
        with pytest.raises(ValueError):recovery._prefix_receipt_namespace(tmp_path,[],retained_names=names)
    else:
        used,n=recovery._prefix_receipt_namespace(tmp_path,[],retained_names=names)
        assert n==count and used==(16777216+delta if dimension=='bytes' else 0)


@pytest.mark.parametrize('damage',[
    'resolution-missing','resolution-wrong','resolution-extra','resolution-pending',
    'dispatch-authority','dispatch-binding','dispatch-runtime','dispatch-phase','dispatch-receipt','dispatch-outcome','dispatch-ACK',
    'job-principal','job-context','job-bundles','job-state','job-attempt','job-generation','job-deadline','job-reap',
    'reservation-category','reservation-configuration','reservation-demand','reservation-state',
    'history','archive','descriptor','media','typed-reference'])
def test_t1_repeated_successor_strict_writers_observed(b23_host,installation,monkeypatch,damage):
    roots,scope,profiles,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,2)
    a=histories[-1]
    _t1_next_owner(b23_host,roots,scope,a,2)
    _lotp_transport(b23_host,monkeypatch)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        if damage=='resolution-missing':db.execute('DELETE FROM dispatch_resolutions')
        elif damage=='resolution-wrong':db.execute("UPDATE dispatch_resolutions SET digest=?",('0'*64,))
        elif damage=='resolution-extra':db.execute('INSERT INTO dispatch_resolutions VALUES(?,?,?,?)',(999,a['id'],'{}','0'*64))
        elif damage=='resolution-pending':db.execute("DELETE FROM p1_recovery_steps WHERE step='core-complete'")
        elif damage.startswith('dispatch-'):
            column={'authority':'authority','binding':'binding','runtime':'runtime','phase':'phase','receipt':'receipt','outcome':'outcome','ACK':'acked'}[damage[9:]]
            value={'authority':'[]','binding':'0'*64,'runtime':'changed','phase':'changed','receipt':'{}','outcome':'{}','acked':1}[column]
            db.execute('UPDATE dispatches SET '+column+'=?',(value,))
        elif damage.startswith('job-'):
            if damage=='job-principal':db.execute("UPDATE jobs SET principal='changed'")
            elif damage in ('job-context','job-bundles'):
                value=json.loads(db.execute('SELECT context FROM jobs').fetchone()[0])
                if damage=='job-context':value['plan']['changed']=True
                else:value['bundles']=[]
                db.execute('UPDATE jobs SET context=?',(encode(value),))
            else:
                value=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0])
                if damage=='job-state':value['state']='running'
                elif damage=='job-attempt':value['attempts'][-1]['id']='0'*32
                elif damage=='job-generation':value['generation']+=1
                elif damage=='job-deadline':value['deadline']='2099-01-01T00:00:00Z'
                elif damage=='job-reap':value['attempts'][-1]['reaped']=False
                db.execute('UPDATE jobs SET job=?',(encode(value),))
        elif damage.startswith('reservation-'):
            if damage=='reservation-category':db.execute("UPDATE resource_reservations SET category='changed'")
            elif damage=='reservation-state':db.execute("UPDATE resource_reservations SET state='executing'")
            else:
                value=json.loads(db.execute('SELECT demand FROM resource_reservations').fetchone()[0])
                if damage=='reservation-configuration':value['configuration']='0'*64
                else:value['disk_bytes']+=1
                db.execute('UPDATE resource_reservations SET demand=?',(encode(value),))
        elif damage=='history':db.execute("UPDATE events SET state='changed'")
        elif damage=='typed-reference':
            value=json.loads(db.execute('SELECT authority FROM dispatches').fetchone()[0]);value[0]['artifact_ids']=['new-reference']
            db.execute('UPDATE dispatches SET authority=?',(encode(value),))
    if damage=='archive':next((roots['control']/'core').glob('*.manifest.json')).write_bytes(b'{}')
    elif damage=='descriptor':next(roots['receipts'].glob('*.core-certificate.intent.json')).write_bytes(b'{}')
    elif damage=='media':(roots['staging']/'new-defect').write_bytes(b'new')
    before=logical(roots);files=retained_bytes(roots)
    trap=roots['control'].parent/'unexpected-writer'
    def forbidden(*a,**k):
        trap.write_text('strict refusal reached writer');raise AssertionError('strict refusal reached writer')
    with monkeypatch.context() as m:
        for name in ('_prefix_file','_prefix_transaction','_pending_prefix_locked','_rehearse_preparation'):
            m.setattr(recovery,name,forbidden)
        for _ in range(2):
            with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
            assert not trap.exists()
            assert logical(roots)==before
            for name,raw in files.items():
                if Path(name.split(':')[0]).name not in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
                    assert retained_bytes(roots)[name]==raw



@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('population',[1,16])
def test_t1_actual_population_certificate_capacity(b23_host,installation,monkeypatch,same_consumer,population,tmp_path):
    from service.admission import Admission
    from service.dispatch_store import DispatchStore
    from tests.service.test_admission import fixture
    roots,scope,profiles,_=installation
    core=Core(roots['control']/'core')
    try:
        f=fixture();admission=Admission(core,roots['staging']);store=DispatchStore(core,admission)
        for i in range(population):
            core.submit_synthetic('p','capacity-job-'+str(i),f['context'],_admission=admission,
                _profiles=f['profiles'],_dispatch_settings=('silence',8))
        store.allocate(f['profiles'],'silence',8)
        with core.transaction():
            core.db.execute('UPDATE resource_accounting SET highwater=?',(20*1024**2,))
            core.db.execute("UPDATE resource_monitor SET state='checking'")
            core.db.execute("UPDATE p1_integrity SET state='checking',token='inherited-A2'")
    finally:core.close()
    _lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked;rehearse=recovery._rehearse_preparation
    def observed(session,plan,intent,frozen):
        chain=[dict(recovery_id=intent['id'],**e,lane=intent['new_lane'],epoch=intent['new_epoch']) for e in intent['expected'][:4]]
        cert=recovery._certificate(intent,chain[-1])
        sizes=dict(manifest=len(encode(recovery.archive_manifest(intent)).encode()),
            certificate=len(encode(cert).encode()),receipt=len(encode(dict(certificate=cert,steps=chain)).encode()))
        (tmp_path/'documents.json').write_text(encode(sizes))
        return rehearse(session,plan,intent,frozen)
    monkeypatch.setattr(recovery,'_rehearse_preparation',observed)
    trap=tmp_path/'unexpected-writer'
    def forbidden(*a,**k):trap.write_text('acceptance');raise AssertionError('capacity reached acceptance')
    def consume(session,**request):
        try:r=prepare(session,**request)
        except ValueError as e:
            assert not session._live;(tmp_path/'error').write_text(str(e));raise
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    if population==16:
        for name in ('_prefix_file','_pending_prefix_locked'):monkeypatch.setattr(recovery,name,forbidden)
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert not trap.exists();_t1_assert_unaccepted(roots,1)
        assert (tmp_path/'error').read_text()=='b2_maintenance_document'
        assert max(json.loads((tmp_path/'documents.json').read_text()).values())>65536
    else:
        r=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert max(json.loads((tmp_path/'documents.json').read_text()).values())<65536
        recovery._validate_preparation_result(r)


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_preparation_free_reserve_boundary(b23_host,installation,monkeypatch,same_consumer,delta,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    monkeypatch.setattr(recovery.os,'statvfs',lambda p:type('Free',(),dict(f_bavail=16777216+2097152+delta,f_frsize=1))())
    prepare=recovery._prepare_abandon_locked
    def consume(session,**request):
        r=prepare(session,**request)
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    if delta<0:
        trap=tmp_path/'unexpected-writer'
        def forbidden(*a,**k):trap.write_text('write');raise AssertionError('free reserve reached writer')
        for name in ('_prefix_file','_prefix_transaction','_pending_prefix_locked'):monkeypatch.setattr(recovery,name,forbidden)
        before=logical(roots);files=retained_bytes(roots)
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert not trap.exists() and logical(roots)==before and retained_bytes(roots)==files
        _t1_assert_unaccepted(roots,0)
    else:recovery._validate_preparation_result(recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[]))


def test_t1_ABC_completed_later_activity_observational(b23_host,installation,monkeypatch):
    roots,scope,_,_=installation
    histories=_t1_completed_nonempty(installation,b23_host,monkeypatch,3)
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db,db:
        db.execute("UPDATE resource_monitor SET state='checking'")
        db.execute('UPDATE resource_accounting SET highwater=highwater+1')
    (roots['staging']/'later-ordinary-file').write_bytes(b'later activity')
    (roots['receipts']/'journal.json').write_bytes(b'later ordinary journal')
    before=logical(roots);files=retained_bytes(roots)
    for r in histories:_kfifth_fresh(roots,scope,target=r['id'],setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
    assert logical(roots)==before and retained_bytes(roots)==files


@pytest.mark.parametrize('same_consumer',[False,True])
@pytest.mark.parametrize('dimension',['entries','bytes'])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_rehearsal_receipt_reserve_isolated_namespace(b23_host,installation,monkeypatch,same_consumer,dimension,delta,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked;rehearse=recovery._rehearse_preparation
    namespace=recovery._prefix_receipt_namespace
    directory=tmp_path/'isolated-retained-namespace';directory.mkdir()
    trap=tmp_path/'unexpected-writer'
    def observed(session,plan,intent,frozen):
        chain=[dict(recovery_id=intent['id'],**e,lane=intent['new_lane'],epoch=intent['new_epoch']) for e in intent['expected'][:4]]
        cert=recovery._certificate(intent,chain[-1])
        documents=[encode(intent).encode(),encode(recovery.archive_manifest(intent)).encode(),encode(cert).encode(),
            encode(dict(certificate=cert,steps=chain)).encode(),bytes.fromhex(intent['before']['journal']['bytes_hex'])]
        required=len(frozen._raw)+sum(map(len,documents))+18*65536
        count=2048-18+delta if dimension=='entries' else 1
        names=[]
        for i in range(count):
            name='retained-'+str(i);names.append(name);p=directory/name
            p.write_bytes(b'x'*(16777216-required+delta) if dimension=='bytes' else b'');p.chmod(0o600)
        measured=namespace(directory,[],retained_names=names)
        assert measured==((16777216-required+delta,1) if dimension=='bytes' else (0,count))
        # Isolate only the namespace measurement at the REAL rehearsal guard.
        # These files are never imported into source/history or acceptance.
        # Actual production capacity failures are covered separately above.
        with monkeypatch.context() as m:
            m.setattr(recovery,'_prefix_receipt_namespace',lambda *a,**k:measured)
            return rehearse(session,plan,intent,frozen)
    monkeypatch.setattr(recovery,'_rehearse_preparation',observed)
    def consume(session,**request):
        try:r=prepare(session,**request)
        except ValueError as e:
            assert not session._live and str(e)=='b2_prefix_receipt_budget'
            (tmp_path/'burned').write_text(str(e));raise
        if same_consumer:assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',consume)
    if delta>0:
        def forbidden(*a,**k):trap.write_text('acceptance');raise AssertionError('reserve failure reached acceptance')
        for name in ('_prefix_file','_pending_prefix_locked'):monkeypatch.setattr(recovery,name,forbidden)
        with pytest.raises(ValueError):recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
        assert not trap.exists() and (tmp_path/'burned').exists();_t1_assert_unaccepted(roots,1)
    else:recovery._validate_preparation_result(recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[]))


def test_t1_actual_pipe_overflow_precedes_decoder(monkeypatch,tmp_path):
    import resource
    prior=resource.getrlimit(resource.RLIMIT_FSIZE);owned=_t1_pipe_model(monkeypatch)
    size=1048577;value='a'*(size-13)
    raw=encode(dict(result=value)).encode();assert len(raw)==size
    # Fault injection at the sender boundary: receiver must independently cap.
    monkeypatch.setattr(recovery,'_maintenance_result_frame',lambda result:bytearray(raw))
    trap=tmp_path/'decoded-overflow'
    def forbidden(*a,**k):trap.write_text('decode');raise AssertionError('overflow decoded')
    monkeypatch.setattr(recovery,'_decode_maintenance_result',forbidden)
    with pytest.raises(ValueError,match='result_budget'):recovery._run_bounded_maintenance({}, {},lambda s:value)
    assert not trap.exists() and resource.getrlimit(resource.RLIMIT_FSIZE)==prior
    with pytest.raises(ChildProcessError):os.waitpid(owned[0],os.WNOHANG)


@pytest.mark.parametrize('delta',[-1,0,1])
def test_t1_actual_file_stat_guard_boundary(tmp_path,delta):
    p=tmp_path/'core.sqlite';p.write_bytes(b'x'*(4194304+delta));p.chmod(0o600)
    # Physical file-length guard only; arbitrary bytes are NOT a SQLite or
    # recovery semantic positive and create no authority.
    if delta>0:
        with pytest.raises(ValueError,match='maintenance_file'):recovery._maintenance_files(tmp_path)
    else:recovery._maintenance_files(tmp_path)


def test_t1_durable_completion_lost_result_no_false_success(b23_host,installation,monkeypatch,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def finish_then_lose(session,**request):
        r=prepare(session,**request)
        assert session.continue_kernel(r['intent'],prepared=r._snapshot)['status']=='complete'
        assert r._snapshot._session is session and r._snapshot._used
        assert len(r['intent']['id'])==32
        (tmp_path/'durable-completion').write_text(r['intent']['id'])
        def broken(*a):raise BrokenPipeError('modeled lost result pipe')
        os.write=broken  # child only, AFTER all real durable completion writers
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',finish_then_lose)
    with pytest.raises(ValueError,match='child_failed'):
        recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    identity=(tmp_path/'durable-completion').read_text()
    with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM p1_recoveries').fetchone()==(1,)
        assert db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone()==(5,)
        assert db.execute('SELECT id FROM p1_recoveries').fetchone()==(identity,)
    before=logical(roots);files=retained_bytes(roots)
    _kfifth_fresh(roots,scope,target=identity,setup=_KFIFTH_OBSERVE+_T1_CURRENT_UNION)
    assert logical(roots)==before and retained_bytes(roots)==files


def test_t1_owned_proposal_reads_cannot_mutate_sender(b23_host,installation,monkeypatch,tmp_path):
    roots,scope,profiles,_=installation
    _lotp_population(installation,'running');_lotp_transport(b23_host,monkeypatch)
    prepare=recovery._prepare_abandon_locked
    def inspect(session,**request):
        r=prepare(session,**request)
        plan=r['plan'];intent=r['intent']
        plan['new_epoch']='0'*32;intent['new_epoch']='0'*32
        assert r['plan']['new_epoch']!=plan['new_epoch']
        assert r['intent']['new_epoch']!=intent['new_epoch']
        recovery._validate_preparation_result(r)
        # Arbitrary mutation through the private dict storage is still rejected
        # by the strict codec, rather than acquiring snapshot/plan authority.
        bad=dict(r);bad['intent']=intent
        with pytest.raises(ValueError):recovery._decode_maintenance_result(recovery._maintenance_result_frame(bad),preparation=True)
        with pytest.raises(ValueError):session.continue_kernel(intent,prepared=r._snapshot)
        assert not session._live
        assert not list(roots['receipts'].glob('p1-recovery-*'))
        (tmp_path/'original').write_text(digest(r['intent']))
        return r
    monkeypatch.setattr(recovery,'_prepare_abandon_locked',inspect)
    result=recovery._prepare_abandon(roots,scope,profiles=profiles,budget=DEFAULT_BUDGET,diagnostic_decisions=[])
    recovery._validate_preparation_result(result)
    assert digest(result['intent'])==(tmp_path/'original').read_text()
    _t1_assert_unaccepted(roots,1)


@pytest.mark.parametrize('repeat',range(2))
@pytest.mark.parametrize('site,timing',[(site,timing)
    for site in ('parent-write','cleanup-read','fork-read','fork-write','child-read')
    for timing in ('before','after','reuse','pipe-reuse')
    if timing!='pipe-reuse' or site=='parent-write'])
@pytest.mark.parametrize('failure',[InterruptedError,KeyboardInterrupt])
def test_t1_close_ownership_cleanup(monkeypatch,tmp_path,repeat,site,timing,failure):
    """Injected close faults, not evidence of kernel EINTR behavior.

    Actual fork/fds, bounded child reap, durable child-operation marker, and
    a reused fd sentinel distinguish release from a blind close retry.
    """
    import signal,time,resource
    prior=resource.getrlimit(resource.RLIMIT_FSIZE)
    owned=_t1_pipe_model(monkeypatch)
    parent=os.getpid();pipe=os.pipe;close=os.close;fstat=os.fstat
    ends=[];fired=[];sentinels=[];statuses=[];pipe_id=[]
    marker=tmp_path/'operation-entered'
    def tracked_pipe():
        pair=pipe();ends.extend(pair);pipe_id.append(fstat(pair[0]).st_ino);return pair
    def broken_close(fd):
        target=(os.getpid()!=parent and site=='child-read' and fd==ends[0]) or (
            os.getpid()==parent and site!='child-read' and fd==ends[0 if site in ('cleanup-read','fork-read') else 1])
        if target and not fired:
            fired.append(fd)
            if timing!='before':close(fd)
            if timing=='pipe-reuse':
                os.dup2(ends[1 if fd==ends[0] else 0],fd)
                sentinels.append(fd)
            if timing=='reuse':
                replacement=os.open(tmp_path/'sentinel',os.O_RDWR|os.O_CREAT,0o600)
                if replacement!=fd:
                    os.dup2(replacement,fd);close(replacement)
                sentinels.append(fd)
            raise failure('injected close failure '+timing)
        return close(fd)
    if site.startswith('fork-'):
        def failed_fork():raise OSError('injected fork failure')
        monkeypatch.setattr(os,'fork',failed_fork)
    monkeypatch.setattr(os,'pipe',tracked_pipe)
    monkeypatch.setattr(os,'close',broken_close)
    if site=='cleanup-read':
        def failed_read(*a):raise ValueError('injected read failure')
        monkeypatch.setattr(os,'read',failed_read)
    if timing=='pipe-reuse':
        # Linux pipe ends share inode identity. Model that identity on macOS;
        # the descriptors, opposite access modes and dup2 remain actual.
        fstat=os.fstat
        def linux_pipe_identity(fd):
            current=fstat(fd)
            if fd in ends:
                fields=list(current);fields[1]=pipe_id[0]
                return os.stat_result(fields)
            return current
        monkeypatch.setattr(os,'fstat',linux_pipe_identity)
    waitpid=os.waitpid
    def tracked_waitpid(pid,flags):
        result=waitpid(pid,flags)
        if result[0]:statuses.append(result[1])
        return result
    monkeypatch.setattr(os,'waitpid',tracked_waitpid)
    def operation(session):
        marker.write_text('operation entered')
        if site!='cleanup-read':time.sleep(60)
        return dict(success=True)
    start=time.monotonic()
    try:
        with pytest.raises((failure,OSError,ValueError)):
            recovery._run_bounded_maintenance({}, {},operation)
        assert time.monotonic()-start<5
        assert resource.getrlimit(resource.RLIMIT_FSIZE)==prior
        for fd in ends:
            if fd in sentinels:
                if timing=='pipe-reuse':assert os.fstat(fd).st_ino==pipe_id[0]
                else:assert os.fstat(fd).st_ino==(tmp_path/'sentinel').stat().st_ino
            else:
                with pytest.raises(OSError):os.fstat(fd)
        for pid in owned:
            with pytest.raises(ChildProcessError):waitpid(pid,os.WNOHANG)
        if site=='child-read':
            assert not marker.exists()
            assert len(statuses)==1 and os.WIFEXITED(statuses[0]) and os.WEXITSTATUS(statuses[0])==74
        if site.startswith('fork-'):assert not owned and not marker.exists()
    finally:
        # Harness owns these exact fd numbers throughout; never opens anything
        # after the call except the deliberate sentinel, which it also owns.
        for fd in ends:
            try:close(fd)
            except OSError:pass
        for pid in owned:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:waitpid(pid,0)
            except ChildProcessError:pass


@pytest.mark.parametrize('repeat',range(2))
@pytest.mark.parametrize('error_type',[InterruptedError,KeyboardInterrupt,BaseException])
@pytest.mark.parametrize('phase',['before','after'])
@pytest.mark.parametrize('site',['kill','cleanup-wait','cleanup-probe','normal-wait'])
def test_t1_reap_interruption_ownership(monkeypatch,tmp_path,repeat,error_type,phase,site):
    """Actual children/fds; injected control flow, not Linux syscall qualification."""
    import signal,time,resource
    parent=os.getpid();fork=os.fork;kill=os.kill;wait=os.waitpid;close=os.close;pipe=os.pipe
    unrelated=fork()
    if unrelated==0:
        time.sleep(60);os._exit(0)
    owned=[];ends=[];fired=[];released=[]
    marker=tmp_path/'operation';original=ValueError('original close refusal')
    interruption=error_type('injected '+site+' '+phase)
    prior=resource.getrlimit(resource.RLIMIT_FSIZE)
    try:
        owned=_t1_pipe_model(monkeypatch)
        def tracked_pipe():
            pair=pipe();ends.extend(pair);return pair
        def badclose(fd):
            if os.getpid()==parent and fd==ends[1] and site!='normal-wait':
                until=time.monotonic()+3
                while not marker.with_suffix('.ready').exists() and time.monotonic()<until:time.sleep(.001)
                assert marker.with_suffix('.ready').exists(),'sleeping child must enter operation before cleanup'
                close(fd);raise original
            return close(fd)
        def badkill(pid,sig):
            assert pid!=unrelated and not released,'never signal after an actual reap'
            if site=='kill' and not fired:
                fired.append(site)
                if phase=='after':kill(pid,sig)
                raise interruption
            return kill(pid,sig)
        def badwait(pid,flags):
            assert pid!=unrelated
            target=(site=='normal-wait' or (site=='cleanup-wait' and flags==0) or
                    (site=='cleanup-probe' and flags==os.WNOHANG))
            if os.getpid()==parent and target and not fired:
                fired.append(site)
                if phase=='after':
                    if site=='cleanup-probe':kill(pid,signal.SIGKILL)
                    result=wait(pid,0)
                    assert result[0]==pid
                    released.append(pid)
                raise interruption
            return wait(pid,flags)
        monkeypatch.setattr(os,'pipe',tracked_pipe)
        monkeypatch.setattr(os,'close',badclose)
        monkeypatch.setattr(os,'kill',badkill)
        monkeypatch.setattr(os,'waitpid',badwait)
        def operation(session):
            with marker.open('a') as stream:
                stream.write('entered\n');stream.flush();os.fsync(stream.fileno())
            marker.with_suffix('.ready').write_text('ready')
            if site!='normal-wait':time.sleep(60)
            return True
        start=time.monotonic()
        if site=='normal-wait' and phase=='before' and error_type is InterruptedError:
            assert recovery._run_bounded_maintenance({}, {},operation) is True
        else:
            with pytest.raises(BaseException) as caught:
                recovery._run_bounded_maintenance({}, {},operation)
            assert caught.value is (interruption if site=='normal-wait' else original)
        assert fired==[site] and len(owned)==1 and time.monotonic()-start<5
        assert marker.read_text()=='entered\n' and resource.getrlimit(resource.RLIMIT_FSIZE)==prior
        for fd in ends:
            with pytest.raises(OSError):os.fstat(fd)
        with pytest.raises(ChildProcessError):wait(owned[0],os.WNOHANG)
        assert wait(unrelated,os.WNOHANG)==(0,0),'unrelated child remains owned by harness'
    finally:
        for fd in ends:
            try:close(fd)
            except OSError:pass
        # Ownership probe precedes every harness signal too, including RED paths.
        for pid in owned+[unrelated]:
            try:ended,_=wait(pid,os.WNOHANG)
            except ChildProcessError:continue
            if not ended:
                try:kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:wait(pid,0)
                except ChildProcessError:pass


@pytest.mark.parametrize('site',['kill-ESRCH','cleanup-ECHILD'])
def test_t1_cleanup_missing_process_receipts(monkeypatch,tmp_path,site):
    import signal,time
    owned=_t1_pipe_model(monkeypatch)
    parent=os.getpid();pipe=os.pipe;close=os.close;kill=os.kill;wait=os.waitpid
    ends=[];fired=[];reaped=[];original=KeyboardInterrupt('original parent interruption')
    marker=tmp_path/'operation'
    def trackedpipe():
        pair=pipe();ends.extend(pair);return pair
    def badclose(fd):
        if os.getpid()==parent and fd==ends[1]:
            until=time.monotonic()+3
            while not marker.with_suffix('.ready').exists() and time.monotonic()<until:time.sleep(.001)
            assert marker.with_suffix('.ready').exists()
            close(fd);raise original
        return close(fd)
    def badkill(pid,sig):
        assert not reaped,'ECHILD must prevent PID reuse signalling'
        result=kill(pid,sig)
        if site=='kill-ESRCH' and not fired:
            fired.append(site);raise ProcessLookupError()
        return result
    def badwait(pid,flags):
        if flags==0 and site=='cleanup-ECHILD' and not fired:
            fired.append(site);wait(pid,flags);reaped.append(pid);raise ChildProcessError()
        return wait(pid,flags)
    monkeypatch.setattr(os,'pipe',trackedpipe);monkeypatch.setattr(os,'close',badclose)
    monkeypatch.setattr(os,'kill',badkill);monkeypatch.setattr(os,'waitpid',badwait)
    def operation(session):
        with marker.open('a') as stream:
            stream.write('once');stream.flush();os.fsync(stream.fileno())
        marker.with_suffix('.ready').write_text('ready')
        time.sleep(60)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:recovery._run_bounded_maintenance({}, {},operation)
        assert caught.value is original and fired==[site] and marker.read_text()=='once'
        with pytest.raises(ChildProcessError):wait(owned[0],os.WNOHANG)
        for fd in ends:
            with pytest.raises(OSError):os.fstat(fd)
    finally:
        for fd in ends:
            try:close(fd)
            except OSError:pass
        for pid in owned:
            try:ended,_=wait(pid,os.WNOHANG)
            except ChildProcessError:continue
            if not ended:
                try:kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
                wait(pid,0)


@pytest.mark.parametrize('error_type',[InterruptedError,KeyboardInterrupt,BaseException])
def test_t1_cleanup_refuses_inside_caller_exception_handler(monkeypatch,error_type):
    """An already handled caller exception must not hide a new cleanup failure."""
    import signal
    owned=_t1_pipe_model(monkeypatch)
    parent=os.getpid();pipe=os.pipe;close=os.close;wait=os.waitpid;kill=os.kill
    ends=[];fired=[];interruption=error_type('read close after successful child')
    def trackedpipe():
        pair=pipe();ends.extend(pair);return pair
    def badclose(fd):
        if os.getpid()==parent and fd==ends[0] and not fired:
            fired.append(fd);close(fd);raise interruption
        return close(fd)
    monkeypatch.setattr(os,'pipe',trackedpipe);monkeypatch.setattr(os,'close',badclose)
    try:
        try:raise ValueError('caller already handled this')
        except ValueError:
            with pytest.raises(BaseException) as caught:
                recovery._run_bounded_maintenance({}, {},lambda session:True)
            assert caught.value is interruption
        assert fired and len(owned)==1
        with pytest.raises(ChildProcessError):wait(owned[0],os.WNOHANG)
        for fd in ends:
            with pytest.raises(OSError):os.fstat(fd)
    finally:
        for fd in ends:
            try:close(fd)
            except OSError:pass
        for pid in owned:
            try:ended,_=wait(pid,os.WNOHANG)
            except ChildProcessError:continue
            if not ended:
                kill(pid,signal.SIGKILL);wait(pid,0)


@pytest.mark.parametrize('reader', ['startup_read', 'enrollment_gate'])
@pytest.mark.parametrize('case', ['early', 'sql', 'schema', 'connect', 'success', 'double-failure'])
def test_lifetime_private_reader_closes_before_teardown(tmp_path, monkeypatch, reader, case):
    from contextlib import contextmanager
    original_connect = sqlite3.connect
    source = tmp_path/'core.sqlite'
    with closing(original_connect(source)) as db:
        if reader == 'startup_read' and case == 'early': db.execute('CREATE TABLE ordinary(id)')
        else:
            for table, columns in recovery.B1_COLUMNS.items():
                db.execute('CREATE TABLE '+table+'('+','.join(c+' TEXT' for c in columns.split())+')')
            host = dict()  # Validators below isolate ownership from enrollment policy.
            installation = dict(deployment_id='a'*32, revision=1, scope='owned-synthetic', host=host,
                profile_budget_digest=recovery.digest(dict(profiles=[], budget={})))
            db.execute('INSERT INTO p1_installation(id,record,digest) VALUES(?,?,?)', ('a'*32, encode(installation), recovery.digest(installation)))
            record = dict(deployment_id='a'*32, scope=host, evidence='actual-host-inspection', lane='b'*32, boot='c'*32)
            db.execute('INSERT INTO p1_registrations(id,record,digest) VALUES(?,?,?)', (recovery.digest(record), encode(record), recovery.digest(record)))
            if case == 'schema': db.execute('DROP TABLE p1_registrations')
            if case == 'early' and reader == 'enrollment_gate': db.execute('DELETE FROM p1_installation')
            db.commit()
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    opened = []; events = []; primary = sqlite3.OperationalError('reader SQL failure')
    class Connection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if case in ('sql', 'double-failure') and sql == 'PRAGMA query_only=ON': raise primary
            return super().execute(sql, *args, **kwargs)
        def close(self):
            super().close(); events.append('closed')
            if case == 'double-failure': raise RuntimeError('close failure')
    def connect(path, **kwargs):
        assert '?mode=ro' in path and str(source) not in path
        if case == 'connect': raise primary
        db = original_connect(path, factory=Connection, **kwargs); opened.append(db); return db
    private_database = recovery.private_database
    @contextmanager
    def private(path):
        with private_database(path) as target:
            try: yield target
            finally:
                events.append('teardown')
                for db in opened:
                    with pytest.raises(sqlite3.ProgrammingError, match='closed'): original_execute(db, 'SELECT 1')
                if case == 'double-failure': raise RuntimeError('private failure')
    original_execute = sqlite3.Connection.execute
    monkeypatch.setattr(recovery, 'private_database', private)
    monkeypatch.setattr(recovery.sqlite3, 'connect', connect)
    monkeypatch.setattr(recovery, 'validate_scope', lambda _:None)
    monkeypatch.setattr(recovery, 'validate_registration', lambda _:None)
    monkeypatch.setattr(recovery, 'check_completion_hello', lambda *a:None)
    monkeypatch.setattr(recovery, 'completion_binding', lambda *a:None)
    def run():
        if reader == 'startup_read': return recovery.startup_read(tmp_path)
        return recovery.enrollment_gate(tmp_path, recovery.digest(record), dict(lane='b'*32, boot='c'*32, quarantined=False), [], {})
    if case == 'schema' or (case == 'early' and reader == 'enrollment_gate'):
        with pytest.raises(ValueError, match='schema|not_enrolled'): run()
    elif case in ('sql', 'connect', 'double-failure'):
        with pytest.raises(sqlite3.OperationalError) as caught: run()
        assert caught.value is primary
    else:
        result = run()
        assert result == (None if reader == 'startup_read' and case == 'early' else
            dict(deployment_id='a'*32, completion=None) if reader == 'startup_read' else 'a'*32)
    assert events == (['teardown'] if case == 'connect' else ['closed', 'teardown'])
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('case', ['semantic', 'death73', 'unrelated', 'signal', 'timeout', 'missing-semantic'])
def test_lifetime_wait_guard_unit(case, maintenance_wait_guard):
    from types import SimpleNamespace
    MaintenanceWaitGuard = type(maintenance_wait_guard)
    import signal
    session = SimpleNamespace(shouldstop=False)
    node = SimpleNamespace(nodeid='synthetic-owned-wait')
    guard = MaintenanceWaitGuard(node, session); guard.owned.add(123)
    pid = 456 if case == 'unrelated' else 123
    status = signal.SIGSEGV if case == 'signal' else signal.SIGKILL if case == 'timeout' else (73 if case == 'death73' else 74)<<8
    result = (pid, status)
    assert guard.observe(result, witness=b'V' if case == 'semantic' else b'D' if case == 'death73' else b'',
        expected_signal=signal.SIGKILL if case == 'timeout' else None) is result
    assert bool(guard.failures) == (case in ('signal', 'missing-semantic'))
    assert bool(session.shouldstop) == bool(guard.failures)
    assert guard.receipts == ([] if case == 'unrelated' else [result])
    assert guard.owned == ({123} if case == 'unrelated' else set())


def test_lifetime_guard_real_semantic_refusal_and_unrelated_child(monkeypatch, maintenance_wait_guard):
    import sys
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(recovery, '_install_maintenance_limit', lambda:None)
    @__import__('contextlib').contextmanager
    def session(*a, **k): yield None
    monkeypatch.setattr(recovery, '_maintenance_session', session)
    unrelated = os.fork()
    if unrelated == 0: os._exit(0)
    def refused(_): raise ValueError('synthetic semantic refusal')
    with pytest.raises(ValueError, match='child_failed'):
        recovery._run_bounded_maintenance({}, {}, refused)
    assert os.waitpid(unrelated, 0) == (unrelated, 0)
    assert not maintenance_wait_guard.failures
    assert len(maintenance_wait_guard.receipts) == 1
    assert maintenance_wait_guard.receipts[0][1] == 74<<8
    assert not maintenance_wait_guard.owned


def test_lifetime_guard_reporting_overrides_caught_negative(maintenance_wait_guard):
    from types import SimpleNamespace
    node = SimpleNamespace(nodeid='synthetic-negative', _maintenance_guard=None)
    session = SimpleNamespace(shouldstop=False)
    guard = type(maintenance_wait_guard)(node, session); node._maintenance_guard = guard
    guard.owned.add(123)
    with pytest.raises(ValueError):
        guard.observe((123, 11))  # Synthetic SIGSEGV wait status; no signal is generated.
        raise ValueError('sanitized child failure')
    report = SimpleNamespace(when='call', outcome='passed', longrepr=None)
    hook = maintenance_wait_guard.report_hook(node, None)
    next(hook)
    with pytest.raises(StopIteration) as result: hook.send(report)
    assert result.value.value is report and report.outcome == 'failed'
    assert session.shouldstop and 'status=11' in report.longrepr


def test_lifetime_startup_lock_discovery_never_creates_paths(tmp_path):
    from service.recovery import _startup_read_lock
    root = tmp_path/'absent'
    with _startup_read_lock(root) as readable: assert readable is False
    assert not root.exists()
    root.mkdir()
    before = list(root.iterdir())
    with _startup_read_lock(root) as readable: assert readable is False
    assert list(root.iterdir()) == before
    (root/'core.sqlite').write_bytes(b'not opened')
    with pytest.raises(ValueError, match='missing_control_lock'):
        with _startup_read_lock(root): pytest.fail('unchecked database')
    assert (root/'core.sqlite').read_bytes() == b'not opened'


def test_lifetime_startup_lock_release_does_not_unlock_other_owner(tmp_path):
    import fcntl
    from service.core import Core
    from service.recovery import _startup_read_lock
    core = Core(tmp_path/'core'); core.close()
    with _startup_read_lock(core.root) as readable:
        assert readable
        with pytest.raises(ValueError, match='control_busy'): Core(core.root)
    # The read-side/Core gap is not transfer: Core must independently acquire.
    owner = Core(core.root)
    try:
        with pytest.raises(ValueError, match='control_busy'):
            with _startup_read_lock(core.root): pytest.fail('reader bypass')
        with pytest.raises(ValueError, match='control_busy'): Core(core.root)
    finally: owner.close()
    with _startup_read_lock(core.root) as readable: assert readable


@pytest.mark.parametrize('state', ['constructing', 'live', 'closing', 'uncertain', 'owner-only'])
@pytest.mark.parametrize('same_root', [False, True])
def test_bounded_core_maintenance_refuses_before_effects(tmp_path, monkeypatch, state, same_root):
    import threading, sys
    from service import core as module
    baseline = tuple(module._core_registry)
    record = module._reserve_core(tmp_path/'held')
    record.state = 'uncertain' if state == 'owner-only' else state
    effects = []
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(threading, 'active_count', lambda: 1)
    def forbidden(*a, **kw):
        effects.append(True); raise AssertionError('pre-admission effect')
    monkeypatch.setattr(os, 'pipe', forbidden)
    monkeypatch.setattr(os, 'fork', forbidden)
    monkeypatch.setattr(recovery, '_maintenance_session', forbidden)
    try:
        with pytest.raises(ValueError, match='b2_maintenance_core_ownership_busy'):
            recovery._run_bounded_maintenance({'control': tmp_path/('held' if same_root else 'other')}, {}, forbidden, _prewrite=forbidden)
        assert effects == []
    finally:
        record.state = 'constructing'; record.close()
    assert tuple(module._core_registry) == baseline


@pytest.mark.parametrize('failure', ['pipe', 'fork'])
def test_bounded_core_prefork_lease_failure(tmp_path, monkeypatch, failure):
    import threading, sys
    from service import core as module
    baseline = tuple(module._core_registry)
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(threading, 'active_count', lambda: 1)
    pipe = os.pipe; ends = []; errors = []
    def contender():
        try: module._reserve_core(tmp_path/'concurrent')
        except ValueError as e: errors.append(str(e))
    def check():
        with pytest.raises(ValueError, match='core_ownership_busy'): module._reserve_core(tmp_path/'nested')
        before = len(errors)
        thread = threading.Thread(target=contender); thread.start(); thread.join(2)
        assert not thread.is_alive() and len(errors) == before+1
    def piping():
        check()
        if failure == 'pipe': raise OSError('owned pipe failure')
        pair = pipe(); ends.extend(pair); return pair
    def forking():
        check()
        raise OSError('owned fork failure')
    monkeypatch.setattr(os, 'pipe', piping); monkeypatch.setattr(os, 'fork', forking)
    with pytest.raises(OSError, match='owned .* failure'):
        recovery._run_bounded_maintenance({}, {}, lambda s: True)
    for fd in ends:
        with pytest.raises(OSError): os.fstat(fd)
    r = module._reserve_core(tmp_path/'after'); r.close()
    assert tuple(module._core_registry) == baseline


def test_bounded_core_lease_parent_release_child_pid_refusal(monkeypatch, tmp_path):
    from service import core as module
    baseline = tuple(module._core_registry)
    _t1_pipe_model(monkeypatch)
    def operation(session):
        # Actual one-shot child: PID guard must run before the locked mutex.
        for action in (lambda: module._reserve_core(tmp_path/'child'),
                       lambda: module._core_maintenance_lease().__enter__()):
            try: action()
            except ValueError as e: assert 'inherited_process' in str(e)
            else: raise AssertionError('child inherited registry accepted')
        return 'child refused inherited registry'
    assert recovery._run_bounded_maintenance({}, {}, operation) == 'child refused inherited registry'
    r = module._reserve_core(tmp_path/'parent'); r.close()
    assert tuple(module._core_registry) == baseline


def test_bounded_core_prefork_deterministic_parent_release(tmp_path, monkeypatch):
    import select, threading, sys
    from service import core as module
    baseline = tuple(module._core_registry); effects = []; waits = []
    pipe = os.pipe; ends = []; raw = iter([b'{"result":true}', b''])
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(threading, 'active_count', lambda: 1)
    def piping():
        with pytest.raises(ValueError, match='core_ownership_busy'): module._reserve_core(tmp_path/'at-pipe')
        pair = pipe(); ends.extend(pair); effects.append('pipe'); return pair
    def fork():
        with pytest.raises(ValueError, match='core_ownership_busy'): module._reserve_core(tmp_path/'at-fork')
        effects.append('fork'); return 987654
    def read(*a):
        # Parent releases immediately after successful fork, before any read.
        r = module._reserve_core(tmp_path/'after-fork'); r.close()
        return next(raw)
    def wait(pid, flags): waits.append((pid, flags)); return pid, 0
    monkeypatch.setattr(os, 'pipe', piping); monkeypatch.setattr(os, 'fork', fork)
    monkeypatch.setattr(os, 'read', read); monkeypatch.setattr(os, 'waitpid', wait)
    monkeypatch.setattr(select, 'select', lambda r, w, x, t: (r, w, x))
    monkeypatch.setattr(os, 'kill', lambda *a: pytest.fail('successful double signalled'))
    assert recovery._run_bounded_maintenance({}, {}, lambda s: pytest.fail('parent operation')) is True
    assert effects == ['pipe', 'fork'] and waits == [(987654, os.WNOHANG)]
    for fd in ends:
        with pytest.raises(OSError): os.fstat(fd)
    assert tuple(module._core_registry) == baseline
