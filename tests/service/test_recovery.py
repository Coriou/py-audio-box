"""B1 E1 only: real disposable roots/SQLite/locks, mocked host projection."""
import copy
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
    with sqlite3.connect(roots['control']/'core'/'core.sqlite') as db:
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
    spec=importlib.util.spec_from_file_location('p1_operator',Path('scripts/p1-operator.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
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
    with pytest.raises(ValueError,match='inspection_budget'):recovery.inspect(roots,scope)


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
    class NoRuntime:
        def __getattr__(self,name):pytest.fail('B1 must never invoke runtime '+name)
    dispatcher=Dispatcher(roots['control']/'core',staging=roots['staging'],runtime=NoRuntime())
    try:
        assert dispatcher.call('status')['persistent_activation'] is False
        with pytest.raises(ValueError,match='persistent_activation_closed'):dispatcher.call('submit','p','new')
        with pytest.raises(ValueError,match='persistent_activation_closed'):dispatcher.call('cancel','unknown')
        assert dispatcher.call('jobs','p')==[]
    finally:dispatcher.close()


def test_unknown_inventory_refuses_plan_but_is_visible_in_inspect(installation):
    roots,scope,_,_=installation
    observation,_=enrolled_plan(installation)
    (roots['staging']/'unexplained').write_bytes(b'unknown')
    state=recovery.inspect(roots,scope)
    assert state['inventory']['counts']['unknown_foreign']==1
    with pytest.raises(ValueError,match='unknown_recovery_scope'):recovery.plan(roots,scope,observation)
