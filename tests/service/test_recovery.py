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
    path=next(roots['receipts'].glob('p1-recovery-*.intent.json'))
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
    result=subprocess.run([sys.executable,'-c',script],input=json.dumps(dict(roots={k:str(v) for k,v in roots.items()},scope=scope,id=identity)),text=True,capture_output=True)
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
    child=subprocess.run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid,epoch=epoch)),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
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
    pid=os.fork()
    if pid==0:
        core=Core(root);store=ResourceStore(core,DEFAULT_BUDGET,(root,roots['staging']))
        def observe():
            assert store.state()=='checking'
            core.db.execute('PRAGMA query_only=ON')
            raise ValueError('transient-output-growth')
        try:store.observe(observe)
        except sqlite3.OperationalError:
            # No close/finally/later loop can write a witness before death.
            os._exit(73)
        os._exit(74)
    assert os.waitpid(pid,0)[1]==73<<8
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
    child=subprocess.run([sys.executable,'-c',code],input=encode(dict(roots={k:str(p) for k,p in roots.items()},scope=scope,rid=rid,epoch=epoch,lane=lane,quarantine=point.endswith('before-replace'))),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
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
