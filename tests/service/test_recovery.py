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


def b2_intent(installation, *, complete=False,recovery_id='9'*32,archive_id='8'*32):
    """Test-only evidence construction; no production transition/apply writer."""
    import hashlib
    roots,scope,_,_=installation
    observation,planned=b2_plan(installation);p=planned['plan'];root=roots['control']/'core'
    proof=dict(kind='E1-mock',observation=observation,prior_epoch=p['prior_epoch'],old_lane=p['old_lane'])
    p['proof_digest']=digest(proof)
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
    snapshot_path.write_bytes(snapshot_bytes);snapshot_path.chmod(0o600)
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
        job=json.loads(row[6]);job['state']='failed' if action['action']=='fail' else 'cancelled';row[6]=encode(job)
    for row in core_rows['resource_reservations']:row[3]='retained'
    core_rows['resource_monitor']=[[1,'healthy']];core_rows['resource_faults']=[]
    outputs=[dict(recovery_id=r['id'],plan_digest=r['plan_digest'],proof_digest=r['proof_digest'],before_digest=r['before_digest']),
        dict(archive_id=r['archive_id'],snapshot_digest=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),manifest_digest='0'*64,journal_state=p['journal_state'],journal_digest=p['journal_digest']),
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
    with closing(sqlite3.connect(root/'core.sqlite')) as db,db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('INSERT INTO p1_recoveries VALUES(?,?,?,?)',(r['id'],r['plan_id'],encode(r),digest(r)))
        if complete:
            for expected in r['expected'][:4]:
                step=dict(recovery_id=r['id'],**expected,lane=r['new_lane'],epoch=r['new_epoch'])
                db.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',(r['id'],step['step'],encode(step),digest(step)))
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
        db.execute('INSERT INTO p1_retired_publications VALUES(?,?,?,?)',(job_id,r['id'],encode(value),digest(value)))
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
