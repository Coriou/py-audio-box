"""Finite r372 format matrix. Tiny SQLite/media; no child or conversion."""
import copy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contracts import validate as oracle
from service import persistence as p
from service.core import Core,encode,new_id
from service.publication_format import migrate,_validate,_model


def fixture(name='speech_generate'):
    return json.loads(Path('contracts/fixtures/v1/transcript-'+name+'.json').read_text())


def direct_history(core, operation='speech_generate', *, pending=False):
    """Use the inert synthetic producer directly, without a dispatch journal."""
    from service.worker_transport import Authority
    from worker.synthetic import fixture_outputs,fixture_manifest
    f=fixture('voice_prepare-roles' if operation=='voice_prepare' else operation);context=f['context']
    if operation=='voice_prepare':
        # Exact catalogue fixture input, privately owned by this test principal.
        asset=context['plan']['assets'][0]
        asset['principal_id']='alice'
        raw=Path('contracts/fixtures/v1/impulse.wav').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==asset['sha256']
        (core.root/'assets'/asset['id']).write_bytes(raw)
        with core.transaction():
            core.db.execute('INSERT INTO assets VALUES(?,?)',(asset['id'],encode(asset)))
    job=core.submit_execution('alice','direct-'+operation,context)
    attempt=dict(id=new_id('attempt'),generation=1,owner_id=new_id('owner'),staging_id=new_id('stage'),reaped=False)
    job.update(state='running',stage='synthetic',generation=1,attempts=[attempt],deadline='2099-01-01T00:00:00+00:00')
    with core.transaction(): core._save(job)
    a=Authority(job,context,core._input_assets('alice',context['plan']),f['profiles'])
    message=a.message();outputs,files=fixture_outputs(message)
    stage=core.root/'staging'/attempt['staging_id'];stage.mkdir(mode=0o700)
    for name,raw in files:(stage/name).write_bytes(raw)
    manifest=fixture_manifest(message,outputs)
    message.update(kind='result',stage='completed',result_asset_ids=[o['id'] for o in outputs])
    def fault(point):
        if pending and point=='before_adoption':raise RuntimeError('retained-pending')
    try: core._publish(job['id'],dict(message=message,manifest=manifest),fault=fault)
    except RuntimeError:
        if not pending:raise
    return job,context,manifest


class PublicationFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pab-r373-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()/'core'
    def core(self,version=2):
        core=Core(self.root,_publication_creation=version);self.addCleanup(core.close);return core
    def state(self):
        with closing(sqlite3.connect(self.root/'core.sqlite')) as db:
            return db.execute('PRAGMA user_version').fetchone(),list(db.execute('SELECT * FROM p1_format')),list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'))
    def test_fresh_default_and_explicit_v1_and_no_auto_upgrade(self):
        core=self.core();self.assertEqual(p.store_format(core.db),2);core.close()
        with self.assertRaisesRegex(ValueError,'v2_unsupported'):Core(self.root,_publication_creation=1)
        other=Path(self.tmp.name)/'old'
        with closing(Core(other,_publication_creation=1)) as old:self.assertEqual(old.publication_format,1)
        with closing(Core(other)) as old:self.assertEqual(old.publication_format,1)
    def test_mixed_versions_and_exact_ddl_refuse(self):
        for version in (1,2):
            for defect in ('marker','pragma','ddl','index','trigger','float-detached','bool-detached'):
                with self.subTest(version=version,defect=defect),closing(sqlite3.connect(':memory:')) as db:
                    db.execute(p.FORMAT_DDL);db.execute(p.PUBLICATION_V1 if version==1 else p.PUBLICATION_V2)
                    db.execute("INSERT INTO p1_format VALUES (?,'synthetic-core')",(version,));db.execute('PRAGMA user_version='+str(version))
                    schema=list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'))
                    if defect.endswith('detached'):
                        with self.assertRaises(ValueError):p.detached_format(schema,[[float(version) if defect.startswith('float') else True,'synthetic-core']])
                        continue
                    if defect=='marker':db.execute('UPDATE p1_format SET revision=?',(3-version,))
                    elif defect=='pragma':db.execute('PRAGMA user_version='+str(3-version))
                    elif defect=='ddl':
                        db.execute('DROP TABLE publication_work');db.execute(p.PUBLICATION_V2 if version==1 else p.PUBLICATION_V1)
                    elif defect=='index':db.execute('CREATE INDEX extra ON publication_work(sha256)')
                    else:db.execute('CREATE TRIGGER extra AFTER INSERT ON publication_work BEGIN SELECT 1; END')
                    with self.assertRaises(ValueError):p.store_format(db)
    def test_new_v1_export_core_gate_precedes_every_allocation_and_replay(self):
        core=self.core(1);context=fixture('audio_export')['context']
        before=list(core.db.iterdump())
        with patch('service.native_renewal.preallocate',side_effect=AssertionError('preallocated')):
            for retry in (None,'missing'):
                with self.assertRaisesRegex(ValueError,'publication_format_migration_required'):
                    core.submit_execution('alice','export',context,retry_of=retry)
        self.assertEqual(list(core.db.iterdump()),before)
        replay=dict(id='retained')
        with patch.object(core,'execution_duplicate',return_value=replay),patch.object(core,'require_export_format',side_effect=AssertionError('gate-before-replay')):
            self.assertIs(core.submit_execution('alice','old',context),replay)
    def test_v1_speech_public_read_and_migration_preserve_rows_rowids(self):
        core=self.core(1);job,context,manifest=direct_history(core)
        artifact=core.artifact(manifest['artifact_id']);before=_validate(core.db,self.root);core.close()
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'migrated'))
        with closing(Core(self.root)) as reopened:
            after=_validate(reopened.db,self.root)
            self.assertEqual(reopened.artifact(artifact['id']),artifact)
            self.assertEqual(reopened.submit_execution('alice','direct-speech_generate',context)['id'],job['id'])
            expected=copy.deepcopy(before[2]);expected['p1_format']=[(1,2,'synthetic-core')]
            self.assertEqual(encode(after[2]),encode(expected))
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'already-v2'))
    def test_empty_migration_and_already_v2(self):
        core=self.core(1);core.close()
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'migrated'))
        before=self.state();self.assertEqual(migrate(self.root,expect_v1=True),(0,'already-v2'));self.assertEqual(self.state(),before)
    def test_live_owner_and_control_lock_exclusion(self):
        import fcntl
        core=self.core(1);self.assertEqual(migrate(self.root,expect_v1=True)[0],2);core.close()
        with open(self.root/'control.lock','r+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
        self.assertEqual(self.state()[0],(1,))
    def test_pending_and_running_prestate_refuse_unchanged(self):
        core=self.core(1);direct_history(core,pending=True);before=list(core.db.iterdump());core.close()
        self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
        with closing(Core(self.root)) as reopened:self.assertEqual(list(reopened.db.iterdump()),before)
    def test_dispatch_history_unknown_scope_and_partial_groups_refuse(self):
        from service.dispatch_store import DispatchStore
        core=self.core(1);DispatchStore(core)
        # A retained settings row tied to a direct terminal job is still history.
        job,_,_=direct_history(core)
        core.db.execute("INSERT INTO dispatch_settings VALUES(?,'silence',8)",(job['id'],))
        before=list(core.db.iterdump());core.close();self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
        with closing(sqlite3.connect(self.root/'core.sqlite')) as db:self.assertEqual(list(db.iterdump()),before)
    def test_unknown_namespace_and_permissions_refuse(self):
        core=self.core(1);core.close();before=self.state()
        for name in ('native-renewal','p1-init.json','core.sqlite-journal','unknown'):
            path=self.root/name;path.write_bytes(b'')
            self.assertEqual(migrate(self.root,expect_v1=True)[0],2);path.unlink()
        (self.root/'control.lock').chmod(0o644)
        self.assertEqual(migrate(self.root,expect_v1=True)[0],2);self.assertEqual(self.state(),before)
    def test_schema_variants_refuse_without_mutation(self):
        from service.publication_format import _refused,Refusal
        self.assertEqual(_refused(p.PublicationQuarantine('publication_version_pair')),(2,'invalid-prestate:publication_version_pair'))
        self.assertEqual(_refused(Refusal('dispatch-history')),(2,'refused:dispatch-history'))
        for sql in ('CREATE TABLE resource_faults(reason TEXT PRIMARY KEY)',
                    'CREATE TABLE p1_installation(id TEXT)',
                    'CREATE INDEX extra ON publication_work(sha256)',
                    'CREATE TRIGGER extra AFTER INSERT ON publication_work BEGIN SELECT 1; END',
                    'CREATE TABLE foreign_ref(id TEXT REFERENCES publication_work(job_id))'):
            with self.subTest(sql=sql),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'core'
                with closing(Core(root,_publication_creation=1)) as core:core.db.execute(sql);before=list(core.db.iterdump())
                self.assertEqual(migrate(root,expect_v1=True)[0],2)
                with closing(sqlite3.connect(root/'core.sqlite')) as db:self.assertEqual(list(db.iterdump()),before)
    def test_each_transaction_step_rolls_back(self):
        for point in ('before-rename','after-rename','after-create','after-copy','after-drop','after-marker','after-version','before-commit'):
            with self.subTest(point=point),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'core'
                with closing(Core(root,_publication_creation=1)) as core:before=list(core.db.iterdump())
                def fault(where):
                    if where==point:raise sqlite3.OperationalError('injected')
                result=migrate(root,expect_v1=True,_fault=fault)
                self.assertEqual(result[0],3,result)
                with closing(Core(root)) as core:self.assertEqual(core.publication_format,1);self.assertEqual(list(core.db.iterdump()),before)
    def test_commit_ambiguity_is_unknown_and_explicit_reinspection(self):
        core=self.core(1);core.close()
        def fault(point):
            if point=='after-commit':raise OSError('lost-reply')
        self.assertEqual(migrate(self.root,expect_v1=True,_fault=fault)[0],4)
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'already-v2'))
    def test_exact_detached_models_select_marker_and_ddl(self):
        from service.recovery import _INSTALLED_V1_TABLE_DDL,_PUBLICATION_V2_TABLE_DDL,_installed_detached_rows
        from service.voice_store import _TABLE_DDL,_TRIGGER_DDL
        for version,ddl in ((1,_INSTALLED_V1_TABLE_DDL),(2,_PUBLICATION_V2_TABLE_DDL)):
            with closing(sqlite3.connect(':memory:')) as db:
                for sql in (*ddl,*_TABLE_DDL,*_TRIGGER_DDL):db.execute(sql)
                db.execute("INSERT INTO p1_format VALUES (?,'synthetic-core')",(version,))
                schema=list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'))
                tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                columns={t:' '.join(r[1] for r in db.execute('PRAGMA table_info('+t+')')) for t in tables}
                data={t:[list(r) for r in db.execute('SELECT * FROM '+t)] for t in tables}
                _installed_detached_rows(dict(schema=schema,data=data),columns)
                data['resource_accounting']=[[1,'1']]
                with self.assertRaises(ValueError):_installed_detached_rows(dict(schema=schema,data=data),columns)
                data['resource_accounting']=[]
                data['publication_work']=[['absent','{}','asset','adoption-'+'0'*32,'0'*64,1]]
                with self.assertRaises(ValueError):_installed_detached_rows(dict(schema=schema,data=data),columns)
                data['publication_work']=[]
                data['p1_format'][0][0]=float(version)
                with self.assertRaises(ValueError):_installed_detached_rows(dict(schema=schema,data=data),columns)
    def test_policy_scalars_never_allocate_near_ceiling(self):
        context=fixture('audio_export')['context'];job=dict(operation='audio_export',input_sha256=oracle.plan_hash(context['plan']))
        with patch('worker.audio_export.execution',return_value=SimpleNamespace(max_output_bytes=37748736)):
            self.assertEqual(p.publication_limit(2,job,context),37748736)
            self.assertEqual(p.publication_limit(1,job,context),1048576)
            for size in (1048576,1048577,37748736,37748737):
                with patch('service.persistence.os.open',side_effect=RuntimeError('reached-inode')):
                    expected='reached-inode' if size<=37748736 else 'publication_length_limit'
                    with self.assertRaisesRegex((RuntimeError,ValueError),expected):p.verify_file(Path('/unused'), '0'*64,size,links=1,publication_format=2,job=job,context=context)
        for op in ('speech_generate','voice_prepare'):
            c=fixture(op)['context'];j=dict(operation=op,input_sha256=oracle.plan_hash(c['plan']))
            self.assertEqual(p.publication_limit(2,j,c),1048576)
            with self.assertRaisesRegex(ValueError,'publication_length_limit'):p.verify_file(Path('/unused'),'0'*64,1048577,links=1,publication_format=2,job=j,context=c)
        forged=copy.deepcopy(context);forged['execution']['converter']['reported_build']='forged'
        with self.assertRaises(ValueError):p.publication_limit(2,job,forged)
    def test_media_and_scalar_accounting_are_independent(self):
        from service.reconciliation import Reconciliation,Incomplete
        core=self.core();scan=Reconciliation(core)
        scan.entries=scan.bytes=scan.media_bytes=0;scan.end=float('inf');scan.media_limit=160*1024**2
        scan._charge(148*1024**2,media=True);self.assertEqual(scan.bytes,0)
        with self.assertRaises(Incomplete):scan._charge(16*1024**2+1)
        scan.entries=scan.bytes=scan.media_bytes=0
        with self.assertRaises(Incomplete):scan._charge(160*1024**2+1,media=True)
        core.close()
        other=Path(self.tmp.name)/'v1'
        with closing(Core(other,_publication_creation=1)) as old:self.assertEqual(Reconciliation(old).media_limit,16*1024**2)
    def test_old_source_refusals_and_registration_limitation(self):
        # Source proof only; no old binary/subprocess authorization.
        # Pin the retained historical source with git outside this test, rather
        # than spawning a binary here. Current role gates are explicit literals.
        source=Path('service/persistent_init.py').read_text()
        self.assertIn("Core(roots['control']/'core', _publication_creation=1)",source)
        # The separately checked export family creates v2; the legacy branch remains v1.
        self.assertIn("Core(Path(control)/'core', _publication_creation=2 if checked else 1",source)
        native=Path('service/native_activation.py').read_text()
        self.assertIn("data['p1_format'] == [[1,'synthetic-core']]",native)
        self.assertNotIn('user_version',Path('service/install_child.py').read_text())

class PublicationFormatHTTPTests(unittest.TestCase):
    def test_v1_public_409_readiness_and_no_allocation(self):
        import test_audio_export as e
        f=e.AudioExportTests()
        with patch('service.dispatcher.Core',side_effect=lambda root,**kw:Core(root,**dict(kw,_publication_creation=1))):f.setUp()
        self.addCleanup(f.doCleanups)
        _,asset,request=f.upload()
        status=f.dispatcher.call('status')
        response=f.request('POST','/v1/operations/audio_export',payload=request,key='needs-format')
        self.assertEqual(response.status_code,409,response.text)
        self.assertEqual(response.json()['code'],'publication_format_migration_required')
        self.assertEqual(f.dispatcher.call('status')['reservations'],status['reservations'])
        self.assertFalse(f.dispatcher.call('status')['persistence_fault'])
        self.assertEqual(f.runtime.authorities,[])
        detail=f.request('GET','/skeleton/operations/audio_export')
        self.assertEqual(detail.json()['availability'],'publication_format_migration_required')
        f.dispatcher.close();f.start(exports=False)
        detail=f.request('GET','/skeleton/operations/audio_export')
        self.assertEqual(detail.json()['availability'],'unavailable_converter')

    def test_v1_retry_gate_and_retained_key_replay(self):
        import test_audio_export as e
        from service.admission import Admission
        f=e.AudioExportTests()
        with patch('service.dispatcher.Core',side_effect=lambda root,**kw:Core(root,**dict(kw,_publication_creation=1))):f.setUp()
        self.addCleanup(f.doCleanups)
        _,_,request=f.upload();f.dispatcher.close()
        with closing(Core(f.root)) as core:
            context=f.exports.resolve(core,'alice',request);h=oracle.plan_hash(context['plan'])
            job=dict(id='old-export',operation='audio_export',input_sha256=h,principal_id='alice',idempotency_key='old',state='failed',generation=0,attempts=[],artifact_ids=[],stage='failed',deadline='')
            oracle.validate('job',job)
            demand=Admission.demand(SimpleNamespace(native_availability=None),context,[f.profile])
            with core.transaction():
                core.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context,retry_of) VALUES(?,?,?,?,?,?,?,NULL)',('old-export','alice','audio_export','old',oracle.fingerprint('alice','audio_export',request),encode(job),encode(context)))
                core.db.execute("INSERT INTO events(job_id,state) VALUES('old-export','failed')")
                core.db.execute("INSERT INTO dispatch_settings VALUES('old-export','audio_export',25)")
                core.db.execute("INSERT INTO resource_reservations VALUES('old-export','inference',?,'retained')",(encode(demand),))
        f.start()
        self.assertEqual(f.submit('audio_export',request,'old')['job_id'],'old-export')
        before=f.dispatcher.call('status')['reservations']
        response=f.request('POST','/v1/operations/job_retry',payload=dict(job_id='old-export',input_sha256=h),key='new-retry')
        self.assertEqual(response.status_code,409,response.text)
        self.assertEqual(response.json()['code'],'publication_format_migration_required')
        self.assertEqual(f.dispatcher.call('status')['reservations'],before)
        self.assertFalse(f.dispatcher.call('status')['persistence_fault'])


class PublicationMigrationFaultTests(PublicationFormatTests):
    # Only the newly declared cases are selected below; inherited tests are
    # suppressed by the runner-independent unittest loader hook at file end.
    def test_two_file_prepare_v1_migration_and_canonical_intent(self):
        core=self.core(1);_,_,manifest=direct_history(core,'voice_prepare')
        before=core.artifact(manifest['artifact_id']);rows=_validate(core.db,self.root)[2]
        self.assertEqual(len(before['asset_ids']),2);core.close()
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'migrated'))
        with closing(Core(self.root)) as reopened:
            self.assertEqual(reopened.artifact(before['id']),before)
            after=_validate(reopened.db,self.root)[2]
            rows['p1_format']=[(1,2,'synthetic-core')]
            self.assertEqual(encode(after),encode(rows))
    def test_close_unknown_retains_ownership_until_confirmed_test_teardown(self):
        from service import core as c
        core=self.core(1);core.close()
        def fault(point):
            if point=='before-close':raise OSError('close-uncertain')
        self.assertEqual(migrate(self.root,expect_v1=True,_fault=fault)[0],4)
        owner=c._core_registry[0]
        try:
            self.assertIsNotNone(owner.db);self.assertIsNotNone(owner.owner)
            self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
            with self.assertRaises(ValueError):Core(self.root)
        finally:
            # Disposable injected fixture only: independently confirm both close
            # returns before releasing its in-process record. No product repair.
            owner.db.close();owner.owner.close()
            with c._core_bookkeeping():c._core_registry[0]=None
    def test_active_owner_lock_refused_without_reaping(self):
        import fcntl
        core=self.core(1);job,_,_=direct_history(core)
        path=self.root/'owners'/job['attempts'][-1]['owner_id']
        path.touch(mode=0o600);core.close();before=self.state()
        with open(path,'r+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
        self.assertEqual(self.state(),before)
        self.assertEqual(migrate(self.root,expect_v1=True),(0,'migrated'))

    def test_wrong_uid_refused(self):
        core=self.core(1);core.close();uid=os.getuid();before=self.state()
        with patch('service.publication_format.os.getuid',return_value=uid+1):
            self.assertEqual(migrate(self.root,expect_v1=True)[0],2)
        self.assertEqual(self.state(),before)
    def test_retained_owner_modes_refuse_before_original_writer(self):
        import stat
        for mode in (0o666,0o644,0o660,0o640,0o400,0o700,0o1600,0o2600,0o4600):
            with self.subTest(mode=oct(mode)),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'core'
                with closing(Core(root,_publication_creation=1)) as core:
                    job,_,_=direct_history(core)
                    path=root/'owners'/job['attempts'][-1]['owner_id']
                    path.touch(mode=0o600);path.chmod(mode)
                    before=list(core.db.iterdump())
                evidence=path.stat();connect=sqlite3.connect
                def guarded_connect(database,*args,**kwargs):
                    self.assertNotIn('?mode=rw',str(database))
                    return connect(database,*args,**kwargs)
                with patch('service.publication_format.sqlite3.connect',side_effect=guarded_connect):
                    result=migrate(root,expect_v1=True)
                    self.assertEqual(result[0],2,result)
                    self.assertTrue(result[1].startswith('refused:'),result)
                    if mode!=0o400:self.assertEqual(result[1],'refused:foreign-owner-file')
                with closing(connect(root/'core.sqlite')) as db:self.assertEqual(list(db.iterdump()),before)
                after=path.stat()
                self.assertEqual((after.st_dev,after.st_ino,after.st_mode,after.st_uid,after.st_gid,after.st_nlink,after.st_size,after.st_mtime_ns),
                                 (evidence.st_dev,evidence.st_ino,evidence.st_mode,evidence.st_uid,evidence.st_gid,evidence.st_nlink,evidence.st_size,evidence.st_mtime_ns))
                self.assertEqual(stat.S_IMODE(after.st_mode),mode);self.assertEqual(path.read_bytes(),b'')

    def test_held_owner_descriptor_mode_revalidated_before_writer_and_ddl(self):
        from service import publication_format as fmt
        for boundary in ('before-writer','before-ddl'):
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'core'
                with closing(Core(root,_publication_creation=1)) as core:
                    job,_,_=direct_history(core)
                    path=root/'owners'/job['attempts'][-1]['owner_id'];path.touch(mode=0o600)
                    before=list(core.db.iterdump())
                hold=fmt._hold_owners;validate=fmt._validate;held=[];calls=0
                def hold_then_change(*args):
                    hold(*args);held.append(args[2].locks[0].fileno())
                    if boundary=='before-writer':os.fchmod(held[0],0o666)
                def validate_then_change(*args):
                    nonlocal calls
                    result=validate(*args);calls+=1
                    if boundary=='before-ddl' and calls==2:os.fchmod(held[0],0o666)
                    return result
                with patch.object(fmt,'_hold_owners',side_effect=hold_then_change),patch.object(fmt,'_validate',side_effect=validate_then_change):
                    self.assertEqual(migrate(root,expect_v1=True),(2,'refused:foreign-owner-file'))
                with closing(sqlite3.connect(root/'core.sqlite')) as db:self.assertEqual(list(db.iterdump()),before)
                self.assertEqual(path.stat().st_mode & 0o7777,0o666);self.assertEqual(path.read_bytes(),b'')
    def test_scalar_sql_preflight_before_materialization(self):
        core=self.core(1)
        class SizeProxy:
            def set_progress_handler(self,*args):return core.db.set_progress_handler(*args)
            def execute(self,sql,*args):
                if sql.startswith('SELECT count(*),') and sql.endswith(' FROM catalogue_requests'):
                    return SimpleNamespace(fetchone=lambda:(1,16*1024**2+1))
                if sql.startswith('SELECT rowid,* FROM catalogue_requests'):raise AssertionError('oversized materialization')
                return core.db.execute(sql,*args)
        with self.assertRaisesRegex(ValueError,'inspection-budget'):_validate(SizeProxy(),self.root)
    def test_v2_incomplete_initialization_refuses(self):
        for version in (1,2):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'core'
                with closing(Core(root,_publication_creation=version)) as old:
                    old.db.execute('DROP TABLE publication_work');old.db.execute('DROP TABLE p1_format')
                if version==2:
                    with self.assertRaisesRegex(ValueError,'partial_p1_format'):Core(root)
                else:
                    with closing(Core(root)) as old:
                        self.assertFalse(old.p1);self.assertEqual(old.db.execute('PRAGMA query_only').fetchone(),(1,))
        core=self.core();core.db.execute('DROP TABLE catalogue_requests');core.close()
        with self.assertRaises(ValueError):Core(self.root)


def load_tests(loader, tests, pattern):
    suite=unittest.TestSuite()
    for cls in (PublicationFormatTests,PublicationFormatHTTPTests,PublicationMigrationFaultTests):
        for name in sorted(cls.__dict__):
            if name.startswith('test_'):suite.addTest(cls(name))
    return suite
