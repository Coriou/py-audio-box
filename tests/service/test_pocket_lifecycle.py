"""Saved-reference vertical slice; injected transport/backend, no ML or processes."""
import copy
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import wave
from unittest.mock import patch
from types import SimpleNamespace
from contracts import validate as oracle
from lib.tts_backends.contract import AudioResult
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import PocketContexts
from service.ingest import ingest_local
from service.production import Production
from worker.engines.pocket import execute as engine_execute
from worker.engines.pocket_execution import PocketExecution
from worker.runtime_child import run_generation
from worker.runtime_protocol import execution, generated_source
from worker.transport import Refused


def assert_configuration_mutations(test, state, journal):
    """Mutate joined detached evidence coherently, leaving only demand stale."""
    from service import recovery
    from service.core import encode
    from service.worker_transport import Authority
    data=state['evidence_projection']['data']
    context=json.loads(data['jobs'][0][7])
    actual=json.loads(data['resource_reservations'][0][2])['configuration']
    wrong=['0'*64]
    if execution(context) is not None:
        wrong.append(recovery.digest(context['plan']['resolved']['identity']['effective']))
    for configuration in wrong:
        test.assertNotEqual(configuration,actual)
        dirty=copy.deepcopy(state);projection=dirty['evidence_projection'];rows=projection['data']
        demand=json.loads(rows['resource_reservations'][0][2]);demand['configuration']=configuration
        rows['resource_reservations'][0][2]=encode(demand)
        dirty['reservations'][0]['demand']=demand
        dirty['source_digest']=recovery.digest(projection)
        test.assertFalse(recovery._allocation_coherent(rows))
        with test.assertRaisesRegex(ValueError,'installed_history_snapshot_binding'):
            recovery.installed_clean_history(dirty,journal)
    if execution(context) is None:return
    dirty=copy.deepcopy(state);modified_journal=copy.deepcopy(journal)
    rows=dirty['evidence_projection']['data']
    snapshot=json.loads(rows['dispatches'][0][2])
    # Semantically valid descriptor-only mutation: same effective identity,
    # different closed execution and Authority. Keep reservation digest frozen.
    snapshot[1]['execution']['descriptor']['max_text_bytes']+=1
    test.assertIsNotNone(execution(snapshot[1]))
    authority=Authority(*snapshot)
    rows['jobs'][0][7]=encode(authority.context)
    rows['dispatches'][0][2]=encode(snapshot);rows['dispatches'][0][3]=authority.binding
    receipt=json.loads(rows['dispatches'][0][7]);receipt['binding']=authority.binding
    rows['dispatches'][0][7]=encode(receipt)
    record=modified_journal['records'][0]
    record['allocation'].update(snapshot=snapshot,binding=authority.binding)
    record['receipt']=receipt
    dirty['dispatches'][0].update(binding=authority.binding,receipt_digest=recovery.digest(receipt))
    dirty['journal_records'][0]['binding']=authority.binding
    dirty['journal_digest']=recovery.digest(dict(payload=modified_journal,sha256=recovery.digest(modified_journal)))
    dirty['evidence_projection']['journal']=dirty['journal_digest']
    dirty['source_digest']=recovery.digest(dirty['evidence_projection'])
    test.assertFalse(recovery._allocation_coherent(rows))
    with test.assertRaisesRegex(ValueError,'installed_history_snapshot_binding'):
        recovery.installed_clean_history(dirty,modified_journal)


class PocketLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        # This Pocket cloning fixture exercises the existing publication-v1 lane.
        self.core = Core(self.root/'core', _publication_creation=1); self.addCleanup(lambda: self.core.close())
        self.stage = self.root/'stage'; self.stage.mkdir()
        self.admission = Admission(self.core, self.stage); self.store = DispatchStore(self.core,self.admission)
        b = io.BytesIO()
        with wave.open(b,'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\0\0'*4)
        self.raw = b.getvalue(); self.asset = ingest_local(self.core,'alice',self.raw)
        self.bundle = self.root/'bundle'; self.bundle.mkdir()
        entries=[]
        for name in ('config.yaml','model.safetensors'):
            data=b'inert'; (self.bundle/name).write_bytes(data)
            entries.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        raw=json.dumps(dict(schema_version=1,backend='pocket-tts',checkpoint='test',revision='rev1',language='fr',files=entries)).encode()
        (self.bundle/'manifest.json').write_bytes(raw)
        self.spec=PocketExecution('bundle',hashlib.sha256(raw).hexdigest(),'test','rev1','fr',2,1024,100,244,192*1024**2,self.asset['id'],self.asset['sha256'],len(self.raw),4,24000,22050)
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
        self.context=fixture['context']; plan=self.context['plan']; r=plan['resolved']
        ident=dict(engine='pocket-tts',model='test',bundle_sha256=self.spec.manifest_sha256,device='cpu',precision='float32')
        settings=dict(kind='pocket_clone',language='fr',seed=None)
        version=plan['voice_versions'][0]
        version.update(origin='clone',identity=ident,settings=settings,settings_sha256=oracle.canonical_hash(settings),configuration=dict(mode='pocket_clone',language='fr',reference_asset_id=self.asset['id']),reference_asset_ids=[self.asset['id']],reference_asset_sha256={self.asset['id']:self.asset['sha256']})
        self.version=copy.deepcopy(version)
        self.core.voices.register('alice','saved',version,expected_revision=0)
        plan['request'].update(identity=ident,text=' Exact\n texte ')
        plan['assets']=[self.asset]
        profile=fixture['profiles'][0]
        profile.update(id='pocket-cpu',support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INJECTED-ONLY-NOT-ACTUAL')
        profile['compatibility']=[dict(operation='speech_generate',mode='pocket_clone',identity=ident)]
        profile['limits'].update(ram_bytes=self.spec.memory_bytes,disk_bytes=2*1024**2,threads=2)
        r.update(identity=dict(requested=ident,effective=ident,fallback='forbid',reason='',profile_id='pocket-cpu'),profile=oracle.profile_snapshot(profile),settings=settings,settings_sha256=oracle.canonical_hash(settings))
        self.context['bundles']=[dict(instructions=False,engine='pocket-tts',model='test',mode='pocket_clone',bundle_sha256=self.spec.manifest_sha256,languages=['fr'],selectors=[])]
        self.context['execution']=dict(descriptor=self.spec.identity(),bundles={'bundle':'/models/bundles/bundle'},references={self.asset['id']:'/references/assets/'+self.asset['id']})
        self.profiles=[profile]; self.mapping={'bundle':self.bundle}; self.refs={self.asset['id']:self.core.root/'assets'/self.asset['id']}
        oracle.frozen_plan(plan,self.context['bundles'])
        self.contexts=PocketContexts({'alice':self.context},self.mapping,self.refs)

    def submit(self,key='generate'):
        c=self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        return Production(self.core).generate('alice',key,c,_admission=self.admission,_profiles=self.profiles)

    def proposal(self, *, authority=None):
        if authority is None:
            self.submit(); authority=self.store.allocate(self.profiles,'pocket',8)['authority']
        stage=self.stage/authority.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        (stage/'conditioning').mkdir(mode=0o700)  # Injected setup; no engine-side mkdir.
        self.calls=[];sent=[];outer=self
        class Backend:
            def __init__(inner,root,**kw):outer.calls.append((root,kw))
            def synthesise(inner,text,**kw):
                outer.calls.append((text,kw['reference'].read_bytes()))
                return AudioResult([0.,.5,-.5],22050)
            def close(inner):outer.calls.append('closed')
        def execute(spec,text,bundles,references,**kw):
            self.assertEqual(bundles,authority.context['execution']['bundles']);self.assertEqual(references,authority.context['execution']['references'])
            self.assertIs(kw.pop('managed_namespace'),True)
            kw['managed_owner_uid']=os.getuid()  # injected fixture ownership; installed owner is 10001
            return engine_execute(spec,text,self.mapping,self.refs,**kw)
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        with patch('lib.tts_backends.pocket_snapshot.managed_directory',return_value=nullcontext()), patch('worker.transport.authenticate') as auth,patch('worker.transport.receive',return_value=authority.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.engines.pocket.execute',side_effect=execute):
            run_generation(None,config,self.stage,time.monotonic()+5,backend_factory=Backend)
            self.assertEqual(auth.call_args.args[1].binding,authority.binding)
        return authority,sent[0],stage

    def test_saved_upload_generation_measured_rate_and_reuse_restart(self):
        authority,proposal,stage=self.proposal()
        self.assertEqual(self.calls[1],(' Exact\n texte ',self.raw));self.assertEqual(self.calls[-1],'closed')
        asset=proposal['manifest']['outputs'][0]
        self.assertEqual((asset['sample_rate'],asset['frames']),(22050,3))
        self.assertEqual(asset['provenance']['source_id'],generated_source(authority.context))
        self.core._publish(authority.job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.core.asset(asset['id'])['sha256'],asset['sha256'])
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        with self.core.transaction():self.admission.store.release_tx(authority.job['id'],confirmed_reap=True)
        root=self.core.root;self.core.close();self.core=Core(root)
        self.admission=Admission(self.core,self.stage);self.store=DispatchStore(self.core,self.admission)
        self.assertEqual(self.core.voices.version('alice',self.version['id']),self.version)
        second=self.submit('again');self.assertEqual(second,self.submit('again'))
        self.assertEqual(self.core._load(second['id'])[1]['execution']['descriptor'],self.spec.identity())

    def test_closed_identity_foreign_missing_tampered(self):
        for mutate in (lambda c:c['execution']['descriptor'].update(reference_sha256='0'*64),lambda c:c['execution']['references'].update({self.asset['id']:'/tmp/arbitrary'}),lambda c:c['execution']['descriptor'].update(revision='other',path='bad'),lambda c:c['plan']['request'].update(fallback='allow')):
            c=copy.deepcopy(self.context);mutate(c)
            with self.assertRaises(ValueError):PocketContexts({'alice':c},self.mapping,self.refs)
        with self.assertRaises(ValueError):self.contexts.resolve('bob','speech_generate',self.context['plan']['request'])
        with self.assertRaises(ValueError):Production(self.core).generate('bob','foreign',self.context,_admission=self.admission,_profiles=self.profiles)
        for path in (self.refs[self.asset['id']],self.bundle/'manifest.json',self.bundle/'model.safetensors'):
            raw=path.read_bytes();path.write_bytes(b'tampered')
            with self.assertRaises(ValueError):PocketContexts({'alice':self.context},self.mapping,self.refs)
            path.write_bytes(raw)
        self.refs[self.asset['id']].unlink()
        with self.assertRaises((ValueError,OSError)):self.submit()
        self.assertEqual(self.core.jobs('alice'),[])

    def test_cancel_reap_retains_namespace_and_refuses_stale_output(self):
        authority,proposal,stage=self.proposal()
        namespace=stage/'conditioning';self.assertEqual(namespace.stat().st_mode&0o777,0o700)
        (namespace/'descendant-sentinel').write_bytes(b'retained')
        self.core.request_cancel(authority.job['id'])
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=self.core;owner._fail(self.core.job(authority.job['id']),'cancelled')
        with self.assertRaises(ValueError):self.core._publish(authority.job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        with self.assertRaises(ValueError):
            with self.core.transaction():self.admission.store.release_tx(authority.job['id'])
        self.assertTrue((namespace/'descendant-sentinel').exists())
        with self.core.transaction():self.admission.store.release_tx(authority.job['id'],confirmed_reap=True)
        self.assertTrue((namespace/'descendant-sentinel').exists())

    def test_oracle_provenance_rate_and_unknown_namespace(self):
        authority,proposal,stage=self.proposal()
        for mutate in (lambda p:p['manifest']['outputs'][0].update(sample_rate=24000),
                       lambda p:p['manifest']['outputs'][0]['provenance'].update(source_id='unbound'),
                       lambda p:p['message'].update(input_sha256='0'*64)):
            changed=copy.deepcopy(proposal);mutate(changed)
            with self.assertRaises(ValueError):self.core._verified(self.core.job(authority.job['id']),changed,stage_root=self.stage,engine_uid=os.getuid())
        from service.reconciliation import Reconciliation
        (stage/'conditioning'/'opaque').write_bytes(b'retain')
        summary=Reconciliation(self.core,self.stage).reconcile_startup()
        self.assertTrue(summary['blocked']);self.assertIn('unknown_namespace',summary['reasons'])
        self.assertTrue((stage/'conditioning'/'opaque').exists())

    def test_empty_conditioning_persists_unknown_and_blocks_reopen(self):
        from service.reconciliation import Reconciliation
        authority, proposal, stage = self.proposal()
        namespace = stage/'conditioning'
        for nonempty in (False, True):
            if nonempty:
                (namespace/'opaque').write_bytes(b'retained')
            scan = Reconciliation(self.core, self.stage)
            summary = scan.reconcile_startup()
            row, = [r for r in scan.rows.values() if r['name']=='conditioning']
            self.assertEqual((row['classification'], row['reason']),
                             ('unknown_foreign', 'unknown_namespace'))
            self.assertTrue(summary['blocked'])
            persisted = json.loads(self.core.db.execute(
                'SELECT record FROM p1_inventory_objects WHERE id=?', (row['id'],)).fetchone()[0])
            self.assertEqual(persisted, row)
            history = json.loads(self.core.db.execute(
                'SELECT record FROM p1_inventory_history WHERE object_id=? ORDER BY seq DESC LIMIT 1',
                (row['id'],)).fetchone()[0])
            self.assertEqual(history, row)
            # Real recovery eligibility consumer, with the actual persisted row.
            # UNKNOWN cannot be acknowledged as a retained asset/namespace.
            from service.recovery import _validate_diagnostic_eligibility
            source = dict(inventory=[row], data=dict(p1_inventory_objects=[
                [row['id'], 0, 0, scan.scan_id, json.dumps(persisted)]]))
            with self.assertRaisesRegex(ValueError, 'b2_unresolved_diagnostic'):
                _validate_diagnostic_eligibility(source, dict(dispatches=[]), [])
            self.assertTrue(namespace.is_dir())
        self.assertEqual((namespace/'opaque').read_bytes(), b'retained')
        self.assertTrue(Reconciliation(self.core,self.stage).reconcile_startup()['blocked'])

    def test_interrupted_wav_yaml_leftovers_stay_unknown(self):
        from service.reconciliation import Reconciliation
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        authority,proposal,stage=self.proposal()
        for name in ('wrapper.wav','backend.wav','translated.yaml'):
            sentinel=stage/'conditioning'/name;sentinel.write_bytes(b'opaque fixture')
            with filesystem_policy(self.stage,stage):
                with self.assertRaises(OSError):remove_empty(stage.name,20000)
            self.assertEqual(sentinel.read_bytes(),b'opaque fixture')
            summary=Reconciliation(self.core,self.stage).reconcile_startup()
            self.assertTrue(summary['blocked']);self.assertIn('unknown_namespace',summary['reasons'])
        self.assertTrue(Reconciliation(self.core,self.stage).reconcile_startup()['blocked'])

    def test_qualification_and_dispatch_dependency_recheck(self):
        self.profiles[0]['verification']='synthetic'
        with self.assertRaisesRegex(ValueError,'qualification'):self.submit()
        self.profiles[0]['verification']='measured';self.submit()
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'kokoro',8)
        self.refs[self.asset['id']].write_bytes(b'tamper')
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'pocket',8)
        self.assertEqual(self.admission.store.rows()[0]['state'],'queued')

    def test_public_overrides_and_managed_owner_refuse(self):
        from jsonschema.exceptions import ValidationError
        for field in ('execution','reference_path','reference_overrides','context'):
            request=copy.deepcopy(self.context['plan']['request']);request[field]='/tmp/caller'
            with self.assertRaises(ValidationError):self.contexts.resolve('alice','speech_generate',request)
        from worker.engines.pocket import verify_inputs
        with self.assertRaisesRegex(Refused,'ownership'):
            verify_inputs(self.spec,self.mapping,self.refs,owner_uid=os.getuid()+1)
        resolved=self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        resolved['execution']['descriptor']['reference_sha256']='0'*64
        self.assertEqual(execution(self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])),self.spec)

    def test_child_limits_precede_generation_without_actual_rlimits(self):
        from worker.runtime_child import contain_kokoro, main
        import resource
        with patch.dict(os.environ),patch('resource.setrlimit') as limits:
            contain_kokoro(self.context)
            self.assertEqual(limits.call_args_list[0].args,(resource.RLIMIT_AS,(self.spec.memory_bytes,)*2))
            self.assertEqual(limits.call_args_list[1].args,(resource.RLIMIT_FSIZE,(max(self.spec.max_output_bytes,self.spec.reference_bytes),)*2))
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'2')
        calls=[]
        config=dict(fixture='pocket',context=self.context,job=dict(attempts=[dict(staging_id='stage')]))
        with patch('sys.argv',['worker','3']),patch('worker.runtime_child.socket.socket'),patch('worker.runtime_child.receive',return_value=config),patch('worker.runtime_child.contain_kokoro',side_effect=lambda c:calls.append('limits')),patch('worker.runtime_child.pocket_cutoff',return_value=100),patch('worker.runtime_child.run_generation',side_effect=lambda *a:calls.append('generation')),patch('worker.runtime_child.os.chmod'):
            main()
        self.assertEqual(calls,['limits','generation'])

    def test_supervisor_independent_mapping_and_fixture(self):
        self.submit();a=self.store.allocate(self.profiles,'pocket',8)['authority']
        from worker.runtime_supervisor import RuntimeSupervisor
        s=RuntimeSupervisor.__new__(RuntimeSupervisor);s.bundles=a.context['execution']['bundles'];s.references={}
        body=dict(snapshot=json.loads(a._snapshot),binding=a.binding,fixture='pocket')
        with self.assertRaisesRegex(Refused,'reference_mapping'):
            s.operation('prepare',body)
        s.references=a.context['execution']['references']
        with patch('worker.engines.pocket.verify_inputs',side_effect=Refused('verified-before-journal')) as verify:
            with self.assertRaisesRegex(Refused,'verified-before-journal'):s.operation('prepare',body)
            verify.assert_called_once()
        body['fixture']='kokoro'
        with self.assertRaisesRegex(Refused,'dispatch'):s.operation('prepare',body)

    def assert_closed_history(self, root, runtime, history):
        """Actual complete detached reader over the completed Pocket DB/inventory.

        Supply test enrollment and injected ownership, without changing installed
        configuration or any production reader. No historical/installed root.
        """
        import sqlite3
        from service import recovery
        from worker.runtime_journal import journal_bytes
        import shutil
        from service.reconciliation import Reconciliation
        authority=runtime.authority
        target=history.roots['control']/'core'
        # Copy only this small owned fixture's DB and explicitly named assets;
        # no production root is enrolled or altered for the reader test.
        with sqlite3.connect(target/'core.sqlite') as enrolled:
            enrollment={table:enrolled.execute('SELECT * FROM '+table).fetchall()
                        for table in ('p1_installation','p1_registrations')}
            schemas=[(n,sql) for n,sql in enrolled.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            with sqlite3.connect(root/'core.sqlite') as db:db.backup(enrolled)
            for name,sql in schemas:
                if not enrolled.execute("SELECT 1 FROM sqlite_master WHERE name=?",(name,)).fetchone():enrolled.execute(sql)
            for table,rows in enrollment.items():
                enrolled.executemany('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in rows[0])+')',rows)
            receipt=json.loads(enrolled.execute('SELECT receipt FROM dispatches').fetchone()[0])
            asset_ids=[r[0] for r in enrolled.execute('SELECT id FROM assets')]
        for identity in asset_ids:
            shutil.copyfile(root/'assets'/identity,target/'assets'/identity)
            (target/'assets'/identity).chmod(0o600)
        stage_id=authority.job['attempts'][-1]['staging_id']
        stage=history.roots['staging']/stage_id;stage.mkdir(mode=0o770)
        if (self.stage/stage_id/'output.wav').exists():
            shutil.copyfile(self.stage/stage_id/'output.wav',stage/'output.wav')
        journal=dict(version=1,lane=history.lane,records=[dict(id=self.history_runtime_id,
            allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='pocket'),
            phase='terminal',acked=True,uid=20000,boot=history.hello['boot'],receipt=receipt,proposal=None)])
        (history.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
        with patch.object(os,'getuid',return_value=10001):
            core=Core(target, _publication_creation=1)
            try:
                scan=Reconciliation(core,history.roots['staging'])
                self.assertFalse(scan.reconcile_startup()['blocked'])
                for identity,row in scan.rows.items():
                    persisted=json.loads(core.db.execute('SELECT record FROM p1_inventory_objects WHERE id=?',(identity,)).fetchone()[0])
                    latest=json.loads(core.db.execute('SELECT record FROM p1_inventory_history WHERE object_id=? ORDER BY seq DESC LIMIT 1',(identity,)).fetchone()[0])
                    self.assertEqual(row,persisted);self.assertEqual(row,latest)
            finally:core.close()
        with patch.object(os,'getuid',return_value=10001), sqlite3.connect(target/'core.sqlite',isolation_level=None) as db:
            from service import install_policy as ip
            state=recovery.snapshot(history.roots,db,history.scope,policy=ip.SPLIT)
        state=json.loads(json.dumps(state))
        recovery._installed_detached_source(state,journal)
        self.assertTrue(state['allocation_coherent'])
        self.assertTrue(recovery.installed_clean_history(state,journal))
        assert_configuration_mutations(self,state,journal)
        recovery._validate_diagnostic_eligibility(state['evidence_projection'],state['affected'],[])
        self.assertFalse(state['inventory']['reasons'])

    def test_dispatcher_generate_artifact_and_cancel_reap(self):
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation, State
        import threading
        outer=self
        def remove_namespace(stage):
            from tests.worker.test_runtime_conditioning import filesystem_policy
            from worker.runtime_conditioning import remove_empty
            helper=threading.get_ident()
            originals={name:getattr(os,name) for name in ('open','stat','fstat')}
            with filesystem_policy(outer.stage,stage):
                # Core's owner scans concurrently; only this inert helper thread
                # may see synthetic UID/mode records for the housekeeping test.
                for name,original in originals.items():
                    mocked=getattr(os,name);injected=mocked.side_effect
                    mocked.side_effect=lambda *args,_injected=injected,_original=original,**kwargs: (_injected if threading.get_ident()==helper else _original)(*args,**kwargs)
                remove_empty(stage.name,20000)
        publication_errors=[]
        original_publish=Core._publish
        def publishing(core,*args,**kwargs):
            try:return original_publish(core,*args,**kwargs)
            except (ValueError,AssertionError) as exc:
                publication_errors.append(type(exc).__name__+':'+str(exc)[:256])
                raise
        self.enterContext(patch.object(Core,'_publish',new=publishing))
        class Runtime:
            def __init__(inner):
                inner.state=State.ALIVE;inner.authority=None
                inner.finish=threading.Event();inner.stopped=threading.Event();inner.cancel=False
            def call(inner,op,body):
                return dict(quarantined=False) if op=='hello' else dict(acked=True)
            def prepare(inner,authority,fixture):
                outer.assertEqual(fixture,'pocket');inner.authority=authority
                outer.history_runtime_id=history.lane+'/'+history.hello['boot']+'/'+hashlib.sha256(authority.job['id'].encode()).hexdigest()[:32]
                return outer.history_runtime_id
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):
                return Observation(identity,State.STOPPED if inner.finish.is_set() else State.ALIVE)
            def terminate(inner,identity,timeout):inner.cancel=True;inner.stopped.set()
            def collect(inner,authority,identity):
                receipt=dict(runtime_id=identity,binding=authority.binding,reaped=True,engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=0,timely=not inner.cancel,original_deadline=authority.job['deadline'])
                stage=outer.stage/authority.job['attempts'][-1]['staging_id']
                if inner.cancel:
                    remove_namespace(stage)
                    return receipt,None
                _,proposal,stage=outer.proposal(authority=authority)
                remove_namespace(stage)
                return receipt,TransportObservation('proposal','injected-test-only',message=proposal['message'],manifest=proposal['manifest'])
        def wait(predicate):
            end=time.monotonic()+5
            while time.monotonic()<end:
                value=predicate()
                if value:return value
                time.sleep(.02)
            self.fail('bounded dispatcher wait expired')
        # Separate persistent owner root, seeded with one explicitly owned preset.
        for cancel in (False,True):
            # Enrollment is inert test support only; installed config stays Kokoro.
            from tests.service.test_installed_registration import Registration
            history=Registration();history.setUp()
            self.addCleanup(history.doCleanups)
            history.run_registration()
            base_translate=history.fixture.translate
            def engine_translate(info,path=None):
                if path is not None and path.name=='output.wav' and self.stage in path.parents:
                    return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=20000,st_gid=20000))
                return base_translate(info,path)
            history.fixture.translate=engine_translate
            # Each scenario reopens its exact completed root and staging root.
            self.stage=self.root/('stage-cancel' if cancel else 'stage-success')
            self.stage.mkdir()
            root=self.root/('dispatcher-cancel' if cancel else 'dispatcher-success')/'core'
            seed=Core(root, _publication_creation=1);self.core.db.backup(seed.db);(seed.root/'assets'/self.asset['id']).write_bytes(self.raw);seed.close()
            runtime=Runtime()
            contexts=PocketContexts({'alice':self.context},self.mapping,self.refs)
            dispatcher=Dispatcher(root,staging=self.stage,runtime=runtime,profiles=self.profiles,production_contexts=contexts)
            try:
                wait(lambda:dispatcher.call('status')['reconciled'])
                self.assertFalse(dispatcher.call('status')['integrity_fault'],dispatcher.call('status'))
                result=dispatcher.call('production_generate','alice',self.context['plan']['request'],key='cancel' if cancel else 'success')
                wait(lambda:runtime.authority is not None)
                self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                if cancel:
                    attempt_stage=self.stage/runtime.authority.job['attempts'][-1]['staging_id']
                    attempt_stage.mkdir(mode=0o770);attempt_stage.chmod(0o770)
                    namespace=attempt_stage/'conditioning';namespace.mkdir(mode=0o700)
                    dispatcher.call('production_cancel','alice',dict(job_id=result['job_id']),key='cancel-intent')
                    wait(runtime.stopped.is_set)
                    self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                    self.assertEqual(dispatcher.call('job',result['job_id'])['state'],'cancel_requested')
                    self.assertTrue(namespace.is_dir())
                runtime.finish.set()
                job=wait(lambda:(j if (j:=dispatcher.call('job',result['job_id']))['state'] in ('succeeded','cancelled','failed') else None))
                self.assertEqual(job['state'],'cancelled' if cancel else 'succeeded',dict(job=job,publication_errors=publication_errors))
                wait(lambda:dispatcher.call('status')['reservations'][-1]['state']=='retained')
                wait(lambda:dispatcher.call('status')['phase']=='idle')
                if not cancel:
                    artifact=dispatcher.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0]))
                    self.assertEqual(len(artifact['asset_ids']),1)
            finally:dispatcher.close()
            self.assert_closed_history(root, runtime, history)
            reopened_runtime=Runtime()
            reopened=Dispatcher(root,staging=self.stage,runtime=reopened_runtime,profiles=self.profiles,production_contexts=contexts)
            try:
                status=wait(lambda:(v if (v:=reopened.call('status'))['reconciled'] else None))
                self.assertFalse(status['integrity_fault'],status)
                self.assertEqual(reopened.call('job',result['job_id'])['state'],job['state'])
                self.assertEqual(reopened.call('voice_version','alice',self.version['id']),self.version)
                with self.assertRaises(ValueError):reopened.call('voice_version','bob',self.version['id'])
                self.assertFalse((self.stage/runtime.authority.job['attempts'][-1]['staging_id']/'conditioning').exists())
                if not cancel:
                    self.assertEqual(reopened.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0])),artifact)
                next_job=reopened.call('production_generate','alice',self.context['plan']['request'],key='after-reopen')
                self.assertNotEqual(next_job['job_id'],result['job_id'])
                wait(lambda:reopened_runtime.authority is not None)
                self.assertEqual(reopened.call('status')['reservations'][-1]['state'],'executing')
            finally:reopened.close()
            history.doCleanups()


class ExecutionCleanHistoryTests(unittest.TestCase):
    def setUp(self):
        from tests.service.test_installed_restart import Restart
        Restart.setUp(self)
    def history(self, typed):
        from service import recovery, host_install
        from service.core import encode
        from worker.runtime_journal import journal_bytes
        f=self.f
        if typed:
            context=copy.deepcopy(f.config['contexts']['alice']);profiles=f.config['profiles'];fixture='kokoro'
        else:
            transcript=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_bytes())
            context=transcript['context'];profiles=transcript['profiles'];fixture='silence'
            for profile in profiles:profile.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='legacy inert fixture')
        with patch.object(os,'getuid',return_value=10001):
            core=Core(f.roots['control']/'core')
            try:
                admission=Admission(core,f.roots['staging']);store=DispatchStore(core,admission)
                job=core.submit_execution('p','clean-history',context,_admission=admission,_profiles=profiles,_dispatch_settings=(fixture,8))
                row=store.allocate(profiles,fixture,8);authority=row['authority']
                runtime=f.lane+'/'+f.hello['boot']+'/'+'d'*32
                receipt=dict(runtime_id=runtime,binding=authority.binding,reaped=True,timely=False,
                    original_deadline=authority.job['deadline'],engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=0)
                job=core.job(job['id']);job.update(state='failed',stage='failed');job['attempts'][-1]['reaped']=True
                core._save(job)
                with core.transaction():admission.store.release_tx(job['id'],confirmed_reap=True)
                store.update(row,runtime=runtime,phase='acknowledged',receipt=encode(receipt),outcome=encode(job),acked=1)
                journal=dict(version=1,lane=f.lane,records=[dict(id=runtime,
                    allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture=fixture),
                    phase='terminal',acked=True,uid=20000,boot=f.hello['boot'],receipt=receipt,proposal=None)])
                (f.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
            finally:core.close()
        state=json.loads(json.dumps(host_install._registration_snapshot(f.roots,f.scope)))
        self.assertTrue(recovery.installed_clean_history(state,journal))
        assert_configuration_mutations(self,state,journal)
    def test_kokoro_exact_execution_configuration_clean_history(self):self.history(True)
    def test_legacy_effective_identity_clean_history_unchanged(self):self.history(False)


if __name__=='__main__':unittest.main()
