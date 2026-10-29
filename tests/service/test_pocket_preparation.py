"""Managed PCM -> prepared reference -> spoken audition -> immutable saved reuse.

Backend/transport/death observations are injected; real Core/SQLite/media/oracle.
No ML, network, listener, real sandbox or operational housekeeping.
"""
import copy
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import unittest
import wave
from unittest.mock import patch
from contracts import validate as oracle
from jsonschema.exceptions import ValidationError
from lib.tts_backends.contract import AudioResult
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import PocketContexts
from service.ingest import ingest_local
from service.production import Production
from worker.engines.pocket_prepare import PocketPreparation, execute, selected_reference
from worker.runtime_child import run_generation
from worker.runtime_protocol import execution
from tests.service import test_pocket_lifecycle as lifecycle


class PreparationFixture(unittest.TestCase):
    def setUp(self):
        self.base=lifecycle.PocketLifecycleTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        b=self.base
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed'))
            w.writeframes(struct.pack('<8h',101,-203,305,-407,509,-611,713,-815))
        self.raw=buffer.getvalue();self.asset=ingest_local(b.core,'alice',self.raw)
        self.spec=PocketPreparation(**dict(b.spec.identity(),reference_id=self.asset['id'],reference_sha256=self.asset['sha256'],reference_bytes=len(self.raw),reference_frames=8),in_tick=4,out_tick=12)
        self.context=copy.deepcopy(b.context);p=self.context['plan'];r=p['resolved']
        p.update(operation='voice_prepare',voice_versions=[],assets=[self.asset])
        p['request']=dict(asset_id=self.asset['id'],in_tick=4,out_tick=12,reference_text='',audition_text='  Exact audition.\nDeux phrases ! ',identity=r['identity']['effective'])
        settings=dict(kind='prepare',conditioning='pocket',output_format='wav_pcm16')
        r.update(kind='prepare',settings=settings,settings_sha256=oracle.canonical_hash(settings),reuse_settings=dict(kind='pocket_clone',language='fr',seed=None),reuse_configuration=dict(mode='pocket_clone',language='fr',reference_asset_id='pending'))
        p['reference_roles']=dict(version='reference-roles-v1',reference_kind='prepared_reference',reference_asset_id='pending',audition_asset_id='pending-audition',audition_text='  Exact audition.\nDeux phrases ! ')
        self.profiles=copy.deepcopy(b.profiles)
        self.profiles[0]['compatibility'].append(dict(operation='voice_prepare',mode='prepare',identity=r['identity']['effective']))
        r['profile']=oracle.profile_snapshot(self.profiles[0])
        self.context['execution']=dict(descriptor=self.spec.identity(),bundles={'bundle':'/models/bundles/bundle'},references={self.asset['id']:'/references/assets/'+self.asset['id']})
        self.refs={self.asset['id']:b.core.root/'assets'/self.asset['id']}
        self.contexts=PocketContexts({'alice':self.context},b.mapping,self.refs)
        self.calls=[];self.fail_audition=False;outer=self
        class Backend:
            def __init__(inner,root,**kw):outer.calls.append(('construct',root,kw))
            def synthesise(inner,text,**kw):
                outer.calls.append((text,kw['reference'].read_bytes()))
                if outer.fail_audition:raise RuntimeError('audition failed')
                return AudioResult([.25,-.25,.5],22050)
            def close(inner):outer.calls.append('closed')
        self.backend=Backend
        oracle.frozen_plan(p,self.context['bundles']);self.assertEqual(execution(self.context),self.spec)

    def submit(self,key='prepare',principal='alice',context=None):
        b=self.base
        return Production(b.core).prepare(principal,key,context or self.context,_admission=b.admission,_profiles=self.profiles)

    def proposal(self,authority=None):
        b=self.base
        if authority is None:
            self.submit();authority=b.store.allocate(self.profiles,'pocket',8)['authority']
        stage=b.stage/authority.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        (stage/'conditioning').mkdir(mode=0o700)  # Injected setup; no engine-side mkdir.
        sent=[]
        def engine(spec,text,bundles,references,**kw):
            self.assertEqual(bundles,authority.context['execution']['bundles']);self.assertEqual(references,authority.context['execution']['references'])
            self.assertIs(kw.pop('managed_namespace'),True)
            kw['managed_owner_uid']=os.getuid()
            return execute(spec,text,b.mapping,self.refs,**kw)
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        with patch('lib.tts_backends.pocket_snapshot.managed_directory',return_value=nullcontext()), patch('worker.transport.authenticate'),patch('worker.transport.receive',return_value=authority.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.engines.pocket_prepare.execute',side_effect=engine):
            run_generation(None,config,b.stage,time.monotonic()+5,backend_factory=self.backend)
        return authority,sent[0],stage


class PocketPreparationTests(PreparationFixture):
    def test_exact_segment_pair_provenance_transcript_save_restart_generation(self):
        b=self.base;a,proposal,stage=self.proposal()
        expected,frames=selected_reference(self.spec,self.raw)
        self.assertEqual(frames,4)
        with wave.open(io.BytesIO(expected)) as wav:
            self.assertEqual(wav.readframes(4),self.raw[48:56])
        self.assertEqual(self.calls[1],(self.context['plan']['reference_roles']['audition_text'],expected))
        self.assertEqual((stage/'reference.wav').read_bytes(),expected)
        ref,audition=proposal['manifest']['outputs']
        self.assertEqual((ref['sample_rate'],ref['frames'],audition['sample_rate'],audition['frames']),(24000,4,22050,3))
        self.assertNotEqual(ref['sha256'],audition['sha256'])
        roles=proposal['manifest']['candidates'][0]['reference_roles']
        self.assertEqual(roles['reference']['original'],self.asset)
        self.assertEqual((roles['reference']['in_tick'],roles['reference']['out_tick'],roles['reference']['transcript']),(4,12,''))
        self.assertEqual(ref['provenance'],dict(kind=self.asset['provenance']['kind'],source_id=self.asset['id'],source_sha256=self.asset['sha256'],missing_fields=[]))
        b.core._publish(a.job['id'],proposal,stage_root=b.stage,engine_uid=os.getuid())
        with b.core.transaction():b.admission.store.release_tx(a.job['id'],confirmed_reap=True)
        # Direct allocator seam: injected completed transport ACK, no process claim.
        b.store.update(b.store.current(),acked=1,phase='terminal')
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        with filesystem_policy(b.stage,stage):remove_empty(stage.name,20000)
        domain=Production(b.core);candidate=domain.results('alice',a.job['id'])[0]
        saved=domain.save('alice','save',a.job['id'],candidate['id'],'prepared-voice',expected_revision=0)['version']
        self.assertEqual(saved['reference_asset_ids'],[ref['id']]);self.assertEqual(saved['reference_text'],'')
        self.assertEqual(saved['provenance'],ref['provenance'])
        root=b.core.root;b.core.close();b.core=Core(root)
        b.admission=Admission(b.core,b.stage);b.store=DispatchStore(b.core,b.admission)
        self.assertEqual(Production(b.core).results('alice',a.job['id'])[0],candidate)
        self.assertEqual(b.core.voices.version('alice',saved['id']),saved)
        c=copy.deepcopy(b.context);p=c['plan'];r=p['resolved'];p['voice_versions']=[saved];p['assets']=[ref]
        p['request'].update(voice_version_id=saved['id']);r['profile']=oracle.profile_snapshot(self.profiles[0])
        spec=replace(b.spec,reference_id=ref['id'],reference_sha256=ref['sha256'],reference_bytes=len(expected),reference_frames=4)
        c['execution']=dict(descriptor=spec.identity(),bundles=c['execution']['bundles'],references={ref['id']:'/references/assets/'+ref['id']})
        b.context=c;b.spec=spec;b.profiles=self.profiles;b.refs={ref['id']:b.core.root/'assets'/ref['id']}
        b.contexts=PocketContexts({'alice':c},b.mapping,b.refs)
        generation,generated,_=b.proposal()
        self.assertEqual(b.calls[1],(' Exact\n texte ',expected))
        b.core._publish(generation.job['id'],generated,stage_root=b.stage,engine_uid=os.getuid())
        self.assertEqual(b.core.job(generation.job['id'])['state'],'succeeded')
        self.assertNotIn(audition['id'],generation.allowed_assets)

    def test_dynamic_owner_and_independent_supervisor_refusals(self):
        b=self.base
        request=copy.deepcopy(self.context['plan']['request'])
        resolve=lambda p='alice',r=request:self.contexts.resolve_owned(b.core,p,'voice_prepare',r)
        context=resolve()
        self.assertEqual(context['plan']['request'],request)
        for principal,change in [('bob',{}),('alice',{'asset_id':'absent'}),
                ('alice',{'audition_text':''}),('alice',{'audition_text':'   '}),
                ('alice',{'identity':dict(request['identity'],model='foreign')})]:
            with self.assertRaises((ValueError,OSError,ValidationError)):
                resolve(principal,dict(request,**change))
        foreign=ingest_local(b.core,'bob',self.raw)
        with self.assertRaisesRegex(ValueError,'forbidden'):
            resolve('alice',dict(request,asset_id=foreign['id']))
        path=b.core.root/'assets'/self.asset['id'];raw=path.read_bytes()
        path.write_bytes(raw[:-2]+b'xx')
        with self.assertRaises(ValueError):resolve()
        path.write_bytes(raw)
        from worker.runtime_supervisor import RuntimeSupervisor
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
        supervisor.bundles=b.mapping;supervisor.references={}
        supervisor.pocket_reference_roots={'alice':b.core.root/'assets'}
        supervisor.verify_pocket(context,'alice',owner_uid=os.getuid())
        with self.assertRaisesRegex(ValueError,'reference_mapping'):
            supervisor.verify_pocket(context,'bob',owner_uid=os.getuid())
        bad=copy.deepcopy(context);bad['execution']['references'][self.asset['id']]='/tmp/caller'
        with self.assertRaises(ValueError):supervisor.verify_pocket(bad,'alice',owner_uid=os.getuid())
        path.write_bytes(raw[:-2]+b'xx')
        with self.assertRaises(ValueError):supervisor.verify_pocket(context,'alice',owner_uid=os.getuid())
        path.write_bytes(raw);path.rename(path.with_suffix('.retained'))
        with self.assertRaisesRegex(ValueError,'missing_media'):resolve()
        with self.assertRaises(OSError):supervisor.verify_pocket(context,'alice',owner_uid=os.getuid())
        path.with_suffix('.retained').rename(path)
        profiles=copy.deepcopy(self.profiles);profiles[0]['compatibility']=[c for c in profiles[0]['compatibility'] if c['operation']!='voice_prepare']
        with self.assertRaises(ValueError):Production(b.core).prepare('alice','incompatible',context,_admission=b.admission,_profiles=profiles)
        self.assertEqual(b.core.jobs('alice'),[])
        self.assertEqual(b.admission.store.rows(),[])

    def test_refusals_before_reservation_and_closed_binding(self):
        b=self.base
        for principal,c in [('bob',self.context),('alice',dict(self.context))]:
            if principal=='alice':
                c=copy.deepcopy(c);c['plan']['request']['reference_text']='do not discard'
            with self.assertRaises(ValueError):self.submit(principal=principal,context=c)
        c=copy.deepcopy(self.context);c['plan']['request']['reference_text']='nonempty'
        with self.assertRaisesRegex(ValueError,'unsupported_parameter'):self.submit(context=c)
        with self.assertRaisesRegex(ValueError,'unsupported_parameter'):self.contexts.resolve('alice','voice_prepare',c['plan']['request'])
        for change in (dict(in_tick=1),dict(out_tick=17),dict(transform='denoise'),dict(phase='other')):
            with self.assertRaises(ValueError):replace(self.spec,**change)
        for change in (dict(reference_sha256='0'*64),dict(reference_rate=48000),dict(reference_frames=9)):
            c=copy.deepcopy(self.context);c['execution']['descriptor'].update(change)
            with self.assertRaises(ValueError):self.submit(context=c)
        for path in (self.refs[self.asset['id']],b.bundle/'model.safetensors'):
            raw=path.read_bytes();path.write_bytes(b'tampered')
            with self.assertRaises(ValueError):PocketContexts({'alice':self.context},b.mapping,self.refs)
            path.write_bytes(raw)
        for offset,fmt,value in ((22,'<H',2),(34,'<H',24)):
            raw=bytearray(self.raw);struct.pack_into(fmt,raw,offset,value)
            spec=replace(self.spec,reference_sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError,'reference_format'):selected_reference(spec,bytes(raw))
        self.assertEqual(b.core.jobs('alice'),[]);self.assertEqual(b.admission.store.rows(),[])
        self.assertEqual(self.calls,[])

    def test_explicit_audition_qualification_caps_and_supervisor_binding(self):
        b=self.base
        for mutate in (lambda c:c['plan']['request'].pop('audition_text'),
                       lambda c:c['plan']['request'].update(audition_text=' '),
                       lambda c:c['execution']['references'].update({self.asset['id']:'/tmp/caller'}),
                       lambda c:c['execution']['descriptor'].update(max_output_bytes=1048577),
                       lambda c:c['execution']['descriptor'].update(threads=3)):
            c=copy.deepcopy(self.context);mutate(c)
            with self.assertRaises(ValueError):self.submit(context=c)
        profiles=copy.deepcopy(self.profiles);profiles[0]['qualification_receipt']=''
        with self.assertRaises(ValueError):Production(b.core).prepare('alice','unqualified',self.context,_admission=b.admission,_profiles=profiles)
        self.assertEqual(b.admission.store.rows(),[])
        self.submit();a=b.store.allocate(self.profiles,'pocket',8)['authority']
        from worker.runtime_supervisor import RuntimeSupervisor
        from worker.transport import Refused
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
        supervisor.bundles=a.context['execution']['bundles'];supervisor.references={};supervisor.resident_qwen=False
        body=dict(snapshot=json.loads(a._snapshot),binding=a.binding,fixture='pocket')
        with self.assertRaisesRegex(ValueError,'reference_mapping'):supervisor.operation('prepare',body)
        supervisor.references=a.context['execution']['references']
        with patch('worker.engines.pocket.verify_inputs',side_effect=Refused('verified-before-journal')) as verify:
            with self.assertRaisesRegex(ValueError,'verified-before-journal'):supervisor.operation('prepare',body)
            self.assertEqual(verify.call_args.args[0],self.spec)
        from service.worker_transport import Authority
        changed=copy.deepcopy(a.context);changed['plan']['reference_roles']['audition_text']+='!'
        with self.assertRaisesRegex(ValueError,'audition_transcript_mismatch'):Authority(a.job,changed,a.allowed_assets,a.profiles)
        with self.assertRaisesRegex(ValueError,'audition_transcript_mismatch'):oracle.plan_hash(changed['plan'])
        changed_job=copy.deepcopy(a.job)
        changed['plan']['request']['audition_text']=changed['plan']['reference_roles']['audition_text']
        changed_job['input_sha256']=oracle.plan_hash(changed['plan'])
        self.assertNotEqual(Authority(changed_job,changed,a.allowed_assets,a.profiles).binding,a.binding)
        self.assertEqual(self.calls,[])

    def test_failed_audition_writes_no_pair_and_cancel_retains_until_reap(self):
        b=self.base;self.fail_audition=True
        with self.assertRaisesRegex(RuntimeError,'audition failed'):self.proposal()
        stage=next(b.stage.iterdir());self.assertEqual(list(stage.iterdir()),[stage/'conditioning'])
        self.assertEqual(list((stage/'conditioning').iterdir()),[])
        self.assertEqual(self.calls[-1],'closed')
        self.assertEqual(b.core.db.execute('SELECT count(*) FROM publications').fetchone()[0],0)
        b.core.request_cancel(b.core.jobs('alice')[0]['job']['id'])
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=b.core;owner._fail(b.core.jobs('alice')[0]['job'],'cancelled')
        with self.assertRaises(ValueError):
            with b.core.transaction():b.admission.store.release_tx(b.core.jobs('alice')[0]['job']['id'])
        (stage/'conditioning'/'interrupted.wav').write_bytes(b'opaque')
        with b.core.transaction():b.admission.store.release_tx(b.core.jobs('alice')[0]['job']['id'],confirmed_reap=True)
        from service.reconciliation import Reconciliation
        scan=Reconciliation(b.core,b.stage).reconcile_startup()
        self.assertTrue(scan['blocked']);self.assertIn('unknown_namespace',scan['reasons'])
        self.assertEqual((stage/'conditioning'/'interrupted.wav').read_bytes(),b'opaque')

    def test_both_outputs_measured_tamper_and_pair_atomic_failure(self):
        b=self.base;a,proposal,stage=self.proposal()
        for name in ('reference.wav','audition.wav'):
            path=stage/name;raw=path.read_bytes();path.write_bytes(raw[:-2]+b'zz')
            with self.assertRaises(ValueError):b.core._verified(b.core.job(a.job['id']),proposal,stage_root=b.stage,engine_uid=os.getuid())
            path.write_bytes(raw)
        wrong=copy.deepcopy(proposal)
        wrong['manifest']['candidates'][0]['reference_roles']['audition']['text']='invented'
        with self.assertRaises(ValueError):b.core._verified(b.core.job(a.job['id']),wrong,stage_root=b.stage,engine_uid=os.getuid())
        # Coherently rehashed fabricated reference still fails exact original PCM.
        wrong=copy.deepcopy(proposal);path=stage/'reference.wav';raw=path.read_bytes();bad=raw[:-2]+b'zz';path.write_bytes(bad)
        digest=hashlib.sha256(bad).hexdigest();wrong['manifest']['outputs'][0]['sha256']=digest
        candidate=wrong['manifest']['candidates'][0];candidate['reference_roles']['reference']['asset']['sha256']=digest
        with self.assertRaisesRegex(ValueError,'prepared_policy'):b.core._verified(b.core.job(a.job['id']),wrong,stage_root=b.stage,engine_uid=os.getuid())
        path.write_bytes(raw)
        def fault(point):
            if point=='after_link_0':raise RuntimeError('second output not adopted')
        with self.assertRaises(RuntimeError):b.core._publish(a.job['id'],proposal,stage_root=b.stage,engine_uid=os.getuid(),fault=fault)
        ids=[v['id'] for v in proposal['manifest']['outputs']]
        self.assertEqual(b.core.db.execute('SELECT count(*) FROM assets WHERE id IN (?,?)',ids).fetchone()[0],0)
        self.assertEqual(b.core.job(a.job['id'])['artifact_ids'],[])


class PocketPublicJourneyTests(PreparationFixture):
    installed_journey = False

    def journey_paths(self):
        root=self.base.root/'public';stage=self.base.root/'public-stage';stage.mkdir()
        return root,stage

    def journey_hello(self):
        return dict(quarantined=False)

    def journey_identity(self, count):
        return 'inert/'+str(count)

    def journey_supervisor(self, supervisor, authority, root):
        supervisor.pocket_reference_roots={'alice':root/'assets'}
        supervisor.verify_pocket(authority.context,authority.job['principal_id'],owner_uid=os.getuid())

    def journey_execute(self, runner, spec, text, bundles, references, kwargs):
        return runner(spec,text,bundles,references,**kwargs)

    def journey_contexts(self):
        return PocketContexts({'alice':self.context},self.base.mapping)

    def journey_owner(self, root, stage, runtime, configured):
        from service.dispatcher import Dispatcher
        return Dispatcher(root,staging=stage,runtime=runtime,profiles=self.profiles,production_contexts=configured)

    def journey_app(self, owner, credentials):
        from service.api import create_voice_app
        return create_voice_app(owner,credentials=credentials)

    def test_http_cli_mcp_dispatcher_prepare_save_generate_and_reopen(self):
        import asyncio
        import contextlib
        import httpx
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        from service.api import create_voice_app
        from service.auth import Credential, StaticCredentials
        from service.catalogue import OPERATIONS
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation, State
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        outer=self;b=self.base
        root,stage=self.journey_paths()
        def remove_namespace(attempt_stage):
            # Inject identities only on the helper thread. The real owner scan
            # runs concurrently and must see actual fixture metadata.
            import threading
            helper=threading.get_ident()
            originals={name:getattr(os,name) for name in ('open','stat','fstat')}
            with filesystem_policy(stage,attempt_stage):
                for name,original in originals.items():
                    mocked=getattr(os,name);injected=mocked.side_effect
                    mocked.side_effect=lambda *args,_injected=injected,_original=original,**kw: (_injected if threading.get_ident()==helper else _original)(*args,**kw)
                remove_empty(attempt_stage.name,20000)
        class Runtime:
            def __init__(inner):inner.authorities=[];inner.hold=False;inner.cancelled=False;inner.reaped=False
            def call(inner,op,body):return outer.journey_hello() if op=='hello' else dict(acked=True)
            def prepare(inner,a,fixture):
                outer.assertEqual(fixture,'pocket')
                from worker.runtime_supervisor import RuntimeSupervisor
                supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
                supervisor.bundles=b.mapping;supervisor.references={}
                outer.journey_supervisor(supervisor,a,root)
                inner.authorities.append(a);return outer.journey_identity(len(inner.authorities))
            def start(inner,identity):
                if inner.hold:
                    attempt_stage=stage/inner.authorities[-1].job['attempts'][-1]['staging_id']
                    attempt_stage.mkdir(mode=0o770);attempt_stage.chmod(0o770)
                    (attempt_stage/'conditioning').mkdir(mode=0o700)
                return dict(state='alive')
            def observe(inner,identity,timeout):return Observation(identity,State.ALIVE if inner.hold and not inner.reaped else State.STOPPED)
            def terminate(inner,*args):inner.cancelled=True
            def collect(inner,a,identity):
                sent=[];attempt_stage=stage/a.job['attempts'][-1]['staging_id']
                if inner.cancelled:
                    remove_namespace(attempt_stage)
                    return dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=0,timely=False,original_deadline=a.job['deadline']),None
                attempt_stage.mkdir(mode=0o770);attempt_stage.chmod(0o770)
                (attempt_stage/'conditioning').mkdir(mode=0o700)  # Injected setup boundary.
                spec=execution(a.context);refs={spec.reference_id:root/'assets'/spec.reference_id}
                from worker.engines.pocket import execute as generate
                def engine(spec,text,bundles,references,**kw):
                    outer.assertIs(kw.pop('managed_namespace'),True)
                    kw['managed_owner_uid']=os.getuid()
                    runner=execute if type(spec) is PocketPreparation else generate
                    return outer.journey_execute(runner,spec,text,b.mapping,refs,kw)
                config=dict(job=a.job,context=a.context,allowed_assets=list(a.allowed_assets),profiles=list(a.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
                with patch('lib.tts_backends.pocket_snapshot.managed_directory',return_value=nullcontext()), patch('worker.transport.authenticate'),patch('worker.transport.receive',return_value=a.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.engines.pocket_prepare.execute',side_effect=engine),patch('worker.engines.pocket.execute',side_effect=engine):
                    run_generation(None,config,stage,time.monotonic()+5,backend_factory=outer.backend)
                # Injected confirmed death, then actual exact empty-dir helper
                # filesystem logic under same-UID policy fixtures.
                remove_namespace(attempt_stage)
                receipt=dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=0,timely=True,original_deadline=a.job['deadline'])
                return receipt,TransportObservation('proposal','INJECTED ONLY',message=sent[0]['message'],manifest=sent[0]['manifest'])
        configured=self.journey_contexts()
        runtime=Runtime();owner=self.journey_owner(root,stage,runtime,configured)
        self.addCleanup(lambda current=owner: current.close() if current.thread.is_alive() else None)
        credentials=StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'})), Credential('reader','alice',frozenset({'read'})), Credential('foreign','bob',frozenset({'read','production-write'}))])
        app=self.journey_app(owner,credentials)
        def wait(predicate):
            end=time.monotonic()+5
            while time.monotonic()<end:
                result=predicate()
                if result:return result
                time.sleep(.01)
            self.fail(str(owner.call('status')))
        class ASGIClient:
            def __init__(inner,**kw):pass
            def __enter__(inner):return inner
            def __exit__(inner,*args):pass
            def request(inner,method,path,**kw):
                if kw.get('content') is not None and not isinstance(kw['content'],bytes):kw['content']=b''.join(kw['content'])
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:return await client.request(method,path,**kw)
                return asyncio.run(run())
            def get(inner,path,**kw):return inner.request('GET',path,**kw)
            def post(inner,path,**kw):return inner.request('POST',path,**kw)
        def cli(args):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):code=main(args)
            self.assertEqual(code,0,output.getvalue());return json.loads(output.getvalue())
        def tool(op,payload,key=None):
            args=dict(payload)
            if key:args['idempotency_key']=key
            result=asyncio.run(mcp.call_tool(OPERATIONS[op]['mcp'],args))
            self.assertFalse(result.isError,result.structuredContent);return result.structuredContent
        wait(lambda:owner.call('status')['reconciled'])
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer','SPEECH_UPLOAD_ROOT':b.tmp.name}):
            http=HTTP(credential='writer');path=b.root/'source.wav';path.write_bytes(self.raw)
            from service.catalogue import Failure
            self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'],[])
            caps=http.call('discover',{})
            self.assertEqual(caps['engines'],['pocket-tts'])
            self.assertIn('voice_prepare',caps['operations']);self.assertIn('job_get',caps['operations'])
            self.assertNotIn('voice_preset_register',caps['operations'])
            oracle.validate('capabilities',caps)
            self.assertFalse(http.get('/skeleton')['synthetic'])
            self.assertEqual(http.get('/skeleton')['installed'],self.installed_journey)
            self.assertEqual(http.get('/skeleton/operations/voice_preset_register')['availability'],'unavailable_configured_profile')
            voices=[]; frozen=[]; references=[]; preparations=[]
            for index,audition_text in enumerate(('  First audition.\nExact! ', 'Deuxième audition distincte.')):
                if index:
                    raw=bytearray(self.raw);raw[50:52]=b'zz';path.write_bytes(raw)
                asset=http.upload_file(path)
                request=dict(self.context['plan']['request'], asset_id=asset['id'], audition_text=audition_text)
                spec=replace(self.spec,reference_id=asset['id'],reference_sha256=asset['sha256'])
                for credential in (None,'reader','foreign'):
                    with self.assertRaises(Failure):HTTP(credential=credential).call('voice_prepare',request,key='denied')
                for change in ({'audition_text':' '},{'path':'/tmp/caller'},{'asset_id':'missing'}):
                    with self.assertRaises((Failure,ValidationError)):http.call('voice_prepare',dict(request,**change),key='invalid')
                reservations=owner.call('status')['reservations']
                bad=dict(request,reference_text='unsupported source transcript')
                result=asyncio.run(mcp.call_tool(OPERATIONS['voice_prepare']['mcp'],dict(bad,idempotency_key='bad')))
                self.assertTrue(result.isError);self.assertEqual(result.structuredContent['code'],'unsupported_control')
                self.assertIn('reference_text',json.dumps(result.structuredContent))
                self.assertEqual(owner.call('status')['reservations'],reservations)
                receipt=cli(['voice','prepare','--request',json.dumps({k:v for k,v in request.items() if k!='audition_text'}),'--audition-text',audition_text,'--idempotency-key','prep'+str(index)])
                self.assertEqual(tool('voice_prepare',request,'prep'+str(index)),receipt)
                self.assertEqual(http.call('voice_prepare',request,key='prep'+str(index)),receipt)
                preparations.append((copy.deepcopy(request),copy.deepcopy(receipt)))
                with self.assertRaises(Failure):http.call('voice_prepare',dict(request,audition_text='changed'),key='prep'+str(index))
                wait(lambda:owner.call('job',receipt['job_id'])['state']=='succeeded');wait(lambda:owner.call('status')['phase']=='idle')
                result=cli(['voice','results',receipt['job_id']]);self.assertEqual(result,tool('voice_results',dict(job_id=receipt['job_id'])))
                candidate=result['candidates'][0];ref=candidate['reference_roles']['reference']['asset'];audition=candidate['reference_roles']['audition']['asset']
                prepared,_=selected_reference(spec,path.read_bytes())
                self.assertEqual(self.calls[-2],(audition_text,prepared))
                self.assertEqual((root/'assets'/ref['id']).read_bytes(),prepared)
                self.assertEqual(http.call('asset_get',dict(asset_id=audition['id']))['sha256'],audition['sha256'])
                save=dict(job_id=receipt['job_id'],candidate_id=candidate['id'],voice_id='public-voice'+str(index),expected_revision=0)
                if index:
                    with self.assertRaises(Failure) as conflict:http.call('voice_save',dict(save,voice_id='public-voice0'),key='stale-cas')
                    self.assertEqual(conflict.exception.body['code'],'revision_conflict')
                saved=cli(['voice','save','--request',json.dumps(save),'--idempotency-key','save'+str(index)])
                self.assertEqual(saved,tool('voice_save',save,'save'+str(index)))
                with self.assertRaises(Failure):http.call('voice_save',dict(save,expected_revision=1),key='save'+str(index))
                with self.assertRaises(Failure):http.call('voice_save',save,key='cas'+str(index))
                version=saved['version'];self.assertEqual(version['reference_asset_ids'],[ref['id']])
                g=copy.deepcopy(b.context);g['plan']['request']['voice_version_id']=version['id']
                speech=cli(['speech','generate','--request',json.dumps(g['plan']['request']),'--idempotency-key','gen'+str(index)])
                self.assertEqual(speech,tool('speech_generate',g['plan']['request'],'gen'+str(index)))
                terminal=wait(lambda:(j if (j:=owner.call('job',speech['job_id']))['state'] in ('succeeded','failed','cancelled') else None))
                self.assertEqual(terminal['state'],'succeeded',dict(job=terminal,status=owner.call('status')))
                wait(lambda:owner.call('status')['phase']=='idle')
                self.assertEqual(runtime.authorities[-1].allowed_assets,(ref['id'],))
                self.assertEqual(self.calls[-2],(g['plan']['request']['text'],prepared))
                voices.append(copy.deepcopy(g['plan']['request']))
                frozen.append(runtime.authorities[-1]._snapshot)
                references.append(ref['sha256'])
            self.assertNotEqual(references[0],references[1])
            self.assertNotEqual(voices[0]['voice_version_id'],voices[1]['voice_version_id'])
            with self.assertRaises(Failure):HTTP(credential='foreign').call('speech_generate',voices[0],key='foreign-version')
            with self.assertRaises(Failure):http.call('speech_generate',dict(voices[0],voice_version_id='missing'),key='missing-version')
            for index, generation in enumerate((voices[0], voices[1], voices[0])):
                reply=http.call('speech_generate',generation,key='alternate'+str(index))
                wait(lambda:owner.call('job',reply['job_id'])['state']=='succeeded')
                wait(lambda:owner.call('status')['phase']=='idle')
            self.assertEqual([a._snapshot for a in runtime.authorities if a._snapshot in frozen],frozen)
            job=http.call('job_get',dict(job_id=speech['job_id']));artifact=cli(['artifact','get',job['artifact_ids'][0]])
            self.assertEqual(artifact,tool('artifact_get',dict(artifact_id=artifact['id'])))
            owner.close()
            owner=self.journey_owner(root,stage,Runtime(),configured)
            self.addCleanup(lambda current=owner: current.close() if current.thread.is_alive() else None);app=self.journey_app(owner,credentials)
            wait(lambda:owner.call('status')['reconciled'])
            self.assertFalse(owner.call('status')['integrity_fault'],owner.call('status'))
            self.assertEqual(http.call('voice_results',dict(job_id=receipt['job_id'])),result)
            for index,generation in enumerate(voices):
                old_request,old_receipt=preparations[index]
                self.assertEqual(http.call('voice_prepare',old_request,key='prep'+str(index)),old_receipt)
                again=http.call('speech_generate',generation,key='after-restart'+str(index))
                wait(lambda:owner.call('job',again['job_id'])['state']=='succeeded')
                wait(lambda:owner.call('status')['phase']=='idle')

            cancelled_runtime=owner.runtime
            cancelled_runtime.hold=True
            cancel_job=http.call('voice_prepare',request,key='cancel-preparation')
            wait(lambda:len(cancelled_runtime.authorities)==3)
            namespace=stage/cancelled_runtime.authorities[-1].job['attempts'][-1]['staging_id']/'conditioning'
            wait(namespace.is_dir)
            tool('job_cancel',dict(job_id=cancel_job['job_id']),'cancel-intent')
            wait(lambda:cancelled_runtime.cancelled)
            self.assertTrue(namespace.is_dir())
            self.assertEqual(owner.call('status')['reservations'][-1]['state'],'executing')
            cancelled_runtime.reaped=True
            wait(lambda:owner.call('job',cancel_job['job_id'])['state']=='cancelled')
            wait(lambda:owner.call('status')['phase']=='idle')
            self.assertFalse(namespace.exists())
            self.assertEqual(owner.call('job',cancel_job['job_id'])['artifact_ids'],[])
            owner.close()
            owner=self.journey_owner(root,stage,Runtime(),configured)
            self.addCleanup(lambda current=owner: current.close() if current.thread.is_alive() else None)
            app=self.journey_app(owner,credentials)
            wait(lambda:owner.call('status')['reconciled'])
            self.assertFalse(owner.call('status')['integrity_fault'],owner.call('status'))
            self.assertEqual(http.call('job_get',dict(job_id=cancel_job['job_id']))['state'],'cancelled')
            final=http.call('speech_generate',g['plan']['request'],key='after-cancel-restart')
            wait(lambda:owner.call('job',final['job_id'])['state']=='succeeded')
            wait(lambda:owner.call('status')['phase']=='idle')


class PocketPreparationInventoryTests(PreparationFixture):
    def test_detached_preparation_files_require_exact_typed_owner_and_namespace(self):
        from service.reconciliation import Reconciliation, typed_references
        from service import recovery
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        b=self.base;a,proposal,stage=self.proposal()
        b.core._publish(a.job['id'],proposal,stage_root=b.stage,engine_uid=os.getuid())
        with b.core.transaction():b.admission.store.release_tx(a.job['id'],confirmed_reap=True)
        b.store.update(b.store.current(),acked=1,phase='terminal')
        with filesystem_policy(b.stage,stage):remove_empty(stage.name,20000)
        scan=Reconciliation(b.core,b.stage)
        self.assertFalse(scan.reconcile_startup()['blocked'])
        data={table:[dict(zip([column[0] for column in cursor.description],r)) for r in cursor.fetchall()]
              for table in ('jobs','assets','artifacts','publications','publication_work','dispatches','dispatch_settings','resource_reservations')
              for cursor in [b.core.db.execute('SELECT * FROM '+table)]}
        recovery._installed_detached_documents(data)
        joined=typed_references(scan,data,None,None,oracle.SCHEMA)
        contexts=joined[3];owners=list(joined[5].values())
        rows=[r for r in scan.rows.values() if r['name'] in ('reference.wav','audition.wav')]
        self.assertEqual({r['name'] for r in rows},{'reference.wav','audition.wav'})
        for row in rows:
            self.assertEqual(row['classification'],'owned_synthetic_transient')
            self.assertTrue(recovery._installed_prepared_inventory(row,contexts,owners))
            for change in (dict(owner=None),dict(owner=dict(row['owner'],job_id='foreign')),
                           dict(namespace='worker_staging/foreign'),dict(namespace='assets'),
                           dict(name='conditioning'),dict(name='other.wav')):
                with self.subTest(file=row['name'],change=change):
                    self.assertFalse(recovery._installed_prepared_inventory(dict(row,**change),contexts,owners))
            alias=copy.deepcopy(row);alias['identity'][5]=2
            self.assertFalse(recovery._installed_prepared_inventory(alias,contexts,owners))
            self.assertFalse(recovery._installed_prepared_inventory(row,{},owners))
            changed=copy.deepcopy(contexts)
            changed[row['owner']['job_id']]['plan']['reference_roles']['reference_kind']='other'
            self.assertFalse(recovery._installed_prepared_inventory(row,changed,owners))

    def test_failed_prepare_plans_are_not_assets_but_original_still_required(self):
        from service.reconciliation import Reconciliation, typed_references
        from service import recovery
        b=self.base;job=self.submit()
        row=b.store.allocate(self.profiles,'pocket',8);a=row['authority']
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=b.core;owner._fail(b.core.job(job['id']),'injected-failure')
        with b.core.transaction():b.admission.store.release_tx(job['id'],confirmed_reap=True)
        b.store.update(row,acked=1,phase='terminal')
        scan=Reconciliation(b.core,b.stage)
        self.assertFalse(scan.reconcile_startup()['blocked'])
        data={table:[dict(zip([column[0] for column in cursor.description],r)) for r in cursor.fetchall()]
              for table in ('jobs','assets','artifacts','publications','publication_work','dispatches','dispatch_settings','resource_reservations')
              for cursor in [b.core.db.execute('SELECT * FROM '+table)]}
        # The detached installed history reader imports this exact shared walker
        # after its closed document pass; exercise both functions with real rows.
        recovery._installed_detached_documents(data)
        refs=typed_references(scan,data,None,None,oracle.SCHEMA)[0]
        roles=a.context['plan']['reference_roles']
        self.assertNotIn(roles['reference_asset_id'],refs);self.assertNotIn(roles['audition_asset_id'],refs)
        self.assertIn(self.asset['id'],refs)
        (b.core.root/'assets'/self.asset['id']).unlink()
        self.assertTrue(scan.reconcile_startup()['blocked'])
        self.assertTrue(any(r['name']==self.asset['id'] and r['classification']=='missing_hash_mismatched_terminal' for r in scan.rows.values()))

    def test_committed_pair_missing_output_is_never_excused(self):
        from service.reconciliation import Reconciliation
        b=self.base;a,p,stage=self.proposal()
        b.core._publish(a.job['id'],p,stage_root=b.stage,engine_uid=os.getuid())
        with b.core.transaction():b.admission.store.release_tx(a.job['id'],confirmed_reap=True)
        b.store.update(b.store.current(),acked=1,phase='terminal')
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        with filesystem_policy(b.stage,stage):remove_empty(stage.name,20000)
        output=p['manifest']['outputs'][1]['id'];(b.core.root/'assets'/output).unlink()
        scan=Reconciliation(b.core,b.stage);self.assertTrue(scan.reconcile_startup()['blocked'])
        self.assertTrue(any(r['name']==output and r['classification']=='missing_hash_mismatched_terminal' for r in scan.rows.values()))


class PocketResourceRaceTests(PreparationFixture):
    def running_namespace(self,name='conditioning'):
        b=self.base;self.submit();a=b.store.allocate(self.profiles,'pocket',8)['authority']
        stage=b.stage/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        child=stage/name;child.mkdir(mode=0o700)
        return stage,child

    def test_exact_postreap_helper_removal_interleaving_keeps_monitor_healthy(self):
        stage,child=self.running_namespace();b=self.base
        original=os.open;triggered=[]
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        def interleave(path,flags,*args,**kwargs):
            if path=='conditioning' and not triggered:
                triggered.append(True)
                with filesystem_policy(b.stage,stage):remove_empty(stage.name,20000)
                self.assertFalse(child.exists())
            return original(path,flags,*args,**kwargs)
        high=b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]
        with patch('service.resource_store.os.open',side_effect=interleave):b.admission.check()
        self.assertTrue(triggered);self.assertEqual(b.admission.store.state(),'healthy')
        self.assertEqual(b.core.db.execute('SELECT * FROM resource_faults').fetchall(),[])
        self.assertGreaterEqual(b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0],high)
        self.assertEqual(b.admission.store.rows()[0]['state'],'executing')
        with self.assertRaises(ValueError):
            with b.core.transaction():b.admission.store.release_tx(b.admission.store.rows()[0]['id'])

    def test_unexpected_directory_disappearance_still_faults(self):
        stage,child=self.running_namespace('unexpected');b=self.base;original=os.open
        def interleave(path,flags,*args,**kwargs):
            if path=='unexpected':child.rmdir()
            return original(path,flags,*args,**kwargs)
        with patch('service.resource_store.os.open',side_effect=interleave):
            with self.assertRaises(FileNotFoundError):b.admission.check()
        self.assertEqual(b.admission.store.state(),'fault')

    def test_nonempty_permission_and_substitution_are_not_waived(self):
        for scenario in ('nonempty','permission','substitute','foreign-stage-mode'):
            case=PocketResourceRaceTests();case.setUp();self.addCleanup(case.doCleanups)
            stage,child=case.running_namespace();b=case.base;original=os.open;hit=[]
            if scenario=='nonempty':(child/'opaque').write_bytes(b'retain')
            if scenario=='foreign-stage-mode':stage.chmod(0o750)
            def interleave(path,flags,*args,**kwargs):
                if path=='conditioning' and not hit:
                    hit.append(True)
                    if scenario=='permission':raise PermissionError('injected no access')
                    if scenario=='nonempty':child.rmdir() # real ENOTEMPTY, bytes retained
                    elif scenario=='substitute':
                        child.rename(stage/'retained');child.mkdir(mode=0o700)
                    else:child.rmdir()
                return original(path,flags,*args,**kwargs)
            with patch('service.resource_store.os.open',side_effect=interleave):
                with self.assertRaises((OSError,ValueError)):b.admission.check()
            self.assertEqual(b.admission.store.state(),'fault')
            if scenario=='nonempty':self.assertEqual((child/'opaque').read_bytes(),b'retain')

    def test_missing_root_and_symlink_fault(self):
        b=self.base;b.stage.rmdir()
        with self.assertRaisesRegex(ValueError,'resource_root_missing'):b.admission.check()
        self.assertEqual(b.admission.store.state(),'fault')
        other=PocketResourceRaceTests();other.setUp();self.addCleanup(other.doCleanups)
        stage,child=other.running_namespace();child.rmdir();child.symlink_to(other.base.bundle,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'resource_namespace_unknown'):other.base.admission.check()
        self.assertEqual(other.base.admission.store.state(),'fault')
