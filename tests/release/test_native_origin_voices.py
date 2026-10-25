"""Origin families through public clients; inert TLS, backend and owned handles."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from clients.http import HTTP
from clients import mcp
from contracts.scenarios import preset_registration
from service import native_launch, native_renewal as r
from tests.release.test_native_managed_clone import ManagedCloneTests
from tests.release.test_native_onboarding import NativeOnboardingTests
from tests.release.test_native_renewal import RenewalCompletionTests
from worker import native_protocol as p
from worker.transport import canonical, Refused


class RenewalPublic:
    successor_inputs = RenewalCompletionTests.successor_inputs

    def renew_public(self, epoch):
        # Reuse the owning transaction fixture with the actual public owner root.
        root = self.root
        try:
            self.root = self.owner
            old, new = RenewalCompletionTests.renew(self, epoch)
        finally:
            self.root = root
        def before(q):
            if q['op'] in ('observe','collect'):
                if self.f.s.active:self.f.finish()
                self.f.s.tick()
        self.f.before=before
        self.assertTrue((old.journal.root/'retirement').exists())
        return native_launch.create_native_app(record_file=str(self.record),
            configuration_file=self.kw['configuration_file'],templates_file=self.kw['templates_file'],
            owner_root=str(self.owner),staging=str(self.f.stage),credential_file=str(self.credential))

    def complete(self, app, http, receipt):
        job=self.wait(lambda:(j if (j:=http.call('job_get',dict(job_id=receipt['job_id'])))['state'] in ('succeeded','failed') else None))
        self.assertEqual(job['state'],'succeeded',job)
        self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
        return job

    def verify_origin_refusals(self):
        original,current,mapping=r.origin_authorization(self.owner)
        for mutate in (lambda a,b,m:m[0].update(origin_sha256='0'*64),
                       lambda a,b,m:m[0].update(principal='bob'),
                       lambda a,b,m:b[0]['context']['execution']['native'].update(runtime_manifest_sha256='0'*64),
                       lambda a,b,m:b[0]['context']['plan']['resolved']['settings'].update(seed=True)):
            a,b,m=copy.deepcopy((original,current,mapping));mutate(a,b,m)
            # Coherent current digest forgery still cannot change origin semantics.
            m[0]['current_sha256']=r.digest(b[0])
            with self.assertRaises((Refused,ValueError)):p.origin_template_pairs(a,b,m)
        with self.assertRaises(Refused):p.origin_template_pairs(original+original,current+current,mapping+mapping)
        bad=copy.deepcopy(self.f.s.journal.value['records'][-1]['snapshot'])
        bad[1]['execution']['native']['boot']='0'*32
        before=self.f.s.launches
        with self.assertRaises(Refused):self.f.s.register_snapshot(bad)
        self.assertEqual(before,self.f.s.launches)
        registered=self.f.s._preset_templates
        self.f.s._preset_templates=dict(registered, duplicate=copy.deepcopy(next(iter(registered.values()))))
        try:
            snapshot=self.f.s.journal.value['records'][-1]['snapshot']
            with self.assertRaises(Refused):self.f.s.register_snapshot(snapshot)
        finally:self.f.s._preset_templates=registered
        from types import SimpleNamespace
        from worker.native_supervisor import NativeSupervisor
        retained=copy.deepcopy(self.f.s.journal.retained)
        retained['current_templates'][0]['principal']='bob'
        # Even a recomputed mapping cannot substitute another provisioned principal.
        retained['origins'][0]['current_sha256']=r.digest(retained['current_templates'][0])
        for coherent in (False, True):
            if coherent:
                retained['origin_templates'][0]['principal']='bob'
                retained['origins'][0].update(principal='bob',origin_sha256=r.digest(retained['origin_templates'][0]))
            with patch('worker.native_supervisor.Journal',return_value=SimpleNamespace(retained=retained)):
                with self.assertRaises(Refused):
                    NativeSupervisor(self.root/'unused-inert-host',deployment_id=self.f.native['deployment_id'],
                        configuration_sha256=self.f.native['configuration_sha256'],peer_pin=self.f.s.pin,
                        runtime_manifest_sha256=self.f.s.manifest,registrations=registered,
                        python=self.f.s.python,child_entry=self.f.s.entry)


class OriginCloneTests(RenewalPublic, ManagedCloneTests):
    def journey(self, terminal):
        self.public_journey(terminal=terminal)
        import sqlite3
        def immutable():
            with sqlite3.connect(self.owner/'core.sqlite') as db:
                return db.execute('SELECT * FROM jobs ORDER BY id').fetchall(), db.execute('SELECT * FROM dispatches ORDER BY binding').fetchall()
        old_jobs, old_authorities = immutable()
        old_assets={path:path.read_bytes() for path in (self.owner/'assets').iterdir() if path.is_file()}
        old_record=self.record.read_bytes()
        old_version=canonical(self.saved['version'])
        for epoch in (1,2):
            app=self.renew_public(epoch)
            async def run():
                async with app.router.lifespan_context(app):await asyncio.to_thread(check)
            def check():
                with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                    http=HTTP(credential='x'*64)
                    launches=self.f.s.launches
                    self.assertEqual(http.call('speech_generate',self.generate,key='generate'),self.generated)
                    self.assertEqual(http.call('voice_prepare',self.request,key='prepare'),self.prep)
                    self.assertEqual(http.call('voice_save',self.save,key='save'),self.saved)
                    self.assertEqual(launches,self.f.s.launches)
                    self.assertEqual(canonical(http.call('voice_version_get',dict(voice_version_id=self.saved['version']['id']))),old_version)
                    result=self.invoke_cli(['speech','generate','--request',json.dumps(self.generate),'--idempotency-key','renew'+str(epoch)])
                    job=self.complete(app,http,result)
                    tool=asyncio.run(mcp.call_tool('speech_speech_generate',dict(self.generate,idempotency_key='renew'+str(epoch))))
                    self.assertFalse(tool.isError,tool.structuredContent);self.assertEqual(tool.structuredContent,result)
                    with sqlite3.connect(self.owner/'core.sqlite') as db:
                        context=json.loads(db.execute('SELECT context FROM jobs WHERE id=?',(job['id'],)).fetchone()[0])
                    self.assertEqual(context['execution']['native'],self.f.native)
                    self.assertEqual(canonical(context['plan']['voice_versions'][0]),old_version)
                    if epoch == 1:
                        self.f.f.h=type(self.f.f.h)()
                        prepared=http.call('voice_prepare',self.request,key='fresh-prepare')
                        self.complete(app,http,prepared)
                        candidates=http.call('voice_results',dict(job_id=prepared['job_id']))
                        candidate=candidates[0] if isinstance(candidates,list) else candidates['candidates'][0]
                        saved=http.call('voice_save',dict(job_id=prepared['job_id'],candidate_id=candidate['id'],voice_id='new.clone',expected_revision=0),key='fresh-save')
                        self.assertEqual(saved['version']['reference_text'],self.saved['version']['reference_text'])

            with self.connector(),patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):asyncio.run(run())
            self.verify_origin_refusals()
            jobs, authorities = immutable()
            self.assertTrue(set(old_jobs) <= set(jobs))
            self.assertTrue(set(old_authorities) <= set(authorities))
        self.assertEqual((self.root/'enrollment').read_bytes(),old_record)
        self.assertTrue(all(path.read_bytes()==raw for path,raw in old_assets.items()))

    def test_cancelled_clone_two_renewals(self):self.journey('cancelled')
    def test_failed_clone_two_renewals(self):self.journey('failed')


class OriginPresetTests(RenewalPublic, NativeOnboardingTests):
    def test_public_preset_stable_across_two_renewals(self):
        app=self.prepare();request=preset_registration(self.f.f.e.preset,'stable.voice')
        saved=None;receipt=None
        for epoch in (0,1,2):
            if epoch:app=self.renew_public(epoch)
            def check():
                nonlocal saved,receipt
                with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                    http=HTTP(credential='x'*64);launches=self.f.s.launches
                    value=http.call('voice_preset_register',request,key='preset')
                    if saved is None:saved=value
                    self.assertEqual(saved,value)
                    payload=dict(self.f.snapshot[1]['plan']['request'],voice_version_id=saved['version']['id'])
                    if receipt:self.assertEqual(http.call('speech_generate',payload,key='initial'),receipt)
                    self.assertEqual(launches,self.f.s.launches)
                    new=http.call('speech_generate',payload,key='initial' if epoch==0 else 'fresh'+str(epoch))
                    if receipt is None:receipt=new
                    self.complete(app,http,new)
                    self.assertEqual(http.call('voice_preset_register',dict(request,voice_id='new.voice'),key='new') ['version']['id'],
                        app.state.dispatcher.production_contexts.preset_version('alice',request['preset'],'new.voice')['id'])
            async def run():
                async with app.router.lifespan_context(app):await asyncio.to_thread(check)
            with self.connector(),patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):asyncio.run(run())
            if epoch:self.verify_origin_refusals()
