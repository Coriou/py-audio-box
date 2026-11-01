"""Finite public SOURCE journeys, injected freshness clock, inert provider/TLS/ML."""
import asyncio
import copy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import json
import sqlite3
import unittest
from unittest.mock import patch

from clients.http import HTTP
from clients import mcp
from contracts.scenarios import preset_registration
from service import native_launch
from tests.release import test_native_onboarding as presets
from tests.release import test_native_managed_clone as clones
from tests.release import test_native_enrollment as enrollment
from worker.transport import canonical, Refused


class LiveJourney:
    def instrument(self):
        self.now=datetime.now(timezone.utc);self.provider_calls=0
        cache=self.f.s.availability
        cache.utc=lambda:self.now
        def provider(enrolled):
            self.provider_calls+=1
            return replace(enrolled,devices=tuple(replace(d,available=True,probed_at=self.now.isoformat()) for d in enrolled.devices))
        cache.provider=provider
        # Factories used on control reopen receive the same injected clock.
        from service.native_runtime import NativeRuntime
        saved=NativeRuntime.__init__
        def init(runtime,*args,**kwargs):
            kwargs['utc']=lambda:self.now
            saved(runtime,*args,**kwargs)
        patcher=patch.object(NativeRuntime,'__init__',init);patcher.start();self.addCleanup(patcher.stop)

    def reopen_app(self):
        return native_launch.create_native_app(record_file=str(self.record),
            configuration_file=str(self.root/'control'),templates_file=str(self.root/'templates'),
            owner_root=str(self.owner),staging=str(self.f.stage),credential_file=str(self.credential))

    def immutable(self):
        with sqlite3.connect(self.owner/'core.sqlite') as db:
            rows={table:db.execute('SELECT * FROM '+table).fetchall() for table in
                  ('jobs','dispatches','resource_reservations','catalogue_versions','publications')}
        files={p:p.read_bytes() for root in (self.owner/'assets',self.f.spool,self.f.stage)
               for p in root.rglob('*') if p.is_file()}
        return rows,files

    def retained(self, before):
        rows,files=before
        with sqlite3.connect(self.owner/'core.sqlite') as db:
            for table,old in rows.items():
                current=db.execute('SELECT * FROM '+table).fetchall()
                self.assertTrue(all(row in current for row in old),table)
        for path,raw in files.items():self.assertEqual(path.read_bytes(),raw,str(path))

    def complete(self,app,http,receipt):
        job=self.wait(lambda:(j if (j:=http.call('job_get',dict(job_id=receipt['job_id'])))['state'] in ('succeeded','failed') else None))
        self.assertEqual(job['state'],'succeeded',job)
        self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
        return job

    def run_app(self,app,check):
        async def run():
            async with app.router.lifespan_context(app):await asyncio.to_thread(check)
        with self.connector(),patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):asyncio.run(run())
        self.assertFalse(app.state.dispatcher.thread.is_alive())


class LivePresetTests(LiveJourney,unittest.TestCase):
    setUp=presets.NativeOnboardingTests.setUp
    write=presets.NativeOnboardingTests.write
    enroll=presets.NativeOnboardingTests.enroll
    app=presets.NativeOnboardingTests.app
    prepare=presets.NativeOnboardingTests.prepare
    connector=presets.NativeOnboardingTests.connector
    bridge=presets.NativeOnboardingTests.bridge
    wait=presets.NativeOnboardingTests.wait
    invoke_cli=presets.NativeOnboardingTests.invoke_cli

    def test_first_ever_preset_after_nine_windows_same_boot_reopen_and_identity(self):
        app=self.prepare();self.instrument();self.now+=timedelta(seconds=2701)
        template=(self.root/'templates').read_bytes();enrolled=self.record.read_bytes();boot=self.f.s.boot
        def first():
            with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                http=HTTP(credential='x'*64)
                self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'],[])
                registered=self.invoke_cli(['voice','preset-register','--request',json.dumps(preset_registration(self.f.f.e.preset,'elapsed')),'--idempotency-key','first'])
                self.payload=dict(self.f.snapshot[1]['plan']['request'],voice_version_id=registered['version']['id'])
                self.old=http.call('speech_generate',self.payload,key='old');self.complete(app,http,self.old)
                self.before=self.immutable();self.measurement=canonical(sorted(app.state.dispatcher.production_contexts._contexts.values()))
                count=self.provider_calls;self.now+=timedelta(seconds=1501)
                # Discovery is cached metadata only, including after expiry.
                http.get('/skeleton');self.assertEqual(self.provider_calls,count)
                self.f.f.h=type(self.f.f.h)()
                result=asyncio.run(mcp.call_tool('speech_speech_generate',dict(self.payload,idempotency_key='later')))
                self.assertFalse(result.isError,result.structuredContent);self.complete(app,http,result.structuredContent)
                self.retained(self.before)
                self.assertEqual(self.measurement,canonical(sorted(app.state.dispatcher.production_contexts._contexts.values())))
        self.run_app(app,first)
        self.now+=timedelta(seconds=1501);app=self.reopen_app()
        def reopened():
            with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                http=HTTP(credential='x'*64)
                self.assertEqual(http.call('speech_generate',self.payload,key='old'),self.old)
                count=self.provider_calls;before=self.immutable()
                http.call('speech_generate',self.payload,key='old')
                self.assertEqual(count,self.provider_calls);self.retained(before)
                self.f.f.h=type(self.f.f.h)()
                result=self.invoke_cli(['speech','generate','--request',json.dumps(self.payload),'--idempotency-key','reopened'])
                self.complete(app,http,result);self.retained(self.before)
        self.run_app(app,reopened)
        self.assertEqual(self.provider_calls,3);self.assertEqual(self.f.s.launches,3)
        self.assertEqual(boot,self.f.s.boot);self.assertIsNone(self.f.s.journal.retained)
        self.assertFalse((self.owner/'native-renewal').exists())
        self.assertEqual(template,(self.root/'templates').read_bytes());self.assertEqual(enrolled,self.record.read_bytes())


class LiveCloneTests(LiveJourney,unittest.TestCase):
    setUp=clones.ManagedCloneTests.setUp
    write=clones.ManagedCloneTests.write
    enroll=clones.ManagedCloneTests.enroll
    app=clones.ManagedCloneTests.app
    prepare=clones.ManagedCloneTests.prepare
    connector=clones.ManagedCloneTests.connector
    bridge=clones.ManagedCloneTests.bridge
    wait=clones.ManagedCloneTests.wait
    invoke_cli=clones.ManagedCloneTests.invoke_cli
    public_journey=clones.ManagedCloneTests.public_journey

    def test_saved_clone_elapsed_refresh_preserves_origin_media_measurement_and_charges(self):
        self.instrument();self.public_journey()
        before=self.immutable();template=(self.root/'templates').read_bytes();enrolled=self.record.read_bytes();boot=self.f.s.boot
        for index in range(2):
            self.now+=timedelta(seconds=1501);app=self.reopen_app()
            def check():
                with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                    http=HTTP(credential='x'*64)
                    self.assertEqual(http.call('speech_generate',self.generate,key='generate'),self.generated)
                    self.assertEqual(http.call('voice_version_get',dict(voice_version_id=self.saved['version']['id'])),self.saved['version'])
                    # Execution-bound measurement digest remains the immutable context digest.
                    old_context=copy.deepcopy(app.state.dispatcher.production_contexts._contexts)
                    self.f.f.h=type(self.f.f.h)()
                    self.complete(app,http,http.call('speech_generate',self.generate,key='elapsed'+str(index)))
                    self.assertEqual(old_context,app.state.dispatcher.production_contexts._contexts)
                    self.retained(before)
            self.run_app(app,check)
        self.assertEqual(self.provider_calls,3);self.assertEqual(self.f.s.launches,4)
        self.now+=timedelta(seconds=301)
        self.f.s.availability.provider=lambda enrolled: (_ for _ in ()).throw(OSError('inert unavailable'))
        app=self.reopen_app()
        def history_only():
            from service.catalogue import Failure
            with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                http=HTTP(credential='x'*64)
                self.assertEqual(http.call('speech_generate',self.generate,key='generate'),self.generated)
                http.call('job_get',dict(job_id=self.generated['job_id']))
                http.call('artifact_get',dict(artifact_id=self.generated_job['artifact_ids'][0]))
                with self.assertRaises(Failure):http.call('speech_generate',self.generate,key='unavailable')
                self.retained(before)
        self.run_app(app,history_only)
        self.assertEqual(self.f.s.launches,4)
        self.assertEqual(boot,self.f.s.boot);self.assertFalse((self.owner/'native-renewal').exists())
        self.assertEqual(template,(self.root/'templates').read_bytes());self.assertEqual(enrolled,self.record.read_bytes())


class LiveActivationTests(unittest.TestCase):
    setUp=enrollment.NativeEnrollmentTests.setUp
    write=enrollment.NativeEnrollmentTests.write
    enroll=enrollment.NativeEnrollmentTests.enroll

    def test_activation_rejects_failed_current_association_but_preserves_enrollment(self):
        from service import native_package as n
        self.enroll();before=self.record.read_bytes()
        self.f.s.availability.provider=lambda enrolled: (_ for _ in ()).throw(OSError('inert'))
        values=n.native_startup_inputs(self.record,**self.kw)
        request=self.templates['registrations'][0]['context']['plan']['request']
        with self.assertRaises(Refused):values['production_contexts'].resolve('alice','speech_generate',request)
        self.assertEqual(self.record.read_bytes(),before);self.assertEqual(self.f.s.launches,0)
