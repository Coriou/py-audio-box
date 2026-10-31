"""Real Core/publication/ACK source fixtures, inert audio; no model qualification."""
import copy
from datetime import datetime,timezone
import json
import time
import unittest
from unittest.mock import patch
from contracts import validate as oracle
from service import native_cpu_measurement as m
from tests.service import test_native_cpu_public_source as fixtures
from worker.transport import canonical,Refused


class MeasurementTests(unittest.TestCase):
    # Reuse only the inert fixture helpers; legacy tests stay in their module.
    connect=fixtures.CPUPublicSourceTests.connect
    exchange=fixtures.CPUPublicSourceTests.exchange
    open=fixtures.CPUPublicSourceTests.open
    close=fixtures.CPUPublicSourceTests.close
    wait=fixtures.CPUPublicSourceTests.wait
    request=fixtures.CPUPublicSourceTests.request
    reopen_transport=fixtures.CPUPublicSourceTests.reopen_transport
    # Reuse fixture setup/helpers, not its test suite.
    def setUp(self):
        fixtures.CPUPublicSourceTests.setUp(self)
        # The fixture sampler has no sampling thread. Model a healthy sample
        # immediately before this inert Popen poll confirms exit, rather than
        # accidentally testing scheduler/RSS loss during expensive source proof.
        original_poll=self.fixture.h.poll
        def poll():
            active=self.s.active
            if active is not None and active.get('sampler') is not None:
                active['sampler'].stamp=time.monotonic()
            return original_poll()
        self.fixture.h.poll=poll
        state=self.root/'state';state.mkdir(mode=0o700)
        self.control.rename(state/'control');self.control=state/'control';self.prepared.control=self.control
        self.s.journal.root.rename(state/'worker');self.s.journal.root=state/'worker';self.f.root=state/'worker'
        self.prepared.root=self.root;self.prepared.identity='b'*64;self.prepared.runtime_manifest='d'*64
        self.prepared.descriptor=self.f.source.expected;self.prepared.baseline_profile=copy.deepcopy(self.f.profiles[0])
        self.observed=dict(configuration='b'*64,runtime_manifest='d'*64,profile=oracle.canonical_hash(self.prepared.baseline_profile),model=self.f.spec.manifest_sha256,
            host=dict(system='Darwin',machine='arm64',os='15.fixture',boot='fixture-boot'))
        self.session_hash=oracle.canonical_hash(self.session.value)
        from service.native_cpu_install import write
        write(self.root/'measurement-intent.json',self.session.value)
        self.binding_patch=patch.object(m,'binding',return_value=self.observed);self.binding_patch.start();self.addCleanup(self.binding_patch.stop)

    def completion(self):
        from service.native_cpu_configuration import ALIAS
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open();voice=d.call('voice_resolve','local',ALIAS)
            version=d.call('voice_version','local',voice['version_ids'][-1])
            submitted=d.call('production_generate','local',self.request(version),key='measurement-source')
            job=self.wait(lambda:(j if (j:=d.call('job',submitted['job_id']))['state']=='succeeded' else None))
            self.wait(lambda:d.call('status')['phase']=='idle')
        self.close()
        return job

    def value(self,job):
        proof=m.retained(self.root,job['id'],configuration='b'*64,runtime_manifest='d'*64,descriptor=self.prepared.descriptor,baseline=self.prepared.baseline_profile)
        return dict(version=m.VERSION,binding=copy.deepcopy(self.observed),proof=proof,elapsed_seconds=3.5,
            checked_at=datetime.now(timezone.utc).isoformat(),cleanup='owned_lifecycle_closed',session_sha256=self.session_hash)

    def test_actual_public_proof_derives_profile_without_private_scope(self):
        from worker.native_cpu_qualification import current
        job=self.completion();value=self.value(job);measured=m.validate(value,self.prepared)
        self.assertIsNone(current());self.assertEqual(measured['verification'],'measured')
        self.assertEqual(oracle.profile_snapshot(measured),oracle.profile_snapshot(self.prepared.baseline_profile))
        snapshot=self.s.journal.value['records'][0]['snapshot']
        oracle.admission(snapshot[0],snapshot[1],[measured])
        self.assertFalse(self.root.joinpath('measurement.json').exists())
        # The finite scope is absent on ordinary activation and dispatch.
        self.session=None;self.s.qualification=None;self.prepared.qualification=None
        self.f.profiles[:]=[measured];self.prepared.profiles=self.f.profiles
        self.reopen_transport();self.contexts.qualification=None
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open();self.assertEqual(d.call('configured_profiles')[0]['readiness'],'ready')
            import asyncio,httpx
            from service.api import create_voice_app
            from service.auth import Credential,StaticCredentials
            app=create_voice_app(d,credentials=StaticCredentials([Credential('fixture-only','local',frozenset({'read','production-write'}))]),native_installed=True)
            async def public_status():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://fixture') as client:
                    return (await client.get('/v1/status',headers={'Authorization':'Bearer fixture-only'})).json()
            status=asyncio.run(public_status());self.assertTrue(status['ready_for_speech'])
            self.assertEqual(status['qualification'],'local_selftest');self.assertIsNone(status['configured_capabilities']['qualification_session'])
            from service.native_cpu_configuration import ALIAS
            voice=d.call('voice_resolve','local',ALIAS);version=d.call('voice_version','local',voice['version_ids'][-1])
            result=d.call('production_generate','local',self.request(version),key='ordinary-no-scope')
            self.wait(lambda:d.call('job',result['job_id'])['state']=='succeeded')
            self.wait(lambda:d.call('status')['phase']=='idle')
        self.close();self.assertTrue(all(r['acked'] for r in self.s.journal.value['records']))

    def selftest_app(self,*,remove_alias=False):
        # Source fixture: real HTTP routes/Core/mTLS adapter/encoder, inert model
        # and process handles. This cannot qualify an actual installed host.
        import asyncio
        import threading
        from service.native_cpu_install import write
        from service.auth import Credential,StaticCredentials
        from service.api import create_voice_app
        from service.entrypoint import _owned_app
        write(self.root/'credential.json',dict(credentials=[dict(token='fixture-only',principal='local',scopes=['read','production-write'])]))
        write(self.root/'owner.lock',b'')
        prepared=self.prepared;prepared.selftest=True
        info=(self.root/'owner.lock').stat();prepared.owner_identity=(info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_nlink)
        prepared.closed=False;prepared.errors=[];prepared.thread=threading.Thread(target=lambda:None)
        prepared.thread.start();prepared.thread.join()
        def owned_close():
            self.assertFalse(self.d.thread.is_alive())
            self.d=None;self.activation.release();prepared.closed=True
        prepared.close=owned_close
        def acquire():
            owner=self.open()
            if remove_alias:
                from service.native_cpu_configuration import VOICE
                voice=owner.call('voice_get','local',VOICE)
                owner.call('production_aliases','local',dict(voice_id=VOICE,aliases=[],expected_revision=voice['revision']),key='fixture-remove-alias')
            return owner
        def dependencies():
            return prepared,None,acquire
        def mount(owner):
            return create_voice_app(owner,credentials=StaticCredentials([Credential('fixture-only','local',frozenset({'read','production-write'}))]),native_installed=True)
        return _owned_app(dependencies,mount)

    def test_doctor_selftest_actual_asgi_routes_publish_only_after_owned_close(self):
        import asyncio
        from service.native_cpu_selftest import run
        prepared=self.prepared;app=self.selftest_app()
        with patch('service.native_cpu_setup.create_app',return_value=app),patch.object(self.s,'exchange',side_effect=self.exchange):
            result=asyncio.run(run(self.root))
        self.assertTrue(result['ready_for_speech']);self.assertTrue(prepared.closed)
        self.assertEqual(m.load(prepared)['verification'],'measured')
        self.assertEqual(len(self.s.journal.value['records']),1)

    def test_selftest_uses_owned_voice_when_default_alias_removed(self):
        import asyncio
        from service.native_cpu_selftest import run
        app=self.selftest_app(remove_alias=True)
        with patch('service.native_cpu_setup.create_app',return_value=app),patch.object(self.s,'exchange',side_effect=self.exchange):
            result=asyncio.run(run(self.root))
        self.assertTrue(result['ready_for_speech']);self.assertEqual(result['history_jobs_used'],1)
        self.assertEqual(result['history_jobs_remaining'],62)
        self.assertEqual(len(self.s.journal.value['records']),1)

    def test_cancelled_selftest_reaps_and_never_publishes_measurement(self):
        import asyncio
        from service.native_cpu_selftest import run
        self.finish=False;app=self.selftest_app()
        async def exercise():
            task=asyncio.create_task(run(self.root));deadline=asyncio.get_running_loop().time()+4
            while self.s.active is None:
                if task.done():await task
                if asyncio.get_running_loop().time()>deadline:self.fail('source child did not start')
                await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        with patch('service.native_cpu_setup.create_app',return_value=app),patch.object(self.s,'exchange',side_effect=self.exchange):
            asyncio.run(exercise())
        self.assertTrue(self.prepared.closed);self.assertIsNone(self.s.active)
        self.assertFalse((self.root/'measurement.json').exists())
        record=self.s.journal.value['records'][0]
        self.assertTrue(record['receipt']['reaped']);self.assertTrue(record['acked']);self.assertFalse(record['receipt']['timely'])

    def test_crash_pending_each_recovers_private_regular_leaves_before_job(self):
        from service.native_cpu_install import write
        original=(self.root/'measurement-intent.json').read_bytes()
        for leaf in ('measurement-intent.pending','measurement.pending'):
            write(self.root/leaf,b'incomplete crash bytes')
            self.assertEqual(m.recover_pending(self.root),[leaf])
            self.assertFalse((self.root/leaf).exists())
            self.assertEqual(m.history(self.root)['history_jobs_used'],0)
        self.assertEqual((self.root/'measurement-intent.json').read_bytes(),original)

    def test_pending_symlink_owner_mode_and_replacement_refuse_without_job(self):
        import os
        from types import SimpleNamespace
        from service.native_cpu_install import Prepared
        from service.native_cpu_configuration import expected
        outside=self.root.parent/'untouched-outside';outside.write_bytes(b'outside');outside.chmod(0o444)
        path=self.root/'measurement.pending';path.symlink_to(outside)
        (self.root/'state'/'spool').mkdir(mode=0o700)
        (self.root/'state'/'staging').mkdir(mode=0o770);(self.root/'state'/'staging').chmod(0o770)
        config=dict(source_files={},python_sha256='e'*64,python_version='fixture',dependencies={},
            deployment_id='a'*32,installed_at=datetime.now(timezone.utc).isoformat(),principal='local')
        with patch('service.native_cpu_install.read_installation',return_value=(self.root,config,expected(),self.root/'unused')),                patch.object(Prepared,'_prepare') as start:
            with self.assertRaisesRegex(Refused,'pending_unsafe:measurement.pending'):Prepared(self.root,_selftest=True)
            start.assert_not_called()
        self.assertTrue(path.is_symlink());self.assertEqual(outside.read_bytes(),b'outside');self.assertEqual(outside.stat().st_mode&0o777,0o444)
        path.unlink();path.write_bytes(b'crash');path.chmod(0o644)
        with self.assertRaisesRegex(Refused,'pending_unsafe:measurement.pending'):m.recover_pending(self.root)
        path.chmod(0o600);original=os.stat
        def changed(name,*args,**kwargs):
            value=original(name,*args,**kwargs)
            if name=='measurement.pending':return SimpleNamespace(st_mode=value.st_mode,st_uid=os.getuid()+1,st_nlink=1,st_size=value.st_size)
            return value
        with patch.object(m.os,'stat',side_effect=changed),self.assertRaisesRegex(Refused,'pending_unsafe:measurement.pending'):m.recover_pending(self.root)
        self.assertTrue(path.exists());self.assertEqual(m.history(self.root)['history_jobs_used'],0)

    def test_history_counter_reports_actual_used_remaining_and_full_recovery(self):
        import sqlite3
        path=self.control/'core.sqlite'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE jobs(id INTEGER)');db.executemany('INSERT INTO jobs VALUES (?)',[(i,) for i in range(63)])
        counts=m.history(self.root)
        self.assertEqual((counts['history_jobs_used'],counts['history_jobs_remaining'],counts['history_jobs_capacity']),(63,0,63))
        self.assertIn('new directory',counts['next_action']);self.assertIn('acquired again',counts['next_action'])
        self.assertEqual(m.history(self.root),counts)

    def test_doctor_drift_is_structured_and_retains_causal_diagnostic(self):
        import contextlib,io
        from service.native_cpu_setup import main
        cause=ValueError('private-value-must-not-appear')
        error=Refused('native_cpu_measurement_revalidation_required');error.__cause__=cause
        output=io.StringIO()
        with patch('service.native_cpu_install.Prepared',side_effect=error),contextlib.redirect_stdout(output):
            self.assertEqual(main(['doctor','--root',str(self.root)]),1)
        result=json.loads(output.getvalue());self.assertFalse(result['ready_for_speech'])
        self.assertEqual(result['status'],'revalidation_required');self.assertEqual(result['history_jobs_used'],0)
        self.assertIn('doctor --selftest',result['next_action']);self.assertIn('ValueError',json.dumps(result['error']))
        self.assertNotIn('private-value',output.getvalue())

    def test_forged_stale_binding_and_invalid_measurements_refused(self):
        job=self.completion();value=self.value(job)
        mutations=[lambda v:v.update(cleanup='unknown'),lambda v:v.update(elapsed_seconds=0),lambda v:v.update(elapsed_seconds=91),
            lambda v:v.update(elapsed_seconds=float('nan')),lambda v:v.update(session_sha256='0'*64),
            lambda v:v['proof']['wav'].update(sha256='0'*64),lambda v:v['proof'].update(terminal_sha256='0'*64),
            lambda v:v.update(checked_at='2999-01-01T00:00:00+00:00')]
        for field in ('configuration','runtime_manifest','profile','model'):
            mutations.append(lambda v,f=field:v['binding'].update({f:'0'*64}))
        for field in ('boot','os','machine'):
            mutations.append(lambda v,f=field:v['binding']['host'].update({f:'changed'}))
        for mutate in mutations:
            bad=copy.deepcopy(value);mutate(bad)
            with self.subTest(value=bad),self.assertRaises((Refused,ValueError)):m.validate(bad,self.prepared)

    def test_before_ack_partial_publication_and_changed_wav_refused(self):
        import sqlite3
        job=self.completion();value=self.value(job)
        path=self.control/'core.sqlite'
        with sqlite3.connect(path) as db:db.execute('UPDATE dispatches SET acked=0')
        with self.assertRaises(Refused):m.validate(value,self.prepared)
        with sqlite3.connect(path) as db:
            db.execute('UPDATE dispatches SET acked=1');db.execute('DELETE FROM publications')
        with self.assertRaises(Refused):m.validate(value,self.prepared)

    def test_changed_pcm_or_unknown_custody_cannot_issue_receipt(self):
        job=self.completion();value=self.value(job)
        self.prepared.closed=False
        with self.assertRaises(Refused):m.publish(self.prepared,value)
        self.assertFalse((self.root/'measurement.json').exists())
        asset=next((self.control/'assets').iterdir())
        raw=asset.read_bytes();asset.chmod(0o600);asset.write_bytes(raw[:-2]+b'\x00\x00')
        with self.assertRaises((Refused,ValueError)):m.validate(value,self.prepared)

    def test_no_measurement_is_not_ready_and_failed_new_session_invalidates_old(self):
        self.assertIsNone(m.load(self.prepared))
        job=self.completion();value=self.value(job)
        from service.native_cpu_install import write
        write(self.root/'measurement.json',value)
        self.assertEqual(m.load(self.prepared)['readiness'],'ready')
        intent=copy.deepcopy(self.session.value);intent['session_id']='2'*32
        path=self.root/'measurement-intent.json';path.write_bytes(canonical(intent))
        with self.assertRaises(Refused):m.load(self.prepared)
