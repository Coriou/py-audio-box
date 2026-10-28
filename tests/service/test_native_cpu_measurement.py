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
            from service.native_cpu_configuration import ALIAS
            voice=d.call('voice_resolve','local',ALIAS);version=d.call('voice_version','local',voice['version_ids'][-1])
            result=d.call('production_generate','local',self.request(version),key='ordinary-no-scope')
            self.wait(lambda:d.call('job',result['job_id'])['state']=='succeeded')
            self.wait(lambda:d.call('status')['phase']=='idle')
        self.close();self.assertTrue(all(r['acked'] for r in self.s.journal.value['records']))

    def selftest_app(self):
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
        prepared=self.prepared
        info=(self.root/'owner.lock').stat();prepared.owner_identity=(info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_nlink)
        prepared.closed=False;prepared.errors=[];prepared.thread=threading.Thread(target=lambda:None)
        prepared.thread.start();prepared.thread.join()
        def owned_close():
            self.assertFalse(self.d.thread.is_alive())
            self.d=None;self.activation.release();prepared.closed=True
        prepared.close=owned_close
        def dependencies():
            return prepared,None,self.open
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

