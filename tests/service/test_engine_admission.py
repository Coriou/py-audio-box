"""Injected source fixtures only: no installed qualification or model inference."""
import copy
import json
from pathlib import Path
import sqlite3
import unittest
from contracts import validate as oracle
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import KokoroContexts, verify_local
from service.production import Production
from service.provisioning import identity, manifest_sha256
from tests.service import test_provisioning as provisioning_fixtures
from worker.engines.execution import Execution
from worker.runtime_protocol import execution


def configuration(expected):
    context = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    plan = context['plan']; r = plan['resolved']
    ident = dict(engine='kokoro', model=expected['checkpoint'], bundle_sha256=manifest_sha256(expected), device='cpu', precision='float32')
    settings = dict(kind='kokoro_preset', language='en')
    version = plan['voice_versions'][0]
    version.update(identity=ident, settings=settings, settings_sha256=oracle.canonical_hash(settings), configuration=dict(mode='kokoro_preset', preset='af_heart', language='en'))
    plan['request'].update(identity=ident, text=' Exact\n authored text ')
    profile = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['profiles'][0]
    profile.update(id='kokoro-cpu', support='qualified', installation='installed', readiness='ready', verification='measured', qualification_receipt='INJECTED-TEST-ASSERTION-NOT-ACTUAL-EVIDENCE')
    profile['compatibility'] = [dict(operation='speech_generate', mode='kokoro_preset', identity=ident)]
    profile['limits'].update(ram_bytes=192*1024**2, disk_bytes=2*1024**2, threads=2)
    r.update(identity=dict(requested=ident, effective=ident, fallback='forbid', reason='', profile_id='kokoro-cpu'), profile=oracle.profile_snapshot(profile), settings=settings, settings_sha256=oracle.canonical_hash(settings))
    context['bundles'] = [dict(instructions=False, engine='kokoro', model=expected['checkpoint'], mode='kokoro_preset', bundle_sha256=manifest_sha256(expected), languages=['en'], selectors=[dict(value='af_heart', languages=['en'])])]
    spec = Execution(identity(expected), manifest_sha256(expected), expected['checkpoint'], expected['revision'], 'af_heart', 'en', 2, 1024, 100, 244, 192*1024**2)
    context['execution'] = dict(descriptor=spec.identity(), bundles={spec.bundle_id:'/models/bundles/'+spec.bundle_id})
    oracle.plan_hash(plan); oracle.frozen_plan(plan, context['bundles'])
    return context, [profile]


class EngineAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=provisioning_fixtures.ProvisioningTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.bundle=self.fixture.run_provision()
        self.context,self.profiles=configuration(self.fixture.expected)
        self.mapping={identity(self.fixture.expected):self.bundle}
        self.core=Core(Path(self.fixture.temp.name)/'core'); self.addCleanup(self.core.close)
        self.stage=Path(self.fixture.temp.name)/'stage';self.stage.mkdir()
        self.admission=Admission(self.core,self.stage);self.store=DispatchStore(self.core,self.admission)
        self.version=self.context['plan']['voice_versions'][0]
        self.core.voices.register('alice','register',self.version,expected_revision=0)

    def submit(self, key='generate'):
        return Production(self.core).generate('alice',key,self.context,_admission=self.admission,_profiles=self.profiles)

    def test_verified_snapshot_owned_version_atomic_authority(self):
        contexts=KokoroContexts({'alice':self.context},self.mapping)
        frozen=contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        self.assertEqual(frozen,self.context)
        job=self.submit();self.assertEqual(job,self.submit())
        self.assertEqual(self.core.db.execute('SELECT fixture FROM dispatch_settings').fetchone(),('kokoro',))
        row=self.store.allocate(self.profiles,'kokoro',8);authority=row['authority']
        self.context['execution']['descriptor']['preset']='af_other'
        self.assertEqual(execution(authority.context).preset,'af_heart')
        self.assertEqual(len(self.admission.store.rows()),1)
        for field in ('generation','deadline','input_sha256'):
            message=authority.message();message[field]=0 if field=='generation' else ('0'*64 if field=='input_sha256' else '2000-01-01T00:00:00+00:00')
            with self.assertRaises(ValueError):authority.check(message,'execute')
        message=authority.message();message['plan']['request']['text']='substituted'
        with self.assertRaises(ValueError):authority.check(message,'execute')
        with self.assertRaises(ValueError):contexts.resolve('bob','speech_generate',frozen['plan']['request'])
        with self.assertRaises(ValueError):self.core.voices.version('bob',self.version['id'])

    def test_no_authority_qualification_and_closed_mapping(self):
        with self.assertRaisesRegex(ValueError,'authenticated_dispatch'):
            self.core.submit_execution('alice','no-authority',self.context)
        for mutate in (lambda c:c.pop('execution'), lambda c:c['execution']['bundles'].update({identity(self.fixture.expected):'/tmp/arbitrary'}),lambda c:c['execution']['descriptor'].update(revision='wrong',seed=1)):
            c=copy.deepcopy(self.context);mutate(c)
            with self.assertRaises(ValueError):self.admission.demand(c,self.profiles)
        self.profiles[0]['verification']='synthetic'
        with self.assertRaisesRegex(ValueError,'qualification'):self.submit()
        self.assertEqual(self.core.jobs('alice'),[])
        with self.assertRaises(ValueError):verify_local(self.context,{'wrong':self.bundle})
        changed=copy.deepcopy(self.context);changed['execution']['descriptor']['revision']='wrong'
        with self.assertRaises(ValueError):verify_local(changed,self.mapping)

    def test_reservation_failure_rolls_back_everything(self):
        self.core.db.execute("CREATE TRIGGER refuse BEFORE INSERT ON resource_reservations BEGIN SELECT RAISE(ABORT,'refuse'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.submit()
        for table in ('jobs','events','dispatch_settings','resource_reservations'):
            self.assertEqual(self.core.db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)

    def test_fixture_route_cannot_execute_real_context(self):
        self.submit()
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'silence',8)
        self.assertEqual(self.core.job(self.core.jobs('alice')[0]['job']['id'])['generation'],0)
        self.assertEqual(self.admission.store.rows()[0]['state'],'queued')

    def test_reconciliation_execution_and_reservation_tamper(self):
        from service.reconciliation import Reconciliation
        self.submit();row=self.store.allocate(self.profiles,'kokoro',8)
        original=self.core.db.execute('SELECT context FROM jobs').fetchone()[0]
        for mutate in (lambda c:c.update(unknown=True),lambda c:c.pop('execution'),lambda c:c['execution']['descriptor'].update(preset='wrong'),lambda c:c['execution']['bundles'].clear()):
            context=json.loads(original);mutate(context)
            with self.core.transaction():self.core.db.execute('UPDATE jobs SET context=?',(json.dumps(context),))
            summary=Reconciliation(self.core,self.stage).reconcile_startup()
            self.assertTrue(summary['blocked'])
            self.assertTrue(set(summary['reasons']) & {'unknown_reference_schema','unknown_reference_kind'})
        with self.core.transaction():
            self.core.db.execute('UPDATE jobs SET context=?',(original,))
            demand=json.loads(self.core.db.execute('SELECT demand FROM resource_reservations').fetchone()[0]);demand['configuration']='0'*64
            self.core.db.execute('UPDATE resource_reservations SET demand=?',(json.dumps(demand),))
        summary=Reconciliation(self.core,self.stage).reconcile_startup()
        self.assertTrue(summary['blocked'])
        self.assertIn('reservation_binding_mismatch',summary['reasons'])

if __name__=='__main__':unittest.main()
