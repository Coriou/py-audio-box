"""Private qualification authority is finite, scoped and never a ready profile."""
import copy
import json
from types import SimpleNamespace
import unittest
from contracts import validate as oracle
from worker.native_cpu_qualification import Session,VERSION,SECONDS,MAX_JOBS,permitted,current
from worker.transport import Refused
from tests.worker.native_cpu_fixtures import Fixture


class CPUQualificationTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture(self);self.t=100.
        self.profile=self.f.profiles[0]
        self.profile.update(support='unqualified',readiness='not_ready',verification='synthetic',qualification_receipt='')
        self.value=dict(version=VERSION,session_id='1'*32,boot_id='00000000-0000-0000-0000-000000000001',started_monotonic=100.,seconds=SECONDS,max_jobs=MAX_JOBS)
        self.s=Session(self.value,configuration_sha256='b'*64,runtime_manifest_sha256='d'*64,deployment_id='a'*32,
            profile=self.profile,manifest_sha256=self.f.spec.manifest_sha256,principal='owner',clock=lambda:self.t,boot=lambda:self.value['boot_id'])
        self.job=dict(operation='speech_generate',input_sha256=oracle.plan_hash(self.f.context['plan']))

    def test_unqualified_profile_requires_private_scope_without_promotion(self):
        with self.assertRaisesRegex(ValueError,'not_ready'):oracle.admission(self.job,self.f.context,self.f.profiles)
        with self.s.scope():
            oracle.admission(self.job,self.f.context,self.f.profiles)
            self.assertIs(current(),self.s)
        self.assertIsNone(current())
        self.assertEqual((self.profile['support'],self.profile['readiness'],self.profile['qualification_receipt']),('unqualified','not_ready',''))
        self.assertFalse(permitted(self.f.context,self.profile))

    def test_expiry_restart_boot_and_context_substitution_refuse(self):
        self.t=641.
        with self.s.scope(),self.assertRaisesRegex(Refused,'expired'):oracle.admission(self.job,self.f.context,self.f.profiles)
        self.s.authorize(self.f.context,self.profile,live=False)  # cleanup/history, never allocation
        self.t=100.
        for mutate in (lambda c:c['execution']['native'].update(configuration_sha256='f'*64),
                lambda c:c['plan']['resolved']['profile'].update(config_revision='changed')):
            context=copy.deepcopy(self.f.context);mutate(context)
            with self.assertRaises(Refused):self.s.authorize(context,self.profile)
        with self.assertRaises(Refused):self.s.authorize(self.f.context,self.profile,dict(operation='speech_generate',principal_id='another'))
        self.s.boot=lambda:'00000000-0000-0000-0000-000000000002'
        with self.assertRaisesRegex(Refused,'expired'):self.s.authorize(self.f.context,self.profile)
        with self.assertRaises(Refused):Session(self.value,configuration_sha256='b'*64,runtime_manifest_sha256='d'*64,deployment_id='a'*32,
            profile=self.profile,manifest_sha256=self.f.spec.manifest_sha256,principal='owner',clock=lambda:99.,boot=lambda:self.value['boot_id'])

    def test_reopened_descriptor_keeps_original_monotonic_budget(self):
        self.t=620.
        reopened=Session(self.value,configuration_sha256='b'*64,runtime_manifest_sha256='d'*64,deployment_id='a'*32,
            profile=self.profile,manifest_sha256=self.f.spec.manifest_sha256,principal='owner',clock=lambda:self.t,boot=lambda:self.value['boot_id'])
        self.assertEqual(reopened.value['started_monotonic'],100.)
        self.t=641.
        with self.assertRaises(Refused):reopened.authorize(self.f.context,self.profile)

    def test_durable_job_count_and_queue_deadline_are_not_renewed(self):
        rows=[]
        class DB:
            def execute(inner,sql,args):
                self.assertEqual(args,(MAX_JOBS+1,))
                return SimpleNamespace(fetchall=lambda:rows[:MAX_JOBS+1])
        core=SimpleNamespace(db=DB())
        row=lambda state:(json.dumps(self.f.context),json.dumps(dict(state=state)))
        rows.extend(row('succeeded') for _ in range(MAX_JOBS))
        self.s.reserve(core,self.f.context,self.profile,self.job)
        rows.append(row('queued'))
        with self.assertRaisesRegex(Refused,'job_limit'):self.s.reserve(core,self.f.context,self.profile,self.job)
        rows[:]=[row('running'),row('queued')];self.t=600.
        with self.assertRaisesRegex(Refused,'queue_budget'):self.s.reserve(core,self.f.context,self.profile,self.job)
        rows[:]=[row('cancelled'),row('queued')]
        self.s.reserve(core,self.f.context,self.profile,self.job)


    def test_explicit_selftest_exempts_only_already_validated_history_without_reset(self):
        old=[('ignored old measured context',json.dumps(dict(id='old-'+str(i),state='succeeded'))) for i in range(12)]
        rows=list(old)
        class DB:
            def execute(inner,sql,args):
                self.assertEqual(args,(64,));return SimpleNamespace(fetchall=lambda:rows[:64])
        self.s.prior_jobs=frozenset('old-'+str(i) for i in range(12))
        core=SimpleNamespace(db=DB())
        self.s.reserve(core,self.f.context,self.profile,self.job)
        rows.extend((json.dumps(self.f.context),json.dumps(dict(id='new-'+str(i),state='succeeded'))) for i in range(7))
        with self.assertRaisesRegex(Refused,'job_limit'):self.s.reserve(core,self.f.context,self.profile,self.job)
        self.assertEqual(rows[:12],old)


if __name__=='__main__':unittest.main()
