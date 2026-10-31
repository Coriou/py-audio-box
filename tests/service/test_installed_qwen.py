"""Injected source fixtures only; never acquire weights or import an ML runtime."""
import copy
from datetime import datetime, timezone, timedelta
import json
import os
import unittest
from unittest.mock import patch
from service import install_policy as ip, installed_qwen_daemon as qd
from service import execution_context as ec
from worker.engines import pocket
from worker.engines.qwen_execution import inventory_from_wire, installed_cpu_hardware
from worker.runtime_protocol import execution
from tests.service.test_qwen_preparation import CPUPreparationFixture
from tests.service.test_installed_pocket import configuration as base_configuration


def configuration(fixture):
    context=copy.deepcopy(fixture.template)
    def pins(value):
        if isinstance(value,dict): return {k:(qd.PUBLIC_MODEL if k == 'model' and v == 'test' else pins(v)) for k,v in value.items()}
        if isinstance(value,list): return [pins(v) for v in value]
        return {'test':'Qwen/Qwen3-TTS-12Hz-0.6B-Base','rev1':'5d83992436eae1d760afd27aff78a71d676296fc'}.get(value,value) if isinstance(value,str) else value
    context=pins(context)
    profiles=pins(copy.deepcopy(fixture.profiles))
    config=base_configuration(context,profiles)
    config['ownership_policy']=qd.POLICY
    return config


class InstalledQwenTests(CPUPreparationFixture):
    def setUp(self):
        super().setUp()
        self.config=configuration(self)
        # Inert fixture manifest; actual configuration requires the canonical pin.
        self.enterContext(patch.object(qd,'MANIFEST_SHA256',execution(self.config['contexts']['alice']).manifest_sha256))

    def test_closed_configuration_family_and_cpu_tuple(self):
        qd.configuration(self.config)
        self.assertIs(ip.family_policy(qd.FAMILY,self.config),ip.QWEN_SPLIT)
        self.assertEqual(ip.expectation(ip.QWEN_SPLIT,'staging',directory=True),(0,10001,0o1753))
        for key,value in [('device','mps'),('dtype','bfloat16'),('attention','flash_attention_2'),('revision','wrong'),('checkpoint','other'),('x_vector_only',False)]:
            bad=copy.deepcopy(self.config);bad['contexts']['alice']['execution']['descriptor'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):qd.configuration(bad)
        with self.assertRaises(ValueError):ip.family_configuration(self.config,object())
        with self.assertRaises(ValueError):ip.family_configuration(self.config,__import__('service.installed_pocket_daemon',fromlist=['FAMILY']).FAMILY)

    def test_real_schema_and_exact_public_binding(self):
        from contracts import validate as oracle
        from jsonschema.exceptions import ValidationError
        context=self.config['contexts']['alice']
        self.assertEqual(context['plan']['request']['identity']['model'],qd.PUBLIC_MODEL)
        oracle.validate('profile',self.config['profiles'][0])
        oracle.frozen_plan(context['plan'],context['bundles'])
        qd.configuration(self.config)
        # Exercise the real eleven-file production manifest pin as metadata;
        # no acquisition or model bytes are needed by this closed parser.
        from pathlib import Path
        import hashlib
        from service.provisioning import bundle_manifest, manifest_bytes
        expected=json.loads((Path(__file__).resolve().parents[2]/'requirements/tts/qwen-0.6b-base.expected.json').read_text())
        canonical=hashlib.sha256(manifest_bytes(bundle_manifest(expected))).hexdigest()
        old=context['execution']['descriptor']['manifest_sha256']
        pinned=json.loads(json.dumps(self.config).replace(old,canonical))
        with patch.object(qd,'MANIFEST_SHA256',canonical):qd.configuration(pinned)
        for model in (qd.CHECKPOINT,'other-model'):
            bad=json.loads(json.dumps(self.config).replace('"'+qd.PUBLIC_MODEL+'"','"'+model+'"'))
            rejection=ValidationError if model == qd.CHECKPOINT else ValueError
            with self.subTest(model=model),self.assertRaises(rejection):qd.configuration(bad)
        for key,value in [('revision','wrong'),('checkpoint',qd.PUBLIC_MODEL),('dtype','bfloat16'),('attention','flash_attention_2')]:
            bad=copy.deepcopy(self.config);bad['contexts']['alice']['execution']['descriptor'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):qd.configuration(bad)

    def test_live_cpu_observation_does_not_renew_frozen_identity(self):
        context=self.config['contexts']['alice']
        old=(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
        context['execution']['hardware']['devices'][0]['probed_at']=old
        before=copy.deepcopy(context)
        inv=inventory_from_wire(context['execution']['hardware'])
        with patch('sys.platform','linux'),patch('os.sched_getaffinity',return_value={0,1},create=True):
            installed_cpu_hardware(execution(context),inv,family=qd.FAMILY)
            self.assertEqual(context,before)
            with self.assertRaises(ValueError):installed_cpu_hardware(execution(context),inv,family=object())
        with patch('sys.platform','linux'),patch('os.sched_getaffinity',return_value=set(),create=True):
            with self.assertRaises(ValueError):installed_cpu_hardware(execution(context),inv,family=qd.FAMILY)

    def test_installed_factory_keeps_managed_owner_and_explicit_family(self):
        with patch.object(ec,'QwenContexts') as factory:
            ec.installed_contexts(self.config,{'bundle':'/sealed'})
        self.assertIs(factory.call_args.kwargs['installed_family'],qd.FAMILY)
        self.assertEqual(factory.call_args.args[4],self.config['profiles'])


# Run LotA's complete public journey through the installed context permission.
# Descriptor and filesystem policy are tested above; this fixture injects their
# authority because it deliberately uses tiny bytes owned by the test UID.
from tests.service.test_qwen_preparation import CPUPreparationTests


class InstalledQwenManagedJourneyTests(CPUPreparationTests):
    def setUp(self):
        super().setUp()
        self.contexts.installed_family=qd.FAMILY
        # Installed assets belong to UID 10001; this injected Core writes them
        # as the local test UID. Keep the real bounded, digest-checked read.
        real_read=pocket._read
        def injected_asset_read(path,limit,digest=None,*,owner_uid=None):
            if owner_uid == 10001:
                self.assertEqual(path.parent,self.b.core.root/'assets')
                self.assertTrue(digest)
                owner_uid=os.getuid()
            return real_read(path,limit,digest,owner_uid=owner_uid)
        self.enterContext(patch.object(pocket,'_read',side_effect=injected_asset_read))
        self.enterContext(patch.object(qd,'validate_spec',return_value=None))
        self.enterContext(patch('sys.platform','linux'))
        self.enterContext(patch('os.sched_getaffinity',return_value={0,1},create=True))
        real=ec.verify_qwen_local
        def injected_owner(context,bundles,references,inventory,**kwargs):
            return real(context,bundles,references,inventory)
        self.enterContext(patch.object(ec,'verify_qwen_local',side_effect=injected_owner))
