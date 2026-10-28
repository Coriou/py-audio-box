"""Actual typed catalogue resolution; tiny sealed bundles, never model weights."""
import copy
import unittest
from unittest.mock import Mock
from service.native_cpu_context import NativeCPUContexts
from tests.worker.native_cpu_fixtures import Fixture
from worker import native_cpu_protocol as p
from worker.transport import Refused


class NativeCPUContextTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture(self)
        self.contexts=NativeCPUContexts({'owner':self.f.context},{self.f.spec.bundle_id:self.f.bundle})
        self.contexts.availability=Mock()

    def test_owned_alias_version_routes_exact_cpu_descriptor(self):
        version=self.contexts.preset_version('owner','af_heart','default')
        request=copy.deepcopy(self.f.context['plan']['request']);request['voice_version_id']=version['id']
        context=self.contexts.resolve_registered('owner',request,version)
        spec=p.native_execution(context)
        self.assertEqual((spec.voice_id,spec.voice_revision),('default',version['id']))
        self.assertEqual(version['reference_asset_sha256'],{})
        self.contexts.availability.assert_called()
        self.assertEqual(version,self.contexts.preset_version('owner','af_heart','default'))

    def test_ownership_reference_mixed_family_and_availability_refuse(self):
        version=self.contexts.preset_version('owner','af_heart','default')
        request=copy.deepcopy(self.f.context['plan']['request']);request['voice_version_id']=version['id']
        for key,value in (('reference_asset_ids',['ref']),('origin','clone'),('reference_text','source')):
            changed=copy.deepcopy(version);changed[key]=value
            with self.subTest(key=key),self.assertRaises(Refused):self.contexts.resolve_registered('owner',request,changed)
        with self.assertRaises(Refused):self.contexts.preset_version('another','af_heart','default')
        with self.assertRaises(Refused):self.contexts.preset_version('owner','Ryan','default')
        self.contexts.availability=None
        with self.assertRaisesRegex(Refused,'native_availability_required'):self.contexts.resolve_registered('owner',request,version)
        with self.assertRaises(Refused):self.contexts.resolve('owner','voice_prepare',{})


if __name__=='__main__':unittest.main()
