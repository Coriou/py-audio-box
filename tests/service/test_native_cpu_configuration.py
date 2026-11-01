"""Production configuration generator uses real pinned metadata, no model files."""
import unittest
from service.native_cpu_configuration import expected,profile,context,template
from worker.hardware import HardwareInventory,Host,Device,DeviceFamily
from contracts import validate as oracle
from worker import native_cpu_protocol as p


class CPUConfigurationTests(unittest.TestCase):
    def test_generated_installed_metadata_is_unqualified_and_closed(self):
        descriptor=expected();configured=profile(descriptor,revision='test-source',installed_at='2026-10-28T00:00:00+00:00')
        native=dict(deployment_id='a'*32,configuration_sha256='b'*64,lane='c'*32,boot='d'*32,runtime_manifest_sha256='e'*64)
        inventory=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.CPU,0,True,True,None),))
        result=context(descriptor,configured,native,inventory)
        self.assertEqual(p.native_execution(result).preset,'af_heart')
        self.assertEqual(result['plan']['assets'],[])
        self.assertEqual((configured['support'],configured['readiness'],configured['qualification_receipt']),('unqualified','not_ready',''))
        snapshot=template(result,[configured],'local')
        with self.assertRaisesRegex(ValueError,'not_ready'):oracle.admission(snapshot[0],result,[configured])
        self.assertEqual(snapshot[2],[])


if __name__=='__main__':unittest.main()
