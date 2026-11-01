"""CPU metadata freshness only, not runtime or performance qualification."""
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import unittest
from unittest.mock import Mock,patch
from worker.hardware import HardwareInventory,Host,Device,DeviceFamily
from worker.native_cpu_availability import Availability,validate,digest
from worker.native_cpu_protocol import inventory_wire
from worker.transport import Refused


class CPUAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        self.enrolled=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.CPU,0,True,True,None),))
        self.observed=replace(self.enrolled,devices=(replace(self.enrolled.devices[0],probed_at=self.now.isoformat()),))

    def test_cache_discovery_does_not_probe_and_fresh_observation_is_cpu_only(self):
        provider=Mock(return_value=self.observed);clock=[self.now]
        value=Availability(self.enrolled,'a'*64,provider=provider,utc=lambda:clock[0])
        self.assertIsNone(value.cached());provider.assert_not_called()
        result=value.get();self.assertEqual(validate(result,self.enrolled,'a'*64,now=self.now),self.observed)
        value.get();self.assertEqual(provider.call_count,1)
        clock[0]+=timedelta(seconds=301);self.assertIsNone(value.cached())
        broken=dict(result,runtime_manifest_sha256='b'*64)
        with self.assertRaises(Refused):validate(broken,self.enrolled,'a'*64,now=self.now)

    def test_stale_future_unavailable_and_foreign_host_refuse(self):
        for seconds in (-1,301):
            observed=replace(self.observed,devices=(replace(self.observed.devices[0],probed_at=(self.now-timedelta(seconds=seconds)).isoformat()),))
            value=dict(enrollment_sha256=digest(self.enrolled),runtime_manifest_sha256='a'*64,inventory=inventory_wire(observed))
            with self.assertRaises(Refused):validate(value,self.enrolled,'a'*64,now=self.now)
        for foreign in (replace(self.enrolled,host=Host('linux','arm64','docker')),replace(self.enrolled,devices=(Device(DeviceFamily.MPS,0,True,True,None),))):
            with self.assertRaises(Refused):Availability(foreign,'a'*64)

    def test_friendly_macos_minimum_without_dependency_or_model_import(self):
        from lib.tts_backends import kokoro_native as k
        with patch.object(k.platform,'system',return_value='Darwin'),patch.object(k.platform,'machine',return_value='arm64'),patch.object(k.platform,'mac_ver',return_value=('13.7.9',(),'')):
            with self.assertRaisesRegex(ValueError,'macOS 14'):k.validate_host()


if __name__=='__main__':unittest.main()
