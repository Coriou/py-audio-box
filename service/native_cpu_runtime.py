"""Explicit CPU transport policy on the existing native control lifecycle."""
import json
from service.native_runtime import NativeRuntime
from worker import native_cpu_protocol as p
from worker.native_cpu_availability import validate
from worker.native_availability import digest
from worker.engines.native_kokoro import JOB_SECONDS
from worker.transport import Refused, canonical


class NativeCPURuntime(NativeRuntime):
    protocol=p
    fixture='kokoro'
    job_seconds=JOB_SECONDS

    def cached_availability(self):
        if self._availability_metadata is None or self._availability_inventory is None:return None
        try:
            return validate(self._availability_metadata,p.inventory(self._availability_inventory),
                self.native['runtime_manifest_sha256'],now=self.utc())
        except Refused:return None

    def check_availability(self,context):
        if canonical(p.native_binding(context))!=canonical(self.native):raise Refused('native_configuration')
        enrolled=p.inventory(context['execution']['hardware'])
        if self._availability_inventory is not None and canonical(context['execution']['hardware'])!=canonical(self._availability_inventory):
            raise Refused('hardware_identity')
        value=self._rpc('availability',dict(runtime_manifest_sha256=self.native['runtime_manifest_sha256']),
            availability_binding=digest(enrolled))
        observed=validate(value,enrolled,self.native['runtime_manifest_sha256'],now=self.utc())
        self._availability_metadata=value
        self._availability_inventory=json.loads(canonical(context['execution']['hardware']))
        return observed

    def prepare(self,authority,fixture='kokoro',detached=b''):
        return super().prepare(authority,fixture,detached)
