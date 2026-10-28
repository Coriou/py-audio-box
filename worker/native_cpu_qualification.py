"""Private, finite CPU qualification authority; never a readiness receipt.

Only the checked native CPU activation/custodian enters this scope. HTTP payloads
cannot install one. Profiles remain unqualified/not_ready with no measurement.
The immutable session uses the host boot and system monotonic clock, so service
restart cannot renew it. Ordinary admission remains unchanged outside this scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import copy
import time
import uuid

from worker.transport import Refused,canonical

_current=ContextVar('native_cpu_qualification',default=None)
SECONDS=600
MAX_JOBS=6
VERSION='native-cpu-qualification-session-v1'


def boot_identity():
    import ctypes
    import sys
    if sys.platform!='darwin':raise Refused('native_cpu_darwin_boot_required')
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
    query=lib.sysctlbyname
    query.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t]
    query.restype=ctypes.c_int
    value=ctypes.create_string_buffer(64);size=ctypes.c_size_t(64)
    if query(b'kern.bootsessionuuid',value,ctypes.byref(size),None,0) or not 1<size.value<=64:
        raise OSError(ctypes.get_errno(),'native_cpu_boot_observation_failed')
    text=value.value.decode('ascii').lower()
    if str(uuid.UUID(text))!=text:raise Refused('native_cpu_boot_identity')
    return text


def descriptor():
    return dict(version=VERSION,session_id=uuid.uuid4().hex,boot_id=boot_identity(),
        started_monotonic=time.monotonic(),seconds=SECONDS,max_jobs=MAX_JOBS)


class Session:
    def __init__(self,value,*,configuration_sha256,runtime_manifest_sha256,deployment_id,
                 profile,manifest_sha256,principal,clock=time.monotonic,boot=None):
        from worker.native_protocol import _closed,_hex
        from contracts import validate as oracle
        import math
        if boot is None:boot=boot_identity
        _closed(value,('version','session_id','boot_id','started_monotonic','seconds','max_jobs'))
        if value['version']!=VERSION or type(value['seconds']) is not int or value['seconds']!=SECONDS or type(value['max_jobs']) is not int or value['max_jobs']!=MAX_JOBS:
            raise Refused('native_cpu_qualification_shape')
        _hex(value['session_id'],32)
        if type(value['boot_id']) is not str or str(uuid.UUID(value['boot_id']))!=value['boot_id'] or value['boot_id']!=boot():
            raise Refused('native_cpu_qualification_boot')
        if type(value['started_monotonic']) not in (int,float) or not math.isfinite(value['started_monotonic']) or value['started_monotonic']>clock():
            raise Refused('native_cpu_qualification_clock')
        for digest in (configuration_sha256,runtime_manifest_sha256,manifest_sha256):_hex(digest,64)
        _hex(deployment_id,32)
        oracle.validate('profile',profile)
        if (profile['id'],profile['support'],profile['readiness'],profile['installation'],profile['configuration'],profile['verification'],profile['qualification_receipt'])!=('kokoro-native-cpu','unqualified','not_ready','installed','valid','synthetic',''):
            raise Refused('native_cpu_qualification_profile')
        if type(principal) is not str or not principal:raise Refused('native_cpu_qualification_principal')
        self.value=copy.deepcopy(value);self.configuration=configuration_sha256;self.manifest=runtime_manifest_sha256
        self.deployment=deployment_id;self.profile=canonical(profile);self.bundle=manifest_sha256;self.principal=principal
        self.clock,self.boot=clock,boot

    @contextmanager
    def scope(self):
        token=_current.set(self)
        try:yield
        finally:_current.reset(token)

    def authorize(self,context,profile,job=None,*,live=True):
        from contracts import validate as oracle
        from worker import native_cpu_protocol as p
        from worker.engines.native_kokoro import JOB_SECONDS
        spec=p.native_execution(context);binding=p.native_binding(context)
        if (canonical(profile)!=self.profile or oracle.profile_snapshot(profile)!=context['plan']['resolved']['profile'] or spec.manifest_sha256!=self.bundle or
                (binding['deployment_id'],binding['configuration_sha256'],binding['runtime_manifest_sha256'])!=(self.deployment,self.configuration,self.manifest)):
            raise Refused('native_cpu_qualification_binding')
        if job is not None and (job.get('principal_id',self.principal)!=self.principal or job.get('operation')!='speech_generate'):
            raise Refused('native_cpu_qualification_principal')
        if live:
            remaining=self.value['started_monotonic']+SECONDS-self.clock()
            budget=JOB_SECONDS
            if job and job.get('deadline'):
                budget=(datetime.fromisoformat(job['deadline'])-datetime.now(timezone.utc)).total_seconds()
            if self.boot()!=self.value['boot_id'] or not 0<budget<=remaining<=SECONDS:
                raise Refused('native_cpu_qualification_expired')
        return True

    def reserve(self,core,context,profile,job):
        self.authorize(context,profile,job)
        rows=core.db.execute('SELECT context,job FROM jobs LIMIT ?', (MAX_JOBS+1,)).fetchall()
        if len(rows)>MAX_JOBS:raise Refused('native_cpu_qualification_job_limit')
        import json
        live=0
        for raw,stored in rows:
            self.authorize(json.loads(raw),profile,live=False)
            if json.loads(stored)['state'] in ('queued','running','cancel_requested'):live+=1
        from worker.engines.native_kokoro import JOB_SECONDS
        if live*(JOB_SECONDS+2)>self.value['started_monotonic']+SECONDS-self.clock():
            raise Refused('native_cpu_qualification_queue_budget')


def current():
    value=_current.get()
    return value if type(value) is Session else None


def permitted(context,profile,job=None):
    session=current()
    if session is None:return False
    session.authorize(context,profile,job)
    return True
