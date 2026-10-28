"""Tiny inert native CPU fixtures. Never hardware/model/installed evidence."""
import copy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from contracts import validate as oracle
from service import provisioning
from tests.service.test_provisioning import ProvisioningTests
from tests.service.test_engine_admission import configuration
from worker.engines.native_kokoro import NativeKokoroExecution,MEMORY_BYTES,JOB_SECONDS
from worker.hardware import HardwareInventory,Host,Device,DeviceFamily
from worker.transport import canonical


class Fixture:
    def __init__(self,case):
        self.source=ProvisioningTests();self.source.setUp();case.addCleanup(self.source.doCleanups)
        self.bundle=self.source.run_provision();self.context,self.profiles=configuration(self.source.expected)
        from worker.runtime_protocol import execution
        old=execution(self.context);voice=self.context['plan']['voice_versions'][0]
        self.spec=NativeKokoroExecution(replace(old,memory_bytes=MEMORY_BYTES,max_text_bytes=256,max_frames=480000,max_output_bytes=960044),voice['voice_id'],voice['id'])
        profile=self.profiles[0];profile.update(id='kokoro-native-cpu',topology='packaged_native')
        profile['limits'].update(ram_bytes=MEMORY_BYTES,deadline_seconds=JOB_SECONDS,disk_bytes=16*1024**2,upload_bytes=1048576,decoded_ticks=960000)
        resolved=self.context['plan']['resolved'];resolved['identity']['profile_id']=profile['id'];resolved['profile']=oracle.profile_snapshot(profile)
        self.context['execution']=dict(descriptor=self.spec.identity(),bundles={self.spec.bundle_id:'/native-models/bundles/'+self.spec.bundle_id},
            native=dict(deployment_id='a'*32,configuration_sha256='b'*64,lane='0'*32,boot='0'*32,runtime_manifest_sha256='d'*64))
        oracle.frozen_plan(self.context['plan'],self.context['bundles'])
        self.now=datetime.now(timezone.utc);self.t=100.
        self.inventory=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.CPU,0,True,True,None),))
        from worker.engines.qwen_execution import inventory_wire
        self.context['execution']['hardware']=inventory_wire(self.inventory)
        self.root=Path(self.source.temp.name).resolve()/'cpu-worker';self.root.mkdir(mode=0o700)
        self.snapshot=self.snapshot_for(self.context)

    def snapshot_for(self,context):
        context=copy.deepcopy(context)
        sample=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
        job=sample['job'];job.update(input_sha256=oracle.plan_hash(context['plan']),deadline=(self.now+timedelta(seconds=60)).isoformat())
        return [job,context,[],copy.deepcopy(self.profiles)]

    def registration(self,snapshot=None):
        snapshot=self.snapshot if snapshot is None else snapshot
        from worker import native_cpu_protocol as p
        return dict(snapshot=snapshot,execution=p.native_execution(snapshot[1]),bundles={self.spec.bundle_id:self.bundle},inventory=self.inventory)

    def observed(self,enrolled):
        return replace(enrolled,devices=(replace(enrolled.devices[0],available=True,probed_at=self.now.isoformat()),))


class Handle:
    pid=123
    code=None
    waited=0
    def poll(self):return self.code
    def wait(self,timeout):self.waited+=1;return self.code


class Channel:
    def __init__(self):self.fd=42
    def fileno(self):return self.fd
    def close(self):self.fd=-1


class Sampler:
    def __init__(self,pid,fixture):
        self.fixture=fixture;self.pid=pid;self.error=None;self.peak=100;self.count=1;self.stamp=fixture.t
        self.thread=SimpleNamespace(is_alive=lambda:False)
    def start(self):pass
    def snapshot(self):return dict(count=self.count,peak=self.peak,error=self.error,
        last=None if not self.count else dict(pid=self.pid,identity=4,rss_bytes=100,method='proc_pid_rusage_v0',observed_at=self.stamp))
    def request_stop(self):pass
    def stop(self):return True
