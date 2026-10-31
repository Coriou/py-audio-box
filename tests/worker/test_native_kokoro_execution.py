"""Typed native CPU source contracts; no imports of a model or native runtime."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from lib.spoken_text import freeze
from worker.engines.execution import Execution
from worker.engines.native_kokoro import NativeKokoroExecution, relation, MEMORY_BYTES, JOB_SECONDS
from worker.transport import Refused


def fixture():
    context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    plan=context['plan'];resolved=plan['resolved'];profile=resolved['profile'];voice=plan['voice_versions'][0]
    spec=NativeKokoroExecution(Execution('bundle','a'*64,'Kokoro-82M-v1.0-onnx-fp32','model-files-v1.0',
        'af_heart','en',2,256,24000*20,960044,MEMORY_BYTES),voice['voice_id'],voice['id'])
    identity=dict(engine='kokoro',model=spec.checkpoint,bundle_sha256=spec.manifest_sha256,device='cpu',precision='float32')
    settings=dict(kind='kokoro_preset',language='en')
    voice.update(identity=identity,origin='preset',reference_asset_ids=[],configuration=dict(mode='kokoro_preset',preset='af_heart',language='en'),settings=settings)
    profile.update(id='kokoro-native-cpu',device='cpu',topology='packaged_native')
    profile['limits'].update(threads=2,ram_bytes=MEMORY_BYTES,deadline_seconds=JOB_SECONDS,decoded_ticks=960000,upload_bytes=1048576,disk_bytes=2097152)
    resolved.update(settings=settings,identity=dict(requested=identity,effective=identity,fallback='forbid',reason='',profile_id=profile['id']))
    plan['assets']=[];plan['request'].update(voice_version_id=voice['id'],identity=identity,fallback='forbid',variants=1,reuse='never',profile_override=None,device_override=None)
    context['execution']=dict(descriptor=spec.identity(),bundles={'bundle':'/native-models/bundles/bundle'},native=dict(deployment_id='b'*32,configuration_sha256='c'*64,lane='d'*32,boot='e'*32,runtime_manifest_sha256='f'*64))
    from worker.engines.qwen_execution import inventory_wire
    from worker.hardware import HardwareInventory,Host,Device,DeviceFamily
    context['execution']['hardware']=inventory_wire(HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.CPU,0,True,True,None),)))
    freeze(plan)
    return context,spec


class NativeCPUExecutionTests(unittest.TestCase):
    def test_closed_roundtrip_and_actual_engine_value_remains_distinct(self):
        context,spec=fixture();self.assertEqual(relation(context),spec)
        self.assertEqual(NativeKokoroExecution.from_wire(spec.identity()),spec)
        self.assertIs(type(spec.model),Execution)
        with self.assertRaises(Refused):Execution.from_wire(spec.identity())
        from worker.native_protocol import native_execution
        with self.assertRaises(Refused):native_execution(context)
        from worker.runtime_protocol import execution
        self.assertEqual(execution(context),spec)

    def test_family_version_and_receipt_cannot_cross_lanes(self):
        from worker import native_protocol as qwen, native_cpu_protocol as cpu
        from worker.native_family import protocol
        context,spec=fixture();self.assertIs(protocol(context),cpu)
        q=dict(version=cpu.VERSION,deployment_id='a'*32,configuration_sha256='b'*64,
            lane=None,boot=None,request_id='c'*32,runtime_id=None,binding=None,op='hello',body={})
        self.assertEqual(cpu.request(q),q)
        with self.assertRaises(Refused):qwen.request(q)
        q['version']=qwen.VERSION
        with self.assertRaises(Refused):cpu.request(q)
        expected=dict(runtime_id='a'*32+'/'+'b'*32+'/'+'c'*32,binding='d'*64,deployment_id='e'*32,
            configuration_sha256='f'*64,lane='a'*32,boot='b'*32,original_deadline='2026-10-28T00:00:00+00:00')
        from worker.native_cpu_resources import assess
        resources=assess(dict(last=None,error=None,peak=0,count=0),started=0,now=0)
        resources['thread_stopped']=True
        value=dict(expected,version=cpu.RECEIPT_VERSION,reaped=True,timely=False,
            reap_method='darwin-owned-exec-v1',processes=[],receiver_closed=True,resources=resources,error=None,termination=dict(reason=None,signals=[],signal_error=None,observed_exit_at=None))
        self.assertEqual(cpu.receipt(value,expected=expected),value)
        with self.assertRaises(Refused):qwen.receipt(value,expected=expected)
        value['version']='native-receipt-v1'
        with self.assertRaises(Refused):cpu.receipt(value,expected=expected)

    def test_character_profile_and_encoded_backend_limits_are_distinct(self):
        from service.native_cpu_configuration import expected,profile
        from worker.engines.kokoro import execute
        context,spec=fixture()
        configured=profile(expected(),revision='source-limits',installed_at='2026-10-28T00:00:00+00:00')
        self.assertEqual(configured['limits']['text_chars'],256)
        self.assertEqual(spec.max_text_bytes,256)
        with self.assertRaisesRegex(Refused,'unauthorized_bundle'):execute(spec.model,'a'*256,{})
        with self.assertRaisesRegex(Refused,'text_limit'):execute(spec.model,'é'*256,{})

    def test_cpu_inventory_excludes_mps_and_linux(self):
        from worker import native_cpu_protocol as cpu
        from worker.hardware import HardwareInventory, Host, Device, DeviceFamily
        value=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.CPU,0,True,True,None),))
        self.assertEqual(cpu.inventory(cpu.inventory_wire(value)),value)
        for modified in (replace(value,host=Host('linux','arm64','docker')),
                replace(value,devices=(Device(DeviceFamily.MPS,0,True,True,None),))):
            with self.assertRaises(Refused):cpu.inventory(cpu.inventory_wire(modified))

    def test_no_qwen_clone_reference_path_or_extra_controls(self):
        context,spec=fixture()
        for key,value in (('task','clone'),('reference_bytes',42),('dtype','float16'),('path','/untrusted')):
            wire=spec.identity();wire[key]=value
            with self.subTest(key=key),self.assertRaises(Refused):NativeKokoroExecution.from_wire(wire)
        for change in (dict(language='fr',preset='ff_siwis'),dict(preset='af_bella'),dict(threads=1),dict(memory_bytes=6*1024**3)):
            with self.subTest(change=change),self.assertRaises(Refused):replace(spec,model=replace(spec.model,**change))
        context['execution']['references']={}
        with self.assertRaises(Refused):relation(context)

    def test_context_identity_topology_mapping_voice_and_budget_must_match(self):
        context,spec=fixture()
        changes=(lambda c:c['plan']['resolved']['profile'].update(id='kokoro-cpu',topology='networkless_container'),
            lambda c:c['execution']['bundles'].update(bundle='/models/bundles/bundle'),
            lambda c:c['plan']['voice_versions'][0]['configuration'].update(preset='af_bella'),
            lambda c:c['plan']['request'].update(fallback='allow_cpu'),
            lambda c:c['plan']['resolved']['profile']['limits'].update(deadline_seconds=150),
            lambda c:c['plan']['voice_versions'][0].update(reference_asset_ids=['asset']))
        for change in changes:
            modified=copy.deepcopy(context);change(modified)
            with self.subTest(change=change),self.assertRaises(Refused):relation(modified)


if __name__=='__main__':unittest.main()
