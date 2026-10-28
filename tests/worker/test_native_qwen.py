"""Native execution primitives with injected upstream; no process/hardware probe."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import os
import json
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

from worker import native_child
from worker.engines.qwen_execution import current_hardware, current_native_hardware
from worker.hardware import HardwareInventory, Host, Device, DeviceFamily
from worker.transport import Refused
from tests.worker import test_engine_qwen as qwen_fixture


@contextmanager
def inert_guard():
    # Exercise the real installer, restoring Python hooks afterward. Register no
    # permanent interpreter audit hook in the unittest owner process.
    from multiprocessing.process import BaseProcess
    with ExitStack() as stack:
        for name in ('fork','forkpty','posix_spawn','posix_spawnp','system',
                     'execl','execle','execlp','execlpe','execv','execve',
                     'execvp','execvpe','spawnl','spawnle','spawnlp','spawnlpe',
                     'spawnv','spawnve','spawnvp','spawnvpe'):
            if hasattr(os,name): stack.enter_context(patch.object(os,name,getattr(os,name)))
        stack.enter_context(patch.object(subprocess,'Popen',subprocess.Popen))
        stack.enter_context(patch.object(BaseProcess,'start',BaseProcess.start))
        stack.enter_context(patch.object(native_child,'_guard_installed',False))
        audit=stack.enter_context(patch.object(sys,'addaudithook'))
        native_child.install_spawn_guard()
        yield audit


class NativeQwenTests(unittest.TestCase):
    def setUp(self):
        self.fixture=qwen_fixture.QwenTests('test_cpu_exact_forwarding_and_pcm')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.spec=replace(self.fixture.e,device='mps',timeout_s=25)
        self.now=datetime.now(timezone.utc)
        self.inventory=HardwareInventory(Host('darwin','arm64','native'),
            (Device(DeviceFamily.MPS,0,True,True,self.now.isoformat()),))

    def test_native_freshness_and_linux_separation(self):
        current_native_hardware(self.spec,self.inventory,now=self.now)
        with self.assertRaises(Refused): current_hardware(self.spec,self.inventory)
        for age in (-1,301):
            device=replace(self.inventory.devices[0],probed_at=(self.now-timedelta(seconds=age)).isoformat())
            with self.assertRaises(Refused):
                current_native_hardware(self.spec,replace(self.inventory,devices=(device,)),now=self.now)
        for host in (Host('linux','arm64','docker'), Host('darwin','x86_64','native'), Host('darwin','arm64','docker')):
            with self.assertRaises(Refused): current_native_hardware(self.spec,replace(self.inventory,host=host),now=self.now)
        for mutation in (dict(dtype='float16'),dict(attention='eager')):
            with self.assertRaises(Refused): current_native_hardware(replace(self.spec,**mutation),self.inventory,now=self.now)

    def test_guard_all_python_launch_routes_and_audit(self):
        from multiprocessing.process import BaseProcess
        with inert_guard() as audit:
            for call in (lambda:subprocess.Popen(['never']), lambda:os.system('never'),
                         lambda:os.execv('/never',[]),lambda:BaseProcess.start(None)):
                with self.assertRaises(native_child.SpawnRefused): call()
            for name in ('fork','forkpty','posix_spawn','posix_spawnp'):
                if hasattr(os,name):
                    with self.assertRaises(native_child.SpawnRefused): getattr(os,name)()
            hook=audit.call_args.args[0]
            for event in ('subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.exec','os.system','os.spawn'):
                with self.assertRaises(native_child.SpawnRefused): hook(event,())
            hook('open',())

    def test_inference_requires_guard(self):
        with self.assertRaisesRegex(Refused,'native_spawn_guard_required'):
            native_child.infer(self.spec,'hello',{}, {},self.inventory)

    def test_corrupt_bundle_cannot_reach_backend_or_loaded(self):
        model=self.fixture.bundle/'model.safetensors'
        model.write_bytes(b'corrupt fixture')
        for path in self.fixture.bundle.iterdir(): path.chmod(0o400)
        self.fixture.bundle.chmod(0o500)
        loaded=Mock()
        with inert_guard(),patch('lib.tts_backends.qwen_native.NativeQwenBackend') as backend:
            with self.assertRaises(Refused):
                native_child.infer(self.spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory,on_loaded=loaded)
        backend.assert_not_called();loaded.assert_not_called()

    def test_actual_inference_function_complete_loader_and_encoder(self):
        # Seal only the tiny source fixtures; no real model or installed tree.
        for path in self.fixture.bundle.iterdir(): path.chmod(0o400)
        self.fixture.bundle.chmod(0o500)
        events=[]
        torch=types.ModuleType('torch');torch.dtype=str
        torch.float32='float32';torch.float16='float16';torch.bfloat16='bfloat16'
        torch.backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda:True))
        torch.set_num_threads=lambda n:events.append(('threads',n))
        def generate(**kwargs):
            events.append(('generate',kwargs))
            return [[.1,-.1]],24000
        model=types.SimpleNamespace(generate_custom_voice=generate,
                                    device=types.SimpleNamespace(type='mps'),
                                    model=types.SimpleNamespace(dtype='torch.float32'))
        factory=Mock(return_value=model)
        hub=types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError('download')))
        # Force the complete loader to use injected Torch even if another test
        # already imported lib.tts. No file/function AST extraction.
        saved=sys.modules.pop('lib.tts',None)
        try:
            with inert_guard(), patch.dict(sys.modules,dict(torch=torch,
                    qwen_tts=types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory)),
                    huggingface_hub=hub)), patch.dict(os.environ), \
                    patch('platform.system',return_value='Darwin'),patch('platform.machine',return_value='arm64'):
                result=native_child.infer(self.spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory,on_loaded=lambda:events.append(('loaded',None)))
                self.assertEqual(result['data'][:4],b'RIFF')
                self.assertEqual(result['size_bytes'],48)
                self.assertEqual(result['identity']['device'],'mps')
                self.assertEqual(events[0],('threads',2))
                self.assertEqual([event[0] for event in events],['threads','loaded','generate'])
                self.assertEqual(factory.call_args.kwargs,dict(device_map='mps',dtype='float32',attn_implementation='sdpa',local_files_only=True))
                from lib.tts_backends.qwen_native import NativeQwenBackend
                long_spec=replace(self.spec,timeout_s=120)
                with self.assertRaisesRegex(Refused,'native_execution_limit'):
                    native_child.infer(long_spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory)
                received=[]; original=NativeQwenBackend._parameters
                def parameters(backend,params,timeout_s):
                    received.append((backend._request_seconds,timeout_s))
                    return original(backend,params,timeout_s)
                with patch.object(NativeQwenBackend,'_parameters',parameters):
                    diagnostic=native_child.infer(long_spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=120)
                self.assertEqual(received,[(120,120)])
                self.assertEqual(diagnostic['data'][:4],b'RIFF')
                for value in (True,119,121):
                    with self.assertRaisesRegex(Refused,'native_diagnostic_budget'):
                        native_child.infer(long_spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=value)
                default_backend=NativeQwenBackend(model_name=str(self.fixture.bundle),num_threads=2,dtype_str='float32',revision=self.spec.revision,offline=True,device='mps',attention='sdpa')
                params=dict(text='hello',language='English',temperature=.9,top_p=1.,top_k=50,repetition_penalty=1.05,max_new_tokens=256)
                with self.assertRaisesRegex(ValueError,'native_budget_required'):default_backend._parameters(params,120)
                default_backend.close()
                model.model.dtype='torch.float16'
                with self.assertRaisesRegex(ValueError,'native_dtype_mismatch'):
                    NativeQwenBackend(model_name=str(self.fixture.bundle),num_threads=2,dtype_str='float32',revision=self.spec.revision,offline=True,device='mps',attention='sdpa')
                model.model.dtype='torch.float32'
                # FP16 is a private explicit diagnostic selection, never fallback.
                import hashlib
                manifest=self.fixture.bundle/'manifest.json';original_manifest=manifest.read_bytes()
                changed=json.loads(original_manifest);changed['execution']['devices'].append(dict(device='mps',dtype='float16',attention='sdpa'))
                raw=json.dumps(changed).encode();manifest.chmod(0o600);manifest.write_bytes(raw);manifest.chmod(0o400)
                fp16=replace(long_spec,dtype='float16',manifest_sha256=hashlib.sha256(raw).hexdigest())
                model.model=types.SimpleNamespace(dtype='torch.float16')
                try:
                    with self.assertRaises(Refused):native_child.infer(fp16,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=120)
                    result=native_child.infer(fp16,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=120,diagnostic_dtype='float16')
                    self.assertEqual(result['data'][:4],b'RIFF')
                    self.assertEqual(factory.call_args.kwargs['dtype'],'float16')
                    for value in ('auto','bfloat16',True):
                        with self.assertRaisesRegex(Refused,'native_diagnostic_precision'):
                            native_child.infer(fp16,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=120,diagnostic_dtype=value)
                    model.model.dtype='torch.float32'
                    with self.assertRaisesRegex(ValueError,'native_dtype_mismatch'):
                        native_child.infer(fp16,'hello',{'bundle':self.fixture.bundle},{},self.inventory,diagnostic_request_seconds=120,diagnostic_dtype='float16')
                finally:
                    manifest.chmod(0o600);manifest.write_bytes(original_manifest);manifest.chmod(0o400)
                # Upstream ordinary Python spawn is refused before execution.
                model.generate_custom_voice=lambda **kw:subprocess.Popen(['never'])
                with self.assertRaises(native_child.SpawnRefused):
                    native_child.infer(self.spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory)
                hub.snapshot_download.assert_not_called()
                model.device.type='cpu'
                from lib.tts_backends.qwen_native import NativeQwenBackend
                with self.assertRaisesRegex(ValueError,'native_device_mismatch'):
                    NativeQwenBackend(model_name=str(self.fixture.bundle),num_threads=2,
                        dtype_str='float32',revision=self.spec.revision,offline=True,
                        device='mps',attention='sdpa')
        finally:
            sys.modules.pop('lib.tts',None)
            if saved is not None:sys.modules['lib.tts']=saved

    def test_sealed_native_inputs_are_fully_verified_once_before_load(self):
        from worker.engines import qwen
        for path in self.fixture.bundle.iterdir():path.chmod(0o400)
        self.fixture.bundle.chmod(0o500)
        backend=Mock(side_effect=RuntimeError('stop before model load'))
        with inert_guard(),patch.object(qwen,'verify_inputs',wraps=qwen.verify_inputs) as verify, \
                patch('lib.tts_backends.qwen_native.NativeQwenBackend',backend):
            with self.assertRaisesRegex(RuntimeError,'stop before model load'):
                native_child.infer(self.spec,'hello',{'bundle':self.fixture.bundle},{},self.inventory)
        verify.assert_called_once();backend.assert_called_once()

    def test_clone_prompt_once_exact_reference_transcript_language_and_failure(self):
        # Exercise the actual native backend method, with no upstream/weights.
        from lib.tts_backends.qwen_native import NativeQwenBackend
        backend=object.__new__(NativeQwenBackend)
        backend._request_seconds=30;backend._checkpoint='/sealed/base';backend._revision='pinned'
        model=Mock();backend._model=model
        prompt=object();model.create_voice_clone_prompt.return_value=prompt
        model.generate_voice_clone.return_value=([[.1,-.1]],24000)
        params=dict(text='audition',language='English',temperature=.9,top_p=1.,top_k=50,repetition_penalty=1.05,max_new_tokens=256)
        for language,transcript in (('English','Reference words.'),('French','Les mots de référence.')):
            model.reset_mock();params['language']=language
            result=backend.synthesise_clone(params,'/sealed/reference.wav',transcript,False,30)
            self.assertEqual(result.sample_rate,24000)
            model.create_voice_clone_prompt.assert_called_once_with(ref_audio='/sealed/reference.wav',ref_text=transcript,x_vector_only_mode=False)
            model.generate_voice_clone.assert_called_once_with(**params,voice_clone_prompt=prompt)
        for transcript in (None,'','   '):
            model.reset_mock()
            with self.assertRaisesRegex(ValueError,'reference_conditioning'):
                backend.synthesise_clone(params,'/sealed/reference.wav',transcript,False,30)
            model.create_voice_clone_prompt.assert_not_called();model.generate_voice_clone.assert_not_called()
        model.generate_voice_clone.side_effect=RuntimeError('retained upstream cause')
        with self.assertRaisesRegex(RuntimeError,'retained upstream cause'):
            backend.synthesise_clone(params,'/sealed/reference.wav','Exact transcript',False,30)


if __name__ == '__main__': unittest.main()
