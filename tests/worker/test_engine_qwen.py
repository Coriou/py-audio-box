import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import wave
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from worker.engines.qwen import execute
from worker.engines.qwen_execution import QwenExecution
from worker.hardware import Host, Device, DeviceFamily, HardwareInventory
from worker.transport import Refused
from lib.tts_backends.contract import AudioResult


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.root.chmod(0o700)
        self.bundle = self.root/'bundle'; self.bundle.mkdir()
        self.backend = Mock()
        self.backend.synthesise_preset.return_value = AudioResult([0.1, -0.1],24000)
        self.backend.synthesise_clone.return_value = AudioResult([0.1],24000)
        self.backend.synthesise_design.return_value = AudioResult([0.1],24000)
        self.factory = Mock(return_value=self.backend)
        self.inventory = HardwareInventory(Host('linux','x86_64','native'),(Device(DeviceFamily.CPU,0,True,True,'trusted'),))
        self.make_bundle()

    def make_bundle(self, mode='custom_voice', size='1b7'):
        files = {'config.json':json.dumps(dict(model_type='qwen3_tts',tts_model_type=mode,tts_model_size=size)).encode(), 'generation_config.json':b'{}','model.safetensors':b'inert fixture'}
        entries=[]
        for name,raw in files.items():
            (self.bundle/name).write_bytes(raw)
            entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.manifest=dict(schema_version=1,backend='qwen',checkpoint='operator-checkpoint',revision='exact-revision',language='English',files=entries,execution=dict(sample_rate=24000,languages=['English'],presets=['Ryan'],devices=[dict(device=d,dtype='float32',attention='sdpa') for d in ('cpu','cuda:1','mps')]))
        self.save_manifest()

    def save_manifest(self):
        raw=json.dumps(self.manifest).encode(); (self.bundle/'manifest.json').write_bytes(raw)
        self.e=QwenExecution('bundle',hashlib.sha256(raw).hexdigest(),'operator-checkpoint','exact-revision','custom_voice','English','voice','v1',2,100,100,244,4096,24000,preset='Ryan')

    def run_engine(self,e=None,refs=None):
        for method in ('synthesise_preset','synthesise_clone','synthesise_design'):
            result=getattr(self.backend,method).return_value
            if not result.metadata: result.metadata=dict(checkpoint=str(self.bundle),revision='exact-revision')
        return execute(e or self.e,'hello',{'bundle':self.bundle},refs or {},self.inventory,reference_temp_dir=self.root,backend_factory=self.factory)

    def test_cpu_exact_forwarding_and_pcm(self):
        result=self.run_engine()
        self.assertEqual(result['frames'],2); self.assertEqual(result['size_bytes'],48)
        self.assertEqual(result['identity']['revision'],'exact-revision')
        self.factory.assert_called_once_with(model_name=str(self.bundle),num_threads=2,dtype_str='float32',revision='exact-revision',offline=True,device='cpu',attention='sdpa')
        self.backend.close.assert_called_once()
        with wave.open(io.BytesIO(result['data'])) as w: self.assertEqual((w.getnchannels(),w.getsampwidth(),w.getframerate()),(1,2,24000))

    def test_cuda_and_native_metal(self):
        for device,family,host in [('cuda:1',DeviceFamily.CUDA,Host('linux','x86_64','docker')),('mps',DeviceFamily.MPS,Host('darwin','arm64','native'))]:
            with self.subTest(device=device):
                self.inventory=HardwareInventory(host,(Device(family,1 if family==DeviceFamily.CUDA else 0,True,True,'trusted'),))
                self.run_engine(replace(self.e,device=device))
                self.assertEqual(self.factory.call_args.kwargs['device'],device)

    def test_metal_docker_and_unavailable_refuse_without_fallback(self):
        for host,available in [(Host('darwin','arm64','docker'),True),(Host('linux','arm64','docker'),True),(Host('darwin','arm64','native'),False)]:
            self.inventory=HardwareInventory(host,(Device(DeviceFamily.MPS,0,True,available,'trusted'),))
            with self.assertRaises(Refused): self.run_engine(replace(self.e,device='mps'))
        self.factory.assert_not_called()

    def test_closed_controls(self):
        for changes in [dict(seed=1),dict(device='auto'),dict(device='cuda'),dict(dtype='auto'),dict(dtype='float16'),dict(attention='flash_attention_2'),dict(task='other'),dict(reference_id='x'),dict(threads=True)]:
            with self.subTest(changes=changes), self.assertRaises(Refused): replace(self.e,**changes)
        with self.assertRaises(Refused): QwenExecution.from_wire(dict(self.e.identity(),extra=1))

    def test_manifest_identity_hash_capabilities_and_unlisted_files(self):
        for mutation in ['revision','rate','language','device','preset','hash','extra']:
            with self.subTest(mutation=mutation):
                self.make_bundle()
                if mutation=='revision': self.manifest['revision']='wrong'
                if mutation=='rate': self.manifest['execution']['sample_rate']=22050
                if mutation=='language': self.manifest['execution']['languages']=[]
                if mutation=='device': self.manifest['execution']['devices']=[]
                if mutation=='preset': self.manifest['execution']['presets']=[]
                if mutation=='hash': self.manifest['files'][0]['sha256']='0'*64
                if mutation=='extra': (self.bundle/'unlisted.json').write_text('{}')
                self.save_manifest()
                with self.assertRaises(Refused): self.run_engine()
        self.factory.assert_not_called()

    def test_checkpoint_instruction_design(self):
        self.make_bundle(size='0b6')
        with self.assertRaises(Refused): self.run_engine(replace(self.e,instruct='warm'))
        self.make_bundle(mode='voice_design')
        self.run_engine(replace(self.e,task='voice_design',preset=None,instruct='warm'))
        self.backend.synthesise_design.assert_called_once()
        with self.assertRaises(Refused): self.run_engine()

    def test_clone_exact_reference_and_conditioning(self):
        self.make_bundle(mode='base')
        out=io.BytesIO()
        with wave.open(out,'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\x00\x00'*4)
        raw=out.getvalue(); path=self.root/'ref.wav'; path.write_bytes(raw)
        e=replace(self.e,task='clone',preset=None,reference_id='ref',reference_sha256=hashlib.sha256(raw).hexdigest(),reference_bytes=len(raw),reference_frames=4,reference_rate=24000,reference_text='source')
        r=dict(path=path,voice_id='voice',voice_revision='v1',checkpoint=e.checkpoint,revision=e.revision,text='source')
        self.run_engine(e,{'ref':r})
        self.backend.synthesise_clone.assert_called_once()
        for key in ('voice_id','voice_revision','checkpoint','revision','text'):
            with self.subTest(key=key),self.assertRaises(Refused): self.run_engine(e,{'ref':dict(r,**{key:'wrong'})})
        path.write_bytes(raw[:-1]+b'x')
        with self.assertRaises(Refused): self.run_engine(e,{'ref':r})

    def test_managed_preparation_pair_has_no_named_snapshot_and_original_remaining_time(self):
        import os
        from contextlib import nullcontext
        from unittest.mock import patch
        from worker.engines.qwen_prepare import QwenPreparation
        self.make_bundle(mode='base')
        out=io.BytesIO()
        with wave.open(out,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','none'));w.writeframes(b'\0\0'*4)
        raw=out.getvalue();path=self.root/'ref.wav';path.write_bytes(raw);path.chmod(0o600)
        e=replace(self.e,task='clone',preset=None,reference_id='ref',reference_sha256=hashlib.sha256(raw).hexdigest(),
            reference_bytes=len(raw),reference_frames=4,reference_rate=24000,reference_text=None,x_vector_only=True,timeout_s=1800)
        spec=QwenPreparation(**e.identity())
        refs={'ref':dict(path='/must/not/open',voice_id=e.voice_id,voice_revision=e.voice_revision,checkpoint=e.checkpoint,revision=e.revision,text=None)}
        self.backend.synthesise_clone.return_value.metadata=dict(checkpoint=str(self.bundle),revision=e.revision)
        fd=os.open(path,os.O_RDONLY)
        try:
            with patch('lib.tts_backends.pocket_snapshot.managed_directory',side_effect=lambda p:nullcontext()), \
                    patch('worker.engines.qwen.tempfile.NamedTemporaryFile',side_effect=AssertionError('named snapshot')), \
                    patch('time.monotonic',return_value=140.):
                result=execute(spec,'audition',{'bundle':self.bundle},refs,self.inventory,reference_temp_dir=self.root,
                    backend_factory=self.factory,reference_fd=fd,managed_namespace=True,managed_owner_uid=os.getuid(),cutoff=1900.)
            self.assertEqual(result['reference']['data'],raw)
            self.assertNotEqual(result['reference']['sha256'],result['audition']['sha256'])
            call=self.backend.synthesise_clone.call_args.args
            self.assertEqual(call[1:],(raw,None,True,1760.))
            self.factory.reset_mock()
            with patch('time.monotonic',return_value=1901.),self.assertRaisesRegex(Refused,'expired'):
                execute(spec,'audition',{'bundle':self.bundle},refs,self.inventory,reference_temp_dir=self.root,
                    backend_factory=self.factory,reference_fd=fd,managed_namespace=True,managed_owner_uid=os.getuid(),cutoff=1900.)
            self.factory.assert_not_called()
        finally:os.close(fd)

    def test_result_identity_mismatch(self):
        self.backend.synthesise_preset.return_value.metadata=dict(checkpoint='wrong',revision='exact-revision')
        with self.assertRaises(Refused): self.run_engine()
        self.backend.close.assert_called_once()

    def test_output_bounds_and_close_failure(self):
        for samples,rate in [([0.]*101,24000),([1.1],24000),([0.],22050)]:
            self.backend.synthesise_preset.return_value=AudioResult(samples,rate)
            with self.assertRaises(Refused): self.run_engine()
        result=AudioResult([0.],24000); result.samples=[float('nan')]
        self.backend.synthesise_preset.return_value=result
        with self.assertRaises(Refused): self.run_engine()
        self.backend.synthesise_preset.return_value=AudioResult([0.],24000)
        self.backend.close.side_effect=RuntimeError('reap failed')
        with self.assertRaisesRegex(RuntimeError,'reap failed'): self.run_engine()

    def test_local_loader_exact_device_and_no_auto_fallback(self):
        import types
        from unittest.mock import patch
        tree=ast.parse((ROOT/'lib/tts.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_load_tts_model_local')
        bootstrap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qwen_model_class')
        torch=Mock(); torch.cuda.is_available.return_value=True; torch.cuda.device_count.return_value=2
        torch.backends.mps.is_available.return_value=True
        model=Mock(); model.from_pretrained.return_value=types.SimpleNamespace()
        env=dict(Path=Path,torch=torch,_DTYPE_MAP={'float32':'fp32'},best_dtype=Mock(return_value='fp32'),get_device=Mock(side_effect=AssertionError('auto policy called')),_has_flash_attn=Mock(side_effect=AssertionError('auto attention called')))
        exec(compile(ast.Module(body=[bootstrap,node],type_ignores=[]),'loader','exec'),env)
        with patch.dict(sys.modules,{'qwen_tts':types.SimpleNamespace(Qwen3TTSModel=model),'huggingface_hub':types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError('acquisition called')))}), patch('platform.system',return_value='Darwin'),patch('platform.machine',return_value='arm64'):
            for device in ('cpu','cuda:1','mps'):
                env['_load_tts_model_local'](str(self.bundle),2,'float32','exact-revision',True,device,'eager')
                self.assertEqual(model.from_pretrained.call_args.kwargs['device_map'],device)
                self.assertEqual(model.from_pretrained.call_args.kwargs['attn_implementation'],'eager')
            torch.cuda.is_available.return_value=False
            with self.assertRaises(ValueError): env['_load_tts_model_local'](str(self.bundle),2,'float32','r',True,'cuda:1','sdpa')
        self.assertEqual(model.from_pretrained.call_count,3)

    def test_loader_forwarding_without_importing_torch(self):
        tree=ast.parse((ROOT/'lib/tts.py').read_text())
        loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_tts_model')
        # Execute only the forwarding function with controlled globals.
        env=dict(QwenBackend=Mock(),os=Mock(),env_flag_enabled=Mock(return_value=False))
        env['os'].getenv.return_value=None
        exec(compile(ast.Module(body=[loader],type_ignores=[]),'loader','exec'),env)
        env['load_tts_model']('local',2,revision='r',offline=True)
        self.assertNotIn('device',env['QwenBackend'].call_args.kwargs)
        env['load_tts_model']('local',2,'float32',revision='r',offline=True,device='mps',attention='eager')
        self.assertEqual(env['QwenBackend'].call_args.kwargs['device'],'mps')

if __name__=='__main__': unittest.main()
