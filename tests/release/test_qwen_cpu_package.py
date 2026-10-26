"""Source closure only. Does not create a release artifact from dirty source."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]

class QwenCPUArchiveTests(unittest.TestCase):
    def test_archive_contains_existing_recipe_dependencies_and_family(self):
        files=json.loads((ROOT/'headless-dist-manifest.json').read_text())['files']
        required={'service/installed_qwen_daemon.py','docs/agent-interface/QWEN-CPU.md','requirements/qwen-cpu.in','requirements/qwen-cpu.lock','requirements/qwen-installed.lock','requirements/qwen-sox-build.lock','requirements/tts/qwen-0.6b-base.expected.json','service/Dockerfile.release'}
        self.assertTrue(required<=set(files))
        self.assertEqual(len(files),len(set(files)))
        for name in files:self.assertTrue((ROOT/name).is_file(),name)
        for name in ('service/installed_qwen_daemon.py','worker/runtime_supervisor.py','service/daemon_helper.py','service/entrypoint.py'):
            ast.parse((ROOT/name).read_text())

    def test_canonical_manifest_pin_and_eleven_file_descriptor(self):
        from service import installed_qwen_daemon as qd
        from service.provisioning import bundle_manifest,manifest_bytes
        value=json.loads((ROOT/'requirements/tts/qwen-0.6b-base.expected.json').read_text())
        self.assertEqual(len(value['files']),11)
        self.assertEqual(hashlib.sha256(manifest_bytes(bundle_manifest(value))).hexdigest(),qd.MANIFEST_SHA256)
        qd.manifest(bundle_manifest(value))
        value['files'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):qd.manifest(bundle_manifest(value))

    def test_extracted_stdlib_entrypoints_resolve_without_ml(self):
        import shutil
        import subprocess
        import sys
        import tempfile
        files=json.loads((ROOT/'headless-dist-manifest.json').read_text())['files']
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'speech-box';target.mkdir()
            for name in files:
                path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,path)
            code="""
import importlib, importlib.abc, pathlib, sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torchaudio','numpy','soundfile','transformers','qwen_tts','huggingface_hub'}:
            raise AssertionError('ML import forbidden: '+fullname)
sys.meta_path.insert(0,Block())
for name in ('service.installed_qwen_daemon','service.daemon_client','service.daemon_helper','service.daemon_runtime','service.install_child','service.entrypoint','worker.runtime_supervisor','worker.runtime_conditioning'):
    module=importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(sys.argv[1])
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(target)],cwd=tmp,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
