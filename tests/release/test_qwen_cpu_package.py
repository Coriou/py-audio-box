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
        required={'service/shared_models.py','service/shared_model_helper.py','service/shared_model_client.py','service/installed_qwen_daemon.py','docs/agent-interface/QWEN-CPU.md','requirements/qwen-cpu.in','requirements/qwen-cpu.lock','requirements/qwen-installed.lock','requirements/qwen-sox-build.lock','requirements/tts/qwen-0.6b-base.expected.json','service/Dockerfile.release'}
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
        import zipfile
        import subprocess
        import sys
        import tempfile
        files=json.loads((ROOT/'headless-dist-manifest.json').read_text())['files']
        with tempfile.TemporaryDirectory() as tmp:
            from service.package_install import archive
            package=Path(tmp)/'headless.zip'
            archive(ROOT,package)
            with zipfile.ZipFile(package) as source:
                self.assertEqual(set(source.namelist()),{'speech-box/'+name for name in files}|{'speech-box/speech-box'})
                source.extractall(tmp)
            target=Path(tmp)/'speech-box'
            code="""
import importlib, importlib.abc, pathlib, sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torchaudio','numpy','soundfile','transformers','qwen_tts','huggingface_hub'}:
            raise AssertionError('ML import forbidden: '+fullname)
sys.meta_path.insert(0,Block())
for name in ('service.shared_models','service.shared_model_helper','service.shared_model_client','service.installed_qwen_daemon','service.daemon_client','service.daemon_helper','service.daemon_runtime','service.install_child','service.entrypoint','worker.runtime_supervisor','worker.runtime_conditioning'):
    module=importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(sys.argv[1])
from service.package_install import main
import runpy
launcher=str(pathlib.Path(sys.argv[1])/'speech-box')
for route in ('docker-models','docker-qwen','docker','docker-pocket','docker-export'):
    sys.argv=[launcher,route,'--help']
    try: runpy.run_path(launcher,run_name='__main__')
    except SystemExit as exc: assert exc.code == 0
    else: raise AssertionError('help did not exit: '+route)
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(target)],cwd=tmp,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_documented_shared_preparation_uses_actual_descriptor_and_selection(self):
        """Execute the published preparation scripts; no payload/daemon/ML needed."""
        import os
        import re
        import tempfile
        from unittest.mock import patch
        from service import shared_models as sm
        from service.daemon_client import encoded
        document=(ROOT/'docs/agent-interface/DOCKER-VOLUMES.md').read_text()
        scripts=re.findall(r"python3 - <<'PY_SHARED_(?:REQUEST|SELECTION)'\n(.*?)\nPY_SHARED_(?:REQUEST|SELECTION)",document,re.S)
        self.assertEqual(len(scripts),2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'requirements/tts').mkdir(parents=True)
            descriptor=ROOT/'requirements/tts/qwen-0.6b-base.expected.json'
            (root/'requirements/tts'/descriptor.name).write_bytes(descriptor.read_bytes())
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ,BUNDLE=str(root/'preprovided')):
                    exec(compile(scripts[0],'<documented-request>','exec'),{})
                request=json.loads(Path('model-import.json').read_text())
                sm.closed(request,'descriptor sources')
                key,raw,files,dirs=sm.descriptor(request['descriptor'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),key)
                self.assertEqual(set(request['sources']),{e['path'] for e in request['descriptor']['files']})
                self.assertTrue(all(Path(v).is_absolute() for v in request['sources'].values()))
                self.assertEqual(json.loads(Path('model-check.json').read_text()),dict(descriptor=request['descriptor'],sources={}))
                attempt='a'*32
                def volume(name,labels):
                    return dict(Name=name,Driver='local',Options=None,Scope='local',Mountpoint='/volumes/'+name,CreatedAt='retained-incarnation',Labels=labels)
                catalog=volume(sm.CATALOG,sm.labels())
                models=volume('pab-model-sha256-'+key+'-'+attempt,sm.labels(key,attempt))
                # Selection construction must preserve the entire returned object
                # in its digest, rather than hashing only the candidate or K.
                published=dict(catalog=catalog,candidate=models,key=key,attempt=attempt,observation={'retained':'proof'})
                Path('publication.json').write_bytes(encoded(published))
                exec(compile(scripts[1],'<documented-selection>','exec'),{})
                selection=json.loads(Path('shared-selection.json').read_text())
                sm.closed(selection,'descriptor catalog models publication_sha256')
                self.assertEqual(selection['descriptor'],request['descriptor'])
                self.assertEqual(sm.volume(selection['catalog']),catalog)
                self.assertEqual(sm.volume(selection['models'],key,attempt),models)
                self.assertEqual(selection['publication_sha256'],sm.digest(published))
                self.assertNotEqual(selection['publication_sha256'],key)
                # Exclusive creation is part of the documented operator contract.
                with self.assertRaises(FileExistsError):
                    exec(compile(scripts[1],'<documented-selection>','exec'),{})
            finally: os.chdir(previous)

    def test_extracted_native_help_closure_without_daemon_modules_or_ml(self):
        import zipfile
        import subprocess
        import sys
        import tempfile
        files=json.loads((ROOT/'native-dist-manifest.json').read_text())['files']
        self.assertTrue(set(files).isdisjoint({'service/shared_models.py','service/shared_model_helper.py','service/shared_model_client.py'}))
        self.assertEqual(len(files),len(set(files)))
        with tempfile.TemporaryDirectory() as tmp:
            from service.native_package import archive
            package=Path(tmp)/'native.zip'
            archive(package)
            with zipfile.ZipFile(package) as source:
                self.assertEqual(set(source.namelist()),{'native-worker/'+name for name in files}|{'native-worker/native-worker.py'})
                source.extractall(tmp)
            target=Path(tmp)/'native-worker'
            code="""
import importlib.abc, pathlib, sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torchaudio','numpy','soundfile','transformers','qwen_tts','huggingface_hub','packaging','mcp'}:
            raise AssertionError('optional dependency imported: '+fullname)
sys.meta_path.insert(0,Block())
from service import native_package
assert pathlib.Path(native_package.__file__).is_relative_to(sys.argv[1])
import runpy
sys.argv=[str(pathlib.Path(sys.argv[1])/'native-worker.py'),'--help']
try: runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc: assert exc.code == 0
else: raise AssertionError('help did not exit')
assert not any(name.startswith('service.shared_model') for name in sys.modules)
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(target)],cwd=tmp,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('native-qwen-v1',result.stdout)
            self.assertIn('runtime-manifest',result.stdout)

    def test_documented_store_commands_reach_real_parser_and_intent(self):
        import contextlib
        import io
        import os
        import shlex
        import tempfile
        from unittest.mock import patch
        from service import shared_model_client as client
        document=(ROOT/'docs/agent-interface/DOCKER-VOLUMES.md').read_text()
        commands=[line for line in document.splitlines() if line.startswith('python3 speech-box docker-models ')]
        self.assertEqual(len(commands),4)
        expected=json.loads((ROOT/'requirements/tts/qwen-0.6b-base.expected.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); previous=Path.cwd()
            try:
                os.chdir(root)
                for name in ('model-import.json','model-check.json'):
                    Path(name).write_text(json.dumps(dict(descriptor=expected,sources={})))
                seen=[]
                def run(own,operation,attempt=None):
                    seen.append((operation,attempt,own.expected,own.sources))
                    return {'parser_boundary':'no daemon invoked'}
                # Real main/parser, request admission, source admission and durable
                # intent. Replace only Docker and execution beyond that boundary.
                with patch.object(client,'Docker'), patch.object(client.Client,'run',run):
                    for line in commands:
                        line=line.replace('"$CONTEXT"','orbstack').replace('"$IMAGE"','sha256:'+'b'*64).replace('"$ATTEMPT"','c'*32)
                        args=shlex.split(line)[3:]
                        if '>' in args: args=args[:args.index('>')]
                        with contextlib.redirect_stdout(io.StringIO()): client.main(args)
                        output=Path(args[args.index('--output')+1])
                        intent=json.loads((output/'intent.json').read_text())
                        self.assertEqual(intent['operation'],args[0])
                        self.assertEqual(intent['attempt'],'c'*32 if args[0]=='recover-import' else None)
                        self.assertEqual(output.stat().st_mode & 0o777,0o700)
                self.assertEqual([v[0] for v in seen],['init','import','verify','recover-import'])
                self.assertIsNone(seen[0][2])
                self.assertTrue(all(v[2]==expected and v[3]=={} for v in seen[1:]))
                # Recovery is an exact-attempt selection, not an implicit retry.
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    client.main(['recover-import','--context','orbstack','--image','sha256:'+'b'*64,'--request','model-check.json','--output','missing-attempt'])
                self.assertFalse(Path('missing-attempt').exists())
            finally: os.chdir(previous)
