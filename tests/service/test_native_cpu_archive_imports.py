"""Real extracted control/lifespan imports, no models or installed admission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from service import native_cpu_package as package


PROBE=r'''
import asyncio,json,os,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]).resolve();installation=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root));os.chdir(root)
assert sys.flags.isolated and not sys.flags.no_site
from service.native_cpu_setup import _control_imports
_control_imports()
from service import native_cpu_setup as setup
from service.native_cpu_install import Prepared,locked_versions
from service.dispatcher import Dispatcher
from service.native_cpu_activation import NativeCPUActivation
from service.entrypoint import _INSTALLED_LIFECYCLES
from worker.native_child import install_spawn_guard,SpawnRefused
import subprocess
cached=subprocess.Popen
install_spawn_guard()
# Only host/model ownership is inert. Product factory, acquisition import,
# activation type gate, ASGI mounting, credentials and shutdown are real paths.
installation.mkdir(mode=0o700)
credential=installation/'credential.json'
credential.write_text(json.dumps({'version':1,'credentials':[{'token':'f'*96,'principal':'local','scopes':['read','production-write']}]}))
credential.chmod(0o600)
closed=[]
def init(self,path,*,qualification=False):
    assert Path(path)==installation and qualification is True
    self.root=installation;self.qualification=None;self.selftest=False;self.config={'principal':'local'}
    self.control=installation/'control';self.staging=installation/'staging'
    self.revalidate=lambda:'inert-source-only'
class Owner:
    installed_checked=True
    def call(self,op):
        assert op=='status';return {'reconciled':True,'persistent_activation':True}
    def close(self):closed.append('owner')
def dispatcher(*args,**kwargs):
    owner=Owner();owner.installed_activation=kwargs['installed_activation']
    assert type(owner.installed_activation) is NativeCPUActivation
    return owner
async def exercise():
    with patch.object(Prepared,'__init__',init),patch.object(Prepared,'close',lambda self:closed.append('prepared')),patch('service.dispatcher.Dispatcher',dispatcher):
        app=setup.create_app(installation,qualification=True)
        async with app.router.lifespan_context(app):
            assert app.state.dispatcher.installed_checked
            assert getattr(app.state,'installed_mount',None) is not None
        assert app.state.installed_lifecycle['closed'] and not _INSTALLED_LIFECYCLES
asyncio.run(exercise())
assert closed==['owner','prepared'],closed
for spawn in (subprocess.Popen,cached):
    try:spawn(['never-execute-extracted-lifespan-test'])
    except SpawnRefused:pass
    else:raise AssertionError('permanent spawn guard lost')
counts={name:0 for name in ('service','clients','contracts','worker','lib')}
for name,module in tuple(sys.modules.items()):
    family=name.split('.')[0];path=getattr(module,'__file__',None)
    if family in counts and path:
        assert Path(path).resolve().is_relative_to(root),(family,Path(path).name)
        counts[family]+=1
    elif path and not str(path).startswith('<'):
        resolved=Path(path).resolve()
        assert any(resolved.is_relative_to(Path(prefix).resolve()) for prefix in (sys.prefix,sys.base_prefix)),('foreign_dependency_path',Path(path).name)
assert all(counts.values()),counts
# Cheap Apple installs exactly all three locks. Normal Unit has its independent
# locked test image; it runs the same extracted source/lifespan path.
if sys.platform=='darwin':
    from importlib.metadata import version
    pins=locked_versions(root)
    assert len(pins)==44 and all(version(name)==value for name,value in pins.items())
print(json.dumps({'extracted_lifespan':'passed','modules':counts,'spawn_refused':True,'weights_loaded':False}))
'''


EXTRACTED_CORE_PROBE=r'''
import importlib.abc,importlib.machinery,os,sys,types,unittest,json
from pathlib import Path
root=Path(sys.argv[1]).resolve();fixture_root=Path(sys.argv[2]).resolve()
os.chdir(root);sys.path.insert(0,str(root))
products={'service','worker','clients','contracts','lib'}
class ClosedSource(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        family=fullname.split('.')[0]
        if family=='tests':return None
        search=([str(root.joinpath(*fullname.split('.')[:-1]))] if path is not None else [str(root)]) if family in products else path
        spec=importlib.machinery.PathFinder.find_spec(fullname,search)
        if spec is None:
            if family in products:raise ModuleNotFoundError('module absent from selected extraction',name=fullname)
            return None
        paths=[] if spec.origin in (None,'built-in','frozen') else [Path(spec.origin).resolve()]
        paths.extend(Path(p).resolve() for p in (spec.submodule_search_locations or ()))
        allowed=[root] if family in products else [Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
        if any(not any(p.is_relative_to(a) for a in allowed) for p in paths):
            raise ModuleNotFoundError('source outside selected extraction/interpreter',name=fullname)
        return spec
sys.meta_path.insert(0,ClosedSource())
fixtures=types.ModuleType('tests');fixtures.__path__=[str(fixture_root/'tests')];sys.modules['tests']=fixtures
# Fixtures may import only their test helpers. Product imports are forcibly
# resolved against the extracted archive even if a helper prepends checkout.
from tests.service.test_native_cpu_public_source import CPUPublicSourceTests
sys.path[:]=[p for p in sys.path if p!=str(fixture_root)]
assert not (root/'service/installed_pocket_daemon.py').exists()
assert not (root/'service/installed_qwen_daemon.py').exists()
from tests.service.test_native_cpu_measurement import MeasurementTests
cases=[CPUPublicSourceTests('test_real_tls_custodian_large_core_publication_and_shutdown'),
       MeasurementTests('test_doctor_selftest_actual_asgi_routes_publish_only_after_owned_close'),
       MeasurementTests('test_actual_public_proof_derives_profile_without_private_scope')]
result=unittest.TestResult()
for case in cases:case.run(result)
if not result.wasSuccessful():
    for _,failure in result.errors+result.failures:print(failure,file=sys.stderr)
    raise SystemExit(1)
assert not {'service.installed_pocket_daemon','service.installed_qwen_daemon'}.intersection(sys.modules)
for name,module in tuple(sys.modules.items()):
    path=getattr(module,'__file__',None)
    if name.split('.')[0] in products and path:assert Path(path).resolve().is_relative_to(root)
if sys.platform=='darwin':
    from service.native_cpu_install import locked_versions
    from importlib.metadata import version
    pins=locked_versions(root);assert len(pins)==44 and all(version(k)==v for k,v in pins.items())
print(json.dumps({'extracted_real_core_tls':'passed','extracted_selftest_ordinary':'passed','foreign_family_imports':False,'weights_loaded':False}))
'''


class CPUArchiveImportTests(unittest.TestCase):
    def test_extracted_isolated_control_lifespan_and_permanent_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve();archive=directory/'cpu.zip';package.archive(archive);package.verify_archive(archive)
            with zipfile.ZipFile(archive) as source:source.extractall(directory/'extracted')
            root=directory/'extracted/native-worker'
            env={key:value for key,value in os.environ.items() if key in ('PATH','SYSTEMROOT','TMPDIR')}
            env.update(HOME=str(directory),HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1')
            result=subprocess.run([sys.executable,'-I','-B','-c',PROBE,str(root),str(directory/'inert-installation')],
                cwd=directory,env=env,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr.decode('utf-8','replace')[:6000])
            report=json.loads(result.stdout);self.assertEqual(report['extracted_lifespan'],'passed')
            self.assertFalse(report['weights_loaded']);self.assertTrue(report['spawn_refused'])

    def test_dispatcher_lazy_foreign_family_imports_have_typed_cpu_guard(self):
        import ast
        tree=ast.parse((package.common.ROOT/'service/dispatcher.py').read_text())
        expected={'_checked_export_family','_local_qwen_active','_qwen_state_poll','_observation_interval'};found=set()
        for function in (node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)):
            imports=[node for node in ast.walk(function) if isinstance(node,ast.ImportFrom) and
                (node.module in ('service.installed_export','service.installed_qwen_daemon') or
                 node.module=='service' and any(alias.name=='installed_pocket_daemon' for alias in node.names))]
            if not imports:continue
            found.add(function.name)
            guard=next(node for node in function.body if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Constant))
            self.assertIsInstance(guard,ast.If);self.assertIsInstance(guard.test,ast.Call)
            self.assertEqual(guard.test.func.attr,'_native_cpu_relation')
            self.assertIsInstance(guard.body[0],ast.Return)
            self.assertTrue(all(guard.lineno<node.lineno for node in imports))
        self.assertEqual(found,expected)
        manifest=package.members()
        self.assertNotIn('service/installed_pocket_daemon.py',manifest)
        self.assertNotIn('service/installed_qwen_daemon.py',manifest)
        tree=ast.parse((package.common.ROOT/'service/production_selection.py').read_text())
        discovery=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='discovery')
        guarded=[node for node in discovery.body if isinstance(node,ast.If) and
            any(isinstance(child,ast.ImportFrom) and any(alias.name=='installed_qwen_daemon' for alias in child.names) for child in ast.walk(node))]
        self.assertEqual(len(guarded),1);self.assertIn('installed_family',ast.unparse(guarded[0].test))

    def test_extracted_real_dispatcher_core_tls_publication_ack_and_custodian_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve();archive=directory/'cpu.zip';package.archive(archive);package.verify_archive(archive)
            with zipfile.ZipFile(archive) as source:source.extractall(directory/'extracted')
            root=directory/'extracted/native-worker'
            env={key:value for key,value in os.environ.items() if key in ('PATH','SYSTEMROOT','TMPDIR')}
            env.update(HOME=str(directory),HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1')
            result=subprocess.run([sys.executable,'-I','-B','-c',EXTRACTED_CORE_PROBE,str(root),str(package.common.ROOT)],
                cwd=directory,env=env,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr.decode('utf-8','replace')[:8000])
            self.assertEqual(json.loads(result.stdout)['extracted_real_core_tls'],'passed')

    def test_extracted_archive_missing_shared_helper_retains_exact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve();archive=directory/'cpu.zip';package.archive(archive)
            with zipfile.ZipFile(archive) as source:source.extractall(directory/'extracted')
            root=directory/'extracted/native-worker';(root/'lib/tts_backends/pocket_snapshot.py').unlink()
            code="""import sys,json
sys.path.insert(0,sys.argv[1])
from worker.native_child import install_spawn_guard
install_spawn_guard()
try:
 from service import installed_export
except ModuleNotFoundError as exc:
 assert exc.name=='lib.tts_backends.pocket_snapshot',exc.name
 from service.entrypoint import _startup_diagnostic
 print(json.dumps(_startup_diagnostic(exc)))
else:raise AssertionError('missing archive helper unexpectedly imported')
"""
            env={key:value for key,value in os.environ.items() if key in ('PATH','SYSTEMROOT','TMPDIR')}
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(root)],cwd=directory,env=env,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr.decode('utf-8','replace')[:3000])
            report=json.loads(result.stdout);self.assertEqual(report['chain'][0]['module'],'lib.tts_backends.pocket_snapshot')
            self.assertEqual((report['chain'][0]['file'],report['chain'][0]['line']),('pocket.py',17))

    def test_startup_diagnostic_has_bounded_safe_source_and_explicit_cause(self):
        from service.entrypoint import _startup_diagnostic
        try:
            try:raise ModuleNotFoundError('secret arbitrary path /private/token',name='lib.tts_backends.pocket_snapshot')
            except ModuleNotFoundError as cause:raise RuntimeError('do not publish this message') from cause
        except RuntimeError as exc:report=_startup_diagnostic(exc)
        raw=json.dumps(report);self.assertNotIn('secret',raw);self.assertNotIn('do not publish',raw);self.assertNotIn('/private',raw)
        self.assertEqual(report['chain'][1]['module'],'lib.tts_backends.pocket_snapshot')
        self.assertTrue(all(set(item)<= {'type','file','function','line','module'} for item in report['chain']))
        exc=ModuleNotFoundError('hidden',name='x'*96);self.assertEqual(_startup_diagnostic(exc)['chain'][0]['module'],'redacted')
        exc.__cause__=exc;self.assertTrue(_startup_diagnostic(exc)['truncated'])
        previous=exc
        for _ in range(8):
            current=RuntimeError('hidden');current.__cause__=previous;previous=current
        report=_startup_diagnostic(previous);self.assertEqual(len(report['chain']),4);self.assertTrue(report['truncated'])
        self.assertLess(len(json.dumps(report)),2000)


if __name__=='__main__':unittest.main()
