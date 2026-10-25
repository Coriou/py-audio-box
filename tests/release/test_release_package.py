"""Inert package/finite-coordinator tests. Docker/host observations are injected."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from service import package_install as p, host_install as h, entrypoint as e
from tests.service import test_installed_host_policy as host_fixtures

ROOT=Path(__file__).resolve().parents[2]


class Transport:
    def __init__(self, fixture):
        self.context=fixture.scope['context']; self.deadline=0
        self.calls=[]; self.containers={}; self.marker={'phase':'waiting'}
        self.exit=0; self.auto_exit=True; self.fail=None
        self.config=fixture.config
    def _read(self,args):
        self.calls.append(args)
        if self.fail and self.fail(args): raise ValueError('injected_unknown')
        if args[:2]==['image','inspect']: return args[2]
        if args[:2]==['container','create']:
            role='control' if len(self.containers)==0 else 'worker'
            identity=('1' if role=='control' else '2')*64
            self.containers[identity]=dict(role=role,running=False,started='2026-10-24T01:00:00Z',exit=0)
            return identity
        if args[:2]==['container','start']:
            self.containers[args[2]]['running']=True
            if self.containers[args[2]]['role']=='control':self.marker={'phase':'waiting'}
            self.containers[args[2]]['started']='2026-10-24T02:00:00Z'
            return args[2]
        if args[:2]==['container','kill']:
            if self.auto_exit:
                self.containers[args[-1]]['running']=False
                self.containers[args[-1]]['exit']=self.exit
            return args[-1]
        if args[:2]==['container','inspect']:
            c=self.containers[args[2]]
            if '"exit"' in args[-1]:
                return json.dumps(dict(running=c['running'],restarting=False,exit=c['exit']))
            result=dict(id=args[2],image=self.config['images'][c['role']],started=c['started'])
            # Tests exercise the actual closed inspect decoder too.
            fields='finished running restarting pid restart pidmode privileged capadd cgroupns mounts user capdrop security readonly groups devices device_requests device_rules userns ipcns utsns memory swap pids nanocpus period quota cpuset network ports publish_all tmpfs'.split()
            result.update({k:None for k in fields})
            return json.dumps(result)
        if args[0]=='exec': return json.dumps(self.marker)
        raise AssertionError(args)


class Authority:
    def __init__(self, transport): self.events=[]; self.transport=transport; self.fail=None
    def event(self,name):
        self.events.append(name)
        if self.fail==name: raise ValueError('dirty_unknown')
    def initialize_authority(self): self.event('fresh')
    def preflight(self,owners): self.event('preflight')
    def wait_worker(self): self.event('hello')
    def register_and_notify(self,owners):
        self.event('register_notify')
        assert all(c['running'] for c in self.transport.containers.values())
        self.transport.marker=dict(phase='ready',registration_id='a'*64)
        return 'a'*64
    def confirm_ready(self,owners,rid): self.event('observe')
    def clean_shutdown(self,owners): self.event('clean_shutdown')


class FixtureLauncher(p.Launcher):
    """Only privileged protected document I/O is replaced; real disposable files."""
    def document(self,name,value=None):
        path=self.directory/name
        if value is not None:
            with path.open('x') as out: json.dump(value,out)
        return json.loads(path.read_text())


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        fixture=self.fixture=host_fixtures.InstalledHost();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.transport=Transport(fixture); self.authority=Authority(self.transport)
        self.now=0
        def sleep(seconds): self.now+=seconds
        self.launcher=FixtureLauncher(fixture.config,fixture.scope,fixture.policy,self.authority,self.temp.name,
            docker=self.transport,clock=lambda:self.now,sleep=sleep)
    def test_start_register_readiness_and_clean_same_id_restart(self):
        first=self.launcher.up(); ids=first['containers']
        self.assertEqual(first['phase'],'ready')
        self.assertEqual(self.authority.events,['fresh','hello','register_notify','observe'])
        starts=[a[2] for a in self.transport.calls if a[:2]==['container','start']]
        self.assertEqual(starts,[ids['control'],ids['worker']])
        self.launcher.stop()
        self.assertLess(self.authority.events.index('clean_shutdown'),self.authority.events.index('preflight'))
        creates=sum(a[:2]==['container','create'] for a in self.transport.calls)
        second=self.launcher.up()
        self.assertEqual(second['containers'],ids)
        self.assertEqual(sum(a[:2]==['container','create'] for a in self.transport.calls),creates)
        self.assertTrue(all('stop' not in a and 'rm' not in a and 'pull' not in a for a in self.transport.calls))
    def test_already_live_is_observed_without_registration(self):
        self.launcher.up(); self.authority.events.clear()
        self.launcher.up(); self.assertEqual(self.authority.events,['observe'])
    def test_partial_create_is_retained_and_never_retried(self):
        self.transport.fail=lambda a:a[:2]==['container','create'] and len(self.transport.containers)==1
        with self.assertRaises(ValueError): self.launcher.up()
        calls=len(self.transport.calls); self.transport.fail=None
        with self.assertRaises(FileExistsError): self.launcher.up()
        self.assertEqual(calls,len(self.transport.calls))
        self.assertTrue((Path(self.temp.name)/'created-control.json').exists())
    def test_register_failure_never_polls_ready_or_stops(self):
        self.authority.fail='register_notify'
        with self.assertRaises(ValueError): self.launcher.up()
        self.assertEqual(sum(a[0]=='exec' for a in self.transport.calls),1)  # waiting phase only
        self.assertFalse(any(a[:2]==['container','kill'] for a in self.transport.calls))
        self.assertTrue(all(c['running'] for c in self.transport.containers.values()))
    def test_dirty_preflight_does_not_start_retained_containers(self):
        self.launcher.up(); self.launcher.stop(); self.transport.calls.clear()
        self.authority.fail='preflight'
        with self.assertRaises(ValueError): self.launcher.up()
        self.assertFalse(any(a[:2]==['container','start'] for a in self.transport.calls))
    def test_dirty_shutdown_never_signals_worker(self):
        self.launcher.up(); self.authority.fail='clean_shutdown'
        with self.assertRaises(ValueError): self.launcher.stop()
        kills=[a for a in self.transport.calls if a[:2]==['container','kill']]
        self.assertEqual(len(kills),1)
        self.assertTrue(self.transport.containers['2'*64]['running'])
    def test_timeout_never_escalates(self):
        self.launcher.up(); self.transport.auto_exit=False
        self.launcher.end=self.now+.15
        with self.assertRaisesRegex(ValueError,'timeout'): self.launcher.stop()
        kills=[a for a in self.transport.calls if a[:2]==['container','kill']]
        self.assertEqual(kills,[['container','kill','--signal=TERM','1'*64]])
        self.assertNotIn('clean_shutdown',self.authority.events)
    def test_nonzero_exit_retains_worker(self):
        self.launcher.up();self.transport.exit=2
        with self.assertRaisesRegex(ValueError,'unknown'):self.launcher.stop()
        self.assertTrue(self.transport.containers['2'*64]['running'])
    def test_readiness_must_bind_registration(self):
        self.launcher.up(); self.transport.marker['registration_id']='b'*64
        with self.assertRaisesRegex(ValueError,'binding'):self.launcher.ready(self.launcher.retained(),'a'*64)
    def test_readiness_timeout_retains_all(self):
        self.launcher.up(); self.transport.marker={'phase':'waiting'};self.launcher.end=self.now+.1
        with self.assertRaisesRegex(ValueError,'timeout'):self.launcher.ready(self.launcher.retained(),'a'*64)
        self.assertTrue(all(c['running'] for c in self.transport.containers.values()))
    def test_command_matches_closed_observer_policy(self):
        f=self.fixture
        for role in ('control','worker'):
            args=p.command(role,f.config,f.policy,f.scope)
            self.assertEqual(args[:3],['container','create','--pull=never'])
            self.assertIn('--read-only',args);self.assertIn('--cap-drop=ALL',args)
            self.assertIn(f.config['images'][role],args)
            self.assertFalse(any('docker.sock' in a or 'p1-host-authority' in a for a in args))
            self.assertNotIn('NEVER-RETURN-THIS-CREDENTIAL',str(args))
            self.assertEqual(sum(a.startswith('--cap-add=') for a in args),3 if role=='worker' else 0)
            sources=[args[i+1] for i,a in enumerate(args) if a=='--mount']
            self.assertEqual(len(sources),6 if role=='control' else 5)
            self.assertEqual(any('control.key' in a for a in sources),role=='control')
    def test_invalid_policy_refuses_before_transport(self):
        policy=copy.deepcopy(self.fixture.policy);policy['cpu']['worker']=0
        with self.assertRaises(ValueError):p.command('worker',self.fixture.config,policy,self.fixture.scope)
        self.assertEqual(self.transport.calls,[])
    def test_cli_errors_are_redacted(self):
        with patch.object(Path,'read_bytes',side_effect=ValueError('TOP-SECRET')):
            with self.assertRaises(SystemExit) as error:p.main(['up','--configuration','missing','--scope','missing'])
        self.assertNotIn('TOP-SECRET',str(error.exception))


class CLIResultPublicationTests(unittest.TestCase):
    """Real CLI/public input parsing; held authority and operations are inert."""
    def check_publication(self, command, fail_exit):
        import contextlib
        import io
        from types import SimpleNamespace
        f=host_fixtures.InstalledHost();f.setUp();self.addCleanup(f.doCleanups)
        scope=f.root/'scope.json';scope.write_text(json.dumps(f.scope))
        policy=f.root/'policy.json';policy.write_text(json.dumps(f.policy))
        argv=[command,'--configuration',str(f.root/'install.json'),'--scope',str(scope)]
        if command=='init':argv+=['--host-policy',str(policy)]
        if command=='complete':argv+=['--initialization-digest','b'*64]
        events=[]
        class Output(io.StringIO):
            def write(self, value):
                if value:events.append('output')
                return super().write(value)
        output=Output()
        # Retained state models an obligation that must survive exit refusal.
        retained=f.root/'retained-obligation';retained.write_bytes(b'unchanged')
        @contextlib.contextmanager
        def held(config, scope):
            self.assertEqual(config,f.config);self.assertEqual(scope,f.scope)
            events.append('enter')
            yield SimpleNamespace(policy=f.policy,initialize_authority=lambda:None)
            events.append('exit-check')
            if fail_exit:raise ValueError('installed_host_policy_changed PRIVATE-ERROR')
            events.append('exited')
        def operation(*args, **kwargs):
            self.assertEqual(events,['enter'])
            events.append('computed')
            return dict(phase='stopped' if command=='stop' else 'ready',registration_id='a'*64)
        # All transport is forbidden even if a CLI branch accidentally reaches it.
        with patch.object(p.sys,'version_info',(3,11)), \
             patch.object(h,'installed_launch_authority',side_effect=held), \
             patch.object(h,'create_installed_authority') as create, \
             patch('service.persistent_init.initialize_split',side_effect=operation) as initialize, \
             patch.object(p.Launcher,command if command in ('up','stop','status') else 'up',side_effect=operation) as launch, \
             patch.object(h.Docker,'_read',side_effect=AssertionError('transport forbidden')), \
             contextlib.redirect_stdout(output):
            original=os.umask(0o077)
            try:
                if fail_exit:
                    with self.assertRaises(SystemExit) as error:p.main(argv)
                    self.assertEqual(str(error.exception),'speech-box refused; retained state requires review')
                else:self.assertIsNone(p.main(argv))
            finally:os.umask(original)
        self.assertEqual(retained.read_bytes(),b'unchanged')
        self.assertEqual(initialize.call_count, int(command in ('init','complete')))
        self.assertEqual(launch.call_count, int(command in ('up','stop','status')))
        self.assertEqual(create.call_count,int(command=='init'))
        if fail_exit:
            self.assertEqual(output.getvalue(),'')
            self.assertEqual(events,['enter','computed','exit-check'])
        else:
            value=json.loads(output.getvalue())
            self.assertEqual(len(output.getvalue().splitlines()),1)
            self.assertEqual(value['phase'],'initialized' if command in ('init','complete') else
                             'stopped' if command=='stop' else 'ready')
            self.assertEqual(events[:4],['enter','computed','exit-check','exited'])
            self.assertTrue(events[4:]);self.assertEqual(set(events[4:]),{'output'})

    def test_context_exit_failure_emits_no_result(self):
        for command in ('status','up','stop','init','complete'):
            with self.subTest(command=command):self.check_publication(command,True)

    def test_success_is_published_only_after_context_exit(self):
        for command in ('status','up','stop','init','complete'):
            with self.subTest(command=command):self.check_publication(command,False)


class ArchiveTests(unittest.TestCase):
    def test_deterministic_extracted_help_without_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'one.zip';second=Path(tmp)/'two.zip'
            self.assertEqual(p.archive(ROOT,first),p.archive(ROOT,second))
            with zipfile.ZipFile(first) as z:
                self.assertFalse(any('__pycache__' in n or '/tests/' in n or '/.git/' in n for n in z.namelist()))
                for name in json.loads((ROOT/'headless-dist-manifest.json').read_text())['files']:
                    self.assertEqual(z.read('speech-box/'+name),(ROOT/name).read_bytes(),name)
                z.extractall(Path(tmp)/'extracted')
            package=Path(tmp)/'extracted/speech-box'
            result=subprocess.run([sys.executable,'-I','-B',str(package/'speech-box'),'--help'],cwd=tmp,
                env={'PATH':os.defpath},capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('init,complete,up,stop,status,archive',result.stdout)
            self.assertTrue((package/'contracts/v1/schema.json').is_file())
            self.assertTrue((package/'requirements/kokoro.lock').is_file())
            code="import sys,pathlib;sys.path.insert(0,"+repr(str(package))+");import service.host_install,service.entrypoint,worker.runtime_supervisor,contracts.validate,lib.tts_backends.assets;assert all(str(pathlib.Path(m.__file__)).startswith("+repr(str(package))+") for m in (service.host_install,service.entrypoint,worker.runtime_supervisor,contracts.validate,lib.tts_backends.assets));print('packaged imports only')"
            imported=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp,
                env={'PATH':os.defpath},capture_output=True,text=True,timeout=5)
            self.assertEqual(imported.returncode,0,imported.stderr)
    def test_pocket_archive_bytes_and_isolated_dependency_closure(self):
        modules = ('clients.cli', 'clients.http', 'clients.mcp', 'service.execution_context',
                   'worker.managed_reference', 'service.pocket_install', 'worker.runtime_conditioning', 'worker.engines.pocket',
                   'worker.engines.pocket_execution', 'worker.engines.pocket_prepare')
        manifest = json.loads((ROOT/'headless-dist-manifest.json').read_text())
        files = set(manifest['files'])
        # Inspect lazy imports too, without executing generation or housekeeping.
        pending = list(modules) + ['worker.runtime_supervisor', 'worker.runtime_child']
        visited = set()
        while pending:
            module = pending.pop()
            if module in visited:
                continue
            visited.add(module)
            name = module.replace('.', '/') + '.py'
            if not (ROOT/name).is_file():
                name = module.replace('.', '/') + '/__init__.py'
            self.assertIn(name, files, module)
            package = module.split('.') if name.endswith('/__init__.py') else module.split('.')[:-1]
            for node in ast.walk(ast.parse((ROOT/name).read_bytes())):
                dependencies = []
                if isinstance(node, ast.Import):
                    dependencies = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = (package[:len(package)-node.level+1] if node.level else [])
                    base += (node.module or '').split('.') if node.module else []
                    prefix = '.'.join(base)
                    dependencies = [prefix] + [prefix+'.'+alias.name for alias in node.names]
                for dependency in dependencies:
                    parts = dependency.split('.')
                    for end in range(1, len(parts)+1):
                        local = '.'.join(parts[:end])
                        path = ROOT/local.replace('.', '/')
                        if path.with_suffix('.py').is_file() or (path/'__init__.py').is_file():
                            pending.append(local)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'pocket.zip'
            p.archive(ROOT, archive)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),
                    {'speech-box/'+name for name in files} | {'speech-box/speech-box'})
                self.assertEqual(len(z.namelist()), len(files)+1)
                for module in modules:
                    name = module.replace('.', '/')+'.py'
                    self.assertEqual(z.read('speech-box/'+name), (ROOT/name).read_bytes())
                z.extractall(Path(tmp)/'extracted')
            package = Path(tmp)/'extracted/speech-box'
            for module in modules:
                name = module.replace('.', '/')+'.py'
                self.assertEqual((package/name).read_bytes(), (ROOT/name).read_bytes())
            # Resolve the literal -m target from the archived supervisor, but never
            # execute __main__, remove_empty, engine generation, or a backend.
            code = """
import ast, importlib, importlib.abc, importlib.util, pathlib, sys
package = pathlib.Path(sys.argv[1])
checkout = pathlib.Path(sys.argv[2])
assert all(not pathlib.Path(p).is_relative_to(checkout) for p in sys.path if p)
sys.path.insert(0, str(package))
class ForbidHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'numpy', 'soundfile',
                'pocket_tts', 'kokoro', 'transformers', 'huggingface_hub', 'onnxruntime'}:
            raise AssertionError('heavy import forbidden: '+fullname)
sys.meta_path.insert(0, ForbidHeavy())
supervisor = importlib.import_module('worker.runtime_supervisor')
tree = ast.parse(pathlib.Path(supervisor.__file__).read_bytes())
assert any(isinstance(n, ast.List) and len(n.elts) >= 3
    and isinstance(n.elts[1], ast.Constant) and n.elts[1].value == '-m'
    and isinstance(n.elts[2], ast.Constant) and n.elts[2].value == 'worker.runtime_conditioning'
    for n in ast.walk(tree))
for name in ('clients.cli', 'clients.http', 'clients.mcp', 'service.execution_context',
             'worker.managed_reference', 'service.pocket_install', 'worker.runtime_conditioning', 'worker.engines.pocket',
             'worker.engines.pocket_execution', 'worker.engines.pocket_prepare', 'worker.runtime_child'):
    spec = importlib.util.find_spec(name)
    assert pathlib.Path(spec.origin).is_relative_to(package), name
    importlib.import_module(name)
for name, module in tuple(sys.modules.items()):
    if name.split('.')[0] in {'clients', 'worker', 'service', 'contracts', 'lib'}:
        assert pathlib.Path(module.__file__).is_relative_to(package), name
from clients import mcp
schema = mcp.input_schema('voice_prepare')
assert schema['properties']['audition_text']['maxLength'] == 10000
assert schema['additionalProperties'] is False
assert 'idempotency_key' in schema['required']
print('isolated Pocket and authenticated client archive closure')
"""
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                str(package), str(ROOT)], cwd=tmp, env={'PATH': os.defpath},
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'isolated Pocket and authenticated client archive closure')

    def test_manifest_exact_files_and_no_legacy_metadata_change(self):
        manifest=json.loads((ROOT/'headless-dist-manifest.json').read_text())
        self.assertEqual(len(manifest['files']),len(set(manifest['files'])))
        self.assertTrue(all((ROOT/n).is_file() for n in manifest['files']))
        import tomllib
        meta=tomllib.loads((ROOT/'pyproject.toml').read_text())
        self.assertEqual(meta['tool']['poetry']['name'],'voice-tools')
        self.assertFalse(meta['tool']['poetry']['package-mode'])
    def test_release_context_and_cpu_base(self):
        recipe=(ROOT/'service/Dockerfile.release').read_text()
        cpu=(ROOT/'Dockerfile.cpu').read_text().splitlines()[0].removeprefix('FROM ')
        self.assertEqual(recipe.count('FROM '+cpu+' AS '),4)
        self.assertNotIn('3.14',recipe);self.assertNotIn('service.api:app',recipe)
        self.assertIn('-r kokoro.lock -r requirements.lock',recipe)
        self.assertNotIn('tini',recipe)  # supervisor and control must remain PID1
        ignore=(ROOT/'service/Dockerfile.release.dockerignore').read_text().splitlines()
        self.assertEqual(ignore[0],'**')
        self.assertFalse(any(n.startswith('!tests') or n.startswith('!studio') for n in ignore))
        self.assertNotIn('build:',(ROOT/'service/compose.release.yml').read_text())
    def test_pocket_worker_lock_is_exact_existing_hash_metadata_union(self):
        import re
        def entries(path):
            raw=path.read_text()
            starts=list(re.finditer(r'^([a-zA-Z0-9_-]+)==',raw,re.M))
            return {m[1]:raw[m.start():(starts[i+1].start() if i+1<len(starts) else len(raw))]
                    for i,m in enumerate(starts)}
        pocket=entries(ROOT/'requirements/pocket.lock')
        service=entries(ROOT/'service/requirements.lock')
        actual=entries(ROOT/'requirements/pocket-installed.lock')
        self.assertEqual(actual,{**service,**pocket})
        for entry in actual.values():self.assertIn('--hash=sha256:',entry)
        self.assertTrue(actual['fastapi'].startswith('fastapi==0.143.1 '))
        self.assertNotIn('kokoro-onnx',actual)
        from service.pocket_install import runtime_versions
        for name,version in runtime_versions().items():
            self.assertTrue(actual[name].startswith(name+'=='+version+' '))
        recipe=(ROOT/'service/Dockerfile.release').read_text().split(' AS worker-pocket',1)[1]
        self.assertIn('--require-hashes -r pocket-installed.lock',recipe)
        self.assertNotIn('kokoro.lock',recipe)

    def test_lock_versions_preserved_and_hashes_present(self):
        lock=(ROOT/'service/requirements.lock').read_text()
        self.assertIn('--python-version 3.11',lock)
        self.assertIn('fastapi==0.143.2',lock)
        self.assertIn('starlette==1.8.0',lock)
        self.assertIn('--hash=sha256:',lock)


class ProtectedLaunchRecordsTests(unittest.TestCase):
    def setUp(self):
        from tests.service import test_installed_registration
        f=self.fixture=test_installed_registration.Registration(); f.setUp(); self.addCleanup(f.doCleanups)
    def documents(self):
        f=self.fixture
        ids={c['role']:c['id'] for c in f.owners}
        values={'creation-intent.json':dict(version=1,configuration_digest=h.digest(f.config)),
                'containers.json':dict(version=1,configuration_digest=h.digest(f.config),containers=ids),
                **{'created-'+role+'.json':dict(id=identity) for role,identity in ids.items()}}
        for name,value in values.items():h._host_write(f.authority/name,value)
        return ids
    def test_real_registration_accepts_only_complete_bound_launcher_records(self):
        self.documents();rid=self.fixture.run_registration()
        self.assertEqual(self.fixture.rows()[0][0],rid)
    def test_partial_creation_blocks_before_registration_child(self):
        f=self.fixture
        h._host_write(f.authority/'creation-intent.json',dict(version=1,configuration_digest=h.digest(f.config)))
        with self.assertRaisesRegex(ValueError,'partial_creation'):f.run_registration()
        self.assertEqual(f.children,[]);self.assertEqual(f.rows(),[])
    def test_changed_created_identity_blocks_before_registration_child(self):
        self.documents();f=self.fixture
        path=f.authority/'created-control.json';path.write_text(h.encode(dict(id='f'*64)))
        with self.assertRaisesRegex(ValueError,'creation_identity'):f.run_registration()
        self.assertEqual(f.children,[])
    def test_unknown_authority_document_still_refuses(self):
        self.documents();f=self.fixture
        h._host_write(f.authority/'unknown.json',{})
        with self.assertRaisesRegex(ValueError,'unknown_authority'):f.run_registration()
        self.assertEqual(f.children,[])


class InstalledCommandTests(unittest.TestCase):
    def test_factory_pid1_startup_publication_and_clean_exit_no_listener(self):
        import asyncio
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            self.addCleanup(os.close,fd)
            life=dict(notification=SimpleNamespace(fd=fd,read=lambda:dict(registration_id='a'*64)),closed=False,error=None)
            app=SimpleNamespace(state=SimpleNamespace(installed_lifecycle=life))
            class Server:
                def __init__(self,config):self.started=False;self.should_exit=False
                async def serve(self):
                    self.started=True
                    while not (directory/e.STARTUP_NAME).exists():await asyncio.sleep(.001)
                    life['closed']=True
            with patch.object(e,'_control_identity'),patch.object(e,'create_installed_app',return_value=app) as factory, \
                 patch.object(e,'local_incarnation',return_value={'injected':True}), \
                 patch('uvicorn.Config'),patch('uvicorn.Server',Server):
                original=os.umask(0o077)
                try:e.installed_main()
                finally:os.umask(original)
            factory.assert_called_once_with(configuration_file=e.ip.CONFIG_DESTINATION,credential_file=e.ip.CREDENTIAL_DESTINATION)
            self.assertEqual(json.loads((directory/e.STARTUP_NAME).read_text()),
                dict(phase='ready',registration_id='a'*64,incarnation={'injected':True}))
    def test_factory_failure_exits_nonzero_without_publishing(self):
        from types import SimpleNamespace
        app=SimpleNamespace(state=SimpleNamespace())
        class Server:
            def __init__(self,config):self.started=False;self.should_exit=False
            async def serve(self):return
        with patch.object(e,'_control_identity'),patch.object(e,'create_installed_app',return_value=app), \
             patch('uvicorn.Config'),patch('uvicorn.Server',Server):
            original=os.umask(0o077)
            try:
                with self.assertRaises(SystemExit) as result:e.installed_main()
            finally:os.umask(original)
        self.assertEqual(result.exception.code,2)


class SignalPolicyTests(unittest.TestCase):
    def test_repeated_signal_only_requests_normal_shutdown(self):
        import signal
        handlers={};calls=[]
        def install(sig,handler):
            previous=handlers.get(sig,signal.SIG_DFL);handlers[sig]=handler;calls.append((sig,handler));return previous
        with patch('signal.signal',side_effect=install),patch('signal.raise_signal') as rethrow:
            server=e._installed_server(__import__('uvicorn').Config('unused'))
            with server.capture_signals():
                handlers[signal.SIGTERM](signal.SIGTERM,None)
                handlers[signal.SIGINT](signal.SIGINT,None)
                self.assertTrue(server.should_exit);self.assertFalse(server.force_exit)
            rethrow.assert_not_called()
            self.assertEqual(handlers[signal.SIGTERM],signal.SIG_DFL)


class GeneratedTopologyTests(unittest.TestCase):
    def test_generated_create_arguments_satisfy_real_closed_topology_validator(self):
        from decimal import Decimal
        f=host_fixtures.InstalledHost();f.setUp();self.addCleanup(f.doCleanups)
        consumers=[]
        for template,owner in zip(f.containers,f.owners):
            role=owner['role'];args=p.command(role,f.config,f.policy,f.scope)
            options={};i=2
            flags=('--read-only',)
            while i<len(args) and args[i].startswith('--'):
                option=args[i]
                if '=' in option:
                    key,value=option.split('=',1);i+=1
                elif option in flags:
                    key,value=option,True;i+=1
                else:
                    key,value=option,args[i+1];i+=2
                options.setdefault(key,[]).append(value)
            self.assertEqual(args[i],f.config['images'][role])
            c=copy.deepcopy(template)
            c.update(user=options['--user'][0],restart=options['--restart'][0],readonly=options['--read-only'][0],
                security=options['--security-opt'],capdrop=options['--cap-drop'],capadd=options.get('--cap-add',[]),
                ipcns=options['--ipc'][0],cgroupns=options['--cgroupns'][0],network=options['--network'][0],
                memory=int(options['--memory'][0]),swap=int(options['--memory-swap'][0]),pids=int(options['--pids-limit'][0]),
                nanocpus=int(Decimal(options['--cpus'][0])*10**9))
            destination,settings=options['--tmpfs'][0].split(':',1);c['tmpfs']={destination:settings}
            c['ports']={}
            if '--publish' in options:
                host,port,container=options['--publish'][0].rsplit(':',2)
                c['ports']={container+'/tcp':[dict(HostIp=host,HostPort=port)]}
            c['mounts']=[]
            for mount in options['--mount']:
                parts=dict(item.split('=',1) for item in mount.split(',') if '=' in item)
                name=parts['source'] if parts['type']=='volume' else ''
                source=f.policy['roots'][name]['source'] if name else parts['source']
                c['mounts'].append(dict(type=parts['type'],name=name,source=source,
                    destination=parts['destination'],rw='readonly' not in mount.split(',')))
            consumers.append(c)
        topology=dict(scope=f.scope,volumes=f.volumes,consumers=consumers,policy=h.ip.HOST_POLICY,
            configuration_digest=h.digest(f.config),host_policy_digest=h.digest(f.policy))
        self.assertIs(h.validate_installed_topology(topology,f.scope,f.owners,f.config,f.policy),topology)


class ListenerProbeTests(unittest.TestCase):
    def test_probe_binds_live_activation_and_never_returns_token_or_body(self):
        from types import SimpleNamespace
        token='PRIVATE-TOKEN-NEVER-OUTPUT'
        body=dict(synthetic=False,ready_for_speech=True,configuration_digest='a'*64,
            status=dict(persistent_activation=True),private_rows=['never-return'])
        requests=[]
        class Connection:
            def __init__(self,*args,**kw):self.closed=False
            def request(self,*args,**kwargs):requests.append((args,kwargs))
            def getresponse(self):return SimpleNamespace(status=200,read=lambda limit:json.dumps(body).encode())
            def close(self):self.closed=True
        with patch('service.auth.private_input',return_value=json.dumps(dict(credentials=[dict(token=token)])).encode()), \
             patch('http.client.HTTPConnection',Connection):
            self.assertIs(e._listener_ready('a'*64),True)
            body['configuration_digest']='b'*64
            self.assertIs(e._listener_ready('a'*64),False)
            body['configuration_digest']='a'*64;body['status']['persistent_activation']=False
            self.assertIs(e._listener_ready('a'*64),False)
        self.assertEqual(requests[0][0],('GET','/skeleton'))
        self.assertEqual(requests[0][1]['headers']['Authorization'],'Bearer '+token)


class ReadinessReaderTests(unittest.TestCase):
    def test_exec_reader_checks_pid1_gate_and_listener_without_claiming_to_be_pid1(self):
        import contextlib
        import io
        f=host_fixtures.InstalledHost();f.setUp();self.addCleanup(f.doCleanups)
        directory=f.root/'activation';directory.mkdir(mode=0o700);f.owner(directory,10001,10001)
        result=dict(phase='ready',registration_id='a'*64,incarnation={'injected-pid1':True})
        path=directory/e.STARTUP_NAME;path.write_text(h.encode(result));path.chmod(0o600);f.owner(path,10001,10001)
        output=io.StringIO()
        with patch.object(e,'GATE_DIRECTORY',directory),patch.object(e,'_control_identity'), \
             patch.object(e,'_read_pid1_incarnation',return_value={'injected-pid1':True}), \
             patch.object(e.ip,'read_private',return_value=b'gate'),patch.object(e,'decode_projection',return_value=dict(registration_id='a'*64,configuration_digest='b'*64)), \
             patch.object(e,'_listener_ready',return_value=True),contextlib.redirect_stdout(output):
            e.launch_status()
        self.assertEqual(json.loads(output.getvalue()),dict(phase='ready',registration_id='a'*64))


class ProvisioningTests(unittest.TestCase):
    def test_create_only_authority_has_exact_files_and_retains_second_attempt(self):
        import stat
        from types import SimpleNamespace
        f=host_fixtures.InstalledHost();f.setUp();self.addCleanup(f.doCleanups)
        real_directory=h._root_authority_directory
        directory=f.authority_fixture()
        # Remove only this test fixture's prepared authority leaf to exercise
        # creation. Root identities/trusted ancestry are injected, not changed.
        (directory/'launch.lock').unlink();(directory/'host-policy.json').unlink();directory.rmdir()
        real_fstat=os.fstat
        def root_observation(fd):
            info=real_fstat(fd)
            fields={k:getattr(info,k) for k in dir(info) if k.startswith('st_')}
            fields.update(st_uid=0,st_gid=0)
            if stat.S_ISDIR(info.st_mode):fields['st_mode']=info.st_mode & ~0o022
            return SimpleNamespace(**fields)
        observed=[]
        observer=SimpleNamespace(initialize_authority=lambda:observed.append('checked-empty-inputs'))
        with patch.object(h,'InstalledObserver',return_value=observer),patch.object(os,'fstat',side_effect=root_observation), \
             patch.object(h,'_root_authority_directory',side_effect=real_directory):
            h.create_installed_authority(f.config,f.scope,f.policy)
            before={n.name:n.read_bytes() for n in directory.iterdir()}
            with self.assertRaises(FileExistsError):h.create_installed_authority(f.config,f.scope,f.policy)
            self.assertEqual(before,{n.name:n.read_bytes() for n in directory.iterdir()})
        self.assertEqual(set(before),{'host-policy.json','launch.lock'})
        self.assertEqual(before['launch.lock'],b'')
        self.assertEqual(json.loads(before['host-policy.json']),f.policy)
        self.assertEqual(observed,['checked-empty-inputs','checked-empty-inputs'])
        self.assertNotIn(b'NEVER-RETURN-THIS-CREDENTIAL',b''.join(before.values()))


if __name__=='__main__':unittest.main()
