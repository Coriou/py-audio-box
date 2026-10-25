"""Nonqualifying inert package manifests; no interpreter, socket or ML launch."""
import copy
from email.parser import Parser
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from service import native_package as n
from worker.transport import Refused, canonical


class NativePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.root.chmod(0o700)
        self.python = self.file('managed-python', b'INERT NOT A PYTHON BUILD', 0o700)
        self.source = self.root/'source.zip'; digest = n.archive(self.source)
        self.identity = dict(executable=str(self.python), version='INERT', build=['INERT','INERT'],
                             implementation='CPython', machine='arm64', system='Darwin')
        self.versions = dict(packaging='1', torch='1', transformers='1', **{'qwen-tts':'1'})
        artifacts=[]
        for name in self.versions:
            wheel=self.make_wheel(name)
            artifacts.append(dict(path=str(wheel),sha256=n.file_hash(str(wheel)),distributions=[name]))
        self.wheel=Path(artifacts[0]['path'])
        self.manifest = dict(version=1, variant=n.VARIANT, qualification='unverified',
            python=dict(self.identity, sha256=n.file_hash(str(self.python))), distributions=self.versions,
            artifacts=artifacts,
            source_archive=dict(path=str(self.source),sha256=digest),
            models=[dict(bundle_id='bundle',path=str(self.root/'model'),manifest_sha256='a'*64,
                checkpoint='INERT',revision='INERT',license='INERT-NONQUALIFYING',access='INERT')])

    def make_wheel(self,name,tag='py3-none-any'):
        # Metadata-only inert ZIP, deliberately no importable package or binary.
        stem=name.replace('-','_')+'-1'
        path=self.root/(stem+'-'+tag+'.whl')
        with zipfile.ZipFile(path,'w') as wheel:
            wheel.writestr(stem+'.dist-info/METADATA','Metadata-Version: 2.1\nName: '+name+'\nVersion: 1\n')
            wheel.writestr(stem+'.dist-info/WHEEL','Wheel-Version: 1.0\nTag: '+tag+'\n')
        path.chmod(0o400);return path

    def file(self, name, raw, mode=0o600):
        path=self.root/name; path.write_bytes(raw); path.chmod(mode); return path

    def validate(self, value=None):
        with patch.object(n,'python_identity',return_value=self.identity), \
             patch.object(n,'dependency_closure',return_value={name:frozenset() for name in self.versions}):
            return n.runtime_manifest(value or self.manifest)

    def test_manifest_exact_hash_and_nonqualifying_status(self):
        self.assertEqual(self.validate(), hashlib.sha256(canonical(self.manifest)).hexdigest())
        with patch.object(n,'preflight'), contextlib.redirect_stdout(io.StringIO()) as out:
            n.main(['check','--config','/explicit/config','--runtime-manifest','/explicit/runtime'])
        result=json.loads(out.getvalue())
        self.assertEqual((result['abi'],result['speech'],result['qualification']),('unrun','unavailable','unqualified'))

    def test_wrong_arch_system_build_or_executable_refuses(self):
        for key,value in [('machine','x86_64'),('system','Linux'),('version','other'),('build',['other']),('executable','/other')]:
            manifest=copy.deepcopy(self.manifest);manifest['python'][key]=value
            with self.subTest(key=key), self.assertRaises(Refused):self.validate(manifest)

    def test_python_and_artifact_hash_mismatch(self):
        for key in ('python','source_archive'):
            manifest=copy.deepcopy(self.manifest);manifest[key]['sha256']='f'*64
            with self.subTest(key=key), self.assertRaises(Refused):self.validate(manifest)
        self.wheel.chmod(0o600);self.wheel.write_bytes(b'changed')
        with self.assertRaises(Refused):self.validate()

    def test_missing_artifact_coverage_and_fake_qualification_refuse(self):
        for change in ('missing','duplicate','qualified','extra'):
            manifest=copy.deepcopy(self.manifest)
            if change=='missing':manifest['artifacts'][0]['distributions'].pop()
            if change=='duplicate':manifest['artifacts']*=2
            if change=='qualified':manifest['qualification']='qualified'
            if change=='extra':manifest['unknown']=True
            with self.subTest(change=change), self.assertRaises(Refused):self.validate(manifest)

    def test_private_owner_mode_symlink_hardlink_and_parent_refusal(self):
        path=self.file('credential',b'PRIVATE')
        self.assertEqual(n.read(path),b'PRIVATE')
        path.chmod(0o644)
        with self.assertRaises(Refused):n.read(path)
        path.chmod(0o600)
        with patch.object(n.os,'getuid',return_value=os.getuid()+1), self.assertRaises(Refused):n.read(path)
        os.link(path,self.root/'alias')
        with self.assertRaises(Refused):n.read(path)
        (self.root/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(OSError):n.read(self.root/'link/credential')
        with self.assertRaises(OSError):n.directory(str(self.file('not-directory',b'')))

    def test_canonical_document_duplicate_and_nonfinite_refuse(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{ "x": 1 }'):
            path=self.file('document',raw)
            with self.assertRaises(Refused):n.document(path)
        path=self.file('document',canonical(dict(x=1)))
        self.assertEqual(n.document(path),dict(x=1))

    def test_archive_exact_members_secrets_excluded_extracted_help(self):
        self.file('operator-key',b'SECRET-NOT-IN-ARCHIVE')
        with zipfile.ZipFile(self.source) as source:
            self.assertEqual(set(source.namelist()),{'native-worker/'+x for x in n.members()}|{'native-worker/native-worker.py'})
            self.assertFalse(any(x.endswith('.lock') or 'operator-key' in x for x in source.namelist()))
            self.assertFalse(any(b'SECRET-NOT-IN-ARCHIVE' in source.read(x) for x in source.namelist()))
            extracted=self.root/'extracted';source.extractall(extracted)
        package=extracted/'native-worker'
        spec=importlib.util.spec_from_file_location('extracted_native_package',package/'service/native_package.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.ROOT,package)
        module.verify_archive(self.source)
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as result:
            module.main(['--help'])
        self.assertEqual(result.exception.code,0);self.assertIn('native-qwen-v1',out.getvalue())
        original_path=sys.path[:]
        try:
            with patch.dict(sys.modules,{'service.native_package':module}), \
                 patch.object(sys,'argv',[str(package/'native-worker.py'),'--help']), \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as result:
                runpy.run_path(str(package/'native-worker.py'),run_name='__main__')
            self.assertEqual(result.exception.code,0)
        finally:sys.path[:]=original_path
        (package/'worker/native_child.py').write_text('changed')
        with self.assertRaises(Refused):module.verify_archive(self.source)

    def test_extra_archive_member_refuses(self):
        with zipfile.ZipFile(self.source,'a') as source:source.writestr('native-worker/secret',b'bad')
        with self.assertRaises(Refused):n.verify_archive(self.source)

    def test_dependency_missing_wrong_version_cpu_or_transitive_refuses(self):
        with self.assertRaises(Refused):n.dependency_closure({'torch':'1'})
        with self.assertRaises(Refused):n.dependency_closure(dict(self.versions,torch='1+cpu'))
        with patch.object(n,'distributions',return_value=dict(self.versions,torch='2')):
            with self.assertRaises(Refused):n.dependency_closure(self.versions)
        from types import SimpleNamespace
        with patch.object(n,'distributions',return_value=self.versions), \
             patch.object(n.importlib.metadata,'distributions',return_value=[SimpleNamespace(metadata=Parser().parsestr('Name: torch\nVersion: 1\n'),requires=['missing>=1'])]):
            with self.assertRaises(Refused):n.dependency_closure(self.versions)
        with patch.object(n,'distributions',return_value=self.versions), \
             patch.object(n.importlib.metadata,'distributions',return_value=[SimpleNamespace(metadata=Parser().parsestr('Name: torch\nVersion: 1\n'),requires=['torch==1'])]):
            n.dependency_closure(self.versions)

    def test_wheel_platform_name_version_and_metadata_refuse(self):
        from packaging.tags import Tag
        with patch('packaging.tags.sys_tags',return_value=iter([Tag('cp311','cp311','macosx_14_0_arm64')])):
            path=self.make_wheel('torch','cp311-cp311-manylinux_2_17_x86_64')
            with self.assertRaisesRegex(Refused,'wheel_platform'):n.wheel_identity(str(path),['torch'],self.versions)
        with self.assertRaises(Refused):n.wheel_identity(str(self.wheel),['torch'],self.versions)
        self.wheel.chmod(0o600)
        with zipfile.ZipFile(self.wheel,'w') as wheel:wheel.writestr('bad','NO METADATA')
        with self.assertRaises(Refused):n.wheel_identity(str(self.wheel),['packaging'],self.versions)

    def rewrite_metadata(self, headers, *, base="Name: packaging\nVersion: 1\n", wheel_headers="Tag: py3-none-any\n"):
        self.wheel.chmod(0o600)
        with zipfile.ZipFile(self.wheel, 'w') as wheel:
            wheel.writestr('packaging-1.dist-info/METADATA',
                base + headers)
            wheel.writestr('packaging-1.dist-info/WHEEL', wheel_headers)
        self.manifest['artifacts'][0]['sha256'] = n.file_hash(str(self.wheel))

    def test_selected_wheel_active_dependencies_refuse(self):
        for raw in ('missing>=99', 'torch>=99', 'torch @ https://example.invalid/a.whl',
                    'torch[extra]>=1', 'torch; invalid_marker == "x"'):
            with self.subTest(raw=raw):
                self.rewrite_metadata('Requires-Dist: ' + raw + '\n')
                with self.assertRaises(Refused): self.validate()

    def test_selected_wheel_and_installed_graph_agreement(self):
        from types import SimpleNamespace
        installed = [SimpleNamespace(metadata=Parser().parsestr('Name: '+name+'\nVersion: 1\n'), version='1', requires=[])
                     for name in self.versions]
        self.rewrite_metadata('Requires-Dist: torch>=1\n')
        with patch.object(n, 'python_identity', return_value=self.identity), \
             patch.object(n.importlib.metadata, 'distributions', return_value=installed):
            with self.assertRaisesRegex(Refused, 'metadata_disagreement'):
                n.runtime_manifest(self.manifest)
            installed[0].requires = ['Torch >= 1; python_version >= "0"']
            self.assertEqual(n.runtime_manifest(self.manifest), hashlib.sha256(canonical(self.manifest)).hexdigest())
            installed[0].requires = ['torch<2']
            with self.assertRaisesRegex(Refused, 'metadata_disagreement'):
                n.runtime_manifest(self.manifest)

    def test_inactive_valid_markers_and_requires_python(self):
        inactive = ['missing>=99; python_version < "0"',
                    'torch[extra] @ https://example.invalid/a ; extra == "optional"']
        self.rewrite_metadata(''.join('Requires-Dist: '+raw+'\n' for raw in inactive))
        self.validate()
        self.assertEqual(n.active_edges(inactive, self.versions), frozenset())
        for raw in ('torch>=99', 'torch[extra]', 'torch @ https://example.invalid/a',
                    'torch; invalid_marker == "x"'):
            with self.assertRaises(Refused): n.active_edges([raw], self.versions)
        self.rewrite_metadata('Requires-Python: >=99\n')
        with self.assertRaises(Refused): self.validate()

    def test_wheel_singleton_duplicates_before_dependency_acceptance(self):
        for field, original, other in (('Name', 'packaging', 'contradictory'),
                                       ('Version', '1', 'NaN'),
                                       ('Requires-Python', '>=0', '>=99')):
            for spelling in (field, field.swapcase()):
                for value in (original, other):
                    for folded in ('', '\n '):
                        with self.subTest(field=spelling, value=value, folded=bool(folded)):
                            base = 'Name: packaging\nVersion: 1\n'
                            if field == 'Requires-Python': base += 'Requires-Python: >=0\n'
                            self.rewrite_metadata(spelling+': '+folded+value+'\n', base=base)
                            with patch.object(n, 'active_edges', side_effect=AssertionError('edges entered')), \
                                 self.assertRaisesRegex(Refused, 'metadata_singleton'):
                                self.validate()

    def test_wheel_missing_identity_and_single_python_constraint(self):
        for base in ('Name: packaging\n', 'Version: 1\n'):
            self.rewrite_metadata('', base=base)
            with self.assertRaisesRegex(Refused, 'metadata_singleton'): self.validate()
        self.rewrite_metadata('Requires-Python: >=0\n')
        self.validate()
        self.rewrite_metadata('Requires-Python: >=99\n')
        with self.assertRaisesRegex(Refused, 'python_constraint'): self.validate()

    def test_wheel_headers_singletons_and_repeated_tags_requirements(self):
        for field, value, other in (('Wheel-Version', '1.0', '2.0'),
                                    ('Root-Is-Purelib', 'true', 'false')):
            for duplicate in (value, other):
                self.rewrite_metadata('', wheel_headers='Tag: py3-none-any\n'+field+': '+value+'\n'+
                                      field.swapcase()+':\n '+duplicate+'\n')
                with self.subTest(field=field, duplicate=duplicate), \
                     self.assertRaisesRegex(Refused, 'metadata_singleton'): self.validate()
        for field, value in (('Wheel-Version', '2.0'), ('Root-Is-Purelib', 'maybe')):
            self.rewrite_metadata('', wheel_headers='Tag: py3-none-any\n'+field+': '+value+'\n')
            with self.assertRaisesRegex(Refused, 'wheel_metadata'): self.validate()
        self.rewrite_metadata('Requires-Dist: missing>=99; python_version < "0"\n'
                              'Requires-Dist: torch[extra]; extra == "optional"\n',
                              wheel_headers='Wheel-Version: 1.0\nRoot-Is-Purelib: true\n'
                                            'Tag: py3-none-any\ntAg: py3-none-any\n')
        self.validate()
        self.rewrite_metadata('Requires-Dist: torch>=1\nRequires-Dist: transformers>=1\n')
        self.assertEqual(n.wheel_identity(str(self.wheel), ['packaging'], self.versions),
                         frozenset({('torch', '>=1'), ('transformers', '>=1')}))

    def test_installed_metadata_singletons(self):
        from types import SimpleNamespace
        installed = [SimpleNamespace(metadata=Parser().parsestr('Name: '+name+'\nVersion: 1\n'),
                                     requires=[]) for name in self.versions]
        with patch.object(n.importlib.metadata, 'distributions', return_value=installed):
            self.assertEqual(n.distributions(), self.versions)
            for field, value, other in (('Name', 'packaging', 'other'), ('Version', '1', '2'),
                                        ('Requires-Python', '>=0', '>=99')):
                for duplicate in (value, other):
                    raw = 'Name: packaging\nVersion: 1\n'
                    if field == 'Requires-Python': raw += 'Requires-Python: >=0\n'
                    installed[0].metadata = Parser().parsestr(raw+field.swapcase()+':\n '+duplicate+'\n')
                    with self.subTest(field=field, duplicate=duplicate), \
                         self.assertRaisesRegex(Refused, 'metadata_singleton'):
                        n.dependency_closure(self.versions)
            for raw in ('Name: packaging\n', 'Version: 1\n',
                        'Name: packaging\nVersion: 1\nRequires-Python: >=99\n'):
                installed[0].metadata = Parser().parsestr(raw)
                with self.assertRaises(Refused): n.dependency_closure(self.versions)

    def test_source_and_wheel_eocd_count_mismatch_before_member_reads(self):
        import struct
        # 96 tiny wheel entries reproduce a low lying count without a large archive.
        self.rewrite_metadata('')
        with zipfile.ZipFile(self.wheel, 'a') as wheel:
            for index in range(94): wheel.writestr('inert/'+str(index), b'')
        for path, source in ((self.source, True), (self.wheel, False)):
            original = path.read_bytes()
            with zipfile.ZipFile(path) as archive: actual = len(archive.infolist())
            self.assertEqual(n.zip_preflight(path, source=source), actual)
            if not source: self.assertEqual(actual, 96)
            for count in (0, 1, actual-1, actual+1):
                changed = bytearray(original); offset = changed.rfind(b'PK\x05\x06')
                struct.pack_into('<HH', changed, offset+8, count, count)
                path.chmod(0o600); path.write_bytes(changed)
                with self.subTest(source=source, count=count), \
                     patch.object(n, 'zip_read', side_effect=AssertionError('member read entered')) as reads, \
                     self.assertRaisesRegex(Refused, 'zip_bounds'):
                    if source: n.verify_archive(path)
                    else: n.wheel_identity(str(path), ['packaging'], self.versions)
                reads.assert_not_called()
            path.write_bytes(original)
            if source: n.verify_archive(path)
            else: n.wheel_identity(str(path), ['packaging'], self.versions)

    def test_launcher_declared_overlimit_never_opens_member(self):
        bad = self.root/'bad.zip'
        with zipfile.ZipFile(self.source) as source, zipfile.ZipFile(bad, 'w') as out:
            for info in source.infolist():
                out.writestr(info, source.read(info) if not info.filename.endswith('/native-worker.py')
                             else n.LAUNCHER + b'X')
        original = n.zip_read
        def spy(archive, info, limit):
            self.assertFalse(info.filename.endswith('/native-worker.py'))
            return original(archive, info, limit)
        with patch.object(n, 'zip_read', side_effect=spy), \
             self.assertRaisesRegex(Refused, 'launcher'):
            n.verify_archive(bad)

    def test_metadata_declared_overlimit_never_reads_member(self):
        self.rewrite_metadata('X: ' + 'X'*65536)
        with patch.object(n, 'zip_read', side_effect=AssertionError('read entered')) as spy, \
             self.assertRaisesRegex(Refused, 'wheel_metadata'):
            n.wheel_identity(str(self.wheel), ['packaging'], self.versions)
        spy.assert_not_called()

    def test_zip_bounds_before_constructor_and_bounded_stream(self):
        import struct
        raw = bytearray(self.source.read_bytes())
        offset = raw.rfind(b'PK\x05\x06')
        for field, value, fmt in ((10, 513, '<H'), (12, 4194305, '<I')):
            changed = bytearray(raw); struct.pack_into(fmt, changed, offset+field, value)
            bad = self.file('bounds.zip', changed)
            with patch.object(n.zipfile, 'ZipFile', side_effect=AssertionError('constructor entered')), \
                 self.assertRaisesRegex(Refused, 'zip_bounds'):
                n.verify_archive(bad)
        class Stream(io.BytesIO):
            def read(inner, size=-1):
                self.assertEqual(size, 4)
                return super().read(size)
        from types import SimpleNamespace
        with patch.object(n.zipfile.ZipFile, 'open', return_value=Stream(b'xxxx')):
            with zipfile.ZipFile(self.wheel) as wheel, self.assertRaises(Refused):
                n.zip_read(wheel, SimpleNamespace(file_size=3), 3)

    def test_archive_deterministic_bytes(self):
        other = self.root/'second.zip'
        self.assertEqual(n.archive(other), hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(other.read_bytes(), self.source.read_bytes())

    def preflight_fixture(self):
        from tests.service.test_native_runtime import NativeFixture
        from worker.engines.qwen_execution import inventory_wire
        f=NativeFixture(self); e=f.f.e
        self.manifest['models']=[dict(bundle_id=e.bundle_id,path=str(f.f.fixture.bundle),
            manifest_sha256=e.manifest_sha256,checkpoint=e.checkpoint,revision=e.revision,
            license='INERT-NONQUALIFYING',access='INERT')]
        for name in ('ca','cert','key'):self.file(name,b'INERT-NOT-A-CERTIFICATE')
        config=dict(root=str(f.f.root),deployment_id='a'*32,configuration_sha256='b'*64,
            peer_pin='c'*64,runtime_manifest_sha256='d'*64,python=str(self.python),
            child_entry=str(n.ROOT/'worker/native_child.py'),host='192.0.2.1',port=4443,
            proven_host_network=False,cafile=str(self.root/'ca'),certfile=str(self.root/'cert'),keyfile=str(self.root/'key'),
            registrations=[dict(snapshot=f.snapshot,execution=e.identity(),
                bundles={'bundle':str(f.f.fixture.bundle)},inventory=inventory_wire(f.f.inventory))])
        return config

    def check_config(self,config):
        self.file('config.json',canonical(config));self.file('runtime.json',canonical(self.manifest))
        with patch.object(n,'runtime_manifest',return_value='d'*64),patch.object(n.sys,'executable',str(self.python)):
            return n.preflight(str(self.root/'config.json'),str(self.root/'runtime.json'))

    def test_preflight_existing_native_template_and_actual_bundle_verifier(self):
        config=self.preflight_fixture()
        self.assertEqual(self.check_config(config),config)
        bundle=Path(config['registrations'][0]['bundles']['bundle'])
        model=bundle/'model.safetensors';model.chmod(0o600);model.write_bytes(b'altered')
        with self.assertRaises(Refused):self.check_config(config)

    def test_preflight_pin_topology_runtime_and_template_binding_refuse(self):
        original=self.preflight_fixture()
        for key,value in [('peer_pin',''),('host','0.0.0.0'),('host','127.0.0.1'),
                          ('host','host.docker.internal'),('runtime_manifest_sha256','f'*64),('deployment_id','f'*32)]:
            config=copy.deepcopy(original);config[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(Refused):self.check_config(config)
        (self.root/'key').chmod(0o644)
        with self.assertRaises(Refused):self.check_config(original)

    def test_control_consumer_actual_native_runtime_fake_tls_exact_hello(self):
        from tests.service.test_native_runtime import NativeFixture
        from service.native_runtime import NativeRuntime
        f=NativeFixture(self)
        for name in ('ca','cert','key'):self.file(name,b'INERT-NOT-A-CERTIFICATE')
        config=dict(version=1,variant=n.VARIANT,native=f.native,host='127.0.0.1',port=4443,
            peer_sha256=f.pin,cafile=str(self.root/'ca'),certfile=str(self.root/'cert'),keyfile=str(self.root/'key'),
            spool=str(f.spool),staging=str(f.stage))
        path=self.file('control.json',canonical(config))
        with patch.object(n.p,'tls_context',return_value=None):
            runtime=n.enrolled_runtime(str(path),connector=f.connect)
            self.assertIsInstance(runtime,NativeRuntime)
            self.assertEqual([q['op'] for q in f.requests],['hello'])
            config['native']=dict(f.native,boot='f'*32);path.write_bytes(canonical(config))
            with self.assertRaisesRegex(Refused,'control_hello'):n.enrolled_runtime(str(path),connector=f.connect)
            config['native']=f.native;path.write_bytes(canonical(config))
            f.cert=b'wrong peer'
            with self.assertRaisesRegex(Refused,'native_pin'):n.enrolled_runtime(str(path),connector=f.connect)

    def test_native_configuration_does_not_enter_linux_policy(self):
        from service.install_policy import configuration
        with self.assertRaises((ValueError,KeyError)):
            configuration(dict(version=1,ownership_policy=n.VARIANT))

    def test_serve_handoff_checked_snapshot_no_listener_or_process(self):
        config=self.preflight_fixture()
        with patch('worker.native_supervisor.NativeSupervisor') as supervisor, \
             patch('worker.native_supervisor.serve') as serve, \
             patch.object(n.p,'tls_context',return_value='INERT-TLS-CONTEXT'):
            n.serve_checked(config)
            serve.assert_called_once_with(supervisor.return_value,'INERT-TLS-CONTEXT',
                config['host'],config['port'],proven_host_network=False)
            self.assertEqual(supervisor.call_args.kwargs['python'],str(self.python))
            self.assertNotIn('keyfile',supervisor.call_args.kwargs)
            config['registrations']*=2
            with self.assertRaises(Refused):n.serve_checked(config)


if __name__=='__main__':unittest.main()
