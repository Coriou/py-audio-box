"""Nonqualifying inert package manifests; no interpreter, socket or ML launch."""
import copy
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
             patch.object(n,'dependency_closure'):
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
             patch.object(n.importlib.metadata,'distributions',return_value=[SimpleNamespace(requires=['missing>=1'])]):
            with self.assertRaises(Refused):n.dependency_closure(self.versions)
        with patch.object(n,'distributions',return_value=self.versions), \
             patch.object(n.importlib.metadata,'distributions',return_value=[SimpleNamespace(requires=['torch==1'])]):
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
