"""Explicit daemon-side observations, disposable files; no Docker/child execution."""
import copy
import json
import tempfile
import zipfile
import unittest
from unittest.mock import patch
from service import cpu_setup as c
from tests.service.test_installed_host_policy import InstalledHost


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.f = f = InstalledHost(); f.setUp(); self.addCleanup(f.doCleanups)
        for r in f.policy['roots'].values():
            p=c.Path(r['source']); p.chmod(0o700); f.owner(p,0,0)
        self.request=dict(layout='linux-local', daemon_root=str(f.root), configuration=f.config,
            scope=f.scope, inputs={k:v['source'] for k,v in f.policy['inputs'].items()},
            cpu=f.policy['cpu'],tmpfs=f.policy['tmpfs'],control_port=8765)
        self.calls=[]
        self.root=str(f.root);self.arch='x86_64'
        class Docker:
            context=f.scope['context']
            def _read(_, args):
                self.calls.append(args)
                if args[0]=='image':return args[2]
                if args[-1]=='{{.DockerRootDir}}':return self.root
                if args[-1]=='{{.Architecture}}':return self.arch
                raise AssertionError(args)
        self.docker=Docker()

    def generate(self):
        with patch.object(c.sys,'platform','linux'),patch.object(c.os,'getuid',return_value=0),patch.object(c.os,'geteuid',return_value=0), \
             patch.object(c.h,'_host_scope',return_value=self.f.volumes), \
             patch.object(c.h.InstalledObserver,'initialize_authority',return_value={}) as check:
            result=c.generate(self.request,docker=self.docker)
            self.assertEqual(check.call_count,2)
            return result

    def test_two_layouts_exact_generated_policy_without_credentials(self):
        for layout in ('linux-local','intel-mac-linux-vm'):
            self.request['layout']=layout
            result=self.generate()
            self.assertEqual(result['phase'],'configuration-only')
            self.assertEqual(result['host_policy'],self.f.policy)
            self.assertNotIn('NEVER-RETURN',c.encode(result))
        self.assertTrue(all(a[0] in ('info','image') for a in self.calls))

    def test_wrong_root_arch_layout_owner_credential_and_nonempty_refuse(self):
        original=copy.deepcopy(self.request)
        for field,value in [('daemon_root','/wrong'),('layout','desktop')]:
            self.request=copy.deepcopy(original);self.request[field]=value
            with self.assertRaises(ValueError):self.generate()
        self.request=original;self.arch='aarch64'
        with self.assertRaisesRegex(ValueError,'x86_64'):self.generate()
        self.arch='x86_64'
        credential=c.Path(self.request['inputs']['credential']);credential.chmod(0o644)
        with self.assertRaises(ValueError):self.generate()
        credential.chmod(0o640)
        p=c.Path(self.f.policy['roots']['control']['source']);self.f.owner(p,10001,10001)
        with self.assertRaisesRegex(ValueError,'fresh_roots'):self.generate()
        self.f.owner(p,0,0);(p/'unexplained').write_text('retain')
        with self.assertRaisesRegex(ValueError,'empty_roots'):self.generate()
        self.assertTrue((p/'unexplained').exists())

    def test_real_observer_setup_and_denials(self):
        real = c.h.InstalledObserver
        base = self.f.observer([]).docker
        read = self.docker._read
        def observed(args):
            return read(args) if args[0]=='image' or args[-1] in ('{{.DockerRootDir}}','{{.Architecture}}') else base._read(args)
        self.docker._read = observed
        def factory(docker, config, scope, policy):
            return real(docker,config,scope,policy,mountinfo=lambda:self.f.mounts)
        with patch.object(c.sys,'platform','linux'),patch.object(c.os,'getuid',return_value=0),patch.object(c.os,'geteuid',return_value=0),patch.object(c.h,'InstalledObserver',side_effect=factory):
            result=c.generate(self.request,docker=self.docker)
            self.assertEqual(result['host_policy'],self.f.policy)
            for endpoint in ('ssh://remote','unix:///run/user/1/docker.sock'):
                self.docker._read=lambda args:json.dumps(endpoint) if args[0]=='context' else observed(args)
                with self.assertRaisesRegex(ValueError,'unsupported_host_pid'):c.generate(self.request,docker=self.docker)
            self.docker._read=lambda args:c.encode(dict(id='wrong',os='linux')) if args[0]=='info' and '"id"' in args[-1] else observed(args)
            with self.assertRaisesRegex(ValueError,'daemon_mismatch'):c.generate(self.request,docker=self.docker)

    def test_bundle_cli_status_reuses_held_authority(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from service import package_install as package
        result=self.generate();output=self.f.root/'setup-bundle.json';c.write_bundle(result,output)
        @contextlib.contextmanager
        def authority(config,scope):
            self.assertEqual(config,self.f.config);self.assertEqual(scope,self.f.scope)
            yield SimpleNamespace(policy=self.f.policy)
        stream=io.StringIO()
        with patch.object(package.sys,'version_info',(3,11)),patch.object(c.h,'installed_launch_authority',side_effect=authority),patch.object(package.Launcher,'status',return_value={'phase':'retained'}),patch.object(c.h.Docker,'_read',side_effect=AssertionError('no Docker')),contextlib.redirect_stdout(stream):
            previous=c.os.umask(0o077)
            try:package.main(['status','--setup-bundle',str(output)])
            finally:c.os.umask(previous)
        self.assertEqual(json.loads(stream.getvalue()),{'phase':'retained'})

    def test_mac_client_refuses_before_transport(self):
        with patch.object(c.sys,'platform','darwin'):
            with self.assertRaisesRegex(ValueError,'inside_linux'):c.generate(self.request,docker=self.docker)
        self.assertEqual(self.calls,[])

    def test_archive_allowlist_bytes_and_determinism_without_child(self):
        from service.package_install import archive
        root=c.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            a=c.Path(tmp)/'a.zip'; b=c.Path(tmp)/'b.zip'
            self.assertEqual(archive(root,a),archive(root,b))
            with zipfile.ZipFile(a) as z:
                manifest=json.loads((root/'headless-dist-manifest.json').read_text())
                self.assertEqual(set(z.namelist()),{'speech-box/speech-box'} | {'speech-box/'+n for n in manifest['files']})
                for n in manifest['files']:self.assertEqual(z.read('speech-box/'+n),(root/n).read_bytes())

    def test_exclusive_result_does_not_replace(self):
        output=self.f.root/'bundle.json';result=self.generate()
        c.write_bundle(result,output);before=output.read_bytes()
        with self.assertRaises(FileExistsError):c.write_bundle(result,output)
        self.assertEqual(output.read_bytes(),before)


if __name__=='__main__':unittest.main()
