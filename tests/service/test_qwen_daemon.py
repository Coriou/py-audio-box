"""Finite Qwen daemon command and authority fences, without Docker or ML."""
import copy
from unittest.mock import patch
from service import daemon_volumes as dv, daemon_client as dc, package_install
from service import installed_qwen_daemon as qd, installed_pocket_daemon as pd
from tests.service.test_installed_qwen import InstalledQwenTests
from tests.service.test_daemon_volumes import binding, container


class QwenDaemonTests(InstalledQwenTests):
    def test_real_worker_entrypoint_configures_qwen_before_serve(self):
        from types import SimpleNamespace
        from service import execution_context as ec, install_policy as ip
        from worker import runtime_supervisor as rs
        from worker.engines import qwen_execution as qe
        from worker.runtime_protocol import execution

        supervisor = SimpleNamespace(qwen_contexts={})
        mapping = {'inert-bundle': '/sealed/inert-bundle'}
        with patch('sys.argv', ['worker', '--installed-policy', qd.POLICY,
                                '--configuration-file', '/config/install.json']), \
                patch.object(rs.os, 'getuid', return_value=0), \
                patch.object(rs.os, 'geteuid', return_value=0), \
                patch.object(rs.os, 'getgid', return_value=10001), \
                patch.object(rs.os, 'getegid', return_value=10001), \
                patch.object(rs.os, 'getgroups', return_value=[]), \
                patch.object(rs.os, 'umask'), \
                patch.object(ec, 'installed_configuration', return_value=self.config) as load, \
                patch.object(ec, 'installed_mapping', return_value=mapping), \
                patch.object(ip, 'directory_identity'), patch.object(ip, 'read_private'), \
                patch.object(rs, 'RuntimeSupervisor', return_value=supervisor) as factory, \
                patch.object(qe, 'installed_cpu_hardware', wraps=qe.installed_cpu_hardware) as hardware, \
                patch('sys.platform', 'linux'), \
                patch.object(rs.os, 'sched_getaffinity', return_value={0, 1}, create=True):
            from unittest.mock import Mock
            def served():
                self.assertTrue(supervisor._qwen_daemon)
                self.assertFalse(supervisor._pocket_daemon)
                self.assertEqual(supervisor.qwen_profiles, self.config['profiles'])
                for principal, context in self.config['contexts'].items():
                    self.assertEqual(supervisor.qwen_contexts[(principal, 'qwen-cpu', 'managed-clone')], context)
                self.assertIsNotNone(supervisor.inventory)
                self.assertEqual(hardware.call_count, len(self.config['contexts']))
            supervisor.serve = Mock(side_effect=served)
            rs.qwen_daemon_main()
            supervisor.serve.assert_called_once_with()
            load.assert_called_once_with('/config/install.json', policy=ip.SPLIT)
            self.assertIs(factory.call_args.kwargs['installed_policy'], ip.QWEN_SPLIT)
            self.assertEqual(factory.call_args.kwargs['bundles'], mapping)
            context = self.config['contexts']['alice']
            hardware.assert_called_once_with(execution(context), supervisor.inventory, family=qd.FAMILY)

    def test_exact_commands_and_readonly_subpath(self):
        b=binding(self.config);b['policy']=qd.LAUNCH
        dv.binding(b,family=qd.FAMILY)
        for role,selector in [('helper','qwen_main'),('control','daemon_qwen_main'),('worker','qwen_daemon_main')]:
            args=dv.command(role,self.config,b,phase='status' if role=='helper' else None,nonce='a'*32 if role=='helper' else None,family=qd.FAMILY)
            self.assertTrue(any(selector in x for x in args))
            self.assertIn('--pull=never',args)
            c=container(role,self.config,b,phase='status' if role=='helper' else None,nonce='a'*32 if role=='helper' else None,family=qd.FAMILY)
            dv.inspect_container(c,role,self.config,b,phase='status' if role=='helper' else None,nonce='a'*32 if role=='helper' else None,family=qd.FAMILY)
            bad=copy.deepcopy(c);bad['Image']='sha256:'+'f'*64
            with self.assertRaises(ValueError):dv.inspect_container(bad,role,self.config,b,phase='status' if role=='helper' else None,nonce='a'*32 if role=='helper' else None,family=qd.FAMILY)
        for family in (None,pd.FAMILY,object()):
            with self.assertRaises(ValueError):dv.binding(b,family=family)
        assets=next(m for m in dv.mounts('worker',b,family=qd.FAMILY) if m['Destination']=='/references/assets')
        self.assertIs(assets['RW'],False)

    def test_fixed_package_route_and_no_repair(self):
        with patch.object(dc,'qwen_main') as command:
            package_install.main(['docker-qwen','status','--handle','explicit.json'])
            command.assert_called_once_with(['status','--handle','explicit.json'])
        with patch.object(dc,'Docker') as docker,self.assertRaises(SystemExit):
            dc.qwen_main(['repair','--handle','explicit.json'])
        docker.assert_not_called()

    def test_large_model_import_is_explicit_family_only(self):
        entries=[dict(role='models',path='bundles/b/model.safetensors',source='unused',bytes=1024**3,sha256='a'*64),dict(role='config',path='install.json',source='unused',bytes=1,sha256='b'*64),dict(role='credential',path='control.key',source='unused',bytes=1)]
        dc.manifest(entries,family=qd.FAMILY)
        for size in (2*1024**3+1,4*1024**3+1):
            oversized=copy.deepcopy(entries);oversized[0]['bytes']=size
            with self.assertRaises(ValueError):dc.manifest(oversized,family=qd.FAMILY)
        oversized=copy.deepcopy(entries)
        oversized[0]['bytes']=2*1024**3
        oversized.insert(1,dict(oversized[0],path='bundles/b/second.safetensors'))
        with self.assertRaises(ValueError):dc.manifest(oversized,family=qd.FAMILY)
        for family in (None,pd.FAMILY):
            with self.assertRaises(ValueError):dc.manifest(entries,family=family)

    def session(self):
        """Existing daemon fixture with tiny inert Qwen bytes and injected pin."""
        import hashlib
        import json
        from service.core import encode
        from service import install_policy as ip
        from service.provisioning import manifest_bytes
        from tests.service.test_daemon_volumes import Session
        s=Session(self)
        manifest=json.loads((self.b.bundle/'manifest.json').read_bytes())
        manifest.update(checkpoint='Qwen/Qwen3-TTS-12Hz-0.6B-Base',revision='5d83992436eae1d760afd27aff78a71d676296fc')
        raw=manifest_bytes(manifest)
        old=self.config['contexts']['alice']['execution']['descriptor']['manifest_sha256']
        new=hashlib.sha256(raw).hexdigest()
        # Every occurrence of the execution pin is replaced together, never a
        # relaxation in production validation. No actual weight bytes exist.
        config=json.loads(json.dumps(self.config).replace(old,new))
        self.enterContext(patch.object(qd,'MANIFEST_SHA256',new))
        s.config=config;s.f.config=config;s.family=qd.FAMILY
        s.b=binding(config);s.b['policy']=qd.LAUNCH
        s.scope=dv.scope(s.b,family=qd.FAMILY);s.f.scope=s.scope
        bid=next(iter(config['bundles']))
        credential=next(raw for e,raw in zip(s.inputs,s.bytes) if e['role']=='credential')
        data=[('models','bundles/'+bid+'/'+entry['path'],(self.b.bundle/entry['path']).read_bytes()) for entry in manifest['files']]
        data += [('models','bundles/'+bid+'/manifest.json',raw),('config','install.json',encode(config).encode()),('credential','control.key',credential)]
        s.inputs=[];s.bytes=[]
        for role,path,content in data:
            entry=dict(role=role,path=path,bytes=len(content))
            if role!='credential':entry['sha256']=hashlib.sha256(content).hexdigest()
            s.inputs.append(entry);s.bytes.append(content)
        return s

    def test_injected_install_register_status_stop_restart(self):
        import json
        from service import install_policy as ip
        s=self.session();s.run('install');s.run('init')
        public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        qd.installation(public)
        self.assertEqual(public['scope'],ip.QWEN_SCOPE)
        first=s.run('up');self.assertEqual(s.run('status'),first)
        registration=s.gate['registration_id']
        self.assertEqual(s.run('stop')['phase'],'stopped')
        again=s.run('up')
        self.assertEqual(again['containers'],first['containers'])
        self.assertNotEqual(s.gate['registration_id'],registration)
        self.assertEqual(s.run('stop')['phase'],'stopped')

    def test_unacknowledged_or_unknown_reap_refuses_clean_stop(self):
        from service import daemon_helper as dh
        for reason in ('unacknowledged_dispatch','unknown_reap'):
            s=self.session();s.install()
            with patch.object(dh.Helper,'clean',side_effect=ValueError(reason)):
                with self.assertRaisesRegex(ValueError,reason):s.run('stop')
            self.assertFalse(list(s.roots['authority'].glob('stopped-*.json')))

    def test_tampered_retained_configuration_and_failed_notify_remain_retained(self):
        s=self.session();s.run('install');s.run('init');s.fail_action='notify'
        with self.assertRaises(OSError):s.run('up')
        self.assertTrue((s.roots['authority']/'containers.json').exists())
        s.fail_action=None
        with self.assertRaisesRegex(ValueError,'daemon_incomplete_operation_retained'):s.run('up')
