"""SOURCE-only Pocket daemon transport fixtures; no Docker or ML execution."""
import copy
import json
import sqlite3
import unittest

from service import daemon_volumes as dv, daemon_helper as dh, entrypoint as ep
from service import install_policy as ip, installed_pocket_daemon as pd, installed_export as ex
from service import recovery
from tests.service.test_daemon_volumes import Session, container


class PocketDaemonTests(unittest.TestCase):
    def setUp(self):
        self.s = Session(self, family=pd.FAMILY)

    def test_install_init_up_status_stop_and_same_id_restart_v1(self):
        s = self.s
        self.assertEqual(s.run('install'), dict(phase='imported'))
        s.run('init')
        public = json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        self.assertEqual(public['scope'], ip.POCKET_V2_SCOPE)
        self.assertEqual(public['deployment_id'], s.b['installation'])
        self.assertEqual(public['roles']['staging']['uid'], 0)
        self.assertEqual(public['roles']['staging']['gid'], 10001)
        self.assertEqual(public['roles']['staging']['mode'], 0o1753)
        self.assertEqual(ip.expectation(ip.POCKET_SPLIT,'staging',directory=True), (0,10001,0o1753))
        path = s.roots['control']/'core/core.sqlite'
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone(), (1,))
            self.assertEqual(db.execute('SELECT * FROM p1_format').fetchall(), [(1,'synthetic-core')])
        first = s.run('up')
        self.assertEqual(s.run('status'), first)
        self.assertEqual(s.gate['version'], 1)
        self.assertEqual(s.gate['launch_policy'], pd.LAUNCH)
        self.assertEqual(s.gate['registration']['evidence'], dv.EVIDENCE)
        self.assertEqual(set(s.startup()['hello']), {'lane','boot','quarantined','launches'})
        first_gate = copy.deepcopy(s.gate)
        self.assertEqual(s.run('stop')['phase'], 'stopped')
        again = s.run('up')
        self.assertEqual(again['containers'], first['containers'])
        self.assertNotEqual(s.gate['registration']['boot'], first_gate['registration']['boot'])
        self.assertNotEqual(again['registration_id'], first['registration_id'])
        self.assertEqual(s.run('stop')['phase'], 'stopped')

    def test_family_configuration_and_binding_are_closed(self):
        s = self.s
        pd.configuration(s.config)
        dv.binding(s.b, family=pd.FAMILY)
        for family in (None, ex.FAMILY, object()):
            with self.assertRaises(ValueError): dv.binding(s.b, family=family)
        for policy in (ip.POLICY, ip.POCKET_POLICY, ex.POLICY):
            with self.assertRaises(ValueError): pd.configuration(dict(s.config,ownership_policy=policy))
        for key in ('configuration_digest','helper_image'):
            bad = copy.deepcopy(s.b); bad[key]='changed'
            with self.assertRaises(ValueError): dv.binding(bad,family=pd.FAMILY)
        bad = copy.deepcopy(s.config)
        next(iter(bad['bundles'].values()))['manifest_sha256']='0'*64
        with self.assertRaises(ValueError): pd.configuration(bad)

    def test_exact_assets_subpath_and_readonly_worker_mount(self):
        s = self.s
        c = container('worker',s.config,s.b,family=pd.FAMILY)
        dv.inspect_container(c,'worker',s.config,s.b,family=pd.FAMILY)
        mounts = c['HostConfig']['Mounts']
        assets = next(m for m in mounts if m['Target']=='/references/assets')
        self.assertEqual(assets['VolumeOptions'],dict(NoCopy=True,Subpath='core/assets'))
        self.assertIs(assets['ReadOnly'],True)
        for mutation in ('missing','whole-core','write','copy','extra'):
            bad=copy.deepcopy(c)
            m=next(m for m in bad['HostConfig']['Mounts'] if m['Target']=='/references/assets')
            if mutation=='missing':m['VolumeOptions'].pop('Subpath')
            elif mutation=='whole-core':m['VolumeOptions']['Subpath']='core'
            elif mutation=='write':m['ReadOnly']=False
            elif mutation=='copy':m['VolumeOptions']['NoCopy']=False
            else:bad['HostConfig']['Mounts'].append(copy.deepcopy(m))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                dv.inspect_container(bad,'worker',s.config,s.b,family=pd.FAMILY)
        bad=copy.deepcopy(c);bad['Image']='sha256:'+'f'*64
        with self.assertRaises(ValueError):dv.inspect_container(bad,'worker',s.config,s.b,family=pd.FAMILY)

    def test_changed_assets_inode_and_stale_boot_refuse(self):
        s=self.s;s.install()
        gate=copy.deepcopy(s.gate)
        public=pd.installation(json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public'])
        for mutation in ('assets','boot','config','binding'):
            bad=copy.deepcopy(gate)
            if mutation=='assets':bad['observation']['local']['worker']['roots']['assets'][1]+=1
            elif mutation=='boot':bad['observation']['hello']['boot']='f'*32
            elif mutation=='config':bad['configuration_digest']='f'*64
            else:bad['observation']['binding']='f'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                dh.gate_projection(bad,family=pd.FAMILY,configuration=s.config,binding=s.b,public=public)
        ep.projection(gate,family=pd.FAMILY)
        for family in (None,ex.FAMILY,object()):
            with self.assertRaises(ValueError):ep.projection(gate,family=family)
        s.run('stop')

    def test_control_startup_read_requires_pocket_family_and_live_incarnation(self):
        s=self.s;s.install()
        control=s.gate['observation']['local']['control']
        incarnation={k:control[k] for k in ('start_ticks','pid_namespace','cgroup')}
        def read(gate=s.gate, hello=None, observed=None, family=pd.FAMILY, policy=ip.POCKET_SPLIT):
            return recovery.installed_startup_read(s.roots['control']/'core',s.config,gate,
                s.startup()['hello'] if hello is None else hello,incarnation if observed is None else observed,
                policy=policy,launch_policy=pd.LAUNCH,family=family)
        checked=read()
        self.assertEqual(checked['deployment_id'],s.b['installation'])
        self.assertEqual(checked['registration_id'],s.gate['registration_id'])
        for family,policy in ((None,ip.POCKET_SPLIT),(ex.FAMILY,ip.EXPORT_SPLIT),(object(),ip.POCKET_SPLIT)):
            with self.assertRaises(ValueError):read(family=family,policy=policy)
        with self.assertRaises(ValueError):read(hello=dict(s.startup()['hello'],boot='f'*32))
        with self.assertRaises(ValueError):read(observed=dict(incarnation,start_ticks=incarnation['start_ticks']+1))
        s.run('stop')

    def test_changed_enrollment_id_and_role_tuple_refuse(self):
        s=self.s;s.run('install');s.run('init');public=pd.installation(json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public'])
        for mutate in (lambda p:p.update(deployment_id='f'*32),
                       lambda p:p['roles']['staging'].update(uid=10001),
                       lambda p:p['roles']['staging'].update(mode=0o1733),
                       lambda p:p['launch'].update(policy=dv.POLICY)):
            bad=copy.deepcopy(public);mutate(bad)
            with self.assertRaises(ValueError):pd.installation(bad)

    def test_partial_creation_and_notification_failure_are_retained(self):
        s=self.s;s.run('install');s.run('init')
        s.fail_action='notify'
        with self.assertRaises(OSError):s.run('up')
        self.assertTrue((s.roots['authority']/'containers.json').exists())
        self.assertTrue(any(s.roots['authority'].glob('registration-*.json')))
        s.fail_action=None
        with self.assertRaisesRegex(ValueError,'daemon_incomplete_operation_retained'):s.run('up')

    def test_partial_container_creation_retains_exact_first_id(self):
        s=self.s;s.run('install');s.run('init')
        original=s.action
        def fail_worker(action,payload):
            if action=='create' and payload['role']=='worker':raise OSError('injected creation failure')
            return original(action,payload)
        s.action=fail_worker
        with self.assertRaises(OSError):s.run('up')
        self.assertTrue((s.roots['authority']/'created-control.json').exists())
        self.assertFalse((s.roots['authority']/'containers.json').exists())
        with self.assertRaisesRegex(ValueError,'daemon_incomplete_operation_retained'):s.run('up')

    def test_pocket_fixed_entrypoints_and_export_are_separate(self):
        s=self.s
        for role,selector in (('control','daemon_pocket_main'),('worker','pocket_daemon_main'),('helper','pocket_main')):
            args=dv.command(role,s.config,s.b,phase='status' if role=='helper' else None,
                            nonce='a'*32 if role=='helper' else None,family=pd.FAMILY)
            self.assertTrue(any(selector in x for x in args))
            self.assertFalse(any('export_main' in x for x in args))
        with self.assertRaises(ValueError):ip.family_policy(object())


class PocketPackageRoutingTests(unittest.TestCase):
    def test_public_fixed_docker_commands_route_to_exact_family(self):
        from unittest.mock import patch
        from service import package_install as package, daemon_client as client
        for command,selected in (('docker','main'),('docker-export','export_main'),('docker-pocket','pocket_main')):
            with self.subTest(command=command), patch.object(client,'main') as ordinary, patch.object(client,'export_main') as export, patch.object(client,'pocket_main') as pocket:
                handlers=dict(main=ordinary,export_main=export,pocket_main=pocket)
                arguments=['status','--handle','explicit-handle.json']
                package.main([command,*arguments])
                handlers[selected].assert_called_once_with(arguments)
                for name,handler in handlers.items():
                    if name!=selected: handler.assert_not_called()

    def test_pocket_dispatch_uses_process_arguments(self):
        from unittest.mock import patch
        from service import package_install as package, daemon_client as client
        with patch('sys.argv',['speech-box','docker-pocket','stop','--handle','explicit.json']), patch.object(client,'pocket_main') as pocket:
            package.main()
            pocket.assert_called_once_with(['stop','--handle','explicit.json'])

    def test_package_and_pocket_help_expose_finite_commands(self):
        from contextlib import redirect_stdout
        import io
        from service import package_install as package, daemon_client as client
        for function,expected in ((package.main,'docker-pocket'),(client.pocket_main,'Pocket CPU')):
            output=io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
                function(['--help'])
            self.assertEqual(caught.exception.code,0)
            self.assertIn(expected,output.getvalue())
        self.assertIn('install,up,status,stop',output.getvalue())

    def test_pocket_rejects_unlisted_actions_before_daemon_access(self):
        from contextlib import redirect_stderr
        from unittest.mock import patch
        import io
        from service import package_install as package, daemon_client as client
        with patch.object(client,'Docker') as docker, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            package.main(['docker-pocket','repair','--handle','explicit.json'])
        self.assertEqual(caught.exception.code,2)
        docker.assert_not_called()
