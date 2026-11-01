"""Injected Docker/UID observations; tiny SQLite, no real child or daemon."""
import copy
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

from service import daemon_volumes as dv, daemon_helper as dh, daemon_client as dc
from service import installed_export as ex, install_child as ic
from service.core import encode
from tests.service.test_daemon_volumes import Session, container, assert_quarantined_stop


class ExportDaemonTests(unittest.TestCase):
    def session(self): return Session(self,family=ex.FAMILY)

    def test_export_quarantined_stop_preserves_fault_and_refuses_restart(self):
        assert_quarantined_stop(self,self.session())

    def initialized(self):
        s=self.session(); s.run('install'); s.run('init'); return s

    def registration(self,s,hook=None):
        # Exercise the real helper registration boundary while the ordinary
        # runtime entrypoint remains unavailable pending owner activation.
        def runtime(h):
            public=h.public(); ids={}
            for role in ('control','worker'):
                c=h.rpc('create',dict(role=role)); ids[role]=c['Id']
                h.inspect(role,ids,'start')
            if hook: hook(h,ids,public)
            rid,record=h.register(ids,public,dict(initial_fixture=True))
            return dict(phase='registered-fixture',registration_id=rid)
        # Do not run held() final snapshot with unexplained fixture runtime IDs;
        # persist the same normal creation receipts used by runtime().
        original=runtime
        def prepared(h):
            h.write('container-creation-intent.json',dict(binding=dv.digest(h.b)))
            create=h.rpc
            def rpc(action,payload):
                result=create(action,payload)
                if action=='create': h.write('created-'+payload['role']+'.json',dict(id=result['Id']))
                return result
            h.rpc=rpc
            return original(h)
        with patch.object(dh.Helper,'runtime',prepared): return s.run('up')

    def test_install_init_new_core_writer_sealed_registration(self):
        s=self.initialized()
        self.assertEqual(len(s.f.calls),1)
        result=self.registration(s)
        self.assertEqual(result['phase'],'registered-fixture')
        self.assertEqual(s.gate['version'],2)
        self.assertEqual(s.gate['launch_policy'],ex.LAUNCH)
        dh.gate_projection(s.gate,family=ex.FAMILY,configuration=s.config,binding=s.b,public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public'])
        public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        for version in (True,2.0,1):
            bad=copy.deepcopy(s.gate); bad['version']=version
            with self.assertRaises(ValueError): dh.gate_projection(bad,family=ex.FAMILY,configuration=s.config,binding=s.b,public=public)
        bad=dict(s.gate,extra=True)
        with self.assertRaises(ValueError): dh.gate_projection(bad,family=ex.FAMILY,configuration=s.config,binding=s.b,public=public)
        with self.assertRaises(ValueError): dh.gate_projection(s.gate)
        with self.assertRaises(ValueError): dv.binding(s.b)
        with self.assertRaises(ValueError): dv.command('worker',s.config,s.b)
        self.assertEqual(s.run('status')['phase'],'initialized')

    def test_notification_follows_release_of_real_core_exclusion(self):
        import fcntl
        s=self.initialized()
        observed=[]
        def hook(helper, ids, public):
            original=helper.rpc
            def rpc(action,payload):
                if action=='notify':
                    fd=os.open(s.roots['control']/'core/control.lock',os.O_RDONLY)
                    try:
                        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        observed.append(payload['gate']['registration_id'])
                        fcntl.flock(fd,fcntl.LOCK_UN)
                    finally: os.close(fd)
                return original(action,payload)
            helper.rpc=rpc
        result=self.registration(s,hook)
        self.assertEqual(observed,[result['registration_id']])

    def test_exact_subpath_commands_and_inspection_refusals(self):
        s=self.session(); good=container('worker',s.config,s.b,family=ex.FAMILY)
        args=dv.command('worker',s.config,s.b,family=ex.FAMILY)
        self.assertIn('--expected-worker-image',args)
        self.assertEqual(args[args.index('--expected-worker-image')+1],s.config['images']['worker'])
        self.assertTrue(any('volume-subpath=core/assets' in a and ',readonly' in a and 'volume-nocopy' in a for a in args))
        dv.inspect_container(good,'worker',s.config,s.b,family=ex.FAMILY)
        self.assertEqual(good['Mounts'][-1]['Source'],s.b['volumes']['control']['Mountpoint'])
        changes=[lambda c:c['HostConfig']['Mounts'][-1]['VolumeOptions'].pop('Subpath'),
            lambda c:c['HostConfig']['Mounts'][-1]['VolumeOptions'].update(Subpath='core'),
            lambda c:c['HostConfig']['Mounts'][-1]['VolumeOptions'].update(NoCopy=False),
            lambda c:c['HostConfig']['Mounts'][-1]['VolumeOptions'].update(NoCopy=1),
            lambda c:c['HostConfig']['Mounts'][-1].update(ReadOnly=1),
            lambda c:c['HostConfig']['Mounts'][-1]['VolumeOptions'].update(unknown=True),
            lambda c:c['Mounts'][-1].update(Source=s.b['volumes']['control']['Mountpoint']+'/foreign'),
            lambda c:c['Mounts'][-1].update(Name='copied-control'),
            lambda c:c['Mounts'][-1].update(RW=True),
            lambda c:c['Mounts'].append(copy.deepcopy(c['Mounts'][-1])),
            lambda c:c.update(Image='sha256:'+'f'*64),
            lambda c:c['Config']['Cmd'].pop()]
        for change in changes:
            bad=copy.deepcopy(good); change(bad)
            with self.assertRaises(ValueError): dv.inspect_container(bad,'worker',s.config,s.b,family=ex.FAMILY)
        helper=dv.command('helper',s.config,s.b,phase='init',nonce='a'*32,family=ex.FAMILY)
        self.assertTrue(any('export_main' in a for a in helper))

    def test_up_uses_existing_coordinator(self):
        s=self.initialized()
        self.assertEqual(s.run('up')['phase'],'ready')
        self.assertEqual([a for a,p in s.calls].count('create'),2)
        self.assertEqual([a for a,p in s.calls].count('notify'),1)

    def test_credential_principal_refuses_before_core(self):
        s=self.session(); i=next(i for i,e in enumerate(s.inputs) if e['role']=='credential')
        value=json.loads(s.bytes[i]); value['credentials'][0]['scopes']=['read']
        s.bytes[i]=encode(value).encode(); s.inputs[i]['bytes']=len(s.bytes[i])
        with self.assertRaisesRegex(ValueError,'credential_principals'): s.run('install')
        self.assertFalse((s.roots['control']/'core').exists())

    def test_seal_profile_principal_image_boot_and_assets_brackets_refuse(self):
        for fault in ('seal','profile','principals','image','boot','assets','start'):
            with self.subTest(fault=fault):
                s=self.initialized()
                def hook(h,ids,public):
                    startup=s.startup; calls=[0]
                    def changed():
                        value=startup(); calls[0]+=1
                        if calls[0]>=3:
                            if fault=='seal': value['hello']['export_seal']['converter_digest']='f'*64
                            elif fault=='profile': value['hello']['export_seal']['profile_digest']='f'*64
                            elif fault=='principals': value['hello']['export_seal']['principals_digest']='f'*64
                            elif fault=='boot': value['hello']['boot']='f'*32
                            elif fault=='assets': value['local']['roots']['assets'][1]+=1
                            elif fault=='image': s.cs[ids['worker']]['Image']='sha256:'+'f'*64
                            else: s.cs[ids['worker']]['State']['StartedAt']='2026-10-25T01:00:00Z'
                        return value
                    s.startup=changed
                with self.assertRaises(ValueError): self.registration(s,hook)
                self.assertIsNone(s.gate)

    def test_fixed_child_entrypoint_and_closed_family_request(self):
        s=self.initialized(); public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        request=json.loads(s.f.calls[0] if isinstance(s.f.calls[0],str) else encode(s.f.calls[0]))
        raw=encode(request).encode()
        with patch.object(ic.sys,'platform','linux'),patch.object(os,'getuid',return_value=0),patch.object(os,'geteuid',return_value=0),\
                patch.object(ic.sys,'stdin',Mock(buffer=io.BytesIO(raw))),\
                patch.object(ic,'prepare_container',return_value=(s.roots['control'],s.roots['staging'])) as prepare,\
                patch.object(ic,'close_other_descriptors'),patch.object(ic,'drop_control'),patch.object(ic,'verify_dropped_capabilities'),\
                patch('service.persistent_init._split_core') as core:
            ic.export_container_entry('core')
        self.assertIs(prepare.call_args.kwargs['family'],ex.FAMILY)
        self.assertIs(core.call_args.kwargs['family'],ex.FAMILY)
        with self.assertRaises(ValueError): ic.prepare_container(public,request['control_fd'],request['staging_fd'])

    def test_private_client_install_exact_family_and_preflight_refusal(self):
        s=self.session(); entries=[]
        for index,(entry,raw) in enumerate(zip(s.inputs,s.bytes)):
            path=s.root/('input-'+str(index)); path.write_bytes(raw)
            entries.append(dict(entry,source=str(path)))
        request=dict(configuration=s.config,helper_image=s.b['helper_image'],inputs=entries,
            cpu=s.b['cpu'],tmpfs=s.b['tmpfs'],control_port=s.b['control_port'])
        request_path=s.root/'request.json'; request_path.write_text(encode(request))
        calls=[]; phases=[]
        class Docker:
            frozen=s.b['daemon']
            def __init__(self,context,*,family):
                self_test.assertIs(family,ex.FAMILY); self_test.assertEqual(context,'fixture')
            def call(self,args):
                calls.append(args)
                if args[:2]==['image','inspect']: return args[2]
                if args[:2]==['volume','ls']: return ''
                if args[:2]==['volume','create']: return args[-1]
                if args[:2]==['container','create']:
                    self_test.assertTrue(any('export_main' in a for a in args)); return '3'*64
                raise AssertionError(args)
            def json(self,args):
                name=args[2]; role=name.rsplit('-',1)[1]
                value=copy.deepcopy(s.b['volumes'][role]); value['Name']=name
                installation=name.split('-')[1]; value['Labels']['speech-box.installation']=installation
                return [value]
            def check(self): pass
            def exchange(self,identity,phase,nonce,config,b,source):
                dv.binding(b,family=ex.FAMILY); ex.configuration(config); phases.append(phase)
                return dict(phase=phase)
        self_test=self
        argv=['install','--context','fixture','--request',str(request_path),'--handle',str(s.root/'handle')]
        with patch.object(dc,'Docker',Docker),patch('builtins.print'):
            dc.export_main(argv)
        self.assertEqual(phases,['install','init'])
        retained=json.loads((s.root/'handle.binding').read_bytes())
        self.assertEqual(retained['binding']['policy'],ex.LAUNCH)
        for fault in ('unknown','principal','crossfamily'):
            bad=copy.deepcopy(request)
            if fault=='unknown': bad['configuration']['extra']=True
            elif fault=='principal':
                # A configured principal without a production-write credential.
                value=json.loads(s.bytes[-1]); value['credentials'][0]['scopes']=['read']
                Path(entries[-1]['source']).write_text(encode(value))
            else: bad['configuration']['version']=1
            request_path.write_text(encode(bad))
            with patch.object(dc,'Docker') as docker,self.assertRaises(SystemExit): dc.export_main(argv)
            docker.assert_not_called()

    def test_child_adapter_selects_fixed_new_entry_and_hello_configuration(self):
        config=self.session().config
        request=dict(ipc_fd=9,configuration=config)
        fake=Mock()
        with patch.object(dh.subprocess,'Popen',return_value=fake) as spawn,patch.object(dh,'collect',return_value=b'{}') as collect:
            self.assertEqual(dh.child('hello',request,(9,),family=ex.FAMILY),{})
        self.assertIn("export_container_entry('hello')",spawn.call_args.args[0][-1])
        self.assertEqual(json.loads(collect.call_args.args[1]),request)
        with patch.object(ic.sys,'platform','linux'),patch.object(os,'getuid',return_value=0),patch.object(os,'geteuid',return_value=0),\
                patch.object(ic.sys,'stdin',Mock(buffer=io.BytesIO(encode(request).encode()))),\
                patch('service.install_policy.check'),\
                patch.object(os,'fstat',return_value=Mock()),patch.object(ic,'close_other_descriptors'),\
                patch.object(ic,'drop_control'),patch.object(ic,'verify_dropped_capabilities'),\
                patch('service.supervisor_runtime.installed_hello_at',return_value={}) as hello,patch('builtins.print'):
            ic.export_container_entry('hello')
        self.assertIs(hello.call_args.kwargs['family'],ex.FAMILY)
        self.assertEqual(hello.call_args.kwargs['configuration'],config)

    def test_subpath_daemon_failure_has_no_fallback(self):
        s=self.session(); d=object.__new__(dc.Docker); d.family=ex.FAMILY
        d.check=Mock(); d.call=Mock(side_effect=ValueError('injected unsupported volume-subpath'))
        with self.assertRaisesRegex(ValueError,'unsupported volume-subpath'):
            d.action('create',dict(role='worker'),s.config,s.b,[])
        self.assertEqual(d.call.call_count,1)
        self.assertTrue(any('volume-subpath=core/assets' in arg for arg in d.call.call_args.args[0]))

class ExportSnapshotTests(unittest.TestCase):
    session=ExportDaemonTests.session
    initialized=ExportDaemonTests.initialized
    # Select only these methods in the inert runner (do not duplicate inherited cases).
    def snapshot_fixture(self):
        s=self.initialized()
        public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        return s,public,s.roots['control']/'core/core.sqlite'

    def source_state(self,path):
        return {p.name:(p.read_bytes(),tuple(getattr(p.stat(),k) for k in
            ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')))
            for p in path.parent.iterdir() if p.is_file()}

    def checked_snapshot(self,s,public,path,*,refusal=False,application_id=0):
        import sqlite3
        from service import install_policy as ip
        before=self.source_state(path); namespace=sorted(p.name for p in path.parent.iterdir())
        connect=sqlite3.connect; targets=[]
        def opened(target,*args,**kwargs):
            targets.append(str(target))
            self.assertNotIn(str(path.parent),str(target))
            db=connect(target,*args,**kwargs)
            if str(target)!=':memory:':
                self.assertTrue(str(target).startswith('file:'))
                self.assertEqual(db.execute('PRAGMA application_id').fetchone(),(application_id,))
            return db
        with ip.locked_file(path.parent/'control.lock',policy=ip.SPLIT,role='control'),patch.object(sqlite3,'connect',side_effect=opened):
            if refusal:
                with self.assertRaises(ValueError): s.helper.registration_snapshot(public)
            else:
                result=s.helper.registration_snapshot(public)
                self.assertEqual(result['protocol'],2)
        self.assertEqual(self.source_state(path),before)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()),namespace)
        return [t for t in targets if t!=':memory:']

    def test_snapshot_preserves_source_without_and_with_live_wal(self):
        import sqlite3
        s,public,path=self.snapshot_fixture()
        self.assertFalse(path.with_name('core.sqlite-wal').exists())
        self.assertEqual(len(self.checked_snapshot(s,public,path)),1)
        db=sqlite3.connect(path); self.addCleanup(db.close)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA application_id=386')
        db.commit()
        # Real committed WAL bytes must be copied, not ignored or checkpointed.
        self.assertGreater(path.with_name('core.sqlite-wal').stat().st_size,0)
        self.assertEqual(len(self.checked_snapshot(s,public,path,application_id=386)),1)
        from service import recovery
        # Cumulative DB+WAL and DB+WAL+SHM limits, using real tiny sidecars.
        total=path.stat().st_size
        for name in ('core.sqlite-wal','core.sqlite-shm'):
            total+=(path.parent/name).stat().st_size
            with patch.object(recovery,'MAX_BYTES',total-1):
                self.assertEqual(self.checked_snapshot(s,public,path,refusal=True),[])

    def test_snapshot_refusal_preserves_source_and_checks_private_target(self):
        s,public,path=self.snapshot_fixture()
        from service import installed_export as export
        original=export.database
        def bad(db,public):
            data=original(db,public); data['resource_accounting']=[(1,1)]; return data
        with patch.object(export,'database',side_effect=bad):
            self.assertEqual(len(self.checked_snapshot(s,public,path,refusal=True)),1)

    def test_snapshot_preopen_finite_byte_and_namespace_caps(self):
        from service import recovery
        s,public,path=self.snapshot_fixture()
        for budget in (1,path.stat().st_size-1):
            with patch.object(recovery,'MAX_BYTES',budget):
                self.assertEqual(self.checked_snapshot(s,public,path,refusal=True),[])
        with patch.object(recovery,'MAX_ROWS',1):
            self.assertEqual(self.checked_snapshot(s,public,path,refusal=True),[])

    def test_snapshot_detects_source_change_before_open_and_after_query(self):
        from service import recovery, installed_export as export, install_policy as ip
        s,public,path=self.snapshot_fixture()
        # A legitimate external namespace change is detected; the snapshot must
        # retain that change rather than repairing/removing it.
        marker=path.parent/'external-change'
        before=self.source_state(path)
        read=recovery.read_file; calls=[]
        def changed(*args,**kwargs):
            data=read(*args,**kwargs); calls.append(1)
            if len(calls)==1: marker.write_bytes(b'external')
            return data
        with ip.locked_file(path.parent/'control.lock',policy=ip.SPLIT,role='control'),patch.object(recovery,'read_file',side_effect=changed),self.assertRaisesRegex(ValueError,'snapshot_source_changed'):
            s.helper.registration_snapshot(public)
        self.assertEqual(marker.read_bytes(),b'external')
        self.assertEqual({k:v for k,v in self.source_state(path).items() if k!=marker.name},before)
        marker.unlink()  # fixture-owned mutation only
        original=export.database
        def after(db,public):
            data=original(db,public); marker.write_bytes(b'after-query'); return data
        with ip.locked_file(path.parent/'control.lock',policy=ip.SPLIT,role='control'),patch.object(export,'database',side_effect=after),self.assertRaisesRegex(ValueError,'snapshot_source_changed'):
            s.helper.registration_snapshot(public)
        self.assertEqual(marker.read_bytes(),b'after-query')
        self.assertEqual({k:v for k,v in self.source_state(path).items() if k!=marker.name},before)
