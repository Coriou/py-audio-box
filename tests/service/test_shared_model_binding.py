"""Detached Slice B policy only; no store reference or installation shortcut."""
import copy
import unittest
from service import daemon_volumes as dv, installed_qwen_daemon as qd
from service import installed_pocket_daemon as pd, installed_export as ex
from service import shared_models as sm, install_policy as ip
from tests.service.test_installed_qwen import InstalledQwenTests

# Existing command inspection fixture, kept local to avoid importing the
# unrelated MCP-dependent legacy test aggregation.
def binding(config):
    installation='a'*32
    return dict(policy=dv.POLICY,installation=installation,daemon=dict(endpoint='unix:///Users/operator/.docker/run/docker.sock',
        id='fixture-desktop-linux',os='linux',architecture='amd64',security=['name=seccomp,profile=builtin'],cgroup='2'),
        volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
            Mountpoint='/opaque-daemon/volumes/'+r,CreatedAt='2026-10-25T00:00:00Z',
            Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES},
        helper_image=config['images']['control'],configuration_digest=dv.digest(config),cpu=dict(control=10**9,
            worker=max(c['execution']['descriptor']['threads'] for c in config['contexts'].values())*10**9),
        tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765)


def container(role,config,b,*,phase=None,nonce=None,identity=None,family=None):
    args=dv.command(role,config,b,phase=phase,nonce=nonce,family=family)
    helper=role=='helper'; t=b['tmpfs']['control' if helper else role]
    return dict(Id=identity or ('1' if role=='control' else '2')*64,Image=b['helper_image'] if helper else config['images'][role],
        Config=dict(User='0:0' if helper else '10001:10001' if role=='control' else '0:10001',Entrypoint=['python'],
            Cmd=args[args.index('--entrypoint')+3:]),
        HostConfig=dict(Privileged=False,ReadonlyRootfs=True,Init=None,PidMode='',IpcMode='private',CgroupnsMode='private',UsernsMode='',
            NetworkMode='bridge' if role=='control' else 'none',CapDrop=['ALL'],
            CapAdd=list(dv.INIT_CAPS if phase in ('install','init') else dv.LATER_CAPS) if helper else [] if role=='control' else list(dv.WORKER_CAPS),
            SecurityOpt=['no-new-privileges:true'],RestartPolicy=dict(Name='no',MaximumRetryCount=0),Devices=[],DeviceRequests=None,Binds=None,
            Memory=config['budget']['ram_bytes'],MemorySwap=config['budget']['ram_bytes'],PidsLimit=config['budget']['pids'],
            Mounts=[dict(Type='volume',Source=m['Name'],Target=m['Destination'],ReadOnly=not m['RW'],VolumeOptions=dict(NoCopy=True,**({'Subpath':'core/assets'} if m['Destination']=='/references/assets' else {}))) for m in dv.mounts(role,b,phase,family=family)],
            NanoCpus=b['cpu']['control' if helper else role],Tmpfs={'/tmp':'rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777'},
            PortBindings={'8000/tcp':[dict(HostIp='127.0.0.1',HostPort=str(b['control_port']))]} if role=='control' else {}),
        State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',StartedAt='0001-01-01T00:00:00Z',Pid=0),RestartCount=0,
        Mounts=dv.mounts(role,b,phase,family=family))



def shared(config):
    b = binding(config); b['policy'] = qd.LAUNCH
    key = qd.MANIFEST_SHA256; attempt = 'b'*32
    def volume(name, labels):
        return dict(Name=name, Driver='local', Options=None, Scope='local',
                    Mountpoint='/opaque-store/'+name, CreatedAt='2026-10-27T00:00:00Z', Labels=labels)
    b['volumes']['models'] = volume('pab-model-sha256-'+key+'-'+attempt, sm.labels(key, attempt))
    b['model_storage'] = dict(kind=sm.STORAGE, manifest_sha256=key, bundle_id=key,
        catalog=volume(sm.CATALOG, sm.labels()), publication_sha256='e'*64, reference_id=b['installation'])
    return b


class SharedBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = InstalledQwenTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.config
        # Reauthor the inert config consistently; never change frozen fields singly.
        old = next(iter(self.config['bundles']))
        import json
        def rename(v):
            if isinstance(v, dict): return {(qd.MANIFEST_SHA256 if k == old else k): rename(x) for k,x in v.items()}
            if isinstance(v, list): return [rename(x) for x in v]
            if isinstance(v, str):
                return qd.MANIFEST_SHA256 if v == old else v.replace('/bundles/'+old, '/bundles/'+qd.MANIFEST_SHA256)
            return v
        self.config = rename(self.config)
        self.b = shared(self.config)

    def test_real_builders_inspectors_all_phases(self):
        qd.binding_configuration(self.b, self.config)
        for role, phase in [('helper', p) for p in dv.PHASES]+[('control', None), ('worker', None)]:
            kw = dict(phase=phase, nonce='c'*32 if phase else None, family=qd.FAMILY)
            with self.subTest(role=role, phase=phase):
                c = container(role, self.config, self.b, **kw)
                dv.inspect_container(c, role, self.config, self.b, **kw)
                mounts = dv.mounts(role, self.b, phase, family=qd.FAMILY)
                self.assertFalse(next(m['RW'] for m in mounts if m['Destination']=='/models'))
                self.assertEqual(any(m['Destination']==sm.CATALOG_ROOT for m in mounts), role=='helper')
                for actual in (True, False):
                    bad = copy.deepcopy(c)
                    if actual:
                        next(m for m in bad['Mounts'] if m['Destination']=='/models')['RW'] = True
                    else:
                        next(m for m in bad['HostConfig']['Mounts'] if m['Target']=='/models')['ReadOnly'] = False
                    with self.assertRaises(ValueError): dv.inspect_container(bad, role, self.config, self.b, **kw)
                bad=copy.deepcopy(c);bad['Mounts'].append(dict(c['Mounts'][0],Destination='/alias'))
                with self.assertRaises(ValueError):dv.inspect_container(bad,role,self.config,self.b,**kw)
                bad=copy.deepcopy(c);bad['HostConfig']['Mounts'][0]['VolumeOptions']['NoCopy']=False
                with self.assertRaises(ValueError):dv.inspect_container(bad,role,self.config,self.b,**kw)

    def test_closed_shapes_mixing_and_aliases(self):
        for family in (None,pd.FAMILY,ex.FAMILY,object()):
            with self.subTest(family=family),self.assertRaises(ValueError):dv.binding(self.b,family=family)
        mutations = []
        def change(path, value):
            bad=copy.deepcopy(self.b); cursor=bad
            for name in path[:-1]:cursor=cursor[name]
            cursor[path[-1]]=value;mutations.append(bad)
        for field in ('kind','manifest_sha256','bundle_id','reference_id','publication_sha256'):
            change(('model_storage',field),'forged')
        change(('model_storage','extra'),True)
        for field,value in [('Driver','foreign'),('Options',{'device':'/alias'}),('Scope','global'),('Labels',{})]:
            change(('volumes','models',field),value)
            change(('model_storage','catalog',field),value)
        change(('volumes','models'),binding(self.config)['volumes']['models'])
        change(('volumes','config'),self.b['volumes']['models'])
        change(('volumes','control','Mountpoint'),'/opaque-store/../control')
        for role in ('models','control'):
            for suffix in ('','/nested'):
                change(('model_storage','catalog','Mountpoint'),self.b['volumes'][role]['Mountpoint']+suffix)
        bad=copy.deepcopy(self.b);del bad['model_storage'];mutations.append(bad)
        for bad in mutations:
            with self.subTest(bad=bad),self.assertRaises(ValueError):qd.binding(bad)

    def test_original_mountpoint_aliases_refused_by_real_parsers(self):
        public = self.public()
        roots = list(self.b['volumes']) + ['catalog']
        def volume(b, role):
            return b['model_storage']['catalog'] if role == 'catalog' else b['volumes'][role]
        cases = []
        # Every original string must be canonical, even when disjoint.
        for role in roots:
            path = volume(self.b, role)['Mountpoint']
            for alias in ('/'+path, path+'/', path+'/./leaf', path+'//leaf',
                          path+'/../leaf', path.lstrip('/'), '/', '///root'):
                cases.append((role, alias))
        # Equal and ancestor aliases across private, model and catalog roots,
        # in both directions, must reach the actual enrollment parsers.
        for target in ('control', 'models', 'catalog'):
            for source in ('control', 'models', 'catalog'):
                if target == source: continue
                path = volume(self.b, source)['Mountpoint']
                for alias in ('/'+path, '/'+path+'/nested', '//opaque-store'):
                    cases.append((target, alias))
        for role, alias in cases:
            bad = copy.deepcopy(public)
            volume(bad['launch'], role)['Mountpoint'] = alias
            before = copy.deepcopy(bad)
            for parser, value in ((qd.binding, bad['launch']),
                                  (ip.installation, bad), (qd.installation, bad)):
                with self.subTest(role=role, alias=alias, parser=parser.__qualname__):
                    with self.assertRaisesRegex(ValueError, '^(daemon_volume_alias|daemon_fresh_local_volume_required)$'):
                        parser(value)
                    self.assertEqual(bad, before)
        self.assertEqual(qd.binding(self.b), self.b)
        self.assertEqual(ip.installation(public), public)
        self.assertEqual(qd.installation(public), public)

    def public(self, config=None):
        config=self.config if config is None else config
        return dict(deployment_id=self.b['installation'],revision=1,scope=ip.QWEN_SCOPE,
            host=dv.scope(self.b,family=qd.FAMILY),configuration=config,
            configuration_digest=dv.digest(config),ownership_policy=qd.POLICY,
            roles={r:dict(device=1,inode=i+1,uid=ip.expectation(ip.QWEN_SPLIT,r,directory=True)[0],
                gid=ip.expectation(ip.QWEN_SPLIT,r,directory=True)[1],mode=ip.expectation(ip.QWEN_SPLIT,r,directory=True)[2])
                for i,r in enumerate(('control','staging','receipts','ipc'))},launch=copy.deepcopy(self.b))

    def test_real_enrollment_detached_copy_and_join(self):
        public=self.public()
        self.assertEqual(ip.installation(public), public)
        self.assertEqual(qd.installation(public), public)
        detached=qd.binding(self.b);detached['model_storage']['catalog']['Labels'].clear()
        qd.binding(self.b)
        bad=copy.deepcopy(public);bad['launch']['model_storage']['reference_id']='f'*32
        for parser in (ip.installation,qd.installation):
            with self.assertRaises(ValueError):parser(bad)
        # A valid legacy bundle ID is inconsistent with shared K, though every
        # configuration digest is corroborated. Both enrollment parsers refuse.
        import json
        config=json.loads(json.dumps(self.config).replace(qd.MANIFEST_SHA256,'legacy-bundle'))
        # Restore manifest pins after replacing the ID only.
        def restore(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k in ('manifest_sha256','bundle_sha256') and x=='legacy-bundle':v[k]=qd.MANIFEST_SHA256
                    else:restore(x)
            elif isinstance(v,list):
                for x in v:restore(x)
        restore(config)
        qd.configuration(config)
        self.b['configuration_digest']=dv.digest(config)
        bad=self.public(config)
        with self.assertRaisesRegex(ValueError,'shared_configuration_binding'):qd.installation(bad)
        with self.assertRaisesRegex(ValueError,'shared_configuration_binding'):
            dv.command('control',config,self.b,family=qd.FAMILY)
        with self.assertRaisesRegex(ValueError,'shared_configuration_binding'):ip.installation(bad)

    def test_legacy_enrollment_bytes_and_no_launch(self):
        import json
        self.b = binding(self.fixture.config)
        self.b['policy'] = qd.LAUNCH
        public = self.public(self.fixture.config)
        before = json.dumps(public, sort_keys=True, separators=(',', ':'))
        for parser in (ip.installation, qd.installation):
            self.assertEqual(json.dumps(parser(public), sort_keys=True, separators=(',', ':')), before)
        # Existing host enrollment has no daemon launch to join.
        del public['launch']
        self.assertEqual(ip.installation(public), public)

    def test_configuration_digest_refusal_precedes_join(self):
        public = self.public()
        public['configuration_digest'] = public['launch']['configuration_digest'] = 'f'*64
        with self.assertRaisesRegex(ValueError, '^installed_configuration_digest$'):
            ip.installation(public)
