"""Actual coordinator/client/helper code, inert fake Docker metadata only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import unittest
from service import shared_models as sm, shared_model_helper as sh, shared_model_client as sc
from service import package_install, installed_qwen_daemon as qd
from tests.service.test_shared_models import fixture, metadata, ownership, IMAGE, DAEMON, new_bundle, SharedModelCase


class FakeDocker:
    def __init__(self,root):
        self.root=Path(root);self.frozen=copy.deepcopy(DAEMON);self.volumes={};self.containers={};self.retained=[]
        self.env={};self.allocations=0;self.started=[];self.end=time.monotonic()+20
    def check(self): pass
    def container(self,identity): return copy.deepcopy(self.containers[identity])
    def json(self,args):
        assert args[:2]==['volume','inspect']
        return [copy.deepcopy(self.volumes[args[2]])]
    def call(self,args):
        if args[:2]==['image','inspect']: return IMAGE
        if args[:2]==['volume','ls']: return '\n'.join(self.volumes)
        if args[:2]==['container','ls']: return '\n'.join(self.containers)
        if args[:2]==['volume','create']:
            name=args[-1];path=self.root/name;path.mkdir()
            labels=dict(args[i+1].split('=',1) for i,a in enumerate(args) if a=='--label')
            key=labels.get('speech-box.manifest');attempt=labels.get('speech-box.attempt')
            self.volumes[name]=metadata(path,key,attempt)
            if key:self.allocations+=1
            return name
        if args[:2]==['container','create']:
            identity=format(len(self.containers)+1,'064x')
            c=container_from_command(args,identity,self.volumes)
            self.containers[identity]=c;return identity
        raise AssertionError(args)


def container_from_command(args,identity,volumes):
    i=args.index('--entrypoint');image=args[i+2]
    requested=[];actual=[]
    for n,arg in enumerate(args):
        if arg!='--mount':continue
        bits=args[n+1].split(',');v=volumes[bits[1].split('=',1)[1]];target=bits[2].split('=',1)[1];rw='readonly' not in bits
        requested.append(dict(Type='volume',Source=v['Name'],Target=target,ReadOnly=not rw,VolumeOptions={'NoCopy':True}))
        actual.append(dict(Type='volume',Name=v['Name'],Source=v['Mountpoint'],Destination=target,RW=rw,Propagation=''))
    return dict(Id=identity,Image=image,RestartCount=0,
        Config=dict(User='0:0',Entrypoint=['python'],Cmd=args[i+3:]),
        HostConfig=dict(Privileged=False,ReadonlyRootfs=True,Init=False,PidMode='',IpcMode='private',CgroupnsMode='private',UsernsMode='',
            NetworkMode='none',CapDrop=['ALL'],CapAdd=[a.split('=',1)[1] for a in args if a.startswith('--cap-add=')],
            SecurityOpt=['no-new-privileges:true'],RestartPolicy={'Name':'no','MaximumRetryCount':0},
            Devices=[],DeviceRequests=[],Binds=None,Memory=268435456,MemorySwap=268435456,PidsLimit=32,NanoCpus=10**9,
            Tmpfs={'/tmp':sh.TMPFS},PortBindings={},Mounts=requested),
        Mounts=actual,State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',Pid=0,OOMKilled=False,Dead=False,Error='',
            StartedAt='0001-01-01T00:00:00Z',FinishedAt='0001-01-01T00:00:00Z'))


def exit_container(c,code=0):
    c['State'].update(Running=False,ExitCode=code,Status='exited',Pid=0,
        StartedAt='2026-10-27T10:00:00Z',FinishedAt='2026-10-27T10:00:01Z')


class LocalClient(sc.Client):
    """Only process/mount boundary injected; real controller/import/hash/RPC."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.bytes=0;self.fault=lambda _:None
    def input(self,payload):
        value=super().input(payload)
        import base64
        self.bytes+=len(base64.b64decode(value['data']));return value
    def exchange(self,identity,role,nonce,operation,candidate=None,attempt=None):
        self.check();c=self.docker.containers[identity]
        sh.inspect(c,role,self.image,nonce,self.catalog,candidate)
        self.docker.started.append(identity)
        c['State'].update(Running=True,Status='running',Pid=123,StartedAt='2026-10-27T10:00:00Z')
        try:
            if role=='controller':
                catalog=sm.Catalog(self.catalog['Mountpoint'],self.docker.frozen,self.catalog,self.end)
                controller=sh.Controller(catalog,self.action,self.image,self.expected,fault=self.fault)
                result=catalog.init() if operation=='init' else controller.coordinate(operation,self.selection) if operation in ('reserve','bind','attachment-intent','attachment-created') else controller.run(operation,attempt)
            elif role=='writer':
                import base64
                result=sh.import_tree(candidate['Mountpoint'],self.expected,
                    lambda name,offset:base64.b64decode(self.action('input',dict(path=name,offset=offset))['data']),self.end,fault=self.fault)
            else: result=sm.verify_tree(candidate['Mountpoint'],self.expected,self.end)
        except BaseException:
            exit_container(c,1);raise
        exit_container(c)
        sh.inspect(c,role,self.image,nonce,self.catalog,candidate,terminal=True)
        return dict(inspection=copy.deepcopy(c),result=result)


def client(root,docker,expected,data,index):
    output=root/('receipts-'+str(index));output.mkdir()
    key=sm.descriptor(expected)[0];sources={}
    source=root/'source'
    if not source.exists():
        source.mkdir()
        for name,raw in data.items():
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    for name in data:sources['bundles/'+key+'/'+name]=str(source/name)
    return LocalClient(docker,IMAGE,output,time.monotonic()+20,time.time()+20,expected,sources)


class SharedModelTransportTests(SharedModelCase):
    def test_import_terminal_writer_independent_ro_reuse(self):
        bundle=self.bundle
        expected,data,key,root=bundle
        docker=FakeDocker(root);c=client(root,docker,expected,data,1)
        c.run('init');published=c.run('import')
        assert docker.allocations==1 and c.bytes==sum(map(len,data.values()))
        assert published['writer']['id']!=published['verifier']['id']
        writer=docker.containers[published['writer']['id']];reader=docker.containers[published['verifier']['id']]
        assert writer['Mounts'][0]['RW'] is True and reader['Mounts'][0]['RW'] is False
        other=client(root,docker,expected,data,2)
        assert other.run('import')==published
        assert other.bytes==0 and docker.allocations==1
        assert other.run('verify')==published and other.bytes==0
        assert len([v for v in docker.volumes.values() if v['Name']!=sm.CATALOG])==1
        with self.assertRaisesRegex(ValueError,'fresh_allocation'):
            other.create_helper(dict(role='writer',nonce='a'*32,image=IMAGE,candidate=published['candidate']))


    def crash_case(self,bundle,point):
        expected,data,key,root=bundle;docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init')
        def fail(where):
            if where==point:raise ValueError('injected_crash')
        c.fault=fail
        with self.assertRaisesRegex(ValueError,'injected_crash'):c.run('import')
        catalog=sm.Catalog(c.catalog['Mountpoint'],DAEMON,c.catalog,time.monotonic()+20)
        with catalog.held(key) as store:
            attempts=store.attempts();assert len(attempts)==1;attempt=attempts[0]
            before=store.publication()
        other=client(root,docker,expected,data,2)
        if point in ('candidate_intent','candidate_receipt','writer_created','verifier_created','stream_chunk','file_sealed','tree_sealed'):
            with self.assertRaises((ValueError,FileNotFoundError)):other.run('recover-import',attempt)
            assert not list((Path(c.catalog['Mountpoint'])/'published').glob('*.json'))
        else:
            result=other.run('recover-import',attempt)
            assert result['attempt']==attempt and other.bytes==0 and docker.allocations==1
            if before is not None:assert result==before


    def tamper_case(self,bundle,mutate):
        expected,data,key,root=bundle;docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init');pub=c.run('import')
        writer=docker.containers[pub['writer']['id']];reader=docker.containers[pub['verifier']['id']]
        if mutate=='running-writer':writer['State']['Running']=True
        elif mutate=='restarted':writer['RestartCount']=1
        elif mutate=='rw-reader':reader['Mounts'][0]['RW']=True
        elif mutate=='requested-rw':reader['HostConfig']['Mounts'][0]['ReadOnly']=False
        elif mutate=='alias':reader['Mounts'].append(dict(reader['Mounts'][0],Destination='/alias'))
        elif mutate=='wrong-created':docker.volumes[pub['candidate']['Name']]['CreatedAt']='changed'
        elif mutate=='unknown':docker.containers['f'*64]=dict(copy.deepcopy(reader),Id='f'*64)
        elif mutate=='oom':writer['State']['OOMKilled']=True
        elif mutate=='nonzero':writer['State']['ExitCode']=1
        else:reader['Config']['Cmd'][-1]='f'*32
        other=client(root,docker,expected,data,2)
        with self.assertRaises(ValueError):other.run('import')
        assert other.bytes==0 and docker.allocations==1


    def test_transport_allowlist_and_source_mutation(self):
        bundle=self.bundle
        expected,data,key,root=bundle;docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init')
        for action in ('start','term','delete','verify','exec'):
            with self.assertRaises(ValueError):c.action(action,{})
        with self.assertRaises(ValueError):c.action('input',dict(path='/etc/passwd',offset=0))
        path=next(iter(c.sources));Path(c.sources[path]).write_bytes(b'x'*len(data[path.split('/',2)[2]]))
        with self.assertRaisesRegex(ValueError,'source_changed'):c.input(dict(path=path,offset=0))


    def test_cli_dispatch_help_and_stdlib_import(self):
        with patch.object(sc,'main',return_value='route') as command:
            assert package_install.main(['docker-models','verify'])=='route';command.assert_called_once_with(['verify'])
        code="import sys;sys.path.insert(0,sys.argv[1]);from service.package_install import main;main(['docker-models','--help'])"
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2])],capture_output=True,text=True,timeout=20)
        assert result.returncode==0 and 'recover-import' in result.stdout


    def test_real_process_key_exclusion_timeout_and_other_key(self):
        bundle=self.bundle
        expected,data,key,root=bundle
        catalog_root=root/'catalog';catalog_root.mkdir();meta=metadata(catalog_root)
        catalog=sm.Catalog(catalog_root,DAEMON,meta,time.monotonic()+20);catalog.init()
        # Child reports readiness before flock. Parent holds same key; a different
        # key succeeds while the same-key child reaches a finite timeout. No sleeps.
        code=r'''
import json,os,sys,time
sys.path.insert(0,sys.argv[1])
from service import shared_models as sm
from tests.service.test_shared_models import ownership
path,daemon,meta,key=json.loads(sys.argv[2])
with ownership():
    c=sm.Catalog(path,daemon,meta,time.monotonic()+.4)
    print('ready',flush=True)
    sys.stdin.readline()
    try:
        with c.held(key): print('acquired',flush=True)
    except ValueError: print('timeout',flush=True)
'''
        with catalog.held(key):
            for selected,want in ((key,'timeout'),('f'*64,'acquired')):
                p=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),json.dumps([str(catalog_root),DAEMON,meta,selected])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                assert p.stdout.readline().strip()=='ready'
                output,error=p.communicate('\n',timeout=10)
                assert p.returncode==0,error
                assert output.strip()==want
        with catalog.held(key):pass

    def test_crash_matrix(self):
        for point in ('candidate_intent','candidate_receipt','writer_created','stream_chunk','file_sealed','tree_sealed','writer_terminal','verifier_created','verifier_terminal','publication_temp_fsync','publication_rename','publication_parent_fsync'):
            with self.subTest(point=point),new_bundle() as bundle:
                self.crash_case(bundle,point)

    def test_adversarial_consumers(self):
        for mutate in ('running-writer','restarted','rw-reader','requested-rw','alias','wrong-created','unknown','oom','nonzero','nonce'):
            with self.subTest(mutate=mutate),new_bundle() as bundle:
                self.tamper_case(bundle,mutate)

    def test_full_client_wire_and_helper_entrypoints(self):
        expected,data,key,root=self.bundle
        class ProcessDocker(FakeDocker):
            def argv(inner,args):
                assert args[:4]==['container','start','--attach','--interactive']
                identity=args[4];c=inner.containers[identity]
                role,nonce=c['Config']['Cmd'][-2:]
                candidate=None if role=='controller' else inner.volumes[c['Mounts'][0]['Name']]
                catalog=inner.volumes[sm.CATALOG]
                code=r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from service import shared_models as sm,shared_model_helper as sh,installed_qwen_daemon as qd
from tests.service.test_shared_models import ownership
root,catalog,key,role,nonce=json.loads(sys.argv[2])
original_catalog=sm.Catalog
sm.Catalog=lambda path,*args,**kwargs:original_catalog(catalog,*args,**kwargs)
if root is not None: sm.MODEL_ROOT=root
qd.MANIFEST_SHA256=key
sh.mount_policy=lambda path,readonly:None
sys.argv=['helper',role,nonce]
with ownership(): sh.main()
'''
                # Docker terminal observations are injected; Client still must
                # drain exact EOF and reap the real helper before inspecting it.
                exit_container(c)
                return [sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),
                    json.dumps([None if candidate is None else candidate['Mountpoint'],catalog['Mountpoint'],key,role,nonce])]
        docker=ProcessDocker(root)
        inert=client(root,docker,expected,data,1)
        c=sc.Client(docker,IMAGE,inert.directory,time.monotonic()+30,time.time()+30,expected,inert.sources)
        c.run('init');pub=c.run('import')
        assert pub['key']==key and docker.allocations==1 and not docker.retained
        assert c.run('verify')==pub and docker.allocations==1 and not docker.retained
        # Exercise the new coordinator vocabulary over the actual Client/Wire
        # and real helper entrypoint, including protected reference writes.
        from tests.service.test_installed_qwen import InstalledQwenTests
        from tests.service.test_shared_model_binding import shared,container
        f=InstalledQwenTests(); f.setUp(); self.addCleanup(f.doCleanups)
        old=next(iter(f.config['bundles'])); old_hash=f.config['contexts']['alice']['execution']['descriptor']['manifest_sha256']
        def rename(v):
            if isinstance(v,dict): return {(key if k==old else k):rename(x) for k,x in v.items()}
            if isinstance(v,list): return [rename(x) for x in v]
            if isinstance(v,str): return key if v in (old,old_hash) else v.replace('/bundles/'+old,'/bundles/'+key)
            return v
        config=rename(f.config)
        with patch.object(qd,'MANIFEST_SHA256',key):
            assert c.coordinate('reserve',dict(installation='a'*32,configuration=config,catalog=pub['catalog'],publication_sha256=sm.digest(pub)))==pub
            b=shared(config); b['daemon']=docker.frozen; b['volumes']['models']=pub['candidate']
            b['model_storage'].update(catalog=pub['catalog'],publication_sha256=sm.digest(pub))
            assert c.coordinate('bind',dict(binding=b,configuration=config))==dict(binding=sm.digest(b))
            nonce='d'*32
            c.coordinate('attachment-intent',dict(binding=b,configuration=config,role='helper',phase='install',nonce=nonce))
            identity=format(len(docker.containers)+1,'064x')
            own=container('helper',config,b,phase='install',nonce=nonce,identity=identity,family=qd.FAMILY)
            docker.containers[identity]=own
            assert c.coordinate('attachment-created',dict(binding=b,nonce=nonce,inspection=own))==dict(id=identity)
        assert docker.allocations==1 and not docker.retained

    def test_two_process_import_and_bootstrap_race(self):
        expected,data,key,root=self.bundle
        # Persistent fake daemon state is touched only by the real key-lock
        # owner. Both clients bootstrap against the same precreated root inode.
        catalog_root=root/'catalog';catalog_root.mkdir();meta=metadata(catalog_root)
        state=root/'fake-state.json';state.write_text(json.dumps(dict(volumes={sm.CATALOG:meta},containers={},allocations=0,bytes=0)))
        code=r'''
import json,sys,time,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from service import shared_models as sm,shared_model_helper as sh,installed_qwen_daemon as qd
from tests.service.test_shared_models import ownership,fixture,metadata,IMAGE,DAEMON
from tests.service.test_shared_model_transport import FakeDocker,LocalClient,exit_container
root=Path(sys.argv[2]); catalog_root=root/'catalog';meta=metadata(catalog_root)
expected,data=fixture();key=sm.descriptor(expected)[0]
class RPC:
    def __init__(self):self.transferred=0
    def __call__(self,action,payload):
        state=json.loads((root/'fake-state.json').read_text())
        docker=FakeDocker(root);docker.volumes=state['volumes'];docker.containers=state['containers'];docker.allocations=state['allocations']
        # Build a real Client to exercise its fixed action grammar. Each client
        # retains exact newly-created IDs locally; persisted daemon state is read
        # under the Controller's actual OS lock.
        if not hasattr(self,'client'):
            out=root/('client-'+str(os.getpid()));out.mkdir()
            sources={}
            for name in data:sources['bundles/'+key+'/'+name]=str(root/'source'/name)
            self.client=LocalClient(docker,IMAGE,out,time.monotonic()+20,time.time()+20,expected,sources)
        c=self.client;c.docker=docker;c.catalog=meta
        # Controller itself runs in this child; allocate/run actions remain the
        # real ordinary-client code with inert helper process boundary only.
        result=c.action(action,payload)
        state.update(volumes=docker.volumes,containers=docker.containers,allocations=docker.allocations,bytes=state['bytes']+c.bytes)
        self.transferred+=c.bytes;c.bytes=0
        (root/'fake-state.json').write_text(json.dumps(state))
        return result
with ownership():
    qd.MANIFEST_SHA256=key
    print('ready',flush=True);sys.stdin.readline()
    catalog=sm.Catalog(catalog_root,DAEMON,meta,time.monotonic()+20)
    catalog.init()
    rpc=RPC();result=sh.Controller(catalog,rpc,IMAGE,expected).run('import')
    print(json.dumps(dict(key=result['key'],candidate=result['candidate'],bytes=rpc.transferred)),flush=True)
'''
        source=root/'source';source.mkdir()
        for name,raw in data.items():
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        children=[]
        try:
            for _ in range(2):
                p=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),str(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                children.append(p)
                assert p.stdout.readline().strip()=='ready'
            for p in children:p.stdin.write('\n');p.stdin.flush()
            results=[]
            for p in children:
                # communicate closes input and reaps every finite child.
                out,err=p.communicate(timeout=30)
                assert p.returncode==0,err
                results.append(json.loads(out))
            assert results[0]['candidate']==results[1]['candidate']
            assert sorted(r['bytes'] for r in results)==[0,sum(map(len,data.values()))]
            final=json.loads(state.read_text());assert final['allocations']==1
        finally:
            for p in children:
                if p.poll() is None:p.communicate(timeout=30)

    def test_partial_failed_writer_retained_new_generation(self):
        expected,data,key,root=self.bundle
        docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init')
        original=c.input
        count=0
        def broken(payload):
            nonlocal count
            count+=1
            if count==3: raise ValueError('disconnect')
            return original(payload)
        c.input=broken
        with self.assertRaisesRegex(ValueError,'disconnect'):c.run('import')
        old_names=set(docker.volumes)
        other=client(root,docker,expected,data,2)
        pub=other.run('import')
        assert docker.allocations==2 and old_names<=set(docker.volumes)
        assert pub['candidate']['Name'] not in old_names
        assert other.bytes==sum(map(len,data.values()))

    def test_malformed_publication_never_replaced(self):
        expected,data,key,root=self.bundle
        docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init');pub=c.run('import')
        final=Path(c.catalog['Mountpoint'])/'published'/(key+'.json')
        final.write_bytes(b'{"malformed":true}')
        other=client(root,docker,expected,data,2)
        with self.assertRaises(ValueError):other.run('import')
        assert final.read_bytes()==b'{"malformed":true}' and other.bytes==0 and docker.allocations==1

    def test_real_pipe_terminal_failures_never_publish_success(self):
        expected,data,key,root=self.bundle
        for mode in ('trailing','nonzero','missing','oom'):
            with self.subTest(mode=mode),new_bundle() as b:
                expected,data,key,root=b
                class FaultDocker(FakeDocker):
                    def argv(inner,args):
                        identity=args[-1];c=inner.containers[identity];role,nonce=c['Config']['Cmd'][-2:]
                        exit_container(c)
                        if mode=='oom':c['State']['OOMKilled']=True
                        code=r'''
import os,sys
sys.path.insert(0,sys.argv[1])
from service.daemon_client import Wire
role,nonce,mode=sys.argv[2:]
w=Wire(sys.stdin.buffer,sys.stdout.buffer,nonce,role)
w.receive()
if mode!='missing': w.send('complete',dict(result={}))
if mode=='trailing': os.write(sys.stdout.fileno(),b'extra')
sys.stdin.buffer.read()
raise SystemExit(2 if mode=='nonzero' else 0)
'''
                        return [sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),role,nonce,mode]
                docker=FaultDocker(root);inert=client(root,docker,expected,data,1)
                c=sc.Client(docker,IMAGE,inert.directory,time.monotonic()+10,time.time()+10,expected,inert.sources)
                with self.assertRaises(ValueError):c.run('init')
                assert not list(c.directory.glob('*-complete.json'))
                assert not docker.retained

    def test_verify_and_recover_need_no_host_payload(self):
        expected,data,key,root=self.bundle
        docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init');pub=c.run('import')
        output=root/'reader-only';output.mkdir()
        other=LocalClient(docker,IMAGE,output,time.monotonic()+20,time.time()+20,expected,None)
        assert other.run('verify')==pub and other.bytes==0
        assert other.run('recover-import',pub['attempt'])==pub and other.bytes==0
        with self.assertRaises(ValueError):other.run('recover-import','f'*32)

    def test_actual_cli_offline_routes(self):
        import contextlib
        import io
        expected,data,key,root=self.bundle
        docker=FakeDocker(root)
        prepared=client(root,docker,expected,data,'sources')
        request=root/'request.json'
        request.write_text(json.dumps(dict(descriptor=expected,sources={name.split('/',2)[2]:path for name,path in prepared.sources.items()})))
        common=['--context','inert','--image',IMAGE]
        results=[]
        with patch.object(sc,'Docker',return_value=docker),patch.object(sc,'Client',LocalClient):
            for index,operation in enumerate(('init','import','verify','recover-import')):
                argv=['docker-models',operation]+common+['--output',str(root/('cli-'+str(index)))]
                if operation!='init':argv+=['--request',str(request)]
                if operation=='recover-import':argv+=['--attempt',results[1]['attempt']]
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    package_install.main(argv)
                results.append(json.loads(output.getvalue()))
        assert results[1]==results[2]==results[3] and docker.allocations==1
        assert str(root/'source') not in json.dumps(results[1])

    def test_lost_create_responses_refuse_blind_repetition(self):
        for lost in ('allocate','create-helper'):
            with self.subTest(lost=lost),new_bundle() as b:
                expected,data,key,root=b;docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init')
                action=c.action
                def uncertain(name,payload):
                    result=action(name,payload)
                    if name==lost:raise ValueError('lost_response')
                    return result
                c.action=uncertain
                with self.assertRaisesRegex(ValueError,'lost_response'):c.run('import')
                other=client(root,docker,expected,data,2)
                with self.assertRaises((ValueError,FileNotFoundError)):other.run('import')
                assert docker.allocations==1
                assert not list((Path(c.catalog['Mountpoint'])/'published').glob('*.json'))

    def test_reused_owner_retains_custody_across_transport_faults(self):
        from service.daemon_client import Owner,collect
        for fault in ('close','selector','wait','deadline','output'):
            with self.subTest(fault=fault):
                code="import sys;sys.stdout.buffer.write(b'x'*131072)" if fault=='output' else 'pass'
                p=subprocess.Popen([sys.executable,'-I','-B','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
                retained=[];end=time.monotonic()+5
                if fault=='output':
                    with self.assertRaises(ValueError):collect(p,b'',16,end,'shared_test',retained)
                else:
                    original_wait=p.wait
                    if fault=='wait':
                        called=[False]
                        def wait(*a,**kw):
                            if not called[0]:called[0]=True;raise OSError('wait fault')
                            return original_wait(*a,**kw)
                        p.wait=wait
                    if fault=='close':
                        class CloseOnce:
                            def __init__(self,stream):self.stream=stream;self.first=True
                            @property
                            def closed(self):return self.stream.closed
                            def close(self):
                                if self.first:self.first=False;raise OSError('close fault')
                                self.stream.close()
                        p.stdin=CloseOnce(p.stdin)
                    class SelectorOnce:
                        first=True
                        def close(self):
                            if self.first:self.first=False;raise OSError('selector fault')
                    with self.assertRaises((OSError,ValueError)):
                        with Owner(p,'shared_test',time.monotonic()-1 if fault=='deadline' else end,retained) as owner:
                            if fault=='selector':owner.selectors.append(SelectorOnce())
                assert p.returncode==0 and not retained and p.stdin.closed and p.stdout.closed

    def test_missing_attempt_authority_never_silently_reconstructed(self):
        expected,data,key,root=self.bundle
        docker=FakeDocker(root);c=client(root,docker,expected,data,1);c.run('init');pub=c.run('import')
        intent=Path(c.catalog['Mountpoint'])/'attempts'/(key+'-'+pub['attempt']+'-intent.json')
        intent.rename(root/'retained-intent.json')
        other=client(root,docker,expected,data,2)
        with self.assertRaises((ValueError,FileNotFoundError)):other.run('verify')
        assert other.bytes==0 and docker.allocations==1
