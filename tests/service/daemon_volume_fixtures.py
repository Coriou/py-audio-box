"""Synthetic Docker inspections accepted by the real daemon validators."""
from service import daemon_volumes as dv


def binding(config):
    installation='a'*32
    return dict(policy=dv.POLICY,installation=installation,daemon=dict(endpoint='unix:///Users/operator/.docker/run/docker.sock',
        id='fixture-desktop-linux',os='linux',architecture='amd64',security=['name=seccomp,profile=builtin'],cgroup='2'),
        volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
            Mountpoint='/opaque-daemon/volumes/'+r,CreatedAt='2026-10-25T00:00:00Z',
            Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES},
        helper_image=config['images']['control'],configuration_digest=dv.digest(config),cpu=dict(control=10**9,
            worker=max(c['execution']['descriptor']['threads'] for context in config['contexts'].values()
                for c in (context.values() if 'execution' not in context else (context,)))*10**9),
        tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765)


def container(role,config,b,*,phase=None,nonce=None,identity=None,family=None):
    args=dv.command(role,config,b,phase=phase,nonce=nonce,family=family)
    helper=role=='helper'; t=b['tmpfs']['control' if helper else role]
    return dict(Id=identity or ('1' if role=='control' else '2')*64,Image=b['helper_image'] if helper else config['images'][role],
        Config=dict(User='0:0' if helper else '10001:10001' if role=='control' else '0:10001',Entrypoint=['python'],
            Cmd=args[args.index('--entrypoint')+3:]),
        HostConfig=dict(Privileged=False,ReadonlyRootfs=True,Init=None,PidMode='',IpcMode='private',CgroupnsMode='host' if role=='worker' and 'custody' in b else 'private',UsernsMode='',
            NetworkMode='bridge' if role=='control' else 'none',CapDrop=['ALL'],
            CapAdd=list(dv.INIT_CAPS if phase in ('install','init') else dv.LATER_CAPS) if helper else [] if role=='control' else list(dv.WORKER_CAPS),
            SecurityOpt=['no-new-privileges:true'],RestartPolicy=dict(Name='no',MaximumRetryCount=0),Devices=[],DeviceRequests=None,Binds=None,
            CgroupParent=b['custody'].removeprefix('/sys/fs/cgroup') if role=='worker' and 'custody' in b else None,
            Memory=config['budget']['ram_bytes'],MemorySwap=config['budget']['ram_bytes'],PidsLimit=config['budget']['pids'],
            Mounts=[dict(Type=m['Type'],Source=m['Source'] if m['Type']=='bind' else m['Name'],Target=m['Destination'],ReadOnly=not m['RW'],
                **(dict(BindOptions=dict(Propagation='rprivate')) if m['Type']=='bind' else dict(VolumeOptions=dict(NoCopy=True,**({'Subpath':'core/assets'} if m['Destination']=='/references/assets' else {})))))
                for m in dv.mounts(role,b,phase,family=family)],
            NanoCpus=b['cpu']['control' if helper else role],Tmpfs={'/tmp':'rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777'},
            PortBindings={'8000/tcp':[dict(HostIp='127.0.0.1',HostPort=str(b['control_port']))]} if role=='control' else {}),
        State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',StartedAt='0001-01-01T00:00:00Z',Pid=0),RestartCount=0,
        Mounts=dv.mounts(role,b,phase,family=family))
