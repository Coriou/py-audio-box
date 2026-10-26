"""Closed Docker-volume launch policy. No Docker execution or host PID inference.

Client metadata is trusted operator/daemon input, not administrator-resistant
attestation. Selection of POLICY is explicit in packaged entrypoints.
"""
import hashlib
import json
import re

POLICY = 'installed-daemon-volumes-v1'
EVIDENCE = 'daemon-volumes-inspection-v1'
ROLES = ('control', 'receipts', 'staging', 'ipc', 'authority', 'models', 'config', 'credential')
ROOTS = {r: '/'+r for r in ROLES}
ROOTS.update(config='/etc/speech-box/config', credential='/etc/speech-box/credentials')
CONFIG = ROOTS['config']+'/install.json'
CREDENTIAL = ROOTS['credential']+'/control.key'
INIT_CAPS = ('CHOWN', 'FOWNER', 'DAC_OVERRIDE', 'SETUID', 'SETGID', 'KILL')
WORKER_CAPS = ('SETUID', 'SETGID', 'KILL')
# Later helpers still read UID10001's private Core under exclusion. DAC bypass
# is confined to these finite trusted narrow-volume helpers, never the worker.
LATER_CAPS = ('DAC_OVERRIDE', *WORKER_CAPS)
HEX = re.compile('[0-9a-f]{64}')
IMAGE = re.compile('sha256:[0-9a-f]{64}')
PHASES = ('install', 'init', 'up', 'status', 'stop')
HELPER_ROLES = dict(install=('authority','models','config','credential'),
    init=('authority','control','receipts','staging','ipc'),
    up=('authority','control','receipts','staging','ipc'),
    status=('authority','control','receipts','staging','ipc'),
    stop=('authority','control','receipts','staging','ipc'))


def closed(value, fields):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError('daemon_closed_record')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def daemon(value):
    closed(value, 'endpoint id os architecture security cgroup')
    if (not isinstance(value['endpoint'],str) or len(value['endpoint'])>256 or not re.fullmatch(r'unix:///[^\s\x00]+',value['endpoint']) or
        not isinstance(value['id'],str) or not 1 <= len(value['id']) <= 256 or
        value['os']!='linux' or value['architecture'] not in ('amd64','x86_64') or
        value['cgroup']!='2' or type(value['security']) is not list or
        any(not isinstance(s,str) for s in value['security']) or
        any('rootless' in s.lower() or 'userns' in s.lower() for s in value['security'])):
        raise ValueError('daemon_local_rootful_linux_amd64_required')
    return value


def volume(value, role, installation):
    closed(value, 'Name Driver Options Scope Mountpoint CreatedAt Labels')
    if (not re.fullmatch('pab-'+installation+'-'+role,value['Name']) or value['Driver']!='local' or
        value['Options'] not in (None,{}) or value['Scope']!='local' or
        not isinstance(value['Mountpoint'],str) or not value['Mountpoint'].startswith('/') or
        not isinstance(value['CreatedAt'],str) or not 1<=len(value['CreatedAt'])<=128 or
        value['Labels']!={'speech-box.installation':installation,'speech-box.role':role}):
        raise ValueError('daemon_fresh_local_volume_required')
    return value


def binding(value, *, family=None):
    if family is not None:
        from service import installed_export as ex, installed_pocket_daemon as pd
        from service import installed_qwen_daemon as qd
        if family is qd.FAMILY: return qd.binding(value)
        if family is pd.FAMILY: return pd.binding(value)
        if family is not ex.FAMILY: raise ValueError('explicit_export_family_required')
        return ex.binding(value)
    closed(value, 'policy installation daemon volumes helper_image configuration_digest cpu tmpfs control_port')
    if value['policy']!=POLICY or not re.fullmatch('[0-9a-f]{32}',value['installation']):
        raise ValueError('daemon_new_installation_policy')
    daemon(value['daemon']); closed(value['volumes'],' '.join(ROLES))
    paths=[]
    for role,v in value['volumes'].items():
        volume(v,role,value['installation']); paths.append(v['Mountpoint'].rstrip('/'))
    if len(set(paths))!=8 or any(a.startswith(b+'/') for a in paths for b in paths if a!=b):
        raise ValueError('daemon_volume_alias')
    if not IMAGE.fullmatch(value['helper_image']) or not HEX.fullmatch(value['configuration_digest']):
        raise ValueError('daemon_image_configuration_binding')
    closed(value['cpu'],'control worker'); closed(value['tmpfs'],'control worker')
    for role in ('control','worker'):
        if type(value['cpu'][role]) is not int or not 1<=value['cpu'][role]<=1024*10**9:
            raise ValueError('daemon_cpu_budget')
        t=value['tmpfs'][role]; closed(t,'bytes inodes')
        if type(t['bytes']) is not int or not 65536<=t['bytes']<=1024**3 or type(t['inodes']) is not int or not 16<=t['inodes']<=65536:
            raise ValueError('daemon_tmpfs_budget')
    if type(value['control_port']) is not int or not 1<=value['control_port']<=65535:
        raise ValueError('daemon_port_budget')
    return value


def scope(value, *, family=None):
    binding(value, family=family)
    return dict(daemon=value['daemon']['id'],context=value['daemon']['endpoint'],volumes={r:dict(
        name=value['volumes'][r]['Name'],driver='local',created=value['volumes'][r]['CreatedAt'])
        for r in ('control','receipts','staging','ipc')})


def mounts(role, b, phase=None, *, family=None):
    binding(b, family=family)
    if role=='helper': names=HELPER_ROLES[phase]
    elif role=='control': names=('control','ipc','staging','models','config','credential')
    elif role=='worker': names=('receipts','ipc','staging','models','config')
    else: raise ValueError('daemon_role')
    writable=set(names) if role=='helper' else ({'control'} if role=='control' else {'receipts','ipc','staging'})
    result = [dict(Type='volume',Name=b['volumes'][r]['Name'],Source=b['volumes'][r]['Mountpoint'],
        Destination=ROOTS[r],RW=r in writable,Propagation='') for r in names]
    if family is not None and role=='worker':
        result.append(dict(Type='volume',Name=b['volumes']['control']['Name'],
            # Docker reports the volume root here, including for subpath mounts.
            # HostConfig.VolumeOptions.Subpath is independently checked exactly;
            # live local-root identities corroborate the mounted asset inode.
            Source=b['volumes']['control']['Mountpoint'],
            Destination='/references/assets',RW=False,Propagation=''))
    return result


def command(role, config, b, *, phase=None, nonce=None, family=None):
    from service import installed_pocket_daemon as pd
    binding(b, family=family)
    from service import installed_qwen_daemon as qd
    export = pd.is_export(family)
    if role=='helper' and (phase not in PHASES or not re.fullmatch('[0-9a-f]{32}',nonce or '')):
        raise ValueError('daemon_helper_phase')
    if family is not None:
        from service import install_policy as ip
        if export:
            from service import installed_export as ex
        ip.family_configuration(config, family)
        if digest(config)!=b['configuration_digest']: raise ValueError('daemon_configuration_binding')
    helper=role=='helper'; budget=config['budget']; t=b['tmpfs']['control' if helper else role]
    args=['container','create','--pull=never','--restart=no','--read-only','--user',
        '0:0' if helper else ('10001:10001' if role=='control' else '0:10001'),
        '--cap-drop=ALL','--security-opt=no-new-privileges:true','--ipc=private','--cgroupns=private',
        '--network','bridge' if role=='control' else 'none','--memory',str(budget['ram_bytes']),
        '--memory-swap',str(budget['ram_bytes']),'--pids-limit',str(budget['pids']),
        '--cpus',format(b['cpu']['control' if helper else role]/10**9,'.9f'),
        '--tmpfs','/tmp:rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777']
    caps=(INIT_CAPS if phase in ('install','init') else LATER_CAPS) if helper else (() if role=='control' else WORKER_CAPS)
    for cap in caps: args+=['--cap-add='+cap]
    if helper: args+=['--interactive','--attach=stdin','--attach=stdout']
    if role=='control': args+=['--publish','127.0.0.1:'+str(b['control_port'])+':8000']
    for m in mounts(role,b,phase,family=family):
        args+=['--mount','type=volume,source='+m['Name']+',destination='+m['Destination']+',volume-nocopy'+('' if m['RW'] else ',readonly')+(',volume-subpath=core/assets' if m['Destination']=='/references/assets' else '')]
    if helper:
        code=('from service.daemon_helper import qwen_main;qwen_main()' if family is qd.FAMILY else 'from service.daemon_helper import pocket_main;pocket_main()' if family is pd.FAMILY else 'from service.daemon_helper import export_main;export_main()' if export else 'from service.daemon_helper import main;main()'); extra=[phase,nonce]
    elif role=='control':
        code=('from service.entrypoint import daemon_qwen_main;daemon_qwen_main()' if family is qd.FAMILY else 'from service.entrypoint import daemon_pocket_main;daemon_pocket_main()' if family is pd.FAMILY else 'from service.entrypoint import daemon_export_main;daemon_export_main()' if export else 'from service.entrypoint import daemon_main;daemon_main()'); extra=[]
    else:
        code=('from worker.runtime_supervisor import qwen_daemon_main;qwen_daemon_main()' if family is qd.FAMILY else 'from worker.runtime_supervisor import pocket_daemon_main;pocket_daemon_main()' if family is pd.FAMILY else 'from worker.runtime_supervisor import main;main()')
        extra=['--installed-policy',qd.POLICY if family is qd.FAMILY else pd.POLICY if family is pd.FAMILY else ex.POLICY if export else 'docker-kokoro-split-v1','--configuration-file',CONFIG]
        if export: extra+=['--expected-worker-image',config['images']['worker']]
    args += ['--entrypoint','python',b['helper_image'] if helper else config['images'][role],'-I','-B','-c',
        "import sys;sys.path.insert(0,'/opt/speech');"+code]+extra
    return args


def inspect_container(c, role, config, b, *, phase=None, nonce=None, family=None):
    """Validate full trusted Docker inspection, including absence of extra mounts."""
    binding(b, family=family)
    expected=command(role,config,b,phase=phase,nonce=nonce,family=family)
    hc=c['HostConfig']; cc=c['Config']; state=c['State']
    helper=role=='helper'; t=b['tmpfs']['control' if helper else role]
    caps=list((INIT_CAPS if phase in ('install','init') else LATER_CAPS) if helper else (() if role=='control' else WORKER_CAPS))
    actual_caps = hc['CapAdd']
    if actual_caps is None:
        actual_caps = []
    if type(actual_caps) is not list or any(type(cap) is not str for cap in actual_caps):
        raise ValueError('daemon_container_security_binding')
    # Docker canonicalizes Linux capabilities with CAP_; older responses and
    # fixtures use the bare kernel name. Both still match the same closed set.
    actual_caps = [cap.removeprefix('CAP_') for cap in actual_caps]
    if (not HEX.fullmatch(c['Id']) or c['Image']!=(b['helper_image'] if helper else config['images'][role]) or
        cc['User']!=('0:0' if helper else '10001:10001' if role=='control' else '0:10001') or
        cc['Entrypoint']!=['python'] or cc['Cmd']!=expected[expected.index('--entrypoint')+3:] or
        hc['Privileged'] is not False or hc['ReadonlyRootfs'] is not True or (hc.get('Init') is not None and hc.get('Init') is not False) or
        hc['PidMode'] not in ('',None) or hc['IpcMode']!='private' or hc['CgroupnsMode']!='private' or
        hc['UsernsMode'] not in ('',None) or hc['NetworkMode']!=('bridge' if role=='control' else 'none') or
        hc['CapDrop']!=['ALL'] or sorted(actual_caps)!=sorted(caps) or
        hc['SecurityOpt']!=['no-new-privileges:true'] or hc['RestartPolicy']!={'Name':'no','MaximumRetryCount':0} or
        hc['Devices'] or hc['DeviceRequests'] or hc['Binds'] or hc.get('VolumesFrom') or hc.get('GroupAdd') or
        hc.get('DeviceCgroupRules') or hc.get('Sysctls') or hc.get('CgroupParent') or hc.get('UTSMode') or
        hc.get('CpuPeriod',0) or hc.get('CpuQuota',0) or hc.get('CpusetCpus','') or hc.get('PublishAllPorts',False) or
        hc['Memory']!=config['budget']['ram_bytes'] or hc['MemorySwap']!=config['budget']['ram_bytes'] or
        hc['PidsLimit']!=config['budget']['pids'] or hc['NanoCpus']!=b['cpu']['control' if helper else role] or
        hc['Tmpfs']!={'/tmp':'rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777'} or
        hc['PortBindings']!=({'8000/tcp':[{'HostIp':'127.0.0.1','HostPort':str(b['control_port'])}]} if role=='control' else {}) or
        state['Restarting'] is not False or c['RestartCount']!=0):
        raise ValueError('daemon_container_security_binding')
    if (type(state['Running']) is not bool or type(state['ExitCode']) is not int or state['Status'] not in ('created','running','exited') or
        state.get('OOMKilled',False) or state.get('Dead',False) or state.get('Error','')):
        raise ValueError('daemon_container_state')
    if state['Status']!='created':
        from datetime import datetime
        started=state['StartedAt']
        if not isinstance(started,str) or not started.endswith('Z') or len(started)>64:
            raise ValueError('daemon_container_started_at')
        datetime.fromisoformat(started.replace('Z','+00:00'))
        if started.startswith('0001-'): raise ValueError('daemon_container_started_at')
    requested=hc.get('Mounts')
    if type(requested) is not list or len(requested)!=len(mounts(role,b,phase,family=family)):
        raise ValueError('daemon_explicit_volume_mounts_required')
    projected=[]
    for m in requested:
        if family is not None:
            options=dict(NoCopy=True,**({'Subpath':'core/assets'} if role=='worker' and m.get('Target')=='/references/assets' else {}))
            if digest(m.get('VolumeOptions'))!=digest(options) or type(m.get('ReadOnly',False)) is not bool:
                raise ValueError('daemon_export_exact_mount_options')
        if (set(m)-{'Type','Source','Target','ReadOnly','Consistency','VolumeOptions'} or
            m.get('VolumeOptions')!=dict(NoCopy=True,**({'Subpath':'core/assets'} if family is not None and role=='worker' and m.get('Target')=='/references/assets' else {})) or m.get('Consistency','') not in ('','default')):
            raise ValueError('daemon_no_copyup_mount_required')
        projected.append(dict(Type=m['Type'],Source=m['Source'],Target=m['Target'],ReadOnly=m.get('ReadOnly',False)))
    expected_mounts=[dict(Type='volume',Source=m['Name'],Target=m['Destination'],ReadOnly=not m['RW']) for m in mounts(role,b,phase,family=family)]
    if sorted(projected,key=lambda m:m['Target'])!=sorted(expected_mounts,key=lambda m:m['Target']):
        raise ValueError('daemon_requested_mount_binding')
    # Reject source/path aliases as well as extra or changed role mounts.
    if family is not None and any(type(m.get('RW')) is not bool for m in c['Mounts']):
        raise ValueError('daemon_export_exact_mount_options')
    actual=[{k:m.get(k,'') for k in ('Type','Name','Source','Destination','RW','Propagation')} for m in c['Mounts']]
    if sorted(actual,key=lambda m:m['Destination'])!=sorted(mounts(role,b,phase,family=family),key=lambda m:m['Destination']):
        raise ValueError('daemon_container_mount_binding')
    return c


def local(value, role, *, family=None):
    if family is not None:
        from service import installed_pocket_daemon as pd
        pd.require(family)
    closed(value,'role pid start_ticks pid_namespace cgroup uid gid groups capabilities roots')
    expected={'control':(10001,10001,0),'worker':(0,10001,(1<<5)|(1<<6)|(1<<7))}[role]
    if (any(type(value[k]) is not int for k in ('uid','gid','capabilities')) or
        type(value['groups']) is not list or any(type(g) is not int for g in value['groups']) or value['role']!=role or value['pid']!=1 or type(value['pid']) is not int or
        any(type(value[k]) is not int or not 1<=value[k]<2**64 for k in ('start_ticks','pid_namespace')) or
        value['cgroup']!='0::/' or (value['uid'],value['gid'],value['capabilities'])!=expected or
        value['groups'] not in ([],[expected[1]])):
        raise ValueError('daemon_container_local_identity')
    names=('control','ipc','staging','models','config','credential') if role=='control' else ('receipts','ipc','staging','models','config')
    if family is not None: names=(*names,'assets')
    closed(value['roots'],' '.join(names))
    for pair in value['roots'].values():
        if type(pair) is not list or len(pair)!=2 or any(type(n) is not int or n<1 for n in pair):
            raise ValueError('daemon_container_local_roots')
    if len({tuple(p) for p in value['roots'].values()})!=len(names): raise ValueError('daemon_container_local_alias')
    return value


def observation(value, registration, config, b, public, *, family=None):
    from service import installed_export as ex
    binding(b, family=family)
    evidence=ex.EVIDENCE if family is ex.FAMILY else EVIDENCE
    closed(value,'kind binding inspections local hello')
    if value['kind']!=evidence or value['binding']!=digest(b) or registration['evidence']!=evidence or public['launch']!=b:
        raise ValueError('daemon_evidence_binding')
    closed(value['inspections'],'control worker'); closed(value['local'],'control worker')
    from service.host_install import _registration_hello
    _registration_hello(value['hello'],registration,family=family,configuration=config if family is not None else None)
    for role in ('control','worker'):
        c=inspect_container(value['inspections'][role],role,config,b,family=family); facts=local(value['local'][role],role,family=family)
        owner=next(c for c in registration['containers'] if c['role']==role)
        if (owner['id']!=c['Id'] or owner['image']!=c['Image'] or owner['started']!=c['State']['StartedAt'] or c['State']['Running'] is not True):
            raise ValueError('daemon_evidence_incarnation')
        for r,record in public['roles'].items():
            if r in facts['roots'] and facts['roots'][r]!=[record['device'],record['inode']]:
                raise ValueError('daemon_evidence_role_identity')
    for r in (('ipc','staging','models','config','assets') if family is not None else ('ipc','staging','models','config')):
        if value['local']['control']['roots'][r]!=value['local']['worker']['roots'][r]:
            raise ValueError('daemon_evidence_cross_view_identity')
    return value
