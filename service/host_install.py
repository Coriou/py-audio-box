"""Host-only inspection and explicitly selected bounded freeze; apply unavailable."""
import argparse
import json
import os
import re
import stat
import sqlite3
from contextlib import closing, ExitStack
import uuid
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from service.core import encode
from service.persistent_init import digest, read_file, roots_checked, validate_scope
from service import recovery


class Docker:
    def __init__(self, context):
        if not isinstance(context,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',context):
            raise ValueError('docker_context_invalid')
        self.context=context
        self.started=time.monotonic()
        self.deadline=self.started+recovery.HOST_COLLECTION_SECONDS

    def _read(self, arguments):
        # Each internal caller supplies one fixed operation. Never a shell proxy.
        if time.monotonic() >= self.deadline: raise ValueError('docker_global_deadline')
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process=None
            try:
                process=subprocess.Popen(['docker','--context',self.context,*arguments],
                    shell=False,stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
                end=min(self.deadline,time.monotonic()+5)
                while process.poll() is None:
                    if (os.fstat(output.fileno()).st_size>1024*1024 or
                            os.fstat(errors.fileno()).st_size>65536 or time.monotonic()>=end):
                        raise ValueError('bounded_docker_operation_failed')
                    time.sleep(.01)
                if (process.returncode or os.fstat(output.fileno()).st_size>1024*1024 or
                        os.fstat(errors.fileno()).st_size>65536 or time.monotonic()>=end):
                    raise ValueError('bounded_docker_operation_failed')
                output.seek(0)
                return output.read(1024*1024).decode('utf-8').strip()
            except (OSError,UnicodeError):
                raise ValueError('docker_operation_unavailable') from None
            finally:
                if process is not None:
                    if process.poll() is None: process.kill()
                    try: process.wait(timeout=2)
                    except (OSError,subprocess.TimeoutExpired):
                        raise ValueError('docker_process_reap_failed') from None

    def observe(self, scope, registration):
        validate_scope(scope); recovery.validate_registration(registration)
        if self.context!=scope['context']: raise ValueError('docker_context_mismatch')
        daemon=self._read(['info','--format','{{.ID}}'])
        if daemon!=scope['daemon']: raise ValueError('docker_daemon_mismatch')
        for role,v in scope['volumes'].items():
            raw=self._read(['volume','inspect',v['name'],'--format','{{json .Name}} {{json .Driver}} {{json .CreatedAt}}'])
            decoder=json.JSONDecoder(); values=[]
            while raw:
                value,n=decoder.raw_decode(raw); values.append(value); raw=raw[n:].lstrip()
            if values!=[v['name'],v['driver'],v['created']]: raise ValueError('docker_volume_mismatch')
        ids=self._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()
        if len(ids)>128 or any(not recovery.HEX.fullmatch(i) for i in ids): raise ValueError('docker_container_budget')
        known={c['id']:c for c in registration['containers']}; observed=[]; states=[]
        names={v['name']:role for role,v in scope['volumes'].items()}
        for identity in sorted(ids):
            # Select only identity/topology fields; never Config.Env or raw receipt.
            raw=self._read(['container','inspect',identity,'--format',
                '{"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}},"running":{{json .State.Running}},"restarting":{{json .State.Restarting}},"pid":{{json .State.Pid}},"finished":{{json .State.FinishedAt}},"restart":{{json .HostConfig.RestartPolicy.Name}},"pidmode":{{json .HostConfig.PidMode}},"mounts":[{{range $i,$m := .Mounts}}{{if $i}},{{end}}{"type":{{json $m.Type}},"name":{{json $m.Name}},"destination":{{json $m.Destination}},"rw":{{json $m.RW}}}{{end}}]}'])
            value=json.loads(raw)
            mounts=[m for m in value['mounts'] if m['type']=='volume' and m['name'] in names]
            if not mounts: continue
            if identity not in known: raise ValueError('unknown_volume_consumer')
            if len(mounts)!=len(value['mounts']): raise ValueError('unexpected_container_mount')
            expected=known[identity]
            roles=sorted(names[m['name']] for m in mounts)
            if len(roles)!=len(set(roles)) or value['restart'] not in ('no','') or value['pidmode']!='': raise ValueError('unsupported_container_topology')
            for m in mounts:
                role=names[m['name']]
                if m['destination']!='/'+role or (expected['role']=='control' and role in ('ipc','staging') and m['rw']):
                    raise ValueError('container_mount_mismatch')
            actual=dict(id=identity,image=value['image'],started=value['started'],role=expected['role'],volumes=roles)
            if actual!={**expected,'volumes':sorted(expected['volumes'])}: raise ValueError('container_incarnation_mismatch')
            observed.append(expected)
            states.append({k:value[k] for k in ('id','running','restarting','pid','started','finished')})
        if {c['id'] for c in observed}!=set(known): raise ValueError('missing_registered_container')
        return dict(scope=scope,containers=registration['containers'],states=states,evidence='actual-host-inspection')



# Closed new host projection. The original B1 observe/plan representation above
# stays unchanged; none of its caller-supplied strings authenticates B2 authority.
CONTAINER_FORMAT = ('{"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}},'
    '"finished":{{json .State.FinishedAt}},"running":{{json .State.Running}},'
    '"restarting":{{json .State.Restarting}},"pid":{{json .State.Pid}},'
    '"restart":{{json .HostConfig.RestartPolicy.Name}},"pidmode":{{json .HostConfig.PidMode}},'
    '"privileged":{{json .HostConfig.Privileged}},"capadd":{{json .HostConfig.CapAdd}},'
    '"cgroupns":{{json .HostConfig.CgroupnsMode}},'
    '"mounts":[{{range $i,$m := .Mounts}}{{if $i}},{{end}}{"type":{{json $m.Type}},'
    '"name":{{json $m.Name}},"source":{{json $m.Source}},"destination":{{json $m.Destination}},'
    '"rw":{{json $m.RW}}}{{end}}]}')
VOLUME_FORMAT = ('{"name":{{json .Name}},"driver":{{json .Driver}},"created":{{json .CreatedAt}},'
    '"source":{{json .Mountpoint}},"options":{{json .Options}}}')


def _json_output(raw):
    return recovery.host_json(raw.encode())


def _volume_identity(source):
    if not isinstance(source,str) or not source.startswith('/') or str(Path(source).resolve())!=source:
        raise ValueError('host_volume_alias')
    st=os.stat(source)
    if not stat.S_ISDIR(st.st_mode): raise ValueError('host_volume_directory')
    return st.st_dev,st.st_ino


def _mount_touches(mount,volumes):
    if mount['name'] in {v['name'] for v in volumes}: return True
    source=mount['source']
    if mount['type'] not in ('bind','volume'): return False
    if not isinstance(source,str) or not source.startswith('/'): raise ValueError('host_incomplete_mount_enumeration')
    canonical=str(Path(source).resolve())
    if any(canonical==v['source'] or canonical.startswith(v['source']+'/') or v['source'].startswith(canonical.rstrip('/')+'/') for v in volumes):
        return True
    st=os.stat(source)
    if (st.st_dev,st.st_ino) in {(v['device'],v['inode']) for v in volumes}: return True
    # Neither realpath nor a different root inode proves disjointness: an
    # ancestor bind alias can reach a deployment descendant with another inode.
    # No host filesystem-topology resolver is qualified in this lot. Refuse ALL
    # remaining bind/volume sources, even a different device, before any stop.
    # This rule runs on every bounded all-container inventory, including stopped
    # containers; no source is silently excluded on incomplete OS evidence.
    raise ValueError('unsupported_host_storage_disjointness')


def _host_scope(observer,scope):
    validate_scope(scope)
    if sys.platform!='linux': raise ValueError('unsupported_host_pid_provenance')
    if observer.context!=scope['context']: raise ValueError('docker_context_mismatch')
    endpoint=_json_output(observer._read(['context','inspect',observer.context,'--format','{{json .Endpoints.docker.Host}}']))
    if endpoint!='unix:///var/run/docker.sock' or sys.platform!='linux':
        raise ValueError('unsupported_host_pid_provenance')
    info=_json_output(observer._read(['info','--format','{"id":{{json .ID}},"os":{{json .OSType}}}']))
    recovery.closed(info,'id os')
    if info!={'id':scope['daemon'],'os':'linux'}: raise ValueError('docker_daemon_mismatch')
    volumes=[]
    for role in sorted(scope['volumes']):
        value=_json_output(observer._read(['volume','inspect',scope['volumes'][role]['name'],'--format',VOLUME_FORMAT]))
        recovery.closed(value,'name driver created source options')
        if value.pop('options') not in (None,{}): raise ValueError('unsupported_volume_options')
        device,inode=_volume_identity(value['source'])
        volumes.append(dict(role=role,device=device,inode=inode,**value))
    return volumes


def _container(observer,identity):
    if not isinstance(identity,str) or not recovery.HEX.fullmatch(identity): raise ValueError('host_container_id')
    value=_json_output(observer._read(['container','inspect',identity,'--format',CONTAINER_FORMAT]))
    recovery.closed(value,'id image started finished running restarting pid restart pidmode privileged capadd cgroupns mounts')
    if value['id']!=identity: raise ValueError('host_inspect_identity')
    return value


def host_inventory(observer,scope,owners,*,stopped=False):
    volumes=_host_scope(observer,scope)
    ids=observer._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()
    if (len(ids)>recovery.HOST_OWNER_LIMIT or len(set(ids))!=len(ids) or
            any(not recovery.HEX.fullmatch(i) for i in ids)): raise ValueError('host_enumeration_budget')
    consumers=[]
    for identity in sorted(ids):
        current=_container(observer,identity)
        if not isinstance(current['mounts'],list) or len(current['mounts'])>16: raise ValueError('host_mount_budget')
        for mount in current['mounts']: recovery.closed(mount,'type name source destination rw')
        # Bind aliases to the same source count as consumers and then refuse the
        # exact-volume topology validator. Unknown stopped consumers count too.
        if any(_mount_touches(m,volumes) for m in current['mounts']): consumers.append(current)
    # Detect changes during enumeration; another complete pass occurs before
    # publication. This is a bounded admin-maintenance protocol, not a daemon lock.
    if observer._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()!=ids:
        raise ValueError('host_enumeration_changed')
    if _host_scope(observer,scope)!=volumes: raise ValueError('host_scope_changed')
    topology=dict(scope=scope,volumes=volumes,consumers=consumers)
    return recovery.validate_host_topology(topology,scope,owners,stopped=stopped)


def _proc_observation(pid,identity,ipc):
    """Narrow local Linux cgroup-v2 observation; never cgroup-parent inference.

    A pinned C1 launcher must qualify these actual OS paths. Desktop/remote,
    delegated/unknown cgroups and mixed PID namespace trees refuse this mode.
    """
    if sys.platform!='linux' or type(pid) is not int or pid<1: raise ValueError('unsupported_host_pid_provenance')
    end=time.monotonic()+2; count=0
    def bounded(path):
        nonlocal count
        count+=1
        if count>2048 or time.monotonic()>end: raise ValueError('host_proc_budget')
        with open(path,'rb') as stream: raw=stream.read(65537)
        if len(raw)>65536: raise ValueError('host_proc_budget')
        return raw.decode('ascii')
    base=Path('/proc')/str(pid)
    def process(path):
        raw=bounded(path/'stat'); fields=raw[raw.rfind(')')+2:].split()
        ticks=int(fields[19]); ns=os.stat(path/'ns/pid').st_ino
        group=bounded(path/'cgroup').strip()
        return ticks,ns,group
    before=process(base)
    if before[1]==os.stat('/proc/self/ns/pid').st_ino: raise ValueError('host_shared_pid_namespace')
    allowed=['/docker/'+identity,'/system.slice/docker-'+identity+'.scope']
    if before[2] not in ['0::'+v for v in allowed]: raise ValueError('host_cgroup_association')
    group=before[2][3:]
    root=Path('/sys/fs/cgroup')/group.lstrip('/')
    if bounded('/proc/self/cgroup').strip().split(':',2)[-1]==group:
        raise ValueError('host_cgroup_namespace_unsupported')
    if not Path('/sys/fs/cgroup/cgroup.controllers').is_file(): raise ValueError('host_cgroup_v2_required')
    visited=[]; tasks=set(); pending=[root]
    while pending:
        directory=pending.pop(); visited.append(directory)
        if len(visited)>128 or time.monotonic()>end: raise ValueError('host_proc_budget')
        for task in bounded(directory/'cgroup.procs').splitlines():
            if not task.isdecimal() or len(tasks)>=2048: raise ValueError('host_proc_budget')
            taskpid=int(task); tasks.add(taskpid)
            _,namespace,taskgroup=process(Path('/proc')/str(taskpid))
            if namespace!=before[1] or not (taskgroup=='0::'+group or taskgroup.startswith('0::'+group+'/')):
                raise ValueError('host_execution_tree_changed')
        with os.scandir(directory) as entries:
            for entry in entries:
                count+=1
                if count>2048 or time.monotonic()>end: raise ValueError('host_proc_budget')
                if entry.is_symlink(): raise ValueError('host_cgroup_symlink')
                if entry.is_dir(follow_symlinks=False): pending.append(Path(entry.path))
    if pid not in tasks or process(base)!=before: raise ValueError('host_proc_identity_changed')
    observed=os.stat(base/'root/ipc'); expected=os.stat(ipc)
    if (observed.st_dev,observed.st_ino)!=(expected.st_dev,expected.st_ino): raise ValueError('host_ipc_association')
    return dict(pid=pid,start_ticks=before[0],pid_namespace=before[1],cgroup=group)


def _protected_directory(path,*,create=False):
    # Check every ancestor, disallow symlinks and group/other writable parents.
    path=Path(path)
    for parent in reversed(path.parents):
        st=parent.lstat()
        if not stat.S_ISDIR(st.st_mode) or stat.S_IMODE(st.st_mode)&0o022:
            raise ValueError('host_unsafe_parent')
    if create:
        try: path.mkdir(mode=0o700)
        except FileExistsError: pass
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    st=os.fstat(fd)
    if st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700:
        os.close(fd); raise ValueError('host_unsafe_directory')
    return fd


def _host_write(path,value):
    data=encode(value).encode()
    if len(data)>recovery.HOST_CAPSULE_BYTES: raise ValueError('host_document_budget')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        if stream.write(data)!=len(data): raise OSError('host_short_write')
        stream.flush(); os.fsync(stream.fileno())
    recovery.sync_dir(Path(path).parent)


def _provenance_read(registrations):
    fd=_protected_directory(recovery.HOST_AUTHORITY_DIRECTORY)
    try:
        values=[]; size=0; end=time.monotonic()+2
        for r in registrations:
            value=recovery._host_read_at(fd,'registration-'+digest(r)+'.json',os.getuid())
            size+=len(encode(value).encode())
            if size>recovery.HOST_CAPSULE_BYTES or time.monotonic()>end: raise ValueError('host_provenance_budget')
            values.append(value)
        for value,r in zip(values,registrations): recovery.validate_host_provenance(value,r)
        return values
    finally: os.close(fd)


def register_host(observer,roots,scope,registration):
    from service.supervisor_runtime import SupervisorRuntime
    recovery.host_registration_union([registration],scope,registration['deployment_id'])
    retained=host_state(roots,scope,_allow_empty=True)['registrations']
    registrations={digest(r):r for r in retained};registrations[digest(registration)]=registration
    owners=recovery.host_registration_union(list(registrations.values()),scope,registration['deployment_id'])
    selected={c['id'] for c in registration['containers']}
    first=host_inventory(observer,scope,owners)
    candidates=[c for c in first['consumers'] if c['id'] in selected]
    if any(not c['running'] for c in candidates): raise ValueError('host_registration_requires_live_owners')
    # Exactly one live worker can own the authenticated IPC/lane/boot association.
    if (sum(c['role']=='worker' for c in registration['containers'])!=1 or
            sum(c['running'] and next(o for o in owners if o['id']==c['id'])['role']=='worker' for c in first['consumers'])!=1): raise ValueError('host_unique_worker_required')
    proc=[]
    for c in candidates:
        proc.append({k:c[k] for k in ('id','image','started')} | _proc_observation(c['pid'],c['id'],roots['ipc']))
    hello=SupervisorRuntime(roots['ipc']).call('hello',{})
    if (hello.get('lane')!=registration['lane'] or hello.get('boot')!=registration['boot'] or
            hello.get('quarantined') is not False): raise ValueError('authenticated_supervisor_registration_mismatch')
    second=host_inventory(observer,scope,owners)
    repeated=[{k:c[k] for k in ('id','image','started')} | _proc_observation(c['pid'],c['id'],roots['ipc']) for c in second['consumers'] if c['id'] in selected]
    if first!=second or proc!=repeated: raise ValueError('host_registration_bracket_changed')
    ipc=os.stat(roots['ipc'])
    value=dict(revision=1,kind='local-linux-proc-v1',registration_digest=digest(registration),
        ipc_identity=[ipc.st_dev,ipc.st_ino],hello={k:hello[k] for k in ('lane','boot')},owners=proc)
    recovery.validate_host_provenance(value,registration)
    fd=_protected_directory(recovery.HOST_AUTHORITY_DIRECTORY,create=True); os.close(fd)
    # Retain a partial/crash obligation, never overwrite a previous certificate.
    _host_write(Path(recovery.HOST_AUTHORITY_DIRECTORY)/('registration-'+digest(registration)+'.json'),value)
    return recovery.register(roots,scope,registration)


def _purpose(value):
    recovery.closed(value,'kind plan_id plan_digest')
    if value['kind']=='pre-plan':
        if value['plan_id'] is not None or value['plan_digest'] is not None: raise ValueError('host_preplan_purpose')
    elif value['kind'] in ('initial','resume'):
        recovery._host_id(value['plan_id']); recovery._hash(value['plan_digest'])
    else: raise ValueError('host_purpose')


def host_preflight(observer,state,request):
    recovery.closed(request,'deployment_id scope registration_digest purpose original_invocation')
    _purpose(request['purpose'])
    if (request['deployment_id']!=state['deployment_id'] or request['scope']!=state['host_scope'] or
            request['registration_digest']!=recovery.host_registration_digest(state['registrations'])):
        raise ValueError('host_freeze_scope_changed')
    owners=recovery.host_registration_union(state['registrations'],request['scope'],state['deployment_id'])
    provenance=_provenance_read(state['registrations'])
    # Validate ambiguity before stop, rather than waiting for final capsule checks.
    bindings={}
    for r in state['registrations']:
        for c in r['containers']:
            if c['role']=='worker':
                binding=(r['lane'],r['boot'])
                if c['id'] in bindings and bindings[c['id']]!=binding: raise ValueError('host_ambiguous_worker_boot')
                bindings[c['id']]=binding
    original=None
    if state.get('unfinished') and request['original_invocation'] is None: raise ValueError('host_original_resume_required')
    if request['original_invocation'] is not None:
        recovery._host_id(request['original_invocation'])
        fd=_protected_directory(Path(recovery.HOST_AUTHORITY_DIRECTORY)/'invocations'/request['original_invocation'])
        try: original=recovery._host_read_at(fd,'capsule.json',os.getuid())
        finally: os.close(fd)
        # Old issuance can expire; original immutable proof/identities cannot.
        recovery.validate_host_capsule(original,now=original['issued'])
        if (request['purpose']['kind']!='resume' or original['registration_digest']!=request['registration_digest'] or
                original['scope']!=request['scope'] or original['deployment_id']!=request['deployment_id'] or
                original['provenance']!=provenance or
                (original['purpose']['kind']!='pre-plan' and
                 any(original['purpose'][k]!=request['purpose'][k] for k in ('plan_id','plan_digest')))): raise ValueError('host_resume_original_changed')
    elif request['purpose']['kind']=='resume': raise ValueError('host_resume_original_required')
    topology=host_inventory(observer,request['scope'],owners,stopped=original is not None)
    if original is not None and topology!=original['topology']: raise ValueError('host_resume_incarnation_changed')
    # Running owners must still have the exact registration-time process identity.
    for c in topology['consumers']:
        if c['running']:
            current={k:c[k] for k in ('id','image','started')} | _proc_observation(c['pid'],c['id'],state['roots']['ipc'])
            retained=[o for p in provenance for o in p['owners'] if o['id']==c['id']]
            if not retained or any(o!=current for o in retained): raise ValueError('host_proc_identity_changed')
    # Conservative prospective capsule ceiling, including a repeated frozen
    # observation, bounded finish strings and wait records. Refuse before stop.
    size=len(encode(dict(registrations=state['registrations'],provenance=provenance,
        topology=topology,owners=owners)).encode())+len(encode(topology).encode())
    if size+65536+1024*len(owners)>recovery.HOST_CAPSULE_BYTES: raise ValueError('host_capsule_population_budget')
    if time.monotonic()>=observer.deadline: raise ValueError('docker_global_deadline')
    return owners,provenance,topology,original


def freeze_host(observer,state,request):
    """An explicitly selected finite stop/wait attempt. No kernel invocation."""
    owners,provenance,before,original=host_preflight(observer,state,request)
    if host_inventory(observer,request['scope'],owners,stopped=original is not None)!=before:
        raise ValueError('host_before_stop_changed')
    waits=[]
    # All admission/control writers stop before any worker is stopped.
    for owner in sorted(owners,key=lambda o:(o['role']!='control',o['id'])):
        if original is None:
            response=observer._read(['container','stop','--time','2',owner['id']])
            if response!=owner['id']: raise ValueError('host_stop_result')
        status=observer._read(['container','wait',owner['id']])
        if not re.fullmatch(r'[0-9]{1,3}',status) or int(status)>255: raise ValueError('host_wait_result')
        waits.append(dict(id=owner['id'],command_status=0,container_status=int(status)))
    final=host_inventory(observer,request['scope'],owners,stopped=True)
    if original is not None and final!=original['topology']: raise ValueError('host_resume_incarnation_changed')
    # Already stopped incarnations must retain their original meaningful finish.
    for old in before['consumers']:
        if not old['running'] and next(c for c in final['consumers'] if c['id']==old['id'])!=old:
            raise ValueError('host_stopped_incarnation_changed')
    if final!=host_inventory(observer,request['scope'],owners,stopped=True): raise ValueError('host_final_topology_changed')
    # Re-read retained registration authority immediately before publishing.
    current=host_state(state['roots'],request['scope'])
    if (current['registrations']!=state['registrations'] or current['lane']!=state['lane'] or
            current['prior_epoch']!=state['prior_epoch']): raise ValueError('host_registration_set_changed')
    if final!=host_inventory(observer,request['scope'],owners,stopped=True): raise ValueError('host_final_topology_changed')
    observation=dict(scope=request['scope'],containers=owners,evidence='actual-host-inspection',
        states=[{k:c[k] for k in ('id','running','restarting','pid','started','finished')} for c in final['consumers']])
    proof=original['kernel_proof'] if original else dict(kind='host-admin-container-teardown-v1',
        observation=observation,prior_epoch=state['prior_epoch'],old_lane=state['lane'])
    issued=time.time(); elapsed=time.monotonic()-observer.started
    capsule=dict(revision=1,kind=recovery.HOST_CAPSULE_KIND,invocation_id=uuid.uuid4().hex,
        deployment_id=state['deployment_id'],scope=request['scope'],registrations=state['registrations'],original_invocation=request['original_invocation'],
        registration_digest=request['registration_digest'],kernel_proof=proof,kernel_proof_digest=digest(proof),
        purpose=request['purpose'],issued=issued,expires=issued+recovery.TTL,collection_seconds=elapsed,
        topology=final,provenance=provenance,waits=waits)
    recovery.validate_host_capsule(capsule,now=issued)
    if time.monotonic()>=observer.deadline: raise ValueError('docker_global_deadline')
    base=Path(recovery.HOST_AUTHORITY_DIRECTORY)
    fd=_protected_directory(base/'invocations',create=True); os.close(fd)
    path=base/'invocations'/capsule['invocation_id']
    fd=_protected_directory(path,create=True); os.close(fd)
    _host_write(path/'capsule.json',capsule)
    recovery.sync_dir(path.parent)
    return capsule


def host_state(roots,scope,*,_allow_empty=False):
    """Advisory retained-owner inventory while normal owners may hold locks.

    SQLite opens only a stable private WAL copy. This is not a kernel source
    projection or preparation/acceptance gate. B2.3b/c own those locked checks.
    """
    roots=roots_checked(roots);validate_scope(scope)
    with recovery.private_database(roots['control']/'core/core.sqlite') as path:
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=0)) as db:
            db.execute('PRAGMA query_only=ON'); db.execute('PRAGMA trusted_schema=OFF')
            end=time.monotonic()+2
            db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('p1_installation','p1_registrations'):
                if table not in tables: raise ValueError('host_registration_schema')
                if [r[1] for r in db.execute('PRAGMA table_info('+table+')')]!=recovery.B1_COLUMNS[table].split():
                    raise ValueError('host_registration_schema')
                n,size=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM '+table).fetchone()
                if n>recovery.HOST_OWNER_LIMIT or size>recovery.HOST_CAPSULE_BYTES: raise ValueError('host_registration_budget')
            rows=db.execute('SELECT id,record,digest FROM p1_installation').fetchall()
            if len(rows)!=1: raise ValueError('host_installation_missing')
            identity,raw,hashed=rows[0];installation=recovery.host_json(raw.encode())
            recovery.closed(installation,'deployment_id revision scope host profile_budget_digest')
            if (identity!=installation['deployment_id'] or digest(installation)!=hashed or
                    type(installation['revision']) is not int or installation['revision']!=1 or
                    installation['scope']!='owned-synthetic' or installation['host']!=scope):
                raise ValueError('host_installation_changed')
            registrations=[]
            for rid,raw,hashed in db.execute('SELECT id,record,digest FROM p1_registrations ORDER BY rowid'):
                registration=recovery.host_json(raw.encode())
                if rid!=hashed or digest(registration)!=hashed: raise ValueError('host_registration_changed')
                registrations.append(registration)
            if registrations or not _allow_empty: recovery.host_registration_union(registrations,scope,identity)
            prior_epoch=identity;unfinished=False
            if 'p1_recovery_steps' in tables:
                n,size=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM p1_recovery_steps').fetchone()
                if n>recovery.MAX_ROWS or size>recovery.MAX_BYTES: raise ValueError('host_history_budget')
                for raw,hashed in db.execute("SELECT record,digest FROM p1_recovery_steps WHERE step='core-complete' ORDER BY rowid"):
                    step=recovery.host_json(raw.encode());recovery.validate_b2_record('p1_recovery_steps',step)
                    if digest(step)!=hashed: raise ValueError('host_epoch_changed')
                    prior_epoch=step['epoch']
                unfinished=db.execute("SELECT 1 FROM p1_recoveries r WHERE NOT EXISTS (SELECT 1 FROM p1_recovery_steps s WHERE s.recovery_id=r.id AND s.step='receipt-complete') LIMIT 1").fetchone() is not None
    private=recovery.host_json(read_file(roots['receipts']/'p1-init.json'))
    marker=recovery.host_json(read_file(roots['receipts']/'p1-enrolled.json'))
    if ({k:v for k,v in private.items() if k not in ('roots','seed')}!=installation or
            private.get('roots')!={k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()} or
            marker!=dict(deployment_id=identity,digest=digest(installation))):
        raise ValueError('host_enrollment_identity_changed')
    # Host death coverage uses retained identities, including damaged/missing
    # journals. It makes no claim about kernel preparation or a healthy epoch.
    lane=registrations[-1]['lane'] if registrations else None
    fence_path=roots['receipts']/'p1-maintenance.json'
    if fence_path.exists() or fence_path.is_symlink():
        from worker.runtime_journal import maintenance_record
        fence=maintenance_record(recovery.host_json(read_file(fence_path)))
        if fence['deployment_id']!=identity: raise ValueError('host_maintenance_scope_changed')
        unfinished=unfinished or fence['step']!='complete'
    return dict(roots=roots,host_scope=scope,deployment_id=identity,registrations=registrations,
        lane=lane,prior_epoch=prior_epoch,unfinished=unfinished)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('installed-register','register','inspect','plan','preflight','freeze','apply'))
    parser.add_argument('--request',required=True)
    args=parser.parse_args()
    if args.command=='apply': recovery.apply()
    request=recovery.host_json(read_file(Path(args.request)))
    if args.command=='installed-register':
        recovery.closed(request,'configuration scope owners')
        with installed_launch_authority(request['configuration'],request['scope']) as authority:
            result=dict(registration_id=authority.register(request['owners']))
    elif args.command in ('preflight','freeze'):
        recovery.closed(request,'roots scope freeze')
        state=host_state(request['roots'],request['scope']); observer=Docker(request['scope']['context'])
        if args.command=='freeze': result=freeze_host(observer,state,request['freeze'])
        else:
            owners,_,topology,_=host_preflight(observer,state,request['freeze'])
            result=dict(apply_available=False,owners=owners,topology=topology,registration_digest=recovery.host_registration_digest(state['registrations']))
    else:
        recovery.closed(request,'roots scope registration')
        observer=Docker(request['scope']['context'])
        if args.command=='register':
            result=dict(registration_id=register_host(observer,request['roots'],request['scope'],request['registration']),evidence='actual-host-inspection')
        else:
            state=recovery.inspect(request['roots'],request['scope'])
            # Union ALL retained registrations; the supplied registration is not
            # a selector capable of omitting an old owner.
            owners={}
            for r in state['registrations']:
                for c in r['containers']:
                    if c['id'] in owners and owners[c['id']]!=c: raise ValueError('ambiguous_incarnation')
                    owners[c['id']]=c
            projection={**request['registration'],'containers':list(owners.values())}
            observation=observer.observe(request['scope'],projection)
            if args.command=='plan': result=recovery.plan(request['roots'],request['scope'],observation)
            else:
                recovery.check_consumers(state,observation)
                result={k:v for k,v in state.items() if k!='registrations'}; result['evidence']='actual-host-inspection'
    print(encode(result))




# Installed observer is separate from the historical synthetic projection.
# Observation remains separate from the protected registration transaction below.
from contextlib import contextmanager
from service import install_policy as ip
import hashlib
import fcntl

INSTALLED_FIELDS = {
    'user': '.Config.User', 'capdrop': '.HostConfig.CapDrop',
    'security': '.HostConfig.SecurityOpt', 'readonly': '.HostConfig.ReadonlyRootfs',
    'network': '.HostConfig.NetworkMode', 'groups': '.HostConfig.GroupAdd',
    'devices': '.HostConfig.Devices', 'device_requests': '.HostConfig.DeviceRequests',
    'device_rules': '.HostConfig.DeviceCgroupRules', 'userns': '.HostConfig.UsernsMode',
    'ipcns': '.HostConfig.IpcMode', 'utsns': '.HostConfig.UTSMode',
    'memory': '.HostConfig.Memory', 'swap': '.HostConfig.MemorySwap',
    'pids': '.HostConfig.PidsLimit', 'nanocpus': '.HostConfig.NanoCpus',
    'period': '.HostConfig.CpuPeriod', 'quota': '.HostConfig.CpuQuota',
    'cpuset': '.HostConfig.CpusetCpus', 'tmpfs': '.HostConfig.Tmpfs',
    'ports': '.HostConfig.PortBindings', 'publish_all': '.HostConfig.PublishAllPorts',
}
INSTALLED_FORMAT = CONTAINER_FORMAT[:-1] + ''.join(
    ',"'+key+'":{{json '+field+'}}' for key,field in INSTALLED_FIELDS.items()) + '}'


def _installed_container(observer, identity):
    if not isinstance(identity,str) or not recovery.HEX.fullmatch(identity):
        raise ValueError('host_container_id')
    value = _json_output(observer._read(['container','inspect',identity,'--format',INSTALLED_FORMAT]))
    recovery.closed(value, 'id image started finished running restarting pid restart pidmode privileged capadd cgroupns mounts ' + ' '.join(INSTALLED_FIELDS))
    if value['id'] != identity: raise ValueError('host_inspect_identity')
    return value


def _source_stat(path, *, directory):
    # Descriptor traversal preserves literal no-follow ancestry, including /tmp.
    with ip.parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK |
                     (os.O_DIRECTORY if directory else 0), dir_fd=parent)
        try:
            before = os.fstat(fd)
            if not (stat.S_ISDIR(before.st_mode) if directory else
                    stat.S_ISREG(before.st_mode) and before.st_nlink == 1):
                raise ValueError('installed_source_type')
            if ip.identity(os.stat(name,dir_fd=parent,follow_symlinks=False)) != ip.identity(before):
                raise ValueError('installed_source_changed')
            return before
        finally: os.close(fd)


def _public_file(path, *, uid, gid, mode, limit, deadline):
    with ip.parent_fd(path) as (parent,name):
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        try:
            before=os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                    (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode)) != (uid,gid,mode) or
                    before.st_size > limit): raise ValueError('installed_public_file_policy')
            result=[]; size=0
            while size <= before.st_size:
                if time.monotonic() > deadline: raise ValueError('installed_source_budget')
                block=os.read(fd,min(1024*1024,before.st_size+1-size))
                if not block: break
                size+=len(block); result.append(block)
            if size!=before.st_size or ip.identity(os.fstat(fd))!=ip.identity(before):
                raise ValueError('installed_source_changed')
            with ip.parent_fd(path) as (current,leaf):
                if ip.identity(os.stat(leaf,dir_fd=current,follow_symlinks=False))!=ip.identity(before):
                    raise ValueError('installed_source_changed')
            return b''.join(result)
        finally: os.close(fd)


def _model_tree(path, config, deadline):
    """Bounded sealed tree inventory; hash every file, verify enrolled manifest."""
    records=[]; total=0; pending=[Path(path)]; files={}; directories=[]; file_identities=[]
    entry_count=1  # Root plus every discovered child, including queued directories.
    while pending:
        if time.monotonic()>deadline: raise ValueError('installed_source_budget')
        current=pending.pop(); before=_source_stat(current,directory=True)
        if (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode))!=(10001,10001,0o555):
            raise ValueError('installed_model_mode')
        directories.append((current,ip.identity(before)))
        records.append([str(current.relative_to(path)), 'directory', list(ip.identity(before))])
        if time.monotonic()>deadline: raise ValueError('installed_source_budget')
        names=[]
        with os.scandir(current) as entries:
            iterator=iter(entries)
            while True:
                if time.monotonic()>deadline: raise ValueError('installed_source_budget')
                try: entry=next(iterator)
                except StopIteration: break
                if time.monotonic()>deadline or entry_count>=4096:
                    raise ValueError('installed_source_budget')
                names.append(entry.name); entry_count+=1
        names.sort()
        for name in names:
            child=current/name
            with ip.parent_fd(child) as (parent,leaf): s=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
            if stat.S_ISDIR(s.st_mode): pending.append(child); continue
            file_before=ip.identity(_source_stat(child,directory=False))
            raw=_public_file(child,uid=10001,gid=10001,mode=0o444,
                             limit=min(512*1024**2,2*1024**3-total),deadline=deadline)
            total+=len(raw); relative=str(child.relative_to(path)); hashed=hashlib.sha256(raw).hexdigest()
            files[relative]=(hashed,len(raw))
            if ip.identity(_source_stat(child,directory=False))!=file_before:
                raise ValueError('installed_source_changed')
            file_identities.append((child,file_before))
            records.append([relative,hashed,len(raw),list(file_before)])
            if child.name=='manifest.json':
                if len(raw)>65536: raise ValueError('installed_manifest_budget')
                files[relative]+=(recovery.host_json(raw),)
    expected={'bundles/'+bid+'/manifest.json': bundle for bid,bundle in config['bundles'].items()}
    if {n for n in files if n.endswith('/manifest.json')} != set(expected):
        raise ValueError('installed_model_manifest')
    allowed=set(expected)
    for name,bundle in expected.items():
        hashed,_,manifest=files[name]
        if hashed!=bundle['manifest_sha256'] or manifest.get('backend')!='kokoro' or manifest.get('schema_version')!=1 or (manifest.get('checkpoint'),manifest.get('revision'))!=(bundle['checkpoint'],bundle['revision']):
            raise ValueError('installed_model_manifest')
        if not isinstance(manifest.get('files'),list) or not manifest['files']:
            raise ValueError('installed_model_manifest')
        for entry in manifest['files']:
            relative=entry.get('path')
            if not isinstance(relative,str) or str(Path(relative))!=relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('installed_model_manifest')
            target=str(Path(name).parent/relative)
            if target in allowed or target not in files or files[target][0]!=entry.get('sha256') or ('bytes' in entry and files[target][1]!=entry['bytes']):
                raise ValueError('installed_model_manifest')
            allowed.add(target)
    if set(files)!=allowed: raise ValueError('installed_model_extra_file')
    for child,before in file_identities:
        if ip.identity(_source_stat(child,directory=False))!=before:
            raise ValueError('installed_source_changed')
    for current,before in directories:
        if ip.identity(_source_stat(current,directory=True))!=before:
            raise ValueError('installed_source_changed')
    return digest(sorted(records))


def _installed_sources(policy,config):
    deadline=time.monotonic()+recovery.HOST_COLLECTION_SECONDS
    identities={}; paths=[]; records={**policy['roots'],**policy['inputs']}
    for name,record in records.items():
        s=_source_stat(record['source'],directory=name not in ('config','credential'))
        if (s.st_dev,s.st_ino)!=(record['device'],record['inode']):
            raise ValueError('installed_source_identity')
        if name in policy['inputs'] and (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))!=(record['uid'],record['gid'],record['mode']):
            raise ValueError('installed_input_identity')
        if name in policy['inputs'] and any(getattr(s,'st_'+k)!=record[k] for k in ('size','mtime_ns','ctime_ns')):
            raise ValueError('installed_input_changed')
        identities[name]=ip.identity(s); paths.append(Path(record['source']))
    if len({(s[0],s[1]) for s in identities.values()})!=len(records) or any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('installed_source_overlap')
    # Credential descriptor open/fstat is allowed; never read or hash its contents.
    raw=_public_file(policy['inputs']['config']['source'],uid=0,gid=10001,mode=0o440,
                     limit=65536,deadline=deadline)
    if (hashlib.sha256(raw).hexdigest()!=policy['inputs']['config']['sha256'] or
            recovery.host_json(raw)!=config): raise ValueError('installed_config_changed')
    if _model_tree(policy['inputs']['models']['source'],config,deadline)!=policy['inputs']['models']['sha256']:
        raise ValueError('installed_models_changed')
    return identities


def _installed_mount_proof(policy, mounts):
    """Only same ordinary observed filesystem, no intervening/bind aliases."""
    sources=[v['source'] for v in (*policy['roots'].values(),*policy['inputs'].values())]
    selected=[]
    for source in sources:
        covering=[m for m in mounts if source==m['target'] or source.startswith(m['target'].rstrip('/')+'/')]
        if not covering: raise ValueError('installed_mount_unproved')
        mount=max(covering,key=lambda m:len(m['target']))
        if mount['root']!='/': raise ValueError('installed_mount_alias')
        if any(m['target']==source or m['target'].startswith(source+'/') for m in mounts):
            raise ValueError('installed_nested_mount')
        selected.append(mount)
    if any(m!=selected[0] for m in selected): raise ValueError('installed_same_mount_required')
    chosen=selected[0]
    if any(m['id']!=chosen['id'] and m['device']==chosen['device'] for m in mounts):
        raise ValueError('installed_mount_alias')
    if any(str(v['device']) != str(os.makedev(*map(int,chosen['device'].split(':'))))
           for v in (*policy['roots'].values(),*policy['inputs'].values())):
        raise ValueError('installed_mount_device')
    return chosen


def validate_installed_topology(topology, scope, owners, config, policy, *, stopped=False):
    """Content validator only; no authority or actual qualification assertion."""
    import copy
    ip.host_policy(policy,config,scope)
    recovery.closed(topology,'scope volumes consumers policy configuration_digest host_policy_digest')
    if (topology['policy']!=ip.HOST_POLICY or topology['configuration_digest']!=digest(config) or
            topology['host_policy_digest']!=digest(policy)):
        raise ValueError('installed_topology_binding')
    projected=copy.deepcopy({k:topology[k] for k in ('scope','volumes','consumers')})
    expected={c['id']:c for c in owners}
    for current,legacy in zip(topology['consumers'],projected['consumers']):
        recovery.closed(current,'id image started finished running restarting pid restart pidmode privileged capadd cgroupns mounts '+' '.join(INSTALLED_FIELDS))
        if current['id'] not in expected: raise ValueError('host_unknown_consumer')
        owner=expected[current['id']]; role=owner['role']
        if role not in ('control','worker'): raise ValueError('installed_container_role')
        if (current['user'] != ('10001:10001' if role=='control' else '0:10001') or
                current['capdrop']!=['ALL'] or
                sorted(current['capadd'] or []) != ([] if role=='control' else ['KILL','SETGID','SETUID']) or
                current['security'] != ['no-new-privileges:true'] or current['readonly'] is not True or
                current['groups'] not in (None,[]) or current['devices'] not in (None,[]) or
                current['device_requests'] not in (None,[]) or current['device_rules'] not in (None,[]) or
                current['userns']!='' or current['ipcns']!='private' or current['utsns']!='' or
                current['cgroupns']!='private' or current['image']!=config['images'][role]):
            raise ValueError('installed_container_policy')
        # No alternative quota/cpuset path, unlimited swap, or implicit default.
        if (any(type(current[k]) is not int for k in ('memory','swap','pids','nanocpus','period','quota')) or
                current['memory']!=config['budget']['ram_bytes'] or current['swap']!=current['memory'] or
                current['pids']!=config['budget']['pids'] or current['nanocpus']!=policy['cpu'][role] or
                current['period']!=0 or current['quota']!=0 or current['cpuset']!=''):
            raise ValueError('installed_resource_policy')
        ports={'8000/tcp':[{'HostIp':'127.0.0.1','HostPort':str(policy['control_port'])}]} if role=='control' else {}
        if (current['network']!=('bridge' if role=='control' else 'none') or
                (current['ports'] or {})!=ports or current['publish_all'] is not False):
            raise ValueError('installed_network_policy')
        tmp=policy['tmpfs'][role]
        options='rw,noexec,nosuid,nodev,size='+str(tmp['bytes'])+',nr_inodes='+str(tmp['inodes'])+',mode=1777'
        if current['tmpfs']!={'/tmp':options}: raise ValueError('installed_tmpfs_policy')
        wanted_roles=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging']
        if sorted(owner['volumes'])!=wanted_roles: raise ValueError('installed_authority_mounts')
        allowed={ '/models':policy['inputs']['models'], ip.CONFIG_DESTINATION:policy['inputs']['config'] }
        if role=='control': allowed[ip.CREDENTIAL_DESTINATION]=policy['inputs']['credential']
        mounts=[]; seen=set(); tmp_seen=False
        for mount in current['mounts']:
            recovery.closed(mount,'type name source destination rw')
            destination=mount['destination']
            if destination in allowed:
                if (destination in seen or mount['type']!='bind' or mount['name'] not in ('',None) or
                        mount['source']!=allowed[destination]['source'] or mount['rw'] is not False):
                    raise ValueError('installed_input_mount')
                seen.add(destination)
            elif destination=='/tmp':
                if tmp_seen or mount['type']!='tmpfs' or mount['rw'] is not True or mount['source'] not in ('',None) or mount['name'] not in ('',None):
                    raise ValueError('installed_tmpfs_mount')
                tmp_seen=True
            else: mounts.append(mount)
        # Docker versions may omit tmpfs from .Mounts; HostConfig.Tmpfs is exact.
        if seen!=set(allowed): raise ValueError('installed_input_mount_coverage')
        legacy['mounts']=mounts; legacy['capadd']=[]
        for key in INSTALLED_FIELDS: legacy.pop(key)
    recovery.validate_host_topology(projected,scope,owners,stopped=stopped)
    for volume in topology['volumes']:
        if {k:volume[k] for k in ('source','device','inode')}!=policy['roots'][volume['role']]:
            raise ValueError('installed_volume_binding')
    return topology


class InstalledObserver:
    """Trusted internal adapter; injection is for tests, never a CLI evidence flag.

    A protected launch session constructs this object. Returned JSON is content,
    not proof that an actual host was qualified. Activation remains separate.
    """
    def __init__(self, docker, config, scope, policy, *, mountinfo=recovery._host_mount_topology,
                 proc=_proc_observation):
        self.config=ip.configuration(config)
        # Detach caller data; all calls corroborate the frozen authority.
        self.policy=json.loads(encode(ip.host_policy(policy,self.config,scope)))
        self.scope=json.loads(encode(scope)); self.docker=docker
        self.mountinfo=mountinfo; self.proc=proc

    def inventory(self, owners, *, stopped=False):
        policy=self.policy
        if owners:
            ip.roots_checked({r:v['source'] for r,v in policy['roots'].items()},policy=ip.SPLIT)
        mounts=self.mountinfo(); _installed_mount_proof(policy,mounts)
        before=_installed_sources(policy,self.config)
        security=_json_output(self.docker._read(['info','--format','{"security":{{json .SecurityOptions}},"cgroup":{{json .CgroupVersion}},"name":{{json .Name}},"os":{{json .OperatingSystem}}}']))
        recovery.closed(security,'security cgroup name os')
        if (security['cgroup']!='2' or not isinstance(security['security'],list) or
                any('rootless' in s.lower() or 'userns' in s.lower() for s in security['security']) or
                'desktop' in (str(security['name'])+' '+str(security['os'])).lower()):
            raise ValueError('installed_local_rootful_required')
        volumes=_host_scope(self.docker,self.scope)
        ids=self.docker._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()
        if len(ids)>recovery.HOST_OWNER_LIMIT or len(set(ids))!=len(ids) or any(not recovery.HEX.fullmatch(i) for i in ids):
            raise ValueError('host_enumeration_budget')
        consumers=[]
        exact_inputs={r['source'] for r in policy['inputs'].values()}
        for identity in sorted(ids):
            c=_installed_container(self.docker,identity)
            if not isinstance(c['mounts'],list) or len(c['mounts'])>16: raise ValueError('host_mount_budget')
            touches=False
            for m in c['mounts']:
                recovery.closed(m,'type name source destination rw')
                if m['type']=='bind' and m['source'] in exact_inputs:
                    touches=True  # Exact narrow disjointness proof already held.
                elif _mount_touches(m,volumes): touches=True
                elif m['type']!='tmpfs': raise ValueError('installed_unknown_storage')
            if touches: consumers.append(c)
        topology=dict(scope=self.scope,volumes=volumes,consumers=consumers,policy=ip.HOST_POLICY,
                      configuration_digest=digest(self.config),host_policy_digest=digest(policy))
        validate_installed_topology(topology,self.scope,owners,self.config,policy,stopped=stopped)
        if (self.docker._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()!=ids or
                _host_scope(self.docker,self.scope)!=volumes or self.mountinfo()!=mounts or
                _installed_sources(policy,self.config)!=before):
            raise ValueError('installed_observation_changed')
        return topology

    def initialize_authority(self):
        topology=self.inventory([])
        return dict(host=self.scope,roots={v['role']:[v['device'],v['inode']] for v in topology['volumes']},consumers=[])

    def current_owners(self, claims):
        """Inspect each retained ID; no caller-selected StartedAt projection."""
        result=[]
        for claim in claims:
            actual=_installed_container(self.docker,claim['id'])
            result.append(dict(claim,started=actual['started'],image=actual['image']))
        return result

    def observe_owners(self, owners, enrollment):
        enrollment_digest=ip.host_enrollment(enrollment,self.policy,self.config,self.scope)
        first=self.inventory(owners)
        proc=[{k:c[k] for k in ('id','image','started')} | self.proc(c['pid'],c['id'],self.policy['roots']['ipc']['source'])
              for c in first['consumers'] if c['running']]
        second=self.inventory(owners)
        repeated=[{k:c[k] for k in ('id','image','started')} | self.proc(c['pid'],c['id'],self.policy['roots']['ipc']['source'])
                  for c in second['consumers'] if c['running']]
        if first!=second or proc!=repeated: raise ValueError('installed_owner_bracket_changed')
        result=dict(kind=ip.HOST_POLICY,enrollment_digest=enrollment_digest,configuration_digest=digest(self.config),host_policy_digest=digest(self.policy),
                    topology=first,processes=proc)
        return recovery.validate_installed_observation(result,self.scope,owners,self.config,self.policy,enrollment)


def _root_authority_directory(path):
    path=Path(path)
    # Root-owned private leaf; every ancestor is trusted and non-writable.
    for parent in (path,*path.parents):
        with ip.parent_fd(parent/'__authority_probe__') as (fd,_):
            s=os.fstat(fd)
            if s.st_uid!=0 or not stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode)&0o022:
                raise ValueError('installed_unsafe_authority')
            if parent==path and (stat.S_IMODE(s.st_mode)!=0o700 or s.st_gid!=0):
                raise ValueError('installed_unsafe_authority')
    return os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)


@contextmanager
def installed_launch_authority(config, scope):
    """Existing protected per-install policy + finite nonblocking exclusion.

    Provisioning is separate and create-only. No arbitrary policy path or
    metadata selection here, no creating locks, no policy repair. Hold through init/observation/registration.
    """
    if sys.platform!='linux' or os.getuid()!=0 or os.geteuid()!=0:
        raise ValueError('installed_linux_root_authority_required')
    config=ip.configuration(config); validate_scope(scope)
    directory=Path(recovery.HOST_AUTHORITY_DIRECTORY)/'installed'/digest(scope)
    fd=_root_authority_directory(directory)
    lock=None
    try:
        lock=os.open('launch.lock',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        original=os.fstat(lock)
        def stable():
            s=os.fstat(lock)
            if (not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=0 or s.st_gid!=0 or
                    stat.S_IMODE(s.st_mode)!=0o600 or s.st_size!=0 or ip.identity(s)!=ip.identity(original) or
                    ip.identity(os.stat('launch.lock',dir_fd=fd,follow_symlinks=False))!=ip.identity(original)):
                raise ValueError('installed_launch_lock_changed')
            current=_root_authority_directory(directory)
            try:
                if ip.identity(os.fstat(current))!=ip.identity(os.fstat(fd)):
                    raise ValueError('installed_authority_changed')
            finally: os.close(current)
        stable()
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('installed_launch_busy') from None
        raw=_public_file(directory/'host-policy.json',uid=0,gid=0,mode=0o600,limit=65536,
                         deadline=time.monotonic()+2)
        policy_identity=ip.identity(_source_stat(directory/'host-policy.json',directory=False))
        policy=ip.host_policy(recovery.host_json(raw),config,scope)
        observer=InstalledObserver(Docker(scope['context']),config,scope,policy)
        active=True
        def check():
            if not active: raise ValueError('installed_launch_authority_closed')
            stable()
            if ip.identity(_source_stat(directory/'host-policy.json',directory=False))!=policy_identity:
                raise ValueError('installed_host_policy_changed')
            if _public_file(directory/'host-policy.json',uid=0,gid=0,mode=0o600,limit=65536,
                            deadline=time.monotonic()+2)!=raw:
                raise ValueError('installed_host_policy_changed')
        class HeldAuthority:
            @property
            def policy(self):
                check(); return json.loads(encode(policy))
            def _public(self):
                roots={r:Path(v['source']) for r,v in observer.policy['roots'].items()}
                raw=read_file(roots['receipts']/'p1-init.json',policy=ip.SPLIT,role='receipts')
                private=recovery.host_json(raw)
                ip.initialization(private)
                public=private['public']
                ip.host_enrollment(public,observer.policy,observer.config,observer.scope)
                enrolled=recovery.host_json(read_file(roots['receipts']/'p1-enrolled.json',policy=ip.SPLIT,role='receipts'))
                if enrolled != dict(deployment_id=public['deployment_id'],digest=digest(public)):
                    raise ValueError('installed_enrollment_binding')
                return roots,public
            def preflight(self, owners):
                check(); roots,public=self._public()
                observer.inventory(owners,stopped=True)
                with ip.locked_file(roots['control']/'core/control.lock',policy=ip.SPLIT,role='control'):
                    with ip.locked_file(roots['receipts']/'lane.lock',policy=ip.SPLIT,role='receipts'):
                        state=_registration_snapshot(roots,observer.scope)
                        from worker.runtime_journal import read_snapshot
                        journal,maintenance=read_snapshot(roots['receipts'],policy=ip.SPLIT)
                        if maintenance is not None: raise ValueError('installed_maintenance_pending')
                        recovery.installed_clean_history(state,journal)
                        retained,_,_,_=_installed_history_read(observer,public,fd,state['registrations'])
                        current=_installed_current(observer,list(retained.values()),observer.scope,public['deployment_id']) if retained else []
                        if retained and sorted(current,key=lambda c:c['id'])!=sorted(owners,key=lambda c:c['id']):
                            raise ValueError('installed_retained_owner_coverage')
                check()
            def clean_shutdown(self, owners):
                # Live worker retains lane.lock. Stable read brackets plus healthy
                # hello and exact stopped-control topology corroborate its idle
                # terminal history; this is never a writable maintenance session.
                check(); roots,public=self._public()
                first=observer.inventory(owners)
                by_id={c['id']:c for c in first['consumers']}
                if any(by_id[c['id']]['running'] != (c['role']=='worker') for c in owners):
                    raise ValueError('installed_shutdown_owner_state')
                with ip.locked_file(roots['control']/'core/control.lock',policy=ip.SPLIT,role='control'):
                    state=_registration_snapshot(roots,observer.scope)
                    records=[r for r in state['registrations'] if r['containers']==owners]
                    if len(records)!=1: raise ValueError('installed_shutdown_registration')
                    with ip.parent_fd(roots['ipc']) as (parent,name):
                        ipc=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                    try:
                        hello=_registration_hello(_registration_child('hello',dict(ipc_fd=ipc),(ipc,)),records[0])
                        from worker.runtime_journal import read_snapshot
                        journal,maintenance=read_snapshot(roots['receipts'],policy=ip.SPLIT)
                        if maintenance is not None: raise ValueError('installed_maintenance_pending')
                        recovery.installed_clean_history(state,journal)
                        if (read_snapshot(roots['receipts'],policy=ip.SPLIT)!=(journal,maintenance) or
                                _registration_hello(_registration_child('hello',dict(ipc_fd=ipc),(ipc,)),records[0])!=hello):
                            raise ValueError('installed_shutdown_journal_changed')
                    finally: os.close(ipc)
                if observer.inventory(owners)!=first: raise ValueError('installed_shutdown_topology_changed')
                check()
            def observe_owners(self, owners, enrollment=None):
                check()
                if enrollment is None:
                    roots,public=self._public()
                else: public=enrollment
                result=observer.observe_owners(owners,public)
                check(); return result
            def initialize_authority(self):
                check(); result=observer.initialize_authority(); check(); return result
            def wait_worker(self):
                check(); roots,public=self._public()
                with ip.parent_fd(roots['ipc']) as (parent,name):
                    ipc=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                try:
                    ip.check(os.fstat(ipc),ip.SPLIT,'ipc',directory=True)
                    deadline=time.monotonic()+10
                    while True:
                        check()
                        try:
                            hello=_registration_hello(_registration_child('hello',dict(ipc_fd=ipc),(ipc,)))
                            check(); return hello
                        except ValueError:
                            if time.monotonic()>=deadline: raise ValueError('installed_worker_wait_timeout') from None
                            time.sleep(.05)
                finally: os.close(ipc)
            def confirm_ready(self, owners, rid):
                check(); roots,public=self._public()
                launch_ids=_installed_launcher_records(observer,fd)
                if launch_ids is not None and launch_ids!={c['role']:c['id'] for c in owners}:
                    raise ValueError('installed_readiness_owner_binding')
                if not isinstance(rid,str) or not re.fullmatch('[0-9a-f]{64}',rid):
                    raise ValueError('installed_readiness_registration')
                name='registration-'+rid+'.json'
                raw,identity=_registration_read_at(fd,name)
                value=recovery.host_json(raw)
                _installed_provenance(value,observer,public)
                if digest(value['registration'])!=rid or value['registration']['containers']!=owners:
                    raise ValueError('installed_readiness_registration')
                first=observer.observe_owners(owners,public)
                if first!=value['observation']: raise ValueError('installed_readiness_observation')
                with ip.parent_fd(roots['ipc']) as (parent,leaf):
                    ipc=os.open(leaf,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                try:
                    hello=_registration_hello(_registration_child('hello',dict(ipc_fd=ipc),(ipc,)),value['registration'])
                    if any(hello[k]!=value['hello'][k] for k in ('lane','boot')):
                        raise ValueError('installed_readiness_hello')
                finally: os.close(ipc)
                check()
                if (_registration_read_at(fd,name)!=(raw,identity) or
                        observer.observe_owners(owners,public)!=first):
                    raise ValueError('installed_readiness_changed')
                return rid
            def register(self, owners):
                check()
                return _register_installed(observer,owners,fd,check)
            def register_and_notify(self, owners):
                # No notification after any failed/unknown registration outcome.
                rid = self.register(owners)
                check()
                return _installed_handoff(observer, owners, rid, fd, check)
        try:
            yield HeldAuthority()
        except _RegistrationChildUnreaped:
            _registration_retained.append((os.dup(lock),os.dup(fd)))
            raise
        finally:
            active=False
            stable()
            if ip.identity(_source_stat(directory/'host-policy.json',directory=False))!=policy_identity:
                raise ValueError('installed_host_policy_changed')
            if _public_file(directory/'host-policy.json',uid=0,gid=0,mode=0o600,limit=65536,
                            deadline=time.monotonic()+2)!=raw:
                raise ValueError('installed_host_policy_changed')
    finally:
        if lock is not None: os.close(lock)
        os.close(fd)



_registration_retained=[]


class _RegistrationChildUnreaped(RuntimeError):
    """Uncertain helper retains duplicate parent exclusions; never success."""


def _registration_child(kind, request, descriptors):
    raw=encode(request).encode()
    if kind not in ('hello','writer') or len(raw)>65536:
        raise ValueError('installed_child_request_budget')
    if sys.platform!='linux' or os.getuid()!=0 or os.geteuid()!=0:
        raise ValueError('installed_linux_root_authority_required')
    with tempfile.TemporaryFile() as output:
        child=subprocess.Popen([sys.executable,'-I','-B','-c',
            'import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent.parent))+');'
            'from service.install_child import registration_entry;registration_entry('+repr(kind)+')'],
            stdin=subprocess.PIPE,stdout=output,stderr=subprocess.DEVNULL,
            close_fds=True,pass_fds=tuple(descriptors),cwd='/',env={})
        try: child.communicate(raw,timeout=10)
        except BaseException as primary:
            try: child.kill()
            except BaseException as cleanup: primary.add_note('registration kill failed: '+str(cleanup))
            try: child.wait(timeout=2)
            except BaseException:
                _registration_retained.append(child)
                raise _RegistrationChildUnreaped('installed_registration_child_unreaped') from primary
            try:
                if child.stdin is not None: child.stdin.close()
            except BaseException as cleanup: primary.add_note('registration pipe close failed: '+str(cleanup))
            raise
        if child.stdin is not None: child.stdin.close()
        if child.returncode!=0: raise ValueError('installed_registration_child_failed')
        output.seek(0); data=output.read(65537)
        if len(data)>65536: raise ValueError('installed_child_output_budget')
        if kind=='writer':
            if data: raise ValueError('installed_writer_output')
            return
        return recovery.host_json(data)


def _registration_read_at(directory,name):
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or
                before.st_uid!=0 or before.st_gid!=0 or stat.S_IMODE(before.st_mode)!=0o600 or
                before.st_size>recovery.HOST_CAPSULE_BYTES):
            raise ValueError('installed_provenance_identity')
        raw=os.read(fd,recovery.HOST_CAPSULE_BYTES+1)
        if (len(raw)!=before.st_size or ip.identity(os.fstat(fd))!=ip.identity(before) or
                ip.identity(os.stat(name,dir_fd=directory,follow_symlinks=False))!=ip.identity(before)):
            raise ValueError('installed_provenance_changed')
        return raw,ip.identity(before)
    finally: os.close(fd)


def _registration_snapshot(roots,scope):
    with recovery.private_database(roots['control']/'core/core.sqlite',policy=ip.SPLIT) as path:
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=0,isolation_level=None)) as db:
            db.execute('PRAGMA query_only=ON'); db.execute('PRAGMA trusted_schema=OFF')
            end=time.monotonic()+2
            db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
            state=recovery.snapshot(roots,db,scope,policy=ip.SPLIT)
    data=state['evidence_projection']['data']
    if (state['faults'] or state['inventory']['reasons'] or not state['allocation_coherent'] or
            any(r['phase']!='terminal' for r in state['journal_records']) or
            data['p1_recovery_plans'] or any(data.get(t) for t in recovery.B2_COLUMNS)):
        raise ValueError('installed_registration_unhealthy_state')
    return state


def _registration_hello(hello,registration=None):
    recovery.closed(hello,'lane boot quarantined launches')
    if (any(not isinstance(hello[k],str) or not re.fullmatch('[0-9a-f]{32}',hello[k]) for k in ('lane','boot')) or
            hello['quarantined'] is not False or type(hello['launches']) is not int or not 0<=hello['launches']<64 or
            (registration is not None and any(hello[k]!=registration[k] for k in ('lane','boot')))):
        raise ValueError('installed_registration_hello_changed')
    return hello


def _installed_provenance(value,observer,public):
    version=value.get('kind')
    recovery.closed(value,'kind registration observation hello ipc_identity'+(' history' if version=='installed-registration-v2' else ''))
    if version not in ('installed-registration-v1','installed-registration-v2'): raise ValueError('installed_provenance_kind')
    if version=='installed-registration-v2':
        history=value['history']; recovery.closed(history,'predecessors state journal')
        state=history['state']
        recovery.closed(state,'prior_epoch deployment_id source_digest affected bytes evidence_projection monitor protocol integrity reservations faults dispatches jobs publications allocation_coherent inventory journal_digest lane journal_records registrations versions apply_available')
        if (history['predecessors']!=sorted(digest(r) for r in state['registrations']) or
                not history['predecessors'] or len(set(history['predecessors']))!=len(history['predecessors']) or
                digest(state['evidence_projection'])!=state['source_digest'] or
                state['deployment_id']!=public['deployment_id']):
            raise ValueError('installed_history_linkage')
        if digest(dict(payload=history['journal'],sha256=digest(history['journal'])))!=state['journal_digest']:
            raise ValueError('installed_history_journal_binding')
        if digest(value['registration']) in history['predecessors']:
            raise ValueError('installed_history_self_link')
        recovery.installed_clean_history(state,history['journal'])
        if json.loads(state['evidence_projection']['data']['p1_installation'][0][3])!=public:
            raise ValueError('installed_history_enrollment')
    registration=value['registration']; recovery.validate_registration(registration)
    if (registration['deployment_id']!=public['deployment_id'] or registration['scope']!=observer.scope or
            registration['evidence']!='actual-host-inspection'):
        raise ValueError('installed_provenance_binding')
    _registration_hello(value['hello'],registration)
    if value['ipc_identity']!=[public['roles']['ipc'][k] for k in ('device','inode')]:
        raise ValueError('installed_provenance_ipc')
    projection=[]
    for current in value['observation']['topology']['consumers']:
        role='control' if any(m['destination']=='/control' for m in current['mounts']) else 'worker'
        if role is None: raise ValueError('installed_provenance_role')
        projection.append(dict(id=current['id'],image=current['image'],started=current['started'],role=role,
            volumes=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging']))
    recovery.validate_installed_observation(value['observation'],observer.scope,projection,
        observer.config,observer.policy,public)
    if any(c not in projection for c in registration['containers']): raise ValueError('installed_provenance_coverage')
    return value


LAUNCH_DOCUMENTS = ('creation-intent.json','created-control.json','created-worker.json','containers.json')


def _installed_launcher_records(observer, authority_fd):
    values={}
    for name in LAUNCH_DOCUMENTS:
        try: raw,_=_registration_read_at(authority_fd,name)
        except FileNotFoundError: continue
        value=recovery.host_json(raw)
        if raw!=encode(value).encode(): raise ValueError('installed_launch_document_changed')
        values[name]=value
    if not values: return None  # Earlier protected registration adapters unchanged.
    if set(values)!=set(LAUNCH_DOCUMENTS): raise ValueError('installed_partial_creation')
    expected=dict(version=1,configuration_digest=digest(observer.config))
    if values['creation-intent.json']!=expected: raise ValueError('installed_creation_binding')
    containers=values['containers.json']
    recovery.closed(containers,'version configuration_digest containers')
    if {k:containers[k] for k in expected}!=expected: raise ValueError('installed_creation_binding')
    recovery.closed(containers['containers'],'control worker')
    ids=containers['containers']
    if len(set(ids.values()))!=2 or any(not isinstance(i,str) or not recovery.HEX.fullmatch(i) for i in ids.values()):
        raise ValueError('installed_creation_identity')
    for role in ('control','worker'):
        if values['created-'+role+'.json']!=dict(id=ids[role]): raise ValueError('installed_creation_identity')
    return ids


def _installed_history_read(observer, public, authority_fd, registrations):
    """Validate retained historical claims, without pretending to inspect old boots."""
    _installed_launcher_records(observer,authority_fd)
    retained={digest(r):r for r in registrations}; provenance={}; identities={}; retained_bytes=0
    with os.scandir(authority_fd) as entries:
        for index,entry in enumerate(entries):
            if index>=recovery.HOST_OWNER_LIMIT+2+len(LAUNCH_DOCUMENTS): raise ValueError('installed_authority_budget')
            if entry.name in ('launch.lock','host-policy.json',*LAUNCH_DOCUMENTS): continue
            if not re.fullmatch(r'registration-[0-9a-f]{64}\.json',entry.name):
                raise ValueError('installed_unknown_authority_file')
            raw,identity=_registration_read_at(authority_fd,entry.name)
            retained_bytes+=len(raw)
            if retained_bytes>recovery.HOST_CAPSULE_BYTES: raise ValueError('installed_provenance_budget')
            value=_installed_provenance(recovery.host_json(raw),observer,public)
            rid=digest(value['registration'])
            if entry.name!='registration-'+rid+'.json' or raw!=encode(value).encode():
                raise ValueError('installed_provenance_changed')
            provenance[rid]=raw; retained[rid]=value['registration']
            identities[rid]=identity
    if any(digest(r) not in provenance for r in registrations):
        raise ValueError('installed_missing_provenance')
    _installed_history_graph(retained,provenance)
    return retained,provenance,identities,retained_bytes


def _installed_history_graph(retained, provenance):
    """Bounded DAG of exact retained pre-insert registration closures."""
    limit=recovery.HOST_OWNER_LIMIT
    if len(retained)>limit or len(provenance)>limit: raise ValueError('installed_history_graph_budget')
    values={rid:recovery.host_json(raw) for rid,raw in provenance.items()}
    edges={rid:value.get('history',{}).get('predecessors',[]) for rid,value in values.items()}
    if sum(map(len,edges.values()))>limit*limit: raise ValueError('installed_history_graph_budget')
    for rid,predecessors in edges.items():
        if any(p not in values for p in predecessors): raise ValueError('installed_history_missing_predecessor')
        if any(digest(r) not in retained or retained[digest(r)]!=r
                for r in values[rid].get('history',{}).get('state',{}).get('registrations',[])):
            raise ValueError('installed_history_missing_predecessor')
    # Iterative topological elimination: bounded nodes, edges AND depth, no
    # recursion on untrusted historical claims.
    done=set();depth={}
    for _ in range(limit):
        ready=[rid for rid,parents in edges.items() if rid not in done and set(parents)<=done]
        if not ready: break
        if done and len(ready)!=1: raise ValueError('installed_history_preinsert_branch')
        for rid in ready:
            if edges[rid] and set(edges[rid])!=done: raise ValueError('installed_history_preinsert_closure')
        for rid in ready:
            parents=edges[rid];depth[rid]=1+max((depth[p] for p in parents),default=0)
            if depth[rid]>limit: raise ValueError('installed_history_graph_budget')
            done.add(rid)
    if done!=set(values): raise ValueError('installed_history_cycle')
    for rid,parents in edges.items():
        value=values[rid];new=value['registration']
        for pid in parents:
            old=values[pid]['registration']
            if 'history' in values[pid]:
                previous=values[pid]['history'];current=value['history']
                if (current['journal']['records'][:len(previous['journal']['records'])]!=previous['journal']['records'] or
                        current['state']['evidence_projection']['schema']!=previous['state']['evidence_projection']['schema']):
                    raise ValueError('installed_history_predecessor_evidence')
            if not set(edges[pid])<=set(parents): raise ValueError('installed_history_preinsert_closure')
            oldowners={c['id']:c for c in old['containers']}
            changed=False
            for owner in new['containers']:
                previous=oldowners.get(owner['id'])
                if previous is None or any(previous[k]!=owner[k] for k in ('image','role','volumes')):
                    raise ValueError('installed_history_binding')
                before=recovery.host_timestamp(previous['started']);after=recovery.host_timestamp(owner['started'])
                if before>after: raise ValueError('installed_history_future_prestate')
                changed |= before<after
                if owner['role']=='worker' and ((before==after)!=(old['boot']==new['boot'])):
                    raise ValueError('installed_history_boot_order')
            if not changed or old['lane']!=new['lane']:
                raise ValueError('installed_history_replay')
            # Bind process claims to the exact predecessor provenance, not just
            # a registration that happens to be present in the final directory.
            for process in values[pid]['observation']['processes']:
                actual=next((p for p in value['observation']['processes'] if p['id']==process['id']),None)
                if actual is not None and actual['started']!=process['started'] and (
                        actual['start_ticks']==process['start_ticks'] or actual['pid_namespace']==process['pid_namespace']):
                    raise ValueError('installed_history_process_reused')
    return True


def _installed_current(observer, registrations, scope, deployment_id):
    claims={}
    for record in registrations:
        for owner in record["containers"]:
            if owner["id"] in claims and any(claims[owner["id"]][k]!=owner[k] for k in ("image","role","volumes")):
                raise ValueError("installed_history_binding")
            claims[owner["id"]]=owner
    if len(claims)>recovery.HOST_OWNER_LIMIT: raise ValueError("host_owner_budget")
    owners=observer.current_owners([claims[k] for k in sorted(claims)])
    current=[dict(r,containers=[next(c for c in owners if c["id"]==old["id"]) for old in r["containers"]]) for r in registrations]
    current=list({digest(r):r for r in current}.values())
    return recovery.installed_history_current(registrations,scope,deployment_id,current)


def _register_installed(observer,selected,authority_fd,check,*,child=_registration_child,fault=lambda _:None):
    """Protected held authority only. No caller-supplied observation receipt.

    Core exclusion spans private copies, children/reap and the post-insert bracket.
    Provenance-only interruptions remain in the complete retained owner union.
    """
    selected=json.loads(encode(selected))
    launch_ids=_installed_launcher_records(observer,authority_fd)
    if launch_ids is not None and {c['role']:c['id'] for c in selected}!=launch_ids:
        raise ValueError('installed_launch_container_binding')
    roots=ip.roots_checked({r:v['source'] for r,v in observer.policy['roots'].items()},policy=ip.SPLIT)
    with ip.locked_file(roots['control']/'core/control.lock',policy=ip.SPLIT,role='control') as lock_fd:
        with ExitStack() as stack:
            try:
                fds={}
                for role in ('control','staging','ipc'):
                    with ip.parent_fd(roots[role]) as (parent,name):
                        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                    stack.callback(os.close,fd); fds[role]=fd
                    ip.check(os.fstat(fd),ip.SPLIT,role,directory=True)
                    expected=observer.policy['roots'][role]
                    if (os.fstat(fd).st_dev,os.fstat(fd).st_ino)!=(expected['device'],expected['inode']):
                        raise ValueError('installed_registration_roots_changed')
                check(); before=_registration_snapshot(roots,observer.scope)
                public=json.loads(before['evidence_projection']['data']['p1_installation'][0][3])
                ip.host_enrollment(public,observer.policy,observer.config,observer.scope)
                retained,provenance,identities,retained_bytes=_installed_history_read(observer,public,authority_fd,before['registrations'])
                candidate=dict(deployment_id=public['deployment_id'],scope=observer.scope,containers=selected,
                               lane=before['lane'],boot='0'*32,evidence='actual-host-inspection')
                recovery.validate_registration(candidate)
                # Retained claims are immutable. Inspect the current identity for
                # every ID before deriving the one current projection.
                owners=_installed_current(observer,list({digest(r):r for r in [*retained.values(),candidate]}.values()),observer.scope,public['deployment_id'])
                if any(c not in owners for c in selected): raise ValueError('installed_selected_not_current')
                changed=any(c not in owners for r in retained.values() for c in r['containers'])
                from worker.runtime_journal import read_snapshot
                journal,maintenance=read_snapshot(roots['receipts'],policy=ip.SPLIT)
                if maintenance is not None or len(journal['records'])>=64:
                    raise ValueError('installed_history_lane_budget')
                if digest(dict(payload=journal,sha256=digest(journal)))!=before['journal_digest']:
                    raise ValueError('installed_history_journal_changed')
                history=None
                if changed:
                    if any(digest(r) not in {digest(x) for x in before['registrations']} and
                           any(c not in owners for c in r['containers']) for r in retained.values()):
                        raise ValueError('installed_history_unregistered_provenance')
                    recovery.installed_clean_history(before,journal)
                    history=dict(predecessors=sorted(retained),state=before,journal=journal)
                first=observer.observe_owners(owners,public);check()
                live={c['id']:c for c in first['topology']['consumers'] if c['running']}
                if (len(selected)!=2 or {c['role'] for c in selected}!={'control','worker'} or
                        any(c['id'] not in live for c in selected)):
                    raise ValueError('installed_registration_requires_live_pair')
                if changed:
                    for raw in provenance.values():
                        old=recovery.host_json(raw)
                        for process in old['observation']['processes']:
                            actual=next((p for p in first['processes'] if p['id']==process['id']),None)
                            if actual is not None and actual['started']!=process['started'] and (
                                    actual['start_ticks']==process['start_ticks'] or actual['pid_namespace']==process['pid_namespace']):
                                raise ValueError('installed_history_process_reused')
                hello=_registration_hello(child('hello',dict(ipc_fd=fds['ipc']),(fds['ipc'],)))
                if hello['lane']!=before['lane']: raise ValueError('installed_registration_lane')
                candidate.update(lane=hello['lane'],boot=hello['boot']); rid=digest(candidate)
                for old in retained.values():
                    old_worker=next(c for c in old['containers'] if c['role']=='worker')
                    new_worker=next(c for c in candidate['containers'] if c['role']=='worker')
                    if old_worker==new_worker and old['boot']!=hello['boot']:
                        raise ValueError('installed_history_boot_without_restart')
                    if old_worker['id']==new_worker['id'] and old_worker['started']!=new_worker['started'] and old['boot']==hello['boot']:
                        raise ValueError('installed_history_worker_boot')

                if rid not in retained and len(retained)>=recovery.HOST_OWNER_LIMIT: raise ValueError('host_registration_budget')
                value=dict(kind='installed-registration-v1',registration=candidate,observation=first,hello=hello,
                           ipc_identity=[public['roles']['ipc'][k] for k in ('device','inode')])
                if rid in provenance:
                    previous=recovery.host_json(provenance[rid])
                    if previous['kind']=='installed-registration-v2':
                        value.update(kind='installed-registration-v2',history=previous['history'])
                elif history is not None:
                    value.update(kind='installed-registration-v2',history=history)
                _installed_provenance(value,observer,public)
                _installed_history_graph({**retained,rid:candidate},{**provenance,rid:encode(value).encode()})
                if observer.observe_owners(owners,public)!=first: raise ValueError('installed_registration_prebracket_changed')
                if _registration_hello(child('hello',dict(ipc_fd=fds['ipc']),(fds['ipc'],)),candidate)!=hello:
                    raise ValueError('installed_registration_hello_changed')
                check(); data=encode(value).encode(); name='registration-'+rid+'.json'
                if rid not in provenance and retained_bytes+len(data)>recovery.HOST_CAPSULE_BYTES:
                    raise ValueError('installed_provenance_budget')
                if rid in provenance:
                    if provenance[rid]!=data: raise ValueError('installed_provenance_changed')
                else:
                    fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=authority_fd)
                    with os.fdopen(fd,'wb') as out:
                        info=os.fstat(out.fileno())
                        if (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode))!=(0,0,0o600) or info.st_nlink!=1:
                            raise ValueError('installed_provenance_identity')
                        if out.write(data)!=len(data): raise OSError('installed_provenance_short_write')
                        out.flush();os.fsync(out.fileno())
                        identities[rid]=ip.identity(os.fstat(out.fileno()))
                    os.fsync(authority_fd)
                provenance[rid]=data
                def retained_exact():
                    for identity,raw in provenance.items():
                        leaf='registration-'+identity+'.json'
                        if _registration_read_at(authority_fd,leaf)!=(raw,identities[identity]):
                            raise ValueError('installed_provenance_changed')
                retained_exact();fault('provenance');check()
                prestate=[list(r) for r in before['evidence_projection']['data']['p1_registrations']]
                request=dict(public=public,registration=candidate,prestate=prestate,
                    control_fd=fds['control'],staging_fd=fds['staging'],lock_fd=lock_fd)
                if len(encode(request).encode())>65536: raise ValueError('installed_child_request_budget')
                if _registration_snapshot(roots,observer.scope)['source_digest']!=before['source_digest']:
                    raise ValueError('installed_registration_prestate_changed')
                child('writer',request,(fds['control'],fds['staging'],lock_fd))
                fault('insert');check()
                after=_registration_snapshot(roots,observer.scope)
                expected=prestate if any(r[0]==rid for r in prestate) else prestate+[[rid,encode(candidate),rid]]
                if [list(r) for r in after['evidence_projection']['data']['p1_registrations']]!=expected:
                    raise ValueError('installed_registration_poststate')
                def stable(state):
                    projection=json.loads(encode(state['evidence_projection']))
                    projection['data'].pop('p1_registrations')
                    return projection
                if stable(before)!=stable(after): raise ValueError('installed_registration_state_changed')
                if observer.observe_owners(owners,public)!=first: raise ValueError('installed_registration_postbracket_changed')
                if _registration_hello(child('hello',dict(ipc_fd=fds['ipc']),(fds['ipc'],)),candidate)!=hello:
                    raise ValueError('installed_registration_hello_changed')
                retained_exact()
                fault('bracket');check()
                return rid
            except _RegistrationChildUnreaped:
                _registration_retained.append(os.dup(lock_fd))
                raise


if __name__=='__main__': main()



def _installed_handoff(observer, owners, rid, authority_fd, check, *, notify=None):
    """Held root authority only: project successful retained registration to control.

    Does not claim that an owner/listener started. Full provenance remains private.
    No receipt keys, seed, credential contents or authority paths cross the exec.
    """
    from service.entrypoint import projection
    if not isinstance(rid,str) or not re.fullmatch('[0-9a-f]{64}',rid):
        raise ValueError('installed_notification_registration')
    name = 'registration-'+rid+'.json'
    check(); raw,identity = _registration_read_at(authority_fd,name)
    value = recovery.host_json(raw)
    roots = {r:Path(v['source']) for r,v in observer.policy['roots'].items()}
    state = _registration_snapshot(roots,observer.scope)
    public = json.loads(state['evidence_projection']['data']['p1_installation'][0][3])
    _installed_provenance(value,observer,public)
    record = value['registration']
    if (digest(record) != rid or record['containers'] != owners or
            record not in state['registrations']):
        raise ValueError('installed_notification_registration')
    retained,provenance,identities,_=_installed_history_read(observer,public,authority_fd,state['registrations'])
    all_owners = _installed_current(observer,list(retained.values()),observer.scope,public['deployment_id'])
    def history_exact():
        for identity,raw in provenance.items():
            if _registration_read_at(authority_fd,'registration-'+identity+'.json')!=(raw,identities[identity]):
                raise ValueError('installed_provenance_changed')
    history_exact()
    if observer.observe_owners(all_owners,public) != value['observation']:
        raise ValueError('installed_notification_incarnation_changed')
    selected = {c['id'] for c in record['containers']}
    gate = projection(dict(version=1,registration_id=rid,registration=record,
        configuration_digest=digest(observer.config),enrollment_digest=digest(public),
        provenance_digest=digest(value),processes=[p for p in value['observation']['processes'] if p['id'] in selected]))
    check()
    if _registration_read_at(authority_fd,name) != (raw,identity):
        raise ValueError('installed_notification_provenance_changed')
    control = next(c['id'] for c in owners if c['role']=='control')
    (notify or _installed_notify)(observer.scope['context'],control,encode(gate).encode())
    check()
    history_exact()
    if (_registration_read_at(authority_fd,name) != (raw,identity) or
            observer.observe_owners(all_owners,public) != value['observation']):
        raise ValueError('installed_notification_postbracket_changed')
    return rid  # Notification delivered only; never an installed-startup receipt.


def _installed_notify(context, control, raw):
    from service.entrypoint import decode_projection
    decode_projection(raw)
    Docker(context)  # Reuse closed context parser; no environment-selected daemon.
    if not isinstance(control,str) or not recovery.HEX.fullmatch(control):
        raise ValueError('installed_notification_control')
    command = ['docker','--context',context,'exec','--user','10001:10001','-i',control,
        'python','-I','-B','-c',
        "import sys;sys.path.insert(0,'/opt/speech');from service.entrypoint import publish,GATE_LIMIT;publish(sys.stdin.buffer.read(GATE_LIMIT+1))"]
    output = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.DEVNULL,
                                   close_fds=True,env={'PATH':os.defpath},cwd='/')
        try:
            process.communicate(raw,timeout=5)
        except BaseException as primary:
            try: process.kill()
            except BaseException as cleanup: primary.add_note('notification kill failed: '+type(cleanup).__name__)
            try: process.wait(timeout=2)
            except BaseException:
                _registration_retained.append(process)
                raise _RegistrationChildUnreaped('installed_notification_child_unreaped') from primary
            raise
        finally:
            primary = sys.exception()
            try:
                if process.stdin is not None: process.stdin.close()
            except BaseException as cleanup:
                if primary is None: raise
                primary.add_note('notification pipe close failed: '+type(cleanup).__name__)
        if process.returncode != 0 or os.fstat(output.fileno()).st_size:
            raise ValueError('installed_notification_failed')
    finally:
        primary = sys.exception()
        try: output.close()
        except BaseException as cleanup:
            if primary is None: raise
            primary.add_note('notification output close failed: '+type(cleanup).__name__)


def create_installed_authority(config, scope, policy):
    """One explicit create-only protected policy/lock provisioning boundary.

    Inputs and named volumes must already be explicitly provisioned with their
    required identities/modes. Never copy, read or log the credential contents.
    Existing authority refuses; partial creation is retained for operator review.
    """
    if sys.platform!='linux' or os.getuid()!=0 or os.geteuid()!=0:
        raise ValueError('installed_linux_root_authority_required')
    config=ip.configuration(config); policy=ip.host_policy(policy,config,scope)
    observer=InstalledObserver(Docker(scope['context']),config,scope,policy)
    observer.initialize_authority()  # Exact inputs, empty consumer set, mount proof.
    directory=Path(recovery.HOST_AUTHORITY_DIRECTORY)/'installed'/digest(scope)
    for parent in reversed(directory.parents):
        try: parent.mkdir(mode=0o700)
        except FileExistsError: pass
        with ip.parent_fd(parent/'__probe__') as (fd,_):
            info=os.fstat(fd)
            if info.st_uid!=0 or stat.S_IMODE(info.st_mode)&0o022:
                raise ValueError('installed_unsafe_authority')
    directory.mkdir(mode=0o700)  # No adoption of an existing installation leaf.
    fd=_root_authority_directory(directory)
    try:
        _host_write(directory/'host-policy.json',policy)
        lock=os.open('launch.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        try:
            info=os.fstat(lock)
            if (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode),info.st_nlink)!=(0,0,0o600,1):
                raise ValueError('installed_launch_lock_policy')
            os.fsync(lock)
        finally: os.close(lock)
        os.fsync(fd)
        recovery.sync_dir(directory.parent)
    finally: os.close(fd)
