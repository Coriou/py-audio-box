#!/usr/bin/env python3
"""Host-only inspection and explicitly selected bounded freeze; apply unavailable."""
import argparse
import json
import os
import re
import stat
import sqlite3
from contextlib import closing
import uuid
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
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
    return (st.st_dev,st.st_ino) in {(v['device'],v['inode']) for v in volumes}


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
    parser.add_argument('command',choices=('register','inspect','plan','preflight','freeze','apply'))
    parser.add_argument('--request',required=True)
    args=parser.parse_args()
    if args.command=='apply': recovery.apply()
    request=recovery.host_json(read_file(Path(args.request)))
    if args.command in ('preflight','freeze'):
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


if __name__=='__main__': main()
