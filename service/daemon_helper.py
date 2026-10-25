"""Finite networkless volume helper, with held authority exclusion.

Docker actions stay in the ordinary client. Every prescribed action is bracketed
by nonce/phase/step frames while this process holds the volume lock. Incomplete
intent is a durable refusal, never a request to repair or retry a mutation.
"""
import base64
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
import time

from service import daemon_volumes as dv, install_policy as ip
from service.core import encode
from service.daemon_client import Wire, encoded, manifest, collect, INPUT_CHUNK
from service.persistent_init import digest

MAX_HELPERS=128


def gate_projection(value, *, family=None, configuration=None, binding=None, public=None):
    from service import recovery, installed_export as ex
    export=family is ex.FAMILY
    if family is not None and not export: raise ValueError('explicit_export_family_required')
    policy=ex.LAUNCH if export else dv.POLICY
    evidence=ex.EVIDENCE if export else dv.EVIDENCE
    dv.closed(value,'version launch_policy registration_id registration configuration_digest enrollment_digest provenance_digest observation')
    if type(value['version']) is not int or value['version']!=(2 if export else 1) or value['launch_policy']!=policy or len(encoded(value))>65536:
        raise ValueError('daemon_gate_version_budget')
    for k in ('registration_id','configuration_digest','enrollment_digest','provenance_digest'):
        if not isinstance(value[k],str) or not dv.HEX.fullmatch(value[k]): raise ValueError('daemon_gate_digest')
    r=value['registration']
    (ex.host_registration if export else recovery.validate_registration)(r)
    if digest(r)!=value['registration_id'] or r['evidence']!=evidence or len(r['containers'])!=2 or {c['role'] for c in r['containers']}!={'control','worker'}:
        raise ValueError('daemon_gate_registration')
    observation=value['observation']; dv.closed(observation,'kind binding inspections local hello')
    if observation['kind']!=evidence: raise ValueError('daemon_gate_evidence')
    dv.closed(observation['local'],'control worker'); dv.closed(observation['inspections'],'control worker')
    for role in ('control','worker'): dv.local(observation['local'][role],role,family=family)
    from service.host_install import _registration_hello
    _registration_hello(observation['hello'],r,family=family,configuration=configuration)
    if export:
        ex.installation(public)
        if (digest(configuration)!=value['configuration_digest'] or digest(public)!=value['enrollment_digest'] or
                ex._encoded(public['configuration'])!=ex._encoded(configuration)):
            raise ValueError('daemon_gate_configuration_enrollment')
        dv.observation(observation,r,configuration,binding,public,family=family)
    return value


def child(kind, request, descriptors, *, family=None):
    """Retain all enclosing exclusions through actual exit/reap, even on failure."""
    from service import installed_export as ex
    if family is not None and family is not ex.FAMILY: raise ValueError('explicit_export_family_required')
    entry='export_container_entry' if family is not None else 'container_entry'
    if kind not in ('core','hello','writer'): raise ValueError('daemon_child_kind')
    raw=encoded(request)
    if len(raw)>65536: raise ValueError('daemon_child_budget')
    deadline=time.monotonic()+30
    p=subprocess.Popen([sys.executable,'-I','-B','-c',
        "import sys;sys.path.insert(0,'/opt/speech');from service.install_child import "+entry+";"+entry+"("+repr(kind)+')'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,close_fds=True,
        pass_fds=tuple(descriptors),cwd='/',env={},bufsize=0)
    result=collect(p,raw,65536,deadline,'daemon_child')
    if kind=='hello': return json.loads(result)
    if result: raise ValueError('daemon_writer_unexpected_output')


class Helper:
    def __init__(self,phase,nonce,request,wire,*,roots=None,child_runner=child,family=None):
        dv.closed(request,'configuration binding helper manifest')
        self.phase,self.nonce,self.wire=phase,nonce,wire
        from service import installed_export as ex
        self.family=family; self.export=family is ex.FAMILY
        self.policy=ex.LAUNCH if self.export else dv.POLICY
        self.evidence=ex.EVIDENCE if self.export else dv.EVIDENCE
        self.config=ip.family_configuration(request['configuration'],family); self.b=dv.binding(request['binding'],family=family)
        if self.config['ownership_policy']!=(ex.POLICY if self.export else ip.POLICY) or digest(self.config)!=self.b['configuration_digest']:
            raise ValueError('daemon_kokoro_configuration_required')
        if (self.b['cpu']['worker']<max(c['execution']['descriptor']['threads'] for c in self.config['contexts'].values())*10**9 or
                any(t['bytes']>self.config['budget']['ram_bytes'] for t in self.b['tmpfs'].values())):
            raise ValueError('daemon_resource_budget')
        self.own=dv.inspect_container(request['helper'],'helper',self.config,self.b,phase=phase,nonce=nonce,family=family)
        self.manifest=request['manifest']; self.root={r:Path(dv.ROOTS[r]) for r in dv.HELPER_ROLES[phase]} if roots is None else roots
        self.child=(lambda kind,request,fds:child(kind,request,fds,family=family)) if child_runner is child else child_runner; self.fds={}; self.original={}; self.scope=dv.scope(self.b,family=family)
        self.authority=self.root['authority']; self.end=time.monotonic()+100

    def rpc(self,action,payload):
        if time.monotonic()>self.end: raise ValueError('daemon_helper_deadline')
        self.stable(); self.wire.send(action,payload)
        returned,value=self.wire.receive()
        if returned!=action: raise ValueError('daemon_phase_action_mismatch')
        self.stable(); return value

    def stable(self):
        for role,fd in self.fds.items():
            s=os.fstat(fd); p=self.root[role]
            with ip.parent_fd(p) as (parent,name): current=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if (s.st_dev,s.st_ino)!=self.original[role] or (current.st_dev,current.st_ino)!=self.original[role]:
                raise ValueError('daemon_helper_root_changed')

    def write(self,name,value):
        if '/' in name or len(name)>128: raise ValueError('daemon_authority_leaf')
        raw=encoded(value)
        from service.recovery import HOST_CAPSULE_BYTES
        if len(raw)>HOST_CAPSULE_BYTES: raise ValueError('daemon_authority_record_budget')
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fds['authority'])
        with os.fdopen(fd,'wb') as f:
            before=os.fstat(f.fileno())
            if (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode),before.st_nlink)!=(0,0,0o600,1):
                raise ValueError('daemon_authority_file_identity')
            f.write(raw); f.flush(); os.fsync(f.fileno())
            after=os.fstat(f.fileno())
            if ip.identity(os.stat(name,dir_fd=self.fds['authority'],follow_symlinks=False))!=ip.identity(after):
                raise ValueError('daemon_authority_file_changed')
        os.fsync(self.fds['authority']); self.stable()

    def read(self,name):
        from service.host_install import _registration_read_at
        raw,_=_registration_read_at(self.fds['authority'],name)
        value=json.loads(raw)
        if raw!=encoded(value): raise ValueError('daemon_authority_encoding')
        return value

    def exists(self,name):
        try: self.read(name); return True
        except FileNotFoundError: return False

    def terminal_helpers(self):
        return {self.read(n)['helper'] for n in ip.directory_entries(self.fds['authority']) if n.startswith('operation-') and n.endswith('.done')}

    def snapshot(self):
        value=self.rpc('snapshot',{})
        dv.closed(value,'daemon volumes consumers')
        if value['daemon']!=self.b['daemon'] or value['volumes']!=self.b['volumes']:
            raise ValueError('daemon_snapshot_binding')
        known={self.own['Id']: (self.phase,self.nonce,self.own['State']['StartedAt'])}
        for name in ip.directory_entries(self.fds['authority']):
            if name.startswith('operation-') and name.endswith('.done'):
                terminal=self.read(name); dv.closed(terminal,'phase nonce helper started result')
                known[terminal['helper']]=(terminal['phase'],terminal['nonce'],terminal['started'])
        ids={}
        if self.exists('containers.json'): ids=self.read('containers.json')
        else:
            for role in ('control','worker'):
                if self.exists('created-'+role+'.json'): ids[role]=self.read('created-'+role+'.json')['id']
        seen=set()
        if type(value['consumers']) is not list or len(value['consumers'])>MAX_HELPERS+3: raise ValueError('daemon_consumer_budget')
        for c in value['consumers']:
            identity=c['Id']
            if identity in seen: raise ValueError('daemon_duplicate_consumer')
            seen.add(identity)
            if identity in known:
                phase,nonce,started=known[identity]
                dv.inspect_container(c,'helper',self.config,self.b,phase=phase,nonce=nonce,family=self.family)
                if c['State']['StartedAt']!=started: raise ValueError('daemon_helper_incarnation_changed')
                if identity!=self.own['Id'] and (c['State']['Running'] is not False or c['State']['ExitCode']!=0):
                    raise ValueError('daemon_prior_helper_not_terminal')
            elif identity in ids.values():
                role=next(r for r,i in ids.items() if i==identity)
                dv.inspect_container(c,role,self.config,self.b,family=self.family)
            else: raise ValueError('daemon_unexplained_volume_consumer')
        if not set(known).union(ids.values())<=seen: raise ValueError('daemon_missing_retained_consumer')
        return value

    @contextmanager
    def held(self):
        with ExitStack() as stack:
            # Literal role roots; their daemon Mountpoints are never client paths.
            for role in dv.HELPER_ROLES[self.phase]:
                path=self.root[role]
                with ip.parent_fd(path) as (parent,name):
                    fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                stack.callback(os.close,fd); s=os.fstat(fd)
                if not stat.S_ISDIR(s.st_mode) or (s.st_dev,s.st_ino) in self.original.values():
                    raise ValueError('daemon_helper_distinct_roots')
                self.fds[role]=fd; self.original[role]=(s.st_dev,s.st_ino)
            try: fcntl.flock(self.fds['authority'],fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('daemon_launch_lock_busy') from None
            # Real mountinfo corroborates exact mounted destinations and no child
            # mounts/aliases. Called by installed entrypoint; fixtures adapt only
            # this filesystem observation, never manufacture host PID evidence.
            self.mounts_checked()
            authority=os.fstat(self.fds['authority'])
            if self.phase=='install':
                if ip.directory_entries(self.fds['authority']): raise ValueError('daemon_new_authority_required')
                if (authority.st_uid,authority.st_gid,stat.S_IMODE(authority.st_mode)) not in ((0,0,0o755),(0,0,0o700)):
                    raise ValueError('daemon_fresh_authority_identity')
                # First exact snapshot can only contain this helper. Bootstrap
                # normalization intent is durably written before chmod.
                self.write('normalization-intent.json',dict(binding=self.b))
                self.snapshot()
                os.fchmod(self.fds['authority'],0o700); os.fsync(self.fds['authority'])
                self.write('binding.json',self.b)
                self.write('launch.lock',dict(policy=self.policy))
            else:
                if (authority.st_uid,authority.st_gid,stat.S_IMODE(authority.st_mode))!=(0,0,0o700) or self.read('binding.json')!=self.b:
                    raise ValueError('daemon_authority_binding')
            if self.read('launch.lock')!=dict(policy=self.policy): raise ValueError('daemon_launch_lock_binding')
            stack.enter_context(ip.locked_file(self.authority/'launch.lock',policy=ip.SPLIT,role='receipts'))
            entries=ip.directory_entries(self.fds['authority'])
            import re
            allowed={'launch.lock','normalization-intent.json','binding.json','import-intent.json','import-done.json',
                'roots-normalization-intent.json','init-done.json','containers.json','container-creation-intent.json',
                'created-control.json','created-worker.json'}
            if len(entries)>1024 or any(n not in allowed and not re.fullmatch(r'(operation-[0-9a-f]{32}\.(intent|done)|registration-[0-9a-f]{64}\.json|stopped-[0-9a-f]{32}\.json)',n) for n in entries):
                raise ValueError('daemon_unknown_authority_entry')
            total=0
            for n in entries:
                st=os.stat(n,dir_fd=self.fds['authority'],follow_symlinks=False)
                total+=st.st_size
                if total>64*1024**2: raise ValueError('daemon_authority_budget')
                self.read(n)
            intents=[n for n in entries if n.startswith('operation-') and n.endswith('.intent')]
            if len(intents)>=MAX_HELPERS: raise ValueError('daemon_terminal_helper_budget')
            if any(n[:-7]+'.done' not in entries for n in intents): raise ValueError('daemon_incomplete_operation_retained')
            self.snapshot()
            self.write('operation-'+self.nonce+'.intent',dict(phase=self.phase,nonce=self.nonce,helper=self.own['Id'],started=self.own['State']['StartedAt']))
            yield
            self.snapshot(); self.stable()

    def mounts_checked(self):
        from service.recovery import _host_mount_topology
        observed=_host_mount_topology()
        targets={m['target']:m for m in observed}
        for role in dv.HELPER_ROLES[self.phase]:
            path=str(self.root[role])
            if path not in targets or 'rw' not in targets[path]['options'] or any(m['target'].startswith(path+'/') for m in observed):
                raise ValueError('daemon_helper_mountinfo')
        selected=[targets[str(self.root[r])] for r in dv.HELPER_ROLES[self.phase]]
        if any(a['device']==b['device'] and (a['root']==b['root'] or a['root'].startswith(b['root'].rstrip('/')+'/')) for a in selected for b in selected if a is not b):
            raise ValueError('daemon_helper_mount_alias')
        # Docker's inspected role map was checked independently in __init__.
        # Neither filesystem id nor daemon Source is treated as a host identity.

    def input_import(self):
        # Manifest validation is shared with client; source paths never enter helper.
        entries=manifest([dict(e,source='unused') for e in self.manifest])
        self.write('import-intent.json',dict(files=[{k:v for k,v in e.items() if k!='source'} for e in entries]))
        for role in ('models','config','credential'):
            fd=self.fds[role]; s=os.fstat(fd)
            if ip.directory_entries(fd) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) not in ((0,0,0o755),(0,0,0o700)):
                raise ValueError('daemon_empty_input_volume_required')
            os.fchmod(fd,0o700); os.fsync(fd)
        directories=set()
        for index,e in enumerate(entries):
            role=e['role']; parts=e['path'].split('/'); root=self.root[role]
            path=root
            for part in parts[:-1]:
                path=path/part
                if not path.exists(): path.mkdir(mode=0o700); directories.add(path)
                if path.is_symlink() or not path.is_dir(): raise ValueError('daemon_input_ancestry')
            path=path/parts[-1]
            with ip.parent_fd(path) as (parent,name):
                fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
                with os.fdopen(fd,'wb') as f:
                    offset=0; hashed=hashlib.sha256()
                    while offset<e['bytes']:
                        response=self.rpc('input',dict(index=index,offset=offset)); dv.closed(response,'data')
                        raw=base64.b64decode(response['data'],validate=True)
                        if not 1<=len(raw)<=min(INPUT_CHUNK,e['bytes']-offset): raise ValueError('daemon_input_stream_budget')
                        f.write(raw); hashed.update(raw); offset+=len(raw)
                    if role!='credential' and hashed.hexdigest()!=e['sha256']: raise ValueError('daemon_input_content_changed')
                    os.fchown(f.fileno(),10001 if role=='models' else 0,10001)
                    os.fchmod(f.fileno(),{'models':0o444,'config':0o440,'credential':0o640}[role])
                    f.flush(); os.fsync(f.fileno())
                os.fsync(parent)
        for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                os.fchown(fd,10001,10001); os.fchmod(fd,0o555); os.fsync(fd)
            finally: os.close(fd)
        for role in ('models','config','credential'):
            os.fchown(self.fds[role],10001 if role=='models' else 0,10001)
            os.fchmod(self.fds[role],0o555 if role=='models' else 0o750); os.fsync(self.fds[role])
        from service.host_install import _model_tree, _public_file
        model_digest=_model_tree(self.root['models'],self.config,time.monotonic()+30)
        config_raw=_public_file(self.root['config']/'install.json',uid=0,gid=10001,mode=0o440,limit=65536,deadline=time.monotonic()+2)
        if config_raw!=encoded(self.config): raise ValueError('daemon_config_content')
        from service.auth import installed_credentials
        # Existing credential parser validates closed principals/scopes; bytes and
        # credential digest never leave this phase or enter an authority record.
        credentials=installed_credentials(self.root['credential']/'control.key',self.config['contexts'])
        if self.export:
            admitted={c.principal for c in credentials._credentials if {'read','production-write'} <= c.scopes}
            if not set(self.config['audio_export']['principals']) <= admitted:
                raise ValueError('daemon_export_credential_principals')
        result=dict(identities={r:list(self.original[r]) for r in ('models','config','credential')},model_digest=model_digest)
        self.write('import-done.json',result)
        return dict(phase='imported')

    def initialize(self):
        if not self.exists('import-done.json') or self.exists('init-done.json'): raise ValueError('daemon_new_initialization_required')
        self.write('roots-normalization-intent.json',dict(roots={r:list(self.original[r]) for r in ip.ROLES}))
        for role in ip.ROLES:
            fd=self.fds[role]; s=os.fstat(fd)
            if ip.directory_entries(fd) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) not in ((0,0,0o755),(0,0,0o700)):
                raise ValueError('daemon_fresh_data_volume_required')
            os.fchmod(fd,0o700); os.fsync(fd)
        from service.persistent_init import initialize_split
        def observe():
            snapshot=self.snapshot()
            if any(c['Id'] != self.own['Id'] and c['Id'] not in self.terminal_helpers() for c in snapshot['consumers']):
                raise ValueError('daemon_initializer_unexplained_consumer')
            return dict(host=self.scope,roots={r:list(self.original[r]) for r in ip.ROLES},consumers=[])
        result=initialize_split({r:self.root[r] for r in ip.ROLES},self.scope,self.config,
            authority=observe,policy=ip.family_policy(self.family),daemon_binding=self.b,family=self.family,
            launch_child=lambda raw:self.child('core',json.loads(raw),tuple(json.loads(raw)[k] for k in ('control_fd','staging_fd'))))
        self.write('init-done.json',result)
        return dict(phase='initialized',**result)

    def public(self):
        private=json.loads(ip.read_private(self.root['receipts']/'p1-init.json',1024*1024,policy=ip.SPLIT,role='receipts'))
        public=private['public']
        if self.export:
            from service import installed_export as ex
            ip._closed(private,'public seed')
            ex.installation(public)
            if not isinstance(private['seed'],str) or not re.fullmatch('[0-9a-f]{64}',private['seed']):
                raise ValueError('daemon_initialization_seed')
        else: ip.initialization(private)
        marker=json.loads(ip.read_private(self.root['receipts']/'p1-enrolled.json',65536,policy=ip.SPLIT,role='receipts'))
        if (public.get('launch')!=self.b or public['configuration']!=self.config or ip.role_records({r:self.root[r] for r in ip.ROLES},policy=ip.family_policy(self.family))!=public['roles'] or
            marker!=dict(deployment_id=public['deployment_id'],digest=digest(public))): raise ValueError('daemon_enrollment_changed')
        completed=self.read('init-done.json')
        if completed!=dict(deployment_id=public['deployment_id'],digest=digest(public),initialization_digest=digest(private)):
            raise ValueError('daemon_initialization_completion_changed')
        return public

    def inspect(self,role,ids,action='inspect'):
        c=self.rpc(action,dict(id=ids[role],role=role))
        dv.inspect_container(c,role,self.config,self.b,family=self.family)
        if c['Id']!=ids[role]: raise ValueError('daemon_retained_id_changed')
        return c

    def pair(self,ids): return {r:self.inspect(r,ids) for r in ('control','worker')}

    def owners(self,pair):
        return [dict(id=c['Id'],image=c['Image'],started=c['State']['StartedAt'],role=r,
            volumes=['control','ipc','staging'] if r=='control' else ['ipc','receipts','staging']) for r,c in pair.items()]

    def startup(self):
        return self.child('hello',dict(ipc_fd=self.fds['ipc'],**({'configuration':self.config} if self.export else {})),(self.fds['ipc'],))

    def healthy(self,state):
        if (state['protocol']!=2 or json.loads(encode(state['monitor']))!=[[1,'healthy']] or
                json.loads(encode(state['integrity']))!=[[1,1,'healthy',None]]):
            raise ValueError('daemon_unhealthy_control_state')

    def clean(self,*,lane_lock=False):
        from worker.runtime_journal import read_snapshot
        from service.recovery import installed_clean_history
        with ExitStack() as stack:
            stack.enter_context(ip.locked_file(self.root['control']/'core/control.lock',policy=ip.SPLIT,role='control'))
            if lane_lock: stack.enter_context(ip.locked_file(self.root['receipts']/'lane.lock',policy=ip.SPLIT,role='receipts'))
            state=self.retained_snapshot()
            self.healthy(state)
            journal,maintenance=read_snapshot(self.root['receipts'],policy=ip.SPLIT)
            if maintenance is not None: raise ValueError('daemon_maintenance_pending')
            installed_clean_history(state,journal,family=self.family)
            return json.loads(encode(dict(state=state,journal=journal)))

    def observe(self,ids,public):
        first=self.pair(ids); startup=self.startup()
        facts=self.rpc('facts',dict(id=ids['control'],role='control'))
        value=dict(kind=self.evidence,binding=digest(self.b),inspections=first,local=dict(control=facts,worker=startup['local']),hello=startup['hello'])
        # Include every imported input view, not only the four mutable roles.
        inputs=self.read('import-done.json')['identities']
        for r,identity in inputs.items():
            for local in value['local'].values():
                if r in local['roots'] and local['roots'][r]!=identity: raise ValueError('daemon_input_view_identity')
        if self.export:
            assets=self.root['control']/'core/assets'
            info=ip.directory_identity(assets,policy=ip.SPLIT,role='control')
            if any(local['roots']['assets']!=list(info[:2]) for local in value['local'].values()):
                raise ValueError('daemon_assets_view_identity')
        if self.pair(ids)!=first or self.startup()!=startup: raise ValueError('daemon_observation_bracket')
        return value

    def validate_registration(self, record):
        from service import installed_export as ex, recovery
        return (ex.host_registration if self.export else recovery.validate_registration)(record)

    def retained_snapshot(self):
        if self.export:
            from service import installed_export as ex
            return ex.retained_snapshot(self.root,self.public(),family=self.family)
        from service.host_install import _registration_snapshot
        return _registration_snapshot({r:self.root[r] for r in ip.ROLES},self.scope)

    def registration_snapshot(self, public, *, retained=False):
        if retained and self.export:
            from service import installed_export as ex
            return ex.retained_snapshot(self.root,public,family=self.family)
        if not self.export:
            from service.host_install import _registration_snapshot
            return _registration_snapshot({r:self.root[r] for r in ip.ROLES},self.scope)
        # Deliberately narrow fresh reader. Retained facts use the separate
        # explicitly selected reader; this path keeps its empty-history checks.
        from service import installed_export as ex, recovery
        import sqlite3
        ex.expectation(public,self.family).check_root(self.root['control']/'core')
        ex.fresh_namespace(self.root)
        path=self.root['control']/'core/core.sqlite'
        with recovery.private_database(path,policy=ip.SPLIT) as private:
            db=sqlite3.connect(private.as_uri()+'?mode=ro',uri=True,timeout=0,isolation_level=None)
            try:
                db.execute('PRAGMA query_only=ON'); db.execute('PRAGMA trusted_schema=OFF')
                data=ex.database(db,public)
                fixed=dict(p1_format=[(2,'synthetic-core')],voice_catalogue_format=[(1,)],
                    resource_accounting=[(1,0)],resource_protocol=[(1,2)],
                    resource_monitor=[(1,'healthy')],p1_integrity=[(1,1,'healthy',None)],
                    p1_installation=[(public['deployment_id'],1,ex.SCOPE,encode(public),digest(public))])
                if any(rows!=fixed.get(table,[]) for table,rows in data.items() if table!='p1_registrations'):
                    raise ValueError('daemon_export_initial_history_only')
                schema=list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master LIMIT 2049'))
            finally: db.close()
        return dict(protocol=2,monitor=data['resource_monitor'],integrity=data['p1_integrity'],
            registrations=[json.loads(row[1]) for row in data['p1_registrations']],
            evidence_projection=dict(schema=schema,data=data))

    def register(self,ids,public,history):
        with ip.locked_file(self.root['control']/'core/control.lock',policy=ip.SPLIT,role='control') as lock:
            before=self.registration_snapshot(public,retained=self.export)
            self.healthy(before)
            if self.export:
                from worker.runtime_journal import read_snapshot
                from service import recovery
                journal,maintenance=read_snapshot(self.root['receipts'],policy=ip.SPLIT)
                if maintenance is not None: raise ValueError('daemon_maintenance_pending')
                recovery.installed_clean_history(before,journal,family=self.family)
            obs=self.observe(ids,public)
            if self.export and obs['hello']['lane']!=before['lane']: raise ValueError('installed_history_lane')
            record=dict(deployment_id=public['deployment_id'],scope=self.scope,containers=self.owners(obs['inspections']),
                lane=obs['hello']['lane'],boot=obs['hello']['boot'],evidence=self.evidence)
            self.validate_registration(record); dv.observation(obs,record,self.config,self.b,public,family=self.family)
            for old in before['registrations']:
                if old['evidence']!=self.evidence: raise ValueError('daemon_mixed_registration_history')
                old_worker=next(c for c in old['containers'] if c['role']=='worker')
                new_worker=next(c for c in record['containers'] if c['role']=='worker')
                if old_worker['id']!=new_worker['id'] or old['lane']!=record['lane']:
                    raise ValueError('daemon_restart_identity')
                if (old_worker['started']==new_worker['started'])!=(old['boot']==record['boot']):
                    raise ValueError('daemon_restart_boot_binding')
            rid=digest(record)
            provenance=dict(kind=self.evidence,registration=record,observation=obs,enrollment_digest=digest(public),history=history)
            self.write('registration-'+rid+'.json',provenance)
            prestate=[list(r) for r in before['evidence_projection']['data']['p1_registrations']]
            self.child('writer',dict(public=public,registration=record,prestate=prestate,control_fd=self.fds['control'],
                staging_fd=self.fds['staging'],lock_fd=lock),(self.fds['control'],self.fds['staging'],lock))
            after=self.registration_snapshot(public,retained=self.export)
            if [list(r) for r in after['evidence_projection']['data']['p1_registrations']]!=prestate+[[rid,encode(record),rid]]:
                raise ValueError('daemon_registration_poststate')
            def stable(state):
                projection=json.loads(encode(state['evidence_projection'])); projection['data'].pop('p1_registrations'); return projection
            if stable(before)!=stable(after) or self.observe(ids,public)!=obs: raise ValueError('daemon_registration_bracket')
            gate=gate_projection(dict(version=2 if self.export else 1,launch_policy=self.policy,registration_id=rid,registration=record,
                configuration_digest=digest(self.config),enrollment_digest=digest(public),provenance_digest=digest(provenance),observation=obs),family=self.family,
                configuration=self.config if self.export else None,binding=self.b,public=public)
        # Durable registration and its final bracket are complete. Release the
        # registrar's Core exclusion BEFORE publishing activation: the notified
        # control immediately acquires that same lock for its private precheck.
        # The outer authority lock still excludes another installer/registrar.
        self.rpc('notify',dict(id=ids['control'],gate=gate))
        if self.observe(ids,public)!=obs: raise ValueError('daemon_notification_bracket')
        return rid,record

    def wait(self,function,predicate):
        while time.monotonic()<self.end:
            value=function()
            if predicate(value): return value
            time.sleep(.05)
        raise ValueError('daemon_runtime_wait_timeout_retained')

    def ready(self,ids,public,rid,record):
        self.wait(lambda:self.rpc('marker',dict(id=ids['control'],role='control')),lambda v:v==dict(phase='ready',registration_id=rid))
        current=self.observe(ids,public); dv.observation(current,record,self.config,self.b,public,family=self.family)
        return dict(phase='ready',registration_id=rid,containers=ids)

    def runtime(self):
        public=self.public()
        if self.exists('containers.json'):
            ids=self.read('containers.json'); dv.closed(ids,'control worker')
        elif self.phase=='up':
            if self.exists('container-creation-intent.json'): raise ValueError('daemon_partial_container_creation')
            self.write('container-creation-intent.json',dict(binding=digest(self.b)))
            ids={}
            for role in ('control','worker'):
                self.snapshot()
                c=self.rpc('create',dict(role=role)); dv.inspect_container(c,role,self.config,self.b,family=self.family)
                if c['State']['Running'] is not False or c['State']['Status']!='created': raise ValueError('daemon_created_container_state')
                ids[role]=c['Id']; self.write('created-'+role+'.json',dict(id=c['Id']))
            if len(set(ids.values()))!=2: raise ValueError('daemon_duplicate_runtime_id')
            self.write('containers.json',ids)
        elif self.phase=='status': return dict(phase='initialized')
        else: raise ValueError('daemon_not_launched')
        if any(not isinstance(i,str) or not dv.HEX.fullmatch(i) for i in ids.values()): raise ValueError('daemon_retained_ids')
        if self.read('container-creation-intent.json')!=dict(binding=digest(self.b)) or any(self.read('created-'+r+'.json')!=dict(id=i) for r,i in ids.items()):
            raise ValueError('daemon_creation_receipts_changed')
        pair=self.pair(ids)
        live=[c['State']['Running'] for c in pair.values()]
        if all(live):
            marker=self.rpc('marker',dict(id=ids['control'],role='control'))
            if marker.get('phase')!='ready': raise ValueError('daemon_live_not_ready')
            rid=marker['registration_id']; provenance=self.read('registration-'+rid+'.json')
            record=provenance['registration']
            if digest(record)!=rid or record['evidence']!=self.evidence or provenance['enrollment_digest']!=digest(public): raise ValueError('daemon_retained_registration')
            result=self.ready(ids,public,rid,record)
            if self.phase!='stop': return result
            self.rpc('term',dict(id=ids['control'],role='control'))
            self.wait(lambda:self.inspect('control',ids),lambda c:not c['State']['Running'] and c['State']['ExitCode']==0)
            hello=self.startup(); self.clean()
            if self.startup()!=hello: raise ValueError('daemon_shutdown_hello_changed')
            self.rpc('term',dict(id=ids['worker'],role='worker'))
            self.wait(lambda:self.inspect('worker',ids),lambda c:not c['State']['Running'] and c['State']['ExitCode']==0)
            history=self.clean(lane_lock=True)
            stopped=self.pair(ids)
            if any(c['State']['Running'] or c['State']['ExitCode']!=0 for c in stopped.values()): raise ValueError('daemon_stopped_unknown')
            self.write('stopped-'+self.nonce+'.json',dict(containers=stopped,history=history))
            return dict(phase='stopped',containers=ids)
        if any(live) or any(c['State']['ExitCode']!=0 for c in pair.values()): raise ValueError('daemon_dirty_partial_runtime')
        history=self.clean(lane_lock=True)
        if self.phase=='status': return dict(phase='stopped',containers=ids)
        if self.phase!='up': raise ValueError('daemon_stop_requires_ready')
        # A stopped retained pair needs a matching clean stop receipt, unless it
        # is the never-started pair from this fresh exclusive create transaction.
        if any(c['State']['Status']!='created' for c in pair.values()):
            receipts=[self.read(n) for n in ip.directory_entries(self.fds['authority']) if n.startswith('stopped-') and n.endswith('.json')]
            if self.export:
                from service.installed_export import stopped_receipt
                matches = []
                for receipt in receipts:
                    try: stopped_receipt(receipt,pair,history,public=public,family=self.family)
                    except ValueError: continue
                    matches.append(receipt)
                if not matches: raise ValueError('daemon_clean_stop_receipt_required')
            elif not any(r['containers']==pair and r['history']==history for r in receipts): raise ValueError('daemon_clean_stop_receipt_required')
        self.inspect('control',ids,'start')
        self.wait(lambda:self.rpc('marker',dict(id=ids['control'],role='control')),lambda v:v==dict(phase='waiting'))
        self.inspect('worker',ids,'start')
        # Lifecycle connection may not yet be available: refuse transient failure
        # only after bounded retry; all frames remain under the launch lock.
        while True:
            try: self.startup(); break
            except (ValueError,OSError):
                if time.monotonic()>self.end: raise
                time.sleep(.05)
        rid,record=self.register(ids,public,history)
        return self.ready(ids,public,rid,record)

    def run(self):
        with self.held():
            if self.phase=='install': result=self.input_import()
            elif self.phase=='init': result=self.initialize()
            else: result=self.runtime()
            self.snapshot()
            self.write('operation-'+self.nonce+'.done',dict(phase=self.phase,nonce=self.nonce,helper=self.own['Id'],started=self.own['State']['StartedAt'],result=result))
        # No success frame until final held-context checks have completed.
        return result


def main(*, family=None):
    try:
        if sys.platform!='linux' or os.getresuid()!=(0,0,0) or os.getresgid()!=(0,0,0): raise ValueError('daemon_helper_linux_root')
        phase,nonce=sys.argv[1:]
        if phase not in dv.PHASES or not __import__('re').fullmatch('[0-9a-f]{32}',nonce): raise ValueError('daemon_helper_phase')
        os.umask(0o077)
        with open('/proc/self/status','rb') as stream: status=stream.read(65537)
        if len(status)>65536: raise ValueError('daemon_proc_budget')
        fields=dict(line.split(':',1) for line in status.decode('ascii').splitlines() if ':' in line)
        permitted=sum(1<<n for n in ((0,1,3,5,6,7) if phase in ('install','init') else (1,5,6,7)))
        if (fields['NoNewPrivs'].strip()!='1' or any(int(fields[k].strip(),16)!=permitted for k in ('CapEff','CapPrm')) or
                any(int(fields[k].strip(),16) for k in ('CapInh','CapAmb'))):
            raise ValueError('daemon_helper_capability_identity')
        wire=Wire(sys.stdin.buffer,sys.stdout.buffer,nonce,phase)
        action,request=wire.receive()
        if action!='begin': raise ValueError('daemon_helper_begin')
        result=Helper(phase,nonce,request,wire,family=family).run()
        wire.send('complete',dict(result=result))
    except (Exception,KeyboardInterrupt) as exc:
        # Source location only: never serialize exception messages, request data,
        # backend paths, manifests or credentials into the diagnostic channel.
        trace = exc.__traceback__
        while trace is not None and trace.tb_next is not None:
            trace = trace.tb_next
        if trace is not None:
            code = trace.tb_frame.f_code
            print('daemon_helper_refused:'+type(exc).__name__+':'+
                code.co_name+':'+str(trace.tb_lineno),file=sys.stderr)
        raise SystemExit(2) from None


def export_main():
    from service.installed_export import FAMILY
    return main(family=FAMILY)
