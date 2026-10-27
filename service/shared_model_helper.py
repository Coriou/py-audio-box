"""Networkless store controller, single-candidate writer and independent RO reader.

No Docker socket. The controller retains the permanent key flock across every
client RPC, including terminal process custody and independent verification.
"""
import base64
import hashlib
import os
import re
import secrets
import stat
import sys
import time

from service import shared_models as sm
from service.daemon_client import Wire, encoded

ROLES = ('controller', 'writer', 'verifier')
CAPS = {'controller': [], 'writer': ['CHOWN','FOWNER','DAC_OVERRIDE'], 'verifier': []}
TMPFS = 'rw,noexec,nosuid,nodev,size=16777216,nr_inodes=1024,mode=1777'



# Closed diagnostic vocabulary: never serialize exception messages or chains.
_DIAGNOSTIC_CODES = frozenset((
    'daemon_action_not_allowed', 'daemon_action_role', 'daemon_closed_record',
    'daemon_configuration_binding', 'daemon_configuration_budget', 'daemon_consumer_budget',
    'daemon_container_changed', 'daemon_container_local_alias',
    'daemon_container_local_identity', 'daemon_container_local_roots',
    'daemon_container_mount_binding', 'daemon_container_security_binding',
    'daemon_container_started_at', 'daemon_container_state', 'daemon_context',
    'daemon_control_reader_only', 'daemon_cpu_budget', 'daemon_duplicate_field',
    'daemon_endpoint_identity_changed', 'daemon_evidence_binding',
    'daemon_evidence_cross_view_identity', 'daemon_evidence_incarnation',
    'daemon_evidence_role_identity', 'daemon_exact_container_id',
    'daemon_existing_volume_refused', 'daemon_explicit_context_required',
    'daemon_explicit_volume_mounts_required', 'daemon_export_credential_principals',
    'daemon_export_exact_mount_options', 'daemon_frame_budget',
    'daemon_fresh_local_volume_required', 'daemon_gate_budget', 'daemon_handle_context_changed',
    'daemon_handle_selection_only', 'daemon_helper_exit_unknown', 'daemon_helper_id',
    'daemon_helper_phase', 'daemon_helper_start_unknown', 'daemon_image_configuration_binding',
    'daemon_input_budget', 'daemon_input_changed', 'daemon_input_file', 'daemon_input_frame',
    'daemon_input_manifest', 'daemon_install_inputs_required', 'daemon_loaded_image_required',
    'daemon_local_rootful_linux_amd64_required', 'daemon_new_installation_policy',
    'daemon_no_copyup_mount_required', 'daemon_phase_eof_retained', 'daemon_phase_replay',
    'daemon_phase_timeout', 'daemon_phase_write_timeout', 'daemon_pinned_images_required',
    'daemon_port_budget', 'daemon_private_input_budget', 'daemon_requested_mount_binding',
    'daemon_resource_budget', 'daemon_role', 'daemon_terminal_eof_timeout',
    'daemon_tmpfs_budget', 'daemon_trailing_protocol', 'daemon_volume_alias',
    'daemon_volume_changed', 'daemon_volume_create_changed', 'explicit_export_family_required',
    'shared_absolute_source_required', 'shared_actor_id_changed',
    'shared_actor_incarnation_changed', 'shared_actor_intent', 'shared_actual_mounts',
    'shared_attachment_binding', 'shared_attachment_outcome_unknown',
    'shared_attachment_record', 'shared_attachment_role', 'shared_attempt_budget',
    'shared_attempt_descriptor', 'shared_attempt_grammar', 'shared_begin',
    'shared_candidate_incarnation', 'shared_candidate_not_empty', 'shared_candidate_not_fresh',
    'shared_candidate_required', 'shared_catalog_directory_changed', 'shared_catalog_identity',
    'shared_catalog_incarnation', 'shared_catalog_inventory', 'shared_catalog_root_changed',
    'shared_config_credential_only', 'shared_consumer_budget', 'shared_consumer_changed',
    'shared_controller_payload_forbidden', 'shared_deadline', 'shared_duplicate_attachment',
    'shared_duplicate_consumer', 'shared_duplicate_helper', 'shared_enrolled_binding',
    'shared_enrollment_binding', 'shared_enrollment_required', 'shared_event',
    'shared_exact_published_attempt', 'shared_exact_recovery_attempt', 'shared_expiry',
    'shared_explicit_init_required', 'shared_explicit_request', 'shared_explicit_sources',
    'shared_explicit_writer_sources', 'shared_extra_directory', 'shared_extra_file',
    'shared_file_extended', 'shared_file_hash', 'shared_file_identity',
    'shared_filesystem_policy', 'shared_fresh_attachment', 'shared_fresh_catalog_owner',
    'shared_fresh_helper_required', 'shared_fresh_qwen_only', 'shared_helper_allowlist',
    'shared_helper_failed_retained', 'shared_helper_role', 'shared_helper_security',
    'shared_helper_timeout_retained', 'shared_identity', 'shared_image',
    'shared_immutable_image', 'shared_incomplete_tree', 'shared_init_request',
    'shared_input_allowlist', 'shared_input_chunk', 'shared_input_hash',
    'shared_inspect_allowlist', 'shared_installation_operation',
    'shared_installation_record_grammar', 'shared_inventory_budget', 'shared_key_allowlist',
    'shared_literal_mount_policy', 'shared_local_image_required', 'shared_lock_changed',
    'shared_lock_contents', 'shared_locks_directory_changed', 'shared_manifest_bytes',
    'shared_not_published', 'shared_operation', 'shared_orphan_attempt_record',
    'shared_orphan_reader_record', 'shared_partial_bootstrap', 'shared_payload_budget',
    'shared_publication_actor_record', 'shared_publication_attempt',
    'shared_publication_changed', 'shared_publication_conflict', 'shared_publication_exists',
    'shared_publication_identity', 'shared_published_tree_changed', 'shared_qwen_descriptor',
    'shared_reader_budget', 'shared_reader_candidate', 'shared_reader_intent',
    'shared_reader_outcome_unknown', 'shared_reader_record', 'shared_record_budget',
    'shared_record_changed', 'shared_record_encoding', 'shared_reference_binding',
    'shared_reference_grammar', 'shared_requested_mounts', 'shared_reservation_identity',
    'shared_root_changed', 'shared_root_identity', 'shared_rpc_action',
    'shared_rpc_not_allowed', 'shared_selected_candidate', 'shared_selected_catalog',
    'shared_selected_publication', 'shared_snapshot_identity', 'shared_source_changed',
    'shared_source_file', 'shared_sources', 'shared_start_allowlist', 'shared_storage_binding',
    'shared_terminal_required', 'shared_terminal_time', 'shared_tree_changed',
    'shared_unknown_attempt', 'shared_unknown_consumer', 'shared_verifier_rpc', 'shared_volume',
    'shared_volume_alias', 'shared_volume_create_unknown', 'shared_volume_inspection',
    'shared_volume_inventory_budget', 'shared_writer_fresh_allocation_required',
    'shared_writer_rpc',
))
_DIAGNOSTIC_FUNCTIONS = frozenset((
    '__enter__', '__exit__', '__init__', '_binding', '_create', 'action', 'actor', 'append',
    'argv', 'attachment', 'attempts', 'bind', 'binding', 'bound', 'cadence', 'call',
    'canonical_mountpoint', 'catalog_volume', 'check', 'chunk', 'classify', 'closed', 'collect',
    'command', 'consumers', 'container', 'coordinate', 'create_helper', 'create_volume',
    'created', 'daemon', 'deadline', 'descriptor', 'digest', 'directory', 'disjoint', 'encoded',
    'enrolled', 'event', 'exchange', 'exclusive', 'export_main', 'fail', 'file_identity',
    'finish', 'flock', 'held', 'helper_roles', 'identity', 'import_tree', 'init', 'input',
    'inspect', 'inspect_container', 'inspect_volume', 'json', 'labels', 'local', 'main',
    'manifest', 'mount_policy', 'mounts', 'name', 'names_at', 'new_reader', 'observation',
    'open_root', 'pairs', 'parse', 'pocket_main', 'policy', 'publication', 'publish',
    'qwen_main', 'read', 'read_at', 'readers', 'receipt', 'receive', 'reference_name',
    'remaining', 'reservation', 'reserve', 'reuse', 'root_fd', 'rpc', 'run', 'scope',
    'selector', 'send', 'snapshot', 'source_overlap', 'stable', 'storage', 'terminal', 'token',
    'validate', 'verify_tree', 'volume', 'walk', 'write_at',
))
_DIAGNOSTIC_FILES = frozenset(('shared_models.py', 'shared_model_helper.py', 'shared_model_client.py', 'daemon_client.py', 'daemon_volumes.py'))
_DIAGNOSTIC_TYPES = (ValueError, OSError, RuntimeError, TimeoutError,
    FileNotFoundError, PermissionError, BlockingIOError, BrokenPipeError,
    EOFError, TypeError, KeyError, KeyboardInterrupt, SystemExit)
OPERATIONS = ('init','import','verify','recover-import','reserve','bind',
    'attachment-intent','attachment-created')
_DIAGNOSTIC_STAGES = ('unknown','receive','validated','mount','operation',
    'complete','eof','start','begin-sent','response','reply-sent','terminal')


def failure_record(exc, *, role='unknown', nonce='unknown', operation='unknown',
                   stage='unknown', identity='unknown'):
    """At most eight frames, no source/locals/paths/messages/chained errors."""
    kind=type(exc)
    known=kind in _DIAGNOSTIC_TYPES
    code='unknown'
    if kind is ValueError and len(exc.args)==1 and type(exc.args[0]) is str and exc.args[0] in _DIAGNOSTIC_CODES:
        code=exc.args[0]
    number=exc.errno if known and isinstance(exc,OSError) else None
    if type(number) is not int or not -2147483648<=number<=2147483647: number=None
    frames=[]; trace=exc.__traceback__
    for _ in range(64):
        if trace is None: break
        source=trace.tb_frame.f_code
        basename=source.co_filename.rsplit('/',1)[-1]
        if (basename in _DIAGNOSTIC_FILES and source.co_name in _DIAGNOSTIC_FUNCTIONS
                and 0<trace.tb_lineno<=1000000):
            frames.append(dict(file=basename,function=source.co_name,line=trace.tb_lineno))
            frames=frames[-8:]
        trace=trace.tb_next
    return dict(version=1,schema='shared-model-failure-v1',
        role=role if type(role) is str and role in ROLES else 'unknown',
        nonce=nonce if type(nonce) is str and sm.ATTEMPT.fullmatch(nonce) else 'unknown',
        id=identity if type(identity) is str and sm.HEX.fullmatch(identity) else 'unknown',
        operation=operation if type(operation) is str and operation in OPERATIONS else 'unknown',
        stage=stage if type(stage) is str and stage in _DIAGNOSTIC_STAGES else 'unknown',
        exception=kind.__name__ if known else 'Exception',code=code,errno=number,
        frames=frames,frames_available=bool(frames))


def diagnostic(exc, **context):
    # Best effort must never replace the original refusal, including on EPIPE.
    try:
        raw=encoded(failure_record(exc,**context))
        if len(raw)<=4096: sys.stderr.write(raw.decode('ascii')+'\n')
    except BaseException:
        pass


def mounts(role, catalog, candidate=None):
    if role not in ROLES: raise ValueError('shared_helper_role')
    if role == 'controller':
        sm.volume(catalog); return [(catalog, sm.CATALOG_ROOT, True)]
    if candidate is None: raise ValueError('shared_candidate_required')
    # Caller validates key/attempt against its durable intent.
    return [(candidate, sm.MODEL_ROOT, role == 'writer')]


def command(role, image, nonce, catalog, candidate=None):
    sm.token(nonce, sm.ATTEMPT)
    if not sm.dv.IMAGE.fullmatch(image): raise ValueError('shared_immutable_image')
    args = ['container','create','--pull=never','--restart=no','--read-only','--user','0:0',
            '--cap-drop=ALL','--security-opt=no-new-privileges:true','--ipc=private','--cgroupns=private',
            '--network','none','--memory','268435456','--memory-swap','268435456','--pids-limit','32',
            '--cpus','1.000000000','--tmpfs','/tmp:'+TMPFS,
            '--interactive','--attach=stdin','--attach=stdout']
    for cap in CAPS[role]: args += ['--cap-add='+cap]
    for v, target, rw in mounts(role, catalog, candidate):
        args += ['--mount','type=volume,source='+v['Name']+',destination='+target+',volume-nocopy'+('' if rw else ',readonly')]
    args += ['--entrypoint','python',image,'-I','-B','-c',
             "import sys;sys.path.insert(0,'/opt/speech');from service.shared_model_helper import main;main()",role,nonce]
    return args


def inspect(c, role, image, nonce, catalog, candidate=None, *, terminal=False, successful=True):
    expected = command(role,image,nonce,catalog,candidate)
    hc, cc, state = c['HostConfig'], c['Config'], c['State']
    caps = [v.removeprefix('CAP_') for v in (hc['CapAdd'] or [])]
    if (not sm.HEX.fullmatch(c['Id']) or c['Image'] != image or cc['User'] != '0:0' or
            cc['Entrypoint'] != ['python'] or cc['Cmd'] != expected[expected.index('--entrypoint')+3:] or
            hc['Privileged'] is not False or hc['ReadonlyRootfs'] is not True or hc.get('Init') not in (None,False) or
            hc['PidMode'] not in ('',None) or hc['IpcMode'] != 'private' or hc['CgroupnsMode'] != 'private' or
            hc['UsernsMode'] not in ('',None) or hc['NetworkMode'] != 'none' or
            hc['CapDrop'] != ['ALL'] or sorted(caps) != sorted(CAPS[role]) or
            hc['SecurityOpt'] != ['no-new-privileges:true'] or hc['RestartPolicy'] != {'Name':'no','MaximumRetryCount':0} or
            any(hc.get(k) for k in ('Devices','DeviceRequests','Binds','VolumesFrom','GroupAdd','DeviceCgroupRules','Sysctls',
                                      'CgroupParent','UTSMode','CpuPeriod','CpuQuota','CpusetCpus','PublishAllPorts','PortBindings')) or
            hc['Memory'] != 268435456 or hc['MemorySwap'] != 268435456 or hc['PidsLimit'] != 32 or hc['NanoCpus'] != 10**9 or
            hc['Tmpfs'] != {'/tmp': TMPFS} or c['RestartCount'] != 0 or state['Restarting'] is not False or
            type(state['Running']) is not bool or type(state['ExitCode']) is not int or
            state.get('OOMKilled',False) or state.get('Dead',False) or state.get('Error','') or
            state['Status'] not in ('created','running','exited')):
        raise ValueError('shared_helper_security')
    requested = hc.get('Mounts')
    wanted = mounts(role,catalog,candidate)
    if type(requested) is not list or len(requested) != len(wanted): raise ValueError('shared_requested_mounts')
    for m, (v,target,rw) in zip(requested,wanted):
        if (set(m)-{'Type','Source','Target','ReadOnly','Consistency','VolumeOptions'} or
                m.get('Type') != 'volume' or m.get('Source') != v['Name'] or m.get('Target') != target or
                type(m.get('ReadOnly',False)) is not bool or m.get('ReadOnly',False) != (not rw) or
                m.get('VolumeOptions') != {'NoCopy':True} or m.get('Consistency','') not in ('','default')):
            raise ValueError('shared_requested_mounts')
    actual = [dict(Type='volume',Name=v['Name'],Source=v['Mountpoint'],Destination=target,RW=rw,Propagation='') for v,target,rw in wanted]
    if [{k:m.get(k) for k in actual[0]} for m in c['Mounts']] != actual:
        raise ValueError('shared_actual_mounts')
    if terminal:
        if (state['Running'] is not False or state['Status'] != 'exited' or (successful and state['ExitCode'] != 0) or state.get('Pid') != 0):
            raise ValueError('shared_terminal_required')
        for field in ('StartedAt','FinishedAt'):
            from datetime import datetime
            value = state[field]
            if type(value) is not str or not value.endswith('Z') or value.startswith('0001-') or len(value)>64:
                raise ValueError('shared_terminal_time')
            datetime.fromisoformat(value.replace('Z','+00:00'))
    return c


def terminal(c, role, image, nonce, catalog, candidate, *, successful=True):
    inspect(c,role,image,nonce,catalog,candidate,terminal=True,successful=successful)
    # Bind only policy-relevant fields, not volatile Docker diagnostics/size.
    evidence = {k:c[k] for k in ('Id','Image','Config','HostConfig','Mounts','State','RestartCount')}
    return dict(id=c['Id'],nonce=nonce,image=image,inspection_sha256=sm.digest(evidence),
                started=c['State']['StartedAt'],finished=c['State']['FinishedAt'])


def mount_policy(path, readonly):
    # Literal mount, not a writable subdirectory of a differently mounted root.
    found = []
    with open('/proc/self/mountinfo',encoding='ascii') as f:
        for line in f:
            fields = line.split()
            if len(fields)>6 and fields[4] == path:
                found.append(fields[5].split(','))
    if len(found) != 1 or ('ro' in found[0]) != readonly or ('rw' in found[0]) == readonly:
        raise ValueError('shared_literal_mount_policy')


def import_tree(path, expected, get_chunk, end, *, fault=lambda _: None):
    key, raw, files, dirs = sm.descriptor(expected)
    with sm.root_fd(path) as root:
        if sm.names_at(root,1,end): raise ValueError('shared_candidate_not_empty')
        opened = {'':root}
        try:
            for name in sorted(dirs-{''}, key=lambda n:(n.count('/'),n)):
                parent, _, leaf = name.rpartition('/')
                os.mkdir(leaf,0o700,dir_fd=opened[parent])
                opened[name] = os.open(leaf,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=opened[parent])
            for name,(size,hashed) in sorted(files.items()):
                sm.remaining(end)
                parent,_,leaf = name.rpartition('/')
                fd = os.open(leaf,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=opened[parent])
                try:
                    count=0; h=hashlib.sha256()
                    while count < size:
                        sm.remaining(end)
                        data = raw[count:count+sm.CHUNK] if leaf=='manifest.json' else get_chunk(name,count)
                        if type(data) is not bytes or not 0<len(data)<=min(sm.CHUNK,size-count):
                            raise ValueError('shared_input_chunk')
                        offset=0
                        while offset<len(data): offset+=os.write(fd,data[offset:])
                        h.update(data); count+=len(data); fault('stream_chunk')
                    if h.hexdigest()!=hashed: raise ValueError('shared_input_hash')
                    os.fchown(fd,*sm.OWNER); os.fchmod(fd,0o444); os.fsync(fd); fault('file_sealed')
                finally: os.close(fd)
            for name in sorted(dirs,key=lambda n:(n.count('/'),n),reverse=True):
                os.fchown(opened[name],*sm.OWNER); os.fchmod(opened[name],0o555); os.fsync(opened[name])
            fault('tree_sealed')
        finally:
            for name,fd in opened.items():
                if name: os.close(fd)
    # Writer's result is never publication authority; separate reader repeats.
    return {'sealed':True}


class Controller:
    def __init__(self, catalog, rpc, image, expected=None, *, fault=lambda _: None):
        self.catalog,self.rpc,self.image,self.expected,self.fault = catalog,rpc,image,expected,fault

    def actor(self, store, attempt, role, candidate, *, recovery=False, successful=True):
        image=self.image
        if recovery:
            intent=store.event(attempt,role+'-intent'); sm.closed(intent,'role nonce image candidate')
            if intent['role']!=role or intent['candidate']!=candidate: raise ValueError('shared_actor_intent')
            nonce=intent['nonce']; image=intent['image']
            created=store.event(attempt,role+'-created'); sm.closed(created,'id')
            c=self.rpc('inspect',dict(id=created['id'],role=role,image=image,nonce=nonce,candidate=candidate))
            proof=terminal(c,role,image,nonce,self.catalog.metadata,candidate,successful=successful)
            if proof['id']!=created['id']: raise ValueError('shared_actor_id_changed')
            # A missing terminal receipt is recoverable only via exact inspection.
            try: old=store.event(attempt,role+'-terminal')
            except FileNotFoundError: store.append(attempt,role+'-terminal',proof)
            else:
                if old != proof: raise ValueError('shared_actor_incarnation_changed')
            return proof
        nonce=secrets.token_hex(16)
        intent=dict(role=role,nonce=nonce,image=image,candidate=candidate)
        store.append(attempt,role+'-intent',intent)
        c=self.rpc('create-helper',intent)
        inspect(c,role,image,nonce,self.catalog.metadata,candidate)
        store.append(attempt,role+'-created',dict(id=c['Id']))
        self.fault(role+'_created')
        result=self.rpc('run-helper',dict(id=c['Id'],role=role,nonce=nonce,image=image,candidate=candidate,expected=self.expected))
        sm.closed(result,'inspection result')
        proof=terminal(result['inspection'],role,image,nonce,self.catalog.metadata,candidate)
        if proof['id'] != c['Id']: raise ValueError('shared_actor_id_changed')
        store.append(attempt,role+'-terminal',proof)
        self.fault(role+'_terminal')
        return proof,result['result']

    def consumers(self, candidate, actors, store=None, *, installations=None):
        snapshot=self.rpc('snapshot',dict(candidate=candidate))
        sm.closed(snapshot,'daemon candidate consumers')
        if snapshot['daemon']!=self.catalog.daemon or snapshot['candidate']!=candidate:
            raise ValueError('shared_snapshot_identity')
        consumers=snapshot['consumers']
        if type(consumers) is not list or len(consumers)>512: raise ValueError('shared_consumer_budget')
        known={a['id']:a for a in actors}
        authority=installations or (sm.Installations(store) if store is not None and store.publication() is not None else None)
        readers=authority.readers() if authority is not None else {}
        seen=set()
        for c in consumers:
            identity=c['Id']
            if identity in seen: raise ValueError('shared_duplicate_consumer')
            seen.add(identity)
            if identity in known:
                a=known[identity]
                role='writer' if a is actors[0] else 'verifier'
                if terminal(c,role,a['image'],a['nonce'],self.catalog.metadata,candidate)!=a:
                    raise ValueError('shared_consumer_changed')
            else:
                if authority is None: raise ValueError('shared_unknown_consumer')
                authority.classify(c,readers)
        if len(known)!=len(actors) or not set(known)<=seen or not set(readers)<=seen:
            raise ValueError('shared_unknown_consumer')

    def run(self, operation, attempt=None):
        key=sm.descriptor(self.expected)[0]
        with self.catalog.held(key) as store:
            published=store.publication()
            if published is not None:
                if operation=='recover-import' and attempt!=published['attempt']: raise ValueError('shared_exact_published_attempt')
                # Verify all historical actors, then fresh full RO read. Reader
                # receipts are permanent holds too, included on every later scan.
                return self.reuse(store,published,recover=operation=='recover-import')
            if operation=='verify': raise ValueError('shared_not_published')
            attempts=store.attempts()
            if operation=='recover-import':
                sm.token(attempt,sm.ATTEMPT)
                if attempt not in attempts: raise ValueError('shared_unknown_attempt')
                intent=store.event(attempt,'intent')
                if intent != dict(schema=sm.SCHEMA,key=key,attempt=attempt,expected=self.expected):
                    raise ValueError('shared_attempt_descriptor')
                candidate=sm.volume(store.event(attempt,'candidate'),key,attempt); sm.disjoint(self.catalog.metadata,candidate)
                writer=self.actor(store,attempt,'writer',candidate,recovery=True)
                self.readers(store,candidate,recover=True)
                try: store.event(attempt,'verifier-intent')
                except FileNotFoundError:
                    verifier,observation=self.actor(store,attempt,'verifier',candidate)
                else:
                    verifier=self.actor(store,attempt,'verifier',candidate,recovery=True)
                    _,observation=self.new_reader(store,candidate,attempt)
            elif operation=='import':
                # Before replacement, every old writer must have exact terminal
                # custody. Missing candidate/create response remains uncertainty.
                for old in attempts:
                    candidate=sm.volume(store.event(old,'candidate'),key,old)
                    self.actor(store,old,'writer',candidate,recovery=True,successful=False)
                    try: store.event(old,'verifier-intent')
                    except FileNotFoundError: pass
                    else: self.actor(store,old,'verifier',candidate,recovery=True,successful=False)
                attempt=secrets.token_hex(16)
                store.append(attempt,'intent',dict(schema=sm.SCHEMA,key=key,attempt=attempt,expected=self.expected))
                self.fault('candidate_intent')
                candidate=self.rpc('allocate',dict(key=key,attempt=attempt))
                sm.volume(candidate,key,attempt); sm.disjoint(self.catalog.metadata,candidate); store.append(attempt,'candidate',candidate)
                self.fault('candidate_receipt')
                writer,_=self.actor(store,attempt,'writer',candidate)
                verifier,observation=self.actor(store,attempt,'verifier',candidate)
            else: raise ValueError('shared_operation')
            actors=[writer,verifier]+self.readers(store,candidate)
            self.consumers(candidate,actors,store)
            result=dict(schema=sm.SCHEMA,daemon=self.catalog.daemon,catalog=self.catalog.metadata,key=key,attempt=attempt,
                        candidate=candidate,observation=observation,writer=writer,verifier=verifier)
            return store.publish(result,self.fault)

    def coordinate(self,operation,selection):
        if operation=='reserve': key=sm.descriptor(self.expected)[0]
        else:
            from service import installed_qwen_daemon as qd
            key=qd.binding(selection['binding'])['model_storage']['manifest_sha256']
        with self.catalog.held(key) as store:
            authority=sm.Installations(store)
            if operation=='reserve':
                sm.closed(selection,'installation configuration catalog publication_sha256')
                if selection['catalog']!=self.catalog.metadata or selection['publication_sha256']!=sm.digest(authority.published): raise ValueError('shared_selected_publication')
                self.reuse(store,authority.published)
                return authority.reserve(selection['installation'],selection['configuration'])
            if operation=='bind':
                sm.closed(selection,'binding configuration')
                return authority.bind(selection['binding'],selection['configuration'])
            if operation=='attachment-intent':
                sm.closed(selection,'binding configuration role phase nonce')
                return authority.attachment(selection['binding'],selection['configuration'],selection['role'],selection['phase'],selection['nonce'])
            if operation=='attachment-created':
                sm.closed(selection,'binding nonce inspection')
                return authority.created(selection['binding'],selection['nonce'],selection['inspection'])
            raise ValueError('shared_installation_operation')

    def readers(self,store,candidate,*,recover=False):
        with store.directory('references') as fd:
            names=sm.names_at(fd,sm.MAX_ENTRIES,self.catalog.end)
            if len(names)>sm.MAX_ENTRIES: raise ValueError('shared_reader_budget')
            values=[]
            for name in names:
                if not sm.reference_name(name):
                    raise ValueError('shared_reference_grammar')
                if name.startswith(store.key+'-'):
                    base=name.rsplit('-',1)[0]
                    if base+'-intent.json' not in names: raise ValueError('shared_orphan_reader_record')
            for name in sorted(names):
                if not name.startswith(store.key+'-') or not name.endswith('-intent.json'): continue
                intent=sm.read_at(fd,name)
                sm.closed(intent,'role nonce image candidate')
                if intent['candidate']!=candidate: raise ValueError('shared_reader_candidate')
                nonce=sm.token(intent['nonce'],sm.ATTEMPT)
                if name != store.key+'-'+nonce+'-intent.json' or intent['role']!='verifier':
                    raise ValueError('shared_reader_intent')
                base=store.key+'-'+nonce
                created=sm.read_at(fd,base+'-created.json'); sm.closed(created,'id')
                try: value=sm.read_at(fd,base+'-done.json')
                except FileNotFoundError:
                    if not recover: raise ValueError('shared_reader_outcome_unknown')
                    c=self.rpc('inspect',dict(id=created['id'],role='verifier',image=intent['image'],nonce=nonce,candidate=candidate))
                    proof=terminal(c,'verifier',intent['image'],nonce,self.catalog.metadata,candidate)
                    value=dict(candidate=candidate,proof=proof,observation=None)
                    sm.write_at(fd,base+'-done.json',value)
                sm.closed(value,'candidate proof observation')
                if value['candidate']!=candidate or value['proof']['id']!=created['id'] or value['proof']['nonce']!=nonce or value['proof']['image']!=intent['image']:
                    raise ValueError('shared_reader_record')
                values.append(value['proof'])
            return values

    def new_reader(self,store,candidate,attempt=None):
        # Intent and exact ID are durable before start, including recovery reads.
        nonce=secrets.token_hex(16)
        intent=dict(role='verifier',nonce=nonce,image=self.image,candidate=candidate)
        with store.directory('references') as fd:
            sm.write_at(fd,store.key+'-'+nonce+'-intent.json',intent)
        c=self.rpc('create-helper',intent)
        inspect(c,'verifier',self.image,nonce,self.catalog.metadata,candidate)
        with store.directory('references') as fd:
            sm.write_at(fd,store.key+'-'+nonce+'-created.json',dict(id=c['Id']))
        result=self.rpc('run-helper',dict(id=c['Id'],role='verifier',nonce=nonce,image=self.image,candidate=candidate,expected=self.expected))
        sm.closed(result,'inspection result')
        proof=terminal(result['inspection'],'verifier',self.image,nonce,self.catalog.metadata,candidate)
        if proof['id']!=c['Id']: raise ValueError('shared_actor_id_changed')
        with store.directory('references') as fd:
            sm.write_at(fd,store.key+'-'+nonce+'-done.json',dict(candidate=candidate,proof=proof,observation=result['result']))
        return proof,result['result']

    def reuse(self,store,published,*,recover=False):
        attempt=published['attempt']
        if (store.event(attempt,'intent')!=dict(schema=sm.SCHEMA,key=store.key,attempt=attempt,expected=self.expected) or
                store.event(attempt,'candidate')!=published['candidate']):
            raise ValueError('shared_publication_attempt')
        store.attempts()
        for role in ('writer','verifier'):
            proof=published[role]
            if (store.event(attempt,role+'-intent')!=dict(role=role,nonce=proof['nonce'],image=proof['image'],candidate=published['candidate']) or
                    store.event(attempt,role+'-created')!=dict(id=proof['id']) or store.event(attempt,role+'-terminal')!=proof):
                raise ValueError('shared_publication_actor_record')
        candidate=published['candidate']
        actors=[published['writer'],published['verifier']]+self.readers(store,candidate,recover=recover)
        self.consumers(candidate,actors,store)
        proof,observation=self.new_reader(store,candidate)
        if observation!=published['observation']: raise ValueError('shared_published_tree_changed')
        self.consumers(candidate,actors+[proof],store)
        if store.publication()!=published: raise ValueError('shared_publication_changed')
        return published


def main():
    role,nonce=sys.argv[1:]
    if role not in ROLES: raise SystemExit('shared helper refused')
    wire=Wire(sys.stdin.buffer,sys.stdout.buffer,nonce,role)
    operation='unknown'; stage='receive'
    try:
        action,request=wire.receive()
        if action!='begin': raise ValueError('shared_begin')
        sm.closed(request,'expires operation daemon catalog candidate expected attempt image'+(' selection' if request.get('operation') in ('reserve','bind','attachment-intent','attachment-created') else ''))
        end=sm.deadline(request['expires']); wire.deadline=end
        sm.dv.daemon(request['daemon']); sm.volume(request['catalog'])
        if type(request['operation']) is str and request['operation'] in OPERATIONS:
            operation=request['operation']
        stage='validated'
        def rpc(action,payload):
            sm.remaining(end); wire.send(action,payload)
            returned,value=wire.receive()
            if returned!=action: raise ValueError('shared_rpc_action')
            sm.remaining(end); return value
        stage='mount'
        if role=='controller':
            mount_policy(sm.CATALOG_ROOT,False)
            catalog=sm.Catalog(sm.CATALOG_ROOT,request['daemon'],request['catalog'],end)
            controller=Controller(catalog,rpc,request['image'],request['expected'])
            stage='operation'
            result=(catalog.init() if request['operation']=='init' else controller.coordinate(request['operation'],request['selection']) if 'selection' in request else controller.run(request['operation'],request['attempt']))
        else:
            key=sm.descriptor(request['expected'])[0]
            sm.volume(request['candidate'],key,request['attempt'])
            mount_policy(sm.MODEL_ROOT,role=='verifier')
            stage='operation'
            if role=='writer':
                def chunk(name,offset):
                    result=rpc('input',dict(path=name,offset=offset)); sm.closed(result,'data')
                    return base64.b64decode(result['data'],validate=True)
                result=import_tree(sm.MODEL_ROOT,request['expected'],chunk,end)
            else: result=sm.verify_tree(sm.MODEL_ROOT,request['expected'],end)
        wire.send('complete',dict(result=result))
        stage='complete'
        wire.finish()
    except BaseException as exc:
        diagnostic(exc,role=role,nonce=nonce,operation=operation,stage=stage)
        raise SystemExit('shared helper refused; retained state requires review') from None
