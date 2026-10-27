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
    try:
        action,request=wire.receive()
        if action!='begin': raise ValueError('shared_begin')
        sm.closed(request,'expires operation daemon catalog candidate expected attempt image'+(' selection' if request.get('operation') in ('reserve','bind','attachment-intent','attachment-created') else ''))
        end=sm.deadline(request['expires']); wire.deadline=end
        sm.dv.daemon(request['daemon']); sm.volume(request['catalog'])
        def rpc(action,payload):
            sm.remaining(end); wire.send(action,payload)
            returned,value=wire.receive()
            if returned!=action: raise ValueError('shared_rpc_action')
            sm.remaining(end); return value
        if role=='controller':
            mount_policy(sm.CATALOG_ROOT,False)
            catalog=sm.Catalog(sm.CATALOG_ROOT,request['daemon'],request['catalog'],end)
            controller=Controller(catalog,rpc,request['image'],request['expected'])
            result=(catalog.init() if request['operation']=='init' else controller.coordinate(request['operation'],request['selection']) if 'selection' in request else controller.run(request['operation'],request['attempt']))
        else:
            key=sm.descriptor(request['expected'])[0]
            sm.volume(request['candidate'],key,request['attempt'])
            mount_policy(sm.MODEL_ROOT,role=='verifier')
            if role=='writer':
                def chunk(name,offset):
                    result=rpc('input',dict(path=name,offset=offset)); sm.closed(result,'data')
                    return base64.b64decode(result['data'],validate=True)
                result=import_tree(sm.MODEL_ROOT,request['expected'],chunk,end)
            else: result=sm.verify_tree(sm.MODEL_ROOT,request['expected'],end)
        wire.send('complete',dict(result=result))
        wire.finish()
    except BaseException:
        raise SystemExit('shared helper refused; retained state requires review') from None
