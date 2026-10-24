"""Single owned exec native worker. No Core, Linux containment or restart adoption."""
import base64
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import socket
import stat
import subprocess
import threading
import time
import wave

from worker import native_protocol as p
from worker.transport import Refused, canonical

QUOTA = 64 * 1024 * 1024


def available_ram():
    try:
        pages=os.sysconf('SC_AVPHYS_PAGES'); size=os.sysconf('SC_PAGE_SIZE')
        return pages*size if pages>0 and size>0 else 0
    except (ValueError,OSError): return 0


def private(path, directory=False):
    path = Path(path).absolute()
    for parent in reversed(path.parents):
        if parent.is_symlink(): raise Refused('native_storage')
    info = path.lstat()
    if (path.is_symlink() or info.st_uid != os.getuid() or
        (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)) or
        (directory and not stat.S_ISDIR(info.st_mode)) or
        stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)):
        raise Refused('native_storage')
    return path


def read_private(path, cap):
    private(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if info.st_nlink != 1 or info.st_size > cap: raise Refused('native_storage')
        return stream.read(cap + 1)


def pcm(data, spec, reference=False):
    try:
        with wave.open(io.BytesIO(data), 'rb') as wav:
            frames = spec.reference_frames if reference else None
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, spec.reference_rate if reference else spec.sample_rate, 'NONE'):
                raise ValueError()
            n = wav.getnframes()
            if n <= 0 or (frames is not None and n != frames) or (not reference and n > spec.max_frames): raise ValueError()
            if len(wav.readframes(n + 1)) != n * 2 or len(data) != 44 + n * 2: raise ValueError()
    except (ValueError, EOFError, wave.Error): raise Refused('native_pcm') from None


class Journal:
    def __init__(self, root, deployment, configuration):
        self.root = private(root, True)
        fd = os.open(self.root/'native.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, 'a+b')
        private(self.root/'native.lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.deployment, self.configuration = deployment, configuration
        path = self.root/'native.json'
        if path.exists():
            raw = read_private(path, 1048576)
            value = json.loads(raw)
            p._closed(value, ('payload', 'sha256'))
            if hashlib.sha256(canonical(value['payload'])).hexdigest() != value['sha256']: raise Refused('native_journal')
            self.value = value['payload']; self.validate(self.value)
            from worker.native_child import validate_proposal
            for r in self.value['records']:
                if r['output'] is not None:
                    data=read_private(private(self.root/r['token'],True)/'output',1048576)
                    if len(data)!=r['output']['bytes'] or hashlib.sha256(data).hexdigest()!=r['output']['sha256']: raise Refused('native_journal')
                    validate_proposal(r['snapshot'],r['proposal'],data,p.native_execution(r['snapshot'][1]))
        else:
            if set(x.name for x in self.root.iterdir())!={'native.lock'}: raise Refused('native_storage')
            self.save(dict(version='native-journal-v1', deployment_id=deployment,
                configuration_sha256=configuration, lane=secrets.token_hex(16), records=[]))
        expected={'native.lock','native.json'} | {r['token'] for r in self.value['records']}
        if set(x.name for x in self.root.iterdir())!=expected: raise Refused('native_storage')
        for r in self.value['records']: private(self.root/r['token'],True)

    def validate(self, value):
        p._closed(value, ('version','deployment_id','configuration_sha256','lane','records'))
        if (value['version'],value['deployment_id'],value['configuration_sha256']) != ('native-journal-v1',self.deployment,self.configuration): raise Refused('native_journal')
        p._hex(value['lane'],32)
        if type(value['records']) is not list or len(value['records']) > 64: raise Refused('native_journal')
        seen = set()
        for r in value['records']:
            p._closed(r, ('id','boot','binding','snapshot','phase','acked','input_size','receipt','proposal','output','token'))
            p.runtime_id(r['id'],value['lane'],r['boot']); p._hex(r['token'],32)
            if r['id'] in seen or r['phase'] not in ('prepared','launch-intent','terminal') or type(r['acked']) is not bool: raise Refused('native_journal')
            seen.add(r['id'])
            if hashlib.sha256(canonical(r['snapshot'])).hexdigest() != r['binding']: raise Refused('native_journal')
            p._int(r['input_size'],0,24000000)
            from contracts import validate as oracle
            oracle.stored_context(r['snapshot'][0],r['snapshot'][1])
            spec=p.native_execution(r['snapshot'][1])
            if r['input_size']>spec.reference_bytes: raise Refused('native_journal')
            if r['phase'] == 'terminal':
                p.receipt(r['receipt'],expected=self.expected(r,value['lane']))
                if r['receipt']['timely'] and (r['proposal'] is None or r['output'] is None): raise Refused('native_journal')
                if (r['proposal'] is None)!=(r['output'] is None): raise Refused('native_journal')
                if r['output'] is not None:
                    p._closed(r['output'],('sha256','bytes')); p._hex(r['output']['sha256'],64); p._int(r['output']['bytes'],46,1048576)
            elif r['receipt'] is not None or r['proposal'] is not None or r['output'] is not None or r['acked']: raise Refused('native_journal')
        if sum(not r['acked'] for r in value['records']) > 8: raise Refused('native_journal')

    def expected(self,r,lane=None):
        return dict(runtime_id=r['id'],binding=r['binding'],deployment_id=self.deployment,
            configuration_sha256=self.configuration,lane=lane or self.value['lane'],boot=r['boot'],original_deadline=r['snapshot'][0]['deadline'])

    def save(self,value):
        self.validate(value)
        raw = canonical(dict(payload=value,sha256=hashlib.sha256(canonical(value)).hexdigest()))
        if len(raw)>1048576: raise Refused('native_journal_limit')
        # Never truncate a pre-existing temporary file after an uncertain write.
        fd = os.open(self.root/'native.tmp',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(self.root/'native.tmp',self.root/'native.json')
        fd = os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
        self.value = value


class NativeSupervisor:
    """Registrations are trusted exact snapshots, not a wire-selected validator.

    Each registration contains snapshot, typed execution, sealed bundles and
    enrolled inventory. Future foundation wiring constructs these registrations.
    """
    def __init__(self, root, *, deployment_id, configuration_sha256, peer_pin,
                 runtime_manifest_sha256, registrations, python, child_entry,
                 popen=subprocess.Popen, pair=socket.socketpair,
                 signal_group=os.killpg, identity=lambda handle: None,
                 clock=time.monotonic, utc=lambda: datetime.now(timezone.utc),
                 free_bytes=None, free_ram=available_ram):
        for v,n in ((deployment_id,32),(configuration_sha256,64),(peer_pin,64),(runtime_manifest_sha256,64)): p._hex(v,n)
        if not Path(python).is_absolute() or child_entry!=str(Path(__file__).with_name('native_child.py').resolve()): raise Refused('native_command')
        self.journal=Journal(root,deployment_id,configuration_sha256)
        self.boot=secrets.token_hex(16); self.pin=peer_pin; self.manifest=runtime_manifest_sha256
        if type(registrations) is not dict or not 1<=len(registrations)<=64: raise Refused('native_registration')
        self.registrations=copy.deepcopy(registrations)
        self.python,self.entry=python,child_entry
        self.popen,self.pair,self.signal_group,self.identity=popen,pair,signal_group,identity
        self.clock,self.utc=clock,utc
        self.free_ram=free_ram
        self.free_bytes=free_bytes or (lambda: shutil.disk_usage(self.journal.root).free)
        self.lock=threading.RLock(); self.transfer=threading.Lock()
        # Exact handle selection/signalling and the terminal decision use this lock.
        self.handle_lock=threading.Lock(); self.uncertain=set()
        self.active=None; self.failed=False; self.launches=0
        self.cutoffs={}; self.cancelled=set(); self.slots=threading.BoundedSemaphore(3)
        self.shutdown_event=threading.Event(); self.transfer_slot=threading.BoundedSemaphore(1)

    @property
    def quarantined(self):
        return self.failed or any(r['phase']!='terminal' and r['id'] not in self.cutoffs for r in self.journal.value['records'])

    def persist(self,r,**changes):
        with self.lock:
            value=copy.deepcopy(self.journal.value)
            target=next(x for x in value['records'] if x['id']==r['id']); target.update(changes)
            try: self.journal.save(value)
            except Exception:
                self.failed=True; self.uncertain.add(r['id'])
                raise Refused('native_unknown') from None
            return target

    def registration(self,r,verify=False):
        reg=self.registrations.get(r['binding'])
        if reg is None or reg['snapshot'] != r['snapshot']: raise Refused('native_registration')
        e=reg['execution']
        from contracts import validate as oracle
        job,context,assets,profiles=r['snapshot']
        oracle.admission(job,context,profiles)
        attempt=oracle.active_attempt(job)
        if job['state'] not in ('running','cancel_requested') or attempt['reaped']: raise Refused('native_attempt')
        if type(assets) is not list or len(assets)>256 or any(type(a) is not str for a in assets) or len(set(assets))!=len(assets) or not set(oracle.required_asset_ids(context['plan']))<=set(assets): raise Refused('native_assets')
        if p.native_execution(r['snapshot'][1]) != e: raise Refused('native_registration')
        from worker.engines.qwen_execution import inventory_wire
        if r['snapshot'][1]['execution']['hardware'] != inventory_wire(reg['inventory']): raise Refused('native_registration')
        if (e.device,e.dtype,e.attention,e.task) not in (('mps','float32','sdpa','custom_voice'),('mps','float32','sdpa','clone')) or e.timeout_s>30 or e.max_output_bytes>1048576: raise Refused('native_registration')
        from worker.engines.qwen_execution import current_native_hardware
        current_native_hardware(e,reg['inventory'],now=self.utc())
        if verify:
            from worker.engines.qwen import verify_service_inputs
            verify_service_inputs(e,reg['bundles'],self.references(r,reg),reg['inventory'])
        return reg

    def register_snapshot(self, snapshot):
        """Internal provisioning helper; deliberately absent from the wire OPS.

        Extend an existing trusted principal/version/profile template to one exact
        owner allocation. It cannot add model mappings, hardware or descriptors.
        The caller must be the trusted native configuration owner, before prepare.
        """
        from contracts import validate as oracle
        from service.worker_transport import Authority
        authority=Authority(*snapshot)
        def template(value):
            context=copy.deepcopy(value[1]);context['plan'].pop('request')
            return value[0]['principal_id'],context,value[3]
        with self.lock:
            matches=[r for r in self.registrations.values() if template(r['snapshot'])==template(snapshot)]
            if not matches: raise Refused('native_registration')
            oracle.admission(authority.job,authority.context,authority.profiles)
            authority.check(authority.message(),'execute')
            reg=copy.deepcopy(matches[0]);reg['snapshot']=json.loads(authority._snapshot)
            binding=authority.binding
            if binding in self.registrations:
                if self.registrations[binding]!=reg: raise Refused('native_registration')
                return binding
            if len(self.registrations)>=64: raise Refused('native_registration')
            self.registrations[binding]=reg
            return binding

    def directory(self,r): return self.journal.root/r['token']
    def references(self,r,reg):
        e=reg['execution']
        if e.task!='clone': return {}
        return {e.reference_id:dict(path=str(self.directory(r)/('input-'+e.reference_sha256)),voice_id=e.voice_id,voice_revision=e.voice_revision,checkpoint=e.checkpoint,revision=e.revision,text=e.reference_text)}

    def dispatch(self,request):
        q=p.request(request)
        if q['deployment_id']!=self.journal.deployment or q['configuration_sha256']!=self.journal.configuration: raise Refused('native_session')
        op=q['op']
        # Stop must reach the owned handle even while a previous ACK holds the
        # state lock in fsync. Journal values are replaced, never edited in place.
        if op=='stop':
            r=next((r for r in self.journal.value['records'] if r['id']==q['runtime_id']),None)
            if (q['lane']!=self.journal.value['lane'] or r is None or
                (q['binding'],q['boot'])!=(r['binding'],r['boot']) or q['boot']!=self.boot): raise Refused('native_session')
            with self.handle_lock:
                if self.active and self.active['id']==r['id']:
                    a=self.active
                    if a.get('decision') is not None:
                        if self.failed or r['id'] in self.uncertain: raise Refused('native_unknown')
                        return dict(state='terminal' if r['phase']=='terminal' else 'committing',timely=a['decision'])
                    a['cancel']=True
                    return {}
        with self.lock:
            if op=='hello':
                from worker.engines.qwen_execution import inventory_wire
                inventories=[inventory_wire(r['inventory']) for r in self.registrations.values()]
                if not inventories or any(i!=inventories[0] for i in inventories): raise Refused('native_registration')
                return dict(deployment_id=self.journal.deployment,configuration_sha256=self.journal.configuration,lane=self.journal.value['lane'],boot=self.boot,quarantined=self.quarantined,launches=self.launches,runtime_manifest_sha256=self.manifest,inventory=inventories[0])
            if q['lane']!=self.journal.value['lane']: raise Refused('native_session')
            if op=='prepare': return self.prepare(q)
            r=next((r for r in self.journal.value['records'] if r['id']==q['runtime_id']),None)
            if r is None or (q['binding'],q['boot'])!=(r['binding'],r['boot']): raise Refused('native_session')
            if op not in ('observe','collect','ack','get_output') and q['boot']!=self.boot: raise Refused('native_session')
            if op=='observe': return dict(runtime_id=r['id'],state=self.state(r))
            if op=='collect':
                if r['id'] in self.uncertain: return dict(receipt=None,proposal=None,output=None)
                return dict(receipt=r['receipt'],proposal=r['proposal'],output=r['output'])
            if op=='ack':
                if self.failed or r['id'] in self.uncertain: raise Refused('native_unknown')
                if r['phase']!='terminal': raise Refused('native_state')
                self.persist(r,acked=True) # Keep bounded transfer evidence; no automatic cleanup.
                return {}
            if self.quarantined and op!='get_output': raise Refused('native_unknown')
            if op=='stop':
                if r['phase']=='terminal': return dict(state='terminal',timely=r['receipt']['timely'])
                if r['phase']=='prepared':
                    self.cancelled.add(r['id'])
                    if self.transfer.acquire(False):
                        try: self.terminal(r,[],False)
                        finally: self.transfer.release()
                else:
                    with self.handle_lock:
                        if self.active and self.active['id']==r['id']: self.active['cancel']=True
                return {}
        if op=='start': return self.start(r)
        # Transfer never holds the launch/watchdog lock while hashing or fsync.
        if not self.transfer.acquire(blocking=False): raise Refused('native_busy')
        try:
            if op=='put_input':
                try: return self.put(r,q['body'])
                except OSError:
                    self.failed=True; raise Refused('native_unknown') from None
            if op=='get_output':
                b=q['body']
                if r['id'] in self.uncertain or r['phase']!='terminal' or r['output'] is None or b['sha256']!=r['output']['sha256'] or b['offset']+b['bytes']>r['output']['bytes']: raise Refused('native_output')
                data=read_private(self.directory(r)/'output',1048576)
                if hashlib.sha256(data).hexdigest()!=b['sha256']: self.failed=True; raise Refused('native_output')
                return dict(sha256=b['sha256'],offset=b['offset'],data=base64.b64encode(data[b['offset']:b['offset']+b['bytes']]).decode())
        finally: self.transfer.release()
        raise Refused('native_protocol')

    def state(self,r):
        if r['id'] in self.uncertain: return 'unknown'
        if r['phase']=='terminal': return 'stopped-and-reaped'
        if r['id'] in self.cutoffs and not self.failed: return 'alive'
        return 'unknown'

    def prepare(self,q):
        if q['boot']!=self.boot: raise Refused('native_session')
        snapshot=json.loads(canonical(q['body']['snapshot'])); binding=hashlib.sha256(canonical(snapshot)).hexdigest()
        if binding!=q['binding']: raise Refused('native_binding')
        for r in self.journal.value['records']:
            if r['binding']==binding and r['snapshot']==snapshot: return dict(runtime_id=r['id'],state=self.state(r))
            a,b=r['snapshot'][0],snapshot[0]
            if (a['id'],a['generation'],a['attempts'][-1])==(b['id'],b['generation'],b['attempts'][-1]): raise Refused('native_binding')
        if self.quarantined or any(r['phase']!='terminal' for r in self.journal.value['records']) or len(self.journal.value['records'])>=64 or sum(not r['acked'] for r in self.journal.value['records'])>=8: raise Refused('native_busy')
        r=dict(id='/'.join((self.journal.value['lane'],self.boot,secrets.token_hex(16))),boot=self.boot,binding=binding,snapshot=snapshot,phase='prepared',acked=False,input_size=0,receipt=None,proposal=None,output=None,token=secrets.token_hex(16))
        reg=self.registration(r); e=reg['execution']
        reserved=sum(2*(p.native_execution(x['snapshot'][1]).reference_bytes+p.native_execution(x['snapshot'][1]).max_output_bytes) for x in self.journal.value['records'])
        if reserved+2*(e.reference_bytes+e.max_output_bytes)>QUOTA: raise Refused('native_quota')
        if self.free_bytes()<6*1024**3+snapshot[1]['plan']['resolved']['profile']['limits']['disk_bytes']: raise Refused('native_disk_gate')
        if self.free_ram()<e.memory_bytes: raise Refused('native_ram_gate')
        from worker.engines.qwen import verify_bundle
        if set(reg['bundles'])!={e.bundle_id}: raise Refused('native_registration')
        verify_bundle(reg['bundles'][e.bundle_id],e.manifest_sha256,e.checkpoint,e.revision,e.language,owner_uid=os.getuid(),execution=e)
        budget=(datetime.fromisoformat(snapshot[0]['deadline'])-self.utc()).total_seconds()
        if not 0<budget<=30: raise Refused('native_deadline')
        cutoff=self.clock()+budget
        value=copy.deepcopy(self.journal.value); value['records'].append(r)
        try:
            self.journal.save(value)
            os.mkdir(self.directory(r),0o700)
            fd=os.open(self.journal.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try: os.fsync(fd)
            finally: os.close(fd)
        except Exception:
            self.failed=True; self.uncertain.add(r['id']); raise Refused('native_unknown') from None
        self.cutoffs[r['id']]=cutoff
        return dict(runtime_id=r['id'],state='alive')

    def put(self,r,b):
        with self.lock:
            r=next(x for x in self.journal.value['records'] if x['id']==r['id'])
            if r['id'] in self.cancelled: raise Refused('native_cancelled')
        reg=self.registration(r); e=reg['execution']
        if r['phase']!='prepared' or self.clock()>=self.cutoffs[r['id']] or (b['sha256'],b['bytes'])!=(e.reference_sha256,e.reference_bytes): raise Refused('native_input')
        data=p.chunk(b['data']); path=self.directory(r)/('input-'+e.reference_sha256)
        if b['offset']<r['input_size']:
            old=read_private(path,24000000)
            if b['offset']+len(data)>r['input_size'] or old[b['offset']:b['offset']+len(data)]!=data: raise Refused('native_replay')
            return {}
        if b['offset']!=r['input_size']: raise Refused('native_offset')
        flags=os.O_WRONLY|os.O_NOFOLLOW|(os.O_CREAT|os.O_EXCL if r['input_size']==0 else os.O_APPEND)
        if r['input_size']: private(path)
        fd=os.open(path,flags,0o600)
        try:
            if os.fstat(fd).st_size!=r['input_size']: raise Refused('native_input')
            with os.fdopen(fd,'ab',closefd=False) as out: out.write(data); out.flush(); os.fsync(fd)
        finally: os.close(fd)
        size=r['input_size']+len(data)
        if size==e.reference_bytes:
            raw=read_private(path,24000000)
            if hashlib.sha256(raw).hexdigest()!=e.reference_sha256: self.failed=True; raise Refused('native_input')
            try: pcm(raw,e,True)
            except Exception: self.failed=True; raise
        directory=os.open(self.directory(r),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: os.fsync(directory)
        finally: os.close(directory)
        self.persist(r,input_size=size)
        return {}

    def start(self,r):
        if r['phase']!='prepared': return {}
        if not self.transfer.acquire(False): raise Refused('native_busy')
        try:
            reg=self.registration(r,True); e=reg['execution']
            if self.failed or r['phase']!='prepared': raise Refused('native_unknown')
            if r['input_size']!=e.reference_bytes or self.clock()>=self.cutoffs[r['id']]: raise Refused('native_deadline')
            with self.lock:
                r=next(x for x in self.journal.value['records'] if x['id']==r['id'])
                if self.failed or r['phase']!='prepared' or r['id'] in self.cancelled: return {}
                conditioning=self.directory(r)/'conditioning'
                if e.task=='clone': os.mkdir(conditioning,0o700)
                r=self.persist(r,phase='launch-intent')
                parent,child=self.pair(); token=secrets.token_hex(16)
                a=dict(id=r['id'],handle=None,parent=parent,receiver=None,closed=False,result=None,token=token,cancel=False,term=None,killed=None,exit=None,start_identity=None)
                with self.handle_lock: self.active=a
                try:
                    handle=self.popen([self.python,'-I','-B',self.entry,str(child.fileno())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,pass_fds=(child.fileno(),),start_new_session=True,env={'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
                    with self.handle_lock:
                        a['handle']=handle; a['start_identity']=self.identity(handle); self.launches+=1
                except Exception:
                    self.failed=True
                    try: parent.close()
                    finally: child.close()
                    raise Refused('native_unknown') from None
                child.close()
                from worker.engines.qwen_execution import inventory_wire
                config=dict(token=token,binding=r['binding'],snapshot=r['snapshot'],execution=e.identity(),bundles={k:str(v) for k,v in reg['bundles'].items()},references=self.references(r,reg),inventory=inventory_wire(reg['inventory']),conditioning=str(conditioning) if e.task=='clone' else None,cutoff=self.cutoffs[r['id']])
                def receive():
                    from worker.native_child import parent_exchange
                    try: a['result']=parent_exchange(parent,config,self.cutoffs[r['id']])
                    except Exception: a['result']=None
                    finally:
                        try: parent.close(); a['closed']=parent.fileno()==-1
                        except Exception: self.failed=True
                a['receiver']=threading.Thread(target=receive,daemon=True); a['receiver'].start()
                return {}
        finally: self.transfer.release()

    def terminal(self,r,processes,timely,proposal=None,output=None):
        if self.failed or r['id'] in self.uncertain: raise Refused('native_unknown')
        receipt=dict(self.journal.expected(r),version='native-receipt-v1',reaped=True,timely=timely,reap_method='darwin-owned-exec-v1',processes=processes,receiver_closed=True)
        self.persist(r,phase='terminal',receipt=receipt,proposal=proposal,output=output)

    def signal_tick(self):
        # No journal, hashing, Popen, receiver join or blocking wait here.
        with self.handle_lock:
            a=self.active
            if a is None or a['handle'] is None: return
            h=a['handle']
            try:
                if a['exit'] is not None: return
                status=h.poll()
                if status is not None:
                    a['exit']=status
                    return
                if a['cancel'] or self.clock()>=self.cutoffs[a['id']]:
                    if self.identity(h)!=a['start_identity']: raise Refused('native_identity')
                    if a['term'] is None:
                        self.signal_group(h.pid,signal.SIGTERM); a['term']=self.clock()
                    elif self.clock()-a['term']>=.2 and a['killed'] is None:
                        self.signal_group(h.pid,signal.SIGKILL); a['killed']=self.clock()
                    elif a['killed'] is not None and self.clock()-a['killed']>=1:
                        self.failed=True
            except Exception: self.failed=True

    def tick(self):
        self.signal_tick()
        # A blocked durable operation never delays the next signalling pass.
        if not self.lock.acquire(False): return
        try: self.finish_tick()
        finally: self.lock.release()

    def finish_tick(self):
        a=self.active
        if a is None:
            if self.failed: return
            for r in self.journal.value['records']:
                if r['phase']=='prepared' and r['id'] in self.cutoffs and (self.clock()>=self.cutoffs[r['id']] or r['id'] in self.cancelled):
                    if self.transfer.acquire(False):
                        try: self.terminal(r,[],False)
                        except Exception: self.failed=True
                        finally: self.transfer.release()
            return
        h=a['handle']
        if h is None: return
        r=next(r for r in self.journal.value['records'] if r['id']==a['id'])
        try:
            if a['exit'] is None: return
            if not a.get('waited'):
                if a.get('wait_attempted'): return
                a['wait_attempted']=True
                code=h.wait(timeout=1)
                if type(code) is not int or code!=a['exit']: raise Refused('native_wait')
                a['waited']=True
            if self.failed: return
            receiver=a['receiver']
            if receiver is None: return
            receiver.join(timeout=0)
            if receiver.is_alive():
                if self.clock()>self.cutoffs[r['id']]+1: self.failed=True
                return
            if not a['closed']: self.failed=True; return
            reg=self.registration(r); e=reg['execution']
            if e.task=='clone':
                directory=private(self.directory(r)/'conditioning',True)
                if list(directory.iterdir()): self.failed=True; return
                directory.rmdir()
            proposal=output=None
            result=a['result']
            timely=False
            if result is not None and a['exit']==0 and not a['cancel']:
                proposal,data=result
                from worker.native_child import validate_proposal
                validate_proposal(r['snapshot'],proposal,data,e)
                fd=os.open(self.directory(r)/'output',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
                directory=os.open(self.directory(r),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try: os.fsync(directory)
                finally: os.close(directory)
                output=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            # Linearization point: stop either latched cancellation before this
            # decision, or observes this immutable decision without acknowledging
            # cancellation. All validation/output I/O precedes it; terminal journal
            # I/O follows it without holding handle_lock. Ownership lasts through
            # durable persistence, and failure never grants receipt authority.
            with self.handle_lock:
                timely=proposal is not None and self.clock()<self.cutoffs[r['id']]
                if a['cancel']:
                    timely=False; proposal=output=None
                a['decision']=timely
            process=dict(role='model',spawn_token=a['token'],pid=h.pid,pgid=h.pid,start_identity=a['start_identity'],exit_code=a['exit'])
            self.terminal(r,[process],timely,proposal,output)
            with self.handle_lock:
                self.active=None
        except Exception: self.failed=True

    def watchdog(self):
        while not self.shutdown_event.wait(.02): self.tick()

    def serve_connection(self,connection):
        slot=None
        try:
            cutoff=self.clock()+2
            q=p.request(p.receive(connection,cutoff,peer_sha256=self.pin))
            slot=self.transfer_slot if q['op'] in ('put_input','get_output') else self.slots
            if not slot.acquire(False): slot=None; raise Refused('native_busy')
            envelope={k:v for k,v in q.items() if k!='body'}
            try:
                value=self.dispatch(q)
                response=p.response(dict(envelope,ok=True,value=value),request=q)
            except Refused:
                response=p.response(dict(envelope,ok=False,error='native_unknown' if self.quarantined else 'native_refused'),request=q)
            except Exception:
                self.failed=True
                response=p.response(dict(envelope,ok=False,error='native_unknown'),request=q)
            p.send(connection,response,cutoff,peer_sha256=self.pin)
        finally:
            connection.close()
            if slot is not None: slot.release()


def serve(supervisor, context, host, port, *, proven_host_network=False,
          socket_factory=socket.socket):
    """Production listener; never invoked by inert tests or imported for side effects."""
    import ssl
    host,port=p.numeric_bind(host,port,proven_host_network=proven_host_network)
    if context.minimum_version!=ssl.TLSVersion.TLSv1_3 or context.maximum_version!=ssl.TLSVersion.TLSv1_3 or context.verify_mode!=ssl.CERT_REQUIRED: raise Refused('native_tls')
    watchdog=threading.Thread(target=supervisor.watchdog,daemon=True); watchdog.start()
    connections=threading.BoundedSemaphore(4)
    with socket_factory(socket.AF_INET6 if ':' in host else socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind((host,port)); listener.listen(8); listener.settimeout(.2)
        try:
            while not supervisor.shutdown_event.is_set():
                try: raw,_=listener.accept()
                except socket.timeout: continue
                if not connections.acquire(False): raw.close(); continue
                raw.settimeout(2)
                try: connection=context.wrap_socket(raw,server_side=True)
                except Exception: raw.close(); connections.release(); continue
                def handle(connection=connection):
                    try: supervisor.serve_connection(connection)
                    finally: connections.release()
                threading.Thread(target=handle,daemon=True).start()
        finally:
            with supervisor.handle_lock:
                if supervisor.active and supervisor.active.get('decision') is None: supervisor.active['cancel']=True
            end=time.monotonic()+2
            while supervisor.active and time.monotonic()<end: supervisor.tick(); time.sleep(.02)
            supervisor.shutdown_event.set(); watchdog.join(timeout=1)
            if supervisor.active: raise Refused('native_unknown')


def main():
    """Explicit private operator configuration; no installed-policy activation."""
    import argparse
    from worker.engines.qwen_execution import QwenExecution, inventory_from_wire
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=json.loads(read_private(args.config,1048576))
    p._closed(config,('root','deployment_id','configuration_sha256','peer_pin','runtime_manifest_sha256','registrations','python','child_entry','host','port','proven_host_network','cafile','certfile','keyfile'))
    for key in ('cafile','certfile','keyfile'): private(config[key])
    for key in ('python','child_entry'):
        path=Path(config[key])
        if not path.is_absolute() or path.is_symlink(): raise Refused('native_command')
        info=path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode & 0o022: raise Refused('native_command')
    registrations={}
    for r in config['registrations']:
        p._closed(r,('snapshot','execution','bundles','inventory'))
        binding=hashlib.sha256(canonical(r['snapshot'])).hexdigest()
        if binding in registrations: raise Refused('native_registration')
        registrations[binding]=dict(r,execution=QwenExecution.from_wire(r['execution']),inventory=inventory_from_wire(r['inventory']))
    supervisor=NativeSupervisor(**{k:config[k] for k in ('root','deployment_id','configuration_sha256','peer_pin','runtime_manifest_sha256','python','child_entry')},registrations=registrations)
    context=p.tls_context(server=True,**{k:config[k] for k in ('cafile','certfile','keyfile')})
    serve(supervisor,context,config['host'],config['port'],proven_host_network=config['proven_host_network'])


if __name__=='__main__': main()
