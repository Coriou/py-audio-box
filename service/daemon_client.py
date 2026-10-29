"""Stdlib-only ordinary Mac/Linux client. Never imports service dependencies.

All daemon traffic uses the explicitly resolved Unix endpoint, with ambient
Docker overrides removed. No shell, acquisition, rollback, rm or KILL path.
"""
from contextlib import contextmanager
import argparse
import base64
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import stat
import subprocess
import sys
import threading
import time

from service import daemon_volumes as dv

LIMIT=1024*1024
TOTAL=2*1024**3
INPUT_CHUNK=256*1024  # Base64 plus framing remains below the 1 MiB wire limit.
_HELPER_DIAGNOSTIC=re.compile(r'daemon_helper_diagnostic:(install|init|up|status|stop):(operation_failed|prior_failure|final_refusal):(ValueError|OSError|RuntimeError|TimeoutError|FileNotFoundError|PermissionError|BlockingIOError|KeyboardInterrupt|Exception):([a-z_]+|unknown):([0-9]{1,5})(?::(daemon_helper\.py|daemon_client\.py|install_policy\.py|supervisor_runtime\.py|persistent_init\.py|install_child\.py|unknown):([0-9]{1,3}|unknown))?')
# Linux errno values emitted by the root-only container helper.
_HELPER_ERRNOS=frozenset((1,2,5,6,9,11,12,13,16,17,19,20,22,23,24,28,30,32,38,40,110,122))
_HELPER_CAUSE=re.compile(r'daemon_helper_cause:(install|init|up|status|stop):(operation_failed|prior_failure|final_refusal):([a-z][a-z0-9_]{1,79})')
_HELPER_CHANGE=re.compile(r'daemon_helper_change:(install|init|up|status|stop):(operation_failed|prior_failure|final_refusal):([a-z0-9_.,]{1,255})')
_HOST_REFUSAL_CODES=frozenset(('daemon_helper_failed_retained','daemon_helper_start_unknown',
    'daemon_helper_exit_unknown','daemon_helper_timeout_retained',
    'daemon_phase_eof_retained','daemon_phase_timeout','daemon_phase_write_timeout',
    'daemon_terminal_eof_timeout','daemon_trailing_protocol'))
_HELPER_TAIL_LIMIT=4096
_HELPER_LINE_LIMIT=512


class HelperDiagnostics:
    """Drain Docker stderr without retaining untrusted output or blocking its pipe."""
    def __init__(self,phase):
        self.phase=phase; self.tail=bytearray(); self.line=bytearray()
        self.overlong=False; self.details={}; self.causes={}; self.changes={}
        self.stop_event=threading.Event(); self.thread=None; self.stream=None

    def feed(self,data):
        self.tail.extend(data)
        if len(self.tail)>_HELPER_TAIL_LIMIT: del self.tail[:-_HELPER_TAIL_LIMIT]
        for byte in data:
            if byte==10:
                if not self.overlong: self.record(bytes(self.line))
                self.line.clear(); self.overlong=False
            elif not self.overlong:
                if len(self.line)<_HELPER_LINE_LIMIT: self.line.append(byte)
                else: self.line.clear(); self.overlong=True

    def record(self,raw):
        try: line=raw.decode('ascii')
        except UnicodeDecodeError: return
        detail=_HELPER_DIAGNOSTIC.fullmatch(line)
        cause=_HELPER_CAUSE.fullmatch(line)
        change=_HELPER_CHANGE.fullmatch(line)
        if (detail and detail[1]==self.phase and
                (detail[6] is None or (detail[3] in ('OSError','TimeoutError','FileNotFoundError','PermissionError','BlockingIOError')
                    and (detail[7]=='unknown' or int(detail[7]) in _HELPER_ERRNOS)))):
            self.details.setdefault(detail[2],detail)
        if cause and cause[1]==self.phase and cause[3] in dv.HELPER_REFUSAL_CODES:
            self.causes.setdefault(cause[2],cause)
        if change and change[1]==self.phase:
            labels=change[3].split(',')
            if 0<len(labels)<=8 and len(labels)==len(set(labels)) and all(label in dv.OBSERVATION_CHANGE_LABELS for label in labels):
                self.changes.setdefault(change[2],tuple(labels))

    def refusal(self):
        for stage in ('operation_failed','prior_failure','final_refusal'):
            matched=self.details.get(stage)
            if matched:
                reason=self.causes[stage][3] if stage in self.causes else 'unclassified'
                parts=(self.phase,stage,reason,matched[3],matched[4],matched[5])
                if matched[6] is not None: parts+=('file='+matched[6],'errno='+matched[7])
                if reason in ('daemon_observation_pair_changed','daemon_observation_startup_changed') and stage in self.changes:
                    parts+=('changed='+','.join(self.changes[stage]),)
                return ':'.join(parts)
        return None

    def drain(self):
        fd=self.stream.fileno()
        try:
            os.set_blocking(fd,False)
            with selectors.DefaultSelector() as selector:
                selector.register(fd,selectors.EVENT_READ)
                stop_by=None
                while True:
                    if self.stop_event.is_set():
                        if stop_by is None: stop_by=time.monotonic()+.5
                        if time.monotonic()>=stop_by: break
                    if not selector.select(.1):
                        if self.stop_event.is_set(): break
                        continue
                    try: data=os.read(fd,65536)
                    except BlockingIOError: continue
                    if not data: break
                    self.feed(data)
        except (OSError,ValueError):
            # A broken diagnostic pipe cannot replace the owned host failure.
            pass

    def start(self,stream):
        self.stream=stream
        self.thread=threading.Thread(target=self.drain,name='daemon-helper-stderr')
        try: self.thread.start()
        except BaseException:
            stream.close()
            self.thread=None
            raise

    def stop(self):
        if self.thread is not None:
            self.stop_event.set(); self.thread.join()
        if self.stream is not None: self.stream.close()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def parse(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('daemon_duplicate_field')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('daemon_nonfinite')))


def file_identity(s):
    return tuple(getattr(s,k) for k in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns'))


def read(path, limit=LIMIT):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>limit:
            raise ValueError('daemon_input_file')
        raw=f.read(limit+1)
        if len(raw)!=before.st_size or file_identity(os.fstat(f.fileno()))!=file_identity(before) or file_identity(os.stat(path,follow_symlinks=False))!=file_identity(before):
            raise ValueError('daemon_input_changed')
        return raw


def exclusive(path, value):
    path=Path(path)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(encoded(value)); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


class Owner:
    """Synchronous local Popen custody; finite faults cannot release an owner.

    Only the first exception is retained, without its traceback/context. Cleanup
    uses the stream objects, never a numeric descriptor that could be reused.
    """
    def __init__(self,p,prefix,deadline,retained=None):
        self.p=p; self.prefix=prefix; self.deadline=deadline
        self.retained=retained; self.failure=None; self.status=None
        self.selectors=[]

    def fail(self,exc):
        if self.failure is None:
            exc.__traceback__=None; exc.__context__=None; exc.__cause__=None
            self.failure=exc

    def cadence(self):
        end=None
        while True:
            try:
                now=time.monotonic()
                if end is None: end=now+.1
                remaining=end-now
                if remaining<=0: return
                time.sleep(remaining)
            except BaseException as exc: self.fail(exc)

    def __enter__(self):
        # Registration is inside custody, including an append failure.
        try:
            if self.retained is not None: self.retained.append(self.p)
        except BaseException as exc:
            self.fail(exc); self.__exit__(None,None,None)
        return self

    def __exit__(self,kind,exc,tb):
        if exc is not None: self.fail(exc)
        closed=[False,False]
        while self.status is None or not all(closed) or self.selectors:
            for i,name in enumerate(('stdin','stdout')):
                if closed[i]: continue
                try:
                    stream=getattr(self.p,name,None)
                    if stream is not None and not stream.closed: stream.close()
                    closed[i]=stream is None or stream.closed
                except BaseException as error: self.fail(error)
            if self.selectors:
                try:
                    self.selectors[-1].close(); self.selectors.pop()
                except BaseException as error: self.fail(error)
            if self.status is None:
                try:
                    if time.monotonic()>=self.deadline:
                        self.fail(ValueError(self.prefix+'_timeout_retained'))
                    status=self.p.wait(timeout=.1)
                    if type(status) is int: self.status=status
                except subprocess.TimeoutExpired: pass
                except BaseException as error: self.fail(error)
            if self.status is None or not all(closed) or self.selectors: self.cadence()
        if self.retained is not None:
            while True:
                try:
                    if self.p in self.retained: self.retained.remove(self.p)
                    break
                except BaseException as error:
                    self.fail(error); self.cadence()
        if self.failure is not None: raise self.failure from None
        if self.status!=0: raise ValueError(self.prefix+'_failed_retained')
        return False


def collect(p, raw, limit, deadline, prefix, retained=None):
    """Capped live collection; after failure discard through EOF when usable."""
    with Owner(p,prefix,deadline,retained) as owner:
        output=bytearray(); offset=0
        sel=selectors.DefaultSelector()
        owner.selectors.append(sel)
        try:
            if p.stdout is not None:
                os.set_blocking(p.stdout.fileno(),False)
                sel.register(p.stdout,selectors.EVENT_READ)
            if p.stdin is not None:
                os.set_blocking(p.stdin.fileno(),False)
                if raw: sel.register(p.stdin,selectors.EVENT_WRITE)
                else: p.stdin.close()
            while sel.get_map():
                if owner.failure is None and time.monotonic()>=deadline:
                    owner.fail(ValueError(prefix+'_timeout_retained'))
                    output.clear()
                    if p.stdin is not None and not p.stdin.closed:
                        sel.unregister(p.stdin); p.stdin.close()
                for key,_ in sel.select(.1):
                    if key.fileobj is p.stdout:
                        try: data=os.read(p.stdout.fileno(),65536)
                        except BlockingIOError: continue
                        if not data:
                            sel.unregister(p.stdout); p.stdout.close()
                        elif owner.failure is None:
                            if len(data)>limit-len(output):
                                owner.fail(ValueError(prefix+'_output_budget')); output.clear()
                                if p.stdin is not None and not p.stdin.closed:
                                    sel.unregister(p.stdin); p.stdin.close()
                            else: output.extend(data)
                    else:
                        try: offset+=os.write(p.stdin.fileno(),raw[offset:offset+65536])
                        except BlockingIOError: continue
                        except BrokenPipeError: offset=len(raw)
                        if offset==len(raw): sel.unregister(p.stdin); p.stdin.close()
        except BaseException as exc:
            output.clear(); owner.fail(exc)
            raise
    return bytes(output)


class Docker:
    def __init__(self, context=None, *, endpoint=None, family=None):
        self.family=family
        self.env={k:v for k,v in os.environ.items() if not k.startswith('DOCKER_')}
        self.endpoint=endpoint
        self.end=time.monotonic()+120
        self.retained=[]
        if endpoint is None:
            if not isinstance(context,str) or not 1<=len(context)<=128 or context.startswith('-'):
                raise ValueError('daemon_explicit_context_required')
            c=self.json(['context','inspect',context])
            if type(c) is not list or len(c)!=1: raise ValueError('daemon_context')
            self.endpoint=c[0]['Endpoints']['docker']['Host']
        dv.daemon(dict(endpoint=self.endpoint,id='pending',os='linux',architecture='amd64',security=[],cgroup='2'))
        self.frozen=self.identity()

    def argv(self,args):
        return ['docker']+(['--host',self.endpoint] if self.endpoint else [])+args

    def call(self,args):
        deadline=min(self.end,time.monotonic()+15)
        p=subprocess.Popen(self.argv(args),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,env=self.env,close_fds=True,bufsize=0)
        return collect(p,b'',LIMIT,deadline,'daemon_client',self.retained).decode().strip()

    def json(self,args): return parse(self.call(args))

    def identity(self):
        info=self.json(['info','--format','{{json .}}'])
        self.cgroup_driver=info.get('CgroupDriver')
        return dv.daemon(dict(endpoint=self.endpoint,id=info['ID'],os=info['OSType'],architecture=info['Architecture'],
            security=info['SecurityOptions'],cgroup=info['CgroupVersion']))

    def require_resident_cgroupfs(self):
        if self.cgroup_driver!='cgroupfs':
            raise ValueError('qwen_service_cgroupfs_required: Docker info CgroupDriver must be cgroupfs')
        self.resident_cgroupfs=True

    def check(self):
        if self.identity()!=self.frozen: raise ValueError('daemon_endpoint_identity_changed')
        if getattr(self,'resident_cgroupfs',False): self.require_resident_cgroupfs()

    def container(self,identity):
        if not isinstance(identity,str) or not dv.HEX.fullmatch(identity): raise ValueError('daemon_exact_container_id')
        values=self.json(['container','inspect',identity])
        if len(values)!=1 or values[0]['Id']!=identity: raise ValueError('daemon_container_changed')
        return values[0]

    def snapshot(self,b):
        self.check()
        volumes={r:self.json(['volume','inspect',v['Name']])[0] for r,v in b['volumes'].items()}
        if 'model_storage' in b:
            selected=b['model_storage']['catalog']
            current=self.json(['volume','inspect',selected['Name']])
            if len(current)!=1 or {k:current[0][k] for k in selected}!=selected: raise ValueError('shared_catalog_incarnation')
        # Reduce only Docker volume records; no client host stat of Mountpoints.
        volumes={r:{k:v[k] for k in ('Name','Driver','Options','Scope','Mountpoint','CreatedAt','Labels')} for r,v in volumes.items()}
        if volumes!=b['volumes']: raise ValueError('daemon_volume_changed')
        ids=self.call(['container','ls','--all','--no-trunc','--format','{{.ID}}']).splitlines()
        if len(ids)>512: raise ValueError('daemon_consumer_budget')
        paths=[v['Mountpoint'].rstrip('/') for v in volumes.values()]; names={v['Name'] for v in volumes.values()}
        if 'custody' in b:paths.append(b['custody'])
        consumers=[]
        if 'model_storage' in b:
            from service import shared_models as sm
        for identity in ids:
            c=self.container(identity)
            if any(m.get('Name') in names or ('model_storage' in b and sm.source_overlap(m.get('Source'),paths)) or (bool(m.get('Source')) and any(m.get('Source','').rstrip('/')==p or
                m.get('Source','').startswith(p+'/') or p.startswith(m.get('Source','').rstrip('/')+'/')
                for p in paths)) for m in c['Mounts']) or ('custody' in b and c['HostConfig'].get('CgroupParent')==b['custody'].removeprefix('/sys/fs/cgroup')):
                consumers.append(c)
        self.check()
        return dict(daemon=self.frozen,volumes=volumes,consumers=consumers)

    def action(self,action,payload,config,b,source):
        from service import installed_pocket_daemon as pd
        from service import installed_qwen_daemon as qd
        export = pd.is_export(getattr(self,'family',None))
        # Input chunks do no daemon action. The held phase's exact snapshots
        # bracket the stream; avoid a Docker-info process for every input chunk.
        if action!='input': self.check()
        if action=='snapshot':
            dv.closed(payload,''); result=self.snapshot(b)
        elif action=='create':
            dv.closed(payload,'role')
            if payload['role'] not in ('control','worker'): raise ValueError('daemon_action_role')
            if getattr(self,'resident_cgroupfs',False): self.check()
            identity=self.call(dv.command(payload['role'],config,b,family=getattr(self,'family',None)))
            result=self.container(identity)
        elif action in ('start','term','inspect','marker','facts'):
            dv.closed(payload,'id role')
            role=payload['role']; identity=payload['id']
            if role not in ('control','worker'): raise ValueError('daemon_action_role')
            dv.inspect_container(self.container(identity),role,config,b,family=getattr(self,'family',None))
            if action in ('start','term') and getattr(self,'resident_cgroupfs',False): self.check()
            if action=='start': self.call(['container','start',identity])
            elif action=='term': self.call(['container','kill','--signal=TERM',identity])
            if action in ('start','term','inspect'): result=self.container(identity)
            else:
                code=(('from service.entrypoint import qwen_launch_status;qwen_launch_status()' if getattr(self,'family',None) is qd.FAMILY else 'from service.entrypoint import pocket_launch_status;pocket_launch_status()' if getattr(self,'family',None) is pd.FAMILY else 'from service.entrypoint import export_launch_status;export_launch_status()' if export else 'from service.entrypoint import launch_status;launch_status()') if action=='marker' else
                    ('from service.daemon_runtime import read_qwen_control;read_qwen_control()' if getattr(self,'family',None) is qd.FAMILY else 'from service.daemon_runtime import read_pocket_control;read_pocket_control()' if getattr(self,'family',None) is pd.FAMILY else 'from service.daemon_runtime import read_export_control;read_export_control()' if export else 'from service.daemon_runtime import read_control;read_control()'))
                if role!='control': raise ValueError('daemon_control_reader_only')
                result=parse(self.call(['exec','--user','10001:10001',identity,'python','-I','-B','-c',
                    "import sys;sys.path.insert(0,'/opt/speech');"+code]))
        elif action=='notify':
            # Payload is nonsecret. Fixed private notification entrypoint only.
            dv.closed(payload,'id gate')
            identity=payload['id']; dv.inspect_container(self.container(identity),'control',config,b,family=getattr(self,'family',None))
            raw=encoded(payload['gate'])
            if len(raw)>65536: raise ValueError('daemon_gate_budget')
            deadline=min(self.end,time.monotonic()+10)
            if getattr(self,'resident_cgroupfs',False): self.check()
            p=subprocess.Popen(self.argv(['exec','--interactive','--user','10001:10001',identity,'python','-I','-B','-c',
                "import sys;sys.path.insert(0,'/opt/speech');from service.entrypoint import "+("publish_qwen" if getattr(self,"family",None) is qd.FAMILY else "publish_pocket" if getattr(self,"family",None) is pd.FAMILY else "publish_export" if export else "publish")+" as publish;publish(sys.stdin.buffer.read(65537))"]),
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=self.env)
            collect(p,raw,0,deadline,'daemon_notify',self.retained)
            result={}
        elif action=='input':
            dv.closed(payload,'index offset')
            index,offset=payload['index'],payload['offset']
            if type(index) is not int or not 0<=index<len(source) or type(offset) is not int or offset<0:
                raise ValueError('daemon_input_frame')
            entry=source[index]
            fd=os.open(entry['source'],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as f:
                before=os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size!=entry['bytes'] or offset>before.st_size:
                    raise ValueError('daemon_input_changed')
                f.seek(offset); raw=f.read(min(INPUT_CHUNK,before.st_size-offset))
                if file_identity(os.fstat(f.fileno()))!=file_identity(before) or file_identity(os.stat(entry['source'],follow_symlinks=False))!=file_identity(before):
                    raise ValueError('daemon_input_changed')
            result=dict(data=base64.b64encode(raw).decode())
        else: raise ValueError('daemon_action_not_allowed')
        if action!='input': self.check()
        return result

    def exchange(self,identity,phase,nonce,config,b,source):
        diagnostics=HelperDiagnostics(phase)
        try:
            return self._exchange_attached(identity,phase,nonce,config,b,source,diagnostics)
        except (Exception,KeyboardInterrupt) as exc:
            detail=diagnostics.refusal()
            if detail is not None:
                if type(exc) is ValueError and str(exc) in _HOST_REFUSAL_CODES:
                    raise ValueError('daemon_helper_refusal:'+detail+':host='+str(exc)) from exc
                exc.add_note('daemon_helper_diagnostic:'+detail)
            raise

    def _exchange_attached(self,identity,phase,nonce,config,b,source,diagnostics):
        from service import installed_qwen_daemon as qd
        qwen_import=getattr(self,'family',None) is qd.FAMILY and phase=='install'
        expires=time.time()+max(0,self.end-time.monotonic()) if qwen_import else None
        own=self.container(identity); dv.inspect_container(own,'helper',config,b,phase=phase,nonce=nonce,family=getattr(self,'family',None))
        if getattr(self,'resident_cgroupfs',False): self.check()
        p=subprocess.Popen(self.argv(['container','start','--attach','--interactive',identity]),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=self.env,bufsize=0,close_fds=True)
        try:
            with Owner(p,'daemon_helper',self.end,self.retained) as owner:
                diagnostics.start(p.stderr)
                wire=Wire(p.stdout,p.stdin,nonce,phase,deadline=self.end,owner=owner)
                while True:
                    own=self.container(identity)
                    dv.inspect_container(own,'helper',config,b,phase=phase,nonce=nonce,family=getattr(self,'family',None))
                    if own['State']['Running']: break
                    if p.poll() is not None or time.monotonic()>=self.end: raise ValueError('daemon_helper_start_unknown')
                    time.sleep(.02)
                manifest=[{k:v for k,v in e.items() if k!='source'} for e in source]
                if qwen_import: wire.send('import-deadline',dict(expires=expires))
                wire.send('begin',dict(configuration=config,binding=b,helper=own,manifest=manifest))
                while True:
                    action,payload=wire.receive()
                    if action=='complete':
                        dv.closed(payload,'result'); result=payload['result']; break
                    response=self.action(action,payload,config,b,source)
                    wire.send(action,response)
                p.stdin.close()
                wire.finish()
                owner.deadline=min(self.end,time.monotonic()+10)
        finally:
            diagnostics.stop()
        final=self.container(identity)
        dv.inspect_container(final,'helper',config,b,phase=phase,nonce=nonce,family=getattr(self,'family',None))
        if final['State']['Running'] or final['State']['ExitCode']!=0: raise ValueError('daemon_helper_exit_unknown')
        if action!='input': self.check()
        return result


class Wire:
    def __init__(self,reader,writer,nonce,phase,*,deadline=None,owner=None):
        self.reader,self.writer,self.nonce,self.phase=reader,writer,nonce,phase
        self.owner=owner
        self.step=0; self.deadline=deadline or time.monotonic()+110; self.pending=b''
        os.set_blocking(self.writer.fileno(),False)
    @contextmanager
    def selector(self):
        sel=selectors.DefaultSelector()
        if self.owner is not None: self.owner.selectors.append(sel)
        try:
            yield sel
        except BaseException as exc:
            if self.owner is not None: self.owner.fail(exc)
            raise
        finally:
            sel.close()
            if self.owner is not None: self.owner.selectors.remove(sel)

    def send(self,action,payload):
        self.step+=1
        raw=encoded(dict(version=1,nonce=self.nonce,phase=self.phase,step=self.step,action=action,payload=payload))+b'\n'
        if len(raw)>LIMIT: raise ValueError('daemon_frame_budget')
        offset=0
        with self.selector() as sel:
            sel.register(self.writer,selectors.EVENT_WRITE)
            while offset<len(raw):
                remaining=self.deadline-time.monotonic()
                if remaining<=0 or not sel.select(remaining): raise ValueError('daemon_phase_write_timeout')
                try: offset+=os.write(self.writer.fileno(),raw[offset:])
                except BlockingIOError: continue
    def finish(self):
        """A terminal frame is successful only with bounded, exact stdout EOF."""
        if self.pending: raise ValueError('daemon_trailing_protocol')
        with self.selector() as sel:
            sel.register(self.reader,selectors.EVENT_READ)
            remaining=min(10,self.deadline-time.monotonic())
            if remaining<=0 or not sel.select(remaining):
                raise ValueError('daemon_terminal_eof_timeout')
            if os.read(self.reader.fileno(),1):
                raise ValueError('daemon_trailing_protocol')

    def receive(self):
        raw=self.pending
        with self.selector() as sel:
            sel.register(self.reader,selectors.EVENT_READ)
            while b'\n' not in raw:
                if len(raw)>=LIMIT: raise ValueError('daemon_frame_budget')
                remaining=self.deadline-time.monotonic()
                if remaining<=0 or not sel.select(remaining): raise ValueError('daemon_phase_timeout')
                data=os.read(self.reader.fileno(),min(65536,LIMIT-len(raw)))
                if not data: raise ValueError('daemon_phase_eof_retained')
                raw+=data
        line,_,self.pending=raw.partition(b'\n'); raw=line
        if len(raw)>=LIMIT: raise ValueError('daemon_frame_budget')
        value=parse(raw); dv.closed(value,'version nonce phase step action payload')
        self.step+=1
        if (type(value['version']) is not int or value['version']!=1 or value['nonce']!=self.nonce or value['phase']!=self.phase or
            type(value['step']) is not int or value['step']!=self.step): raise ValueError('daemon_phase_replay')
        return value['action'],value['payload']


def manifest(entries, *, family=None, service=False):
    from service import installed_pocket_daemon as pd, installed_qwen_daemon as qd
    pd.require(family)
    qwen = family is qd.FAMILY
    if type(entries) is not list or not 2<=len(entries)<=4096: raise ValueError('daemon_input_manifest')
    seen=set(); total=0; out=[]
    for e in entries:
        dv.closed(e,'role path source bytes sha256' if e.get('role')!='credential' else 'role path source bytes')
        role=e['role']; path=e['path']; size=e['bytes']
        if (role not in ('models','config','credential') or not isinstance(path,str) or len(path)>256 or
            path.startswith('/') or any(p in ('','.','..') for p in path.split('/')) or
            (role=='config' and path!='install.json') or (role=='credential' and path!='control.key') or
            (role,path) in seen or type(size) is not int or not 1<=size<=(2*1024**3 if qwen and role=='models' else 512*1024**2) or
            (role!='credential' and not dv.HEX.fullmatch(e['sha256']))): raise ValueError('daemon_input_manifest')
        total+=size; seen.add((role,path)); out.append(e)
    if total>(8*1024**3 if qwen and service else 4*1024**3 if qwen else TOTAL) or sum(e['role']=='config' for e in out)!=1 or sum(e['role']=='credential' for e in out)!=1:
        raise ValueError('daemon_input_budget')
    return out


def main(argv=None, *, family=None):
    # Family comes only from the distinct private entrypoint, never a request.
    from service import install_policy as ip, installed_pocket_daemon as pd, installed_qwen_daemon as qd
    pd.require(family)
    export = pd.is_export(family)
    p=argparse.ArgumentParser(description=('Explicit fresh Pocket CPU Docker volume installation; actual qualification remains separate.' if family is pd.FAMILY else 'Explicit local Docker volume installation; no acquisition or repair.'))
    p.add_argument('--shared-models',help='Fresh Qwen selection JSON: descriptor, catalog, models, publication_sha256'); p.add_argument('command',choices=('install','up','status','stop')); p.add_argument('--context'); p.add_argument('--request'); p.add_argument('--handle',required=True)
    a=p.parse_args(argv)
    try:
        shared=None; store_client=None
        if a.shared_models and (a.command!='install' or family is not qd.FAMILY): raise ValueError('shared_fresh_qwen_only')
        if a.shared_models:
            from service import shared_models as sm
            shared=parse(read(a.shared_models)); dv.closed(shared,'descriptor catalog models publication_sha256')
            key=sm.descriptor(shared['descriptor'])[0]; sm.volume(shared['catalog']); sm.token(shared['publication_sha256'])
            sm.volume(shared['models'],key,shared['models'].get('Labels',{}).get('speech-box.attempt'))
        if a.command=='install':
            if not a.context or not a.request: raise ValueError('daemon_install_inputs_required')
            request=parse(read(a.request))
            service=family is qd.FAMILY and ip.qwen_service(request['configuration'])
            dv.closed(request,'configuration helper_image inputs cpu tmpfs control_port'+(' custody' if service else ''))
            config=request['configuration']
            inputs=manifest(request['inputs'],family=family,service=service)
            if shared and any(e['role']=='models' for e in inputs): raise ValueError('shared_config_credential_only')
            if family is not None:
                config=ip.family_configuration(config,family)
            if export or service:
                # Corroborate the actual supplied closed credential document
                # before the first Docker call; helper repeats after import.
                from service.auth import decode_credentials
                credentials=decode_credentials(read(next(e['source'] for e in inputs if e['role']=='credential'),48000),config['contexts'])
                admitted={c.principal for c in credentials._credentials if {'read','production-write'} <= c.scopes}
                required=set(config['audio_export']['principals']) if export else set(config['contexts'])
                if not required <= admitted:
                    raise ValueError('daemon_export_credential_principals' if export else 'daemon_service_credential_principals')
            # Stdlib admission before any daemon mutation. Full existing service
            # configuration/model validation runs again in the pinned helper.
            if type(config) is not dict or len(encoded(config))>48000: raise ValueError('daemon_configuration_budget')
            dv.closed(config['images'],'control worker')
            if any(not isinstance(i,str) or not dv.IMAGE.fullmatch(i) for i in [request['helper_image'],*config['images'].values()]):
                raise ValueError('daemon_pinned_images_required')
            if any(type(config['budget'][k]) is not int or config['budget'][k]<1 for k in ('ram_bytes','pids')):
                raise ValueError('daemon_resource_budget')
            if any(e['bytes']>48000 for e in inputs if e['role'] in ('config','credential')):
                raise ValueError('daemon_private_input_budget')
            d=Docker(a.context,family=family)
            if service: d.require_resident_cgroupfs()
            installation=secrets.token_hex(16)
            # Exclusive client intent before ANY daemon mutation. Partial volumes
            # and exact IDs remain in append-only sibling receipts, never rollback.
            b=dict(policy=pd.launch(family) if family is not None else dv.POLICY,installation=installation,daemon=d.frozen,volumes={},helper_image=request['helper_image'],
                configuration_digest=dv.digest(config),cpu=request['cpu'],tmpfs=request['tmpfs'],control_port=request['control_port'])
            if service:b['custody']=request['custody']
            candidate=dict(b,volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
                Mountpoint='/pending/'+r,CreatedAt='pending',Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES})
            if shared:
                # Detached metadata permits only pre-mutation syntax/resource
                # admission. The protected reservation below authenticates it.
                b['model_storage']=dict(kind=sm.STORAGE,manifest_sha256=key,bundle_id=key,
                    catalog=shared['catalog'],publication_sha256=shared['publication_sha256'],reference_id=installation)
                b['volumes']['models']=shared['models']
                candidate['model_storage']=b['model_storage']; candidate['volumes']['models']=shared['models']
                qd.binding_configuration(candidate,config)
            else: dv.binding(candidate,family=family)
            for image in [b['helper_image'],*config['images'].values()]:
                if d.call(['image','inspect',image,'--format','{{.Id}}'])!=image: raise ValueError('daemon_loaded_image_required')
            exclusive(a.handle,dict(kind='creation-intent',binding=b,configuration=config))
            if shared:
                from service.shared_model_client import Client
                directory=Path(a.handle+'.shared'); directory.mkdir(mode=0o700)
                d.end=time.monotonic()+qd.IMPORT_SECONDS
                store_client=Client(d,b['helper_image'],directory,d.end,time.time()+qd.IMPORT_SECONDS,shared['descriptor'])
                store_client.catalog_volume(False)
                if store_client.catalog!=shared['catalog']: raise ValueError('shared_selected_catalog')
                published=store_client.coordinate('reserve',dict(installation=installation,configuration=config,
                    catalog=shared['catalog'],publication_sha256=shared['publication_sha256']))
                if published['candidate']!=shared['models']: raise ValueError('shared_selected_candidate')
            for role in (r for r in dv.ROLES if not shared or r!='models'):
                name='pab-'+installation+'-'+role
                existing=d.call(['volume','ls','--format','{{.Name}}']).splitlines()
                if name in existing: raise ValueError('daemon_existing_volume_refused')
                exclusive(a.handle+'.intent-'+role,dict(name=name))
                # Read/list and receipt writes can intervene after admission.
                if service: d.check()
                result=d.call(['volume','create','--driver','local','--label','speech-box.installation='+installation,
                    '--label','speech-box.role='+role,name])
                if result!=name: raise ValueError('daemon_volume_create_changed')
                v=d.json(['volume','inspect',name])[0]
                v={k:v[k] for k in ('Name','Driver','Options','Scope','Mountpoint','CreatedAt','Labels')}
                dv.volume(v,role,installation); b['volumes'][role]=v
                exclusive(a.handle+'.created-'+role,v); d.check()
            dv.binding(b,family=family)
            if shared: store_client.coordinate('bind',dict(binding=b,configuration=config))
            exclusive(a.handle+'.binding',dict(binding=b,configuration=config))
            phases=('install','init')
        else:
            if a.request or a.context: raise ValueError('daemon_handle_selection_only')
            retained=parse(read(a.handle+'.binding')); dv.closed(retained,'binding configuration')
            b=dv.binding(retained['binding'],family=family); config=retained['configuration']; inputs=[]
            d=Docker(endpoint=b['daemon']['endpoint'],family=family)
            if d.frozen!=b['daemon']: raise ValueError('daemon_handle_context_changed')
            if 'custody' in b: d.require_resident_cgroupfs()
            if 'model_storage' in b:
                from service import shared_models as sm
                from service.shared_model_client import Client
                # Lifecycle selection comes only from the retained binding.
                # Helpers read the descriptor from protected publication attempts.
                directory=Path(a.handle+'.shared-'+secrets.token_hex(16)); directory.mkdir(mode=0o700)
                store_client=Client(d,b['helper_image'],directory,time.monotonic()+120,time.time()+120)
                store_client.catalog_volume(False)
                if store_client.catalog!=b['model_storage']['catalog']: raise ValueError('shared_selected_catalog')
            phases=(a.command,)
        for phase in phases:
            if family is qd.FAMILY:
                # Anchor before helper creation, never on file/chunk progress.
                if not (store_client and phase=='install'): d.end=time.monotonic()+(qd.IMPORT_SECONDS if phase=='install' else 120)
            nonce=secrets.token_hex(16)
            intent=a.handle+'.helper-'+nonce
            exclusive(intent,dict(phase=phase,nonce=nonce,binding=dv.digest(b)))
            if store_client:
                store_client.end=d.end; store_client.expires=time.time()+max(0,d.end-time.monotonic())
                store_client.coordinate('attachment-intent',dict(binding=b,configuration=config,role='helper',phase=phase,nonce=nonce))
            if 'custody' in b: d.check()
            identity=d.call(dv.command('helper',config,b,phase=phase,nonce=nonce,family=family))
            if not dv.HEX.fullmatch(identity): raise ValueError('daemon_helper_id')
            exclusive(intent+'.id',dict(id=identity))
            if store_client: store_client.coordinate('attachment-created',dict(binding=b,nonce=nonce,inspection=d.container(identity)))
            result=d.exchange(identity,phase,nonce,config,b,inputs if phase=='install' else [])
            exclusive(intent+'.terminal',dict(id=identity,result=result))
        print(encoded(result).decode())
    except (Exception,KeyboardInterrupt) as exc:
        if isinstance(exc,ValueError) and str(exc).startswith('qwen_service_cgroupfs_required:'):
            raise SystemExit(str(exc)) from None
        if isinstance(exc,ValueError) and str(exc).startswith('daemon_helper_refusal:'):
            raise SystemExit('speech-box docker refused; retain handle, volumes and containers for review; '+str(exc)) from None
        raise SystemExit('speech-box docker refused; retain handle, volumes and containers for review') from None


def export_main(argv=None):
    """Explicit private fresh-family installer; no public/default selection."""
    from service.installed_export import FAMILY
    return main(argv,family=FAMILY)


def pocket_main(argv=None):
    from service.installed_pocket_daemon import FAMILY
    return main(argv, family=FAMILY)


def qwen_main(argv=None):
    from service.installed_qwen_daemon import FAMILY
    return main(argv, family=FAMILY)
