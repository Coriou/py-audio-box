"""One explicit installed EN journey; fixed product roles, never a generic runner.

No invocation without --run-installed-journey acquires weights. Internal exec
continues the original system monotonic deadline in the copied installed Python.
MCP uses its real stdio JSON-RPC protocol with an explicitly owned Popen handle:
the pinned SDK stdio_client hides that handle and creates another process group.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ci_process_rss import Sampler

TOTAL=600
RESERVE=20
LOG_LIMIT=1024**2
RSS_LIMIT=6*1024**3
MAX_WAV=128*1024
_LIVE={}
_STOP=False


class Failure(RuntimeError):pass


class CleanupFailure(Failure):
    def __init__(self,code,relative='',*,cause=None,progress=None):
        super().__init__(code)
        self.detail=dict(code=code,relative=str(relative)[:256],type=type(cause).__name__ if cause else type(self).__name__,
            errno=getattr(cause,'errno',None),progress=dict(progress or {}))


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()


def installed_modules(installed):
    counts={name:0 for name in ('service','clients','worker','contracts','lib')}
    for name,module in tuple(sys.modules.items()):
        family=name.split('.')[0];path=getattr(module,'__file__',None)
        if family in counts and path is not None:
            if not Path(path).resolve().is_relative_to(installed.resolve()):raise Failure('installed_source_closure')
            counts[family]+=1
    return counts


def directory_identity(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        value=os.fstat(fd)
        if value.st_uid!=os.getuid():raise CleanupFailure('root_owner_mismatch')
        return dict(dev=value.st_dev,ino=value.st_ino,uid=value.st_uid)
    finally:os.close(fd)


def remove_fresh_installation(output,state):
    """Only this journey's captured fresh root; fd-relative, no-follow removal.

    Validate the whole owned namespace before changing any permissions. Only
    directory fds get owner-write added; regular files/links are never chmodded.
    The separate controller log/state directory is retained and never uploaded.
    """
    private=output/'.private';root=private/'installation';progress=dict(checked=0,writable_dirs=0,files_removed=0,dirs_removed=0)
    def fail(code,relative='',cause=None):raise CleanupFailure(code,relative,cause=cause,progress=progress)
    owners=state.get('owners',[])
    if (_LIVE or state.get('live_owned_pids') or state.get('cleanup_errors') or state.get('fresh_installation') is not True or
            state.get('setup_cleanup_confirmed') is not True or state.get('family_cleanup_confirmed') is not True or
            state.get('workload_completed') is not True or not owners or any(r.get('reaped') is not True or
            r.get('sampler_stopped') is not True or r.get('cleanup_unknown') is not False or type(r.get('exit')) is not int for r in owners)):
        fail('cleanup_custody_unknown')
    if state.get('root')!=str(root) or not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (private,output,*output.parents)):
        fail('cleanup_path_escape')
    for key in ('private_identity','installation_identity'):
        value=state.get(key)
        if not isinstance(value,dict) or set(value)!= {'dev','ino','uid'} or any(type(v) is not int for v in value.values()):
            fail('cleanup_identity_missing')
        if value['uid']!=os.getuid():fail('cleanup_owner_mismatch')
    identities={};children={};parent=None;rootfd=None
    def identity(info):return dict(dev=info.st_dev,ino=info.st_ino,uid=info.st_uid)
    def check(info,relative):
        if time.monotonic()>=state['deadline']:fail('cleanup_deadline',relative)
        if info.st_uid!=os.getuid():fail('cleanup_owner_mismatch',relative)
        if info.st_dev!=state['installation_identity']['dev']:fail('cleanup_cross_device',relative)
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):fail('cleanup_node_type',relative)
        if stat.S_ISREG(info.st_mode) and info.st_nlink!=1:fail('cleanup_shared_file',relative)
    def inspect(fd,relative):
        if relative.count('/')>128:fail('cleanup_depth_limit',relative)
        info=os.fstat(fd);check(info,relative);identities[relative]=(identity(info),stat.S_IFMT(info.st_mode))
        names=os.listdir(fd);children[relative]=names
        for name in names:
            rel=relative+'/'+name if relative else name;info=os.stat(name,dir_fd=fd,follow_symlinks=False);check(info,rel)
            progress['checked']+=1
            if progress['checked']>50000:fail('cleanup_entry_limit',rel)
            identities[rel]=(identity(info),stat.S_IFMT(info.st_mode))
            if stat.S_ISDIR(info.st_mode):
                child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                try:
                    if identity(os.fstat(child))!=identity(info):fail('cleanup_identity_changed',rel)
                    inspect(child,rel)
                finally:os.close(child)
    def same(info,relative):
        check(info,relative)
        if (identity(info),stat.S_IFMT(info.st_mode))!=identities[relative]:fail('cleanup_identity_changed',relative)
    def remove(fd,relative):
        info=os.fstat(fd);same(info,relative)
        if set(os.listdir(fd))!=set(children[relative]):fail('cleanup_namespace_changed',relative)
        if not info.st_mode&stat.S_IWUSR:
            os.fchmod(fd,stat.S_IMODE(info.st_mode)|stat.S_IWUSR);progress['writable_dirs']+=1
        if set(os.listdir(fd))!=set(children[relative]):fail('cleanup_namespace_changed',relative)
        for name in children[relative]:
            rel=relative+'/'+name if relative else name;info=os.stat(name,dir_fd=fd,follow_symlinks=False);same(info,rel)
            if stat.S_ISDIR(info.st_mode):
                child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                try:remove(child,rel)
                finally:os.close(child)
                same(os.stat(name,dir_fd=fd,follow_symlinks=False),rel);os.rmdir(name,dir_fd=fd);progress['dirs_removed']+=1
            else:os.unlink(name,dir_fd=fd);progress['files_removed']+=1
    try:
        parent=os.open(private,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        if identity(os.fstat(parent))!=state.get('private_identity'):fail('cleanup_parent_identity_changed')
        rootfd=os.open('installation',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        if identity(os.fstat(rootfd))!=state.get('installation_identity'):fail('cleanup_root_identity_changed')
        check(os.fstat(rootfd),'');inspect(rootfd,'')
        if identity(os.stat('installation',dir_fd=parent,follow_symlinks=False))!=state['installation_identity']:fail('cleanup_root_identity_changed')
        remove(rootfd,'')
        if identity(os.stat('installation',dir_fd=parent,follow_symlinks=False))!=state['installation_identity']:fail('cleanup_root_identity_changed')
        os.rmdir('installation',dir_fd=parent);progress['dirs_removed']+=1
        return dict(progress,complete=True)
    except CleanupFailure:raise
    except OSError as exc:fail('cleanup_filesystem_failure',cause=exc)
    finally:
        if rootfd is not None:os.close(rootfd)
        if parent is not None:os.close(parent)


def save(path,value):
    temporary=path.with_name(path.name+'.new')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def sanitized(raw):
    text=raw.decode('utf-8','replace')
    text=re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----','[PRIVATE KEY REDACTED]',text,flags=re.S)
    text=re.sub(r'(?i)\b[0-9a-f]{96}\b','[INSTALLATION TOKEN REDACTED]',text)
    return '\n'.join('[AUTH LINE REDACTED]' if re.search(r'(?i)authorization|bearer|speech_credential|"token"',line) else line for line in text.splitlines())


def environment(private):
    return dict(PATH=os.defpath,HOME=str(private),TMPDIR=str(private),PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        **{k:os.environ[k] for k in ('GITHUB_SHA','GITHUB_REF','ImageOS','ImageVersion') if k in os.environ})


def budget(deadline,minimum=0):
    remaining=deadline-RESERVE-time.monotonic()
    if _STOP or remaining<=minimum:raise Failure('controller_total_deadline_or_interrupt')
    for child in tuple(_LIVE.values()):child.observe()
    return remaining


class Child:
    """Only setup/service/CLI/MCP roles in this journey; original handle custody."""
    def __init__(self,role,argv,private,deadline,records,*,pipes=False,sampler_factory=Sampler):
        if role not in ('setup','service','cli','mcp'):raise Failure('controller_role')
        budget(deadline);self.role=role;self.deadline=deadline;self.start=time.monotonic()
        self.path=private/(role+'-'+str(len(records))+'.log');self.log=self.path.open('xb')
        self.stderr_path=self.path.with_suffix('.stderr.log');self.stderr=self.stderr_path.open('xb')
        self.handle=subprocess.Popen(argv,stdin=subprocess.PIPE if pipes else subprocess.DEVNULL,
            stdout=subprocess.PIPE if pipes else self.log,stderr=self.stderr,env=environment(private),
            start_new_session=True,close_fds=True,bufsize=0)
        self.record=dict(role=role,pid=self.handle.pid,identity=None,signals=[],exit=None,reaped=False,
            sampler_stopped=False,sampled_peak_rss_bytes=None,samples=0,rss_limit_bytes=RSS_LIMIT,cleanup_unknown=False)
        records.append(self.record);_LIVE[self.handle.pid]=self
        self.sampler=sampler_factory(self.handle.pid);self.sampler.start()

    def observe(self):
        code=self.handle.poll()
        state=self.sampler.snapshot();last=state['last'];now=time.monotonic()
        self.record.update(sampled_peak_rss_bytes=state['peak'],samples=state['count'],sampler_error=state['error'],
            last_sample_age=None if last is None else now-last['observed_at'])
        if last is not None:
            if last['pid']!=self.handle.pid or self.record['identity'] not in (None,last['identity']):raise Failure('controller_identity_changed')
            self.record['identity']=last['identity']
        if state['error'] and state['error']['type']!='ProcessGone':raise Failure('controller_sampler_lost')
        if code is not None:
            self.record['exit']=code;return code
        if (self.path.stat().st_size+self.stderr_path.stat().st_size>LOG_LIMIT or (state['peak'] or 0)>RSS_LIMIT):raise Failure('controller_log_or_rss_limit')
        if state['error'] or last is None or now-last['observed_at']>.5:
            # ProcessGone before poll exposes exit is provisional, never death
            # proof or a renewed deadline. Original Popen wait remains authority.
            fresh=now-(last['observed_at'] if last else self.start)<=.5
            gone=state['error'] and state['error']['type']=='ProcessGone'
            if not fresh or state['error'] and not gone:raise Failure('controller_sampler_lost')
        return None

    def send(self,number):
        if self.handle.poll() is not None:return
        state=self.sampler.snapshot();last=state['last']
        if (last is None or last['pid']!=self.handle.pid or last['identity']!=self.record['identity'] or
                time.monotonic()-last['observed_at']>.5 or state['error'] and state['error']['type']!='ProcessGone'):
            raise Failure('controller_signal_identity_unknown')
        item=dict(signal=number.name,accepted=False,error=None);self.record['signals'].append(item)
        try:
            if os.getpgid(self.handle.pid)!=self.handle.pid:raise Failure('controller_process_group_changed')
            os.killpg(self.handle.pid,number);item['accepted']=True
        except ProcessLookupError as exc:item['error']=dict(type=type(exc).__name__,errno=exc.errno)

    def wait(self,seconds,*,work=True):
        cutoff=min(self.deadline-(RESERVE if work else 0),time.monotonic()+seconds)
        while self.observe() is None:
            if work:budget(self.deadline)
            if time.monotonic()>=cutoff:raise Failure('controller_child_deadline')
            time.sleep(.05)
        if work:
            budget(self.deadline)
            if time.monotonic()>=cutoff:raise Failure('controller_child_deadline')
        return self.reap()

    def reap(self):
        code=self.handle.wait(timeout=max(.001,min(1,self.deadline-time.monotonic())))
        self.record.update(exit=code,reaped=True,sampler_stopped=self.sampler.stop())
        self.log.close();self.stderr.close()
        for stream in (self.handle.stdin,self.handle.stdout):
            if stream is not None and not stream.closed:stream.close()
        if not self.record['sampler_stopped']:raise Failure('controller_sampler_cleanup_unknown')
        _LIVE.pop(self.handle.pid,None)
        if self.path.stat().st_size+self.stderr_path.stat().st_size>LOG_LIMIT:raise Failure('controller_log_limit')
        return code

    def close(self,*,force_allowed=False):
        try:
            if self.handle.poll() is None:
                if self.role=='mcp' and self.handle.stdin is not None:self.handle.stdin.close()
                else:self.send(signal.SIGTERM)
                try:self.wait(14 if self.role=='service' else 3,work=False)
                except Failure:
                    if self.handle.poll() is None:
                        allowed=force_allowed() if callable(force_allowed) else force_allowed
                        if not allowed:raise Failure('controller_nested_custody_unknown')
                        self.send(signal.SIGKILL)
                        self.wait(1,work=False)
                    else:self.reap()
            else:self.reap()
        except BaseException:
            self.record['cleanup_unknown']=True;raise


class MCP:
    """Closed real stdio calls; explicit child handle, bounded frames/round trips."""
    def __init__(self,child,deadline):
        self.child=child;self.deadline=deadline;self.seq=0;self.buffer=b''
        os.set_blocking(child.handle.stdin.fileno(),False);os.set_blocking(child.handle.stdout.fileno(),False)
        self.selector=selectors.DefaultSelector();self.selector.register(child.handle.stdout,selectors.EVENT_READ)
        initialized=self.call('initialize',dict(protocolVersion='2025-11-25',capabilities={},clientInfo=dict(name='native-installed-ci',version='1')))
        if initialized.get('protocolVersion')!='2025-11-25':raise Failure('mcp_protocol_version')
        self.write(dict(jsonrpc='2.0',method='notifications/initialized'))

    def write(self,value,*,cutoff=None):
        budget(self.deadline);raw=canonical(value)+b'\n'
        if len(raw)>32768:raise Failure('mcp_request_limit')
        cutoff=min(self.deadline-RESERVE,time.monotonic()+10,cutoff if cutoff is not None else self.deadline)
        with selectors.DefaultSelector() as writer:
            writer.register(self.child.handle.stdin,selectors.EVENT_WRITE)
            while raw:
                budget(self.deadline)
                if time.monotonic()>=cutoff:raise Failure('mcp_write_deadline')
                if writer.select(max(0,min(.1,cutoff-time.monotonic()))):
                    try:raw=raw[os.write(self.child.handle.stdin.fileno(),raw):]
                    except BlockingIOError:pass

    def call(self,method,params,*,cutoff=None):
        cutoff=min(self.deadline-RESERVE,time.monotonic()+10,cutoff if cutoff is not None else self.deadline)
        self.seq+=1;self.write(dict(jsonrpc='2.0',id=self.seq,method=method,params=params),cutoff=cutoff)
        while True:
            self.child.observe();budget(self.deadline)
            if time.monotonic()>=cutoff:raise Failure('mcp_roundtrip_deadline')
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1);value=json.loads(line)
                if value.get('id')!=self.seq:raise Failure('mcp_unexpected_frame')
                if value.get('jsonrpc')!='2.0' or 'error' in value:raise Failure('mcp_rpc_error')
                return value['result']
            remaining=cutoff-time.monotonic()
            if remaining<=0:raise Failure('mcp_roundtrip_deadline')
            if self.selector.select(min(.1,remaining)):
                block=os.read(self.child.handle.stdout.fileno(),65536)
                if not block:raise Failure('mcp_unexpected_eof')
                self.buffer+=block
                if len(self.buffer)>256*1024:raise Failure('mcp_response_limit')

    def tool(self,name,arguments,*,cutoff=None):
        result=self.call('tools/call',dict(name=name,arguments=arguments),cutoff=cutoff)
        if result.get('isError'):raise Failure('mcp_tool_error')
        return result['structuredContent']


def wav_info(raw):
    if not 44<len(raw)<=MAX_WAV:raise Failure('wav_size_limit')
    with wave.open(io.BytesIO(raw),'rb') as wav:
        if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype())!=(1,2,24000,'NONE'):raise Failure('wav_format')
        frames=wav.getnframes();pcm=wav.readframes(frames)
    if frames<=0 or len(pcm)!=frames*2 or not any(pcm):raise Failure('wav_empty_or_silent')
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),frames=frames,duration_seconds=frames/24000,ending_verified=False)


def download(call,asset,deadline,*,mcp=False,known_size=None):
    end=min(deadline-RESERVE,time.monotonic()+20)
    size=known_size if known_size is not None else call(dict(asset_id=asset['id'],start=0,end_exclusive=1),cutoff=end)['total_bytes']
    if not 44<size<=MAX_WAV:raise Failure('download_size_limit')
    parts=[];count=0
    for start in range(0,size,3072 if mcp else 65536):
        budget(deadline)
        if time.monotonic()>=end:raise Failure('download_total_deadline')
        stop=min(size,start+(3072 if mcp else 65536));result=call(dict(asset_id=asset['id'],start=start,end_exclusive=stop),cutoff=end);count+=1
        raw=base64.b64decode(result['content_base64'],validate=True)
        if time.monotonic()>end:raise Failure('download_total_deadline')
        if (result['start'],result['end_exclusive'],result['total_bytes'],result['content_sha256'])!=(start,stop,size,asset['sha256']) or len(raw)!=stop-start:raise Failure('download_metadata')
        parts.append(raw)
        if mcp and count>43:raise Failure('mcp_range_count')
    raw=b''.join(parts)
    if hashlib.sha256(raw).hexdigest()!=asset['sha256']:raise Failure('download_whole_hash')
    return raw


class Journey:
    def __init__(self,output,state):
        self.output,self.state=output,state;self.private=output/'.private';self.root=self.private/'installation'
        self.deadline=state['deadline'];self.records=state['owners'];self.service=None;self.mcp_child=None
        installed=self.root/'source/native-worker';sys.path.insert(0,str(installed))
        from service.native_cpu_setup import client_environment
        from clients.http import HTTP
        if not Path(sys.modules['service.native_cpu_setup'].__file__).is_relative_to(installed):raise Failure('installed_role_source')
        state['installed_source_modules_checked']=installed_modules(installed)
        sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=ROOT.resolve()]
        self.client_environment=client_environment;self.HTTP=HTTP
        self.http_thread=None
        self.command=[str(self.root/'venv/bin/python'),'-I','-B',str(installed/'native-worker.py')]

    def cli(self,*args,seconds=20):
        child=Child('cli',[*self.command,*args],self.private,self.deadline,self.records)
        code=child.wait(min(seconds,budget(self.deadline)))
        if code:raise Failure('product_cli_failed')
        return json.loads(child.path.read_bytes()) if child.path.stat().st_size else None

    def http(self,op,payload=None,*,key=None,phase_cutoff=None):
        budget(self.deadline,10)
        if self.http_thread is not None and self.http_thread.is_alive():raise Failure('prior_http_custody_unknown')
        cutoff=min(self.deadline-RESERVE,time.monotonic()+10,phase_cutoff if phase_cutoff is not None else self.deadline)
        if time.monotonic()>=cutoff:raise Failure('http_roundtrip_deadline')
        result={};done=threading.Event()
        def request():
            try:
                with self.client_environment(self.root):
                    transport=self.HTTP(credential=os.environ['SPEECH_CREDENTIAL'])
                    result['value']=transport.get('/v1/status') if op=='status' else transport.call(op,payload or {},**({'key':key} if key else {}))
            except BaseException as exc:result['error']=exc
            finally:done.set()
        # One non-replacing daemon per sequential HTTP operation. Main custody
        # keeps observing processes/deadline while the existing client blocks.
        self.http_thread=threading.Thread(target=request,daemon=True);self.http_thread.start()
        while not done.wait(.05):
            budget(self.deadline)
            if time.monotonic()>=cutoff:raise Failure('http_roundtrip_deadline')
        self.http_thread.join(0)
        budget(self.deadline)
        if time.monotonic()>=cutoff:raise Failure('http_roundtrip_deadline')
        if 'error' in result:raise result['error']
        return result['value']

    def close_http(self):
        if self.http_thread is not None:
            self.http_thread.join(max(0,min(10,self.deadline-time.monotonic())))
            if self.http_thread.is_alive():raise Failure('http_cleanup_unknown')

    def ready(self):
        cutoff=min(self.deadline-RESERVE,time.monotonic()+30)
        while time.monotonic()<cutoff:
            if self.service.observe() is not None:raise Failure('service_startup_exit')
            try:
                status=self.http('status',phase_cutoff=cutoff)
                if time.monotonic()>=cutoff:raise Failure('service_readiness_deadline')
                if status['activation_ready']:
                    if status['ready_for_speech'] is not False:raise Failure('unexpected_readiness_promotion')
                    return
            except ConnectionError:pass  # bounded startup reads, never a write retry
            time.sleep(.2)
        raise Failure('service_readiness_deadline')

    def start_service(self):
        session=hashlib.sha256((self.root/'qualification.json').read_bytes()).hexdigest()
        if self.state.get('qualification_session_sha256',session)!=session:raise Failure('qualification_session_changed')
        self.state['qualification_session_sha256']=session
        self.service=Child('service',[*self.command,'serve','--root',str(self.root),'--qualification-session'],self.private,self.deadline,self.records)
        self.ready()

    def start_mcp(self):
        self.mcp_child=Child('mcp',[*self.command,'mcp','--root',str(self.root)],self.private,self.deadline,self.records,pipes=True)
        self.mcp=MCP(self.mcp_child,self.deadline)

    def stop_mcp(self):
        if self.mcp_child is not None:
            self.mcp.selector.close();self.mcp_child.close(force_allowed=True);self.mcp_child=None

    def family_stopped(self):
        """Read-only exact CPU journal/Core closure; never discover/kill model PIDs."""
        try:
            if self.http_thread is not None and self.http_thread.is_alive():return False
            from worker.native_cpu_supervisor import CPUJournal
            from service.native_cpu_install import read_installation
            _,config,_,_=read_installation(self.root)
            envelope=json.loads((self.root/'state/worker/native.json').read_bytes());value=envelope['payload']
            if hashlib.sha256(canonical(value)).hexdigest()!=envelope['sha256']:
                self.state['family_cleanup_error']='journal_hash';return False
            checker=CPUJournal.__new__(CPUJournal);checker.deployment=config['deployment_id'];checker.configuration=hashlib.sha256(canonical(config)).hexdigest()
            checker.validate(value)
            if any(r['phase']!='terminal' or not r['receipt']['reaped'] for r in value['records']):
                self.state['family_cleanup_error']='journal_not_terminal';return False
            path=self.root/'state/control/core.sqlite'
            if not path.exists():return not value['records'] and not self.state.get('submitted_jobs')
            with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
                jobs=[json.loads(row[0]) for row in db.execute('SELECT job FROM jobs')]
                runtimes={row[0] for row in db.execute('SELECT runtime FROM dispatches')}
            return (len(jobs)<=4 and all(j['state'] in ('succeeded','failed','cancelled') and all(a['reaped'] for a in j['attempts']) for j in jobs)
                and runtimes=={r['id'] for r in value['records']})
        except Exception as exc:
            self.state['family_cleanup_error']=dict(type=type(exc).__name__,message=sanitized(str(exc).encode())[:300]);return False

    def stop_service(self):
        if self.service is not None:
            self.service.close(force_allowed=self.family_stopped)
            if self.service.record['exit']!=0 or self.service.record['signals'] and any(s['signal']=='SIGKILL' for s in self.service.record['signals']):raise Failure('service_unclean_shutdown')
            if not self.family_stopped():raise Failure('family_cleanup_unknown')
            self.service=None

    def take_job(self):
        count=self.state.get('submitted_jobs',0)
        if count>=4:raise Failure('four_job_limit')
        # Increment before a write: unknown HTTP outcomes never buy another try.
        self.state['submitted_jobs']=count+1

    def submit(self,key,request):
        self.take_job()
        return self.http('speech_generate',request,key=key)

    def wait_job(self,jid):
        cursor='';cutoff=min(self.deadline-RESERVE,time.monotonic()+65)
        while time.monotonic()<cutoff:
            result=self.http('job_wait',dict(job_id=jid,cursor=cursor,wait_seconds=5));cursor=result['next_cursor'];job=result['job']
            if job['state'] in ('succeeded','failed','cancelled'):return job
        raise Failure('public_job_wait_deadline')

    def clip(self,job,name):
        if job['state']!='succeeded':raise Failure('public_job_not_succeeded')
        artifact=self.http('artifact_get',dict(artifact_id=job['artifact_ids'][0]));asset=self.http('asset_get',dict(asset_id=artifact['asset_ids'][0]))
        raw=download(lambda p,*,cutoff:self.http('asset_content',p,phase_cutoff=cutoff),asset,self.deadline)
        result=self.cli('client','--root',str(self.root),'asset','content',asset['id'],'--output',str(self.output/name))
        if result['sha256']!=asset['sha256'] or (self.output/name).read_bytes()!=raw:raise Failure('cli_http_download_disagreement')
        return raw,asset,artifact

    def model_identity(self):
        from service.native_cpu_install import read_installation
        _,config,expected,bundle=read_installation(self.root)
        files={}
        for entry in expected['files']:
            info=(bundle/entry['path']).stat()
            files[entry['path']]=dict(inode=info.st_ino,size=info.st_size,mtime_ns=info.st_mtime_ns,expected_sha256=entry['sha256'])
        return dict(bundle=str(bundle),manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),files=files,
            configuration_sha256=hashlib.sha256(canonical(config)).hexdigest())

    def evidence(self):
        path=self.root/'state/worker/native.json'
        if path.stat().st_size>LOG_LIMIT:raise Failure('journal_report_limit')
        envelope=json.loads(path.read_bytes());value=envelope['payload']
        if hashlib.sha256(canonical(value)).hexdigest()!=envelope['sha256']:raise Failure('journal_report_hash')
        return [dict(runtime_id=r['id'],job_id=r['snapshot'][0]['id'],phase=r['phase'],acked=r['acked'],receipt=r['receipt'],output=r['output']) for r in value['records']]

    def first_public(self):
        voice=self.http('voice_resolve',dict(name='default.narration'))
        version=self.http('voice_version_get',dict(voice_version_id=voice['version_ids'][-1]));self.version=version
        request=dict(voice_version_id=version['id'],text='Hello. The test is complete.',identity=version['identity'],fallback='forbid',variants=1,reuse='never');self.request=request
        started=time.monotonic();submitted=self.submit('installed-en-1',request)
        replay=self.cli('client','--root',str(self.root),'speech','generate','--request',canonical(request).decode(),'--idempotency-key','installed-en-1')
        if replay!=submitted:raise Failure('cli_replay_disagreement')
        self.start_mcp()
        if self.mcp.tool('speech_speech_generate',dict(request,idempotency_key='installed-en-1'))!=submitted:raise Failure('mcp_replay_disagreement')
        job=self.wait_job(submitted['job_id']);completed=time.monotonic()-started
        raw,asset,artifact=self.clip(job,'English.wav')
        if self.mcp.tool('speech_artifact_get',dict(artifact_id=artifact['id']))!=artifact:raise Failure('mcp_artifact_disagreement')
        mcp_raw=download(lambda p,*,cutoff:self.mcp.tool('speech_asset_content',p,cutoff=cutoff),asset,self.deadline,mcp=True,known_size=len(raw))
        if mcp_raw!=raw:raise Failure('mcp_download_disagreement')
        info=wav_info(raw);info.update(public_completion_seconds=completed,end_to_end_rtf=completed/info['duration_seconds'])
        self.state.update(first_job=job,first_asset=asset,first_voice=voice,first_wav=info)

    def cancellations(self):
        submitted=self.submit('installed-en-2',self.request);jid=submitted['job_id']
        cutoff=min(self.deadline-RESERVE,time.monotonic()+5)
        while time.monotonic()<cutoff:
            status=self.http('status',phase_cutoff=cutoff)['status']
            current=self.http('job_get',dict(job_id=jid),phase_cutoff=cutoff)
            if time.monotonic()>=cutoff:raise Failure('cancel_observing_deadline')
            if current['state']=='running' and status['phase']=='observing':break
            if current['state'] in ('succeeded','failed','cancelled'):
                self.state['pre_cancel_terminal_job']=current
                reason={'succeeded':'cancel_completion_won','failed':'job_failed_before_cancel','cancelled':'job_cancelled_before_request'}[current['state']]
                raise Failure(reason)
            time.sleep(.1)
        else:raise Failure('cancel_observing_deadline')
        observing=time.monotonic()
        queued=self.submit('installed-en-3',self.request)
        self.http('job_cancel',dict(job_id=queued['job_id']),key='installed-cancel-queued')
        self.state['running_cancel_request_after_observing_seconds']=time.monotonic()-observing
        self.mcp.tool('speech_job_cancel',dict(job_id=jid,idempotency_key='installed-cancel-running'))
        self.state['running_cancel_response_after_observing_seconds']=time.monotonic()-observing
        a=self.wait_job(jid);b=self.wait_job(queued['job_id'])
        self.state['cancel_jobs']=[a,b]
        if a['state']=='succeeded':raise Failure('cancel_completion_won')
        if b['state']=='succeeded' or b['attempts']:raise Failure('queued_cancel_lost_job_started')
        if a['state']=='failed':raise Failure('running_job_failed_not_cancelled')
        if a['state']!='cancelled' or not a['attempts'] or not all(v['reaped'] for v in a['attempts']) or a['artifact_ids']:raise Failure('running_cancel_unqualified')
        if (b['state'],b['stage'],b['attempts'],b['artifact_ids'])!=('cancelled','never_started',[],[]):raise Failure('queued_cancel_unqualified')
        receipt=next(r['receipt'] for r in self.evidence() if r['job_id']==jid)
        if not receipt['reaped'] or receipt['timely'] or len(receipt['processes'])!=1 or receipt['termination']['reason']!='cancelled':raise Failure('cancel_process_proof_missing')

    def restarted(self):
        if self.model_identity()!=self.state['model_before']:raise Failure('model_identity_changed')
        voice=self.http('voice_resolve',dict(name='default.narration'))
        if voice!=self.state['first_voice']:raise Failure('retained_alias_changed')
        if self.http('job_get',dict(job_id=self.state['first_job']['id']))!=self.state['first_job']:raise Failure('retained_job_changed')
        for job in self.state['cancel_jobs']:
            if self.http('job_get',dict(job_id=job['id']))!=job:raise Failure('retained_cancel_changed')
        raw=download(lambda p,*,cutoff:self.http('asset_content',p,phase_cutoff=cutoff),self.state['first_asset'],self.deadline)
        if hashlib.sha256(raw).hexdigest()!=self.state['first_wav']['sha256']:raise Failure('retained_wav_changed')
        self.take_job()
        submitted=self.cli('client','--root',str(self.root),'speech','generate','--request',canonical(self.request).decode(),'--idempotency-key','installed-en-4')
        raw,_,_=self.clip(self.wait_job(submitted['job_id']),'English-restart.wav')
        self.state['restart_wav']=wav_info(raw)


def product_flow(journey):
    """Fixed role ordering and four jobs; no retries or session renewal."""
    def phase(name):journey.state['phase']=name;print('phase '+name,flush=True)
    phase('doctor')
    journey.state['doctor']=journey.cli('doctor','--root',str(journey.root))
    journey.state['model_before']=journey.model_identity()
    phase('qualification_session')
    journey.cli('qualification-session','--root',str(journey.root))
    phase('service_start');journey.start_service()
    phase('public_en');journey.first_public()
    phase('cancellation');journey.cancellations()
    phase('first_shutdown');journey.stop_mcp();journey.stop_service()
    phase('restart');journey.start_service()
    phase('restart_en');journey.restarted()
    phase('final_shutdown');journey.stop_service()


def setup_clean(root,pid):
    try:
        value=json.loads((root/('bootstrap-cleanup-'+str(pid)+'.json')).read_bytes())
        return value['pid']==pid and value['cleanup_confirmed'] is True
    except (OSError,ValueError,KeyError):return False


def public(value):
    if isinstance(value,dict):
        return {k:public(v) for k,v in value.items() if not any(s in k.lower() for s in ('credential','authorization','private_key','environment','token'))}
    if isinstance(value,list):return [public(v) for v in value]
    return sanitized(value.encode()) if isinstance(value,str) else value


def publish(output,state):
    logs=output/'safe-logs';logs.mkdir(mode=0o700,exist_ok=True)
    private=output/'.private';root=private/'installation'
    sources=[(p,p.name) for p in private.glob('*.log')]
    sources.extend((root/name,'setup-'+name) for name in ('venv.log','pip-minimum.log','dependencies.log','dependency-check.log','finish-setup.log'))
    sources.append((root/'preflight/child.log','setup-preflight-child.log'))
    for path,name in sources:
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=LOG_LIMIT:(logs/name).write_text(sanitized(path.read_bytes()))
    for name,key in (('preflight/report.json','setup_preflight'),('preflight/owner.json','setup_preflight_owner'),('state/worker/owner-failure.json','worker_owner_failure')):
        path=root/name
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=LOG_LIMIT:
            try:state[key]=json.loads(path.read_bytes())
            except (ValueError,OSError):state[key]=dict(error='evidence_unreadable')
    save(output/'report.json',public(state))


def _run(args):
    output=args.output.absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise Failure('output_symlink')
    if os.environ.get('GITHUB_REF') not in ('refs/heads/feat/apple-silicon-native','refs/heads/main') or not re.fullmatch('[a-f0-9]{40}',os.environ.get('GITHUB_SHA','')):
        raise Failure('trusted_ci_ref_required')
    private=output/'.private';root=private/'installation';state_path=private/'controller-state.json'
    journey=None;state=None
    if not args.installed_phase:
        if args.deadline is not None:raise Failure('external_deadline_override')
        output.mkdir(mode=0o700,parents=True,exist_ok=False);private.mkdir(mode=0o700)
        state=dict(version=1,phase='setup',source_commit=os.environ['GITHUB_SHA'],controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runner_image=os.environ.get('ImageOS'),runner_version=os.environ.get('ImageVersion'),started=time.monotonic(),deadline=time.monotonic()+TOTAL,
            python=sys.version,system=platform.system(),architecture=platform.machine(),macos=platform.mac_ver()[0],
            total_seconds=TOTAL,cleanup_reserve_seconds=RESERVE,root=str(root),private_identity=directory_identity(private),
            owners=[],submitted_jobs=0,status='running',ending_verified=False)
    else:
        state=json.loads(state_path.read_bytes())
        if (state['phase']!='setup_complete' or state['deadline']!=args.deadline or state['root']!=str(root) or
                state['source_commit']!=os.environ['GITHUB_SHA'] or state['controller_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):raise Failure('installed_exec_binding')
    try:
        if not args.installed_phase:
            print('phase setup',flush=True)
            child=Child('setup',[sys.executable,'-B',str(ROOT/'scripts/native-kokoro.py'),'setup','--root',str(root),
                '--download-model','--setup-download-seconds','120'],private,state['deadline'],state['owners'])
            if child.wait(300):raise Failure('fresh_setup_failed')
            state['setup_cleanup_confirmed']=setup_clean(root,child.handle.pid)
            if not state['setup_cleanup_confirmed']:raise Failure('setup_nested_cleanup_unknown')
            state.update(installation_identity=directory_identity(root),fresh_installation=True)
            state['phase']='setup_complete';save(state_path,state);publish(output,state)
            python=str(root/'venv/bin/python')
            os.execve(python,[python,'-I','-B','-u',str(Path(__file__).resolve()),'--run-installed-journey','--installed-phase',
                '--output',str(output),'--deadline',str(state['deadline'])],environment(private))
        journey=Journey(output,state);state['phase']='public_journey'
        state['install_preflight_before_download']=json.loads((root/'preflight/report.json').read_bytes())
        product_flow(journey)
        if not journey.family_stopped() or _LIVE:raise Failure('final_family_custody_unknown')
        from service.native_cpu_install import read_installation
        _,config,_,_=read_installation(root)
        state.update(status='pending_cleanup',phase='cleanup',workload_completed=True,dependencies=config['dependencies'],
            source_provenance=config['source_provenance'],device='cpu',precision='float32',provider='CPUExecutionProvider (child asserted)',
            model_after=journey.model_identity(),family_cleanup_confirmed=True)
        # Only validated short clips; no credential/config/key/model links.
        import html
        items=[]
        for name in ('English.wav','English-restart.wav'):
            raw=(output/name).read_bytes();info=wav_info(raw)
            items.append('<p>'+html.escape(name)+' · '+info['sha256']+'</p><audio controls src="data:audio/wav;base64,'+base64.b64encode(raw).decode()+'"></audio>')
        (output/'LISTEN.html').write_text('<!doctype html><meta charset="utf-8"><title>Native Kokoro installed EN evidence</title><p>Hello. The test is complete. Human ending/quality unverified. Preset only; not cloning.</p>'+''.join(items))
    except BaseException as exc:
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)
        location=None if not frames else dict(file=Path(frames[-1].filename).name,line=frames[-1].lineno,function=frames[-1].name)
        state.update(status='failed',failure=dict(type=type(exc).__name__,code=str(exc) if isinstance(exc,Failure) else 'controller_exception',
            message=sanitized(str(exc).encode())[:500],location=location,
            cause=None if exc.__cause__ is None else dict(type=type(exc.__cause__).__name__,message=sanitized(str(exc.__cause__).encode())[:300])))
    finally:
        cleanup=[]
        if journey is not None:
            try:journey.close_http()
            except Exception as exc:cleanup.append(dict(role='http',type=type(exc).__name__,code='http_cleanup_unknown'))
        for child in sorted(tuple(_LIVE.values()),key=lambda c:c.role=='service'):
            try:
                allowed=journey.family_stopped if child.role=='service' and journey else (lambda:setup_clean(root,child.handle.pid)) if child.role=='setup' else child.role in ('cli','mcp')
                child.close(force_allowed=allowed)
            except BaseException as exc:cleanup.append(dict(role=child.role,type=type(exc).__name__,code=str(exc) if isinstance(exc,Failure) else 'cleanup_unknown'))
        if journey is not None:
            state['family_cleanup_confirmed']=journey.family_stopped()
            if not state['family_cleanup_confirmed']:cleanup.append(dict(role='model-family',type='Unknown',code='family_cleanup_unknown'))
            try:state['native_records']=journey.evidence()
            except Exception as exc:state['native_evidence_error']=type(exc).__name__;state['status']='failed'
        state.update(cleanup_errors=cleanup,live_owned_pids=list(_LIVE),elapsed_seconds=time.monotonic()-state['started'],private_tree_removed=False)
        if cleanup or _LIVE or time.monotonic()>state['deadline']:state['status']='failed'
        publish(output,state)
        finish_installation_cleanup(output,state)
        publish(output,state)
    return 0 if state['status']=='succeeded' else 1


def finish_installation_cleanup(output,state):
    state['journey_completed']=False
    if state['status']!='pending_cleanup':return
    try:
        state['installation_cleanup']=remove_fresh_installation(output,state)
        state.update(installation_tree_removed=True,journey_completed=True,status='succeeded',phase='complete')
    except CleanupFailure as exc:
        state.update(cleanup_failure=exc.detail,status='failed',phase='cleanup_failed',installation_tree_removed=False)


def run(args):
    global _STOP
    saved_stop=_STOP;_STOP=False
    def stop(number,frame):
        global _STOP
        _STOP=True
    previous={number:signal.signal(number,stop) for number in (signal.SIGTERM,signal.SIGINT)}
    try:return _run(args)
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
        _STOP=saved_stop


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-installed-journey',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--installed-phase',action='store_true');parser.add_argument('--deadline',type=float)
    args=parser.parse_args(argv)
    if not args.run_installed_journey:parser.error('Explicit --run-installed-journey required; no model acquisition by default.')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
