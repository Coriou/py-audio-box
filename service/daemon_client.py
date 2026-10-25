"""Stdlib-only ordinary Mac/Linux client. Never imports service dependencies.

All daemon traffic uses the explicitly resolved Unix endpoint, with ambient
Docker overrides removed. No shell, acquisition, rollback, rm or KILL path.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import selectors
import stat
import subprocess
import sys
import tempfile
import time

from service import daemon_volumes as dv

LIMIT=1024*1024
TOTAL=2*1024**3


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


class Docker:
    def __init__(self, context=None, *, endpoint=None):
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
        # File-backed bounded collection avoids unbounded communicate output.
        with tempfile.TemporaryFile() as output:
            p=subprocess.Popen(self.argv(args),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.DEVNULL,env=self.env,close_fds=True)
            self.retained.append(p)
            try: p.wait(timeout=max(.01,min(15,self.end-time.monotonic())))
            except subprocess.TimeoutExpired: raise ValueError('daemon_client_timeout_retained') from None
            if p.returncode!=0: raise ValueError('daemon_cli_refused_retained')
            if output.tell()>LIMIT: raise ValueError('daemon_output_budget')
            output.seek(0); return output.read(LIMIT+1).decode().strip()

    def json(self,args): return parse(self.call(args))

    def identity(self):
        info=self.json(['info','--format','{{json .}}'])
        return dv.daemon(dict(endpoint=self.endpoint,id=info['ID'],os=info['OSType'],architecture=info['Architecture'],
            security=info['SecurityOptions'],cgroup=info['CgroupVersion']))

    def check(self):
        if self.identity()!=self.frozen: raise ValueError('daemon_endpoint_identity_changed')

    def container(self,identity):
        if not isinstance(identity,str) or not dv.HEX.fullmatch(identity): raise ValueError('daemon_exact_container_id')
        values=self.json(['container','inspect',identity])
        if len(values)!=1 or values[0]['Id']!=identity: raise ValueError('daemon_container_changed')
        return values[0]

    def snapshot(self,b):
        self.check()
        volumes={r:self.json(['volume','inspect',v['Name']])[0] for r,v in b['volumes'].items()}
        # Reduce only Docker volume records; no client host stat of Mountpoints.
        volumes={r:{k:v[k] for k in ('Name','Driver','Options','Scope','Mountpoint','CreatedAt','Labels')} for r,v in volumes.items()}
        if volumes!=b['volumes']: raise ValueError('daemon_volume_changed')
        ids=self.call(['container','ls','--all','--no-trunc','--format','{{.ID}}']).splitlines()
        if len(ids)>512: raise ValueError('daemon_consumer_budget')
        paths=[v['Mountpoint'].rstrip('/') for v in volumes.values()]; names={v['Name'] for v in volumes.values()}
        consumers=[]
        for identity in ids:
            c=self.container(identity)
            if any(m.get('Name') in names or (bool(m.get('Source')) and any(m.get('Source','').rstrip('/')==p or
                m.get('Source','').startswith(p+'/') or p.startswith(m.get('Source','').rstrip('/')+'/')
                for p in paths)) for m in c['Mounts']): consumers.append(c)
        self.check()
        return dict(daemon=self.frozen,volumes=volumes,consumers=consumers)

    def action(self,action,payload,config,b,source):
        # Input chunks do no daemon action. The held phase's exact snapshots
        # bracket the stream; avoid a Docker-info process for every 48 KiB.
        if action!='input': self.check()
        if action=='snapshot':
            dv.closed(payload,''); result=self.snapshot(b)
        elif action=='create':
            dv.closed(payload,'role')
            if payload['role'] not in ('control','worker'): raise ValueError('daemon_action_role')
            identity=self.call(dv.command(payload['role'],config,b))
            result=self.container(identity)
        elif action in ('start','term','inspect','marker','facts'):
            dv.closed(payload,'id role')
            role=payload['role']; identity=payload['id']
            if role not in ('control','worker'): raise ValueError('daemon_action_role')
            dv.inspect_container(self.container(identity),role,config,b)
            if action=='start': self.call(['container','start',identity])
            elif action=='term': self.call(['container','kill','--signal=TERM',identity])
            if action in ('start','term','inspect'): result=self.container(identity)
            else:
                code=('from service.entrypoint import launch_status;launch_status()' if action=='marker' else
                    'from service.daemon_runtime import read_control;read_control()')
                if role!='control': raise ValueError('daemon_control_reader_only')
                result=parse(self.call(['exec','--user','10001:10001',identity,'python','-I','-B','-c',
                    "import sys;sys.path.insert(0,'/opt/speech');"+code]))
        elif action=='notify':
            # Payload is nonsecret. Fixed private notification entrypoint only.
            dv.closed(payload,'id gate')
            identity=payload['id']; dv.inspect_container(self.container(identity),'control',config,b)
            raw=encoded(payload['gate'])
            if len(raw)>65536: raise ValueError('daemon_gate_budget')
            p=subprocess.Popen(self.argv(['exec','--interactive','--user','10001:10001',identity,'python','-I','-B','-c',
                "import sys;sys.path.insert(0,'/opt/speech');from service.entrypoint import publish;publish(sys.stdin.buffer.read(65537))"]),
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=self.env)
            self.retained.append(p)
            try: p.communicate(raw,timeout=10)
            except subprocess.TimeoutExpired: raise ValueError('daemon_notify_timeout_retained') from None
            if p.returncode!=0: raise ValueError('daemon_notify_failed_retained')
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
                f.seek(offset); raw=f.read(min(49152,before.st_size-offset))
                if file_identity(os.fstat(f.fileno()))!=file_identity(before) or file_identity(os.stat(entry['source'],follow_symlinks=False))!=file_identity(before):
                    raise ValueError('daemon_input_changed')
            result=dict(data=base64.b64encode(raw).decode())
        else: raise ValueError('daemon_action_not_allowed')
        if action!='input': self.check()
        return result

    def exchange(self,identity,phase,nonce,config,b,source):
        own=self.container(identity); dv.inspect_container(own,'helper',config,b,phase=phase,nonce=nonce)
        p=subprocess.Popen(self.argv(['container','start','--attach','--interactive',identity]),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=self.env,bufsize=0,close_fds=True)
        self.retained.append(p)
        wire=Wire(p.stdout,p.stdin,nonce,phase,deadline=self.end)
        while True:
            own=self.container(identity)
            dv.inspect_container(own,'helper',config,b,phase=phase,nonce=nonce)
            if own['State']['Running']: break
            if p.poll() is not None or time.monotonic()>=self.end: raise ValueError('daemon_helper_start_unknown')
            time.sleep(.02)
        manifest=[{k:v for k,v in e.items() if k!='source'} for e in source]
        wire.send('begin',dict(configuration=config,binding=b,helper=own,manifest=manifest))
        while True:
            action,payload=wire.receive()
            if action=='complete':
                dv.closed(payload,'result'); result=payload['result']; break
            response=self.action(action,payload,config,b,source)
            wire.send(action,response)
        p.stdin.close()
        try: p.wait(timeout=max(.01,min(10,self.end-time.monotonic())))
        except subprocess.TimeoutExpired: raise ValueError('daemon_helper_exit_unknown') from None
        if p.returncode!=0: raise ValueError('daemon_helper_failed_retained')
        final=self.container(identity)
        dv.inspect_container(final,'helper',config,b,phase=phase,nonce=nonce)
        if final['State']['Running'] or final['State']['ExitCode']!=0: raise ValueError('daemon_helper_exit_unknown')
        if action!='input': self.check()
        return result


class Wire:
    def __init__(self,reader,writer,nonce,phase,*,deadline=None):
        self.reader,self.writer,self.nonce,self.phase=reader,writer,nonce,phase
        self.step=0; self.deadline=deadline or time.monotonic()+110; self.pending=b''
        os.set_blocking(self.writer.fileno(),False)
    def send(self,action,payload):
        self.step+=1
        raw=encoded(dict(version=1,nonce=self.nonce,phase=self.phase,step=self.step,action=action,payload=payload))+b'\n'
        if len(raw)>LIMIT: raise ValueError('daemon_frame_budget')
        offset=0
        with selectors.DefaultSelector() as sel:
            sel.register(self.writer,selectors.EVENT_WRITE)
            while offset<len(raw):
                remaining=self.deadline-time.monotonic()
                if remaining<=0 or not sel.select(remaining): raise ValueError('daemon_phase_write_timeout')
                try: offset+=os.write(self.writer.fileno(),raw[offset:])
                except BlockingIOError: continue
    def receive(self):
        raw=self.pending
        with selectors.DefaultSelector() as sel:
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


def manifest(entries):
    if type(entries) is not list or not 2<=len(entries)<=4096: raise ValueError('daemon_input_manifest')
    seen=set(); total=0; out=[]
    for e in entries:
        dv.closed(e,'role path source bytes sha256' if e.get('role')!='credential' else 'role path source bytes')
        role=e['role']; path=e['path']; size=e['bytes']
        if (role not in ('models','config','credential') or not isinstance(path,str) or len(path)>256 or
            path.startswith('/') or any(p in ('','.','..') for p in path.split('/')) or
            (role=='config' and path!='install.json') or (role=='credential' and path!='control.key') or
            (role,path) in seen or type(size) is not int or not 1<=size<=512*1024**2 or
            (role!='credential' and not dv.HEX.fullmatch(e['sha256']))): raise ValueError('daemon_input_manifest')
        total+=size; seen.add((role,path)); out.append(e)
    if total>TOTAL or sum(e['role']=='config' for e in out)!=1 or sum(e['role']=='credential' for e in out)!=1:
        raise ValueError('daemon_input_budget')
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='Explicit local Docker volume installation; no acquisition or repair.')
    p.add_argument('command',choices=('install','up','status','stop')); p.add_argument('--context'); p.add_argument('--request'); p.add_argument('--handle',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='install':
            if not a.context or not a.request: raise ValueError('daemon_install_inputs_required')
            request=parse(read(a.request)); dv.closed(request,'configuration helper_image inputs cpu tmpfs control_port')
            inputs=manifest(request['inputs']); config=request['configuration']
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
            d=Docker(a.context)
            installation=secrets.token_hex(16)
            # Exclusive client intent before ANY daemon mutation. Partial volumes
            # and exact IDs remain in append-only sibling receipts, never rollback.
            b=dict(policy=dv.POLICY,installation=installation,daemon=d.frozen,volumes={},helper_image=request['helper_image'],
                configuration_digest=dv.digest(config),cpu=request['cpu'],tmpfs=request['tmpfs'],control_port=request['control_port'])
            candidate=dict(b,volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
                Mountpoint='/pending/'+r,CreatedAt='pending',Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES})
            dv.binding(candidate)
            for image in [b['helper_image'],*config['images'].values()]:
                if d.call(['image','inspect',image,'--format','{{.Id}}'])!=image: raise ValueError('daemon_loaded_image_required')
            exclusive(a.handle,dict(kind='creation-intent',binding=b,configuration=config))
            for role in dv.ROLES:
                name='pab-'+installation+'-'+role
                existing=d.call(['volume','ls','--format','{{.Name}}']).splitlines()
                if name in existing: raise ValueError('daemon_existing_volume_refused')
                exclusive(a.handle+'.intent-'+role,dict(name=name))
                result=d.call(['volume','create','--driver','local','--label','speech-box.installation='+installation,
                    '--label','speech-box.role='+role,name])
                if result!=name: raise ValueError('daemon_volume_create_changed')
                v=d.json(['volume','inspect',name])[0]
                v={k:v[k] for k in ('Name','Driver','Options','Scope','Mountpoint','CreatedAt','Labels')}
                dv.volume(v,role,installation); b['volumes'][role]=v
                exclusive(a.handle+'.created-'+role,v); d.check()
            dv.binding(b)
            exclusive(a.handle+'.binding',dict(binding=b,configuration=config))
            phases=('install','init')
        else:
            if a.request or a.context: raise ValueError('daemon_handle_selection_only')
            retained=parse(read(a.handle+'.binding')); dv.closed(retained,'binding configuration')
            b=dv.binding(retained['binding']); config=retained['configuration']; inputs=[]
            d=Docker(endpoint=b['daemon']['endpoint'])
            if d.frozen!=b['daemon']: raise ValueError('daemon_handle_context_changed')
            phases=(a.command,)
        for phase in phases:
            nonce=secrets.token_hex(16)
            intent=a.handle+'.helper-'+nonce
            exclusive(intent,dict(phase=phase,nonce=nonce,binding=dv.digest(b)))
            identity=d.call(dv.command('helper',config,b,phase=phase,nonce=nonce))
            if not dv.HEX.fullmatch(identity): raise ValueError('daemon_helper_id')
            exclusive(intent+'.id',dict(id=identity))
            result=d.exchange(identity,phase,nonce,config,b,inputs if phase=='install' else [])
            exclusive(intent+'.terminal',dict(id=identity,result=result))
        print(encoded(result).decode())
    except (Exception,KeyboardInterrupt):
        raise SystemExit('speech-box docker refused; retain handle, volumes and containers for review') from None
