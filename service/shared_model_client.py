"""Stdlib ordinary-client transport for explicit offline shared-store operations.

Only fixed store roles/actions are admitted. All subprocesses use existing
Owner/Wire custody; no retry, cleanup, downloader or installation shortcut.
"""
import argparse
import base64
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time

from service import shared_models as sm
from service import shared_model_helper as sh
from service.daemon_client import Docker, Owner, Wire, read, parse, encoded, exclusive, file_identity


class Client:
    def __init__(self,docker,image,directory,end,expires,expected=None,sources=None):
        if not sm.dv.IMAGE.fullmatch(image): raise ValueError('shared_immutable_image')
        self.docker,self.image,self.directory,self.end,self.expires=docker,image,Path(directory),end,expires
        self.expected,self.sources=expected,sources or {}
        self.failure_context={}
        self.catalog=None; self.created={}; self.candidates={}; self.writer_claimed=set(); self.sequence=0; self.source_identities={}
        if expected is not None:
            key,_,files,_=sm.descriptor(expected)
            wanted=set(files)-{'bundles/'+key+'/manifest.json'}
            if self.sources and set(self.sources)!=wanted: raise ValueError('shared_explicit_sources')
            for name,path in self.sources.items():
                if type(path) is not str or not os.path.isabs(path): raise ValueError('shared_absolute_source_required')
                with sm.root_fd(Path(path).parent) as parent:
                    fd=os.open(Path(path).name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
                    try:
                        s=os.fstat(fd)
                        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size!=files[name][0]:
                            raise ValueError('shared_source_file')
                        self.source_identities[name]=file_identity(s)
                    finally: os.close(fd)

    def receipt(self,kind,value):
        sm.remaining(self.end)
        self.sequence+=1
        exclusive(self.directory/(str(self.sequence).zfill(6)+'-'+kind+'.json'),value)

    def check(self):
        sm.remaining(self.end); self.docker.end=self.end; self.docker.check()
        if self.catalog is not None and self.inspect_volume(sm.CATALOG)!=self.catalog:
            raise ValueError('shared_catalog_incarnation')

    def inspect_volume(self,name):
        values=self.docker.json(['volume','inspect',name])
        if type(values) is not list or len(values)!=1: raise ValueError('shared_volume_inspection')
        return {k:values[0][k] for k in ('Name','Driver','Options','Scope','Mountpoint','CreatedAt','Labels')}

    def create_volume(self,key=None,attempt=None):
        name=sm.CATALOG if key is None else 'pab-model-sha256-'+sm.token(key)+'-'+sm.token(attempt,sm.ATTEMPT)
        self.receipt('volume-intent',dict(name=name,labels=sm.labels(key,attempt)))
        args=['volume','create','--driver','local']
        for k,v in sorted(sm.labels(key,attempt).items()): args+=['--label',k+'='+v]
        # Caller established absence from a complete bounded inventory. An
        # ambiguous create response is retained, never retried here.
        returned=self.docker.call(args+[name])
        if returned!=name: raise ValueError('shared_volume_create_unknown')
        value=sm.volume(self.inspect_volume(name),key,attempt)
        self.receipt('volume-created',value)
        if key is not None: self.candidates[value['Name']]=value
        return value

    def catalog_volume(self,initialize):
        self.check()
        names=self.docker.call(['volume','ls','--format','{{.Name}}']).splitlines()
        if len(names)>4096 or len(set(names))!=len(names): raise ValueError('shared_volume_inventory_budget')
        if sm.CATALOG in names:
            self.catalog=sm.volume(self.inspect_volume(sm.CATALOG))
        elif initialize:
            # Concurrent volume create is idempotent only for the fixed catalog;
            # daemon-side root flock/schema join owns bootstrap authority.
            self.catalog=self.create_volume()
        else: raise ValueError('shared_explicit_init_required')
        return self.catalog

    def input(self,payload):
        sm.closed(payload,'path offset')
        name,offset=payload['path'],payload['offset']
        if name not in self.sources or type(offset) is not int or offset<0:
            raise ValueError('shared_input_allowlist')
        path=Path(self.sources[name]); identity=self.source_identities[name]
        with sm.root_fd(path.parent) as parent:
            fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
            try:
                before=os.fstat(fd)
                if file_identity(before)!=identity or offset>before.st_size: raise ValueError('shared_source_changed')
                os.lseek(fd,offset,os.SEEK_SET)
                raw=os.read(fd,min(sm.CHUNK,before.st_size-offset))
                if file_identity(os.fstat(fd))!=identity or file_identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))!=identity:
                    raise ValueError('shared_source_changed')
            finally: os.close(fd)
        return dict(data=base64.b64encode(raw).decode())

    def create_helper(self,payload):
        sm.closed(payload,'role nonce image candidate')
        role=payload['role']; nonce=sm.token(payload['nonce'],sm.ATTEMPT)
        if role not in ('writer','verifier') or payload['image']!=self.image: raise ValueError('shared_helper_allowlist')
        key=sm.descriptor(self.expected)[0]
        candidate=payload['candidate']; attempt=candidate['Labels'].get('speech-box.attempt')
        sm.volume(candidate,key,attempt); sm.disjoint(self.catalog,candidate)
        if role=='writer':
            if self.candidates.get(candidate['Name'])!=candidate or candidate['Name'] in self.writer_claimed:
                raise ValueError('shared_writer_fresh_allocation_required')
            self.writer_claimed.add(candidate['Name'])
            files=sm.descriptor(self.expected)[2]
            if set(self.sources)!=set(files)-{'bundles/'+key+'/manifest.json'}: raise ValueError('shared_explicit_writer_sources')
        if self.inspect_volume(candidate['Name'])!=candidate: raise ValueError('shared_candidate_incarnation')
        return self._create(role,nonce,candidate)

    def _create(self,role,nonce,candidate):
        command=sh.command(role,self.image,nonce,self.catalog,candidate)
        self.receipt('helper-intent',dict(role=role,nonce=nonce,command_sha256=sm.digest(command),candidate=candidate))
        identity=self.docker.call(command)
        sm.token(identity)
        # Record exact ID before inspection/start, even when inspection fails.
        self.receipt('helper-created',dict(id=identity,role=role,nonce=nonce))
        c=self.docker.container(identity)
        sh.inspect(c,role,self.image,nonce,self.catalog,candidate)
        if c['State']['Status']!='created' or c['State']['Running']:
            raise ValueError('shared_fresh_helper_required')
        if identity in self.created: raise ValueError('shared_duplicate_helper')
        self.created[identity]=dict(role=role,nonce=nonce,candidate=candidate,started=False)
        return c

    def snapshot(self,candidate):
        key=sm.descriptor(self.expected)[0]
        sm.volume(candidate,key,candidate['Labels'].get('speech-box.attempt'))
        current=self.inspect_volume(candidate['Name'])
        if current!=candidate: raise ValueError('shared_candidate_incarnation')
        ids=self.docker.call(['container','ls','--all','--no-trunc','--format','{{.ID}}']).splitlines()
        if len(ids)>512 or len(ids)!=len(set(ids)): raise ValueError('shared_consumer_budget')
        consumers=[]; root=candidate['Mountpoint'].rstrip('/')
        for identity in ids:
            c=self.docker.container(identity)
            for m in c['Mounts']:
                source=m.get('Source','').rstrip('/')
                if m.get('Name')==candidate['Name'] or sm.source_overlap(source,[root]):
                    consumers.append(c); break
        return dict(daemon=self.docker.frozen,candidate=current,consumers=consumers)

    def action(self,action,payload):
        if action=='input': return self.input(payload)
        self.check()
        if action=='allocate':
            sm.closed(payload,'key attempt')
            if payload['key']!=sm.descriptor(self.expected)[0]: raise ValueError('shared_key_allowlist')
            name='pab-model-sha256-'+sm.token(payload['key'])+'-'+sm.token(payload['attempt'],sm.ATTEMPT)
            names=self.docker.call(['volume','ls','--format','{{.Name}}']).splitlines()
            if len(names)>4096 or name in names: raise ValueError('shared_candidate_not_fresh')
            result=self.create_volume(payload['key'],payload['attempt'])
        elif action=='create-helper': result=self.create_helper(payload)
        elif action=='run-helper':
            sm.closed(payload,'id role nonce image candidate expected')
            own=self.created.get(payload['id'])
            if (own is None or own['started'] or payload['image']!=self.image or payload['expected']!=self.expected or
                    any(payload[k]!=own[k] for k in ('role','nonce','candidate'))):
                raise ValueError('shared_start_allowlist')
            own['started']=True
            parent_context=self.failure_context
            result=self.exchange(payload['id'],own['role'],own['nonce'],'import' if own['role']=='writer' else 'verify',own['candidate'])
            self.failure_context=parent_context
        elif action=='inspect':
            sm.closed(payload,'id role image nonce candidate')
            if payload['role'] not in ('writer','verifier') or payload['image']!=self.image:
                raise ValueError('shared_inspect_allowlist')
            key=sm.descriptor(self.expected)[0]
            sm.volume(payload['candidate'],key,payload['candidate']['Labels'].get('speech-box.attempt'))
            result=self.docker.container(sm.token(payload['id']))
            sh.inspect(result,payload['role'],payload['image'],payload['nonce'],self.catalog,payload['candidate'])
        elif action=='snapshot':
            sm.closed(payload,'candidate'); result=self.snapshot(payload['candidate'])
        else: raise ValueError('shared_rpc_not_allowed')
        self.check(); return result

    def exchange(self,identity,role,nonce,operation,candidate=None,attempt=None):
        context=dict(identity=identity,role=role,nonce=nonce,operation=operation,stage='start')
        self.failure_context=context
        self.check()
        sh.inspect(self.docker.container(identity),role,self.image,nonce,self.catalog,candidate)
        self.receipt('start-intent',dict(id=identity,role=role,nonce=nonce))
        p=subprocess.Popen(self.docker.argv(['container','start','--attach','--interactive',identity]),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=self.docker.env,bufsize=0,close_fds=True)
        with Owner(p,'shared_helper',self.end,self.docker.retained) as owner:
            wire=Wire(p.stdout,p.stdin,nonce,role,deadline=self.end,owner=owner)
            request=dict(expires=self.expires,operation=operation,daemon=self.docker.frozen,catalog=self.catalog,
                candidate=candidate,expected=self.expected,attempt=attempt if candidate is None else candidate['Labels']['speech-box.attempt'],image=self.image)
            if operation in ('reserve','bind','attachment-intent','attachment-created'): request['selection']=self.selection
            wire.send('begin',request)
            context['stage']='begin-sent'
            while True:
                action,payload=wire.receive()
                context['stage']='response'
                if action=='complete':
                    sm.closed(payload,'result'); result=payload['result']; break
                if role=='writer':
                    if action!='input': raise ValueError('shared_writer_rpc')
                elif role=='verifier': raise ValueError('shared_verifier_rpc')
                elif action=='input': raise ValueError('shared_controller_payload_forbidden')
                reply=self.action(action,payload)
                self.failure_context=context
                wire.send(action,reply)
                context['stage']='reply-sent'
            p.stdin.close(); wire.finish()
            context['stage']='eof'
        c=self.docker.container(identity)
        sh.inspect(c,role,self.image,nonce,self.catalog,candidate,terminal=True)
        self.check()
        value=dict(inspection=c,result=result)
        context['stage']='terminal'
        self.receipt('terminal',dict(id=identity,role=role,inspection_sha256=sm.digest(c)))
        return value

    def coordinate(self,operation,selection):
        if operation not in ('reserve','bind','attachment-intent','attachment-created'):
            raise ValueError('shared_installation_operation')
        self.selection=selection
        return self.run(operation)

    def run(self,operation,attempt=None):
        self.failure_context={}
        if self.docker.call(['image','inspect',self.image,'--format','{{.Id}}'])!=self.image:
            raise ValueError('shared_local_image_required')
        self.catalog_volume(operation=='init')
        nonce=secrets.token_hex(16)
        own=self._create('controller',nonce,None)
        result=self.exchange(own['Id'],'controller',nonce,operation,attempt=attempt)['result']
        self.receipt('complete',result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline immutable daemon model store; retained failures require explicit review.')
    parser.add_argument('command',choices=('init','import','verify','recover-import'))
    parser.add_argument('--context',required=True)
    parser.add_argument('--image',required=True,help='Existing immutable sha256 image ID')
    parser.add_argument('--request',help='Reviewed JSON {descriptor, sources: {manifest-relative path: absolute local file}}; sources may be empty for verify/recovery/reuse')
    parser.add_argument('--attempt',help='Exact retained 32-hex attempt for recover-import')
    parser.add_argument('--output',required=True,help='Fresh private intent/receipt directory')
    args=parser.parse_args(argv)
    client=None; output=None; output_owned=False
    try:
        end=time.monotonic()+600; expires=time.time()+600
        if (args.command=='recover-import') != (args.attempt is not None): raise ValueError('shared_exact_recovery_attempt')
        if args.attempt: sm.token(args.attempt,sm.ATTEMPT)
        if args.command=='init':
            if args.request: raise ValueError('shared_init_request')
            expected=sources=None
        else:
            if not args.request: raise ValueError('shared_explicit_request')
            request=parse(read(args.request)); sm.closed(request,'descriptor sources')
            expected=request['descriptor']; key,_,files,_=sm.descriptor(expected)
            if type(request['sources']) is not dict: raise ValueError('shared_sources')
            sources={'bundles/'+key+'/'+name:path for name,path in request['sources'].items()}
        output=Path(args.output)
        output.mkdir(mode=0o700,parents=False,exist_ok=False)
        output_owned=True
        os.chmod(output,0o700)
        exclusive(output/'intent.json',dict(schema=sm.SCHEMA,operation=args.command,context=args.context,image=args.image,
                    key=None if expected is None else key,attempt=args.attempt,expires=expires))
        docker=Docker(args.context); docker.end=end
        client=Client(docker,args.image,output,end,expires,expected,sources)
        print(encoded(client.run(args.command,args.attempt)).decode())
    except (Exception,KeyboardInterrupt) as exc:
        context=client.failure_context if client is not None else {}
        sh.diagnostic(exc,**context)
        if output_owned:
            try: exclusive(output/'failure.json',sh.failure_record(exc,**context))
            except BaseException: pass
        raise SystemExit('shared models refused; retained state requires review') from None
