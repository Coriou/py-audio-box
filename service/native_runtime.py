"""Bounded pinned native client and control-owned detached media import.

No Core/SQLite object crosses into lifecycle I/O. Enrollment/activation is Lot B.
The private spool retains evidence; uncertain files are never deleted or retried
as a new allocation. Native receipts never supply a local filesystem UID.
"""
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import stat
import threading
import time

from service.worker_transport import TransportObservation
from worker import native_protocol as p
from worker.lifecycle import Observation, State
from worker.transport import Refused, canonical

QUOTA = 64 * 1024**2


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_nlink,
            info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _directory(path, mode):
    path=Path(path).absolute()
    for parent in (*path.parents,path):
        if parent.is_symlink(): raise Refused('native_control_storage')
    info=path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=mode:
        raise Refused('native_control_storage')
    return path


def _read(path, cap):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_nlink!=1 or before.st_size>cap:
            raise Refused('native_control_storage')
        data=b''
        while len(data)<before.st_size:
            part=os.read(fd,min(32768,before.st_size-len(data)))
            if not part: raise Refused('native_control_storage')
            data+=part
        if _identity(before)!=_identity(os.fstat(fd)) or _identity(before)!=_identity(path.lstat()):
            raise Refused('native_control_storage')
        return data
    finally: os.close(fd)


def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def _write(path, data):
    if path.exists() or path.is_symlink():
        if _read(path,len(data))!=data: raise Refused('native_control_conflict')
        return
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    _sync(path.parent)


class NativeRuntime:
    def __init__(self, *, host, port, tls, peer_sha256, native, spool, staging,
                 local_import_uid, connector=None):
        p._hex(peer_sha256,64)
        p._closed(native,('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
        for key in ('deployment_id','lane','boot'): p._hex(native[key],32)
        for key in ('configuration_sha256','runtime_manifest_sha256'): p._hex(native[key],64)
        if type(host) is not str or not host or len(host)>253: raise Refused('native_endpoint')
        # Trusted enrollment must resolve gateway names outside the RPC path.
        try: endpoint=ipaddress.ip_address(host)
        except ValueError: raise Refused('native_numeric_endpoint') from None
        if getattr(endpoint,'scope_id',None) is not None: raise Refused('native_numeric_endpoint')
        p._int(port,1,65535)
        if local_import_uid!=os.getuid() or type(local_import_uid) is not int: raise Refused('native_import_uid')
        if connector is None and (type(tls) is not ssl.SSLContext or tls.verify_mode!=ssl.CERT_REQUIRED or
                tls.minimum_version!=ssl.TLSVersion.TLSv1_3 or tls.maximum_version!=ssl.TLSVersion.TLSv1_3):
            raise Refused('native_tls')
        self.host,self.port,self.tls,self.pin=str(endpoint),port,tls,peer_sha256
        self.native=json.loads(canonical(native));self.local_import_uid=local_import_uid
        self.spool=_directory(spool,0o700);self.staging=_directory(staging,0o770)
        if self.spool==self.staging or self.staging in self.spool.parents or self.spool in self.staging.parents:
            raise Refused('native_spool_separation')
        if self.spool.stat().st_dev!=self.staging.stat().st_dev: raise Refused('native_import_volume')
        self.connector=connector or self._connect
        self.records={};self.lock=threading.RLock()
        self.lifecycle=threading.BoundedSemaphore(3);self.transfer=threading.BoundedSemaphore(1)
        self.failed=False
        self._account()

    @contextmanager
    def _connect(self, cutoff):
        def remaining():
            value=cutoff-time.monotonic()
            if value<=0: raise Refused('native_deadline')
            return value
        address=ipaddress.ip_address(self.host)
        # A single numeric sockaddr: no resolver or multi-address retry loop.
        raw=socket.socket(socket.AF_INET6 if address.version==6 else socket.AF_INET,socket.SOCK_STREAM)
        connection=None
        try:
            raw.settimeout(remaining())
            raw.connect((self.host,self.port))
            raw.settimeout(remaining())
            connection=self.tls.wrap_socket(raw,server_hostname=self.host,do_handshake_on_connect=False)
            connection.settimeout(remaining())
            connection.do_handshake()
            remaining()
            yield connection
        finally:
            try:
                if connection is not None: connection.close()
            finally: raw.close()

    def _account(self, extra=0, *, new=False):
        total=0;count=0
        for directory in self.spool.iterdir():
            p._hex(directory.name,64);_directory(directory,0o700);count+=1
            actual=0
            for file in directory.iterdir():
                if file.name not in ('snapshot','input','output','complete','import'): raise Refused('native_control_storage')
                info=file.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o600:
                    raise Refused('native_control_storage')
                actual+=info.st_size
            snapshot=json.loads(_read(directory/'snapshot',48000))
            from service.worker_transport import Authority
            authority=Authority(*snapshot)
            if authority.binding!=directory.name: raise Refused('native_control_storage')
            spec=p.native_execution(snapshot[1])
            from worker.native_supervisor import pcm
            if spec.reference_bytes:
                if not (directory/'input').exists(): self.failed=True
                else:
                    raw=_read(directory/'input',24000000)
                    if len(raw)!=spec.reference_bytes or hashlib.sha256(raw).hexdigest()!=spec.reference_sha256: self.failed=True
                    else: pcm(raw,spec,True)
            if (directory/'complete').exists():
                output=json.loads(_read(directory/'complete',256))
                p._closed(output,('sha256','bytes'));p._hex(output['sha256'],64);p._int(output['bytes'],46,min(spec.max_output_bytes,1048576))
                raw=_read(directory/'output',1048576)
                if len(raw)!=output['bytes'] or hashlib.sha256(raw).hexdigest()!=output['sha256']: self.failed=True
                else: pcm(raw,spec)
                final=self.staging/authority.job['attempts'][-1]['staging_id']/'output.wav'
                if final.exists() or final.is_symlink():
                    _directory(final.parent,0o770)
                    if _read(final,1048576)!=raw: self.failed=True
            # Charge declared worst case, including detached input and output copy.
            total+=max(actual,2*spec.reference_bytes+3*min(spec.max_output_bytes,1048576)+96000)
            if (directory/'import').exists() or ((directory/'output').exists() and not (directory/'complete').exists()):
                self.failed=True
        if count+int(new)>64 or total+extra>QUOTA: raise Refused('native_control_quota')
        return total

    @property
    def admission_open(self):
        return not self.failed

    def detach(self, authority, core):
        """Owner-loop only: authenticate managed input, then freeze immutable bytes."""
        spec=p.native_execution(authority.context)
        data=b''
        if spec.task=='clone':
            from worker.engines.pocket import _read as managed_read, _reference_pcm16
            core.voices._asset(authority.job['principal_id'],spec.reference_id)
            data=managed_read(core.root/'assets'/spec.reference_id,spec.reference_bytes,spec.reference_sha256)
            _reference_pcm16(data,spec.reference_rate,spec.reference_frames)
        self.allocate_stage(authority)
        return bytes(data)

    def allocate_stage(self, authority):
        """Owner only, serialized with resource observation; never adopt a collision.

        Completion (including parent fsync) precedes executor submission. A failed
        creation retains its evidence and cannot be retried as a fresh stage.
        """
        _directory(self.staging,0o770)
        stage=self.staging/authority.job['attempts'][-1]['staging_id']
        stage.mkdir(mode=0o770)
        fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fchmod(fd,0o770)
            _directory(stage,0o770)
            if os.fstat(fd)!=stage.stat(): raise Refused('native_control_storage')
            os.fsync(fd)
            _sync(self.staging)
        finally: os.close(fd)

    def bind(self, authority, runtime_id):
        config=p.native_binding(authority.context)
        if config!=self.native and any(config[k]!=self.native[k] for k in config if k!='boot'):
            raise Refused('native_configuration')
        p.runtime_id(runtime_id,config['lane'],config['boot'])
        with self.lock:
            old=self.records.get(runtime_id)
            if old and old['authority']._snapshot!=authority._snapshot: raise Refused('native_binding')
            if old: return old
            record=dict(authority=authority,config=config,event=threading.Event(),cutoff=authority.cutoff(30),uploaded=False)
            self.records[runtime_id]=record
            return record

    def _rpc(self, op, body, *, authority=None, runtime_id=None, timeout=2):
        if type(timeout) not in (int,float) or not 0<timeout<=2: raise Refused('native_timeout')
        config=p.native_binding(authority.context) if authority else self.native
        if authority and any(config[k]!=self.native[k] for k in config if k!='boot'): raise Refused('native_configuration')
        cutoff=time.monotonic()+timeout
        if op in ('prepare','put_input','start','get_output'):
            if self.failed: raise Refused('native_control_uncertain')
            cutoff=min(cutoff,authority.cutoff(30))
        if cutoff<=time.monotonic(): raise Refused('native_deadline')
        q=dict(version=p.VERSION,deployment_id=config['deployment_id'],configuration_sha256=config['configuration_sha256'],
            lane=None if op=='hello' else config['lane'],boot=None if op=='hello' else config['boot'],
            request_id=secrets.token_hex(16),runtime_id=runtime_id,binding=authority.binding if authority else None,op=op,body=body)
        p.request(q)
        semaphore=self.transfer if op in ('put_input','get_output') else self.lifecycle
        if not semaphore.acquire(False): raise Refused('native_busy')
        try:
            with self.connector(cutoff) as connection:
                p.send(connection,q,cutoff,peer_sha256=self.pin)
                reply=p.response(p.receive(connection,cutoff,peer_sha256=self.pin),request=q)
            if not reply['ok']: raise Refused(reply['error'])
            return reply['value']
        finally: semaphore.release()

    def call(self, op, body, timeout=2):
        if op=='hello':
            if body!={}: raise Refused('native_shape')
            value=self._rpc(op,{ },timeout=timeout)
            for key in ('deployment_id','configuration_sha256','lane','runtime_manifest_sha256'):
                if value[key]!=self.native[key]: raise Refused('native_hello_binding')
            return value
        if op!='ack' or set(body)!={'runtime_id'}: raise Refused('native_operation')
        r=self.records[body['runtime_id']]
        self._rpc('ack',{},authority=r['authority'],runtime_id=body['runtime_id'],timeout=timeout)
        return dict(acked=True)

    def prepare(self, authority, fixture='qwen', detached=b''):
        if fixture!='qwen': raise Refused('native_fixture')
        spec=p.native_execution(authority.context)
        config=p.native_binding(authority.context)
        if config!=self.native: raise Refused('native_configuration')
        from worker.engines.qwen_execution import current_native_hardware, inventory_from_wire
        current_native_hardware(spec,inventory_from_wire(authority.context['execution']['hardware']))
        if type(detached) is not bytes or len(detached)!=spec.reference_bytes: raise Refused('native_input')
        if detached:
            from worker.native_supervisor import pcm
            if hashlib.sha256(detached).hexdigest()!=spec.reference_sha256: raise Refused('native_input')
            pcm(detached,spec,True)
        directory=self.spool/authority.binding
        with self.lock:
            if self.failed: raise Refused('native_control_uncertain')
            if not directory.exists():
                self._account(2*spec.reference_bytes+3*min(spec.max_output_bytes,1048576)+96000,new=True)
                directory.mkdir(mode=0o700)
            _directory(directory,0o700)
            _write(directory/'snapshot',authority._snapshot)
            if detached: _write(directory/'input',detached)
            self._account()
        value=self._rpc('prepare',dict(snapshot=json.loads(authority._snapshot)),authority=authority)
        self.bind(authority,value['runtime_id'])
        return value['runtime_id']

    def interrupt(self, runtime_id):
        # Called by owner while executor is transferring. No owner-loop network I/O.
        with self.lock:
            if runtime_id in self.records: self.records[runtime_id]['event'].set()

    def _active(self, r, runtime_id):
        if r['event'].is_set():
            self._rpc('stop',{},authority=r['authority'],runtime_id=runtime_id)
            raise Refused('native_cancelled')
        if time.monotonic()>=r['cutoff']: raise Refused('native_deadline')

    def start(self, runtime_id):
        r=self.records[runtime_id];a=r['authority'];spec=p.native_execution(a.context)
        from worker.engines.qwen_execution import current_native_hardware, inventory_from_wire
        current_native_hardware(spec,inventory_from_wire(a.context['execution']['hardware']))
        if spec.reference_bytes and not r['uploaded']:
            raw=_read(self.spool/a.binding/'input',24000000)
            if len(raw)!=spec.reference_bytes or hashlib.sha256(raw).hexdigest()!=spec.reference_sha256: raise Refused('native_input')
            for offset in range(0,len(raw),p.CHUNK_LIMIT):
                self._active(r,runtime_id)
                self._rpc('put_input',dict(sha256=spec.reference_sha256,bytes=len(raw),offset=offset,
                    data=base64.b64encode(raw[offset:offset+p.CHUNK_LIMIT]).decode()),authority=a,runtime_id=runtime_id)
        r['uploaded']=True
        self._active(r,runtime_id)
        return self._rpc('start',{},authority=a,runtime_id=runtime_id)

    def observe(self, runtime_id, timeout=2):
        r=self.records[runtime_id]
        value=self._rpc('observe',{},authority=r['authority'],runtime_id=runtime_id,timeout=timeout)
        return Observation(runtime_id,State(value['state']))

    def terminate(self, runtime_id, timeout=2):
        self.interrupt(runtime_id)
        r=self.records[runtime_id]
        # All three stop variants are observations only; collect supplies proof.
        return self._rpc('stop',{},authority=r['authority'],runtime_id=runtime_id,timeout=timeout)

    kill=terminate

    def collect(self, authority, runtime_id):
        try: return self._collect(authority,runtime_id)
        except Refused:
            r=self.records.get(runtime_id)
            if r and (r["event"].is_set() or time.monotonic()>=r["cutoff"]):
                value=self._rpc("collect",{},authority=authority,runtime_id=runtime_id)
                if value["receipt"] is not None:
                    p.terminal_receipt(value["receipt"],authority=authority,runtime_id=runtime_id)
                    return value["receipt"],None
            raise

    def _collect(self, authority, runtime_id):
        r=self.bind(authority,runtime_id)
        value=self._rpc('collect',{},authority=authority,runtime_id=runtime_id)
        receipt=value['receipt']
        if receipt is None: return None
        p.terminal_receipt(receipt,authority=authority,runtime_id=runtime_id,local_import_uid=self.local_import_uid)
        if r['event'].is_set(): return receipt,None
        proposal=value['proposal'];output=value['output']
        if proposal is None:
            if output is not None: raise Refused('native_output')
            return receipt,None
        if output is None or not receipt['timely']: raise Refused('native_output')
        authority.check(proposal['message'],'result')
        spec=p.native_execution(authority.context)
        directory=self.spool/authority.binding
        _directory(directory,0o700)
        destination=self.staging/authority.job['attempts'][-1]['staging_id']/'output.wav'
        complete=directory/'complete';partial=directory/'output'
        if complete.exists():
            raw=_read(partial,1048576)
            if _read(complete,256)!=canonical(output): raise Refused('native_control_conflict')
        else:
            if partial.exists(): self.failed=True;raise Refused('native_control_uncertain')
            fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try:
                with os.fdopen(fd,'wb') as stream:
                    offset=0
                    while offset<output['bytes']:
                        self._active(r,runtime_id)
                        count=min(p.CHUNK_LIMIT,output['bytes']-offset)
                        chunk=self._rpc('get_output',dict(sha256=output['sha256'],offset=offset,bytes=count),authority=authority,runtime_id=runtime_id)
                        data=p.chunk(chunk['data']);stream.write(data);offset+=len(data)
                    stream.flush();os.fsync(stream.fileno())
                _sync(directory)
                raw=_read(partial,1048576)
                self._validate_output(authority,proposal,output,raw,spec)
                self._active(r,runtime_id)
                _write(complete,canonical(output))
            except Exception:
                self.failed=True
                raise
        self._validate_output(authority,proposal,output,raw,spec)
        stage=self.staging/authority.job['attempts'][-1]['staging_id']
        _directory(stage,0o770)
        expected_stage=stage.lstat()
        stagefd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        def verify_stage():
            _directory(stage,0o770)
            def identity(info):
                return info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid
            if not identity(expected_stage)==identity(os.fstat(stagefd))==identity(stage.lstat()):
                raise Refused('native_control_storage')
        try:
            verify_stage()
            destination=stage/'output.wav'
            if destination.exists() or destination.is_symlink():
                if _read(destination,1048576)!=raw: raise Refused('native_control_conflict')
            else:
                self._active(r,runtime_id)
                temporary=directory/'import'
                if temporary.exists(): self.failed=True;raise Refused('native_control_uncertain')
                _write(temporary,raw)
                # Atomic no-replace publication into staging; Core sees only complete
                # bytes. Drop the private link and fsync before returning ownership.
                os.link(temporary,'output.wav',dst_dir_fd=stagefd,follow_symlinks=False)
                temporary.unlink()
                _sync(directory);os.fsync(stagefd)
                self._active(r,runtime_id)
            if _read(destination,1048576)!=raw: raise Refused('native_control_conflict')
            verify_stage()
            return receipt,TransportObservation('proposal','native_owned_receipt',None,
                proposal['message'],proposal['manifest'],time.monotonic())
        finally: os.close(stagefd)

    @staticmethod
    def _validate_output(authority,proposal,output,raw,spec):
        if len(raw)!=output['bytes'] or len(raw)>min(spec.max_output_bytes,1048576) or hashlib.sha256(raw).hexdigest()!=output['sha256']:
            raise Refused('native_output')
        from worker.native_child import validate_proposal
        validate_proposal(json.loads(authority._snapshot),proposal,raw,spec)
