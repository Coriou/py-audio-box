"""Checked native CPU installation and owned existing mTLS service lifecycle."""
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import sys
import threading

from service import native_package as files
from service.native_cpu_package import VARIANT
from service.native_cpu_configuration import expected,profile,context,template
from service.provisioning import identity,manifest_sha256
from worker import native_cpu_protocol as p
from worker.transport import Refused,canonical


def write(path,value):
    from service.native_runtime import _sync
    raw=value if type(value) is bytes else canonical(value)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    _sync(Path(path).parent)


def locked_versions(source):
    result={}
    for name in ('requirements/kokoro.lock','requirements/client.lock','service/requirements.lock'):
        for package,version in re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)',(source/name).read_text(),re.M):
            if package in result and result[package]!=version:raise Refused('native_cpu_dependency_lock_conflict')
            result[package]=version
    return result


def read_installation(root):
    from lib.tts_backends.kokoro_native import validate_host,validate_versions
    validate_host();validate_versions()
    root=files.directory(str(Path(root).absolute()))
    config=files.document(root/'install.json')
    p._closed(config,('version','variant','root','source','source_files','python','python_sha256','python_version','dependencies',
        'deployment_id','principal','installed_at','port','tls_sha256','credential_sha256','qualification_session_id'))
    if type(config['version']) is not int or config['version']!=1 or config['variant']!=VARIANT or config['root']!=str(root):raise Refused('native_cpu_installation')
    source=root/'source'/'native-worker'
    if config['source']!=str(source) or config['python']!=str(root/'venv'/'bin'/'python'):raise Refused('native_cpu_installation_paths')
    from service.native_cpu_package import members
    if type(config['source_files']) is not dict or set(config['source_files'])!=set(members())|{'native-worker.py'}:raise Refused('native_cpu_source_members')
    for name,digest in config['source_files'].items():
        p._hex(digest,64)
        if files.file_hash(str(source/name))!=digest:raise Refused('native_cpu_source_changed')
    if files.file_hash(config['python'],executable=True)!=config['python_sha256'] or sys.version!=config['python_version']:raise Refused('native_cpu_interpreter_changed')
    pins=locked_versions(source)
    if config['dependencies']!=pins or any(metadata.version(name)!=version for name,version in pins.items()):raise Refused('native_cpu_dependencies_changed')
    p._hex(config['deployment_id'],32);p._hex(config['qualification_session_id'],32);p._int(config['port'],1,65535)
    if config['principal']!='local':raise Refused('native_cpu_principal')
    from datetime import datetime
    if datetime.fromisoformat(config['installed_at']).tzinfo is None:raise Refused('native_cpu_installation_clock')
    if set(config['tls_sha256'])!={'ca.pem','server.pem','server.key','client.pem','client.key'}:raise Refused('native_cpu_tls_files')
    files.directory(str(root/'tls'))
    for name,digest in config['tls_sha256'].items():
        if hashlib.sha256(files.read(root/'tls'/name,65536)).hexdigest()!=digest:raise Refused('native_cpu_tls_changed')
    credential=files.read(root/'credential.json',48000)
    if hashlib.sha256(credential).hexdigest()!=config['credential_sha256']:raise Refused('native_cpu_credential_changed')
    from service.auth import decode_credentials
    decode_credentials(credential,{'local'})
    descriptor=expected();bundle=root/'models'/'bundles'/identity(descriptor)
    return root,config,descriptor,bundle


def create_tls(root):
    """Local CA key stays in memory; all retained keys are newly generated."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
    from datetime import datetime,timezone,timedelta
    import ipaddress
    directory=root/'tls';directory.mkdir(mode=0o700)
    now=datetime.now(timezone.utc);key=ec.generate_private_key(ec.SECP256R1())
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'py-audio-box native CPU local CA')])
    def builder(subject,public):
        return x509.CertificateBuilder().subject_name(subject).issuer_name(name).public_key(public).serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=90))
    ca=builder(name,key.public_key()).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).add_extension(
        x509.KeyUsage(digital_signature=True,content_commitment=False,key_encipherment=False,data_encipherment=False,key_agreement=False,key_cert_sign=True,crl_sign=True,encipher_only=None,decipher_only=None),critical=True).sign(key,hashes.SHA256())
    write(directory/'ca.pem',ca.public_bytes(serialization.Encoding.PEM))
    for role,usage in (('server',ExtendedKeyUsageOID.SERVER_AUTH),('client',ExtendedKeyUsageOID.CLIENT_AUTH)):
        private=ec.generate_private_key(ec.SECP256R1())
        subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'py-audio-box native CPU '+role)])
        certificate=builder(subject,private.public_key()).add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True).add_extension(x509.ExtendedKeyUsage([usage]),critical=True).add_extension(
            x509.KeyUsage(digital_signature=True,content_commitment=False,key_encipherment=False,data_encipherment=False,key_agreement=False,key_cert_sign=False,crl_sign=False,encipher_only=None,decipher_only=None),critical=True)
        if role=='server':certificate=certificate.add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False)
        certificate=certificate.sign(key,hashes.SHA256())
        write(directory/(role+'.pem'),certificate.public_bytes(serialization.Encoding.PEM))
        write(directory/(role+'.key'),private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))


def certificate_pin(root,role):
    import ssl
    raw=files.read(root/'tls'/(role+'.pem'),65536).decode('ascii')
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(raw)).hexdigest()


class Prepared:
    """Own the actual custodian/ports/locks; no Core or bearer data in model child."""
    def __init__(self,root,*,qualification=False):
        if type(qualification) is not bool:raise Refused('native_cpu_qualification_flag')
        self.root,self.config,self.descriptor,self.bundle=read_installation(root)
        self.identity=hashlib.sha256(canonical(self.config)).hexdigest()
        self.runtime_manifest=hashlib.sha256(canonical(dict(variant=VARIANT,source=self.config['source_files'],python=self.config['python_sha256'],python_version=self.config['python_version'],dependencies=self.config['dependencies'],bundle=manifest_sha256(self.descriptor)))).hexdigest()
        self.profile=profile(self.descriptor,revision='native-cpu-'+self.identity[:32],installed_at=self.config['installed_at'])
        self.profiles=[self.profile];self.qualification=None
        if qualification:
            from worker.native_cpu_qualification import Session
            session=files.document(self.root/'qualification.json')
            if session['session_id']!=self.config['qualification_session_id']:raise Refused('native_cpu_qualification_installation')
            self.qualification=Session(session,configuration_sha256=self.identity,runtime_manifest_sha256=self.runtime_manifest,
                deployment_id=self.config['deployment_id'],profile=self.profile,manifest_sha256=manifest_sha256(self.descriptor),principal=self.config['principal'])
        self.control=self.root/'state'/'control';self.staging=self.root/'state'/'staging'
        for role in ('control','worker','spool'):
            files.directory(str(self.root/'state'/role))
        from service.native_runtime import _directory
        _directory(self.staging,0o770)
        self.lock=os.open(self.root/'owner.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
        info=os.fstat(self.lock)
        if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o600:
            os.close(self.lock);raise Refused('native_cpu_owner_lock')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(self.lock);raise
        self.supervisor=None;self.listener=None;self.thread=None;self.errors=[];self.closed=False
        try:self._prepare()
        except BaseException:
            self.close()
            raise

    def revalidate(self):
        _,config,descriptor,bundle=read_installation(self.root)
        if canonical(config)!=canonical(self.config) or descriptor!=self.descriptor or bundle!=self.bundle:raise Refused('native_cpu_installation_changed')
        return self.identity

    def _prepare(self):
        from worker.hardware import HardwareInventory,Device,DeviceFamily,host_metadata
        from worker.native_cpu_supervisor import CPUJournal,NativeCPUSupervisor
        from service.execution_context import verify_local
        from service.native_cpu_context import NativeCPUContexts
        from service.native_cpu_runtime import NativeCPURuntime
        self.inventory=HardwareInventory(host_metadata('native'),(Device(DeviceFamily.CPU,0,True,True,None),))
        self.inventory_wire=p.inventory_wire(self.inventory);p.inventory(self.inventory_wire)
        journal=CPUJournal(self.root/'state'/'worker',self.config['deployment_id'],self.identity)
        try:lane=journal.value['lane']
        finally:journal.lock.close()
        boot=secrets.token_hex(16)
        self.native=dict(deployment_id=self.config['deployment_id'],configuration_sha256=self.identity,lane=lane,boot=boot,runtime_manifest_sha256=self.runtime_manifest)
        owned=context(self.descriptor,self.profile,self.native,self.inventory)
        mapping={identity(self.descriptor):self.bundle};verify_local(owned,mapping)
        snapshot=template(owned,self.profiles,self.config['principal'])
        binding=hashlib.sha256(canonical(snapshot)).hexdigest()
        self.supervisor=NativeCPUSupervisor(self.root/'state'/'worker',deployment_id=self.config['deployment_id'],configuration_sha256=self.identity,
            peer_pin=certificate_pin(self.root,'client'),runtime_manifest_sha256=self.runtime_manifest,
            registrations={binding:dict(snapshot=snapshot,execution=p.native_execution(owned),bundles=mapping,inventory=self.inventory)},
            python=self.config['python'],child_entry=str(Path(self.config['source'])/'worker'/'native_cpu_child.py'),qualification=self.qualification,_boot=boot)
        self.contexts=NativeCPUContexts({self.config['principal']:owned},mapping)
        self.contexts._inventory=self.inventory;self.contexts.qualification=self.qualification
        self.budget=dict(slots=1,resident_models=1,ram_bytes=2*1024**3,vram_bytes=0,pids=1,queue=2,disk_bytes=128*1024**2,free_bytes=32*1024**2)
        self.listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM);self.listener.bind(('127.0.0.1',0));self.listener.listen(8)
        host,port=self.listener.getsockname()
        tls=p.tls_context(server=True,cafile=str(self.root/'tls'/'ca.pem'),certfile=str(self.root/'tls'/'server.pem'),keyfile=str(self.root/'tls'/'server.key'))
        from worker.native_supervisor import serve
        def run():
            # Both listener and control live in this checked native host process;
            # no Docker namespace/loopback address is being reinterpreted.
            try:serve(self.supervisor,tls,host,port,proven_host_network=True,_listener=self.listener)
            except BaseException as exc:self.errors.append(exc)
            finally:self.supervisor.shutdown_event.set()
        self.thread=threading.Thread(target=run,name='native-cpu-custodian',daemon=True);self.thread.start()
        client=p.tls_context(server=False,cafile=str(self.root/'tls'/'ca.pem'),certfile=str(self.root/'tls'/'client.pem'),keyfile=str(self.root/'tls'/'client.key'))
        self.runtime=NativeCPURuntime(host=host,port=port,tls=client,peer_sha256=certificate_pin(self.root,'server'),native=self.native,
            spool=self.root/'state'/'spool',staging=self.staging,local_import_uid=os.getuid())
        self.contexts.bind_availability(self.runtime)
        self.runtime.check_availability(owned)
        if self.errors:raise self.errors[0]

    def close(self):
        if self.closed:return
        if self.supervisor is not None:self.supervisor.shutdown_event.set()
        if self.thread is not None:
            self.thread.join(5)  # existing 2s model reap + 1s watchdog + 1s RPC join
            if self.thread.is_alive() or self.supervisor.active is not None:raise Refused('native_cpu_shutdown_unknown')
        if self.listener is not None:self.listener.close()
        if self.errors:raise Refused('native_cpu_custodian_failed') from self.errors[0]
        if self.supervisor is not None:self.supervisor.journal.lock.close()
        os.close(self.lock);self.closed=True
        activation=getattr(self,'activation',None)
        if activation is not None:activation.release()
