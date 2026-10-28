"""Installed native Kokoro commands. Installation never manufactures readiness."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
import zipfile

from service import native_package as files
from service.native_cpu_install import write,read_installation,locked_versions,create_tls
from service.native_cpu_package import VARIANT,archive,verify_archive,members
from service.native_cpu_configuration import expected,VOICE,ALIAS
from worker.transport import Refused,canonical

ROOT=Path(__file__).resolve().parents[1]


def _control_imports():
    from service.api import create_voice_app
    from clients import cli,mcp


def _preflight_child(output):
    result=dict(status='failed',weights_loaded=False,synthesis_qualified=False,installed_qualified=False)
    try:
        # Control clients legitimately own subprocesses (MCP stdio) and evaluate
        # Popen[bytes] annotations at import. They are not model-child imports.
        # Check this closure first, then permanently guard all backend imports.
        _control_imports()
        result.update(control_imports=True,control_imports_guarded=False)
        from worker.native_child import install_spawn_guard
        install_spawn_guard()
        from lib.tts_backends.kokoro_native import preflight
        result.update(preflight(),backend_imports_guarded=True)
        from worker.native_cpu_qualification import boot_identity
        boot_identity()
        with tempfile.TemporaryDirectory(prefix='native-cpu-tls-check-') as temporary:
            root=Path(temporary).resolve();create_tls(root)
            from worker.native_cpu_protocol import tls_context
            tls_context(server=True,cafile=str(root/'tls'/'ca.pem'),certfile=str(root/'tls'/'server.pem'),keyfile=str(root/'tls'/'server.key'))
        from worker.native_child import SpawnRefused
        import subprocess
        try:subprocess.Popen(['never-execute-native-cpu-preflight'])
        except SpawnRefused:result['explicit_spawn_refused']=True
        else:raise Refused('native_cpu_spawn_guard_lost')
        result.update(status='passed',control_imports=True,tls_configuration=True,boot_identity_valid=True,
            lock_sha256=hashlib.sha256((ROOT/'requirements/kokoro.lock').read_bytes()).hexdigest(),
            commit=os.environ.get('GITHUB_SHA','unknown'),runner_image=os.environ.get('ImageOS','unknown'),runner_version=os.environ.get('ImageVersion','unknown'))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[:1000],cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None)
        raise
    finally:write(output,result)


def preflight(output):
    from lib.tts_backends.kokoro_native import validate_host
    validate_host();output=Path(output).absolute();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('native_cpu_preflight_owner',ROOT/'scripts/ci-apple-speech.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    observation={}
    env={k:v for k,v in os.environ.items() if k in ('PATH','TMPDIR','GITHUB_SHA','ImageOS','ImageVersion')}
    with tempfile.TemporaryDirectory(prefix='native-cpu-preflight-home-') as home:
        env.update(HOME=home,HF_HUB_OFFLINE='1',HF_HOME=home,OMP_NUM_THREADS='2')
        try:
            owner.owned_child([sys.executable,'-I','-B','-u',str(ROOT/'scripts/native-kokoro.py'),'preflight','--child-report',str(output/'report.json')],
                env,output/'child.log',observation,seconds=60)
        finally:write(output/'owner.json',observation)
    return files.document(output/'report.json')


def finish_setup(root,*,resume=False,download_seconds=None):
    """Only the fresh copied venv may finish its create-only bootstrap."""
    if resume:
        return _complete_setup(root,download_seconds=600 if download_seconds is None else download_seconds,resume=True)
    root=files.directory(str(Path(root).absolute()));marker=files.document(root/'bootstrap.json')
    if (marker.get('version')!=1 or marker.get('root')!=str(root) or marker.get('source')!=str(ROOT) or
            marker.get('download_model') is not True or Path(sys.prefix)!=root/'venv'):
        raise Refused('native_cpu_bootstrap_binding')
    for name,digest in marker['locks'].items():
        if name not in ('requirements/kokoro.lock','requirements/client.lock','service/requirements.lock') or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise Refused('native_cpu_bootstrap_lock_changed')
    if set(marker['locks'])!={'requirements/kokoro.lock','requirements/client.lock','service/requirements.lock'}:raise Refused('native_cpu_bootstrap_lock_closure')
    source_zip=root/'source.zip';archive(source_zip);verify_archive(source_zip)
    destination=root/'source';destination.mkdir(mode=0o700)
    # Exact verified archive; never general extractall or path adoption.
    with zipfile.ZipFile(source_zip) as source:
        for entry in source.infolist():
            target=destination/entry.filename;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
            with os.fdopen(fd,'wb') as stream:stream.write(source.read(entry))
    installed=destination/'native-worker'
    for directory in sorted((p for p in installed.rglob('*') if p.is_dir()),reverse=True):directory.chmod(0o555)
    installed.chmod(0o555)
    # Preflight is guarded, bounded and weight-free. No acquisition precedes it.
    preflight(root/'preflight')
    create_tls(root)
    descriptor=expected()
    credential=dict(version=1,credentials=[dict(token=secrets.token_hex(48),principal='local',scopes=['read','production-write'])])
    write(root/'credential.json',credential)
    from importlib import metadata
    pins=locked_versions(installed)
    if any(metadata.version(name)!=version for name,version in pins.items()):raise Refused('native_cpu_dependency_identity')
    record=dict(version=1,variant=VARIANT,root=str(root),source=str(installed),
        source_files={name:files.file_hash(str(installed/name)) for name in [*members(),'native-worker.py']},
        python=str(root/'venv'/'bin'/'python'),python_sha256=files.file_hash(str(root/'venv'/'bin'/'python'),executable=True),python_version=sys.version,
        dependencies=pins,deployment_id=secrets.token_hex(16),principal='local',installed_at=datetime.now(timezone.utc).isoformat(),port=marker['port'],
        tls_sha256={path.name:hashlib.sha256(files.read(path,65536)).hexdigest() for path in (root/'tls').iterdir()},
        credential_sha256=hashlib.sha256(files.read(root/'credential.json',48000)).hexdigest(),qualification_session_id=secrets.token_hex(16),
        source_provenance=marker['source_provenance'])
    # Durable acquisition-ready checkpoint. Retried setup validates these exact
    # artifacts rather than rebuilding/adopting an arbitrary existing directory.
    write(root/'setup-record.json',record)
    return _complete_setup(root,download_seconds=marker['download_seconds'],resume=False)


def _complete_setup(root,*,download_seconds,resume):
    if type(download_seconds) is not int or not 120<=download_seconds<=600:raise Refused('native_cpu_setup_download_budget')
    root,record,descriptor,bundle=read_installation(root,_pending=True)
    if Path(sys.prefix)!=root/'venv':raise Refused('native_cpu_bootstrap_interpreter')
    # Serialize explicit retries; partial/verified provision files remain managed
    # by the existing transactional provisioner, never hand-adopted or redownloaded
    # merely because a later file failed. No automatic retry is scheduled.
    import fcntl
    lock=os.open(root/'setup-owner.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        import stat
        info=os.fstat(lock)
        if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o600:raise Refused('native_cpu_setup_lock')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:preflight(root/('resume-preflight-'+secrets.token_hex(8)))
        from service.provisioning import provision,download
        started=time.monotonic()
        def acquire(url,target):
            entries=[e for e in descriptor['files'] if e['origin']==url]
            if len(entries)!=1:raise Refused('native_cpu_asset_origin')
            remaining=download_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('native_cpu_acquisition_deadline')
            download(url,target,entries[0]['bytes'],timeout=remaining,
                allowed_origins=['https://github.com','https://release-assets.githubusercontent.com'],public_github_assets=True)
        provision(root/'models',descriptor,runtime_versions=descriptor['versions'],downloader=acquire,lock_timeout=5,transfer_timeout=download_seconds)
        state=root/'state'
        if not state.exists():state.mkdir(mode=0o700)
        files.directory(str(state))
        from service.native_runtime import _directory
        for role in ('control','worker','spool','staging'):
            path=state/role;mode=0o770 if role=='staging' else 0o700
            if not path.exists():path.mkdir(mode=mode);path.chmod(mode)
            _directory(path,mode)
            if any(path.iterdir()):raise Refused('native_cpu_pending_state_not_empty')
        write(root/'install.json',record)
    finally:os.close(lock)


def qualification_session(root):
    root,config,_,_=read_installation(root)
    lock=os.open(root/'owner.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        import fcntl
        import stat
        info=os.fstat(lock)
        if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or stat.S_IMODE(info.st_mode)!=0o600:raise Refused('native_cpu_owner_lock')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from worker.native_cpu_qualification import descriptor
        value=descriptor();value['session_id']=config['qualification_session_id']
        write(root/'qualification.json',value)  # O_EXCL: restart never renews it.
    finally:os.close(lock)
    return dict(qualification='unqualified',ready_for_speech=False,session_seconds=value['seconds'],max_jobs=value['max_jobs'])


def create_app(root,*,qualification=False):
    from service.native_cpu_install import Prepared
    from service.entrypoint import _owned_app
    holder={}
    def dependencies():
        prepared=Prepared(root,qualification=qualification);holder['prepared']=prepared
        def acquire():
            from service.native_cpu_activation import NativeCPUActivation
            from service.dispatcher import Dispatcher
            prepared.activation=NativeCPUActivation(prepared)
            return Dispatcher(prepared.control,staging=prepared.staging,installed_activation=prepared.activation)
        return prepared,None,acquire
    def mount(owner):
        from service.auth import decode_credentials
        from service.api import create_voice_app
        prepared=holder['prepared'];prepared.revalidate()
        credentials=decode_credentials(files.read(prepared.root/'credential.json',48000),{'local'})
        return create_voice_app(owner,credentials=credentials,native_installed=True)
    return _owned_app(dependencies,mount)


@contextmanager
def client_environment(root):
    root,config,_,_=read_installation(root)
    credentials=files.document(root/'credential.json')['credentials']
    if len(credentials)!=1 or credentials[0]['principal']!='local':raise Refused('native_cpu_credential')
    values=dict(SPEECH_URL='http://127.0.0.1:'+str(config['port']),SPEECH_CREDENTIAL=credentials[0]['token'],SPEECH_UPLOAD_ROOT=None)
    before={key:os.environ.get(key) for key in values}
    try:
        for key,value in values.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        yield
    finally:
        for key,value in before.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def speak(root,text,output,voice=ALIAS):
    from clients.http import HTTP
    from clients.cli import main as cli
    with client_environment(root):
        http=HTTP(credential=os.environ['SPEECH_CREDENTIAL'])
        selected=http.call('voice_resolve',dict(name=voice))
        version=http.call('voice_version_get',dict(voice_version_id=selected['version_ids'][-1]))
        request=dict(voice_version_id=version['id'],text=text,identity=version['identity'],fallback='forbid',variants=1,reuse='never')
        submitted=http.call('speech_generate',request,key='native-speak-'+secrets.token_hex(12))
        deadline=time.monotonic()+65;cursor=''
        while True:
            seconds=int(min(5,deadline-time.monotonic()))
            if seconds<1:raise TimeoutError('Job wait ended; job outcome may be unknown. Use client job get; do not automatically retry.')
            result=http.call('job_wait',dict(job_id=submitted['job_id'],cursor=cursor,wait_seconds=seconds))
            job=result['job'];cursor=result['next_cursor']
            if job['state'] in ('succeeded','failed','cancelled'):break
        if job['state']!='succeeded':raise Refused('native_cpu_job_'+job['state'])
        artifact=http.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
        return cli(['asset','content',artifact['asset_ids'][0],'--output',str(output)])


def main(argv=None):
    parser=argparse.ArgumentParser(description='Native CPU Kokoro preset preview; cloning is not supported and readiness is never inferred from installation.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('finish-setup','doctor','qualification-session','serve','client','mcp','speak'):
        command=commands.add_parser(name);command.add_argument('--root',required=True,type=Path)
        if name=='finish-setup':
            command.add_argument('--resume',action='store_true')
            command.add_argument('--setup-download-seconds',type=int)
        if name=='serve':command.add_argument('--qualification-session',action='store_true')
        if name=='client':command.add_argument('arguments',nargs=argparse.REMAINDER)
        if name=='speak':
            command.add_argument('--text',required=True);command.add_argument('--voice',default=ALIAS);command.add_argument('--output',required=True,type=Path)
    check=commands.add_parser('preflight');mode=check.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path);mode.add_argument('--child-report',type=Path)
    args=parser.parse_args(argv)
    if args.command=='preflight':
        if args.child_report:_preflight_child(args.child_report)
        else:preflight(args.output)
        return 0
    if args.command=='finish-setup':return finish_setup(args.root,resume=args.resume,download_seconds=args.setup_download_seconds)
    if args.command=='doctor':
        from service.native_cpu_install import Prepared
        prepared=Prepared(args.root)
        try:
            prepared.revalidate()
            from service.native_cpu_install import certificate_validity
            print(canonical(dict(status='prepared',device='cpu',preset='af_heart',model_store='verified',tls='verified',
                certificates_valid_until=certificate_validity(prepared.root),source_provenance=prepared.config['source_provenance'],
                ready_for_speech=False,qualification='unqualified')).decode())
        finally:prepared.close()
        return
    if args.command=='qualification-session':print(canonical(qualification_session(args.root)).decode());return
    if args.command=='client':
        from clients.cli import main as cli
        with client_environment(args.root):return cli(args.arguments)
    if args.command=='mcp':
        import asyncio
        with client_environment(args.root):
            from clients.mcp import main as mcp
            return asyncio.run(mcp())
    if args.command=='speak':return speak(args.root,args.text,args.output,args.voice)
    if args.command=='serve':
        import asyncio
        import uvicorn
        from service.entrypoint import _installed_server
        _,config,_,_=read_installation(args.root)
        app=create_app(args.root,qualification=args.qualification_session)
        server=_installed_server(uvicorn.Config(app,host='127.0.0.1',port=config['port'],access_log=False,log_level='critical',timeout_graceful_shutdown=10))
        async def run():
            await server.serve()
            lifecycle=getattr(app.state,'installed_lifecycle',None)
            if not server.started or lifecycle is None or not lifecycle['closed'] or lifecycle['error'] is not None:raise Refused('native_cpu_shutdown_unknown')
        asyncio.run(run())
