"""Stdlib-only create-only Mac bootstrap; model work needs explicit download flag."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
_OWNED={}


def host():
    major=platform.mac_ver()[0].split('.')[0]
    if (platform.system(),platform.machine(),platform.python_implementation(),sys.version_info[:2])!=('Darwin','arm64','CPython',(3,11)) or not major.isdigit() or int(major)<14:
        raise ValueError('Use native Apple arm64 CPython 3.11 on macOS 14 or later; leave Rosetta before setup.')
    if sys.version_info[:3]<(3,11,4):raise ValueError('Use CPython 3.11.4 or later in the 3.11 series; bundled pip must support --keyring-provider (pip23.1+).')


def root_path(value):
    path=Path(value).absolute()
    if '..' in path.parts or path==Path('/') or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Use an explicit installation directory without symlink components.')
    return path


def environment(root):
    return dict(PATH=os.defpath,HOME=str(root/'bootstrap-home'),TMPDIR=str(root/'tmp'),PYTHONNOUSERSITE='1',
        PIP_CONFIG_FILE=os.devnull,PIP_NO_INPUT='1',PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')


def bounded(argv,root,name,seconds):
    """One explicit binary-only setup phase; no shell, automatic retry or orphan."""
    path=root/(name+'.log')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    process=None
    try:
        with os.fdopen(fd,'wb') as log:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                close_fds=True,start_new_session=True,env=environment(root))
            _OWNED[process.pid]=process;cutoff=time.monotonic()+seconds
            while process.poll() is None:
                if time.monotonic()>=cutoff:raise TimeoutError(name+' phase deadline')
                if path.stat().st_size>1024**2:raise ValueError(name+' log limit')
                time.sleep(.05)
            code=process.wait(timeout=1)
            if path.stat().st_size>1024**2:raise ValueError(name+' log limit')
            if code:raise RuntimeError(name+' failed; retained '+str(path))
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        if process.poll() is None:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                        process.wait(timeout=1)
                else:process.wait(timeout=1)
            except BaseException as cleanup:
                raise RuntimeError('setup process cleanup unknown; retain installation and logs') from cleanup
            else:_OWNED.pop(process.pid,None)


def provenance(root):
    result=dict(commit=None,dirty=None)
    if not (ROOT/'.git').exists():return result
    env=environment(root);env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull)
    try:
        prefix=['git','-c','core.fsmonitor=false','-C',str(ROOT)]
        commit=subprocess.run([*prefix,'rev-parse','--verify','HEAD'],env=env,capture_output=True,timeout=2,check=True).stdout.decode('ascii').strip()
        if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):return result
        dirty=subprocess.run([*prefix,'status','--porcelain','--untracked-files=no'],env=env,capture_output=True,timeout=2,check=True).stdout
        result.update(commit=commit,dirty=bool(dirty))
    except (OSError,subprocess.SubprocessError,UnicodeError):pass
    return result


def setup(root,*,download_model=False,port=8788,resume=False,download_seconds=600):
    host();root=root_path(root)
    if type(download_seconds) is not int or not 120<=download_seconds<=600:raise ValueError('Setup download budget must be120..600seconds, separate from synthesis deadlines.')
    if resume:
        if not root.is_dir() or not (root/'setup-record.json').is_file() or (root/'install.json').exists():raise ValueError('Resume requires a retained acquisition-ready setup; completed or earlier partial setup is not adopted.')
        root_info=root.stat()
        if root_info.st_uid!=os.getuid() or stat.S_IMODE(root_info.st_mode)!=0o700:raise ValueError('Retained setup must be private and owned by this user.')
        # The sealed installed source/interpreter validates the entire pending
        # record before using existing immutable provision staging. No new venv.
        for relative in ('venv/bin/python','source/native-worker/native-worker.py'):
            path=root_path(root/relative);info=path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_mode&0o022:raise ValueError('Unsafe retained setup artifact')
        bounded([str(root/'venv/bin/python'),'-I','-B',str(root/'source/native-worker/native-worker.py'),
            'finish-setup','--root',str(root),'--resume','--setup-download-seconds',str(download_seconds)],root,'resume-'+uuid.uuid4().hex,download_seconds+130)
        return
    if root.exists():raise ValueError('Installation directory exists; use doctor/serve or explicit setup --resume after an acquisition failure. Partial setup is retained, never silently overwritten.')
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Use a TCP port between 1 and 65535.')
    os.umask(0o077);root.mkdir(mode=0o700,parents=True)
    for name in ('bootstrap-home','tmp'):(root/name).mkdir(mode=0o700)
    marker=dict(version=1,source=str(ROOT),root=str(root),download_model=download_model,port=port,download_seconds=download_seconds,source_provenance=provenance(root),
        locks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('requirements/kokoro.lock','requirements/client.lock','service/requirements.lock')})
    with (root/'bootstrap.json').open('x') as stream:json.dump(marker,stream,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    bounded([sys.executable,'-m','venv','--copies',str(root/'venv')],root,'venv',60)
    python=str(root/'venv'/'bin'/'python')
    bounded([python,'-c','from importlib.metadata import version; v=version("pip"); assert tuple(map(int,v.split(".")[:2])) >= (23,1), "Use native CPython3.11.4+ with bundled pip23.1+; setup does not upgrade pip automatically."'],root,'pip-minimum',15)
    bounded([python,'-m','pip','install','--require-hashes','--only-binary=:all:','--retries','0','--timeout','15','--keyring-provider','disabled',
        '--index-url','https://pypi.org/simple','--report',str(root/'pip-install.json'),
        *sum((['-r',str(ROOT/name)] for name in marker['locks']),[])],root,'dependencies',180)
    bounded([python,'-m','pip','check'],root,'dependency-check',30)
    bounded([python,str(ROOT/'scripts/native-kokoro.py'),'finish-setup','--root',str(root)],root,'finish-setup',download_seconds+130)
    print(json.dumps(dict(installed=True,ready_for_speech=False,root=str(root),
        next_action='Use the installed native-worker.py doctor, then explicitly create a finite qualification session.'),sort_keys=True))


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ('--help','-h'):
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command',choices=('setup','doctor','qualification-session','serve','client','mcp','speak','preflight'))
        parser.print_help();return
    if arguments and arguments[0]=='setup':
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command',choices=['setup']);parser.add_argument('--root',required=True)
        parser.add_argument('--download-model',action='store_true',help='Explicitly download the pinned public 353.75 MB Kokoro model/voices, after guarded preflight.')
        parser.add_argument('--port',type=int,default=8788)
        parser.add_argument('--resume',action='store_true',help='Explicit retry only from an acquisition-ready retained setup; reverify source, dependencies and immutable staging.')
        parser.add_argument('--setup-download-seconds',type=int,default=600)
        args=parser.parse_args(arguments)
        if not args.download_model:parser.error('Setup needs explicit --download-model; it is not a gated model and accepts no tokens.')
        try:setup(args.root,download_model=args.download_model,port=args.port,resume=args.resume,download_seconds=args.setup_download_seconds)
        except (Exception,KeyboardInterrupt) as exc:raise SystemExit(str(exc)+'; setup stopped, no automatic retry.') from exc
        return
    # Non-setup commands use the installed interpreter/source when invoked from
    # a checkout, preventing parent/child source drift after a checkout update.
    if arguments and arguments[0]!='finish-setup' and '--root' in arguments:
        index=arguments.index('--root')+1
        if index>=len(arguments):raise SystemExit('--root requires a path')
        root=root_path(arguments[index]);entry=root/'source'/'native-worker'/'native-worker.py';python=root/'venv'/'bin'/'python'
        if ROOT!=entry.parent or not sys.flags.isolated or Path(sys.prefix)!=root/'venv':
            for path in (entry,python):
                info=path.lstat()
                if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1:
                    raise SystemExit('Installed source/interpreter is missing or unsafe; retain the installation.')
            os.execv(str(python),[str(python),'-I','-B',str(entry),*arguments])
    from service.native_cpu_setup import main as installed_main
    return installed_main(arguments)
