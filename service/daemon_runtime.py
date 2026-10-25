"""Private container-local startup observations. Never a host PID projection."""
import os
from pathlib import Path
import stat

from service import daemon_volumes as dv
from service.core import encode

LOCAL_FILE='/tmp/speech-box-local.json'


def facts(role, *, family=None):
    from service.entrypoint import local_incarnation
    incarnation=local_incarnation()  # actually running PID1, never the exec reader
    with open('/proc/self/status','rb') as stream: raw=stream.read(65537)
    if len(raw)>65536: raise ValueError('daemon_proc_budget')
    status=raw.decode('ascii')
    fields=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
    # Saved IDs and every capability set matter, not just the effective UID.
    uid=[int(n) for n in fields['Uid'].split()]; gid=[int(n) for n in fields['Gid'].split()]
    if len(set(uid))!=1 or len(set(gid))!=1 or fields['NoNewPrivs'].strip()!='1':
        raise ValueError('daemon_runtime_saved_identity')
    caps=[int(fields[k].strip(),16) for k in ('CapInh','CapPrm','CapEff','CapAmb')]
    if caps[0] or caps[3] or caps[1]!=caps[2]: raise ValueError('daemon_runtime_capabilities')
    names=('control','ipc','staging','models','config','credential') if role=='control' else ('receipts','ipc','staging','models','config')
    if family is not None:
        from service.installed_export import FAMILY
        if family is not FAMILY: raise ValueError('explicit_export_family_required')
        names=(*names,'assets')
    roots={}
    from service import install_policy as ip
    for r in names:
        path=Path(('/control/core/assets' if role=='control' else '/references/assets') if r=='assets' else dv.ROOTS[r])
        with ip.parent_fd(path) as (parent,name):
            fd=os.open(name,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            try:
                s=os.fstat(fd)
                if (s.st_dev,s.st_ino)!=(os.stat(name,dir_fd=parent,follow_symlinks=False).st_dev,os.stat(name,dir_fd=parent,follow_symlinks=False).st_ino):
                    raise ValueError('daemon_runtime_root_changed')
                roots[r]=[s.st_dev,s.st_ino]
            finally: os.close(fd)
    value=dict(role=role,pid=os.getpid(),**incarnation,uid=uid[0],gid=gid[0],groups=os.getgroups(),capabilities=caps[2],roots=roots)
    return dv.local(value,role,family=family)


def publish_control(*, family=None):
    value=facts('control',family=family)
    fd=os.open(LOCAL_FILE,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(encode(value).encode()); f.flush(); os.fsync(f.fileno())


def read_control(*, family=None):
    from service import install_policy as ip
    from service.entrypoint import _read_pid1_incarnation, _control_identity
    import json
    _control_identity()
    value=json.loads(ip.read_private(LOCAL_FILE,8192,policy=ip.SPLIT,role='control'))
    dv.local(value,'control',family=family)
    current=_read_pid1_incarnation()
    if any(value[k]!=current[k] for k in current): raise ValueError('daemon_control_local_incarnation')
    print(encode(value))


def read_export_control():
    from service.installed_export import FAMILY
    return read_control(family=FAMILY)
