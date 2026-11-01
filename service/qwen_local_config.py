"""Create-only local Qwen client identity for the existing authenticated API."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import stat

DEFAULT_ROOT = Path.home() / '.local' / 'state' / 'speech-box' / 'qwen-service'
ENV_ROOT = 'SPEECH_BOX_QWEN_HOME'


def root_path(explicit=None):
    selected = explicit if explicit is not None else os.environ.get(ENV_ROOT, DEFAULT_ROOT)
    path = Path(selected).expanduser()
    if not path.is_absolute() or path == Path('/') or '..' in path.parts:
        raise ValueError('qwen_local_root_required')
    return path


def _private_file(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def create(root=None, *, port=8000, principal='local'):
    """Mint a bearer credential once; never print or overwrite it."""
    path = root_path(root)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('qwen_local_port')
    if type(principal) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', principal):
        raise ValueError('qwen_local_principal')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('qwen_local_symlink')
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if stat.S_IMODE(path.stat().st_mode) != 0o700 or path.stat().st_uid != os.getuid():
        raise ValueError('qwen_local_private_root')
    credential = path / 'control.key'
    _private_file(credential, dict(version=1, credentials=[dict(
        token=secrets.token_urlsafe(48), principal=principal,
        scopes=['read', 'production-write'])]))
    _private_file(path / 'client.json', dict(version=1, endpoint=f'http://127.0.0.1:{port}',
                                              credential_file=str(credential), principal=principal))
    return path


def discover(root=None):
    """Return local connection metadata only after file and credential checks."""
    from service.auth import decode_credentials
    from service.daemon_client import file_identity, parse
    path = root_path(root)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('qwen_local_symlink')
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('qwen_local_private_root')
    result = {}
    for name in ('client.json', 'control.key'):
        fd = os.open(path / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                    before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600 or
                    before.st_size > 48000):
                raise ValueError('qwen_local_private_file')
            raw = stream.read(48001)
            if (len(raw) != before.st_size or file_identity(os.fstat(stream.fileno())) != file_identity(before) or
                    file_identity(os.stat(path / name, follow_symlinks=False)) != file_identity(before)):
                raise ValueError('qwen_local_file_changed')
            result[name] = parse(raw)
            if name == 'control.key':
                credential_raw = raw
    client = result['client.json']
    if (type(client) is not dict or set(client) != {'version','endpoint','credential_file','principal'} or
            client['version'] != 1 or client['credential_file'] != str(path / 'control.key') or
            type(client['endpoint']) is not str or not re.fullmatch(r'http://127\.0\.0\.1:[0-9]{1,5}',client['endpoint'])):
        raise ValueError('qwen_local_client_configuration')
    port = int(client['endpoint'].rsplit(':',1)[1])
    if not 1 <= port <= 65535:
        raise ValueError('qwen_local_client_configuration')
    credentials = decode_credentials(credential_raw, {client['principal']: None})
    if (len(credentials._credentials) != 1 or
            credentials._credentials[0].scopes != frozenset(('read','production-write'))):
        raise ValueError('qwen_local_client_credential')
    return client


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mint protected local Qwen credentials and client discovery once.')
    parser.add_argument('--root')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--principal', default='local')
    args = parser.parse_args(argv)
    print(create(args.root, port=args.port, principal=args.principal))


if __name__ == '__main__':
    main()
