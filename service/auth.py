"""Explicit operator-owned, revocable bearer credentials; no default secret."""
from dataclasses import dataclass
import hmac
import threading
from service.catalogue import Failure

@dataclass(frozen=True)
class Credential:
    token: str
    principal: str
    scopes: frozenset[str]

class StaticCredentials:
    def __init__(self, credentials):
        self._lock = threading.Lock()
        self.replace(credentials)

    def replace(self, credentials):
        from re import fullmatch
        records = tuple(credentials)
        if not records or any(not isinstance(c, Credential) or not c.token or
                not c.token.isascii() or len(c.token) > 4096 or
                not fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', c.principal) or
                not isinstance(c.scopes, frozenset) for c in records):
            raise ValueError('invalid_credential_configuration')
        if len({c.token for c in records}) != len(records):
            raise ValueError('duplicate_credential')
        with self._lock:
            self._credentials = records

    def revoke(self, token):
        with self._lock:
            self._credentials = tuple(c for c in self._credentials if not hmac.compare_digest(c.token, token))

    def resolve(self, authorization, *, required_scope="read"):
        token = authorization[7:] if isinstance(authorization, str) and authorization.startswith('Bearer ') else ''
        if not token.isascii() or len(token) > 4096:
            token = ''
        found = None
        with self._lock:
            for record in self._credentials:
                if hmac.compare_digest(record.token, token):
                    found = record
        if found is None or not {'read', required_scope} <= found.scopes:
            raise Failure('forbidden', 'A valid credential with the required scopes is required.', action='Configure an active bearer credential with read scope.')
        return found.principal


def private_input(path, *, mode, limit):
    """Read an explicit fixed-policy input without following any path symlink."""
    import os
    import stat
    from pathlib import Path
    from service import install_policy as ip
    path = Path(path)
    with ip.parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or
                    (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode), before.st_nlink) != (0, 10001, mode, 1) or
                    not 1 <= before.st_size <= limit):
                raise ValueError('installed_input_policy')
            raw = os.read(fd, limit+1)
            if (len(raw) != before.st_size or ip.identity(os.fstat(fd)) != ip.identity(before) or
                    ip.identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != ip.identity(before)):
                raise ValueError('installed_input_changed')
        finally:
            os.close(fd)
    with ip.parent_fd(path) as (parent, name):
        if ip.identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != ip.identity(before):
            raise ValueError('installed_input_changed')
    return raw


def installed_credentials(path, principals):
    from service.recovery import host_json, closed
    from re import fullmatch
    value = host_json(private_input(path, mode=0o640, limit=48000))
    closed(value, 'version credentials')
    if type(value['version']) is not int or value['version'] != 1 or type(value['credentials']) is not list or not 1 <= len(value['credentials']) <= 32:
        raise ValueError('invalid_credential_configuration')
    records = []
    for record in value['credentials']:
        closed(record, 'token principal scopes')
        if (type(record['token']) is not str or not fullmatch(r'[A-Za-z0-9._:-]{32,4096}', record['token']) or
                record['principal'] not in principals or type(record['scopes']) is not list or
                not record['scopes'] or len(set(record['scopes'])) != len(record['scopes']) or
                not set(record['scopes']) <= {'read', 'production-write'} or 'read' not in record['scopes']):
            raise ValueError('invalid_credential_configuration')
        records.append(Credential(record['token'], record['principal'], frozenset(record['scopes'])))
    return StaticCredentials(records)
