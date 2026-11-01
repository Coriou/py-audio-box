"""Explicit local installation discovery; never searches user directories."""
import json
import os
from pathlib import Path
import stat
import re


class ConfigurationFailure(ValueError):
    """Safe, actionable error for an endpoint/credential pairing refusal."""
    code = 'invalid_request'
    message = 'Discovered installation credential requires its configured endpoint.'
    action = 'Use the configured installation URL or supply an explicit credential for a different URL.'

    def __init__(self):
        super().__init__(self.message)

    @property
    def body(self):
        from service.catalogue import problem
        return problem(self.code, self.message, 'url', self.action)


class InstallationFailure(ValueError):
    """A safe local discovery failure with the failing configuration boundary."""
    code = 'invalid_request'
    action = ('Place the installation under a directory owned by root or the client caller, '
              'with a private root and credential files owned by that caller; then retry discovery.')

    def __init__(self, message, field='installation'):
        self.message, self.field = message, field
        super().__init__(message)

    @property
    def body(self):
        from service.catalogue import problem
        action = ('Check the local installation document and its endpoint and credential references.'
                  if self.field == 'config' else self.action)
        return problem(self.code, self.message, self.field, action)


def installation(root):
    try:
        return _installation(root)
    except InstallationFailure:
        raise
    except OSError as exc:
        raise InstallationFailure('Local installation path or file cannot be read.') from exc
    except ValueError as exc:
        # Messages below are fixed client strings; no paths, credentials or
        # document contents are copied into the structured response.
        message = str(exc)
        field = 'config' if message.startswith(('Invalid ', 'Expected one local')) else 'installation'
        raise InstallationFailure(message, field) from exc


def _installation(root):
    root = Path(root)
    if not root.is_absolute(): raise ValueError('Installation path must be absolute.')
    current = Path('/')
    for part in root.parts[1:]:
        current = current / part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise InstallationFailure('Installation ancestor is not a directory.')
        if info.st_uid not in (0, os.getuid()):
            raise InstallationFailure('Installation ancestor owner is neither root nor the client caller.')
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise InstallationFailure('Installation ancestor mode permits unsafe writes.')
    def read(name):
        path = root / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077 or not 0 < info.st_size <= 48000:
            raise ValueError('Installation file owner or mode is unsafe.')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_nlink) != (
                    info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink):
                raise ValueError('Installation file changed before reading.')
            raw = os.read(fd, 48001)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size) or len(raw) != before.st_size:
                raise ValueError('Installation file changed while reading.')
            final = path.lstat()
            if (after.st_dev, after.st_ino) != (final.st_dev, final.st_ino):
                raise ValueError('Installation file changed after reading.')
            try:
                return json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                raise InstallationFailure('Invalid local installation document.', 'config') from None
        finally:
            os.close(fd)
    if (root/'client.json').exists():
        if stat.S_IMODE(root.lstat().st_mode) != 0o700:
            raise ValueError('Qwen local installation root must be private.')
        config, credentials = read('client.json'), read('control.key')
        if (type(config) is not dict or set(config) != {'version','endpoint','credential_file','principal'}
                or config['version'] != 1 or config['credential_file'] != str(root/'control.key')
                or type(config['endpoint']) is not str
                or not re.fullmatch(r'http://127\.0\.0\.1:([0-9]{1,5})',config['endpoint'])):
            raise ValueError('Invalid Qwen local client configuration.')
        port = int(config['endpoint'].rsplit(':',1)[1])
        if (not 1 <= port <= 65535 or type(config['principal']) is not str
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}',config['principal'])):
            raise ValueError('Invalid Qwen local client port.')
        endpoint = config['endpoint']
        principal = config['principal']
    else:
        config, credentials = read('install.json'), read('credential.json')
        if type(config) is not dict:
            raise InstallationFailure('Invalid local installation configuration.', 'config')
        port = config.get('port')
        if config.get('root') != str(root) or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid local installation configuration.')
        endpoint, principal = 'http://127.0.0.1:' + str(port), 'local'
    if type(credentials) is not dict or credentials.get('version') != 1:
        raise ValueError('Invalid local credential document.')
    records = credentials.get('credentials')
    if (type(records) is not list or len(records) != 1 or type(records[0]) is not dict
            or records[0].get('principal') != principal
            or type(records[0].get('scopes')) is not list
            or set(records[0]['scopes']) != {'read','production-write'}):
        raise ValueError('Expected one local write credential.')
    token = records[0].get('token')
    if type(token) is not str or not re.fullmatch(r'[A-Za-z0-9._:-]{32,4096}', token):
        raise ValueError('Invalid local credential.')
    return endpoint, token


def resolve(url=None, credential=None, root=None):
    # Explicit endpoint/credential > explicit installation > environment > default.
    if root:
        local_url, local_token = installation(root) if not url or not credential else (None, None)
        if local_token and url and url != local_url and not credential:
            raise ConfigurationFailure()
        return url or local_url, credential or local_token
    preferred_url = url or os.environ.get('SPEECH_URL')
    preferred_token = credential or os.environ.get('SPEECH_CREDENTIAL')
    selected = os.environ.get('SPEECH_INSTALLATION')
    if not selected and not preferred_url and not preferred_token:
        from service.qwen_local_config import DEFAULT_ROOT
        if (DEFAULT_ROOT / 'client.json').exists():
            selected = str(DEFAULT_ROOT)
    local_url, local_token = installation(selected) if selected and (not preferred_url or not preferred_token) else (None, None)
    if local_token and preferred_url and preferred_url != local_url and not preferred_token:
        raise ConfigurationFailure()
    return preferred_url or local_url, preferred_token or local_token
