"""Shared bounded HTTP transport. No client-owned business state."""
import os
import time
from urllib.parse import quote
import httpx
from contracts.validate import validate
from service.catalogue import Failure, IMPLEMENTED, OPERATIONS, VOICE_READS, PRODUCTION_OPERATIONS

def _timeout(deadline, cap):
    if deadline is None: return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError('workflow_deadline')
    return min(cap, remaining)


class _Bounded:
    def __init__(self, transport, deadline):
        self.transport, self.deadline = transport, deadline

    def get(self, path, params=None):
        return self.transport.get(path, params, deadline=self.deadline)

    def call(self, operation, payload, *, key=None):
        return self.transport.call(operation, payload, key=key, deadline=self.deadline)

    def upload_file(self, path, *, root=None):
        return self.transport.upload_file(path, root=root, deadline=self.deadline)


class HTTP:
    def with_deadline(self, deadline):
        return _Bounded(self, deadline)

    def __init__(self, base_url=None, *, credential=None):
        self.credential = credential
        self.base_url = base_url or os.environ.get('SPEECH_URL', 'http://127.0.0.1:8780')

    def get(self, path, params=None, *, deadline=None):
        try:
            with httpx.Client(base_url=self.base_url, timeout=_timeout(deadline, 10), trust_env=False, follow_redirects=False) as client:
                response = client.get(path, params=params, headers={'Authorization': 'Bearer ' + self.credential} if self.credential else {})
                body = response.json()
                _timeout(deadline, 10)
        except (httpx.HTTPError, ValueError):
            raise ConnectionError('Speech HTTP transport unavailable.') from None
        if response.status_code != 200:
            validate('error', body)
            error = Failure(body['code'], body['message'], body['field'], body['action'])
            error.body = body
            raise error
        return body

    def upload_file(self, path, *, root=None, deadline=None):
        """Read a client-host file only. MCP must supply an operator-configured root.

        Walk every component with no-follow descriptors; never send paths to HTTP.
        """
        import stat
        from pathlib import Path
        maximum = 24*1024*1024
        candidate = Path(path)
        if root is not None:
            if candidate.is_absolute() or '..' in candidate.parts:
                raise Failure('invalid_request', 'Use a relative path inside the configured upload root.')
            base = Path(root)
            if not base.is_absolute(): raise Failure('invalid_request', 'Upload root must be absolute.')
            parts = base.parts[1:] + candidate.parts
        else:
            candidate = candidate.absolute()
            parts = candidate.parts[1:]
        fd = None
        try:
            fd = os.open('/', os.O_RDONLY|os.O_DIRECTORY)
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = next_fd
            leaf = os.open(parts[-1], os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(leaf, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                    raise Failure('resource_limit', 'Upload file exceeds media budget or is not regular.')
                def chunks():
                    remaining = info.st_size
                    while remaining:
                        _timeout(deadline, 20)
                        data = stream.read(min(65536, remaining))
                        if not data: raise ValueError('changed_file')
                        remaining -= len(data)
                        yield data
                    after = os.fstat(stream.fileno())
                    if stream.read(1) or (info.st_size,info.st_mtime_ns,info.st_ctime_ns) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                        raise ValueError('changed_file')
                with httpx.Client(base_url=self.base_url, timeout=_timeout(deadline, 20), trust_env=False, follow_redirects=False) as client:
                    response = client.post('/v1/assets', content=chunks(), headers={
                        'Authorization':'Bearer '+self.credential if self.credential else '',
                        'Content-Type':'audio/wav', 'Content-Length':str(info.st_size)})
                    body = response.json()
                    _timeout(deadline, 20)
            if response.status_code != 200:
                validate('error', body)
                error = Failure(body['code'],body['message'],body['field'],body['action'])
                error.body = body
                raise error
            return validate('asset',body)
        except (OSError, ValueError, IndexError):
            raise Failure('invalid_request', 'Client-local upload file unavailable or changed.') from None
        except httpx.HTTPError:
            raise Failure('not_ready', 'Upload transport failed; outcome may be unknown.', action='Do not automatically retry uploads.') from None
        finally:
            if fd is not None: os.close(fd)

    def post(self, path, payload, key, *, deadline=None):
        try:
            with httpx.Client(base_url=self.base_url, timeout=_timeout(deadline, 10), trust_env=False, follow_redirects=False) as client:
                response = client.post(path, json=payload, headers={
                    'Authorization':'Bearer '+self.credential if self.credential else '',
                    'Idempotency-Key':key})
                body = response.json()
                _timeout(deadline, 10)
        except (httpx.HTTPError, ValueError):
            raise ConnectionError('Production HTTP outcome unknown; do not automatically retry.') from None
        if response.status_code != 200:
            validate('error',body)
            error = Failure(body['code'],body['message'],body['field'],body['action'])
            error.body = body
            raise error
        return body

    def call(self, operation, payload, *, key=None, deadline=None):
        _timeout(deadline, 10)
        if operation not in (*IMPLEMENTED, *VOICE_READS, *PRODUCTION_OPERATIONS):
            raise Failure('not_found', 'Operation unavailable in the synthetic evaluation service.', 'operation')
        op = OPERATIONS[operation]
        validate(op['request'], payload)
        if op['http']['method'] == 'POST':
            from service.voice_store import _id
            try: _id(key)
            except ValueError: raise Failure('invalid_request', 'Supply a valid idempotency key.') from None
            return validate(op['response'], self.post(op['http']['route'], payload, key, deadline=deadline))
        route = op['http']['route']
        query = dict(payload)
        for key in tuple(query):
            marker = '{' + key + '}'
            if marker in route:
                route = route.replace(marker, quote(str(query.pop(key)), safe=''))
        return validate(op['response'], self.get(route, query, deadline=deadline))
