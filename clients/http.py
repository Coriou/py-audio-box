"""Shared bounded HTTP transport. No client-owned business state."""
import os
import json
import time
import ipaddress
import socket
from contextlib import nullcontext
from urllib.parse import quote
import httpx
from httpcore import ConnectError, ConnectTimeout, WriteError, WriteTimeout
from httpcore._backends.sync import SyncBackend, SyncStream
from httpcore._exceptions import map_exceptions
from contracts.validate import validate
from service.catalogue import Failure, IMPLEMENTED, OPERATIONS, VOICE_READS, PRODUCTION_OPERATIONS

def _timeout(deadline, cap):
    if deadline is None: return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError('workflow_deadline')
    return min(cap, remaining) if cap is not None else remaining


_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _DeadlineStream:
    """Clamp every socket operation, including TLS and successive body reads."""
    def __init__(self, stream, deadline):
        self.stream, self.deadline = stream, deadline

    def read(self, max_bytes, timeout=None):
        return self.stream.read(max_bytes, _timeout(self.deadline, timeout))

    def write(self, buffer, timeout=None):
        if isinstance(self.stream, SyncStream):
            # SyncStream.write reuses its timeout across partial sends. Its
            # socket may also be an SSLSocket after start_tls.
            if not buffer:
                return
            while buffer:
                remaining = _timeout(self.deadline, timeout)
                with map_exceptions({socket.timeout: WriteTimeout, OSError: WriteError}):
                    self.stream._sock.settimeout(remaining)
                    buffer = buffer[self.stream._sock.send(buffer):]
            _timeout(self.deadline, timeout)
            return
        return self.stream.write(buffer, _timeout(self.deadline, timeout))

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return _DeadlineStream(self.stream.start_tls(
            ssl_context, server_hostname, _timeout(self.deadline, timeout)), self.deadline)

    def get_extra_info(self, info):
        return self.stream.get_extra_info(info)

    def close(self):
        return self.stream.close()


class _DeadlineBackend(SyncBackend):
    def __init__(self, deadline):
        self.deadline = deadline

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        address = _numeric_address(host)
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.settimeout(_timeout(self.deadline, timeout))
            if local_address is not None:
                source = _numeric_address(local_address)
                if source.version != address.version:
                    raise ConnectError('Local address family differs from endpoint.')
                sock.bind((str(source), 0))
            for option in socket_options or ():
                sock.setsockopt(*option)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(_timeout(self.deadline, timeout))
            sock.connect((str(address), port))
            _timeout(self.deadline, timeout)
            return _DeadlineStream(SyncStream(sock), self.deadline)
        except (OSError, TimeoutError) as exc:
            sock.close()
            if isinstance(exc, TimeoutError) and str(exc) == 'workflow_deadline':
                raise
            raise (ConnectTimeout if isinstance(exc, socket.timeout) else ConnectError)(str(exc)) from exc
        except BaseException:
            sock.close()
            raise

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        stream = super().connect_unix_socket(path, _timeout(self.deadline, timeout), socket_options)
        return _DeadlineStream(stream, self.deadline)


def _numeric_address(host):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        raise Failure('invalid_request', 'A workflow deadline requires a numeric IP endpoint.',
                      'url', 'Use 127.0.0.1 (or [::1]) for a local service; hostname URLs are available for unbounded low-level calls.') from None


def _client(base_url, deadline, cap):
    client = httpx.Client(base_url=base_url, timeout=_timeout(deadline, cap),
                          trust_env=False, follow_redirects=False)
    if deadline is not None and isinstance(getattr(client, '_transport', None), httpx.HTTPTransport):
        try:
            _numeric_address(client.base_url.host)
        except BaseException:
            client.close()
            raise
        # HTTPX's per-read timeout resets on each chunk. The connection pool's
        # network backend enforces the same absolute deadline on every socket call.
        client._transport._pool._network_backend = _DeadlineBackend(deadline)
    return client


def _body(client, method, path, *, deadline, cap, **kwargs):
    if hasattr(client, 'stream'):
        context = client.stream(method, path, **kwargs)
    else:
        context = nullcontext(getattr(client, method.lower())(path, **kwargs))
    with context as response:
        data = bytearray()
        chunks = (response.content,) if response.is_stream_consumed else response.iter_bytes(chunk_size=65536)
        for chunk in chunks:
            _timeout(deadline, cap)
            if len(data) + len(chunk) > _MAX_RESPONSE_BYTES:
                raise httpx.DecodingError('HTTP response exceeds client limit.')
            data.extend(chunk)
        _timeout(deadline, cap)
        try:
            body = json.loads(data)
        except ValueError:
            raise httpx.DecodingError('Invalid HTTP JSON response.') from None
        _timeout(deadline, cap)
        return response.status_code, body


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
            with _client(self.base_url, deadline, 10) as client:
                status, body = _body(client, 'GET', path, deadline=deadline, cap=10,
                    params=params, headers={'Authorization': 'Bearer ' + self.credential} if self.credential else {})
        except (httpx.HTTPError, ValueError):
            _timeout(deadline, 10)
            raise ConnectionError('Speech HTTP transport unavailable.') from None
        if status != 200:
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
                with _client(self.base_url, deadline, 20) as client:
                    status, body = _body(client, 'POST', '/v1/assets', deadline=deadline, cap=20,
                        content=chunks(), headers={
                        'Authorization':'Bearer '+self.credential if self.credential else '',
                        'Content-Type':'audio/wav', 'Content-Length':str(info.st_size)})
            if status != 200:
                validate('error', body)
                error = Failure(body['code'],body['message'],body['field'],body['action'])
                error.body = body
                raise error
            return validate('asset',body)
        except (OSError, ValueError, IndexError):
            raise Failure('invalid_request', 'Client-local upload file unavailable or changed.') from None
        except httpx.HTTPError:
            _timeout(deadline, 20)
            raise Failure('not_ready', 'Upload transport failed; outcome may be unknown.', action='Do not automatically retry uploads.') from None
        finally:
            if fd is not None: os.close(fd)

    def post(self, path, payload, key, *, deadline=None):
        try:
            with _client(self.base_url, deadline, 10) as client:
                status, body = _body(client, 'POST', path, deadline=deadline, cap=10,
                    json=payload, headers={
                    'Authorization':'Bearer '+self.credential if self.credential else '',
                    'Idempotency-Key':key})
        except (httpx.HTTPError, ValueError):
            _timeout(deadline, 10)
            raise ConnectionError('Production HTTP outcome unknown; do not automatically retry.') from None
        if status != 200:
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
