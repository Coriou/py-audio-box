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

    def resolve(self, authorization):
        token = authorization[7:] if isinstance(authorization, str) and authorization.startswith('Bearer ') else ''
        if not token.isascii() or len(token) > 4096:
            token = ''
        found = None
        with self._lock:
            for record in self._credentials:
                if hmac.compare_digest(record.token, token):
                    found = record
        if found is None or 'read' not in found.scopes:
            raise Failure('forbidden', 'A valid read credential is required.', action='Configure an active bearer credential with read scope.')
        return found.principal
