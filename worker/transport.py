"""Linux one-exchange transport. No store, lifecycle or publication authority."""
from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import socket
import struct
import time

LIMIT = 65536

class Refused(ValueError):
    pass

class Uncertain(OSError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def remaining(connection, cutoff):
    seconds = cutoff - time.monotonic()
    if seconds <= 0:
        raise Uncertain('timeout')
    connection.settimeout(seconds)


def exact(connection, count, cutoff):
    result = bytearray()
    while len(result) < count:
        remaining(connection, cutoff)
        chunk = connection.recv(count - len(result))
        if not chunk:
            raise Uncertain('disconnect')
        result.extend(chunk)
    return bytes(result)


def send(connection, value, cutoff):
    data = canonical(value)
    if not 0 < len(data) <= LIMIT:
        raise Refused('frame_limit')
    remaining(connection, cutoff)
    connection.sendall(struct.pack('!I', len(data)) + data)


def receive(connection, cutoff):
    size, = struct.unpack('!I', exact(connection, 4, cutoff))
    if not 0 < size <= LIMIT:
        raise Refused('frame_limit')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refused('duplicate_key')
            result[key] = value
        return result
    try:
        return json.loads(exact(connection, size, cutoff), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Refused('nonfinite')))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise Refused('invalid_frame') from error


@dataclass(frozen=True)
class PeerEvidence:
    pid: int
    uid: int
    gid: int
    binding: str


@dataclass(frozen=True)
class Session:
    secret: str
    binding: str
    peer_uid: int
    peer_gid: int


def peer(connection, session):
    if not hasattr(socket, 'SO_PEERCRED'):
        raise Refused('linux_peer_credentials_required')
    pid, uid, gid = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if (uid, gid) != (session.peer_uid, session.peer_gid):
        raise Refused('peer_mismatch')
    return PeerEvidence(pid, uid, gid, session.binding)


def proof(session, role, worker_nonce, control_nonce):
    return hmac.new(bytes.fromhex(session.secret), canonical([
        'w1-session/1', session.binding, role, worker_nonce, control_nonce]), hashlib.sha256).hexdigest()


def auth_frame(value, role):
    if not isinstance(value, dict) or set(value) != {'role', 'nonce', 'proof'} or value['role'] != role:
        raise Refused('session_mismatch')
    if not all(isinstance(value[k], str) and len(value[k]) == 64 and
               all(c in '0123456789abcdef' for c in value[k]) for k in ('nonce', 'proof')):
        raise Refused('session_mismatch')
    return value


def authenticate(connection, session, cutoff, *, control):
    evidence = peer(connection, session)
    if control:
        hello = auth_frame(receive(connection, cutoff), 'worker')
        wn = hello['nonce']
        if not hmac.compare_digest(hello['proof'], proof(session, 'hello', wn, '')):
            raise Refused('session_mismatch')
        cn = secrets.token_hex(32)
        send(connection, dict(role='control', nonce=cn, proof=proof(session, 'control', wn, cn)), cutoff)
        ack = auth_frame(receive(connection, cutoff), 'worker')
        if ack['nonce'] != wn or not hmac.compare_digest(ack['proof'], proof(session, 'worker', wn, cn)):
            raise Refused('session_mismatch')
    else:
        wn = secrets.token_hex(32)
        send(connection, dict(role='worker', nonce=wn, proof=proof(session, 'hello', wn, '')), cutoff)
        hello = auth_frame(receive(connection, cutoff), 'control')
        cn = hello['nonce']
        if not hmac.compare_digest(hello['proof'], proof(session, 'control', wn, cn)):
            raise Refused('session_mismatch')
        send(connection, dict(role='worker', nonce=wn, proof=proof(session, 'worker', wn, cn)), cutoff)
    return evidence
