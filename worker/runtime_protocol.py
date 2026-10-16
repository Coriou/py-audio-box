"""Closed private lifecycle-v1. Canonical execution/result remains 1.3.0."""
import hashlib
import hmac
from worker.transport import Session, Refused, authenticate, receive, send

VERSION = 'lifecycle-v1'
OPS = frozenset(('hello', 'prepare', 'start', 'observe', 'stop', 'collect', 'ack'))


def session(secret, uid, gid):
    key = hmac.new(bytes.fromhex(secret), b'private-lifecycle-v1/auth', hashlib.sha256).hexdigest()
    return Session(key, hashlib.sha256(b'private-lifecycle-v1').hexdigest(), uid, gid)


def request(value):
    if (not isinstance(value, dict) or set(value) != {'version', 'op', 'body'} or
            value['version'] != VERSION or value['op'] not in OPS or not isinstance(value['body'], dict)):
        raise Refused('closed_lifecycle_frame_required')
    return value['op'], value['body']


def execution_session(secret, binding, uid, gid):
    key = hmac.new(bytes.fromhex(secret), ('execution-allocation-v1/'+binding).encode(), hashlib.sha256).hexdigest()
    return Session(key, binding, uid, gid)
