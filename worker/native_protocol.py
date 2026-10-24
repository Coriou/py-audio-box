"""Private native-qwen-v1 codecs. No Linux peer/authentication substitutions.

These pure checks do not enroll a server or establish OS process death proof.
"""
import base64
import binascii
from datetime import datetime
import hashlib
import hmac
import ipaddress
import json
import re
import ssl
import struct

from worker.transport import Refused, canonical, exact, remaining

VERSION = 'native-qwen-v1'
FRAME_LIMIT = 65536
CHUNK_LIMIT = 32768
OPS = frozenset(('hello', 'prepare', 'put_input', 'start', 'observe', 'stop',
                 'collect', 'get_output', 'ack'))


def _closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise Refused('native_shape')


def _hex(value, size):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{%d}' % size, value):
        raise Refused('native_identity')


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise Refused('native_limit')


def runtime_id(value, lane, boot):
    if type(value) is not str or not value.startswith(lane + '/' + boot + '/'):
        raise Refused('native_runtime')
    _hex(value[len(lane) + len(boot) + 2:], 32)


def chunk(value):
    if type(value) is not str or not 4 <= len(value) <= 43692:
        raise Refused('native_chunk')
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise Refused('native_chunk') from None
    if not 1 <= len(data) <= CHUNK_LIMIT or base64.b64encode(data).decode() != value:
        raise Refused('native_chunk')
    return data


def request(value):
    _closed(value, ('version', 'deployment_id', 'configuration_sha256', 'lane',
                    'boot', 'request_id', 'runtime_id', 'binding', 'op', 'body'))
    if value['version'] != VERSION or type(value['op']) is not str or value['op'] not in OPS:
        raise Refused('native_protocol')
    for key in ('deployment_id', 'request_id'):
        _hex(value[key], 32)
    _hex(value['configuration_sha256'], 64)
    op, body = value['op'], value['body']
    if op == 'hello':
        if any(value[k] is not None for k in ('lane', 'boot', 'runtime_id', 'binding')):
            raise Refused('native_binding')
    else:
        for key in ('lane', 'boot'):
            _hex(value[key], 32)
        _hex(value['binding'], 64)
        if op == 'prepare':
            if value['runtime_id'] is not None:
                raise Refused('native_runtime')
        else:
            runtime_id(value['runtime_id'], value['lane'], value['boot'])
    if op == 'prepare':
        _closed(body, ('snapshot',))
        if type(body['snapshot']) is not list or len(body['snapshot']) != 4:
            raise Refused('native_snapshot')
        try:
            raw = canonical(body['snapshot'])
        except (TypeError, ValueError, RecursionError):
            raise Refused('native_snapshot') from None
        if len(raw) > 48000:
            raise Refused('native_snapshot')
    elif op == 'put_input':
        _closed(body, ('sha256', 'bytes', 'offset', 'data'))
        _hex(body['sha256'], 64)
        _int(body['bytes'], 1, 24000000)
        _int(body['offset'], 0, body['bytes'] - 1)
        if body['offset'] + len(chunk(body['data'])) > body['bytes']:
            raise Refused('native_chunk')
    elif op == 'get_output':
        _closed(body, ('sha256', 'offset', 'bytes'))
        _hex(body['sha256'], 64)
        _int(body['offset'], 0, 1048575)
        _int(body['bytes'], 1, CHUNK_LIMIT)
        if body['offset'] + body['bytes'] > 1048576:
            raise Refused('native_limit')
    else:
        _closed(body, ())
    return value


def receipt(value, *, expected):
    """Expected tuple comes from frozen authority plus trusted native config.

    Callers must select this validator from the frozen lane, never the receipt.
    It deliberately returns no engine_uid for a local control file.
    """
    _closed(value, ('version', 'runtime_id', 'binding', 'deployment_id',
        'configuration_sha256', 'lane', 'boot', 'reaped', 'timely',
        'original_deadline', 'reap_method', 'processes', 'receiver_closed'))
    binding_keys = ('runtime_id', 'binding', 'deployment_id',
                    'configuration_sha256', 'lane', 'boot', 'original_deadline')
    _closed(expected, binding_keys)
    if any(value[k] != expected[k] for k in binding_keys):
        raise Refused('native_receipt_binding')
    if value['version'] != 'native-receipt-v1' or value['reap_method'] != 'darwin-owned-exec-v1':
        raise Refused('native_receipt')
    for key in ('deployment_id', 'lane', 'boot'):
        _hex(value[key], 32)
    for key in ('configuration_sha256', 'binding'):
        _hex(value[key], 64)
    runtime_id(value['runtime_id'], value['lane'], value['boot'])
    try:
        deadline = value['original_deadline']
        if type(deadline) is not str or len(deadline) > 128 or datetime.fromisoformat(deadline).tzinfo is None:
            raise ValueError()
    except (ValueError, TypeError):
        raise Refused('native_deadline') from None
    if (value['reaped'] is not True or value['receiver_closed'] is not True
            or type(value['timely']) is not bool):
        raise Refused('native_receipt')
    processes = value['processes']
    if type(processes) is not list or len(processes) > 1:
        raise Refused('native_processes')
    for process in processes:
        _closed(process, ('role', 'spawn_token', 'pid', 'pgid', 'start_identity', 'exit_code'))
        if process['role'] != 'model':
            raise Refused('native_processes')
        _hex(process['spawn_token'], 32)
        for key in ('pid', 'pgid'):
            _int(process[key], 1, 2**31 - 1)
        _int(process['exit_code'], -2**31, 2**31 - 1)
        identity = process['start_identity']
        if identity is not None and (type(identity) is not str or not 0 < len(identity) <= 256):
            raise Refused('native_processes')
    if value['timely'] and (not processes or processes[0]['exit_code'] != 0):
        raise Refused('native_receipt')
    return value


def numeric_bind(host, port, *, proven_host_network=False):
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise Refused('native_bind') from None
    _int(port, 1, 65535)
    if address.is_unspecified or address.is_multicast or address.is_loopback and proven_host_network is not True:
        raise Refused('native_bind')
    return str(address), port


def tls_context(*, server, cafile, certfile, keyfile):
    """Trust-file ownership/no-follow validation belongs to future enrollment."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=cafile)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if hasattr(ssl, 'OP_NO_RENEGOTIATION'):
        context.options |= ssl.OP_NO_RENEGOTIATION
    return context


def pinned(connection, peer_sha256):
    _hex(peer_sha256, 64)
    if connection.version() != 'TLSv1.3':
        raise Refused('native_tls')
    certificate = connection.getpeercert(binary_form=True)
    if not certificate or not hmac.compare_digest(hashlib.sha256(certificate).hexdigest(), peer_sha256):
        raise Refused('native_pin')


def send(connection, value, cutoff, *, peer_sha256):
    pinned(connection, peer_sha256)
    try:
        data = canonical(value)
    except (ValueError, TypeError, RecursionError):
        raise Refused('native_frame') from None
    if not 0 < len(data) <= FRAME_LIMIT:
        raise Refused('frame_limit')
    remaining(connection, cutoff)
    connection.sendall(struct.pack('!I', len(data)) + data)


def receive(connection, cutoff, *, peer_sha256):
    pinned(connection, peer_sha256)
    size, = struct.unpack('!I', exact(connection, 4, cutoff))
    if not 0 < size <= FRAME_LIMIT:
        raise Refused('frame_limit')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refused('duplicate_key')
            result[key] = value
        return result
    def nonfinite(_):
        raise Refused('nonfinite')
    raw = exact(connection, size, cutoff)
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        if canonical(value) != raw:
            raise Refused('native_canonical')
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise Refused('native_frame') from None
