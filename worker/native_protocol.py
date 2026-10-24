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


ERRORS = frozenset(('native_refused', 'native_unknown', 'native_busy'))


def response(value, *, request):
    """Closed responses bind every reply to the originating session and nonce."""
    if type(value) is not dict: raise Refused('native_response')
    keys = ('version','deployment_id','configuration_sha256','lane','boot',
            'request_id','runtime_id','binding','op')
    _closed(value, (*keys,'ok','value') if value.get('ok') is True else (*keys,'ok','error'))
    if type(value['ok']) is not bool or any(value[k] != request[k] for k in keys):
        raise Refused('native_response_binding')
    if not value['ok']:
        if type(value['error']) is not str or value['error'] not in ERRORS: raise Refused('native_response')
        return value
    v, op = value['value'], request['op']
    if op in ('start','ack','put_input'): _closed(v, ())
    elif op=='stop':
        # Empty means cancellation accepted (or prepared cancellation completed).
        # A frozen decision cannot acknowledge a new cancellation.
        if v == {}: _closed(v, ())
        else:
            _closed(v, ('state','timely'))
            if v['state'] not in ('committing','terminal') or type(v['timely']) is not bool: raise Refused('native_response')
    elif op in ('prepare','observe'):
        _closed(v, ('runtime_id','state'))
        if type(v['runtime_id']) is not str or v['state'] not in ('alive','stopped-and-reaped','unknown'): raise Refused('native_response')
        if op=='prepare': runtime_id(v['runtime_id'],request['lane'],request['boot'])
        elif v['runtime_id'] != request['runtime_id']: raise Refused('native_response')
    elif op=='hello':
        _closed(v, ('deployment_id','configuration_sha256','lane','boot','quarantined','launches','runtime_manifest_sha256','inventory'))
        if (v['deployment_id'],v['configuration_sha256']) != (request['deployment_id'],request['configuration_sha256']): raise Refused('native_response')
        for k in ('lane','boot'): _hex(v[k],32)
        _hex(v['runtime_manifest_sha256'],64); _int(v['launches'],0,64)
        if type(v['quarantined']) is not bool: raise Refused('native_response')
        from worker.engines.qwen_execution import inventory_from_wire
        inventory_from_wire(v['inventory'])
    elif op=='get_output':
        _closed(v, ('sha256','offset','data'))
        if (v['sha256'],v['offset']) != (request['body']['sha256'],request['body']['offset']) or len(chunk(v['data'])) != request['body']['bytes']: raise Refused('native_response')
    elif op=='collect':
        _closed(v, ('receipt','proposal','output'))
        if v['receipt'] is None:
            if v['proposal'] is not None or v['output'] is not None: raise Refused('native_response')
        else:
            expected={k:request[k] for k in ('runtime_id','binding','deployment_id','configuration_sha256','lane','boot')}
            expected['original_deadline']=v['receipt']['original_deadline']
            receipt(v['receipt'],expected=expected)
            if v['proposal'] is not None:
                _closed(v['proposal'],('message','manifest'))
                from contracts import validate as oracle
                try:
                    oracle.validate('worker_message',v['proposal']['message'])
                    oracle.validate('result_manifest',v['proposal']['manifest'])
                except Exception: raise Refused('native_response') from None
                message,manifest=v['proposal']['message'],v['proposal']['manifest']
                if message['kind']!='result' or any(message[k]!=manifest[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')) or len(manifest['outputs'])!=1 or manifest['candidates']: raise Refused('native_response')
            if v['output'] is not None:
                _closed(v['output'],('sha256','bytes')); _hex(v['output']['sha256'],64); _int(v['output']['bytes'],46,1048576)
    return value


def native_execution(context):
    _closed(context, ('plan','bundles','execution'))
    from worker.runtime_protocol import qwen_relation
    return qwen_relation(context, lane='native')


def generated_source(context):
    spec=native_execution(context)
    return spec.engine+'-execution-sha256:'+hashlib.sha256(canonical(spec.identity())).hexdigest()


def frozen_native(context):
    """Select only from the frozen profile, never from received evidence."""
    return context['plan']['resolved']['profile']['id'] == 'qwen-mps'


def native_binding(context):
    native_execution(context)
    value = context['execution']['native']
    _closed(value, ('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
    for key in ('deployment_id','lane','boot'): _hex(value[key],32)
    for key in ('configuration_sha256','runtime_manifest_sha256'): _hex(value[key],64)
    return value


def terminal_receipt(value, *, authority, runtime_id, local_import_uid=None):
    """One pure live/history validator. Local UID is trusted configuration only."""
    if frozen_native(authority.context):
        config = native_binding(authority.context)
        expected = {k:config[k] for k in ('deployment_id','configuration_sha256','lane','boot')}
        expected.update(runtime_id=runtime_id,binding=authority.binding,original_deadline=authority.job['deadline'])
        receipt(value, expected=expected)
        if local_import_uid is not None: _int(local_import_uid,0,2**31-1)
        return local_import_uid
    _closed(value, ('runtime_id','binding','reaped','timely','original_deadline','engine_uid','reap_method','reaped_children'))
    if (value['runtime_id'] != runtime_id or value['binding'] != authority.binding or
        value['original_deadline'] != authority.job['deadline'] or value['reaped'] is not True or
        value['reap_method'] != 'pid1-waitpid-echild' or type(value['timely']) is not bool or
        type(value['reaped_children']) is not int or value['reaped_children'] < 0 or
        type(value['engine_uid']) is not int): raise Refused('unbound_receipt')
    return value['engine_uid']
