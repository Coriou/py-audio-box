"""Domain entry points: retained 0C fixture reads and trusted 1A durable core.

Fixture reads remain separate from the authenticated borrowed-owner adapter.
"""
import base64
import hashlib
from jsonschema.exceptions import ValidationError
from contracts.validate import validate, ContractError
from service.catalogue import (ASSETS, ASSET_FILES, Failure, HTTP_BYTES, IMPLEMENTED,
                               OPERATIONS, capabilities)


def read(operation, payload):
    if operation not in IMPLEMENTED:
        raise Failure('not_found', 'Operation is unavailable in the 0C skeleton.', 'operation')
    op = OPERATIONS[operation]
    try:
        validate(op['request'], payload)
    except (ValidationError, ContractError):
        raise Failure('invalid_request', 'Request does not match the named canonical schema.', 'request') from None
    if operation == 'discover':
        result = capabilities()
    elif operation == 'voices_list':
        if payload.get('cursor'):
            raise Failure('invalid_request', 'No cursor exists for the empty fixture catalogue.', 'cursor')
        result = {'items': [], 'next_cursor': None}
    else:
        asset_id = payload['asset_id']
        if asset_id not in ASSET_FILES:
            raise Failure('not_found', 'Unknown synthetic fixture asset.', 'asset_id')
        asset = ASSETS[asset_id]
        if operation == 'asset_get':
            result = asset
        else:
            start, end = payload['start'], payload['end_exclusive']
            data = ASSET_FILES[asset_id].read_bytes()
            if not 0 <= start < end <= len(data) or end - start > HTTP_BYTES:
                raise Failure('invalid_request', 'Use an explicit nonempty in-file range of at most 16 MiB.', 'range')
            if hashlib.sha256(data).hexdigest() != asset['sha256']:
                raise RuntimeError('Packaged synthetic fixture hash mismatch')
            result = {'schema_version': '1.3.0', 'asset_id': asset_id, 'start': start,
                'end_exclusive': end, 'total_bytes': len(data), 'content_sha256': asset['sha256'],
                'content_base64': base64.b64encode(data[start:end]).decode('ascii')}
    return validate(op['response'], result)


def voice_read(dispatcher, principal, operation, payload):
    """Adapt declared reads through the borrowed owner; never access its Core."""
    import queue
    import sqlite3
    from concurrent.futures import TimeoutError
    from service.catalogue import VOICE_READS, VOICE_RESPONSE_BYTES, compact
    if operation not in VOICE_READS:
        raise Failure('not_found', 'Voice operation unavailable.', 'operation')
    op = OPERATIONS[operation]
    try:
        validate(op['request'], payload)
        if operation == 'voices_list':
            cursor = payload['cursor']
            if cursor and (not cursor.isascii() or not cursor.isdecimal() or len(cursor) > 16):
                raise Failure('invalid_request', 'Use the returned numeric cursor.', 'cursor')
            after = int(cursor or '0')
            if after > 9007199254740991:
                raise Failure('invalid_request', 'Cursor exceeds supported range.', 'cursor')
            rows = dispatcher.call('voice_list', principal, after=after, limit=payload['limit'])
            result = dict(items=[r['voice']['id'] for r in rows],
                          next_cursor=str(rows[-1]['cursor']) if len(rows) == payload['limit'] else None)
        elif operation == 'voice_get':
            result = dispatcher.call('voice_get', principal, payload['voice_id'])
        else:
            result = dispatcher.call('voice_version', principal, payload['voice_version_id'])
        if len(compact(result).encode('utf-8')) > VOICE_RESPONSE_BYTES:
            raise Failure('resource_limit', 'Voice response exceeds the 2048-byte transport budget.',
                          action='Use a smaller list page; oversized details require compact domain reads in a later service version.')
        return validate(op['response'], result)
    except (ValidationError, ContractError):
        raise Failure('invalid_request', 'Request or response does not match the voice contract.', 'request') from None
    except ValueError as exc:
        code = str(exc) if str(exc) in ('not_found', 'invalid_request', 'resource_limit', 'voice_archived') else 'not_ready'
        raise Failure(code, 'Voice read unavailable.', action='Check the requested ID and service readiness.') from None
    except (sqlite3.Error, OSError, TimeoutError, queue.Full):
        raise Failure('not_ready', 'Voice store temporarily unavailable.', action='Retry the read when the owner is available.') from None


def __getattr__(name):
    # Preserve the historical export without loading the owner on discovery.
    if name == 'DurableCore':
        from service.core import Core
        return Core
    raise AttributeError(name)
