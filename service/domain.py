"""Domain entry points: retained 0C fixture reads and trusted 1A durable core.

HTTP remains explicitly fixture-only until transport lot 1B.
"""
from service.core import Core as DurableCore
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
