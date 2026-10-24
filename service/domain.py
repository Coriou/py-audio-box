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
        if operation in ('voices_list', 'voice_versions_list'):
            cursor = payload['cursor']
            if cursor and (not cursor.isascii() or not cursor.isdecimal() or len(cursor) > 16):
                raise Failure('invalid_request', 'Use the returned numeric cursor.', 'cursor')
            after = int(cursor or '0')
            if after > 9007199254740991:
                raise Failure('invalid_request', 'Cursor exceeds supported range.', 'cursor')
            rows = (dispatcher.call('voice_summaries', principal, after=after, limit=payload['limit'])
                    if operation == 'voices_list' else dispatcher.call('voice_version_summaries', principal, payload['voice_id'], after=after, limit=payload['limit']))
            result = dict(items=[(r['voice']['id'] if operation == 'voices_list' else r['id']) for r in rows],
                          next_cursor=str(rows[-1]['cursor']) if len(rows) == payload['limit'] else None)
        elif operation == 'voice_get':
            result = dispatcher.call('voice_get', principal, payload['voice_id'])
        else:
            result = dispatcher.call('voice_version', principal, payload['voice_version_id'])
        if len(compact(result).encode('utf-8')) > VOICE_RESPONSE_BYTES:
            raise Failure('resource_limit', 'Voice response exceeds the 2048-byte transport budget.',
                          action='Use a smaller list page; oversized details require compact domain reads in a later service version.')
        return validate(op['response'], result)
    except ValidationError:
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


def owned_asset(core, principal, asset_id, *, content=False, start=0, end_exclusive=0):
    """Owner-only managed upload reads. No fixture or legacy ownership fallback.

    Hash the same no-follow descriptor used for the bounded range; close before
    returning detached bytes. No descriptor survives into the network response.
    """
    import os
    import stat
    from service.ingest import MAX_BYTES, open_directory
    import json
    row = core.db.execute('SELECT metadata FROM assets WHERE id=?', (asset_id,)).fetchone()
    if row is None: raise ValueError('not_found')
    owner = json.loads(row[0]).get('principal_id')
    if owner is not None and owner != principal: raise ValueError('not_found')
    try:
        # Published audition/generation assets inherit the producing job owner;
        # uploads carry explicit managed ownership. Use the shared owner oracle.
        core.voices._asset(principal, asset_id)
    except ValueError as exc:
        if str(exc) == 'forbidden': raise ValueError('not_found') from None
        raise
    metadata = core.asset(asset_id)
    if not content:
        return metadata
    if not 0 <= start < end_exclusive or end_exclusive-start > 65536:
        raise ValueError('invalid_request')
    directory = open_directory(core.root/'assets')
    try:
        fd = os.open(metadata['id'], os.O_RDONLY|os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1 or before.st_size > MAX_BYTES:
            raise ValueError('missing_media')
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = stream.read(min(65536, MAX_BYTES+1-total))
            if not chunk: break
            total += len(chunk)
            if total > MAX_BYTES: raise ValueError('missing_media')
            digest.update(chunk)
        if digest.hexdigest() != metadata['sha256'] or end_exclusive > total:
            raise ValueError('missing_media')
        stream.seek(start)
        data = stream.read(end_exclusive-start)
        after = os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or len(data) != end_exclusive-start:
            raise ValueError('missing_media')
    return dict(schema_version='1.3.0', asset_id=asset_id, start=start,
                end_exclusive=end_exclusive, total_bytes=total,
                content_sha256=metadata['sha256'], content_base64=base64.b64encode(data).decode('ascii'))


def asset_call(dispatcher, principal, operation, payload):
    import queue
    import sqlite3
    op = OPERATIONS[operation]
    try:
        if operation == 'asset_upload':
            if type(payload) is not bytes: raise ValueError('invalid_request')
            result = dispatcher.call('asset_ingest', principal, payload)
        else:
            validate(op['request'], payload)
            result = dispatcher.call('asset_metadata' if operation == 'asset_get' else 'asset_content', principal, **payload)
        from service.catalogue import compact
        if operation != 'asset_content' and len(compact(result).encode('utf-8')) > 2048:
            raise Failure('resource_limit', 'Asset metadata exceeds compact response budget.')
        return validate(op['response'], result)
    except ValidationError:
        raise Failure('invalid_request', 'Asset contract unavailable.') from None
    except ValueError as exc:
        code = {'invalid_audio':'invalid_media', 'foreign_media':'missing_media'}.get(str(exc), str(exc))
        if code not in ('invalid_request','invalid_media','resource_limit','not_found','missing_media'): code='not_ready'
        raise Failure(code, 'Asset operation unavailable.', action='Check media and service readiness; uploads are never automatically retried.') from None
    except (OSError, sqlite3.Error, TimeoutError, queue.Full):
        raise Failure('not_ready', 'Asset owner unavailable; upload outcome may be unknown.', action='Do not automatically retry uploads.') from None


class SyntheticContexts:
    """Operator configuration only: detached immutable snapshots keyed by principal/op.

    This provider advertises no engine readiness. Runtime profile admission remains
    independent; synthetic test qualification must be explicitly supplied by owner.
    """
    def __init__(self, contexts):
        import json
        from types import MappingProxyType
        frozen = {}
        for (principal, operation), context in contexts.items():
            if operation not in ('voice_prepare', 'speech_generate'):
                raise ValueError('unsupported_control')
            snapshot = json.loads(json.dumps(context, allow_nan=False))
            plan = snapshot['plan']
            if plan['operation'] != operation or plan['resolved']['profile']['id'] != 'cpu-synthetic':
                raise ValueError('unsupported_control')
            identity = plan['resolved']['identity']['effective']
            if (identity['engine'], identity['model'], identity['device'], identity['precision']) != ('qwen', 'Qwen3-TTS-12Hz-0.6B-Base', 'cpu', 'float32'):
                raise ValueError('unsupported_control')
            if plan['resolved']['settings']['kind'] != ('prepare' if operation == 'voice_prepare' else 'qwen_clone'):
                raise ValueError('unsupported_control')
            frozen[(principal, operation)] = json.dumps(snapshot, allow_nan=False).encode()
        self._contexts = MappingProxyType(frozen)

    def resolve(self, principal, operation, request):
        import json
        raw = self._contexts.get((principal, operation))
        if raw is None: raise ValueError('not_ready')
        context = json.loads(raw)
        plan = context['plan']; resolved = plan['resolved']
        if request['identity'] != resolved['identity']['requested']:
            raise ValueError('unsupported_control')
        if operation == 'speech_generate' and (request['fallback'] != 'forbid' or request['variants'] != 1 or request['reuse'] != 'never'):
            raise ValueError('unsupported_control')
        plan['request'] = json.loads(json.dumps(request))
        if operation == 'voice_prepare':
            resolved['reuse_configuration']['reference_text'] = request['reference_text']
        return context


def production_call(dispatcher, principal, operation, payload, key=None):
    """Typed bounded adapter, explicit owner commands only; no Core access here."""
    import queue
    import sqlite3
    from service.catalogue import PRODUCTION_OPERATIONS, compact
    from service.voice_store import _id
    if operation not in PRODUCTION_OPERATIONS:
        raise Failure('not_found', 'Production operation unavailable.')
    op = OPERATIONS[operation]
    try:
        validate(op['request'], payload)
        if op['idempotency']: _id(key)
        command = {'voice_prepare':'production_prepare', 'speech_generate':'production_generate',
                   'voice_save':'production_save', 'voice_results':'production_results',
                   'job_get':'production_job', 'job_cancel':'production_cancel',
                   'artifact_get':'production_artifact'}[operation]
        result = dispatcher.call(command, principal, payload, key=key)
        if len(compact(result).encode()) > 12288:
            raise Failure('resource_limit', 'Production response exceeds 12 KiB budget.')
        return validate(op['response'], result)
    except ValidationError:
        raise Failure('invalid_request', 'Production contract unavailable.', 'request') from None
    except (ValueError, KeyError, NotImplementedError) as exc:
        code = str(exc)
        code = {'candidate_missing':'not_found','voice_version_missing':'not_found',
                'unpublished_candidate':'not_ready','stale_candidate':'missing_media',
                'foreign_media':'missing_media','asset_metadata_mismatch':'missing_media',
                'invalid_trim':'invalid_request','invalid_range':'invalid_request',
                'settings_mismatch':'unsupported_control','unsupported_parameter':'unsupported_control','resource_input_limit':'resource_limit',
                'resource_plan_limit':'resource_limit','plan_limit':'resource_limit',
                'queue_full':'resource_limit','resource_runtime_unsupported':'unsupported_control'}.get(code, code)
        if code not in op['errors']: code = 'not_ready'
        detail = 'Pocket reference_text is unsupported; use an empty string.' if str(exc) == 'unsupported_parameter' else 'Production operation unavailable.'
        raise Failure(code, detail, action='Check IDs, selection and operator configuration; do not automatically retry mutations.') from None
    except (OSError, sqlite3.Error, TimeoutError, queue.Full):
        raise Failure('not_ready', 'Owner unavailable; mutation outcome may be unknown.', action='Do not automatically retry mutations.') from None
