"""Domain entry points: retained 0C fixture reads and trusted 1A durable core.

Fixture reads remain separate from the authenticated borrowed-owner adapter.
"""
import base64
import hashlib
import logging
from pathlib import Path
from jsonschema.exceptions import ValidationError
from contracts.validate import validate, ContractError
from service.catalogue import (ASSETS, ASSET_FILES, Failure, HTTP_BYTES, IMPLEMENTED,
                               OPERATIONS, capabilities)

_DIAGNOSTIC_SOURCES = (
    'lib/spoken_text.py',
    'contracts/__init__.py',
    'contracts/generate.py',
    'contracts/media.py',
    'contracts/scenarios.py',
    'contracts/validate.py',
    'service/__init__.py',
    'service/admission.py',
    'service/api.py',
    'service/audio_export.py',
    'service/auth.py',
    'service/catalogue.py',
    'service/core.py',
    'service/cpu_setup.py',
    'service/daemon_client.py',
    'service/daemon_helper.py',
    'service/daemon_runtime.py',
    'service/daemon_volumes.py',
    'service/dispatch_store.py',
    'service/dispatcher.py',
    'service/domain.py',
    'service/entrypoint.py',
    'service/execution_context.py',
    'service/host_install.py',
    'service/ingest.py',
    'service/install_child.py',
    'service/install_policy.py',
    'service/installed_export.py',
    'service/installed_pocket_daemon.py',
    'service/installed_qwen_daemon.py',
    'service/job_control.py',
    'service/lifecycle_cli.py',
    'service/native_activation.py',
    'service/native_cpu_activation.py',
    'service/native_cpu_bootstrap.py',
    'service/native_cpu_capacity.py',
    'service/native_cpu_configuration.py',
    'service/native_cpu_context.py',
    'service/native_cpu_install.py',
    'service/native_cpu_measurement.py',
    'service/native_cpu_package.py',
    'service/native_cpu_runtime.py',
    'service/native_cpu_selftest.py',
    'service/native_cpu_setup.py',
    'service/native_launch.py',
    'service/native_package.py',
    'service/native_renewal.py',
    'service/native_runtime.py',
    'service/owner_store.py',
    'service/package_install.py',
    'service/persistence.py',
    'service/persistent_init.py',
    'service/pocket_install.py',
    'service/production.py',
    'service/production_selection.py',
    'service/profiles.py',
    'service/provisioning.py',
    'service/publication_format.py',
    'service/qwen_local_config.py',
    'service/qwen_local_lane.py',
    'service/qwen_service_config.py',
    'service/qwen_service_setup.py',
    'service/qwen_snapshot.py',
    'service/reconciliation.py',
    'service/recovery.py',
    'service/resource_store.py',
    'service/runtime_adapter.py',
    'service/shared_model_client.py',
    'service/shared_model_helper.py',
    'service/shared_models.py',
    'service/supervisor_runtime.py',
    'service/voice_store.py',
    'service/worker_supervisor.py',
    'service/worker_transport.py',
    'worker/__init__.py',
    'worker/audio_export.py',
    'worker/boundary.py',
    'worker/engines/__init__.py',
    'worker/engines/execution.py',
    'worker/engines/kokoro.py',
    'worker/engines/native_kokoro.py',
    'worker/engines/pocket.py',
    'worker/engines/pocket_execution.py',
    'worker/engines/pocket_prepare.py',
    'worker/engines/qwen.py',
    'worker/engines/qwen_execution.py',
    'worker/engines/qwen_prepare.py',
    'worker/hardware.py',
    'worker/isolated_synthetic.py',
    'worker/lifecycle.py',
    'worker/managed_reference.py',
    'worker/native_availability.py',
    'worker/native_budgets.py',
    'worker/native_child.py',
    'worker/native_cpu_availability.py',
    'worker/native_cpu_child.py',
    'worker/native_cpu_protocol.py',
    'worker/native_cpu_qualification.py',
    'worker/native_cpu_resources.py',
    'worker/native_cpu_supervisor.py',
    'worker/native_family.py',
    'worker/native_memory.py',
    'worker/native_protocol.py',
    'worker/native_supervisor.py',
    'worker/qwen_resident.py',
    'worker/runtime_child.py',
    'worker/runtime_conditioning.py',
    'worker/runtime_journal.py',
    'worker/runtime_protocol.py',
    'worker/runtime_supervisor.py',
    'worker/supervised_synthetic.py',
    'worker/synthetic.py',
    'worker/transport.py',
)
_DIAGNOSTIC_PATHS = {str(Path(__file__).parent.parent / label): label
                     for label in _DIAGNOSTIC_SOURCES}


def _refusal_source(exc):
    """Return bounded origin, distinct repository caller, and trusted call chain."""
    frames = []
    trace = exc.__traceback__
    while trace is not None:
        label = _DIAGNOSTIC_PATHS.get(trace.tb_frame.f_code.co_filename)
        line = trace.tb_lineno
        frames.append((label, line if type(line) is int and 0 < line <= 999999 else 0))
        trace = trace.tb_next
    origin = frames[-1] if frames else (None, 0)
    caller = next(((label, line) for label, line in reversed(frames[:-1]) if label),
                  (None, 0))
    # Keep the final repository path, including domain and dispatcher when present.
    trusted = [f'{label}:{line}' for label, line in frames if label]
    chain = trusted if len(trusted) <= 16 else trusted[:2] + trusted[-14:]
    return ((origin[0] or 'unknown'), origin[1] if origin[0] else 0,
            (caller[0] or 'unknown'), caller[1], ','.join(chain) or 'unknown')


def read(operation, payload):
    if operation not in IMPLEMENTED:
        raise Failure('not_found', 'Operation is unavailable in the synthetic evaluation service.', 'operation')
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
        elif operation == 'voice_resolve':
            result = dispatcher.call('voice_resolve', principal, payload['name'])
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
    if metadata['provenance']['kind']=='audio_export':
        from worker.audio_export import MAX_OUTPUT
        MAX_BYTES=MAX_OUTPUT
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
        from lib.spoken_text import freeze
        freeze(plan)
        if operation == 'voice_prepare':
            resolved['reuse_configuration']['reference_text'] = request['reference_text']
        return context


def production_call(dispatcher, principal, operation, payload, key=None):
    """Typed bounded adapter, explicit owner commands only; no Core access here."""
    import queue
    import sqlite3
    from service.catalogue import PRODUCTION_OPERATIONS, PRODUCTION_RESPONSE_BYTES, ARTIFACT_RESPONSE_BYTES, compact
    from service.voice_store import _id
    if operation not in PRODUCTION_OPERATIONS:
        raise Failure('not_found', 'Production operation unavailable.')
    op = OPERATIONS[operation]
    try:
        validate(op['request'], payload)
        if op['idempotency']: _id(key)
        command = {'audio_export':'production_audio_export', 'voice_preset_register':'production_preset_register', 'voice_prepare':'production_prepare', 'speech_generate':'production_generate',
                   'voice_save':'production_save', 'voice_results':'production_results',
                   'job_get':'production_job', 'job_cancel':'production_cancel',
                   'artifact_get':'production_artifact', 'jobs_list':'production_jobs', 'job_events':'production_events',
                   'job_retry':'production_retry', 'voice_aliases':'production_aliases', 'voice_archive':'production_archive', 'profile_get':'production_profile'}[operation]
        result = dispatcher.call(command, principal, payload, key=key)
        response_bytes = ARTIFACT_RESPONSE_BYTES if operation == 'artifact_get' else PRODUCTION_RESPONSE_BYTES
        if len(compact(result).encode('utf-8')) > response_bytes:
            raise Failure('resource_limit', 'Production response exceeds '+str(response_bytes//1024)+' KiB budget.')
        return validate(op['response'], result)
    except ValidationError:
        raise Failure('invalid_request', 'Production contract unavailable.', 'request') from None
    except (ValueError, KeyError, NotImplementedError) as exc:
        raw_code = str(exc)
        # Log only exact, declared refusal tokens. Exception text can contain
        # media, paths or credentials and must never reach diagnostics.
        known_refusals = frozenset(op['errors']) | frozenset((
            'candidate_missing', 'voice_version_missing', 'unpublished_candidate',
            'stale_candidate', 'foreign_media', 'asset_metadata_mismatch',
            'invalid_trim', 'invalid_range', 'invalid_text_rendering',
            'speech_text_mismatch', 'speech_text_language_mismatch', 'profile_limit',
            'settings_mismatch', 'unsupported_parameter', 'resource_input_limit',
            'resource_plan_limit', 'plan_limit', 'queue_full', 'disk_quota',
            'disk_free_reserve', 'resource_runtime_unsupported',
            # Exact refusals from Qwen resolve_owned, local input verification,
            # and admission. These remain internal diagnostic tokens only.
            'hardware_identity', 'execution_reference_mismatch',
            'owned_mapping_required', 'saved_version_required',
            'qwen_execution_required', 'explicit_installed_family_required',
            'owned_bundle_required', 'sealed_bundle_required',
            'bundle_identity', 'bundle_digest', 'bundle_changed',
            'unauthorized_bundle', 'unauthorized_reference',
            'reference_digest', 'reference_size', 'reference_format',
            'canonical_reference_required', 'cpu_clone_required',
            'cpu_xvector_required', 'typed_preparation_required',
            'execution_identity', 'execution_device', 'device_topology',
            'device_unavailable', 'stale_or_unavailable_device',
            'closed_hardware_required', 'hardware_required',
            'installed_cpu_topology', 'installed_cpu_execution_required',
            'preparation_reference_limit', 'explicit_transcript_required',
            'resident_cpu_execution_required', 'resident_model_identity',
            'reference_identity'))
        diagnostic_code = (exc.args[0] if type(exc) is KeyError and len(exc.args) == 1 and
                           type(exc.args[0]) is str else raw_code)
        reason = diagnostic_code if diagnostic_code in known_refusals else 'unclassified'
        source, line, caller, caller_line, chain = _refusal_source(exc)
        exception_type = ('ContractError' if type(exc) is ContractError else
                          'KeyError' if isinstance(exc, KeyError) else
                          'NotImplementedError' if isinstance(exc, NotImplementedError) else 'ValueError')
        logging.getLogger(__name__).warning(
            'production refusal operation=%s reason=%s type=%s source=%s line=%d caller=%s caller_line=%d chain=%s',
            operation, reason, exception_type, source, line, caller, caller_line, chain)
        code = raw_code
        code = {'candidate_missing':'not_found','voice_version_missing':'not_found',
                'unpublished_candidate':'not_ready','stale_candidate':'missing_media',
                'foreign_media':'missing_media','asset_metadata_mismatch':'missing_media',
                'invalid_trim':'invalid_request','invalid_range':'invalid_request',
                'invalid_text_rendering':'invalid_request','speech_text_mismatch':'invalid_request','speech_text_language_mismatch':'unsupported_control','profile_limit':'resource_limit','settings_mismatch':'unsupported_control','unsupported_parameter':'unsupported_control','resource_input_limit':'resource_limit',
                'resource_plan_limit':'resource_limit','plan_limit':'resource_limit',
                'queue_full':'resource_limit','disk_quota':'resource_limit','disk_free_reserve':'resource_limit',
                'resource_runtime_unsupported':'unsupported_control'}.get(code, code)
        if code not in op['errors']: code = 'not_ready'
        detail = 'Pocket reference_text is unsupported; use an empty string.' if str(exc) == 'unsupported_parameter' else 'Production operation unavailable.'
        if code=='publication_format_migration_required':
            detail='This local store requires an explicit publication format migration before new audio exports.'
        from service.production_selection import SelectionUnavailable
        action = ('Operator must refresh/qualify the exact installed profile or correct the override: '
                  + ', '.join(exc.reasons)) if isinstance(exc, SelectionUnavailable) else 'Check IDs, selection and operator configuration; do not automatically retry mutations.'
        if str(exc) == 'disk_quota':
            detail = 'Retained storage and reservations exceed the configured disk quota.'
            action = 'Operator must review retained data and reservations under the retention policy or increase the disk quota; do not automatically retry mutations.'
        elif str(exc) == 'disk_free_reserve':
            detail = 'Available disk space cannot preserve the configured free-space reserve.'
            action = 'Operator must restore free disk space under the retention policy or review the configured free-space reserve; do not automatically retry mutations.'
        elif str(exc)=='native_cpu_qualification_expired':
            detail='The finite native CPU qualification session has expired or cannot fit another bounded job.'
            action='Retained job/artifact reads remain available. Keep the evidence; sessions are not automatically renewed.'
        elif str(exc)=='native_cpu_history_capacity':
            detail='The bounded native CPU retained-history capacity has been reached.'
            action='Retain this installation and its history; run setup into a new directory. The model will be acquired again; no rollover is available.'
        elif str(exc) in ('native_cpu_qualification_job_limit','native_cpu_qualification_queue_budget'):
            detail='The finite native CPU qualification resource or job limit has been reached.'
            action='Keep retained state and review qualification evidence; do not automatically retry or renew the session.'
        raise Failure(code, detail, action=action) from None
    except (OSError, sqlite3.Error, TimeoutError, queue.Full):
        raise Failure('not_ready', 'Owner unavailable; mutation outcome may be unknown.', action='Do not automatically retry mutations.') from None
