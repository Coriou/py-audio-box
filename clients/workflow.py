"""Bounded, replayable client composition of the public production operations."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import wave

from service.catalogue import Failure


def key(phase, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return 'workflow-' + phase + '-' + hashlib.sha256(raw).hexdigest()


def wait(http, job_id, deadline, operation, expected_key, receipt_hash=None):
    while True:
        _remaining(deadline)
        job = http.call('job_get', {'job_id': job_id})
        if (job['id'] != job_id or job['operation'] != operation
                or job['idempotency_key'] != expected_key
                or (receipt_hash is not None and job['input_sha256'] != receipt_hash)):
            raise Failure('invalid_request', 'Job does not match this workflow request.',
                          action='Check the job ID and exact original inputs; no result was used.')
        if job['state'] in ('succeeded', 'failed', 'cancelled'):
            return job
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('workflow_deadline')
        time.sleep(min(1, remaining))


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError('workflow_deadline')
    return remaining


class _PublishedAfterDeadline(TimeoutError):
    def __init__(self, published):
        super().__init__('workflow_deadline')
        self.published = published


def _budget(http, seconds):
    if type(seconds) not in (int, float) or not 0 <= seconds <= 600:
        raise ValueError('wait_seconds must be between 0 and 600')
    deadline = time.monotonic() + seconds
    return http.with_deadline(deadline), deadline


def _outcome(phase, exc, handles, *, mutation=False):
    if isinstance(exc, TimeoutError):
        code, message, state = 'deadline_exceeded', 'Workflow time budget expired.', 'timed_out'
        action = 'Resume with the exact inputs and confirmed handles; do not assume the mutation did not happen.'
    elif isinstance(exc, ConnectionError):
        code, message, state = 'not_ready', str(exc), phase + '_unknown' if mutation else 'read_unavailable'
        action = 'Resume with the exact inputs and confirmed handles. Do not automatically retry an ambiguous mutation.'
    else:
        body = exc.body
        code, message, action = body['code'], body['message'], body.get('action', '')
        unknown = mutation and code == 'not_ready' and 'unknown' in (message + ' ' + action).lower()
        state = phase + '_unknown' if unknown else 'failed'
        if unknown: action = 'Resume with the exact inputs and confirmed handles. Do not automatically retry an ambiguous mutation.'
    if 'asset_id' in handles and handles['asset_id'] and 'prepare_key' in handles:
        action += ' Enrollment resume: pass --asset-id with the same name, reference text and audition text; include --job-id only when confirmed.'
    if 'generation_key' in handles:
        action += ' Speech resume: pass --voice-version-id with the same text; include --job-id only when confirmed.'
    if phase == 'upload' and state == 'upload_unknown' and not handles.get('asset_id'):
        action = 'Upload outcome is unknown and no asset ID was confirmed. Inspect service assets before any new upload; do not automatically repeat the WAV command.'
    if handles.get('saved'):
        action = ('The verified output was saved at the reported path after the deadline. '
                  'Use that file; do not repeat the download to this path. ' + action)
    return dict(handles, state=state, code=code, message=message, action=action, phase=phase)


def _checked_receipt(receipt, job_id=None):
    if job_id is not None and receipt['job_id'] != job_id:
        raise Failure('invalid_request', 'Submission receipt references another job.')
    return receipt['job_id'], receipt['input_sha256']


def _identity(http, operation):
    status = http.get('/v1/status')
    configured = status.get('configured_capabilities') or {}
    if operation not in configured.get('operations', []):
        raise Failure('not_ready', 'Configured service does not offer this voice operation.')
    matches = [c['identity'] for c in configured['profile']['compatibility'] if c['operation'] == operation]
    if len(matches) != 1:
        raise Failure('not_ready', 'Expected one configured voice profile for this operation.')
    return matches[0]


def _wav_ticks(path):
    try:
        with wave.open(str(path), 'rb') as stream:
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
                raise ValueError('Reference must be mono PCM16 WAV.')
            frames, rate = stream.getnframes(), stream.getframerate()
            if rate <= 0 or frames <= 0 or frames * 48000 % rate:
                raise ValueError('Reference duration must resolve to whole 48 kHz ticks.')
            return frames * 48000 // rate
    except wave.Error:
        raise ValueError('Reference must be a complete mono PCM16 WAV.') from None


def enroll(http, *, name, reference_text, audition_text, file=None, asset_id=None,
           job_id=None, seconds=120):
    if not name or not reference_text or not audition_text:
        raise ValueError('Name, exact reference transcript and audition text are required.')
    if job_id and not asset_id:
        raise ValueError('Resume requires the confirmed --asset-id returned by the first run.')
    http, deadline = _budget(http, seconds)
    handles = dict(asset_id=asset_id, job_id=job_id)
    phase = 'preflight'
    try:
        identity = _identity(http, 'voice_prepare')
        if file is not None:
            ticks = _wav_ticks(file)
            if asset_id is None:
                phase = 'upload'
                uploaded = http.upload_file(file)
                asset_id = uploaded['id']; handles['asset_id'] = asset_id
                asset = uploaded
            else:
                phase = 'asset_verify'
                asset = http.call('asset_get', {'asset_id': asset_id})
            if asset['sha256'] != hashlib.sha256(Path(file).read_bytes()).hexdigest():
                raise Failure('invalid_request', 'WAV bytes differ from the confirmed uploaded asset.')
        elif asset_id is None:
            raise ValueError('Supply a WAV or a confirmed uploaded asset ID.')
        if file is None:
            phase = 'asset_verify'
            asset = http.call('asset_get', {'asset_id': asset_id})
            if asset['channels'] != 1 or asset['media_format'] != 'wav' or asset['media_subtype'] != 'PCM_16':
                raise ValueError('Asset must be mono PCM16 WAV.')
            if asset['sample_rate'] <= 0 or asset['frames'] * 48000 % asset['sample_rate']:
                raise ValueError('Asset duration must resolve to whole 48 kHz ticks.')
            ticks = asset['frames'] * 48000 // asset['sample_rate']
        request = dict(asset_id=asset_id, in_tick=0, out_tick=ticks,
                       reference_text=reference_text, audition_text=audition_text, identity=identity)
        prepare_key = key('prepare', dict(name=name, request=request))
        handles['prepare_key'] = prepare_key
        receipt_hash = None
        if job_id is None:
            phase = 'prepare_submit'
            receipt = http.call('voice_prepare', request, key=prepare_key)
            job_id, receipt_hash = _checked_receipt(receipt)
            handles['job_id'] = job_id
        phase = 'job_get'
        job = wait(http, job_id, deadline, 'voice_prepare', prepare_key, receipt_hash)
        if job['state'] != 'succeeded':
            return dict(handles, state=job['state'], code='job_'+job['state'],
                        message='Preparation job ended without success.',
                        action='Inspect this job and its events; do not save a candidate.')
        phase = 'voice_results'
        candidates = http.call('voice_results', {'job_id': job_id})['candidates']
        if len(candidates) != 1:
            raise Failure('not_ready', 'Preparation returned an ambiguous candidate set.')
        phase = 'voice_resolve'
        try:
            voice = http.call('voice_resolve', {'name': name})
            revision, voice_id = voice['revision'], voice['id']
        except Failure as exc:
            if exc.body['code'] != 'not_found': raise
            revision, voice_id = 0, name
        from contracts.validate import canonical_hash
        saved_id = 'saved-' + canonical_hash([job_id, candidates[0]['id'], voice_id])
        handles.update(candidate_id=candidates[0]['id'], voice_version_id=saved_id)
        phase = 'voice_version_get'
        try:
            previous = http.call('voice_version_get', {'voice_version_id': saved_id})
            if previous['voice_id'] != voice_id or previous['reference_text'] != reference_text:
                raise Failure('invalid_request', 'Saved version does not match this enrollment.')
            return dict(handles, voice_id=voice_id, state='succeeded')
        except Failure as exc:
            if exc.body['code'] != 'not_found': raise
        save = dict(job_id=job_id, candidate_id=candidates[0]['id'],
                    voice_id=voice_id, expected_revision=revision)
        save_key = key('save', save)
        handles['save_key'] = save_key
        phase = 'voice_save'
        saved = http.call('voice_save', save, key=save_key)
        if saved['version']['id'] != saved_id or saved['version']['voice_id'] != voice_id:
            raise Failure('invalid_request', 'Save response does not match this enrollment.')
        return dict(handles, voice_id=voice_id, state='succeeded')
    except (ConnectionError, TimeoutError, Failure) as exc:
        return _outcome(phase, exc, handles, mutation=phase in ('upload','prepare_submit','voice_save'))


def download(http, asset_id, output, deadline):
    _remaining(deadline)
    asset = http.call('asset_get', {'asset_id': asset_id})
    first = http.call('asset_content', {'asset_id': asset_id, 'start': 0, 'end_exclusive': 1})
    size = first['total_bytes']
    if not 0 < size <= 24 * 1024 * 1024:
        raise ValueError('Output exceeds client transfer limit.')
    output = Path(output)
    temporary = None
    published = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.py-audio-box-', delete=False) as stream:
            temporary = Path(stream.name)
            for start in range(0, size, 65536):
                _remaining(deadline)
                end = min(start + 65536, size)
                chunk = http.call('asset_content', {'asset_id': asset_id, 'start': start, 'end_exclusive': end})
                data = base64.b64decode(chunk['content_base64'], validate=True)
                if (chunk['start'], chunk['end_exclusive'], chunk['total_bytes'], chunk['content_sha256']) != (start, end, size, asset['sha256']) or len(data) != end-start:
                    raise ValueError('Unexpected transfer metadata.')
                stream.write(data); digest.update(data)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('Downloaded hash mismatch.')
        _remaining(deadline)
        os.link(temporary, output)
        published = dict(asset_id=asset_id, bytes=size, sha256=digest.hexdigest(),
                         output=str(output), saved=True)
        try:
            _remaining(deadline)
        except TimeoutError as exc:
            raise _PublishedAfterDeadline(published) from exc
        return published
    finally:
        if temporary is not None:
            primary = sys.exception()
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                warning = f'Could not confirm removal of temporary download file {temporary}: {exc}'
                if published is not None:
                    published['cleanup_warning'] = warning
                    published['temporary_path'] = str(temporary)
                elif primary is not None:
                    primary.add_note(warning)
                    if isinstance(primary, (ConnectionError, TimeoutError, Failure)):
                        primary.cleanup_warning = warning
                        primary.temporary_path = str(temporary)
                else:
                    raise


def speak(http, *, voice, text, output=None, seconds=120, job_id=None, voice_version_id=None):
    if not voice or not text: raise ValueError('Voice and text are required.')
    if job_id and not voice_version_id:
        raise ValueError('Resume requires --voice-version-id returned by the first run.')
    http, deadline = _budget(http, seconds)
    handles = dict(job_id=job_id, voice_version_id=voice_version_id)
    phase = 'voice_resolve'
    try:
        if voice_version_id is None:
            try:
                selected = http.call('voice_resolve', {'name': voice})
            except Failure as exc:
                if exc.body['code'] != 'not_found': raise
                phase = 'preset_status'
                status = http.get('/v1/status')
                presets = [p for p in status.get('configured_presets', []) if p['preset'].casefold() == voice.casefold()]
                if len(presets) != 1: raise
                register = dict(preset=presets[0]['preset'], voice_id=voice.casefold(), expected_revision=0)
                handles['preset_key'] = key('preset', register)
                phase = 'preset_register'
                http.call('voice_preset_register', register, key=handles['preset_key'])
                phase = 'voice_resolve'
                selected = http.call('voice_resolve', {'name': voice.casefold()})
            if not selected['version_ids']: raise Failure('not_found', 'Voice has no saved version.')
            voice_version_id = selected['version_ids'][-1]
            handles['voice_version_id'] = voice_version_id
        phase = 'voice_version_get'
        version = http.call('voice_version_get', {'voice_version_id': voice_version_id})
        if version['id'] != voice_version_id:
            raise Failure('invalid_request', 'Voice version response does not match selected version.')
        request = dict(voice_version_id=version['id'], text=text, identity=version['identity'],
                       fallback='forbid', variants=1, reuse='never')
        generation_key = key('speak', request)
        handles['generation_key'] = generation_key
        receipt_hash = None
        if job_id is None:
            phase = 'generation_submit'
            receipt = http.call('speech_generate', request, key=generation_key)
            job_id, receipt_hash = _checked_receipt(receipt)
            handles['job_id'] = job_id
        phase = 'job_get'
        job = wait(http, job_id, deadline, 'speech_generate', generation_key, receipt_hash)
        if job['state'] != 'succeeded':
            return dict(handles, state=job['state'], code='job_'+job['state'],
                        message='Generation job ended without success.', action='Inspect this job and its events.')
        if len(job['artifact_ids']) != 1:
            raise Failure('not_ready', 'Generation has no single published artifact.')
        phase = 'artifact_get'
        artifact = http.call('artifact_get', {'artifact_id': job['artifact_ids'][0]})
        if artifact['job_id'] != job_id or artifact['input_sha256'] != job['input_sha256'] or len(artifact['asset_ids']) != 1:
            raise Failure('invalid_request', 'Published artifact does not match generation job.')
        handles.update(artifact_id=artifact['id'], asset_id=artifact['asset_ids'][0])
        if output is not None:
            phase = 'download'
            handles.update(download(http, artifact['asset_ids'][0], output, deadline))
        return dict(handles, state='succeeded')
    except (ConnectionError, TimeoutError, Failure) as exc:
        if isinstance(exc, _PublishedAfterDeadline):
            handles.update(exc.published)
        if hasattr(exc, 'cleanup_warning'):
            handles.update(cleanup_warning=exc.cleanup_warning,
                           temporary_path=exc.temporary_path)
        return _outcome(phase, exc, handles, mutation=phase in ('preset_register','generation_submit'))
