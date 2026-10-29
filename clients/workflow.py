"""Bounded, replayable client composition of the public production operations."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import wave

from service.catalogue import Failure


def key(phase, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return 'workflow-' + phase + '-' + hashlib.sha256(raw).hexdigest()


def wait(http, job_id, seconds):
    if not 0 <= seconds <= 600:
        raise ValueError('wait_seconds must be between 0 and 600')
    end = time.monotonic() + seconds
    while True:
        job = http.call('job_get', {'job_id': job_id})
        if job['state'] in ('succeeded', 'failed', 'cancelled'):
            return job
        remaining = end - time.monotonic()
        if remaining <= 0:
            return {'job_id': job_id, 'state': job['state'], 'timed_out': True,
                    'action': 'Read job get JOB_ID or repeat this command with the same inputs; the same idempotency key is reused. No cancellation was requested.'}
        time.sleep(min(1, remaining))


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
    identity = _identity(http, 'voice_prepare')
    if file is not None:
        ticks = _wav_ticks(file)
        if asset_id is None:
            asset_id = http.upload_file(file)['id']
    elif asset_id is None:
        raise ValueError('Supply a WAV or an already uploaded asset ID.')
    if file is None:
        asset = http.call('asset_get', {'asset_id': asset_id})
        if asset['channels'] != 1 or asset['media_format'] != 'wav' or asset['media_subtype'] != 'PCM_16':
            raise ValueError('Asset must be mono PCM16 WAV.')
        if asset['sample_rate'] <= 0 or asset['frames'] * 48000 % asset['sample_rate']:
            raise ValueError('Asset duration must resolve to whole 48 kHz ticks.')
        ticks = asset['frames'] * 48000 // asset['sample_rate']
    request = dict(asset_id=asset_id, in_tick=0, out_tick=ticks,
                   reference_text=reference_text, audition_text=audition_text, identity=identity)
    prepare_key = key('prepare', dict(name=name, request=request))
    if job_id is None:
        try:
            receipt = http.call('voice_prepare', request, key=prepare_key)
        except ConnectionError:
            return dict(asset_id=asset_id, prepare_key=prepare_key, state='submission_unknown',
                        action='Repeat with the same inputs and --asset-id; the same idempotency key will be reused.')
        job_id = receipt['job_id']
    job = wait(http, job_id, seconds)
    if job.get('timed_out') or job['state'] != 'succeeded':
        return dict(job_id=job_id, prepare_key=prepare_key, asset_id=asset_id,
                    state=job['state'], timed_out=job.get('timed_out', False))
    candidates = http.call('voice_results', {'job_id': job_id})['candidates']
    if len(candidates) != 1:
        raise Failure('not_ready', 'Preparation returned an ambiguous candidate set.')
    try:
        voice = http.call('voice_resolve', {'name': name})
        revision = voice['revision']
        voice_id = voice['id']
    except Failure as exc:
        if exc.body['code'] != 'not_found': raise
        revision, voice_id = 0, name
    # The service derives this stable version ID from the prepared candidate and
    # target voice. A lost save response can therefore be recovered by a read.
    from contracts.validate import canonical_hash
    saved_id = 'saved-' + canonical_hash([job_id, candidates[0]['id'], voice_id])
    try:
        previous = http.call('voice_version_get', {'voice_version_id': saved_id})
        if previous['voice_id'] != voice_id:
            raise Failure('not_ready', 'Saved version target does not match enrollment.')
        return dict(voice_id=voice_id, voice_version_id=saved_id, job_id=job_id,
                    asset_id=asset_id, state='succeeded')
    except Failure as exc:
        if exc.body['code'] != 'not_found': raise
    save = dict(job_id=job_id, candidate_id=candidates[0]['id'],
                voice_id=voice_id, expected_revision=revision)
    save_key = key('save', save)
    try:
        saved = http.call('voice_save', save, key=save_key)
    except ConnectionError:
        return dict(job_id=job_id, asset_id=asset_id, candidate_id=candidates[0]['id'],
                    save_key=save_key, state='save_unknown',
                    action='Repeat with the same name, asset, transcript and audition; the saved version is checked before another save.')
    return dict(voice_id=saved['voice']['id'], voice_version_id=saved['version']['id'],
                job_id=job_id, asset_id=asset_id, state='succeeded')


def download(http, asset_id, output):
    asset = http.call('asset_get', {'asset_id': asset_id})
    first = http.call('asset_content', {'asset_id': asset_id, 'start': 0, 'end_exclusive': 1})
    size = first['total_bytes']
    if not 0 < size <= 24 * 1024 * 1024:
        raise ValueError('Output exceeds client transfer limit.')
    output = Path(output)
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.py-audio-box-', delete=False) as stream:
            temporary = Path(stream.name)
            for start in range(0, size, 65536):
                end = min(start + 65536, size)
                chunk = http.call('asset_content', {'asset_id': asset_id, 'start': start, 'end_exclusive': end})
                data = base64.b64decode(chunk['content_base64'], validate=True)
                if (chunk['start'], chunk['end_exclusive'], chunk['total_bytes'], chunk['content_sha256']) != (start, end, size, asset['sha256']) or len(data) != end-start:
                    raise ValueError('Unexpected transfer metadata.')
                stream.write(data); digest.update(data)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('Downloaded hash mismatch.')
        os.link(temporary, output)
        return dict(asset_id=asset_id, bytes=size, sha256=digest.hexdigest(), saved=True)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def speak(http, *, voice, text, output=None, seconds=120, job_id=None, voice_version_id=None):
    if not voice or not text: raise ValueError('Voice and text are required.')
    if voice_version_id is None:
        try:
            selected = http.call('voice_resolve', {'name': voice})
        except Failure as exc:
            if exc.body['code'] != 'not_found': raise
            status = http.get('/v1/status')
            presets = [p for p in status.get('configured_presets', []) if p['preset'].casefold() == voice.casefold()]
            if len(presets) != 1:
                raise
            register = dict(preset=presets[0]['preset'], voice_id=voice.casefold(), expected_revision=0)
            try:
                http.call('voice_preset_register', register, key=key('preset', register))
            except ConnectionError:
                return dict(preset_key=key('preset', register), state='registration_unknown',
                            action='Repeat with the same voice and text to replay the preset registration key.')
            selected = http.call('voice_resolve', {'name': voice.casefold()})
        if len(selected['version_ids']) < 1: raise Failure('not_found', 'Voice has no saved version.')
        voice_version_id = selected['version_ids'][-1]
    version = http.call('voice_version_get', {'voice_version_id': voice_version_id})
    request = dict(voice_version_id=version['id'], text=text, identity=version['identity'],
                   fallback='forbid', variants=1, reuse='never')
    generation_key = key('speak', request)
    if job_id is None:
        try:
            job_id = http.call('speech_generate', request, key=generation_key)['job_id']
        except ConnectionError:
            return dict(generation_key=generation_key, voice_version_id=voice_version_id,
                        state='submission_unknown',
                        action='Repeat with --voice-version-id and the same text; the same idempotency key will be reused.')
    job = wait(http, job_id, seconds)
    if job.get('timed_out') or job['state'] != 'succeeded':
        return dict(job_id=job_id, generation_key=generation_key, voice_version_id=voice_version_id, state=job['state'],
                    timed_out=job.get('timed_out', False))
    if len(job['artifact_ids']) != 1:
        raise Failure('not_ready', 'Generation has no single published artifact.')
    artifact = http.call('artifact_get', {'artifact_id': job['artifact_ids'][0]})
    if artifact['job_id'] != job_id or artifact['input_sha256'] != job['input_sha256'] or len(artifact['asset_ids']) != 1:
        raise Failure('not_ready', 'Published artifact does not match generation job.')
    body = dict(job_id=job_id, artifact_id=artifact['id'], asset_id=artifact['asset_ids'][0], state='succeeded')
    if output is not None: body.update(download(http, artifact['asset_ids'][0], output))
    return body
