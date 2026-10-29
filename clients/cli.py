"""Small named HTTP read CLI; media goes to an exclusive local file, not stdout."""
import argparse
import base64
import hashlib
import sys
from pathlib import Path
from jsonschema.exceptions import ValidationError
from service.catalogue import Failure, compact
from clients.http import HTTP, network_os_error
from clients.config import ConfigurationFailure, InstallationFailure


def strict_json(raw):
    import json
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Named HTTP reads; capabilities describe configured service availability')
    parser.add_argument('--url', default=None)
    parser.add_argument('--credential', default=None, help='Explicit bearer credential; prefer SPEECH_CREDENTIAL environment for process privacy.')
    parser.add_argument('--installation', help='Explicit local installation root containing install.json and credential.json.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('discover')
    commands.add_parser('status')
    enroll = commands.add_parser('enroll', help='Upload, prepare, audition and save one trusted reference voice.')
    enroll.add_argument('--name', required=True)
    enroll.add_argument('--wav', type=Path)
    enroll.add_argument('--asset-id')
    enroll.add_argument('--reference-text', required=True)
    enroll.add_argument('--audition-text', required=True)
    enroll.add_argument('--job-id', help='Resume an already submitted preparation job.')
    enroll.add_argument('--wait-seconds', type=int, default=120)
    speak = commands.add_parser('speak', help='Generate one output from a saved named voice.')
    speak.add_argument('--voice', required=True)
    speak.add_argument('--text', required=True)
    speak.add_argument('--output', type=Path, required=True)
    speak.add_argument('--job-id', help='Resume an already submitted generation job.')
    speak.add_argument('--voice-version-id', help='Pin the saved version when replaying an uncertain submission.')
    speak.add_argument('--wait-seconds', type=int, default=120)
    detail = commands.add_parser('operation-detail')
    detail.add_argument('operation_id')
    voices = commands.add_parser('voices')
    vs = voices.add_subparsers(dest='verb', required=True).add_parser('list')
    vs.add_argument('--limit', type=int, default=20)
    vs.add_argument('--cursor', default='')
    voice = commands.add_parser('voice')
    vsubs = voice.add_subparsers(dest='verb', required=True)
    vl = vsubs.add_parser('versions'); vl.add_argument('voice_id'); vl.add_argument('--limit',type=int,default=20); vl.add_argument('--cursor',default='')
    vg = vsubs.add_parser('get'); vg.add_argument('voice_id')
    vr = vsubs.add_parser('resolve'); vr.add_argument('name')
    version = vsubs.add_parser('version')
    vget = version.add_subparsers(dest='version_verb', required=True).add_parser('get')
    vget.add_argument('voice_version_id')
    for name in ('prepare', 'save', 'preset-register', 'aliases', 'archive'):
        action = vsubs.add_parser(name)
        action.add_argument('--request', required=True, help='Canonical JSON request object.')
        action.add_argument('--idempotency-key', required=True)
        if name == 'prepare': action.add_argument('--audition-text', help='Exact spoken audition text; required by configured Pocket preparation.')
    results = vsubs.add_parser('results'); results.add_argument('job_id')
    speech = commands.add_parser('speech').add_subparsers(dest='verb', required=True)
    generate = speech.add_parser('generate'); generate.add_argument('--request', required=True); generate.add_argument('--idempotency-key', required=True)
    audio = commands.add_parser('audio').add_subparsers(dest='verb',required=True).add_parser('export')
    audio.add_argument('--request',required=True); audio.add_argument('--idempotency-key',required=True)
    jobs = commands.add_parser('job').add_subparsers(dest='verb', required=True)
    jobs.add_parser('get').add_argument('job_id')
    for name in ('events','wait'):
        action = jobs.add_parser(name); action.add_argument('job_id'); action.add_argument('--cursor',default='')
        action.add_argument('--limit' if name == 'events' else '--wait-seconds',type=int,default=20 if name == 'events' else 5)
    retry = jobs.add_parser('retry'); retry.add_argument('job_id'); retry.add_argument('--input-sha256',required=True); retry.add_argument('--idempotency-key',required=True)
    listing = commands.add_parser('jobs').add_subparsers(dest='verb',required=True).add_parser('list')
    listing.add_argument('--cursor',default=''); listing.add_argument('--limit',type=int,default=20)
    commands.add_parser('profile').add_subparsers(dest='verb',required=True).add_parser('get').add_argument('profile_id')
    cancel = jobs.add_parser('cancel'); cancel.add_argument('job_id'); cancel.add_argument('--idempotency-key', required=True)
    artifacts = commands.add_parser('artifact').add_subparsers(dest='verb', required=True)
    artifacts.add_parser('get').add_argument('artifact_id')
    asset = commands.add_parser('asset')
    subs = asset.add_subparsers(dest='verb', required=True)
    upload = subs.add_parser('upload'); upload.add_argument('file',type=Path)
    get = subs.add_parser('get'); get.add_argument('asset_id')
    download = subs.add_parser('content')
    download.add_argument('asset_id')
    download.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from clients.config import resolve
        url, credential = resolve(args.url, args.credential, args.installation)
        http = HTTP(url, credential=credential)
        if args.command == 'enroll':
            from clients.workflow import enroll
            result = enroll(http, name=args.name, file=args.wav, asset_id=args.asset_id,
                            reference_text=args.reference_text, audition_text=args.audition_text,
                            job_id=args.job_id, seconds=args.wait_seconds)
        elif args.command == 'speak':
            from clients.workflow import speak
            result = speak(http, voice=args.voice, text=args.text, output=args.output,
                           job_id=args.job_id, voice_version_id=args.voice_version_id,
                           seconds=args.wait_seconds)
        elif args.command == 'discover': result = http.call('discover', {})
        elif args.command == 'status': result = http.get('/v1/status')
        elif args.command == 'operation-detail':
            from urllib.parse import quote
            result = http.get('/v1/operations/' + quote(args.operation_id, safe=''))
        elif args.command in ('speech','audio') or (args.command == 'voice' and args.verb in ('prepare','save','preset-register','aliases','archive')):
            import json
            operation = 'audio_export' if args.command == 'audio' else 'speech_generate' if args.command == 'speech' else 'voice_'+args.verb.replace('-', '_')
            payload = strict_json(args.request)
            if operation == 'voice_prepare' and args.audition_text is not None:
                if 'audition_text' in payload and payload['audition_text'] != args.audition_text: raise ValueError('conflicting_audition_text')
                payload['audition_text'] = args.audition_text
            result = http.call(operation, payload, key=args.idempotency_key)
        elif args.command == 'voice' and args.verb == 'resolve': result = http.call('voice_resolve',dict(name=args.name))
        elif args.command == 'voice' and args.verb == 'results': result = http.call('voice_results',dict(job_id=args.job_id))
        elif args.command == 'jobs': result = http.call('jobs_list',dict(cursor=args.cursor,limit=args.limit))
        elif args.command == 'profile': result = http.call('profile_get',dict(profile_id=args.profile_id))
        elif args.command == 'job':
            payload = dict(job_id=args.job_id)
            if args.verb == 'retry': payload['input_sha256'] = args.input_sha256
            if args.verb in ('events','wait'):
                payload['cursor'] = args.cursor
                payload.update(dict(limit=args.limit) if args.verb == 'events' else dict(wait_seconds=args.wait_seconds))
            result = http.call('job_'+args.verb,payload, **({'key':args.idempotency_key} if args.verb in ('cancel','retry') else {}))
        elif args.command == 'artifact': result = http.call('artifact_get',dict(artifact_id=args.artifact_id))
        elif args.command == 'voices': result = http.call('voices_list', {'limit': args.limit, 'cursor': args.cursor})
        elif args.command == 'voice' and args.verb == 'versions':
            result = http.call('voice_versions_list',dict(voice_id=args.voice_id,limit=args.limit,cursor=args.cursor))
        elif args.command == 'voice':
            result = http.call('voice_get', {'voice_id': args.voice_id}) if args.verb == 'get' else http.call('voice_version_get', {'voice_version_id': args.voice_version_id})
        elif args.verb == 'upload': result = http.upload_file(args.file)
        elif args.verb == 'get': result = http.call('asset_get', {'asset_id': args.asset_id})
        else:
            asset = http.call('asset_get', {'asset_id': args.asset_id})
            first = http.call('asset_content', {'asset_id':args.asset_id, 'start':0, 'end_exclusive':1})
            total_bytes = first['total_bytes']
            # Use a sibling temporary file; publish only complete verified bytes.
            import tempfile
            import os
            temporary = None
            try:
                digest = hashlib.sha256(); total = 0
                with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix='.py-audio-box-', delete=False) as out:
                    temporary = Path(out.name)
                    for start in range(0, total_bytes, 65536):
                        end = min(start + 65536, total_bytes)
                        chunk = http.call('asset_content', {'asset_id':args.asset_id, 'start':start, 'end_exclusive':end})
                        data = base64.b64decode(chunk['content_base64'], validate=True)
                        if (chunk['start'], chunk['end_exclusive'], chunk['total_bytes'], chunk['content_sha256']) != (start,end,total_bytes,asset['sha256']) or len(data) != end-start:
                            raise ValueError('Unexpected byte transfer metadata.')
                        out.write(data); digest.update(data); total += len(data)
                if digest.hexdigest() != asset['sha256']: raise ValueError('Downloaded hash mismatch.')
                os.link(temporary, args.output)  # refuses an existing destination
                result = {'asset_id':args.asset_id, 'bytes':total, 'sha256':digest.hexdigest(), 'saved':True}
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
        print(compact(result)); return 2 if args.command in ('enroll','speak') and result.get('state') != 'succeeded' else 0
    except Failure as exc:
        print(compact(exc.body)); return 2
    except (ConfigurationFailure, InstallationFailure) as exc:
        print(compact(exc.body)); return 2
    except ValidationError:
        from service.catalogue import problem
        print(compact(problem('invalid_request', 'Request does not match canonical schema.', 'request'))); return 2
    except ConnectionError:
        print(compact({'transport_error':'Speech HTTP transport unavailable.'})); return 3
    except OSError as exc:
        if network_os_error(exc):
            print(compact({'transport_error':'Speech HTTP transport unavailable.'})); return 3
        from service.catalogue import problem
        print(compact(problem('invalid_request', 'Invalid request or local media transfer; check arguments and paths.', 'output'))); return 2
    except ValueError:
        # No local paths, payloads, URLs or tracebacks in model-facing errors.
        from service.catalogue import problem
        print(compact(problem('invalid_request', 'Invalid request or local media transfer; check arguments and paths.', 'output'))); return 2

if __name__ == '__main__':
    sys.exit(main())
