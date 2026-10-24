"""Small named HTTP read CLI; media goes to an exclusive local file, not stdout."""
import argparse
import base64
import hashlib
import sys
from pathlib import Path
from jsonschema.exceptions import ValidationError
from service.catalogue import Failure, compact
from clients.http import HTTP


def main(argv=None):
    parser = argparse.ArgumentParser(description='Named HTTP reads; capabilities describe configured service availability')
    parser.add_argument('--url', default=None)
    parser.add_argument('--credential', default=None, help='Explicit bearer credential; prefer SPEECH_CREDENTIAL environment for process privacy.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('discover')
    commands.add_parser('status')
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
    version = vsubs.add_parser('version')
    vget = version.add_subparsers(dest='version_verb', required=True).add_parser('get')
    vget.add_argument('voice_version_id')
    for name in ('prepare', 'save', 'preset-register'):
        action = vsubs.add_parser(name)
        action.add_argument('--request', required=True, help='Canonical JSON request object.')
        action.add_argument('--idempotency-key', required=True)
        if name == 'prepare': action.add_argument('--audition-text', help='Exact spoken audition text; required by configured Pocket preparation.')
    results = vsubs.add_parser('results'); results.add_argument('job_id')
    speech = commands.add_parser('speech').add_subparsers(dest='verb', required=True)
    generate = speech.add_parser('generate'); generate.add_argument('--request', required=True); generate.add_argument('--idempotency-key', required=True)
    jobs = commands.add_parser('job').add_subparsers(dest='verb', required=True)
    jobs.add_parser('get').add_argument('job_id')
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
    import os
    http = HTTP(args.url, credential=args.credential or os.environ.get('SPEECH_CREDENTIAL'))
    try:
        if args.command == 'discover': result = http.call('discover', {})
        elif args.command == 'status': result = http.get('/skeleton')
        elif args.command == 'operation-detail':
            from urllib.parse import quote
            result = http.get('/skeleton/operations/' + quote(args.operation_id, safe=''))
        elif args.command == 'speech' or (args.command == 'voice' and args.verb in ('prepare','save','preset-register')):
            import json
            operation = 'speech_generate' if args.command == 'speech' else 'voice_'+args.verb.replace('-', '_')
            payload = json.loads(args.request)
            if operation == 'voice_prepare' and args.audition_text is not None:
                if 'audition_text' in payload and payload['audition_text'] != args.audition_text: raise ValueError('conflicting_audition_text')
                payload['audition_text'] = args.audition_text
            result = http.call(operation, payload, key=args.idempotency_key)
        elif args.command == 'voice' and args.verb == 'results': result = http.call('voice_results',dict(job_id=args.job_id))
        elif args.command == 'job': result = http.call('job_'+args.verb,dict(job_id=args.job_id), **({'key':args.idempotency_key} if args.verb == 'cancel' else {}))
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
                with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix='.speech-0c-', delete=False) as out:
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
        print(compact(result)); return 0
    except Failure as exc:
        print(compact(exc.body)); return 2
    except ValidationError:
        from service.catalogue import problem
        print(compact(problem('invalid_request', 'Request does not match canonical schema.', 'request'))); return 2
    except ConnectionError:
        print(compact({'transport_error':'Skeleton HTTP transport unavailable.'})); return 3
    except (OSError, ValueError) as exc:
        # No local paths, payloads, URLs or tracebacks in model-facing errors.
        from service.catalogue import problem
        print(compact(problem('invalid_request', 'Invalid request or local media transfer; check arguments and paths.', 'output'))); return 2

if __name__ == '__main__':
    sys.exit(main())
