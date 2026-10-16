"""Small named fixture CLI; media goes to an exclusive local file, not stdout."""
import argparse
import base64
import hashlib
import sys
from pathlib import Path
from jsonschema.exceptions import ValidationError
from service.catalogue import Failure, compact
from clients.http import HTTP


def main(argv=None):
    parser = argparse.ArgumentParser(description='0C synthetic fixture reads; no durable service')
    parser.add_argument('--url', default=None)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('discover')
    commands.add_parser('status')
    detail = commands.add_parser('operation-detail')
    detail.add_argument('operation_id')
    voices = commands.add_parser('voices')
    vs = voices.add_subparsers(dest='verb', required=True).add_parser('list')
    vs.add_argument('--limit', type=int, default=20)
    asset = commands.add_parser('asset')
    subs = asset.add_subparsers(dest='verb', required=True)
    get = subs.add_parser('get'); get.add_argument('asset_id')
    download = subs.add_parser('content')
    download.add_argument('asset_id')
    download.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    http = HTTP(args.url)
    try:
        if args.command == 'discover': result = http.call('discover', {})
        elif args.command == 'status': result = http.get('/skeleton')
        elif args.command == 'operation-detail':
            from urllib.parse import quote
            result = http.get('/skeleton/operations/' + quote(args.operation_id, safe=''))
        elif args.command == 'voices': result = http.call('voices_list', {'limit': args.limit})
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
                result = {'fixture':True, 'asset_id':args.asset_id, 'bytes':total, 'sha256':digest.hexdigest(), 'saved':True}
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
        print(compact(problem('invalid_request', 'Local byte transfer failed; use a new writable output file.', 'output'))); return 2

if __name__ == '__main__':
    sys.exit(main())
