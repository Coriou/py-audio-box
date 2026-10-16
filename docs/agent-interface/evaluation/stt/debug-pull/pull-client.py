#!/usr/bin/env python3
"""One anonymous pinned pull through an explicit Docker Engine Unix socket.

Only progress counts are observable here: they are not network-wire accounting.
An external watchdog must own this process. Closing this request cancels its
client stream; check readiness separately and never restart the Docker daemon.
"""
import argparse
import datetime
import http.client
import json
from pathlib import Path
import socket
import sys
import urllib.parse

class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=30)
        self.path = path
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)

def emit(**row):
    row['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(row, sort_keys=True), flush=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--socket', required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--reference', required=True)
parser.add_argument('--progress-budget', type=int, default=550000000)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
assert '@sha256:' in args.reference
assert args.reference.split('@', 1)[1] == 'sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647'
expected = {v['digest'].split(':')[1][:12]: v['size'] for v in manifest['layers']}
assert sum(expected.values()) == 504943686
conn = UnixConnection(args.socket)
progress = {}
completed = set()
observed = 0
try:
    conn.request('POST', '/v1.54/images/create?' + urllib.parse.urlencode({'fromImage': args.reference}),
                 body=b'', headers={'Content-Length': '0'})
    response = conn.getresponse()
    emit(event='response', status=response.status,
         declared_compressed_bytes=sum(expected.values()), progress_abort_budget_bytes=args.progress_budget)
    if response.status != 200:
        emit(event='error', kind='engine_http_status')
        sys.exit(1)
    for line in response:
        row = json.loads(line)
        status = row.get('status', '')
        ident = row.get('id')
        detail = row.get('progressDetail', {})
        if 'error' in row:
            # Engine errors can contain signed registry URLs. Retain only the
            # category; do not persist the raw error body.
            emit(event='error', kind='engine_stream_error', code=row.get('errorDetail', {}).get('code'))
            sys.exit(1)
        if status.startswith('Retrying'):
            emit(event='error', kind='unexpected_engine_retry')
            sys.exit(2)
        if status == 'Downloading':
            current = detail.get('current', 0)
            previous = progress.get(ident, 0)
            observed += current - previous if current >= previous else current
            progress[ident] = current
        elif status == 'Download complete' and ident not in completed:
            observed += max(0, expected.get(ident, 0) - progress.get(ident, 0))
            completed.add(ident)
        emit(event='progress', status=status, layer=ident, detail=detail,
             observed_compressed_progress_bytes=observed)
        if observed > args.progress_budget:
            emit(event='error', kind='compressed_progress_budget_exceeded')
            sys.exit(2)
    emit(event='end_of_stream', observed_compressed_progress_bytes=observed,
         completed_layers=len(completed))
except (TimeoutError, OSError, ValueError, http.client.HTTPException) as error:
    emit(event='error', kind=type(error).__name__)
    sys.exit(1)
finally:
    conn.close()
