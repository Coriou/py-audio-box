"""One bounded synthetic exchange, no descendants, engines or service imports."""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4
import wave

from datetime import datetime, timezone
from contracts import validate as oracle
from worker.transport import Session, Refused, authenticate, canonical, receive, send
from jsonschema.exceptions import ValidationError


def run(connection, config, staging, cutoff):
    job, context = config['job'], config['context']
    attempt = job['attempts'][-1]
    cutoff = min(cutoff, time.monotonic() + max(0, (datetime.fromisoformat(job['deadline']) - datetime.now(timezone.utc)).total_seconds()))
    binding = hashlib.sha256(canonical([job, context, config['allowed_assets'], config['profiles']])).hexdigest()
    session = Session(config['secret'], binding, config['control_uid'], config['gid'])
    authenticate(connection, session, cutoff, control=False)
    message = receive(connection, cutoff)
    oracle.worker_message(message, job, config['allowed_assets'], transport_owner=attempt['owner_id'],
                          context=context, current_profiles=config['profiles'])
    if message['kind'] != 'execute':
        raise Refused('unexpected_message')
    plan = oracle.stored_context(job, context)
    request = plan['request']
    if plan['operation'] not in ('voice_prepare', 'speech_generate'):
        raise Refused('synthetic_only')
    if plan['operation'] == 'speech_generate' and (request['variants'] != 1 or request['reuse'] != 'never'):
        raise Refused('synthetic_only')
    from worker.synthetic import fixture_outputs, fixture_manifest
    outputs, files = fixture_outputs(message)
    if Path(attempt['staging_id']).name != attempt['staging_id']:
        raise Refused('invalid_stage')
    directory = os.open(Path(staging) / attempt['staging_id'], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name, data in files:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640, dir_fd=directory)
            with os.fdopen(fd, 'wb') as output:
                output.write(data);output.flush();os.fsync(output.fileno())
    finally:
        os.close(directory)
    manifest = fixture_manifest(message, outputs)
    message.update(kind='result', stage='completed', result_asset_ids=[a['id'] for a in outputs])
    send(connection, dict(message=message, manifest=manifest), cutoff)


def main():
    os.umask(0o027)
    cutoff = time.monotonic() + 8
    path = Path('/ipc/session.json')
    while not path.exists():
        if time.monotonic() >= cutoff:
            raise TimeoutError('session_unavailable')
        time.sleep(.02)
    config = json.loads(path.read_text())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(max(.01, cutoff-time.monotonic()))
        connection.connect('/ipc/worker.sock')
        try:
            run(connection, config, '/staging', cutoff)
        except (Refused, oracle.ContractError, ValidationError):
            print('W1 worker: refused session or execute', flush=True)
            return
        except OSError:
            print('W1 worker: uncertain disconnect or timeout', flush=True)
            raise SystemExit(2)
    print('W1 worker: synthetic proposal sent', flush=True)


if __name__ == '__main__':
    main()
