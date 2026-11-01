"""Bounded separate-container control caller; synthetic fixture only."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import wave

from contracts import validate as oracle
from service.worker_transport import Authority, exchange
from worker.transport import Session, authenticate, receive, send, Uncertain


def fixture_authority():
    fixture = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
    # E1 synthetic mock admission only; never saved as capability/readiness.
    for profile in fixture['profiles']:
        profile.update(support='qualified', installation='installed', readiness='ready',
                       verification='measured', qualification_receipt='E1-w1-synthetic-mock-only')
    job = fixture['job']
    job['deadline'] = (datetime.now(timezone.utc) + timedelta(seconds=8)).isoformat()
    job['state'] = 'running'
    job['attempts'][-1]['reaped'] = False
    return Authority(job, fixture['context'], tuple(fixture['authorized_assets']), tuple(fixture['profiles']))


def verify(observation, authority, staging, worker_uid):
    assert observation.status == 'proposal', observation
    message, manifest = observation.message, observation.manifest
    attempt = authority.job['attempts'][-1]
    assert len(manifest['outputs']) == 1
    output = manifest['outputs'][0]
    fd = os.open(Path(staging)/attempt['staging_id']/'output.wav', os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        assert info.st_uid == worker_uid and info.st_size <= 1024*1024
        data = stream.read(1024*1024+1)
    with wave.open(io.BytesIO(data), 'rb') as wav:
        assert (wav.getnframes(), wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (4800, 48000, 1, 2)
        assert wav.readframes(4801) == b'\0' * 9600
    digest = hashlib.sha256(data).hexdigest()
    assert {k: output[k] for k in ('sha256', 'frames', 'sample_rate', 'channels', 'media_format', 'media_subtype')} == dict(sha256=digest, frames=4800, sample_rate=48000, channels=1, media_format='wav', media_subtype='PCM_16')
    assert output['provenance'] == dict(kind='synthetic', source_sha256=digest, source_id='phase-1a-silence', missing_fields=[])
    artifact = dict(id=manifest['artifact_id'], job_id=authority.job['id'], generation=authority.job['generation'],
                    asset_ids=[output['id']], input_sha256=authority.job['input_sha256'], project_revision=0,
                    manifest_sha256=oracle.canonical_hash(manifest), identity=authority.context['plan']['resolved']['identity'],
                    attempt_id=attempt['id'], owner_id=attempt['owner_id'], staging_id=attempt['staging_id'])
    oracle.publication(authority.job, attempt, artifact, {output['id']: digest},
                       transport_owner=attempt['owner_id'], context=authority.context, manifest=manifest,
                       verified_result=message, allowed_assets=authority.allowed_assets)
    return digest


def main():
    os.umask(0o027)
    authority = fixture_authority()
    attempt = authority.job['attempts'][-1]
    stage = Path('/staging')/attempt['staging_id']
    stage.mkdir(mode=0o770)
    stage.chmod(0o770)
    secret = secrets.token_hex(32)
    fault = os.environ.get('W1_FAULT', 'none')
    assert fault in ('none', 'session', 'tuple')
    config = dict(job=authority.job, context=authority.context, allowed_assets=authority.allowed_assets,
                  profiles=authority.profiles, secret=secret, control_uid=10001, gid=10001)
    if fault == 'session':
        config['secret'] = secrets.token_hex(32)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind('/ipc/worker.sock')
        Path('/ipc/worker.sock').chmod(0o660)
        listener.listen(1)
        listener.settimeout(8)
        # Atomic publication after the listener and private attempt stage exist.
        Path('/ipc/session.tmp').write_text(json.dumps(config))
        Path('/ipc/session.tmp').chmod(0o640)
        Path('/ipc/session.tmp').rename('/ipc/session.json')
        connection, _ = listener.accept()
        with connection:
            session = Session(secret, authority.binding, 10002, 10001)
            cutoff = authority.cutoff()
            if fault == 'tuple':
                authenticate(connection, session, cutoff, control=True)
                message = authority.message()
                message['attempt_id'] = 'wrong-attempt'
                send(connection, message, cutoff)
                try:
                    receive(connection, cutoff)
                except Uncertain:
                    pass
                else:
                    raise AssertionError('wrong tuple returned a result')
            else:
                result = exchange(connection, authority, session, cutoff)
    if fault == 'none':
        digest = verify(result, authority, '/staging', 10002)
        print(json.dumps(dict(status='verified_synthetic_proposal', peer=asdict(result.peer),
                              sha256=digest, bytes=9644, protocol='1.3.0', interface='w1-session/1')))
    else:
        if fault == 'session':
            assert result.status == 'refused', result
        assert not list(stage.iterdir()), 'refused execution wrote output'
        print(json.dumps(dict(status='verified_refusal', fault=fault, protocol='1.3.0')))

    # Keep topology inspectable after the exchange; external harness releases it.
    import time
    for _ in range(200):
        if Path('/ipc/release').exists():
            return
        time.sleep(.05)
    raise TimeoutError('inspection_release_missing')


def test_worker_source_excludes_service_imports():
    import ast
    tree = ast.parse(Path('worker/isolated_synthetic.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('service') for n in ast.walk(tree))


if __name__ == '__main__':
    main()
