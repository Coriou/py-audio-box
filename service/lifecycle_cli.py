"""Installed fixed synthetic diagnostic. Not a dispatcher/public API."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import time
import wave

from service.owner_store import OwnerStore
from service.worker_supervisor import Supervisor
from service.worker_transport import Authority
from service.supervisor_runtime import SupervisorRuntime
from worker.lifecycle import State


def fixture_authority(seconds=8):
    value = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
    for profile in value['profiles']:
        profile.update(support='qualified', installation='installed', readiness='ready',
            verification='measured', qualification_receipt='E1-w2c-synthetic-only')
    job = value['job']
    nonce = secrets.token_hex(8)
    job['id'] = 'job-'+nonce
    attempt = job['attempts'][-1]
    attempt.update(id='attempt-'+nonce, owner_id='owner-'+nonce, staging_id='stage-'+nonce, reaped=False)
    job.update(deadline=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(), state='running')
    return Authority(job, value['context'], value['authorized_assets'], value['profiles'])


def authority_from_store(store):
    current = store.current()
    # Full immutable snapshot is control-owned, written BEFORE dispatch intent.
    value = json.loads(Path('/control/authority.json').read_bytes())
    authority = Authority(*value)
    if current and current['authority']['binding'] != authority.binding:
        raise RuntimeError('control_snapshot_mismatch')
    return authority


def verify(observation, authority, engine_uid):
    from contracts import validate as oracle
    import stat
    attempt = authority.job['attempts'][-1]
    fd = os.open('/staging/'+attempt['staging_id']+'/output.wav', os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 9644 and info.st_uid == engine_uid
        data = stream.read(9645)
    with wave.open(io.BytesIO(data), 'rb') as wav:
        assert (wav.getnframes(),wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) == (4800,48000,1,2)
        assert wav.readframes(4801) == b'\0'*9600
    digest = hashlib.sha256(data).hexdigest()
    manifest = observation.manifest
    output = manifest['outputs'][0]
    assert {k:output[k] for k in ('sha256','frames','sample_rate','channels','media_format','media_subtype')} == dict(sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16')
    assert output['provenance'] == dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[])
    artifact = dict(id=manifest['artifact_id'],job_id=authority.job['id'],generation=authority.job['generation'],
        asset_ids=[output['id']],input_sha256=authority.job['input_sha256'],project_revision=0,
        manifest_sha256=oracle.canonical_hash(manifest),identity=authority.context['plan']['resolved']['identity'],
        attempt_id=attempt['id'],owner_id=attempt['owner_id'],staging_id=attempt['staging_id'])
    oracle.publication(authority.job,attempt,artifact,{output['id']:digest},transport_owner=attempt['owner_id'],
        context=authority.context,manifest=manifest,verified_result=observation.message,allowed_assets=authority.allowed_assets)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('hello','dispatch','start','execute','observe','stop','collect','ack'))
    parser.add_argument('--fixture', choices=('silence','busy','descendants','permissions'), default='silence')
    parser.add_argument('--seconds', type=int, choices=range(2,26), default=8)
    parser.add_argument('--interrupt', choices=('none','prespawn','preregistration','prestart','postspawn','start-sent','stop-sent'), default='none')
    args = parser.parse_args()
    runtime = SupervisorRuntime(interrupt=args.interrupt)
    if args.action == 'hello':
        print(json.dumps(runtime.call('hello', {}))); return
    store = OwnerStore('/control/owner.db')
    try:
        supervisor = Supervisor(store, runtime)
        if args.action == 'dispatch':
            old = store.current()
            if old and old['state'] != State.STOPPED.value:
                result = supervisor.reconnect()
            else:
                if old:
                    ackfile=Path('/control/acked.json')
                    if not ackfile.exists() or json.loads(ackfile.read_bytes())['runtime_id'] != old['runtime_id']:
                        raise RuntimeError('acknowledge_prior_outcome_before_next_allocation')
                authority = fixture_authority(args.seconds)
                target = Path('/control/authority.json')
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    stream.write(authority._snapshot); stream.flush(); os.fsync(stream.fileno())
                temporary.replace(target)
                fd = os.open('/control', os.O_RDONLY|os.O_DIRECTORY)
                os.fsync(fd); os.close(fd)
                def fault(point):
                    if point == args.interrupt:
                        os._exit(17)
                result = supervisor.dispatch(authority, lambda: runtime.prepare(authority,args.fixture), runtime.start, fault=fault)
            print(json.dumps(dict(runtime_id=result['runtime_id'],state=result['state'])))
        elif args.action == 'execute':
            import socket
            from service.worker_transport import exchange
            from worker.runtime_protocol import execution_session
            current=supervisor.reconnect()
            if current is None or current['state']==State.UNKNOWN.value:
                raise RuntimeError('exact_observed_allocation_required')
            authority=authority_from_store(store)
            secret=Path('/ipc/execution.key').read_text().strip()
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
                connection.settimeout(1.5);connection.connect('/ipc/worker.sock')
                result=exchange(connection,authority,execution_session(secret,authority.binding,0,10001),authority.cutoff(1.5))
            if result.status!='proposal':raise RuntimeError('execute_proposal_unavailable')
            print(json.dumps(dict(status=result.status,proposal_sha256=hashlib.sha256(json.dumps(result.manifest,sort_keys=True).encode()).hexdigest())))
        elif args.action == 'start':
            current = store.current()
            if current is None or current['runtime_id'] is None:
                raise RuntimeError('bound_intent_required')
            print(json.dumps(runtime.start(current['runtime_id'])))
        elif args.action == 'observe':
            result = supervisor.reconnect()
            print(json.dumps(dict(runtime_id=result['runtime_id'],state=result['state'])))
        elif args.action == 'stop':
            supervisor.stop()
            # Poll through bounded communication; parent alone owns escalation.
            end = time.monotonic()+2
            while supervisor.reconnect()['state'] != State.STOPPED.value and time.monotonic()<end:
                time.sleep(.03)
            print(json.dumps(dict(state=supervisor.reconnect()['state'])))
        else:
            current = supervisor.reconnect()
            authority = authority_from_store(store)
            if args.action == 'ack':
                if current['state'] != State.STOPPED.value:
                    raise RuntimeError('durable_outcome_required')
                # Diagnostic records receipt durably before acknowledgement.
                collected = runtime.collect(authority,current['runtime_id'])
                if collected is None:
                    raise RuntimeError('receipt_required')
                target = Path('/control/outcome.json')
                with target.open('wb') as stream:
                    stream.write(json.dumps(collected[0]).encode()); stream.flush(); os.fsync(stream.fileno())
                fd=os.open('/control',os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                answer=runtime.call('ack',dict(runtime_id=current['runtime_id']))
                with Path('/control/acked.json').open('wb') as stream:
                    stream.write(json.dumps(dict(runtime_id=current['runtime_id'])).encode());stream.flush();os.fsync(stream.fileno())
                fd=os.open('/control',os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                print(json.dumps(answer))
            else:
                result = runtime.collect(authority,current['runtime_id'])
                if result is None:
                    print(json.dumps(dict(state=current['state']))); return
                receipt, observation = result
                digest = None
                if observation and receipt['timely']:
                    digest = supervisor.verify(observation,lambda value: verify(value,authority,receipt['engine_uid']))
                print(json.dumps(dict(receipt=receipt,sha256=digest,
                    proposal_sha256=hashlib.sha256(json.dumps(observation.manifest,sort_keys=True).encode()).hexdigest() if observation else None)))
    finally:
        store.close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, AssertionError):
        print(json.dumps(dict(error='lifecycle_unavailable_or_quarantined')))
        raise SystemExit(2)
