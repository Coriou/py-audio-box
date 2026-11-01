"""Container control commands, driven by trusted external harness. No Docker API."""
import json
import hashlib
import os
from pathlib import Path
import secrets
import socket
from dataclasses import asdict
from service.owner_store import OwnerStore
from service.worker_supervisor import Supervisor
from service.worker_transport import Authority, exchange, TransportObservation
from worker.lifecycle import Observation, State
from tests.release.test_worker_boundary import fixture_authority, verify
from worker.transport import Session, canonical


def main():
    action = os.environ.get('W2_ACTION', 'prepare')
    if action == 'prepare':
        os.umask(0o027)
        authority = fixture_authority()
        # Long enough for deterministic harness barriers; worker caps it at 30s.
        from datetime import datetime, timedelta, timezone
        job = authority.job
        job['deadline'] = (datetime.now(timezone.utc)+timedelta(seconds=20)).isoformat()
        authority = Authority(job, authority.context, authority.allowed_assets, authority.profiles)
        stage = Path('/staging')/job['attempts'][-1]['staging_id']
        stage.mkdir(mode=0o770); stage.chmod(0o770)
        config = dict(job=authority.job, context=authority.context, allowed_assets=authority.allowed_assets, profiles=authority.profiles, ignore_term=os.environ.get('W2_IGNORE_TERM') == '1', secret=secrets.token_hex(32), control_uid=10001, gid=10001)
        Path('/ipc/session.json').write_text(json.dumps(config))
        Path('/ipc/session.json').chmod(0o640)
        # Control-only root: never mounted on worker.
        with_store = OwnerStore('/control/owner.sqlite')
        with_store.intent(authority); with_store.close()
        print(json.dumps(dict(status='intent_committed', deadline=authority.job['deadline'], binding=authority.binding)), flush=True)
    else:
        config = json.loads(Path('/ipc/session.json').read_text())
        authority = Authority(config['job'], config['context'], config['allowed_assets'], config['profiles'])
        store = OwnerStore('/control/owner.sqlite')
        identity = os.environ.get('W2_RUNTIME_ID', '')
        class HarnessEvidence:
            # Values injected only by the trusted host operator after Docker inspect/wait.
            def observe(self, runtime_id, timeout):
                assert runtime_id == identity
                return Observation(identity, State(os.environ['W2_STATE']))
        supervisor = Supervisor(store, HarnessEvidence())
        if action in ('status','observe'):
            if action == 'observe': store.observe(Observation(identity, State(os.environ['W2_STATE'])))
            print(json.dumps(store.current())); store.close(); return
        if action == 'bind':
            store.bind(authority, identity)
            print('runtime_bound'); store.close(); return
        if action == 'verify':
            receipt = json.loads(Path('/control/result.json').read_text())
            result = TransportObservation('proposal', 'cached', message=receipt['message'], manifest=receipt['manifest'])
            supervisor.verify(result, lambda observation: verify(observation, authority, '/staging', 10002))
            print('reaped_then_verified'); store.close(); return
        secret = config['secret']
        if action == 'stale': secret = '00'*32
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2); connection.connect('/ipc/worker.sock')
            result = supervisor.exchange(connection, authority, Session(secret, authority.binding, 10002, 10001), fault=lambda point: os._exit(17) if action=='lostack' and point=='postsend' else None)
        if action == 'stale':
            assert result.status == 'refused', result
            print('stale_session_refused', flush=True)
        else:
            assert result.status == 'proposal', result
            # Verification here is byte/oracle evidence only, not adoption/publication.
            digest = verify(result, authority, '/staging', 10002)
            receipt = dict(message=result.message, manifest=result.manifest, digest=digest)
            previous = Path('/control/result.json')
            if previous.exists(): assert json.loads(previous.read_text()) == receipt
            previous.write_text(json.dumps(receipt))
            print(json.dumps(dict(status='same_cached_proposal_verified', deadline=authority.job['deadline'], manifest_sha256=hashlib.sha256(canonical(result.manifest)).hexdigest())), flush=True)
        store.close()


if __name__ == '__main__': main()
