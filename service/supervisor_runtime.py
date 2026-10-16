"""Real bounded UDS lifecycle adapter. No host operator or daemon authority."""
import json
from pathlib import Path
import socket
import time
from worker.lifecycle import Observation, State
from worker.runtime_protocol import VERSION, session, authenticate, send, receive
from service.worker_transport import TransportObservation


class SupervisorRuntime:
    def __init__(self, ipc='/ipc', interrupt='none'):
        self.ipc = Path(ipc)
        self.interrupt = interrupt

    def call(self, op, body, timeout=1.5):
        if not 0 < timeout <= 2:
            raise ValueError('bounded_communication_required')
        cutoff = time.monotonic()+timeout
        secret = (self.ipc/'lifecycle.key').read_text().strip()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.ipc/'lifecycle.sock'))
            authenticate(connection, session(secret, 0, 10001), cutoff, control=True)
            send(connection, dict(version=VERSION, op=op, body=body), cutoff)
            if self.interrupt == op+'-sent':
                import os
                os._exit(17)
            result = receive(connection, cutoff)
        if set(result) != {'ok', 'value'} or result['ok'] is not True:
            raise RuntimeError('lifecycle_refused_or_quarantined')
        return result['value']

    def prepare(self, authority, fixture='silence'):
        return self.call('prepare', dict(snapshot=json.loads(authority._snapshot),
            binding=authority.binding, fixture=fixture))['runtime_id']

    def start(self, identity):
        return self.call('start', dict(runtime_id=identity))

    def observe(self, runtime_id, timeout):
        value = self.call('observe', dict(runtime_id=runtime_id), min(timeout, 2))
        if set(value) != {'runtime_id', 'state'} or value['runtime_id'] != runtime_id:
            raise ValueError('unbound_observation')
        return Observation(runtime_id, State(value['state']))

    def terminate(self, runtime_id, timeout):
        self.call('stop', dict(runtime_id=runtime_id), min(timeout, 2))

    def kill(self, runtime_id, timeout):
        # Escalation remains the parent watchdog's fixed grace, never client signals.
        self.terminate(runtime_id, timeout)

    def collect(self, authority, runtime_id):
        value = self.call('collect', dict(runtime_id=runtime_id))
        receipt = value['receipt']
        if receipt is None:
            return None
        if (receipt['runtime_id'] != runtime_id or receipt['binding'] != authority.binding or
                receipt['reaped'] is not True):
            raise ValueError('unbound_receipt')
        proposal = value['proposal']
        if proposal is None:
            return receipt, None
        authority.check(proposal['message'], 'result')
        from contracts import validate as oracle
        oracle.validate('result_manifest', proposal['manifest'])
        message, manifest = proposal['message'], proposal['manifest']
        for key in ('job_id','attempt_id','generation','owner_id','input_sha256','staging_id'):
            if manifest[key] != message[key]:
                raise ValueError('manifest_tuple')
        return receipt, TransportObservation('proposal', 'durable_parent_receipt', None,
                                            message, manifest, time.monotonic())
