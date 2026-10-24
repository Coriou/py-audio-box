"""Private W1 control adapter. Not connected to Core or any production route."""
from dataclasses import dataclass
import json
from datetime import datetime, timezone
import time

from contracts import validate as oracle
from jsonschema.exceptions import ValidationError
from worker.transport import (PeerEvidence, Session, Refused, Uncertain, authenticate,
                              canonical, receive, send)
import hashlib


@dataclass(frozen=True, init=False)
class Authority:
    _snapshot: bytes

    def __init__(self, job, context, allowed_assets, profiles):
        oracle.stored_context(job, context)
        from worker.runtime_protocol import execution
        execution(context)
        object.__setattr__(self, '_snapshot', canonical([job, context, allowed_assets, profiles]))

    # Return detached values: neither the caller nor an accessor can retarget allocation.
    job = property(lambda self: json.loads(self._snapshot)[0])
    context = property(lambda self: json.loads(self._snapshot)[1])
    allowed_assets = property(lambda self: tuple(json.loads(self._snapshot)[2]))
    profiles = property(lambda self: tuple(json.loads(self._snapshot)[3]))

    @property
    def binding(self):
        return hashlib.sha256(self._snapshot).hexdigest()

    def message(self):
        job = self.job
        attempt = job['attempts'][-1]
        return dict(protocol_version='1.3.0', kind='execute', job_id=job['id'],
                    attempt_id=attempt['id'], generation=job['generation'],
                    input_sha256=job['input_sha256'], input_asset_ids=list(self.allowed_assets),
                    staging_id=attempt['staging_id'], owner_id=attempt['owner_id'],
                    identity=self.context['plan']['resolved']['identity'], deadline=job['deadline'],
                    stage='loading', result_asset_ids=[], plan=self.context['plan'])

    def check(self, message, kind):
        if not isinstance(message, dict) or message.get('kind') != kind:
            raise Refused('unexpected_message')
        oracle.worker_message(message, self.job, self.allowed_assets,
                              transport_owner=self.job['attempts'][-1]['owner_id'],
                              context=self.context, current_profiles=self.profiles)

    def cutoff(self, maximum=5):
        budget = (datetime.fromisoformat(self.job['deadline']) - datetime.now(timezone.utc)).total_seconds()
        return time.monotonic() + min(maximum, max(0, budget))


@dataclass(frozen=True)
class TransportObservation:
    status: str
    reason: str
    peer: PeerEvidence | None = None
    message: dict | None = None
    manifest: dict | None = None
    received_monotonic: float | None = None


def exchange(connection, authority, session, cutoff, *, fault=lambda _: None):
    evidence = None
    try:
        if session.binding != authority.binding:
            raise Refused('unbound_session')
        message = authority.message()
        authority.check(message, 'execute')
        evidence = authenticate(connection, session, cutoff, control=True)
        fault('presend')
        send(connection, message, cutoff)
        fault('postsend')
        proposal = receive(connection, cutoff)
        if not isinstance(proposal, dict) or set(proposal) != {'message', 'manifest'}:
            raise Refused('unexpected_result')
        authority.check(proposal['message'], 'result')
        oracle.validate('result_manifest', proposal['manifest'])
        # This checks relations, never claims actual media verification.
        manifest = proposal['manifest']
        attempt = authority.job['attempts'][-1]
        for key, value in dict(job_id=authority.job['id'], attempt_id=attempt['id'],
                               generation=attempt['generation'], owner_id=attempt['owner_id'],
                               staging_id=attempt['staging_id'], input_sha256=authority.job['input_sha256'],
                               operation=authority.job['operation']).items():
            if manifest[key] != value:
                raise Refused('manifest_tuple')
        if time.monotonic() >= cutoff:
            raise Uncertain('timeout')
        return TransportObservation('proposal', 'received_only', evidence, proposal['message'],
                                    manifest, time.monotonic())
    except (Refused, oracle.ContractError, ValidationError, ValueError, KeyError, TypeError, RecursionError):
        return TransportObservation('refused', 'protocol_or_session_refused', evidence)
    except (Uncertain, OSError, TimeoutError):
        return TransportObservation('uncertain', 'disconnect_or_timeout', evidence)
