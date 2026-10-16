"""Serialized synthetic owner lifecycle, W1 reconnect and verification fencing.

Runtime creation/start are trusted callbacks; publication/cleanup stay with control.
"""
from worker.lifecycle import Observation, State


class Supervisor:
    def __init__(self, store, runtime, *, timeout=1.0):
        if not 0 < timeout <= 5:
            raise ValueError('bounded_timeout_required')
        self.store, self.runtime, self.timeout = store, runtime, timeout

    def reconnect(self):
        current = self.store.current()
        if current is None or current['runtime_id'] is None:
            return current  # Unregistered dispatch remains quarantined.
        if current['state'] == State.STOPPED.value:
            return current
        identity = current['runtime_id']
        try:
            evidence = self.runtime.observe(identity, self.timeout)
            if evidence.runtime_id != identity or not isinstance(evidence.state, State):
                raise ValueError('unbound_runtime_observation')
        except Exception:
            evidence = Observation(identity, State.UNKNOWN)
        return self.store.observe(evidence)

    def stop(self):
        current = self.reconnect()
        if current is None or current['runtime_id'] is None or current['state'] == State.STOPPED.value:
            return current
        identity = current['runtime_id']
        for action in (self.runtime.terminate, self.runtime.kill):
            try:
                action(identity, self.timeout)
            except Exception:
                pass
            current = self.reconnect()
            if current['state'] == State.STOPPED.value:
                break
        return current

    def dispatch(self, authority, prepare, start, *, fault=lambda _: None):
        """Persist before any runtime callback. Prepare creates a stopped runtime;
        bind BEFORE physical start closes the spawn/registration gap. Repeated
        intent never invokes either callback (including after a lost start ACK).
        Unbound interrupted intent stays quarantined for explicit operator recovery.
        """
        from service.owner_store import allocation, Fenced
        from worker.transport import canonical
        old = self.store.current()
        if old and canonical(old['authority']) == canonical(allocation(authority)):
            return self.reconnect()
        self.store.intent(authority)
        fault('prespawn')
        identity = prepare()
        fault('preregistration')
        self.store.bind(authority, identity)
        fault('prestart')
        start(identity)
        fault('postspawn')
        return self.reconnect()

    def exchange(self, connection, authority, session, *, fault=lambda _: None):
        """Serialized control owner receives a proposal; never publishes here."""
        from service.owner_store import allocation, Fenced
        from service.worker_transport import exchange
        from worker.transport import canonical
        current = self.reconnect()
        if current is None or current['state'] != State.ALIVE.value or canonical(current['authority']) != canonical(allocation(authority)):
            raise Fenced('live_exact_allocation_required')
        result = exchange(connection, authority, session, authority.cutoff(), fault=fault)
        return result

    def verify(self, observation, verifier, *, fault=lambda _: None):
        """No adoption API: verification runs on the store's serialized owner only,
        after independently observed termination. Callback remains control-owned.
        """
        from service.owner_store import Fenced
        self.store.assert_owner()
        fault('observation_reap')
        current = self.reconnect()
        if current is None or current['state'] != State.STOPPED.value:
            raise Fenced('termination_required_before_verification')
        if observation.status != 'proposal':
            raise Fenced('proposal_required')
        message = current['authority']['message']
        for field in ('job_id', 'attempt_id', 'generation', 'owner_id', 'input_sha256', 'staging_id', 'deadline'):
            if observation.message[field] != message[field]:
                raise Fenced('stale_proposal')
        fault('preverification')
        return verifier(observation)
