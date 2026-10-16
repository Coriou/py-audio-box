"""Dormant W2 evidence policy. No spawn, transport, publication or cleanup API."""
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
