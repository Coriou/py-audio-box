"""W2a store/evidence policy tests; no claim of runtime supervision."""
import copy
import subprocess
import sys

import pytest

from service.owner_store import OwnerStore, Fenced
from service.worker_supervisor import Supervisor
from service.worker_transport import Authority
from tests.release.test_worker_boundary import fixture_authority
from worker.lifecycle import Observation, State


class Runtime:
    def __init__(self, states):
        self.states = iter(states)
        self.calls = []

    def observe(self, identity, timeout):
        self.calls.append(('observe', identity, timeout))
        value = next(self.states)
        if isinstance(value, Exception):
            raise value
        return Observation(identity, value)

    def terminate(self, identity, timeout):
        self.calls.append(('terminate', identity, timeout))

    def kill(self, identity, timeout):
        self.calls.append(('kill', identity, timeout))


def other(authority):
    job = copy.deepcopy(authority.job)
    job['attempts'][-1]['owner_id'] += '-new'
    return Authority(job, authority.context, authority.allowed_assets, authority.profiles)


def test_intent_survives_process_death_before_registration(tmp_path):
    path = tmp_path/'owners.sqlite'
    script = '''import os,sys
from service.owner_store import OwnerStore
from tests.release.test_worker_boundary import fixture_authority
s=OwnerStore(sys.argv[1]);s.intent(fixture_authority());os._exit(17)
'''
    assert subprocess.run([sys.executable, '-c', script, str(path)], timeout=10).returncode == 17
    store = OwnerStore(path)
    try:
        current = store.current()
        assert current['runtime_id'] is None and current['state'] == 'unknown'
        assert Supervisor(store, Runtime([])).stop() == current
        # Use persisted tuple: fixture_authority generates a fresh deadline.
        with pytest.raises(Fenced): store.intent(fixture_authority())
    finally:
        store.close()


def test_reconnect_preserves_authority_deadline_and_does_not_spawn(tmp_path):
    path = tmp_path/'owners.sqlite'
    authority = fixture_authority()
    store = OwnerStore(path)
    store.intent(authority)
    original = store.bind(authority, 'daemon-A/container-immutable-A')
    store.close()
    store = OwnerStore(path)
    try:
        runtime = Runtime([State.ALIVE, State.ALIVE])
        supervisor = Supervisor(store, runtime)
        for _ in range(2):
            current = supervisor.reconnect()
            assert current['authority'] == original['authority']
            assert current['runtime_id'] == original['runtime_id']
        assert store.intent(authority) == current
        assert all(c[0] == 'observe' for c in runtime.calls)
        with pytest.raises(Fenced): store.intent(other(authority))
        with pytest.raises(Fenced): store.bind(authority, 'replacement')
    finally:
        store.close()


@pytest.mark.parametrize('states,actions,final', [
    ([State.ALIVE, State.STOPPED], ['observe','terminate','observe'], State.STOPPED),
    ([State.ALIVE, State.ALIVE, State.STOPPED], ['observe','terminate','observe','kill','observe'], State.STOPPED),
    ([OSError(), OSError(), OSError()], ['observe','terminate','observe','kill','observe'], State.UNKNOWN),
    ([State.UNKNOWN, State.UNKNOWN, State.UNKNOWN], ['observe','terminate','observe','kill','observe'], State.UNKNOWN),
])
def test_stop_requires_observed_termination(tmp_path, states, actions, final):
    store = OwnerStore(tmp_path/'owners.sqlite')
    try:
        authority = fixture_authority()
        store.intent(authority);store.bind(authority, 'exact-runtime')
        runtime = Runtime(states)
        assert Supervisor(store, runtime, timeout=.2).stop()['state'] == final.value
        assert [c[0] for c in runtime.calls] == actions
        assert all(c[1:] == ('exact-runtime', .2) for c in runtime.calls)
        if final == State.UNKNOWN:
            with pytest.raises(Fenced): store.intent(other(authority))
        else:
            assert store.intent(other(authority))['runtime_id'] is None
    finally:
        store.close()


def test_stale_runtime_cannot_release_or_revive_lane(tmp_path):
    store = OwnerStore(tmp_path/'owners.sqlite')
    try:
        authority = fixture_authority()
        store.intent(authority);store.bind(authority, 'runtime-A')
        with pytest.raises(Fenced): store.observe(Observation('runtime-B', State.STOPPED))
        with pytest.raises(TypeError): store.observe({'runtime_id':'runtime-A','state':'stopped-and-reaped'})
        store.observe(Observation('runtime-A', State.STOPPED))
        with pytest.raises(Fenced): store.observe(Observation('runtime-A', State.ALIVE))
        assert Supervisor(store, Runtime([])).reconnect()['state'] == State.STOPPED.value
        store.intent(other(authority))
        with pytest.raises(Fenced): store.observe(Observation('runtime-A', State.STOPPED))
    finally:
        store.close()


def test_runtime_mismatch_quarantines_and_control_is_exclusive(tmp_path):
    path = tmp_path/'owners.sqlite'
    store = OwnerStore(path)
    try:
        with pytest.raises(BlockingIOError): OwnerStore(path)
        authority = fixture_authority()
        store.intent(authority);store.bind(authority, 'runtime-A')
        runtime = Runtime([])
        runtime.observe = lambda *_: Observation('stale-runtime', State.STOPPED)
        assert Supervisor(store, runtime).reconnect()['state'] == State.UNKNOWN.value
        with pytest.raises(Fenced): store.intent(other(authority))
    finally:
        store.close()
