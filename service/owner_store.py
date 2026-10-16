"""Separate private W2 store. One control writer; no worker or core DB access."""
import fcntl
import json
import sqlite3
import threading
from pathlib import Path
from service.worker_transport import Authority

from worker.transport import canonical
from worker.lifecycle import Observation, State


class Fenced(RuntimeError):
    pass


class OwnerStore:
    def __init__(self, path):
        self._thread = threading.get_ident()
        path = Path(path)
        self._lock = open(str(path) + '.lock', 'a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(path)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS lane (id INTEGER PRIMARY KEY CHECK(id=1), authority BLOB NOT NULL, runtime TEXT, state TEXT NOT NULL)')
            self.db.commit()
        except BaseException:
            self._lock.close()
            raise

    def close(self):
        self.db.close()
        self._lock.close()

    def assert_owner(self):
        if threading.get_ident() != self._thread:
            raise Fenced("serialized_control_owner_required")

    def current(self):
        self.assert_owner()
        row = self.db.execute('SELECT authority,runtime,state FROM lane WHERE id=1').fetchone()
        if row is None:
            return None
        return dict(authority=json.loads(row[0]), runtime_id=row[1], state=row[2])

    def intent(self, authority):
        if not isinstance(authority, Authority):
            raise TypeError("validated_W1_authority_required")
        authority = allocation(authority)
        encoded = canonical(authority)
        with self.db:
            old = self.current()
            if old and canonical(old['authority']) == encoded:
                return old
            if old and old['state'] != State.STOPPED.value:
                raise Fenced('owner_not_confirmed_terminated')
            self.db.execute('INSERT OR REPLACE INTO lane VALUES (1,?,NULL,?)',
                            (encoded, State.UNKNOWN.value))
        return self.current()

    def bind(self, authority, runtime_id):
        if not isinstance(authority, Authority):
            raise TypeError("validated_W1_authority_required")
        authority = allocation(authority)
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError('runtime_identity_required')
        with self.db:
            old = self.current()
            if old is None or canonical(old['authority']) != canonical(authority):
                raise Fenced('stale_authority')
            if old['runtime_id'] is not None or old['state'] == State.STOPPED.value:
                raise Fenced('runtime_already_bound')
            self.db.execute('UPDATE lane SET runtime=? WHERE id=1', (runtime_id,))
        return self.current()

    def observe(self, evidence: Observation):
        if not isinstance(evidence, Observation) or not isinstance(evidence.state, State):
            raise TypeError('typed_runtime_evidence_required')
        with self.db:
            old = self.current()
            if old is None or old['runtime_id'] is None or old['runtime_id'] != evidence.runtime_id:
                raise Fenced('runtime_identity_mismatch')
            if old['state'] == State.STOPPED.value and evidence.state != State.STOPPED:
                raise Fenced('terminated_identity_cannot_revive')
            self.db.execute('UPDATE lane SET state=? WHERE id=1', (evidence.state.value,))
        return self.current()


def allocation(authority):
    """Persist the exact W1 execute plus immutable authority digest."""
    message = authority.message()
    authority.check(message, "execute")
    return dict(message=message, binding=authority.binding)
