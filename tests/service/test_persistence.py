"""A1 real Core publication fault boundaries, disposable synthetic roots only."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from service import persistence
from service.core import Core as ActiveCore
from tests.service.test_core import Core, FIXTURES, context, core, proposal


def pending(core, context, point='before_adoption'):
    job, result = proposal(core, context)
    def stop(where):
        if where == point:
            raise RuntimeError('barrier')
    with pytest.raises(RuntimeError, match='barrier'):
        core._publish(job['id'], result, fault=stop)
    return job, result


def names(core):
    destination, temporary = core.db.execute('SELECT destination,temporary FROM publication_work').fetchone()
    return core.root/'assets'/destination, core.root/'assets'/temporary


def reopen(core):
    root = core.root
    core.close()
    core.__init__(root, historical_local=True)


def test_work_committed_before_exclusive_create(core, context, monkeypatch):
    import builtins
    original = builtins.open
    seen = []
    def check(path, mode='r', *args, **kwargs):
        if mode == 'xb':
            with sqlite3.connect(core.root/'core.sqlite') as reader:
                row = reader.execute('SELECT p.job_id,w.temporary,w.byte_length FROM publications p JOIN publication_work w USING(job_id)').fetchone()
            assert row and row[1] == path.name and row[2] > 0
            seen.append(row)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', check)
    job, result = proposal(core, context)
    core._publish(job['id'], result)
    assert len(seen) == 1 and core.job(job['id'])['state'] == 'succeeded'
    assert core.db.execute('PRAGMA user_version').fetchone() == (1,)
    assert core.db.execute('SELECT * FROM p1_format').fetchall() == [(1, 'synthetic-core')]


@pytest.mark.parametrize('kind', ['short', 'zero', 'partial_error', 'flush', 'file_fsync'])
def test_real_publication_write_failures(core, context, monkeypatch, kind):
    import builtins
    original = builtins.open
    class Stream:
        def __init__(self, stream): self.stream = stream; self.calls = 0
        def __enter__(self): return self
        def __exit__(self, *exc): return self.stream.__exit__(*exc)
        def fileno(self): return self.stream.fileno()
        def flush(self):
            if kind == 'flush': raise OSError('flush denied')
            return self.stream.flush()
        def write(self, data):
            self.calls += 1
            if kind == 'zero': return 0
            if kind == 'partial_error' and self.calls > 1: raise OSError('write denied')
            return self.stream.write(data[:17])
    def wrapped(path, mode='r', *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return Stream(stream) if mode == 'xb' else stream
    job, result = proposal(core, context)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', wrapped)
        if kind == 'file_fsync': patch.setattr(persistence.os, 'fsync', lambda _: (_ for _ in ()).throw(OSError('file sync denied')))
        if kind == 'short': core._publish(job['id'], result)
        else:
            with pytest.raises(OSError): core._publish(job['id'], result)
    if kind == 'short':
        assert core.job(job['id'])['state'] == 'succeeded'
        return
    destination, temporary = names(core)
    assert temporary.exists() and not destination.exists()
    assert not core.db.execute('SELECT * FROM assets').fetchall()
    with pytest.raises(OSError, match='reopen_required'): core._commit_publication(job['id'])
    for _ in range(2):
        reopen(core)
        assert not core.reconcile()
        assert core.job(job['id'])['state'] == 'failed'
        assert temporary.exists()


@pytest.mark.parametrize('point', ['after_link', 'after_adoption'])
def test_actual_process_death_and_repeat_recovery(tmp_path, point):
    root = tmp_path/'fresh'
    script = '''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]),historical_local=True)
x=json.loads(Path(sys.argv[2]).read_text())['context']
c.submit_synthetic('p','a',x);c.start()
def die(point):
    if point==sys.argv[3]:os._exit(79)
c.finish(fault=die)
'''
    assert subprocess.run([sys.executable, '-c', script, str(root), str(FIXTURES/'transcript-speech_generate.json'), point], timeout=10).returncode == 79
    core = Core(root)
    try:
        destination, temporary = names(core)
        assert destination.stat().st_nlink == (2 if point == 'after_link' else 1)
        assert not core.reconcile()
        assert not temporary.exists() and destination.stat().st_nlink == 1
        job = core.jobs('p')[0]['job']; events = core.events(job['id'])
        assert job['state'] == 'succeeded'
        reopen(core)
        assert not core.reconcile() and core.events(job['id']) == events
        core._commit_publication(job['id'])
        assert core.events(job['id']) == events
    finally: core.close()


@pytest.mark.parametrize('alias', ['third', 'wrong', 'symlink', 'unrecorded'])
def test_ambiguous_alias_retained_twice(core, context, alias):
    job, result = pending(core, context, 'after_link')
    destination, temporary = names(core)
    if alias == 'third': os.link(destination, destination.parent/'foreign')
    elif alias == 'wrong':
        temporary.unlink(); temporary.write_bytes(destination.read_bytes())
    elif alias == 'symlink':
        temporary.unlink(); temporary.symlink_to(destination.name)
    else:
        temporary.rename(destination.parent/'unrecorded')
    before = sorted(p.name for p in destination.parent.iterdir())
    for _ in range(2):
        reopen(core)
        assert core.reconcile()
        assert core.job(job['id'])['state'] == 'running'
        assert sorted(p.name for p in destination.parent.iterdir()) == before
        assert not core.db.execute('SELECT * FROM assets').fetchall()


@pytest.mark.parametrize('point', ['after_link', 'after_adoption'])
@pytest.mark.parametrize('barrier', ['file', 'directory'])
def test_recovery_barriers_precede_success(core, context, monkeypatch, point, barrier):
    job, result = pending(core, context, point)
    reopen(core)
    with monkeypatch.context() as patch:
        if barrier == 'file': patch.setattr(persistence.os, 'fsync', lambda _: (_ for _ in ()).throw(OSError('file barrier')))
        else: patch.setattr('service.core.sync_dir', lambda _: (_ for _ in ()).throw(OSError('directory barrier')))
        assert core.reconcile()
        assert core.job(job['id'])['state'] == 'running'
        assert not core.db.execute('SELECT * FROM assets').fetchall()
    reopen(core)
    assert not core.reconcile() and core.job(job['id'])['state'] == 'succeeded'
    reopen(core)
    assert not core.reconcile()


class FaultDB:
    def __init__(self, db, match, *, after=False, rollback=False):
        self.db, self.match, self.after, self.deny_rollback = db, match, after, rollback
        self.fired = False
    def __getattr__(self, name): return getattr(self.db, name)
    def execute(self, sql, *args):
        if self.deny_rollback and sql == 'ROLLBACK': raise sqlite3.OperationalError('rollback denied')
        if not self.fired and sql.startswith(self.match):
            self.fired = True
            if self.after: self.db.execute(sql, *args)
            raise sqlite3.OperationalError('original injected error')
        return self.db.execute(sql, *args)


@pytest.mark.parametrize('statement', ['BEGIN IMMEDIATE', 'INSERT INTO publications', 'INSERT INTO publication_work', 'COMMIT'])
@pytest.mark.parametrize('after', [False, True])
def test_intent_transaction_failure_reopen_state(core, context, statement, after):
    job, result = proposal(core, context)
    real = core.db
    core.db = FaultDB(real, statement, after=after, rollback=True)
    with pytest.raises(sqlite3.OperationalError, match='original injected error'):
        core._publish(job['id'], result)
    assert not list((core.root/'assets').iterdir())
    core.db = real
    reopen(core)
    rows = core.db.execute('SELECT count(*) FROM publications').fetchone()[0]
    work = core.db.execute('SELECT count(*) FROM publication_work').fetchone()[0]
    assert rows == work == (1 if statement == 'COMMIT' and after else 0)
    assert not core.reconcile()
    assert core.job(job['id'])['state'] == 'failed'
    reopen(core)
    assert not core.reconcile()


@pytest.mark.parametrize('after', [False, True])
def test_terminal_commit_ambiguity_no_duplicate_history(core, context, after):
    job, result = pending(core, context, 'after_adoption')
    real = core.db
    core.db = FaultDB(real, 'COMMIT', after=after, rollback=True)
    with pytest.raises(sqlite3.OperationalError, match='original injected error'):
        core._commit_publication(job['id'])
    core.db = real
    reopen(core)
    assert core.job(job['id'])['state'] == ('succeeded' if after else 'running')
    assert not core.reconcile()
    assert [e['state'] for e in core.events(job['id'])] == ['queued', 'running', 'succeeded']
    assert core.db.execute('SELECT count(*) FROM assets').fetchone() == (1,)
    reopen(core)
    assert not core.reconcile()
    assert len(core.events(job['id'])) == 3


def test_old_populated_root_not_enrolled(tmp_path, context):
    root = tmp_path/'old'
    core = Core(root)
    job, result = pending(core, context)
    core.db.execute('DROP TABLE publication_work'); core.db.execute('DROP TABLE p1_format')
    core.close()
    for _ in range(2):
        core = Core(root)
        try:
            assert not core.p1 and core.reconcile()
            assert core.job(job['id'])['state'] == 'running'
            assert not core.db.execute("SELECT name FROM sqlite_master WHERE name='p1_format'").fetchall()
        finally: core.close()


def test_parent_namespace_barriers(tmp_path, monkeypatch):
    calls = []
    from service.core import sync_dir
    def record(path): calls.append(path); sync_dir(path)
    monkeypatch.setattr('service.core.sync_dir', record)
    root = tmp_path/'parent'/'root'
    core = Core(root)
    try:
        assert calls.index(root/'assets') < calls.index(root, calls.index(root/'assets'))
        assert root.parent in calls and tmp_path in calls
        assert calls[-2:] == [root, root.parent]
    finally: core.close()


def test_checkpoint_busy_keeps_committed_wal_success(core, context):
    core.db.execute('PRAGMA wal_autocheckpoint=0')
    reader = sqlite3.connect(core.root/'core.sqlite', isolation_level=None)
    reader.execute('BEGIN'); reader.execute('SELECT * FROM jobs').fetchall()
    job, result = proposal(core, context)
    core._publish(job['id'], result)
    events = core.events(job['id'])
    core.db.execute('PRAGMA busy_timeout=1')
    assert core.db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 1
    assert (core.root/'core.sqlite-wal').stat().st_size > 0
    assert core.job(job['id'])['state'] == 'succeeded'
    # Reopen while the reader pins WAL: no successful checkpoint is required.
    reopen(core)
    assert core.job(job['id'])['state'] == 'succeeded' and core.events(job['id']) == events
    core._commit_publication(job['id'])
    reader.close()
    reopen(core)
    assert core.events(job['id']) == events


def test_recovery_write_denied_retains_intent_twice(core, context):
    job, result = pending(core, context, 'after_adoption')
    for _ in range(2):
        reopen(core)
        core.db.execute('PRAGMA query_only=ON')
        assert core.reconcile()
        assert core.job(job['id'])['state'] == 'running'
        assert not core.db.execute('SELECT * FROM assets').fetchall()
        assert core.db.execute('SELECT count(*) FROM publication_work').fetchone() == (1,)
    reopen(core)
    assert not core.reconcile()
    assert core.job(job['id'])['state'] == 'succeeded'


@pytest.mark.parametrize('fragment', ['control.lock', 'core.sqlite', 'unexplained'])
def test_initialization_fragments_not_reseeded(tmp_path, fragment):
    root = tmp_path/'fragment'; root.mkdir(); (root/fragment).touch()
    for _ in range(2):
        with pytest.raises(persistence.PublicationQuarantine): Core(root)
        assert not (root/'assets').exists()


def test_partial_private_schema_refuses(tmp_path):
    root = tmp_path/'partial'; core = Core(root)
    core.db.execute('DROP TABLE publication_work'); core.close()
    with pytest.raises(persistence.PublicationQuarantine, match='partial_p1_format'): Core(root)


@pytest.mark.parametrize('boundary', ['directory', 'terminal', 'outcome'])
@pytest.mark.parametrize('after', [False, True])
def test_dispatcher_storage_failure_retains_release_ack(tmp_path, monkeypatch, boundary, after):
    from service.dispatcher import Dispatcher
    from tests.service.test_dispatcher import Runtime, ready, wait, running, State
    import service.core as module
    stage = tmp_path/'stage'; stage.mkdir(); runtime = Runtime(stage)
    root = tmp_path/'core'
    fired = []
    real_sync = module.sync_dir
    def fault(point):
        if fired: return
        if boundary == 'directory' and point == 'before_adoption':
            fired.append(point)
            monkeypatch.setattr(module, 'sync_dir', lambda _: (_ for _ in ()).throw(OSError('directory denied')))
        elif (boundary == 'terminal' and point == 'after_adoption') or (boundary == 'outcome' and point == 'after_commit'):
            fired.append(point)
            dispatcher.core.db = FaultDB(dispatcher.core.db, 'COMMIT', after=after, rollback=True)
    dispatcher = Dispatcher(root, staging=stage, runtime=runtime, fault=fault)
    try:
        ready(dispatcher); job = running(dispatcher, runtime)
        runtime.state = State.STOPPED
        wait(dispatcher, lambda: dispatcher.call('status')['persistence_fault'])
        state = dispatcher.call('job', job['id'])['state']
        assert state == ('succeeded' if boundary == 'outcome' or (boundary == 'terminal' and after) else 'running')
        assert runtime.acks == 0 and runtime.starts == 1
        # Repeated owner ticks cannot retry the ambiguous operation or ACK.
        dispatcher.call('status'); dispatcher.call('status')
        assert runtime.acks == 0
    finally:
        dispatcher.close(); monkeypatch.setattr(module, 'sync_dir', real_sync)
    for _ in range(2):
        dispatcher = Dispatcher(root, staging=stage, runtime=runtime)
        try:
            wait(dispatcher, lambda: dispatcher.call('status')['phase'] == 'idle')
            assert dispatcher.call('job', job['id'])['state'] == 'succeeded'
            assert len(dispatcher.call('events', job['id'])) == 3
            assert runtime.starts == runtime.acks == 1
        finally: dispatcher.close()


@pytest.mark.parametrize('boundary', ['root', 'assets', 'database', 'format'])
def test_initialization_barrier_failure_no_reseed(tmp_path, monkeypatch, boundary):
    import service.core as module
    root = tmp_path/'data'
    sync = module.sync_dir
    def fail(path):
        should_fail = ((boundary == 'root' and path == root and not (root/'assets').exists())
            or (boundary == 'assets' and path == root/'assets')
            or (boundary == 'database' and path == root and (root/'core.sqlite').exists() and (root/'core.sqlite').stat().st_size == 0)
            or (boundary == 'format' and path == root and (root/'core.sqlite-wal').exists()))
        if should_fail: raise OSError('initialization barrier')
        sync(path)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'sync_dir', fail)
        with pytest.raises(OSError, match='initialization barrier'): Core(root)
    for _ in range(2):
        if boundary in ('root', 'format'):
            # An empty directory has no identity/DB fragments to reinterpret.
            core = Core(root)
            assert core.p1 and core.db.execute('SELECT * FROM p1_format').fetchall() == [(1, 'synthetic-core')]
            core.close()
        else:
            with pytest.raises(persistence.PublicationQuarantine): Core(root)


def test_old_terminal_history_reads_and_dedup_remain(core, context):
    job, result = proposal(core, context); core._publish(job['id'], result)
    terminal = core.job(job['id']); events = core.events(job['id'])
    artifact = core.artifact(terminal['artifact_ids'][0])
    core.db.execute('DROP TABLE publication_work'); core.db.execute('DROP TABLE p1_format')
    for _ in range(2):
        reopen(core)
        assert not core.p1
        assert core.job(job['id']) == terminal and core.events(job['id']) == events
        assert core.artifact(artifact['id']) == artifact
        assert core.submit_synthetic('p', 'a', context) == terminal
        with pytest.raises(persistence.PublicationQuarantine, match='inspect_only'):
            core.submit_synthetic('p', 'new', context)
        core._commit_publication(job['id'])
        assert core.events(job['id']) == events
