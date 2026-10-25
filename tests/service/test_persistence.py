"""A1 real Core publication fault boundaries, disposable synthetic roots only."""
import hashlib
import json
import os
import sqlite3
import stat
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
    assert core.db.execute('PRAGMA user_version').fetchone() == (2,)
    assert core.db.execute('SELECT * FROM p1_format').fetchall() == [(2, 'synthetic-core')]


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
        if kind == 'file_fsync':
            fsync = os.fsync
            def denied_file(fd):
                if stat.S_ISREG(os.fstat(fd).st_mode): raise OSError('file sync denied')
                fsync(fd)
            patch.setattr(persistence.os, 'fsync', denied_file)
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
        with pytest.raises(sqlite3.OperationalError):
            core.db.execute('CREATE TABLE forbidden_migration(value)')
        assert core.job(job['id']) == terminal and core.events(job['id']) == events
        assert core.artifact(artifact['id']) == artifact
        assert core.submit_synthetic('p', 'a', context) == terminal
        with pytest.raises(persistence.PublicationQuarantine, match='inspect_only'):
            core.submit_synthetic('p', 'new', context)
        core._commit_publication(job['id'])
        assert core.events(job['id']) == events


def test_recovery_stat_io_error_is_not_incomplete_media(core, context, monkeypatch):
    from pathlib import Path
    job, result = pending(core, context, 'after_adoption')
    destination, temporary = names(core)
    lstat = Path.lstat
    def denied(path, *args, **kwargs):
        if path == destination: raise OSError(5, 'stat I/O denied')
        return lstat(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'lstat', denied)
        assert core.reconcile()
        assert core.job(job['id'])['state'] == 'running'
        assert not core.db.execute('SELECT * FROM assets').fetchall()
    reopen(core)
    assert not core.reconcile() and core.job(job['id'])['state'] == 'succeeded'
    reopen(core)
    assert not core.reconcile()


def test_intent_namespace_sync_failure_precedes_temp_create(core, context, monkeypatch):
    import service.core as module
    job, result = proposal(core, context)
    sync = module.sync_dir
    def denied(path):
        if path == core.root: raise OSError('SQLite namespace sync denied')
        sync(path)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'sync_dir', denied)
        with pytest.raises(OSError, match='namespace sync denied'): core._publish(job['id'], result)
    assert core.db.execute('SELECT count(*) FROM publication_work').fetchone() == (1,)
    assert not list((core.root/'assets').iterdir())
    for _ in range(2):
        reopen(core)
        assert not core.reconcile() and core.job(job['id'])['state'] == 'failed'


# A2 uses actual Core rows and managed filesystem names, never a shadow store.
def test_inventory_terminal_integrity_and_history(core, context):
    from service.reconciliation import Reconciliation
    job, result = proposal(core, context); core._publish(job['id'], result)
    terminal = core.job(job['id']); events = core.events(job['id'])
    asset_id = result['manifest']['outputs'][0]['id']
    media = core.root/'assets'/asset_id
    inventory = Reconciliation(core)
    assert not inventory.reconcile_startup()['blocked']
    media.write_bytes(b'corrupt')
    summary = inventory.reconcile_startup()
    assert summary['complete'] and summary['blocked'] and 'unavailable' in summary['reasons']
    assert core.job(job['id']) == terminal and core.events(job['id']) == events
    with pytest.raises(Exception): core.asset(asset_id)
    with pytest.raises(Exception): core.artifact(terminal['artifact_ids'][0])
    media.unlink()
    assert inventory.reconcile_startup()['counts']['missing_hash_mismatched_terminal'] == 1
    media.write_bytes(b'corrupt')
    assert inventory.reconcile_startup()['blocked']
    assert core.db.execute('SELECT count(*) FROM p1_inventory_history').fetchone()[0] >= 4
    assert core.job(job['id']) == terminal and core.events(job['id']) == events


def test_inventory_pages_generation_and_filesystem_changes(core):
    from service.reconciliation import Reconciliation
    for n in range(23): (core.root/'assets'/('asset-orphan-'+str(n))).write_bytes(b'x')
    inventory = Reconciliation(core)
    summary = inventory.reconcile_startup()
    assert summary['complete'] and summary['counts'] == {'adopted_orphan':23}
    page = inventory.inventory_page(summary['scan_id'])
    assert len(page['entries']) == 20 and page['next_cursor']
    assert len(inventory.inventory_page(summary['scan_id'],page['next_cursor'])['entries']) == 3
    with pytest.raises(ValueError, match='cursor'): inventory.inventory_page(summary['scan_id'], 'tampered')
    old = summary['scan_id']; summary = inventory.reconcile_startup()
    with pytest.raises(ValueError, match='scan'): inventory.inventory_page(old)
    (core.root/'assets'/'new').write_bytes(b'x')
    with pytest.raises(ValueError, match='stale'): inventory.inventory_page(summary['scan_id'])


@pytest.mark.parametrize('budget', [{'entries':1}, {'byte_budget':1}, {'seconds':0}])
def test_inventory_incomplete_never_cleanup(core, budget):
    from service.reconciliation import Reconciliation
    inventory = Reconciliation(core, **budget)
    summary = inventory.reconcile_startup()
    assert not summary['complete'] and summary['blocked'] and not summary['cleanup_allowed']
    assert summary['reasons'] == ['inventory_budget']


def test_inventory_unknown_symlink_hardlink_no_traversal(core, tmp_path):
    from service.reconciliation import Reconciliation
    outside = tmp_path/'outside'; outside.mkdir(); (outside/'private').write_bytes(b'preserve')
    (core.root/'staging'/'unowned').symlink_to(outside, target_is_directory=True)
    first=core.root/'assets'/'asset-unknown'; first.write_bytes(b'bytes')
    os.link(first, core.root/'assets'/'third')
    inventory=Reconciliation(core); summary=inventory.reconcile_startup()
    assert summary['complete'] and summary['blocked']
    assert summary['counts']['unknown_foreign'] == 3
    assert not any('private' in r[0] for r in core.db.execute('SELECT record FROM p1_inventory_objects'))
    assert (outside/'private').read_bytes() == b'preserve' and first.stat().st_nlink == 2


def test_inventory_all_reference_sources_and_unknown_future_schema(core, context):
    from service.reconciliation import Reconciliation
    job, result = proposal(core, context); core._publish(job['id'], result)
    asset_id=result['manifest']['outputs'][0]['id']
    # Historical revision and frozen context are deliberately outside the selected take.
    core.db.execute('INSERT INTO revisions VALUES(?,?,?)',('project-evidence',1,json.dumps({'assets':[result['manifest']['outputs'][0]]})))
    core.db.execute('INSERT INTO revisions VALUES(?,?,?)',('project-evidence',2,'{}'))
    inventory=Reconciliation(core); summary=inventory.reconcile_startup()
    row=next(json.loads(r[0]) for r in core.db.execute('SELECT record FROM p1_inventory_objects') if json.loads(r[0])['name']==asset_id)
    assert any(r.startswith('revisions:') for r in row['references'])
    assert any(r.startswith('publications:') for r in row['references'])
    assert any(r.startswith('artifacts:') for r in row['references'])
    core.db.execute('CREATE TABLE future_cleanup_recovery_records(asset_id TEXT)')
    core.db.execute('INSERT INTO future_cleanup_recovery_records VALUES(?)',(asset_id,))
    summary=inventory.reconcile_startup()
    assert summary['blocked'] and 'unknown_reference_schema' in summary['reasons']


def test_inventory_persistence_failure_readable_and_owner_only(core, monkeypatch):
    from service.reconciliation import Reconciliation
    import concurrent.futures
    inventory=Reconciliation(core)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        with pytest.raises(ValueError, match='owner'): pool.submit(inventory.reconcile_startup).result()
    core.db.execute('PRAGMA query_only=ON')
    assert inventory.reconcile_startup()['reasons'] == ['inventory_persistence_failed']
    assert core.jobs('p') == []


def test_inventory_first_observed_and_resolved_diagnostic_retained(core):
    from service.reconciliation import Reconciliation
    media=core.root/'assets'/'asset-orphan';media.write_bytes(b'bytes')
    inventory=Reconciliation(core);inventory.reconcile_startup()
    first=core.db.execute('SELECT first_observed,record FROM p1_inventory_objects').fetchone()
    inventory.reconcile_startup()
    assert core.db.execute('SELECT first_observed FROM p1_inventory_objects').fetchone()[0]==first[0]
    media.unlink()
    summary=inventory.reconcile_startup()
    assert summary['blocked'] and 'unacknowledged_integrity' in summary['reasons']
    assert core.db.execute('SELECT record FROM p1_inventory_objects').fetchone()[0]==first[1]


def test_inventory_database_change_refuses_cursor(core):
    from service.reconciliation import Reconciliation
    inventory=Reconciliation(core);summary=inventory.reconcile_startup()
    core.db.execute('INSERT INTO revisions VALUES(?,?,?)',('project-diagnostic',1,'{}'))
    with pytest.raises(ValueError,match='stale'):inventory.inventory_page(summary['scan_id'])


def test_inventory_managed_ids_without_asset_prefix(core, context):
    from service.reconciliation import Reconciliation
    job,result=proposal(core,context)
    original=result['manifest']['outputs'][0]['id']
    result=json.loads(json.dumps(result).replace(original,'legacy.voice:output.1'))
    core._publish(job['id'],result)
    core.db.execute('INSERT INTO revisions VALUES(?,?,?)',('project-history',1,json.dumps({'clips':[{'asset_id':'legacy.voice:output.1'}]})))
    inventory=Reconciliation(core);summary=inventory.reconcile_startup()
    assert summary['complete'] and not summary['blocked']
    row=next(json.loads(r[0]) for r in core.db.execute('SELECT record FROM p1_inventory_objects') if json.loads(r[0])['name']=='legacy.voice:output.1')
    assert any(r.startswith('revisions:') for r in row['references'])
    assert row['owner']['job_id']==job['id']
    (core.root/'assets'/'legacy.voice:output.1').unlink()
    summary=inventory.reconcile_startup()
    assert summary['counts']['missing_hash_mismatched_terminal']==1


def test_inventory_real_frozen_render_context_and_all_project_revisions(core):
    from service.reconciliation import Reconciliation
    from contracts import validate as oracle
    transcript=json.loads((FIXTURES/'transcript-project_render.json').read_text())
    job=transcript['job'];ctx=transcript['context'];oracle.stored_context(job,ctx)
    with core.transaction():
        core.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context) VALUES(?,?,?,?,?,?,?)',
            (job['id'],'p',job['operation'],'frozen',job['input_sha256'],json.dumps(job),json.dumps(ctx)))
    project=ctx['plan']['project'];core.create_project(project)
    core.edit_text(project['id'],1,project['blocks'][0]['id'],'later text')
    inventory=Reconciliation(core);summary=inventory.reconcile_startup()
    assert summary['complete'] and summary['blocked']
    asset_id=ctx['plan']['assets'][0]['id']
    row=next(json.loads(r[0]) for r in core.db.execute('SELECT record FROM p1_inventory_objects') if json.loads(r[0])['name']==asset_id)
    assert 'jobs:'+job['id'] in row['references']
    assert 'revisions:'+project['id']+':1' in row['references']
    assert 'revisions:'+project['id']+':2' in row['references']


def test_inventory_authored_asset_prefix_is_not_reference(core, context):
    from service.reconciliation import Reconciliation
    context['plan']['request']['text'] = 'asset-management is useful.'
    job = core.submit_synthetic('asset-principal', 'asset-key', context)
    project = json.loads((FIXTURES/'transcript-project_render.json').read_text())['context']['plan']['project']
    # Authored history text is unrelated to managed asset identity.
    project['blocks'][0]['text'] = 'asset-history prose'
    core.create_project(project)
    # Render fixture dependencies are genuine refs; supply no fake prose ref.
    inventory = Reconciliation(core)
    inventory.reconcile_startup()
    records = [json.loads(r[0]) for r in core.db.execute('SELECT record FROM p1_inventory_objects')]
    assert not any(r['name'].startswith(('asset-management','asset-principal','asset-key','asset-history')) for r in records)
    assert core.job(job['id'])['state'] == 'queued'


def test_inventory_clean_text_restart(core, context):
    from service.reconciliation import Reconciliation
    context['plan']['request']['text'] = 'asset-management is useful.'
    job = core.submit_synthetic('asset-principal', 'asset-key', context)
    for _ in range(3):
        assert not Reconciliation(core).reconcile_startup()['blocked']
        assert core.submit_synthetic('asset-principal', 'asset-key', context) == job
        reopen(core)


def test_inventory_begin_denial_has_zero_observations(core, monkeypatch):
    from service.reconciliation import Reconciliation
    inventory = Reconciliation(core)
    observed = []
    monkeypatch.setattr(inventory, '_collect', lambda: observed.append(True))
    core.db.execute('PRAGMA query_only=ON')
    assert inventory.reconcile_startup()['blocked']
    assert observed == []
    core.db.execute('PRAGMA query_only=OFF')
    reopen(core)
    assert not Reconciliation(core).reconcile_startup()['blocked']


@pytest.mark.parametrize('point', ['fault_denial','completion_denial','healthy_death','healthy_mid_death','healthy_after_death','page_denial'])
def test_inventory_precommitted_epoch_immediate_death(tmp_path, point):
    root = tmp_path/'core'
    script = r'''
import os, sys
from service.core import Core
from service.reconciliation import Reconciliation
c=Core(sys.argv[1], historical_local=True)
i=Reconciliation(c)
assert not i.reconcile_startup()['blocked']
point=sys.argv[2]
if point == 'fault_denial': (c.root/'assets'/'asset-transient-orphan').write_bytes(b'x')
original=i._collect
def barrier():
    # Independent reader proves begin COMMITTED before any observation.
    import sqlite3
    with sqlite3.connect(c.root/'core.sqlite') as reader:
        assert reader.execute('SELECT state FROM p1_integrity').fetchone()[0]=='checking'
    if point == 'healthy_death': os._exit(23)
    if point == 'healthy_mid_death':
        charge=i._charge
        def mid(size=0):
            charge(size)
            if i.entries >= 4: os._exit(23)
        i._charge=mid
    result=original()
    if point == 'healthy_after_death': os._exit(23)
    if point == 'fault_denial': assert 'no_publication_provenance' in i.reasons
    c.db.execute('PRAGMA query_only=ON')
    return result
i._collect=barrier
if point == 'page_denial':
    try: i.inventory_page(i.scan_id)
    except ValueError: pass
else:
    assert i.reconcile_startup()['blocked']
os._exit(23)
'''
    result = subprocess.run([sys.executable,'-c',script,str(root),point], env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}, timeout=10)
    assert result.returncode == 23
    (root/'assets'/'asset-transient-orphan').unlink(missing_ok=True)
    for _ in range(2):
        c = ActiveCore(root, historical_local=True)
        from service.reconciliation import Reconciliation
        i = Reconciliation(c)
        i._collect = lambda: pytest.fail('inherited open epoch must not observe')
        assert i.reconcile_startup()['blocked']
        assert c.db.execute('SELECT state FROM p1_integrity').fetchone()[0] == 'checking'
        c.close()


def test_inventory_page_change_durable(core):
    from service.reconciliation import Reconciliation
    i=Reconciliation(core); scan=i.reconcile_startup()['scan_id']
    (core.root/'assets'/'asset-transient').write_bytes(b'x')
    with pytest.raises(ValueError, match='stale'): i.inventory_page(scan)
    (core.root/'assets'/'asset-transient').unlink()
    for _ in range(2):
        reopen(core)
        assert Reconciliation(core).reconcile_startup()['blocked']


def test_inventory_old_diagnostics_without_authority_are_unresolved(core):
    from service.reconciliation import Reconciliation
    assert not Reconciliation(core).reconcile_startup()['blocked']
    core.db.execute('DROP TABLE p1_integrity')
    reopen(core)
    assert Reconciliation(core).reconcile_startup()['blocked']


def test_inventory_design_description_and_instruction_are_prose(core):
    from contracts import validate as oracle
    from service.reconciliation import Reconciliation
    ctx=json.loads((FIXTURES/'transcript-voice_design.json').read_text())['context']
    ctx['plan']['request']['description']='asset-description for a voice'
    ctx['plan']['request']['reference_text']='asset-reference prose'
    ctx['plan']['resolved']['reuse_configuration']['reference_text']='asset-reference prose'
    ctx['plan']['resolved']['settings']['instruction']='asset-description for a voice'
    ctx['plan']['resolved']['settings_sha256']=oracle.canonical_hash(ctx['plan']['resolved']['settings'])
    # Core's synthetic submit only executes speech; retain a valid design snapshot
    # through its real stored jobs shape, as for frozen render coverage.
    job=json.loads((FIXTURES/'transcript-voice_design.json').read_text())['job']
    job['input_sha256']=oracle.plan_hash(ctx['plan'])
    oracle.stored_context(job,ctx)
    core.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context) VALUES(?,?,?,?,?,?,?)',
        (job['id'],'asset-principal',job['operation'],'asset-design-key',job['input_sha256'],json.dumps(job),json.dumps(ctx)))
    for _ in range(2):
        summary=Reconciliation(core).reconcile_startup()
        assert summary['complete'] and summary['blocked']
        assert summary['reasons']==(['unavailable'] if _==0 else ['unacknowledged_integrity','unavailable'])
        assert summary['counts']=={'missing_hash_mismatched_terminal':1}
        names=[json.loads(r[0])['name'] for r in core.db.execute('SELECT record FROM p1_inventory_objects')]
        assert not any(name.startswith(('asset-description','asset-reference prose')) for name in names)
        assert core.job(job['id'])==job
        reopen(core)


def test_inventory_diagnostic_aggregate_refused_before_decode(core, monkeypatch):
    import service.reconciliation as module
    i=module.Reconciliation(core);i.reconcile_startup()
    core.db.execute('INSERT INTO p1_inventory_objects VALUES(?,?,?,?,?)',('oversized',0,0,i.scan_id,'x'*4096))
    original=module.json.loads
    def decode(raw, *args, **kwargs):
        assert raw != 'x'*4096, 'oversized diagnostic materialized for decode'
        return original(raw,*args,**kwargs)
    monkeypatch.setattr(module.json,'loads',decode)
    result=module.Reconciliation(core,byte_budget=2048).reconcile_startup()
    assert result['blocked'] and 'inventory_history_budget' in result['reasons']


def inventory_media(core, metadata):
    """Install owned synthetic bytes with matching retained contract metadata."""
    import hashlib
    metadata['sha256'] = hashlib.sha256(b'x').hexdigest()
    core.db.execute('INSERT OR REPLACE INTO assets VALUES(?,?)', (metadata['id'], json.dumps(metadata)))
    (core.root/'assets'/metadata['id']).write_bytes(b'x')


@pytest.mark.parametrize('operation', ['voice_design','voice_prepare'])
def test_inventory_valid_design_lineage_project_restart(core, operation):
    from contracts import validate as oracle
    from service.reconciliation import Reconciliation
    transcript=json.loads((FIXTURES/('transcript-'+operation+'.json')).read_text())
    version=next(e['response'] for e in transcript['exchanges'] if e['operation']=='voice_version_get')
    asset=next(e['response'] for e in transcript['exchanges'] if e['operation']=='asset_get' and e['response']['id']==version['reference_asset_ids'][0])
    inventory_media(core,asset)
    project=json.loads((FIXTURES/'transcript-project_render.json').read_text())['context']['plan']['project']
    for key in ('speakers','blocks','takes','tracks','clips'): project[key]=[]
    project['voices']=[dict(id=version['voice_id'],aliases=[],version_ids=[version['id']],archived=False,revision=1)]
    project['voice_versions'],project['assets']=[version],[asset]
    assert (version['design_lineage'] is None)==(operation=='voice_prepare')
    oracle.validate('project',project)
    core.create_project(project)
    for _ in range(3):
        summary=Reconciliation(core).reconcile_startup()
        assert summary['complete'] and not summary['blocked'], summary
        assert summary['reasons']==[] and summary['counts']=={'committed_referenced':1}
        assert core.project(project['id'])==project
        reopen(core)


@pytest.mark.parametrize('operation', ['voice_prepare','voice_design'])
@pytest.mark.parametrize('missing', [False,True])
def test_inventory_valid_resolved_snapshot_dependencies(core, operation, missing):
    from contracts import validate as oracle
    from service.reconciliation import Reconciliation
    transcript=json.loads((FIXTURES/('transcript-'+operation+'.json')).read_text())
    ctx,job=transcript['context'],transcript['job']
    resolved=ctx['plan']['resolved']
    assert all(resolved[k] is not None for k in ('reuse_settings','reuse_configuration'))
    assert (resolved['reuse_identity'] is None)==(operation=='voice_prepare')
    prospective=resolved['reuse_configuration']['reference_asset_id']
    for asset in ctx['plan']['assets']: inventory_media(core,asset)
    output=next(e['response'] for e in transcript['exchanges'] if e['operation']=='asset_get' and e['response']['id']==prospective)
    if not missing: inventory_media(core,output)
    job['input_sha256']=oracle.plan_hash(ctx['plan'])
    oracle.stored_context(job,ctx)
    core.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context) VALUES(?,?,?,?,?,?,?)',
        (job['id'],'p',operation,'resolved',job['input_sha256'],json.dumps(job),json.dumps(ctx)))
    for _ in range(3):
        summary=Reconciliation(core).reconcile_startup()
        assert summary['complete'], summary
        expected=['unacknowledged_integrity','unavailable'] if missing and _ else ['unavailable'] if missing else []
        assert summary['reasons']==expected
        assert summary['blocked']==missing
        assert summary['counts'].get('missing_hash_mismatched_terminal',0)==int(missing)
        records=[json.loads(r[0]) for r in core.db.execute('SELECT record FROM p1_inventory_objects')]
        row=next(r for r in records if r['name']==prospective)
        assert row['references']==(['jobs:'+job['id']] if missing else ['assets:'+prospective,'jobs:'+job['id']])
        assert {r['name'] for r in records}=={a['id'] for a in ctx['plan']['assets']}|{prospective}
        reopen(core)


@pytest.mark.parametrize('change', ['variant','field','type','null_type'])
def test_inventory_unknown_snapshot_shapes_refuse(core, context, change):
    from service.reconciliation import Reconciliation
    job=core.submit_synthetic('p','shape',context)
    if change=='variant': context['plan']['resolved']['settings']['kind']='future'
    elif change=='field': context['plan']['resolved']['future_media']={'asset_id':'opaque'}
    elif change=='type': context['plan']['resolved']['reuse_identity']=[]
    else: context['plan']['request']['text']=None
    core.db.execute('UPDATE jobs SET context=? WHERE id=?',(json.dumps(context),job['id']))
    summary=Reconciliation(core).reconcile_startup()
    assert summary['blocked']
    assert set(summary['reasons']) & {'unknown_reference_kind','unknown_reference_schema'}
