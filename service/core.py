"""Phase 1A synthetic durable domain. Trusted control API, never an HTTP payload API.

One Core owns a data root. Canonical dictionaries retain contract 1.3.0 exactly.
Only preset speech (one variant, no reuse) is executable in this bounded lot.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from uuid import uuid4
import wave

from contracts import validate as oracle
from service import persistence


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def new_id(prefix):
    return prefix + '-' + uuid4().hex


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


class CompletionObserver:
    """Control-only, conservative completion evidence; never a child timestamp.

    Observation at/after cutoff fails even if the OS exit happened earlier.
    No SQLite or publication runs on this thread. Lost evidence on control death
    is handled by ordinary interrupted-job recovery, never inferred from output.
    """
    def __init__(self, process, cutoff):
        self.process, self.cutoff = process, cutoff
        self.expired = False
        self.reaped = False
        self.thread = threading.Thread(target=self._observe, daemon=True)
        self.thread.start()

    def _observe(self):
        try:
            self.process.wait(timeout=max(0, self.cutoff-time.monotonic()))
            self.expired = time.monotonic() >= self.cutoff
        except subprocess.TimeoutExpired:
            self.expired = True
        if self.expired and self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=3)
            self.reaped = True
        except subprocess.TimeoutExpired:
            pass  # finish retains active ownership and quarantines the lane

    def collect(self):
        self.thread.join(timeout=max(0, self.cutoff-time.monotonic())+4)
        if self.thread.is_alive() or not self.reaped:
            raise ValueError('lane_quarantined')
        return self.expired


# One slot owns the entire lifetime, including construction and uncertain close.
# Fixed cardinality only: this is not a whole-process FD or byte budget.
CORE_OWNERSHIP_LIMIT = 64
_core_registry_pid = os.getpid()
_core_registry_mutex = threading.Lock()
_core_registry = [None] * CORE_OWNERSHIP_LIMIT
_CORE_ACTION = '; close Cores on their creating threads or restart the process'


@contextmanager
def _core_bookkeeping():
    # In a fork child the inherited mutex may be locked. Never touch it there.
    if os.getpid() != _core_registry_pid:
        raise ValueError('core_ownership_inherited_process'+_CORE_ACTION)
    if not _core_registry_mutex.acquire(blocking=False):
        raise ValueError('core_ownership_busy'+_CORE_ACTION)
    try:
        yield
    finally:
        _core_registry_mutex.release()


class _ConstructorCleanup:
    """One process/thread-bound ownership record, never a Core or exception graph."""
    __slots__ = ('db', 'owner', 'pid', 'thread', 'root_key', 'root_identity',
                 'state', 'slot', 'constructor_failed')

    def __init__(self, root_key, slot):
        self.db = self.owner = None
        self.pid, self.thread = os.getpid(), threading.current_thread()
        self.root_key, self.root_identity = root_key, None
        self.state, self.slot, self.constructor_failed = 'constructing', slot, False

    def _identity(self):
        if os.getpid() != self.pid or os.getpid() != _core_registry_pid:
            raise ValueError('core_ownership_inherited_process'+_CORE_ACTION)
        if threading.current_thread() is not self.thread:
            raise RuntimeError('constructor_cleanup_requires_owning_thread')

    def bind_root(self, identity):
        self._identity()
        with _core_bookkeeping():
            if any(r is not None and r is not self and r.root_identity == identity
                   for r in _core_registry):
                raise ValueError('control_busy')
            self.root_identity = identity

    def close(self):
        self._identity()  # PID before mutex, Thread before any resource calls.
        with _core_bookkeeping():
            if self.state == 'closed': return
            if _core_registry[self.slot] is not self:
                raise RuntimeError('core_ownership_stale_record')
            if self.state == 'closing': raise ValueError('core_ownership_close_reentrant')
            if self.state == 'live':
                raise ValueError('live_owner_close')
            self.state = 'closing'
        try:
            if self.db is not None:
                self.db.close()
                # Successful return is the sole DB-release proof.
                with _core_bookkeeping(): self.db = None
            if self.owner is not None:
                self.owner.close()
            with _core_bookkeeping():
                if _core_registry[self.slot] is not self:
                    raise RuntimeError('core_ownership_stale_record')
                _core_registry[self.slot] = None
                self.state = 'closed'
        except BaseException:
            # No resource operation or fallible diagnostic under this mutex.
            # If bookkeeping is unavailable, closing remains conservatively held.
            try:
                with _core_bookkeeping(): self.state = 'uncertain'
            except BaseException: pass
            raise


def _reserve_core(root):
    key = os.path.normcase(os.path.abspath(root))
    with _core_bookkeeping():
        if any(r is not None and r.root_key == key for r in _core_registry):
            raise ValueError('control_busy')
        slot = next((i for i, r in enumerate(_core_registry) if r is None), None)
        if slot is None:
            raise ValueError('core_ownership_capacity'+_CORE_ACTION)
        record = _ConstructorCleanup(key, slot)
        _core_registry[slot] = record
        return record


@contextmanager
def _core_maintenance_lease():
    # Parent alone releases this lease. Child keeps the inherited locked mutex
    # and rejects all registry entry points by PID before touching that mutex.
    with _core_bookkeeping():
        if any(r is not None for r in _core_registry):
            raise ValueError('b2_maintenance_core_ownership_busy'+_CORE_ACTION)
        yield


# Read-only compatibility view for the existing F2 constructor probes. It owns
# no additional records/resources; all retention is in the fixed registry.
from collections.abc import Set as _AbstractSet
class _FailedConstructorView(_AbstractSet):
    def __iter__(self):
        with _core_bookkeeping():
            return iter(tuple(r for r in _core_registry if r is not None and r.constructor_failed))
    def __len__(self): return sum(1 for _ in self)
    def __contains__(self, record): return any(r is record for r in self)
    def _from_iterable(self, values): return set(values)

_failed_constructor_cleanup = _FailedConstructorView()


class Core:
    def __init__(self, root: Path, *, historical_local=False):
        self.historical_local = historical_local
        self.root = Path(root).absolute()
        self._ownership = _reserve_core(self.root)
        self.owner = self._db = None
        try:
            fresh = not self.root.exists() or (self.root.is_dir() and not any(self.root.iterdir()))
            persistence.mkdir_durable(self.root, sync_dir)
            if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
                raise ValueError('foreign_root')
            if not fresh and not (self.root/'core.sqlite').is_file():
                raise persistence.PublicationQuarantine('incomplete_core_initialization')
            info = self.root.stat()
            self._ownership.bind_root((info.st_dev, info.st_ino))
            self.root.chmod(0o700)
            self.owner = self._ownership.owner = open(self.root / 'control.lock', 'a+b')
            try: fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('control_busy') from None
            for name in ('staging', 'assets', 'owners'):
                path = self.root / name
                if not fresh and not path.is_dir():
                    raise persistence.PublicationQuarantine('incomplete_core_namespace')
                persistence.mkdir_durable(path, sync_dir)
                if path.is_symlink() or path.stat().st_uid != os.getuid():
                    raise ValueError('foreign_namespace')
                path.chmod(0o700)
            database=self.root/'core.sqlite'
            if fresh:
                fd=os.open(database,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                try: os.fsync(fd)
                finally: os.close(fd)
                sync_dir(self.root)
            elif database.is_symlink() or database.stat().st_uid!=os.getuid():
                raise persistence.PublicationQuarantine('foreign_database')
            self._db = self._ownership.db = sqlite3.connect(self.root / 'core.sqlite', timeout=2, isolation_level=None)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA foreign_keys=ON')
            version = self.db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1): raise ValueError('unsupported_store_version')
            if version == 0 and not fresh:
                raise persistence.PublicationQuarantine('incomplete_core_schema')
            if version == 0:
                self.db.executescript('''BEGIN IMMEDIATE;
            CREATE TABLE jobs(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
              principal TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
              fingerprint TEXT NOT NULL, job TEXT NOT NULL, context TEXT NOT NULL,
              retry_of TEXT REFERENCES jobs(id), UNIQUE(principal,operation,key));
            CREATE TABLE revisions(project TEXT NOT NULL, revision INTEGER NOT NULL,
              snapshot TEXT NOT NULL, PRIMARY KEY(project,revision));
            CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id), state TEXT NOT NULL);
            CREATE TABLE publications(job_id TEXT PRIMARY KEY REFERENCES jobs(id), manifest TEXT NOT NULL, artifact TEXT NOT NULL, message TEXT NOT NULL);
            CREATE TABLE assets(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE artifacts(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            PRAGMA user_version=1;
            COMMIT;''')
            self.p1 = persistence.initialize(self.db, fresh=fresh)
            if not self.p1:
                self.db.execute('PRAGMA query_only=ON')  # Also fence direct schema writers.
            # SQLite FULL protects WAL commits; persist the database/WAL namespace too.
            sync_dir(self.root)
            sync_dir(self.root.parent)
            self.active = None
            self._publication_storage_failed = False
            self._transaction_serial = 0
            from service.voice_store import VoiceStore, initialize as initialize_voices
            initialize_voices(self)
            self.voices = VoiceStore(self)
            with _core_bookkeeping(): self._ownership.state = 'live'
        except BaseException as primary:
            self._ownership.constructor_failed = True
            try:
                self._ownership.close()
            except BaseException as secondary:
                primary.add_note('constructor cleanup failed: '+type(secondary).__name__)
            raise

    @property
    def db(self):
        # A failed close makes the Core terminal even if SQLite is still live.
        if self._ownership.state not in ('constructing', 'live', 'closed'):
            raise sqlite3.ProgrammingError('Core closed for normal work')
        return self._db

    def close(self):
        self._ownership._identity()
        if self.active is not None: raise ValueError('live_owner_close')
        with _core_bookkeeping():
            if self._ownership.state == 'live': self._ownership.state = 'permitted'
        self._ownership.close()

    @contextmanager
    def transaction(self, *, _deferred=False):
        if not self.p1:
            raise persistence.PublicationQuarantine('pre_p1_inspect_only')
        self._transaction_serial += 1
        self.db.execute('BEGIN' if _deferred else 'BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            try:
                self.db.execute('ROLLBACK')
            except sqlite3.Error:
                # Do not expose an uncommitted local success after rollback denial.
                # DB-API rollback is a bounded fallback; failure closes this handle.
                try: self.db.rollback()
                except sqlite3.Error: self.db.close()
                # Preserve the original COMMIT/write error, including ambiguity.
            raise

    def _load(self, job_id):
        row = self.db.execute('SELECT job,context FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None: raise ValueError('not_found')
        return json.loads(row[0]), json.loads(row[1])

    def _save(self, job):
        oracle.validate('job', job)
        self.db.execute('UPDATE jobs SET job=? WHERE id=?', (encode(job), job['id']))
        self.db.execute('INSERT INTO events(job_id,state) VALUES (?,?)', (job['id'], job['state']))

    def job(self, job_id: str) -> dict:
        return self._load(job_id)[0]

    def jobs(self, principal: str, after: int = 0, limit: int = 20) -> list:
        if not 0 <= after or not 1 <= limit <= 20: raise ValueError('invalid_range')
        return [dict(cursor=r[0], job=json.loads(r[1])) for r in self.db.execute(
            'SELECT seq,job FROM jobs WHERE principal=? AND seq>? ORDER BY seq LIMIT ?', (principal,after,limit))]

    def events(self, job_id: str, after: int = 0, limit: int = 20) -> list:
        if not 0 <= after or not 1 <= limit <= 20: raise ValueError('invalid_range')
        return [dict(cursor=r[0], state=r[1]) for r in self.db.execute(
            'SELECT seq,state FROM events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?', (job_id,after,limit))]

    def create_project(self, project: dict) -> dict:
        oracle.validate('project', project)
        if project['revision'] != 1: raise ValueError('revision_conflict')
        with self.transaction():
            self.db.execute('INSERT INTO revisions VALUES (?,?,?)', (project['id'],1,encode(project)))
        return copy.deepcopy(project)

    def project(self, project_id: str, revision: int | None = None) -> dict:
        row = self.db.execute('SELECT snapshot FROM revisions WHERE project=? AND (? IS NULL OR revision=?) ORDER BY revision DESC LIMIT 1', (project_id,revision,revision)).fetchone()
        if row is None: raise ValueError('not_found')
        return json.loads(row[0])

    def edit_text(self, project_id: str, expected_revision: int, block_id: str, text: str) -> dict:
        with self.transaction():
            before = self.project(project_id)
            oracle.revision(expected_revision, before['revision'])
            after = copy.deepcopy(before)
            block = next((b for b in after['blocks'] if b['id'] == block_id), None)
            if block is None: raise ValueError('not_found')
            block['text'] = text
            after['revision'] += 1
            oracle.edit_semantics(before, after, 'block_text')
            self.db.execute('INSERT INTO revisions VALUES (?,?,?)', (project_id,after['revision'],encode(after)))
        return after

    def submit_synthetic(self, principal: str, key: str, context: dict, *, retry_of: str | None = None, _dispatch_settings=None, _admission=None, _profiles=None) -> dict:
        """Trusted fixture-only admission; never claims installed/qualified ML readiness.

        Resolve context in control before calling. Duplicate lookup precedes resolution.
        """
        plan = context['plan']
        operation = plan['operation']
        fingerprint = oracle.fingerprint(principal, operation, plan['request'])
        row = self.db.execute('SELECT fingerprint,job FROM jobs WHERE principal=? AND operation=? AND key=?',(principal,operation,key)).fetchone()
        if row:
            oracle.require(row[0] == fingerprint, 'idempotency_conflict')
            return json.loads(row[1])
        if _admission is not None:
            _admission.demand(context,_profiles)
        from contextlib import nullcontext
        with (_admission._preflight() if _admission is not None else nullcontext()), self.transaction():
            row = self.db.execute('SELECT fingerprint,job FROM jobs WHERE principal=? AND operation=? AND key=?', (principal,operation,key)).fetchone()
            if row:
                oracle.require(row[0] == fingerprint, 'idempotency_conflict')
                return json.loads(row[1])
            if operation != 'speech_generate': raise NotImplementedError(operation)
            if len(encode(context).encode()) > 48000: raise ValueError('plan_limit')
            oracle.plan_hash(plan)
            oracle.frozen_plan(plan, context['bundles'])
            if plan['resolved']['settings']['kind'] not in ('qwen_preset','kokoro_preset') or plan['request']['variants'] != 1 or plan['request']['reuse'] != 'never':
                raise NotImplementedError('only single synthetic preset without reuse')
            if plan['assets']: raise NotImplementedError('synthetic preset has no inputs')
            if retry_of:
                old, previous = self._load(retry_of)
                oracle.require(old['state'] in ('failed','cancelled') and old['principal_id'] == principal and encode(previous) == encode(context)
                               and old['input_sha256'] == oracle.plan_hash(previous['plan'])
                               and old['input_sha256'] == oracle.plan_hash(plan), 'invalid_retry')
            count = sum(json.loads(r[0])['state'] in ('queued','running','cancel_requested') for r in self.db.execute('SELECT job FROM jobs'))
            if count >= min(32, plan['resolved']['profile']['limits']['queue']): raise ValueError('queue_full')
            job = dict(id=new_id('job'), operation=operation,input_sha256=oracle.plan_hash(plan),principal_id=principal,idempotency_key=key,state='queued',generation=0,attempts=[],artifact_ids=[],stage='queued',deadline='')
            oracle.validate('job', job)
            self.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context,retry_of) VALUES (?,?,?,?,?,?,?,?)', (job['id'],principal,operation,key,fingerprint,encode(job),encode(context),retry_of))
            self.db.execute('INSERT INTO events(job_id,state) VALUES (?,?)', (job['id'],'queued'))
            if _admission is not None:
                _admission.reserve_tx(job,context,_profiles)
            if _dispatch_settings is not None:
                self.db.execute('INSERT INTO dispatch_settings VALUES (?,?,?)',(job['id'],*_dispatch_settings))
        return job

    def start(self, *, delay: float = 0) -> dict | None:
        if not self.historical_local: raise ValueError('dispatcher_required')
        if self.active is not None: raise ValueError('lane_busy')
        if self.reconcile(): raise ValueError('lane_quarantined')
        with self.transaction():
            row = next((json.loads(r[0]) for r in self.db.execute('SELECT job FROM jobs ORDER BY seq') if json.loads(r[0])['state']=='queued'), None)
            if row is None: return None
            job = row
            attempt = dict(id=new_id('attempt'),generation=job['generation']+1,owner_id=new_id('owner'),staging_id=new_id('stage'),reaped=False)
            stage = self.root / 'staging' / attempt['staging_id']
            stage.mkdir(mode=0o700)
            job.update(state='running',stage='synthetic',generation=attempt['generation'],attempts=[*job['attempts'],attempt],deadline=(datetime.now(timezone.utc)+timedelta(seconds=min(30,self._load(job['id'])[1]['plan']['resolved']['profile']['limits']['deadline_seconds']))).isoformat())
            self._save(job)
        # Anchor persisted wall deadline once to a monotonic control cutoff.
        cutoff = time.monotonic() + (datetime.fromisoformat(job['deadline'])-datetime.now(timezone.utc)).total_seconds()
        lock = open(self.root / 'owners' / attempt['owner_id'], 'a+b')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            process = subprocess.Popen([sys.executable,'-m','worker.synthetic',str(child.fileno()),str(lock.fileno()),str(stage),str(delay)],pass_fds=(child.fileno(),lock.fileno()),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            self.active = (job['id'],process,parent,lock)
            process.completion = CompletionObserver(process, cutoff)
            _, context = self._load(job['id'])
            message = self._message(job, context, 'execute')
            # Synthetic execution deliberately bypasses real-hardware admission only.
            parent.sendall((encode(message)+'\n').encode())
        finally: child.close()
        return job

    def _message(self, job, context, kind):
        a = job['attempts'][-1]
        return dict(protocol_version='1.3.0',kind=kind,job_id=job['id'],attempt_id=a['id'],generation=a['generation'],input_sha256=job['input_sha256'],input_asset_ids=[],staging_id=a['staging_id'],owner_id=a['owner_id'],identity=context['plan']['resolved']['identity'],deadline=job['deadline'],stage='completed' if kind=='result' else 'loading',result_asset_ids=[],plan=context['plan'])

    def request_cancel(self, job_id: str, *, _cancel_unstarted=False, _admission=None) -> dict:
        with self.transaction():
            job,_ = self._load(job_id)
            if job['state'] in ('queued','running'):
                job['state']='cancel_requested'; self._save(job)
                if _cancel_unstarted and not job['attempts']:
                    job.update(state='cancelled',stage='never_started');self._save(job)
                    if _admission is not None:_admission.store.release_tx(job['id'],unstarted=True)
        return job

    def finish(self, *, cancel: bool = False, fault=None) -> dict:
        """Wait/reap known child; verified result can win cancellation. Fault hooks E2 only."""
        if not self.historical_local: raise ValueError('dispatcher_required')
        if self.active is None: raise ValueError('no_owner')
        job_id, process, connection, lock = self.active
        if cancel:
            self.request_cancel(job_id)
            if process.poll() is None: process.terminate()
        expired = process.completion.collect()
        connection.settimeout(2)
        raw = bytearray()
        try:
            while len(raw) <= 1024*1024:
                try: chunk = connection.recv(65536)
                except ConnectionResetError:
                    raw.clear()
                    break
                if not chunk: break
                raw.extend(chunk)
            if len(raw)>1024*1024: raise ValueError('result_limit')
            proposal=json.loads(raw) if raw and not expired else None
            if proposal:
                self._publish(job_id, proposal, fault=fault)
            else:
                with self.transaction():
                    job,_=self._load(job_id)
                    job['state']='cancelled' if job['state']=='cancel_requested' else 'failed'
                    job['stage']='deadline' if expired else 'reaped';job['attempts'][-1]['reaped']=True;self._save(job)
        finally:
            connection.close(); lock.close(); self.active=None
        self.reconcile()
        return self.job(job_id)

    def _verified(self, job, proposal, *, stage_root=None, engine_uid=None):
        message, manifest = proposal['message'], proposal['manifest']
        _,context=self._load(job['id'])
        owner=job['attempts'][-1]['owner_id'] # bound to the private socket inherited only by our child
        oracle.worker_message(message,job,[],transport_owner=owner,context=context)
        oracle.validate('result_manifest',manifest)
        oracle.require(len(manifest['outputs'])==1, 'result_mismatch')
        output=manifest['outputs'][0]
        path=(Path(stage_root) if stage_root is not None else self.root/'staging')/job['attempts'][-1]['staging_id']/'output.wav'
        stage=path.parent
        oracle.require(not stage.is_symlink() and stage.stat().st_uid==os.getuid() and stat.S_IMODE(stage.stat().st_mode)==(0o770 if stage_root is not None else 0o700), 'foreign_stage')
        fd=os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            oracle.require(stat.S_ISREG(info.st_mode) and info.st_uid==(engine_uid if stage_root is not None else os.getuid()) and info.st_nlink==1 and info.st_size<=1024*1024, 'foreign_media')
            data=stream.read(1024*1024+1)
        import io
        with wave.open(io.BytesIO(data),'rb') as wav:
            measured=dict(sha256=hashlib.sha256(data).hexdigest(),frames=wav.getnframes(),sample_rate=wav.getframerate(),channels=wav.getnchannels(),media_format='wav',media_subtype={2:'PCM_16',3:'PCM_24',4:'PCM_32'}.get(wav.getsampwidth(),'OTHER'))
            oracle.require(len(wav.readframes(wav.getnframes()))==measured['frames']*measured['channels']*wav.getsampwidth(), 'missing_media')
        oracle.require(measured=={k:output[k] for k in measured}, 'media_mismatch')
        # Fixed fake-worker policy; provenance is control-derived, never accepted on faith.
        expected=dict(kind='synthetic',source_sha256=measured['sha256'],source_id='phase-1a-silence',missing_fields=[])
        oracle.require(output['provenance']==expected and measured['frames']==4800 and measured['sample_rate']==48000 and measured['channels']==1 and measured['media_subtype']=='PCM_16', 'synthetic_policy')
        artifact=dict(id=manifest['artifact_id'],job_id=job['id'],generation=job['generation'],asset_ids=[output['id']],input_sha256=job['input_sha256'],project_revision=0,manifest_sha256=oracle.canonical_hash(manifest),identity=context['plan']['resolved']['identity'],attempt_id=job['attempts'][-1]['id'],owner_id=owner,staging_id=job['attempts'][-1]['staging_id'])
        oracle.publication(job,job['attempts'][-1],artifact,{output['id']:measured['sha256']},transport_owner=owner,context=context,manifest=manifest,verified_result=message,allowed_assets=[])
        return data,manifest,artifact,message

    @persistence.storage_session
    def _publish(self, job_id, proposal, fault=None, *, stage_root=None, engine_uid=None):
        job,_=self._load(job_id)
        try:
            data,manifest,artifact,message=self._verified(job,proposal,stage_root=stage_root,engine_uid=engine_uid)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ELOOP):
                raise oracle.ContractError('missing_or_symlink_result') from exc
            raise
        if not self.p1:
            raise persistence.PublicationQuarantine('pre_p1_root')
        temporary_name = new_id('adoption')
        output=manifest['outputs'][0]
        with self.transaction():
            self.db.execute('INSERT INTO publications VALUES (?,?,?,?)', (job_id,encode(manifest),encode(artifact),encode(message)))
            self.db.execute('INSERT INTO publication_work VALUES (?,?,?,?,?,?)',
                (job_id,encode(artifact),output['id'],temporary_name,hashlib.sha256(data).hexdigest(),len(data)))
        sync_dir(self.root)  # Persist any newly created SQLite WAL/SHM names too.
        if fault: fault('before_adoption')
        output=manifest['outputs'][0]
        destination=self.root/'assets'/output['id']
        # Same-filesystem atomic no-replace adoption. Keep crash orphans for review.
        temporary=destination.parent/temporary_name
        with open(temporary,'xb') as stream:
            persistence.write_complete(stream, data)
        os.link(temporary,destination,follow_symlinks=False)
        if fault: fault('after_link')
        # The same validated alias completion is used live and after restart.
        self._publication_barrier(job_id, manifest, artifact)
        if fault: fault('after_adoption')
        self._commit_publication(job_id)
        if fault: fault('after_commit')

    def _publication_barrier(self, job_id, manifest, artifact):
        if not self.p1:
            raise persistence.PublicationQuarantine('pre_p1_intent')
        row=self.db.execute('SELECT binding,destination,temporary,sha256,byte_length FROM publication_work WHERE job_id=?',(job_id,)).fetchone()
        if row is None:
            raise persistence.PublicationQuarantine('missing_publication_provenance')
        binding,destination,temporary,digest,length=row
        output=manifest['outputs'][0]
        if (len(manifest['outputs'])!=1 or binding!=encode(artifact) or destination!=output['id']
                or digest!=output['sha256'] or not temporary.startswith('adoption-')
                or len(temporary)!=41 or any(c not in '0123456789abcdef' for c in temporary[9:])):
            raise persistence.PublicationQuarantine('publication_work_mismatch')
        job,context=self._load(job_id)
        attempt=job['attempts'][-1]
        oracle.publication(job,attempt,artifact,{destination:digest},transport_owner=attempt['owner_id'],
            context=context,manifest=manifest,
            verified_result=json.loads(self.db.execute('SELECT message FROM publications WHERE job_id=?',(job_id,)).fetchone()[0]),allowed_assets=[])
        directory=self.root/'assets'
        path=directory/destination
        try: path.lstat()
        except FileNotFoundError:
            raise ValueError('incomplete_adoption') from None
        alias=directory/temporary
        try: alias_info=alias.lstat()
        except FileNotFoundError: alias_info=None
        if alias_info is not None:
            first=persistence.verify_file(path,digest,length,links=2)
            second=persistence.verify_file(alias,digest,length,links=2)
            if (first.st_dev,first.st_ino)!=(second.st_dev,second.st_ino):
                raise persistence.PublicationQuarantine('foreign_publication_alias')
            for (asset_id,) in self.db.execute('SELECT id FROM assets'):
                try:
                    other=(directory/asset_id).lstat()
                except FileNotFoundError:
                    raise persistence.PublicationQuarantine('unavailable_published_inode') from None
                if (other.st_dev,other.st_ino)==(first.st_dev,first.st_ino):
                    raise persistence.PublicationQuarantine('published_inode_alias')
            alias.unlink()  # Only the exact recorded redundant protocol alias.
            sync_dir(directory)
        persistence.verify_file(path,digest,length,links=1)
        sync_dir(directory)

    @persistence.storage_session
    def _commit_publication(self, job_id):
        job,_=self._load(job_id)
        if job['state'] in ('succeeded','failed','cancelled'):
            return  # Read durable state after ambiguous COMMIT; never rewrite history.
        row=self.db.execute('SELECT manifest,artifact FROM publications WHERE job_id=?',(job_id,)).fetchone()
        manifest,artifact=map(json.loads,row)
        self._publication_barrier(job_id,manifest,artifact)
        with self.transaction():
            job,_=self._load(job_id)
            oracle.require(job['state'] in ('running','cancel_requested'), 'invalid_transition')
            for output in manifest['outputs']:
                self.db.execute('INSERT INTO assets VALUES (?,?)',(output['id'],encode(output)))
            self.db.execute('INSERT INTO artifacts VALUES (?,?)',(artifact['id'],encode(artifact)))
            job.update(state='succeeded',stage='completed',artifact_ids=[artifact['id']])
            job['attempts'][-1]['reaped']=True
            self._save(job)

    def reconcile(self) -> bool:
        """No retries, kills or blind deletion. An inherited lock fences a living fake owner."""
        if not self.historical_local: raise ValueError('dispatcher_required')
        if not self.p1:
            return any(json.loads(raw)['state'] not in ('succeeded','failed','cancelled')
                       for (raw,) in self.db.execute('SELECT job FROM jobs'))
        quarantined=False
        for (raw,) in self.db.execute('SELECT job FROM jobs').fetchall():
            job=json.loads(raw)
            if job['state'] not in ('running','cancel_requested'): continue
            if not job['attempts']:
                with self.transaction():
                    job.update(state='cancelled',stage='never_started');self._save(job)
                continue
            if self.active and self.active[0]==job['id']: quarantined=True;continue
            a=job['attempts'][-1]
            with open(self.root/'owners'/a['owner_id'],'a+b') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: quarantined=True;continue
                row=self.db.execute('SELECT manifest FROM publications WHERE job_id=?',(job['id'],)).fetchone()
                if row:
                    try: self._commit_publication(job['id'])
                    except (persistence.PublicationQuarantine, sqlite3.Error):
                        quarantined=True
                    except OSError:
                        quarantined=True
                    except (ValueError,oracle.ContractError):
                        with self.transaction():
                            job.update(state='failed',stage='publication_interrupted');a['reaped']=True;self._save(job)
                else:
                    with self.transaction():
                        job.update(state='cancelled' if job['state']=='cancel_requested' else 'failed',stage='interrupted');a['reaped']=True;self._save(job)
        return quarantined

    def artifact(self, artifact_id: str) -> dict:
        row=self.db.execute('SELECT metadata FROM artifacts WHERE id=?',(artifact_id,)).fetchone()
        if row is None: raise ValueError('not_found')
        artifact=json.loads(row[0])
        for asset_id in artifact['asset_ids']: self.asset(asset_id)
        return artifact

    def asset(self, asset_id: str) -> dict:
        row=self.db.execute('SELECT metadata FROM assets WHERE id=?',(asset_id,)).fetchone()
        if row is None: raise ValueError('not_found')
        metadata=json.loads(row[0])
        oracle.validate('asset', metadata)
        oracle.require(metadata['id']==asset_id and '/' not in asset_id and asset_id not in ('.','..'), 'foreign_media')
        fd=os.open(self.root/'assets'/metadata['id'],os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            oracle.require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1 and info.st_size<=1024*1024, 'foreign_media')
            oracle.require(hashlib.sha256(stream.read(1024*1024+1)).hexdigest()==metadata['sha256'], 'missing_media')
        return metadata
