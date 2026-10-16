"""Phase 1A synthetic durable domain. Trusted control API, never an HTTP payload API.

One Core owns a data root. Canonical dictionaries retain contract 1.3.0 exactly.
Only preset speech (one variant, no reuse) is executable in this bounded lot.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import sys
from uuid import uuid4
import wave

from contracts import validate as oracle


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def new_id(prefix):
    return prefix + '-' + uuid4().hex


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


class Core:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise ValueError('foreign_root')
        self.root.chmod(0o700)
        self.owner = open(self.root / 'control.lock', 'a+b')
        try: fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise ValueError('control_busy') from None
        for name in ('staging', 'assets', 'owners'):
            path = self.root / name
            path.mkdir(mode=0o700, exist_ok=True)
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                self.owner.close()
                raise ValueError('foreign_namespace')
            path.chmod(0o700)
        self.db = sqlite3.connect(self.root / 'core.sqlite', timeout=2, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        version = self.db.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, 1): raise ValueError('unsupported_store_version')
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
        self.active = None

    def close(self):
        if self.active is not None: raise ValueError('live_owner_close')
        self.db.close()
        self.owner.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
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

    def submit_synthetic(self, principal: str, key: str, context: dict, *, retry_of: str | None = None) -> dict:
        """Trusted fixture-only admission; never claims installed/qualified ML readiness.

        Resolve context in control before calling. Duplicate lookup precedes resolution.
        """
        plan = context['plan']
        operation = plan['operation']
        fingerprint = oracle.fingerprint(principal, operation, plan['request'])
        with self.transaction():
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
                oracle.require(old['state'] in ('failed','cancelled') and old['principal_id'] == principal and previous == context, 'invalid_retry')
            count = sum(json.loads(r[0])['state'] in ('queued','running','cancel_requested') for r in self.db.execute('SELECT job FROM jobs'))
            if count >= min(32, plan['resolved']['profile']['limits']['queue']): raise ValueError('queue_full')
            job = dict(id=new_id('job'), operation=operation,input_sha256=oracle.plan_hash(plan),principal_id=principal,idempotency_key=key,state='queued',generation=0,attempts=[],artifact_ids=[],stage='queued',deadline='')
            oracle.validate('job', job)
            self.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context,retry_of) VALUES (?,?,?,?,?,?,?,?)', (job['id'],principal,operation,key,fingerprint,encode(job),encode(context),retry_of))
            self.db.execute('INSERT INTO events(job_id,state) VALUES (?,?)', (job['id'],'queued'))
        return job

    def start(self, *, delay: float = 0) -> dict | None:
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
        lock = open(self.root / 'owners' / attempt['owner_id'], 'a+b')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            process = subprocess.Popen([sys.executable,'-m','worker.synthetic',str(child.fileno()),str(lock.fileno()),str(stage),str(delay)],pass_fds=(child.fileno(),lock.fileno()),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            self.active = (job['id'],process,parent,lock)
            _, context = self._load(job['id'])
            message = self._message(job, context, 'execute')
            # Synthetic execution deliberately bypasses real-hardware admission only.
            parent.sendall((encode(message)+'\n').encode())
        finally: child.close()
        return job

    def _message(self, job, context, kind):
        a = job['attempts'][-1]
        return dict(protocol_version='1.3.0',kind=kind,job_id=job['id'],attempt_id=a['id'],generation=a['generation'],input_sha256=job['input_sha256'],input_asset_ids=[],staging_id=a['staging_id'],owner_id=a['owner_id'],identity=context['plan']['resolved']['identity'],deadline=job['deadline'],stage='completed' if kind=='result' else 'loading',result_asset_ids=[],plan=context['plan'])

    def request_cancel(self, job_id: str) -> dict:
        with self.transaction():
            job,_ = self._load(job_id)
            if job['state'] in ('queued','running'):
                job['state']='cancel_requested'; self._save(job)
        return job

    def finish(self, *, cancel: bool = False, fault=None) -> dict:
        """Wait/reap known child; verified result can win cancellation. Fault hooks E2 only."""
        if self.active is None: raise ValueError('no_owner')
        job_id, process, connection, lock = self.active
        if cancel:
            self.request_cancel(job_id)
            if process.poll() is None: process.terminate()
        expired=False
        remaining=max(.01,min(35,(datetime.fromisoformat(self.job(job_id)['deadline'])-datetime.now(timezone.utc)).total_seconds()))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            expired=True
            process.kill()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: raise ValueError('lane_quarantined') from None
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

    def _verified(self, job, proposal):
        message, manifest = proposal['message'], proposal['manifest']
        _,context=self._load(job['id'])
        owner=job['attempts'][-1]['owner_id'] # bound to the private socket inherited only by our child
        oracle.worker_message(message,job,[],transport_owner=owner,context=context)
        oracle.validate('result_manifest',manifest)
        oracle.require(len(manifest['outputs'])==1, 'result_mismatch')
        output=manifest['outputs'][0]
        path=self.root/'staging'/job['attempts'][-1]['staging_id']/'output.wav'
        stage=path.parent
        oracle.require(not stage.is_symlink() and stage.stat().st_uid==os.getuid() and stat.S_IMODE(stage.stat().st_mode)==0o700, 'foreign_stage')
        fd=os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            oracle.require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1 and info.st_size<=1024*1024, 'foreign_media')
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

    def _publish(self, job_id, proposal, fault=None):
        job,_=self._load(job_id)
        data,manifest,artifact,message=self._verified(job,proposal)
        with self.transaction():
            self.db.execute('INSERT INTO publications VALUES (?,?,?,?)', (job_id,encode(manifest),encode(artifact),encode(message)))
        if fault: fault('before_adoption')
        output=manifest['outputs'][0]
        destination=self.root/'assets'/output['id']
        # Same-filesystem atomic no-replace adoption. Keep crash orphans for review.
        temporary=destination.parent/new_id('adoption')
        with open(temporary,'xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,destination,follow_symlinks=False)
        temporary.unlink()
        sync_dir(destination.parent)
        if fault: fault('after_adoption')
        self._commit_publication(job_id)
        if fault: fault('after_commit')

    def _commit_publication(self, job_id):
        row=self.db.execute('SELECT manifest,artifact FROM publications WHERE job_id=?',(job_id,)).fetchone()
        manifest,artifact=map(json.loads,row)
        for output in manifest['outputs']:
            path=self.root/'assets'/output['id']
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb') as stream:
                info=os.fstat(stream.fileno())
                oracle.require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1 and info.st_size<=1024*1024, 'foreign_media')
                oracle.require(hashlib.sha256(stream.read(1024*1024+1)).hexdigest()==output['sha256'], 'missing_media')
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
                    except (OSError,ValueError,oracle.ContractError):
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
        fd=os.open(self.root/'assets'/metadata['id'],os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            oracle.require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1 and info.st_size<=1024*1024, 'foreign_media')
            oracle.require(hashlib.sha256(stream.read(1024*1024+1)).hexdigest()==metadata['sha256'], 'missing_media')
        return metadata
