"""Private allocator journal, on the sole Core writer/SQLite connection."""
import json
from datetime import datetime, timedelta, timezone
from service.core import encode, new_id
from service.worker_transport import Authority


class DispatchStore:
    def __init__(self, core, admission=None):
        self.core, self.db = core, core.db
        self.admission = admission
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS dispatch_settings(
          job_id TEXT PRIMARY KEY REFERENCES jobs(id), fixture TEXT NOT NULL, seconds INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS dispatches(
          seq INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
          authority TEXT NOT NULL, binding TEXT UNIQUE NOT NULL, fixture TEXT NOT NULL,
          runtime TEXT, phase TEXT NOT NULL, receipt TEXT, outcome TEXT, acked INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS dispatch_commands(
          seq INTEGER PRIMARY KEY, dispatch_id INTEGER NOT NULL REFERENCES dispatches(seq),
          op TEXT NOT NULL, state TEXT NOT NULL);
        ''')

    def current(self):
        from service.persistence import completed_dispatch_resolutions
        resolved={r['dispatch_seq'] for r in completed_dispatch_resolutions(self.db)}
        row = next((r for r in self.db.execute('SELECT seq,job_id,authority,binding,fixture,runtime,phase,receipt,outcome,acked FROM dispatches WHERE acked=0 ORDER BY seq') if r[0] not in resolved),None)
        if row is None: return None
        value = dict(zip(('seq','job_id','authority','binding','fixture','runtime','phase','receipt','outcome','acked'),row))
        value['authority'] = Authority(*json.loads(value['authority']))
        if value['authority'].binding != value['binding']: raise ValueError('authority_store_mismatch')
        if value['receipt']: value['receipt']=json.loads(value['receipt'])
        return value

    def update(self, row, **fields):
        if not set(fields) <= {'runtime','phase','receipt','outcome','acked'}: raise ValueError('private_fields')
        with self.core.transaction():
            self.db.execute('UPDATE dispatches SET '+','.join(k+'=?' for k in fields)+' WHERE seq=?', (*fields.values(),row['seq']))

    def allocate(self, profiles, fixture, seconds, *, replacement=None):
        from service.persistence import check_operator_resolved
        if replacement:check_operator_resolved(self.db,replacement)
        if self.current(): raise ValueError('lane_busy')
        from contextlib import nullcontext
        with (self.admission._preflight() if self.admission is not None else nullcontext()), self.core.transaction():
            if replacement:
                job,context=self.core._load(replacement)
                if self.db.execute('SELECT 1 FROM publications WHERE job_id=?',(replacement,)).fetchone():raise ValueError('publication_reconciliation_required')
                if job['state']!='failed' or not job['attempts'][-1]['reaped']: raise ValueError('confirmed_failed_attempt_required')
            else:
                row=next((json.loads(r[0]) for r in self.db.execute('SELECT job FROM jobs ORDER BY seq') if json.loads(r[0])['state']=='queued'),None)
                if row is None:return None
                job=row; _,context=self.core._load(job['id'])
            check_operator_resolved(self.db,job['id'])
            # Verify the owner's frozen dependencies before resource acquisition.
            allowed_assets = self.core._input_assets(job['principal_id'], context['plan'])
            if self.admission is not None:
                self.admission.demand(context,profiles)
                self.admission.store.acquire_tx(job['id'],replacement=bool(replacement))
            attempt=dict(id=new_id('attempt'),generation=job['generation']+1,owner_id=new_id('owner'),staging_id=new_id('stage'),reaped=False)
            job.update(state='running',stage='synthetic',generation=attempt['generation'],attempts=[*job['attempts'],attempt],deadline=(datetime.now(timezone.utc)+timedelta(seconds=min(seconds,context['plan']['resolved']['profile']['limits']['deadline_seconds']))).isoformat())
            from worker.runtime_protocol import execution
            spec=execution(context)
            if (fixture == 'kokoro') != (spec is not None): raise ValueError('execution_dispatch_mismatch')
            authority=Authority(job,context,allowed_assets,profiles)
            authority.check(authority.message(),'execute')
            self.core._save(job)
            self.db.execute('INSERT INTO dispatches(job_id,authority,binding,fixture,phase) VALUES (?,?,?,?,?)', (job['id'],authority._snapshot.decode(),authority.binding,fixture,'intent'))
        return self.current()

    def command(self,row,op):
        with self.core.transaction():
            return self.db.execute('INSERT INTO dispatch_commands(dispatch_id,op,state) VALUES (?,?,?)',(row['seq'],op,'sent')).lastrowid

    def complete(self,command,success):
        with self.core.transaction():
            self.db.execute('UPDATE dispatch_commands SET state=? WHERE seq=?',('returned' if success else 'uncertain',command))
