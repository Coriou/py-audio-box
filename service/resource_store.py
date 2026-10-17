"""Durable deployment budget. Methods ending _tx require the Core writer transaction."""
import json
import os
from pathlib import Path
import stat
import threading
from contextlib import contextmanager
from service.core import encode


class ResourceStore:
    def __init__(self, core, budget, roots):
        self.core, self.db, self.budget = core, core.db, budget
        self.roots = tuple(Path(p) for p in roots)
        self.owner = threading.get_ident()
        self._token = self._snapshot = None
        self.failed = False
        self.on_fault = lambda:None
        existing = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='resource_monitor'").fetchone()
        tables={r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        populated=any(self.db.execute('SELECT 1 FROM '+name+' LIMIT 1').fetchone()
                      for name in ('jobs','revisions','assets','artifacts','publications') if name in tables)
        # Even an emptied old resource schema is not a new protocol root.
        old_resource_schema=bool(tables & {'resource_reservations','resource_accounting','resource_faults','resource_protocol'})
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS resource_reservations(
          id TEXT PRIMARY KEY, category TEXT NOT NULL, demand TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('queued','executing','retained')));
        CREATE TABLE IF NOT EXISTS resource_accounting(
          id INTEGER PRIMARY KEY CHECK(id=1), highwater INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS resource_faults(reason TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS resource_monitor(
          id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL CHECK(state IN ('healthy','checking','fault')));

        INSERT OR IGNORE INTO resource_accounting VALUES(1,0);
        ''')
        self.db.execute('CREATE TABLE IF NOT EXISTS resource_protocol(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)')
        revision = self.db.execute('SELECT revision FROM resource_protocol WHERE id=1').fetchone()
        monitor = self.db.execute('SELECT state FROM resource_monitor WHERE id=1').fetchone()
        if not existing and not old_resource_schema and not populated:
            with self.core.transaction():
                self.db.execute("INSERT INTO resource_monitor VALUES(1,'healthy')")
                self.db.execute('INSERT INTO resource_protocol VALUES(1,2)')
        self.valid = (not existing and not old_resource_schema and not populated) or bool(revision == (2,) and monitor)

    def state(self):
        row = self.db.execute('SELECT state FROM resource_monitor WHERE id=1').fetchone()
        return row[0] if self.valid and row else 'invalid'

    def _owner(self):
        if threading.get_ident() != self.owner:raise ValueError('resource_owner')

    def observe(self, observation):
        """The begin commit is independent of both observation and consuming transactions."""
        self._owner()
        if self.db.in_transaction or self._token is not None:raise ValueError('resource_reentrant_check')
        self.require_admission()
        token = object()
        try:
            with self.core.transaction():
                if self.db.execute("UPDATE resource_monitor SET state='checking' WHERE id=1 AND state='healthy'").rowcount != 1:
                    raise ValueError('resource_fence_begin')
        except BaseException:
            self.failed = True
            self.on_fault()
            raise
        self._token = token
        self._measured = None
        try:
            result = observation()
        except BaseException as exc:
            self.failed = True
            self.on_fault()
            # Local quarantine is already latched before these fallible writes.
            try:
                # Still the sole writer. Deferred BEGIN lets a write-denied
                # connection reach the fault UPDATE itself; it grants no authority.
                with self.core.transaction(_deferred=True):
                    if self._token is not token:raise ValueError('resource_check_token')
                    if self.db.execute("UPDATE resource_monitor SET state='fault' WHERE id=1 AND state='checking'").rowcount != 1:
                        raise ValueError('resource_fence_fault')
                    self.db.execute('INSERT OR IGNORE INTO resource_faults VALUES(?)',(str(exc)[:160],))
            finally:
                self._token = None
            raise
        try:
            if self._token is not token:raise ValueError('resource_check_token')
            if self._measured is None:raise ValueError('resource_accounting_attestation_required')
            with self.core.transaction():
                if self._measured is not None:
                    self.db.execute('UPDATE resource_accounting SET highwater=max(highwater,?) WHERE id=1',(self._measured[0],))
                if self.db.execute("UPDATE resource_monitor SET state='healthy' WHERE id=1 AND state='checking' AND NOT EXISTS(SELECT 1 FROM resource_faults)").rowcount != 1:
                    raise ValueError('resource_fence_complete')
        except BaseException:
            self.failed = True
            self.on_fault()
            raise
        finally:self._token = None
        return result

    @contextmanager
    def _preflight(self, observation=None):
        self._owner()
        if self._snapshot is not None or self.db.in_transaction:raise ValueError('resource_stale_snapshot')
        self.observe(observation or self.check_disk)
        self._snapshot = (*self._measured, self.core._transaction_serial+1)
        try:yield
        finally:self._snapshot = None

    def _capacity(self, extra):
        self._owner()
        if not self.db.in_transaction or self._snapshot is None:raise ValueError('resource_preflight_required')
        high, free, serial = self._snapshot
        if serial != self.core._transaction_serial:raise ValueError('resource_stale_snapshot')
        charged = sum(r['demand']['disk_bytes'] for r in self.rows())
        if high+charged+extra>self.budget['disk_bytes']:raise ValueError('disk_quota')
        if free<self.budget['free_bytes']+charged+extra:raise ValueError('disk_free_reserve')

    def rows(self):
        return [dict(id=r[0],category=r[1],demand=json.loads(r[2]),state=r[3]) for r in self.db.execute('SELECT * FROM resource_reservations ORDER BY rowid')]

    def measure(self):
        self._owner()
        if self._token is None or self.db.in_transaction:raise ValueError('resource_unfenced_measurement')
        total=0
        for root in self.roots:
            if not root.is_dir() or root.is_symlink():raise ValueError('resource_root_missing')
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda exc: (_ for _ in ()).throw(exc)):
                for name in dirs+files:
                    s=(Path(directory)/name).lstat()
                    if stat.S_ISLNK(s.st_mode) or not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
                        raise ValueError('resource_namespace_unknown')
                    total += s.st_size
        return total

    def check_disk(self):
        if self._token is None:return self.observe(self.check_disk)
        self._owner()
        if self.db.in_transaction:raise ValueError('resource_transactional_observation')
        actual=self.measure()
        high=max(actual,self.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0])
        charged=sum(r['demand']['disk_bytes'] for r in self.rows())
        free=min(v.f_bavail*v.f_frsize for v in (os.statvfs(root) for root in self.roots))
        if high+charged>self.budget['disk_bytes']:raise ValueError('disk_quota')
        if free<self.budget['free_bytes']+charged:raise ValueError('disk_free_reserve')
        self._measured=(high,free)
        return actual

    def require_admission(self):
        if (self.failed or self.state()!='healthy'
                or self.db.execute('SELECT 1 FROM resource_faults').fetchone()):
            raise ValueError('resource_fault_unresolved')

    def reserve_tx(self, identity, category, demand):
        if category not in ('inference','decode','render','provision'):raise ValueError('resource_category')
        keys={'configuration','slots','ram_bytes','vram_bytes','pids','disk_bytes','attempt_disk_bytes'}
        if set(demand)!=keys or not isinstance(demand['configuration'],str) or not demand['configuration']:
            raise ValueError('closed_resource_demand')
        if any(type(demand[k])!=int or demand[k]<0 or demand[k]>2**53-1 for k in keys-{'configuration'}):
            raise ValueError('finite_resource_demand')
        if any(demand[k]==0 for k in ('slots','ram_bytes','pids','disk_bytes','attempt_disk_bytes')):
            raise ValueError('positive_resource_demand')
        old=next((r for r in self.rows() if r['id']==identity),None)
        if old:
            if old['category']!=category or encode(old['demand'])!=encode(demand):raise ValueError('reservation_conflict')
            return old
        self.require_admission()
        for key in ('slots','ram_bytes','vram_bytes','pids'):
            if demand[key]>self.budget[key]:raise ValueError('resource_'+key)
        rows=self.rows()
        if sum(r['state']!='retained' for r in rows)>=self.budget['queue']:raise ValueError('queue_full')
        self._capacity(demand['disk_bytes'])
        self.db.execute('INSERT INTO resource_reservations VALUES(?,?,?,?)',(identity,category,encode(demand),'queued'))

    def acquire_tx(self, identity, *, replacement=False):
        self.require_admission()
        rows=self.rows();row=next((r for r in rows if r['id']==identity),None)
        if row is None:raise ValueError('resource_reservation_missing')
        if row['state']=='executing':raise ValueError('resource_already_executing')
        if row['state']=='retained' and not replacement:raise ValueError('resource_retained')
        queued=[r for r in rows if r['state']=='queued']
        if queued and queued[0]['id']!=identity:raise ValueError('resource_fifo')
        active=[r for r in rows if r['state']=='executing'];d=row['demand']
        for key in ('ram_bytes','vram_bytes','pids','slots'):
            if sum(r['demand'][key] for r in active)+d[key]>self.budget[key]:raise ValueError('resource_'+key)
        configs={r['demand']['configuration'] for r in active}|{d['configuration']}
        if len(configs)>self.budget['resident_models']:raise ValueError('resource_resident_models')
        self._capacity(d['attempt_disk_bytes'] if replacement else 0)
        if replacement:
            # Persist cumulative disk charge without inflating execution charges.
            d['disk_bytes']+=d['attempt_disk_bytes']
        self.db.execute("UPDATE resource_reservations SET state='executing',demand=? WHERE id=?",(encode(d),identity))

    def release_tx(self, identity, *, confirmed_reap=False, unstarted=False):
        row=next((r for r in self.rows() if r['id']==identity),None)
        if row is None:raise ValueError('resource_reservation_missing')
        if row['state']=='retained':return
        if self.state() not in ('healthy','fault') or (self.failed and self.state()!='fault'):
            raise ValueError('resource_fault_persistence_unresolved')
        if not confirmed_reap and not (unstarted and row['state']=='queued'):raise ValueError('resource_reap_required')
        self.db.execute("UPDATE resource_reservations SET state='retained' WHERE id=?",(identity,))


MAINTENANCE_FILE_LIMIT = 4_194_304
MAINTENANCE_SNAPSHOT_LIMIT = 1_048_576
MAINTENANCE_DOCUMENT_LIMIT = 65_536


def _measure_recovery_roots(roots):
    """Existing bounded st_size loop: descendants included, roots excluded."""
    import time
    deadline=time.monotonic()+1;count=total=0
    for root in roots:
        root=Path(root)
        if root.is_symlink() or not root.is_dir():raise ValueError('resource_root_missing')
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=lambda exc: (_ for _ in ()).throw(exc)):
            for name in dirs+files:
                count+=1
                if count>2048 or time.monotonic()>deadline:raise ValueError('b2_resource_scan_budget')
                info=(Path(directory)/name).lstat()
                if info.st_uid!=os.getuid() or stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                    raise ValueError('resource_namespace_unknown')
                total+=info.st_size
                if total>2**53-1:raise ValueError('b2_resource_scan_budget')
    return total


def _maintenance_footprint(roots, highwater, charged, budget):
    """Early fixed-envelope guard; no accounting authority or counter update."""
    from service.admission import finite_budget
    budget=finite_budget(budget)
    if any(type(v) is not int or not 0<=v<=2**53-1 for v in (highwater,charged)):
        raise ValueError('b2_maintenance_arithmetic')
    roots=tuple(roots)
    # Five is conservative even if WAL/SHM already exist: two sidecars,
    # snapshot + manifest before scan, one later flat certificate.
    count=0
    import time
    deadline=time.monotonic()+1
    for root in roots:
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=lambda exc: (_ for _ in ()).throw(exc)):
            count+=len(dirs)+len(files)
            if count+5>2048 or time.monotonic()>deadline:raise ValueError('b2_resource_scan_budget')
    measured=_measure_recovery_roots(roots)
    upper=measured+3*MAINTENANCE_FILE_LIMIT+MAINTENANCE_SNAPSHOT_LIMIT+2*MAINTENANCE_DOCUMENT_LIMIT
    if upper>2**53-1 or highwater+charged>2**53-1:raise ValueError('b2_maintenance_arithmetic')
    if upper>highwater:raise ValueError('b2_maintenance_highwater')
    if highwater+charged>budget['disk_bytes']:raise ValueError('disk_quota')
    free=min(v.f_bavail*v.f_frsize for v in (os.statvfs(root) for root in roots))
    if free<budget['free_bytes']+charged:raise ValueError('disk_free_reserve')
    return dict(measured=measured,upper=upper,highwater=highwater,charged=charged)


def _recovery_scan(db, roots, intent):
    """Private maintenance observation; caller holds both locks and pending proof.

    No ordinary monitor/token/constructor is used. This scan grants no authority
    until its exact accepted output and the whole Core transaction commit.
    """
    import time
    from service.admission import finite_budget
    from service.persistent_init import digest
    budget=finite_budget(intent['budget_after'])
    if budget!=intent['budget_before']:raise ValueError('b2_budget_change_unsupported')
    row=db.execute('SELECT record,digest FROM p1_recoveries WHERE id=?',(intent['id'],)).fetchone()
    if row!=(encode(intent),digest(intent)):raise ValueError('b2_scan_pending_required')
    if db.in_transaction or db.execute("SELECT 1 FROM p1_recovery_steps WHERE recovery_id=? AND step='core-complete'",(intent['id'],)).fetchone():
        raise ValueError('b2_scan_pending_required')
    total=_measure_recovery_roots(roots)
    jobs={r[1]:json.loads(r[6]) for r in intent['before']['source']['data']['jobs']}
    for identity,category,raw,state in intent['before']['source']['data']['resource_reservations']:
        if state!='executing':continue
        demand=json.loads(raw);stage=Path(roots[1])/jobs[identity]['attempts'][-1]['staging_id']
        if stage.exists():
            if sum(p.stat().st_size for p in stage.iterdir() if p.is_file())>demand['attempt_disk_bytes']:raise ValueError('resource_stage_growth')
            output=stage/'output.wav'
            if output.exists() and output.stat().st_size>9644:raise ValueError('resource_output_bytes')
    high=max(total,intent['highwater'])
    charged=sum(json.loads(r[0])['disk_bytes'] for r in db.execute('SELECT demand FROM resource_reservations'))
    free=min(v.f_bavail*v.f_frsize for v in (os.statvfs(root) for root in roots))
    if high+charged>budget['disk_bytes']:raise ValueError('disk_quota')
    if free<budget['free_bytes']+charged:raise ValueError('disk_free_reserve')
    if high!=intent['expected'][3]['output']['highwater']:raise ValueError('b2_scan_output_changed')
    return high
