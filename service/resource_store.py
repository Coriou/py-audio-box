"""Durable deployment budget. Methods ending _tx require the Core writer transaction."""
import json
import os
from pathlib import Path
import stat
from service.core import encode


class ResourceStore:
    def __init__(self, core, budget, roots):
        self.core, self.db, self.budget = core, core.db, budget
        self.roots = tuple(Path(p) for p in roots)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS resource_reservations(
          id TEXT PRIMARY KEY, category TEXT NOT NULL, demand TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('queued','executing','retained')));
        CREATE TABLE IF NOT EXISTS resource_accounting(
          id INTEGER PRIMARY KEY CHECK(id=1), highwater INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS resource_faults(reason TEXT PRIMARY KEY);
        INSERT OR IGNORE INTO resource_accounting VALUES(1,0);
        ''')
        self.check_disk()

    def rows(self):
        return [dict(id=r[0],category=r[1],demand=json.loads(r[2]),state=r[3]) for r in self.db.execute('SELECT * FROM resource_reservations ORDER BY rowid')]

    def measure(self):
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

    def check_disk(self, extra=0):
        actual=self.measure()
        high=max(actual,self.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0])
        if high>self.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]:
            self.db.execute('UPDATE resource_accounting SET highwater=?',(high,))
        # Deliberate conservative double accounting: retained charge PLUS observed bytes.
        charged=sum(r['demand']['disk_bytes'] for r in self.rows())
        if high+charged+extra>self.budget['disk_bytes']:raise ValueError('disk_quota')
        for root in self.roots:
            v=os.statvfs(root)
            if v.f_bavail*v.f_frsize < self.budget['free_bytes']+charged+extra:raise ValueError('disk_free_reserve')
        return actual

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
        for key in ('slots','ram_bytes','vram_bytes','pids'):
            if demand[key]>self.budget[key]:raise ValueError('resource_'+key)
        rows=self.rows()
        if sum(r['state']!='retained' for r in rows)>=self.budget['queue']:raise ValueError('queue_full')
        self.check_disk(demand['disk_bytes'])
        self.db.execute('INSERT INTO resource_reservations VALUES(?,?,?,?)',(identity,category,encode(demand),'queued'))

    def acquire_tx(self, identity, *, replacement=False):
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
        if replacement:
            self.check_disk(d['attempt_disk_bytes'])
            # Persist cumulative disk charge without inflating execution charges.
            d['disk_bytes']+=d['attempt_disk_bytes']
        self.db.execute("UPDATE resource_reservations SET state='executing',demand=? WHERE id=?",(encode(d),identity))

    def release_tx(self, identity, *, confirmed_reap=False, unstarted=False):
        row=next((r for r in self.rows() if r['id']==identity),None)
        if row is None:raise ValueError('resource_reservation_missing')
        if row['state']=='retained':return
        if not confirmed_reap and not (unstarted and row['state']=='queued'):raise ValueError('resource_reap_required')
        self.db.execute("UPDATE resource_reservations SET state='retained' WHERE id=?",(identity,))
