"""Real temporary Core/reservation/Authority proofs, isolated unittest."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contracts import validate as oracle
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.ingest import ingest_local
from service.production import Production
from test_production_api import audio, configuration


class ProductionAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.core=Core(Path(self.tmp.name)/'core'); self.addCleanup(self.core.close)
        stage=Path(self.tmp.name)/'stage'; stage.mkdir()
        self.admission=Admission(self.core,stage); self.store=DispatchStore(self.core,self.admission)
        self.context,_,self.profiles=configuration()
        self.asset=ingest_local(self.core,'alice',audio())
        self.context['plan']['request']['asset_id']=self.asset['id']
        self.production=Production(self.core)

    def submit(self,key='prepare'):
        return self.production.prepare('alice',key,self.context,_admission=self.admission,_profiles=self.profiles,_dispatch_settings=('silence',8))

    def test_atomic_duplicate_and_exact_detached_authority(self):
        job=self.submit(); self.assertEqual(self.submit(),job)
        self.assertEqual(len(self.admission.store.rows()),1)
        extra=ingest_local(self.core,'alice',audio())
        row=self.store.allocate(self.profiles,'silence',8)
        authority=row['authority']
        self.assertEqual(authority.allowed_assets,(self.asset['id'],))
        self.assertNotIn(extra['id'],authority.allowed_assets)
        self.context['plan']['request']['asset_id']=extra['id']
        message=authority.message(); message['input_asset_ids'].append(extra['id'])
        with self.assertRaises(ValueError): authority.check(message,'execute')
        self.assertEqual(authority.allowed_assets,(self.asset['id'],))

    def test_refusal_and_reservation_failure_leave_no_job(self):
        self.profiles[0]['readiness']='not_ready'
        with self.assertRaises(ValueError):self.submit()
        self.assertEqual(self.core.jobs('alice'),[]);self.assertEqual(self.admission.store.rows(),[])
        self.profiles[0]['readiness']='ready'
        self.core.db.execute("CREATE TRIGGER refuse_reservation BEFORE INSERT ON resource_reservations BEGIN SELECT RAISE(ABORT,'refuse'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.submit()
        for table in ('jobs','events','dispatch_settings','resource_reservations'):
            self.assertEqual(self.core.db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)

    def test_missing_foreign_altered_refs_before_allocation(self):
        self.submit()
        path=self.core.root/'assets'/self.asset['id']; original=path.read_bytes()
        path.write_bytes(original[:-1]+b'X')
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'silence',8)
        self.assertEqual(self.core.jobs('alice')[0]['job']['generation'],0)
        self.assertEqual(self.admission.store.rows()[0]['state'],'queued')
        path.unlink()
        with self.assertRaises((ValueError,OSError)):self.store.allocate(self.profiles,'silence',8)
        path.write_bytes(original)
        metadata=self.core.asset(self.asset['id']); metadata['principal_id']='bob'
        with self.core.transaction():self.core.db.execute('UPDATE assets SET metadata=? WHERE id=?',(json.dumps(metadata),self.asset['id']))
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'silence',8)
        self.assertEqual(self.core.jobs('alice')[0]['job']['generation'],0)
        self.assertIsNone(self.store.current())

if __name__=='__main__':unittest.main()
