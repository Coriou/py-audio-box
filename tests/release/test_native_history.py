"""Tiny source histories; actual Core and native proofs, fake TLS/process/backend."""
import copy
import json
import time
import unittest
from unittest.mock import patch

from service import native_package as n
from service.core import Core
from service.dispatcher import Dispatcher
from service.native_activation import NativeActivation
from service.admission import DEFAULT_BUDGET
from tests.service.test_native_lifecycle import NativeLifecycleTests
from worker.engines.qwen_execution import inventory_wire
from worker.transport import canonical


class NativeHistoryTests(unittest.TestCase):
    setUp = NativeLifecycleTests.setUp
    wait = NativeLifecycleTests.wait
    open = NativeLifecycleTests.open
    close = NativeLifecycleTests.close

    def prepare(self):
        self.root.chmod(0o700)
        base = self.f.f.fixture.root
        for name in ('ca','cert','key'):
            path=base/name;path.write_bytes(b'INERT');path.chmod(0o600)
        config=dict(version=1,variant=n.VARIANT,native=self.f.native,host='127.0.0.1',port=4443,
            peer_sha256=self.f.pin,cafile=str(base/'ca'),certfile=str(base/'cert'),keyfile=str(base/'key'),
            spool=str(self.f.spool),staging=str(self.f.stage))
        templates=dict(version=1,variant=n.VARIANT,profiles=self.profiles,budget=DEFAULT_BUDGET,
            inventory=inventory_wire(self.f.f.inventory),registrations=[dict(principal='alice',profile='qwen-mps',voice_version='vv0',context=self.context)])
        for name,value in (('control',config),('templates',templates)):
            path=base/name;path.write_bytes(canonical(value));path.chmod(0o600)
        self.kw=dict(configuration_file=str(base/'control'),templates_file=str(base/'templates'),owner_root=str(self.root),connector=self.f.connect)
        self.record=base/'enrollment'
        tls=patch.object(n.p,'tls_context',return_value=None);tls.start();self.addCleanup(tls.stop)
        n.enroll_native(self.record,**self.kw)
        with patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):
            try:
                d=self.open()
                result=d.call('production_generate','alice',self.context['plan']['request'],key='retained')
                self.job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state']=='succeeded' else None))
                self.wait(lambda:d.call('status')['phase']=='idle')
            finally:self.close()

    def reopen(self):
        self.dispatcher=Dispatcher(self.root,staging=self.f.stage,installed_activation=NativeActivation(self.record,**self.kw))
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        return self.dispatcher

    def test_completed_job_reopens_twice_and_retains_exact_artifact(self):
        self.prepare()
        for _ in range(2):
            try:
                d=self.reopen()
                self.assertFalse(d.call('status')['quarantined'])
                self.assertEqual(d.call('job',self.job['id']),self.job)
                self.assertEqual(d.call('status')['reservations'][0]['state'],'retained')
                self.assertEqual(d.call('production_artifact','alice',dict(artifact_id=self.job['artifact_ids'][0]))['id'],self.job['artifact_ids'][0])
            finally:self.close()
        self.assertEqual(len(self.f.f.calls),1)

    def refusal(self, mutate):
        self.prepare()
        core=Core(self.root)
        try: mutate(core)
        finally:core.close()
        evidence={p:p.read_bytes() for p in self.f.spool.rglob('*') if p.is_file()}
        acks=sum(q['op']=='ack' for q in self.f.requests)
        with self.assertRaises(Exception): self.reopen()
        self.dispatcher=None
        self.assertEqual(evidence,{p:p.read_bytes() for p in self.f.spool.rglob('*') if p.is_file()})
        self.assertEqual(acks,sum(q['op']=='ack' for q in self.f.requests))

    def test_unfinished_reservation_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE resource_reservations SET state='executing'"))

    def test_unknown_schema_refuses(self):
        self.refusal(lambda c:c.db.execute('CREATE TABLE unexpected(evidence TEXT)'))

    def test_changed_schema_refuses(self):
        self.refusal(lambda c:c.db.execute('CREATE INDEX unexpected ON dispatches(job_id)'))

    def test_unconfirmed_ack_refuses(self):
        self.refusal(lambda c:c.db.execute('UPDATE dispatches SET acked=0'))

    def test_wrong_native_terminal_receipt_refuses(self):
        def change(c):
            receipt=json.loads(c.db.execute('SELECT receipt FROM dispatches').fetchone()[0])
            receipt['reap_method']='pid1-waitpid-echild'
            c.db.execute('UPDATE dispatches SET receipt=?',(json.dumps(receipt),))
        self.refusal(change)

    def test_changed_authority_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE dispatches SET binding=?",('f'*64,)))

    def test_missing_import_completion_refuses(self):
        self.refusal(lambda c:next(self.f.spool.glob('*/import-complete')).unlink())

    def test_changed_publication_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE publications SET message='{}'"))

    def test_unknown_staging_evidence_refuses(self):
        self.refusal(lambda c:(self.f.stage/'unknown').write_bytes(b'RETAIN'))

    def test_missing_trust_refuses(self):
        self.refusal(lambda c:__import__('pathlib').Path(self.kw['configuration_file']).with_name('key').unlink())

    def test_changed_boot_refuses_without_learning(self):
        def change(c):
            self.f.mutate=lambda q,v: dict(v,boot='e'*32) if q['op']=='hello' else v
        self.refusal(change)

    def test_remote_terminal_proof_mismatch_refuses(self):
        def change(c):
            def mutate(q,v):
                if q['op']=='collect': v['value']['receipt']['timely']=False
                return v
            self.f.mutate=mutate
        self.refusal(change)

    def test_malformed_job_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE jobs SET job='{}'"))

    def test_changed_import_intent_refuses(self):
        self.refusal(lambda c:next(self.f.spool.glob('*/import-intent')).write_bytes(b'{}'))

    def test_missing_entire_catalogue_refuses_before_core_can_recreate_it(self):
        def remove(c):
            from service.voice_store import _TABLES
            c.db.execute('PRAGMA foreign_keys=OFF')
            for table in _TABLES: c.db.execute('DROP TABLE '+table)
        self.refusal(remove)
        import sqlite3
        with sqlite3.connect(self.root/'core.sqlite') as db:
            self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='voice_catalogue_format'").fetchone())
