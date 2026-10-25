"""Tiny source histories; actual Core and native proofs, fake TLS/process/backend."""
import copy
import json
import sqlite3
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

    def prepare(self, *, observe_count=1):
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
                if observe_count > 1:
                    # Let the real Dispatcher record repeated ALIVE observations;
                    # only the final one gets the fixture's actual terminal proof.
                    observed=0
                    def before(q):
                        nonlocal observed
                        if q['op']=='observe':
                            observed+=1
                            if observed >= observe_count and self.f.s.active:
                                self.f.finish()
                            self.f.s.tick()
                    self.f.before=before
                result=d.call('production_generate','alice',self.context['plan']['request'],key='retained')
                self.job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state']=='succeeded' else None))
                self.wait(lambda:d.call('status')['phase']=='idle')
            finally:self.close()

    def reopen(self):
        self.activation=NativeActivation(self.record,**self.kw)
        self.dispatcher=Dispatcher(self.root,staging=self.f.stage,installed_activation=self.activation)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        return self.dispatcher

    def test_completed_job_reopens_twice_and_retains_exact_artifact(self):
        self.prepare()
        for _ in range(2):
            try:
                d=self.reopen()
                self.assertFalse(d.call('status')['quarantined'])
                self.assertEqual(d.call('production_generate','alice',self.context['plan']['request'],key='retained')['job_id'],self.job['id'])
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
        def evidence():
            return {p:p.read_bytes() for root in (self.f.spool,self.f.stage,self.root/'assets')
                    for p in root.rglob('*') if p.is_file()}
        saved=evidence()
        def database():
            with sqlite3.connect(self.root/'core.sqlite') as db:
                return list(db.iterdump())
        durable=database()
        acks=sum(q['op']=='ack' for q in self.f.requests)
        with self.assertRaises(Exception): self.reopen()
        activation=self.activation
        if hasattr(activation,'runtime'):
            self.assertTrue(activation.runtime.import_closed)
            self.assertFalse(activation.runtime.import_completed)
        self.dispatcher=None
        self.assertEqual(saved,evidence())
        self.assertEqual(durable,database())
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


    def change_document(self, table, column, mutate):
        def change(c):
            value=json.loads(c.db.execute('SELECT '+column+' FROM '+table).fetchone()[0])
            mutate(value)
            c.db.execute('UPDATE '+table+' SET '+column+'=?',(json.dumps(value),))
        self.refusal(change)

    def test_negative_retained_dimensions_refuse(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(
            slots=-1,ram_bytes=-1,pids=-1,disk_bytes=-1,attempt_disk_bytes=-1))

    def test_self_retry_refuses(self):
        self.refusal(lambda c:c.db.execute('UPDATE jobs SET retry_of=id'))

    def test_wrong_request_fingerprint_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE jobs SET fingerprint='not-the-request-fingerprint'"))

    def test_unknown_returned_command_refuses(self):
        self.refusal(lambda c:c.db.execute("INSERT INTO dispatch_commands(dispatch_id,op,state) SELECT seq,'unrecognized-operation','returned' FROM dispatches"))

    def test_unknown_command_state_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE dispatch_commands SET state='unknown' WHERE op='ack'"))

    def test_missing_start_command_refuses(self):
        self.refusal(lambda c:c.db.execute("DELETE FROM dispatch_commands WHERE op='start'"))

    def test_collect_without_observe_refuses(self):
        self.refusal(lambda c:c.db.execute("DELETE FROM dispatch_commands WHERE op='observe'"))

    def rewrite_commands(self, c, ops):
        dispatch=c.db.execute('SELECT seq FROM dispatches').fetchone()[0]
        c.db.execute('DELETE FROM dispatch_commands')
        c.db.executemany('INSERT INTO dispatch_commands(dispatch_id,op,state) VALUES (?, ?, ?)',
                        [(dispatch,op,'returned') for op in ops])

    def test_duplicate_returned_collect_refuses(self):
        self.refusal(lambda c:self.rewrite_commands(c,
            ['prepare','start','observe','collect','collect','ack']))

    def test_stop_between_terminal_observe_and_collect_refuses(self):
        self.refusal(lambda c:self.rewrite_commands(c,
            ['prepare','start','observe','stop','collect','ack']))

    def test_observe_after_collect_refuses(self):
        self.refusal(lambda c:self.rewrite_commands(c,
            ['prepare','start','observe','collect','observe','ack']))

    def test_missing_ack_command_refuses(self):
        self.refusal(lambda c:c.db.execute("DELETE FROM dispatch_commands WHERE op='ack'"))

    def test_actual_repeated_observations_reopen(self):
        self.prepare(observe_count=3)
        with sqlite3.connect(self.root/'core.sqlite') as db:
            ops=[r[0] for r in db.execute('SELECT op FROM dispatch_commands ORDER BY seq')]
        self.assertEqual(ops,['prepare','start','observe','observe','observe','collect','ack'])
        try:
            self.assertFalse(self.reopen().call('status')['quarantined'])
        finally:self.close()

    def test_command_order_refuses(self):
        self.refusal(lambda c:c.db.execute("UPDATE dispatch_commands SET op='ack' WHERE op='prepare'"))

    def test_job_generation_float_refuses(self):
        self.change_document('jobs','job',lambda j:j.update(generation=float(j['generation'])))

    def test_outcome_generation_float_refuses(self):
        self.change_document('dispatches','outcome',lambda j:j.update(generation=float(j['generation'])))

    def test_context_primitive_substitution_refuses(self):
        self.change_document('jobs','context',lambda c:c['execution']['descriptor'].update(
            memory_bytes=float(c['execution']['descriptor']['memory_bytes'])))

    def test_artifact_primitive_substitution_refuses(self):
        self.change_document('artifacts','metadata',lambda a:a.update(generation=float(a['generation'])))

    def test_publication_intent_generation_float_refuses_before_startup(self):
        self.change_document('publication_work','binding',lambda a:a.update(generation=float(a['generation'])))

    def test_publication_intent_revision_float_refuses_before_startup(self):
        self.change_document('publication_work','binding',lambda a:a.update(project_revision=float(a['project_revision'])))

    def change_artifact_copies(self, key):
        # Keep all three copies identical: isolate their join to frozen job/plan.
        def change(c):
            for table,column in (('publications','artifact'),('artifacts','metadata'),('publication_work','binding')):
                value=json.loads(c.db.execute('SELECT '+column+' FROM '+table).fetchone()[0])
                value[key]=float(value[key])
                c.db.execute('UPDATE '+table+' SET '+column+'=?',(json.dumps(value),))
        self.refusal(change)

    def test_identical_publication_copies_generation_float_refuse(self):
        self.change_artifact_copies('generation')

    def test_identical_publication_copies_revision_float_refuse(self):
        self.change_artifact_copies('project_revision')

    def test_retained_float_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(slots=1.0))

    def test_retained_bool_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(slots=True))

    def test_retained_zero_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(pids=0))

    def test_retained_overflow_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(ram_bytes=2**53))

    def test_retained_nonfinite_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(ram_bytes=float('inf')))

    def test_retained_positive_wrong_reservation_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(disk_bytes=d['disk_bytes']+1))

    def test_retained_unknown_dimension_refuses(self):
        self.change_document('resource_reservations','demand',lambda d:d.update(extra=1))
