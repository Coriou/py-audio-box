"""Pre-r335 row compatibility, using only tiny SQLite and never-started jobs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contracts import validate as oracle
from service.core import Core, encode


class HistoricalRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'core'
        self.core = Core(self.root, historical_local=True)
        self.addCleanup(lambda: self.core.close())
        self.context = json.loads((Path(__file__).resolve().parents[2]/
            'contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
        self.source = self.source_job('source')
        self.other = self.source_job('other')

    def source_job(self, key):
        job = self.core.submit_execution('p', key, self.context)
        self.core.request_cancel(job['id']); self.core.reconcile()
        return job['id']

    def snapshot(self):
        return {table: self.core.db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
                for table in ('jobs','events')}

    def legacy(self):
        job = self.core.submit_execution('p','retry',self.context,retry_of=self.source)
        job.pop('retry_of')
        self.core.db.execute('UPDATE jobs SET fingerprint=?,job=? WHERE id=?',
            (oracle.fingerprint('p','speech_generate',self.context['plan']['request']),encode(job),job['id']))
        self.core.close(); self.core = Core(self.root,historical_local=True)
        return job

    def refuse(self, key, context=None, retry_of=None):
        before = self.snapshot()
        with self.assertRaisesRegex(oracle.ContractError,'idempotency_conflict'):
            self.core.submit_execution('p',key,context or self.context,retry_of=retry_of)
        self.assertEqual(self.snapshot(),before)

    def test_prechange_reopen_exact_replay_and_collisions_unchanged(self):
        job = self.legacy(); before = self.snapshot()
        self.assertEqual(self.core.submit_execution('p','retry',self.context,retry_of=self.source),job)
        self.assertEqual(self.snapshot(),before)
        self.refuse('retry')
        self.refuse('retry',retry_of=self.other)
        self.refuse('source',retry_of=self.source)
        changed = copy.deepcopy(self.context); changed['plan']['request']['text']='changed'
        self.refuse('retry',changed,self.source)
        changed = copy.deepcopy(self.context)
        limits = changed['plan']['resolved']['profile']['limits']; limits['queue']=float(limits['queue'])
        self.refuse('retry',changed,self.source)
        self.assertIsNone(self.core.execution_duplicate('foreign','speech_generate','retry',self.context['plan']['request'],context=self.context,retry_of=self.source))
        self.assertIsNone(self.core.execution_duplicate('p','voice_prepare','retry',self.context['plan']['request'],context=self.context,retry_of=self.source))
        self.assertEqual(self.snapshot(),before)

    def test_transaction_duplicate_race_uses_same_lineage_rules(self):
        job = self.legacy(); before = self.snapshot()
        # At transaction entry a committed competing row becomes visible: hide
        # it from the optimistic read only, retaining real transaction checks.
        lookup = self.core.execution_duplicate
        for source, succeeds in ((self.source,True),(self.other,False),(None,False)):
            calls = []
            def race(*args, **kwargs):
                calls.append(1)
                return None if len(calls)==1 else lookup(*args, **kwargs)
            with patch.object(self.core,'execution_duplicate',side_effect=race):
                if succeeds:
                    self.assertEqual(self.core.submit_execution('p','retry',self.context,retry_of=source),job)
                else:
                    self.refuse('retry',retry_of=source)
            self.assertEqual(len(calls),2)
            self.assertEqual(self.snapshot(),before)

    def test_new_fingerprint_and_durable_context_hash_source_guards(self):
        job = self.core.submit_execution('p','new',self.context,retry_of=self.source)
        self.assertEqual(self.core.submit_execution('p','new',self.context,retry_of=self.source),job)
        self.refuse('new'); self.refuse('new',retry_of=self.other)
        self.legacy()
        for column, value in (('retry_of',None),('retry_of',self.other),
                              ('context',encode(dict(self.context, extra=True)))):
            original = self.core.db.execute('SELECT '+column+' FROM jobs WHERE key=?',('retry',)).fetchone()[0]
            self.core.db.execute('UPDATE jobs SET '+column+'=? WHERE key=?',(value,'retry'))
            self.refuse('retry',retry_of=self.source)
            self.core.db.execute('UPDATE jobs SET '+column+'=? WHERE key=?',(original,'retry'))
        source_job = self.core.job(self.source); source_job['input_sha256']='0'*64
        self.core.db.execute('UPDATE jobs SET job=? WHERE id=?',(encode(source_job),self.source))
        self.refuse('retry',retry_of=self.source)
