"""Injected verification delay, real owner queue and durable Qwen submission."""
import threading
import time
from unittest.mock import patch

from service.dispatcher import Dispatcher
from service.production import Production
from tests.service.test_qwen_preparation import CPUPreparationFixture


class OwnerAcknowledgmentTests(CPUPreparationFixture):
    def test_generate_receipt_survives_slow_owned_resolution(self):
        b = self.b
        authority, proposal, stage = self.proposal()
        self.finish(authority, proposal, stage)
        candidate = Production(b.core).results('alice', authority.job['id'])[0]
        version = Production(b.core).save('alice', 'ack-seed', authority.job['id'],
                                           candidate['id'], 'ack-clone', expected_revision=0)['version']
        request = dict(b.context['plan']['request'], voice_version_id=version['id'])
        root = b.core.root
        b.core.close()
        original = self.contexts.resolve_owned

        def delayed(*args, **kwargs):
            time.sleep(3.2)
            return original(*args, **kwargs)

        with patch.object(Dispatcher, '_tick', lambda owner: None):
            d = Dispatcher(root, staging=b.stage, runtime=object(),
                           profiles=self.profiles, production_contexts=self.contexts)
            try:
                b.core = d.core
                d.reconciled = True
                with patch.object(self.contexts, 'resolve_owned', side_effect=delayed):
                    receipt = d.call('production_generate', 'alice', request, key='known-generate')
                    self.assertEqual(receipt['state'], 'queued')
                    self.assertEqual(d.call('production_generate', 'alice', request,
                                            key='known-generate'), receipt)
                    self.assertEqual(d.call('production_job', 'alice',
                                            dict(job_id=receipt['job_id']))['idempotency_key'], 'known-generate')
                    with self.assertRaises(ValueError):
                        d.call('production_job', 'bob', dict(job_id=receipt['job_id']))
            finally:
                d.close()

    def test_prepare_receipt_survives_slow_owned_resolution(self):
        b = self.b
        root = b.core.root
        b.core.close()
        original = self.contexts.resolve_owned

        def delayed(*args, **kwargs):
            time.sleep(3.2)
            return original(*args, **kwargs)

        with patch.object(Dispatcher, '_tick', lambda owner: None):
            d = Dispatcher(root, staging=b.stage, runtime=object(),
                           profiles=self.profiles, production_contexts=self.contexts)
            try:
                b.core = d.core
                d.reconciled = True
                with patch.object(self.contexts, 'resolve_owned', side_effect=delayed):
                    receipt = d.call('production_prepare', 'alice', self.request, key='known-receipt')
                    self.assertEqual(receipt['state'], 'queued')
                    self.assertEqual(d.call('production_prepare', 'alice', self.request,
                                            key='known-receipt'), receipt)
                    self.assertEqual(d.call('production_job', 'alice',
                                            dict(job_id=receipt['job_id']))['idempotency_key'], 'known-receipt')
                    with self.assertRaises(ValueError):
                        d.call('production_job', 'bob', dict(job_id=receipt['job_id']))
                    self.assertEqual(len(d.call('jobs', 'alice')), 1)
            finally:
                d.close()

    def test_verification_can_outlive_caller_without_cancelling_commit(self):
        b = self.b
        root = b.core.root
        b.core.close()
        entered = threading.Event()
        finished = threading.Event()
        original = self.contexts.resolve_owned

        def delayed(*args, **kwargs):
            entered.set()
            time.sleep(3.2)
            result = original(*args, **kwargs)
            finished.set()
            return result

        # Isolate submission from subsequent worker execution: no runtime or model.
        with patch.object(Dispatcher, '_tick', lambda owner: None):
            d = Dispatcher(root, staging=b.stage, runtime=object(),
                           profiles=self.profiles, production_contexts=self.contexts)
            try:
                b.core = d.core
                d.reconciled = True
                with patch.object(self.contexts, 'resolve_owned', side_effect=delayed):
                    with self.assertRaises(TimeoutError):
                        d.call('production_prepare', 'alice', self.request,
                               key='late-receipt', timeout=3)
                    self.assertTrue(entered.is_set())
                    self.assertFalse(finished.is_set())
                    jobs = d.call('jobs', 'alice')
                self.assertEqual(len(jobs), 1)
                job = jobs[0]['job']
                self.assertEqual(job['idempotency_key'], 'late-receipt')
                self.assertEqual(job['state'], 'queued')
                self.assertTrue(finished.is_set())
                receipt = d.call('production_prepare', 'alice', self.request,
                                 key='late-receipt')
                self.assertEqual(receipt['job_id'], job['id'])
                with self.assertRaises(ValueError):
                    d.call('production_job', 'bob', dict(job_id=job['id']))
            finally:
                d.close()
