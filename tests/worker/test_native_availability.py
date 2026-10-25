"""Inert availability providers, fake TLS and handles. Never calls host_observation."""
import copy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import threading
import unittest
from unittest.mock import patch

from worker.native_availability import Availability, digest
from worker.transport import Refused, canonical
from tests.service import test_native_runtime as fixtures
from service.execution_context import NativeQwenContexts
from service.admission import Admission


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.NativeFixture(self)
        self.now = datetime.now(timezone.utc)
        self.calls = 0
        self.result = 'ok'
        self.f.runtime.utc = lambda: self.now
        self.f.s.availability.utc = lambda: self.now
        self.f.s.availability.provider = self.provider
        self.context = self.f.a.context
        self.contexts = NativeQwenContexts({('alice','qwen-mps','vv0'):self.context},
            self.f.f.inventory, list(self.f.a.profiles), self.f.native)
        self.contexts.bind_availability(self.f.runtime)

    def provider(self, enrolled):
        self.calls += 1
        if self.result == 'failed': raise OSError('inert provider failure')
        device = replace(enrolled.devices[0], available=self.result != 'unavailable',
            probed_at=(self.now+timedelta(seconds={'stale':-301,'future':1}.get(self.result,0))).isoformat())
        return replace(enrolled, devices=(device,))

    def refresh(self): return self.f.runtime.check_availability(self.context)

    def expire(self): self.now += timedelta(seconds=301)

    def test_freshness_does_not_spend_capacity_or_mutate_frozen_identity(self):
        before = canonical(self.f.s.journal.value), self.f.a._snapshot, canonical([r['snapshot'] for r in self.f.s.registrations.values()])
        for _ in range(10):
            self.expire(); self.refresh()
        self.assertEqual(self.calls,10)
        self.assertEqual(before,(canonical(self.f.s.journal.value),self.f.a._snapshot,canonical([r['snapshot'] for r in self.f.s.registrations.values()])))
        self.assertEqual(self.f.s.launches,0)
        self.assertEqual(self.context,self.contexts.resolve('alice','speech_generate',self.context['plan']['request']))
        self.assertEqual(self.calls,10)
        self.assertIsNotNone(self.f.runtime.cached_availability())
        self.expire()
        self.assertIsNone(self.f.runtime.cached_availability())
        self.assertEqual(self.calls,10)  # metadata read never probes

    def test_failed_observations_refuse_resolution_admission_and_prepare_without_launch(self):
        admission = Admission.__new__(Admission); admission.native_availability = self.f.runtime.check_availability
        for result in ('failed','unavailable','stale','future'):
            self.result = result
            for entry in (self.refresh,
                lambda:self.contexts.resolve('alice','speech_generate',self.context['plan']['request']),
                lambda:admission.demand(self.context,list(self.f.a.profiles)),
                lambda:self.f.runtime.prepare(self.f.a),
                lambda:self.f.s.prepare(dict(binding=self.f.a.binding,boot=self.f.s.boot,body=dict(snapshot=self.f.snapshot)))):
                with self.subTest(result=result,entry=entry),self.assertRaises(Refused): entry()
            self.assertEqual(self.f.s.launches,0)
            self.assertFalse(self.f.s.journal.value['records'])
            self.assertFalse(self.f.runtime.failed)

    def test_response_replay_foreign_shape_runtime_types_and_peer_refuse(self):
        original = self.f.mutate
        def mutation(change):
            def mutate(q, reply):
                if q['op']=='availability': change(reply)
                return reply
            return mutate
        changes = [
            lambda r:r.update(request_id='0'*32),
            lambda r:r.update(boot='0'*32),
            lambda r:r['value'].update(enrollment_sha256='0'*64),
            lambda r:r['value'].update(runtime_manifest_sha256='0'*64),
            lambda r:r['value']['inventory']['host'].update(architecture='foreign'),
            lambda r:r['value']['inventory']['devices'][0].update(index=True),
            lambda r:r['value']['inventory']['devices'][0].update(available=1),
            lambda r:r['value']['inventory']['devices'][0].update(probed_at=True),
            lambda r:r['value']['inventory']['devices'][0].update(probed_at=(self.now-timedelta(seconds=301)).isoformat()),
            lambda r:r['value']['inventory']['devices'][0].update(probed_at=(self.now+timedelta(seconds=1)).isoformat())]
        for change in changes:
            self.f.mutate=mutation(change)
            with self.assertRaises(Refused): self.refresh()
        saved=[]
        self.f.mutate=lambda q,r:(saved.append(copy.deepcopy(r)) or r) if q['op']=='availability' else r
        self.refresh()
        self.f.mutate=lambda q,r:saved[0] if q['op']=='availability' else r
        with self.assertRaises(Refused): self.refresh()
        self.f.mutate=original
        self.f.runtime.pin='0'*64
        with self.assertRaises(Refused): self.refresh()
        self.assertEqual(self.f.s.launches,0)

    def test_expiry_prepare_to_start_preserves_authority_stop_collect_ack(self):
        self.f.runtime.allocate_stage(self.f.a)
        rid=self.f.runtime.prepare(self.f.a)
        before=self.f.a._snapshot
        self.expire();self.result='unavailable'
        with self.assertRaises(Refused):self.f.runtime.start(rid)
        # Independent supervisor enforcement when a client skips its own check.
        with self.assertRaises(Refused):self.f.runtime._rpc('start',{},authority=self.f.a,runtime_id=rid)
        self.assertEqual(self.f.s.launches,0);self.assertEqual(before,self.f.a._snapshot)
        self.f.runtime.terminate(rid)
        self.assertEqual(self.f.runtime.observe(rid).state.value,'stopped-and-reaped')
        self.f.runtime._rpc('collect',{},authority=self.f.a,runtime_id=rid)
        self.f.runtime.call('ack',dict(runtime_id=rid))
        self.assertTrue(self.f.s.journal.value['records'][0]['acked'])

    def test_concurrent_refresh_coalesces_and_stop_collect_are_independent(self):
        self.f.runtime.allocate_stage(self.f.a);rid=self.f.runtime.prepare(self.f.a)
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.f.runtime.start(rid)
            self.f.s.active['receiver'].join(1)
        self.expire();entered=threading.Event();release=threading.Event()
        def blocked(enrolled):
            entered.set();release.wait(2);return self.provider(enrolled)
        self.f.s.availability.provider=blocked
        outcomes=[]
        def refresh():
            try: outcomes.append(self.refresh())
            except Exception as e: outcomes.append(e)
        arrivals=[];both=threading.Event();arrival_lock=threading.Lock()
        def before(q):
            if q['op']=='availability':
                with arrival_lock:
                    arrivals.append(q['request_id'])
                    if len(arrivals)==2:both.set()
        self.f.before=before
        threads=[threading.Thread(target=refresh) for _ in range(2)]
        for t in threads:t.start()
        self.assertTrue(entered.wait(1))
        self.assertTrue(both.wait(1))
        # Probe traffic cannot spend the lifecycle slots needed by stop/collect.
        with self.assertRaisesRegex(Refused,'native_busy'):self.refresh()
        self.f.runtime.terminate(rid)
        self.assertTrue(self.f.s.active['cancel'])
        self.f.f.h.code=-15;self.f.s.tick()
        self.f.runtime._rpc('collect',{},authority=self.f.a,runtime_id=rid)
        pending=self.f.s.availability.pending
        release.set();self.assertTrue(pending['event'].wait(1))
        for t in threads:t.join(2);self.assertFalse(t.is_alive())
        self.assertTrue(all(not isinstance(v,Exception) for v in outcomes),repr(outcomes))
        self.assertEqual(self.calls,2) # initial observation plus one shared refresh
        self.assertEqual(len(outcomes),2);self.assertTrue(all(not isinstance(v,Exception) for v in outcomes))
        self.f.runtime.call('ack',dict(runtime_id=rid))

    def test_timeout_single_inflight_late_result_discarded_and_retry_bounded(self):
        entered=threading.Event();release=threading.Event()
        def blocked(enrolled):
            entered.set();release.wait(2);return self.provider(enrolled)
        cache=self.f.s.availability;cache.provider=blocked;cache.timeout=.02
        with self.assertRaises(Refused):self.refresh()
        self.assertTrue(entered.is_set());pending=cache.pending
        for _ in range(3):
            with self.assertRaises(Refused):self.refresh()
            self.assertIs(cache.pending,pending)
        release.set();self.assertTrue(pending['event'].wait(1))
        self.assertIsNone(cache.cached())
        with self.assertRaises(Refused):self.refresh() # consume failed timed-out flight
        cache.provider=self.provider
        self.refresh();self.assertEqual(self.calls,2)
        self.assertEqual(self.f.s.launches,0)

    def test_changed_incarnation_and_static_enrollment_never_learned(self):
        for key, value in (('boot','0'*32),('lane','0'*32),('runtime_manifest_sha256','0'*64)):
            context=copy.deepcopy(self.context);context['execution']['native'][key]=value
            with self.assertRaises(Refused):self.f.runtime.check_availability(context)
        context=copy.deepcopy(self.context);context['execution']['hardware']['devices'][0]['index']=1
        with self.assertRaises(Refused):self.f.runtime.check_availability(context)
        self.assertEqual(self.calls,0)

    def test_measurement_binding_survives_refresh_but_changed_measurement_refuses(self):
        from types import SimpleNamespace
        from contracts import validate as oracle
        from service.production_selection import choose, Measurement, SelectionUnavailable
        from service.profiles import Performance, SelectionPolicy
        dispatcher=SimpleNamespace(runtime=self.f.runtime,profiles=list(self.f.a.profiles),
            admission=SimpleNamespace(budget=dict(ram_bytes=1024**3,vram_bytes=0)))
        measurement=Measurement(oracle.canonical_hash(self.context['execution']),Performance('inert',2,1,'INERT'))
        policy=SelectionPolicy(workload='inert')
        def select(m):
            return choose(dispatcher,[('native',self.context)],self.context['plan']['request'],
                measurements={'native':m},policy=policy)
        first=canonical(select(measurement));self.expire()
        self.assertEqual(first,canonical(select(measurement)));self.assertEqual(self.calls,2)
        with self.assertRaises(SelectionUnavailable):select(Measurement('0'*64,measurement.performance))
        self.assertEqual(self.calls,2)

    def test_default_provider_uses_host_metal_api_without_ml_or_process(self):
        from unittest.mock import Mock
        from worker.native_availability import host_observation
        metal=Mock();metal.MTLCreateSystemDefaultDevice.return_value=42
        objc=Mock();objc.sel_registerName.return_value=7
        with patch('worker.native_availability.host_metadata',return_value=self.f.f.inventory.host), \
             patch('worker.native_availability.ctypes.CDLL',side_effect=[metal,objc]) as library:
            value=host_observation(self.f.f.inventory)
        self.assertTrue(value.devices[0].available)
        metal.MTLCreateSystemDefaultDevice.assert_called_once_with()
        objc.objc_msgSend.assert_called_once_with(42,7)
        self.assertEqual(library.call_count,2)

    def test_positive_expiry_at_start_rechecks_without_mutating_prepared_input(self):
        self.f.runtime.allocate_stage(self.f.a);rid=self.f.runtime.prepare(self.f.a)
        before=self.f.a._snapshot
        self.expire()
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.f.runtime.start(rid);self.f.s.active['receiver'].join(1)
        self.assertEqual(self.calls,2);self.assertEqual(self.f.a._snapshot,before)
        self.expire();self.result='failed'
        # Failure after launch cannot prevent the owned terminal proof.
        self.f.runtime.terminate(rid);self.f.f.h.code=-15;self.f.s.tick()
        receipt=self.f.runtime._rpc('collect',{},authority=self.f.a,runtime_id=rid)['receipt']
        self.assertTrue(receipt['reaped']);self.f.runtime.call('ack',dict(runtime_id=rid))
        self.assertEqual(self.calls,2)

    def test_cancel_prepared_request_during_refresh_never_launches(self):
        self.f.runtime.allocate_stage(self.f.a);rid=self.f.runtime.prepare(self.f.a)
        self.expire();entered=threading.Event();release=threading.Event()
        def provider(enrolled):
            entered.set();release.wait(1);return self.provider(enrolled)
        self.f.s.availability.provider=provider
        outcomes=[]
        def start():
            try:self.f.runtime.start(rid)
            except Refused as e:outcomes.append(str(e))
        thread=threading.Thread(target=start);thread.start()
        self.assertTrue(entered.wait(1));self.f.runtime.terminate(rid);release.set()
        thread.join(2);self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes,['native_cancelled']);self.assertEqual(self.f.s.launches,0)
        self.f.runtime._rpc('collect',{},authority=self.f.a,runtime_id=rid)
        self.f.runtime.call('ack',dict(runtime_id=rid))
