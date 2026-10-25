"""Real owner/admission/Authority with inert sealed Qwen bundles; no ML or child."""
import asyncio
import contextlib
import copy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import httpx
from clients.cli import main
from clients import mcp
from contracts import generate, validate as oracle
from service.admission import DEFAULT_BUDGET
from service.api import create_voice_app
from service.auth import Credential, StaticCredentials
from service.dispatcher import Dispatcher
from service.execution_context import QwenContexts
from service.profiles import Performance, SelectionPolicy
from service.production_selection import ContextComposition, Measurement
from tests.service.test_production_api import SyntheticRuntime, ProductionAPIFixture
from tests.service import test_qwen_lifecycle as qwen_fixture
from worker.engines.qwen_execution import inventory_wire
from worker.hardware import Device, DeviceFamily, HardwareInventory


class ProductionSelectionTests(unittest.TestCase):
    request = ProductionAPIFixture.request
    wait = ProductionAPIFixture.wait
    task = 'custom_voice'
    device = 'cuda:0'

    def setUp(self):
        fixture = qwen_fixture.QwenLifecycleTests('test_refusals_before_allocation')
        fixture.task = self.task; fixture.device = self.device
        original = json.dumps
        def manifest(value, *args, **kwargs):
            if type(value) is dict and value.get('backend') == 'qwen':
                value = copy.deepcopy(value)
                value['execution']['devices'].append(dict(device='cuda:1',dtype='float32',attention='sdpa'))
            return original(value, *args, **kwargs)
        with patch('json.dumps', side_effect=manifest): fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.tmp = fixture.tmp
        self.root = fixture.core.root; self.stage = fixture.stage
        fixture.core.close()
        self.profiles = copy.deepcopy(fixture.profiles)
        self.profiles[0]['limits']['expensive_slots'] = 2
        self.inventory = HardwareInventory(fixture.inventory.host, tuple(
            Device(DeviceFamily.CUDA, i, True, True, datetime.now(timezone.utc).isoformat()) for i in (0, 1)))
        self.snapshots = {}
        providers = []
        for i in (0, 1):
            c = copy.deepcopy(fixture.context)
            c['execution']['hardware'] = inventory_wire(self.inventory)
            c['execution']['descriptor'] = replace(fixture.spec, device='cuda:'+str(i)).identity()
            c['plan']['resolved']['profile'] = oracle.profile_snapshot(self.profiles[0])
            self.snapshots['gpu'+str(i)] = c
            providers.append(('gpu'+str(i), QwenContexts({'alice':c}, fixture.mapping, {}, self.inventory, self.profiles)))
        self.runtime = SyntheticRuntime(self.stage); self.runtime.hold = True
        self.measurements = {name:Measurement(oracle.canonical_hash(c['execution']),
            Performance('inert', 4, 2 if name == 'gpu0' else 1, 'SYNTHETIC selection fixture only'))
            for name, c in self.snapshots.items()}
        self.composition = ContextComposition(providers, runtime=self.runtime, inventory=self.inventory,
            measurements=self.measurements, policy=SelectionPolicy(workload='inert'))
        self.dispatcher = Dispatcher(self.root, staging=self.stage, runtime=self.runtime,
            budget=dict(DEFAULT_BUDGET,vram_bytes=4096), profiles=self.profiles,
            production_contexts=self.composition)
        self.addCleanup(self.stop)
        self.credentials = StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'}))])
        self.app = create_voice_app(self.dispatcher, credentials=self.credentials)
        self.payload = copy.deepcopy(fixture.context['plan']['request'])
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])

    def stop(self):
        for row in self.dispatcher.call('jobs','alice'):
            job = row['job']
            if job['state'] in ('queued','running'):
                self.dispatcher.call('cancel',job['id'])
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        self.dispatcher.close()

    def submit(self, payload=None, key='generate'):
        r = self.request('POST','/v1/operations/speech_generate',payload=payload or self.payload,key=key)
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def cancel(self, receipt):
        self.dispatcher.call('cancel',receipt['job_id'])
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')

    def refused(self, reasons, payload=None, key='refuse'):
        before = self.dispatcher.call('jobs','alice')
        response = self.request('POST','/v1/operations/speech_generate',payload=payload or self.payload,key=key)
        self.assertEqual(response.status_code,503,response.text)
        self.assertEqual(response.json()['code'],'not_ready',response.text)
        for reason in reasons: self.assertIn(reason,response.json()['action'])
        self.assertEqual(self.dispatcher.call('jobs','alice'),before)
        self.assertEqual(self.dispatcher.call('status')['reservations'],[])

    def test_measured_ranking_dispatches_selected_identity_and_replay_never_resolves(self):
        receipt = self.submit()
        self.wait(lambda:self.runtime.authorities)
        authority = self.runtime.authorities[0]
        self.assertEqual(authority.context['execution'],self.snapshots['gpu1']['execution'])
        self.assertEqual(authority.context['plan']['voice_versions'],[self.dispatcher.call('voice_version','alice',self.payload['voice_version_id'])])
        self.assertEqual(authority.context['plan']['request'],self.payload)
        self.cancel(receipt)
        # Alter ranking, invalidate inventory and resolver: duplicate lookup precedes all selection.
        self.composition.measurements = {'gpu0':self.measurements['gpu0']}
        self.composition.candidates = ()
        self.dispatcher.close()
        self.dispatcher = Dispatcher(self.root,staging=self.stage,runtime=self.runtime,
            budget=dict(DEFAULT_BUDGET,vram_bytes=4096),profiles=self.profiles,production_contexts=self.composition)
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        self.assertEqual(self.submit(),receipt)
        self.assertEqual(len(self.runtime.authorities),1)
        self.assertEqual(self.dispatcher.call('job',receipt['job_id'])['input_sha256'],receipt['input_sha256'])

    def test_profile_device_override_reaches_dispatch_and_never_falls_back(self):
        payload = dict(self.payload,profile_override='qwen-cuda',device_override=dict(family='cuda',index=0))
        receipt = self.submit(payload)
        self.wait(lambda:self.runtime.authorities)
        self.assertEqual(self.runtime.authorities[0].context['execution']['descriptor']['device'],'cuda:0')
        self.assertEqual(self.runtime.authorities[0].context['plan']['request'],payload)
        self.cancel(receipt)
        # Retained reservations are allowed; no extra job is admitted on mismatch.
        before = len(self.dispatcher.call('jobs','alice'))
        for change in ({'profile_override':'missing'}, {'device_override':dict(family='cpu',index=0)},
                       {'identity':dict(self.payload['identity'],model='other')}):
            r = self.request('POST','/v1/operations/speech_generate',payload=dict(self.payload,**change),key='bad')
            self.assertEqual(r.json()['code'],'not_ready',r.text)
        self.assertEqual(len(self.dispatcher.call('jobs','alice')),before)

    def test_stale_inventory_unqualified_memory_concurrency_and_measurement_refuse(self):
        original = self.composition.inventory
        self.composition.inventory = HardwareInventory(original.host,tuple(replace(d,
            probed_at=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()) for d in original.devices))
        self.refused(['hardware_identity_changed'])
        self.composition.inventory = original
        self.profiles[0]['support'] = 'unqualified'
        # Public adapter uses the owner's detached current profile authority.
        self.dispatcher.close()
        self.dispatcher = Dispatcher(self.root,staging=self.stage,runtime=self.runtime,
            budget=dict(DEFAULT_BUDGET,vram_bytes=4096),profiles=self.profiles,production_contexts=self.composition)
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        self.refused(['profile_unqualified'])

    def test_resource_and_measurement_gates_use_real_admission_path(self):
        self.composition.concurrency = 3
        self.refused(['concurrency_exceeded'])
        self.composition.concurrency = 1
        self.composition.measurements = {}
        self.refused(['measurement_required'])
        self.composition.measurements = dict(self.measurements,gpu1=Measurement('0'*64,self.measurements['gpu1'].performance))
        receipt = self.submit()
        self.wait(lambda:self.runtime.authorities)
        self.assertEqual(self.runtime.authorities[0].context['execution']['descriptor']['device'],'cuda:0')
        self.cancel(receipt)

    def test_memory_and_topology_refuse_without_fake_cpu(self):
        self.dispatcher.close()
        self.dispatcher = Dispatcher(self.root,staging=self.stage,runtime=self.runtime,
            budget=dict(DEFAULT_BUDGET,vram_bytes=0),profiles=self.profiles,production_contexts=self.composition)
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        self.refused(['memory_insufficient'])
        # A different runtime cannot reuse this composition, including a native runtime.
        self.composition.runtime = object()
        self.refused(['runtime_binding_changed'])

    def test_discovery_is_metadata_only_truthful_and_closed_overrides(self):
        with patch.object(QwenContexts,'resolve',side_effect=AssertionError('discovery resolved ML context')):
            caps = self.request('GET','/v1/capabilities').json()
            profile = self.request('GET','/skeleton/profiles/qwen-cuda').json()
        self.assertEqual(caps['profiles'],[profile])
        self.assertEqual(profile['readiness'],'ready')
        self.composition.inventory = HardwareInventory(self.inventory.host,tuple(replace(d,available=False) for d in self.inventory.devices))
        profile = self.request('GET','/skeleton/profiles/qwen-cuda').json()
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('stale_or_unavailable_device',profile['next_action'])
        for override in ({'family':'cuda','index':True},{'family':'cuda','index':0,'path':'/tmp'}, {'family':'auto','index':0}):
            r=self.request('POST','/v1/operations/speech_generate',payload=dict(self.payload,device_override=override),key='bad')
            self.assertEqual(r.json()['code'],'invalid_request')

    def test_cli_mcp_select_through_canonical_http_owner(self):
        outer = self
        class ASGIClient:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def post(self,path,**kwargs): return self.request('POST',path,**kwargs)
            def request(self,method,path,**kwargs):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
        payload = dict(self.payload,profile_override='qwen-cuda',device_override=dict(family='cuda',index=0))
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'writer'}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code=main(['speech','generate','--request',json.dumps(payload),'--idempotency-key','cli'])
            self.assertEqual(code,0,output.getvalue())
            receipt=json.loads(output.getvalue()); self.wait(lambda:self.runtime.authorities)
            self.assertEqual(self.runtime.authorities[-1].context['execution']['descriptor']['device'],'cuda:0')
            self.cancel(receipt)
            result=asyncio.run(mcp.call_tool('speech_speech_generate',dict(payload,idempotency_key='mcp')))
            self.assertFalse(result.isError,result.structuredContent)
            self.wait(lambda:len(self.runtime.authorities)==2)
            self.assertEqual(self.runtime.authorities[-1].context['execution']['descriptor']['device'],'cuda:0')
            self.cancel(result.structuredContent)

    def discovery_profile(self, profile_id='qwen-cuda'):
        with patch.object(QwenContexts,'resolve',side_effect=AssertionError('discovery resolved context')):
            caps = self.request('GET','/v1/capabilities')
            detail = self.request('GET','/skeleton/profiles/'+profile_id)
        self.assertEqual(caps.status_code,200,caps.text)
        self.assertEqual(detail.status_code,200,detail.text)
        profile = detail.json()
        self.assertIn(profile,caps.json()['profiles'])
        return profile

    def test_orphan_measurement_discovery_and_submission_refuse_without_allocation(self):
        self.composition.measurements = {'not-a-candidate':self.measurements['gpu0']}
        profile = self.discovery_profile()
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('measurement_required',profile['next_action'])
        self.refused(['measurement_required'],key='orphan')

    def test_measurement_is_profile_scoped_and_bound_metadata_remains_ready(self):
        self.assertEqual(self.discovery_profile()['readiness'],'ready')
        other = copy.deepcopy(self.profiles[0]); other['id'] = 'other-cuda'
        self.dispatcher.profiles.append(other)
        profile = self.discovery_profile('other-cuda')
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('measurement_required',profile['next_action'])
        self.refused(['candidate_identity_or_readiness_refused'],payload=dict(self.payload,profile_override='other-cuda'),key='other')
        receipt = self.submit(key='bound-positive')
        self.wait(lambda:self.runtime.authorities)
        self.cancel(receipt)

    def test_stale_execution_measurements_never_promise_ready(self):
        self.composition.measurements = {name:Measurement('0'*64,m.performance)
                                         for name,m in self.measurements.items()}
        profile = self.discovery_profile()
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('measurement_required',profile['next_action'])
        self.refused(['measurement_identity_changed'],key='stale-binding')

    def test_unknown_metadata_requires_verification_and_unmeasured_policy_is_explicit(self):
        with patch.object(QwenContexts,'selection_metadata',None):
            profile = self.discovery_profile()
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('measurement_binding_verification_required',profile['next_action'])
        self.composition.policy = SelectionPolicy(workload='different-workload')
        profile = self.discovery_profile()
        self.assertEqual(profile['readiness'],'not_ready')
        self.assertIn('measurement_required',profile['next_action'])
        self.refused(['measurement_required'],key='workload-mismatch')
        self.composition.measurements = {}
        self.composition.policy = SelectionPolicy(allow_unmeasured=True)
        self.assertEqual(self.discovery_profile()['readiness'],'ready')
        receipt = self.submit(key='explicit-unmeasured')
        self.wait(lambda:self.runtime.authorities)
        self.cancel(receipt)

    def assert_discovery_refuses(self, reason):
        profile = self.discovery_profile()
        self.assertEqual(profile['readiness'], 'not_ready')
        self.assertIn(reason, profile['next_action'])

    def test_exact_unavailable_cuda0_never_borrows_available_cuda1(self):
        self.composition.candidates = self.composition.candidates[:1]
        self.composition.measurements = {'gpu0': self.measurements['gpu0']}
        self.composition.inventory = HardwareInventory(self.inventory.host, tuple(
            replace(d, available=False) if d.index == 0 else d for d in self.inventory.devices))
        self.assert_discovery_refuses('hardware_identity_changed')
        self.refused(['hardware_identity_changed'], key='exact-unavailable')

    def test_same_index_changed_hardware_identity_requires_requalification(self):
        self.composition.inventory = replace(self.inventory,
            host=replace(self.inventory.host, architecture='changed-same-indices'))
        self.assert_discovery_refuses('hardware_identity_changed')
        self.refused(['hardware_identity_changed'], key='same-index-changed-host')

    def test_exact_device_predicate_even_with_matching_hardware_wire(self):
        # Detached metadata can describe an unavailable device; a matching receipt
        # alone is insufficient. Keep all hashes and hardware wires consistent.
        inv = replace(self.inventory, devices=tuple(
            replace(d, available=False) if d.index == 0 else d for d in self.inventory.devices))
        self.composition.inventory = inv
        name, provider = self.composition.candidates[0]
        self.composition.candidates = ((name, provider),)
        context = copy.deepcopy(self.snapshots[name])
        context['execution']['hardware'] = inventory_wire(inv)
        provider._contexts['alice'] = json.dumps(context)
        provider._inventory = inv
        self.composition.measurements = {name: Measurement(oracle.canonical_hash(context['execution']),
                                                        self.measurements[name].performance)}
        self.assert_discovery_refuses('device_unavailable')
        self.refused(['candidate_identity_or_readiness_refused'], key='matching-unavailable')

    def test_candidates_cannot_mix_measurement_and_current_hardware_witnesses(self):
        # Only gpu0 is measured but its hardware binding is stale. gpu1 has
        # current hardware and availability but no measurement of its own.
        name, provider = self.composition.candidates[0]
        context = copy.deepcopy(self.snapshots[name])
        old = replace(self.inventory, host=replace(self.inventory.host, architecture='old'))
        context['execution']['hardware'] = inventory_wire(old)
        provider._contexts['alice'] = json.dumps(context)
        self.composition.measurements = {name: Measurement(oracle.canonical_hash(context['execution']),
                                                        self.measurements[name].performance)}
        self.assert_discovery_refuses('hardware_identity_changed')
        self.refused(['measurement_required', 'hardware_identity_changed'], key='mixed-witnesses')
        # One complete valid candidate suffices despite the invalid alternative.
        self.composition.measurements['gpu1'] = self.measurements['gpu1']
        self.assertEqual(self.discovery_profile()['readiness'], 'ready')
        receipt = self.submit(key='same-candidate-valid')
        self.wait(lambda: self.runtime.authorities)
        self.assertEqual(self.runtime.authorities[-1].context['execution']['descriptor']['device'], 'cuda:1')
        self.cancel(receipt)

    def test_metadata_resources_concurrency_and_runtime_composition_agree(self):
        for field in ('ram_bytes', 'vram_bytes'):
            with self.subTest(resource=field):
                original = self.dispatcher.admission.budget[field]
                self.dispatcher.admission.budget[field] = self.profiles[0]['limits'][field] - 1
                self.assert_discovery_refuses('memory_insufficient')
                self.refused(['memory_insufficient'], key='metadata-'+field)
                self.dispatcher.admission.budget[field] = original
        self.composition.concurrency = 3
        self.assert_discovery_refuses('concurrency_exceeded')
        self.refused(['concurrency_exceeded'], key='metadata-concurrency')
        self.composition.concurrency = 1
        self.composition.runtime = object()
        self.assert_discovery_refuses('runtime_binding_changed')
        self.refused(['runtime_binding_changed'], key='metadata-runtime')
        self.composition.runtime = self.runtime
        self.dispatcher.installed_checked = True
        try:
            self.assert_discovery_refuses('installed_composition_unsupported')
            # Exercise the same pre-resolution composition gate directly; this
            # fixture has no installed activation and must not fake enrollment.
            from service.production_selection import SelectionUnavailable
            with self.assertRaises(SelectionUnavailable) as caught:
                self.composition.resolve_selected(self.dispatcher, 'alice', 'speech_generate', self.payload)
            self.assertIn('installed_composition_unsupported', caught.exception.reasons)
        finally:
            self.dispatcher.installed_checked = False

    def test_metadata_runtime_topology_and_unknown_unmeasured_are_conservative(self):
        self.runtime.native = {'changed': 'native binding'}
        self.assert_discovery_refuses('runtime_topology_incompatible')
        self.refused(['runtime_topology_incompatible'], key='metadata-topology')
        del self.runtime.native
        self.composition.policy = SelectionPolicy(allow_unmeasured=True)
        with patch.object(QwenContexts, 'selection_metadata', None):
            self.assert_discovery_refuses('measurement_binding_verification_required')

    def test_selection_metadata_is_detached_and_contains_no_owned_media(self):
        provider = self.composition.candidates[0][1]
        facts = provider.selection_metadata()[0]
        self.assertEqual(set(facts), {'profile', 'execution_sha256', 'engine', 'device',
                                     'memory_bytes', 'threads', 'hardware', 'native'})
        facts['hardware']['host']['architecture'] = 'tampered'
        facts['profile']['id'] = 'tampered'
        self.assertEqual(self.discovery_profile()['readiness'], 'ready')


class SelectionContractTests(unittest.TestCase):
    def test_generator_filename_and_byte_determinism_and_existing_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'contracts'; (root/'v1').mkdir(parents=True)
            for name in ('schema.json','operations.json'): shutil.copyfile(Path('contracts/v1')/name,root/'v1'/name)
            output=root/'fixtures/v1'; output.mkdir(parents=True)
            with patch.object(generate,'ROOT',output):
                generate.generate()
                first={p.name:p.read_bytes() for p in output.iterdir()}
                generate.generate()
                self.assertEqual(first,{p.name:p.read_bytes() for p in output.iterdir()})
            self.assertEqual(first,{p.name:p.read_bytes() for p in Path('contracts/fixtures/v1').iterdir() if p.is_file() and p.name in first})
        for name in ('voice_prepare','speech_generate'):
            schema=oracle.SCHEMA['$defs'][name+'_request']
            self.assertIn('profile_override',schema['properties'])
            self.assertNotIn('profile_override',schema['required'])


if __name__ == '__main__': unittest.main()
