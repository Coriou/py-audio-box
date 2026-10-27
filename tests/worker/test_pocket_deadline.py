"""Pocket-only finite deadline tests; no model/process/container execution."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from contracts import validate as oracle
from service.worker_transport import Authority
from tests.service import test_pocket_lifecycle as lifecycle
from tests.service import test_pocket_preparation as preparation
from worker import runtime_child as child
from worker.runtime_supervisor import RuntimeSupervisor
from worker.transport import canonical, Refused


class PocketDeadlineTests(unittest.TestCase):
    def fixture(self,preparing=False,seconds=180):
        fixture=preparation.PreparationFixture() if preparing else lifecycle.PocketLifecycleTests()
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.profiles[0]['limits']['deadline_seconds']=seconds
        fixture.context['plan']['resolved']['profile']=oracle.profile_snapshot(fixture.profiles[0])
        return fixture

    def test_prepare_and_generation_freeze_pocket_only_180_ceiling(self):
        for preparing in (False,True):
            for seconds,expected in ((30,30),(180,180),(300,180)):
                with self.subTest(preparing=preparing,seconds=seconds):
                    f=self.fixture(preparing,seconds)
                    from service.production import Production
                    core=f.base.core if preparing else f.core
                    admission=f.base.admission if preparing else f.admission
                    operation=Production(core).prepare if preparing else Production(core).generate
                    job=operation('alice','deadline',f.context,_admission=admission,_profiles=f.profiles)
                    self.assertEqual(core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(job['id'],)).fetchone(),('pocket',expected))

    def supervisor(self,snapshot):
        s=RuntimeSupervisor.__new__(RuntimeSupervisor)
        s.bundles=snapshot[1]['execution']['bundles'];s.verify_pocket=Mock()
        s.failed=False;s.active=None;s.cutoffs={};s.boot='b'*32
        s.journal=SimpleNamespace(value=dict(lane='a'*32,records=[]))
        s.tree_lock=threading.Lock()
        s.persist=lambda value:setattr(s.journal,'value',value)
        return s

    def test_supervisor_accepts_long_original_pocket_budget_rejects_above_180(self):
        f=self.fixture(seconds=300)
        from service.production import Production
        Production(f.core).generate('alice','deadline',f.context,_admission=f.admission,_profiles=f.profiles)
        a=f.store.allocate(f.profiles,'pocket',180)['authority']
        original=json.loads(a._snapshot)
        for remaining,accepted in ((30,True),(179,True),(181,False)):
            snapshot=copy.deepcopy(original)
            snapshot[0]['deadline']=(datetime.now(timezone.utc)+timedelta(seconds=remaining)).isoformat()
            s=self.supervisor(snapshot)
            body=dict(snapshot=snapshot,binding=hashlib.sha256(canonical(snapshot)).hexdigest(),fixture='pocket')
            with self.subTest(remaining=remaining):
                if accepted:
                    before=child.time.monotonic()
                    result=s.operation('prepare',body)
                    self.assertLessEqual(s.cutoffs[result['runtime_id']],before+remaining+1)
                    cutoff=s.cutoffs[result['runtime_id']]
                    self.assertEqual(s.operation('prepare',body),result)
                    self.assertEqual(s.cutoffs[result['runtime_id']],cutoff)
                else:
                    with self.assertRaisesRegex(Refused,'bounded_original_deadline_required'):s.operation('prepare',body)
                    self.assertEqual(s.journal.value['records'],[])

    def test_child_uses_original_anchor_wall_deadline_and_profile_without_renewal(self):
        f=self.fixture(seconds=300)
        fixed=datetime(2026,10,25,tzinfo=timezone.utc)
        clock=Mock(wraps=datetime);clock.now.return_value=fixed
        config=dict(context=f.context,job=dict(deadline=(fixed+timedelta(seconds=300)).isoformat()),pocket_cutoff=250)
        with patch.object(child,'datetime',clock),patch.object(child.time,'monotonic',return_value=100):
            self.assertEqual(child.pocket_cutoff(config),250)  # Anchor beats the fresh 180s cap.
            self.assertEqual(child.pocket_cutoff(dict(config,pocket_cutoff=1000)),280)
            config['context']['plan']['resolved']['profile']['limits']['deadline_seconds']=30
            self.assertEqual(child.pocket_cutoff(config),130)
            config['context']['plan']['resolved']['profile']['limits']['deadline_seconds']=180
            config['job']['deadline']=(fixed+timedelta(seconds=10)).isoformat()
            self.assertEqual(child.pocket_cutoff(config),110)
            config['job']['deadline']=(fixed-timedelta(seconds=1)).isoformat()
            self.assertEqual(child.pocket_cutoff(config),100)
            for invalid in (None,True,0,-1,float('inf'),float('nan')):
                with self.subTest(anchor=invalid),self.assertRaises(ValueError):child.pocket_cutoff(dict(config,pocket_cutoff=invalid))

    def test_main_preserves_non_cpu_30_bound_and_cpu_pocket_anchors(self):
        for mode,device,expected in (('kokoro','cpu',130),('qwen','mps',130),('qwen','cuda:0',130),('qwen','cpu',1900),('pocket','cpu',220)):
            config=dict(fixture=mode,context={})
            events=[]
            with self.subTest(mode=mode,device=device),patch('sys.argv',['child','3']),patch.object(child.socket,'socket'),patch.object(child,'receive',return_value=config),patch('worker.runtime_protocol.execution',return_value=SimpleNamespace(engine=mode,device=device)),patch('worker.runtime_protocol.dispatch_mode',return_value=mode),patch.object(child,'contain_kokoro',side_effect=lambda context:events.append('limits')),patch.object(child,'pocket_cutoff',return_value=220) as cutoff,patch.object(child,'qwen_cpu_cutoff',return_value=1900) as qwen_cutoff,patch.object(child.time,'monotonic',return_value=100),patch.object(child,'run_generation',side_effect=lambda *args:events.append(args[-1])):
                child.main()
                self.assertEqual(events,['limits',expected])
                if mode=='pocket':cutoff.assert_called_once_with(config)
                else:cutoff.assert_not_called()
                if mode=='qwen' and device=='cpu':qwen_cutoff.assert_called_once_with(config)
                else:qwen_cutoff.assert_not_called()
