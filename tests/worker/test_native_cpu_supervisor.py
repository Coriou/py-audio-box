"""Existing native lifecycle with inert CPU policy/handles, never real ML."""
import hashlib
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from worker import native_cpu_protocol as p, native_protocol as qwen
from worker.native_cpu_supervisor import NativeCPUSupervisor
from worker.transport import canonical,Refused
from tests.worker.native_cpu_fixtures import Fixture,Handle,Channel,Sampler


class CPUSupervisorTests(unittest.TestCase):
    def test_base_factory_keeps_legacy_journal_injection_separate_from_cpu(self):
        from worker.native_supervisor import NativeSupervisor
        with patch('worker.native_supervisor.Journal',side_effect=Refused('inert_journal_factory')) as factory:
            with self.assertRaisesRegex(Refused,'inert_journal_factory'):
                NativeSupervisor('/unused-inert-host',deployment_id='a'*32,configuration_sha256='b'*64,
                    peer_pin='c'*64,runtime_manifest_sha256='d'*64,registrations={},python='/managed/python',
                    child_entry=str(Path('worker/native_child.py').resolve()))
        factory.assert_called_once_with('/unused-inert-host','a'*32,'b'*64)

    def setUp(self):
        self.f=Fixture(self);self.h=Handle();self.signals=[];self.calls=[]
        binding=hashlib.sha256(canonical(self.f.snapshot)).hexdigest()
        def popen(*args,**kwargs):self.calls.append((args,kwargs));return self.h
        self.s=NativeCPUSupervisor(self.f.root,deployment_id='a'*32,configuration_sha256='b'*64,peer_pin='c'*64,
            runtime_manifest_sha256='d'*64,registrations={binding:self.f.registration()},python='/managed/python',
            child_entry=str(Path('worker/native_cpu_child.py').resolve()),popen=popen,pair=lambda:(Channel(),Channel()),
            signal_group=lambda *v:self.signals.append(v),identity=lambda h:'4',clock=lambda:self.f.t,utc=lambda:self.f.now,
            free_bytes=lambda:100*1024**3,free_ram=lambda:16*1024**3,availability_provider=self.f.observed,
            sampler_factory=lambda pid:Sampler(pid,self.f))
        self.addCleanup(self.s.journal.lock.close)
        self.f.context['execution']['native'].update(lane=self.s.journal.value['lane'],boot=self.s.boot)
        self.f.snapshot=self.f.snapshot_for(self.f.context);self.binding=hashlib.sha256(canonical(self.f.snapshot)).hexdigest()
        self.s.registrations={self.binding:self.f.registration()};self.s._preset_templates=dict(self.s.registrations)

    def q(self,op,body=None):
        return dict(version=p.VERSION,deployment_id='a'*32,configuration_sha256='b'*64,lane=self.s.journal.value['lane'],
            boot=self.s.boot,request_id='e'*32,runtime_id=None if op=='prepare' else getattr(self,'rid',None),binding=self.binding,op=op,body=body or {})

    def prepare(self):
        self.rid=self.s.dispatch(self.q('prepare',dict(snapshot=self.f.snapshot)))['runtime_id']
        return self.s.journal.value['records'][0]

    def start(self):
        self.prepare()
        with patch.object(self.s,'exchange',return_value=None):
            self.s.dispatch(self.q('start'))
            self.s.active['receiver'].join(1)

    def test_native_cpu_wire_command_and_nonrenewed_budgets(self):
        self.start();args,kwargs=self.calls[0]
        self.assertEqual(args[0][3],str(Path('worker/native_cpu_child.py').resolve()))
        self.assertTrue(kwargs['start_new_session']);self.assertEqual(kwargs['pass_fds'],(42,))
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'],'2');self.assertNotIn('PYTORCH_ENABLE_MPS_FALLBACK',kwargs['env'])
        self.assertEqual(self.s.cutoffs[self.rid],130);self.assertEqual(self.s.job_cutoffs[self.rid],160)
        wrong=self.q('observe');wrong['version']=qwen.VERSION
        with self.assertRaises(Refused):self.s.dispatch(wrong)

    def test_cancel_uses_owned_signals_then_reaped_cpu_receipt(self):
        self.start();self.s.dispatch(self.q('stop'));self.s.tick()
        self.assertEqual(self.signals,[(123,signal.SIGTERM)])
        self.f.t+=.21;self.s.tick();self.assertEqual(self.signals[-1],(123,signal.SIGKILL))
        self.h.code=-9;self.s.tick()
        receipt=self.s.dispatch(self.q('collect'))['receipt']
        self.assertFalse(receipt['timely']);self.assertEqual(receipt['version'],p.RECEIPT_VERSION)
        self.assertEqual(receipt['termination']['reason'],'cancelled')
        self.assertEqual([v['accepted'] for v in receipt['termination']['signals']],[True,True])
        self.assertEqual(self.h.waited,1);self.assertTrue(receipt['resources']['thread_stopped'])
        self.assertNotIn('engine_uid',receipt);self.assertFalse((self.s.directory(self.s.journal.value['records'][0])/'tmp').exists())

    def test_sampler_start_failure_cannot_abandon_spawned_child(self):
        self.s.sampler_factory=lambda pid:(_ for _ in ()).throw(RuntimeError('sampler fixture failure'))
        self.start();self.s.tick();self.assertEqual(self.signals,[(123,signal.SIGTERM)])
        self.h.code=-15;self.s.tick()
        receipt=self.s.dispatch(self.q('collect'))['receipt']
        self.assertEqual(receipt['resources']['failure'],'native_cpu_rss_lost')
        self.assertEqual(receipt['resources']['sampler_error']['message'],'sampler fixture failure')
        self.assertFalse(receipt['timely']);self.assertEqual(self.h.waited,1)

    def test_identity_query_failure_retains_and_reaps_owned_handle_without_signal(self):
        self.prepare();self.s.identity=lambda h:(_ for _ in ()).throw(ProcessLookupError('already gone'))
        with self.assertRaisesRegex(Refused,'native_unknown'):self.s.dispatch(self.q('start'))
        self.assertIs(self.s.active['handle'],self.h)
        self.assertIn('rss_started',self.s.active)
        self.h.code=0;self.s.tick()
        self.assertEqual(self.h.waited,1);self.assertEqual(self.signals,[])
        self.assertTrue(self.s.failed);self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])

    def test_memory_limit_stops_without_an_oom_claim(self):
        self.start();self.s.active['sampler'].peak=2*1024**3+1;self.s.tick()
        self.assertEqual(self.signals,[(123,signal.SIGTERM)])
        self.h.code=-15;self.s.tick();receipt=self.s.dispatch(self.q('collect'))['receipt']
        self.assertEqual(receipt['resources']['failure'],'native_cpu_memory_limit')
        self.assertFalse(receipt['resources']['rss_bound_proven']);self.assertFalse(receipt['timely'])

    def test_success_uses_cpu_proposal_receipt_and_reopens_same_completed_journal(self):
        import io
        import wave
        from worker.native_child import proposal
        self.prepare()
        out=io.BytesIO()
        with wave.open(out,'wb') as wav:
            wav.setparams((1,2,24000,0,'NONE','none'));wav.writeframes(b'\x01\0'*10)
        raw=out.getvalue()
        result=dict(data=raw,frames=10,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),identity={})
        value=proposal(self.f.snapshot,result,self.f.spec,protocol=p)
        with patch.object(self.s,'exchange',return_value=(value,raw)):
            self.s.dispatch(self.q('start'));self.s.active['receiver'].join(1)
        self.h.code=0;self.s.tick()
        self.assertFalse(self.s.failed)
        collected=self.s.dispatch(self.q('collect'))
        self.assertTrue(collected['receipt']['timely']);self.assertEqual(collected['proposal'],value)
        self.assertEqual(collected['receipt']['processes'][0]['start_identity'],'4')
        self.assertTrue(collected['receipt']['resources']['thread_stopped'])
        self.s.dispatch(self.q('ack'))
        from worker.native_cpu_supervisor import CPUJournal
        self.s.journal.lock.close()
        reopened=CPUJournal(self.f.root,'a'*32,'b'*64)
        try:
            record=reopened.value['records'][0]
            self.assertEqual(record['receipt'],collected['receipt']);self.assertTrue(record['acked'])
            self.assertEqual(record['proposal'],value)
        finally:reopened.lock.close()


if __name__=='__main__':unittest.main()
