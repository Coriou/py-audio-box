"""Tiny SOURCE fixtures: actual codecs/handlers, fake TLS and owned handles."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import unittest
from unittest.mock import patch

from service.native_runtime import NativeRuntime
from service.worker_transport import Authority
from worker import native_protocol as p
from worker.transport import canonical, Refused
from tests.worker import test_native_supervisor as supervisor_tests


class BytesChannel:
    def __init__(self,raw=b''): self.raw=bytearray(raw);self.sent=b''
    def recv(self,size):
        result=bytes(self.raw[:size]);del self.raw[:len(result)];return result
    def sendall(self,data): self.sent+=data
    def settimeout(self,value): pass


class NativeFixture:
    def __init__(self,case):
        from worker.native_child import parent_exchange
        self.parent_exchange=parent_exchange
        self.case=case
        self.f=supervisor_tests.NativeSupervisorTests('test_prepare_replay_tuple_session_and_stop_ack_restart')
        self.f.setUp();case.addCleanup(self.f.doCleanups)
        self.s=self.f.s;self.f.t=time.monotonic();self.s.clock=time.monotonic
        self.native=dict(deployment_id='a'*32,configuration_sha256='b'*64,
            lane=self.s.journal.value['lane'],boot=self.s.boot,runtime_manifest_sha256='d'*64)
        self.snapshot=copy.deepcopy(self.f.snapshot)
        self.snapshot[1]['execution']['native']=self.native.copy()
        self.a=Authority(*self.snapshot)
        self.s.registrations={self.a.binding:dict(snapshot=self.snapshot,execution=self.f.e,
            bundles={'bundle':self.f.fixture.bundle},inventory=self.f.inventory)}
        self.spool=self.f.fixture.root/'control-spool';self.spool.mkdir(mode=0o700)
        self.stage=self.f.fixture.root/'control-stage';self.stage.mkdir(mode=0o770);self.stage.chmod(0o770)
        self.cert=b'fixture-server-certificate';self.pin=hashlib.sha256(self.cert).hexdigest()
        self.requests=[];self.mutate=lambda q,v:v;self.before=lambda q:None
        self.runtime=self.make_runtime()

    def make_runtime(self):
        return NativeRuntime(host='fixture.invalid',port=4443,tls=None,peer_sha256=self.pin,
            native=self.native,spool=self.spool,staging=self.stage,local_import_uid=os.getuid(),connector=self.connect)

    @contextmanager
    def connect(self,timeout):
        fixture=self
        class TLS(BytesChannel):
            def version(self): return 'TLSv1.3'
            def getpeercert(self,binary_form=False): return fixture.cert
            def sendall(self,raw):
                n,=struct.unpack('!I',raw[:4]);q=json.loads(raw[4:]);fixture.case.assertEqual(len(raw)-4,n)
                fixture.requests.append(q);fixture.before(q)
                try:
                    result=fixture.s.dispatch(q)
                    response={k:v for k,v in q.items() if k!='body'}
                    response.update(ok=True,value=result)
                except Refused:
                    response={k:v for k,v in q.items() if k!='body'}
                    response.update(ok=False,error='native_refused')
                response=fixture.mutate(q,copy.deepcopy(response))
                data=canonical(response);self.raw=bytearray(struct.pack('!I',len(data))+data)
        yield TLS()

    def exchange(self,channel,config,cutoff):
        from worker import native_child
        from tests.worker.test_native_qwen import inert_guard
        raw=canonical(config);child=BytesChannel(struct.pack('!I',len(raw))+raw)
        backend=self.f.fixture.backend
        backend.synthesise_preset.return_value.metadata=dict(checkpoint=str(self.f.fixture.bundle),revision=self.f.e.revision)
        with inert_guard(),patch('lib.tts_backends.qwen_native.NativeQwenBackend',return_value=backend):
            native_child.run_socket(child)
        return self.parent_exchange(BytesChannel(child.sent),config,cutoff)

    def finish(self):
        self.s.active['receiver'].join(2)
        self.case.assertFalse(self.s.active['receiver'].is_alive())
        self.f.h.code=0;self.s.tick()

    def launch(self):
        rid=self.runtime.prepare(self.a)
        with patch('worker.native_child.parent_exchange',side_effect=self.exchange):
            self.runtime.start(rid);self.finish()
        return rid


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self): self.f=NativeFixture(self)

    def test_actual_tls_codecs_handlers_child_transfer_receipt_and_exact_replay(self):
        f=self.f;rid=f.launch()
        receipt,observation=f.runtime.collect(f.a,rid)
        self.assertTrue(receipt['receiver_closed']);self.assertNotIn('engine_uid',receipt)
        self.assertEqual(f.f.h.waited,1)
        raw=(f.stage/f.a.job['attempts'][-1]['staging_id']/'output.wav').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),observation.manifest['outputs'][0]['sha256'])
        self.assertEqual(f.runtime.collect(f.a,rid),(receipt,unittest.mock.ANY))
        self.assertEqual(f.runtime.terminate(rid),dict(state='terminal',timely=True))
        self.assertEqual(f.runtime.call('ack',dict(runtime_id=rid)),dict(acked=True))
        self.assertEqual(len(f.f.calls),1)
        self.assertEqual((f.stage/f.a.job['attempts'][-1]['staging_id']/'output.wav').stat().st_nlink,1)

    def test_wrong_pin_and_every_response_tuple_field_refuse(self):
        f=self.f
        f.runtime.pin='0'*64
        with self.assertRaisesRegex(Refused,'pin'):f.runtime.call('hello',{})
        self.assertEqual(f.requests,[]);f.runtime.pin=f.pin
        for key,value in [('request_id','f'*32),('boot','f'*32),('lane','f'*32),('configuration_sha256','f'*64),
                          ('deployment_id','f'*32),('binding','f'*64),('runtime_id','foreign'),('op','stop')]:
            with self.subTest(key=key):
                f.mutate=lambda q,v:dict(v,**{key:value})
                with self.assertRaises(Refused):f.runtime.call('hello',{})
        self.assertEqual(f.f.calls,[])

    def test_receipt_linux_substitution_and_bad_output_never_import(self):
        f=self.f;rid=f.launch()
        original=f.mutate
        def linux(q,v):
            if q['op']=='collect':v['value']['receipt']['reap_method']='pid1-waitpid-echild'
            return v
        f.mutate=linux
        with self.assertRaises(Refused):f.runtime.collect(f.a,rid)
        self.assertEqual(list(f.stage.iterdir()),[])
        f.mutate=lambda q,v:(dict(v,value=dict(v['value'],data='AAAA')) if q['op']=='get_output' else v)
        with self.assertRaises(Refused):f.runtime.collect(f.a,rid)
        self.assertEqual(list(f.stage.iterdir()),[])
        self.assertTrue(f.runtime.failed)
        with self.assertRaisesRegex(Refused,'uncertain'):f.make_runtime().prepare(f.a)

    def test_stop_and_cancel_during_output_import_are_not_death_proof(self):
        f=self.f;rid=f.launch()
        f.before=lambda q:f.runtime.interrupt(rid) if q['op']=='get_output' else None
        receipt,observation=f.runtime.collect(f.a,rid)
        self.assertIsNone(observation);self.assertTrue(receipt['reaped'])
        self.assertEqual(list(f.stage.iterdir()),[])

    def test_local_import_uid_stage_symlink_and_conflict_refuse(self):
        f=self.f
        with self.assertRaisesRegex(Refused,'uid'):
            NativeRuntime(host='x',port=1,tls=None,peer_sha256=f.pin,native=f.native,
                spool=f.spool,staging=f.stage,local_import_uid=os.getuid()+1,connector=f.connect)
        rid=f.launch();stage=f.stage/f.a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        target=stage/'output.wav';target.symlink_to(f.spool/'foreign')
        with self.assertRaises(OSError):f.runtime.collect(f.a,rid)
        self.assertTrue(target.is_symlink())

    def test_detached_preset_and_configuration_and_quota_fences(self):
        f=self.f
        with self.assertRaises(Refused):f.runtime.prepare(f.a,detached=b'not-authorized')
        changed=copy.deepcopy(f.snapshot);changed[1]['execution']['native']['configuration_sha256']='f'*64
        with self.assertRaises(Refused):f.runtime.prepare(Authority(*changed))
        with patch('service.native_runtime.QUOTA',1):
            with self.assertRaisesRegex(Refused,'quota'):f.runtime.prepare(f.a)
        self.assertEqual(f.f.calls,[])

    def clone(self):
        f=self.f
        with patch.object(f.f,'prepare',return_value={}):raw,digest,spec=f.f.prepare_clone()
        f.snapshot=copy.deepcopy(f.f.snapshot);f.snapshot[1]['execution']['native']=f.native.copy()
        f.a=Authority(*f.snapshot);f.f.e=spec
        f.s.registrations={f.a.binding:dict(snapshot=f.snapshot,execution=spec,bundles={'bundle':f.f.fixture.bundle},inventory=f.f.inventory)}
        return raw

    def test_clone_detached_transfer_actual_child_and_repeated_start(self):
        f=self.f;raw=self.clone()
        f.f.fixture.backend.synthesise_clone.return_value.metadata=dict(checkpoint=str(f.f.fixture.bundle),revision=f.f.e.revision)
        rid=f.runtime.prepare(f.a,detached=raw)
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            f.runtime.start(rid)
            f.runtime.start(rid)
            f.finish()
        receipt,observation=f.runtime.collect(f.a,rid)
        self.assertTrue(receipt['timely']);self.assertIsNotNone(observation)
        self.assertEqual(len(f.f.calls),1)
        self.assertEqual([q['op'] for q in f.requests].count('put_input'),1)
        self.assertEqual(f.f.fixture.backend.synthesise_clone.call_args.args[2],'hello')

    def test_cancel_upload_and_unknown_restart_never_launch_or_release(self):
        f=self.f;raw=self.clone();rid=f.runtime.prepare(f.a,detached=raw)
        f.before=lambda q:f.runtime.interrupt(rid) if q['op']=='put_input' else None
        with self.assertRaisesRegex(Refused,'cancelled'):f.runtime.start(rid)
        self.assertEqual(f.f.calls,[])
        receipt,observation=f.runtime.collect(f.a,rid)
        self.assertFalse(receipt['timely']);self.assertIsNone(observation)
        # A different fresh fixture is reopened with a nonterminal allocation.
        other=NativeFixture(self);rid=other.runtime.prepare(other.a)
        other.s.journal.lock.close();other.f.make();other.s=other.f.s
        replay=other.make_runtime();replay.bind(other.a,rid)
        self.assertEqual(replay.observe(rid).state.value,'unknown')
        with self.assertRaises(Refused):replay.start(rid)
        self.assertIsNone(replay.collect(other.a,rid))
        self.assertEqual(other.f.calls,[])

    def test_first_late_import_fails_but_completed_local_replay_does_not_expire(self):
        f=self.f;rid=f.launch()
        f.runtime.records[rid]['cutoff']=time.monotonic()-1
        receipt,observation=f.runtime.collect(f.a,rid)
        self.assertTrue(receipt['reaped']);self.assertIsNone(observation)
        self.assertEqual(list(f.stage.iterdir()),[])
        other=NativeFixture(self);rid=other.launch()
        first=other.runtime.collect(other.a,rid)
        other.runtime.records[rid]['cutoff']=time.monotonic()-1
        self.assertEqual(other.runtime.collect(other.a,rid)[0],first[0])

    def test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt(self):
        from service.supervisor_runtime import SupervisorRuntime
        runtime=SupervisorRuntime('/inert')
        f=self.f
        with self.assertRaisesRegex(ValueError,'native_transport'):
            with patch.object(runtime,'call',return_value=dict(receipt={'runtime_id':'x'},proposal=None)):
                runtime.collect(f.a,'x')
        # The existing Linux relation decoder remains Linux-only for this profile.
        from tests.service import test_qwen_lifecycle as linux_tests
        fixture=linux_tests.QwenLifecycleTests('test_refusals_before_allocation');fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.submit();a=fixture.store.allocate(fixture.profiles,'qwen',8)['authority']
        receipt=dict(runtime_id='linux',binding=a.binding,reaped=True,timely=False,original_deadline=a.job['deadline'],
            engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=1)
        with patch.object(runtime,'call',return_value=dict(receipt=receipt,proposal=None)):
            self.assertEqual(runtime.collect(a,'linux'),(receipt,None))
        for mutation in ({'extra':True},{'reaped_children':True},{'engine_uid':True},{'reap_method':'darwin-owned-exec-v1'}):
            with patch.object(runtime,'call',return_value=dict(receipt=dict(receipt,**mutation),proposal=None)):
                with self.assertRaises(Refused):runtime.collect(a,'linux')
