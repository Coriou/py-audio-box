"""Inert native state machine: fake process handles/channels only."""
import base64
import copy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

from worker import native_protocol as p
from worker.native_supervisor import NativeSupervisor
from worker.transport import Refused,canonical
from worker.hardware import HardwareInventory,Host,Device,DeviceFamily
from tests.worker import test_engine_qwen as fixtures


class Channel:
    def __init__(self): self.fd=42
    def fileno(self): return self.fd
    def close(self): self.fd=-1


class Handle:
    pid=123
    def __init__(self): self.code=None; self.waited=0
    def poll(self): return self.code
    def wait(self,timeout): self.waited+=1; return self.code


class NativeSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.QwenTests('test_cpu_exact_forwarding_and_pcm'); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root/'native'; self.root.mkdir(mode=0o700)
        self.now=datetime.now(timezone.utc); self.t=100.
        self.e=replace(self.fixture.e,device='mps',timeout_s=25)
        for path in self.fixture.bundle.iterdir(): path.chmod(0o400)
        self.fixture.bundle.chmod(0o500)
        self.inventory=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.MPS,0,True,True,self.now.isoformat()),))
        self.e=replace(self.e,voice_id='voice.0',voice_revision='vv0',memory_bytes=192*1024**2)
        self.snapshot=self.snapshot_for(self.e)
        self.binding=hashlib.sha256(canonical(self.snapshot)).hexdigest()
        self.reg={self.binding:dict(snapshot=self.snapshot,execution=self.e,bundles={'bundle':self.fixture.bundle},inventory=self.inventory)}
        self.calls=[]; self.h=Handle(); self.signals=[]
        self.make()

    def snapshot_for(self,e):
        from contracts import validate as oracle
        from worker.engines.qwen_execution import inventory_wire
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
        context=fixture['context']; plan=context['plan']; resolved=plan['resolved']
        identity=dict(engine='qwen',model=e.checkpoint,bundle_sha256=e.manifest_sha256,device='metal',precision='float32')
        settings=dict(kind='qwen_clone' if e.task=='clone' else 'qwen_preset',language=e.language,seed=None,instruction=e.instruct,**{k:getattr(e,k) for k in ('temperature','top_p','top_k','repetition_penalty','max_new_tokens')})
        v=plan['voice_versions'][0]; v.update(identity=identity,settings=settings,settings_sha256=oracle.canonical_hash(settings),configuration=dict(mode='qwen_preset',speaker=e.preset))
        plan['request'].update(identity=identity,text='hello')
        profile=fixture['profiles'][0]; profile.update(id='qwen-mps',device='metal',topology='packaged_native',support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INERT-SOURCE-FIXTURE-NOT-HARDWARE')
        profile['compatibility']=[dict(operation='speech_generate',mode='qwen_preset',identity=identity)]
        profile['limits'].update(ram_bytes=e.memory_bytes,threads=e.threads)
        resolved.update(identity=dict(requested=identity,effective=identity,fallback='forbid',reason='',profile_id='qwen-mps'),profile=oracle.profile_snapshot(profile),settings=settings,settings_sha256=oracle.canonical_hash(settings))
        context['bundles']=[dict(instructions=True,engine='qwen',model=e.checkpoint,mode='qwen_preset',bundle_sha256=e.manifest_sha256,languages=[e.language],selectors=[dict(value=e.preset,languages=[e.language])])]
        context['execution']=dict(descriptor=e.identity(),bundles={'bundle':'/models/bundles/bundle'},references={},hardware=inventory_wire(self.inventory))
        job=fixture['job']; job.update(input_sha256=oracle.canonical_hash(plan),deadline=(self.now+timedelta(seconds=25)).isoformat())
        return [job,context,[],[profile]]

    def make(self):
        def popen(*args,**kwargs): self.calls.append((args,kwargs)); return self.h
        self.s=NativeSupervisor(self.root,deployment_id='a'*32,configuration_sha256='b'*64,peer_pin='c'*64,runtime_manifest_sha256='d'*64,registrations=self.reg,python='/managed/python',child_entry=str(Path(__file__).resolve().parents[2]/'worker/native_child.py'),popen=popen,pair=lambda:(Channel(),Channel()),signal_group=lambda *v:self.signals.append(v),clock=lambda:self.t,utc=lambda:self.now,free_bytes=lambda:100*1024**3,free_ram=lambda:16*1024**3)
        self.addCleanup(self.s.journal.lock.close)

    def q(self,op,body=None,**changes):
        q=dict(version=p.VERSION,deployment_id='a'*32,configuration_sha256='b'*64,lane=self.s.journal.value['lane'],boot=self.s.boot,request_id='e'*32,runtime_id=getattr(self,'rid',None),binding=self.binding,op=op,body=body or {})
        if op=='prepare': q['runtime_id']=None
        q.update(changes); return q
    def prepare(self):
        self.rid=self.s.dispatch(self.q('prepare',{'snapshot':self.snapshot}))['runtime_id']
        return self.s.journal.value['records'][0]

    def test_prepare_replay_tuple_session_and_stop_ack_restart(self):
        r=self.prepare(); self.assertEqual(self.prepare()['id'],self.rid)
        for changes in ({'boot':'f'*32},{'binding':'f'*64},{'configuration_sha256':'f'*64},{'lane':'f'*32}):
            with self.assertRaises(Refused): self.s.dispatch(self.q('start',**changes))
        self.s.dispatch(self.q('stop')); receipt=self.s.dispatch(self.q('collect'))['receipt']
        self.assertEqual(receipt['processes'],[])
        old=self.q('ack'); self.s.journal.lock.close(); self.make()
        self.s.dispatch(old); self.s.dispatch(old)
        self.assertTrue(self.s.journal.value['records'][0]['acked'])
        self.assertFalse(self.s.quarantined)

    def test_restart_unknown_never_relaunch(self):
        self.prepare(); old=self.q('start'); self.s.journal.lock.close(); self.make()
        self.assertTrue(self.s.quarantined)
        self.assertEqual(self.s.dispatch(dict(old,op='observe'))['state'],'unknown')
        with self.assertRaises(Refused): self.s.dispatch(old)
        self.assertEqual(self.calls,[])

    def test_launch_intent_before_factory_replay_wait_and_receiver(self):
        self.prepare()
        def exchange(*args):
            self.assertEqual(self.s.journal.value['records'][0]['phase'],'launch-intent')
            return None
        with patch('worker.native_child.parent_exchange',exchange):
            self.s.dispatch(self.q('start')); self.s.active['receiver'].join(1)
        self.s.dispatch(self.q('start')); self.assertEqual(len(self.calls),1)
        args,kw=self.calls[0]
        self.assertEqual(kw['pass_fds'],(42,)); self.assertTrue(kw['start_new_session']); self.assertTrue(kw['close_fds'])
        self.assertNotIn('shell',kw); self.assertNotIn('preexec_fn',kw)
        self.s.tick(); self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])
        self.h.code=0; self.s.tick()
        receipt=self.s.dispatch(self.q('collect'))['receipt']
        self.assertFalse(receipt['timely']); self.assertTrue(receipt['receiver_closed']); self.assertEqual(self.h.waited,1)

    def test_watchdog_term_kill_and_identity_refusal(self):
        self.prepare()
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.s.dispatch(self.q('start')); self.s.active['receiver'].join(1)
        self.s.dispatch(self.q('stop')); self.s.tick(); self.t+=.21; self.s.tick()
        self.assertEqual(self.signals,[(123,signal.SIGTERM),(123,signal.SIGKILL)])
        self.s.identity=lambda handle:'reused'; self.s.tick(); self.assertTrue(self.s.failed)
        self.assertEqual(len(self.signals),2)

    def test_popen_and_journal_uncertainty_retains(self):
        self.prepare()
        self.s.popen=lambda *a,**k: (_ for _ in ()).throw(OSError('unknown'))
        with self.assertRaises(Refused): self.s.dispatch(self.q('start'))
        self.assertTrue(self.s.quarantined); self.assertEqual(self.s.journal.value['records'][0]['phase'],'launch-intent')
        with self.assertRaises(Refused): self.s.dispatch(self.q('start'))

    def test_transfer_does_not_block_stop_and_body_refusals(self):
        self.prepare(); self.s.transfer.acquire()
        try: self.s.dispatch(self.q('stop'))
        finally: self.s.transfer.release()
        self.s.tick()
        self.assertEqual(self.s.dispatch(self.q('observe'))['state'],'stopped-and-reaped')
        with self.assertRaises(Refused): self.s.dispatch(self.q('put_input',dict(sha256='f'*64,bytes=4,offset=0,data='MTIzNA==')))

    def test_journal_failure_never_terminal(self):
        self.prepare()
        with patch.object(self.s.journal,'save',side_effect=OSError('fsync')):
            with self.assertRaises(Refused): self.s.dispatch(self.q('stop'))
        self.assertTrue(self.s.quarantined); self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])

    def test_response_exact_nonce_and_closed_variants(self):
        self.prepare(); q=self.q('observe'); envelope={k:v for k,v in q.items() if k!='body'}
        reply=dict(envelope,ok=True,value=self.s.dispatch(q)); p.response(reply,request=q)
        for changes in ({'request_id':'f'*32},{'extra':1},{'error':'native_refused'}):
            with self.assertRaises(Refused): p.response(dict(reply,**changes),request=q)

    def test_real_child_entry_infer_encoder_and_parent_receiver_fake_channel(self):
        import struct
        import time
        from worker import native_child
        from worker.engines.qwen_execution import inventory_wire
        from tests.worker.test_native_qwen import inert_guard
        class BytesChannel:
            def __init__(self,raw=b''): self.raw=bytearray(raw); self.sent=b''
            def recv(self,size):
                data=bytes(self.raw[:min(size,7)]); del self.raw[:len(data)]; return data
            def sendall(self,data): self.sent+=data
            def settimeout(self,timeout): pass
        cutoff=time.monotonic()+20
        config=dict(token='f'*32,binding=self.binding,snapshot=self.snapshot,execution=self.e.identity(),bundles={'bundle':str(self.fixture.bundle)},references={},inventory=inventory_wire(self.inventory),conditioning=None,cutoff=cutoff)
        raw=canonical(config); child=BytesChannel(struct.pack('!I',len(raw))+raw)
        self.fixture.backend.synthesise_preset.return_value.metadata=dict(checkpoint=str(self.fixture.bundle),revision=self.e.revision)
        with inert_guard(), patch('lib.tts_backends.qwen_native.NativeQwenBackend',return_value=self.fixture.backend):
            native_child.run_socket(child)
        parent=BytesChannel(child.sent)
        proposal,data=native_child.parent_exchange(parent,config,cutoff)
        native_child.validate_proposal(self.snapshot,proposal,data,self.e)
        self.assertEqual(len(data),48)
        # Result token, trailing bytes and truncated output cannot authorize receipt.
        for raw in (child.sent+b'x',child.sent[:-3]):
            with self.assertRaises(Exception): native_child.parent_exchange(BytesChannel(raw),config,cutoff)
        changed=dict(config,token='0'*32)
        with self.assertRaises(Refused): native_child.parent_exchange(BytesChannel(child.sent),changed,cutoff)

    def test_stale_inventory_and_descriptor_relation_refuse_before_spawn(self):
        for mutation in ('future','stale','descriptor','topology'):
            reg=copy.deepcopy(self.reg[self.binding])
            if mutation in ('future','stale'):
                reg['inventory']=replace(self.inventory,devices=(replace(self.inventory.devices[0],probed_at=(self.now+timedelta(seconds=10) if mutation=='future' else self.now-timedelta(seconds=301)).isoformat()),))
            elif mutation=='descriptor': reg['execution']=replace(self.e,preset='Other')
            else: reg['snapshot'][1]['plan']['resolved']['profile']['topology']='networkless_container'
            self.s.registrations[self.binding]=reg
            with self.assertRaises(Exception): self.prepare()
        self.assertEqual(self.calls,[])

    def test_receiver_pending_close_wait_failure_and_journal_fence(self):
        self.prepare()
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.s.dispatch(self.q('start')); self.s.active['receiver'].join(1)
        self.h.code=0
        self.s.active['closed']=False; self.s.tick()
        self.assertTrue(self.s.quarantined); self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])
        self.assertIsNotNone(self.s.active)

    def test_full_records_and_expired_prepare_refuse(self):
        self.now+=timedelta(seconds=26)
        with self.assertRaises(Refused): self.prepare()
        self.assertEqual(self.calls,[])

    def test_actual_child_result_terminal_output_and_oldboot_replay(self):
        import struct
        import time
        from worker import native_child
        from tests.worker.test_native_qwen import inert_guard
        class BytesChannel:
            def __init__(self,raw=b''): self.raw=bytearray(raw); self.sent=b''
            def recv(self,size):
                data=bytes(self.raw[:size]); del self.raw[:len(data)]; return data
            def sendall(self,data): self.sent+=data
            def settimeout(self,timeout): pass
        self.t=time.monotonic(); self.s.clock=lambda:time.monotonic()
        original=native_child.parent_exchange
        def exchange(channel,config,cutoff):
            raw=canonical(config); child=BytesChannel(struct.pack('!I',len(raw))+raw)
            self.fixture.backend.synthesise_preset.return_value.metadata=dict(checkpoint=str(self.fixture.bundle),revision=self.e.revision)
            with inert_guard(), patch('lib.tts_backends.qwen_native.NativeQwenBackend',return_value=self.fixture.backend): native_child.run_socket(child)
            return original(BytesChannel(child.sent),config,cutoff)
        self.prepare()
        with patch('worker.native_child.parent_exchange',exchange):
            self.s.dispatch(self.q('start')); self.s.active['receiver'].join(1)
        self.s.tick(); self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])
        self.h.code=0; self.s.tick()
        collected=self.s.dispatch(self.q('collect'))
        self.assertTrue(collected['receipt']['timely']); self.assertIsNotNone(collected['proposal'])
        output=collected['output']
        data=self.s.dispatch(self.q('get_output',dict(sha256=output['sha256'],offset=0,bytes=output['bytes'])))
        self.assertEqual(hashlib.sha256(base64.b64decode(data['data'])).hexdigest(),output['sha256'])
        old=self.q('collect'); ack=self.q('ack')
        self.s.journal.lock.close(); self.make()
        self.assertEqual(self.s.dispatch(old),collected)
        self.s.dispatch(ack); self.assertFalse(self.s.quarantined)

    def test_output_bad_hash_pcm_and_cap(self):
        from worker.native_child import proposal,validate_proposal
        result=self.fixture.run_engine(); value=proposal(self.snapshot,result,self.e)
        for data in (b'',result['data'][:-1],b'x'*245):
            with self.assertRaises(Refused): validate_proposal(self.snapshot,value,data,self.e)
        value['manifest']['outputs'][0]['provenance']['source_id']='wrong'
        with self.assertRaises(Refused): validate_proposal(self.snapshot,value,result['data'],self.e)

    def test_wait_exception_and_launch_intent_restart(self):
        self.prepare()
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.s.dispatch(self.q('start')); self.s.active['receiver'].join(1)
        self.h.code=0
        with patch.object(self.h,'wait',side_effect=OSError('wait unknown')): self.s.tick()
        self.assertTrue(self.s.quarantined); self.assertIsNotNone(self.s.active)
        self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])
        self.s.journal.lock.close(); self.make()
        self.assertTrue(self.s.quarantined); self.assertEqual(len(self.calls),1)

    def prepare_clone(self):
        import io
        import wave
        from contracts import validate as oracle
        from worker.engines.qwen_execution import inventory_wire
        raw=io.BytesIO()
        with wave.open(raw,'wb') as out: out.setparams((1,2,24000,0,'NONE','none')); out.writeframes(b'\0\0'*4)
        raw=raw.getvalue(); digest=hashlib.sha256(raw).hexdigest()
        self.fixture.bundle.chmod(0o700)
        for path in self.fixture.bundle.iterdir(): path.chmod(0o600)
        self.fixture.make_bundle(mode='base')
        for path in self.fixture.bundle.iterdir(): path.chmod(0o400)
        self.fixture.bundle.chmod(0o500)
        e=replace(self.e,manifest_sha256=self.fixture.e.manifest_sha256,task='clone',preset=None,reference_id='reference',reference_sha256=digest,reference_bytes=len(raw),reference_frames=4,reference_rate=24000,reference_text='hello')
        snapshot=self.snapshot_for(e); plan=snapshot[1]['plan']; v=plan['voice_versions'][0]
        asset=dict(id='reference',sha256=digest,frames=4,sample_rate=24000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='fixture',missing_fields=[]))
        v.update(origin='clone',reference_asset_ids=['reference'],reference_asset_sha256={'reference':digest},reference_text='hello',configuration=dict(mode='qwen_clone',reference_asset_id='reference',reference_text='hello',conditioning='icl'))
        plan['assets']=[asset]
        plan['resolved']['profile']['compatibility'][0]['mode']='qwen_clone'; snapshot[3][0]['compatibility'][0]['mode']='qwen_clone'
        snapshot[1]['bundles'][0].update(mode='qwen_clone',selectors=[])
        snapshot[1]['execution']['references']={'reference':dict(path='/references/assets/reference',voice_id=e.voice_id,voice_revision=e.voice_revision,checkpoint=e.checkpoint,revision=e.revision,text='hello')}
        snapshot[0]['input_sha256']=oracle.plan_hash(plan)
        snapshot[2]=['reference']
        self.snapshot=snapshot; self.binding=hashlib.sha256(canonical(snapshot)).hexdigest()
        self.s.registrations={self.binding:dict(snapshot=snapshot,execution=e,bundles={'bundle':self.fixture.bundle},inventory=self.inventory)}
        self.prepare()
        return raw,digest,e

    def test_input_complete_replay_conflict_gap_and_hash(self):
        raw,digest,e=self.prepare_clone()
        def put(offset,data,**extra):
            body=dict(sha256=digest,bytes=len(raw),offset=offset,data=base64.b64encode(data).decode());body.update(extra)
            return self.s.dispatch(self.q('put_input',body))
        with self.assertRaises(Refused): put(1,raw[:10])
        put(0,raw[:10]); put(0,raw[:10])
        with self.assertRaises(Refused): put(0,b'x'*10)
        with self.assertRaises(Refused): put(10,raw[10:],sha256='0'*64)
        put(10,raw[10:]); put(10,raw[10:])
        self.assertEqual(self.s.journal.value['records'][0]['input_size'],len(raw))
        # Exact replay is allowed; no extension of a completed blob.
        with self.assertRaises(Refused): put(len(raw)-1,b'xx')

    def test_storage_gate_and_quota_refuse_before_reservation(self):
        self.s.free_bytes=lambda:6*1024**3-1
        with self.assertRaises(Refused): self.prepare()
        self.assertEqual(self.s.journal.value['records'],[])
        self.s.free_bytes=lambda:100*1024**3
        with patch('worker.native_supervisor.QUOTA',1):
            with self.assertRaises(Refused): self.prepare()
        self.assertEqual(self.s.journal.value['records'],[])
        self.s.free_ram=lambda:0
        with self.assertRaises(Refused): self.prepare()

    def test_fake_tls_routes_real_handler_and_rejects_pin(self):
        import struct
        from tests.worker.test_native_protocol import TLS
        class Connection(TLS):
            def close(self): self.closed=True
        import time
        self.s.clock=time.monotonic
        self.s.pin=hashlib.sha256(b'fake enrolled cert').hexdigest()
        q=self.q('hello',lane=None,boot=None,runtime_id=None,binding=None)
        raw=canonical(q); conn=Connection(struct.pack('!I',len(raw))+raw)
        self.s.serve_connection(conn)
        import time
        response=p.receive(TLS(conn.sent),time.monotonic()+2,peer_sha256=self.s.pin)
        p.response(response,request=q); self.assertTrue(conn.closed)
        conn=Connection(struct.pack('!I',len(raw))+raw);conn.cert=b'wrong'
        with self.assertRaises(Refused): self.s.serve_connection(conn)
        self.assertTrue(conn.closed);self.assertEqual(self.calls,[])

    def test_escalation_bounded_unknown_no_stale_group_signal(self):
        self.prepare()
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.s.dispatch(self.q('start'));self.s.active['receiver'].join(1)
        self.s.dispatch(self.q('stop'));self.s.tick();self.t+=.21;self.s.tick();self.t+=1.01;self.s.tick()
        self.assertTrue(self.s.quarantined);self.s.tick()
        self.assertEqual(len(self.signals),2);self.assertIsNotNone(self.s.active)
        self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])

    def test_launch_intent_journal_failure_prevents_popen(self):
        self.prepare()
        with patch.object(self.s.journal,'save',side_effect=OSError('intent fsync')):
            with self.assertRaises(Refused): self.s.dispatch(self.q('start'))
        self.assertEqual(self.calls,[]);self.assertTrue(self.s.quarantined)

    def test_clone_conditioning_nonempty_blocks_terminal(self):
        raw,digest,e=self.prepare_clone()
        self.s.dispatch(self.q('put_input',dict(sha256=digest,bytes=len(raw),offset=0,data=base64.b64encode(raw).decode())))
        with patch('worker.native_child.parent_exchange',return_value=None):
            self.s.dispatch(self.q('start'));self.s.active['receiver'].join(1)
        row=self.s.journal.value['records'][0]
        (self.s.directory(row)/'conditioning'/'retained').write_bytes(b'private snapshot')
        self.h.code=-15; self.s.tick()
        self.assertTrue(self.s.quarantined);self.assertIsNone(self.s.dispatch(self.q('collect'))['receipt'])
        self.assertTrue((self.s.directory(row)/'conditioning'/'retained').exists())

    def test_input_wrong_complete_hash_quarantines(self):
        raw,digest,e=self.prepare_clone()
        with self.assertRaises(Refused): self.s.dispatch(self.q('put_input',dict(sha256=digest,bytes=len(raw),offset=0,data=base64.b64encode(b'x'*len(raw)).decode())))
        self.assertTrue(self.s.quarantined)
        with self.assertRaises(Refused): self.s.dispatch(self.q('start'))
        self.assertEqual(self.calls,[])

    def test_spool_symlink_and_hardlink_refuse(self):
        import os
        from worker.native_supervisor import read_private
        target=self.root/'tiny';target.write_bytes(b'x');target.chmod(0o600)
        link=self.root/'link';link.symlink_to(target)
        with self.assertRaises(Refused): read_private(link,10)
        hard=self.root/'hard';os.link(target,hard)
        with self.assertRaises(Refused): read_private(target,10)

    def test_completed_oldboot_then_new_registered_work(self):
        self.prepare();self.s.dispatch(self.q('stop'));ack=self.q('ack')
        self.s.journal.lock.close();self.make();self.s.dispatch(ack)
        snapshot=copy.deepcopy(self.snapshot);snapshot[0]['id']='new-job'
        binding=hashlib.sha256(canonical(snapshot)).hexdigest()
        reg=copy.deepcopy(self.reg[self.binding]);reg['snapshot']=snapshot
        self.s.registrations[binding]=reg
        self.snapshot=snapshot;self.binding=binding
        result=self.s.dispatch(self.q('prepare',{'snapshot':snapshot}))
        self.assertEqual(result['state'],'alive');self.assertIn(self.s.boot,result['runtime_id'])
        self.assertEqual(len(self.s.journal.value['records']),2)
