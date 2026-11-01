"""No-weight actual private socket/handshake/proposal and failure cause."""
import hashlib
import io
import socket
import threading
import time
import unittest
import wave
from worker import native_cpu_child as child
from worker import native_cpu_protocol as p
from worker.transport import Refused,canonical
from tests.worker.native_cpu_fixtures import Fixture


class CPUChildTests(unittest.TestCase):
    def setUp(self):self.f=Fixture(self)

    def config(self):
        return dict(token='a'*32,binding=hashlib.sha256(canonical(self.f.snapshot)).hexdigest(),
            snapshot=self.f.snapshot,execution=self.f.spec.identity(),bundles={self.f.spec.bundle_id:str(self.f.bundle)},
            inventory=p.inventory_wire(self.f.inventory),cutoff=time.monotonic()+60,references={},conditioning=None)

    def exchange(self,inference):
        parent,owned=socket.socketpair();self.addCleanup(parent.close);self.addCleanup(owned.close)
        failures=[]
        def run():
            try:child.run_socket(owned,inference=inference,guard=False)
            except BaseException as exc:failures.append(exc)
            finally:owned.close()
        thread=threading.Thread(target=run);thread.start()
        try:return child.parent_exchange(parent,self.config(),time.monotonic()+30,on_loaded=lambda v:None)
        finally:
            parent.close();thread.join(2)
            self.assertFalse(thread.is_alive());self.failures=failures

    def test_actual_preset_pcm_proposal_and_original_deadline(self):
        def infer(context,bundles,*,on_loaded):
            on_loaded()
            out=io.BytesIO()
            with wave.open(out,'wb') as wav:
                wav.setparams((1,2,24000,0,'NONE','none'));wav.writeframes(b'\x01\0'*10)
            raw=out.getvalue()
            return dict(data=raw,frames=10,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),identity={})
        proposal,raw=self.exchange(infer)
        self.assertEqual(raw[:4],b'RIFF');self.assertEqual(proposal['manifest']['operation'],'speech_generate')
        self.assertFalse(self.failures)

    def test_load_error_preserves_stage_and_original_cause(self):
        def infer(*args,**kwargs):
            try:raise OSError('dependency detail')
            except OSError as exc:raise ValueError('friendly load failure') from exc
        with self.assertRaises(child.ChildFailure) as caught:self.exchange(infer)
        self.assertEqual(caught.exception.error['code'],'native_cpu_startup_failed')
        self.assertEqual(caught.exception.error['cause'],'dependency detail')
        self.assertIsInstance(self.failures[0],ValueError)

    def test_no_model_import_without_guard_or_reference_admission(self):
        with self.assertRaisesRegex(Refused,'native_spawn_guard_required'):child.infer(self.f.context,{},on_loaded=lambda:None)
        for extra in ({'references':{'ref':'untrusted'}},{'conditioning':'clone'}):
            config=self.config();config.update(extra)
            with self.assertRaisesRegex(Refused,'native_cpu_preset_only'):
                child.parent_exchange(None,config,time.monotonic()+30,on_loaded=lambda v:None)
        for candidate in (float('inf'),time.monotonic()-1,time.monotonic()+121,True):
            with self.assertRaises(Refused):child.validate_cutoff(candidate,time.monotonic(),60)


if __name__=='__main__':unittest.main()
