"""Tiny SOURCE fixtures: actual codecs/handlers, fake TLS and owned handles."""
from contextlib import contextmanager
import copy
import contextlib
import hashlib
import io
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
    def __init__(self,case, *, clone=False):
        from worker.native_child import parent_exchange
        self.parent_exchange=parent_exchange
        self.case=case
        self.f=supervisor_tests.NativeSupervisorTests('test_prepare_replay_tuple_session_and_stop_ack_restart')
        self.f.setUp();case.addCleanup(self.f.doCleanups)
        if clone: self.reference_raw,_,_=self.f.prepare_clone(prepare=False)
        self.s=self.f.s;self.f.t=time.monotonic();self.s.clock=time.monotonic;self.s.availability.clock=time.monotonic
        self.native=dict(deployment_id='a'*32,configuration_sha256='b'*64,
            lane=self.s.journal.value['lane'],boot=self.s.boot,runtime_manifest_sha256='d'*64)
        self.snapshot=copy.deepcopy(self.f.snapshot)
        self.snapshot[1]['execution']['native']=self.native.copy()
        self.a=Authority(*self.snapshot)
        self.s.registrations={self.a.binding:dict(snapshot=self.snapshot,execution=self.f.e,
            bundles={'bundle':self.f.fixture.bundle},inventory=self.f.inventory)}
        # This post-constructor native binding is explicit fixture enrollment.
        self.s._preset_templates=copy.deepcopy(self.s.registrations)
        self.spool=self.f.fixture.root/'control-spool';self.spool.mkdir(mode=0o700)
        self.stage=self.f.fixture.root/'control-stage';self.stage.mkdir(mode=0o770);self.stage.chmod(0o770)
        self.cert=b'fixture-server-certificate';self.pin=hashlib.sha256(self.cert).hexdigest()
        self.requests=[];self.mutate=lambda q,v:v;self.before=lambda q:None
        self.runtime=self.make_runtime()

    def make_runtime(self):
        return NativeRuntime(host='127.0.0.1',port=4443,tls=None,peer_sha256=self.pin,
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

    def exchange(self,channel,config,cutoff,**kwargs):
        from worker import native_child
        from tests.worker.test_native_qwen import inert_guard
        raw=canonical(config);ack=canonical(dict(kind='run',token=config['token'],binding=config['binding'],cutoff=config['cutoff']));child=BytesChannel(struct.pack('!I',len(raw))+raw+struct.pack('!I',len(ack))+ack)
        backend=self.f.fixture.backend
        for method in ('synthesise_preset','synthesise_clone'):
            getattr(backend,method).return_value.metadata=dict(checkpoint=str(self.f.fixture.bundle),revision=self.f.e.revision)
        with inert_guard(),patch('lib.tts_backends.qwen_native.NativeQwenBackend',return_value=backend):
            native_child.run_socket(child)
        return self.parent_exchange(BytesChannel(child.sent),config,cutoff,**kwargs)

    def finish(self):
        self.s.active['receiver'].join(2)
        self.case.assertFalse(self.s.active['receiver'].is_alive())
        self.f.h.code=0;self.s.tick()

    def launch(self):
        self.runtime.allocate_stage(self.a)
        rid=self.runtime.prepare(self.a)
        with patch('worker.native_child.parent_exchange',side_effect=self.exchange):
            self.runtime.start(rid);self.finish()
        return rid


class NativePrepareDiagnosticTests(unittest.TestCase):
    def test_prepare_failure_reports_bounded_source_and_preserves_exception(self):
        runtime=object.__new__(NativeRuntime)
        output=io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(Refused) as raised:
                runtime.prepare(None,fixture='private payload and path /secret/media.wav')
        self.assertEqual(str(raised.exception),'native_fixture')
        line=output.getvalue().strip()
        self.assertTrue(line.startswith('native_prepare_failure:'))
        report=json.loads(line.partition(':')[2])
        self.assertEqual(report['chain'][0]['type'],'Refused')
        self.assertEqual(report['chain'][0]['file'],'native_runtime.py')
        self.assertNotIn('private payload',line)
        self.assertNotIn('/secret/',line)


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self): self.f=NativeFixture(self)

    def test_owner_finalizes_stage_before_scanner_and_executor_under_0077(self):
        from service.resource_store import ResourceStore
        f=self.f;stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        from tests.service.test_pocket_preparation import PocketResourceRaceTests
        owner=PocketResourceRaceTests();owner.setUp();self.addCleanup(owner.doCleanups)
        owner.running_namespace()
        b=owner.base;f.stage=b.stage.resolve();f.stage.chmod(0o770);f.runtime.staging=f.stage
        stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        scanner=b.admission.store
        real=os.fchmod; transitions=[]
        def chmod(fd,mode):
            transitions.append(os.fstat(fd).st_mode & 0o777)
            self.assertEqual(transitions,[0o700])
            return real(fd,mode)
        mask=os.umask(0o077)
        try:
            # This is the owner detach seam, before any executor or owner scan.
            with patch('service.native_runtime.os.fchmod',side_effect=chmod):
                self.assertEqual(f.runtime.detach(f.a,None),b'')
            b.admission.check()
            before=b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]
            rid=f.runtime.prepare(f.a)
            with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
                f.runtime.start(rid);f.finish()
            opened=[];collected=[];real_open=os.open
            def scan_open(path,flags,*args,**kwargs):
                if path==stage.name and kwargs.get('dir_fd') is not None:
                    opened.append(True)
                    self.assertEqual(stage.stat().st_mode & 0o777,0o770)
                    if not collected:collected.append(f.runtime.finalize_collect(f.runtime.collect(f.a,rid)))
                return real_open(path,flags,*args,**kwargs)
            with patch('service.native_runtime.os.chmod',side_effect=AssertionError('visible chmod')),patch('service.native_runtime.os.fchmod',side_effect=AssertionError('visible fchmod')),patch('service.resource_store.os.open',side_effect=scan_open):
                b.admission.check()
                receipt,result=collected[0]
                after=b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]
            self.assertTrue(opened);self.assertTrue(receipt['reaped']);self.assertIsNotNone(result)
            self.assertGreaterEqual(after,before)
            self.assertGreaterEqual(scanner._measure_staging(f.stage),(stage/'output.wav').stat().st_size)
            self.assertEqual(scanner.state(),'healthy')
            self.assertEqual(b.core.db.execute('SELECT * FROM resource_faults').fetchall(),[])
        finally: os.umask(mask)

    def test_private_enumeration_cannot_retire_executor_import(self):
        import threading
        from tests.service.test_pocket_preparation import PocketResourceRaceTests
        f=self.f;rid=f.launch()
        owner=PocketResourceRaceTests();owner.setUp();self.addCleanup(owner.doCleanups)
        b=owner.base;scanner=b.admission.store
        scanner.roots=(*scanner.roots,f.spool)
        alias=f.spool/f.a.binding/'import'
        ready=threading.Event();release=threading.Event();done=threading.Event()
        pending=[];errors=[];real_write=__import__('service.native_runtime',fromlist=['_write'])._write
        def write(path,data):
            real_write(path,data)
            if path==alias:
                ready.set();self.assertTrue(release.wait(3))
        def collect():
            try: pending.append(f.runtime.collect(f.a,rid))
            except BaseException as exc: errors.append(exc)
            finally: done.set()
        real_walk=os.walk;listed=[]
        def walk(root,*args,**kwargs):
            for directory,dirs,files in real_walk(root,*args,**kwargs):
                if Path(directory)==alias.parent and 'import' in files:
                    listed.append(True);release.set();self.assertTrue(done.wait(3))
                    self.assertTrue(alias.exists())
                    # Even a completed executor cannot retire the scanned name.
                    self.assertFalse((f.stage/f.a.job['attempts'][-1]['staging_id']/'output.wav').exists())
                yield directory,dirs,files
        with patch('service.native_runtime._write',side_effect=write):
            thread=threading.Thread(target=collect);thread.start()
            try:
                self.assertTrue(ready.wait(3))
                with patch('service.resource_store.os.walk',side_effect=walk):b.admission.check()
            finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[]);self.assertTrue(listed)
        before=b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]
        receipt,observation=f.runtime.finalize_collect(pending[0])
        self.assertFalse(alias.exists());self.assertTrue(receipt['reaped']);self.assertIsNotNone(observation)
        b.admission.check()
        self.assertEqual(scanner.state(),'healthy')
        self.assertEqual(b.core.db.execute('SELECT * FROM resource_faults').fetchall(),[])
        self.assertGreaterEqual(b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0],before)

    def test_pending_owner_cancel_and_partial_restart_retain(self):
        import threading
        f=self.f;rid=f.launch();pending=f.runtime.collect(f.a,rid)
        errors=[]
        def wrong_owner():
            try:f.runtime.finalize_collect(pending)
            except Refused as exc:errors.append(str(exc))
        thread=threading.Thread(target=wrong_owner);thread.start();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,['native_owner_required'])
        self.assertFalse(f.make_runtime().admission_open)
        f.runtime.interrupt(rid)
        with patch.object(f.runtime,'_rpc',side_effect=AssertionError('owner network')):
            with self.assertRaisesRegex(Refused,'cancelled'):f.runtime.finalize_collect(pending)
        self.assertTrue((f.spool/f.a.binding/'import').exists())
        self.assertEqual(list((f.stage/f.a.job['attempts'][-1]['staging_id']).iterdir()),[])

    def test_owner_stage_collision_and_fsync_uncertainty_retain_without_overwrite(self):
        f=self.f;stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        with patch('service.native_runtime.os.fsync',side_effect=OSError('uncertain')):
            with self.assertRaisesRegex(OSError,'uncertain'):f.runtime.detach(f.a,None)
        self.assertEqual(stage.stat().st_mode & 0o777,0o770)
        marker=stage/'retained';marker.write_bytes(b'evidence')
        with self.assertRaises(FileExistsError):f.runtime.detach(f.a,None)
        self.assertEqual(marker.read_bytes(),b'evidence');self.assertEqual(f.requests,[])

    def test_executor_refuses_wrong_mode_and_substituted_stage(self):
        f=self.f;rid=f.launch();stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        pending=f.runtime.collect(f.a,rid)
        stage.chmod(0o700)
        with self.assertRaises(Refused):f.runtime.finalize_collect(pending)
        f=NativeFixture(self);rid=f.launch();stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        pending=f.runtime.collect(f.a,rid)
        real_open=os.open
        def replaced(path,flags,*args,**kwargs):
            fd=real_open(path,flags,*args,**kwargs)
            if path==stage:
                stage.rename(stage.with_name(stage.name+'-retained'))
                stage.mkdir(mode=0o770);stage.chmod(0o770)
            return fd
        with patch('service.native_runtime.os.open',side_effect=replaced):
            with self.assertRaises(Refused):f.runtime.finalize_collect(pending)
        self.assertFalse((stage/'output.wav').exists())

    def owner_import_fault(self, point):
        import stat
        from service import native_runtime as module
        f=self.f;rid=f.launch();pending=f.runtime.collect(f.a,rid)
        directory=f.spool/f.a.binding;real_write=module._write;real_sync=module._sync;injected=[]
        def write(path,data):
            if point=='complete-write' and path==directory/'import-complete':
                injected.append(point);raise OSError('single completion write uncertainty')
            return real_write(path,data)
        def sync(path):
            if point=='complete-dir' and path==directory and (directory/'import-complete').exists() and not injected:
                injected.append(point);raise OSError('single completion directory uncertainty')
            return real_sync(path)
        def link(*args,**kwargs):
            injected.append(point);raise OSError('single prelink uncertainty')
        with patch('service.native_runtime._write',side_effect=write),patch('service.native_runtime._sync',side_effect=sync):
            if point=='prelink':
                with patch('service.native_runtime.os.link',side_effect=link):
                    with self.assertRaises(OSError):f.runtime.finalize_collect(pending)
            else:
                with self.assertRaises(OSError):f.runtime.finalize_collect(pending)
        self.assertEqual(injected,[point]);self.assertTrue(f.runtime.failed)
        retained={x.name:x.read_bytes() for x in directory.iterdir()}
        for _ in range(5):
            f.runtime.observe(rid);f.runtime.terminate(rid)
            with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.collect(f.a,rid)
            with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.finalize_collect(pending)
            with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.call('ack',dict(runtime_id=rid))
        self.assertEqual(retained,{x.name:x.read_bytes() for x in directory.iterdir()})
        reopened=f.make_runtime();self.assertFalse(reopened.admission_open);reopened.bind(f.a,rid)
        with self.assertRaisesRegex(Refused,'uncertain'):reopened.collect(f.a,rid)
        self.assertNotIn('ack',[q['op'] for q in f.requests])

    def test_prelink_failure_is_closed_on_replay_and_reopen(self):self.owner_import_fault('prelink')
    def test_completion_write_failure_is_closed_on_replay_and_reopen(self):self.owner_import_fault('complete-write')
    def test_completion_directory_failure_is_closed_even_with_visible_complete(self):self.owner_import_fault('complete-dir')

    def test_output_file_fsync_failure_is_closed_before_import(self):
        import stat
        f=self.f;rid=f.launch();real=os.fsync;injected=[]
        def sync(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode) and not injected:
                injected.append(True);raise OSError('single output file sync uncertainty')
            return real(fd)
        with patch('service.native_runtime.os.fsync',side_effect=sync):
            with self.assertRaises(OSError):f.runtime.collect(f.a,rid)
        self.assertEqual(injected,[True]);self.assertTrue(f.runtime.failed)
        for _ in range(3):
            f.runtime.terminate(rid);f.runtime.observe(rid)
            with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.collect(f.a,rid)
        self.assertFalse(f.make_runtime().admission_open)
        self.assertFalse((f.stage/f.a.job['attempts'][-1]['staging_id']/'output.wav').exists())

    def test_completed_but_unhanded_off_reopen_is_conservative(self):
        f=self.f;rid=f.launch();receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertIsNotNone(observation)
        self.assertIsNotNone(f.runtime.finalize_collect(f.runtime.collect(f.a,rid))[1])
        reopened=f.make_runtime();self.assertFalse(reopened.admission_open);reopened.bind(f.a,rid)
        self.assertEqual(reopened.observe(rid).state.value,'stopped-and-reaped')
        with self.assertRaisesRegex(Refused,'uncertain'):reopened.collect(f.a,rid)

    def test_production_connector_uses_remaining_absolute_budget(self):
        f=self.f; now=[100.]; timeouts=[]; closed=[]; handshakes=[]
        class Raw:
            def settimeout(self,value): timeouts.append(value)
            def connect(self,address): now[0]+=1.5
            def close(self): closed.append('raw')
        class TLS:
            def settimeout(self,value): timeouts.append(value)
            def do_handshake(self): handshakes.append(True);now[0]+=.25
            def close(self): closed.append('tls')
        class Context:
            def wrap_socket(self,raw,**kw):
                self.kw=kw;return TLS()
        f.runtime.tls=Context()
        with patch('service.native_runtime.time.monotonic',side_effect=lambda:now[0]),patch('service.native_runtime.socket.socket',return_value=Raw()),patch('service.native_runtime.socket.getaddrinfo',side_effect=AssertionError('DNS')):
            with f.runtime._connect(102.): pass
        self.assertEqual(timeouts,[2.,.5,.5]);self.assertEqual(handshakes,[True])
        self.assertEqual(closed,['tls','raw']);self.assertFalse(f.runtime.tls.kw['do_handshake_on_connect'])
        self.assertEqual(f.requests,[])

    def test_production_connector_expiry_closes_without_handshake_or_wire(self):
        f=self.f
        for tcp_time in (2.,3.):
            now=[100.];closed=[]
            class Raw:
                def settimeout(self,value): pass
                def connect(self,address): now[0]+=tcp_time
                def close(self): closed.append(True)
            tls=unittest.mock.Mock();f.runtime.tls=tls
            with patch('service.native_runtime.time.monotonic',side_effect=lambda:now[0]),patch('service.native_runtime.socket.socket',return_value=Raw()) as create,patch('service.native_runtime.socket.getaddrinfo',side_effect=AssertionError('DNS')):
                with self.assertRaisesRegex(Refused,'deadline'):
                    with f.runtime._connect(102.): self.fail('expired connector yielded')
            tls.wrap_socket.assert_not_called();self.assertEqual(create.call_count,1)
            self.assertEqual(closed,[True]);self.assertEqual(f.requests,[])

    def test_expired_connector_does_not_attempt_tcp(self):
        f=self.f;raw=unittest.mock.Mock()
        with patch('service.native_runtime.time.monotonic',return_value=102.),patch('service.native_runtime.socket.socket',return_value=raw):
            with self.assertRaisesRegex(Refused,'deadline'):
                with f.runtime._connect(102.): self.fail('expired')
        raw.connect.assert_not_called();raw.close.assert_called_once()
        self.assertEqual(f.requests,[])

    def test_hostname_requires_separate_trusted_numeric_enrollment(self):
        f=self.f
        with self.assertRaisesRegex(Refused,'numeric_endpoint'):
            NativeRuntime(host='host.docker.internal',port=4443,tls=None,peer_sha256=f.pin,
                native=f.native,spool=f.spool,staging=f.stage,local_import_uid=os.getuid(),connector=f.connect)

    def test_future_allocation_authorizes_from_template_without_control_hook(self):
        f=self.f;snapshot=copy.deepcopy(f.snapshot)
        snapshot[0]['id']='f'*32
        a=Authority(*snapshot)
        self.assertNotIn(a.binding,f.s.registrations)
        rid=f.runtime.prepare(a)
        self.assertIn(a.binding,f.s.registrations)
        self.assertEqual(f.s.journal.value['records'][0]['snapshot'],snapshot)
        self.assertEqual(f.runtime.prepare(a),rid)

    def test_changed_trusted_template_fields_refuse_without_journal(self):
        f=self.f
        paths=[(0,'principal_id'),(1,'plan','voice_versions',0,'id'),
            (1,'execution','descriptor','revision'),(1,'execution','descriptor','checkpoint'),
            (1,'execution','descriptor','reference_sha256'),(1,'execution','descriptor','voice_revision'),(1,'execution','bundles','bundle'),
            (1,'execution','references'),(1,'execution','hardware','host','architecture'),
            (1,'execution','native','configuration_sha256'),(1,'bundles',0,'instructions'),
            (3,0,'qualification_receipt'),(3,0,'id'),
            (1,'execution','hardware','host','deployment'),(1,'bundles',0,'languages')]
        for path in paths:
            with self.subTest(path=path):
                snapshot=copy.deepcopy(f.snapshot);snapshot[0]['id']='f'*32
                target=snapshot
                for key in path[:-1]: target=target[key]
                key=path[-1];old=target.get(key) if isinstance(target,dict) else target[key]
                target[key]=not old if type(old) is bool else ({'foreign':'value'} if type(old) is dict else 'foreign')
                if path==(1,'plan','voice_versions',0,'id'):
                    snapshot[1]['plan']['request']['voice_version_id']='foreign'
                from contracts import validate as oracle
                snapshot[0]['input_sha256']=oracle.plan_hash(snapshot[1]['plan'])
                binding=hashlib.sha256(canonical(snapshot)).hexdigest()
                q=f.f.q('prepare',{'snapshot':snapshot},binding=binding)
                try: f.s.dispatch(q)
                except Exception: pass
                else: self.fail('altered template accepted')
                self.assertEqual(f.s.journal.value['records'],[])
                self.assertNotIn(binding,f.s.registrations)

    def test_actual_tls_codecs_handlers_child_transfer_receipt_and_exact_replay(self):
        f=self.f;rid=f.launch()
        receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertTrue(receipt['receiver_closed']);self.assertNotIn('engine_uid',receipt)
        self.assertEqual(f.f.h.waited,1)
        raw=(f.stage/f.a.job['attempts'][-1]['staging_id']/'output.wav').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),observation.manifest['outputs'][0]['sha256'])
        self.assertEqual(f.runtime.finalize_collect(f.runtime.collect(f.a,rid)),(receipt,unittest.mock.ANY))
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
        with self.assertRaises(Refused):f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertEqual(list((f.stage/f.a.job['attempts'][-1]['staging_id']).iterdir()),[])
        f.mutate=lambda q,v:(dict(v,value=dict(v['value'],data='AAAA')) if q['op']=='get_output' else v)
        with self.assertRaises(Refused):f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertEqual(list((f.stage/f.a.job['attempts'][-1]['staging_id']).iterdir()),[])
        self.assertTrue(f.runtime.failed)
        for _ in range(3):
            f.runtime.terminate(rid);f.runtime.observe(rid)
            with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.collect(f.a,rid)
        with self.assertRaisesRegex(Refused,'uncertain'):f.make_runtime().prepare(f.a)

    def test_stop_and_cancel_during_output_import_are_not_death_proof(self):
        f=self.f;rid=f.launch()
        f.before=lambda q:f.runtime.interrupt(rid) if q['op']=='get_output' else None
        receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertIsNone(observation);self.assertTrue(receipt['reaped'])
        self.assertEqual(list((f.stage/f.a.job['attempts'][-1]['staging_id']).iterdir()),[])

    def test_local_import_uid_stage_symlink_and_conflict_refuse(self):
        f=self.f
        with self.assertRaisesRegex(Refused,'uid'):
            NativeRuntime(host='127.0.0.1',port=1,tls=None,peer_sha256=f.pin,native=f.native,
                spool=f.spool,staging=f.stage,local_import_uid=os.getuid()+1,connector=f.connect)
        rid=f.launch();stage=f.stage/f.a.job['attempts'][-1]['staging_id']
        target=stage/'output.wav';target.symlink_to(f.spool/'foreign')
        with self.assertRaises(OSError):f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
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
        f.runtime.allocate_stage(f.a)
        rid=f.runtime.prepare(f.a,detached=raw)
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            f.runtime.start(rid)
            f.runtime.start(rid)
            f.finish()
        receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertTrue(receipt['timely']);self.assertIsNotNone(observation)
        self.assertEqual(len(f.f.calls),1)
        self.assertEqual([q['op'] for q in f.requests].count('put_input'),1)
        self.assertEqual(f.f.fixture.backend.synthesise_clone.call_args.args[2],'hello')

    def test_cancel_upload_and_unknown_restart_never_launch_or_release(self):
        f=self.f;raw=self.clone();rid=f.runtime.prepare(f.a,detached=raw)
        f.before=lambda q:f.runtime.interrupt(rid) if q['op']=='put_input' else None
        with self.assertRaisesRegex(Refused,'cancelled'):f.runtime.start(rid)
        self.assertEqual(f.f.calls,[])
        receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
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
        receipt,observation=f.runtime.finalize_collect(f.runtime.collect(f.a,rid))
        self.assertTrue(receipt['reaped']);self.assertIsNone(observation)
        self.assertEqual(list((f.stage/f.a.job['attempts'][-1]['staging_id']).iterdir()),[])
        other=NativeFixture(self);rid=other.launch()
        first=other.runtime.finalize_collect(other.runtime.collect(other.a,rid))
        other.runtime.records[rid]['cutoff']=time.monotonic()-1
        self.assertEqual(other.runtime.finalize_collect(other.runtime.collect(other.a,rid))[0],first[0])

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
