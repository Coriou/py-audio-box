"""Installed activation source tests; injected identity/host/reap/backend, no listener."""
import sys
import asyncio
import copy
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import httpx

from service import entrypoint as e, install_policy as ip
from service.core import Core, encode
from service.auth import installed_credentials
from service.dispatcher import Dispatcher
from service.execution_context import KokoroContexts
from service.persistent_init import digest
from tests.release import test_headless_install as gate_tests
from tests.worker import test_engine_lifecycle as engine_tests
from tests.service import test_provisioning as provisioning_tests
from worker.lifecycle import Observation, State
from service.worker_transport import TransportObservation


class ActivationTests(unittest.TestCase):
    notification_directory = gate_tests.GateTests.notification_directory
    proposal = engine_tests.EngineLifecycleTests.proposal

    def setUp(self):
        gate_tests.GateTests.setUp(self)
        self.stage = self.fixture.roots['staging']
        bundle = provisioning_tests.ProvisioningTests(); bundle.setUp(); self.addCleanup(bundle.doCleanups)
        self.bundle = bundle.run_provision()
        self.context = self.fixture.config['contexts']['alice']
        self.profiles = self.fixture.config['profiles']
        self.mapping = {next(iter(self.fixture.config['bundles'])): self.bundle}
        path = self.notification_directory()
        self.waiting = e.WaitingControl(directory=path)
        self.addCleanup(lambda:self.waiting.close())
        e.publish(encode(self.gate).encode(), directory=path)
        f = self.fixture.fixture; translate = f.translate
        def owner(info, current=None):
            if current is not None and any(current == p or p in current.parents for p in (self.bundle, self.stage)):
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            return translate(info,current)
        f.translate = owner
        real_lstat = os.lstat
        self.enterContext(patch.object(os,'lstat',side_effect=lambda path,*a,**kw: owner(real_lstat(path,*a,**kw),Path(path))))
        for name in ('getuid','geteuid','getgid','getegid'):
            self.enterContext(patch.object(os,name,return_value=10001))
        self.enterContext(patch.object(os,'getgroups',return_value=[]))
        self.enterContext(patch.object(e,'local_incarnation',return_value=self.incarnation))
        self.contexts = KokoroContexts({'alice': self.context}, self.mapping)
        outer = self
        class Runtime:
            def __init__(inner):
                inner.authority=None; inner.finish=threading.Event(); inner.stopped=threading.Event(); inner.cancel=False
                inner.hellos=0; inner.changed_at=None
            def call(inner,op,body):
                if op=='hello':
                    inner.hellos+=1
                    value=copy.deepcopy(outer.fixture.hello)
                    if inner.changed_at and inner.hellos>=inner.changed_at: value['boot']='d'*32
                    return value
                return dict(acked=True)
            def prepare(inner,authority,fixture):
                outer.assertEqual(fixture,'kokoro');inner.authority=authority
                return outer.fixture.lane+'/'+outer.fixture.hello['boot']+'/allocation'
            def start(inner,identity): return dict(state='alive')
            def observe(inner,identity,timeout):
                return Observation(identity,State.STOPPED if inner.finish.is_set() else State.ALIVE)
            def terminate(inner,identity,timeout): inner.cancel=True;inner.stopped.set()
            def collect(inner,authority,identity):
                receipt=dict(runtime_id=identity,binding=authority.binding,reaped=True,engine_uid=10001,
                    reap_method='pid1-waitpid-echild',reaped_children=0,timely=not inner.cancel,original_deadline=authority.job['deadline'])
                if inner.cancel:return receipt,None
                _,proposal,_=outer.proposal(authority=authority)
                return receipt,TransportObservation('proposal','INJECTED ONLY',message=proposal['message'],manifest=proposal['manifest'])
        self.runtime = Runtime()
        if getattr(self, 'bootstrap', True):
            bootstrap_owner = self.owner()
            try:
                self.wait(lambda:bootstrap_owner.call('status')['persistent_activation'])
                from service.domain import production_call
                saved = production_call(bootstrap_owner, 'alice', 'voice_preset_register',
                    dict(preset=self.context['execution']['descriptor']['preset'],
                         voice_id=self.context['plan']['voice_versions'][0]['voice_id'], expected_revision=0), 'register')
                self.context = copy.deepcopy(self.context)
                self.context['plan']['voice_versions'] = [saved['version']]
                self.context['plan']['request']['voice_version_id'] = saved['version']['id']
            finally: bootstrap_owner.close()
            self.runtime.hellos = 0

    def owner(self):
        return Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=self.profiles,
            budget=self.fixture.config['budget'],production_contexts=self.contexts,
            installed_activation=e.InstalledActivation(self.waiting,self.fixture.config))

    def wait(self, fn):
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            value=fn()
            if value:return value
            time.sleep(.02)
        self.fail('bounded source wait expired')

    def test_post_core_private_reader_under_lifetime_lock(self):
        from service import recovery
        reader=recovery.installed_startup_read; phases=[]
        def read(*args,**kwargs):
            import fcntl
            fd=os.open(self.root/'control.lock',os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError): fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
            phases.append(self.runtime.hellos)
            if len(phases)==1: self.assertNotIn('owned_db',kwargs)
            else: self.assertIsInstance(kwargs.get('owned_db'),sqlite3.Connection)
            return reader(*args,**kwargs)
        with patch.object(recovery,'installed_startup_read',side_effect=read):
            owner=self.owner()
        try:
            self.wait(lambda:owner.call('status')['persistent_activation'])
            self.assertEqual(len(phases),2)
            with self.assertRaisesRegex(ValueError,'production_commands'):owner.call('submit','alice','synthetic')
        finally:owner.close()

    def test_changed_post_acquire_hello_refuses_and_releases_core(self):
        self.runtime.changed_at=3
        with self.assertRaisesRegex(ValueError,'startup_boot'):self.owner()
        core=Core(self.root);core.close()
        self.assertIsNone(self.runtime.authority)

    def test_changed_post_acquire_second_hello_refuses(self):
        self.runtime.changed_at=4
        with self.assertRaisesRegex(ValueError,'post_owner'):self.owner()
        core=Core(self.root);core.close()

    def test_late_reconciliation_hello_change_stays_closed(self):
        self.runtime.changed_at=5
        owner=self.owner()
        try:
            status=self.wait(lambda:(s if (s:=owner.call('status'))['reconciled'] else None))
            self.assertFalse(status['persistent_activation'])
            with self.assertRaises(ValueError):owner.call('production_generate','alice',self.context['plan']['request'],key='closed')
        finally:owner.close()

    def test_integrity_fault_never_activates(self):
        (self.root/'assets'/'unknown').write_bytes(b'x')
        owner=self.owner()
        try:
            status=self.wait(lambda:(s if (s:=owner.call('status'))['reconciled'] else None))
            self.assertTrue(status['integrity_fault']);self.assertFalse(status['persistent_activation'])
        finally:owner.close()

    def test_resource_fault_never_activates(self):
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute("UPDATE resource_monitor SET state='fault'")
        owner=self.owner()
        try:
            status=self.wait(lambda:(s if (s:=owner.call('status'))['reconciled'] else None))
            self.assertTrue(status['resource_fault']);self.assertFalse(status['persistent_activation'])
        finally:owner.close()

    def inputs(self):
        config=self.fixture.fixture.root/'install.json'
        key=self.fixture.fixture.root/'control.key'
        config.write_text(encode(self.fixture.config));config.chmod(0o440)
        key.write_text(encode(dict(version=1,credentials=[dict(token='x'*64,principal='alice',scopes=['read','production-write'])])))
        key.chmod(0o640)
        for path in (config,key):
            st=path.stat();self.fixture.fixture.virtual[(st.st_dev,st.st_ino)]=(0,10001)
        return config,key

    def test_preowner_refusal_never_constructs_core(self):
        cases = [('UPDATE p1_installation SET scope=?', ('owned-synthetic',)),
                 ('UPDATE p1_registrations SET digest=?', ('f'*64,)),
                 ('INSERT INTO p1_recovery_plans VALUES(?,?,?)', ('pending','{}','x'))]
        for sql,values in cases:
            with self.subTest(sql=sql):
                db=sqlite3.connect(self.root/'core.sqlite')
                db.close()
                with sqlite3.connect(self.root/'core.sqlite') as db:db.execute(sql,values)
                with patch('service.dispatcher.Core') as core:
                    with self.assertRaises(ValueError):self.owner()
                    core.assert_not_called()
                # Restore only this disposable fixture, preserving no product state.
                with sqlite3.connect(self.root/'core.sqlite') as db:
                    if 'installation' in sql:db.execute('UPDATE p1_installation SET scope=?',(ip.SCOPE,))
                    elif 'registrations' in sql:db.execute('UPDATE p1_registrations SET digest=id')
                    else:db.execute('DELETE FROM p1_recovery_plans')
        self.assertIsNone(self.runtime.authority)

    def test_queued_cancel_release_only_after_checked_healthy_startup(self):
        from service.production import Production
        from service.admission import Admission
        core=Core(self.root)
        admission=Admission(core,self.stage,self.fixture.config['budget'])
        job=Production(core).generate('alice','queued-cancel',self.context,_admission=admission,_profiles=self.profiles)
        job.update(state='cancelled',stage='never_started')
        with core.transaction():core._save(job)
        core.close()
        owner=self.owner()
        try:
            self.wait(lambda:owner.call('status')['persistent_activation'])
            self.assertEqual(owner.call('status')['reservations'][-1]['state'],'retained')
            self.assertIsNone(self.runtime.authority)
        finally:owner.close()

    def test_queued_cancel_late_hello_keeps_accounting(self):
        from service.production import Production
        from service.admission import Admission
        core=Core(self.root)
        job=Production(core).generate('alice','late-cancel',self.context,
            _admission=Admission(core,self.stage,self.fixture.config['budget']),_profiles=self.profiles)
        job.update(state='cancelled',stage='never_started')
        with core.transaction():core._save(job)
        core.close()
        self.runtime.changed_at=5
        owner=self.owner()
        try:
            status=self.wait(lambda:(s if (s:=owner.call('status'))['reconciled'] else None))
            self.assertFalse(status['persistent_activation'])
            self.assertEqual(status['reservations'][-1]['state'],'queued')
        finally:owner.close()

    def cancellation_case(self, phase, fail_close=False, direct=False, wrapper=False, reject_callback=False):
        entered=threading.Event(); release=threading.Event(); captured=[]; closes=[]
        original_prepare=e.InstalledActivation.prepare
        def prepare(activation,*args):
            result=original_prepare(activation,*args)
            if phase=='preparation':
                entered.set();self.assertTrue(release.wait(4))
            return result
        def acquire(*args,**kwargs):
            owner=Dispatcher(*args,**kwargs);captured.append(owner)
            original_close=owner.close
            def close():
                closes.append(True)
                if fail_close:raise RuntimeError('injected_close_unknown')
                if phase=='close':
                    entered.set();self.assertTrue(release.wait(4))
                original_close()
            owner.close=close
            if phase=='acquisition':
                entered.set();self.assertTrue(release.wait(4))
            return owner
        closed=[];original_notification_close=self.waiting.close
        def notification_close():
            if not closed:closed.append(True);original_notification_close()
        self.waiting.close=notification_close
        with patch('service.dispatcher.Dispatcher',side_effect=acquire), patch.object(e.InstalledActivation,'prepare',prepare):
            app=self.factory()
            async def run():
                context=app.router.lifespan_context(app)
                startup=asyncio.create_task(context.__aenter__())
                if phase=='close':
                    await startup
                    startup=asyncio.create_task(context.__aexit__(None,None,None))
                self.assertTrue(await asyncio.to_thread(entered.wait,4))
                startup.cancel()
                await asyncio.sleep(.02)
                startup.cancel()
                with self.assertRaises(asyncio.CancelledError):await startup
                self.assertFalse(closed)
                record=app.state.installed_lifecycle
                task=record['task']
                if direct:
                    if wrapper:
                        record['operations'][phase].wrapper.cancel()
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(.01)
                        self.assertFalse(closed)
                        self.assertIn(id(record),e._INSTALLED_LIFECYCLES)
                        self.assertTrue(task.cancelled())
                        self.assertIs(record['task'],task)
                    if phase=='acquisition':
                        self.assertIs(record['operations'][phase].result.done(),False)
                if reject_callback:
                    loop=asyncio.get_running_loop()
                    schedule=loop.call_soon_threadsafe
                    def rejected(callback,*args,**kwargs):
                        if getattr(callback,'__name__',None)=='advance':
                            raise RuntimeError('event loop is closed')
                        return schedule(callback,*args,**kwargs)
                    with patch.object(loop,'call_soon_threadsafe',side_effect=rejected):
                        release.set()
                        op=record['operations'][phase]
                        await asyncio.to_thread(self.wait,op.result.done)
                        await asyncio.sleep(.02)
                    self.assertIs(op.result.result(),captured[0])
                    self.assertIn(id(record),e._INSTALLED_LIFECYCLES)
                    self.assertFalse(closed)
                    self.assertFalse(record['closed'])
                    self.assertEqual(closes,[])
                    e._INSTALLED_LIFECYCLES.pop(id(record),None)
                    return
                release.set()
                if direct:
                    await asyncio.to_thread(self.wait,lambda:id(record) not in e._INSTALLED_LIFECYCLES)
                    task=record['task']
                try:
                    if fail_close:
                        with self.assertRaisesRegex(RuntimeError,'injected_close_unknown'):
                            await asyncio.wait_for(asyncio.shield(task),4)
                        self.assertIs(record['owner'],captured[0])
                        self.assertIs(e._INSTALLED_LIFECYCLES[id(record)],record)
                        self.assertFalse(closed)
                    else:
                        await asyncio.wait_for(asyncio.shield(task),4)
                        self.assertNotIn(id(record),e._INSTALLED_LIFECYCLES)
                        self.assertTrue(closed)
                    self.assertEqual(len(closes),int(phase!='preparation'))
                finally:
                    for owner in captured:
                        if owner.thread.is_alive(): Dispatcher.close(owner)
                    e._INSTALLED_LIFECYCLES.pop(id(record),None)
                    notification_close()
            try:asyncio.run(run())
            finally:
                release.set()
                for owner in captured:
                    if owner.thread.is_alive(): Dispatcher.close(owner)
                notification_close()

    def test_repeated_acquisition_cancellation_retains_owner(self):
        self.cancellation_case('acquisition')

    def test_repeated_preparation_cancellation_retains_dependency(self):
        self.cancellation_case('preparation')

    def test_cancelled_acquisition_close_failure_retains_obligations(self):
        self.cancellation_case('acquisition',fail_close=True)

    def test_direct_cleanup_cancel_acquisition(self):
        self.cancellation_case('acquisition',direct=True)

    def test_direct_cleanup_cancel_preparation(self):
        self.cancellation_case('preparation',direct=True)

    def test_direct_cleanup_cancel_close_thread(self):
        self.cancellation_case('close',direct=True)

    def test_direct_acquisition_wrapper_and_cleanup_cancel(self):
        self.cancellation_case('acquisition',direct=True,wrapper=True)

    def test_callback_closed_retains_eventual_owner_obligation(self):
        self.cancellation_case('acquisition',direct=True,reject_callback=True)

    def test_orphan_guard_prevents_activation(self):
        from service.production import Production
        from service.admission import Admission
        core=Core(self.root)
        job=Production(core).generate('alice','orphan',self.context,_admission=Admission(core,self.stage,self.fixture.config['budget']),_profiles=self.profiles)
        job.update(state='running',stage='loading')
        with core.transaction():core._save(job)
        core.close()
        owner=self.owner()
        try:
            status=self.wait(lambda:(s if (s:=owner.call('status'))['reconciled'] else None))
            self.assertFalse(status['persistent_activation']);self.assertTrue(owner.orphaned)
            self.assertIsNone(self.runtime.authority)
        finally:owner.close()

    def factory(self):
        config,key=self.inputs()
        # Only the installed /models mount is substituted with a sealed local fixture.
        with patch('service.execution_context.installed_mapping',return_value=self.mapping):
            return e.create_installed_app(configuration_file=config,credential_file=key,
                root=self.root,staging=self.stage,runtime=self.runtime,waiting=self.waiting)

    def test_missing_credential_refuses_before_models_owner_or_lifespan(self):
        config,key=self.inputs();key.unlink()
        with patch('service.execution_context.installed_mapping') as mapping:
            with self.assertRaises(FileNotFoundError):
                e.create_installed_app(configuration_file=config,credential_file=key,
                    root=self.root,staging=self.stage,runtime=self.runtime,waiting=self.waiting)
            mapping.assert_not_called()
        self.assertEqual(self.runtime.hellos,0)

    def test_default_model_mount_refuses_without_discovery(self):
        from service.execution_context import installed_mapping
        with self.assertRaises((OSError,ValueError)):
            installed_mapping(self.fixture.config,policy=ip.SPLIT)
        bad=copy.deepcopy(self.fixture.config)
        next(iter(bad['bundles'].values()))['path']='/tmp/arbitrary'
        with self.assertRaisesRegex(ValueError,'bundle_binding'):installed_mapping(bad,policy=ip.SPLIT)

    def credentials(self):
        from service.auth import Credential, StaticCredentials
        return StaticCredentials([Credential('x'*64,'alice',frozenset({'read','production-write'}))])

    def test_factory_lifespan_authenticated_job_artifact_journey(self):
        app=self.factory()
        # Lifespan owns waiting.close; replace the registered test cleanup with an idempotent wrapper.
        close=self.waiting.close; done=[]
        def once():
            if not done: done.append(True);close()
        self.waiting.close=once
        async def journey():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://in-process') as client:
                    self.assertEqual((await client.get('/skeleton')).status_code,403)
                    headers={'authorization':'Bearer '+'x'*64}
                    status=(await client.get('/skeleton',headers=headers)).json()
                    self.assertTrue(status['ready_for_speech']);self.assertFalse(status['synthetic'])
                    caps=(await client.get('/v1/capabilities',headers=headers)).json()
                    self.assertEqual(caps['profiles'][0]['qualification_receipt'],self.profiles[0]['qualification_receipt'])
                    self.assertNotIn('voice_prepare',caps['operations'])
                    reply=await client.post('/v1/operations/speech_generate',headers=dict(headers,**{'idempotency-key':'activated'}),json=self.context['plan']['request'])
                    self.assertEqual(reply.status_code,200,reply.text)
                    job_id=reply.json()['job_id']
                    await asyncio.to_thread(self.wait,lambda:self.runtime.authority)
                    self.runtime.finish.set()
                    owner=app.state.dispatcher
                    job=await asyncio.to_thread(self.wait,lambda:(j if (j:=owner.call('job',job_id))['state']=='succeeded' else None))
                    artifact=await client.get('/v1/artifacts/'+job['artifact_ids'][0],headers=headers)
                    self.assertEqual(artifact.status_code,200,artifact.text)
                    self.assertEqual(len(artifact.json()['asset_ids']),1)
            self.assertFalse(app.state.dispatcher.thread.is_alive())
        asyncio.run(journey())

    def test_close_requests_stop_but_waits_for_receipt(self):
        owner=self.owner()
        self.wait(lambda:owner.call('status')['persistent_activation'])
        job=owner.call('production_generate','alice',self.context['plan']['request'],key='stop')
        self.wait(lambda:self.runtime.authority)
        owner.call('close')
        self.wait(self.runtime.stopped.is_set)
        self.assertTrue(owner.thread.is_alive())
        self.assertEqual(owner.call('status')['reservations'][-1]['state'],'executing')
        self.runtime.finish.set()
        owner.thread.join(4)
        self.assertFalse(owner.thread.is_alive())
        with sqlite3.connect(self.root/'core.sqlite') as db:
            self.assertEqual(db.execute('SELECT state FROM resource_reservations').fetchone()[0],'retained')


class PrivateInputs(unittest.TestCase):
    def setUp(self):
        import tempfile
        tmp=tempfile.TemporaryDirectory(dir='/private/tmp' if sys.platform == 'darwin' else '/tmp');self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.path=self.root/'credential'
        self.raw=encode(dict(version=1,credentials=[dict(token='s'*64,principal='alice',scopes=['read','production-write'])])).encode()
        self.path.write_bytes(self.raw);self.path.chmod(0o640)
        original=os.fstat
        def fstat(fd):
            s=original(fd)
            return SimpleNamespace(**dict({k:getattr(s,k) for k in dir(s) if k.startswith('st_')},st_uid=0,st_gid=10001))
        original_stat=os.stat
        def stat(path,*args,**kwargs):
            s=original_stat(path,*args,**kwargs)
            return SimpleNamespace(**dict({k:getattr(s,k) for k in dir(s) if k.startswith('st_')},st_uid=0,st_gid=10001))
        self.enterContext(patch.object(os,'fstat',side_effect=fstat))
        self.enterContext(patch.object(os,'stat',side_effect=stat))

    def test_explicit_credential_and_scope(self):
        resolver=installed_credentials(self.path,{'alice'})
        self.assertEqual(resolver.resolve('Bearer '+'s'*64,required_scope='production-write'),'alice')
        with self.assertRaises(ValueError):installed_credentials(self.path,{'bob'})

    def test_missing_default_symlink_hardlink_mode_refuse(self):
        with self.assertRaises(FileNotFoundError):installed_credentials(self.root/'missing',{'alice'})
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):installed_credentials(self.path,{'alice'})
        self.path.chmod(0o640)
        alias=self.root/'alias';alias.symlink_to(self.path)
        with self.assertRaises(OSError):installed_credentials(alias,{'alice'})
        alias.unlink();os.link(self.path,alias)
        with self.assertRaises(ValueError):installed_credentials(self.path,{'alice'})

    def test_no_secret_in_fixed_configuration_errors(self):
        self.path.write_text(self.raw.decode().replace('s'*64,'tiny'))
        with self.assertRaises(ValueError) as caught:installed_credentials(self.path,{'alice'})
        self.assertNotIn('tiny',str(caught.exception))



class SelectedRestart(unittest.TestCase):
    def test_reviewed_history_new_registration_activates_exact_selection(self):
        from tests.service import test_installed_restart as history
        from service import host_install as h
        fixture=history.ClosedRestart();fixture.setUp();self.addCleanup(fixture.doCleanups)
        history.TerminalRestart.terminal_fixture(fixture)
        fixture.restart();second=fixture.f.run_registration()
        self.assertNotEqual(second,fixture.first)
        value=json.loads((fixture.f.authority/('registration-'+second+'.json')).read_bytes())
        gate=e.projection(dict(version=1,registration_id=second,registration=value['registration'],
            configuration_digest=digest(fixture.f.config),enrollment_digest=digest(fixture.f.public),
            provenance_digest=digest(value),processes=value['observation']['processes']))
        delivered=[]
        h._installed_handoff(fixture.f.observer,fixture.f.owners,second,fixture.f.fd,lambda:None,
            notify=lambda *args:delivered.append(args))
        self.assertEqual(len(delivered),1)
        class Waiting:
            def wait(self):return copy.deepcopy(gate)
            read=wait
        runtime=Mock(spool=None);runtime.call.side_effect=lambda op,body:copy.deepcopy(fixture.f.hello)
        process=gate['processes'][0]
        incarnation=dict(start_ticks=process['start_ticks'],pid_namespace=process['pid_namespace'],cgroup='0::/')
        for name in ('getuid','geteuid','getgid','getegid'):
            self.enterContext(patch.object(os,name,return_value=10001))
        self.enterContext(patch.object(os,'getgroups',return_value=[]))
        self.enterContext(patch.object(e,'local_incarnation',return_value=incarnation))
        owner=Dispatcher(fixture.f.roots['control']/'core',staging=fixture.f.roots['staging'],runtime=runtime,
            profiles=fixture.f.config['profiles'],budget=fixture.f.config['budget'],production_contexts=Mock(),
            installed_activation=e.InstalledActivation(Waiting(),fixture.f.config))
        try:
            deadline=time.monotonic()+3
            while not owner.call('status')['reconciled'] and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(owner.call('status')['persistent_activation'],owner.call('status'))
            self.assertEqual(owner.installed_activation.checked['registration_id'],second)
            self.assertEqual(len(fixture.f.rows()),2)
            self.assertEqual(fixture.path.read_bytes(),fixture.raw)
        finally:owner.close()


if __name__=='__main__':unittest.main()
