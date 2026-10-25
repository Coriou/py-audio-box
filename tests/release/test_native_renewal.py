"""Stopped SOURCE preparation only. Tiny files/SQLite, injected TLS/processes."""
import copy
import os
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from service import native_renewal as r
from service import native_package as n
from service.core import Core, native_maintenance
from tests.release.test_native_history import NativeHistoryTests
from worker.transport import Refused, canonical
from worker.native_supervisor import Journal


class RenewalFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.root.chmod(0o700)

    def test_publish_exact_resume_and_conflict_retention(self):
        path=self.root/'document'
        r.publish(path,dict(value=1))
        r.publish(path,dict(value=1))
        saved=path.read_bytes()
        with self.assertRaises(Refused): r.publish(path,dict(value=True))
        self.assertEqual(path.read_bytes(),saved)

    def test_partial_stage_retained(self):
        path=self.root/'document'; stage=self.root/'document.stage'
        stage.write_bytes(b'{'); stage.chmod(0o600)
        with self.assertRaises(Refused): r.publish(path,dict(value=1))
        self.assertEqual(stage.read_bytes(),b'{'); self.assertFalse(path.exists())

    def test_exact_stage_resumes(self):
        path=self.root/'document'; stage=self.root/'document.stage'
        stage.write_bytes(canonical(dict(value=1))); stage.chmod(0o600)
        r.publish(path,dict(value=1))
        self.assertFalse(stage.exists()); self.assertEqual(r.document(path),dict(value=1))

    def test_rename_failure_preserves_owned_staging(self):
        path=self.root/'document'
        with patch.object(r.os,'rename',side_effect=OSError('injected rename')):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertEqual((self.root/'document.stage').read_bytes(),canonical(dict(value=1)))
        r.publish(path,dict(value=1))

    def test_fsync_failure_retains_then_explicit_resume(self):
        path=self.root/'document'
        with patch.object(r.os,'fsync',side_effect=OSError('injected fsync')):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertTrue((self.root/'document.stage').exists())
        r.publish(path,dict(value=1))

    def test_write_failure_keeps_partial_without_repair(self):
        path=self.root/'document'
        real=os.fdopen
        class FailedWrite:
            def __init__(self, stream): self.stream=stream
            def __enter__(self): return self
            def __exit__(self,*args): self.stream.close()
            def write(self,raw): raise OSError('injected write')
        with patch.object(r.os,'fdopen',side_effect=lambda *a,**k:FailedWrite(real(*a,**k))):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertEqual((self.root/'document.stage').read_bytes(),b'')
        with self.assertRaises(Refused): r.publish(path,dict(value=1))

    def test_visible_final_fsync_failure_explicit_resume(self):
        path=self.root/'document';real=os.fsync;calls=0
        def fail(fd):
            nonlocal calls
            calls+=1
            if calls==3: raise OSError('injected final fsync')
            return real(fd)
        with patch.object(r.os,'fsync',side_effect=fail):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertTrue(path.exists());self.assertFalse((self.root/'document.stage').exists())
        r.publish(path,dict(value=1))

    def test_symlink_hardlink_and_duplicate_keys(self):
        source=self.root/'source'; source.write_bytes(b'{"a":1,"a":1}');source.chmod(0o600)
        with self.assertRaises(Refused): r.document(source)
        path=self.root/'document';path.symlink_to(source)
        with self.assertRaises(OSError) as raised: r.publish(path,dict(value=1))
        import errno
        self.assertEqual(raised.exception.errno,errno.ELOOP)
        self.assertTrue(path.is_symlink())
        self.assertEqual(source.read_bytes(),b'{"a":1,"a":1}')
        path.unlink();os.link(source,path)
        with self.assertRaises(Refused): r.publish(path,dict(value=1))

    def test_empty_unknown_namespace_blocks_core_before_schema(self):
        owner=self.root/'owner'; core=Core(owner);core.close()
        directory=owner/'native-renewal';directory.mkdir(mode=0o700)
        with self.assertRaises(Refused): Core(owner)
        marker=directory/'foreign';marker.write_bytes(b'RETAIN')
        with self.assertRaises(Refused): Core(owner)
        self.assertEqual(marker.read_bytes(),b'RETAIN')

    def test_maintenance_same_registry_readonly_and_no_second_core(self):
        owner=self.root/'owner';core=Core(owner)
        with self.assertRaisesRegex(ValueError,'control_busy'):
            with native_maintenance(owner): pass
        core.close()
        with native_maintenance(owner) as view:
            with self.assertRaises(sqlite3.OperationalError): view.db.execute('CREATE TABLE forbidden(x)')
            with self.assertRaisesRegex(ValueError,'control_busy'): Core(owner)
        core=Core(owner);core.close()

    def test_partial_retirement_blocks_journal(self):
        root=self.root/'host';root.mkdir(mode=0o700)
        journal=Journal(root,'a'*32,'b'*64);journal.lock.close()
        marker=root/'retirement.stage';marker.write_bytes(b'{');marker.chmod(0o600)
        with self.assertRaisesRegex(Refused,'native_root_retired'): Journal(root,'a'*32,'b'*64)
        self.assertEqual(marker.read_bytes(),b'{')

    def test_capacity_actionable_and_canonical_types(self):
        r.capacity('history',62,63)
        with self.assertRaisesRegex(Refused,'native_retention_capacity.*used=63.*retain owner'):
            r.capacity('history',63,63)
        with self.assertRaises(Refused): r.capacity('history',True,63)
        with self.assertRaises(Refused): r.capacity('history',-1,63)


class RenewalHistoryTests(unittest.TestCase):
    setUp = NativeHistoryTests.setUp
    prepare = NativeHistoryTests.prepare
    wait = NativeHistoryTests.wait
    open = NativeHistoryTests.open
    close = NativeHistoryTests.close

    def begin(self):
        self.intent = self.root/'native-renewal'/'001.intent'
        return r.begin(self.record,configuration_file=self.kw['configuration_file'],
            templates_file=self.kw['templates_file'],owner_root=self.root,request_id='e'*32)

    def retire(self):
        self.f.s.journal.lock.close()
        return r.retire(self.intent,host_root=self.f.s.journal.root)

    def test_stopped_offline_success_seal_and_closed_commit(self):
        self.prepare(); old=self.record.read_bytes()
        intent=self.begin();self.assertEqual(self.begin(),intent)
        export=self.retire();self.assertEqual(self.retire(),export)
        self.kw['connector']=lambda _: (_ for _ in ()).throw(AssertionError('old TLS unavailable'))
        with patch.object(n.p,'tls_context',side_effect=AssertionError('no old private key required')):
            seal=r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export')
            self.assertEqual(r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export'),seal)
        self.assertEqual(self.record.read_bytes(),old)
        with self.assertRaisesRegex(Refused,'native_renewal_pending'): n.native_startup_inputs(self.record,**self.kw)
        with self.assertRaisesRegex(Refused,'native_root_retired'):
            self.f.s.dispatch(dict(op='hello'))

    def test_live_host_lock_refuses_retirement_without_marker(self):
        self.prepare();self.begin()
        with self.assertRaises(BlockingIOError): r.retire(self.intent,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_changed_request_cannot_fork_pending(self):
        self.prepare();self.begin();saved=self.intent.read_bytes()
        with self.assertRaises(Refused):
            r.begin(self.record,configuration_file=self.kw['configuration_file'],
                templates_file=self.kw['templates_file'],owner_root=self.root,request_id='f'*32)
        self.assertEqual(self.intent.read_bytes(),saved)

    def test_unacked_record_refuses_without_retirement(self):
        self.prepare();self.begin()
        record=self.f.s.journal.value['records'][0]
        self.f.s.persist(record,acked=False)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(self.intent,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_unknown_host_evidence_retained(self):
        self.prepare();self.begin();root=self.f.s.journal.root
        file=root/'native.tmp';file.write_bytes(b'RETAIN');file.chmod(0o600)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(self.intent,host_root=root)
        self.assertEqual(file.read_bytes(),b'RETAIN')

    def test_altered_membership_refuses(self):
        self.prepare();intent=self.begin();intent['membership']=[]
        path=self.intent.parent/'forged';path.write_bytes(canonical(intent));path.chmod(0o600)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(path,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_changed_retained_media_refuses_seal(self):
        self.prepare();self.begin();self.retire()
        file=next((self.root/'assets').iterdir());raw=file.read_bytes();file.write_bytes(raw+b'RETAIN')
        with self.assertRaises(Exception): r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export')
        self.assertFalse((self.intent.parent/'001.seal').exists())
        self.assertEqual(file.read_bytes(),raw+b'RETAIN')

    def test_canonical_compatibility_refuses_profile_and_types(self):
        self.prepare()
        config=n.document(self.kw['configuration_file']);templates=n.document(self.kw['templates_file'])
        for field,value in (('version',True),('version',1.0)):
            changed=copy.deepcopy(config);changed[field]=value
            with self.assertRaises(Refused): r.compatible(config,templates,changed,templates)
        changed=copy.deepcopy(templates);changed['profiles'][0]['limits']['disk_bytes']+=1
        with self.assertRaises(Refused): r.compatible(config,templates,config,changed)
        changed=copy.deepcopy(config);changed['native']['runtime_manifest_sha256']='f'*64
        with self.assertRaises(Refused): r.compatible(config,templates,changed,templates)


class RenewalNegativeHistoryTests(RenewalHistoryTests):
    terminal_mode='cancelled'
    # Select only the offline proof method explicitly; inherited loader is unused.


class RenewalFailedHistoryTests(RenewalHistoryTests):
    terminal_mode='failed'


class RenewalCompletionTests(RenewalHistoryTests):
    """Fixed explicit version only; Lot B public origin authorization is absent."""
    def successor_inputs(self, epoch):
        from datetime import datetime, timezone
        base=Path(self.kw['configuration_file']).parent
        config=n.document(self.kw['configuration_file'])
        templates=n.document(self.kw['templates_file'])
        native=dict(config['native'],configuration_sha256=f'{epoch:064x}',lane=f'{epoch:032x}',boot=f'{epoch+8:032x}')
        config['native']=native
        for key in ('cafile','certfile','keyfile'):
            path=base/(key+str(epoch));path.write_bytes(b'INERT-NEW-'+str(epoch).encode());path.chmod(0o600)
            config[key]=str(path)
        self.f.cert=b'fixture-server-certificate-'+str(epoch).encode()
        import hashlib
        self.f.pin=hashlib.sha256(self.f.cert).hexdigest();config['peer_sha256']=self.f.pin
        templates['inventory']['devices'][0]['probed_at']=datetime.now(timezone.utc).isoformat()
        for row in templates['registrations']:
            row['context']['execution']['native']=native.copy()
            row['context']['execution']['hardware']=copy.deepcopy(templates['inventory'])
        paths={}
        for key,value in (('configuration_file',config),('templates_file',templates)):
            path=base/(key+str(epoch));path.write_bytes(canonical(value));path.chmod(0o600);paths[key]=str(path)
        return paths,config,templates

    def renew(self, epoch, *, fault=None):
        from worker.native_supervisor import NativeSupervisor
        from worker.engines.qwen_execution import inventory_from_wire
        paths,config,templates=self.successor_inputs(epoch)
        intent=r.begin(self.record,**paths,owner_root=str(self.root),request_id=str(epoch)*32)
        intent_path=self.root/'native-renewal'/f'{epoch:03d}.intent'
        old=self.f.s;old.journal.lock.close()
        r.retire(intent_path,host_root=old.journal.root)
        export_file=old.journal.root/'retirement-export'
        host=old.journal.root.parent/('successor'+str(epoch));host.mkdir(mode=0o700)
        r.provision(intent_path,export_file=export_file,host_root=host)
        registered=copy.deepcopy(old._preset_templates)
        for reg in registered.values():
            reg['snapshot'][1]=copy.deepcopy(templates['registrations'][0]['context'])
            reg['inventory']=inventory_from_wire(templates['inventory'])
        self.f.f.h=type(self.f.f.h)()
        successor=NativeSupervisor(host,deployment_id=config['native']['deployment_id'],
            configuration_sha256=config['native']['configuration_sha256'],peer_pin=old.pin,
            runtime_manifest_sha256=old.manifest,registrations=registered,python=old.python,child_entry=old.entry,
            popen=old.popen,pair=old.pair,signal_group=old.signal_group,identity=old.identity,
            clock=old.clock,utc=old.utc,free_bytes=old.free_bytes,free_ram=old.free_ram,availability_provider=old.availability.provider)
        self.addCleanup(successor.journal.lock.close)
        self.f.s=successor;self.f.native=config['native'];self.f.before=lambda q:None
        if fault: fault(intent_path,export_file)
        result=r.commit(intent_path,export_file=export_file,connector=self.f.connect)
        self.assertEqual(r.commit(intent_path,export_file=export_file,connector=self.f.connect),result)
        self.record=intent_path.with_suffix('.enrollment')
        self.kw.update(paths)
        return old,successor

    def reopen(self):
        from service.native_activation import NativeActivation
        from service.dispatcher import Dispatcher
        self.activation=NativeActivation(self.record,**self.kw)
        self.dispatcher=Dispatcher(self.root,staging=self.f.stage,installed_activation=self.activation)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        return self.dispatcher

    def next_job(self, key):
        with patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):
            try:
                d=self.reopen()
                self.f.before=lambda q:self.f.finish() if q['op'] in ('observe','collect') and self.f.s.active else None
                result=d.call('production_generate','alice',self.context['plan']['request'],key=key)
                job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state'] in ('succeeded','failed') else None))
                self.assertEqual(job['state'],'succeeded',job)
                self.wait(lambda:d.call('status')['phase']=='idle')
                return job
            finally:self.close()

    def test_two_epochs_offline_old_history_new_allocation_and_stale_path(self):
        self.prepare();original=self.record;raw=original.read_bytes();old_job=self.job
        old,first=self.renew(1)
        with self.assertRaisesRegex(Refused,'stale_enrollment'):
            n.native_startup_inputs(original,**self.kw)
        self.next_job('epoch-one')
        old2,second=self.renew(2)
        self.next_job('epoch-two')
        self.assertEqual(original.read_bytes(),raw)
        self.assertEqual(second.journal.retained['charges']['journal_records'],2)
        self.assertEqual(second.journal.retained['charges']['allocation_registrations'],3)
        try:
            d=self.reopen()
            self.assertEqual(d.call('job',old_job['id']),old_job)
            self.assertEqual(d.call('production_generate','alice',self.context['plan']['request'],key='retained')['job_id'],old_job['id'])
            self.assertEqual(second.launches,1)
        finally:self.close()
        self.assertTrue((old.journal.root/'retirement').exists())
        self.assertTrue((old2.journal.root/'retirement').exists())

    def intent_fault(self, epoch, operation):
        self.prepare()
        if epoch == 2: self.renew(1)
        paths,_,_=self.successor_inputs(epoch)
        args=dict(**paths,owner_root=str(self.root),request_id=str(epoch)*32)
        final=self.root/'native-renewal'/f'{epoch:03d}.intent'
        stage=final.with_name(final.name+'.stage')
        real_rename=r.os.rename;real_fsync=r.os.fsync
        def rename(a,b):
            if Path(b)==final: raise OSError('injected intent rename')
            return real_rename(a,b)
        def fsync(fd):
            target=stage if operation=='stage-fsync' else final
            if target.exists() and os.fstat(fd).st_ino==target.stat().st_ino:
                raise OSError('injected intent fsync')
            return real_fsync(fd)
        target=patch.object(r.os,'rename',side_effect=rename) if operation=='rename' else patch.object(r.os,'fsync',side_effect=fsync)
        with target:
            with self.assertRaisesRegex(OSError,'injected intent'): r.begin(self.record,**args)
        retained=stage if stage.exists() else final
        before=retained.read_bytes()
        self.assertEqual(r.document(retained)['request_id'],args['request_id'])
        with self.assertRaises(Refused): r.selected_source(self.record,self.root)
        with self.assertRaises(Refused): Core(self.root)
        self.assertEqual(retained.read_bytes(),before)
        return args,final,stage,before

    def resume_intent_fault(self, epoch, operation):
        args,final,stage,before=self.intent_fault(epoch,operation)
        synced=[];real=r.os.fsync
        def fsync(fd):
            synced.append(os.fstat(fd).st_ino);return real(fd)
        with patch.object(r.os,'fsync',side_effect=fsync):
            value=r.begin(self.record,**args)
        self.assertFalse(stage.exists());self.assertEqual(final.read_bytes(),before)
        self.assertEqual(value,r.document(final))
        self.assertIn(final.stat().st_ino,synced)
        self.assertIn(final.parent.stat().st_ino,synced)
        self.assertEqual(r.begin(self.record,**args),value)

    def test_first_intent_rename_resumes(self): self.resume_intent_fault(1,'rename')
    def test_later_intent_rename_resumes(self): self.resume_intent_fault(2,'rename')
    def test_first_intent_stage_fsync_resumes(self): self.resume_intent_fault(1,'stage-fsync')
    def test_later_intent_stage_fsync_resumes(self): self.resume_intent_fault(2,'stage-fsync')
    def test_first_intent_final_fsync_resumes(self): self.resume_intent_fault(1,'final-fsync')
    def test_later_intent_final_fsync_resumes(self): self.resume_intent_fault(2,'final-fsync')

    def refuse_intent_changes(self, epoch):
        args,final,stage,before=self.intent_fault(epoch,'rename')
        directory=final.parent
        def evidence():
            return {str(f.relative_to(self.root)):f.read_bytes() for f in self.root.rglob('*') if f.is_file()}
        def refused(call):
            saved=evidence()
            with self.assertRaises((Refused,ValueError)): call()
            self.assertEqual(evidence(),saved)
            self.assertFalse(final.exists())
        refused(lambda:r.begin(self.record,**dict(args,request_id='f'*32)))
        # A changed explicit boot in both supplied documents must not retarget intent.
        config_path=Path(args['configuration_file']);templates_path=Path(args['templates_file'])
        config_raw=config_path.read_bytes();templates_raw=templates_path.read_bytes()
        config=n.document(config_path);templates=n.document(templates_path)
        config['native']['boot']='a'*32
        for row in templates['registrations']: row['context']['execution']['native']['boot']='a'*32
        config_path.write_bytes(canonical(config));templates_path.write_bytes(canonical(templates))
        refused(lambda:r.begin(self.record,**args))
        config_path.write_bytes(config_raw);templates_path.write_bytes(templates_raw)
        # Changed payload, truncated document, foreign request and forked epoch stay retained.
        original=r.document(stage)
        for change in ('payload','partial','request','fork'):
            value=copy.deepcopy(original)
            if change=='payload': value['stopped_sha256']='0'*64
            if change=='request': value['request_id']='f'*32
            if change=='fork': value['epoch']=epoch+1
            stage.write_bytes(b'{' if change=='partial' else canonical(value))
            refused(lambda:r.begin(self.record,**args))
        stage.write_bytes(before)
        # A changed stopped database fingerprint is not silently resealed.
        with patch.object(r,'stopped_fingerprint',return_value='0'*64):
            refused(lambda:r.begin(self.record,**args))
        self.assertEqual(r.begin(self.record,**args),original)

    def test_first_staged_intent_changed_inputs_retained(self): self.refuse_intent_changes(1)
    def test_later_staged_intent_changed_inputs_retained(self): self.refuse_intent_changes(2)

    def test_commit_rename_failure_resumes_exact_owned_phase(self):
        self.prepare()
        def fault(intent,export):
            real=r.os.rename
            def rename(a,b):
                if str(b).endswith('.commit'): raise OSError('injected commit rename')
                return real(a,b)
            with patch.object(r.os,'rename',side_effect=rename):
                with self.assertRaisesRegex(OSError,'injected'):r.commit(intent,export_file=export,connector=self.f.connect)
            self.assertTrue(intent.with_suffix('.commit.stage').exists())
        self.renew(1,fault=fault)
        self.next_job('after-resume')

    def test_second_boot_during_commit_refuses_without_commit(self):
        self.prepare()
        def fault(intent,export):
            calls=0
            def mutate(q,v):
                nonlocal calls
                if q['op']=='hello':
                    calls+=1
                    if calls>=3: v['value']['boot']='f'*32
                return v
            self.f.mutate=mutate
            with self.assertRaises(Refused):r.commit(intent,export_file=export,connector=self.f.connect)
            self.assertFalse(intent.with_suffix('.commit').exists())
            self.f.mutate=lambda q,v:v
        self.renew(1,fault=fault)
        self.next_job('after-explicit-resume')

    def test_catalogue_alias_change_does_not_invalidate_terminal_seal(self):
        self.prepare();self.renew(1)
        try:
            d=self.reopen()
            version=self.context['plan']['voice_versions'][0]
            # Normal public mutation and cursor read extend mutable catalogue data.
            voice=d.call('voice_get','alice',version['voice_id'])
            d.call('production_aliases','alice',dict(voice_id=version['voice_id'],aliases=['renewed-alias'],expected_revision=voice['revision']),key='alias')
            d.call('voice_list','alice',limit=20)
        finally:self.close()
        self.next_job('after-alias')

    def test_host_cumulative_limits_and_no_restart_identity_reuse(self):
        self.prepare();old,new=self.renew(1)
        retained=copy.deepcopy(new.journal.retained)
        for key,limit in (('journal_records',63),('allocation_registrations',63),('host_bytes',64*1024**2),('journal_bytes',1048576)):
            value=copy.deepcopy(retained);value['charges'][key]=limit
            with self.assertRaisesRegex(Refused,'native_retention_capacity'):r.validate_host_predecessor(value)
            value['charges'][key]=True
            with self.assertRaises(Refused):r.validate_host_predecessor(value)
        self.assertEqual(new.journal.retained,retained)
        new.journal.lock.close()
        from worker.native_supervisor import NativeSupervisor
        with self.assertRaisesRegex(Refused,'new_boot_required'):
            NativeSupervisor(new.journal.root,deployment_id=new.journal.deployment,configuration_sha256=new.journal.configuration,
                peer_pin=new.pin,runtime_manifest_sha256=new.manifest,registrations=new._preset_templates,
                python=new.python,child_entry=new.entry)

    def test_sealed_row_change_and_unknown_namespace_refuse_without_repair(self):
        self.prepare();self.renew(1)
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute("UPDATE jobs SET fingerprint='altered'")
        with self.assertRaises(Refused):self.reopen()
        with sqlite3.connect(self.root/'core.sqlite') as db:
            self.assertEqual(db.execute('SELECT fingerprint FROM jobs').fetchone()[0],'altered')
        file=self.root/'native-renewal'/'foreign';file.write_bytes(b'RETAIN');file.chmod(0o600)
        with self.assertRaises(Refused):r.namespace(self.root)
        self.assertEqual(file.read_bytes(),b'RETAIN')

    def phase_fault(self, suffix, operation):
        self.prepare()
        def fault(intent,export):
            if operation=='rename':
                real=r.os.rename
                def rename(a,b):
                    if str(b).endswith('.'+suffix):raise OSError('injected '+suffix)
                    return real(a,b)
                target=patch.object(r.os,'rename',side_effect=rename)
            else:
                real=r.os.fsync
                # Failure after the named final rename is visible; explicit resume
                # must re-fsync it. No host crash or real child is involved.
                def fsync(fd):
                    if intent.with_suffix('.'+suffix).exists():raise OSError('injected '+suffix)
                    return real(fd)
                target=patch.object(r.os,'fsync',side_effect=fsync)
            with target:
                with self.assertRaisesRegex(OSError,'injected'):r.commit(intent,export_file=export,connector=self.f.connect)
        self.renew(1,fault=fault)
        self.next_job('phase-resumed')

    def test_enrollment_rename_resumes(self):self.phase_fault('enrollment','rename')
    def test_seal_final_fsync_resumes(self):self.phase_fault('seal','fsync')
    def test_commit_final_fsync_resumes(self):self.phase_fault('commit','fsync')

    def test_eight_total_epochs_and_ninth_refuses_before_intent(self):
        self.prepare()
        for epoch in range(1,8):self.renew(epoch)
        cert,pin=self.f.cert,self.f.pin
        paths,_,_=self.successor_inputs(8)
        self.f.cert,self.f.pin=cert,pin
        with self.assertRaisesRegex(Refused,'native_retention_capacity.*epochs'):
            r.begin(self.record,**paths,owner_root=str(self.root),request_id='8'*32)
        self.assertEqual(len([k for k in r.namespace(self.root) if k.endswith('.commit')]),7)
        self.assertFalse((self.root/'native-renewal'/'008.intent').exists())
        self.next_job('last-epoch')

    def test_commit_replay_after_later_job_does_not_reseal_database(self):
        self.prepare();old,new=self.renew(1)
        path=self.root/'native-renewal'/'001.intent'
        committed=path.with_suffix('.commit').read_bytes()
        self.next_job('later')
        self.assertEqual(canonical(r.commit(path,export_file=old.journal.root/'retirement-export')),committed)


    def test_control_spool_capacity_refuses_before_queued_job_and_replay_is_free(self):
        self.prepare();self.renew(1)
        try:
            d=self.reopen()
            before=d.call('jobs','alice')
            with patch('service.native_runtime.QUOTA',1):
                # Public replay precedes context/admission and spends nothing.
                self.assertEqual(d.call('production_generate','alice',self.context['plan']['request'],key='retained')['job_id'],self.job['id'])
                with self.assertRaises(Refused):d.call('production_generate','alice',self.context['plan']['request'],key='capacity-refused')
            self.assertEqual(d.call('jobs','alice'),before)
            self.assertEqual(self.f.s.launches,0)
        finally:self.close()


    def test_current_peer_inventory_mismatch_refuses_activation(self):
        self.prepare();self.renew(1)
        def mutate(q,v):
            if q['op']=='hello':v['value']['inventory']['host']['architecture']='x86_64'
            return v
        self.f.mutate=mutate
        with self.assertRaisesRegex(Refused,'native_activation_hello'):self.reopen()
        self.assertEqual(self.f.s.launches,0)



class RenewalCompletionCancelledTests(RenewalCompletionTests):
    terminal_mode='cancelled'


class RenewalCompletionFailedTests(RenewalCompletionTests):
    terminal_mode='failed'
