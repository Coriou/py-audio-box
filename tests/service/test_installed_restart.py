"""Installed history only; disposable DB/provenance and injected host/UID/hello."""
import copy
import json
import unittest
from service import recovery, host_install as h
from service.core import encode
from service.persistent_init import digest
from tests.service import test_installed_registration as fixtures


class Restart(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.Registration();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.first=self.f.run_registration()
        self.oldrows=self.f.rows()
        self.path=self.f.authority/('registration-'+self.first+'.json')
        self.raw=self.path.read_bytes();self.inode=self.path.stat().st_ino

    def restart(self):
        for c in self.f.owners:c['started']='2026-10-24T02:00:00Z'
        self.f.hello['boot']='c'*32
        self.f.observer.current_owners=lambda claims:[dict(c,started='2026-10-24T02:00:00Z') for c in claims]

    def test_restart_retains_exact_history_and_retry(self):
        self.restart();second=self.f.run_registration()
        self.assertNotEqual(second,self.first)
        self.assertEqual(self.f.rows()[:1],self.oldrows)
        self.assertEqual(self.path.read_bytes(),self.raw);self.assertEqual(self.path.stat().st_ino,self.inode)
        saved=(self.f.authority/('registration-'+second+'.json')).read_bytes()
        self.assertEqual(json.loads(saved)['kind'],'installed-registration-v2')
        self.assertEqual(self.f.run_registration(),second)
        self.assertEqual((self.f.authority/('registration-'+second+'.json')).read_bytes(),saved)
        with self.assertRaisesRegex(ValueError,'host_ambiguous_incarnation'):
            recovery.host_registration_union([json.loads(r[1]) for r in self.f.rows()],self.f.scope,self.f.public['deployment_id'])

    def test_restart_exact_interruption_retries(self):
        self.restart()
        for point in ('provenance','insert','bracket'):
            with self.subTest(point=point):
                def fault(where):
                    if where==point:raise OSError('restart interruption')
                with self.assertRaisesRegex(OSError,'restart interruption'):self.f.run_registration(fault=fault)
                rid=self.f.run_registration()
                self.assertEqual(len(self.f.rows()),2)
                self.assertEqual(self.path.read_bytes(),self.raw)
                self.assertEqual(self.f.run_registration(),rid)

    def test_lane_and_history_exhaustion(self):
        registration=json.loads(self.oldrows[0][1]);other=copy.deepcopy(registration)
        other['lane']='d'*32
        with self.assertRaisesRegex(ValueError,'history_lane'):
            recovery.installed_history_current([registration,other],self.f.scope,self.f.public['deployment_id'],[registration])
        with self.assertRaisesRegex(ValueError,'registration_budget'):
            recovery.installed_history_current([registration]*(recovery.HOST_OWNER_LIMIT+1),self.f.scope,self.f.public['deployment_id'],[registration])
        state=h._registration_snapshot(self.f.roots,self.f.scope)
        with self.assertRaisesRegex(ValueError,'history_unclean'):
            recovery.installed_clean_history(state,dict(version=1,lane=self.f.lane,records=[{}]*64))

    def test_missing_provenance(self):
        self.path.unlink();self.restart()
        with self.assertRaisesRegex(ValueError,'missing_provenance'):self.f.run_registration()
        self.assertEqual(self.f.rows(),self.oldrows)

    def test_stale_selected(self):
        self.restart();self.f.owners[0]['started']='2026-10-24T01:00:00Z'
        with self.assertRaisesRegex(ValueError,'selected_not_current'):self.f.run_registration()

    def test_reused_boot(self):
        self.restart();self.f.hello['boot']='b'*32
        with self.assertRaisesRegex(ValueError,'worker_boot'):self.f.run_registration()

    def test_changed_image(self):
        self.restart();self.f.owners[0]['image']='sha256:'+'a'*64
        with self.assertRaisesRegex(ValueError,'history_binding'):self.f.run_registration()

    def test_unregistered_orphan_cannot_be_superseded(self):
        self.restart()
        row=json.loads(self.raw);row['registration']['boot']='e'*32;row['hello']['boot']='e'*32
        path=self.f.authority/('registration-'+digest(row['registration'])+'.json')
        path.write_text(encode(row));path.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'unregistered_provenance'):self.f.run_registration()

    def test_clean_state_negatives(self):
        state=h._registration_snapshot(self.f.roots,self.f.scope)
        journal=json.loads((self.f.roots['receipts']/'journal.json').read_bytes())['payload']
        for field,value in [('faults',['unknown']),('allocation_coherent',False),
                            ('reservations',[dict(state='executing')]),
                            ('jobs',[dict(state='running',attempts=[])]),
                            ('dispatches',[dict(acked=False,receipt_digest=None)])]:
            with self.subTest(field=field):
                dirty=copy.deepcopy(state);dirty[field]=value
                with self.assertRaises(ValueError):recovery.installed_clean_history(dirty,journal)

class TerminalRestart(unittest.TestCase):
    setUp=Restart.setUp
    restart=Restart.restart
    def terminal_fixture(self):
        from contextlib import ExitStack
        from pathlib import Path
        from unittest.mock import patch
        import os
        from service.core import Core
        from service.admission import Admission
        from service.dispatch_store import DispatchStore
        from worker.runtime_journal import journal_bytes
        f=self.f
        transcript=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_bytes())
        for profile in transcript['profiles']:
            profile.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='E1 synthetic only')
        with ExitStack() as stack:
            stack.enter_context(patch.object(os,'getuid',return_value=10001))
            core=Core(f.roots['control']/'core');stack.callback(core.close)
            admission=Admission(core,f.roots['staging'])
            job=core.submit_execution('p','terminal-restart',transcript['context'],_admission=admission,
                _profiles=transcript['profiles'],_dispatch_settings=('silence',8))
            store=DispatchStore(core,admission);row=store.allocate(transcript['profiles'],'silence',8)
            authority=row['authority'];runtime=f.lane+'/'+f.hello['boot']+'/'+'a'*32
            receipt=dict(runtime_id=runtime,binding=authority.binding,reaped=True,timely=False,
                original_deadline=authority.job['deadline'],engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=0)
            job=core.job(job['id']);job['state']='failed';job['stage']='failed';job['attempts'][-1]['reaped']=True
            core._save(job)
            core.db.execute("UPDATE resource_reservations SET state='retained'")
            store.update(row,runtime=runtime,phase='acked',receipt=encode(receipt),outcome=encode(job),acked=1)
            journal=dict(version=1,lane=f.lane,records=[dict(id=runtime,allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='silence'),
                phase='terminal',acked=True,uid=20000,boot=f.hello['boot'],receipt=receipt,proposal=None)])
            (f.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
        return journal

    def test_terminal_confirmed_reap_restart(self):
        self.terminal_fixture();self.restart();second=self.f.run_registration()
        self.assertNotEqual(second,self.first);self.assertEqual(self.path.read_bytes(),self.raw)

    def test_stale_ack_refuses(self):
        from worker.runtime_journal import journal_bytes
        journal=self.terminal_fixture();journal['records'][0]['acked']=False
        (self.f.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal));self.restart()
        with self.assertRaisesRegex(ValueError,'unconfirmed_reap_ack'):self.f.run_registration()

    def test_dirty_old_attempt_refuses(self):
        from worker.runtime_journal import journal_bytes
        journal=self.terminal_fixture();row=journal['records'][0]
        row.update(phase='started',acked=False);row.pop('receipt');row.pop('proposal')
        (self.f.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal));self.restart()
        with self.assertRaisesRegex(ValueError,'unhealthy_state'):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_unknown_owner_refuses(self):
        import sqlite3
        self.terminal_fixture()
        with sqlite3.connect(self.f.roots['control']/'core/core.sqlite') as db:
            job=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0])
            job['attempts'][-1]['owner_id']='owner-unknown'
            db.execute('UPDATE jobs SET job=?',(encode(job),))
        self.restart()
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_uid_reuse_refuses(self):
        from worker.runtime_journal import journal_bytes
        journal=self.terminal_fixture();row=journal['records'][0]
        row['uid']=row['receipt']['engine_uid']=20001
        (self.f.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal));self.restart()
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)


class ClosedRestart(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.f=fixtures.Registration();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.host=self.f.use_closed_observer()
        self.enterContext(patch.object(h.sys,'platform','linux'))
        self.first=self.f.run_registration()
        self.path=self.f.authority/('registration-'+self.first+'.json')
        self.raw=self.path.read_bytes()

    def restart(self):
        f=self.f
        for owner,container in zip(f.owners,self.host.containers):
            owner['started']=container['started']='2026-10-24T02:00:00Z'
            container['pid']+=100
        f.observer.proc=lambda pid,identity,ipc:dict(pid=pid,start_ticks=200,pid_namespace=pid,cgroup='/docker/'+identity)
        f.hello['boot']='c'*32

    def test_real_provenance_new_startup_and_handoff(self):
        from service import install_policy as ip
        from service.entrypoint import projection
        TerminalRestart.terminal_fixture(self)
        journal=(self.f.roots['receipts']/'journal.json').read_bytes()
        oldrows=self.f.rows()
        self.restart();second=self.f.run_registration();self.assertNotEqual(second,self.first)
        self.assertEqual(self.path.read_bytes(),self.raw)
        self.assertEqual(self.f.rows()[:1],oldrows)
        self.assertEqual((self.f.roots['receipts']/'journal.json').read_bytes(),journal)
        value=json.loads((self.f.authority/('registration-'+second+'.json')).read_bytes())
        record=value['registration']
        gate=projection(dict(version=1,registration_id=second,registration=record,
            configuration_digest=digest(self.f.config),enrollment_digest=digest(self.f.public),
            provenance_digest=digest(value),processes=value['observation']['processes']))
        process=gate['processes'][0]
        result=recovery.installed_startup_read(self.f.roots['control']/'core',self.f.config,gate,self.f.hello,
            dict(start_ticks=process['start_ticks'],pid_namespace=process['pid_namespace'],cgroup='0::/'),policy=ip.SPLIT)
        self.assertEqual(result['registration_id'],second)
        delivered=[]
        h._installed_handoff(self.f.observer,self.f.owners,second,self.f.fd,lambda:None,
            notify=lambda *args:delivered.append(args))
        self.assertEqual(len(delivered),1)
        self.assertEqual(self.f.run_registration(),second)

    def test_missing_retained_container(self):
        self.restart();self.host.containers.pop()
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_changed_volume(self):
        self.restart();self.host.containers[0]['mounts'][0]['source']+='/changed'
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_changed_config(self):
        self.restart();self.f.observer.config['images']['worker']='sha256:'+'f'*64
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_process_must_be_new(self):
        self.restart()
        self.f.observer.proc=lambda pid,identity,ipc:dict(pid=pid,start_ticks=100,pid_namespace=pid,cgroup='/docker/'+identity)
        with self.assertRaisesRegex(ValueError,'process_reused'):self.f.run_registration()

    def test_unknown_consumer(self):
        self.restart()
        unknown=copy.deepcopy(self.host.containers[0]);unknown['id']='f'*64
        self.host.containers.append(unknown)
        with self.assertRaises(ValueError):self.f.run_registration()
        self.assertEqual(len(self.f.rows()),1)

    def test_old_provenance_missing_after_restart(self):
        self.restart();self.f.run_registration();self.path.unlink()
        with self.assertRaisesRegex(ValueError,'missing_provenance'):self.f.run_registration()

    def test_closed_history_rejects_unknown_shape(self):
        self.restart();rid=self.f.run_registration()
        path=self.f.authority/('registration-'+rid+'.json');value=json.loads(path.read_bytes())
        value['history']['fake_actual_inspection']=True;path.write_text(encode(value))
        with self.assertRaisesRegex(ValueError,'unknown_record_fields'):self.f.run_registration()


class HistoricalValidation(unittest.TestCase):
    setUp=ClosedRestart.setUp
    restart=ClosedRestart.restart
    # Inherit the actual closed reader, never the permissive registration mock.
    def history_fixture(self, terminal=True):
        if terminal: TerminalRestart.terminal_fixture(self)
        self.restart();rid=self.f.run_registration()
        path=self.f.authority/('registration-'+rid+'.json')
        return rid,path,json.loads(path.read_bytes())

    def reject_both(self,rid):
        from unittest.mock import patch
        from jsonschema.exceptions import ValidationError
        oldrows=self.f.rows();delivered=[]
        original=self.f.child
        def child(kind,*args):
            self.assertNotEqual(kind,'writer')
            return original(kind,*args)
        with patch.object(self.f,'child',child):
            with self.assertRaises((ValueError,ValidationError)): self.f.run_registration()
        with self.assertRaises((ValueError,ValidationError)):
            h._installed_handoff(self.f.observer,self.f.owners,rid,self.f.fd,lambda:None,
                notify=lambda *args:delivered.append(args))
        self.assertEqual(delivered,[]);self.assertEqual(self.f.rows(),oldrows)

    def test_review_nested_examples_retry_and_handoff(self):
        rid,path,value=self.history_fixture();original=copy.deepcopy(value)
        def unknown(v):v['history']['journal']['records'][0]['invented_owner_claim']='accepted'
        def authority(v):
            r=v['history']['journal']['records'][0];r['allocation']['snapshot']=['not an authority'];r['boot']='f'*32
        def schema(v):
            source=v['history']['state']['evidence_projection'];source['schema']=[]
            source['data']['p1_recovery_plans']=[['invented pending maintenance']]
        def schema_only(v):v['history']['state']['evidence_projection']['schema']=[]
        def pending_only(v):v['history']['state']['evidence_projection']['data']['p1_recovery_plans']=[['invented pending maintenance']]
        def hidden_monitor(v):v['history']['state']['monitor']=[[1,'invented']]
        def foreign(v):v['history']['journal']['records'][0]['boot']='f'*32
        def proposal(v):v['history']['journal']['records'][0]['proposal']={'message':{},'manifest':{}}
        def hidden_fault(v):v['history']['state']['evidence_projection']['data']['resource_faults']=[['hidden']]
        for mutate in (unknown,authority,schema,schema_only,pending_only,foreign,proposal,hidden_fault,hidden_monitor):
            with self.subTest(mutation=mutate.__name__):
                value=copy.deepcopy(original);mutate(value);hist=value['history'];state=hist['state']
                state['journal_digest']=digest(dict(payload=hist['journal'],sha256=digest(hist['journal'])))
                state['evidence_projection']['journal']=state['journal_digest']
                state['source_digest']=digest(state['evidence_projection']);path.write_text(encode(value))
                self.reject_both(rid)
        path.write_text(encode(original))
        self.assertEqual(self.f.run_registration(),rid)

    def test_review_two_node_cycle_retry_and_handoff(self):
        rid,path,value=self.history_fixture(False)
        old=json.loads(self.raw);old['kind']='installed-registration-v2';old['history']=copy.deepcopy(value['history'])
        state=old['history']['state'];registration=value['registration']
        state['registrations']=[registration];old['history']['predecessors']=[rid]
        state['evidence_projection']['data']['p1_registrations']=[[rid,encode(registration),rid]]
        state['source_digest']=digest(state['evidence_projection']);self.path.write_text(encode(old))
        self.reject_both(rid)

    def third_restart(self):
        for owner,container in zip(self.f.owners,self.host.containers):
            owner['started']=container['started']='2026-10-24T03:00:00Z';container['pid']+=100
        self.f.observer.proc=lambda pid,identity,ipc:dict(pid=pid,start_ticks=300,pid_namespace=pid,cgroup='/docker/'+identity)
        self.f.hello['boot']='d'*32
        return self.f.run_registration()

    def test_multiple_clean_restarts_immutable(self):
        second,path,value=self.history_fixture();saved=path.read_bytes()
        third=self.third_restart();self.assertNotEqual(second,third)
        self.assertEqual(self.path.read_bytes(),self.raw);self.assertEqual(path.read_bytes(),saved)
        self.assertEqual(self.f.run_registration(),third)
        delivered=[]
        h._installed_handoff(self.f.observer,self.f.owners,third,self.f.fd,lambda:None,
            notify=lambda *args:delivered.append(args))
        self.assertEqual(len(delivered),1)

    def test_multinode_cycle_future_and_replay(self):
        second,path,b=self.history_fixture(False);third=self.third_restart()
        cpath=self.f.authority/('registration-'+third+'.json');c=json.loads(cpath.read_bytes())
        originals={self.path:self.raw,path:path.read_bytes(),cpath:cpath.read_bytes()}
        for mode in ('cycle','future','replay'):
            with self.subTest(mode=mode):
                for p,raw in originals.items():p.write_bytes(raw)
                if mode=='cycle':
                    a=json.loads(self.raw);a['kind']='installed-registration-v2';a['history']=copy.deepcopy(b['history'])
                    hist=a['history'];hist['predecessors']=[third];hist['state']['registrations']=[c['registration']]
                    hist['state']['evidence_projection']['data']['p1_registrations']=[[third,encode(c['registration']),third]]
                    hist['state']['source_digest']=digest(hist['state']['evidence_projection']);self.path.write_text(encode(a))
                else:
                    # A coherent DAG shape can still claim a later incarnation
                    # as an earlier prestate, or replay an equal incarnation.
                    a=json.loads(self.raw);a['kind']='installed-registration-v2';a['history']=copy.deepcopy(b['history'])
                    if mode=='future':
                        root=copy.deepcopy(b);root['kind']='installed-registration-v1';root.pop('history')
                        path.write_text(encode(root))
                    parent=copy.deepcopy(b['registration'])
                    if mode=='replay': parent=copy.deepcopy(a['registration'])
                    pid=digest(parent)
                    if mode=='replay':
                        # Exact self replay is a cycle of length one.
                        parent=a['registration'];pid=self.first
                    hist=a['history'];hist['predecessors']=[pid];hist['state']['registrations']=[parent]
                    hist['state']['evidence_projection']['data']['p1_registrations']=[[pid,encode(parent),pid]]
                    hist['state']['source_digest']=digest(hist['state']['evidence_projection']);self.path.write_text(encode(a))
                self.reject_both(third)

    def test_typed_snapshot_equivalence(self):
        import sqlite3
        from jsonschema.exceptions import ValidationError
        journal=TerminalRestart.terminal_fixture(self)
        state=h._registration_snapshot(self.f.roots,self.f.scope)
        for field,index in (('job',6),('context',7)):
            with self.subTest(field=field):
                detached=copy.deepcopy(state)
                rows=detached['evidence_projection']['data']['jobs']
                rows[0]=list(rows[0]);original=rows[0][index]
                value=json.loads(original);value['invented_owner_claim']='accepted'
                rows[0][index]=encode(value)
                detached['source_digest']=digest(detached['evidence_projection'])
                with self.assertRaises((ValueError,ValidationError)):
                    recovery.installed_clean_history(detached,journal)
                with sqlite3.connect(self.f.roots['control']/'core/core.sqlite') as db:
                    db.execute('UPDATE jobs SET '+field+'=?',(encode(value),))
                try:
                    with self.assertRaises((ValueError,ValidationError)):
                        h._registration_snapshot(self.f.roots,self.f.scope)
                finally:
                    with sqlite3.connect(self.f.roots['control']/'core/core.sqlite') as db:
                        db.execute('UPDATE jobs SET '+field+'=?',(original,))

    def test_shared_journal_loader_equivalence(self):
        from worker.runtime_journal import Journal,validate_payload,journal_bytes
        from unittest.mock import patch
        journal=TerminalRestart.terminal_fixture(self)
        path=self.f.roots['receipts']/'journal.json'
        def unknown(v):v['records'][0]['unknown']=True
        def binding(v):v['records'][0]['allocation']['binding']='0'*64
        def receipt(v):v['records'][0]['receipt']['reaped_children']=-1
        def proposal(v):v['records'][0]['proposal']={'message':{},'manifest':{}}
        def uid(v):v['records'][0]['uid']=True
        for mutate in (lambda v:None,unknown,binding,receipt,proposal,uid):
            with self.subTest(mutation=mutate.__name__):
                value=copy.deepcopy(journal);mutate(value);path.write_bytes(journal_bytes(value))
                outcomes=[]
                for operation in (lambda:validate_payload(value),lambda:Journal._load(reader)):
                    reader=Journal.__new__(Journal);reader.root=self.f.roots['receipts']
                    try:
                        with patch('worker.runtime_journal.os.getuid',return_value=path.stat().st_uid):operation()
                        outcomes.append(None)
                    except Exception as exc:outcomes.append((type(exc),str(exc)))
                self.assertEqual(outcomes[0],outcomes[1])
                if mutate.__name__!='<lambda>':self.assertIsNotNone(outcomes[0])


if __name__=='__main__':unittest.main()
