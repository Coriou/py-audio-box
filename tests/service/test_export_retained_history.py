"""Disposable enrolled SQL/PCM; injected identity and journal, no installed journey."""
import copy
import json
import sqlite3
import unittest
from unittest.mock import patch
from contextlib import contextmanager

@contextmanager
def database(path):
    db=sqlite3.connect(path)
    try:
        with db: yield db
    finally: db.close()
from service import installed_export as ex, recovery
from service.core import encode
from tests.service.test_export_daemon import ExportDaemonTests


class ExportRetainedHistoryTests(unittest.TestCase):
    def fixture(self):
        owner=ExportDaemonTests(); self.addCleanup(owner.doCleanups)
        s=owner.initialized()
        owner.registration(s)
        return s,json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']

    def test_fresh_enrolled_retained_reader_and_clean_projection(self):
        s,public=self.fixture()
        before=s.f.bytes()
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        journal=json.loads((s.roots['receipts']/'journal.json').read_bytes())['payload']
        self.assertTrue(recovery.installed_clean_history(state,journal,family=ex.FAMILY))
        self.assertEqual(s.f.bytes(),before)
        with patch.object(s.helper,'public',return_value=public):
            self.assertEqual(s.helper.clean()['state']['source_digest'],state['source_digest'])
        for family in (None,object()):
            with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=family)
        with self.assertRaises(ValueError): recovery.installed_clean_history(state,journal)

    def test_family_version_and_unreaped_mutations_refuse(self):
        s,public=self.fixture()
        path=s.roots['control']/'core/core.sqlite'
        for version in (1,3):
            with database(path) as db:db.execute('PRAGMA user_version='+str(version))
            before=s.f.bytes()
            with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
            self.assertEqual(s.f.bytes(),before)
        with database(path) as db:db.execute('PRAGMA user_version=2')
        bad=copy.deepcopy(public);bad['scope']='owned-kokoro-v1'
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,bad,family=ex.FAMILY)
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        journal=json.loads((s.roots['receipts']/'journal.json').read_bytes())['payload']
        for key,value in [('faults',['unknown']),('allocation_coherent',False),('inventory',dict(reasons=['pending'],counts={}))]:
            bad=copy.deepcopy(state);bad[key]=value
            with self.assertRaises(ValueError): recovery.installed_clean_history(bad,journal,family=ex.FAMILY)

    def retained_export(self):
        """Bind completed LOCAL inert SQL/files to disposable exact enrollment.

        No installed dispatcher is activated. Runtime IDs/UID receipts are
        explicit injected journal facts, never real descendant-reap evidence.
        """
        import os
        from contextlib import closing
        from pathlib import Path
        from service.audio_export import AudioExports
        from worker.runtime_journal import journal_bytes
        import test_audio_export as local
        s,public=self.fixture()
        f=local.AudioExportTests();f.setUp();self.addCleanup(f.doCleanups)
        f.dispatcher.close()
        f.exports=AudioExports(public['configuration']['audio_export']['converter'],f.profile,['alice'])
        f.start()
        # Complete inert local export before split fixture installs UID observations.
        source=local.audio(frames=96000)
        asset=f.request('POST','/v1/assets',content=source).json()
        receipt=f.submit('audio_export',dict(asset_id=asset['id'],rate=48000,channels=2,loudness=None),'retained-large')
        job=f.terminal(receipt)
        self.assertEqual(job['state'],'succeeded')
        f.dispatcher.close()
        target=s.roots['control']/'core';path=target/'core.sqlite'
        # Exact fixture seam: copy common local history rows, preserving the
        # initialized enrollment, registrations and format/catalogue fixed rows.
        copied=('jobs','events','assets','artifacts','publications','publication_work',
                'resource_reservations','dispatch_settings','dispatches','dispatch_commands')
        with database(f.root/'core.sqlite') as src,database(path) as dst:
            for table in copied:
                rows=list(src.execute('SELECT * FROM '+table))
                if rows:dst.executemany('INSERT INTO '+table+' VALUES('+','.join('?' for _ in rows[0])+')',rows)
            dst.execute('UPDATE resource_accounting SET highwater=(SELECT max(seq) FROM dispatches)')
            rid=s.gate['registration']['lane']+'/'+s.gate['registration']['boot']+'/'+'a'*32
            authority,binding,raw=dst.execute('SELECT authority,binding,receipt FROM dispatches').fetchone()
            terminal=json.loads(raw);terminal.update(runtime_id=rid,engine_uid=20000)
            dst.execute('UPDATE dispatches SET runtime=?,receipt=?',(rid,encode(terminal)))
            # No production configuration is changed by this history fixture.
        for p in (f.root/'assets').iterdir():
            (target/'assets'/p.name).write_bytes(p.read_bytes());(target/'assets'/p.name).chmod(0o600)
        for p in (f.root/'owners').iterdir():
            (target/'owners'/p.name).write_bytes(p.read_bytes());(target/'owners'/p.name).chmod(0o600)
        journal=dict(version=1,lane=s.gate['registration']['lane'],records=[dict(id=rid,
            allocation=dict(snapshot=json.loads(authority),binding=binding,fixture='audio_export'),
            phase='terminal',acked=True,uid=20000,boot=s.gate['registration']['boot'],receipt=terminal,proposal=None)])
        (s.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
        return s,public,journal,asset,source

    def test_completed_large_export_retained_reopen_and_canonical_mutations(self):
        s,public,journal,asset,source=self.retained_export()
        path=s.roots['control']/'core/core.sqlite'
        before=s.f.bytes()
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertTrue(recovery.installed_clean_history(state,journal,family=ex.FAMILY))
        self.assertEqual((path.parent/'assets'/asset['id']).read_bytes(),source)
        self.assertEqual(s.f.bytes(),before)
        self.assertEqual(ex.retained_snapshot(s.roots,public,family=ex.FAMILY)['source_digest'],state['source_digest'])
        self.assertGreater(max(p.stat().st_size for p in (path.parent/'assets').iterdir()),1024**2)
        import os
        from contextlib import closing
        from service.core import Core
        with patch.object(os,'getuid',return_value=10001),closing(Core(path.parent,_installed_family=ex.expectation(public,ex.FAMILY))) as core:
            self.assertEqual(core.publication_format,2)
            self.assertEqual(core.job(state['jobs'][0]['id'])['state'],'succeeded')
            artifact_id=json.loads(state['evidence_projection']['data']['artifacts'][0][1])['id']
            self.assertEqual(core.artifact(artifact_id)['id'],artifact_id)
        data=copy.deepcopy(state['evidence_projection']['data'])
        frozen=json.loads(data['dispatches'][0][2]);frozen[0]['generation']=float(frozen[0]['generation'])
        data['dispatches'][0]=list(data['dispatches'][0]);data['dispatches'][0][2]=encode(frozen)
        with self.assertRaises(ValueError):ex._retained_contexts(public,data)
        # SQL copies are changed one at a time and restored, with no repair path.
        for table,column,change in [('artifacts','metadata',lambda x:x['audio_export'].update(source_sha256='0'*64)),
                                    ('publications','manifest',lambda x:x['audio_export']['converter'].update(reported_version='mutated')),
                                    ('jobs','context',lambda x:x['plan']['assets'][0].update(sha256='0'*64))]:
            with self.subTest(table=table):
                with database(path) as db:
                    original=db.execute('SELECT '+column+' FROM '+table).fetchone()[0]
                    value=json.loads(original);change(value)
                    db.execute('UPDATE '+table+' SET '+column+'=?',(encode(value),))
                changed=s.f.bytes()
                with self.assertRaises((ValueError,RuntimeError)):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
                self.assertEqual(s.f.bytes(),changed)
                with database(path) as db:db.execute('UPDATE '+table+' SET '+column+'=?',(original,))
        for mutate in (lambda j:j['records'][0].update(acked=False),
                       lambda j:j['records'][0]['receipt'].update(reaped=False),
                       lambda j:j['records'][0].update(boot='f'*32)):
            bad=copy.deepcopy(journal);mutate(bad)
            with self.assertRaises(ValueError):recovery.installed_clean_history(state,bad,family=ex.FAMILY)

    def test_dispatch_phase_refusal_across_clean_history_readers(self):
        from tests.service.test_export_daemon import ExportSnapshotTests
        s,public,journal,asset,source=self.retained_export()
        path=s.roots['control']/'core/core.sqlite'
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertEqual([d['phase'] for d in state['dispatches']],['acknowledged'])
        hello=s.gate['observation']['hello']
        facts=s.gate['observation']['local']['control']
        incarnation={k:facts[k] for k in ('start_ticks','pid_namespace')};incarnation['cgroup']='0::/'
        pair=copy.deepcopy(s.gate['observation']['inspections'])
        for container in pair.values():container['State'].update(Running=False,ExitCode=0,Status='exited',Pid=0)
        clean_history=json.loads(encode(dict(state=state,journal=journal)))
        clean_receipt=dict(containers=pair,history=clean_history)
        self.assertEqual(ex.stopped_receipt(clean_receipt,pair,clean_history,public=public,family=ex.FAMILY),clean_receipt)
        # Keep a WAL writer open: every reader must preserve source bytes,
        # sidecars, identities, timestamps and namespace after deliberate mutation.
        with database(path) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            for phase in ('unknown','started','bound','observing','receipt','ack_pending','acked','idle'):
                with self.subTest(phase=phase):
                    writer.execute('UPDATE dispatches SET phase=?',(phase,));writer.commit()
                    bad=json.loads(encode(state))
                    bad['dispatches'][0]['phase']=phase
                    bad['evidence_projection']['data']['dispatches'][0][6]=phase
                    bad['source_digest']=recovery.digest(bad['evidence_projection'])
                    history=dict(state=bad,journal=journal)
                    receipt=dict(containers=pair,history=history)
                    before=ExportSnapshotTests.source_state(self,path)
                    namespace=sorted(p.name for p in path.parent.iterdir())
                    files=s.f.bytes()
                    readers=(
                        lambda:ex.retained_snapshot(s.roots,public,family=ex.FAMILY),
                        lambda:ex.startup_read(path.parent,s.config,s.gate,hello,incarnation,family=ex.FAMILY),
                        lambda:recovery._installed_detached_source(bad,journal,family=ex.FAMILY),
                        lambda:recovery.installed_clean_history(bad,journal,family=ex.FAMILY),
                        lambda:s.helper.clean(),
                        lambda:ex.stopped_receipt(receipt,pair,history,public=public,family=ex.FAMILY))
                    with patch.object(s.helper,'public',return_value=public):
                        for read in readers:
                            with self.assertRaisesRegex(ValueError,'^installed_history_unclean$'):read()
                            self.assertEqual(ExportSnapshotTests.source_state(self,path),before)
                            self.assertEqual(sorted(p.name for p in path.parent.iterdir()),namespace)
                            self.assertEqual(s.f.bytes(),files)
                    self.assertEqual(writer.execute('SELECT phase,acked FROM dispatches').fetchall(),[(phase,1)])
            # Restoration is explicit fixture setup, never a reader repair.
            writer.execute("UPDATE dispatches SET phase='acknowledged'");writer.commit()
            restored=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
            self.assertTrue(recovery.installed_clean_history(restored,journal,family=ex.FAMILY))
            ex.startup_read(path.parent,s.config,s.gate,hello,incarnation,family=ex.FAMILY)

    def test_precore_private_read_boot_and_incarnation(self):
        s,public,journal,asset,source=self.retained_export()
        path=s.roots['control']/'core/core.sqlite'
        hello=s.gate['observation']['hello']
        facts=s.gate['observation']['local']['control']
        incarnation={k:facts[k] for k in ('start_ticks','pid_namespace')};incarnation['cgroup']='0::/'
        before=s.f.bytes()
        result=recovery.installed_startup_read(path.parent,s.config,s.gate,hello,incarnation,
            policy=ex.ip.EXPORT_SPLIT,launch_policy=ex.LAUNCH,family=ex.FAMILY)
        self.assertEqual(result['registration_id'],s.gate['registration_id'])
        self.assertEqual(s.f.bytes(),before)
        for changed in (dict(hello,boot='f'*32),dict(hello,launches=True)):
            with self.assertRaises(ValueError):ex.startup_read(path.parent,s.config,s.gate,changed,incarnation,family=ex.FAMILY)
        with self.assertRaises(ValueError):ex.startup_read(path.parent,s.config,s.gate,hello,dict(incarnation,start_ticks=999),family=ex.FAMILY)
        with self.assertRaises(ValueError):recovery.installed_startup_read(path.parent,s.config,s.gate,hello,incarnation,policy=ex.ip.SPLIT,launch_policy=ex.LAUNCH)
        with database(path) as db:db.execute('UPDATE dispatches SET acked=0')
        with self.assertRaises(ValueError):ex.startup_read(path.parent,s.config,s.gate,hello,incarnation,family=ex.FAMILY)
        with database(path) as db:db.execute('UPDATE dispatches SET acked=1')

    def test_selected_counter_budgets_and_all_hardlink_names(self):
        import time
        from service.reconciliation import Reconciliation,Incomplete
        scanner=Reconciliation.__new__(Reconciliation)
        scanner.entry_limit=2048;scanner.byte_limit=scanner.scalar_limit=ex.SCALAR_BYTES
        scanner.media_limit=ex.MEDIA_BYTES;scanner.separate_media=True
        scanner.entries=scanner.bytes=scanner.media_bytes=0;scanner.end=time.monotonic()+1
        # Each name charges fully, even when it represents the same inode.
        for _ in range(4):scanner._charge(36*1024**2,media=True)
        self.assertEqual(scanner.media_bytes,144*1024**2)
        with self.assertRaises(Incomplete):scanner._charge(24*1024**2,media=True)
        scanner.media_bytes=0
        scanner._charge(ex.SCALAR_BYTES)
        with self.assertRaises(Incomplete):scanner._charge(1)
        scanner.bytes=0;scanner.entries=2048
        with self.assertRaises(Incomplete):scanner._charge()
        scanner.entries=0;scanner.end=0
        with self.assertRaises(Incomplete):scanner._charge()

    def test_current_registration_family_and_boot_lane(self):
        s,public=self.fixture();record=s.gate['registration']
        self.assertEqual(recovery.installed_history_current([record],public['host'],public['deployment_id'],[record],family=ex.FAMILY,public=public),sorted(record['containers'],key=lambda c:c['id']))
        with self.assertRaises(ValueError):recovery.installed_history_current([record],public['host'],public['deployment_id'],[record])
        bad=copy.deepcopy(record);bad['lane']='f'*32
        with self.assertRaises(ValueError):recovery.installed_history_current([record],public['host'],public['deployment_id'],[bad],family=ex.FAMILY,public=public)
        with self.assertRaises(ValueError):recovery.installed_history_current([record,bad],public['host'],public['deployment_id'],[record],family=ex.FAMILY,public=public)

    def test_retained_media_intent_missing_source_and_reader_budgets(self):
        from service.persistence import PublicationQuarantine
        s,public,journal,asset,source=self.retained_export()
        path=s.roots['control']/'core/core.sqlite'
        source_path=path.parent/'assets'/asset['id']
        with database(path) as db:
            original=db.execute('SELECT metadata FROM assets WHERE id=?',(asset['id'],)).fetchone()[0]
            changed=json.loads(original);changed['frames']+=1
            db.execute('UPDATE assets SET metadata=? WHERE id=?',(encode(changed),asset['id']))
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        with database(path) as db:db.execute('UPDATE assets SET metadata=? WHERE id=?',(original,asset['id']))
        source_path.write_bytes(b'X'+source[1:])
        changed=s.f.bytes()
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertEqual(s.f.bytes(),changed)
        source_path.write_bytes(source)
        source_path.rename(source_path.with_name('missing-source'))
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        source_path.with_name('missing-source').rename(source_path)
        with database(path) as db:
            original=db.execute('SELECT binding FROM publication_work').fetchone()[0]
            value=json.loads(original)
            # Valid JSON, changed canonical producer copy.
            value['audio_export']['source_sha256']='0'*64
            db.execute('UPDATE publication_work SET binding=?',(encode(value),))
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        with database(path) as db:db.execute('UPDATE publication_work SET binding=?',(original,))
        for constant in ('MEDIA_BYTES','SCALAR_BYTES'):
            before=s.f.bytes()
            with patch.object(ex,constant,1),self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
            self.assertEqual(s.f.bytes(),before)
        # The initial reader still refuses populated history.
        with self.assertRaises(ValueError):s.helper.registration_snapshot(public)

    def test_legacy_acked_speech_fixture_refused_without_repair(self):
        from types import SimpleNamespace
        from tests.service.test_installed_restart import TerminalRestart
        s,public=self.fixture()
        holder=SimpleNamespace(f=SimpleNamespace(roots={r:s.roots[r] for r in ex.ip.ROLES},
            lane=s.gate['registration']['lane'],hello=dict(boot=s.gate['registration']['boot'])))
        TerminalRestart.terminal_fixture(holder)
        path=s.roots['control']/'core/core.sqlite'
        with database(path) as db:
            self.assertEqual(db.execute('SELECT phase,acked FROM dispatches').fetchall(),[('acked',1)])
        before=s.f.bytes()
        with self.assertRaisesRegex(ValueError,'^installed_history_unclean$'):
            ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertEqual(s.f.bytes(),before)

    def test_retained_wal_private_targets_and_source_identity(self):
        from tests.service.test_export_daemon import ExportSnapshotTests
        s,public=self.fixture();path=s.roots['control']/'core/core.sqlite'
        with database(path) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            writer.execute('PRAGMA application_id=388');writer.commit()
            before=ExportSnapshotTests.source_state(self,path)
            namespace=sorted(p.name for p in path.parent.iterdir())
            connect=sqlite3.connect;targets=[]
            def opened(target,*args,**kwargs):
                targets.append(str(target));self.assertNotIn(str(path.parent),str(target))
                db=connect(target,*args,**kwargs)
                if str(target)!=':memory:':self.assertEqual(db.execute('PRAGMA application_id').fetchone(),(388,))
                return db
            with patch.object(sqlite3,'connect',side_effect=opened):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
            self.assertTrue(any(t.startswith('file:') for t in targets))
            self.assertEqual(ExportSnapshotTests.source_state(self,path),before)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()),namespace)

    def test_detached_versions_and_schema_mutations(self):
        s,public=self.fixture()
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        journal=json.loads((s.roots['receipts']/'journal.json').read_bytes())['payload']
        for marker in (1,2.0,True,3):
            bad=json.loads(encode(state));bad['evidence_projection']['data']['p1_format'][0][0]=marker
            with self.assertRaises(ValueError):recovery._installed_detached_source(bad,journal,family=ex.FAMILY)
        for version in (True,1.0):
            with self.assertRaises(ValueError):recovery._installed_detached_source(state,dict(journal,version=version),family=ex.FAMILY)
        path=s.roots['control']/'core/core.sqlite'
        with database(path) as db:db.execute('CREATE INDEX alien_retained ON assets(id)')
        before=s.f.bytes()
        with self.assertRaises(ValueError):ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertEqual(s.f.bytes(),before)

    def test_retained_clean_registration_prestate_and_reopen(self):
        from service import daemon_helper as dh
        s,public,journal,asset,source=self.retained_export()
        old=s.gate['registration_id']
        before=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        history=json.loads(encode(dict(state=before,journal=journal)))
        def runtime(h):
            ids={role:c['Id'] for role,c in s.gate['observation']['inspections'].items()}
            for role in ('control','worker'):h.inspect(role,ids,'start')
            rid,record=h.register(ids,h.public(),history)
            return dict(registration_id=rid)
        with patch.object(dh.Helper,'runtime',runtime): result=s.run('up')
        self.assertNotEqual(result['registration_id'],old)
        state=ex.retained_snapshot(s.roots,public,family=ex.FAMILY)
        self.assertEqual(len(state['registrations']),2)
        self.assertTrue(recovery.installed_clean_history(state,journal,family=ex.FAMILY))
        self.assertEqual(state['evidence_projection']['data']['jobs'],before['evidence_projection']['data']['jobs'])
        self.assertEqual((s.roots['control']/'core/assets'/asset['id']).read_bytes(),source)

    def test_live_unknown_reap_ack_and_uncertain_command_refuse(self):
        from worker.runtime_journal import journal_bytes
        s,public,journal,asset,source=self.retained_export()
        path=s.roots['control']/'core/core.sqlite'
        for mutate in (lambda j:j['records'][0].update(acked=False),
                       lambda j:j['records'][0]['receipt'].update(reaped=False),
                       lambda j:j['records'][0].update(boot='f'*32)):
            bad=copy.deepcopy(journal);mutate(bad)
            (s.roots['receipts']/'journal.json').write_bytes(journal_bytes(bad))
            changed=s.f.bytes()
            with patch.object(s.helper,'public',return_value=public),self.assertRaises(ValueError):s.helper.clean()
            self.assertEqual(s.f.bytes(),changed)
        (s.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
        with database(path) as db:
            original=list(db.execute('SELECT seq,state FROM dispatch_commands'))
            db.execute("UPDATE dispatch_commands SET state='pending' WHERE seq=(SELECT min(seq) FROM dispatch_commands)")
        with patch.object(s.helper,'public',return_value=public),self.assertRaises(ValueError):s.helper.clean()
        with database(path) as db:
            for seq,state in original:db.execute('UPDATE dispatch_commands SET state=? WHERE seq=?',(state,seq))
        self.assertTrue(recovery.installed_clean_history(ex.retained_snapshot(s.roots,public,family=ex.FAMILY),journal,family=ex.FAMILY))
