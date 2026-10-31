"""Small owned synthetic fixtures; no runtime, hardware, or crash invocation."""
import copy
import json
import os
import unittest
from unittest.mock import patch
import test_production as base
from service.core import Core
from service import persistence, recovery


class ReferencePublicationTests(unittest.TestCase):
    def fixture(self):
        case=base.ProductionTests();case.setUp()
        self.addCleanup(case.tearDown)
        return case

    def test_each_file_interruption_restart_is_all_or_none(self):
        for point,complete in [('before_adoption',False),('before_file_0',False),
                ('after_file_0',False),('after_link_0',False),('before_file_1',False),
                ('after_file_1',False),('after_link_1',True),('after_adoption',True),('after_commit',True)]:
            with self.subTest(point=point):
                c=self.fixture();job=c.domain.prepare('p','prepare',c.context);c.core.start()
                def fault(actual):
                    if actual==point: raise RuntimeError(point)
                with self.assertRaisesRegex(RuntimeError,point): c.core.finish(fault=fault)
                manifest=json.loads(c.core.db.execute('SELECT manifest FROM publications').fetchone()[0])
                ids=[o['id'] for o in manifest['outputs']]
                self.assertEqual(c.core.db.execute('SELECT count(*) FROM assets WHERE id IN (?,?)',ids).fetchone()[0],2 if point=='after_commit' else 0)
                root=c.core.root;c.core.close();c.core=Core(root,historical_local=True)
                c.core.reconcile()
                self.assertEqual(c.core.job(job['id'])['state'],'succeeded' if complete else 'failed')
                self.assertEqual(c.core.db.execute('SELECT count(*) FROM assets WHERE id IN (?,?)',ids).fetchone()[0],2 if complete else 0)
                if complete:
                    c.core._commit_publication(job['id'])
                    self.assertEqual(len(c.core.artifact(c.core.job(job['id'])['artifact_ids'][0])['asset_ids']),2)

    def test_closed_intent_and_detached_snapshot_tampering(self):
        c=self.fixture();c.prepared()
        row=c.core.db.execute('SELECT binding,destination,temporary,sha256,byte_length FROM publication_work').fetchone()
        artifact,entries=persistence.publication_intent(row)
        self.assertEqual(len(entries),2)
        record=dict(zip(('binding','destination','temporary','sha256','byte_length'),row),job_id=artifact['job_id'])
        data={table:[dict(zip([c[0] for c in cursor.description],r)) for r in cursor.fetchall()] for table in ('publications','jobs') for cursor in [c.core.db.execute('SELECT * FROM '+table)]}
        data['publication_work']=[record]
        recovery._installed_detached_documents(data)
        for mutate in [lambda v:v.update(version='unknown'),lambda v:v.update(extra=1),
                lambda v:v['files'].append(v['files'][0]),
                lambda v:v['files'].__setitem__(1,v['files'][0]),
                lambda v:v['files'][1].update(sha256='0'*64),
                lambda v:v['files'][1].update(byte_length=9),
                lambda v:v['artifact'].update(attempt_id='foreign')]:
            value=json.loads(row[0]);mutate(value)
            bad=(json.dumps(value),*row[1:])
            with self.assertRaises(ValueError):persistence.publication_intent(bad)
            changed=dict(record,binding=bad[0])
            with self.assertRaises(ValueError):recovery._installed_detached_documents(dict(data,publication_work=[changed]))
        for index,value in [(1,'foreign'),(2,'adoption-'+'f'*32),(3,'0'*64),(4,1)]:
            bad=list(row);bad[index]=value
            with self.assertRaises(ValueError):persistence.publication_intent(bad)

    def test_file_write_link_and_fsync_failure_retains_intents(self):
        for operation in ('write','link','fsync'):
            for fail_at in (1,2):
                with self.subTest(operation=operation,fail_at=fail_at):
                    c=self.fixture();job=c.domain.prepare('p','prepare',c.context);c.core.start()
                    target='service.core.os.link' if operation=='link' else 'service.persistence.write_complete'
                    original=os.link if operation=='link' else persistence.write_complete
                    calls=0
                    def failing(*args,**kwargs):
                        nonlocal calls
                        calls+=1
                        if calls==fail_at:
                            if operation=='fsync':
                                with patch('service.persistence.os.fsync',side_effect=OSError('fixture_fault')):
                                    return original(*args,**kwargs)
                            raise OSError('fixture_fault')
                        return original(*args,**kwargs)
                    with patch(target,side_effect=failing):
                        with self.assertRaises(OSError):c.core.finish()
                    self.assertNotEqual(c.core.job(job['id'])['state'],'succeeded')
                    self.assertEqual(c.core.db.execute('SELECT count(*) FROM assets').fetchone()[0],1)
                    # Retain the original intent and names; restart may complete
                    # only if both links and their durable bytes already exist.
                    root=c.core.root;c.core.close();c.core=Core(root,historical_local=True);c.core.reconcile()
                    count=c.core.db.execute('SELECT count(*) FROM assets').fetchone()[0]
                    self.assertIn(count,(1,3))

    def test_partial_hash_foreign_provenance_and_duplicate_publication(self):
        c=self.fixture();captured=[]
        original=c.core._publish
        def capture(job,proposal,**kwargs):
            captured.append(copy.deepcopy(proposal));return original(job,proposal,**kwargs)
        with patch.object(c.core,'_publish',side_effect=capture):job=c.prepared()
        proposal=captured[0]
        with self.assertRaises(ValueError):
            c.core._publish(job['id'],proposal)
        self.assertEqual(c.core.db.execute('SELECT count(*) FROM assets').fetchone()[0],3)
        # Recreate a running fixture, retain verified pre-adoption proposal.
        c=self.fixture();job=c.domain.prepare('p','prepare',c.context);c.core.start();captured=[]
        def retain(job_id,proposal,**kwargs):
            captured.append(copy.deepcopy(proposal));raise RuntimeError('retained')
        with patch.object(c.core,'_publish',side_effect=retain):
            with self.assertRaises(RuntimeError):c.core.finish()
        proposal=captured[0];running=c.core.job(job['id'])
        stage=c.core.root/'staging'/running['attempts'][-1]['staging_id']
        for filename in ('reference.wav','audition.wav'):
            path=stage/filename;data=path.read_bytes()
            for invalid in (data[:-2],data[:-2]+b'\xff\xff'):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):c.core._verified(running,proposal)
            path.write_bytes(data)
        bad=copy.deepcopy(proposal);bad['manifest']['outputs'][0]['provenance']['source_id']='foreign'
        with self.assertRaises(ValueError):c.core._verified(running,bad)
        path=stage/'audition.wav';data=path.read_bytes();path.unlink();path.symlink_to(stage/'reference.wav')
        with self.assertRaises(OSError):c.core._verified(running,proposal)
        path.unlink();path.write_bytes(data)

    def test_metadata_commit_ambiguity_restarts_without_duplicate_history(self):
        import sqlite3
        class FaultDB:
            def __init__(self,db,after): self.db=db;self.after=after;self.fired=False
            def __getattr__(self,name): return getattr(self.db,name)
            def execute(self,sql,*args):
                if sql=='COMMIT' and not self.fired:
                    self.fired=True
                    if self.after:self.db.execute(sql,*args)
                    raise sqlite3.OperationalError('fixture_commit')
                return self.db.execute(sql,*args)
        for after in (False,True):
            c=self.fixture();job=c.domain.prepare('p','prepare',c.context);c.core.start()
            def stop(point):
                if point=='after_adoption':raise RuntimeError('retained')
            with self.assertRaises(RuntimeError):c.core.finish(fault=stop)
            real=c.core.db;c.core._db=c.core._ownership.db=FaultDB(real,after)
            with self.assertRaises(sqlite3.OperationalError):c.core._commit_publication(job['id'])
            c.core._db=c.core._ownership.db=real
            root=c.core.root;c.core.close();c.core=Core(root,historical_local=True)
            c.core.reconcile();c.core._commit_publication(job['id'])
            self.assertEqual(c.core.job(job['id'])['state'],'succeeded')
            self.assertEqual(c.core.db.execute('SELECT count(*) FROM assets').fetchone()[0],3)
            self.assertEqual([e['state'] for e in c.core.events(job['id'])],['queued','running','succeeded'])

    def test_legacy_intent_snapshot_is_not_reinterpreted(self):
        c=self.fixture()
        context=json.loads((base.FIXTURES/'transcript-speech_generate.json').read_text())['context']
        job=c.core.submit_synthetic('p','legacy',context);c.core.start();c.core.finish()
        row=c.core.db.execute('SELECT binding,destination,temporary,sha256,byte_length FROM publication_work').fetchone()
        artifact,entries=persistence.publication_intent(row)
        self.assertNotIn('version',json.loads(row[0]));self.assertEqual(len(entries),1)
        data={table:[dict(zip([col[0] for col in cursor.description],r)) for r in cursor.fetchall()] for table in ('publications','jobs','publication_work') for cursor in [c.core.db.execute('SELECT * FROM '+table)]}
        recovery._installed_detached_documents(data)
        root=c.core.root;c.core.close();c.core=Core(root,historical_local=True);c.core.reconcile()
        self.assertEqual(c.core.job(job['id'])['state'],'succeeded')
        self.assertEqual(c.core.db.execute('SELECT binding,destination,temporary,sha256,byte_length FROM publication_work').fetchone(),row)

    def test_exact_frozen_role_text_and_no_audition_reuse(self):
        c=self.fixture()
        c.context['plan']['reference_roles']={'audition_text':' Explicit\n audition text '}
        c.context['plan']['request']['audition_text']=' Explicit\n audition text '
        c.context['plan']['request']['reference_text']=' Original\n transcript '
        c.context['plan']['resolved']['reuse_configuration']['reference_text']=c.context['plan']['request']['reference_text']
        job=c.domain.prepare('p','prepare',c.context)
        c.context['plan']['reference_roles']['audition_text']='mutated'
        c.context['plan']['request']['audition_text']='mutated'
        c.core.start();self.assertEqual(c.core.finish()['state'],'succeeded')
        candidate=c.domain.results('p',job['id'])[0]
        self.assertEqual(candidate['reference_roles']['audition']['text'],' Explicit\n audition text ')
        self.assertEqual(candidate['reference_roles']['reference']['transcript'],' Original\n transcript ')
        self.assertEqual(candidate['reference_roles']['reference']['original'],c.asset)
        self.assertNotEqual(candidate['version']['reference_asset_ids'],[candidate['audition_asset_id']])
        saved=c.domain.save('p','save',job['id'],candidate['id'],'voice',expected_revision=0)
        self.assertEqual(saved['version']['reference_asset_ids'],[candidate['reference_roles']['reference']['asset']['id']])
