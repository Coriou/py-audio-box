"""Inert SQLite integration. UID/role observations injected, no child executed."""
import copy
import fcntl
import json
import os
import sqlite3
import unittest
from contextlib import ExitStack, closing
from unittest.mock import patch

from service import installed_export as ex, install_policy as ip, persistent_init as init, install_child
from service.core import Core, encode
from tests.service.test_split_init import SplitInit
from tests.service.test_daemon_volumes import binding


class ExportInitialization(unittest.TestCase):
    # Reuse fixture infrastructure only; no inherited test collection.
    def setUp(self):
        self.fixture = SplitInit()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root; self.roots=self.fixture.roots
        self.config=self.fixture.config; self.calls=self.fixture.calls
        from pathlib import Path
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())
        self.config.update(version=2,ownership_policy=ex.POLICY,
            audio_export=dict(converter=fixture['context']['execution']['converter'],profile=fixture['profiles'][0],principals=['alice']))
        self.config['audio_export']['converter']['worker_image_digest']=self.config['images']['worker']
        self.b=binding(self.config);self.b['policy']=ex.LAUNCH
        self.scope=ex.host_scope(self.b)

    def authority(self):
        return dict(host=copy.deepcopy(self.scope), roots={r:[p.stat().st_dev,p.stat().st_ino] for r,p in self.roots.items()},consumers=[])

    def bytes(self):return self.fixture.bytes()
    def completion(self):return self.fixture.completion()
    def crash(self,point):
        def fault(where):
            if where==point:raise OSError('injected '+point)
        with self.assertRaisesRegex(OSError,'injected'):self.run_init(fault=fault)

    def run_init(self,**kwargs):
        return init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,
            launch_child=self.child,daemon_binding=self.b,family=ex.FAMILY,**kwargs)

    def identity(self):
        stack=ExitStack()
        for name in ('getuid','geteuid','getgid','getegid'):stack.enter_context(patch.object(os,name,return_value=10001))
        stack.enter_context(patch.object(os,'getgroups',return_value=[]))
        return stack

    def child(self,raw):
        request=json.loads(raw);self.calls.append(raw)
        old=os.umask(0o077)
        try:
            with self.identity(): init._split_core(request['public'],self.roots['control'],self.roots['staging'],family=ex.FAMILY)
        finally: os.umask(old)

    def public(self):return json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
    def db(self):return closing(sqlite3.connect(self.roots['control']/'core/core.sqlite',isolation_level=None))

    def test_fresh_exact_and_same_family_reopen(self):
        self.run_init();public=self.public()
        with self.db() as db:ex.database(db,public,fresh=True)
        with self.identity(), closing(Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))) as core:
            self.assertEqual(core.publication_format,2)
        with self.assertRaises(ValueError):ip.installation(public)
        with self.assertRaises(ValueError):ip.configuration(self.config)

    def test_same_intent_completion(self):
        self.crash('database');before=len(self.calls)
        self.run_init(complete=self.completion())
        self.assertEqual(len(self.calls),before)
        with self.db() as db:ex.database(db,self.public(),fresh=True)

    def test_changed_completion_refuses_unchanged(self):
        self.crash('database');before=self.bytes()
        self.config['audio_export']['principals']=['bob']
        with self.assertRaises(ValueError):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes())

    def test_schema_and_marker_refusal_before_core_mutation(self):
        self.run_init();public=self.public()
        for sql,undo in [('PRAGMA user_version=1','PRAGMA user_version=2'),
                         ('CREATE INDEX alien ON p1_format(scope)','DROP INDEX alien')]:
            with self.db() as db:db.execute(sql)
            before=self.bytes()
            with self.identity(),self.assertRaises(ValueError):Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))
            self.assertEqual(before,self.bytes())
            with self.db() as db:db.execute(undo)

    def test_copied_enrollment_and_old_family_refuse(self):
        self.run_init();public=self.public();wrong=copy.deepcopy(public);wrong['deployment_id']='b'*32
        with self.assertRaises(ValueError):ex.expectation(wrong,ex.FAMILY)
        with self.identity(),self.assertRaises(ValueError):Core(self.roots['control']/'core',_publication_creation=1)

    def test_writer_transaction_and_prestate(self):
        self.run_init();public=self.public()
        reg=dict(deployment_id=public['deployment_id'],scope=self.scope,lane='a'*32,boot='b'*32,evidence=ex.EVIDENCE,
            containers=[dict(id=str(i)*64,image=self.config['images'][role],role=role,started='fixture',
                volumes=['control','staging','ipc'] if role=='control' else ['staging','receipts','ipc']) for i,role in ((1,'control'),(2,'worker'))])
        with ip.locked_file(self.roots['control']/'core/control.lock',policy=ip.SPLIT,role='control') as fd, self.identity():
            with self.db() as db:before=ex.database(db,public)
            install_child.registration_writer(public,reg,[],self.roots['control'],fd,family=ex.FAMILY)
            with self.db() as db:after=ex.database(db,public)
            self.assertEqual({k:v for k,v in before.items() if k!='p1_registrations'}, {k:v for k,v in after.items() if k!='p1_registrations'})
            with self.assertRaisesRegex(ValueError,'prestate_changed'):
                install_child.registration_writer(public,reg,[],self.roots['control'],fd,family=ex.FAMILY)
            rows=[list(r) for r in after['p1_registrations']]
            install_child.registration_writer(public,reg,rows,self.roots['control'],fd,family=ex.FAMILY)
            with self.assertRaises(ValueError):install_child.registration_writer(public,reg,rows,self.roots['control'],fd)


    def test_lost_child_reply_completion(self):
        self.crash('child_reaped')
        self.run_init(complete=self.completion())
        self.assertEqual(len(self.calls),1)
        with self.db() as db:ex.database(db,self.public(),fresh=True)

    def test_partial_core_refuses_completion_without_repair(self):
        self.crash('child_reaped')
        with self.db() as db:db.execute('DROP TABLE p1_registrations')
        before=self.bytes()
        with self.assertRaises(ValueError):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes())

    def test_mixed_ddl_unknown_trigger_b2_and_copied_root_refuse(self):
        self.run_init();public=self.public()
        mutations=[('UPDATE p1_format SET revision=1','UPDATE p1_format SET revision=2'),
            ('CREATE TRIGGER alien AFTER INSERT ON p1_registrations BEGIN SELECT 1; END','DROP TRIGGER alien'),
            ('CREATE TABLE b2_unknown(id)','DROP TABLE b2_unknown')]
        for sql,undo in mutations:
            with self.db() as db:db.execute(sql)
            before=self.bytes()
            with self.identity(),self.assertRaises(ValueError):Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))
            self.assertEqual(before,self.bytes())
            with self.db() as db:db.execute(undo)
        other=self.root/'copied';other.mkdir(mode=0o700)
        with self.identity(),self.assertRaises(ValueError):
            Core(other/'core',_installed_family=ex.expectation(public,ex.FAMILY))
        self.assertEqual(list(other.iterdir()),[])

    def test_writer_lock_and_transaction_schema_change_refuse(self):
        self.run_init();public=self.public()
        reg=dict(deployment_id=public['deployment_id'],scope=self.scope,lane='a'*32,boot='b'*32,evidence=ex.EVIDENCE,
            containers=[dict(id=str(i)*64,image=self.config['images'][role],role=role,started='fixture',
                volumes=['control','staging','ipc'] if role=='control' else ['staging','receipts','ipc']) for i,role in ((1,'control'),(2,'worker'))])
        fd=os.open(self.roots['control']/'core/control.lock',os.O_RDONLY)
        try:
            with self.identity(),self.assertRaisesRegex(ValueError,'lock_not_held'):
                install_child.registration_writer(public,reg,[],self.roots['control'],fd,family=ex.FAMILY)
        finally:os.close(fd)
        with ip.locked_file(self.roots['control']/'core/control.lock',policy=ip.SPLIT,role='control') as fd, self.identity():
            def fault(where):
                if where=='inserted':raise OSError('injected rollback')
            with self.assertRaisesRegex(OSError,'rollback'):
                install_child.registration_writer(public,reg,[],self.roots['control'],fd,family=ex.FAMILY,fault=fault)
            with self.db() as db:
                self.assertEqual(db.execute('SELECT * FROM p1_registrations').fetchall(),[])
                db.execute('PRAGMA user_version=1')
            with self.assertRaises(ValueError):install_child.registration_writer(public,reg,[],self.roots['control'],fd,family=ex.FAMILY)

    def test_wrong_publication_ddl_and_native_table_refuse(self):
        self.run_init();public=self.public()
        with self.db() as db:
            db.execute('DROP TABLE publication_work')
            from service.persistence import PUBLICATION_V1
            db.execute(PUBLICATION_V1)
        before=self.bytes()
        with self.identity(),self.assertRaises(ValueError):Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))
        self.assertEqual(before,self.bytes())
        with self.db() as db:
            from service.persistence import PUBLICATION_V2
            db.execute('DROP TABLE publication_work');db.execute(PUBLICATION_V2)
            db.execute('CREATE TABLE native_activation(id)')
        before=self.bytes()
        with self.identity(),self.assertRaises(ValueError):Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))
        self.assertEqual(before,self.bytes())

    def test_extra_fresh_namespace_refuses_completion(self):
        self.crash('database')
        retained=self.roots['control']/'core/assets/foreign'
        retained.write_bytes(b'inert retained');retained.chmod(0o600)
        before=self.bytes()
        with self.assertRaises(ValueError):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes())

    def test_open_owner_rechecks_family_before_new_admission(self):
        self.run_init();public=self.public()
        with self.identity(),closing(Core(self.roots['control']/'core',_installed_family=ex.expectation(public,ex.FAMILY))) as core:
            core.db.execute('PRAGMA user_version=1')
            with self.assertRaises(ValueError):core.submit_execution('alice','inert',self.config['contexts']['alice'])
            self.assertEqual(core.db.execute('SELECT * FROM jobs').fetchall(),[])
