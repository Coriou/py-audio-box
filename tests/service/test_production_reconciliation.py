"""Focused H0/H2 inventory integration; no cleanup/repair authority."""
import json
from pathlib import Path
import tempfile
import unittest
from service.api import create_voice_app
from service.core import Core
from service.dispatcher import Dispatcher
from service.reconciliation import Reconciliation
from service.ingest import ingest_local
from test_production_api import ProductionAPIFixture, audio


class ProductionReconciliationTests(ProductionAPIFixture,unittest.TestCase):
    def test_saved_voice_restart_retains_owned_reference_and_admits(self):
        _,_,receipt,result=self.prepared()
        saved=self.submit('voice_save',dict(job_id=receipt['job_id'],candidate_id=result['candidates'][0]['id'],voice_id='voice',expected_revision=0),'save')
        reference=saved['version']['reference_asset_ids'][0]
        self.dispatcher.close()
        self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=self.profiles,production_contexts=self.contexts)
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        status=self.dispatcher.call('status')
        self.assertFalse(status['integrity_fault'],status)
        self.assertTrue((self.root/'assets'/reference).is_file())
        self.assertEqual(self.dispatcher.call('voice_version','alice',saved['version']['id']),saved['version'])
        generated=self.submit('speech_generate',dict(self.generate['plan']['request'],voice_version_id=saved['version']['id']),'after-restart')
        self.wait(lambda:self.dispatcher.call('job',generated['job_id'])['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')


class CatalogueSchemaTests(unittest.TestCase):
    def test_unknown_table_trigger_format_and_malformed_version_fail_closed(self):
        for defect in ('table','trigger','format','version','foreign','request'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                core=Core(Path(tmp)/'core')
                try:
                    if defect=='table':core.db.execute('CREATE TABLE surprise(asset_id TEXT)')
                    elif defect=='trigger':core.db.execute('CREATE TRIGGER surprise AFTER INSERT ON catalogue_voices BEGIN SELECT 1; END')
                    elif defect=='format':core.db.execute('UPDATE voice_catalogue_format SET revision=99')
                    elif defect=='request':core.db.execute("INSERT INTO catalogue_requests VALUES('alice','unknown','key',?, '{}')",('0'*64,))
                    else:
                        core.db.execute("INSERT INTO catalogue_voices VALUES(1,'alice','voice',1,0)")
                        version=json.loads(Path('contracts/fixtures/v1/voice-catalogue.json').read_text())['register']['version']
                        version.update(id='version',voice_id='voice')
                        if defect=='version':version['unsupported_field']='asset'
                        else:
                            asset=ingest_local(core,'bob',audio())
                            version.update(origin='clone',reference_asset_ids=[asset['id']],reference_asset_sha256={asset['id']:asset['sha256']},reference_text='reference',configuration=dict(mode='qwen_clone',reference_asset_id=asset['id'],reference_text='reference',conditioning='icl'))
                            version['settings']['kind']='qwen_clone'
                            from contracts import validate as oracle
                            version['settings_sha256']=oracle.canonical_hash(version['settings'])
                            oracle.validate('voice_version',version)
                        core.db.execute('INSERT INTO catalogue_versions VALUES(1,?,?,?,?)',('alice','version','voice',json.dumps(version)))
                    inventory=Reconciliation(core)
                    summary=inventory.reconcile_startup()
                    self.assertTrue(summary['blocked'],summary)
                    self.assertFalse(summary['cleanup_allowed'])
                finally:core.close()

if __name__=='__main__':unittest.main()
