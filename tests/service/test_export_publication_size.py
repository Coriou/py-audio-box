"""r371 inert fixtures, completed for r373; no converter/application child."""
import asyncio
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import test_audio_export as export_fixture
from worker.audio_export import execution, pcm, provenance, verify_output, MAX_OUTPUT

audio=export_fixture.audio


class ExportSizeDescriptorTests(unittest.TestCase):
    def test_four_second_stereo_pcm24_is_admitted_above_one_mib(self):
        context=json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())['context']
        source=context['plan']['assets'][0]
        source.update(pcm(audio(frames=96000)))
        spec=execution(context)
        raw=audio(rate=48000,channels=2,width=3,frames=192000)
        self.assertEqual(len(raw),1152044)
        self.assertGreater(len(raw),1024**2)
        self.assertEqual(spec.max_output_bytes,1156102)
        self.assertLess(len(raw),spec.max_output_bytes)
        verify_output(context,raw,dict(pcm(raw),provenance=provenance(context['plan'])))

    def test_descriptor_stays_bounded_without_allocating_ceiling_bytes(self):
        context=json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())['context']
        self.assertEqual(MAX_OUTPUT,37748736)
        rate=context['plan']['assets'][0]['sample_rate']
        for frames in (rate*60+1,10**9):
            changed=copy.deepcopy(context)
            changed['plan']['assets'][0]['frames']=frames
            with self.assertRaises(ValueError):execution(changed)


class ExportPublicationSizeTests(unittest.TestCase):
    def test_large_public_master_verified_download_reopen_replay_reconciliation(self):
        self._journey()

    def test_migrated_empty_v1_large_public_master(self):
        self._journey(migrated=True)

    def _journey(self, *, migrated=False):
        from clients.cli import main
        from service.core import Core
        from service.reconciliation import Reconciliation
        fixture=export_fixture.AudioExportTests()
        self.addCleanup(fixture.doCleanups)
        if migrated:
            with patch('service.dispatcher.Core',side_effect=lambda root,**kw:Core(root,**dict(kw,_publication_creation=1))):fixture.setUp()
            fixture.dispatcher.close()
            from service.publication_format import migrate
            self.assertEqual(migrate(fixture.root.resolve(),expect_v1=True),(0,'migrated'))
            fixture.start()
        else:fixture.setUp()
        source=audio(frames=96000)
        response=fixture.request('POST','/v1/assets',content=source)
        self.assertEqual(response.status_code,200,response.text)
        asset=response.json()
        request=dict(asset_id=asset['id'],rate=48000,channels=2,loudness=None)
        receipt=fixture.submit('audio_export',request,'large-master')
        job=fixture.terminal(receipt)
        response=fixture.request('GET','/v1/artifacts/'+job['artifact_ids'][0])
        self.assertEqual(response.status_code,200,response.text)
        artifact=response.json();output_id=artifact['asset_ids'][0]
        output=fixture.request('GET','/v1/assets/'+output_id).json()
        self.assertEqual((output['frames'],output['sample_rate'],output['channels'],output['media_subtype']),
                         (192000,48000,2,'PCM_24'))
        self.assertEqual(artifact['audio_export']['settings'],request)

        class ASGIClient:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def request(self,method,path,**kwargs):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fixture.app),base_url='http://test') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
            def get(self,path,**kwargs):return self.request('GET',path,**kwargs)

        target=Path(fixture.tmp.name)/'large-master.wav'
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'writer'}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['asset','content',output_id,'--output',str(target)]),0)
        self.assertEqual(target.stat().st_size,1152044)
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),output['sha256'])
        self.assertEqual((fixture.root/'assets'/asset['id']).read_bytes(),source)
        status=fixture.dispatcher.call('status')
        self.assertFalse(status['persistence_fault']);self.assertFalse(status['quarantined'])
        self.assertFalse(any(r['state']=='executing' for r in status['reservations']))
        fixture.dispatcher.close()
        with contextlib.closing(Core(fixture.root)) as core:
            scan=Reconciliation(core,fixture.stage).reconcile_startup()
            self.assertTrue(scan['complete'],scan);self.assertFalse(scan['blocked'],scan)
            self.assertFalse(scan['reasons'],scan)
        fixture.start(exports=False)
        self.assertEqual(fixture.submit('audio_export',request,'large-master'),receipt)
        self.assertFalse(fixture.dispatcher.call('status')['integrity_fault'])
        response=fixture.request('GET','/v1/artifacts/'+artifact['id'])
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json(),artifact)
        self.assertEqual(len(fixture.runtime.authorities),0)
