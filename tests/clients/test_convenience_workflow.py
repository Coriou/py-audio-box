"""Public HTTP composition without a runtime or model weights."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from clients.http import HTTP
from clients.workflow import enroll, speak
from clients.cli import main
from clients.config import installation, resolve
from contracts.validate import canonical_hash

ROOT = Path(__file__).resolve().parents[2]


def transcript(name):
    rows = json.loads((ROOT/'contracts'/'fixtures'/'v1'/name).read_text())['exchanges']
    return {row['operation']: row['response'] for row in rows}


class WorkflowHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prep = transcript('transcript-voice_prepare.json')
        self.saved_id = 'saved-' + canonical_hash(['job-prepare','candidate-prepare','bruce'])
        self.prep['voice_save']['version']['id'] = self.saved_id
        self.prep['voice_save']['version']['voice_id'] = 'bruce'
        self.prep['voice_save']['voice']['id'] = 'bruce'
        self.prep['voice_save']['voice']['version_ids'][-1] = self.saved_id
        self.gen = transcript('transcript-speech_generate.json')
        self.calls = []
        self.saved = False
        self.waiting = False
        self.bad_bytes = False
        self.unknown = False
        self.lost_save = False
        self.preset_registered = False
        self.real_client = httpx.Client

        def handle(request):
            path = request.url.path
            self.calls.append((request.method, path, request.headers.get('idempotency-key'), bytes(request.content)))
            if path == '/v1/status':
                identity = self.prep['voice_save']['version']['identity']
                body = {'configured_capabilities':{'operations':['voice_prepare','speech_generate'],
                    'profile':{'compatibility':[{'operation':'voice_prepare','identity':identity}]}},
                    'configured_presets':[{'preset':'Ryan'}]}
            elif path == '/v1/assets/impulse': body = self.prep['asset_get']
            elif path == '/v1/operations/voice_prepare':
                payload = json.loads(request.content)
                self.assertEqual(payload['reference_text'], 'Exact reference.')
                self.assertEqual(payload['audition_text'], 'Audition!')
                self.assertEqual((payload['in_tick'],payload['out_tick']), (0,4800))
                if self.unknown: raise httpx.ConnectError('lost response')
                body = self.prep['voice_prepare']
            elif path == '/v1/jobs/job-prepare':
                body = dict(self.prep['job_get'], state='running' if self.waiting else 'succeeded')
            elif path == '/v1/jobs/job-prepare/voice-results': body = self.prep['voice_results']
            elif path == '/v1/voice-names/bruce':
                if not self.saved:
                    return httpx.Response(404,json={'code':'not_found','message':'Unknown voice.','field':'name','action':'Use a saved voice.',
                        'current_revision':0,'changed_ids':[]})
                body = self.prep['voice_save']['voice']
            elif path == '/v1/voice-names/Ryan':
                return httpx.Response(404,json={'code':'not_found','message':'Unknown voice.','field':'name',
                    'action':'Register configured preset.','current_revision':0,'changed_ids':[]})
            elif path == '/v1/voice-names/ryan':
                if not self.preset_registered:
                    return httpx.Response(404,json={'code':'not_found','message':'Unknown voice.','field':'name',
                        'action':'Register configured preset.','current_revision':0,'changed_ids':[]})
                body = dict(self.prep['voice_save']['voice'],id='ryan',version_ids=['vv0'])
            elif path == '/v1/operations/voice_preset_register':
                self.assertEqual(json.loads(request.content)['preset'],'Ryan')
                self.preset_registered = True
                body = {'voice':dict(self.prep['voice_save']['voice'],id='ryan',version_ids=['vv0']),
                        'version':self.gen['voice_version_get']}
            elif path == '/v1/voice-versions/vv0': body = self.gen['voice_version_get']
            elif path == '/v1/voice-versions/' + self.saved_id:
                if not self.saved:
                    return httpx.Response(404,json={'code':'not_found','message':'Unknown version.','field':'voice_version_id',
                        'action':'Use a saved version.','current_revision':0,'changed_ids':[]})
                body = self.prep['voice_save']['version']
            elif path == '/v1/operations/voice_save':
                self.saved = True
                if self.lost_save: raise httpx.ConnectError('save response lost')
                body = self.prep['voice_save']
            elif path == '/v1/operations/speech_generate': body = self.gen['speech_generate']
            elif path == '/v1/jobs/job-speech': body = self.gen['job_get']
            elif path == '/v1/artifacts/artifact-speech': body = self.gen['artifact_get']
            elif path == '/v1/assets/output-speech':
                body = dict(self.gen['asset_get'],sha256=hashlib.sha256(b'\x00'*64).hexdigest())
            elif path == '/v1/assets/output-speech/content':
                start = int(request.url.params['start']); end = int(request.url.params['end_exclusive'])
                data = b'\x00' * 64
                asset = self.gen['asset_get']
                asset['sha256'] = hashlib.sha256(data).hexdigest()
                part = data[start:end]
                if self.bad_bytes: part = b'X' * len(part)
                body = {'schema_version':'1.3.0','asset_id':'output-speech','start':start,'end_exclusive':end,
                    'total_bytes':len(data),'content_sha256':asset['sha256'],
                    'content_base64':base64.b64encode(part).decode()}
            else: raise AssertionError(path)
            return httpx.Response(200,json=body)

        transport = httpx.MockTransport(handle)
        def factory(**kwargs):
            kwargs.pop('trust_env',None)
            return self.real_client(transport=transport,**kwargs)
        self.patch = patch('clients.http.httpx.Client',factory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.http = HTTP('http://test',credential='writer')

    def test_enroll_speak_verified_file_and_replay(self):
        enrolled = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertEqual(enrolled['voice_version_id'],self.saved_id)
        target = Path(self.tmp.name)/'speech.wav'
        spoken = speak(self.http,voice='bruce',text='Hello.',output=target,seconds=0)
        self.assertEqual(spoken['artifact_id'],'artifact-speech')
        self.assertEqual(target.read_bytes(),b'\x00'*64)
        first = [c[2] for c in self.calls if c[1] == '/v1/operations/speech_generate']
        speak(self.http,voice='bruce',text='Hello.',seconds=0)
        second = [c[2] for c in self.calls if c[1] == '/v1/operations/speech_generate']
        self.assertEqual(first,[second[1]])
        self.assertTrue(first[0].startswith('workflow-speak-'))
        again = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertEqual(again['voice_version_id'],self.saved_id)
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_save']),1)

    def test_deadline_unknown_submission_and_bad_bytes(self):
        self.waiting = True
        pending = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertTrue(pending['timed_out'])
        self.assertFalse(self.saved)
        self.waiting = False
        enroll(self.http,name='bruce',asset_id='impulse',job_id=pending['job_id'],
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.unknown = True
        outcome = enroll(self.http,name='another',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertEqual(outcome['state'],'submission_unknown')
        self.assertIn('prepare_key',outcome)
        self.bad_bytes = True
        target = Path(self.tmp.name)/'bad.wav'
        with self.assertRaises(ValueError):
            speak(self.http,voice='bruce',text='Hello.',output=target,seconds=0)
        self.assertFalse(target.exists())

    def test_lost_save_response_is_recovered_by_stable_version_read(self):
        self.lost_save = True
        uncertain = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertEqual(uncertain['state'],'save_unknown')
        self.assertEqual(uncertain['job_id'],'job-prepare')
        recovered = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0)
        self.assertEqual(recovered['voice_version_id'],self.saved_id)
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_save']),1)

    def test_configured_preset_registration_and_generation(self):
        result = speak(self.http,voice='Ryan',text='Hello.',seconds=0)
        self.assertEqual(result['artifact_id'],'artifact-speech')
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_preset_register']),1)

    def test_cli_and_mcp_schema(self):
        with patch.dict('os.environ',{'SPEECH_URL':'http://test','SPEECH_CREDENTIAL':'writer'}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['enroll','--name','bruce','--asset-id','impulse',
                    '--reference-text','Exact reference.','--audition-text','Audition!',
                    '--wait-seconds','0'])
            self.assertEqual(code,0,output.getvalue())
            self.assertEqual(json.loads(output.getvalue())['voice_version_id'],self.saved_id)
        try:
            from clients import mcp
        except ImportError:
            return  # MCP dependency is separately required for parity execution.
        names = {tool.name:tool for tool in mcp.tools()}
        self.assertIn('speech_enroll',names)
        self.assertIn('speech_speak',names)
        self.assertFalse(names['speech_speak'].inputSchema['additionalProperties'])

    def test_explicit_installation_owner_mode_and_precedence(self):
        root = Path(self.tmp.name).resolve()
        (root/'install.json').write_text(json.dumps({'root':str(root),'port':8780}))
        (root/'credential.json').write_text(json.dumps({'version':1,'credentials':[
            {'token':'a'*32,'principal':'local','scopes':['read','production-write']}]}))
        for name in ('install.json','credential.json'):
            (root/name).chmod(0o600)
        self.assertEqual(installation(root),('http://127.0.0.1:8780','a'*32))
        with patch.dict('os.environ',{'SPEECH_URL':'http://override','SPEECH_CREDENTIAL':'b'*32}):
            self.assertEqual(resolve(root=root),('http://127.0.0.1:8780','a'*32))
            self.assertEqual(resolve('http://flag','c'*32,root),('http://flag','c'*32))
        (root/'credential.json').chmod(0o644)
        with self.assertRaises(ValueError): installation(root)

    def test_qwen_local_init_discovery_format(self):
        root = (Path(self.tmp.name)/'qwen').resolve()
        root.mkdir(mode=0o700)
        (root/'client.json').write_text(json.dumps({'version':1,'endpoint':'http://127.0.0.1:8000',
            'credential_file':str(root/'control.key'),'principal':'operator'}))
        (root/'control.key').write_text(json.dumps({'version':1,'credentials':[
            {'token':'a'*32,'principal':'operator','scopes':['read','production-write']}]}))
        (root/'client.json').chmod(0o600); (root/'control.key').chmod(0o600)
        self.assertEqual(installation(root),('http://127.0.0.1:8000','a'*32))
        (root/'control.key').unlink()
        (root/'control.key').symlink_to(root/'client.json')
        with self.assertRaises(ValueError): installation(root)
