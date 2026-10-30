"""Public HTTP composition without a runtime or model weights."""
import base64
import ast
import asyncio
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from clients.http import HTTP
from clients.workflow import enroll, speak
from clients.cli import main
from clients.config import ConfigurationFailure, InstallationFailure, installation, resolve
from contracts.validate import canonical_hash
from service.catalogue import Failure

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
        self.drop_path = None
        self.service_unknown_path = None

        def handle(request):
            path = request.url.path
            self.calls.append((request.method, path, request.headers.get('idempotency-key'), bytes(request.content)))
            if path == self.drop_path: raise httpx.ConnectError('dropped')
            if path == self.service_unknown_path:
                return httpx.Response(503,json={'code':'not_ready','message':'Owner unavailable; mutation outcome may be unknown.',
                    'field':'','action':'Do not automatically retry mutations.',
                    'current_revision':0,'changed_ids':[]})
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
                self.prep['job_get']['idempotency_key'] = request.headers['idempotency-key']
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
            elif path == '/v1/operations/speech_generate':
                self.gen['job_get']['idempotency_key'] = request.headers['idempotency-key']
                body = self.gen['speech_generate']
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
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(enrolled['voice_version_id'],self.saved_id)
        target = Path(self.tmp.name)/'speech.wav'
        spoken = speak(self.http,voice='bruce',text='Hello.',output=target,seconds=5)
        self.assertEqual(spoken['artifact_id'],'artifact-speech')
        self.assertEqual(target.read_bytes(),b'\x00'*64)
        first = [c[2] for c in self.calls if c[1] == '/v1/operations/speech_generate']
        speak(self.http,voice='bruce',text='Hello.',seconds=5)
        second = [c[2] for c in self.calls if c[1] == '/v1/operations/speech_generate']
        self.assertEqual(first,[second[1]])
        self.assertTrue(first[0].startswith('workflow-speak-'))
        again = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(again['voice_version_id'],self.saved_id)
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_save']),1)

    def test_deadline_unknown_submission_and_bad_bytes(self):
        self.waiting = True
        pending = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=0.01)
        self.assertEqual(pending['state'],'timed_out')
        self.assertFalse(self.saved)
        self.waiting = False
        enroll(self.http,name='bruce',asset_id='impulse',job_id=pending['job_id'],
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.unknown = True
        outcome = enroll(self.http,name='another',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(outcome['state'],'prepare_submit_unknown')
        self.assertIn('prepare_key',outcome)
        self.bad_bytes = True
        target = Path(self.tmp.name)/'bad.wav'
        with self.assertRaises(ValueError):
            speak(self.http,voice='bruce',text='Hello.',output=target,seconds=5)
        self.assertFalse(target.exists())

    def test_lost_save_response_is_recovered_by_stable_version_read(self):
        self.lost_save = True
        uncertain = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(uncertain['state'],'voice_save_unknown')
        self.assertEqual(uncertain['job_id'],'job-prepare')
        recovered = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(recovered['voice_version_id'],self.saved_id)
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_save']),1)

    def test_configured_preset_registration_and_generation(self):
        result = speak(self.http,voice='Ryan',text='Hello.',seconds=5)
        self.assertEqual(result['artifact_id'],'artifact-speech')
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_preset_register']),1)

    def test_cli_and_mcp_schema(self):
        with patch.dict('os.environ',{'SPEECH_URL':'http://test','SPEECH_CREDENTIAL':'writer'}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['enroll','--name','bruce','--asset-id','impulse',
                    '--reference-text','Exact reference.','--audition-text','Audition!',
                    '--wait-seconds','5'])
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

    def test_cli_oversized_wait_budget_is_identified_before_http(self):
        with patch.dict('os.environ', {'SPEECH_URL':'http://test', 'SPEECH_CREDENTIAL':'writer'}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['enroll', '--name','bruce', '--asset-id','impulse',
                             '--reference-text','Exact reference.', '--audition-text','Audition!',
                             '--wait-seconds','601'])
        body = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual((body['code'], body['field']), ('invalid_request', 'wait_seconds'))
        self.assertIn('--wait-seconds', body['action'])
        self.assertEqual(self.calls, [])

    def test_wrong_resumed_job_is_rejected_before_results_or_output(self):
        enrolled = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(enrolled['state'],'succeeded')
        count = len([c for c in self.calls if c[1] == '/v1/operations/voice_save'])
        wrong = enroll(self.http,name='different',asset_id='impulse',job_id='job-prepare',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(wrong['code'],'invalid_request')
        self.assertEqual(len([c for c in self.calls if c[1] == '/v1/operations/voice_save']),count)
        out = Path(self.tmp.name)/'wrong.wav'
        wrong_speech = speak(self.http,voice='bruce',text='Changed text.',job_id='job-speech',
            voice_version_id='vv0',output=out,seconds=5)
        self.assertEqual(wrong_speech['code'],'invalid_request')
        self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            enroll(self.http,name='bruce',job_id='job-prepare',
                reference_text='Exact reference.',audition_text='Audition!',seconds=5)

    def test_dropped_reads_preserve_confirmed_handles(self):
        for path, phase in [('/v1/jobs/job-prepare','job_get'),
                            ('/v1/jobs/job-prepare/voice-results','voice_results'),
                            ('/v1/voice-names/bruce','voice_resolve')]:
            self.drop_path = path
            result = enroll(self.http,name='bruce',asset_id='impulse',
                reference_text='Exact reference.',audition_text='Audition!',seconds=5)
            self.assertEqual(result['phase'],phase)
            self.assertEqual(result['job_id'],'job-prepare')
            self.assertIn('prepare_key',result)
        self.drop_path = '/v1/voice-versions/vv0'
        result = speak(self.http,voice='bruce',text='Hello.',voice_version_id='vv0',seconds=5)
        self.assertEqual(result['voice_version_id'],'vv0')
        self.assertEqual(result['phase'],'voice_version_get')
        self.drop_path = '/v1/artifacts/artifact-speech'
        result = speak(self.http,voice='bruce',text='Hello.',voice_version_id='vv0',seconds=5)
        self.assertEqual(result['phase'],'artifact_get')
        self.assertIn('generation_key',result)

    def test_service_unknown_outcome_keeps_prepare_key(self):
        self.service_unknown_path = '/v1/operations/voice_prepare'
        result = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(result['state'],'prepare_submit_unknown')
        self.assertIn('prepare_key',result)
        self.assertEqual(result['asset_id'],'impulse')

    def test_output_never_overwrites_and_read_loss_keeps_artifact(self):
        output = Path(self.tmp.name)/'existing.wav'
        output.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            speak(self.http,voice='bruce',text='Hello.',voice_version_id='vv0',output=output,seconds=5)
        self.assertEqual(output.read_bytes(),b'original')
        self.drop_path = '/v1/assets/output-speech/content'
        result = speak(self.http,voice='bruce',text='Hello.',voice_version_id='vv0',output=Path(self.tmp.name)/'missing.wav',seconds=5)
        self.assertEqual(result['phase'],'download')
        self.assertEqual(result['artifact_id'],'artifact-speech')
        self.assertEqual(result['asset_id'],'output-speech')
        self.assertFalse((Path(self.tmp.name)/'missing.wav').exists())

    def test_late_successful_link_reports_deadline_and_saved_output(self):
        output = Path(self.tmp.name)/'late.wav'
        clock = [time.monotonic()]
        actual_link = os.link

        def late_link(source, destination):
            actual_link(source, destination)
            clock[0] += 6  # The filesystem call completed after the budget.

        with patch('clients.workflow.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch('clients.workflow.os.link', side_effect=late_link):
            result = speak(self.http, voice='bruce', text='Hello.',
                           voice_version_id='vv0', output=output, seconds=5)
        self.assertEqual((result['state'], result['code'], result['phase']),
                         ('timed_out', 'deadline_exceeded', 'download'))
        self.assertEqual(result['output'], str(output))
        self.assertEqual(result['artifact_id'], 'artifact-speech')
        self.assertEqual(result['asset_id'], 'output-speech')
        self.assertTrue(result['saved'])
        self.assertEqual(result['sha256'], hashlib.sha256(b'\x00'*64).hexdigest())
        self.assertEqual(output.read_bytes(), b'\x00'*64)
        self.assertIn('do not repeat the download', result['action'])
        self.assertEqual(list(output.parent.glob('.py-audio-box-*')), [])
        with self.assertRaises(FileExistsError):
            speak(self.http, voice='bruce', text='Hello.',
                  voice_version_id='vv0', output=output, seconds=5)
        self.assertEqual(output.read_bytes(), b'\x00'*64)

    def test_successful_link_retains_result_when_temporary_cleanup_fails(self):
        output = Path(self.tmp.name)/'saved.wav'
        with patch('clients.workflow.Path.unlink', side_effect=OSError('cleanup denied')):
            result = speak(self.http, voice='bruce', text='Hello.',
                           voice_version_id='vv0', output=output, seconds=5)
        self.assertEqual(result['state'], 'succeeded')
        self.assertTrue(result['saved'])
        self.assertEqual(result['output'], str(output))
        self.assertEqual(result['sha256'], hashlib.sha256(b'\x00'*64).hexdigest())
        self.assertEqual(result['artifact_id'], 'artifact-speech')
        self.assertEqual(result['asset_id'], 'output-speech')
        self.assertEqual(output.read_bytes(), b'\x00'*64)
        self.assertEqual(Path(result['temporary_path']).read_bytes(), b'\x00'*64)
        self.assertIn(result['temporary_path'], result['cleanup_warning'])
        self.assertIn('cleanup denied', result['cleanup_warning'])

    def test_late_link_retains_deadline_and_saved_output_when_cleanup_fails(self):
        output = Path(self.tmp.name)/'late-cleanup.wav'
        clock = [time.monotonic()]
        actual_link = os.link

        def late_link(source, destination):
            actual_link(source, destination)
            clock[0] += 6

        with patch('clients.workflow.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch('clients.workflow.os.link', side_effect=late_link), \
                patch('clients.workflow.Path.unlink', side_effect=OSError('cleanup denied')):
            result = speak(self.http, voice='bruce', text='Hello.',
                           voice_version_id='vv0', output=output, seconds=5)
        self.assertEqual((result['state'], result['code'], result['phase']),
                         ('timed_out', 'deadline_exceeded', 'download'))
        self.assertTrue(result['saved'])
        self.assertEqual(result['output'], str(output))
        self.assertEqual(result['sha256'], hashlib.sha256(b'\x00'*64).hexdigest())
        self.assertEqual(result['artifact_id'], 'artifact-speech')
        self.assertEqual(result['asset_id'], 'output-speech')
        self.assertEqual(output.read_bytes(), b'\x00'*64)
        self.assertEqual(Path(result['temporary_path']).read_bytes(), b'\x00'*64)
        self.assertIn(result['temporary_path'], result['cleanup_warning'])

    def test_prepublication_error_remains_primary_when_cleanup_fails(self):
        output = Path(self.tmp.name)/'bad-cleanup.wav'
        self.bad_bytes = True
        with patch('clients.workflow.Path.unlink', side_effect=OSError('cleanup denied')):
            with self.assertRaisesRegex(ValueError, 'Downloaded hash mismatch') as caught:
                speak(self.http, voice='bruce', text='Hello.',
                      voice_version_id='vv0', output=output, seconds=5)
        self.assertFalse(output.exists())
        retained = list(output.parent.glob('.py-audio-box-*'))
        self.assertEqual(len(retained), 1)
        self.assertIn(str(retained[0]), caught.exception.__notes__[0])
        self.assertIn('cleanup denied', caught.exception.__notes__[0])

    def test_prepublication_read_loss_keeps_outcome_and_cleanup_warning(self):
        output = Path(self.tmp.name)/'read-loss.wav'
        with patch('clients.workflow.base64.b64decode', side_effect=ConnectionError('read lost')), \
                patch('clients.workflow.Path.unlink', side_effect=OSError('cleanup denied')):
            result = speak(self.http, voice='bruce', text='Hello.',
                           voice_version_id='vv0', output=output, seconds=5)
        self.assertEqual((result['state'], result['phase']), ('read_unavailable', 'download'))
        self.assertEqual(result['artifact_id'], 'artifact-speech')
        self.assertEqual(result['asset_id'], 'output-speech')
        self.assertFalse(output.exists())
        self.assertTrue(Path(result['temporary_path']).exists())
        self.assertIn(result['temporary_path'], result['cleanup_warning'])

    def test_changed_file_rejects_confirmed_asset_before_submit(self):
        import wave
        wav = Path(self.tmp.name)/'changed.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(48000)
            stream.writeframes(b'\x00\x00'*4800)
        result = enroll(self.http,name='bruce',asset_id='impulse',file=wav,
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(result['code'],'invalid_request')
        self.assertFalse(any(c[1] == '/v1/operations/voice_prepare' for c in self.calls))

    def test_submission_receipt_hash_must_match_job(self):
        self.prep['voice_prepare']['input_sha256'] = 'f'*64
        result = enroll(self.http,name='bruce',asset_id='impulse',
            reference_text='Exact reference.',audition_text='Audition!',seconds=5)
        self.assertEqual(result['code'],'invalid_request')
        self.assertFalse(any(c[1] == '/v1/operations/voice_save' for c in self.calls))

    def test_zero_budget_performs_no_network_or_publication(self):
        result = speak(self.http,voice='bruce',text='Hello.',output=Path(self.tmp.name)/'none',seconds=0)
        self.assertEqual(result['state'],'timed_out')
        self.assertEqual(self.calls,[])

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

    def test_discovered_credential_is_bound_to_installation_endpoint(self):
        root = Path(self.tmp.name).resolve()
        endpoint = 'http://127.0.0.1:8780'
        token = 'a'*32
        (root/'install.json').write_text(json.dumps({'root':str(root),'port':8780}))
        (root/'credential.json').write_text(json.dumps({'version':1,'credentials':[
            {'token':token,'principal':'local','scopes':['read','production-write']}]}))
        for name in ('install.json','credential.json'):
            (root/name).chmod(0o600)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve(root=root), (endpoint,token))
            self.assertEqual(resolve(endpoint,None,root), (endpoint,token))
            for different in ('http://192.0.2.10:8780', 'http://127.0.0.1:8781',
                              'https://127.0.0.1:8780', endpoint+'/', endpoint+'/other',
                              endpoint+'?other=1', endpoint+'#other'):
                with self.subTest(different=different):
                    with self.assertRaises(ConfigurationFailure) as caught:
                        resolve(different,None,root)
                    self.assertNotIn(token,str(caught.exception))
                    self.assertEqual(caught.exception.body['code'], 'invalid_request')
                    self.assertEqual(caught.exception.body['field'], 'url')
                    self.assertEqual(caught.exception.body['action'],
                                     'Use the configured installation URL or supply an explicit credential for a different URL.')
            self.assertEqual(resolve('http://192.0.2.10:8780','b'*32,root),
                             ('http://192.0.2.10:8780','b'*32))
            self.assertEqual(resolve(None,'b'*32,root), (endpoint,'b'*32))
            with patch.dict(os.environ, {'SPEECH_INSTALLATION':str(root)}):
                self.assertEqual(resolve(), (endpoint,token))
                with patch.dict(os.environ, {'SPEECH_CREDENTIAL':'c'*32}):
                    self.assertEqual(resolve(), (endpoint,'c'*32))
                with self.assertRaises(ValueError):
                    resolve('http://192.0.2.10:8780')
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['--installation',str(root),'--url',
                                           'http://192.0.2.10:8780','status']), 2)
                cli_error = json.loads(output.getvalue())
                self.assertEqual(cli_error['code'], 'invalid_request')
                self.assertEqual(cli_error['message'], ConfigurationFailure.message)
                self.assertEqual(cli_error['action'], ConfigurationFailure.action)
                self.assertNotIn(token, output.getvalue())
                # Execute the MCP handler from source without importing the unavailable MCP SDK.
                source = ast.parse((ROOT/'clients'/'mcp.py').read_text())
                handler = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef)
                               and node.name == 'call_tool')
                handler.decorator_list = []
                module = ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[]))
                namespace = {'Failure':Failure, 'ConfigurationFailure':ConfigurationFailure,
                             'InstallationFailure':InstallationFailure,
                             'ValidationError':ValueError, 'ContractError':TypeError,
                             'result':lambda body,error=False:(body,error)}
                exec(compile(module, str(ROOT/'clients'/'mcp.py'), 'exec'), namespace)
                with patch.dict(os.environ, {'SPEECH_URL':'http://192.0.2.10:8780'}):
                    mcp_error, is_error = asyncio.run(namespace['call_tool']('speech_status', {}))
                self.assertTrue(is_error)
                self.assertEqual(mcp_error, cli_error)
                with patch.dict(os.environ, {'SPEECH_URL':endpoint}):
                    self.assertEqual(resolve(), (endpoint,token))
                with patch.dict(os.environ, {'SPEECH_URL':'http://192.0.2.10:8780'}):
                    with self.assertRaises(ValueError) as caught:
                        resolve()
                    self.assertNotIn(token,str(caught.exception))
                    self.assertEqual(resolve(credential='b'*32),
                                     ('http://192.0.2.10:8780','b'*32))
                    with patch.dict(os.environ, {'SPEECH_CREDENTIAL':'c'*32}):
                        self.assertEqual(resolve(), ('http://192.0.2.10:8780','c'*32))

    def test_qwen_local_init_discovery_format(self):
        root = (Path(self.tmp.name)/'qwen').resolve()
        root.mkdir(mode=0o700)
        (root/'client.json').write_text(json.dumps({'version':1,'endpoint':'http://127.0.0.1:8000',
            'credential_file':str(root/'control.key'),'principal':'operator'}))
        (root/'control.key').write_text(json.dumps({'version':1,'credentials':[
            {'token':'a'*32,'principal':'operator','scopes':['read','production-write']}]}))
        (root/'client.json').chmod(0o600); (root/'control.key').chmod(0o600)
        self.assertEqual(installation(root),('http://127.0.0.1:8000','a'*32))
        self.assertEqual(resolve(root=root),('http://127.0.0.1:8000','a'*32))
        with self.assertRaises(ValueError):
            resolve('http://192.0.2.10:8000',None,root)
        (root/'control.key').unlink()
        (root/'control.key').symlink_to(root/'client.json')
        with self.assertRaises(ValueError): installation(root)

    def test_default_qwen_root_only_when_configured_without_overrides(self):
        root=(Path(self.tmp.name)/'default-qwen').resolve()
        with patch('service.qwen_local_config.DEFAULT_ROOT',root), patch.dict(os.environ,{},clear=True):
            self.assertEqual(resolve(),(None,None))
            root.mkdir(mode=0o700)
            (root/'client.json').write_text(json.dumps({'version':1,'endpoint':'http://127.0.0.1:8000',
                'credential_file':str(root/'control.key'),'principal':'operator'}))
            (root/'control.key').write_text(json.dumps({'version':1,'credentials':[
                {'token':'a'*32,'principal':'operator','scopes':['read','production-write']}]}))
            (root/'client.json').chmod(0o600);(root/'control.key').chmod(0o600)
            self.assertEqual(resolve(),('http://127.0.0.1:8000','a'*32))
            with patch.dict(os.environ,{'SPEECH_URL':'http://192.0.2.10:8000'}):
                self.assertEqual(resolve(),('http://192.0.2.10:8000',None))
            with patch.dict(os.environ,{'SPEECH_CREDENTIAL':'b'*32}):
                self.assertEqual(resolve(),(None,'b'*32))
            self.assertEqual(resolve('http://192.0.2.10:8000'),('http://192.0.2.10:8000',None))
            self.assertEqual(resolve(credential='b'*32),(None,'b'*32))
            with patch.dict(os.environ,{'SPEECH_INSTALLATION':str(root),'SPEECH_URL':'http://192.0.2.10:8000'}):
                with self.assertRaises(ConfigurationFailure): resolve()
