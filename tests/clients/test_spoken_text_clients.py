"""Closed named operation option; duplicate/nonfinite JSON never reaches HTTP."""
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from jsonschema.exceptions import ValidationError
from unittest.mock import patch
from clients.cli import main, strict_json
from clients import mcp
from contracts import validate as oracle


class SpokenClientTests(unittest.TestCase):
    def test_strict_json_nested_duplicate_and_nonfinite(self):
        for raw in ('{"text_rendering":{"version":1,"version":1}}','{"x":NaN}','{"x":Infinity}','{"x":-Infinity}'):
            with self.assertRaises((ValueError, ValidationError)): strict_json(raw)
            with patch('clients.cli.HTTP') as http,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['speech','generate','--request',raw,'--idempotency-key','key']),2)
                http.return_value.call.assert_not_called()

    def test_cli_preserves_option_and_original_text(self):
        request=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate-spoken-v1.json').read_text())['context']['plan']['request']
        with patch('clients.cli.HTTP') as http,contextlib.redirect_stdout(io.StringIO()):
            http.return_value.call.return_value={}
            self.assertEqual(main(['speech','generate','--request',json.dumps(request),'--idempotency-key','key']),0)
            http.return_value.call.assert_called_once_with('speech_generate',request,key='key')

    def test_mcp_uses_same_closed_nested_schema(self):
        from jsonschema import Draft202012Validator
        tool=next(t for t in mcp.tools() if t.name=='speech_speech_generate')
        request=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate-spoken-v1.json').read_text())['context']['plan']['request']
        Draft202012Validator(tool.inputSchema).validate(dict(request,idempotency_key='key'))
        oracle.validate('speech_generate_request',request)
        for extra in ('spoken_text','render_sha256','unknown'):
            changed=copy.deepcopy(request);changed['text_rendering'][extra]='injected'
            with self.assertRaises((ValueError, ValidationError)):oracle.validate('speech_generate_request',changed)
            with self.assertRaises(ValidationError):Draft202012Validator(tool.inputSchema).validate(dict(changed,idempotency_key='key'))

    def test_mcp_forwards_exact_option_and_rejects_semantic_drift(self):
        request=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate-spoken-v1.json').read_text())['context']['plan']['request']
        with patch('clients.mcp.HTTP') as http:
            http.return_value.call.return_value={}
            result=asyncio.run(mcp.call_tool('speech_speech_generate',dict(request,idempotency_key='key')))
            self.assertFalse(result.isError)
            http.return_value.call.assert_called_once_with('speech_generate',request,key='key')
        for change in ({'version':True},{'version':1.0},{'version':2},{'locale':'fr'},{'annotations':request['text_rendering']['annotations']*2}):
            changed=copy.deepcopy(request);changed['text_rendering'].update(change)
            with patch('clients.mcp.HTTP') as http:
                result=asyncio.run(mcp.call_tool('speech_speech_generate',dict(changed,idempotency_key='key')))
                self.assertTrue(result.isError)
                http.return_value.call.assert_not_called()
