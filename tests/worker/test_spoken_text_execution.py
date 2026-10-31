"""Injected child text capture; no model imports, processes, or hardware proof."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
from contracts import validate as oracle
from lib.spoken_text import freeze, speech_text
from worker import native_child, runtime_child
from worker.transport import canonical


class Captured(Exception): pass


class SpokenExecutionTests(unittest.TestCase):
    def context(self):
        return json.loads(Path('contracts/fixtures/v1/transcript-speech_generate-spoken-v1.json').read_text())

    def test_linux_child_passes_retained_text(self):
        f=self.context();context=f['context'];job=f['job']
        # Trusted synthetic bundle input for the injected engine; no model path.
        context['execution']={'bundles':{}}
        from datetime import datetime,timezone,timedelta
        job['deadline']=(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()
        spec=SimpleNamespace(engine='kokoro',sample_rate=48000)
        config=dict(job=job,context=context,allowed_assets=[],profiles=f['profiles'],secret='a'*64,control_uid=0,gid=0)
        seen=[]
        def execute(spec,text,*args,**kwargs): seen.append(text);raise Captured()
        with patch('worker.runtime_protocol.execution',return_value=spec),patch('worker.transport.authenticate'),patch('worker.transport.receive',return_value={'kind':'execute'}),patch.object(oracle,'worker_message'),patch('worker.engines.kokoro.execute',side_effect=execute):
            with self.assertRaises(Captured):runtime_child.run_generation(None,config,Path('/unused'),time.monotonic()+20)
        self.assertEqual(seen,[context['plan']['speech_text']['spoken_text']])
        self.assertEqual(context['plan']['request']['text'],context['plan']['speech_text']['original_text'])

    def test_native_child_passes_retained_text_and_prepare_stays_raw(self):
        from tests.worker.test_native_supervisor import NativeSupervisorTests
        # Only its fixture setup/snapshot builder; no child launch occurs.
        harness=NativeSupervisorTests('runTest');harness.setUp();self.addCleanup(harness.doCleanups)
        snapshot=harness.snapshot
        p=snapshot[1]['plan'];p['request'].update(text='A-2047',text_rendering=dict(profile='numeric-v1',version=1,locale='fr-FR',annotations=[dict(type='identifier',start=0,end=6,literal='A-2047',policy='spell-letters-digits')]))
        # Fixture language is retained explicitly; choose its matching locale.
        p['request']['text_rendering']['locale']='fr-FR' if harness.e.language=='French' else 'en-GB'
        freeze(p);snapshot[0]['input_sha256']=oracle.plan_hash(p)
        config=dict(token='a'*32,binding=hashlib.sha256(canonical(snapshot)).hexdigest(),snapshot=snapshot,execution=harness.e.identity(),bundles={},references={},inventory={},conditioning=None,cutoff=time.monotonic()+20)
        seen=[]
        def capture(spec,text,*a,**kw):seen.append(text);raise Captured()
        with patch.object(native_child,'wire_receive',return_value=config),patch('worker.engines.qwen_execution.inventory_from_wire',return_value=harness.inventory):
            with self.assertRaises(Captured):native_child.run_socket(None,inference=capture,guard=False)
        self.assertEqual(seen,[p['speech_text']['spoken_text']])
        # The same entry point must leave preparation audition/reference roles alone.
        p['operation']='voice_prepare';p['request']={'audition_text':' Exact\n audition! ','reference_text':'Reference stays exact.'};p.pop('speech_text')
        config['binding']=hashlib.sha256(canonical(snapshot)).hexdigest()
        with patch.object(native_child,'wire_receive',return_value=config),patch('worker.native_protocol.native_execution',return_value=harness.e),patch.object(oracle,'stored_context'),patch('worker.engines.qwen_execution.inventory_from_wire',return_value=harness.inventory):
            with self.assertRaises(Captured):native_child.run_socket(None,inference=capture,guard=False)
        self.assertEqual(seen[-1],' Exact\n audition! ')
        self.assertEqual(p['request']['reference_text'],'Reference stays exact.')

    def test_native_and_linux_relations_reject_tamper_before_inference(self):
        from tests.worker.test_native_supervisor import NativeSupervisorTests
        h=NativeSupervisorTests('runTest');h.setUp();self.addCleanup(h.doCleanups)
        context=h.snapshot[1];p=context['plan'];p['request'].update(text='B',text_rendering=dict(profile='numeric-v1',version=1,locale='fr-FR' if h.e.language=='French' else 'en-GB',annotations=[dict(type='identifier',start=0,end=1,literal='B',policy='spell-letters-digits')]))
        freeze(p)
        from worker.native_protocol import native_execution
        native_execution(context)
        p['speech_text']['spoken_text']='substitution'
        with self.assertRaises(ValueError): native_execution(context)
        from worker.runtime_protocol import execution
        with self.assertRaises(ValueError): execution(context)

    def test_expanded_utf8_budget_native(self):
        from dataclasses import replace
        from tests.worker.test_native_supervisor import NativeSupervisorTests
        from worker.native_protocol import native_execution
        from worker.transport import Refused
        h=NativeSupervisorTests('runTest');h.setUp();self.addCleanup(h.doCleanups)
        context=h.snapshot[1];p=context['plan']
        p['request'].update(text='B',text_rendering=dict(profile='numeric-v1',version=1,locale='en-GB',annotations=[dict(type='identifier',start=0,end=1,literal='B',policy='spell-letters-digits')]))
        freeze(p)
        context['execution']['descriptor']=replace(h.e,max_text_bytes=1).identity()
        self.assertEqual(len(p['request']['text'].encode()),1)
        self.assertGreater(len(speech_text(p).encode()),1)
        with self.assertRaisesRegex(Refused,'execution_budget_mismatch'):native_execution(context)
