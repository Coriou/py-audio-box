"""Pinned renderer and retained-plan refusal; no model qualification."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from jsonschema.exceptions import ValidationError
from contracts import validate as oracle
from lib.spoken_text import canonical_json, freeze, speech_text

FIXTURE = Path('contracts/fixtures/v1/transcript-speech_generate-spoken-v1.json')


def fixture():
    return json.loads(FIXTURE.read_text())


class SpokenPlanTests(unittest.TestCase):
    def test_accepted_amount_exact_hash_and_plan(self):
        f = fixture(); plan = f['context']['plan']
        oracle.validate('worker_plan', plan)
        self.assertEqual(plan['speech_text']['render_sha256'], '98f4fd6218346d02ffdf39c0e093992e1a3762b9803ff849ffbddf504e79cd9e')
        self.assertEqual(speech_text(plan), 'Le budget est de mille deux cent cinquante euros et cinquante centimes, sans dépasser ce montant.')
        self.assertEqual(plan['request']['text'], plan['speech_text']['original_text'])
        oracle.stored_context(f['job'], f['context'])
        self.assertEqual(f['message']['plan'], plan)
        oracle.worker_message(f['message'], f['job'], f['authorized_assets'],
            transport_owner='worker1', context=f['context'])
        self.assertEqual(f['artifact']['manifest_sha256'],
            oracle.canonical_hash(f['verified_manifest']))
        oracle.publication(f['job'], f['job']['attempts'][-1], f['artifact'],
            f['present_hashes'], transport_owner='worker1', context=f['context'],
            manifest=f['verified_manifest'], verified_result=f['message'],
            allowed_assets=f['authorized_assets'])

    def test_three_accepted_r414_hashes(self):
        from lib.spoken_text import render
        cases = [({'annotations': [{'amount': '1250.50', 'currency': 'EUR', 'end': 31, 'literal': '1 250,50 euros', 'start': 17, 'type': 'amount'}], 'locale': 'fr-FR', 'original_text': 'Le budget est de 1 250,50 euros, sans dépasser ce montant.', 'profile': 'numeric-v1'}, '98f4fd6218346d02ffdf39c0e093992e1a3762b9803ff849ffbddf504e79cd9e'), ({'annotations': [{'amount': '1250.50', 'currency': 'EUR', 'end': 34, 'literal': '1,250.50 euros', 'start': 20, 'type': 'amount'}], 'locale': 'en-GB', 'original_text': 'The total budget is 1,250.50 euros, with no additional charge.', 'profile': 'numeric-v1'}, '6915e9c6521ffd30792d742299be0e9160454803b9df9ca5118bf40dc8a244a3'), ({'annotations': [{'end': 184, 'literal': 'A-2047', 'policy': 'spell-letters-digits', 'start': 178, 'type': 'identifier'}, {'end': 195, 'literal': 'B', 'policy': 'spell-letters-digits', 'start': 194, 'type': 'identifier'}], 'locale': 'fr-FR', 'original_text': 'La SAS emploie une équipe dédiée à l’IA et à la recherche. Je ne souhaite pas annuler le rendez-vous. Il reste exactement trois dossiers, et non trente. Le numéro de dossier est A-2047, version B.', 'profile': 'numeric-v1'}, '5e457dc89c10e11c2ed961b81373f7e2e0d46bb2fd44290e2fbbaa399e6ec1a2')]
        for request,digest in cases:
            self.assertEqual(render(request)["render_sha256"],digest)

    def test_raw_multiline_byte_identical(self):
        p = fixture()['context']['plan']; p.pop('speech_text'); p['request'].pop('text_rendering')
        p['request']['text'] = '  Exact\n café!\t '
        before = canonical_json(p); freeze(p)
        self.assertEqual(canonical_json(p), before)
        self.assertEqual(speech_text(p), p['request']['text'])

    def test_every_record_field_and_rehashed_substitution_refused(self):
        p = fixture()['context']['plan']
        for field in p['speech_text']:
            for remove in (False, True):
                q = copy.deepcopy(p)
                if remove: del q['speech_text'][field]
                else: q['speech_text'][field] = [] if field == 'annotations' else 'substitution'
                with self.subTest(field=field, remove=remove), self.assertRaises((ValueError, ValidationError)): speech_text(q)
        q = copy.deepcopy(p); q['speech_text']['spoken_text'] = 'substituted'
        record = q['speech_text']; record.pop('render_sha256')
        record['render_sha256'] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
        with self.assertRaises((ValueError, ValidationError)): speech_text(q)
        for field in ('speech_text', 'text_rendering'):
            q = copy.deepcopy(p)
            if field == 'speech_text': q.pop(field)
            else: q['request'].pop(field)
            with self.assertRaises((ValueError, ValidationError)): speech_text(q)

    def test_closed_semantics(self):
        base = fixture()['context']['plan']['request']
        for field,value in [('version',True),('version',1.0),('version',2),('locale','fr'),('profile','numeric-v2'),('spoken_text','injected'),('render_sha256','0'*64)]:
            q = copy.deepcopy(base); q['text_rendering'][field] = value
            with self.subTest(field=field,value=value), self.assertRaises((ValueError, ValidationError)): oracle.validate('speech_generate_request',q)
        for field,value in [('start',False),('end',31.0),('literal','stale'),('extra',1)]:
            q=copy.deepcopy(base);q['text_rendering']['annotations'][0][field]=value
            with self.assertRaises((ValueError, ValidationError)): oracle.validate('speech_generate_request',q)
        for count in (2,65):
            q=copy.deepcopy(base);q['text_rendering']['annotations'] *= count
            with self.assertRaises((ValueError, ValidationError)): oracle.validate('speech_generate_request',q)

    def test_expanded_char_bound_and_locale_agreement(self):
        p=fixture()['context']['plan']; p['resolved']['profile']['limits']['text_chars']=len(p['request']['text'])
        with self.assertRaisesRegex(ValueError,'profile_limit'): speech_text(p)
        p=fixture()['context']['plan'];p['resolved']['settings']['language']='English'
        with self.assertRaisesRegex(ValueError,'language'): speech_text(p)
        p=fixture()['context']['plan'];p['request']['text_rendering']['annotations'][0]['amount']='1250.5'
        freeze(p)
        self.assertEqual(p['speech_text']['spoken_text'],fixture()['context']['plan']['speech_text']['spoken_text'])
        self.assertNotEqual(oracle.plan_hash(p),oracle.plan_hash(fixture()['context']['plan']))

    def test_record_pairing_in_closed_plan_schema(self):
        from jsonschema import Draft202012Validator
        schema={**oracle.SCHEMA,'$ref':'#/$defs/worker_plan'}
        p=fixture()['context']['plan']
        for remove in ('speech_text','text_rendering'):
            q=copy.deepcopy(p)
            if remove=='speech_text':q.pop(remove)
            else:q['request'].pop(remove)
            with self.assertRaises(ValidationError):Draft202012Validator(schema).validate(q)
        q=copy.deepcopy(p);q['request']['spoken_text']='injected'
        with self.assertRaises(ValidationError):oracle.validate('worker_plan',q)
