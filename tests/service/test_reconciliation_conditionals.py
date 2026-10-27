"""Typed inventory traversal of retained conditional speech plans."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from contracts import validate as oracle
from service.reconciliation import Incomplete, typed_references
from lib.spoken_text import freeze


class ConditionalReferencesTests(unittest.TestCase):
    def scan(self, fixture, document=None):
        scanner=SimpleNamespace(reasons=set())
        job=fixture['job']
        data={'jobs':[dict(id=job['id'],job=json.dumps(job),
                           context=json.dumps(fixture['context']))]}
        result=typed_references(scanner,data,None,None,document or oracle.SCHEMA)
        return scanner.reasons,result[0]

    def fixture(self, spoken=True):
        suffix='-spoken-v1' if spoken else ''
        transcript=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate'+suffix+'.json').read_text())
        return {key:transcript[key] for key in ('job','context')}

    def test_retained_plain_and_spoken_plans_keep_references(self):
        for spoken in (False,True):
            with self.subTest(spoken=spoken):
                f=self.fixture(spoken)
                oracle.validate('worker_plan',f['context']['plan'])
                reasons,refs=self.scan(f)
                self.assertFalse(reasons,reasons)
                for asset in f['context']['plan']['assets']:
                    self.assertIn(asset['id'],refs)

    def test_conditional_malformed_and_unknown_references_refused(self):
        for defect in ('missing_speech','missing_rendering','unknown'):
            with self.subTest(defect=defect):
                f=self.fixture();plan=f['context']['plan']
                if defect=='missing_speech':plan.pop('speech_text')
                elif defect=='missing_rendering':plan['request'].pop('text_rendering')
                else:plan['foreign_asset_id']='foreign'
                self.assertTrue(self.scan(f)[0])

    def test_both_active_consequences_validate_and_collect_references(self):
        # A conditional may constrain existing reference fields; both branches
        # must be walked even though predicates themselves declare no assets.
        document=copy.deepcopy(oracle.SCHEMA)
        branch=document['$defs']['worker_plan']['oneOf'][2]
        branch['properties']['reference_asset_id']={'type':'string'}
        branch['allOf'].append(dict(
            **{'if':{'required':['speech_text']}},
            then={'properties':{'reference_asset_id':{'const':'spoken-reference'}}},
            **{'else':{'properties':{'reference_asset_id':{'const':'plain-reference'}}}}))
        for spoken in (False,True):
            with self.subTest(spoken=spoken):
                f=self.fixture(spoken)
                reference='spoken-reference' if spoken else 'plain-reference'
                f['context']['plan']['reference_asset_id']=reference
                reasons,refs=self.scan(f,document)
                self.assertFalse(reasons,reasons)
                self.assertIn(reference,refs)
                # An invalid active consequence cannot be ignored.
                bad=copy.deepcopy(document)
                consequence='then' if spoken else 'else'
                bad['$defs']['worker_plan']['oneOf'][2]['allOf'][-1][consequence]={'required':['foreign_asset_id']}
                self.assertTrue(self.scan(f,bad)[0])

    def test_unsupported_constraint_applicator_fails_closed(self):
        document=copy.deepcopy(oracle.SCHEMA)
        document['$defs']['worker_plan']['oneOf'][2]['allOf'].append(
            {'patternProperties':{'foreign':{'type':'string'}}})
        with self.assertRaisesRegex(Incomplete,'unknown_reference_schema'):
            self.scan(self.fixture(),document)

    def test_amount_alternatives_and_closed_enclosing_properties(self):
        for location in ('request', 'speech_text'):
            for variant in ('amount', 'minor_units', 'both', 'neither', 'illegal', 'reference', 'unpaired_minor_units'):
                with self.subTest(location=location, variant=variant):
                    f=self.fixture()
                    plan=f['context']['plan']
                    rendering=(plan['request']['text_rendering'] if location=='request'
                               else plan['speech_text'])
                    annotation=rendering['annotations'][0]
                    if variant=='minor_units':
                        annotation.pop('amount')
                        annotation['minor_units']=125050
                        if location=='speech_text':
                            plan['request']['text_rendering']['annotations']=copy.deepcopy(rendering['annotations'])
                        freeze(plan)
                        f['job']['input_sha256']=oracle.plan_hash(plan)
                    elif variant=='unpaired_minor_units':
                        annotation.pop('amount')
                        annotation['minor_units']=125050
                    elif variant=='both': annotation['minor_units']=125050
                    elif variant=='neither': annotation.pop('amount')
                    elif variant=='illegal': annotation['illegal']='value'
                    elif variant=='reference': annotation['reference_asset_id']='foreign'
                    if variant=='unpaired_minor_units':
                        # Structurally valid annotations still require the exact
                        # paired frozen record; inventory traversal is not that gate.
                        with self.assertRaisesRegex(ValueError,'^speech_text_mismatch$'):
                            oracle.frozen_plan(plan,f['context']['bundles'])
                        continue
                    reasons,refs=self.scan(f)
                    if variant in ('amount', 'minor_units'):
                        oracle.validate('worker_plan',plan)
                        oracle.stored_context(f['job'],f['context'])
                        self.assertFalse(reasons,reasons)
                        for asset in plan['assets']:
                            self.assertIn(asset['id'],refs)
                    else:
                        self.assertTrue(reasons)

    def test_nested_union_walks_enclosing_and_selected_reference_properties(self):
        document=copy.deepcopy(oracle.SCHEMA)
        node=document['$defs']['worker_plan']['oneOf'][2]
        node['properties']['source_asset_id']={'type':'string'}
        node['allOf'].append({'oneOf':[
            {'required':['speech_text'], 'properties':{'reference_asset_id':{'type':'string'}}},
            {'not':{'required':['speech_text']}}]})
        node['properties']['reference_asset_id']={'type':'string'}
        f=self.fixture()
        f['context']['plan'].update(source_asset_id='enclosing',reference_asset_id='selected')
        reasons,refs=self.scan(f,document)
        self.assertFalse(reasons,reasons)
        self.assertIn('enclosing',refs)
        self.assertIn('selected',refs)
        ambiguous=copy.deepcopy(document)
        ambiguous['$defs']['worker_plan']['oneOf'][2]['allOf'][-1]['oneOf'].append({})
        self.assertTrue(self.scan(f,ambiguous)[0])
