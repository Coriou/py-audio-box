"""Versioned role contract, called by unittest without pytest collection."""
import copy
import json
from pathlib import Path
import unittest
from jsonschema import ValidationError
from contracts import validate as c
from contracts.scenarios import transcript, catalogue, identity

FIX=Path(__file__).resolve().parents[2]/'contracts/fixtures/v1'
def load(name):return json.loads((FIX/name).read_text())
def check(t):
    c.publication(t['job'],t['job']['attempts'][0],t['artifact'],t['present_hashes'],transport_owner='worker1',context=t['context'],manifest=t['verified_manifest'],verified_result=t['message'],allowed_assets=t['authorized_assets'])
def rehash(t):t['artifact']['manifest_sha256']=c.canonical_hash(t['verified_manifest'])

class ReferenceRolesTests(unittest.TestCase):
    def scenario(self,op='voice_prepare'):
        t=load('transcript-'+op+'-roles.json');check(t);return t

    def test_generated_fixtures_and_all_wire_consumers(self):
        ops={o['id']:o for o in json.loads((c.ROOT/'v1/operations.json').read_text())['operations']}
        for op in ('voice_prepare','voice_design'):
            t=self.scenario(op)
            for e in t['exchanges']:
                c.validate(ops[e['operation']]['request'],e['request'])
                c.validate(ops[e['operation']]['response'],e['response'])
            version=t['exchanges'][-1]['response'];p=load('production-30.json')
            p['assets']+=copy.deepcopy(t['verified_manifest']['outputs'])
            reuse=transcript('speech_generate',p,t['profiles'],load('operation-requests.json'),version=version)
            check(reuse)
            self.assertEqual(c.required_asset_ids(reuse['context']['plan']),version['reference_asset_ids'])
            reference=t['verified_manifest']['candidates'][0]['reference_roles']['reference']
            self.assertEqual(reuse['context']['plan']['assets'],[reference['asset']])
            self.assertEqual(reuse['context']['plan']['voice_versions'][0]['reference_text'],reference['transcript'])

    def test_real_prepare_proposal_and_authorized_original_transfer(self):
        t=self.scenario();p=t['context']['plan'];original=p['assets'][0]
        # No synthetic-only requirement on the versioned path. This is a pure
        # proposal, not an assertion that inference or media verification ran.
        original['provenance']['kind']='upload'
        candidate=t['verified_manifest']['candidates'][0];roles=candidate['reference_roles']
        roles['reference']['original']=copy.deepcopy(original)
        roles['reference']['asset']['provenance']['kind']='upload'
        candidate['version']['provenance']=copy.deepcopy(roles['reference']['asset']['provenance'])
        t['verified_manifest']['outputs'][0]=copy.deepcopy(roles['reference']['asset'])
        roles['audition']['asset']['provenance']['kind']='generated'
        t['verified_manifest']['outputs'][1]=copy.deepcopy(roles['audition']['asset'])
        h=c.plan_hash(p)
        for obj in (t['job'],t['message'],t['artifact'],t['verified_manifest']):obj['input_sha256']=h
        t['message']['plan']=copy.deepcopy(p);rehash(t);check(t)
        self.assertEqual(c.input_content(p,original['id'],authorized_assets=t['authorized_assets']),original)
        with self.assertRaisesRegex(c.ContractError,'unmanaged_asset'):
            c.input_content(p,original['id'],authorized_assets=[])

    def test_pocket_preparation_keeps_reference_without_fabricated_transcript(self):
        t=self.scenario();p=t['context']['plan'];r=p['resolved'];v=t['verified_manifest']['candidates'][0]['version']
        ident=identity(catalogue()[4]);p['request'].update(identity=ident,reference_text='')
        r['identity'].update(requested=ident,effective=ident)
        r['settings']['conditioning']='pocket';r['settings_sha256']=c.canonical_hash(r['settings'])
        r['profile']['compatibility']=[dict(operation='voice_prepare',mode='prepare',identity=ident)]
        r['reuse_configuration']=dict(mode='pocket_clone',reference_asset_id=p['reference_roles']['reference_asset_id'],language='fr')
        r['reuse_settings']=dict(kind='pocket_clone',language='fr',seed=1)
        v.update(identity=ident,reference_text='',configuration=copy.deepcopy(r['reuse_configuration']),settings=copy.deepcopy(r['reuse_settings']),settings_sha256=c.canonical_hash(r['reuse_settings']))
        t['verified_manifest']['candidates'][0]['reference_roles']['reference']['transcript']=''
        h=c.plan_hash(p)
        for obj in (t['job'],t['message'],t['artifact'],t['verified_manifest']):obj['input_sha256']=h
        t['message']['plan']=copy.deepcopy(p)
        t['message']['identity']=copy.deepcopy(r['identity']);t['artifact']['identity']=copy.deepcopy(r['identity'])
        rehash(t);check(t)

    def test_closed_publication_mutations(self):
        for mutation in ('conflation','reference_text','audition_text','foreign_original','segment','provenance','metadata','missing_role','missing_output','foreign_output','configuration','stale','hash','missing_hash','settings','audition_source'):
            with self.subTest(mutation=mutation):
                t=self.scenario();m=t['verified_manifest'];candidate=m['candidates'][0];roles=candidate['reference_roles'];ref=roles['reference'];aud=roles['audition']
                if mutation=='conflation':ref['asset']=copy.deepcopy(aud['asset'])
                elif mutation=='reference_text':ref['transcript']='Wrong generated words'
                elif mutation=='audition_text':aud['text']='Wrong original transcript'
                elif mutation=='foreign_original':ref['original']['id']='foreign'
                elif mutation=='segment':ref['in_tick']=1
                elif mutation=='provenance':ref['asset']['provenance']['source_sha256']='f'*64
                elif mutation=='metadata':aud['asset']['frames']+=1
                elif mutation=='missing_role':del candidate['reference_roles']
                elif mutation=='missing_output':m['outputs'].pop()
                elif mutation=='foreign_output':m['outputs'][0]['id']='foreign'
                elif mutation=='configuration':candidate['version']['configuration']['reference_asset_id']='foreign'
                elif mutation=='stale':m['generation']+=1
                elif mutation=='hash':t['present_hashes'][ref['asset']['id']]='f'*64
                elif mutation=='missing_hash':del ref['asset']['sha256']
                elif mutation=='settings':
                    candidate['version']['settings']['seed']=72
                    candidate['version']['settings_sha256']=c.canonical_hash(candidate['version']['settings'])
                elif mutation=='audition_source':
                    aud['asset']['provenance']['kind']='upload'
                    m['outputs'][1]=copy.deepcopy(aud['asset'])
                rehash(t)
                with self.assertRaises((c.ContractError,ValidationError)):check(t)

    def test_plan_membership_and_configuration(self):
        for mutation in ('input_as_reference','same_roles','wrong_kind','wrong_configuration','wrong_design_text'):
            with self.subTest(mutation=mutation):
                t=self.scenario('voice_design' if mutation=='wrong_design_text' else 'voice_prepare');p=t['context']['plan'];r=p['reference_roles']
                if mutation=='input_as_reference':r['reference_asset_id']=p['request']['asset_id'];p['resolved']['reuse_configuration']['reference_asset_id']=r['reference_asset_id']
                elif mutation=='same_roles':r['audition_asset_id']=r['reference_asset_id']
                elif mutation=='wrong_kind':r['reference_kind']='generated_reference'
                elif mutation=='wrong_configuration':r['reference_asset_id']='foreign'
                else:r['audition_text']='Wrong spoken design text'
                with self.assertRaises(c.ContractError):c.plan_hash(p)

    def test_design_lineage_and_actual_spoken_text(self):
        for mutation in ('lineage','text','separate_asset'):
            t=self.scenario('voice_design');roles=t['verified_manifest']['candidates'][0]['reference_roles']
            if mutation=='lineage':roles['reference']['lineage']['description']='other'
            elif mutation=='text':roles['reference']['transcript']='other'
            else:roles['reference']['asset']['id']='separate'
            rehash(t)
            with self.assertRaises(c.ContractError):check(t)

    def test_legacy_same_asset_only_synthetic(self):
        for op in ('voice_prepare','voice_design'):
            t=load('transcript-'+op+'.json');check(t)
            t['verified_manifest']['outputs'][0]['provenance']['kind']='generated';rehash(t)
            with self.assertRaisesRegex(c.ContractError,'reference_roles_required'):check(t)

if __name__=='__main__':unittest.main()
