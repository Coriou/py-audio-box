import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator, ValidationError
from contracts import validate as c

ROOT=Path(__file__).resolve().parents[2]
FIX=ROOT/'contracts/fixtures/v1'
def load(name):return json.loads((FIX/name).read_text())
@pytest.fixture
def p():return load('production-30.json')

def test_schema_and_all_fixtures():
    Draft202012Validator.check_schema(c.SCHEMA)
    for n in (30,100):
        p=load(f'production-{n}.json');c.validate('project',p);c.renderable(p,p['render'])
        assert len(p['blocks'])==n and len(p['speakers'])==3
    assert len(load('production-100.json')['clips'])==200
    caps=load('capabilities.json');c.validate('capabilities',caps)
    for profile in caps['profiles']:c.validate('profile',profile)
    ops=json.loads((ROOT/'contracts/v1/operations.json').read_text())['operations']
    assert len({o['mcp'] for o in ops})==len(ops)
    for o in ops:
        assert o['request'] in c.SCHEMA['$defs'] and o['response'] in c.SCHEMA['$defs']
        assert o['http']['route'].startswith('/v1/')

def test_purposeful_negative_fixtures(p):
    for case in load('negative.json'):
        x=copy.deepcopy(p);target=x
        for key in case['path'][:-1]:target=target[key]
        target[case['path'][-1]]=case['value']
        with pytest.raises(c.ContractError,match=case['error']):c.validate('project',x)

def test_structural_rejections(p):
    for field,value in [('start',0.5),('start',-1),('start',9007199254740992),('gain',None)]:
        x=copy.deepcopy(p);x['clips'][0][field]=value
        with pytest.raises(ValidationError):c.validate('project',x)
    p['clips'][0]['arbitrary_argv']=[]
    with pytest.raises(ValidationError):c.validate('project',p)

def test_exact_text_ids_defaults_and_source_control_coverage():
    ledger=json.loads((ROOT/'contracts/v1/legacy-controls.json').read_text())
    assert {x['source'] for x in ledger['commands']}=={str(x.relative_to(ROOT)) for x in (ROOT/'apps').glob('*/*.py')}
    for entry in ledger['commands']:
        path=ROOT/entry['source'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('add_argument','add_parser','set_defaults')]
        assert len(nodes)==len(entry['controls'])
        assert entry['coverage']=='unchanged_legacy'
    for entry in ledger['shared_sources']+ledger['launchers']:assert hashlib.sha256((ROOT/entry['source']).read_bytes()).hexdigest()==entry['sha256']
    p=load('production-30.json');assert p['blocks'][0]['text']=='  Ligne 000 — café!\nExact text.  '
    assert p['voices'][0]['id']=='voice.0' and p['voices'][0]['aliases']==['legacy_0']

def test_determinism(tmp_path, monkeypatch):
    from contracts import generate

    before={p.name:p.read_bytes() for p in FIX.glob('*')}
    outputs=[]
    for run in ('first', 'second'):
        package=tmp_path/run/'contracts'
        # ROOT also locates schema/operations via its parents; keep that layout.
        shutil.copytree(ROOT/'contracts/v1', package/'v1')
        output=package/'fixtures/v1'
        monkeypatch.setattr(generate, 'ROOT', output)
        generate.generate()
        outputs.append({p.name:p.read_bytes() for p in output.glob('*')})
    assert outputs[0]==outputs[1]
    assert outputs[0]==before

def test_revision_idempotency():
    c.revision(3,3)
    with pytest.raises(c.ContractError,match='revision_conflict'):c.revision(2,3)
    a={'text':' café ','voice':'voice.0'};h=c.fingerprint('principal','speech_generate',a)
    c.idempotency(h,'principal','speech_generate',dict(reversed(list(a.items()))))
    for principal,op,payload in [('other','speech_generate',a),('principal','other',a),('principal','speech_generate',{**a,'text':'café'})]:
        with pytest.raises(c.ContractError,match='idempotency_conflict'):c.idempotency(h,principal,op,payload)

def test_lifecycle_cancel_fence_publication():
    c.transition('running','cancel_requested')
    c.transition('cancel_requested','succeeded',media_verified=True)
    c.transition('cancel_requested','cancelled',reaped=True)
    for old,new,kw,err in [('cancel_requested','cancelled',{},'cancel_pending'),('running','succeeded',{},'missing_media'),('running','succeeded',dict(generation=1,current_generation=2,media_verified=True),'stale_attempt'),('succeeded','running',{},'invalid_transition')]:
        with pytest.raises(c.ContractError,match=err):c.transition(old,new,**kw)

def test_stale_selection_render_and_fixed_timing(p):
    starts=[x['start'] for x in p['clips']];p['blocks'][0]['text']='Changed'
    assert [t['id'] for t in p['takes'] if c.stale(p,t)]==['t0','candidate-long','candidate-short']
    assert starts==[x['start'] for x in p['clips']]
    with pytest.raises(c.ContractError,match='stale_take'):c.renderable(p,p['render'])
    policy={**p['render'],'include_stale':True};c.renderable(p,policy)
    with pytest.raises(c.ContractError,match='clipping'):c.renderable(p,policy,1.01)
    p['clips'][0]['gain']=0;c.validate('project',p)
    c.ripple(p,'speech',0,9600,0,480,['c0','c1'])
    with pytest.raises(c.ContractError,match='affected_ids_mismatch'):c.ripple(p,'speech',0,9600,0,480,['c0'])

def test_device_policy_and_readiness():
    r=copy.deepcopy(load('production-30.json')['takes'][0]['identity']);r['requested']['device']='cuda'
    with pytest.raises(c.ContractError,match='fallback_forbidden'):c.validate('resolution',r)
    r.update(fallback='allow_cpu',reason='operator policy');c.validate('resolution',r)
    r['effective']['model']='different'
    with pytest.raises(c.ContractError,match='identity_substitution'):c.validate('resolution',r)
    p=load('capabilities.json')['profiles'][0];p['support']='qualified'
    with pytest.raises(c.ContractError,match='qualification_required'):c.validate('profile',p)

def test_tick_rounding_and_immutability():
    assert c.ticks('0.00003125')==2
    with pytest.raises(c.ContractError,match='immutable_identity'):c.immutable({'sha':'a'},{'sha':'b'})

def test_replacement_duration_does_not_ripple(p):
    c0=p['clips'][0];following=p['clips'][1]['start']
    c0.update(take_id='candidate-long',asset_id='long',out_tick=9600)
    c.validate('project',p)
    assert c0['start']+c0['out_tick']>following==4800
    c0.update(take_id='candidate-short',asset_id='short')
    with pytest.raises(c.ContractError,match='invalid_trim'):c.validate('project',p)
    c0['out_tick']=2400;c.validate('project',p)
    assert p['clips'][1]['start']==following

def test_migration_and_worker_boundaries():
    report=dict(collisions=[],apply_allowed=True)
    c.migration(report,source_root='/legacy',destination_root='/service')
    with pytest.raises(c.ContractError,match='distinct_root_required'):c.migration(report,source_root='/legacy',destination_root='/legacy')
    with pytest.raises(c.ContractError,match='migration_collision'):c.migration({**report,'collisions':['voice.0']},source_root='/legacy',destination_root='/service')
    with pytest.raises(c.ContractError,match='pending_jobs_not_importable'):c.migration(report,source_root='/legacy',destination_root='/service',pending_policy='retry')

def test_operation_requests_jobs_artifacts_and_error_mapping():
    ops=json.loads((ROOT/'contracts/v1/operations.json').read_text())['operations']
    samples=load('operation-requests.json')
    assert set(samples)=={op['id'] for op in ops}
    for op in ops:c.validate(op['request'],c.fixture_request(op['request'],samples[op['id']]))
    c.validate('job',load('job.json'));c.validate('artifact',load('artifact.json'))
    errors=json.loads((ROOT/'contracts/v1/errors.json').read_text())
    assert set(errors['http_status'])==set(c.SCHEMA['$defs']['error']['properties']['code']['enum'])
    assert errors['http_status']['revision_conflict']==409
    assert errors['http_status']['idempotency_conflict']==409

def test_late_candidates_and_text_edits_cannot_rearrange_or_select(p):
    after=copy.deepcopy(p);after['revision']+=1
    after['takes'].append({**after['takes'][0],'id':'late-candidate'})
    c.edit_semantics(p,after,'candidate_arrival')
    after['blocks'][0]['selected_take_id']='late-candidate'
    with pytest.raises(c.ContractError,match='implicit_selection'):c.edit_semantics(p,after,'candidate_arrival')
    after=copy.deepcopy(p);after['revision']+=1;after['blocks'][0]['text']='Edited'
    c.edit_semantics(p,after,'block_text')
    after['clips'][1]['start']+=4800
    with pytest.raises(c.ContractError,match='implicit_arrangement'):c.edit_semantics(p,after,'block_text')


def test_binary_fixture_boundary():
    sample=load('operation-requests.json')['asset_upload']
    raw=c.fixture_request('asset_upload_request',sample)
    assert type(raw) is bytes and raw==(FIX/'short.wav').read_bytes()
    assert c.upload(raw) is raw
    assert c.fixture_request('voice_versions_list_request',{'cursor':'','limit':1,'voice_id':'voice.0'})=={'cursor':'','limit':1,'voice_id':'voice.0'}
    for value in (sample,{'content_base64':'YQ==','bytes':1,'content_sha256':'0'*64,'source':'upload'},'example',bytearray(raw),memoryview(raw)):
        with pytest.raises(c.ContractError,match='invalid_bytes'):c.upload(value)
    for value in ({}, {'$fixture_binary_base64':'!'}, {'$fixture_binary_base64':'YR=='}, {'$fixture_binary_base64':1}, {**sample,'extra':True}):
        with pytest.raises(c.ContractError,match='invalid_binary_fixture'):c.fixture_request('asset_upload_request',value)
    with pytest.raises(c.ContractError,match='invalid_size'):c.fixture_request('asset_upload_request',{'$fixture_binary_base64':''})
