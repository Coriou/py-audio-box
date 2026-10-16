"""Detached-wire regression matrix. Every mutation proves a valid baseline first."""
import copy
import json
import pytest
from jsonschema import ValidationError
from contracts import validate as c
from contracts.scenarios import transcript, catalogue, identity, qwen_settings
from test_contracts import load

OPS=['voice_prepare','voice_design','speech_generate','project_render']

def wire(t):return json.loads(json.dumps(t))

def scenario(op='speech_generate',version=None,device='cpu'):
    return transcript(op,load('production-30.json'),load('capabilities.json')['profiles'],load('operation-requests.json'),version=version,device=device)

def check(t):
    c.worker_message(t['message'],t['job'],t['authorized_assets'],transport_owner='worker1',context=t['context'])
    c.publication(t['job'],t['job']['attempts'][-1],t['artifact'],t['present_hashes'],transport_owner='worker1',context=t['context'],manifest=t['verified_manifest'],verified_result=t['message'],allowed_assets=t['authorized_assets'])

def refreeze(t):
    """Explicit trusted control resolution for a NEW synthetic test baseline only."""
    p=t['context']['plan'];r=p['resolved'];r['settings_sha256']=c.canonical_hash(r['settings'])
    if p['operation']=='speech_generate':
        for v in p['voice_versions']:v['settings_sha256']=r['settings_sha256']
    h=c.plan_hash(p);t['job']['input_sha256']=h;t['message']['input_sha256']=h;t['artifact']['input_sha256']=h;t['verified_manifest']['input_sha256']=h
    t['message']['plan']=wire(p);t['message']['identity']=wire(r['identity']);t['artifact']['identity']=wire(r['identity'])
    t['message']['input_asset_ids']=c.required_asset_ids(p);t['authorized_assets']=c.required_asset_ids(p)
    t['artifact']['manifest_sha256']=c.canonical_hash(t['verified_manifest'])
    return wire(t)

def cloned(origin='clone',engine='qwen'):
    p=load('production-30.json');v=copy.deepcopy(p['voice_versions'][0]);b=catalogue()[1 if engine=='qwen' else 4]
    v.update(origin=origin,identity=identity(b),reference_asset_ids=['impulse'],reference_text='Exact reference.' if engine=='qwen' else '')
    v['configuration']=dict(mode='qwen_clone',reference_asset_id='impulse',reference_text=v['reference_text'],conditioning='icl') if engine=='qwen' else dict(mode='pocket_clone',reference_asset_id='impulse',language='fr')
    if origin=='design':v['design_lineage']=dict(description='Historical design.',source_candidate_id='historical-candidate',generation_identity=identity(catalogue()[2]))
    return v

@pytest.mark.parametrize('op',OPS)
def test_complete_catalogue_transcripts(op):
    t=load('transcript-'+op+'.json');check(t)
    c.frozen_plan(t['context']['plan'],t['context']['bundles'])
    catalogue_ops=json.loads((c.ROOT/'v1/operations.json').read_text())['operations'];ops={x['id']:x for x in catalogue_ops}
    for e in t['exchanges']:
        c.validate(ops[e['operation']]['request'],e['request'])
        c.validate(ops[e['operation']]['response'],e['response'])
    if op in ('voice_prepare','voice_design'):
        version=t['exchanges'][-1]['response'];asset=t['verified_manifest']['outputs'][0]
        p=load('production-30.json');p['assets'].append(asset)
        reuse=transcript('speech_generate',p,t['profiles'],load('operation-requests.json'),version=version)
        check(reuse);assert reuse['message']['plan']['voice_versions'][0]==version
        assert reuse['message']['plan']['resolved']['settings']['kind']=='qwen_clone'


def test_two_selectors_survive_and_hashes_differ():
    a=scenario();b=scenario(version=load('production-30.json')['voice_versions'][1]);check(a);check(b)
    assert a['job']['input_sha256']!=b['job']['input_sha256']
    for t in [a,b]:
        version=t['exchanges'][0]['response'];assert version==t['message']['plan']['voice_versions'][0]

@pytest.mark.parametrize('selector',['','unknown','ff_siwis'])
def test_selector_membership(selector):
    t=scenario();check(t);bad=wire(t);bad['context']['plan']['voice_versions'][0]['configuration']['speaker']=selector
    # Shape rejection alone is not membership evidence. Empty is deliberately semantic too.
    with pytest.raises(c.ContractError,match='unknown_selector'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])

@pytest.mark.parametrize('field,value,error',[
 ('deadline','2099','deadline_mismatch'),('input_asset_ids',['impulse'],'asset_dependencies_mismatch'),
 ('owner_id','wrong','attempt_mismatch'),('staging_id','wrong','attempt_mismatch'),('attempt_id','wrong','attempt_mismatch'),('job_id','wrong','attempt_mismatch'),('generation',2,'attempt_mismatch')])
def test_worker_claims(field,value,error):
    t=scenario();check(t);bad=wire(t);bad['message'][field]=value
    with pytest.raises(c.ContractError,match=error):check(bad)

@pytest.mark.parametrize('claim',['model','bundle_sha256','profile_id','device','precision','fallback'])
def test_worker_resolution_detached(claim):
    t=scenario();check(t);bad=wire(t);r=bad['message']['identity']
    if claim in ('model','bundle_sha256'):
        for k in ('requested','effective'):r[k][claim]='other' if claim=='model' else 'f'*64
    elif claim=='profile_id':r[claim]='other-profile'
    elif claim=='fallback':r[claim]='allow_cpu'
    else:
        for k in ('requested','effective'):r[k][claim]='cuda' if claim=='device' else 'float16'
    with pytest.raises(c.ContractError,match='resolution_mismatch'):check(bad)

@pytest.mark.parametrize('claim',['model','bundle_sha256','profile_id','device','precision','revision','manifest','result_set','global_asset'])
def test_publication_context(claim):
    t=scenario();check(t);bad=wire(t);a=bad['artifact']
    if claim in ('model','bundle_sha256'):
        for k in ('requested','effective'):a['identity'][k][claim]='other' if claim=='model' else 'f'*64
    elif claim=='profile_id':a['identity'][claim]='other'
    elif claim in ('device','precision'):
        for k in ('requested','effective'):a['identity'][k][claim]='cuda' if claim=='device' else 'float16'
    elif claim=='revision':a['project_revision']=999
    elif claim=='manifest':a['manifest_sha256']='f'*64
    else:
        a['asset_ids']=['unrelated'];bad['present_hashes']['unrelated']='f'*64
    with pytest.raises(c.ContractError,match='resolution_mismatch|revision_conflict|manifest_mismatch|result_mismatch'):check(bad)

@pytest.mark.parametrize('mutation',['text','settings','voice','profile','assets','rehash'])
def test_received_plan_is_never_authority(mutation):
    t=scenario();check(t);bad=wire(t);p=bad['message']['plan']
    if mutation in ('text','rehash'):p['request']['text']='changed text'
    elif mutation=='settings':p['resolved']['settings']['seed']=2;p['resolved']['settings_sha256']=c.canonical_hash(p['resolved']['settings']);p['voice_versions'][0]['settings_sha256']=p['resolved']['settings_sha256']
    elif mutation=='voice':p['voice_versions'][0]['configuration']['speaker']='Serena'
    elif mutation=='profile':p['resolved']['profile']['limits']['threads']=3
    else:p['assets']=[load('production-30.json')['assets'][0]]
    if mutation=='rehash':bad['message']['input_sha256']=c.plan_hash(p)
    with pytest.raises(c.ContractError,match='input_mismatch|attempt_mismatch'):check(bad)

@pytest.mark.parametrize('device',['cpu','cuda','metal'])
def test_mock_admission_and_auto(device):
    t=scenario(device=device);check(t);profiles=wire(t['profiles'])
    with pytest.raises(c.ContractError,match='not_ready'):c.admission(t['job'],t['context'],profiles)
    profiles[0].update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='MOCK-E1-not-real-qualification')
    c.admission(t['job'],t['context'],profiles)
    t['context']['plan']['request']['identity']['device']='auto';t['context']['plan']['resolved']['identity']['requested']['device']='auto';t=refreeze(t);check(t);c.admission(t['job'],t['context'],profiles)
    for field in ['limits','compatibility','config_revision','topology','device']:
        bad=wire(profiles)
        if field=='limits':bad[0][field]['threads']+=1
        elif field=='compatibility':bad[0][field]=[]
        else:bad[0][field]={'config_revision':'config-2','topology':'networkless_container' if device=='metal' else 'packaged_native','device':'cuda' if device=='cpu' else 'cpu'}[field]
        with pytest.raises(c.ContractError,match='profile_mismatch'):c.admission(t['job'],t['context'],bad)
    with pytest.raises(c.ContractError,match='unknown_profile'):c.admission(t['job'],t['context'],[])


def fallback():
    t=scenario();p=t['context']['plan'];p['request'].update(fallback='allow_cpu');p['request']['identity']['device']='cuda';p['resolved']['identity']['requested']['device']='cuda';p['resolved']['identity'].update(fallback='allow_cpu',reason='Explicit permitted CPU fallback');p['resolved']['operator_fallback']='allow_cpu';return refreeze(t)

@pytest.mark.parametrize('change,error',[('caller','fallback_policy_mismatch'),('operator','fallback_policy_mismatch'),('reason','fallback_forbidden'),('precision','precision_substitution'),('auto','unresolved_identity')])
def test_fallback_authority(change,error):
    t=fallback();check(t);bad=wire(t);p=bad['context']['plan']
    if change=='caller':p['request']['fallback']='forbid'
    elif change=='operator':p['resolved']['operator_fallback']='forbid'
    elif change=='reason':p['resolved']['identity']['reason']=''
    elif change=='precision':p['resolved']['identity']['effective']['precision']='float16'
    else:p['resolved']['identity']['effective']['device']='auto'
    with pytest.raises(c.ContractError,match=error):c.frozen_plan(p,bad['context']['bundles'])

@pytest.mark.parametrize('origin,engine',[('clone','qwen'),('design','qwen'),('clone','pocket-tts')])
def test_clone_design_and_dependency_authority(origin,engine):
    t=scenario(version=cloned(origin,engine));check(t)
    for mutation,error in [('empty','asset_dependencies_mismatch'),('unauthorized','unmanaged_asset'),('duplicate','asset_dependencies_mismatch'),('extra','asset_dependencies_mismatch')]:
        bad=wire(t)
        if mutation=='empty':bad['message']['input_asset_ids']=[];bad['authorized_assets']=[]
        elif mutation=='unauthorized':bad['authorized_assets']=[]
        elif mutation=='duplicate':bad['message']['input_asset_ids']*=2
        else:bad['message']['input_asset_ids'].append('sine');bad['authorized_assets'].append('sine')
        with pytest.raises(c.ContractError,match=error):check(bad)
    bad=wire(t);bad['message']['plan']['voice_versions'][0]['configuration']['reference_asset_id']='sine'
    with pytest.raises(c.ContractError,match='reference_mismatch'):check(bad)

@pytest.mark.parametrize('op',['voice_prepare','project_render'])
def test_prepare_render_exact_assets(op):
    t=scenario(op);check(t)
    for field in ('input_asset_ids','authorized_assets'):
        bad=wire(t);bad['message'][field]=[] if field=='input_asset_ids' else bad['message'].get(field,[])
        if field=='authorized_assets':bad['authorized_assets']=[];bad['message'].pop(field,None)
        with pytest.raises(c.ContractError,match='asset_dependencies_mismatch|unmanaged_asset'):check(bad)
    for mutation in ('missing','duplicate','hash'):
        bad=wire(t);p=bad['context']['plan']
        if mutation=='missing':p['assets']=[]
        elif mutation=='duplicate':p['assets'].append(copy.deepcopy(p['assets'][0]))
        else:p['assets'][0]['sha256']='f'*64
        if mutation=='hash' and op=='voice_prepare':
            # Frozen metadata alone is not byte proof; wrong bytes rejected at transfer/adoption.
            with pytest.raises(c.ContractError,match='input_mismatch'):check(bad)
        else:
            with pytest.raises(c.ContractError):c.frozen_plan(p,bad['context']['bundles'])


def test_muted_render_and_history_transfer():
    t=scenario('project_render');t['context']['plan']['project']['clips'][-1]['mute']=True;t=refreeze(t);check(t)
    assert set(c.required_asset_ids(t['context']['plan']))=={'impulse','sine'}
    c.input_content(t['context']['plan'],'sine',authorized_assets=['impulse','sine','long'])
    with pytest.raises(c.ContractError,match='unmanaged_asset'):c.input_content(t['context']['plan'],'long',authorized_assets=['long'])
    bad=wire(t);bad['message']['input_asset_ids'].remove('sine')
    with pytest.raises(c.ContractError,match='asset_dependencies_mismatch'):check(bad)


def test_kokoro_explicit_settings_and_selector():
    v=load('production-30.json')['voice_versions'][0];v.update(identity=identity(catalogue()[3]),configuration=dict(mode='kokoro_preset',preset='ff_siwis',language='fr'))
    t=scenario(version=v);check(t)
    for key,value in [('seed',1),('instruction','whisper'),('top_p',1)]:
        bad=wire(t);bad['context']['plan']['resolved']['settings'][key]=value
        with pytest.raises(ValidationError):c.validate('worker_plan',bad['context']['plan'])
    for key,value in [('preset','Vivian'),('language','en')]:
        bad=wire(t);bad['context']['plan']['voice_versions'][0]['configuration'][key]=value
        with pytest.raises(c.ContractError,match='unknown_selector|language_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])


def test_semantic_settings_and_conditioning():
    t=scenario(version=cloned());check(t)
    for key,value,error in [('instruction','whisper','unsupported_instruction'),('language','Japanese','language_mismatch')]:
        bad=wire(t);p=bad['context']['plan'];p['resolved']['settings'][key]=value;p['resolved']['settings_sha256']=c.canonical_hash(p['resolved']['settings']);p['voice_versions'][0]['settings_sha256']=p['resolved']['settings_sha256']
        with pytest.raises(c.ContractError,match=error):c.frozen_plan(p,bad['context']['bundles'])
    bad=wire(t);bad['context']['plan']['resolved']['settings_sha256']='f'*64
    with pytest.raises(c.ContractError,match='settings_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])
    bad=wire(t);bad['context']['plan']['voice_versions'][0]['configuration']['conditioning']='x_vector'
    with pytest.raises(c.ContractError,match='conditioning_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])
    bad=wire(t);v=bad['context']['plan']['voice_versions'][0];v['identity']=identity(catalogue()[0]);bad['context']['plan']['request']['identity']=v['identity']
    with pytest.raises(c.ContractError,match='incompatible_bundle|identity_substitution'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])


def test_retry_preserves_acoustic_plan_new_deadline_is_trusted():
    t=scenario();check(t);h=t['job']['input_sha256'];old=t['job']['attempts'][0];old['reaped']=True
    new=dict(id='attempt2',generation=2,owner_id='worker2',staging_id='stage2',reaped=False);c.replacement(t['job'],new,owner_dead=True)
    t['job']['attempts'].append(new);t['job'].update(generation=2,deadline='2026-10-16T00:20:00Z')
    m=t['message'];m.update(attempt_id=new['id'],generation=2,owner_id=new['owner_id'],staging_id=new['staging_id'],deadline=t['job']['deadline'])
    c.worker_message(m,t['job'],[],transport_owner='worker2',context=t['context']);assert t['job']['input_sha256']==h

@pytest.mark.parametrize('op',OPS)
def test_execute_admission_is_enforced_at_worker_entrypoint(op):
    t=scenario(op);check(t);m=wire(t['message']);m['kind']='execute';m['result_asset_ids']=[]
    with pytest.raises(c.ContractError,match='unknown_profile'):
        c.worker_message(m,t['job'],t['authorized_assets'],transport_owner='worker1',context=t['context'])
    with pytest.raises(c.ContractError,match='not_ready'):
        c.worker_message(m,t['job'],t['authorized_assets'],transport_owner='worker1',context=t['context'],current_profiles=t['profiles'])
    profiles=wire(t['profiles']);profiles[0].update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='MOCK-E1-only')
    c.worker_message(m,t['job'],t['authorized_assets'],transport_owner='worker1',context=t['context'],current_profiles=profiles)


def test_input_transfer_scope_and_hash():
    from test_contracts import FIX
    t=scenario('voice_prepare');check(t);request=dict(asset_id='impulse',start=0,end_exclusive=44);data=(FIX/'impulse.wav').read_bytes()
    kwargs=dict(context=t['context'],attempt_id=t['job']['attempts'][0]['id'],transport_owner='worker1',authorized_assets=['impulse'])
    c.job_input_content(data,request,t['job'],**kwargs)
    for changed,error in [({'data':b'wrong'},'invalid_hash'),({'authorized_assets':[]},'unmanaged_asset'),({'attempt_id':'other'},'attempt_mismatch'),({'transport_owner':'other'},'attempt_mismatch')]:
        args={**kwargs,**changed};payload=args.pop('data',data)
        with pytest.raises(c.ContractError,match=error):c.job_input_content(payload,request,t['job'],**args)

@pytest.mark.parametrize('op',OPS)
def test_verified_manifest_is_current_attempt_result(op):
    t=scenario(op);check(t)
    for mutation,error in [('attempt','manifest_attempt_mismatch'),('hash','missing_media'),('result_ids','result_mismatch'),('output_metadata','manifest_mismatch')]:
        bad=wire(t)
        if mutation=='attempt':bad['verified_manifest']['attempt_id']='other'
        elif mutation=='hash':bad['present_hashes'][bad['artifact']['asset_ids'][0]]='f'*64
        elif mutation=='result_ids':bad['message']['result_asset_ids']=['unrelated']
        else:bad['verified_manifest']['outputs'][0]['frames']+=1
        with pytest.raises(c.ContractError,match=error):check(bad)

@pytest.mark.parametrize('op',['voice_prepare','voice_design'])
def test_candidate_cannot_reroll_or_replace_realization(op):
    t=scenario(op);check(t)
    for mutation in ('reference','lineage','origin'):
        if op=='voice_prepare' and mutation=='lineage':continue
        bad=wire(t);v=bad['verified_manifest']['candidates'][0]['version']
        if mutation=='reference':v['configuration']['reference_asset_id']='sine';v['reference_asset_ids']=['sine']
        elif mutation=='lineage':v['design_lineage']['description']='different design'
        else:v['origin']='preset'
        bad['artifact']['manifest_sha256']=c.canonical_hash(bad['verified_manifest'])
        with pytest.raises(c.ContractError,match='reference_mismatch|design_lineage_mismatch|origin_mismatch'):check(bad)


def test_clone_finite_single_reference_and_pocket_closed_mode():
    t=scenario(version=cloned());check(t);bad=wire(t);bad['context']['plan']['voice_versions'][0]['reference_asset_ids'].append('sine')
    with pytest.raises(c.ContractError,match='reference_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])
    t=scenario(version=cloned(engine='pocket-tts'));check(t);bad=wire(t);bad['context']['plan']['voice_versions'][0]['configuration']=dict(mode='qwen_preset',speaker='Vivian')
    with pytest.raises(c.ContractError,match='configuration_engine_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])


def test_reference_must_be_wav_and_engine_execution_not_invented():
    t=scenario(version=cloned(engine='pocket-tts'));check(t);bad=wire(t);bad['context']['plan']['assets'][0]['media_format']='mp3'
    with pytest.raises(c.ContractError,match='reference_format_mismatch'):c.frozen_plan(bad['context']['plan'],bad['context']['bundles'])
    bad=wire(t);p=bad['context']['plan'];p['resolved']['identity']['requested']['device']='cuda';p['resolved']['identity']['effective']['device']='cuda';p['resolved']['profile']['device']='cuda'
    with pytest.raises(c.ContractError,match='unsupported_execution'):c.frozen_plan(p,bad['context']['bundles'])


def test_later_render_draft_and_frozen_profile_substitution():
    t=scenario('project_render');check(t);bad=wire(t);bad['message']['plan']['project']['revision']+=1;bad['message']['plan']['request']['expected_revision']+=1
    with pytest.raises(c.ContractError,match='input_mismatch'):check(bad)
    t=scenario();check(t);bad=wire(t);bad['message']['plan']['resolved']['profile']['limits']['ram_bytes']+=1
    with pytest.raises(c.ContractError,match='input_mismatch'):check(bad)
