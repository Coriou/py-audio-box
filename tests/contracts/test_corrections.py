import base64
import copy
import hashlib
import json
from pathlib import Path
import pytest
from contracts import validate as c
from test_contracts import load, FIX


def test_exact_byte_upload_download_transcript():
    data=(FIX/'impulse.wav').read_bytes()
    request=dict(content_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source='microphone',content_base64=base64.b64encode(data).decode())
    assert c.upload(request)==data
    h=c.fingerprint('agent','asset_upload',request)
    c.idempotency(h,'agent','asset_upload',request)
    chunks=[]
    for start,end in [(0,44),(44,len(data))]:
        r=c.content(data,dict(asset_id='impulse',start=start,end_exclusive=end),authorized=True)
        chunks.append(base64.b64decode(r['content_base64']))
        assert r['content_sha256']==request['content_sha256']
    assert b''.join(chunks)==data
    for update,error in [({'bytes':len(data)+1},'invalid_size'),({'content_sha256':'0'*64},'invalid_hash')]:
        with pytest.raises(c.ContractError,match=error):c.upload({**request,**update})
    for start,end in [(1,1),(2,1),(0,len(data)+1)]:
        with pytest.raises(c.ContractError,match='invalid_range'):c.content(data,dict(asset_id='impulse',start=start,end_exclusive=end),authorized=True)
    with pytest.raises(c.ContractError,match='forbidden'):c.content(data,dict(asset_id='impulse',start=0,end_exclusive=1),authorized=False)


def execution():
    t=load('transcript-speech_generate.json')
    return t['job'],t['message'],t['artifact'],t['context'],t['verified_manifest']


def test_authenticated_attempt_reconnect_cancel_and_replacement():
    j,m,artifact,context,manifest=execution();a=j['attempts'][0]
    for state in ['running','cancel_requested']:
        j['state']=state
        c.worker_message(m,j,[],transport_owner='worker1',context=context)
        c.publication(j,a,artifact,{manifest['outputs'][0]['id']:manifest['outputs'][0]['sha256']},transport_owner='worker1',context=context,manifest=manifest,verified_result=m,allowed_assets=[])
    for key,value in [('job_id','other'),('attempt_id','other'),('owner_id','other'),('input_sha256','f'*64),('staging_id','other'),('generation',2)]:
        bad={**m,key:value}
        with pytest.raises(c.ContractError):c.worker_message(bad,j,[],transport_owner='worker1',context=context)
        bad={**artifact,key:value}
        with pytest.raises(c.ContractError):c.publication(j,a,bad,{manifest['outputs'][0]['id']:manifest['outputs'][0]['sha256']},transport_owner='worker1',context=context,manifest=manifest,verified_result=m,allowed_assets=[])
    with pytest.raises(c.ContractError,match='owner_mismatch'):c.worker_message(m,j,[],transport_owner='rogue',context=context)
    with pytest.raises(c.ContractError,match='attempt_mismatch'):c.publication(j,{**a,'id':'rogue'},artifact,{manifest['outputs'][0]['id']:manifest['outputs'][0]['sha256']},transport_owner='worker1',context=context,manifest=manifest,verified_result=m,allowed_assets=[])
    new=dict(id='attempt2',generation=2,owner_id='worker2',staging_id='stage2',reaped=False)
    with pytest.raises(c.ContractError,match='owner_not_reaped'):c.replacement(j,new,owner_dead=True)
    a['reaped']=True
    with pytest.raises(c.ContractError):c.replacement(j,new,owner_dead=False)
    c.replacement(j,new,owner_dead=True)
    j['attempts'].append(new);j['generation']=2
    with pytest.raises(c.ContractError):c.worker_message(m,j,[],transport_owner='worker1',context=context)
    j,m,artifact,context,manifest=execution();j['state']='cancelled'
    with pytest.raises(c.ContractError):c.publication(j,j['attempts'][0],artifact,{manifest['outputs'][0]['id']:manifest['outputs'][0]['sha256']},transport_owner='worker1',context=context,manifest=manifest,verified_result=m,allowed_assets=[])


@pytest.mark.parametrize('kind',['artifact','take','worker_message'])
@pytest.mark.parametrize('change',['model','bundle_sha256','device','precision','fallback'])
def test_intrinsic_identity_at_resource_entrypoints(kind,change):
    j,m,a,context,manifest=execution();value={'artifact':a,'take':load('production-30.json')['takes'][0],'worker_message':m}[kind]
    if change=='fallback':value['identity']['requested']['device']='cuda'
    else:value['identity']['effective'][change]={'model':'substitution','bundle_sha256':'f'*64,'device':'auto','precision':'auto'}[change]
    with pytest.raises(c.ContractError):c.validate(kind,value)


@pytest.mark.parametrize('operation',['block_text','blocks_reorder','candidate_arrival'])
@pytest.mark.parametrize('group',['takes','assets','voice_versions'])
def test_edit_preserves_existing_immutable_resources(operation,group):
    p=load('production-30.json');q=copy.deepcopy(p);q['revision']+=1
    if operation=='block_text':q['blocks'][0]['text']='Edited'
    q['takes'].append({**q['takes'][0],'id':'new-candidate'})
    c.edit_semantics(p,q,operation)
    field={'takes':'text','assets':'sha256','voice_versions':'reference_text'}[group]
    q[group][0][field]='f'*64 if field=='sha256' else 'rewritten'
    with pytest.raises(c.ContractError,match='immutable_identity'):c.edit_semantics(p,q,operation)


def test_ripple_exact_max_and_atomic_overflow():
    p=load('production-30.json');maximum=9007199254740991
    p['clips'][0]['start']=maximum-4801;s=p['clips'][0]['start'];original=copy.deepcopy(p)
    q=c.ripple(p,'speech',s,maximum,s,1,['c0'])
    assert q['clips'][0]['start']+4800==maximum
    with pytest.raises(c.ContractError,match='invalid_range'):c.ripple(p,'speech',s,maximum,s,2,['c0'])
    assert p==original


def test_voice_history_and_frozen_plan_reconnect_workflows():
    p=load('production-30.json');v=p['voices'][0];c.validate('voice',v)
    for operation in ['voice_prepare','voice_design']:
        candidate=dict(schema_version='1.2.0',id='candidate1',job_id='job1',version=p['voice_versions'][0],audition_asset_id='impulse')
        result=dict(schema_version='1.2.0',job_id='job1',candidates=[candidate]);c.validate('voice_results',result)
        save=dict(candidate_id=result['candidates'][0]['id'],voice_id=v['id'],expected_revision=v['revision']);c.validate('voice_save_request',save)
        c.revision(save['expected_revision'],v['revision'])
        with pytest.raises(c.ContractError):c.revision(save['expected_revision'],v['revision']+1)
        c.validate('voice_version',candidate['version'])
    requests=load('operation-requests.json');record=dict(project_id=p['id'],revision=1,parent_revision=0,operation_id='edit1',actor_id='human',state_sha256='0'*64,changed_ids=['b0'],recorded=dict(operation='block_text',request=requests['block_text']),snapshot=p)
    c.validate('revision',record)
    c.validate('history',dict(schema_version='1.2.0',project_id=p['id'],revisions=[record],next_cursor=None))
    restore=requests['project_restore'];restore['source_revision']=record['revision'];c.validate('project_restore_request',restore)
    c.revision(2,2)
    with pytest.raises(c.ContractError):c.revision(2,3)  # reconnect after another writer; no silent undo
    j,m,a,context,manifest=execution();original=c.plan_hash(m['plan']);m['plan']['request']['text']='changed'
    assert c.plan_hash(m['plan'])!=original
    with pytest.raises(c.ContractError,match='input_mismatch'):c.worker_message(m,j,[],transport_owner='worker1',context=context)

@pytest.mark.parametrize('field,value',[('text','rewritten'),('settings_sha256','f'*64),('voice_version_id','vv1')])
def test_block_edit_cannot_rewrite_old_take(field,value):
    p=load('production-30.json');q=copy.deepcopy(p);q['revision']+=1;q['blocks'][0]['text']='Edited'
    q['takes'][0][field]=value
    with pytest.raises(c.ContractError,match='immutable_identity'):c.edit_semantics(p,q,'block_text')


def test_upload_oversize_and_byte_idempotency():
    from jsonschema import ValidationError
    request=dict(content_sha256='0'*64,bytes=16777217,source='upload',content_base64='')
    with pytest.raises(ValidationError):c.upload(request)
    request.update(bytes=1,content_base64='@@')
    with pytest.raises(c.ContractError,match='invalid_bytes'):c.upload(request)
    h=c.fingerprint('agent','asset_upload',request)
    with pytest.raises(c.ContractError,match='idempotency_conflict'):c.idempotency(h,'agent','asset_upload',{**request,'content_base64':'AA=='})


def test_all_frozen_operation_plans_have_executable_inputs():
    for op in ['speech_generate','voice_prepare','voice_design','project_render']:
        t=load('transcript-'+op+'.json');plan=t['message']['plan']
        c.validate('worker_plan',plan)
        frozen=json.loads(json.dumps(plan));digest=c.plan_hash(frozen)
        later=load('production-30.json');later['blocks'][0]['text']='later client edit'
        assert c.plan_hash(frozen)==digest
    plan['request']['expected_revision']=2
    with pytest.raises(c.ContractError,match='revision_conflict'):c.validate('worker_plan',plan)
