"""Complete synthetic catalogue-only E1 exchanges, not installed engine evidence."""
import copy
from . import validate as c


def qwen_settings(mode='qwen_preset'):
    return dict(kind=mode,language='French',seed=1,temperature=0.9,top_p=1,top_k=50,repetition_penalty=1.05,max_new_tokens=2048,instruction=None)


def catalogue():
    entries=[]
    for engine,model,mode,languages,selectors in [
        ('qwen','Qwen3-TTS-12Hz-1.7B-CustomVoice','qwen_preset',['French','English'],['Vivian','Serena']),
        ('qwen','Qwen3-TTS-12Hz-0.6B-Base','qwen_clone',['French','English'],[]),
        ('qwen','Qwen3-TTS-12Hz-1.7B-VoiceDesign','qwen_design',['French','English'],[]),
        ('kokoro','synthetic-kokoro','kokoro_preset',['fr','en'],['ff_siwis','af_heart']),
        ('pocket-tts','synthetic-pocket-fr','pocket_clone',['fr'],[])]:
        entries.append(dict(instructions=mode in ('qwen_preset','qwen_design'),engine=engine,model=model,mode=mode,bundle_sha256=c.canonical_hash([engine,model]),languages=languages,selectors=[dict(value=s,languages=(['fr'] if s=='ff_siwis' else ['en'] if s=='af_heart' else languages)) for s in selectors]))
    return entries


def identity(entry,device='cpu'):
    return {k:entry[k] for k in ('engine','model','bundle_sha256')}|dict(device=device,precision='float32')


def preset_version(i,provenance):
    return dict(id=f'vv{i}',voice_id=f'voice.{i}',origin='preset',reference_asset_ids=[],reference_text='',identity=identity(catalogue()[0]),settings=qwen_settings(),settings_sha256=c.canonical_hash(qwen_settings()),provenance=copy.deepcopy(provenance),configuration=dict(mode='qwen_preset',speaker='Vivian' if i%2==0 else 'Serena'),design_lineage=None)


def transcript(operation, project, profiles, requests, *, version=None, device='cpu', reuse_settings=None, reference_roles=False):
    """Trusted catalogues and previous typed replies are explicit in each transcript."""
    bundles=catalogue();p=copy.deepcopy(project);request=copy.deepcopy(requests[operation]);assets=[];versions=[]
    settings=qwen_settings();kind='speech'
    if operation=='speech_generate':
        version=copy.deepcopy(version or p['voice_versions'][0]);versions=[version]
        request.update(voice_version_id=version['id'],identity=copy.deepcopy(version['identity']),variants=1)
        settings=copy.deepcopy(version['settings'])
        assets=[a for a in p['assets'] if a['id'] in version['reference_asset_ids']]
    elif operation=='voice_prepare':
        kind='prepare';request.update(asset_id=p['assets'][0]['id'],in_tick=0,out_tick=4800,reference_text='Exact reference.',identity=identity(bundles[1]));assets=[p['assets'][0]]
        settings=dict(kind='prepare',conditioning='icl',output_format='wav_pcm16')
    elif operation=='voice_design':
        kind='design';request.update(description='Warm precise voice.',reference_text='Exact design audition.',identity=identity(bundles[2]));settings=qwen_settings('qwen_design');settings['instruction']=request['description']
    else:
        kind='render';request.update(project_id=p['id'],expected_revision=p['revision'],policy=p['render']);settings=dict(kind='render',policy=p['render'])
        assets=[a for a in p['assets'] if a['id'] in {x['asset_id'] for x in p['clips']}];versions=p['voice_versions']
    effective=copy.deepcopy(request.get('identity',dict(engine='pcm-render',model='pcm24-v1',bundle_sha256='0'*64,device=device,precision='float32')));effective['device']=device
    if 'identity' in request:request['identity']['device']=device
    profile=copy.deepcopy(next(x for x in profiles if x['device']==device));profile['compatibility']=[dict(operation=operation,mode=settings['kind'],identity=effective)]
    reuse_cfg=dict(mode='qwen_clone',reference_asset_id='output-'+kind,reference_text=request.get('reference_text',''),conditioning='icl') if kind in ('prepare','design') else None
    reuse_settings=copy.deepcopy(reuse_settings if reuse_settings is not None else qwen_settings('qwen_clone')) if reuse_cfg else None
    resolved=dict(reuse_configuration=reuse_cfg,reuse_settings=reuse_settings,reuse_identity=identity(bundles[1]) if kind=='design' else None,kind=kind,identity=dict(requested=copy.deepcopy(effective),effective=copy.deepcopy(effective),fallback='forbid',reason='',profile_id=profile['id']),profile=c.profile_snapshot(profile),operator_fallback='forbid',settings=settings,settings_sha256=c.canonical_hash(settings))
    plan=dict(schema_version='1.3.0',operation=operation,request=request,project=p if operation=='project_render' else None,voice_versions=versions,assets=assets,resolved=resolved)
    if reference_roles:
        if operation not in ('voice_prepare','voice_design'):raise ValueError('operation_mismatch')
        reference_id='reference-'+kind if kind=='prepare' else 'output-'+kind
        reuse_cfg['reference_asset_id']=reference_id
        resolved['reuse_configuration']=copy.deepcopy(reuse_cfg)
        plan['reference_roles']=dict(version='reference-roles-v1',reference_kind='prepared_reference' if kind=='prepare' else 'generated_reference',reference_asset_id=reference_id,audition_asset_id='output-'+kind,audition_text='Explicit generated audition.' if kind=='prepare' else request['reference_text'])
    job=dict(id='job-'+kind,operation=operation,input_sha256=c.plan_hash(plan),principal_id='principal1',idempotency_key='request-'+kind,state='running',generation=1,attempts=[dict(id='attempt-'+kind,generation=1,owner_id='worker1',staging_id='stage-'+kind,reaped=False)],artifact_ids=[],stage='loading',deadline='2026-10-16T00:10:00Z')
    attempt=job['attempts'][0]
    output=copy.deepcopy(p['assets'][0]);output['id']='output-'+kind
    if kind=='render':
        from .media import synthetic_media
        policy=request['policy'];output.update(synthetic_media(policy['length_ticks']*policy['sample_rate']//48000,policy['sample_rate'],policy['channels'],'WAV','PCM_24')[1])
        output['provenance']=dict(kind='synthetic',source_sha256=output['sha256'],source_id='E1-render-silence',missing_fields=[])
    candidates=[]
    if operation in ('voice_prepare','voice_design'):
        v=copy.deepcopy(p['voice_versions'][0]);v.update(id='prospective-'+kind,origin='clone' if kind=='prepare' else 'design',identity=identity(bundles[1]),reference_asset_ids=[output['id']],reference_text=request['reference_text'],configuration=copy.deepcopy(reuse_cfg),settings=copy.deepcopy(reuse_settings),settings_sha256=c.canonical_hash(reuse_settings))
        if kind=='design':v['design_lineage']=dict(description=request['description'],source_candidate_id='candidate-design',generation_identity=effective)
        candidates=[dict(schema_version='1.3.0',id='candidate-'+kind,job_id=job['id'],version=v,audition_asset_id=output['id'])]
    outputs=[output]
    if reference_roles:
        role_plan=plan['reference_roles'];v=candidates[0]['version']
        # Synthetic PCM fixtures exercise the real role relation, not real speech.
        from .media import synthetic_media
        output.update(synthetic_media(4800,48000,1,'WAV','PCM_16')[1])
        output['provenance']=dict(kind='synthetic',source_sha256=output['sha256'],source_id='E1-audition-silence',missing_fields=[])
        if kind=='prepare':
            reference=copy.deepcopy(assets[0]);reference['id']=role_plan['reference_asset_id']
            reference['provenance']=dict(kind=assets[0]['provenance']['kind'],source_id=assets[0]['id'],source_sha256=assets[0]['sha256'],missing_fields=[])
            ref_role=dict(kind='prepared_reference',asset=reference,original=copy.deepcopy(assets[0]),in_tick=request['in_tick'],out_tick=request['out_tick'],transcript=request['reference_text'])
            outputs=[reference,output]
        else:
            reference=output
            ref_role=dict(kind='generated_reference',asset=reference,transcript=request['reference_text'],lineage=copy.deepcopy(v['design_lineage']))
        v.update(reference_asset_ids=[reference['id']],configuration=copy.deepcopy(reuse_cfg),provenance=copy.deepcopy(reference['provenance']))
        candidates[0]['reference_roles']=dict(version='reference-roles-v1',reference=ref_role,audition=dict(kind='generated_audition',asset=copy.deepcopy(output),text=role_plan['audition_text']))
    manifest=dict(operation=operation,job_id=job['id'],attempt_id=attempt['id'],generation=1,owner_id=attempt['owner_id'],staging_id=attempt['staging_id'],input_sha256=job['input_sha256'],artifact_id='artifact-'+kind,outputs=outputs,candidates=candidates)
    message=dict(protocol_version='1.3.0',kind='result',job_id=job['id'],attempt_id=attempt['id'],generation=1,input_sha256=job['input_sha256'],input_asset_ids=c.required_asset_ids(plan),staging_id=attempt['staging_id'],owner_id=attempt['owner_id'],identity=resolved['identity'],deadline=job['deadline'],stage='completed',result_asset_ids=[a['id'] for a in outputs],plan=plan)
    artifact=dict(id=manifest['artifact_id'],job_id=job['id'],generation=1,asset_ids=message['result_asset_ids'],input_sha256=job['input_sha256'],project_revision=p['revision'] if kind=='render' else 0,manifest_sha256=c.canonical_hash(manifest),identity=resolved['identity'],attempt_id=attempt['id'],owner_id=attempt['owner_id'],staging_id=attempt['staging_id'])
    result=dict(schema_version='1.3.0',job_id=job['id'],candidates=candidates)
    exchanges=[dict(operation=operation,request=request,response=dict(job_id=job['id'],input_sha256=job['input_sha256'],state='queued',status_url='/v1/jobs/'+job['id']))]
    completed=copy.deepcopy(job);completed.update(state='succeeded',artifact_ids=[artifact['id']],stage='completed')
    exchanges += [dict(operation='job_get',request=dict(job_id=job['id']),response=completed),dict(operation='artifact_get',request=dict(artifact_id=artifact['id']),response=artifact),dict(operation='asset_get',request=dict(asset_id=output['id']),response=output)]
    if reference_roles and kind=='prepare':exchanges.append(dict(operation='asset_get',request=dict(asset_id=reference['id']),response=reference))
    if candidates:
        exchanges += [dict(operation='voice_results',request=dict(job_id=job['id']),response=result),dict(operation='voice_get',request=dict(voice_id='voice.0'),response=p['voices'][0])]
        saved=copy.deepcopy(candidates[0]['version']);saved['id']='saved-'+kind
        voice=copy.deepcopy(p['voices'][0]);voice['revision']+=1;voice['version_ids'].append(saved['id'])
        exchanges.append(dict(operation='voice_save', request=dict(job_id=job['id'],candidate_id=candidates[0]['id'],voice_id=voice['id'],expected_revision=p['voices'][0]['revision']), response=dict(voice=voice,version=saved)))
        exchanges += [dict(operation='voice_get',request=dict(voice_id=voice['id']),response=voice),dict(operation='voice_version_get',request=dict(voice_version_id=saved['id']),response=saved)]
    if operation=='project_render':exchanges.insert(0,dict(operation='project_get',request=dict(project_id=p['id']),response=p))
    if operation=='voice_prepare':exchanges.insert(0,dict(operation='asset_get',request=dict(asset_id=assets[0]['id']),response=assets[0]))
    if operation=='speech_generate':exchanges.insert(0,dict(operation='voice_version_get',request=dict(voice_version_id=versions[0]['id']),response=versions[0]))
    # Detach every nested wire object; there are deliberately no shared aliases.
    import json
    return json.loads(json.dumps(dict(evidence='E1 synthetic catalogue only',context=dict(plan=plan,bundles=bundles),profiles=[profile],authorized_assets=c.required_asset_ids(plan),job=job,message=message,artifact=artifact,verified_manifest=manifest,present_hashes={a['id']:a['sha256'] for a in outputs},exchanges=exchanges)))
