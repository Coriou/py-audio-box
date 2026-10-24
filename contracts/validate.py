"""E1 contract oracle. Not a store, executor, URL fetcher or audio renderer."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parent
SCHEMA = json.loads((ROOT / 'v1/schema.json').read_text())

class ContractError(ValueError):
    pass

def require(condition, code):
    if not condition:
        raise ContractError(code)

def validate(kind, value):
    # JSON Schema binary strings describe the wire body, not a JSON value.
    if kind == 'asset_upload_request':
        require(type(value) is bytes, 'invalid_bytes')
        require(0 < len(value) <= 24*1024*1024, 'invalid_size')
        return value
    Draft202012Validator({**SCHEMA, '$ref': '#/$defs/' + kind}).validate(value)
    intrinsic(value)
    if kind == 'project':
        project(value)
    elif kind == 'voice_version':
        voice_configuration(value)
    elif kind == 'worker_plan':
        frozen_plan(value)
    elif kind == 'capabilities':
        for profile in value['profiles']: validate('profile',profile)
    elif kind == 'resolution':
        resolution(value)
    elif kind == 'profile':
        require(value['support'] != 'qualified' or (value['verification'] == 'measured' and bool(value['qualification_receipt'])), 'qualification_required')
        require(value['readiness'] != 'ready' or (value['support'] == 'qualified' and value['installation'] == 'installed' and value['configuration'] == 'valid'), 'not_ready')
    return value

def fingerprint(principal, operation, payload):
    # v1 accepts JSON with finite numbers only; strings are not trimmed/normalized.
    return hashlib.sha256(json.dumps([principal, operation, payload], ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def idempotency(saved_fingerprint, principal, operation, payload):
    require(operation != 'asset_upload', 'non_idempotent')
    require(saved_fingerprint == fingerprint(principal, operation, payload), 'idempotency_conflict')

def revision(expected, current):
    require(expected == current, 'revision_conflict')

def ticks(seconds):
    return int((Decimal(str(seconds)) * 48000).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def resolution(r):
    a,b=r['requested'],r['effective']
    require(all(a[k] == b[k] for k in ('engine','model','bundle_sha256')), 'identity_substitution')
    require(b['device'] != 'auto' and b['precision'] != 'auto', 'unresolved_identity')
    require(a['precision'] == 'auto' or a['precision'] == b['precision'], 'precision_substitution')
    if a['device'] not in ('auto', b['device']):
        require(r['fallback'] == 'allow_cpu' and b['device'] == 'cpu' and bool(r['reason']), 'fallback_forbidden')

def stale(p,t):
    b=next(b for b in p['blocks'] if b['id']==t['block_id'])
    s=next(s for s in p['speakers'] if s['id']==b['speaker_id'])
    return t['settings_sha256'] != b['settings_sha256'] or t['text'] != b['text'] or t['voice_version_id'] != s['voice_version_id']

def project(p):
    groups={}
    for name in ('speakers','blocks','takes','tracks','clips','assets','voices','voice_versions'):
        groups[name]={x['id']:x for x in p[name]}
        require(len(groups[name]) == len(p[name]), 'duplicate_id')
    for v in p['voices']:
        require(all(i in groups['voice_versions'] and groups['voice_versions'][i]['voice_id']==v['id'] for i in v['version_ids']), 'voice_version_missing')
    aliases=[a for v in p['voices'] for a in [v['id']]+v['aliases']]
    require(len(set(aliases))==len(aliases), 'alias_collision')
    for v in p['voice_versions']:
        voice_configuration(v)
        require(v['voice_id'] in groups['voices'] and all(a in groups['assets'] for a in v['reference_asset_ids']), 'reference_missing')
    for s in p['speakers']:
        require(s['voice_version_id'] in groups['voice_versions'], 'voice_version_missing')
    for b in p['blocks']:
        require(b['speaker_id'] in groups['speakers'], 'speaker_missing')
        t=groups['takes'].get(b['selected_take_id'])
        require(b['selected_take_id'] is None or (t is not None and t['block_id']==b['id']), 'selection_invalid')
    for t in p['takes']:
        require(t['block_id'] in groups['blocks'] and t['asset_id'] in groups['assets'] and t['voice_version_id'] in groups['voice_versions'], 'take_reference_missing')
        resolution(t['identity'])
        v=groups['voice_versions'][t['voice_version_id']]
        require(all(t['identity']['effective'][k]==v['identity'][k] for k in ('engine','model','bundle_sha256')), 'identity_substitution')
    for c in p['clips']:
        require(c['track_id'] in groups['tracks'] and c['asset_id'] in groups['assets'], 'missing_media')
        a=groups['assets'][c['asset_id']]
        require(c['in_tick'] < c['out_tick'] <= ticks(Decimal(a['frames'])/a['sample_rate']), 'invalid_trim')
        duration=c['out_tick']-c['in_tick']
        require(c['fade_in']+c['fade_out'] <= duration, 'invalid_fades')
        require(c['start']+duration <= 9007199254740991, 'tick_overflow')
        if c['take_id'] is not None:
            require(c['take_id'] in groups['takes'] and groups['takes'][c['take_id']]['asset_id']==c['asset_id'], 'take_asset_mismatch')

def renderable(p, policy, peak=0):
    validate('project',p);validate('render_policy',policy)
    require(policy['include_stale'] or not any(stale(p,t) for t in p['takes'] if t['id'] in [b['selected_take_id'] for b in p['blocks']]), 'stale_take')
    require(peak <= 1, 'clipping')
    require(policy['length_ticks'] >= max((c['start']+c['out_tick']-c['in_tick'] for c in p['clips']),default=0), 'partial_requires_selection')

def transition(old, new, *, reaped=False, current_generation=0, generation=0, media_verified=False):
    edges={'queued':{'running','cancel_requested','failed'},'running':{'succeeded','failed','cancel_requested'},'cancel_requested':{'cancelled','succeeded','failed'}}
    require(new in edges.get(old,set()), 'invalid_transition')
    require(generation == current_generation, 'stale_attempt')
    if new=='cancelled': require(reaped, 'cancel_pending')
    if new=='succeeded': require(media_verified, 'missing_media')

def active_attempt(job):
    validate('job', job)
    generations=[a['generation'] for a in job['attempts']]
    require(generations == sorted(set(generations)), 'nonmonotonic_generation')
    require(bool(generations) and generations[-1] == job['generation'], 'stale_attempt')
    require(all(a['reaped'] for a in job['attempts'][:-1]), 'owner_not_reaped')
    return job['attempts'][-1]

def publication(job, attempt, artifact, present_hashes, *, transport_owner, context, manifest, verified_result, allowed_assets):
    current=active_attempt(job)
    require(job['state'] in ('running','cancel_requested'), 'invalid_transition')
    require(attempt == current and not current['reaped'], 'attempt_mismatch')
    validate('artifact', artifact)
    require(transport_owner == current['owner_id'], 'owner_mismatch')
    for key, expected in dict(job_id=job['id'], attempt_id=current['id'], generation=job['generation'], input_sha256=job['input_sha256'], staging_id=current['staging_id'], owner_id=current['owner_id']).items():
        require(artifact[key] == expected, 'attempt_mismatch')
    worker_message(verified_result,job,allowed_assets,transport_owner=transport_owner,context=context)
    require(verified_result['kind']=='result', 'result_mismatch')
    plan=context['plan']
    require(artifact['identity']==plan['resolved']['identity'], 'resolution_mismatch')
    require(artifact['project_revision']==(plan['project']['revision'] if plan['operation']=='project_render' else 0), 'revision_conflict')
    validate('result_manifest',manifest)
    for key in ('job_id','attempt_id','generation','input_sha256','staging_id','owner_id'):
        require(manifest[key]==artifact[key], 'manifest_attempt_mismatch')
    require(manifest['operation']==job['operation'] and manifest['artifact_id']==artifact['id'], 'result_mismatch')
    require(artifact['manifest_sha256']==canonical_hash(manifest), 'manifest_mismatch')
    outputs=unique(manifest['outputs'])
    require(set(verified_result['result_asset_ids'])==set(outputs), 'result_mismatch')
    require(bool(outputs) and len(artifact['asset_ids'])==len(set(artifact['asset_ids'])) and set(artifact['asset_ids'])==set(outputs), 'result_mismatch')
    require(len(outputs)==(plan['request']['variants'] if job['operation']=='speech_generate' else 1), 'result_mismatch')
    require(not set(outputs)&set(required_asset_ids(plan)), 'result_mismatch')
    require(all(present_hashes.get(k)==v['sha256'] for k,v in outputs.items()), 'missing_media')
    if job['operation'] in ('voice_prepare','voice_design'):
        require(all(a['media_format']=='wav' and a['media_subtype']=='PCM_16' for a in outputs.values()), 'output_format_mismatch')
    if job['operation']=='project_render':
        policy=plan['request']['policy']
        require(all(a['media_format']=='wav' and a['media_subtype']=='PCM_24' and a['sample_rate']==policy['sample_rate'] and a['channels']==policy['channels'] and a['frames']*48000==policy['length_ticks']*policy['sample_rate'] for a in outputs.values()), 'output_policy_mismatch')
    candidates=unique(manifest['candidates'])
    if job['operation'] in ('voice_prepare','voice_design'):
        require(len(candidates)==1, 'candidate_missing')
        for candidate in candidates.values():
            require(candidate['job_id']==job['id'] and candidate['audition_asset_id'] in outputs, 'result_mismatch')
            version=candidate['version'];voice_configuration(version,context['bundles'])
            cfg=version['configuration']
            require(version['origin']==('design' if job['operation']=='voice_design' else 'clone'), 'origin_mismatch')
            require(cfg.get('reference_asset_id')==candidate['audition_asset_id'], 'reference_mismatch')
            require(version['reference_text']==plan['request']['reference_text'], 'reference_mismatch')
            require(cfg==plan['resolved']['reuse_configuration'] and version['settings']==plan['resolved']['reuse_settings'], 'reuse_configuration_mismatch')
            if job['operation']=='voice_design':
                require(version['identity']==plan['resolved']['reuse_identity'] and cfg.get('conditioning')=='icl', 'identity_substitution')
                lineage=version['design_lineage']
                require(lineage['generation_identity']==plan['resolved']['identity']['effective'] and lineage['description']==plan['request']['description'] and lineage['source_candidate_id']==candidate['id'], 'design_lineage_mismatch')
            else:
                require(all(version['identity'][k]==plan['resolved']['identity']['effective'][k] for k in ('engine','model','bundle_sha256')), 'identity_substitution')
                require(cfg.get('conditioning','pocket')==plan['resolved']['settings']['conditioning'], 'conditioning_mismatch')
    else:
        require(not candidates, 'result_mismatch')

def replacement(job, attempt, *, owner_dead):
    old=active_attempt(job)
    require(owner_dead and old['reaped'], 'owner_not_reaped')
    validate('attempt', attempt)
    require(attempt['generation'] == job['generation']+1 and attempt['id'] not in [a['id'] for a in job['attempts']] and attempt['staging_id'] not in [a['staging_id'] for a in job['attempts']] and not attempt['reaped'], 'invalid_replacement')


def ripple(p, track_id, start, end, anchor, delta, ids):
    require(start<=anchor<=end, 'invalid_range')
    affected=[c for c in p['clips'] if c['track_id']==track_id and start<=c['start']<end]
    require(set(ids)=={c['id'] for c in affected} and len(ids)==len(affected), 'affected_ids_mismatch')
    require(all(0<=c['start']+delta and c['start']+delta+c['out_tick']-c['in_tick']<=9007199254740991 for c in affected), 'invalid_range')
    result=copy.deepcopy(p)
    for clip in result['clips']:
        if clip['id'] in ids: clip['start']+=delta
    validate('project', result)
    return result

def immutable(before, after):
    require(before==after, 'immutable_identity')

def migration(report, *, source_root, destination_root, pending_policy='leave_legacy'):
    require(Path(source_root).resolve()!=Path(destination_root).resolve(), 'distinct_root_required')
    require(not report['collisions'], 'migration_collision')
    require(pending_policy=='leave_legacy', 'pending_jobs_not_importable')
    require(report['apply_allowed'], 'migration_not_approved')

def canonical_hash(value):
    """Canonical preimage: finite JSON, sorted keys, compact UTF-8, exact strings."""
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def unique(items):
    result={v['id']:v for v in items}
    require(len(result)==len(items), 'duplicate_id')
    return result


def voice_configuration(version, bundles=None):
    cfg=version['configuration'];mode=cfg['mode'];identity=version['identity']
    validate('speech_settings',version['settings'])
    require(version['settings_sha256']==canonical_hash(version['settings']), 'settings_mismatch')
    engine={'qwen_preset':'qwen','kokoro_preset':'kokoro','qwen_clone':'qwen','pocket_clone':'pocket-tts'}[mode]
    require(identity['engine']==engine, 'configuration_engine_mismatch')
    require(version['settings']['kind']==mode, 'settings_mode_mismatch')
    require(version['origin']!='design' or mode=='qwen_clone', 'origin_mismatch')
    clone=mode in ('qwen_clone','pocket_clone')
    require(version['origin'] in (('clone','design') if clone else ('preset',)), 'origin_mismatch')
    require(version['reference_asset_ids']==([cfg['reference_asset_id']] if clone else []), 'reference_mismatch')
    require(version['reference_text']==cfg.get('reference_text',''), 'reference_mismatch')
    require((version['design_lineage'] is not None)==(version['origin']=='design'), 'design_lineage_mismatch')
    if mode=='qwen_clone':
        require(version['settings']['instruction'] is None, 'unsupported_instruction')
        require(cfg['conditioning']!='icl' or bool(cfg['reference_text']), 'conditioning_mismatch')
        require(cfg['conditioning']!='x_vector' or cfg['reference_text']=='', 'conditioning_mismatch')
    if bundles is not None:
        if version['origin']=='design':select_bundle(version['design_lineage']['generation_identity'],'qwen_design',bundles)
        bundle=select_bundle(identity,mode,bundles)
        if not clone:
            selector=cfg['speaker'] if mode=='qwen_preset' else cfg['preset']
            matches=[v for v in bundle['selectors'] if v['value']==selector]
            require(len(matches)==1, 'unknown_selector')
            if mode=='kokoro_preset':require(cfg['language'] in matches[0]['languages'], 'language_mismatch')
        require(version['settings']['language'] in bundle['languages'], 'language_mismatch')
        if mode=='qwen_preset':
            require(version['settings']['language'] in matches[0]['languages'], 'language_mismatch')
            require(version['settings']['instruction'] is None or bundle['instructions'], 'unsupported_instruction')
        if 'language' in cfg:
            require(version['settings']['language']==cfg['language'], 'language_mismatch')
        if 'language' in cfg:require(cfg['language'] in bundle['languages'], 'language_mismatch')


def select_bundle(identity,mode,bundles):
    for b in bundles:validate('bundle_catalogue_entry',b)
    matches=[b for b in bundles if b['mode']==mode and all(b[k]==identity[k] for k in ('engine','model','bundle_sha256'))]
    require(len(matches)==1, 'incompatible_bundle')
    return matches[0]


def required_asset_ids(plan):
    op=plan['operation']
    if op=='voice_prepare':return [plan['request']['asset_id']]
    if op=='voice_design':return []
    if op=='project_render':return sorted({c['asset_id'] for c in plan['project']['clips']})
    versions=unique(plan['voice_versions'])
    require(plan['request']['voice_version_id'] in versions, 'voice_version_missing')
    cfg=versions[plan['request']['voice_version_id']]['configuration']
    return [cfg['reference_asset_id']] if 'reference_asset_id' in cfg else []


def profile_snapshot(profile):
    return {k:copy.deepcopy(profile[k]) for k in ('id','config_revision','device','topology','limits','compatibility')}


def frozen_plan(plan, bundles=None):
    request=plan['request'];op=plan['operation'];r=plan['resolved'];identity=r['identity']
    resolution(identity)
    require(r['kind']=={'speech_generate':'speech','voice_prepare':'prepare','voice_design':'design','project_render':'render'}[op], 'operation_mismatch')
    require(r['settings_sha256']==canonical_hash(r['settings']), 'settings_mismatch')
    profile=r['profile'];effective=identity['effective'];settings=r['settings'];mode=settings['kind']
    require(identity['profile_id']==profile['id'] and effective['device']==profile['device'], 'profile_mismatch')
    if effective['engine'] in ('kokoro','pocket-tts','pcm-render'):
        require(effective['device']=='cpu' and effective['precision']=='float32', 'unsupported_execution')
    require(profile['topology']==('packaged_native' if profile['device']=='metal' else 'networkless_container'), 'profile_mismatch')
    require(dict(operation=op,mode=mode,identity=effective) in profile['compatibility'], 'profile_incompatible')
    require(identity['fallback']!='allow_cpu' or (request.get('fallback','forbid')=='allow_cpu' and r['operator_fallback']=='allow_cpu'), 'fallback_policy_mismatch')
    if op in ('voice_prepare','voice_design'):
        require(r['reuse_settings'] is not None and r['reuse_configuration'] is not None, 'reuse_configuration_mismatch')
        cfg=r['reuse_configuration'];reuse=r['reuse_settings']
        target=r['reuse_identity'] if op=='voice_design' else effective
        require(target is not None, 'unresolved_identity')
        prospective=dict(origin='design' if op=='voice_design' else 'clone',identity=target,configuration=cfg,settings=reuse,settings_sha256=canonical_hash(reuse),reference_asset_ids=[cfg.get('reference_asset_id')],reference_text=request['reference_text'],design_lineage=dict(generation_identity=effective) if op=='voice_design' else None)
        voice_configuration(prospective,bundles)
        require(cfg.get('reference_text',request['reference_text'])==request['reference_text'], 'reference_mismatch')
        require(op!='voice_design' or cfg.get('conditioning')=='icl', 'conditioning_mismatch')
        require(op!='voice_prepare' or cfg.get('conditioning','pocket')==settings['conditioning'], 'conditioning_mismatch')
    else:require(r['reuse_settings'] is None and r['reuse_configuration'] is None, 'operation_mismatch')
    if op!='voice_design':require(r['reuse_identity'] is None, 'operation_mismatch')
    if op!='project_render':require(identity['requested']==request['identity'], 'identity_substitution')
    assets=unique(plan['assets']);versions=unique(plan['voice_versions'])
    for version in versions.values():
        voice_configuration(version,bundles);validate('voice_version',version)
    if op=='speech_generate':
        require(plan['project'] is None, 'unrelated_project')
        require(len(versions)==1 and request['voice_version_id'] in versions, 'voice_version_missing')
        voice=versions[request['voice_version_id']];cfg=voice['configuration']
        require(all(effective[k]==voice['identity'][k] for k in ('engine','model','bundle_sha256')), 'identity_substitution')
        require(mode==cfg['mode'], 'settings_mode_mismatch')
        if mode in ('kokoro_preset','pocket_clone'):require(settings['language']==cfg['language'], 'language_mismatch')
        if mode=='qwen_clone':require(settings['instruction'] is None, 'unsupported_instruction')
        require(voice['settings']==settings and voice['settings_sha256']==r['settings_sha256'], 'settings_mismatch')
        if bundles is not None:
            bundle=select_bundle(effective,mode,bundles)
            require(settings['language'] in bundle['languages'], 'language_mismatch')
            if mode=='qwen_preset':
                selector=next(v for v in bundle['selectors'] if v['value']==cfg['speaker'])
                require(settings['language'] in selector['languages'], 'language_mismatch')
                require(settings['instruction'] is None or bundle['instructions'], 'unsupported_instruction')
    elif op=='voice_prepare':
        require(plan['project'] is None and not versions and mode=='prepare', 'operation_mismatch')
        require(request['asset_id'] in assets, 'missing_media')
        asset=assets[request['asset_id']]
        require(request['in_tick']<request['out_tick']<=ticks(Decimal(asset['frames'])/asset['sample_rate']), 'invalid_trim')
        require((effective['engine']=='qwen' and settings['conditioning'] in ('icl','x_vector')) or (effective['engine']=='pocket-tts' and settings['conditioning']=='pocket'), 'conditioning_mismatch')
        require(settings['conditioning']!='icl' or bool(request['reference_text']), 'conditioning_mismatch')
        require(settings['conditioning']=='icl' or request['reference_text']=='', 'conditioning_mismatch')
        if bundles is not None:select_bundle(effective,'qwen_clone' if effective['engine']=='qwen' else 'pocket_clone',bundles)
    elif op=='voice_design':
        require(plan['project'] is None and not versions and mode=='qwen_design' and effective['engine']=='qwen', 'operation_mismatch')
        require(settings['instruction']==request['description'], 'unsupported_instruction')
        require(r['reuse_identity'] is not None and r['reuse_identity']['device']!='auto' and r['reuse_identity']['precision']!='auto', 'unresolved_identity')
        require(bool(request['reference_text']), 'conditioning_mismatch')
        if bundles is not None:
            select_bundle(r['reuse_identity'],'qwen_clone',bundles)
            bundle=select_bundle(effective,'qwen_design',bundles)
            require(settings['language'] in bundle['languages'], 'language_mismatch')
    else:
        p=plan['project'];require(p is not None, 'missing_project');validate('project',p)
        require(mode=='render' and effective['engine']=='pcm-render' and identity['requested']==effective and identity['fallback']=='forbid', 'operation_mismatch')
        require(request['project_id']==p['id'] and request['expected_revision']==p['revision'], 'revision_conflict')
        require(settings['policy']==request['policy'], 'settings_mismatch')
        require(plan['voice_versions']==p['voice_versions'], 'voice_version_missing')
        for a in p['assets']:
            require(a['id'] not in assets or assets[a['id']]==a, 'asset_metadata_mismatch')
        renderable(p,request['policy'])
    required=set(required_asset_ids(plan))
    if op=='speech_generate' and mode in ('qwen_clone','pocket_clone'):
        require(all(assets.get(k,{}).get('media_format')=='wav' for k in required), 'reference_format_mismatch')
    require(set(assets)==required, 'asset_dependencies_mismatch')
    if op=='project_render':
        source=unique(plan['project']['assets'])
        require(all(source.get(k)==v for k,v in assets.items()), 'asset_metadata_mismatch')
    limits=profile['limits']
    require(len(request.get('text',request.get('reference_text','')))<=limits['text_chars'] and request.get('variants',1)<=limits['variants'], 'profile_limit')
    require(all(ticks(Decimal(a['frames'])/a['sample_rate'])<=limits['decoded_ticks'] for a in assets.values()), 'profile_limit')


def plan_hash(plan):
    validate('worker_plan',plan)
    return canonical_hash(plan)


def stored_context(job, context):
    plan=context['plan']
    require(job['input_sha256']==plan_hash(plan) and job['operation']==plan['operation'], 'input_mismatch')
    frozen_plan(plan,context['bundles'])
    return plan


def admission(job, context, profiles):
    """Control-owned installed/qualification state; mocks are E1 only."""
    plan=stored_context(job,context);snapshot=plan['resolved']['profile']
    matches=[p for p in profiles if p['id']==snapshot['id']]
    require(len(matches)==1, 'unknown_profile')
    p=matches[0];validate('profile',p)
    require(profile_snapshot(p)==snapshot, 'profile_mismatch')
    require(p['readiness']=='ready' and p['support']=='qualified' and p['installation']=='installed' and p['configuration']=='valid', 'not_ready')


def worker_message(message, job, allowed_assets, *, transport_owner, context, current_profiles=()):
    validate('worker_message',message)
    attempt=active_attempt(job)
    require(job['state'] in ('running','cancel_requested') and not attempt['reaped'], 'invalid_transition')
    require(transport_owner == attempt['owner_id'], 'owner_mismatch')
    expected=dict(job_id=job['id'],attempt_id=attempt['id'],generation=job['generation'],input_sha256=job['input_sha256'],staging_id=attempt['staging_id'],owner_id=attempt['owner_id'])
    require(all(message[k]==v for k,v in expected.items()), 'attempt_mismatch')
    plan=stored_context(job,context)
    if message['kind']=='execute':admission(job,context,current_profiles)
    require(message['plan']==plan and message['input_sha256']==plan_hash(message['plan']), 'input_mismatch')
    require(message['identity']==plan['resolved']['identity'], 'resolution_mismatch')
    require(message['deadline']==job['deadline'], 'deadline_mismatch')
    ids=message['input_asset_ids'];required=set(required_asset_ids(plan))
    require(len(ids)==len(set(ids)) and set(ids)==required, 'asset_dependencies_mismatch')
    require(required<=set(allowed_assets), 'unmanaged_asset')
    require(len(message['result_asset_ids'])==len(set(message['result_asset_ids'])), 'duplicate_id')


def input_content(plan, asset_id, *, authorized_assets):
    """Transfer authorization relation; runtime must also scope job/attempt and hash bytes."""
    require(asset_id in required_asset_ids(plan) and asset_id in authorized_assets, 'unmanaged_asset')
    return unique(plan['assets'])[asset_id]

def job_input_content(data, request, job, *, context, attempt_id, transport_owner, authorized_assets):
    """E1 transfer relation; the host/runtime supplies actual bytes and current ACL."""
    plan=stored_context(job,context);attempt=active_attempt(job)
    require(job['state'] in ('running','cancel_requested') and not attempt['reaped'], 'invalid_transition')
    require(attempt_id==attempt['id'] and transport_owner==attempt['owner_id'], 'attempt_mismatch')
    asset=input_content(plan,request['asset_id'],authorized_assets=authorized_assets)
    require(hashlib.sha256(data).hexdigest()==asset['sha256'], 'invalid_hash')
    return content(data,request,authorized=True)


def intrinsic(value):
    if isinstance(value, dict):
        if {'requested','effective','fallback','reason','profile_id'} <= value.keys(): resolution(value)
        if {'configuration','origin','voice_id','reference_asset_ids'} <= value.keys(): voice_configuration(value)
        for child in value.values(): intrinsic(child)
    elif isinstance(value,list):
        for child in value: intrinsic(child)

def upload(request):
    """Raw immutable body oracle; media validation belongs to owner ingest.

    No JSON/base64 envelope, caller hash/size, or idempotent receipt exists.
    HTTP separately enforces authorization, length, deadline and admission.
    """
    return validate('asset_upload_request',request)

def content(data, request, *, authorized):
    validate('content_request',request)
    require(authorized, 'forbidden')
    start,end=request['start'],request['end_exclusive']
    require(0<=start<end<=len(data) and end-start<=16777216, 'invalid_range')
    result=dict(schema_version='1.3.0',asset_id=request['asset_id'],total_bytes=len(data),start=start,end_exclusive=end,content_sha256=hashlib.sha256(data).hexdigest(),content_base64=base64.b64encode(data[start:end]).decode())
    validate('content_result',result)
    return result


def edit_semantics(before, after, operation):
    """Validate non-interference on snapshots; consumers still implement atomic edits."""
    validate('project',before)
    require(after['id']==before['id'] and after['revision']==before['revision']+1, 'revision_conflict')
    for group in ('takes','assets','voice_versions'):
        new={x['id']:x for x in after[group]}
        for old in before[group]:
            require(new.get(old['id']) == old, 'immutable_identity')
    validate('project',after)
    if operation in ('block_text','blocks_reorder','candidate_arrival'):
        require(before['clips']==after['clips'] and before['tracks']==after['tracks'], 'implicit_arrangement')
    if operation=='candidate_arrival':
        old={b['id']:b['selected_take_id'] for b in before['blocks']}
        require(old=={b['id']:b['selected_take_id'] for b in after['blocks']}, 'implicit_selection')
        for take in before['takes']:
            require(take in after['takes'], 'immutable_identity')
