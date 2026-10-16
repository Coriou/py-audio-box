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
    Draft202012Validator({**SCHEMA, '$ref': '#/$defs/' + kind}).validate(value)
    intrinsic(value)
    if kind == 'project':
        project(value)
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

def publication(job, attempt, artifact, present_hashes, *, transport_owner):
    current=active_attempt(job)
    require(job['state'] in ('running','cancel_requested'), 'invalid_transition')
    require(attempt == current and not current['reaped'], 'attempt_mismatch')
    validate('artifact', artifact)
    require(transport_owner == current['owner_id'], 'owner_mismatch')
    for key, expected in dict(job_id=job['id'], attempt_id=current['id'], generation=job['generation'], input_sha256=job['input_sha256'], staging_id=current['staging_id'], owner_id=current['owner_id']).items():
        require(artifact[key] == expected, 'attempt_mismatch')
    require(bool(artifact['asset_ids']) and all(x in present_hashes for x in artifact['asset_ids']), 'missing_media')

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

def worker_message(message, job, allowed_assets, *, transport_owner):
    validate('worker_message',message)
    attempt=active_attempt(job)
    require(job['state'] in ('running','cancel_requested') and not attempt['reaped'], 'invalid_transition')
    require(transport_owner == attempt['owner_id'], 'owner_mismatch')
    expected=dict(job_id=job['id'],attempt_id=attempt['id'],generation=job['generation'],input_sha256=job['input_sha256'],staging_id=attempt['staging_id'],owner_id=attempt['owner_id'])
    require(all(message[k]==v for k,v in expected.items()), 'attempt_mismatch')
    require(set(message['input_asset_ids'])<=set(allowed_assets), 'unmanaged_asset')
    require(message['input_sha256']==plan_hash(message['plan']), 'input_mismatch')
    require(message['plan']['operation']==job['operation'], 'input_mismatch')
    require(message['identity']['requested']==message['plan']['request'].get('identity', message['identity']['requested']), 'identity_substitution')

def frozen_plan(plan):
    validate('project',plan['project'])
    request=plan['request'];op=plan['operation']
    assets={a['id']:a for a in plan['assets']}
    versions={v['id']:v for v in plan['voice_versions']}
    require(len(assets)==len(plan['assets']) and len(versions)==len(plan['voice_versions']), 'duplicate_id')
    require(all(x in assets for v in versions.values() for x in v['reference_asset_ids']), 'reference_missing')
    if op=='speech_generate':
        require(request['voice_version_id'] in versions, 'voice_version_missing')
        voice=versions[request['voice_version_id']]
        require(all(request['identity'][k]==voice['identity'][k] for k in ('engine','model','bundle_sha256')), 'identity_substitution')
    elif op=='voice_prepare':
        require(request['asset_id'] in assets, 'missing_media')
        asset=assets[request['asset_id']]
        require(request['in_tick']<request['out_tick']<=ticks(Decimal(asset['frames'])/asset['sample_rate']), 'invalid_trim')
    elif op=='project_render':
        p=plan['project']
        require(request['project_id']==p['id'] and request['expected_revision']==p['revision'], 'revision_conflict')
        require(all(assets.get(a['id'])==a for a in p['assets']), 'missing_media')
        require(all(versions.get(v['id'])==v for v in p['voice_versions']), 'voice_version_missing')
        renderable(p,request['policy'])

def plan_hash(plan):
    validate('worker_plan',plan)
    return hashlib.sha256(json.dumps(plan,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()

def intrinsic(value):
    if isinstance(value, dict):
        if {'requested','effective','fallback','reason','profile_id'} <= value.keys(): resolution(value)
        for child in value.values(): intrinsic(child)
    elif isinstance(value,list):
        for child in value: intrinsic(child)

def upload(request):
    validate('asset_upload_request',request)
    try: data=base64.b64decode(request['content_base64'],validate=True)
    except (ValueError, base64.binascii.Error): raise ContractError('invalid_bytes')
    require(len(data)==request['bytes'] and len(data)<=16777216, 'invalid_size')
    require(hashlib.sha256(data).hexdigest()==request['content_sha256'], 'invalid_hash')
    return data

def content(data, request, *, authorized):
    validate('content_request',request)
    require(authorized, 'forbidden')
    start,end=request['start'],request['end_exclusive']
    require(0<=start<end<=len(data) and end-start<=16777216, 'invalid_range')
    result=dict(schema_version='1.1.0',asset_id=request['asset_id'],total_bytes=len(data),start=start,end_exclusive=end,content_sha256=hashlib.sha256(data).hexdigest(),content_base64=base64.b64encode(data[start:end]).decode())
    validate('content_result',result)
    return result


def edit_semantics(before, after, operation):
    """Validate non-interference on snapshots; consumers still implement atomic edits."""
    validate('project',before);validate('project',after)
    require(after['id']==before['id'] and after['revision']==before['revision']+1, 'revision_conflict')
    for group in ('takes','assets','voice_versions'):
        new={x['id']:x for x in after[group]}
        for old in before[group]:
            require(new.get(old['id']) == old, 'immutable_identity')
    if operation in ('block_text','blocks_reorder','candidate_arrival'):
        require(before['clips']==after['clips'] and before['tracks']==after['tracks'], 'implicit_arrangement')
    if operation=='candidate_arrival':
        old={b['id']:b['selected_take_id'] for b in before['blocks']}
        require(old=={b['id']:b['selected_take_id'] for b in after['blocks']}, 'implicit_selection')
        for take in before['takes']:
            require(take in after['takes'], 'immutable_identity')
