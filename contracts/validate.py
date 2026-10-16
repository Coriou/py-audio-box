"""E1 contract oracle. Not a store, executor, URL fetcher or audio renderer."""
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
    if kind == 'project':
        project(value)
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

def publication(job, attempt, artifact, present_hashes):
    require(job['state'] in ('running','cancel_requested'), 'invalid_transition')
    require(attempt['generation']==job['generation']==artifact['generation'], 'stale_attempt')
    require(artifact['job_id']==job['id'] and artifact['input_sha256']==job['input_sha256'], 'identity_substitution')
    require(bool(artifact['asset_ids']) and all(x in present_hashes for x in artifact['asset_ids']), 'missing_media')

def ripple(p, track_id, start, end, anchor, delta, ids):
    require(start<=anchor<=end, 'invalid_range')
    affected=[c for c in p['clips'] if c['track_id']==track_id and start<=c['start']<end]
    require(set(ids)=={c['id'] for c in affected} and len(ids)==len(affected), 'affected_ids_mismatch')
    require(all(0<=c['start']+delta<=9007199254740991 for c in affected), 'invalid_range')

def immutable(before, after):
    require(before==after, 'immutable_identity')

def migration(report, *, source_root, destination_root, pending_policy='leave_legacy'):
    require(Path(source_root).resolve()!=Path(destination_root).resolve(), 'distinct_root_required')
    require(not report['collisions'], 'migration_collision')
    require(pending_policy=='leave_legacy', 'pending_jobs_not_importable')
    require(report['apply_allowed'], 'migration_not_approved')

def worker_message(message, active_generation, allowed_assets):
    validate('worker_message',message)
    require(message['generation']==active_generation, 'stale_attempt')
    require(set(message['input_asset_ids'])<=set(allowed_assets), 'unmanaged_asset')

def edit_semantics(before, after, operation):
    """Validate non-interference on snapshots; consumers still implement atomic edits."""
    validate('project',before);validate('project',after)
    require(after['id']==before['id'] and after['revision']==before['revision']+1, 'revision_conflict')
    if operation in ('block_text','blocks_reorder','candidate_arrival'):
        require(before['clips']==after['clips'] and before['tracks']==after['tracks'], 'implicit_arrangement')
    if operation=='candidate_arrival':
        old={b['id']:b['selected_take_id'] for b in before['blocks']}
        require(old=={b['id']:b['selected_take_id'] for b in after['blocks']}, 'implicit_selection')
        for take in before['takes']:
            require(take in after['takes'], 'immutable_identity')
