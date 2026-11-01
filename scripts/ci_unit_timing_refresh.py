"""Offline, bounded receipt inspection. Expected hashes are a caller-owned trust root.

No measurement execution. Candidates are timing-only, never Unit acceptance.
The CLI takes an explicit contract, frozen weights and a fresh output directory.
"""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

CAP = 2 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    with Path(path).open('rb') as stream:
        data = stream.read(CAP + 1)
    assert len(data) <= CAP, 'metadata cap'
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, 'duplicate JSON key'
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def inspect(root, expected, frozen, *, _validated_nodes=None):
    """Inspect one complete transport; never combine partial atom prefixes.

    expected binds identity, canonical manifest/plan, source hashes and every
    artifact byte hash. Sources are explicit paths, never inferred from cwd.
    """
    root = Path(root)
    required = {'manifest.json', 'plan.json', 'receipt.json', 'state.json',
                'container-transport.json', 'cleanup-ok.txt', 'log-drain.txt',
                'container-follower-reaped.txt', 'step-exit-code.txt'}
    assert required <= expected['artifacts'].keys(), 'missing artifact authentication'
    for name, sha in expected['artifacts'].items():
        assert Path(name).name == name and not name.endswith('.pending'), 'artifact path/pending'
        with (root / name).open('rb') as stream:
            data = stream.read(CAP + 1)
        assert len(data) <= CAP and hashlib.sha256(data).hexdigest() == sha, 'artifact hash'
    assert not list(root.glob('*.pending')), 'pending receipt'
    for source in expected['sources']:
        with Path(source['path']).open('rb') as stream:
            data = stream.read(CAP + 1)
        assert len(data) <= CAP and hashlib.sha256(data).hexdigest() == source['sha256'], 'source mismatch'
    assert {s['role'] for s in expected['sources']} == {'recovery_test', 'wait_guard'}, 'source closure'
    manifest, plan, receipt = [read(root / n) for n in ('manifest.json', 'plan.json', 'receipt.json')]
    identity = expected['identity']
    for key in ('run', 'attempt', 'sha', 'image', 'archive_sha', 'selection_hash'):
        assert identity.get(key), 'missing identity ' + key
    assert manifest['identity'] == receipt['identity'] == identity, 'identity'
    assert manifest['version'] == receipt['version'] == identity['version'], 'version'
    assert digest(manifest['selection']) == identity['selection_hash'], 'selection hash'
    assert digest(manifest) == receipt['manifest_hash'] == expected['manifest_hash'], 'manifest hash'
    assert digest(plan) == receipt['plan_hash'] == expected['plan_hash'], 'plan hash'
    assert not manifest['errors'], 'collection errors'
    rows = [dict(r, node=manifest['raw'][r['node']], atom=manifest['atoms'][r['atom']]) for r in manifest['rows']]
    assert len(manifest['raw']) == len(set(manifest['raw'])), 'raw duplicates'
    nodes = [r['node'] for r in rows]
    assert len(nodes) == len(set(nodes)), 'duplicate selected rows'
    assert Counter(nodes) + Counter(manifest['deselected']) == Counter(manifest['raw']), 'selection accounting'
    assignments = plan['assignments']
    assert Counter(n for ns in assignments.values() for n in ns) == Counter(nodes), 'assignment union'
    owners = {n: shard for shard, ns in assignments.items() for n in ns}
    for atom in manifest['atoms']:
        assert len({owners[r['node']] for r in rows if r['atom'] == atom}) <= 1, 'split atom'
    assigned = assignments[receipt['shard']]
    assert receipt['assigned'] == assigned, 'assignment'
    state = read(root / 'state.json')
    assert state['Status'] == 'exited' and state['Running'] is False and state['Pid'] == 0, 'terminal identity'
    assert state['OOMKilled'] is False and not state['Error'], 'OOM/error'
    assert type(receipt['exit']) is int and receipt['exit'] >= 0, 'exit'
    assert state['ExitCode'] == receipt['exit'], 'terminal exit'
    assert (root / 'step-exit-code.txt').read_text().strip() == str(receipt['exit']), 'step exit'
    transport = read(root / 'container-transport.json')
    assert transport == {'complete': True, 'errors': []}, 'transport'
    assert (root / 'container-follower-reaped.txt').read_text().strip() == str(receipt['exit']), 'reap'
    assert (root / 'cleanup-ok.txt').read_text().strip() == 'yes', 'cleanup'
    assert (root / 'log-drain.txt').read_text().strip() == 'drain=eof follower_exit=' + str(receipt['exit']), 'natural EOF'
    # retain authenticates frame sequence/hash and natural EOF; its artifact hash
    # above binds that attestation. No reconstructed or pending header accepted.
    assert expected['receipt_origin'] == 'retain-complete', 'receipt origin'
    events = receipt['events']; cursor = 0; samples = {}; subtests = Counter()
    for node in assigned:
        index = nodes.index(node)
        event_start = cursor
        assert events[cursor] == [index, 'start'], 'start'
        cursor += 1; phases = []; total = Decimal(0); passed = True
        while cursor < len(events) and events[cursor][1] != 'finish':
            event = events[cursor]; cursor += 1
            assert len(event) == 5 and event[0] == index, 'report shape/node'
            _, phase, outcome, duration, xfail = event
            assert phase in ('setup', 'call', 'teardown', 'subtest'), 'phase'
            assert outcome in ('passed', 'failed', 'skipped') and type(xfail) is bool, 'outcome'
            assert type(duration) in (int, float), 'duration type'
            seconds = Decimal(str(duration))
            assert seconds.is_finite() and seconds > 0, 'duration'
            passed &= outcome == 'passed' and not xfail
            if phase == 'subtest':
                assert phases == ['setup'], 'subtest placement'
                subtests[outcome] += 1
            else:
                phases.append(phase); total += seconds
        assert cursor < len(events) and events[cursor] == [index, 'finish'], 'finish'
        cursor += 1
        assert phases == ['setup', 'call', 'teardown'], 'phase cardinality'
        samples[node] = total if passed and phases == ['setup', 'call', 'teardown'] else None
        if _validated_nodes is not None:
            _validated_nodes[node] = dict(seconds=total, passed=passed, node_index=index, event_indices=list(range(event_start, cursor)))
    assert cursor == len(events), 'extra events'
    assert receipt.get('subtests', {}) == dict(subtests), 'subtest counts'
    weights = json.loads(json.dumps(frozen)); unresolved = []; qualified = {}
    for atom, members in expected['atoms'].items():
        actual = [r['node'] for r in rows if r['atom'] == atom]
        assert actual == members and len(members) == len(set(members)), 'stale atom membership'
        if not members or any(n not in samples or samples[n] is None for n in members):
            unresolved.append(atom); continue
        cost = int((sum((samples[n] for n in members), Decimal(0)) * Decimal('1.25') * 1000000 / 50000).to_integral_value(rounding=ROUND_CEILING)) * 50000
        prior = frozen['atom_weights'].get(atom, {})
        if prior.get('count') == len(members):
            cost = max(cost, prior['cost_us'])
        provenance = dict(identity=identity, artifacts=expected['artifacts'], sources=expected['sources'],
                          manifest_hash=expected['manifest_hash'], plan_hash=expected['plan_hash'],
                          members=members, timing_only=True)
        qualified[atom] = dict(count=len(members), cost_us=cost, sample=provenance)
        weights['atom_weights'][atom] = qualified[atom]
    return dict(candidate_weights=weights, qualified=qualified, unresolved=unresolved, timing_only=True)


HISTORY_ATOM = 'tests/service/test_recovery.py::test_t1_nonempty_B_C_actual_prefix_history'
POLICY = 'history-per-param-v1'
LEGACY_ROWS_DIGEST = '0f22e85bf246d73d07463b7e4c89c19ca4b3d3ebefce5e82e01697443d458034'
LEGACY_CONTRACT_SHA = '3b870381437890005ce17f7fcdcdac9599f5f4fa4c0924026c52f2be60b2fbac'
ALLOWED_DELTA = {'scripts/ci_unit_shards.py', 'scripts/ci_unit_timing_refresh.py',
                 'scripts/ci_unit_timing_weights.json', 'tests/test_ci_unit_shards.py',
                 'tests/test_ci_unit_timing_refresh.py'}


def _migrate(root, expected, frozen, baseline, target, source_map, current_map,
             *, parent, members, contract_sha):
    """Internal validator shared by the pinned production migration and fixtures."""
    validated = {}
    inspected = inspect(root, expected, frozen, _validated_nodes=validated)
    manifest = read(Path(root) / 'manifest.json')
    assert manifest.get('atom_policy') in (None, 'legacy'), 'old atom policy'
    assert expected['atoms'] == {parent: members}, 'original membership'
    assert len(members) == len(set(members)) and members, 'membership multiplicity'
    assert [r['node'] for r in baseline if r['atom'] == parent] == members, 'baseline membership/order'
    proposed = [dict(r, atom=r['node']) if r['atom'] == parent else dict(r) for r in baseline]
    assert target == proposed, 'target map mismatch'
    assert set(source_map) == set(current_map), 'source map paths'
    changed = {p for p in source_map if source_map[p] != current_map[p]}
    assert changed <= ALLOWED_DELTA, 'fixture/source drift: ' + repr(sorted(changed - ALLOWED_DELTA))
    assert digest(source_map) == expected['identity']['source_snapshot'], 'original source map'
    assert not inspected['unresolved'] and set(validated) == set(members), 'incomplete legacy sample'
    receipt = read(Path(root) / 'receipt.json')
    assert receipt['exit'] == 0 and not receipt.get('subtests'), 'migration requires passing whole nodes without subtests'
    shared = dict(schema='history-timing-migration-v1', old_policy=manifest.get('atom_policy', 'legacy'),
                  target_policy=POLICY, parent_atom=parent, members=members,
                  membership_digest=digest(members), target_rows_digest=digest(target),
                  contract_sha256=contract_sha, identity=expected['identity'],
                  manifest_hash=expected['manifest_hash'], plan_hash=expected['plan_hash'],
                  artifacts=expected['artifacts'], sources=expected['sources'],
                  original_source_map=source_map, original_source_snapshot=digest(source_map),
                  reviewed_changed_path_allowlist=sorted(ALLOWED_DELTA),
                  timing_only=True)
    reference = digest(shared)
    weights = json.loads(json.dumps(frozen))
    weights.setdefault('history_migrations', {})[reference] = shared
    weights.setdefault('retired_atoms', {})[parent] = dict(migration=reference, reason='replaced by exact singleton members; historical timing only')
    children = {}
    for node in members:
        sample = validated[node]
        assert sample['passed'], 'nonpassing node'
        cost = int((sample['seconds'] * Decimal('1.25') * 1000000 / 50000).to_integral_value(rounding=ROUND_CEILING)) * 50000
        provenance = dict(migration=reference, members=[node], atom_policy=POLICY,
                          source_snapshot=digest(source_map), node_index=sample['node_index'],
                          event_indices=sample['event_indices'], timing_only=True)
        prior = frozen['atom_weights'].get(node)
        if prior:
            assert prior.get('count') == 1 and prior.get('sample') == provenance, 'stale singleton provenance'
            cost = max(cost, prior['cost_us'])
        children[node] = dict(count=1, cost_us=cost, sample=provenance)
    floor = inspected['qualified'][parent]['cost_us']
    total = sum(w['cost_us'] for w in children.values())
    assert total >= floor, f'migration floor discrepancy: children={total} parent={floor}'
    weights['atom_weights'].update(children)
    # Parent remains byte-for-byte historical data, unused by target rows.
    assert weights['atom_weights'].get(parent) == frozen['atom_weights'].get(parent)
    assert len(canonical(weights)) <= CAP, 'weights metadata cap'
    return dict(candidate_weights=weights, qualified=children, total_us=total,
                retained_parent_floor_us=floor, migration=reference, timing_only=True)


def migrate_history(root, contract_path, frozen, baseline, target, source_map, current_map, collection):
    assert collection['atom_policy'] == POLICY and collection['source_snapshot'] == digest(current_map), 'fresh collection policy/source'
    original_nodes = {r['node'] for r in baseline}
    assert [r for r in collection['rows'] if r['node'] in original_nodes] == target, 'fresh baseline projection'
    extras = [r for r in collection['rows'] if r['node'] not in original_nodes]
    assert all(r['file'] in ('tests/test_ci_unit_shards.py', 'tests/test_ci_unit_timing_refresh.py') for r in extras), 'unreviewed new selected node'
    assert len(collection['rows']) == len({r['node'] for r in collection['rows']}), 'duplicate fresh nodes'
    raw = Path(contract_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LEGACY_CONTRACT_SHA, 'unreviewed legacy contract'
    expected = read(contract_path)
    assert digest(baseline) == LEGACY_ROWS_DIGEST, 'unreviewed baseline rows'
    legacy = frozen['atom_weights'].get(HISTORY_ATOM, {})
    assert legacy.get('count') == 46 and legacy.get('cost_us', 0) >= 744300000, 'missing retained legacy floor'
    provenance = legacy.get('sample', {})
    assert all(provenance.get(k) == expected[k] for k in ('identity', 'artifacts', 'sources', 'manifest_hash', 'plan_hash')), 'legacy floor provenance'
    assert provenance.get('members') == expected['atoms'][HISTORY_ATOM] and provenance.get('timing_only') is True, 'legacy floor membership'
    members = expected['atoms'][HISTORY_ATOM]
    assert len(members) == 46, 'reviewed history membership'
    return _migrate(root, expected, frozen, baseline, target, source_map, current_map,
                    parent=HISTORY_ATOM, members=members, contract_sha=LEGACY_CONTRACT_SHA)


def replay(rows, weights, unresolved):
    """Use the production allocator unchanged; capacity stays unresolved if unknown."""
    import ci_unit_shards as ci
    previous = ci.TIMING_WEIGHTS
    try:
        ci.TIMING_WEIGHTS = weights
        allocation = ci.plan(rows)
    finally:
        ci.TIMING_WEIGHTS = previous
    counts = Counter(r['atom'] for r in rows)
    files = Counter(r['file'] for r in rows)
    unknown = {r['atom'] for r in rows if
               weights['atom_weights'].get(r['atom'], {}).get('count') != counts[r['atom']] and
               (r['file'] == ci.RECOVERY or weights['file_weights'].get(r['file'], {}).get('count') != files[r['file']])}
    unresolved = sorted(set(unresolved) | unknown)
    loads = [v[0] for v in allocation['totals'].values()]
    return dict(allocation=allocation, unresolved=unresolved,
                admitted=not unresolved and sum(loads) <= len(loads)*600000000 and max(loads) <= 600000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'replay', 'migrate-history'])
    parser.add_argument('--contract', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--artifacts')
    parser.add_argument('--output', required=True)
    for name in ('baseline', 'target', 'source-map', 'current-map', 'collection'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    contract, weights = read(args.contract), read(args.weights)
    if args.mode == 'inspect':
        assert args.artifacts, 'explicit artifacts required'
        result = inspect(args.artifacts, contract, weights)
    elif args.mode == 'migrate-history':
        assert all((args.artifacts, args.baseline, args.target, args.source_map, args.current_map, args.collection)), 'explicit migration inputs required'
        result = migrate_history(args.artifacts, args.contract, weights, read(args.baseline), read(args.target), read(args.source_map), read(args.current_map), read(args.collection))
    else:
        result = replay(contract['rows'], weights, contract['unresolved'])
    data = canonical(result); assert len(data) <= CAP, 'output cap'
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_bytes(data)


if __name__ == '__main__':
    main()
