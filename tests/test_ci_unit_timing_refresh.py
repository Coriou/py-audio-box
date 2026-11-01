"""Small authenticated synthetic receipt contracts; no measurement launcher."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('timing', Path(__file__).resolve().parents[1] / 'scripts/ci_unit_timing_refresh.py')
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def sample(tmp_path):
    identity = dict(run='synthetic', attempt='1', sha='synthetic-sha', image='synthetic-iid',
                    archive_sha='synthetic-archive', version=1, selection_hash=timing.digest(['synthetic']))
    manifest = dict(identity=identity, version=1, selection=['synthetic'], raw=['a','b'],
                    deselected=[], rows=[dict(node=i, atom=0, file='f') for i in range(2)], atoms=['atom'], errors=[])
    plan = dict(assignments={'01':['a','b']})
    events = []
    for i in range(2):
        events += [[i,'start'], [i,'setup','passed',.11,False], [i,'subtest','passed',7,False],
                   [i,'call','passed',.22,False], [i,'teardown','passed',.33,False], [i,'finish']]
    receipt = dict(identity=identity, version=1, manifest_hash=timing.digest(manifest),
                   plan_hash=timing.digest(plan), assigned=['a','b'], shard='01', exit=0,
                   events=events, subtests={'passed':2})
    data = {'manifest.json':manifest, 'plan.json':plan, 'receipt.json':receipt,
            'state.json':dict(Status='exited', Running=False, Pid=0, OOMKilled=False, Error='', ExitCode=0),
            'container-transport.json':dict(complete=True, errors=[])}
    for name, value in data.items():
        (tmp_path/name).write_bytes(timing.canonical(value))
    for name, value in {'cleanup-ok.txt':'yes', 'log-drain.txt':'drain=eof follower_exit=0',
                        'container-follower-reaped.txt':'0', 'step-exit-code.txt':'0'}.items():
        (tmp_path/name).write_text(value)
    sources=[]
    for role in ('recovery_test','wait_guard'):
        path=tmp_path/role; path.write_bytes(b'frozen')
        sources.append(dict(role=role, path=str(path), sha256=hashlib.sha256(b'frozen').hexdigest()))
    contract = dict(identity=identity, manifest_hash=timing.digest(manifest), plan_hash=timing.digest(plan),
                    sources=sources, atoms={'atom':['a','b']}, receipt_origin='retain-complete')
    authenticate(tmp_path, contract)
    return contract


def authenticate(root, contract):
    contract['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
                           if p.suffix in ('.json','.txt')}


def inspect(root, contract, weights=None):
    return timing.inspect(root, contract, weights or dict(atom_weights={}, file_weights={}, seed='preserved'))


def test_whole_phases_padding_rounding_provenance(tmp_path):
    contract=sample(tmp_path)
    result=inspect(tmp_path,contract)
    weight=result['qualified']['atom']
    assert weight['cost_us']==1650000  # 2*(.11+.22+.33)*1.25, no nested +7
    assert weight['sample']['members']==['a','b'] and weight['sample']['timing_only']
    assert result['candidate_weights']['seed']=='preserved' and result['unresolved']==[]
    frozen=dict(atom_weights={'atom':dict(count=2,cost_us=2000000)},file_weights={})
    assert inspect(tmp_path,contract,frozen)['qualified']['atom']['cost_us']==2000000


@pytest.mark.parametrize('mutation', ['setup','call','teardown','finish','duplicate-start','duplicate-finish',
    'duplicate-report','node-set','hash','identity','source','pending','eof','origin','assignment',
    'zero','negative','nan','inf','bool'])
def test_rejects_invalid_sample(tmp_path, mutation):
    contract=sample(tmp_path); path=tmp_path/'receipt.json'; receipt=timing.read(path)
    if mutation in ('setup','call','teardown','finish'):
        receipt['events'].pop(next(i for i,e in enumerate(receipt['events']) if e[1]==mutation))
    elif mutation.startswith('duplicate-'):
        phase='call' if mutation=='duplicate-report' else mutation.split('-')[1]; event=next(e for e in receipt['events'] if e[1]==phase)
        receipt['events'].insert(receipt['events'].index(event),copy.deepcopy(event))
    elif mutation=='node-set':contract['atoms']['atom']=['a','different']
    elif mutation=='identity':receipt['identity']['sha']='other'
    elif mutation=='source':Path(contract['sources'][0]['path']).write_bytes(b'drift')
    elif mutation=='pending':(tmp_path/'receipt.pending').write_bytes(b'{}')
    elif mutation=='eof':(tmp_path/'log-drain.txt').write_text('not eof')
    elif mutation=='origin':contract['receipt_origin']='reconstructed'
    elif mutation=='assignment':receipt['assigned']=['b','a']
    elif mutation in ('zero','negative','nan','inf','bool'):
        receipt['events'][1][3]={'zero':0,'negative':-1,'nan':float('nan'),'inf':float('inf'),'bool':True}[mutation]
    path.write_bytes(timing.canonical(receipt)); authenticate(tmp_path,contract)
    if mutation=='hash':contract['artifacts']['receipt.json']='bad'
    with pytest.raises((AssertionError,IndexError)):
        inspect(tmp_path,contract)


@pytest.mark.parametrize('kind', ['failed-subtest','skipped-subtest','xfail','skipped-phase'])
def test_nonpassing_atom_stays_unresolved(tmp_path,kind):
    contract=sample(tmp_path); path=tmp_path/'receipt.json'; receipt=timing.read(path)
    if kind.endswith('subtest'):
        outcome=kind.split('-')[0]; receipt['events'][2][2]=outcome
        receipt['subtests']={'passed':1,outcome:1}
    elif kind=='xfail':receipt['events'][3][4]=True
    else:receipt['events'][3][2]='skipped'
    path.write_bytes(timing.canonical(receipt));authenticate(tmp_path,contract)
    result=inspect(tmp_path,contract)
    assert result['qualified']=={} and result['unresolved']==['atom']


def test_failed_job_can_supply_timing_only_and_partial_cannot(tmp_path):
    contract=sample(tmp_path);path=tmp_path/'receipt.json';receipt=timing.read(path)
    receipt['exit']=1;path.write_bytes(timing.canonical(receipt))
    state=timing.read(tmp_path/'state.json');state['ExitCode']=1
    (tmp_path/'state.json').write_bytes(timing.canonical(state))
    for name in ('step-exit-code.txt','container-follower-reaped.txt'):(tmp_path/name).write_text('1')
    (tmp_path/'log-drain.txt').write_text('drain=eof follower_exit=1')
    authenticate(tmp_path,contract)
    assert inspect(tmp_path,contract)['qualified']['atom']['sample']['timing_only']
    receipt['events']=receipt['events'][:-1];path.write_bytes(timing.canonical(receipt));authenticate(tmp_path,contract)
    with pytest.raises(AssertionError):inspect(tmp_path,contract)


def test_replay_does_not_admit_unknown_fallback(monkeypatch):
    # Explicit imports stay offline and stdlib-only in the production replay.
    import sys
    monkeypatch.syspath_prepend(str(Path(timing.__file__).parent))
    import ci_unit_shards
    monkeypatch.setattr(ci_unit_shards, 'SHARDS', [f'{i:02}' for i in range(1, 7)])
    rows=[dict(file=f'tests/f{i}.py',atom=f'a{i}',node=f'n{i}') for i in range(6)]
    weights=dict(atom_weights={},file_weights={})
    result=timing.replay(rows,weights,[])
    assert result['admitted'] is False and len(result['unresolved'])==6
    assert result==timing.replay(rows,weights,[])


def migration_sample(tmp_path):
    contract = sample(tmp_path)
    receipt = timing.read(tmp_path/'receipt.json')
    receipt['events'] = [e for e in receipt['events'] if e[1] != 'subtest']
    receipt['events'][6][3] = .11111  # second setup, deliberately off-grid
    receipt['subtests'] = {}
    (tmp_path/'receipt.json').write_bytes(timing.canonical(receipt))
    source_map = {'tests/service/test_recovery.py':'frozen', 'tests/_maintenance_wait_guard.py':'guard'}
    contract['identity']['source_snapshot'] = timing.digest(source_map)
    manifest = timing.read(tmp_path/'manifest.json'); manifest['identity'] = contract['identity']
    (tmp_path/'manifest.json').write_bytes(timing.canonical(manifest))
    contract['manifest_hash'] = timing.digest(manifest)
    receipt['identity'] = contract['identity']; receipt['manifest_hash'] = contract['manifest_hash']
    (tmp_path/'receipt.json').write_bytes(timing.canonical(receipt)); authenticate(tmp_path,contract)
    baseline = [dict(node=n,atom='atom',file='f') for n in ('a','b')]
    target = [dict(r,atom=r['node']) for r in baseline]
    return contract, baseline, target, source_map


def migrate(root, contract, baseline, target, sources, frozen=None):
    return timing._migrate(root, contract, frozen or dict(atom_weights={'atom':dict(count=2,cost_us=1650000)},file_weights={},seed='keep'), baseline, target, sources, sources, parent='atom', members=['a','b'], contract_sha='synthetic')


def test_migration_individual_rounding_and_provenance(tmp_path):
    contract,baseline,target,sources = migration_sample(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    result = migrate(tmp_path,contract,baseline,target,sources)
    assert [w['cost_us'] for w in result['qualified'].values()] == [850000,850000]
    assert result['total_us'] == 1700000
    assert result['qualified']['b']['sample']['event_indices'] == [5,6,7,8,9]
    weights = result['candidate_weights']
    assert weights['seed'] == 'keep' and weights['atom_weights']['atom']['cost_us'] == 1650000
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    frozen = copy.deepcopy(weights); frozen['atom_weights']['a']['cost_us'] = 900000
    assert migrate(tmp_path,contract,baseline,target,sources,frozen)['total_us'] == 1750000


@pytest.mark.parametrize('fault',['member','permutation','class','policy','source-map','floor','singleton','subtest'])
def test_migration_rejects_unreviewed_inputs(tmp_path,fault):
    contract,baseline,target,sources = migration_sample(tmp_path)
    frozen = dict(atom_weights={'atom':dict(count=2,cost_us=1650000)},file_weights={})
    if fault=='member':target[1]['node']='different'
    elif fault=='permutation':target.reverse()
    elif fault=='class':target[0]['atom']='Class'
    elif fault=='source-map':sources['extra']='drift'
    elif fault=='floor':frozen['atom_weights']['atom']['cost_us']=2000000
    elif fault=='singleton':frozen['atom_weights']['a']=dict(count=1,cost_us=900000,sample={})
    else:
        path=tmp_path/('manifest.json' if fault=='policy' else 'receipt.json'); value=timing.read(path)
        if fault=='policy':
            value['atom_policy']=timing.POLICY; contract['manifest_hash']=timing.digest(value)
            receipt=timing.read(tmp_path/'receipt.json');receipt['manifest_hash']=contract['manifest_hash']
            (tmp_path/'receipt.json').write_bytes(timing.canonical(receipt))
        else:value['subtests']={'passed':1}
        path.write_bytes(timing.canonical(value));authenticate(tmp_path,contract)
    with pytest.raises(AssertionError):migrate(tmp_path,contract,baseline,target,sources,frozen)


def test_production_migration_requires_pinned_contract_and_fresh_collection(tmp_path):
    contract,baseline,target,sources=migration_sample(tmp_path)
    path=tmp_path/'contract-input';path.write_bytes(timing.canonical(contract))
    collection=dict(atom_policy=timing.POLICY,source_snapshot=timing.digest(sources),rows=target)
    frozen=dict(atom_weights={},file_weights={})
    with pytest.raises(AssertionError,match='unreviewed legacy contract'):
        timing.migrate_history(tmp_path,path,frozen,baseline,target,sources,sources,collection)
    collection['rows']=list(reversed(target))
    with pytest.raises(AssertionError,match='fresh baseline projection'):
        timing.migrate_history(tmp_path,path,frozen,baseline,target,sources,sources,collection)
