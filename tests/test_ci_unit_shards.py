"""Synthetic scheduling and fail-closed protocol checks; no providers."""
import importlib.util
from collections import Counter
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('ci_unit_shards', Path(__file__).resolve().parents[1] / 'scripts/ci_unit_shards.py')
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def rows(file, atom, count):
    return [dict(file=file, atom=atom, node=f'{atom}[{i}]') for i in range(count)]


def fillers():
    return [dict(file=f'tests/f{i}.py', atom=f'f{i}', node=f'f{i}') for i in range(6)]


def test_heterogeneous_atoms_and_indivisible_soft_cap(monkeypatch):
    weights = {'heavy': dict(count=2, cost_us=200000000),
               'next': dict(count=1, cost_us=100000000)}
    monkeypatch.setattr(ci, 'TIMING_WEIGHTS', dict(atom_weights=weights,
        file_weights={'tests/other.py': dict(count=1, cost_us=190000000)}))
    selected = rows(ci.RECOVERY, 'heavy', 2) + rows(ci.RECOVERY, 'next', 1) + rows('tests/other.py', 'other', 1) + fillers()
    allocation = ci.plan(selected)
    groups = allocation['groups']
    heavy = next(g for g in groups if g['first'] == 0)
    other = next(g for g in groups if g['file'] == 'tests/other.py')
    assert heavy['indices'] == [0, 1] and heavy['cost'] == 200000000
    assert heavy['oversized'] is False
    assert heavy['shard'] != other['shard']
    assert [g['count'] for g in groups if g['file'] == ci.RECOVERY] == [2, 1]
    assert Counter(n for ns in allocation['assignments'].values() for n in ns) == Counter(r['node'] for r in selected)
    owner = {n: s for s, ns in allocation['assignments'].items() for n in ns}
    assert owner['heavy[0]'] == owner['heavy[1]']
    for ns in allocation['assignments'].values():
        assert ns == [r['node'] for r in selected if r['node'] in ns]
    assert allocation == ci.plan(selected)


def test_count_drift_and_missing_weights_use_fallback(monkeypatch):
    monkeypatch.setattr(ci, 'TIMING_WEIGHTS', dict(atom_weights={'a': dict(count=1, cost_us=99000000)},
        file_weights={'tests/drift.py': dict(count=1, cost_us=99000000)}))
    allocation = ci.plan(rows(ci.RECOVERY, 'a', 2) + rows(ci.RECOVERY, 'missing', 1) + rows('tests/drift.py', 'd', 2) + fillers())
    assert [g['cost'] for g in allocation['groups'][:2]] == [2250000, 1500000]


def test_count_cap_keeps_oversized_class_whole(monkeypatch):
    monkeypatch.setattr(ci, 'TIMING_WEIGHTS', dict(atom_weights={}, file_weights={}))
    allocation = ci.plan(rows(ci.RECOVERY, 'Class', 193) + rows(ci.RECOVERY, 'tail', 1) + fillers())
    assert [(g['count'], g['oversized']) for g in allocation['groups'][:2]] == [(193, True), (1, False)]


def test_file_weight_proportion_rounds_up_and_ties_are_stable(monkeypatch):
    monkeypatch.setattr(ci, 'TIMING_WEIGHTS', dict(atom_weights={},
        file_weights={'tests/measured.py': dict(count=193, cost_us=1000001)}))
    selected = rows('tests/measured.py', 'a', 192) + rows('tests/measured.py', 'b', 1) + fillers()
    groups = ci.plan(selected)['groups']
    assert [g['cost'] for g in groups[:2]] == [(1000001*192+192)//193, (1000001+192)//193]
    ties = ci.plan(fillers())
    assert [g['shard'] for g in ties['groups']] == ci.SHARDS


def receipt():
    return dict(assigned=['n'], exit=0, events=[[0,'start']] +
        [[0,p,'passed',0.1,False] for p in ('setup','call','teardown')] + [[0,'finish']])


@pytest.mark.parametrize('mutation', ['absent', 'duplicate', 'unexpected'])
def test_executor_rejects_bad_finishes(mutation):
    value = receipt()
    ci.verify_execution(value, ['n'], ['n'])
    if mutation == 'absent':
        value['events'].pop()
    elif mutation == 'duplicate':
        value['events'].append([0,'finish'])
    else:
        value['events'][-1] = [1,'finish']
    with pytest.raises(AssertionError):
        ci.verify_execution(value, ['n'], ['n'])


def test_incomplete_transport_is_rejected(tmp_path):
    code = "import os; os.write(1, b'PAB_UNIT_V1:{\"name\":\"receipt\",\"seq\":0,\"data\":\"e30=\"}\\n')"
    assert ci.retain(tmp_path, 'synthetic', [sys.executable, '-c', code]) == 125
    transport = ci.read(tmp_path / 'synthetic-transport.json')
    assert transport['complete'] is False and 'incomplete frame' in transport['errors']
    assert (tmp_path / 'receipt.pending').read_bytes() == b'{}'
