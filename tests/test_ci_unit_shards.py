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
    return [dict(file=f'tests/f{i:02}.py', atom=f'f{i:02}', node=f'f{i:02}') for i in range(len(ci.SHARDS))]


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


@pytest.mark.parametrize('fault', ['eintr', 'epipe', 'zero', 'short', 'exit'])
def test_machine_writer_isolated_from_global_faults(tmp_path, fault):
    # Both pipes belong to the isolated child; exit injection cannot kill pytest.
    code = r'''
import importlib.util, json, os, sys
spec=importlib.util.spec_from_file_location('ci',sys.argv[1])
ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
machine_r,machine_w=os.pipe(); result_r,result_w=os.pipe()
ci.MACHINE_FD=machine_w
original=os.write; calls=[]; fault=sys.argv[2]
def injected(fd,data):
    calls.append(fd)
    if fault=='eintr':raise InterruptedError()
    if fault=='epipe':raise BrokenPipeError()
    if fault=='zero':return 0
    if fault=='short':return original(fd,data[:1])
    if fault=='exit':os._exit(74)
os.write=injected
ci.emit('receipt',{'isolated':True})
assert calls==[]
os.close(machine_w)
framed=b''
while True:
    chunk=os.read(machine_r,65536)
    if not chunk:break
    framed+=chunk
os.close(machine_r)
# Before deliberate injection, the unrelated result pipe contains no bytes.
os.set_blocking(result_r,False)
try:os.read(result_r,1)
except BlockingIOError:pass
else:raise AssertionError('leaked bytes or premature EOF')
# Forward actual frames to retain, including a natural child EOF on exit74.
original(1,framed)
try:
    n=os.write(result_w,b'xy')
    assert n==({'zero':0,'short':1}[fault])
except (InterruptedError,BrokenPipeError):
    assert fault in ('eintr','epipe')
finally:
    os.write=original
assert calls==[result_w]
os.close(result_w)
assert os.read(result_r,10)==(b'x' if fault=='short' else b'')
os.close(result_r)
'''
    status = ci.retain(tmp_path, 'isolation', [sys.executable, '-c', code, ci.__file__, fault])
    assert status == (74 if fault == 'exit' else 0)
    assert ci.read(tmp_path / 'receipt.json') == {'isolated': True}
    assert ci.read(tmp_path / 'isolation-transport.json') == {'complete': True, 'errors': []}


@pytest.mark.parametrize('fault', ['none','shared','override','unparametrized'])
def test_real_item_fixture_closure(tmp_path, fault):
    import subprocess
    root = tmp_path/'project'; (root/'tests/service').mkdir(parents=True)
    (root/'tests/conftest.py').write_text('import pytest\n@pytest.fixture(autouse=True)\ndef maintenance_wait_guard(tmp_path_factory, monkeypatch, request): yield\n')
    code = '''import pytest
@pytest.fixture
def installation(tmp_path): return tmp_path
@pytest.fixture
def b23_host(installation, monkeypatch): return installation
@pytest.mark.parametrize('generation',[1])
@pytest.mark.parametrize('point',['a-b-1','a-b-2'])
def test_t1_nonempty_B_C_actual_prefix_history(b23_host): pass
@pytest.mark.parametrize('point',[1,2])
def test_other(point): pass
class TestControl:
    @pytest.fixture(scope='class')
    def shared(self): return object()
    @pytest.mark.parametrize('point',[1,2])
    def test_t1_nonempty_B_C_actual_prefix_history(self,point,shared): pass
'''
    code = code.replace('def test_t1_nonempty_B_C_actual_prefix_history(b23_host):', 'def test_t1_nonempty_B_C_actual_prefix_history(b23_host,generation,point):')
    if fault=='shared':
        code = code.replace('def b23_host(installation, monkeypatch)', 'def b23_host(installation, monkeypatch, shared)') + '\n@pytest.fixture(scope="session")\ndef shared(): return []\n'
    if fault=='override':code = code.replace('@pytest.fixture\ndef installation', '@pytest.fixture(scope="module")\ndef installation')
    if fault=='unparametrized':code = code.replace("@pytest.mark.parametrize('point',['a-b-1','a-b-2'])\n", '').replace("@pytest.mark.parametrize('generation',[1])\n", '').replace('(b23_host,generation,point)', '(b23_host)')
    (root/ci.RECOVERY).write_text(code)
    # Same-looking function in a different module must retain its function atom.
    (root/'tests/test_other.py').write_text("import pytest\n@pytest.mark.parametrize('point',[1,2])\ndef test_t1_nonempty_B_C_actual_prefix_history(point): pass\n")
    driver = '''import importlib.util,sys,pytest
from pathlib import Path
spec=importlib.util.spec_from_file_location('ci',sys.argv[1]); ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
class Plugin:
 def pytest_collection_modifyitems(self,config,items):
  atoms=[ci.item_atom(i,config) for i in items]
  history=[(i,a) for i,a in zip(items,atoms) if i.nodeid.startswith(ci.HISTORY_ATOM+'[')]
  assert len(history)==2 and all(i.nodeid==a for i,a in history)
  classes=[a for a in atoms if 'TestControl' in a]
  assert len(classes)==2 and len(set(classes))==1
  others=[a for i,a in zip(items,atoms) if i.nodeid.startswith('tests/test_other.py')]
  assert len(set(others))==1
raise SystemExit(pytest.main(['tests','--collect-only','-p','no:cacheprovider','-q'],plugins=[Plugin()]))
'''
    result = subprocess.run([sys.executable,'-c',driver,ci.__file__],cwd=root,capture_output=True,text=True,timeout=30)
    assert (result.returncode==0) == (fault=='none'), result.stdout+result.stderr
    if fault!='none':assert 'unreviewed history fixture' in result.stdout+result.stderr or 'requires reviewed parametrization' in result.stdout+result.stderr


def test_scheduler_independent_fixture_owners_and_class_control(tmp_path):
    import subprocess
    root=tmp_path/'owned';(root/'tests/service').mkdir(parents=True)
    (root/'tests/conftest.py').write_text("""import pytest,json,os
from pathlib import Path
@pytest.fixture(autouse=True)
def maintenance_wait_guard(tmp_path_factory,monkeypatch,request):
 assert not hasattr(os,'_synthetic_owner')
 owner=request.node.nodeid
 root=tmp_path_factory.mktemp('guard')
 state=[]
 def finalized():
  assert state==[owner]
  assert not hasattr(os,'_synthetic_owner')
  Path('guards.jsonl').open('a').write(json.dumps(dict(owner=owner,root=str(root),finalized=True))+'\\n')
 request.addfinalizer(finalized)
 state.append(owner)
 yield
 monkeypatch.undo()
""")
    (root/ci.RECOVERY).write_text('''import pytest,sqlite3,subprocess,sys,json,os
from pathlib import Path
@pytest.fixture
def installation(tmp_path):
 db=sqlite3.connect(tmp_path/'history.sqlite');db.execute('create table history (generation integer)')
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 state=dict(root=str(tmp_path),pid=child.pid,empty=db.execute('select count(*) from history').fetchone()[0]==0)
 yield db,state
 child.terminate();child.wait(timeout=3);db.close()
 state['terminal']=child.returncode;state['finalized']=True
 Path('owners.jsonl').open('a').write(json.dumps(state)+'\\n')
@pytest.fixture
def b23_host(installation,monkeypatch):return installation
@pytest.mark.parametrize('generation',[1])
@pytest.mark.parametrize('point',['independent-a','independent-b'])
def test_t1_nonempty_B_C_actual_prefix_history(b23_host,generation,point):
 db,state=b23_host;assert state['empty']
 for generation in (1,2):
  assert db.execute('select count(*) from history').fetchone()[0]==generation-1
  db.execute('insert into history values (?)',(generation,));db.commit()
class TestClassControl:
 @pytest.fixture(scope='class')
 def shared(self):
  Path('class-owners.txt').open('a').write('setup\\n');yield []
  Path('class-owners.txt').open('a').write('teardown\\n')
 @pytest.mark.parametrize('point',[1,2])
 def test_shared(self,shared,point):shared.append(point);assert shared==list(range(1,point+1))
''')
    with (root/ci.RECOVERY).open('a') as stream:
        stream.write("""
@pytest.mark.parametrize('mode',['pipe-a','pipe-b'])
def test_t1_actual_pipe_transport(tmp_path,monkeypatch,mode):
 assert not hasattr(os,'_synthetic_owner')
 monkeypatch.setattr(os,'_synthetic_owner',mode,raising=False)
 read,write=os.pipe()
 child=subprocess.Popen([sys.executable,'-c',"import os;os.write(int(__import__('sys').argv[1]),b'owned')",str(write)],pass_fds=(write,))
 os.close(write)
 assert os.read(read,5)==b'owned' and os.read(read,1)==b''
 os.close(read);assert child.wait(timeout=3)==0
 (tmp_path/'owned').write_text(mode)
@pytest.mark.parametrize('raw',[b'a',b'b'])
def test_t1_strict_closed_frame(raw,monkeypatch):
 assert not hasattr(os,'_synthetic_owner')
 monkeypatch.setattr(os,'_synthetic_owner',raw,raising=False)
""")
    driver='''import sys,json,importlib.util,os
from pathlib import Path
os.chdir(Path(__file__).parent)
spec=importlib.util.spec_from_file_location('ci',sys.argv[1]);ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
ci.SHARDS=['01','02'];ci.SELECTION=['tests','-p','no:cacheprovider','-q'];ci.TIMING_WEIGHTS={'atom_weights':{},'file_weights':{}}
# Keep history items in distinct packs using real measured synthetic scheduling costs.
for p in ('independent-a','independent-b'):ci.TIMING_WEIGHTS['atom_weights'][ci.HISTORY_ATOM+'['+p+'-1]']={'count':1,'cost_us':180000000}
for name,ids in [('test_t1_actual_pipe_transport',('pipe-a','pipe-b')),('test_t1_strict_closed_frame',('a','b'))]:
 for p in ids:ci.TIMING_WEIGHTS['atom_weights'][ci.RECOVERY+'::'+name+'['+p+']']={'count':1,'cost_us':180000000}
from types import SimpleNamespace
mode=sys.argv[2];args=SimpleNamespace(mode=mode,identity='{"synthetic":true}',shard=sys.argv[3])
if mode=='run':
 manifest=ci.read(Path(sys.argv[4])/'manifest.json');plan=ci.read(Path(sys.argv[4])/'plan.json')
 args.manifest_hash=ci.digest(manifest);args.plan_hash=ci.digest(plan)
raise SystemExit(ci.pytest_mode(args))
'''
    script=root/'driver.py';script.write_text(driver)
    collect=root/'collect'
    assert ci.retain(collect,'collect',[sys.executable,str(script),ci.__file__,'collect','01'])==0
    manifest=ci.read(collect/'manifest.json');allocation=ci.read(collect/'plan.json')
    history=[n for n in manifest['raw'] if n.startswith(ci.HISTORY_ATOM+'[')]
    owners={n:s for s,ns in allocation['assignments'].items() for n in ns}
    assert len({owners[n] for n in history})==2
    classes=[n for n in manifest['raw'] if 'TestClassControl' in n]
    assert len({owners[n] for n in classes})==1
    for shard in ('01','02'):
        output=root/shard
        # retain inherits cwd; driver explicitly changes to its private project.
        assert ci.retain(output,'run',[sys.executable,str(script),ci.__file__,'run',shard,str(collect)])==0
        receipt=ci.read(output/'receipt.json')
        ci.verify_execution(receipt,allocation['assignments'][shard],manifest['raw'])
        assert ci.read(output/'run-transport.json')['complete']
    import json
    states=[json.loads(line) for line in (root/'owners.jsonl').read_text().splitlines()]
    assert len(states)==2 and len({s['root'] for s in states})==2 and len({s['pid'] for s in states})==2
    assert all(s['empty'] and s['finalized'] and s['terminal'] is not None for s in states)
    guards=[json.loads(line) for line in (root/'guards.jsonl').read_text().splitlines()]
    assert Counter(g['owner'] for g in guards)==Counter(manifest['raw'])
    assert len({g['root'] for g in guards})==len(guards) and all(g['finalized'] for g in guards)
    for prefix in ('test_t1_actual_pipe_transport','test_t1_strict_closed_frame'):
        family=[n for n in manifest['raw'] if n.startswith(ci.RECOVERY+'::'+prefix+'[')]
        assert len(family)==2 and len({owners[n] for n in family})==2
    assert (root/'class-owners.txt').read_text().splitlines()==['setup','teardown']


def test_history_singleton_pack_boundary_and_order(monkeypatch):
    selected = [dict(file=ci.RECOVERY,node=ci.HISTORY_ATOM+'['+str(i)+']',atom=ci.HISTORY_ATOM+'['+str(i)+']') for i in range(4)] + fillers()
    costs=[90000000,90000000,50000,179950000]
    monkeypatch.setattr(ci,'TIMING_WEIGHTS',dict(atom_weights={r['atom']:dict(count=1,cost_us=c) for r,c in zip(selected,costs)},file_weights={}))
    allocation=ci.plan(selected)
    assert [(g['indices'],g['cost']) for g in allocation['groups'] if g['file']==ci.RECOVERY]==[([0,1],180000000),([2,3],180000000)]
    assert ci.canonical(allocation)==ci.canonical(ci.plan(selected))
    assert Counter(n for ns in allocation['assignments'].values() for n in ns)==Counter(r['node'] for r in selected)
    for ns in allocation['assignments'].values():assert ns==[r['node'] for r in selected if r['node'] in ns]


@pytest.mark.parametrize('profile,fault', [
    ('H','none'),('T','none'),('G','none'),
    ('H','indirect'),('T','scope'),('G','missing'),('H','names'),
    ('H','session'),('T','module'),('G','class'),
    ('H','foreign'),('T','duplicate'),('G','builtin')])
def test_reviewed_remaining_real_items(tmp_path, profile, fault):
    import subprocess
    root=tmp_path/'reviewed';(root/'tests/service').mkdir(parents=True)
    guard='import pytest\n@pytest.fixture(autouse=True)\ndef maintenance_wait_guard(tmp_path_factory,monkeypatch,request): yield\n'
    (root/'tests/conftest.py').write_text(guard)
    code='import pytest\n@pytest.fixture\ndef installation(tmp_path): return tmp_path\n@pytest.fixture\ndef b23_host(installation,monkeypatch): return installation\n'
    keys=[(name,params) for name,(kind,params) in ci.REVIEWED_REMAINING.items() if kind==profile]
    target=keys[0][0]
    for name,params in keys:
        dependencies=['b23_host'] if profile=='H' else ['tmp_path'] if profile=='T' else []
        extra=''
        options=''
        actual=params
        if name==target:
            if fault=='indirect':
                options=',indirect=True'
                code+='@pytest.fixture\ndef '+params[0]+'(request): return request.param\n'
            if fault=='scope':options=",scope='module'"
            if fault=='names':actual=('unreviewed',)
            if fault in ('session','module','class'):
                dependencies+=['mutable']
                code+='@pytest.fixture(scope='+repr(fault)+')\ndef mutable(): return []\n'
        code+='@pytest.mark.parametrize('+repr(','.join(actual))+', [tuple(range('+str(len(actual))+'))], ids=["odd-[a-b]"]'+options+')\n'
        code+='def '+name+'('+','.join(dependencies+list(actual))+'): pass\n'
    if fault=='foreign':
        code=code.replace('@pytest.fixture\ndef installation(tmp_path): return tmp_path\n','')
        with (root/'tests/conftest.py').open('a') as stream:stream.write('@pytest.fixture\ndef installation(tmp_path): return tmp_path\n')
    if fault=='duplicate':
        with (root/'tests/conftest.py').open('a') as stream:stream.write('@pytest.fixture\ndef tmp_path(tmp_path): return tmp_path\n')
    if fault=='builtin':code+='@pytest.fixture\ndef monkeypatch(): return object()\n'
    code+='@pytest.mark.parametrize("value",[1,2])\ndef test_unlisted(value): pass\n'
    code+='class TestOuter:\n class TestNested:\n  @pytest.mark.parametrize("value",[1,2])\n  def '+target+'(self,value): pass\n'
    (root/ci.RECOVERY).write_text(code)
    (root/'tests/test_elsewhere.py').write_text('import pytest\n@pytest.mark.parametrize("value",[1,2])\ndef '+target+'(value): pass\n')
    driver='''import sys,importlib.util,pytest
spec=importlib.util.spec_from_file_location('ci',sys.argv[1]);ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
class Plugin:
 def pytest_collection_modifyitems(self,config,items):
  for item in items:
   if sys.argv[2]=='missing' and item.originalname==sys.argv[3] and item.getparent(pytest.Class) is None and str(item.path).endswith('test_recovery.py'):del item.callspec
   atom=ci.item_atom(item,config)
   if item.getparent(pytest.Class):assert atom==item.getparent(pytest.Class).nodeid
   elif str(item.path).endswith('test_recovery.py') and item.originalname in ci.REVIEWED_REMAINING:assert atom==item.nodeid
   else:assert atom==item.parent.nodeid+'::'+item.originalname
raise SystemExit(pytest.main(['tests','--collect-only','-q','-p','no:cacheprovider'],plugins=[Plugin()]))
'''
    result=subprocess.run([sys.executable,'-c',driver,ci.__file__,fault,target],cwd=root,capture_output=True,text=True,timeout=30)
    assert (result.returncode==0)==(fault=='none'),result.stdout+result.stderr
    if fault!='none':assert 'unreviewed' in result.stdout+result.stderr or 'requires reviewed' in result.stdout+result.stderr


def test_eleven_configuration_matches_explicit_workflow():
    import ast
    import re
    root = Path(ci.__file__).resolve().parents[1]
    workflow = (root / '.github/workflows/unit.yml').read_text()
    assert ci.SHARDS == [f'{i:02}' for i in range(1, 12)]
    assert ast.literal_eval(re.search(r'shard: (\[.*\])', workflow)[1]) == ci.SHARDS
    assert 'max-parallel: 6' in workflow and '-of-06' not in workflow
    assert '-shard-${{ matrix.shard }}-of-11' in workflow
    aggregate = workflow[workflow.index('  unit:'):]
    assert re.findall(r'-shard-(\d+)-of-11', aggregate) == ci.SHARDS
    assert re.findall(r'/receipts/shard-(\d+)/', aggregate) == ci.SHARDS
    assert aggregate.count('uses: actions/download-artifact@') == 12
    runner = (root / 'scripts/ci-unit.sh').read_text()
    assert 'import VERSION, SHARDS, SELECTION,' in runner
    assert 'shards=len(SHARDS)' in runner


def synthetic_aggregate(root):
    identity = dict(version=ci.VERSION, shards=len(ci.SHARDS), selection_hash=ci.digest(ci.SELECTION),
                    sha='synthetic-current-source', run='synthetic', attempt='1', image='inert', archive_sha='synthetic')
    selected = fillers()
    manifest = dict(identity=identity, version=ci.VERSION, atom_policy=ci.ATOM_POLICY,
                    selection=ci.SELECTION, errors=[], raw=[r['node'] for r in selected],
                    atoms=[r['atom'] for r in selected],
                    rows=[dict(file=r['file'], node=i, atom=i) for i,r in enumerate(selected)],
                    deselected=[], skips=[])
    allocation = ci.plan(selected)
    for shard in [None] + ci.SHARDS:
        directory = root / ('prepare' if shard is None else 'shard-'+shard)
        directory.mkdir()
        for name, value in [('manifest',manifest), ('plan',allocation),
                            ('state',dict(Status='exited',Running=False,Pid=0,OOMKilled=False,ExitCode=0,Error='')),
                            ('container-transport',dict(complete=True))]:
            ci.write(directory/(name+'.json'),value)
        for name,value in [('step-exit-code.txt','0'),('log-drain.txt','drain=eof follower_exit=0'),('cleanup-ok.txt','yes')]:
            (directory/name).write_text(value)
        assigned = allocation['assignments'][shard] if shard else []
        events=[]
        for node in assigned:
            index=manifest['raw'].index(node)
            events += [[index,'start']] + [[index,p,'passed',.1,False] for p in ('setup','call','teardown')] + [[index,'finish']]
        ci.write(directory/'receipt.json',dict(identity=identity,version=ci.VERSION,shard=shard,
                 manifest_hash=ci.digest(manifest),plan_hash=ci.digest(allocation),exit=0,assigned=assigned,events=events))
    return identity


def test_pretest_transport_marker_does_not_certify_shard(tmp_path):
    workflow = (Path(ci.__file__).resolve().parents[1] / '.github/workflows/unit.yml').read_text()
    shard_job = workflow.split('  unit-shard:', 1)[1].split('  unit:', 1)[0]
    assert shard_job.index('Record image transfer start before tests') < shard_job.index('id: image')
    assert shard_job.index("steps.image.outcome == 'failure'") < shard_job.index('Verify and load exact image')
    assert 'if: always()' in shard_job and 'path: ${{ runner.temp }}/shard/' in shard_job

    identity = synthetic_aggregate(tmp_path)
    shard = tmp_path / 'shard-11'
    marker = shard / 'transport-stage.json'
    marker.write_text('{"stage":"image-download-started"}\n')
    assert ci.aggregate(tmp_path, identity, dict(prepare='success', shards='success'))['ok']
    for path in shard.iterdir():
        if path != marker:
            path.unlink()
    with pytest.raises(ci.AggregationError) as error:
        ci.aggregate(tmp_path, identity, dict(prepare='success', shards='failure'))
    assert any('11:' in issue for issue in error.value.summary['errors'])


@pytest.mark.parametrize('fault', ['none','count','identity','missing11','extra','assignment','hash','dependency','duplicate'])
def test_eleven_aggregate_complete_and_fail_closed(tmp_path, fault):
    identity=synthetic_aggregate(tmp_path)
    dependencies=dict(prepare='success',shards='success')
    assert ci.aggregate(tmp_path,identity,dependencies)['ok']
    if fault=='none':
        return
    if fault=='count':
        identity=dict(identity,shards=6)
        # Bind every manifest/receipt to this identity so configuration itself must reject it.
        for p in tmp_path.glob('*/*.json'):
            value=ci.read(p)
            if 'identity' in value:
                value['identity']=identity;ci.write(p,value)
    elif fault=='identity':
        identity=dict(identity,attempt='2')
    elif fault=='missing11':
        (tmp_path/'shard-11').rename(tmp_path/'missing-11')
    elif fault=='extra':
        (tmp_path/'unexpected').mkdir()
    elif fault=='dependency':
        dependencies['shards']='failure'
    else:
        p=tmp_path/'shard-11'/'receipt.json';value=ci.read(p)
        if fault=='hash':value['plan_hash']='changed'
        elif fault=='assignment':value['assigned']=[]
        else:value['events'].append(value['events'][-1])
        ci.write(p,value)
    with pytest.raises(ci.AggregationError):
        ci.aggregate(tmp_path,identity,dependencies)


@pytest.mark.parametrize('fault', ['none','failed','negative','nonfinite','order','missing'])
def test_nested_zero_duration_protocol(fault):
    value=receipt()
    value['events'].insert(2,[0,'subtest','passed',0,False])
    value['subtests']={'passed':1}
    ci.verify_execution(value,['n'],['n'])
    if fault=='none':return
    if fault=='failed':value['events'][2][2]='failed'
    elif fault=='negative':value['events'][2][3]=-1
    elif fault=='nonfinite':value['events'][2][3]=float('nan')
    elif fault=='order':value['events'][2],value['events'][3]=value['events'][3],value['events'][2]
    else:value['events'].pop()
    with pytest.raises(AssertionError):ci.verify_execution(value,['n'],['n'])
