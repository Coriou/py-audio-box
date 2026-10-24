"""Standalone stdlib inert HOST checks; never import guard/product or run pytest.

Usage: python -I -S -B check_outcome.py PRIVATE_NEW_DIRECTORY [--expect-bug]
Optional --controls=FILE selects an immutable source copy for --expect-bug.
Optional --actual-pax=FILE checks pinned r131 diagnostic members in memory.
The private directory and every fixture are retained, not cleaned up.
"""
import ast
import copy
import hashlib
import io
import tarfile
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import uuid

REPO = Path(__file__).resolve().parents[2]
controls_arg = next((a.split('=',1)[1] for a in sys.argv[2:] if a.startswith('--controls=')), None)
CONTROLS = Path(controls_arg) if controls_arg else REPO / 'tests/service/test_child_wait_guard.py'
GUARD = REPO / 'tests/_maintenance_wait_guard.py'
NAMES = ('child-refusal.json', 'parent-callback.json', 'directed-outcome.json')
BASE = Path(sys.argv[1])
BASE.mkdir(mode=0o700)
assert BASE.lstat().st_uid == os.getuid() and stat.S_IMODE(BASE.stat().st_mode) == 0o700
BUG = '--expect-bug' in sys.argv[2:]
labels = []
actual = None
pax_arg = next((a.split('=',1)[1] for a in sys.argv[2:] if a.startswith('--actual-pax=')), None)
if pax_arg:
    fd = os.open(pax_arg, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
        chunks = []
        while chunk := os.read(fd, 65536): chunks.append(chunk)
        payload = b''.join(chunks)
        assert len(payload) == info.st_size
    finally: os.close(fd)
    assert hashlib.sha256(payload).hexdigest() == '5f75f562e9827425b1072705c09659c5a2d6afa14fa0851b698a42f0f101ef6b'
    # Read only two bounded diagnostic members into memory. Never extract.
    actual = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
        for member in archive:
            name = member.name.rsplit('/',1)[-1]
            if name in NAMES[:2]:
                assert member.isfile() and member.size <= 4096 and name not in actual
                actual[name] = json.loads(archive.extractfile(member).read())
    assert set(actual) == set(NAMES[:2])

def check(value, label):
    if not value: raise AssertionError(label)
    labels.append(label)

def assignment(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

outer = ast.parse(CONTROLS.read_bytes())
program = ast.literal_eval(assignment(outer, '_B_SEAM_PROGRAM'))
node = ast.literal_eval(assignment(outer, 'B_CONTROL_NODE'))
# Use the exact builder substitution for NODE; no whole program execution.
seam = ast.parse(program.replace('NODE', repr(node)))
start = next(i for i,n in enumerate(seam.body) if isinstance(n, ast.ImportFrom) and n.module == 'types')
end = next(i for i,n in enumerate(seam.body) if isinstance(n, ast.FunctionDef) and n.name == 'publish_outcome')
# Only pure schema constants/functions and publication functions. No setup/fixture.
selected = []
constants = {'BASELINE_IDENTITIES','NOT_REACHED','QUALIFICATION_KEYS','CALLBACK_ENUMS','CALLBACK_KEYS','MILESTONES'}
for n in seam.body[start:end+1]:
    if isinstance(n, (ast.FunctionDef, ast.ImportFrom)) or isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets): selected.append(n)
helper_tree = ast.parse(GUARD.read_bytes())
helper_functions = [n for n in helper_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('atomic','metadata')]
check(len(helper_functions)==2, 'real serializer/read closure extracted')
# Explicit inert private IO adapter: same os APIs, with no-follow on writes too.
class PrivateOS:
    def __getattr__(self, key): return getattr(os, key)
    def open(self, path, flags, mode=0o777):
        path = Path(path)
        assert path.parent.is_relative_to(BASE) and not path.parent.is_symlink()
        return os.open(path, flags | os.O_NOFOLLOW, mode)

helper = dict(os=PrivateOS(), json=json, stat=stat, uuid=uuid)
exec(compile(ast.Module(body=helper_functions,type_ignores=[]), str(GUARD), 'exec'), helper)
real_atomic, real_metadata = helper['atomic'], helper['metadata']

def fresh(identity, case):
    root = BASE / (identity+'-'+case)
    root.mkdir(mode=0o700)
    h = SimpleNamespace(atomic=real_atomic, metadata=real_metadata, _B_SURVIVAL=object())
    g = SimpleNamespace(receipts=[(18,18944)], owned=set(), failures=[], session=SimpleNamespace(shouldstop=False))
    env = dict(h=h, root=root, NODE=node, expected_control=identity, diagnostic_enabled=True,
        diagnostic_io_failed=False, guards=[g], reports=[('setup',True),('call',True),('teardown',True)])
    exec(compile(ast.Module(body=selected,type_ignores=[]), str(CONTROLS)+':extracted', 'exec'), env)
    raw=bytearray(64);raw[:4]=b'BDS\x02'
    for off,val in ((4,1),(6,1),(10,15),(12,24),(14,3),*((24+2*i,i+1) for i in range(5))):raw[off:off+2]=bytes((val,255-val))
    raw[16:24]=bytes((3,2,0,0,2,3,165,90))
    child=env['decode_survival'](bytes(raw),True,identity)
    parent=env['callback_record']('PARENT',identity)
    parent.update(terminal_called=True,status=18944,raw18944=True,exact_ef=True,exact_t=True,observe_entered=True,observe_completed=True,observe_status=True,observe_armed=True,observe_witness=True,observe_return=True)
    q=parent['qualification'];q.update(dict.fromkeys(env['QUALIFICATION_KEYS'],True));q['observe_calls']=1;env['assert_qualification'](q)
    env.update(parent_diagnostic=parent,teardown_snapshot=env['guard_snapshot'](g))
    real_atomic(root/NAMES[0],child);real_atomic(root/NAMES[1],parent)
    return env,child,parent,g

def failed(e,p,g):
    p.update(status=9,raw18944=False,exact_ef='NOT_REACHED',exact_t='NOT_REACHED',child_bytes='E',terminal_result='TK',suffix='TK',observe_status=False,observe_witness=False)
    p['qualification'].update(terminal=False,observe_status=False,observe_witness=False)
    try:e['assert_qualification'](p['qualification'])
    except AssertionError as exc: check(str(exc)=='baseline qualification: terminal','original deferred failure')
    g.receipts[:]=[(18,9)]
    e['reports'][-1]=('teardown',False)
    e['teardown_snapshot']=e['guard_snapshot'](g)
    real_atomic(e['root']/NAMES[1],p)

def records(e): return {n:real_metadata(e['root']/n) for n in NAMES}
def status(e,r):return e['validate_baseline_records'](r,e['expected_control'],node,dict(schema=2,control=e['expected_control'],node=node))

for identity in ('ORIGINAL74_REAL_OBSERVER','COLLECTED_CONTROL'):
    e,c,p,g=fresh(identity,'regression')
    failed(e,p,g)
    if BUG:
        try:e['publish_outcome'](1)
        except UnboundLocalError as exc: check('diagnostic_io_failed' in str(exc),identity+' before-fix UnboundLocalError')
        else: raise AssertionError('bug not reproduced')
        continue
    e['publish_outcome'](1);r=records(e)
    check(status(e,r)==dict(diagnostic='COMPLETE',test_status='FAIL'),identity+' coherent failed qualification COMPLETE/FAIL')
    check(r[NAMES[2]]['pytest_exit']==1 and r[NAMES[2]]['phases']['teardown']=='FAILED',identity+' original nonzero and teardown')
    for case in ('pass','raw9-incomplete','prelatched','missing','malformed','mixed','validation','read','write'):
        e,c,p,g=fresh(identity,case)
        exit_code=0 if case=='pass' else 1
        if case=='raw9-incomplete':
            failed(e,p,g)
            raw=bytearray(64);raw[:4]=b'BDS\x02'
            for off,val in ((4,1),(6,1),(10,2),(24,1),(26,2),(28,3)):raw[off:off+2]=bytes((val,255-val))
            c=e['decode_survival'](bytes(raw),True,identity)
            check(c['state']=='INCOMPLETE' and not c['result_complete'],'r131 incomplete shape')
            if actual is not None:
                c,p = (copy.deepcopy(actual[n]) for n in NAMES[:2])
                # Exact018 records;017 is an explicit inert identity projection.
                c['control']=p['control']=identity
                e['parent_diagnostic']=p
                real_atomic(e['root']/NAMES[1],p)
                check(p['qualification']==dict(policy='BASELINE_AFTER_OBSERVE',terminal=False,observe_status=False,observe_witness=False,observe_return=True,receipt_owned=True,observe_calls=1,assertions='FAILED',first_failed='terminal'),identity+' actual r131 qualification unchanged')
            real_atomic(e['root']/NAMES[0],c)
        if case=='prelatched':e['diagnostic_io_failed']=True
        if case=='missing': (e['root']/NAMES[0]).rename(e['root']/'retained-child.json')
        if case=='malformed':real_atomic(e['root']/NAMES[0],{'bad':'schema'})
        if case=='mixed':c['control']='COLLECTED_CONTROL' if identity=='ORIGINAL74_REAL_OBSERVER' else 'ORIGINAL74_REAL_OBSERVER';real_atomic(e['root']/NAMES[0],c)
        if case=='validation':
            p['qualification']['receipt_owned']=False
            # Deliberate real validator failure: false PASSED; no validator stub.
            real_atomic(e['root']/NAMES[1],p)
        if case=='read':
            def broken_read(path):raise OSError('inert read failure')
            e['h'].metadata=broken_read
        if case=='write':
            def broken_write(path,value):raise OSError('inert write failure')
            e['h'].atomic=broken_write
        saved=copy.deepcopy(p['qualification'])
        e['publish_outcome'](exit_code)
        if case=='write':
            check(e['diagnostic_io_failed'] and not (e['root']/NAMES[2]).exists(),identity+' write failure latch/no outcome/nonzero')
            continue
        out=real_metadata(e['root']/NAMES[2])
        check(out['teardown']==e['teardown_snapshot'] and out['final']==e['guard_snapshot'](g),identity+' '+case+' snapshots preserved')
        check(out['pytest_exit']==exit_code and out['qualification']==saved,identity+' '+case+' saved qualification/exit')
        check(out['qualification']==real_metadata(e['root']/NAMES[1]).get('qualification'),identity+' '+case+' parent equality')
        check(out['diagnostic']==('COMPLETE' if case=='pass' else 'RED'),identity+' '+case+' completeness')
        if case in ('pass','raw9-incomplete','prelatched'):
            check(status(e,records(e))['test_status']==('PASS' if case=='pass' else 'FAIL'),identity+' '+case+' real validator status')
        if case=='prelatched':check(e['diagnostic_io_failed'] and out['io_failed'],'prelatched never reset')
        g.receipts.clear();g.owned.add(99);g.failures.append('late mutation')
        check(real_metadata(e['root']/NAMES[2])==out and p['qualification']==saved,identity+' '+case+' later guard mutation cannot rewrite capture')
        for foreign_node,binding in (() if case != 'pass' else ((node+'foreign',dict(schema=2,control=identity,node=node)),(node,dict(schema=2,control='FOREIGN',node=node)))):
            try:e['validate_baseline_records'](records(e),identity,foreign_node,binding)
            except (ValueError,KeyError):check(True,identity+' external node/private binding rejected')
            else:raise AssertionError('external binding accepted')
    e,c,p,g=fresh(identity,'false-zero')
    failed(e,p,g)
    e['publish_outcome'](0)
    check(e['diagnostic_io_failed'] and real_metadata(e['root']/NAMES[2])['diagnostic']=='RED',identity+' real validator rejects false exit0')
    assertion=next(n for n in seam.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='diagnostic_enabled' and any(isinstance(x,ast.Assert) and 'diagnostic_io_failed' in ast.unparse(x) for x in n.body))
    try:exec(compile(ast.Module(body=[assertion],type_ignores=[]),'exact-post-handback','exec'),e)
    except AssertionError:check(True,identity+' exact IO latch assertion forces nonzero')
    else:raise AssertionError('false zero escaped')
    # Execute exact handback finally subtree, with inert map and no pytest body.
    handback=next(n for n in seam.body if isinstance(n,ast.Try) and n.finalbody and any(isinstance(x,ast.Try) for x in n.finalbody))
    final=handback.finalbody
    for escaped in (False,True):
        e,c,p,g=fresh(identity,'handback-'+str(escaped))
        failed(e,p,g)
        class Mapping:
            closed=False
            def __getitem__(self,key):return bytes.fromhex(c['raw'])
            def close(self):self.closed=True
        mapping=Mapping();original=object()
        e.update(survival=mapping,wait_delivered=True,parent_steps=[],parent_owner=1,mode='inert',binding0=original,result=1)
        marker=RuntimeError('original primary failure')
        body=[ast.Raise(exc=ast.Name(id='marker',ctx=ast.Load()),cause=None)] if escaped else [ast.Pass()]
        e['marker']=marker
        try:exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Try(body=body,handlers=[],orelse=[],finalbody=final)],type_ignores=[])),'exact-handback','exec'),e)
        except RuntimeError as exc:check(escaped and exc is marker,'original primary exception identity preserved')
        check(e['h']._B_SURVIVAL is original and mapping.closed and e['survival'] is None,identity+' exact handback restoration/close')
        check(real_metadata(e['root']/NAMES[2])['pytest_exit']==1,identity+' handback original exit')
    # Unexpected publication exception chains to the original, still restores.
    e,c,p,g=fresh(identity,'exception-context')
    class BadMapping:
        closed=False
        def __getitem__(self,key):raise OSError('inert mapping read')
        def close(self):self.closed=True
    mapping=BadMapping();original=object();marker=RuntimeError('original primary failure')
    e.update(survival=mapping,wait_delivered=True,binding0=original,marker=marker)
    tree=ast.Module(body=[ast.Try(body=[ast.Raise(exc=ast.Name(id='marker',ctx=ast.Load()),cause=None)],handlers=[],orelse=[],finalbody=final)],type_ignores=[])
    try:exec(compile(ast.fix_missing_locations(tree),'exact-handback','exec'),e)
    except OSError as exc:check(exc.__context__ is marker,'original primary exception context visible')
    check(mapping.closed and e['h']._B_SURVIVAL is original,'exception handback restoration')

result=dict(classification='INERT_HOST_ONLY',expect_bug=BUG,actual_pax_consumed=actual is not None,checks=len(labels),labels=labels,controls_sha256=hashlib.sha256(CONTROLS.read_bytes()).hexdigest(),guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest())
real_atomic(BASE/'results.json', {k:v for k,v in result.items() if k!='labels'})
fd=os.open(BASE/'labels.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as stream:json.dump(labels,stream)
print(json.dumps({k:v for k,v in result.items() if k!='labels'}))
