"""Standalone stdlib inert HOST checks; never import guard/product or run pytest.

Usage: python -I -S -B check_final_gap.py PRIVATE_NEW_DIRECTORY
  --baseline=PRIVATE_BASELINE_DIRECTORY --private=PRIVATE_DECODER_DIRECTORY
  --actual-pax=PINNED_R131_PAX
Baseline directory contains guard-before.py and controls-before.py at d8ffc4e.
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
    h = SimpleNamespace(atomic=real_atomic, metadata=real_metadata, _B_SURVIVAL=object(), _B_GAP_ENABLED=False, _B_GAP_CHILD=False, _B_GAP_CLOCK=None, _B_GAP_PARENT=None)
    g = SimpleNamespace(receipts=[(18,18944)], owned=set(), failures=[], session=SimpleNamespace(shouldstop=False))
    env = dict(h=h, root=root, NODE=node, expected_control=identity, diagnostic_enabled=True,
        diagnostic_io_failed=False, guards=[g], reports=[('setup',True),('call',True),('teardown',True)])
    exec(compile(ast.Module(body=selected,type_ignores=[]), str(CONTROLS)+':extracted', 'exec'), env)
    raw=bytearray(64);raw[:4]=b'BDS\x02'
    for off,val in ((4,1),(6,1),(10,15),(12,24),(14,3),*((24+2*i,i+1) for i in range(5))):raw[off:off+2]=bytes((val,255-val))
    raw[16:24]=bytes((3,2,0,0,2,3,165,90))
    if identity == 'COLLECTED_CONTROL':
        raw[:4] = b'BDS\x03'
        for off,val in ((36,100),(44,101),(52,102)):
            raw[off:off+6]=val.to_bytes(6,'little');raw[off+6:off+8]=b'\xa5\x5a'
        raw[60:64]=b'\x01\xfe\x01\xfe'
    child=env['decode_survival'](bytes(raw),True,identity)
    parent=env['callback_record']('PARENT',identity)
    parent.update(terminal_called=True,status=18944,raw18944=True,exact_ef=True,exact_t=True,observe_entered=True,observe_completed=True,observe_status=True,observe_armed=True,observe_witness=True,observe_return=True)
    if identity == 'COLLECTED_CONTROL': parent['gap_parent']=env['gap_parent_record']([99,101,103,1,1,0])
    env.update(gap_parent=([99,101,103,1,1,0] if identity == 'COLLECTED_CONTROL' else None),gap_bindings0=(False,False,None,None))
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
def status(e,r):return e['validate_baseline_records'](r,e['expected_control'],node,dict(schema=2,control=e['expected_control'],node=node,**({'layout':'FINAL_GAP_V1'} if e['expected_control']=='COLLECTED_CONTROL' else {})))

# Only affected018 publication cases; unchanged017 stays in original r133 smoke.
for identity in ('COLLECTED_CONTROL',):
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
            if identity == 'COLLECTED_CONTROL': raw[:4]=b'BDS\x03'
            c=e['decode_survival'](bytes(raw),True,identity)
            check(c['state']=='INCOMPLETE' and not c['result_complete'],'r131 incomplete shape')
            if actual is not None:
                c,p = (copy.deepcopy(actual[n]) for n in NAMES[:2])
                # Exact018 records;017 is an explicit inert identity projection.
                c['control']=p['control']=identity
                if identity == 'COLLECTED_CONTROL':
                    c=e['decode_survival'](bytes.fromhex(c['raw']),True,identity)
                    p['gap_parent']=e['gap_parent_record']([99,101,103,1,1,0])
                    check(not c['result_complete'] and c['state']=='UNKNOWN','actual r131 extended tag rejected')
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


# R136 additional finite coverage groups.
# Group 1: enumerate only the commissioned additions, then compare the full
# original guard AST. No original predicate/exception/scope node is normalized.
baseline_dir=Path(next(a.split('=',1)[1] for a in sys.argv[2:] if a.startswith('--baseline=')))
private_dir=Path(next(a.split('=',1)[1] for a in sys.argv[2:] if a.startswith('--private=')))
old_guard=ast.parse((baseline_dir/'guard-before.py').read_bytes())
new_guard=ast.parse(GUARD.read_bytes())
class StripGap(ast.NodeTransformer):
    removed=0
    def visit_If(self,n):
        if isinstance(n.test,ast.BoolOp) and isinstance(n.test.values[0],ast.Name) and n.test.values[0].id=='_B_GAP_ENABLED':
            self.removed+=1;return None
        return self.generic_visit(n)
    def visit_Global(self,n):
        if n.names==['_B_GAP_CHILD']:self.removed+=1;return None
        return n
    def visit_Assign(self,n):
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_B_GAP_ENABLED','_B_GAP_CHILD','_B_GAP_CLOCK','_B_GAP_PARENT'):
            self.removed+=1;return None
        return self.generic_visit(n)
strip=StripGap();residual=strip.visit(copy.deepcopy(new_guard))
check(strip.removed==12,'exact 12 enumerated guard AST additions')
check(ast.dump(residual,include_attributes=False)==ast.dump(old_guard,include_attributes=False),'full guard residual identical: predicates scopes truth consumption exception order')
old_outer=ast.parse((baseline_dir/'controls-before.py').read_bytes())
old_program=ast.literal_eval(assignment(old_outer,'_B_SEAM_PROGRAM'));old_seam=ast.parse(old_program.replace('NODE',repr(node)))
# The outer source, including exact mode inventory/order and assertions, is frozen.
for tree in (old_outer,outer):
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_B_SEAM_PROGRAM' for t in n.targets):n.value=ast.Constant(value='INERT_PROGRAM_PLACEHOLDER')
check(ast.dump(old_outer)==ast.dump(outer),'all outer controls including mode inventory/order unchanged')
old_plugin=next(n for n in old_seam.body if isinstance(n,ast.ClassDef) and n.name=='Plugin')
new_plugin=copy.deepcopy(next(n for n in seam.body if isinstance(n,ast.ClassDef) and n.name=='Plugin'))
class StripWait(ast.NodeTransformer):
    removed=0
    def visit_If(self,n):
        if ast.unparse(n.test)=='h._B_GAP_ENABLED':self.removed+=1;return None
        return self.generic_visit(n)
sw=StripWait();new_plugin=sw.visit(new_plugin)
check(sw.removed==1 and ast.dump(old_plugin)==ast.dump(new_plugin),'Plugin residual exact: original qualification assertions weakrefs and frames')
# Source placements, exact isolated functions, no guard module import.
witness=next(n for n in new_guard.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='final_child' for m in n.body))
methods={n.name:n for n in witness.body if isinstance(n,ast.FunctionDef)}
hooks=methods['hooks'];final_child=methods['final_child']
check(isinstance(hooks.body[0],ast.If) and isinstance(hooks.body[1],ast.Import),'H0 precedes unchanged import select')
check(ast.unparse(hooks.body[2].test)=='_B_GAP_ENABLED and _B_GAP_CHILD','import returned immediately after import')
# Both owning closures are actual extracted functions; no lifecycle imports.
private_tree=ast.parse((private_dir/'diagnostic-host.py').read_bytes())
private_env=dict(NODE=node)
pure_names={n.name for n in selected if isinstance(n,ast.FunctionDef)}-{'publish_diagnostic','publish_outcome','witness_snapshot','guard_snapshot'}
private_selected=[n for n in private_tree.body if isinstance(n,ast.ImportFrom) and n.module=='types' or isinstance(n,ast.FunctionDef) and n.name in pure_names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
exec(compile(ast.Module(body=private_selected,type_ignores=[]),'private-owning-decoder-extracted','exec'),private_env)
for name in ('callback_record','validate_callback','decode_gap','decode_survival','validate_survival','validate_gap_parent','validate_gap_crosscheck','validate_baseline_records'):
    a=next(n for n in seam.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in private_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    class BindNode(ast.NodeTransformer):
        def visit_Name(self,n): return ast.copy_location(ast.Constant(value=node),n) if n.id=='NODE' and isinstance(n.ctx,ast.Load) else n
    check(ast.dump(a)==ast.dump(BindNode().visit(copy.deepcopy(b))),'owning AST agreement '+name)
binding=json.loads((private_dir/'baseline-binding-draft.json').read_bytes())
check(binding==dict(schema=2,control='COLLECTED_CONTROL',node=node,layout='FINAL_GAP_V1'),'private explicit layout binding')
# Group 2: execute actual inline stores against explicit inert memory and clocks.
store_nodes={}
for n in ast.walk(new_guard):
    if isinstance(n,ast.If) and ast.unparse(n.test)=='_B_GAP_ENABLED and _B_GAP_CHILD':
        assignments=[x for x in ast.walk(n) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='_B_SURVIVAL' for t in x.targets)]
        first=assignments[0].targets[0].slice
        off=first.lower.value if isinstance(first,ast.Slice) else first.value
        check(off not in store_nodes,'one inline store site '+str(off));store_nodes[off]=n
check(set(store_nodes)=={36,44,52,60,62},'exact five child stores')
class InertMap:
    def __init__(self,raw,fail=None,tear=None):self.raw=bytearray(raw);self.writes=0;self.fail=fail;self.tear=tear;self.closed=False;self.reads=0
    def __getitem__(self,k):self.reads+=1;return self.raw[k]
    def __setitem__(self,k,v):
        self.writes+=1
        if self.fail==self.writes:
            if isinstance(k,slice) and self.tear is not None:self.raw[k.start:k.start+self.tear]=v[:self.tear]
            raise OSError('explicit inert store interruption')
        self.raw[k]=v
    def close(self):self.closed=True
class CountClock:
    def __init__(self,value=100000,fail=False):self.calls=0;self.value=value;self.fail=fail
    def __call__(self):
        self.calls+=1
        if self.fail:raise OSError('explicit inert clock failure')
        return self.value

def run_store(off,mapping,clock,active=True):
    env=dict(_B_GAP_ENABLED=True,_B_GAP_CHILD=active,_B_SURVIVAL=mapping,_B_GAP_CLOCK=clock)
    exec(compile(ast.Module(body=[store_nodes[off]],type_ignores=[]),'actual-inline-store','exec'),env)

e,c,p,g=fresh('COLLECTED_CONTROL','boundaries')
full=bytes.fromhex(c['raw'])
# In program execution order: F2, H0, import, ready, H1.
base=bytearray(full);base[36:]=bytes(28);base[10:16]=b'\x02\xfd\x00\x00\x00\x00';base[16:24]=bytes(8);base[30:34]=bytes(4)
for off in (36,44,60,62,52):
    if off==52:base[14:16]=b'\x01\xfe'
    # Every individual payload byte prefix plus each commit write boundary.
    attempts=[(None,None)]+([(1,k) for k in range(7)]+[(2,None),(3,None)] if off<60 else [(1,None),(2,None)])
    for fail,tear in attempts:
        m=InertMap(base,fail,tear);clock=CountClock(100000+off*1000);run_store(off,m,clock)
        check(clock.calls==(0 if off>=60 else 1),'once clock '+str((off,fail,tear)))
        for decoder in (e,private_env):
            decoded=decoder['decode_survival'](bytes(m.raw),True,'COLLECTED_CONTROL')
            check(not decoded['result_complete'],'every interruption/prefix remains incomplete '+str((off,fail,tear)))
            decoder['validate_survival'](decoded,'COLLECTED_CONTROL')
    m=InertMap(base);run_store(off,m,CountClock(100000+off*1000));base=m.raw
# A complete prefix may commit equal timestamps; reorder as time, not byte offset.
for off,v in ((36,100),(44,101),(52,102)):base[off:off+6]=v.to_bytes(6,'little')
for decoder in (e,private_env):
    check(decoder['decode_gap'](bytes(base))['error']==0,'legal complete gap prefix')
for mutate,label in ((lambda b:b.__setitem__(slice(44,50),(99).to_bytes(6,'little')),'reverse'),(lambda b:b.__setitem__(42,0),'partial commit'),(lambda b:b.__setitem__(61,253),'bad progress'),(lambda b:b.__setitem__(slice(0,4),b'BDS\x02'),'mixed tag')):
    bad=bytearray(full);mutate(bad)
    for decoder in (e,private_env):check(not decoder['decode_survival'](bytes(bad),True,'COLLECTED_CONTROL')['result_complete'],'failclosed '+label)
for off in store_nodes:
    m=InertMap(full);cl=CountClock();run_store(off,m,cl)
    check(m.raw[34:36]==b'\x01\xfe' and cl.calls==0,'duplicate fails closed before clock '+str(off))
for value in (-1000,(1<<48)*1000):
    m=InertMap(bytes(64));cl=CountClock(value);run_store(36,m,cl)
    check(m.raw[34:36]==b'\x01\xfe' and m.raw[42:44]==bytes(2),'no wrap/clamp '+str(value))
for off in (36,44,52):
    m=InertMap(bytes(64));cl=CountClock(fail=True);run_store(off,m,cl)
    check(cl.calls==1 and m.raw[34:36]==b'\x01\xfe','clock failure diagnostic only '+str(off))
    m=InertMap(bytes(64));cl=CountClock();run_store(off,m,cl,False)
    check(cl.calls==0 and m.writes==0,'parent role excluded '+str(off))
for decoder in (e,private_env):
    try:decoder['decode_survival'](full,False,'COLLECTED_CONTROL')
    except ValueError:check(True,'no delivery rejects bytes')
    else:raise AssertionError('unobserved bytes accepted')
    check(not decoder['decode_survival'](None,False,'COLLECTED_CONTROL')['result_complete'],'no delivery incomplete')
# Group 3: actual hooks/final functions with original operands and fake IO/OS.
# Use the full extracted functions, with short circuit at the original first
# predicate (False or raising), and compare exact return/exception/finally state.
function_env=dict(_B_GAP_ENABLED=True,_B_GAP_CHILD=True,_B_SURVIVAL=None,_B_GAP_CLOCK=None,sys=sys,os=SimpleNamespace())
exec(compile(ast.Module(body=[copy.deepcopy(hooks),copy.deepcopy(final_child)],type_ignores=[]),'actual-functions-inert','exec'),function_env)
old_witness=next(n for n in old_guard.body if isinstance(n,ast.ClassDef) and n.name==witness.name)
old_methods={n.name:n for n in old_witness.body if isinstance(n,ast.FunctionDef)}
old_env=dict(sys=sys,os=function_env['os'],_B_SURVIVAL=None)
exec(compile(ast.Module(body=[copy.deepcopy(old_methods['hooks']),copy.deepcopy(old_methods['final_child'])],type_ignores=[]),'original-functions-inert','exec'),old_env)
class Truth:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def __bool__(self):
        self.calls+=1
        if self.fail:raise LookupError('original truth exception')
        return False
for fail_clock in (False,True):
    for truth_raises in (False,True):
        observations=[]
        for env in (old_env,function_env):
            truth=Truth(truth_raises)
            d=dict(final_active=False,child_role=True,gate=0,hook_gate=0,kind='UNREACHED')
            w=SimpleNamespace(valid=truth,_refusal=d,phase='timeout')
            m=InertMap(bytes(64));cl=CountClock(fail=fail_clock)
            env.update(_B_SURVIVAL=m,_B_GAP_CLOCK=cl)
            try:answer=env['final_child'](w,None);outcome=('return',answer)
            except BaseException as exc:outcome=('raise',type(exc),str(exc))
            observations.append((outcome if outcome[0]=='raise' else ('return-original-truth' if answer is truth else 'return', answer if answer is not truth else None),truth.calls,w.valid is truth,d['final_active']))
            check(cl.calls==0,'false first gate never calls gap clock')
        check(observations[0]==observations[1],'actual final first false/caught truth consumption preserved')
# Hooks closure preparation executes before first predicate, but authority
# predicates stop at gate1. No operation/Observer/product is invoked.
for closure_raises in (False,True):
    for failed_write in (None,1,2):
        observed=[]
        for env in (old_env,function_env):
            calls=[]
            def closure(value):
                calls.append(value)
                if closure_raises:raise LookupError('original closure failure')
                return {}
            d=dict(final_active=True,child_role=True,hook_gate=0)
            w=SimpleNamespace(_refusal=d,closure=closure,hook_functions=[SimpleNamespace(__code__=1)],hook_codes=[2])
            for key in ('model_fork','session_wrapper','session_body','limit_wrapper','framing_wrapper','observed','ready','blocked','clock'):setattr(w,key,key)
            m=InertMap(bytes(64),failed_write);cl=CountClock()
            env.update(_B_SURVIVAL=m,_B_GAP_CLOCK=cl)
            try:answer=env['hooks'](w);outcome=('return',answer)
            except BaseException as exc:outcome=('raise',type(exc),str(exc))
            observed.append((outcome,calls,d['hook_gate']))
            if env is function_env:check(cl.calls==2,'H0 H1 clock count in actual hooks')
        check(observed[0]==observed[1],'hooks false/caught original exception and operand order preserved')
# Group 4: private owning validator consumes real publication records and exact
# saved captures. Worst finite JSON shapes are measured with the real serializer.
e,c,p,g=fresh('COLLECTED_CONTROL','private-positive');e['publish_outcome'](0);r=records(e)
check(private_env['validate_baseline_records'](r,'COLLECTED_CONTROL',node,binding)==dict(diagnostic='COMPLETE',test_status='PASS'),'private owning validator positive publication')
for mutation in ('layout','clock-before-setup','clock-after-wait','timeout-after-wait','duplicate-count','overflow'):
    bad=copy.deepcopy(r);timing=bad[NAMES[1]]['gap_parent']
    if mutation=='layout':timing['layout']='UNKNOWN'
    elif mutation=='clock-before-setup':timing['setup']=101
    elif mutation=='clock-after-wait':timing['wait']=101
    elif mutation=='timeout-after-wait':timing['timeout']=104
    elif mutation=='duplicate-count':timing['timeout_count']=2
    elif mutation=='overflow':timing['setup']=1<<48
    for decoder in (e,private_env):
        try:result_bad=decoder['validate_baseline_records'](bad,'COLLECTED_CONTROL',node,binding)
        except (ValueError,KeyError):check(True,'invalid publication rejected '+mutation)
        else:check(result_bad['test_status']=='FAIL','invalid publication never positive '+mutation)
max_sizes={}
for leaf,record in r.items():
    worst=copy.deepcopy(record)
    if leaf==NAMES[0]:
        for k in ('f2','h0','h1'):worst['gap'][k]=(1<<48)-1
    if leaf==NAMES[1]:
        for k in ('setup','timeout','wait'):worst['gap_parent'][k]=(1<<48)-1
    if leaf==NAMES[2]:
        for key in ('final','teardown'):
            worst[key]['failures']=['MISSING_RECEIPT']*8;worst[key]['receipt_statuses']=['OTHER']*2
        worst['missing_records']=list(NAMES[:2])
    # Actual serializer uses sorted JSON compact separators; maximum finite
    # shape includes all fields; invalid enums cannot make an accepted leaf.
    real_atomic(e['root']/('maximum-'+leaf),worst)
    size=(e['root']/('maximum-'+leaf)).stat().st_size
    max_sizes[leaf]=size;check(size<=4096,'actual serializer cap '+leaf)
# Group 5: exact handback including no-delivery and setup-failure paths.
handback=next(n for n in seam.body if isinstance(n,ast.Try) and n.finalbody and any(isinstance(x,ast.Try) for x in n.finalbody))
for no_mapping,delivery in ((False,False),(True,False),(False,True)):
    e,c,p,g=fresh('COLLECTED_CONTROL','lifetime-'+str((no_mapping,delivery)))
    oldbindings=(False,False,None,None);m=None if no_mapping else InertMap(bytes.fromhex(c['raw']))
    e.update(survival=m,wait_delivered=delivery,parent_steps=[],parent_owner=1,mode='inert',binding0=object(),gap_bindings0=oldbindings,result='ESCAPED',gap_parent=None)
    e['h']._B_GAP_ENABLED=True;e['h']._B_GAP_CHILD=True;e['h']._B_GAP_CLOCK=CountClock();e['h']._B_GAP_PARENT=[99,None,None,0,0,0]
    exec(compile(ast.Module(body=handback.finalbody,type_ignores=[]),'actual-handback','exec'),e)
    check(tuple(getattr(e['h'],k) for k in ('_B_GAP_ENABLED','_B_GAP_CHILD','_B_GAP_CLOCK','_B_GAP_PARENT'))==oldbindings,'all new bindings restored '+str((no_mapping,delivery)))
    check(e['gap_parent'] is None and e['gap_clock'] is None and e['survival'] is None,'temporary roots cleared')
    if m is not None:check(m.closed and (delivery or m.reads==0),'close and no-read-without-wait')
# Actual final gate2: false hook and caught original exception, while diagnostic
# clock/mapping failures occur inside the new F2 store.
for raises in (False,True):
    for clock_fails,store_fails in ((False,None),(True,None),(False,9),(False,10)):
        observed=[]
        for env in (old_env,function_env):
            call_count=[0]
            def inert_hooks():
                call_count[0]+=1
                if raises:raise LookupError('original hooks exception')
                return False
            d=dict(final_active=False,child_role=True,gate=0,hook_gate=0,kind='UNREACHED')
            w=SimpleNamespace(valid=True,_refusal=d,phase='timeout',hooks=inert_hooks)
            m=InertMap(bytes(64),store_fails);cl=CountClock(fail=clock_fails);env.update(_B_SURVIVAL=m,_B_GAP_CLOCK=cl)
            answer=env['final_child'](w,None)
            observed.append((answer,call_count[0],w.valid,d.copy()))
            if env is function_env:check(cl.calls==1,'F2 clock exactly once despite original/store failure')
        check(observed[0]==observed[1],'final gate2 original caught/false result validity exception scopes unchanged')
# Actual parent stores, extracted from existing trace and terminal assignment.
parent_store=next(n for n in ast.walk(methods['trace_B']) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('_B_GAP_ENABLED and local.get'))
wait_store=next(n for n in ast.walk(next(n for n in seam.body if isinstance(n,ast.ClassDef) and n.name=='Plugin')) if isinstance(n,ast.If) and ast.unparse(n.test)=='h._B_GAP_ENABLED')
for which,fragment in (('timeout',parent_store),('wait',wait_store)):
    for fail_clock in (False,True):
        values=[99,None,None,0,0,0];cl=CountClock(100000,fail_clock)
        env=dict(_B_GAP_ENABLED=True,_B_GAP_PARENT=values,_B_GAP_CLOCK=cl,local={'pid':123},h=SimpleNamespace(_B_GAP_ENABLED=True,_B_GAP_CLOCK=cl),gap_parent=values)
        code=compile(ast.Module(body=[fragment],type_ignores=[]),'actual-parent-store','exec')
        exec(code,env);check(cl.calls==1,'parent one clock '+which)
        exec(code,env);check(cl.calls==1 and values[5]==1,'parent duplicate no second clock '+which)
# Conservative maximum of every finite schema field, including nonpositive
# enums; combinations need not qualify. It bounds all valid combinations.
e,c,p,g=fresh('COLLECTED_CONTROL','maximum-envelope');e['publish_outcome'](0);envelope=records(e)
child=envelope[NAMES[0]];child.update(state='CAUGHT_EXCEPTION',result_complete=False,first_site=13,final_gate=15,hook_gate=24,hook_stage=5,error=1)
for k in ('f2','h0','h1'):child['gap'][k]=(1<<48)-1
child['gap']['import_returned']=child['gap']['diagnostic_ready']=False
parent=envelope[NAMES[1]]
for k in list(parent):
    if k in e['CALLBACK_ENUMS']:
        allowed=e['CALLBACK_ENUMS'][k];parent[k]=max(allowed,key=lambda v:len(json.dumps(v)))
    elif k not in ('qualification','gap_parent'):parent[k]='NOT_REACHED'
parent.update(schema=2,control='COLLECTED_CONTROL',role='PARENT',first_site=13,gate=15,hook_gate=24,status='NOT_REACHED',kind='CAUGHT_EXCEPTION',phase='BACKPRESSURE')
for k in ('setup','timeout','wait'):parent['gap_parent'][k]=(1<<48)-1
for k in e['QUALIFICATION_KEYS']:parent['qualification'][k]='NOT_REACHED'
parent['qualification'].update(assertions='NOT_REACHED',first_failed='observe_witness')
out=envelope[NAMES[2]];out['qualification']=copy.deepcopy(parent['qualification']);out['pytest_exit']='ESCAPED';out['missing_records']=list(NAMES[:2]);out['phases']=dict.fromkeys(('setup','call','teardown'),'NOT_REACHED')
for k in ('teardown','final'):
    out[k]['failures']=['MISSING_RECEIPT']*8;out[k]['receipt_statuses']=['OTHER']*2;out[k]['shouldstop']='MATCH_FIRST';out[k]['owned']=out[k]['receipts']=9
for leaf,record in envelope.items():
    real_atomic(e['root']/('envelope-'+leaf),record);size=(e['root']/('envelope-'+leaf)).stat().st_size
    max_sizes[leaf]=max(max_sizes[leaf],size);check(size<=4096,'conservative maximum finite field envelope '+leaf)
# Actual setup statement under fake mmap. Stdlib builtin clock remains the
# required primitive; no guard, pytest or product executes.
import builtins,time
setup=handback.body[0]
class FakeMmap:
    MAP_SHARED=1;MAP_ANON=2;PROT_READ=4;PROT_WRITE=8
    def __init__(self,fail):self.fail=fail;self.mapping=InertMap(bytes(64));self.calls=0
    def mmap(self,*a,**kw):
        self.calls+=1
        if self.fail:raise OSError('explicit setup allocation failure')
        return self.mapping
for fail in (False,True):
    e,c,p,g=fresh('COLLECTED_CONTROL','actual-setup-'+str(fail));fake=FakeMmap(fail)
    originals=(False,False,None,None);prior=object()
    e.update(survival=None,binding0=prior,gap_bindings0=originals,wait_delivered=False,gap_parent=None,result='ESCAPED',parent_steps=[],parent_owner=1,mode='inert')
    def importing(name,*a,**kw):return fake if name=='mmap' else builtins.__import__(name,*a,**kw)
    e['__builtins__']=dict(vars(builtins),__import__=importing)
    before=compile(ast.Module(body=[setup],type_ignores=[]),'actual-setup-inert','exec')
    try:exec(before,e)
    except OSError:check(fail,'actual setup failure retained')
    else:
        check(not fail and e['h']._B_GAP_CLOCK is time.monotonic_ns,'setup captures builtin clock identity')
        check(e['h']._B_GAP_CHILD is False and e['gap_parent'][0]>=0,'setup role false primitive timestamp')
        check(all(type(v) in (int,type(None)) for v in e['gap_parent']),'parent roots primitive only')
        check(bytes(fake.mapping.raw)==b'BDS\x03'+bytes(60),'exact BDS03 setup mapping')
    exec(compile(ast.Module(body=handback.finalbody,type_ignores=[]),'actual-setup-handback','exec'),e)
    check(e['h']._B_SURVIVAL is prior and e['h']._B_GAP_CLOCK is None and e['h']._B_GAP_PARENT is None,'setup failure/normal restores all bindings')
    check(fail or fake.mapping.closed,'setup mapping closed after no wait')

# Interruption snapshots of actual writes, including every payload-byte boundary,
# with no exception/error publication (models stopped writer, not a caught error).
class SnapshotMap(InertMap):
    def __init__(self,raw):super().__init__(raw);self.snapshots=[]
    def __setitem__(self,k,v):
        if isinstance(k,slice):
            for index,byte in enumerate(v):
                self.raw[k.start+index]=byte;self.snapshots.append(bytes(self.raw))
        else:self.raw[k]=v;self.snapshots.append(bytes(self.raw))
interrupted=bytearray(full);interrupted[36:]=bytes(28);interrupted[10:16]=b'\x02\xfd\x00\x00\x00\x00';interrupted[16:24]=bytes(8);interrupted[30:34]=bytes(4)
for off,value in ((36,100),(44,101),(60,0),(62,0),(52,102)):
    if off==52:interrupted[14:16]=b'\x01\xfe'
    m=SnapshotMap(interrupted);run_store(off,m,CountClock(value*1000))
    check(len(m.snapshots)==(8 if off<60 else 2),'all payload/commit boundaries captured '+str(off))
    for snapshot in [bytes(interrupted)]+m.snapshots:
        for decoder in (e,private_env):
            decoded=decoder['decode_survival'](snapshot,True,'COLLECTED_CONTROL');decoder['validate_survival'](decoded,'COLLECTED_CONTROL')
            check(not decoded['result_complete'],'interrupted actual write snapshot never positive '+str(off))
    interrupted=m.raw
# Zero/equal timestamps legal only with exact commits and same-domain bounds.
e,c,p,g=fresh('COLLECTED_CONTROL','zero-equal');zero=bytearray.fromhex(c['raw'])
for off in (36,44,52):zero[off:off+6]=bytes(6)
c=e['decode_survival'](bytes(zero),True,'COLLECTED_CONTROL');p['gap_parent']=e['gap_parent_record']([0,0,0,1,1,0]);e['parent_diagnostic']=p
real_atomic(e['root']/NAMES[0],c);real_atomic(e['root']/NAMES[1],p);e['publish_outcome'](0)
for decoder in (e,private_env):check(decoder['validate_baseline_records'](records(e),'COLLECTED_CONTROL',node,binding)['test_status']=='PASS','committed zero equal timestamps legal')
# Original full raw74 cannot promote missing timeout, missing progress or stages.
for off in (36,44,52,60,62):
    bad=bytearray(full);bad[off:off+(8 if off<60 else 2)]=bytes(8 if off<60 else 2)
    child=e['decode_survival'](bytes(bad),True,'COLLECTED_CONTROL')
    for decoder in (e,private_env):check(not decoder['validate_gap_crosscheck'](child,records(e)[NAMES[1]]) or not child['result_complete'],'missing extension cannot qualify raw74 '+str(off))
# Host residual: all lifecycle/retention/PAX/capture functions outside named
# owning consumers remain exactly the original AST, including dependencies.
origins=json.loads((private_dir/'origins.json').read_bytes())
old_host=ast.parse(Path(origins['origins']['diagnostic-host.py']['path']).read_bytes())
residual_host=copy.deepcopy(private_tree)
changed={'callback_record','validate_callback','decode_survival','validate_baseline_records'}
added={'gap_parent_record','validate_gap_parent','decode_gap','validate_gap_crosscheck'}
original_funcs={n.name:n for n in old_host.body if isinstance(n,ast.FunctionDef)}
for index,n in enumerate(residual_host.body):
    if isinstance(n,ast.FunctionDef) and n.name in changed:residual_host.body[index]=copy.deepcopy(original_funcs[n.name])
    elif isinstance(n,ast.FunctionDef) and n.name=='validate_binding':n.body[0]=copy.deepcopy(original_funcs[n.name].body[0])
residual_host.body=[n for n in residual_host.body if not isinstance(n,ast.FunctionDef) or n.name not in added]
check(ast.dump(residual_host)==ast.dump(old_host),'private lifecycle capture retention PAX residual exact')
for info in origins['origins'].values():check(hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256'],'private immutable origin rehash '+Path(info['path']).name)
for name,digest in origins['files'].items():check(hashlib.sha256((private_dir/name).read_bytes()).hexdigest()==digest,'private selected leaf rehash '+name)

# Actual setup range/domain refusal statements, no clock fallback.
setup_extension=next(n for n in ast.walk(setup) if isinstance(n,ast.If) and ast.unparse(n.test)=="expected_control == 'COLLECTED_CONTROL'")
range_if=next(n for n in setup_extension.body if isinstance(n,ast.If) and 'gap_setup' in ast.unparse(n.test))
for value in (-1,True,1.5,(1<<48)*1000):
    try:exec(compile(ast.Module(body=[range_if],type_ignores=[]),'actual-setup-range','exec'),dict(gap_setup=value))
    except ValueError:check(True,'actual setup range refusal '+str(value))
    else:raise AssertionError('setup unsupported range accepted')
clock_if=next(n for n in setup_extension.body if isinstance(n,ast.If) and 'gap_clock' in ast.unparse(n.test))
for monotonic,adjustable in ((False,False),(True,True)):
    fake_time=SimpleNamespace(monotonic_ns=time.monotonic_ns,get_clock_info=lambda name:SimpleNamespace(monotonic=monotonic,adjustable=adjustable))
    try:exec(compile(ast.Module(body=[clock_if],type_ignores=[]),'actual-setup-domain','exec'),dict(gap_clock=time.monotonic_ns,time=fake_time,BuiltinFunctionType=type(time.monotonic_ns)))
    except ValueError:check(True,'actual setup domain refusal '+str((monotonic,adjustable)))
    else:raise AssertionError('setup unsupported domain accepted')
if actual is not None:
    for decoder in (e,private_env):
        historical=decoder['decode_survival'](bytes.fromhex(actual[NAMES[0]]['raw']),True,'COLLECTED_CONTROL')
        check(not historical['result_complete'] and historical['state']=='UNKNOWN','actual r131 bytes rejected extended owning decoder')

# Capture simple immutable origins; no lifecycle code executed.
real_atomic(BASE/'coverage.json',dict(groups=5,max_serialized_sizes=max_sizes,guard_residual=True,private_agreement=True))
result=dict(classification='INERT_HOST_ONLY',expect_bug=BUG,actual_pax_consumed=actual is not None,checks=len(labels),labels=labels,controls_sha256=hashlib.sha256(CONTROLS.read_bytes()).hexdigest(),guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest())
real_atomic(BASE/'results.json', {k:v for k,v in result.items() if k!='labels'})
fd=os.open(BASE/'labels.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as stream:json.dump(labels,stream)
print(json.dumps({k:v for k,v in result.items() if k!='labels'}))
