"""Stdlib-only fake external checks. Never calls node/campaign/pytest/product."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

D = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

R = load('b_runner_HOST', D/'runner.py')
F = R.helpers()


def denied(*a, **kw):
    raise AssertionError('HOST external forbidden')


def rejection(call):
    try: call()
    except RuntimeError: return
    raise AssertionError('expected refusal')


def emit(path, data):
    with F.private(path) as out:
        out.write(json.dumps(data).encode())


def fixture(root, *, fail=None, deadline=False, wrong=False, missing=False):
    calls = []
    clock = [0.]
    rows = R.inventory()
    def fake(argv, outputs, end, *, env, observe):
        index = len(calls)
        row = rows[index]
        directory = root/('%03d' % index)
        assert argv == R.node_argv(index, directory)
        assert env['TMPDIR'] == str(directory/'tmp')
        assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        assert env['PYTHONPATH'].split(os.pathsep) == list(map(str, (R.ROOT, R.ROOT/'tests', R.ROOT/'diagnostics/p-one-node')))
        assert end == clock[0]+row['seconds']
        calls.append(row['id'])
        observe()
        for output, raw in zip(outputs, (b'\x00\xffpartial\r\n', b'\x80stderr')):
            with F.private(output) as out: out.write(raw)
        if deadline: clock[0] = end
        if fail == index: return 1, 101+index
        if not missing:
            emit(directory/'selection.json', dict(node='wrong' if wrong else row['node'], count=1, before_fixtures=True))
            emit(directory/'result.json', dict(node=row['node'], pytest_exit=0, collected=True,
                reports=[['setup','passed'],['call','passed'],['teardown','passed']], accepted=True))
        return 0, 101+index
    return fake, calls, lambda: clock[0]


def selection(root):
    row = R.inventory()[0]
    directory = root/'selection'; directory.mkdir(mode=0o700)
    obj = SimpleNamespace(__code__=SimpleNamespace(co_filename=str(R.ROOT/R.CONTROL)))
    item = SimpleNamespace(nodeid=row['node'], path=R.ROOT/R.CONTROL, obj=obj)
    for items in ([], [item,item], [SimpleNamespace(nodeid='wrong')]):
        plugin = R.Selection(row, directory, F)
        rejection(lambda: plugin.pytest_collection_finish(SimpleNamespace(items=items)))
        assert not plugin.collected and not (directory/'selection.json').exists()
    plugin = R.Selection(row, directory, F)
    plugin.pytest_collection_finish(SimpleNamespace(items=[item]))
    assert json.loads((directory/'selection.json').read_bytes())['before_fixtures']
    for when in ('setup','call','teardown'):
        plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=row['node'], when=when, outcome='passed'))
    assert plugin.accepted(0) and not plugin.accepted(5)
    plugin.reports[1][1] = 'skipped'
    assert not plugin.accepted(0)
    plugin.reports = []
    assert not plugin.accepted(0)
    assert R.pytest_argv(row,directory) == ['--noconftest','-p','no:cacheprovider','-q','-x','--tb=short',
        '--rootdir='+str(R.ROOT),'--basetemp='+str(directory/'pytest'),row['node']]


def orchestration(root):
    exhausted = root/'exhausted'; exhausted.mkdir(mode=0o700)
    ticks = iter([0., 2375.])
    rejection(lambda:R.ordered(exhausted,F,capture=denied,clock=lambda:next(ticks)))
    assert not list(exhausted.iterdir())
    residual = root/'residual'; residual.mkdir(mode=0o700)
    fake,calls,clock = fixture(residual)
    rejection(lambda:R.ordered(residual,F,capture=fake,clock=clock,finished=lambda:R.require(False,'residual descendant')))
    assert len(calls) == 1 and not (residual/'001').exists()
    for name, options in [('all100', {}), ('firstred', {'fail':3}),
            ('deadline', {'deadline':True}), ('wrong-selection', {'wrong':True}),
            ('missing-selection', {'missing':True})]:
        directory = root/name; directory.mkdir(mode=0o700)
        fake, calls, clock = fixture(directory, **options)
        try:
            result = R.ordered(directory,F,capture=fake,clock=clock)
        except (RuntimeError, FileNotFoundError):
            assert options
            expected = 4 if name == 'firstred' else 1
            assert len(calls) == expected
            last = directory/('%03d' % (expected-1))
            assert F.slot_read(F.read_regular(last/'receipt.slot', F.SLOT))['state'] == 'STOP'
            assert (last/'stdout').read_bytes() == b'\x00\xffpartial\r\n'
            assert not (directory/('%03d' % expected)).exists()
        else:
            assert not options and result == [r['id'] for r in R.inventory()]
            assert calls == result and len(calls) == len(set(calls)) == 100
            for index in range(100):
                node = directory/('%03d' % index)
                receipt = F.slot_read(F.read_regular(node/'receipt.slot',F.SLOT))
                assert receipt['state'] == 'PASS'
                for stream in receipt['streams']:
                    raw = (node/stream['name']).read_bytes()
                    assert stream['size'] == len(raw) and stream['sha256'] == hashlib.sha256(raw).hexdigest()


def capture(root):
    # Execute unchanged finite capture with fake Popen/selector and real private
    # output leaves. Fake reads are finite; no process, fork, signal or runtime.
    class Selector:
        def __init__(self): self.entries = {}
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def register(self,stream,event,index): self.entries[stream] = SimpleNamespace(fileobj=stream,data=index)
        def unregister(self,stream): del self.entries[stream]
        def get_map(self): return self.entries
        def select(self,timeout): return [(v,1) for v in list(self.entries.values())]
    for mode in ('binary','overflow','timeout'):
        reads = {}
        class Child:
            pid = 123456789
            def __init__(self,argv,**kw):
                assert argv == ['HOST-FAKE'] and kw['cwd'] == '/app' and kw['start_new_session']
                self.stdout = open(os.devnull,'rb'); self.stderr = open(os.devnull,'rb')
                reads[self.stdout.fileno()] = [b'\x00\xff\r\n',b''] if mode != 'overflow' else [b'x'*65536]*(F.META//65536)+[b'z',b'']
                reads[self.stderr.fileno()] = [b'\x80',b'']
            def poll(self): return 0
            def wait(self,timeout): return 0
        outputs = [root/(mode+'.stdout'),root/(mode+'.stderr')]
        read0 = os.read
        def read(fd,size): return reads[fd].pop(0) if fd in reads else read0(fd,size)
        with patch.object(subprocess,'Popen',Child), patch.object(F.selectors,'DefaultSelector',Selector), patch.object(os,'read',read):
            try: rc,pid = F.capture(['HOST-FAKE'],outputs,time.monotonic()+(-1 if mode == 'timeout' else 5))
            except RuntimeError as exc:
                assert mode != 'binary'
                assert ('cap' if mode == 'overflow' else 'timeout') in str(exc)
            else:
                assert mode == 'binary' and rc == 0 and pid == Child.pid
        if mode == 'binary': assert outputs[0].read_bytes() == b'\x00\xff\r\n'
        if mode == 'overflow': assert outputs[0].stat().st_size == F.META
        if mode == 'timeout': assert outputs[0].stat().st_size == 0


def command_binding():
    import copy
    token, invocation = 'a'*48, '00000000-0000-4000-8000-000000000001'
    image, name, volume = 'sha256:'+'b'*64, 'pab-b-host', 'pab-b-volume'
    argv = R.create_argv(name,volume,image,token,invocation)
    assert argv[-6:] == [image]+R.campaign_command(token,invocation)
    assert 'finite-driver.py' not in ' '.join(argv)
    env = [argv[n+1] for n,v in enumerate(argv) if v == '-e']
    obj = dict(Id='c'*64,Name='/'+name,Image=image,
        Config=dict(Labels={'pab.owner':token,'pab.invocation':invocation},
            Cmd=R.campaign_command(token,invocation),User='10001:10001',WorkingDir='/app',Env=env),
        HostConfig=dict(NanoCpus=2000000000,Memory=2*1024**3,PidsLimit=256,ReadonlyRootfs=True,
            NetworkMode='none',CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],
            RestartPolicy={'Name':'no'},LogConfig={'Type':'none'}),
        Mounts=[dict(Type='volume',Name=volume,Destination='/tmp',RW=True)])
    assert R.inspect_campaign(obj,name,volume,image,token,invocation) == obj['Id']
    for section,key,value in [('Config','Cmd',['python','/app/diagnostics/p-one-node/finite-driver.py','writer',token,invocation]),
        ('Config','Entrypoint',['sh']),('Config','Env',env+['PYTHONOPTIMIZE=1']),
        ('Config','Env',env+[env[0]]),('HostConfig','ReadonlyRootfs',False),
        ('HostConfig','Privileged',True),('HostConfig','CapAdd',['SYS_ADMIN']),
        ('HostConfig','Memory',4*1024**3),('HostConfig','NetworkMode','host'),
        ('HostConfig','Tmpfs',{'/tmp':'rw'}),('HostConfig','PidMode','host')]:
        changed = copy.deepcopy(obj); changed[section][key] = value
        rejection(lambda:R.inspect_campaign(changed,name,volume,image,token,invocation))
    changed = copy.deepcopy(obj); changed['Mounts'][0]['Name']='old-volume'
    rejection(lambda:R.inspect_campaign(changed,name,volume,image,token,invocation))


def source():
    manifest = json.loads((R.ROOT/'diagnostics/p-one-node/source-manifest.json').read_bytes())
    assert len(manifest['files']) == 193 and len(manifest['derivative_files']) == 4
    for name, digest in {**manifest['files'],**manifest['derivative_files']}.items():
        assert hashlib.sha256((R.ROOT/name).read_bytes()).hexdigest() == digest, name
    assert manifest['files']['tests/_maintenance_wait_guard.py'] == R.GUARD
    assert manifest['files'][R.CONTROL] == R.CONTROLS
    rows = R.inventory()
    assert [sum(r['id'].startswith(g[1]) for r in rows) for g in R.GROUPS] == [17,39,23,2,2,4,6,7]
    assert sum(r['seconds'] for r in rows)+180 < R.TOTAL
    assert sum(r['id']=='B-S-original74-real-observer' for r in rows) == 1
    assert all('--noconftest' in R.pytest_argv(r,Path('/tmp/x')) for r in rows)
    assert not any(n == 'pytest' or n == 'service' or n.startswith('service.') for n in sys.modules)


def closed_inventory():
    raw = (R.ROOT/R.CONTROL).read_bytes()
    old = raw[:116293]
    assert hashlib.sha256(old).hexdigest() == '1095f7406ec185162c598da8aa7339eade14dc213492bc3b5e6aaf238175d079'
    tree = ast.parse(raw)
    old_tree = ast.parse(old)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == R.SEPARATE_HELPER)
    # Reconstruct original eight-group authority, adding only the separately
    # authenticated helper required by the successor closed inventory.
    old_tree.body.append(copy.deepcopy(helper))
    assert R.inventory_tree(old_tree) == R.inventory() and len(R.inventory()) == 100
    for mode in ('extra', 'reorder', 'missing', 'helper-body', 'helper-duplicate'):
        changed = copy.deepcopy(tree)
        groups = [n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_B_')]
        if mode == 'extra':
            extra = copy.deepcopy(helper); extra.name = 'test_B_unknown'; changed.body.append(extra)
        elif mode == 'reorder':
            a, b = map(changed.body.index, groups[:2]); changed.body[a], changed.body[b] = changed.body[b], changed.body[a]
        elif mode == 'missing': changed.body.remove(groups[0])
        elif mode == 'helper-body': groups[-1].body.append(ast.Pass())
        else: changed.body.append(copy.deepcopy(helper))
        rejection(lambda:R.inventory_tree(changed))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); path = root/R.CONTROL; path.parent.mkdir(parents=True)
        path.write_bytes(raw + b'\n# changed bytes\n')
        rejection(lambda:R.inventory(root))


def main():
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix='pab-B-controls-HOST-',dir='/private/tmp'))
    started = time.monotonic()
    receipt = dict(state='PENDING', evidence='STDLIB-HOST-FAKE-EXTERNAL-ONLY', groups=[], root=str(root))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError('HOST deadline')))
    signal.setitimer(signal.ITIMER_REAL,30)
    try:
        with patch.object(subprocess,'Popen',denied), patch.object(subprocess,'run',denied), \
             patch.object(os,'system',denied), patch.object(os,'fork',denied), patch.object(os,'killpg',denied), patch.object(os,'kill',denied):
            for name, call in [('source',source),('closed-inventory',closed_inventory),('command-binding',command_binding),('selection',lambda:selection(root)),
                    ('ordering-firstred-retention-deadline',lambda:orchestration(root)),('capture',lambda:capture(root))]:
                call(); receipt['groups'].append(name)
        receipt['state'] = 'PASS'
    except BaseException as exc:
        receipt.update(state='RED',failure=type(exc).__name__+': '+str(exc)[:512])
    receipt.update(seconds=time.monotonic()-started,
        pins={str(p.relative_to(R.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (D/'runner.py',D/'HOST.py',R.ROOT/'diagnostics/p-one-node/source-manifest.json')},
        inventory=R.inventory())
    emit(root/'HOST-receipt.json',receipt)
    print(str(root/'HOST-receipt.json'))
    print(receipt['state'], receipt.get('failure',''))
    return 0 if receipt['state']=='PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
