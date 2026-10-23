import shutil
"""Frozen <=12 cases, <=9s each, <=120s HOST; exact fake externals only."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

D = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, D / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

Q = load('qualification', 'qualify.py')
W = load('dummy', 'dummy-writer.py')
H = load('reviewed_host', 'host-check.py')
L, F = Q.L, Q.F
O, IMAGE, CID, VOL = H.OWNER, H.IMAGE, H.CID, H.VOL
CASES = []
ROOT = None
PLAN = None
OBS = []


def case(fn):
    CASES.append(fn)
    return fn


def fresh(name):
    p = ROOT / name
    p.mkdir(mode=0o700)
    return p


def root(name):
    r = fresh(name)
    for n in ('intent.slot', 'final.slot'):
        F.slot_create(r / n)
    return r


def check(ok, message):
    F.require(ok, message)


def reject(fn):
    H.expect(fn)


@case
def dummy_real_io():
    r = fresh('dummy-real')
    state = W.body(r, O['token'], O['invocation'])
    check(state['bytes_written'] == 9269 and state['bytes_written'] < W.CAP, 'dummy counter')
    check((r / 'dummy.bin').read_bytes() == W.BINARY, 'binary NUL/FF/80')
    check(F.slot_read((r / 'pab-terminal.slot').read_bytes()) == state, 'terminal parse')
    p = W.Proof(fresh('dummy-counter'))
    p.written = W.CAP
    reject(lambda: p.write('refused', b'x'))
    check(not (p.root / 'refused').exists(), 'counter before write')
    pending = fresh('dummy-pending')
    with patch.object(W.Proof, 'write', autospec=True, side_effect=RuntimeError('injected')):
        reject(lambda: W.body(pending, O['token'], O['invocation']))
    check(not list(pending.iterdir()), 'failure fabricated no terminal')


@case
def cli_identity_refusal():
    with patch.object(sys, 'argv', ['dummy-writer.py', O['token'], O['invocation']]):
        reject(W.main)
    with patch.object(sys, 'argv', ['qualify.py']):
        reject(Q.main)
    check(os.getpid() != 1, 'HOST identity')


@case
def actual_initialization():
    r = fresh('initialized')
    with F.private(r / 'pab-seed') as f:
        f.write(F.SEED)
    observed = F.initialized(r, uid=os.getuid(), gid=r.stat().st_gid)
    check(observed['logical'] == len(F.SEED), 'seed logical')
    r.chmod(0o755)
    reject(lambda: F.initialized(r, uid=os.getuid(), gid=r.stat().st_gid))
    r.chmod(0o700)
    # HOST model of inherited ordinary/hidden junk and symlink replacement.
    for kind in ('ordinary', 'hidden', 'symlink'):
        tree = fresh('inherited-' + kind)
        if kind == 'symlink':
            tree.rmdir()
            tree.symlink_to(r, target_is_directory=True)
        else:
            (tree / ('junk' if kind == 'ordinary' else '.junk')).write_bytes(b'old')
        if tree.is_symlink():
            tree.unlink()
        else:
            shutil.rmtree(tree)
        tree.mkdir(mode=0o700)
        with F.private(tree / 'pab-seed') as f: f.write(F.SEED)
        F.initialized(tree, uid=os.getuid(), gid=tree.stat().st_gid)
        (tree / '.residual').write_bytes(b'no')
        reject(lambda: F.initialized(tree, uid=os.getuid(), gid=tree.stat().st_gid))


@case
def actual_context():
    global PLAN
    r = root('context')
    b = L.Budget()
    PLAN = Q.prepare(r, O, b)
    Q.validate_plan(r, PLAN)
    altered = json.loads(json.dumps(PLAN))
    altered['build'].insert(2, '--network=host')
    reject(lambda: Q.validate_plan(r, altered))
    manifest = json.loads((r / 'context/source/manifest.json').read_bytes())
    original = json.loads((D / 'source-manifest.json').read_bytes())
    check(len(original['files']) == 193 and manifest['files'] == original['files'], '193 source mapping')
    for name, digest in {**manifest['files'], **manifest['derivative_files']}.items():
        check(L.stream_hash(r / 'context/source' / name) == digest, 'copied input hash')
    recipe = (r / 'context/Dockerfile').read_text()
    check(recipe.startswith('FROM ' + L.base_reference(O) + '\n'), 'owned FROM')
    for change in ('raw', 'tag', 'plan'):
        altered = json.loads(json.dumps(PLAN))
        if change == 'plan':
            altered['prebuild']['expected_id'] = IMAGE
        else:
            (r / 'context/Dockerfile').write_text(recipe.replace(L.base_reference(O), L.BASE if change == 'raw' else 'foreign:tag'))
        reject(lambda: Q.validate_plan(r, altered))
        (r / 'context/Dockerfile').write_text(recipe)
    prior_root = H.ROOT
    H.ROOT = fresh('product-host29')
    try:
        next(fn for name, fn in H.CASES if name.startswith('29-'))()
    finally:
        H.ROOT = prior_root
    product = fresh('product-context')
    c = L.prepare_context(D.parents[1], product, L.Budget(), O)
    proposal = json.loads((product / 'build-proposal.json').read_bytes())
    L.validate_recipe(c, O, proposal['prebuild'])
    (c / 'Dockerfile').write_text(L.base_recipe(O).replace(L.base_reference(O), L.BASE))
    reject(lambda: L.validate_recipe(c, O, proposal['prebuild']))
    check((c / 'Dockerfile').read_text().replace(L.BASE, L.base_reference(O)) == recipe, 'both shared recipes')
    check(recipe.index('RUN rm -rf -- /tmp') < recipe.index('seed /tmp/pab-seed'), 'clean before seed')
    check('bytes.fromhex' in recipe and '--chmod=0600' in recipe, 'exact seed build assertion')
    check('COPY --chown=10001:10001' in recipe and 'USER 10001:10001' in recipe, 'ownership recipe')
    check(PLAN['profile'] == Q.PROFILE and PLAN['build'][2:4] == ['--pull=false', '--network=none'], 'frozen plan')
    OBS.append(dict(source_files=193, derivative_files=len(manifest['derivative_files']), context_charged=b.used))


@case
def no_product_dispatch():
    a = Q.dummy_create('name', VOL, IMAGE, O)
    check(a[-4:] == Q.dummy_command(O), 'dummy argv')
    check('entry.py' not in ' '.join(a) and '/finite-driver.py writer' not in ' '.join(a), 'no product')
    r = root('no-dispatch')
    fake = H.Fake([])
    obj = Q.Qualification(r, r / 'lane', O, fake)
    reject(lambda: obj.dispatch('forbidden', ['python', '/app/diagnostics/p-one-node/finite-driver.py', 'writer']))
    fake.done()
    tree = ast.parse((D / 'dummy-writer.py').read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    check(not any(n and ('sqlite' in n or 'audio' in n) for n in imports), 'no product imports')


def container(mode, status):
    c = H.container(mode, status)
    if mode == 'writer':
        c['Config']['Cmd'] = Q.dummy_command(O)
    return c


@case
def binding_refusals():
    name = 'pab-p-one-' + O['invocation'] + '-writer'
    good = container('writer', 'created')
    check(Q.dummy_binding(good, O, name, VOL, IMAGE) == CID['writer'], 'full ID')
    for path, value in [(('Id',), '1' * 12), (('Config', 'Cmd'), L.command('writer', O)),
                        (('HostConfig', 'Privileged'), True), (('HostConfig', 'PidMode'), 'host')]:
        c = json.loads(json.dumps(good))
        target = c
        for part in path[:-1]: target = target[part]
        target[path[-1]] = value
        reject(lambda: Q.dummy_binding(c, O, name, VOL, IMAGE))
    v = H.volume()
    v['Options'] = {'type': 'nfs'}
    reject(lambda: L.volume_binding(v, O, VOL))


def fixture_files():
    r = fresh('fixture-' + str(len(list(ROOT.iterdir()))))
    state = W.body(r, O['token'], O['invocation'])
    # Explicit synthetic remote identity; never described as native PID1 execution.
    state['identity'].update(pid=1, uid=10001, gid=10001)
    (r / 'pab-terminal.slot').write_bytes(W.payload(state))
    return [('pab-seed', F.SEED)] + [(p.name, p.read_bytes()) for p in r.iterdir()]


def script(r, failure=None):
    files = fixture_files()
    archive = H.tar_bytes(files)
    mapraw = b''.join(L.encode(dict(name=n, sha256=hashlib.sha256(b).hexdigest())) + b'\n' for n, b in files)
    seq = []
    def add(argv, raw=b''): seq.append((argv, dict(raw=raw)))
    def obj(argv, value): add(argv, L.encode([value]))
    obj(['docker', 'image', 'inspect', L.BASE], dict(Id=L.BASE, Os='linux', Architecture='amd64', Config={}))
    plan = Q.prepare(r, O, L.Budget())
    steps = plan['prebuild']
    add(steps['absence'])
    add(steps['tag'])
    for _ in range(2):
        obj(steps['verify'], dict(Id=L.BASE, Os='linux', Architecture='amd64', Config={}))
    plan['build'][-2:] = [str(r / 'image.id'), str(r / 'context')]
    add(plan['build'])
    obj(['docker', 'image', 'inspect', IMAGE], dict(Id=IMAGE, Os='linux', Architecture='amd64',
        Config=dict(User='10001:10001', Labels={'pab.owner': O['token'], 'pab.invocation': O['invocation']})))
    add(['docker', 'volume', 'ls', '--format', '{{.Name}}', '--filter', 'name=^' + VOL + '$'])
    add(['docker', 'volume', 'create', '--driver', 'local', '--label', 'pab.owner=' + O['token'],
         '--label', 'pab.invocation=' + O['invocation'], VOL], VOL.encode())
    obj(['docker', 'volume', 'inspect', VOL], H.volume())
    for mode in ('writer', 'archive', 'hash'):
        name = 'pab-p-one-' + O['invocation'] + '-' + mode
        add(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'])
        add(Q.dummy_create(name, VOL, IMAGE, O) if mode == 'writer' else L.create_argv(name, VOL, IMAGE, mode, O), CID[mode].encode())
        obj(['docker', 'inspect', CID[mode]], container(mode, 'created'))
        if mode == 'writer':
            add(['docker', 'start', '--attach', CID[mode]], b'')
            add(['docker', 'wait', CID[mode]], b'0\n')
        else:
            add(['docker', 'start', '--attach', CID[mode]], archive if mode == 'archive' else mapraw)
        if mode != 'writer':
            add(['docker', 'wait', CID[mode]], b'0\n')
        obj(['docker', 'inspect', CID[mode]], container(mode, 'exited'))
    for mode in ('writer', 'archive', 'hash'):
        obj(['docker', 'inspect', CID[mode]], container(mode, 'exited'))
        add(['docker', 'rm', CID[mode]])
        add(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id=' + CID[mode]])
    if failure:
        target = ['docker', 'start', '--attach', CID['archive']] if failure == 'export' else ['docker', 'start', '--attach', CID['writer']]
        index = next(i for i, (a, _) in enumerate(seq) if a == target)
        seq = seq[:index] + [(target, OSError('FAKE lost acknowledgement'))]

    return seq, plan


class Fake(H.Fake):
    def __init__(self, seq, r):
        super().__init__(seq)
        self.r = r
    def spawn(self, argv, **kw):
        p = super().spawn(argv, **kw)
        if argv[:2] == ['docker', 'build']:
            with F.private(self.r / 'image.id') as f: f.write(IMAGE.encode())
        return p


def full(name, failure=None):
    r = root(name)
    seq, plan = script(r, failure)
    fake = Fake(seq, r)
    q = Q.Qualification(r, r / 'lane', O, fake)
    q.plan = plan
    ok = q.run()
    fake.done()
    return r, q, fake, ok


@case
def complete_success():
    r, q, fake, ok = full('success')
    check(ok and q.retained and not q.lane.exists(), 'complete retention/release')
    check(all(v['state'] == 'TERMINAL' for v in q.operations.values()), 'journal terminal')
    check(q.objects['base_tag']['retained'] and q.objects['base_tag']['expected_id'] == L.BASE, 'tag retained')
    check(not any(a[:3] == ['docker', 'volume', 'rm'] for a in fake.calls), 'volume retained')
    check(len([a for a in fake.calls if a[:2] == ['docker', 'build']]) == 1, 'one build')
    check(json.loads((r / 'detached-wait.json').read_bytes())['local_wait_cannot_extend_remote_deadline'], 'detached receipt')
    final = F.slot_read((r / 'final.slot').read_bytes())
    check(final['acceptance'] is False, 'no product acceptance')
    OBS.append(dict(success_root=str(r), full_ids=[q.objects[m]['id'] for m in CID], retained_volume=q.volume))


@case
def stopped_export_failure():
    r, q, fake, ok = full('export-failure', 'export')
    check(not ok and not q.retained and q.lane.exists(), 'export HOLD')
    check(q.objects['writer']['state'] == 'STOPPED', 'writer stopped before export')
    check('hash' not in q.objects and not any(a[:2] == ['docker', 'rm'] for a in fake.calls), 'no second reader/cleanup')
    check(F.slot_read((r / 'final.slot').read_bytes())['state'] == 'STOP-OR-UNKNOWN-HOLD', 'honest failure')


@case
def lost_start_ack():
    r, q, fake, ok = full('lost-start', 'start')
    check(not ok and q.operations['writer-start']['state'] == 'SENT-UNKNOWN', 'lost ack preserved')
    check(len([a for a in fake.calls if a == ['docker', 'start', '--attach', CID['writer']]]) == 1, 'no retry')
    check('archive' not in q.objects and q.lane.exists(), 'no readers on uncertainty')


@case
def real_readers_and_parsers():
    r = fresh('readers')
    W.body(r, O['token'], O['invocation'])
    before = {p.name: p.read_bytes() for p in r.iterdir()}
    for mode in ('archive', 'hash'):
        sink = io.BytesIO()
        F.reader(r, mode, sink, uid=os.getuid(), gid=r.stat().st_gid)
        check(sink.getvalue(), 'real reader output')
    check(before == {p.name: p.read_bytes() for p in r.iterdir()}, 'readers preserve bytes')
    files = fixture_files()
    tar = ROOT / 'good.tar'
    with F.private(tar) as f: f.write(H.tar_bytes(files))
    audit = L.parse_archive(tar, ROOT / 'inventory.jsonl', L.Budget())
    check(set(audit['hashes']) == {n for n, _ in files}, 'all binary hashes')
    bad = ROOT / 'bad.tar'
    with F.private(bad) as f: f.write(H.tar_bytes(files + [files[0]]))
    reject(lambda: L.parse_archive(bad, ROOT / 'bad-inventory', L.Budget()))
    mapping = ROOT / 'map.jsonl'
    with F.private(mapping) as f:
        for n, b in files: f.write(L.encode(dict(name=n, sha256=hashlib.sha256(b).hexdigest())) + b'\n')
    check(L.parse_map(mapping) == audit['hashes'], 'independent map equality')


@case
def private_capture_and_failure():
    r = fresh('capture')
    fake = H.Fake([(['fixed'], dict(raw=b'\0\xff\x80'))])
    raw = L.call(r, 'binary', ['fixed'], L.Budget(), fake, deadline=10)
    fake.done()
    check(raw == b'\0\xff\x80', 'binary private capture')
    fake = H.Fake([(['overflow'], dict(raw=b'ab'))])
    reject(lambda: L.call(r, 'overflow', ['overflow'], L.Budget(), fake, deadline=10, cap=1))
    fake.done()
    check(json.loads((r / 'overflow.complete.json').read_bytes())['state'] == 'UNKNOWN', 'capture failure UNKNOWN')


@case
def independent_deadline_source():
    source = (D / 'dummy-writer.py').read_text()
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    check(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'setitimer'
              and len(n.args) == 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == 30 for n in calls), 'independent30 alarm')
    check(not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('Popen', 'fork', 'execv', 'system') for n in calls), 'no dummy descendants')
    check('signal.signal(signal.SIGALRM, lambda *_: os._exit(124))' in source, 'hard exit independent of host')
    r = fresh('supervisor')
    class Worker:
        pid = 888888888
        def __init__(self, target): self.status = None
        def poll(self): return self.status
        def terminate(self, sig): self.status = -int(sig)
    tick = [0]
    def now():
        tick[0] += 1
        if tick[0] == 3: tick[0] = 572
        return tick[0]
    check(not Q.supervise(r, lambda: True, factory=Worker, clock=now, sleep=lambda _: None,
                          free=lambda _: 20 * F.GIB), 'supervisor deadline refusal')
    check(json.loads((r / 'supervisor-complete.json').read_bytes())['state'] == 'UNKNOWN', 'supervisor unknown')
    OBS.append(dict(detachment='SOURCE alarm + fake detached-start/wait routing; no actual timing claim'))


@case
def recipe_adapter_failures():
    for failure in ('collision', 'wrong-id', 'startup', 'lost-tag', 'fatal', 'exit1', 'truncated', 'timeout', 'reverify-id', 'reverify-startup', 'recipe-tamper'):
        r = root('repair-' + failure)
        plan = Q.prepare(r, O, L.Budget())
        steps = plan['prebuild']
        base = dict(Id=L.BASE, Os='linux', Architecture='amd64', Config={})
        seq = [(steps['inspect'], dict(raw=L.encode([base]))),
               (steps['absence'], dict(raw=b'collision' if failure == 'collision' else b''))]
        if failure != 'collision':
            seq.append((steps['tag'], OSError('lost tag ack') if failure == 'lost-tag' else dict()))
        if failure not in ('collision', 'lost-tag'):
            bad = json.loads(json.dumps(base))
            if failure == 'wrong-id': bad['Id'] = IMAGE
            if failure == 'startup': bad['Config']['Entrypoint'] = ['unsafe']
            seq.append((steps['verify'], dict(raw=L.encode([bad]))))
        if failure in ('fatal', 'exit1', 'truncated', 'timeout', 'reverify-id', 'reverify-startup'):
            final_base = json.loads(json.dumps(base))
            if failure == 'reverify-id': final_base['Id'] = IMAGE
            if failure == 'reverify-startup': final_base['Config']['Healthcheck'] = {'Test': ['unsafe']}
            seq.append((steps['verify'], dict(raw=L.encode([final_base]))))
        if failure in ('fatal', 'exit1', 'truncated', 'timeout'):
            terminal = b'ERROR: failed to build: failed to solve: source metadata resolution failed\n'
            response = dict(rc=1, err=terminal if failure == 'fatal' else
                            terminal[:-1] if failure == 'truncated' else b'connection lost\n')
            if failure == 'timeout': response = dict(wait_failure=True, rc=1)
            seq.append((plan['build'], response))
            if failure != 'fatal':
                seq.append((['docker', 'image', 'ls', '--no-trunc', '--filter',
                             'label=pab.owner=' + O['token'], '--filter',
                             'label=pab.invocation=' + O['invocation']], dict()))
        class RepairFake(H.Fake):
            def spawn(self, argv, **kw):
                proc = super().spawn(argv, **kw)
                if failure == 'recipe-tamper' and argv == steps['verify']:
                    (r / 'context/Dockerfile').write_text(L.base_recipe(O).replace(L.base_reference(O), L.BASE))
                return proc
        fake = RepairFake(seq)
        q = Q.Qualification(r, r / 'lane', O, fake)
        q.plan = plan
        check(not q.run(), 'failure admitted')
        fake.done()
        check(not any(a[:2] in (['docker', 'volume'], ['docker', 'create'], ['docker', 'start'])
                      for a in fake.calls), 'downstream after failure')
        check(sum(a[:2] == ['docker', 'build'] for a in fake.calls) <= 1, 'retry')
        if failure == 'lost-tag': check(q.operations['base-tag']['state'] == 'SENT-UNKNOWN', 'lost tag unknown')
        if failure in ('fatal', 'exit1', 'truncated', 'timeout'):
            check(q.operations['build']['state'] == ('TERMINAL' if failure == 'fatal' else 'SENT-UNKNOWN'),
                  'terminal versus unknown')
        OBS.append(dict(failure=failure, calls=fake.calls, operations=q.operations))


@case
def affected_product_transport():
    for name, fn in H.CASES:
        if name[:3] in ('04-', '05-', '06-', '07-', '09-', '23-'):
            fn()


@case
def lifecycle_failures():
    for product in (False, True):
        for failure in ('attach', 'status', 'wait-cli', 'inspect', 'timeout', 'overflow', 'lost', 'deadline', 'inspect-deadline', 'oom', 'daemon-error', 'wrong-id'):
            r = root(('product-' if product else 'dummy-') + failure)
            seq, plan = script(r)
            if product:
                seq = H.lifecycle_script()
            target = ['docker', 'start', '--attach', CID['writer']]
            i = next(i for i, (a, _) in enumerate(seq) if a == target)
            seq = seq[i-3:i+4]  # absence, create, binding, attach, wait, inspect, next absence
            seq = seq[:6]
            if failure == 'attach':
                seq[3][1].update(raw=b'out\x00', err=b'trace\xff', rc=7)
            if failure == 'status':
                seq[4][1]['raw'] = b'1\n'
                obj = json.loads(seq[5][1]['raw'])[0]
                obj['State']['ExitCode'] = 1
                seq[5][1]['raw'] = L.encode([obj])
            if failure in ('oom', 'daemon-error'):
                obj = json.loads(seq[5][1]['raw'])[0]
                obj['State']['OOMKilled' if failure == 'oom' else 'Error'] = True if failure == 'oom' else 'daemon failure'
                seq[5][1]['raw'] = L.encode([obj])
            if failure == 'wrong-id':
                obj = json.loads(seq[5][1]['raw'])[0]
                obj['Id'] = 'f' * 64
                seq[5][1]['raw'] = L.encode([obj])
            if failure == 'wait-cli': seq[4][1].update(raw=b'wait-prefix', err=b'wait-error', rc=9)
            if failure == 'inspect': seq[5][1]['rc'] = 1
            if failure in ('timeout', 'overflow', 'lost'):
                seq = seq[:4]
                if failure == 'lost': seq[3] = (target, OSError('lost'))
                elif failure == 'timeout': seq[3][1].update(raw=b'prefix', wait_failure=True)
                else: seq[3][1].update(raw=b'x' * (F.META + 1))
            if failure == 'deadline': seq = seq[:3]
            if failure == 'inspect-deadline': seq = seq[:5]
            fake = Fake(seq, r)
            q = (L.Lifecycle(r, r / 'lane', IMAGE, O, fake) if product else Q.Qualification(r, r / 'lane', O, fake))
            q.image = IMAGE
            if failure in ('deadline', 'inspect-deadline'):
                original = q.dispatch
                def dispatch(label, argv, **kw):
                    result = original(label, argv, **kw)
                    if label == ('writer-binding' if failure == 'deadline' else 'writer-wait'):
                        fake.t = (389 if not product else 1469) if failure == 'deadline' else (1000 if not product else 2000)
                    return result
                q.dispatch = dispatch
            reject(lambda: q.run_container('writer', 1470) if product else q.dummy())
            fake.done()
            check(not any(a[:2] == ['docker', 'rm'] or 'archive' in str(a) for a in fake.calls), 'no downstream')
            if failure in ('attach', 'status', 'wait-cli', 'oom', 'daemon-error'):
                check(q.objects['writer']['state'] == 'STOPPED', 'stop recorded before rejection')
                check(fake.calls[-1] == ['docker', 'inspect', CID['writer']], 'reserved inspect last')
            else:
                check(q.objects['writer']['state'] == 'CREATED', 'no fabricated stop')
            if failure == 'attach':
                receipt = json.loads((r / 'writer-start.complete.json').read_bytes())
                check(receipt['sha256'] == [hashlib.sha256(b'out\x00').hexdigest(), hashlib.sha256(b'trace\xff').hexdigest()], 'stream hashes')
                check((r / 'writer-start.stdout').read_bytes() == b'out\x00' and
                      (r / 'writer-start.stderr').read_bytes() == b'trace\xff', 'exact nonzero streams')
            if failure in ('timeout', 'overflow', 'lost'):
                receipt = json.loads((r / 'writer-start.complete.json').read_bytes())
                check(receipt['state'] == 'UNKNOWN' and not receipt['remote_completion'], 'honest unknown')
                check(len(fake.calls) == 4, 'no unknown reconciliation/retry')
                if failure != 'lost': check((r / 'writer-start.stdout').stat().st_size > 0, 'prefix retained')


def main():
    global ROOT
    F.enabled()
    os.umask(0o077)
    check(len(CASES) == 15, 'frozen matrix13')
    selected = CASES
    if sys.argv[1:]:
        check(sys.argv[1:] in (['--recipe-repair'], ['--r26']), 'fixed HOST selection')
        selected = ([actual_initialization, actual_context, lifecycle_failures, affected_product_transport, complete_success, lost_start_ack, private_capture_and_failure]
                    if sys.argv[1:] == ['--r26'] else [actual_context, recipe_adapter_failures, complete_success])
    ROOT = Path(tempfile.mkdtemp(prefix='pab-dummy-HOST-', dir='/private/tmp'))
    ROOT.chmod(0o700)
    H.ROOT = ROOT
    hashes = {n: L.stream_hash(D / n) for n in (*Q.PINS, 'dummy-writer.py', 'qualify.py', 'qualification-host-check.py')}
    start = time.monotonic()
    results = []
    def forbidden(*a, **kw): raise RuntimeError('HOST forbidden real external')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('HOST9s')))
    with patch.object(subprocess, 'Popen', forbidden), patch.object(L.Real, 'spawn', forbidden), \
         patch.object(H.L.Real, 'spawn', forbidden), patch.object(os, 'system', forbidden), \
         patch.object(os, 'killpg', forbidden), patch.object(os, 'fork', forbidden):
        for fn in selected:
            check(time.monotonic() - start < 120, 'HOST total120')
            before = time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_REAL, 9)
                fn()
                results.append(dict(case=fn.__name__, result='PASS', seconds=time.monotonic() - before))
            except BaseException as exc:
                import traceback
                results.append(dict(case=fn.__name__, result='RED', error=repr(exc), traceback=traceback.format_exc()))
                break  # fixed first unexpected failure boundary
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for fake in H.FAKES:
                    if not fake.finished: fake.done(remaining=True)
    check(hashes == {n: L.stream_hash(D / n) for n in hashes}, 'freeze changed')
    for p in ROOT.rglob('*'):
        check(stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600), 'private evidence')
    receipt = dict(tier='HOST ONLY / FAKE EXTERNALS / NO PRODUCT CLAIM', hashes=hashes,
                   maximum_cases=15, maximum_case_seconds=9, maximum_seconds=120,
                   seconds=time.monotonic() - start, cases=results, observations=OBS,
                   fd_observations=H.OBSERVATIONS, actual_Docker_product_SQLite_pytest_network_Git_mutations=0,
                   result='PASS' if len(results) == len(selected) and all(r['result'] == 'PASS' for r in results) else 'RED')
    L.publish(ROOT / 'HOST-receipt.json', receipt, L.Budget())
    print(json.dumps(dict(root=str(ROOT), result=receipt['result'], cases=len(results))))
    return 0 if receipt['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
