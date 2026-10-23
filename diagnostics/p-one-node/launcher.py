"""BLOCKEDPIPE r16 host lifecycle. Execution requires fresh independent review.

No build is dispatched here. prepare_context writes the reviewable build proposal;
the actual image must be separately qualified and supplied by immutable image ID.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import uuid

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('p_one_finite', D / 'finite-driver.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
BASE = 'sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb'
SOURCE = 'e941c7aee7f53348905659904b8299d5500d1250'
DOCUMENTATION = 'f93b7fef6cc461b756cd9e13ff14e88e243359a4'
LANE = Path('/private/tmp/pab-worker-w1-docker-lane.lock')
HOST = 4 * F.GIB
SUPERVISOR_RESERVE = F.META
SCHEDULE = (120, 60, 1200, 30, 60, 180, 120, 30)
ID = re.compile('[0-9a-f]{64}')
SEED = F.SEED


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class Budget:
    def __init__(self, cap=HOST):
        self.cap, self.used = cap, 0
    def charge(self, count):
        F.require(0 <= count <= self.cap - self.used, 'STOP host logical write budget')
        self.used += count


def publish(path, value, budget):
    raw = encode(value) + b'\n'
    F.require(len(raw) <= F.META, 'STOP metadata cap')
    budget.charge(len(raw))
    with F.private(path) as f:
        F.require(f.write(raw) == len(raw), 'STOP short publication')
        os.fsync(f.fileno())
    F.require(F.read_regular(path, F.META, 0o600) == raw, 'STOP publication readback')


@contextlib.contextmanager
def checkout_umask():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


class Real:
    """Only external boundary; HOST supplies a deny-by-default substitute."""
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    spawn = staticmethod(subprocess.Popen)
    killpg = staticmethod(os.killpg)
    free = staticmethod(lambda path: shutil.disk_usage(path).free)


def call(root, label, argv, budget, adapter, *, deadline, cap=F.META, cwd=None, completed_nonzero=False):
    """Streaming binary capture; no leader-liveness shortcut on group failure."""
    F.require(re.fullmatch('[a-z0-9-]+', label) is not None, 'STOP call label')
    F.require(adapter.clock() < deadline, 'STOP command budget')
    publish(root / (label + '.pending.json'), {'state': 'PENDING', 'argv': argv}, budget)
    proc = None
    selector = None
    streams = []
    captures = []
    counts = [0, 0]
    digests = [hashlib.sha256(), hashlib.sha256()]
    result = {'state': 'PENDING', 'argv': argv, 'exit': None, 'group_kill_attempted': False,
              'leader_reaped': False, 'remote_completion': False}
    error = None
    try:
        for suffix in ('.stdout', '.stderr'):
            captures.append(F.private(root / (label + suffix)))
        proc = adapter.spawn(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        streams = [proc.stdout, proc.stderr]
        publish(root / (label + '.spawn.json'), {'state': 'SPAWNED', 'local_pid': proc.pid,
                'owned_session': proc.pid, 'remote_completion': False}, budget)
        selector = selectors.DefaultSelector()
        for index, stream in enumerate(streams):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, index)
        while selector.get_map() or proc.poll() is None:
            F.require(adapter.clock() < deadline, 'STOP CLI timeout')
            F.require(adapter.free(root) >= 2 * F.GIB, 'STOP host pressure')
            for key, _ in selector.select(min(0.25, max(0, deadline - adapter.clock()))):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                index = key.data
                limit = cap if index == 0 else F.META
                prefix = chunk[:max(0, limit - counts[index])]
                budget.charge(len(prefix))
                F.require(captures[index].write(prefix) == len(prefix), 'STOP short capture')
                digests[index].update(prefix)
                counts[index] += len(prefix)
                F.require(len(prefix) == len(chunk), 'STOP capture overflow')
        result['exit'] = proc.wait(timeout=min(1, max(0.001, deadline - adapter.clock())))
        result['leader_reaped'] = True
        result['state'] = 'COMPLETE'
    except BaseException as exc:
        error = exc
        result['failure'] = type(exc).__name__
        if proc is not None:
            result['group_kill_attempted'] = True
            try:
                adapter.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as kill_error:
                result['kill_failure'] = type(kill_error).__name__
            try:
                result['exit'] = proc.wait(timeout=1)
                result['leader_reaped'] = True
            except BaseException:
                result['leader_reaped'] = False
        result['state'] = 'UNKNOWN'
    finally:
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                error = error or exc
                result['state'] = 'UNKNOWN'
                result['close_failure'] = type(exc).__name__
        for stream in streams + captures:
            try:
                stream.close()
            except BaseException as exc:
                error = error or exc
                result['state'] = 'UNKNOWN'
                result['close_failure'] = type(exc).__name__
        result['bytes'] = counts
        result['sha256'] = [h.hexdigest() for h in digests]
        result['descriptors_closed'] = all(s.closed for s in streams + captures)
        try:
            publish(root / (label + '.complete.json'), result, budget)
        except BaseException:
            # Pending is never interpreted as success.
            raise
    if error is not None:
        raise RuntimeError('STOP command incomplete: ' + label) from error
    F.require(completed_nonzero or result['exit'] == 0, 'STOP command nonzero: ' + label)
    return F.read_regular(root / (label + '.stdout'), min(cap, F.META)) if cap <= F.META else None


def startup(config):
    F.require(not any(config.get(k) for k in ('Entrypoint', 'OnBuild', 'Healthcheck', 'Volumes')),
              'STOP inherited startup config')
    env = config.get('Env') or []
    allowed = {'PATH', 'LANG', 'PYTHON_VERSION', 'PYTHON_SHA256', 'PYTHON_PIP_VERSION',
               'PYTHON_SETUPTOOLS_VERSION', 'PYTHON_GET_PIP_URL', 'PYTHON_GET_PIP_SHA256',
               'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH', 'PAB_MAINTENANCE_WAIT_LOG', 'TIKTOKEN_CACHE_DIR'}
    F.require(all(x.split('=', 1)[0] in allowed for x in env), 'STOP startup environment')
    F.require(not any(x.startswith(('PYTHONOPTIMIZE=', 'PYTHONHOME=', 'PYTHONSTARTUP=')) for x in env),
              'STOP startup override')


def volume_binding(obj, owner, name):
    F.require(obj['Name'] == name and obj['Driver'] == 'local' and not obj.get('Options')
              and obj.get('Scope') == 'local' and bool(obj.get('CreatedAt')),
              'STOP volume configuration')
    F.require(obj.get('Labels') == {'pab.owner': owner['token'], 'pab.invocation': owner['invocation']},
              'STOP volume collision')
    return {k: obj.get(k) for k in ('Name', 'Driver', 'Options', 'Scope', 'CreatedAt', 'Labels', 'Mountpoint')}


def command(mode, owner):
    return ['python', '/app/diagnostics/p-one-node/finite-driver.py', mode,
            owner['token'], owner['invocation']]


def create_argv(name, volume, image, mode, owner):
    readonly = mode != 'writer'
    return ['docker', 'create', '--name', name, '--label', 'pab.owner=' + owner['token'],
            '--label', 'pab.invocation=' + owner['invocation'], '--cpus', '2', '--memory', '2g',
            '--pids-limit', '256', '--network', 'none', '--read-only', '--user', '10001:10001',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--restart', 'no',
            '--log-driver', 'none', '--workdir', '/app',
            '--mount', 'type=volume,src=' + volume + ',dst=/tmp' + (',readonly' if readonly else ''),
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=/app/diagnostics/p-one-node',
            '-e', 'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl', '-e', 'TIKTOKEN_CACHE_DIR=/tokenizer-cache',
            image] + command(mode, owner)


def container_binding(obj, owner, name, volume, image, mode):
    cid = obj['Id']
    F.require(ID.fullmatch(cid) is not None and obj['Name'] == '/' + name and obj['Image'] == image,
              'STOP container identity')
    cfg, hc = obj['Config'], obj['HostConfig']
    startup(cfg)
    F.require(cfg['Labels'] == {'pab.owner': owner['token'], 'pab.invocation': owner['invocation']}
              and cfg['User'] == '10001:10001' and cfg['Cmd'] == command(mode, owner)
              and cfg['WorkingDir'] == '/app', 'STOP container command binding')
    expected_env = {'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/app/diagnostics/p-one-node',
                    'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl', 'TIKTOKEN_CACHE_DIR=/tokenizer-cache'}
    F.require(expected_env.issubset(set(cfg['Env'])), 'STOP container environment')
    F.require(hc['NanoCpus'] == 2000000000 and hc['Memory'] == 2 * F.GIB and hc['PidsLimit'] == 256
              and hc['ReadonlyRootfs'] and hc['NetworkMode'] == 'none' and hc['CapDrop'] == ['ALL']
              and 'no-new-privileges' in hc['SecurityOpt'] and not hc.get('CapAdd')
              and not hc.get('Privileged') and not hc.get('AutoRemove')
              and hc['RestartPolicy']['Name'] == 'no' and hc['LogConfig']['Type'] == 'none',
              'STOP resource configuration')
    F.require(not any(hc.get(k) for k in ('Binds', 'Devices', 'PortBindings', 'Tmpfs', 'VolumesFrom'))
              and not cfg.get('ExposedPorts') and hc.get('PidMode', '') == '', 'STOP extra exposure')
    mounts = obj['Mounts']
    F.require(len(mounts) == 1 and mounts[0]['Type'] == 'volume' and mounts[0]['Name'] == volume
              and mounts[0]['Destination'] == '/tmp' and mounts[0]['RW'] == (mode == 'writer'),
              'STOP mount binding')
    return cid


def stopped(obj):
    s = obj['State']
    F.require(s['Status'] == 'exited' and not any(s.get(k) for k in
              ('Running', 'Restarting', 'Paused', 'Dead')) and s['Pid'] == 0,
              'UNKNOWN remote workload unresolved')


def parse_archive(path, output, budget, *, clock=time.monotonic, deadline=None):
    F.require(path.stat().st_size <= F.EXPORT, 'STOP archive stream cap')
    hashes, seen, slots = {}, set(), {}
    total = 0
    sentinel = None
    with F.private(output) as inventory, open(path, 'rb') as tf:
        while True:
            F.require(deadline is None or clock() < deadline, 'STOP retention parsing deadline')
            header = tf.read(512)
            F.require(len(header) == 512, 'STOP truncated tar header')
            if header == b'\0' * 512:
                F.require(tf.read(512) == b'\0' * 512, 'STOP tar end')
                while True:
                    tail = tf.read(65536)
                    if not tail:
                        break
                    F.require(not tail.strip(b'\0'), 'STOP tar trailing data')
                break
            m = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
            # Reject extension headers before tarfile can allocate/read their payload.
            F.require(m.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE),
                      'STOP tar extension/object')
            F.require(m.size >= 0 and (m.isfile() or m.size == 0), 'STOP tar payload type')
            name = F.safe_name(m.name)
            F.require(name not in seen and len(seen) < F.ENTRIES, 'STOP archive duplicate/count')
            seen.add(name)
            F.require(m.uid == m.gid == 10001 and (m.isfile() or m.isdir() or m.issym()),
                      'STOP archive type/UID')
            row = dict(name=name, size=m.size, uid=m.uid, gid=m.gid, mode=m.mode,
                       kind='file' if m.isfile() else 'dir' if m.isdir() else 'link', link=m.linkname)
            if m.issym():
                F.require(len(m.linkname.encode()) <= 4096, 'STOP archive link bound')
            if m.isfile():
                F.require(m.size <= F.LEAF, 'STOP archive leaf')
                total += m.size
                F.require(total <= F.LOGICAL, 'STOP archive aggregate')
                h = hashlib.sha256()
                count = 0
                small = bytearray()
                remaining = m.size
                while remaining:
                    b = tf.read(min(65536, remaining))
                    F.require(b, 'STOP truncated archive data')
                    count += len(b)
                    remaining -= len(b)
                    h.update(b)
                    if name in ('retention-sentinel', 'pab-terminal.slot', 'pab-intent.slot',
                                'pab-p-one-node-observer/records'):
                        F.require(len(small) + len(b) <= F.SLOT, 'STOP bounded retained metadata')
                        small.extend(b)
                padding = (-m.size) % 512
                F.require(len(tf.read(padding)) == padding, 'STOP truncated tar padding')
                F.require(count == m.size, 'STOP truncated archive')
                hashes[name] = h.hexdigest()
                row['sha256'] = h.hexdigest()
                if name == 'retention-sentinel':
                    sentinel = bytes(small)
                elif small:
                    slots[name] = bytes(small)
            raw = encode(row) + b'\n'
            F.require(inventory.tell() + len(raw) <= F.META, 'STOP inventory metadata cap')
            budget.charge(len(raw))
            F.require(inventory.write(raw) == len(raw), 'STOP inventory write')
    return {'hashes': hashes, 'members': len(seen), 'logical': total,
            'sentinel': sentinel, 'slots': slots}


def parse_map(path):
    F.require(path.stat().st_size <= F.META, 'STOP map cap')
    out = {}
    with open(path, 'rb') as f:
        for _ in range(F.ENTRIES + 1):
            line = f.readline(8193)
            if not line:
                return out
            F.require(len(line) <= 8192 and line.endswith(b'\n'), 'STOP map line')
            row = json.loads(line)
            name = F.safe_name(row['name'])
            F.require(name not in out and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
                      'STOP map duplicate/digest')
            out[name] = row['sha256']
    raise RuntimeError('STOP map entries')


def validate_records(raw):
    F.require(len(raw) <= 16384 and raw.endswith(b'\n'), 'STOP 47-field records cap')
    rows = []
    for line in raw.splitlines():
        F.require(len(line) + 1 <= 512 and re.fullmatch(rb'-?[0-9]+(?:,-?[0-9]+){46}', line),
                  'STOP 47-field schema')
        rows.append([int(x) for x in line.split(b',')])
    F.require(len(rows) == 4 and sorted(r[2] for r in rows) == [1, 2, 3, 4], 'STOP stages')
    child = next(r for r in rows if r[2] == 1)
    wait = next(r for r in rows if r[2] == 2)
    owner = wait[3]
    F.require(all(r[:2] == [1, 1] and r[3] == owner and r[11] == 0 and r[12] == 1 for r in rows)
              and child[4] == wait[6] and child[5] == owner and wait[7] == 18944
              and all(r[4] == owner for r in rows if r[2] != 1)
              and child[16] == child[26] == child[31] == 1, 'STOP record binding')
    return {'structural': True, 'independent_offset_type_guard_report_review_required': True,
            'acceptance': False, 'rows': rows}



def base_reference(owner):
    F.require(re.fullmatch('[0-9a-f]{48}', owner['token']) is not None
              and str(uuid.UUID(owner['invocation'])) == owner['invocation'], 'STOP tag owner')
    return 'pab-local-base/' + owner['invocation'] + ':' + owner['token']


def base_steps(owner):
    tag = base_reference(owner)
    return dict(reference=tag, expected_id=BASE,
                inspect=['docker', 'image', 'inspect', BASE],
                absence=['docker', 'image', 'ls', '--no-trunc', '--format', '{{.ID}}',
                         '--filter', 'reference=' + tag],
                tag=['docker', 'image', 'tag', BASE, tag],
                verify=['docker', 'image', 'inspect', tag], retained=True)


def base_recipe(owner):
    # Fixed build-time image hygiene only; never a mounted-volume initializer.
    program = ('import os,stat; p="/tmp"; s=os.lstat(p); '
               'assert stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode)==0o700 '
               'and (s.st_uid,s.st_gid)==(10001,10001); '
               'assert os.listdir(p)==["pab-seed"]; s=os.lstat(p+"/pab-seed"); '
               'assert stat.S_ISREG(s.st_mode) and stat.S_IMODE(s.st_mode)==0o600 '
               'and (s.st_uid,s.st_gid)==(10001,10001); '
               'assert open(p+"/pab-seed","rb").read()==bytes.fromhex("' + SEED.hex() + '")')
    return ('FROM ' + base_reference(owner) + '\nWORKDIR /app\nCOPY --chown=10001:10001 source/ /app/\n'
            'RUN rm -rf -- /tmp && mkdir -m 0700 /tmp && chown 10001:10001 /tmp\n'
            'COPY --chown=10001:10001 --chmod=0600 seed /tmp/pab-seed\n'
            "RUN python -I -B -c '" + program + "'\nUSER 10001:10001\n")


def validate_recipe(context, owner, steps):
    F.require(steps == base_steps(owner), 'STOP pinned local tag steps')
    F.require(F.read_regular(context / 'Dockerfile', 4096, 0o600) == base_recipe(owner).encode()
              and F.read_regular(context / 'seed', 128, 0o600) == SEED,
              'STOP fixed source metadata recipe')


def admit_base(obj):
    F.require(obj['Id'] == BASE and obj['Os'] == 'linux' and obj['Architecture'] == 'amd64',
              'STOP retained base identity')
    startup(obj['Config'])


def fatal_build(root):
    """Only an intact, reaped explicit BuildKit terminal solve is conclusive failure."""
    try:
        result = json.loads(F.read_regular(root / 'build.complete.json', F.META, 0o600))
        err = F.read_regular(root / 'build.stderr', F.META, 0o600)
        return (result['state'] == 'COMPLETE' and result['exit'] == 1
                and result['leader_reaped'] and result['descriptors_closed']
                and not result['group_kill_attempted'] and result['bytes'][1] == len(err)
                and err.endswith(b'\n') and bool(err.splitlines())
                and err.splitlines()[-1].startswith(b'ERROR: failed to build: failed to solve: ')
                and len(err.splitlines()[-1]) > 48)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def prepare_context(repo, root, budget, owner):
    """No build or Git mutation; immutable payload admitted against historical manifest."""
    manifest = json.loads(F.read_regular(D / 'source-manifest.json', F.META))
    files = {**manifest['files'], **manifest['derivative_files']}
    context = root / 'context'
    context.mkdir(mode=0o700)
    source = context / 'source'
    source.mkdir(mode=0o700)
    for name, digest in files.items():
        F.safe_name(name)
        raw = F.read_regular(repo / name, F.LEAF)
        F.require(hashlib.sha256(raw).hexdigest() == digest, 'STOP source mismatch: ' + name)
        dst = source / name
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        budget.charge(len(raw))
        with F.private(dst) as f:
            F.require(f.write(raw) == len(raw), 'STOP copy write')
        F.require(F.read_regular(dst, F.LEAF, 0o600) == raw, 'STOP copied bytes')
    publish(source / 'manifest.json', manifest, budget)
    seed = context / 'seed'
    budget.charge(len(SEED))
    with F.private(seed) as f:
        f.write(SEED)
    # No RUN chown of source needed; numeric COPY owns every private source component.
    recipe = base_recipe(owner)
    budget.charge(len(recipe.encode()))
    with F.private(context / 'Dockerfile') as f:
        f.write(recipe.encode())
    publish(root / 'build-proposal.json', {'base': BASE, 'prebuild': base_steps(owner), 'argv': ['docker', 'build', '--pull=false',
            '--network=none', str(context)], 'not_executed': True, 'metadata_only_tmp': True,
            'image_layer_cache_metadata_excluded_from_logical_5GiB_claim': True}, budget)
    validate_recipe(context, owner, base_steps(owner))
    return context


class Lifecycle:
    def __init__(self, root, lane, image, owner, adapter):
        self.root, self.lane, self.image, self.owner, self.a = root, lane, image, owner, adapter
        self.budget = Budget(HOST - SUPERVISOR_RESERVE)
        self.start = adapter.clock()
        self.phase_end = self.start + 120
        self.operations = {}
        self.objects = {}
        self.lane_inode = None
        self.retained = False
        self.volume = 'pab-p-one-' + owner['invocation']
        self.error = None
        self.calls = 0

    def journal(self):
        self.budget.charge(F.SLOT)
        F.slot_write(self.root / 'intent.slot', {'state': 'PENDING', 'owner': self.owner,
                     'operations': self.operations, 'objects': self.objects,
                     'volume': self.volume, 'image': self.image, 'host_bytes': self.budget.used})

    def dispatch(self, label, argv, *, mutation=False, cap=F.META, seconds=60, completed_nonzero=False):
        self.calls += 1
        F.require(self.calls <= 64, 'STOP command count')
        if mutation:
            F.require(label not in self.operations, 'STOP no mutation retry')
            self.operations[label] = {'state': 'NOT-SENT', 'argv': argv}
            self.journal()
            self.operations[label]['state'] = 'SENT-UNKNOWN'
            self.journal()
        try:
            raw = call(self.root, label, argv, self.budget, self.a,
                       deadline=min(self.phase_end, self.a.clock() + seconds), cap=cap, completed_nonzero=completed_nonzero)
        except BaseException:
            # One discovery observation, never a second mutation. Negative is not absence proof.
            if mutation and not label.endswith('-start'):
                try:
                    query = ['docker', 'volume', 'inspect', self.volume] if label == 'volume-create' else \
                            ['docker', 'inspect', self.objects.get(label.split('-')[0], {}).get('id',
                             'pab-p-one-' + self.owner['invocation'] + '-' + label.split('-')[0])]
                    call(self.root, label + '-reconcile', query, self.budget, self.a,
                         deadline=min(self.start + 1770, self.a.clock() + 30))
                except BaseException:
                    pass
                self.journal()
            raise
        if mutation:
            self.operations[label]['state'] = 'CONFIRMED'
            self.journal()
        return raw

    def inspect(self, label, identifier):
        return json.loads(self.dispatch(label, ['docker', 'inspect', identifier]))[0]

    def acquire(self):
        F.require(self.a.free(self.root) >= 6 * F.GIB, 'STOP host admission 6GiB shared conservative')
        self.journal()
        self.lane.mkdir(mode=0o700)
        s = self.lane.stat()
        self.lane_inode = [s.st_dev, s.st_ino]
        publish(self.lane / 'owner.json', self.owner, self.budget)
        self.journal()
        image = json.loads(self.dispatch('image', ['docker', 'image', 'inspect', self.image]))[0]
        F.require(image['Id'] == self.image and image['Os'] == 'linux' and image['Architecture'] == 'amd64',
                  'STOP exact derivative image')
        startup(image['Config'])
        absent = self.dispatch('volume-absence', ['docker', 'volume', 'ls', '--format', '{{.Name}}',
                                '--filter', 'name=^' + self.volume + '$'])
        F.require(not absent.strip(), 'STOP existing volume')
        self.dispatch('volume-create', ['docker', 'volume', 'create', '--driver', 'local', '--label',
                      'pab.owner=' + self.owner['token'], '--label', 'pab.invocation=' + self.owner['invocation'],
                      self.volume], mutation=True)
        obj = json.loads(self.dispatch('volume-binding', ['docker', 'volume', 'inspect', self.volume]))[0]
        self.objects['volume'] = volume_binding(obj, self.owner, self.volume)
        self.journal()

    def run_container(self, mode, phase_end):
        self.phase_end = min(phase_end, self.start + 120) if mode == 'writer' else min(phase_end, self.a.clock() + 60)
        name = 'pab-p-one-' + self.owner['invocation'] + '-' + mode
        F.require(not self.dispatch(mode + '-absence', ['docker', 'ps', '-aq', '--filter',
                  'name=^/' + name + '$']).strip(), 'STOP container collision')
        raw = self.dispatch(mode + '-create', create_argv(name, self.volume, self.image, mode, self.owner),
                            mutation=True)
        cid = raw.decode().strip()
        F.require(ID.fullmatch(cid) is not None, 'UNKNOWN malformed create acknowledgement')
        self.objects[mode] = {'id': cid, 'name': name, 'state': 'CREATED'}
        self.journal()
        obj = self.inspect(mode + '-binding', cid)
        F.require(container_binding(obj, self.owner, name, self.volume, self.image, mode) == cid,
                  'STOP create mismatch')
        F.require(obj['State']['Status'] == 'created' and not obj['State']['Running'], 'STOP unexpected prestart')
        # Attach is the only reader transport, never exec. Binary stdout carries helper payload only.
        cap = F.EXPORT if mode == 'archive' else F.META
        if mode == 'writer':
            F.require(self.a.clock() <= self.start + 120, 'STOP prestart admission deadline')
            self.phase_end = phase_end
        # Protect wait, stopped inspection and local reap within the original phase.
        end = self.phase_end
        F.require(end - self.a.clock() > 12, 'STOP lifecycle inspection reserve')
        self.phase_end = end - 12
        self.dispatch(mode + '-start', ['docker', 'start', '--attach', cid], mutation=True, cap=cap,
                      seconds=1350 if mode == 'writer' else 60, completed_nonzero=True)
        self.phase_end = end - 6
        wait = self.dispatch(mode + '-wait', ['docker', 'wait', cid], seconds=4, completed_nonzero=True)
        self.phase_end = end
        obj = self.inspect(mode + '-stopped', cid)
        F.require(container_binding(obj, self.owner, name, self.volume, self.image, mode) == cid,
                  'STOP stopped full ID mismatch')
        stopped(obj)
        self.objects[mode]['state'] = 'STOPPED'
        self.objects[mode]['stopped'] = obj
        self.operations[mode + '-start']['state'] = 'TERMINAL'
        self.journal()
        self.lifecycle_status(mode, wait, obj)
        return cid

    def lifecycle_status(self, mode, wait, obj):
        # Full inspect was durably captured and identity checked before any status gate.
        results = [json.loads(F.read_regular(self.root / (mode + suffix + '.complete.json'), F.META))
                   for suffix in ('-start', '-wait')]
        F.require(all(r['state'] == 'COMPLETE' and r['exit'] == 0 for r in results)
                  and wait == b'0\n' and obj['State']['ExitCode'] == 0
                  and obj['State']['OOMKilled'] is False and obj['State']['Error'] == '',
                  'STOP lifecycle status: ' + mode)

    def retention(self):
        F.require(self.objects.get('writer', {}).get('state') == 'STOPPED', 'STOP no concurrent reader')
        retention_end = min(self.start + 1650, self.a.clock() + 180)
        self.phase_end = retention_end
        self.run_container('archive', retention_end)
        audit = parse_archive(self.root / 'archive-start.stdout', self.root / 'archive-inventory.jsonl', self.budget,
                              clock=self.a.clock, deadline=retention_end)
        F.require(audit['sentinel'] == self.owner['token'].encode() + b'\x00\xff\x80\r\n', 'STOP sentinel')
        terminal = F.slot_read(audit['slots']['pab-terminal.slot'])
        F.require(terminal['state'] == 'TERMINAL' and terminal['token'] == self.owner['token']
                  and terminal['invocation'] == self.owner['invocation'] and terminal['pytest_exit'] == 1
                  and terminal['descendants']['complete'] and terminal['acceptance'] is False,
                  'STOP writer terminal binding')
        self.run_container('hash', retention_end)
        F.require(parse_map(self.root / 'hash-start.stdout') == audit['hashes'], 'STOP quiescent hash mismatch')
        publish(self.root / 'retention.json', {'state': 'VERIFIED', 'volume': self.objects['volume'],
                'logical': audit['logical'], 'members': audit['members'], 'physical_quota': False,
                'archive_sha256': F.regular_hash(self.root / 'archive-start.stdout')
                    if (self.root / 'archive-start.stdout').stat().st_size <= F.LEAF else stream_hash(self.root / 'archive-start.stdout'),
                'all_regular_hashes_equal': True, 'acceptance': False}, self.budget)
        self.retained = True
        publish(self.root / '47-field-structural.json', validate_records(
                audit['slots']['pab-p-one-node-observer/records']), self.budget)

    def cleanup(self):
        F.require(self.retained, 'STOP preserve stopped containers and volume')
        self.phase_end = min(self.start + 1770, self.a.clock() + 120)
        for mode in ('writer', 'archive', 'hash'):
            obj = self.objects[mode]
            current = self.inspect(mode + '-pre-remove', obj['id'])
            container_binding(current, self.owner, obj['name'], self.volume, self.image, mode)
            stopped(current)
            self.dispatch(mode + '-remove', ['docker', 'rm', obj['id']], mutation=True)
            absent = self.dispatch(mode + '-removed', ['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id=' + obj['id']])
            F.require(not absent.strip(), 'UNKNOWN remove absence')
            obj['state'] = 'ABSENT'
            self.operations[mode + '-create']['state'] = 'TERMINAL'
            self.operations[mode + '-remove']['state'] = 'TERMINAL'
        self.operations['volume-create']['state'] = 'TERMINAL'
        self.journal()

    def finish(self):
        # Hash inventory and handback first. PENDING remains if any later publication fails.
        final_end = min(self.start + 1800, self.a.clock() + 30)
        inventory = {}
        for p in self.root.iterdir():
            F.require(self.a.clock() < final_end, 'STOP final publication budget')
            if p.is_file() and p.name not in ('intent.slot', 'final.slot'):
                inventory[p.name] = {'bytes': p.stat().st_size, 'sha256': stream_hash(p)}
        publish(self.root / 'evidence-hashes.json', {'files': inventory,
                'exclusions': ['intent.slot', 'final.slot', 'evidence-hashes.json',
                               'handback.json', 'release-intent.json', 'supervisor-complete.json'],
                'host_logical_charged': self.budget.used,
                'supervisor_reserved': SUPERVISOR_RESERVE, 'invocation_cap': HOST}, self.budget)
        F.require(self.a.clock() < final_end, 'STOP final publication budget')
        success = self.error is None and self.retained and all(o['state'] == 'TERMINAL' for o in self.operations.values())
        state = {'state': 'FINALIZATION-PENDING', 'owner': self.owner, 'objects': self.objects,
                 'operations': self.operations, 'retained': self.retained, 'acceptance': False,
                 'error': self.error, 'lane_released': False, 'host_bytes': self.budget.used}
        publish(self.root / 'handback.json', state, self.budget)
        if success:
            s = self.lane.lstat()
            F.require([s.st_dev, s.st_ino] == self.lane_inode and not stat.S_ISLNK(s.st_mode)
                      and json.loads(F.read_regular(self.lane / 'owner.json', F.META, 0o600)) == self.owner,
                      'STOP lane ownership')
            publish(self.root / 'release-intent.json', {'owner': self.owner, 'inode': self.lane_inode}, self.budget)
            (self.lane / 'owner.json').unlink()
            self.lane.rmdir()
            state['lane_released'] = True
        state['state'] = 'RETAINED-EXPECTED-RED-REVIEW-REQUIRED' if success else 'STOP-OR-UNKNOWN-HOLD'
        self.budget.charge(F.SLOT)
        F.slot_write(self.root / 'final.slot', state)
        F.require(F.slot_read(F.read_regular(self.root / 'final.slot', F.SLOT)) == state, 'STOP final readback')
        return success

    def run(self):
        F.enabled()
        try:
            self.acquire()
            # Admission transport ends at120; writer internally protects60/1200/30/60.
            F.require(self.a.clock() <= self.start + 120, 'STOP admission overrun')
            self.run_container('writer', self.start + 1470)
            self.retention()
            self.cleanup()
        except BaseException as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)[:256]
        try:
            self.phase_end = self.start + 1800
            return self.finish()
        except BaseException as exc:
            print(json.dumps({'state': 'FINALIZATION-FAILED', 'error': type(exc).__name__,
                  'owner': self.owner, 'objects': self.objects, 'operations': self.operations,
                  'retained': self.retained, 'acceptance': False}), file=sys.stderr)
            return False


def stream_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(65536)
            if not b:
                return h.hexdigest()
            h.update(b)


def make_root(parent, owner):
    parent = F.canonical(parent)
    F.require(str(parent) == '/private/tmp', 'STOP canonical private tmp')
    root = parent / ('pab-p-one-r16-' + owner['invocation'])
    root.mkdir(mode=0o700)
    for name in ('intent.slot', 'final.slot'):
        F.slot_create(root / name)
    return root


class ForkWorker:
    """One host-owned worker; no product imports or source hooks."""
    def __init__(self, target):
        self.pid = os.fork()
        self.status = None
        if self.pid == 0:
            try:
                os.setsid()
                code = 0 if target() else 1
            except BaseException:
                code = 1
            os._exit(code)
    def poll(self):
        if self.status is None:
            pid, raw = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.status = os.waitstatus_to_exitcode(raw)
        return self.status
    def terminate(self, sig):
        # Target only this still-owned, unreaped worker PID. Remote state stays UNKNOWN.
        if self.poll() is None:
            os.kill(self.pid, sig)


def supervise(root, target, *, factory=ForkWorker, clock=time.monotonic,
              sleep=time.sleep, free=lambda p: shutil.disk_usage(p).free):
    start = clock()
    worker = factory(target)
    outcome = {'state': 'PENDING', 'worker_pid': worker.pid,
               'remote_completion_inferred': False, 'acceptance': False}
    try:
        while worker.poll() is None:
            F.require(clock() - start < 1770, 'STOP host supervisor protected final budget')
            F.require(free(root) >= 2 * F.GIB, 'STOP host supervisor pressure')
            sleep(0.5)
        outcome.update(state='COMPLETE', worker_exit=worker.poll())
    except BaseException as exc:
        outcome.update(state='UNKNOWN', failure=type(exc).__name__)
        for sig, seconds in ((signal.SIGTERM, 1), (signal.SIGKILL, 1)):
            worker.terminate(sig)
            end = clock() + seconds
            while worker.poll() is None and clock() < end:
                sleep(0.05)
        outcome['worker_exit'] = worker.poll()
        outcome['worker_reaped'] = worker.poll() is not None
        # Spawn receipts preserve every observed local CLI PID/session; no remote inference.
        outcome['local_cli_spawn_receipts'] = sorted(p.name for p in root.glob('*.spawn.json'))
    try:
        publish(root / 'supervisor-complete.json', outcome, Budget(SUPERVISOR_RESERVE))
    except BaseException:
        print(json.dumps(dict(outcome, state='SUPERVISOR-PUBLICATION-FAILED')), file=sys.stderr)
        return False
    return outcome['state'] == 'COMPLETE' and outcome['worker_exit'] == 0


def main():
    F.enabled()
    os.umask(0o077)
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--image')
    p.add_argument('--review')
    p.add_argument('--authorization')
    args = p.parse_args()
    owner = {'token': os.urandom(24).hex(), 'invocation': str(uuid.uuid4()), 'source': SOURCE,
             'documentation': DOCUMENTATION, 'pid': os.getpid()}
    if not args.prepare:
        F.require(args.image is not None and re.fullmatch(r'sha256:[0-9a-f]{64}', args.image), 'STOP exact image')
        # Nonempty files are necessary provenance, not semantic self-approval.
        for name in ('review', 'authorization'):
            path = getattr(args, name)
            F.require(path is not None, 'STOP independent review and parent authority required')
            raw = F.read_regular(path, F.META)
            F.require(raw.strip(), 'STOP empty authority')
            owner[name + '_sha256'] = hashlib.sha256(raw).hexdigest()
    owner['launcher_sha256'] = stream_hash(D / 'launcher.py')
    owner['driver_sha256'] = stream_hash(D / 'finite-driver.py')
    root = make_root('/private/tmp', owner)
    if args.prepare:
        budget = Budget()
        budget.charge(2 * F.SLOT)
        prepare_context(D.parents[1], root, budget, owner)
        print('SOURCE BUILD PROPOSAL ONLY', root)
        return 0
    lifecycle = Lifecycle(root, LANE, args.image, owner, Real())
    lifecycle.budget.charge(2 * F.SLOT)
    ok = supervise(root, lifecycle.run)
    print('RETAINED EXPECTED RED; INDEPENDENT REVIEW REQUIRED' if ok else 'STOP/UNKNOWN', root)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
