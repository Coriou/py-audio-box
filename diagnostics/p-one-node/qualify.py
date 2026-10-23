"""Separate r18 dummy lifecycle. Preparation is source-only; execution needs new authority."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import signal
import time
import sys
import uuid

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qualification_launcher', D / 'launcher.py')
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
F = L.F
EXECUTABLE = 'c3b966beafbb9cb861428e59a81b56adb3ccd127'
DOCUMENTATION = 'fd0bc585d897818c66f3ae62d38c6a8eeda769b3'
PROFILE = dict(total_seconds=600, build_seconds=300, writer_seconds=30,
               reader_seconds=60, readers=2, builds=1, writers=1, dummy_bytes=1048576,
               host_bytes=L.HOST, product_dispatch=False, physical_quota=False)
PINS = {'launcher.py': '0ded93a96a7136112c163d546cb269d6fdcbff2b06e8e87fecc5db5e30616c29',
        'finite-driver.py': 'bdf424d31c54bacdf2d8b9406ee41c521beb8d28b8503a00620c3ff29b705732',
        'host-check.py': 'a78b2b0ba03dc6bc64bf42be5c1494b4dd94d38bb1aa37bc0c996cbc986f582d',
        'source-manifest.json': '1d97c4f899de7a7cafe2d2137c917d7af6c9a17c037022ee93a478bd19ce219a'}


def source_pins():
    for name, digest in PINS.items():
        F.require(L.stream_hash(D / name) == digest, 'STOP reviewed source pin')
    return {name: L.stream_hash(D / name) for name in (*PINS, 'dummy-writer.py', 'qualify.py')}


def dummy_command(owner):
    return ['python', '/app/diagnostics/p-one-node/dummy-writer.py', owner['token'], owner['invocation']]


def dummy_create(name, volume, image, owner):
    # Reuse the exact reviewed resource/env/mount recipe, replace only executable argv.
    argv = L.create_argv(name, volume, image, 'writer', owner)
    F.require(argv[-5:] == L.command('writer', owner), 'STOP reviewed command shape')
    return argv[:-5] + dummy_command(owner)


def dummy_binding(obj, owner, name, volume, image):
    """Dummy-specific command; all security values mirror reviewed container_binding."""
    cid, cfg, hc = obj['Id'], obj['Config'], obj['HostConfig']
    F.require(L.ID.fullmatch(cid) is not None and obj['Name'] == '/' + name and obj['Image'] == image,
              'STOP dummy full ID')
    L.startup(cfg)
    F.require(cfg['Labels'] == {'pab.owner': owner['token'], 'pab.invocation': owner['invocation']}
              and cfg['Cmd'] == dummy_command(owner) and cfg['User'] == '10001:10001'
              and cfg['WorkingDir'] == '/app', 'STOP dummy command')
    F.require({'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/app/diagnostics/p-one-node',
               'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl', 'TIKTOKEN_CACHE_DIR=/tokenizer-cache'}
              .issubset(set(cfg['Env'])), 'STOP dummy env')
    F.require(hc['NanoCpus'] == 2000000000 and hc['Memory'] == 2 * F.GIB and hc['PidsLimit'] == 256
              and hc['ReadonlyRootfs'] and hc['NetworkMode'] == 'none' and hc['CapDrop'] == ['ALL']
              and 'no-new-privileges' in hc['SecurityOpt'] and not hc.get('CapAdd')
              and not hc.get('Privileged') and not hc.get('AutoRemove')
              and hc['RestartPolicy']['Name'] == 'no' and hc['LogConfig']['Type'] == 'none', 'STOP dummy security')
    F.require(not any(hc.get(k) for k in ('Binds', 'Devices', 'PortBindings', 'Tmpfs', 'VolumesFrom'))
              and not cfg.get('ExposedPorts') and hc.get('PidMode', '') == '', 'STOP dummy exposure')
    mounts = obj['Mounts']
    F.require(len(mounts) == 1 and mounts[0]['Type'] == 'volume' and mounts[0]['Name'] == volume
              and mounts[0]['Destination'] == '/tmp' and mounts[0]['RW'], 'STOP dummy mount')
    return cid


def prepare(root, owner, budget):
    pins = source_pins()
    context = L.prepare_context(D.parents[1], root, budget, owner)
    dst = context / 'source/diagnostics/p-one-node/dummy-writer.py'
    raw = F.read_regular(D / 'dummy-writer.py', F.LEAF)
    budget.charge(len(raw))
    with F.private(dst) as f:
        F.require(f.write(raw) == len(raw), 'STOP dummy source copy')
    # Only the generated context manifest is augmented; repository manifest is immutable.
    manifest = json.loads(F.read_regular(D / 'source-manifest.json', F.META))
    manifest['derivative_files']['diagnostics/p-one-node/dummy-writer.py'] = pins['dummy-writer.py']
    target = context / 'source/manifest.json'
    target.unlink()  # owned preparation output only; not retained runtime proof
    L.publish(target, manifest, budget)
    image = '<BUILD-IIDFILE>'
    name = 'pab-p-one-' + owner['invocation'] + '-writer'
    volume = 'pab-p-one-' + owner['invocation']
    plan = dict(profile=PROFILE, owner=owner, inputs=pins, base=L.BASE, prebuild=L.base_steps(owner),
                build=['docker', 'build', '--pull=false', '--network=none', '--label',
                       'pab.owner=' + owner['token'], '--label', 'pab.invocation=' + owner['invocation'],
                       '--iidfile', str(root / 'image.id'), str(context)],
                dummy_create=dummy_create(name, volume, image, owner),
                dummy_start=['docker', 'start', '<FULL-WRITER-ID>'],
                detached_wait=['docker', 'wait', '<FULL-WRITER-ID>'],
                readers={m: L.create_argv('pab-p-one-' + owner['invocation'] + '-' + m,
                                          volume, image, m, owner) for m in ('archive', 'hash')},
                substitution='Only build iidfile full sha256 and create full IDs; no user argv',
                acceptance=False)
    L.publish(root / 'qualification-plan.json', plan, budget)
    return plan


class Qualification(L.Lifecycle):
    """Reuses reviewed journal/capture/acquire/readers/finalization; no product run route."""
    def __init__(self, root, lane, owner, adapter):
        super().__init__(root, lane, '', owner, adapter)
        self.plan = None
        self.budget.charge(2 * F.SLOT)
    def dispatch(self, label, argv, **kw):
        F.require(self.a.clock() < self.start + 570, 'STOP qualification final reserve')
        F.require(not ('writer' in argv and '/app/diagnostics/p-one-node/finite-driver.py' in argv)
                  and '/app/diagnostics/p-one-node/entry.py' not in argv, 'STOP product dispatch')
        self.phase_end = min(self.phase_end, self.start + 570)
        if label == 'build':
            F.require(label not in self.operations, 'STOP no build retry')
            self.operations[label] = dict(state='NOT-SENT', argv=argv)
            self.journal()
            self.operations[label]['state'] = 'SENT-UNKNOWN'
            self.journal()
            try:
                raw = L.call(self.root, label, argv, self.budget, self.a,
                             deadline=self.phase_end, cap=F.META)
            except BaseException:
                if L.fatal_build(self.root):
                    self.operations[label].update(state='TERMINAL', result='FAILED')
                    self.journal()
                    raise
                # A labelled image observation never establishes absent/in-flight build completion.
                try:
                    L.call(self.root, 'build-reconcile', ['docker', 'image', 'ls', '--no-trunc',
                           '--filter', 'label=pab.owner=' + self.owner['token'], '--filter',
                           'label=pab.invocation=' + self.owner['invocation']], self.budget, self.a,
                           deadline=min(self.start + 570, self.a.clock() + 10))
                except BaseException:
                    pass
                self.journal()
                raise
            self.operations[label]['state'] = 'CONFIRMED'
            self.journal()
            return raw
        if label == 'base-tag':
            F.require(label not in self.operations, 'STOP no tag retry')
            self.operations[label] = dict(state='NOT-SENT', argv=argv)
            self.journal()
            self.operations[label]['state'] = 'SENT-UNKNOWN'
            self.journal()
            raw = L.call(self.root, label, argv, self.budget, self.a, deadline=self.phase_end)
            self.operations[label]['state'] = 'CONFIRMED'
            self.journal()
            return raw
        return super().dispatch(label, argv, **kw)
    def build(self):
        # Lane acquired before any actual mutation, including the sole build.
        F.require(self.a.free(self.root) >= 6 * F.GIB, 'STOP build admission')
        self.journal()
        self.lane.mkdir(mode=0o700)
        s = self.lane.stat()
        self.lane_inode = [s.st_dev, s.st_ino]
        L.publish(self.lane / 'owner.json', self.owner, self.budget)
        self.journal()
        self.phase_end = min(self.start + 330, self.a.clock() + 30)
        steps = self.plan['prebuild']
        F.require(steps == L.base_steps(self.owner), 'STOP prebuild plan drift')
        L.admit_base(json.loads(self.dispatch('base-inspect', steps['inspect']))[0])
        F.require(not self.dispatch('tag-absence', steps['absence']).strip(), 'STOP tag collision')
        # Tag mutation is journaled, never overwritten/retried or removed.
        self.dispatch('base-tag', steps['tag'], mutation=True)
        self.objects['base_tag'] = dict(reference=steps['reference'], expected_id=L.BASE, retained=True)
        L.admit_base(json.loads(self.dispatch('tag-inspect', steps['verify']))[0])
        self.operations['base-tag']['state'] = 'TERMINAL'
        self.journal()
        L.validate_recipe(self.root / 'context', self.owner, steps)
        L.admit_base(json.loads(self.dispatch('tag-prebuild', steps['verify']))[0])
        self.phase_end = min(self.start + 330, self.a.clock() + 300)
        self.budget.charge(128)  # reserve bounded Docker iidfile before sole build
        self.dispatch('build', self.plan['build'], mutation=True, seconds=300)
        self.image = F.read_regular(self.root / 'image.id', 128, 0o600).decode().strip()
        F.require(re.fullmatch('sha256:[0-9a-f]{64}', self.image), 'UNKNOWN build IID')
        image = json.loads(self.dispatch('built-image', ['docker', 'image', 'inspect', self.image]))[0]
        F.require(image['Id'] == self.image and image['Os'] == 'linux' and image['Architecture'] == 'amd64'
                  and image['Config'].get('User') == '10001:10001'
                  and image['Config'].get('Labels', {}).get('pab.owner') == self.owner['token']
                  and image['Config'].get('Labels', {}).get('pab.invocation') == self.owner['invocation'],
                  'STOP built image binding')
        L.startup(image['Config'])
        self.operations['build']['state'] = 'TERMINAL'
        self.journal()
    def acquire(self):
        # Reviewed acquire owns lane creation; it cannot be called twice. Reuse its volume
        # admission statements below after the build-owned lane (same checks, no takeover).
        F.require(self.a.free(self.root) >= 6 * F.GIB, 'STOP host admission')
        absent = self.dispatch('volume-absence', ['docker', 'volume', 'ls', '--format', '{{.Name}}',
                               '--filter', 'name=^' + self.volume + '$'])
        F.require(not absent.strip(), 'STOP existing volume')
        self.dispatch('volume-create', ['docker', 'volume', 'create', '--driver', 'local', '--label',
                      'pab.owner=' + self.owner['token'], '--label', 'pab.invocation=' + self.owner['invocation'],
                      self.volume], mutation=True)
        obj = json.loads(self.dispatch('volume-binding', ['docker', 'volume', 'inspect', self.volume]))[0]
        self.objects['volume'] = L.volume_binding(obj, self.owner, self.volume)
        self.journal()
    def dummy(self):
        self.phase_end = min(self.start + 390, self.a.clock() + 60)
        name = 'pab-p-one-' + self.owner['invocation'] + '-writer'
        F.require(not self.dispatch('writer-absence', ['docker', 'ps', '-aq', '--filter',
                  'name=^/' + name + '$']).strip(), 'STOP container collision')
        cid = self.dispatch('writer-create', dummy_create(name, self.volume, self.image, self.owner),
                            mutation=True).decode().strip()
        F.require(L.ID.fullmatch(cid) is not None, 'UNKNOWN dummy create ID')
        self.objects['writer'] = dict(id=cid, name=name, state='CREATED')
        self.journal()
        obj = self.inspect('writer-binding', cid)
        dummy_binding(obj, self.owner, name, self.volume, self.image)
        F.require(obj['State']['Status'] == 'created' and not obj['State']['Running'], 'STOP dummy prestart')
        self.dispatch('writer-start', ['docker', 'start', cid], mutation=True, seconds=10)
        # start is detached. wait is a separate read-only CLI, not writer transport.
        L.publish(self.root / 'detached-wait.json', dict(start_argv=['docker', 'start', cid],
                  wait_argv=['docker', 'wait', cid], independent_remote_alarm_seconds=30,
                  local_wait_cannot_extend_remote_deadline=True, actual_timing_unqualified=True), self.budget)
        wait = self.dispatch('writer-wait', ['docker', 'wait', cid], seconds=40)
        F.require(wait.strip() == b'0', 'STOP dummy wait status')
        obj = self.inspect('writer-stopped', cid)
        dummy_binding(obj, self.owner, name, self.volume, self.image)
        L.stopped(obj)
        F.require(obj['State']['ExitCode'] == 0, 'STOP dummy exit')
        self.objects['writer']['state'] = 'STOPPED'
        self.operations['writer-start']['state'] = 'TERMINAL'
        self.journal()
    def retention(self):
        F.require(self.objects.get('writer', {}).get('state') == 'STOPPED', 'STOP concurrent reader')
        end = min(self.start + 510, self.a.clock() + 120)
        self.run_container('archive', end)
        audit = L.parse_archive(self.root / 'archive-start.stdout', self.root / 'archive-inventory.jsonl',
                                self.budget, clock=self.a.clock, deadline=end)
        F.require(audit['sentinel'] == self.owner['token'].encode() + b'\x00\xff\x80\r\n', 'STOP sentinel')
        terminal = F.slot_read(audit['slots']['pab-terminal.slot'])
        F.require(terminal['state'] == 'TERMINAL' and terminal['token'] == self.owner['token']
                  and terminal['invocation'] == self.owner['invocation'] and terminal['qualification'] is True
                  and terminal['product_dispatched'] is False and terminal['acceptance'] is False
                  and terminal['descendants'] == [] and terminal['write_cap'] == 1048576
                  and terminal['bytes_written'] == 2 * 4096 + 53 + 1024
                  and terminal['identity']['pid'] == 1 and terminal['identity']['uid'] == terminal['identity']['gid'] == 10001
                  and terminal['identity']['optimize'] == 0, 'STOP dummy terminal')
        expected = {'pab-seed', 'retention-sentinel', 'dummy.bin', 'pab-terminal.slot'}
        F.require(set(audit['hashes']) == expected and audit['members'] == 5
                  and audit['hashes']['pab-seed'] == hashlib.sha256(F.SEED).hexdigest()
                  and audit['hashes']['dummy.bin'] == hashlib.sha256(bytes(range(256)) * 4).hexdigest(),
                  'STOP exact dummy inventory')
        self.run_container('hash', end)
        F.require(L.parse_map(self.root / 'hash-start.stdout') == audit['hashes'], 'STOP persistence hash mismatch')
        L.publish(self.root / 'retention.json', dict(state='VERIFIED', qualification=True, acceptance=False,
                  image=self.image, source=self.plan['inputs'], volume=self.objects['volume'],
                  hashes=audit['hashes'], logical=audit['logical'], members=audit['members'], terminal=terminal,
                  archive_sha256=L.stream_hash(self.root / 'archive-start.stdout'), physical_quota=False), self.budget)
        self.retained = True
    def cleanup(self):
        F.require(self.retained, 'STOP preserve volume and containers')
        self.phase_end = min(self.start + 570, self.a.clock() + 60)
        for mode in ('writer', 'archive', 'hash'):
            obj = self.objects[mode]
            current = self.inspect(mode + '-pre-remove', obj['id'])
            if mode == 'writer':
                dummy_binding(current, self.owner, obj['name'], self.volume, self.image)
            else:
                L.container_binding(current, self.owner, obj['name'], self.volume, self.image, mode)
            L.stopped(current)
            self.dispatch(mode + '-remove', ['docker', 'rm', obj['id']], mutation=True)
            F.require(not self.dispatch(mode + '-removed', ['docker', 'ps', '-aq', '--no-trunc',
                      '--filter', 'id=' + obj['id']]).strip(), 'UNKNOWN removal')
            obj['state'] = 'ABSENT'
            for suffix in ('create', 'remove'):
                self.operations[mode + '-' + suffix]['state'] = 'TERMINAL'
        self.operations['volume-create']['state'] = 'TERMINAL'
        self.journal()
    def finish(self):
        end = min(self.start + 600, self.a.clock() + 30)
        ok = super().finish()
        F.require(self.a.clock() < end, 'STOP dummy final publication')
        if ok:
            final = F.slot_read(F.read_regular(self.root / 'final.slot', F.SLOT))
            final.update(state='DUMMY-RETAINED-REVIEW-REQUIRED', qualification=True, acceptance=False)
            self.budget.charge(F.SLOT)
            F.slot_write(self.root / 'final.slot', final)
            F.require(F.slot_read(F.read_regular(self.root / 'final.slot', F.SLOT)) == final,
                      'STOP dummy final readback')
        F.require(self.a.clock() < end, 'STOP dummy final publication')
        return ok
    def run(self):
        F.enabled()
        try:
            self.build()
            self.phase_end = min(self.start + 390, self.a.clock() + 60)
            self.acquire()
            self.dummy()
            self.retention()
            self.cleanup()
        except BaseException as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)[:256]
        # Reviewed finish has its own30s ceiling. Overall qualification bound adds600.
        try:
            F.require(self.a.clock() < self.start + 570, 'STOP qualification publication reserve')
            return self.finish()
        except BaseException:
            return False


def validate_plan(root, plan):
    owner = plan['owner']
    expected_build = ['docker', 'build', '--pull=false', '--network=none', '--label',
                      'pab.owner=' + owner['token'], '--label', 'pab.invocation=' + owner['invocation'],
                      '--iidfile', str(root / 'image.id'), str(root / 'context')]
    F.require(plan['build'] == expected_build and plan['base'] == L.BASE, 'STOP fixed build argv')
    volume = 'pab-p-one-' + owner['invocation']
    F.require(plan['dummy_create'] == dummy_create(volume + '-writer', volume, '<BUILD-IIDFILE>', owner)
              and plan['dummy_start'] == ['docker', 'start', '<FULL-WRITER-ID>']
              and plan['detached_wait'] == ['docker', 'wait', '<FULL-WRITER-ID>']
              and plan['readers'] == {m: L.create_argv(volume + '-' + m, volume, '<BUILD-IIDFILE>', m, owner)
                                      for m in ('archive', 'hash')}, 'STOP fixed lifecycle argv')
    manifest = json.loads(F.read_regular(D / 'source-manifest.json', F.META))
    manifest['derivative_files']['diagnostics/p-one-node/dummy-writer.py'] = plan['inputs']['dummy-writer.py']
    context = root / 'context'
    F.require(json.loads(F.read_regular(context / 'source/manifest.json', F.META, 0o600)) == manifest,
              'STOP context manifest')
    expected_names = set(manifest['files']) | set(manifest['derivative_files']) | {'manifest.json'}
    actual_names = {p.relative_to(context / 'source').as_posix() for p in (context / 'source').rglob('*') if p.is_file()}
    F.require(actual_names == expected_names, 'STOP context extras')
    for name, digest in {**manifest['files'], **manifest['derivative_files']}.items():
        F.require(hashlib.sha256(F.read_regular(context / 'source' / name, F.LEAF, 0o600)).hexdigest() == digest,
                  'STOP context source drift')
    L.validate_recipe(context, owner, plan['prebuild'])
    for p in context.rglob('*'):
        F.require(not p.is_symlink() and stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600),
                  'STOP context private mode')


def supervise(root, target, *, factory=L.ForkWorker, clock=time.monotonic,
              sleep=time.sleep, free=lambda p: L.shutil.disk_usage(p).free):
    """Qualification-specific cutoff; detached daemon state never inferred from local kill."""
    start = clock()
    worker = factory(target)
    outcome = dict(state='PENDING', worker_pid=worker.pid, acceptance=False,
                   qualification=True, remote_completion_inferred=False)
    try:
        while worker.poll() is None:
            F.require(clock() < start + 570 and free(root) >= 2 * F.GIB, 'STOP qualification supervisor')
            sleep(0.25)
        outcome.update(state='COMPLETE', worker_exit=worker.poll())
    except BaseException as exc:
        outcome.update(state='UNKNOWN', failure=type(exc).__name__)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            worker.terminate(sig)
            end = min(start + 575, clock() + 1)
            while worker.poll() is None and clock() < end:
                sleep(0.05)
        outcome.update(worker_exit=worker.poll(), worker_reaped=worker.poll() is not None,
                       local_cli_spawn_receipts=sorted(p.name for p in root.glob('*.spawn.json')))
    try:
        F.require(clock() < start + 600, 'STOP qualification supervisor publication')
        L.publish(root / 'supervisor-complete.json', outcome, L.Budget(L.SUPERVISOR_RESERVE))
        F.require(clock() < start + 600, 'STOP qualification supervisor publication')
    except BaseException:
        return False
    return outcome['state'] == 'COMPLETE' and outcome['worker_exit'] == 0


def main():
    F.enabled()
    os.umask(0o077)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-plan')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--review')
    parser.add_argument('--authorization')
    args = parser.parse_args()
    if args.prepare:
        F.require(not any((args.execute_plan, args.plan_sha256, args.review, args.authorization)), 'STOP prepare argv')
        owner = dict(token=os.urandom(24).hex(), invocation=str(uuid.uuid4()), pid=os.getpid(),
                     source=EXECUTABLE, documentation=DOCUMENTATION, qualification=True)
        root = L.make_root('/private/tmp', owner)
        budget = L.Budget(L.HOST - L.SUPERVISOR_RESERVE)
        budget.charge(2 * F.SLOT)
        plan = prepare(root, owner, budget)
        # Preparation writes share the future invocation account.
        L.publish(root / 'preparation-account.json', dict(charged=budget.used + 4096), budget)
        print('SOURCE ONLY', root, L.stream_hash(root / 'qualification-plan.json'))
        return 0
    F.require(args.execute_plan and args.plan_sha256 and args.review and args.authorization,
              'STOP separately reviewed plan and parent authority required')
    root = F.canonical(args.execute_plan)
    F.require(root.parent == Path('/private/tmp') and stat.S_IMODE(root.stat().st_mode) == 0o700,
              'STOP qualification root')
    raw = F.read_regular(root / 'qualification-plan.json', F.META, 0o600)
    F.require(hashlib.sha256(raw).hexdigest() == args.plan_sha256, 'STOP exact frozen plan')
    plan = json.loads(raw)
    F.require(plan['profile'] == PROFILE and plan['inputs'] == source_pins(), 'STOP plan source/profile drift')
    validate_plan(root, plan)
    owner = plan['owner']
    F.require(re.fullmatch('[0-9a-f]{48}', owner['token']) and str(uuid.UUID(owner['invocation'])) == owner['invocation']
              and owner['source'] == EXECUTABLE and owner['documentation'] == DOCUMENTATION,
              'STOP plan ownership')
    # These external records are provenance requirements, never semantic self-approval.
    authority_hashes = {}
    for kind in ('review', 'authorization'):
        authority_raw = F.read_regular(getattr(args, kind), F.META)
        authority_hashes[kind] = hashlib.sha256(authority_raw).hexdigest()
        value = json.loads(authority_raw)
        F.require(value['plan_sha256'] == args.plan_sha256 and value['scope'] == 'ONE-DUMMY-ONLY'
                  and value['approved'] is True, 'STOP independent review/parent authority binding')
    F.require(not (root / 'execution-once.json').exists(), 'STOP no qualification retry')
    account = json.loads(F.read_regular(root / 'preparation-account.json', F.META, 0o600))
    q = Qualification(root, L.LANE, owner, L.Real())
    q.plan = plan
    q.budget.charge(account['charged'])  # conservative includes initial slots twice
    L.publish(root / 'execution-once.json', dict(plan_sha256=args.plan_sha256, owner=owner, authority_sha256=authority_hashes), q.budget)
    ok = supervise(root, q.run)
    print('DUMMY RECEIPT REQUIRES INDEPENDENT REVIEW' if ok else 'STOP/UNKNOWN; RETAINED', root)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
