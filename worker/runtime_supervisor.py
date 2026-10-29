"""Linux PID1 trusted single-lane parent/reaper; no inference in the parent."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import socket
import subprocess
import threading
import time

from worker.runtime_journal import Journal, MAX_RECORDS, MAX_UNACKED
from worker.runtime_protocol import authenticate, receive, send, session, request, execution, execution_session, managed_qwen_cpu, resident_managed_qwen_cpu, QWEN_CPU_SECONDS
from worker.transport import canonical, Refused, Session


class RuntimeSupervisor:
    def __init__(self, root='/receipts', ipc='/ipc', *, bundles=None, references=None, qwen_contexts=None, inventory=None, qwen_profiles=None, pocket_reference_roots=None, installed_pocket=False, installed_policy=None, audio_converters=None, converter_image_digest=None, export_reference_roots=None, export_configuration=None, expected_worker_image=None, resident_qwen_policy=None):
        if os.getpid() != 1:
            raise RuntimeError('pid1_required')
        from types import MappingProxyType
        self._export_configuration = None
        if export_configuration is not None:
            self._export_configuration = export_startup(export_configuration, expected_worker_image)
        elif expected_worker_image is not None:
            raise ValueError('explicit_export_configuration_required')
        self.audio_converters = copy.deepcopy(audio_converters or {})
        self.converter_image_digest = converter_image_digest
        self.export_reference_roots = dict(export_reference_roots or {})
        if self._export_configuration is not None:
            if installed_pocket or audio_converters or export_reference_roots or converter_image_digest:
                raise ValueError('export_configuration_is_sole_authority')
            self.converter_image_digest = self._export_configuration['images']['worker']
            self.export_reference_roots = {p:'/references/assets' for p in self._export_configuration['audio_export']['principals']}
        from service import install_policy as ip
        if installed_policy is not None and installed_policy is not ip.SPLIT and installed_policy is not ip.POCKET_SPLIT and installed_policy is not ip.QWEN_SPLIT:
            raise ValueError('explicit_fixed_install_policy_required')
        if installed_policy is ip.POCKET_SPLIT and (not installed_pocket or export_configuration is not None):
            raise ValueError('installed_pocket_policy_binding')
        if installed_policy is ip.QWEN_SPLIT and (installed_pocket or export_configuration is not None):
            raise ValueError('installed_qwen_policy_binding')
        self.installed_policy = installed_policy
        if resident_qwen_policy is not None:
            from service.installed_qwen_daemon import SERVICE_POLICY
            if resident_qwen_policy != SERVICE_POLICY or installed_policy is not ip.QWEN_SPLIT:
                raise ValueError('resident_qwen_requires_explicit_installed_policy')
        self.resident_qwen = resident_qwen_policy is not None
        self.resident = None
        self.installed_pocket = installed_pocket
        self.bundles = MappingProxyType(dict(bundles or {}))
        self.references = MappingProxyType(copy.deepcopy(references or {}))
        self.pocket_reference_roots = MappingProxyType(dict(pocket_reference_roots or {}))
        self.qwen_contexts = copy.deepcopy(qwen_contexts or {})
        self.inventory = copy.deepcopy(inventory)
        self.qwen_profiles = copy.deepcopy(qwen_profiles or [])
        self.journal = Journal(root)
        self.ipc = Path(ipc)
        self.boot = secrets.token_hex(16)
        self.secret = (self.ipc / 'lifecycle.key').read_text().strip()
        self.execution_secret = (self.ipc / 'execution.key').read_text().strip()
        self.cutoffs = {}
        self.commands = queue.Queue(8)
        self.slots = threading.BoundedSemaphore(8)
        self.active = None
        self.tree_lock = threading.Lock()
        self.failed = False
        self.launches = 0
        self.received = queue.Queue(1)
        self.halt = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'halt', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'halt', True))
        threading.Thread(target=self.watchdog, daemon=True).start()

    def persist(self, value):
        try:
            self.journal.save(value)
        except Exception:
            self.failed = True
            raise Refused('journal_write_failed_quarantine')

    def lookup(self, identity):
        for row in self.journal.value['records']:
            if row['id'] == identity:
                return row
        raise Refused('unknown_identity')

    def change(self, row, **changes):
        value = copy.deepcopy(self.journal.value)
        updated = next(r for r in value['records'] if r['id'] == row['id'])
        updated.update(changes)
        self.persist(value)
        return updated

    def state(self, row):
        if row['phase'] == 'terminal':
            return 'stopped-and-reaped'
        if self.active and self.active['id'] == row['id'] and not self.failed:
            stopping = self.active.get('stopping')
            if stopping is not None and time.monotonic()-stopping > 1:
                return 'unknown'
            return 'alive'
        return 'unknown'

    def managed_qwen(self, spec):
        return resident_managed_qwen_cpu(spec) if self.resident_qwen else managed_qwen_cpu(spec)

    @staticmethod
    def owned_processes(uid):
        """Snapshot one unique, unprivileged attempt UID; /proc ambiguity quarantines."""
        found = {}
        for entry in os.scandir('/proc'):
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            try:
                status = Path(entry.path, 'status').read_text()
                line = next(line for line in status.splitlines() if line.startswith('Uid:'))
                ids = tuple(map(int, line.split()[1:]))
                if uid not in ids:
                    continue
                stat = Path(entry.path, 'stat').read_text()
                start = int(stat[stat.rfind(')')+2:].split()[19])
                found[pid] = start
            except FileNotFoundError:
                continue
            except (PermissionError, ValueError, StopIteration, IndexError) as exc:
                raise Refused('process_custody_unknown') from exc
        return found

    @staticmethod
    def signal_owned(uid, sig):
        for pid, start in RuntimeSupervisor.owned_processes(uid).items():
            try:
                fd = os.pidfd_open(pid)
                try:
                    stat = Path('/proc', str(pid), 'stat').read_text()
                    actual = int(stat[stat.rfind(')')+2:].split()[19])
                    status = Path('/proc', str(pid), 'status').read_text()
                    ids = tuple(map(int, next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1:]))
                    if actual != start or uid not in ids:
                        raise Refused('process_identity_changed')
                    signal.pidfd_send_signal(fd, sig)
                finally:
                    os.close(fd)
            except (FileNotFoundError, ProcessLookupError):
                continue

    def create_attempt_cgroup(self):
        """A dedicated cgroup v2 makes detached descendants observable atomically."""
        root = Path('/sys/fs/cgroup')
        if not (root / 'cgroup.controllers').is_file():
            raise Refused('resident_cgroup_v2_required')
        group = root / ('py-audio-box-' + self.boot + '-' + secrets.token_hex(8))
        try:
            group.mkdir(mode=0o700)
            if not (group / 'cgroup.kill').is_file():
                raise Refused('resident_cgroup_kill_required')
            return group
        except BaseException:
            self.failed = True
            try: group.rmdir()
            except OSError: pass
            raise

    @staticmethod
    def cgroup_empty(group):
        try:
            values = dict(line.split() for line in (group / 'cgroup.events').read_text().splitlines())
            if values.get('populated') not in ('0','1'):
                raise Refused('resident_cgroup_unknown')
            return values['populated'] == '0'
        except (OSError, ValueError) as exc:
            raise Refused('resident_cgroup_unknown') from exc

    @staticmethod
    def kill_cgroup(group):
        try:
            (group / 'cgroup.kill').write_text('1')
        except OSError as exc:
            raise Refused('resident_cgroup_kill_unknown') from exc

    def ensure_resident(self, cutoff):
        """Start one networkless model owner with a private supervisor control FD."""
        if self.resident is not None:
            if self.resident['process'].poll() is not None:
                self.failed = True
                self.stop_tree()
                raise Refused('resident_lost')
            return self.resident
        from service.installed_qwen_daemon import service_model
        expected = {name:service_model(name)[2] for name in ('clone','Ryan')}
        if set(self.bundles) != set(expected.values()):
            self.failed = True
            self.stop_tree()
            raise Refused('resident_exact_bundles_required')
        parent, child = socket.socketpair()
        token = secrets.token_hex(24)
        roots = {name:dict(path=str(self.bundles[digest]),revision=service_model(name)[1])
                 for name,digest in expected.items()}
        config = dict(address='\0py-audio-box-qwen-'+token, roots=roots)
        proc = None
        try:
            proc = subprocess.Popen(['python','-m','worker.qwen_resident',str(child.fileno())],
                user=19999, group=19999, extra_groups=[], start_new_session=True,
                close_fds=True, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child.close()
            self.resident = dict(process=proc, address=config['address'], control=parent,
                                 roots=roots, invalidating=False)
            send(parent, config, min(cutoff,time.monotonic()+2))
            if receive(parent,min(cutoff,time.monotonic()+2)) != {'ready':True}:
                raise Refused('resident_startup')
            return self.resident
        except BaseException:
            self.failed = True
            self.stop_tree()
            if proc is not None:
                self.signal_owned(19999, signal.SIGKILL)
            parent.close()
            raise
        finally:
            child.close()

    def resident_control(self, op, authority=None):
        from worker.qwen_resident import frame, read
        resident = self.resident
        if resident is None or resident['invalidating']:
            raise Refused('resident_lost')
        cutoff = time.monotonic()+2
        try:
            frame(resident['control'], dict(op=op, authority=authority), cutoff)
            if read(resident['control'], cutoff) != {'ok':True}:
                raise Refused('resident_control')
        except BaseException:
            self.failed = True
            self.stop_tree()
            raise

    def operation(self, op, body):
        if op == 'startup':
            if body: raise Refused('closed_body')
            from service.daemon_runtime import facts
            if getattr(self,'_export_configuration',None) is not None:
                from service.installed_export import FAMILY
                local=facts('worker',family=FAMILY)
            elif getattr(self, '_qwen_daemon', False):
                from service.installed_qwen_daemon import FAMILY
                local=facts('worker',family=FAMILY)
            elif getattr(self, '_pocket_daemon', False):
                from service.installed_pocket_daemon import FAMILY
                local=facts('worker',family=FAMILY)
            else: local=facts('worker')
            return dict(hello=self.operation('hello',{}),local=local)
        if op == 'hello':
            if body:
                raise Refused('closed_body')
            value = dict(boot=self.boot, lane=self.journal.value['lane'], quarantined=self.failed or any(
                r['phase'] != 'terminal' and (not self.active or self.active['id'] != r['id']) for r in self.journal.value['records']), launches=self.launches)
            if getattr(self, '_export_configuration', None) is not None:
                from service.installed_export import export_seal
                value['export_seal'] = export_seal(self._export_configuration)
            if self.journal.maintenance is not None: value['maintenance']=self.journal.maintenance
            return value
        if op == 'prepare':
            if set(body) != {'snapshot', 'binding', 'fixture'} or body['fixture'] not in ('silence', 'busy', 'descendants', 'permissions', 'kokoro', 'pocket', 'qwen', 'audio_export'):
                raise Refused('closed_prepare')
            snapshot = body['snapshot']
            if not isinstance(snapshot, list) or len(snapshot) != 4 or hashlib.sha256(canonical(snapshot)).hexdigest() != body['binding']:
                raise Refused('authority_binding')
            from contracts import validate as oracle
            job, context, assets, profiles = snapshot
            oracle.stored_context(job, context)
            from worker.runtime_protocol import execution, dispatch_mode
            spec = execution(context)
            if (body['fixture'] in ('kokoro','pocket','qwen','audio_export')) != (spec is not None) or (spec is not None and body['fixture'] != dispatch_mode(spec)): raise Refused('execution_dispatch_mismatch')
            anchored = time.monotonic()
            budget = (datetime.fromisoformat(job['deadline']) - datetime.now(timezone.utc)).total_seconds()
            if spec is not None and spec.engine!='audio-export':
                selected = {spec.bundle_id:self.bundles[spec.bundle_id]} if self.resident_qwen and spec.engine == 'qwen' and spec.bundle_id in self.bundles else self.bundles
                if context['execution']['bundles'] != selected:
                    raise Refused('unregistered_mount_mapping')
            if spec is not None and spec.engine == 'audio-export':
                fd=self.verify_export(context,job['principal_id']); os.close(fd)
            if spec is not None and spec.engine == 'pocket-tts':
                self.verify_pocket(context, job['principal_id'])
            if spec is not None and spec.engine == 'qwen':
                self.verify_qwen(context,profiles,job['principal_id'])
            attempt = job['attempts'][-1]
            allocation = dict(snapshot=snapshot, binding=body['binding'], fixture=body['fixture'])
            for row in self.journal.value['records']:
                if row['allocation'] == allocation:
                    return dict(runtime_id=row['id'], state=self.state(row))
            rows = self.journal.value['records']
            if (self.failed or self.active or any(r['phase'] != 'terminal' for r in rows) or
                    len(rows) >= MAX_RECORDS or sum(not r['acked'] for r in rows) >= MAX_UNACKED):
                raise Refused('lane_quarantined_busy_or_full')
            stage = attempt['staging_id']
            if not isinstance(stage, str) or Path(stage).name != stage or stage in ('.', '..'):
                raise Refused('stage_id')
            # Install the original monotonic budget before ACK/start. Never recompute.
            maximum=min(180,context['plan']['resolved']['profile']['limits']['deadline_seconds']) if spec is not None and spec.engine=='pocket-tts' else min(QWEN_CPU_SECONDS,context['plan']['resolved']['profile']['limits']['deadline_seconds']) if spec is not None and spec.engine=='qwen' and spec.device=='cpu' else 30
            if not 0 < budget <= maximum:
                raise Refused('bounded_original_deadline_required')
            cutoff = anchored+budget
            identity = '/'.join((self.journal.value['lane'], self.boot, secrets.token_hex(16)))
            row = dict(id=identity, allocation=allocation, phase='prepared', acked=False,
                       uid=20000+len(rows), boot=self.boot)
            value = copy.deepcopy(self.journal.value)
            value['records'].append(row)
            self.persist(value)  # Reservation before ACK and before stage setup.
            self.cutoffs[identity] = cutoff
            with self.tree_lock:
                self.active = dict(id=identity, cutoff=cutoff, started=False,
                                   stopping=None, proposal=None, proposed_at=None, leader=None, reaped_children=0, receiver_done=threading.Event())
            return dict(runtime_id=identity, state='alive')
        if set(body) != {'runtime_id'} or not isinstance(body['runtime_id'], str):
            raise Refused('closed_identity_body')
        row = self.lookup(body['runtime_id'])
        if op == 'observe':
            return dict(runtime_id=row['id'], state=self.state(row))
        if op == 'collect':
            if row['phase'] != 'terminal':
                return dict(runtime_id=row['id'], state=self.state(row), receipt=None, proposal=None)
            return dict(runtime_id=row['id'], state='stopped-and-reaped', receipt=row['receipt'], proposal=row.get('proposal'))
        if op == 'ack':
            if row['phase'] != 'terminal':
                raise Refused('terminal_receipt_required')
            self.change(row, acked=True, proposal=None)
            return dict(acked=True)
        if op == 'stop':
            if self.active and self.active['id'] == row['id']:
                self.stop_tree()
            return dict(runtime_id=row['id'], state=self.state(row))
        if op == 'start':
            if row['phase'] != 'prepared' or not self.active or self.active['id'] != row['id']:
                # Same attempt start never forks twice, including old tombstones.
                return dict(runtime_id=row['id'], state=self.state(row))
            if self.failed or time.monotonic() >= self.active['cutoff']:
                raise Refused('expired_or_quarantined')
            context = row['allocation']['snapshot'][1]
            from worker.runtime_protocol import execution
            spec = execution(context)
            reference_fd = None
            if spec is not None and spec.engine == 'audio-export':
                fd=self.verify_export(context,row['allocation']['snapshot'][0]['principal_id']); os.close(fd)
            if spec is not None and spec.engine == 'pocket-tts':
                self.verify_pocket(context, row['allocation']['snapshot'][0]['principal_id'])
            if spec is not None and spec.engine == 'qwen':
                self.verify_qwen(context,row['allocation']['snapshot'][3],row['allocation']['snapshot'][0]['principal_id'])
            row = self.change(row, phase='started')  # Fence BEFORE any fork.
            job, context, assets, profiles = row['allocation']['snapshot']
            stage = job['attempts'][-1]['staging_id']
            uid = row['uid']
            # Nonresident setup helper, reaped before engine launch. No extra caps.
            setup='import os,sys; os.umask(0); os.mkdir(sys.argv[1],0o770)'
            if spec is not None and (spec.engine=='pocket-tts' or self.managed_qwen(spec)):
                setup+='; os.mkdir(sys.argv[1]+"/conditioning",0o770)'
            subprocess.run(['python', '-c', setup,
                            '/staging/'+stage], user=10001, group=uid, extra_groups=[],
                           close_fds=True, check=True, timeout=1)
            if time.monotonic() >= self.active['cutoff']:
                raise Refused('expired_before_launch')
            parent, child = socket.socketpair()
            key = secrets.token_hex(32)
            config = dict(job=job, context=context, allowed_assets=assets, profiles=profiles,
                          secret=key, control_uid=0, gid=10001, fixture=row['allocation']['fixture'],
                          protected_stages=[r['allocation']['snapshot'][0]['attempts'][-1]['staging_id'] for r in self.journal.value['records'] if r['id'] != row['id']])
            if spec is not None and spec.engine=='pocket-tts':
                config['pocket_cutoff']=self.active['cutoff']  # Original prepare anchor, never renewed.
            if spec is not None and spec.engine=='qwen' and spec.device=='cpu':
                config['qwen_cpu_cutoff']=self.active['cutoff']
                from service import install_policy as ip
                if getattr(self, 'installed_policy', None) is ip.QWEN_SPLIT:
                    config['installed_qwen_cpu']=True
                if self.resident_qwen:
                    config['resident_qwen_cpu']=True
            group = None
            try:
                if self.managed_qwen(spec):
                    reference_fd=self.verify_qwen(context,profiles,job['principal_id'],retain_fd=True)
                    config['reference_fd']=reference_fd
                if spec is not None and spec.engine=='audio-export':
                    reference_fd=self.verify_export(context,job['principal_id'])
                    config['reference_fd']=reference_fd
                    config['converter_image_digest']=self.converter_image_digest
                if spec is not None and spec.engine == 'pocket-tts' and getattr(self, 'installed_pocket', False):
                    reference_fd = self.verify_pocket(context, job['principal_id'], retain_fd=True)
                    config['reference_fd'] = reference_fd
                if self.resident_qwen and spec is not None and spec.engine == 'qwen':
                    resident = self.ensure_resident(self.active['cutoff'])
                    from worker.runtime_protocol import resident_attempt_authority
                    nonce = secrets.token_hex(32)
                    authority = resident_attempt_authority(spec, context['plan'], resident['roots'], self.active['cutoff'], nonce)
                    self.resident_control('grant', authority)
                    config['qwen_resident'] = dict(address=resident['address'], nonce=nonce)
                if time.monotonic() >= self.active['cutoff']: raise Refused('expired_before_launch')
                group = self.create_attempt_cgroup() if self.resident_qwen else None
                gate_read, gate_write = os.pipe() if self.resident_qwen else (None, None)
                try:
                    proc = subprocess.Popen(['python', '-m', 'worker.runtime_child', str(child.fileno())] +
                        ([str(gate_read)] if gate_read is not None else []),
                        user=uid, group=uid, extra_groups=[], start_new_session=True,
                        close_fds=True, pass_fds=(child.fileno(),) + (() if reference_fd is None else (reference_fd,)) +
                        (() if gate_read is None else (gate_read,)), stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.active.update(started=True, leader=proc.pid, process=proc, cgroup=group)
                    if group is not None:
                        (group / 'cgroup.procs').write_text(str(proc.pid))
                        os.write(gate_write, b'1')
                    self.launches += 1
                finally:
                    if gate_read is not None:
                        os.close(gate_read); os.close(gate_write)
            except BaseException:
                parent.close()
                if group is not None and self.active.get('cgroup') is None:
                    try: group.rmdir()
                    except OSError: self.failed = True
                self.stop_tree()
                raise
            finally:
                child.close()
                if reference_fd is not None: os.close(reference_fd)
            active = self.active
            def collect_internal():
                try:
                    cutoff = active['cutoff']
                    send(parent, config, cutoff)
                    binding = row['allocation']['binding']
                    authenticate(parent, Session(key, binding, 0, 10001), cutoff, control=True)
                    message = dict(protocol_version='1.3.0', kind='execute', job_id=job['id'],
                        attempt_id=job['attempts'][-1]['id'], generation=job['generation'],
                        input_sha256=job['input_sha256'], input_asset_ids=assets,
                        staging_id=stage, owner_id=job['attempts'][-1]['owner_id'],
                        identity=context['plan']['resolved']['identity'], deadline=job['deadline'],
                        stage='loading', result_asset_ids=[], plan=context['plan'])
                    send(parent, message, cutoff)
                    proposal = receive(parent, cutoff)
                    self.received.put_nowait((row['id'], proposal, time.monotonic()))
                except Exception:
                    pass
                finally:
                    parent.close()
                    active['receiver_done'].set()
            threading.Thread(target=collect_internal, daemon=True).start()
            return dict(runtime_id=row['id'], state='alive')
        raise Refused('operation_refused')

    @staticmethod
    def signal_namespace(sig):
        try:
            os.kill(-1, sig)
        except ProcessLookupError:
            pass

    def stop_tree(self):
        with self.tree_lock:
            active = self.active
            if active is not None and not active.get('finalizing') and active['stopping'] is None:
                active['stopping'] = time.monotonic()
                if self.resident_qwen:
                    if self.resident is not None:
                        self.resident['invalidating'] = True
                    try:
                        self.signal_owned(self.lookup(active['id'])['uid'], signal.SIGTERM)
                        if self.resident is not None:
                            self.signal_owned(19999, signal.SIGTERM)
                    except BaseException:
                        self.failed = True
                        self.signal_namespace(signal.SIGKILL)
                else:
                    self.signal_namespace(signal.SIGTERM)

    def watchdog_tick(self):
        with self.tree_lock:
            active = self.active
            if active is None or active.get('finalizing'):
                return
            now = time.monotonic()
            if (now >= active['cutoff'] or self.halt) and active['stopping'] is None:
                active['stopping'] = now
                if self.resident_qwen:
                    if self.resident is not None:
                        self.resident['invalidating'] = True
                    try:
                        self.signal_owned(self.lookup(active['id'])['uid'], signal.SIGTERM)
                        if self.resident is not None:
                            self.signal_owned(19999, signal.SIGTERM)
                    except BaseException:
                        self.failed = True
                        self.signal_namespace(signal.SIGKILL)
                else:
                    self.signal_namespace(signal.SIGTERM)
            if active['stopping'] is not None and now-active['stopping'] >= .2:
                if self.resident_qwen:
                    try:
                        self.kill_cgroup(active['cgroup']) if active.get('cgroup') is not None else self.signal_owned(self.lookup(active['id'])['uid'], signal.SIGKILL)
                        if self.resident is not None:
                            self.signal_owned(19999, signal.SIGKILL)
                    except BaseException:
                        self.failed = True
                        self.signal_namespace(signal.SIGKILL)
                else:
                    self.signal_namespace(signal.SIGKILL)

    def watchdog(self):
        while True:
            self.watchdog_tick()
            time.sleep(.01)

    def tick(self):
        active = self.active
        if active is None:
            if self.resident is not None and self.resident['process'].poll() is not None:
                self.failed = True
                self.signal_owned(19999, signal.SIGKILL)
                return
            return
        if active.get('finalizing'):
            return
        try:
            identity, proposal, when = self.received.get_nowait()
            if identity == active['id']:
                active.update(proposal=proposal, proposed_at=when)
        except queue.Empty:
            pass
        now = time.monotonic()
        if now >= active['cutoff'] or self.halt:
            self.stop_tree()
        empty = False
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                if self.resident is not None and pid == self.resident['process'].pid:
                    resident = self.resident
                    resident['process'].returncode = os.waitstatus_to_exitcode(status)
                    if not resident['invalidating']:
                        self.failed = True
                        self.stop_tree()
                    continue
                active['reaped_children'] += 1
                if pid == active['leader']:
                    active['process'].returncode = os.waitstatus_to_exitcode(status)
                    active['leader_reaped'] = True
                    if not self.resident_qwen:
                        self.stop_tree()
            except ChildProcessError:
                empty = True
                break
        if self.resident_qwen:
            try:
                uid = self.lookup(active['id'])['uid']
                owned = self.owned_processes(uid)
                group_empty = self.cgroup_empty(active['cgroup']) if active.get('cgroup') is not None else not owned
                if group_empty and owned: raise Refused('resident_cgroup_uid_mismatch')
                resident_owned = self.owned_processes(19999) if self.resident is not None else {}
            except BaseException:
                self.failed = True
                self.stop_tree()
                return
            if self.resident is not None and self.resident['invalidating']:
                if not resident_owned and self.resident['process'].returncode is not None:
                    self.resident['control'].close()
                    self.resident = None
            elif self.resident is not None and self.resident['process'].returncode is not None:
                self.failed = True
                self.stop_tree()
                return
            if active['started'] and active.get('leader_reaped') and active['receiver_done'].is_set() and active['stopping'] is None:
                try:
                    identity, proposal, when = self.received.get_nowait()
                    if identity == active['id']:
                        active.update(proposal=proposal, proposed_at=when)
                except queue.Empty:
                    pass
                if not group_empty or owned or active['proposal'] is None or active['process'].returncode != 0:
                    self.stop_tree()
                else:
                    # The one nonce is spent on the successful call. Revoke the
                    # authorization before persisting this attempt's receipt.
                    if self.resident is not None:
                        try:
                            self.resident_control('revoke')
                        except BaseException:
                            return
            empty = (group_empty and not owned and (not active['started'] or active.get('leader_reaped')) and
                     (active['stopping'] is None or self.resident is None))
        if empty and active['started'] and not active['receiver_done'].is_set():
            return
        if empty and (active['started'] or active['stopping'] is not None):
            try:
                identity, proposal, when = self.received.get_nowait()
                if identity == active['id']:
                    active.update(proposal=proposal, proposed_at=when)
            except queue.Empty:
                pass
            reaped_at = time.monotonic()
            row = self.lookup(active['id'])
            receipt = dict(runtime_id=row['id'], binding=row['allocation']['binding'],
                reaped=True, engine_uid=row['uid'], reap_method='pid1-waitpid-echild',
                reaped_children=active['reaped_children'], timely=bool(active['proposal'] is not None and
                active['proposed_at'] < active['cutoff'] and reaped_at < active['cutoff'] and (not self.resident_qwen or active['stopping'] is None)),
                original_deadline=row['allocation']['snapshot'][0]['deadline'])
            with self.tree_lock:
                active['finalizing'] = True
            try:
                if self.resident_qwen and active.get('cgroup') is not None:
                    active['cgroup'].rmdir()
                if row['allocation']['fixture'] in ('pocket','qwen') and row['phase'] != 'prepared':
                    self.conditioning(row)
                self.change(row, phase='terminal', receipt=receipt, proposal=active['proposal'] if not self.resident_qwen or receipt['timely'] else None)
            except Exception:
                self.failed = True
            with self.tree_lock:
                if not active.get('helper_unknown'):
                    self.active = None

    def verify_export(self, context, principal, *, owner_uid=10001):
        from worker.audio_export import execution, verify_converter, pcm, MAX_SOURCE, read_bounded
        from worker.engines.pocket import _open
        execution(context)
        reg=context['execution']['converter']
        profile=context['plan']['resolved']['profile']['id']
        installed = getattr(self, '_export_configuration', None)
        if installed is not None:
            from service import installed_export as ex
            allowed = installed['audio_export']
            if (principal not in allowed['principals'] or
                    ex._encoded(context['plan']['resolved']['profile']) != ex._encoded(ex.oracle.profile_snapshot(allowed['profile'])) or
                    ex._encoded(reg) != ex._encoded(allowed['converter'])):
                raise Refused('unregistered_export_inputs')
            p = allowed['profile']
            if (p['support'] != 'qualified' or p['installation'] != 'installed' or p['configuration'] != 'valid' or
                    p['readiness'] != 'ready' or p['verification'] != 'measured' or not p['qualification_receipt']):
                raise Refused('export_profile_not_ready')
            exe = ex.verify_local_converter(installed); os.close(exe)
        elif self.audio_converters.get((principal,profile))!=reg:
            raise Refused('unregistered_converter')
        if installed is None:
            exe=verify_converter(reg,self.converter_image_digest); os.close(exe)
        root=self.export_reference_roots.get(principal)
        if root is None or str(root)!='/references/assets': raise Refused('unregistered_reference_mapping')
        a=context['plan']['assets'][0]
        if installed is not None:
            from worker.managed_reference import acquire
            try:
                fd = acquire(a['id'])
            except BaseException:
                self.failed = True
                raise
        else:
            fd=_open(Path(root)/a['id'],owner_uid=owner_uid)
        try:
            if installed is not None:
                from worker.managed_reference import check_fd
                check_fd(fd, owner_uid=owner_uid)
            info=os.fstat(fd)
            if not 0<info.st_size<=MAX_SOURCE: raise Refused('missing_media')
            raw=read_bounded(fd,MAX_SOURCE); measured=pcm(raw)
            after=os.fstat(fd)
            if installed is not None and (len(raw) != info.st_size or
                    any(getattr(info,k) != getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))):
                raise Refused('missing_media')
            if any(a[k]!=v for k,v in measured.items()): raise Refused('missing_media')
            os.lseek(fd,0,0)
            return fd
        except BaseException:
            os.close(fd); raise

    def verify_pocket(self, context, principal, *, owner_uid=10001, retain_fd=False):
        """Independent exact reference allowlist under operator-declared RO mounts.

        Roots are supervisor configuration, never allocation/request paths.
        Installed policy fixes the read-only assets mount and uses a reaped
        UID10001 helper plus an O_RDONLY descriptor, never a DB mount.
        """
        from worker.runtime_protocol import execution
        from worker.engines.pocket import verify_inputs, _open
        spec = execution(context)
        roots = getattr(self, 'pocket_reference_roots', {})
        if principal in roots:
            root = Path(roots[principal])
            if not root.is_absolute() or '..' in root.parts: raise Refused('unregistered_reference_mapping')
            if not getattr(self, 'installed_pocket', False):
                fd = _open(root, directory=True, owner_uid=owner_uid)
                os.close(fd)
            expected = {spec.reference_id:'/references/assets/'+spec.reference_id}
            if context['execution']['references'] != expected: raise Refused('unregistered_reference_mapping')
            references = {spec.reference_id:root/spec.reference_id}
        else:
            if context['execution']['references'] != self.references:
                raise Refused('unregistered_reference_mapping')
            references = self.references
        if getattr(self, 'installed_pocket', False):
            if principal not in roots or str(roots[principal]) != '/references/assets':
                raise Refused('unregistered_reference_mapping')
            from worker.managed_reference import acquire
            try:
                fd = acquire(spec.reference_id)
            except BaseException:
                self.failed = True  # No new lane after an uncertain helper boundary.
                raise
            try:
                verify_inputs(spec, self.bundles, references, owner_uid=owner_uid, reference_fd=fd)
                if retain_fd:
                    result, fd = fd, None
                    return result
            finally:
                if fd is not None: os.close(fd)
        else:
            verify_inputs(spec, self.bundles, references, owner_uid=owner_uid)

    def verify_qwen(self, context, profiles, principal, *, retain_fd=False):
        from worker.engines.qwen_execution import current_hardware, inventory_wire
        from worker.engines.qwen import verify_service_inputs
        from worker.runtime_protocol import execution
        from contracts import validate as oracle
        spec=execution(context)
        selected_bundles = {spec.bundle_id:self.bundles[spec.bundle_id]} if self.resident_qwen and spec.bundle_id in self.bundles else self.bundles
        profile=context['plan']['resolved']['profile']
        managed=self.managed_qwen(spec)
        if managed:
            # Explicit operator registration of the WHOLE trusted template. Existing
            # exact-version registrations never implicitly authorize this lane.
            template=self.qwen_contexts.get((principal,'qwen-cpu','managed-clone'))
            if template is None: raise Refused('unregistered_qwen_context')
            from service.execution_context import cpu_clone_context
            plan=context['plan']
            version=plan['voice_versions'][0] if plan['operation']=='speech_generate' else None
            expected=cpu_clone_context(template,principal,plan['request'],plan['assets'][0],version=version)
            # Production assigns pair IDs after owner resolution. Corroborate
            # every other plan field, including frozen spoken text and budgets.
            if plan['operation']=='voice_prepare':
                for field in ('reference_asset_id','audition_asset_id'):
                    expected['plan']['reference_roles'][field]=plan['reference_roles'][field]
                expected['plan']['resolved']['reuse_configuration']['reference_asset_id']=plan['reference_roles']['reference_asset_id']
            if expected != context: raise Refused('unregistered_qwen_context')
        elif self.qwen_contexts.get((principal, profile['id'], context['plan']['request']['voice_version_id'])) != context['execution']:
            raise Refused('unregistered_qwen_context')
        if context['execution']['hardware'] != inventory_wire(self.inventory) or profiles != self.qwen_profiles:
            raise Refused('unregistered_qwen_context')
        oracle.admission(dict(operation=context['plan']['operation'],input_sha256=oracle.plan_hash(context['plan'])),context,self.qwen_profiles)
        selected=next(p for p in self.qwen_profiles if p['id'] == profile['id'])
        if selected['verification'] != 'measured' or not selected['qualification_receipt'] or 'synthetic' in selected['qualification_receipt'].lower(): raise Refused('real_qualification_required')
        from service import install_policy as ip
        if getattr(self, 'installed_policy', None) is ip.QWEN_SPLIT:
            from service.installed_qwen_daemon import FAMILY
            from worker.engines.qwen_execution import installed_cpu_hardware
            if self.resident_qwen:
                from worker.runtime_protocol import resident_qwen_cpu_hardware
                resident_qwen_cpu_hardware(spec,self.inventory)
            else:
                installed_cpu_hardware(spec,self.inventory,family=FAMILY)
        else: current_hardware(spec,self.inventory)
        if managed:
            from worker.managed_reference import acquire
            try: fd=acquire(spec.reference_id)
            except BaseException:
                self.failed=True
                raise
            try:
                verify_service_inputs(spec,selected_bundles,context['execution']['references'],self.inventory,owner_uid=10001,reference_fd=fd)
                if retain_fd:
                    result,fd=fd,None
                    return result
            finally:
                if fd is not None: os.close(fd)
        else:
            if context['execution']['references'] != self.references: raise Refused('unregistered_reference_mapping')
            verify_service_inputs(spec,selected_bundles,self.references,self.inventory,owner_uid=10001)

    def conditioning(self, row):
        stage = row['allocation']['snapshot'][0]['attempts'][-1]['staging_id']
        from service import install_policy as ip
        from worker.runtime_protocol import execution
        spec=execution(row['allocation']['snapshot'][1])
        if spec is not None and (spec.engine=='pocket-tts' or self.managed_qwen(spec)):
            command=['python','-c','from worker.runtime_conditioning import pocket_main; pocket_main()',stage,str(row['uid'])]
        else:
            command = ['python', '-m', 'worker.runtime_conditioning', stage, str(row['uid'])]
        if getattr(self, 'installed_policy', None) in (ip.POCKET_SPLIT, ip.QWEN_SPLIT):
            command.append(self.installed_policy.name)
        proc = subprocess.Popen(command,
            user=10001, group=row['uid'], extra_groups=[], close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            code = proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                self.active['helper_unknown'] = True
            raise Refused('conditioning_timeout')
        except Exception:
            # A failed exact wait is not death proof. Attempt bounded kill/wait.
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                self.active['helper_unknown'] = True
            raise Refused('conditioning_wait_unknown')
        if code != 0:
            raise Refused('conditioning_refused')

    def client(self, connection):
        entered = time.monotonic()
        cutoff = entered+2
        try:
            authenticate(connection, session(self.secret, 10001, 10001), cutoff, control=False)
            op, body = request(receive(connection, cutoff))
            if (op == 'start' and ((getattr(self, '_pocket_daemon', False) and
                    getattr(self, 'installed_pocket', False)) or getattr(self, '_qwen_daemon', False))):
                if set(body) != {'runtime_id'} or not isinstance(body['runtime_id'], str):
                    raise Refused('closed_identity_body')
                from worker.runtime_protocol import POCKET_START_RPC_SECONDS
                cutoff = entered+POCKET_START_RPC_SECONDS
            answer = queue.Queue(1)
            self.commands.put_nowait((op, body, answer))
            result = answer.get(timeout=max(.001, cutoff-time.monotonic()))
            send(connection, result, cutoff)
        except Exception:
            pass
        finally:
            connection.close()
            self.slots.release()

    def execution_client(self, connection):
        """Accepted W1 execute/result endpoint: replay only, never fork/start/stop."""
        try:
            rows = self.journal.value['records']
            if not rows:
                raise Refused('no_allocation')
            row = rows[-1]
            cutoff = min(time.monotonic()+2, self.cutoffs.get(row['id'], 0))
            allocation = row['allocation']
            authenticate(connection, execution_session(self.execution_secret, allocation['binding'], 10001, 10001), cutoff, control=False)
            message = receive(connection, cutoff)
            from contracts import validate as oracle
            job, context, assets, profiles = allocation['snapshot']
            oracle.worker_message(message, job, assets, transport_owner=job['attempts'][-1]['owner_id'], context=context, current_profiles=profiles)
            if message['kind'] != 'execute':
                raise Refused('execute_required')
            # W1 exact validation, plus execute's empty output set/loading stage.
            if message['stage'] != 'loading' or message['result_asset_ids']:
                raise Refused('frozen_execute_required')
            while time.monotonic() < cutoff:
                current = self.lookup(row['id'])
                proposal = current.get('proposal')
                active = self.active
                if active and active['id'] == row['id']:
                    proposal = active['proposal']
                if proposal is not None:
                    send(connection, proposal, cutoff)
                    return
                time.sleep(.01)
        except Exception:
            pass
        finally:
            connection.close()
            self.slots.release()

    def serve(self):
        path = self.ipc / 'lifecycle.sock'
        path.unlink(missing_ok=True)
        execution_path = self.ipc / 'worker.sock'
        execution_path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as execution:
            listener.bind(str(path)); path.chmod(0o660)
            listener.listen(8); listener.settimeout(.02)
            execution.bind(str(execution_path)); execution_path.chmod(0o660)
            execution.listen(8); execution.setblocking(False)
            while True:
                self.tick()
                if self.halt and self.active is None:
                    if self.resident is not None:
                        self.signal_namespace(signal.SIGTERM)
                        limit = time.monotonic()+.2
                        while time.monotonic() < limit:
                            try:
                                pid, _ = os.waitpid(-1, os.WNOHANG)
                            except ChildProcessError:
                                self.resident = None
                                break
                            if pid == 0: time.sleep(.01)
                        if self.resident is not None:
                            self.signal_namespace(signal.SIGKILL)
                            while True:
                                try: os.waitpid(-1, 0)
                                except ChildProcessError: break
                            self.resident = None
                    return
                try:
                    op, body, answer = self.commands.get_nowait()
                    try:
                        result = dict(ok=True, value=self.operation(op, body))
                    except Exception:
                        result = dict(ok=False, error='lifecycle_refused_or_quarantined')
                    answer.put_nowait(result)
                except queue.Empty:
                    pass
                try:
                    connection, _ = listener.accept()
                    if self.slots.acquire(blocking=False):
                        threading.Thread(target=self.client, args=(connection,), daemon=True).start()
                    else:
                        connection.close()
                except socket.timeout:
                    pass
                try:
                    connection, _ = execution.accept()
                    if self.slots.acquire(blocking=False):
                        threading.Thread(target=self.execution_client, args=(connection,), daemon=True).start()
                    else:
                        connection.close()
                except BlockingIOError:
                    pass


def export_startup(value, expected_worker_image, *, owner_uid=0):
    """Explicit trusted startup input; Docker image corroboration belongs upstream.

    No readiness or measured capability is conferred by validating this seal.
    """
    from service import installed_export as ex
    config = ex.configuration(value)
    if type(expected_worker_image) is not str or expected_worker_image != config['images']['worker']:
        raise ValueError('trusted_worker_image_required')
    fd = ex.verify_local_converter(config, owner_uid=owner_uid)
    os.close(fd)
    return config


def main(*, _pocket_daemon=False, _qwen_daemon=False):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed-policy', choices=('docker-kokoro-split-v1','docker-pocket-split-v1','docker-pocket-split-v2','docker-kokoro-export-split-v2','docker-qwen-cpu-split-v1'))
    parser.add_argument('--configuration-file')
    parser.add_argument('--expected-worker-image')
    parser.add_argument('--resident-qwen-policy', choices=('qwen-resident-cpu-v1',))
    args = parser.parse_args()
    if _pocket_daemon and args.installed_policy != 'docker-pocket-split-v2':
        raise ValueError('pocket_daemon_v2_configuration_required')
    if _qwen_daemon != (args.installed_policy == 'docker-qwen-cpu-split-v1'):
        raise ValueError('explicit_qwen_daemon_required')
    if args.resident_qwen_policy is not None and not _qwen_daemon:
        raise ValueError('resident_qwen_requires_explicit_qwen_daemon')
    os.umask(0o077)
    try:
        bundles = None
        pocket_roots = None
        installed_pocket = False
        installed_policy = None
        export_config = None
        if args.installed_policy is not None:
            from service import install_policy as ip
            from service.execution_context import installed_configuration, installed_mapping
            if args.configuration_file is None: raise ValueError('installed_configuration_required')
            if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (0, 0, 10001, 10001) or os.getgroups() not in ([], [10001]):
                raise ValueError('installed_worker_identity')
            from service import installed_export as ex
            if args.installed_policy == ex.POLICY:
                from service.auth import private_input
                from service.recovery import host_json
                export_config = export_startup(host_json(private_input(args.configuration_file, mode=0o440, limit=48000)), args.expected_worker_image)
                config = {k: copy.deepcopy(v) for k,v in export_config.items() if k != 'audio_export'}
                config.update(version=1, ownership_policy=ip.POLICY)
            else:
                if args.expected_worker_image is not None: raise ValueError('explicit_export_family_required')
                config = installed_configuration(args.configuration_file, policy=ip.SPLIT)
            if (export_config or config)['ownership_policy'] != args.installed_policy: raise ValueError('installed_worker_policy_binding')
            if _pocket_daemon:
                from service import installed_pocket_daemon as pd
                pd.configuration(config)
            installed_policy = ip.configuration_policy(config)
            bundles = installed_mapping(config, policy=installed_policy)
            installed_pocket = ip.pocket(config)
            if installed_pocket: pocket_roots = {p:'/references/assets' for p in config['contexts']}
            ip.directory_identity('/receipts', policy=installed_policy, role='receipts')
            ip.directory_identity('/ipc', policy=installed_policy, role='ipc')
            for name, limit in (('lane.lock', 0), ('journal.json', 1024**2)):
                ip.read_private(Path('/receipts')/name, limit, policy=installed_policy, role='receipts')
            for name in ('lifecycle.key', 'execution.key'):
                ip.read_private(Path('/ipc')/name, 4096, policy=installed_policy, role='ipc')
        elif args.configuration_file is not None or args.expected_worker_image is not None:
            raise ValueError('explicit_fixed_install_policy_required')
        supervisor = RuntimeSupervisor(bundles=bundles, pocket_reference_roots=pocket_roots,
            installed_pocket=installed_pocket, installed_policy=installed_policy, export_configuration=export_config,
            expected_worker_image=args.expected_worker_image, resident_qwen_policy=args.resident_qwen_policy)
        if _pocket_daemon and not installed_pocket: raise ValueError('pocket_daemon_v2_configuration_required')
        if _qwen_daemon:
            from service import installed_qwen_daemon as qd
            from worker.engines.qwen_execution import inventory_from_wire, installed_cpu_hardware
            qd.configuration(config)
            inventory=None
            supervisor.qwen_profiles=copy.deepcopy(config['profiles'])
            for principal,context in config['contexts'].items():
                observed=inventory_from_wire(context['execution']['hardware'])
                selected=execution(context)
                if args.resident_qwen_policy is not None:
                    from worker.runtime_protocol import resident_qwen_cpu_hardware
                    resident_qwen_cpu_hardware(selected,observed)
                else:
                    installed_cpu_hardware(selected,observed,family=qd.FAMILY)
                if inventory is not None and inventory != observed: raise ValueError('hardware_identity')
                inventory=observed
                if supervisor.managed_qwen(selected):
                    supervisor.qwen_contexts[(principal,'qwen-cpu','managed-clone')]=copy.deepcopy(context)
                elif selected.task == 'custom_voice':
                    supervisor.qwen_contexts[(principal,'qwen-cpu',selected.voice_revision)]=copy.deepcopy(context['execution'])
                else:
                    raise ValueError('installed_qwen_context_task')
            supervisor.inventory=inventory
        supervisor._qwen_daemon = _qwen_daemon
        supervisor._pocket_daemon = _pocket_daemon
        supervisor.serve()
    except (OSError, ValueError, KeyError, TypeError):
        print('lifecycle supervisor unavailable; lane quarantined', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()


def pocket_daemon_main():
    return main(_pocket_daemon=True)


def qwen_daemon_main():
    return main(_qwen_daemon=True)
