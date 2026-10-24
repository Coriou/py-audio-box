"""Linux PID1 trusted single-lane parent/reaper; never imports service/engine code."""
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
from worker.runtime_protocol import authenticate, receive, send, session, request, execution_session
from worker.transport import canonical, Refused, Session


class RuntimeSupervisor:
    def __init__(self, root='/receipts', ipc='/ipc', *, bundles=None, references=None):
        if os.getpid() != 1:
            raise RuntimeError('pid1_required')
        from types import MappingProxyType
        self.bundles = MappingProxyType(dict(bundles or {}))
        self.references = MappingProxyType(dict(references or {}))
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

    def operation(self, op, body):
        if op == 'hello':
            if body:
                raise Refused('closed_body')
            value = dict(boot=self.boot, lane=self.journal.value['lane'], quarantined=self.failed or any(
                r['phase'] != 'terminal' and (not self.active or self.active['id'] != r['id']) for r in self.journal.value['records']), launches=self.launches)
            if self.journal.maintenance is not None: value['maintenance']=self.journal.maintenance
            return value
        if op == 'prepare':
            if set(body) != {'snapshot', 'binding', 'fixture'} or body['fixture'] not in ('silence', 'busy', 'descendants', 'permissions', 'kokoro', 'pocket'):
                raise Refused('closed_prepare')
            snapshot = body['snapshot']
            if not isinstance(snapshot, list) or len(snapshot) != 4 or hashlib.sha256(canonical(snapshot)).hexdigest() != body['binding']:
                raise Refused('authority_binding')
            from contracts import validate as oracle
            job, context, assets, profiles = snapshot
            oracle.stored_context(job, context)
            from worker.runtime_protocol import execution, dispatch_mode
            spec = execution(context)
            if (body['fixture'] in ('kokoro','pocket')) != (spec is not None) or (spec is not None and body['fixture'] != dispatch_mode(spec)): raise Refused('execution_dispatch_mismatch')
            # Independent direct-integration fence, before allocation or journal
            # mutation. Existing records remain observable/stoppable/collectable.
            if spec is not None and spec.engine == 'pocket-tts':
                raise Refused('pocket_restart_repair_required')
            if spec is not None and context['execution']['bundles'] != self.bundles:
                raise Refused('unregistered_mount_mapping')
            if spec is not None and spec.engine == 'pocket-tts':
                if context['execution']['references'] != self.references:
                    raise Refused('unregistered_reference_mapping')
                # Independent full byte verification before journaling or launch.
                from worker.engines.pocket import verify_inputs
                verify_inputs(spec, self.bundles, self.references, owner_uid=10001)
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
            anchored = time.monotonic()
            budget = (datetime.fromisoformat(job['deadline']) - datetime.now(timezone.utc)).total_seconds()
            if not 0 < budget <= 30:
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
            row = self.change(row, phase='started')  # Fence BEFORE any fork.
            job, context, assets, profiles = row['allocation']['snapshot']
            stage = job['attempts'][-1]['staging_id']
            uid = row['uid']
            # Nonresident setup helper, reaped before engine launch. No extra caps.
            subprocess.run(['python', '-c', 'import os,sys; os.umask(0); os.mkdir(sys.argv[1],0o770)',
                            '/staging/'+stage], user=10001, group=uid, extra_groups=[],
                           close_fds=True, check=True, timeout=1)
            if time.monotonic() >= self.active['cutoff']:
                raise Refused('expired_before_launch')
            parent, child = socket.socketpair()
            key = secrets.token_hex(32)
            config = dict(job=job, context=context, allowed_assets=assets, profiles=profiles,
                          secret=key, control_uid=0, gid=10001, fixture=row['allocation']['fixture'],
                          protected_stages=[r['allocation']['snapshot'][0]['attempts'][-1]['staging_id'] for r in self.journal.value['records'] if r['id'] != row['id']])
            try:
                proc = subprocess.Popen(['python', '-m', 'worker.runtime_child', str(child.fileno())],
                    user=uid, group=uid, extra_groups=[], start_new_session=True,
                    close_fds=True, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.active.update(started=True, leader=proc.pid, process=proc)
                self.launches += 1
            finally:
                child.close()
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
            if active is not None and active['stopping'] is None:
                active['stopping'] = time.monotonic()
                self.signal_namespace(signal.SIGTERM)

    def watchdog_tick(self):
        # No UDS, journal, generator, subprocess wait or receipt I/O in this loop.
        # Lock covers only identity selection and signal; old allocations cannot
        # signal a newly admitted lane after the main reaper retires their handle.
        with self.tree_lock:
            active = self.active
            if active is None:
                return
            now = time.monotonic()
            if (now >= active['cutoff'] or self.halt) and active['stopping'] is None:
                active['stopping'] = now
                self.signal_namespace(signal.SIGTERM)
            if active['stopping'] is not None and now-active['stopping'] >= .2:
                self.signal_namespace(signal.SIGKILL)

    def watchdog(self):
        while True:
            self.watchdog_tick()
            time.sleep(.01)

    def tick(self):
        active = self.active
        if active is None:
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
                active['reaped_children'] += 1
                if pid == active['leader']:
                    active['process'].returncode = os.waitstatus_to_exitcode(status)
                    self.stop_tree()  # Drain even detached orphan descendants.
            except ChildProcessError:
                empty = True
                break
        if empty and active['started'] and not active['receiver_done'].is_set():
            return
        if empty and (active['started'] or active['stopping'] is not None):
            # Ensure internal receiver completed before terminal persistence.
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
                active['proposed_at'] < active['cutoff'] and reaped_at < active['cutoff']),
                original_deadline=row['allocation']['snapshot'][0]['deadline'])
            try:
                self.change(row, phase='terminal', receipt=receipt, proposal=active['proposal'])
            except Exception:
                # Actual reap without durable receipt is NOT STOPPED authority.
                self.failed = True
            with self.tree_lock:
                self.active = None

    def client(self, connection):
        cutoff = time.monotonic()+2
        try:
            authenticate(connection, session(self.secret, 10001, 10001), cutoff, control=False)
            op, body = request(receive(connection, cutoff))
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed-policy', choices=('docker-kokoro-split-v1',))
    parser.add_argument('--configuration-file')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        bundles = None
        if args.installed_policy is not None:
            from service import install_policy as ip
            from service.execution_context import installed_configuration, installed_mapping
            if args.configuration_file is None: raise ValueError('installed_configuration_required')
            if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (0, 0, 10001, 10001) or os.getgroups() not in ([], [10001]):
                raise ValueError('installed_worker_identity')
            config = installed_configuration(args.configuration_file, policy=ip.SPLIT)
            bundles = installed_mapping(config, policy=ip.SPLIT)
            ip.directory_identity('/receipts', policy=ip.SPLIT, role='receipts')
            ip.directory_identity('/ipc', policy=ip.SPLIT, role='ipc')
            for name, limit in (('lane.lock', 0), ('journal.json', 1024**2)):
                ip.read_private(Path('/receipts')/name, limit, policy=ip.SPLIT, role='receipts')
            for name in ('lifecycle.key', 'execution.key'):
                ip.read_private(Path('/ipc')/name, 4096, policy=ip.SPLIT, role='ipc')
        elif args.configuration_file is not None:
            raise ValueError('explicit_fixed_install_policy_required')
        RuntimeSupervisor(bundles=bundles).serve()
    except (OSError, ValueError, KeyError, TypeError):
        print('lifecycle supervisor unavailable; lane quarantined', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
