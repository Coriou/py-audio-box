#!/usr/bin/env python3
"""Run a task command in its own group; the parent enforces wall-clock timeout.

Does not serialize environment, credentials, command arguments, or captured output
into the receipt. The output file should be used only for non-secret commands.
The deadline watches the direct child. Use a non-daemonizing client; surviving
descendants after a normal leader exit are outside this helper's guarantee.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run(command, limit, grace, log_path, receipt_path):
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError('Refusing to overwrite a receipt or log')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    began = time.monotonic()
    timed_out = False
    signals = []
    signal_errors = []
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            try:
                process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                for sig, wait in [(signal.SIGTERM, grace), (signal.SIGKILL, None)]:
                    try:
                        os.killpg(process.pid, sig)
                        signals.append(signal.Signals(sig).name)
                    except (ProcessLookupError, PermissionError) as error:
                        signal_errors.append({'signal': signal.Signals(sig).name,
                                              'error': type(error).__name__})
                    if wait is not None:
                        # Do not reap the leader before the last group signal.
                        # macOS may report EPERM for a group after leader reap;
                        # EPERM is not evidence that the group is absent.
                        time.sleep(wait)
                process.wait(timeout=5)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError) as error:
                    signal_errors.append({'signal': 'SIGKILL-finally',
                                          'error': type(error).__name__})
                    process.kill()  # The direct child is still ours to terminate.
                process.wait(timeout=5)
    inventory = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid='],
                               text=True, capture_output=True, timeout=3)
    group_members = []
    if inventory.returncode == 0:
        for line in inventory.stdout.splitlines():
            pid, ppid, pgid = map(int, line.split())
            if pgid == process.pid:
                group_members.append({'pid': pid, 'ppid': ppid, 'pgid': pgid})
    receipt = {'started_at': started,
               'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'elapsed_seconds': round(time.monotonic() - began, 3),
               'limit_seconds': limit, 'kill_grace_seconds': grace,
               'timed_out': timed_out, 'signals_sent': signals,
               'group_signal_errors': signal_errors,
               'client_returncode': process.returncode, 'client_reaped': True,
               'group_inventory_available': inventory.returncode == 0,
               'group_members_after_client_reap': group_members,
               'log_bytes': log_path.stat().st_size,
               'scope': 'client process group; daemon-side completion requires separate verification'}
    with receipt_path.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    return 124 if timed_out else process.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=float, required=True)
    parser.add_argument('--grace', type=float, default=2)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    if not args.command or args.limit <= 0 or args.grace < 0:
        parser.error('A command, positive limit and nonnegative grace are required')
    raise SystemExit(run(args.command, args.limit, args.grace, args.log, args.receipt))
