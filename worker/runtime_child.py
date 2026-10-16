"""Fixed synthetic engine child. No management capability or service imports."""
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from worker.transport import receive
from worker.isolated_synthetic import run


def main():
    connection = socket.socket(fileno=int(sys.argv[1]))
    config = receive(connection, time.monotonic() + 2)
    mode = config.pop('fixture')
    protected_stages = config.pop('protected_stages', [])
    if mode in ('busy', 'descendants'):
        leader = os.getpid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mode == 'descendants':
            if os.fork() == 0:
                os.setsid()
                if os.fork() != 0:
                    os._exit(0)
        if os.getpid() == leader:
            Path('/staging/'+config['job']['attempts'][-1]['staging_id']+'/ready').write_text('synthetic_fault_ready')
        while True:
            pass
    if mode == 'permissions':
        status = Path('/proc/self/status').read_text()
        assert 'CapEff:\t0000000000000000' in status
        assert 'CapPrm:\t0000000000000000' in status
        assert 'NoNewPrivs:\t1' in status
        assert os.getgroups() == []
        for stage in protected_stages:
            try:
                os.rename('/staging/'+stage, '/staging/'+stage+'-forbidden')
            except PermissionError:
                pass
            else:
                raise RuntimeError('other_attempt_rename_allowed')
            try:
                open('/staging/'+stage+'/output.wav', 'rb').close()
            except (PermissionError, FileNotFoundError):
                continue
            raise RuntimeError('other_attempt_accessible')
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect('/ipc/lifecycle.sock')
            except PermissionError:
                pass
            else:
                raise RuntimeError('management_socket_accessible')
        for path in ('/ipc/lifecycle.key', '/ipc/execution.key', '/receipts/journal.json', '/control/owner.db', '/publication'):
            try:
                open(path, 'rb').close()
            except (PermissionError, FileNotFoundError):
                continue
            raise RuntimeError('protected_path_accessible')
        try:
            os.kill(1, 0)
        except PermissionError:
            pass
        else:
            raise RuntimeError('supervisor_signal_allowed')
        assert not any(int(p.name) > 2 and int(p.name) != connection.fileno() for p in Path('/proc/self/fd').iterdir() if p.name.isdigit() and Path(p).exists())
    run(connection, config, '/staging', time.monotonic() + 30)
    os.chmod('/staging/'+config['job']['attempts'][-1]['staging_id']+'/output.wav', 0o644)


if __name__ == '__main__':
    main()
