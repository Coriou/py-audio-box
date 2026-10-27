"""Fixed non-descendant synthetic runtime. One execution; reconnect is replay only.

No DB, Docker API, service credentials or publication authority. A process restart
is forbidden by the harness operator (restart policy=no), never recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from datetime import datetime, timezone

from contracts import validate as oracle
from worker.transport import Session, authenticate, canonical, receive, send, Refused, Uncertain


def proposal(message, config, staging):
    # Reuse the accepted W1 generator over an internal authenticated socketpair.
    # It remains the sole fixed synthetic byte generator, invoked exactly once.
    import threading
    from worker.isolated_synthetic import run
    left, right = socket.socketpair()
    errors = []
    local = dict(config, control_uid=os.getuid(), gid=os.getgid())
    binding = hashlib.sha256(canonical([config['job'], config['context'], config['allowed_assets'], config['profiles']])).hexdigest()
    cutoff = time.monotonic() + 2
    def generate():
        try:
            with right:
                run(right, local, staging, cutoff)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    try:
        with left:
            authenticate(left, Session(config['secret'], binding, os.getuid(), os.getgid()), cutoff, control=True)
            send(left, message, cutoff)
            result = receive(left, cutoff)
    finally:
        thread.join(2)
    if errors or thread.is_alive():
        raise RuntimeError('synthetic_generation_failed')
    return result


def serve(config, ipc='/ipc', staging='/staging'):
    if config.get('ignore_term', False):
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    job = config['job']
    # Anchored once. No reconnect or clock update can extend this runtime.
    budget = max(0, (datetime.fromisoformat(job['deadline']) - datetime.now(timezone.utc)).total_seconds())
    cutoff = time.monotonic() + min(budget, 30)
    binding = hashlib.sha256(canonical([job, config['context'], config['allowed_assets'], config['profiles']])).hexdigest()
    session = Session(config['secret'], binding, config['control_uid'], config['gid'])
    # Process-wide watchdog also bounds blocked generation/authentication during
    # control absence. This fixed worker creates threads only, no descendants.
    import threading
    watchdog = threading.Timer(max(0, cutoff-time.monotonic()), lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    cached = None
    path = Path(ipc)/'worker.sock'
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        path.chmod(0o660)
        listener.listen(2)
        (Path(ipc)/'ready').touch()
        while time.monotonic() < cutoff:
            listener.settimeout(max(.001, cutoff-time.monotonic()))
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                break
            try:
                with connection:
                    authenticate(connection, session, cutoff, control=False)
                    message = receive(connection, cutoff)
                    oracle.worker_message(message, job, config['allowed_assets'], transport_owner=job['attempts'][-1]['owner_id'], context=config['context'], current_profiles=config['profiles'])
                    if message['kind'] != 'execute':
                        raise Refused('execute_required')
                    if cached is None:
                        # Exclusive marker also makes accidental process restart fail closed.
                        fd = os.open(Path(staging)/job['attempts'][-1]['staging_id']/'executed', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
                        os.close(fd)
                        cached = proposal(message, config, staging)
                    send(connection, cached, cutoff)
            except (OSError, ValueError, oracle.ContractError):
                # Never regenerate after a lost send. An invalid first execute writes nothing.
                continue


def main():
    os.umask(0o027)
    config = json.loads(Path('/ipc/session.json').read_text())
    serve(config)


if __name__ == '__main__':
    main()
