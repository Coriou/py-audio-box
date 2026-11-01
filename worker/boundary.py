"""0C UDS connectivity probe ONLY; not a job worker or alternate worker schema.

The future execute/result protocol is contracts 1.3.0. No JSON execution request
is accepted here. No DB, credentials, managed publication root or models.
"""
import os
from pathlib import Path
import socket


def main():
    os.umask(0o077)
    directory = Path('/run/speech-ipc')
    directory.mkdir(exist_ok=True)
    directory.chmod(0o700)
    path = directory / 'boundary.sock'
    # Sole container owns this namespace; stale socket from its previous stop.
    path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path)); path.chmod(0o600); listener.listen(1)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(2)
                try:
                    if connection.recv(16) == b'0C-PING\n':
                        connection.sendall(b'0C-BOUNDARY-ONLY\n')
                except (TimeoutError, OSError):
                    pass

if __name__ == '__main__': main()
