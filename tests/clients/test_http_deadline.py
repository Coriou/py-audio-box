"""Real socket coverage for an HTTP body that never finishes."""
import socket
import threading
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from httpcore import WriteError, WriteTimeout
from httpcore._backends.sync import SyncStream

from clients.http import HTTP, _DeadlineStream
from clients.workflow import speak
from service.catalogue import Failure


class HTTPDeadlineTests(unittest.TestCase):
    def test_partial_sends_recheck_absolute_deadline(self):
        class PartialSocket:
            def __init__(self):
                self.timeouts = []
                self.sent = 0

            def settimeout(self, timeout):
                self.timeouts.append(timeout)

            def send(self, buffer):
                time.sleep(.02)  # Intentionally ignores the socket timeout.
                self.sent += 1
                return 1

        sock = PartialSocket()
        start = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, 'workflow_deadline'):
            _DeadlineStream(SyncStream(sock), start + .05).write(b'abcd', timeout=1)
        self.assertLessEqual(sock.sent, 3)
        self.assertGreaterEqual(sock.sent, 2)
        self.assertLess(sock.timeouts[-1], sock.timeouts[0])

    def test_socket_write_errors_keep_httpcore_mapping(self):
        class BrokenSocket:
            def __init__(self, error):
                self.error = error

            def settimeout(self, timeout):
                pass

            def send(self, buffer):
                raise self.error

        for error, expected in ((socket.timeout('slow'), WriteTimeout),
                                (OSError('broken'), WriteError)):
            with self.subTest(error=error), self.assertRaises(expected):
                _DeadlineStream(SyncStream(BrokenSocket(error)), time.monotonic() + 1).write(b'x')

    def _assert_trickling_response(self, operation):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2)
        closed = threading.Event()
        errors = []

        def serve():
            try:
                with listener:
                    conn, _ = listener.accept()
                    with conn:
                        conn.settimeout(2)
                        request = b''
                        while b'\r\n\r\n' not in request:
                            request += conn.recv(4096)
                        conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                                     b'Content-Length: 100000\r\n\r\n')
                        for _ in range(100):
                            time.sleep(0.02)
                            try:
                                conn.sendall(b' ')
                            except (BrokenPipeError, ConnectionResetError):
                                closed.set()
                                return
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=serve, name='loopback-http-deadline')
        worker.start()
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(TimeoutError, 'workflow_deadline'):
                operation(HTTP(f'http://127.0.0.1:{listener.getsockname()[1]}'), start + 0.15)
            self.assertLess(time.monotonic() - start, 0.5)
        finally:
            listener.close()
            worker.join(2)
        self.assertFalse(worker.is_alive(), 'loopback server did not terminate')
        self.assertTrue(closed.is_set(), 'client socket was not closed at deadline')
        self.assertFalse(errors, errors)

    def test_get_trickling_body_stops_at_absolute_deadline(self):
        self._assert_trickling_response(lambda client, deadline:
            client.get('/slow', deadline=deadline))

    def test_upload_trickling_response_stops_at_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root).resolve() / 'reference.wav'
            source.write_bytes(b'RIFF' + b'\0' * 64)
            self._assert_trickling_response(lambda client, deadline:
                client.upload_file(source, deadline=deadline))

    def test_numeric_endpoint_skips_delayed_system_resolver(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        def serve():
            with listener:
                conn, _ = listener.accept()
                with conn:
                    conn.recv(4096)
                    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                                 b'Content-Length: 2\r\n\r\n{}')
        worker = threading.Thread(target=serve, name='numeric-http-deadline')
        worker.start()
        start = time.monotonic()
        try:
            with patch('socket.getaddrinfo', side_effect=lambda *args, **kwargs: (time.sleep(.271), [])[1]) as resolver:
                self.assertEqual(HTTP(f'http://127.0.0.1:{listener.getsockname()[1]}').get(
                    '/status', deadline=start + .05), {})
            resolver.assert_not_called()
            self.assertLess(time.monotonic() - start, .2)
        finally:
            listener.close()
            worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_hostname_deadline_rejected_without_resolution_or_mutation(self):
        with patch('socket.getaddrinfo', side_effect=lambda *args, **kwargs: time.sleep(.271)) as resolver:
            start = time.monotonic()
            for host in ('localhost', 'example.invalid'):
                with self.assertRaises(Failure) as raised:
                    HTTP(f'http://{host}:8780').get('/status', deadline=time.monotonic() + .05)
                self.assertEqual(raised.exception.body['code'], 'invalid_request')
                self.assertIn('127.0.0.1', raised.exception.body['action'])
            result = speak(HTTP('http://localhost:8780'), voice='bruce', text='Hello', seconds=.05)
            self.assertEqual((result['state'], result['code'], result['phase']),
                             ('failed', 'invalid_request', 'voice_resolve'))
            resolver.assert_not_called()
            self.assertLess(time.monotonic() - start, .2)


if __name__ == '__main__':
    unittest.main()
