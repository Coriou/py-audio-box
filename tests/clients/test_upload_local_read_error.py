"""A socket-like errno from a WAV read is not a network outcome."""
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clients.http import HTTP, _WorkflowDeadline
from clients.workflow import _outcome
from service.catalogue import Failure


class LocalReadErrorTests(unittest.TestCase):
    def test_local_timeout_before_first_chunk_is_file_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / 'reference.wav'
            source.write_bytes(b'RIFF' + b'\0' * 64)
            real_fdopen = os.fdopen

            class TimedOutFile:
                def __init__(self, fd):
                    self.file = real_fdopen(fd, 'rb')

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.file.close()

                def fileno(self):
                    return self.file.fileno()

                def read(self, size):
                    raise TimeoutError(errno.ETIMEDOUT, 'local read timeout')

            class Client:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            def consume(client, method, path, *, content, **kwargs):
                next(iter(content))
                self.fail('local read should have failed')

            with patch('clients.http.os.fdopen', side_effect=lambda fd, mode: TimedOutFile(fd)), \
                    patch('clients.http._client', return_value=Client()), \
                    patch('clients.http._body', side_effect=consume):
                with self.assertRaises(Failure) as caught:
                    HTTP().upload_file(source)
            self.assertEqual(caught.exception.body['code'], 'invalid_request')
            self.assertEqual(caught.exception.body['field'], 'file', caught.exception.body)
            self.assertNotIn('unknown', caught.exception.body['message'])
            outcome = _outcome('upload', caught.exception, {}, mutation=True)
            self.assertEqual(outcome['state'], 'failed')
            self.assertNotIn('unknown', outcome['action'].lower())

    def test_local_error_after_first_chunk_keeps_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / 'reference.wav'
            source.write_bytes(b'RIFF' + b'\0' * 65536)
            real_fdopen = os.fdopen

            class InterruptedFile:
                def __init__(self, fd):
                    self.file = real_fdopen(fd, 'rb')
                    self.reads = 0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.file.close()

                def fileno(self):
                    return self.file.fileno()

                def read(self, size):
                    self.reads += 1
                    if self.reads > 1:
                        raise TimeoutError(errno.ETIMEDOUT, 'local read timeout')
                    return self.file.read(size)

            class Client:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            def consume(client, method, path, *, content, **kwargs):
                list(content)
                self.fail('second local read should have failed')

            with patch('clients.http.os.fdopen', side_effect=lambda fd, mode: InterruptedFile(fd)), \
                    patch('clients.http._client', return_value=Client()), \
                    patch('clients.http._body', side_effect=consume):
                with self.assertRaises(Failure) as caught:
                    HTTP().upload_file(source)
            self.assertEqual(caught.exception.body['code'], 'not_ready')
            self.assertIn('unknown', caught.exception.body['message'])

    def test_absolute_deadline_before_and_after_first_chunk_reaches_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / 'reference.wav'
            source.write_bytes(b'RIFF' + b'\0' * 65536)
            real_fdopen = os.fdopen

            class Client:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            for expire_after in (0, 1):
                with self.subTest(expire_after=expire_after):
                    opened = []
                    received = []

                    def tracked_fdopen(fd, mode):
                        stream = real_fdopen(fd, mode)
                        opened.append(stream)
                        return stream

                    def consume(client, method, path, *, content, **kwargs):
                        iterator = iter(content)
                        if expire_after:
                            received.append(next(iterator))
                        next(iterator)
                        self.fail('deadline should have expired')

                    with patch('clients.http.os.fdopen', side_effect=tracked_fdopen), \
                            patch('clients.http._client', return_value=Client()), \
                            patch('clients.http._body', side_effect=consume), \
                            patch('clients.http.time.monotonic', side_effect=([122.0] * expire_after + [124.0])) as clock:
                        with self.assertRaisesRegex(_WorkflowDeadline, 'workflow_deadline') as caught:
                            HTTP().with_deadline(123.0).upload_file(source)
                    self.assertEqual(clock.call_count, expire_after + 1)
                    self.assertEqual(len(received), expire_after)
                    self.assertEqual(len(opened), 1)
                    self.assertTrue(opened[0].closed)
                    outcome = _outcome('upload', caught.exception, {}, mutation=True)
                    self.assertEqual((outcome['state'], outcome['code']),
                                     ('timed_out', 'deadline_exceeded'))
