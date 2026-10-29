"""A socket-like errno from a WAV read is not a network outcome."""
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clients.http import HTTP
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
                    raise OSError(errno.ETIMEDOUT, 'local read timeout')

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
                        raise OSError(errno.ETIMEDOUT, 'local read timeout')
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
