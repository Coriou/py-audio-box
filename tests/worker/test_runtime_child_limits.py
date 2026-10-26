"""Trusted Kokoro library allowance does not alter the audio result budget."""
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lib.tts_backends.espeak_memfd import MAX_LIBRARY_BYTES
from worker.runtime_child import contain_kokoro


class RuntimeChildLimitsTests(unittest.TestCase):
    def test_kokoro_library_ceiling_is_trusted_and_audio_budget_unchanged(self):
        import resource
        spec = SimpleNamespace(engine='kokoro', memory_bytes=2 * 1024**3,
                               max_output_bytes=480044, threads=1)
        with patch('worker.runtime_protocol.execution', return_value=spec), \
             patch('resource.setrlimit') as limits, patch.dict('os.environ'):
            contain_kokoro({})
        self.assertEqual(limits.call_args_list[1].args,
                         (resource.RLIMIT_FSIZE, (MAX_LIBRARY_BYTES,) * 2))
        self.assertEqual(spec.max_output_bytes, 480044)
        self.assertEqual(limits.call_args_list[0].args,
                         (resource.RLIMIT_AS, (2 * 1024**3,) * 2))

    def test_other_engines_keep_their_existing_file_ceiling(self):
        import resource
        spec = SimpleNamespace(engine='pocket-tts', memory_bytes=2 * 1024**3,
                               max_output_bytes=480044, reference_bytes=900000, threads=1)
        with patch('worker.runtime_protocol.execution', return_value=spec), \
             patch('resource.setrlimit') as limits, patch.dict('os.environ'):
            contain_kokoro({})
        self.assertEqual(limits.call_args_list[1].args,
                         (resource.RLIMIT_FSIZE, (900000,) * 2))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux memfd kernel evidence only')
    def test_linux_memfd_obeys_file_limit_and_trusted_allowance(self):
        code = '''
import errno, json, os, resource, signal, sys
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
limit = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
fd = os.memfd_create('bounded-library-probe', os.MFD_CLOEXEC)
try:
    block = b'x' * 65536
    while os.fstat(fd).st_size < 600000:
        os.write(fd, block[:600000 - os.fstat(fd).st_size])
    print(json.dumps({'size': os.fstat(fd).st_size, 'error': None}))
except OSError as exc:
    print(json.dumps({'size': os.fstat(fd).st_size, 'error': exc.errno}))
finally:
    os.close(fd)
'''
        import errno
        def probe(limit):
            result = subprocess.run([sys.executable, '-c', code, str(limit)],
                                    capture_output=True, text=True, timeout=5, check=True)
            return json.loads(result.stdout)
        self.assertEqual(probe(480044), {'size': 480044, 'error': errno.EFBIG})
        self.assertEqual(probe(MAX_LIBRARY_BYTES), {'size': 600000, 'error': None})


class QwenOriginalCutoffTests(unittest.TestCase):
    def test_long_cpu_attempt_keeps_original_anchor_and_expiry(self):
        from datetime import datetime,timezone,timedelta
        from worker.runtime_child import qwen_cpu_cutoff
        now=datetime.now(timezone.utc)
        config=dict(qwen_cpu_cutoff=1800.,job=dict(deadline=(now+timedelta(seconds=1800)).isoformat()),
            context=dict(plan=dict(resolved=dict(profile=dict(limits=dict(deadline_seconds=1800))))))
        spec=SimpleNamespace(engine='qwen',device='cpu')
        with patch('worker.runtime_protocol.execution',return_value=spec),patch('worker.runtime_child.time.monotonic',return_value=100.):
            original=qwen_cpu_cutoff(config)
        with patch('worker.runtime_protocol.execution',return_value=spec),patch('worker.runtime_child.time.monotonic',return_value=140.):
            self.assertEqual(qwen_cpu_cutoff(config),original)
        with patch('worker.runtime_protocol.execution',return_value=spec),patch('worker.runtime_child.time.monotonic',return_value=1901.):
            self.assertLess(qwen_cpu_cutoff(config),1901.)
        # Restart cannot invent an anchor from the remaining wall-clock deadline.
        del config['qwen_cpu_cutoff']
        with patch('worker.runtime_protocol.execution',return_value=spec),self.assertRaisesRegex(ValueError,'original_deadline'):
            qwen_cpu_cutoff(config)
