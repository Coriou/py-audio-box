"""Isolate Linux eSpeak wrappers in sealed anonymous memory.

Preserve noexec mounts and the upstream per-wrapper library state isolation.
"""
import ctypes
import _ctypes
import fcntl
import os
import weakref
import sys
from functools import lru_cache
from importlib.metadata import version


@lru_cache(maxsize=1)
def install():
    if sys.platform != 'linux':
        return
    if version('phonemizer') != '3.4.0':
        raise RuntimeError('Unsupported phonemizer API for sealed eSpeak loading')
    from phonemizer.backend.espeak import wrapper
    original = wrapper.EspeakAPI

    class MemoryAPI(original):
        def __init__(self, library, data_path):
            self._library = None
            source_library = ctypes.CDLL(str(library))
            try:
                path = self._shared_library_path(source_library)
            finally:
                _ctypes.dlclose(source_library._handle)
            fd = os.memfd_create('speech-espeak', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                with open(path, 'rb') as source:
                    total = 0
                    while block := source.read(65536):
                        total += len(block)
                        if total > 32 * 1024 * 1024:
                            raise ValueError('espeak library exceeds bound')
                        view = memoryview(block)
                        while view:
                            written = os.write(fd, view)
                            if written <= 0:
                                raise OSError('Incomplete eSpeak library copy')
                            view = view[written:]
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                            fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                self._library = ctypes.CDLL('/proc/self/fd/' + str(fd))
                data = None if data_path is None else str(data_path).encode()
                if self._library.espeak_Initialize(2, 0, data, 0) <= 0:
                    raise RuntimeError('espeak initialization failed')
                self._library_path = path
                weakref.finalize(self, self._release, self._library, fd)
            except BaseException:
                if self._library is not None:
                    _ctypes.dlclose(self._library._handle)
                os.close(fd)
                raise

        @staticmethod
        def _release(library, fd):
            try:
                library.espeak_Terminate()
            finally:
                try:
                    _ctypes.dlclose(library._handle)
                finally:
                    os.close(fd)

    wrapper.EspeakAPI = MemoryAPI
