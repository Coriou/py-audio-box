"""Lazy backend discovery: importing this module loads no inference runtime."""
from .contract import AudioResult, BackendError, Capabilities


def get_backend(name: str):
    if name in ('qwen', 'qwen3'):
        from .qwen import QwenBackend
        return QwenBackend
    if name == 'pocket-tts':
        from .pocket_tts import PocketBackend
        return PocketBackend
    if name == 'kokoro':
        from .kokoro import KokoroBackend
        return KokoroBackend
    raise BackendError('backend_unavailable', f'Backend {name!r} is not installed/validated')
