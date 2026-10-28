"""No-weight native Mac CPU prerequisites; no worker admission or synthesis claim."""
import ctypes
from importlib import metadata
from pathlib import Path
import platform
import sys

from .contract import BackendError

EXPECTED = {'kokoro-onnx': '0.6.1', 'onnxruntime': '1.31.0',
            'espeakng-loader': '0.2.4', 'phonemizer': '3.4.0'}
INSTALL = 'Use native CPython 3.11 and pip install --require-hashes --only-binary=:all: -r requirements/kokoro.lock in a fresh virtual environment.'

# ONNX IR8/opset13: float32[2] X -> Add(X,X) -> Y. No initializers/weights.
# Fixed protobuf fixture following onnx/onnx.proto; avoids a new onnx dependency.


def cpu_fixture():
    # Each field is <one-byte tag, one-byte length, payload>; all payloads <128B.
    def field(tag, payload):
        return bytes((tag, len(payload))) + payload
    tensor_type = field(0x0a, b'\x08\x01' + field(0x12, field(0x0a, b'\x08\x02')))
    def value(name):
        return field(0x0a, name) + field(0x12, tensor_type)
    node = field(0x0a, b'X') * 2 + field(0x12, b'Y') + field(0x22, b'Add')
    graph = field(0x0a, node) + field(0x12, b'cpu-preflight') + field(0x5a, value(b'X')) + field(0x62, value(b'Y'))
    return b'\x08\x08' + field(0x42, b'\x10\x0d') + field(0x3a, graph)


def validate_host():
    if (platform.system(), platform.machine(), platform.python_implementation(), sys.version_info[:2]) != ('Darwin', 'arm64', 'CPython', (3, 11)):
        raise BackendError('native_kokoro_platform', 'Kokoro CPU preflight requires native Apple arm64 CPython 3.11; leave Rosetta. ' + INSTALL)


def validate_versions():
    for name, expected in EXPECTED.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise BackendError('native_kokoro_dependency', f'Missing {name}. ' + INSTALL) from exc
        if actual != expected:
            raise BackendError('native_kokoro_dependency', f'{name} requires pinned {expected}; found {actual}. ' + INSTALL)


def espeak_paths(loader):
    root = Path(loader.__file__).resolve().parent
    library = Path(loader.get_library_path()).resolve()
    data = Path(loader.get_data_path()).resolve()
    if (library.parent != root or library.name != 'libespeak-ng.dylib' or
            not library.is_file() or data.parent != root or data.name != 'espeak-ng-data' or
            not (data/'en_dict').is_file()):
        raise BackendError('native_kokoro_espeak', 'Pinned eSpeak NG library/data are incomplete. ' + INSTALL)
    try:
        ctypes.CDLL(str(library))
    except OSError as exc:
        raise BackendError('native_kokoro_espeak', 'Cannot load the bundled arm64 eSpeak NG dylib; reinstall the locked wheel in a native interpreter. ' + INSTALL) from exc
    return library, data


def preflight():
    validate_host()
    validate_versions()
    try:
        import numpy as np
        import onnxruntime as ort
        import espeakng_loader
        from kokoro_onnx import Kokoro
        from kokoro_onnx.config import EspeakConfig
        from kokoro_onnx.tokenizer import Tokenizer
        from .kokoro import KokoroBackend
        library, data = espeak_paths(espeakng_loader)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(cpu_fixture(), sess_options=options, providers=['CPUExecutionProvider'])
        providers = session.get_providers()
        if providers != ['CPUExecutionProvider']:
            raise BackendError('native_kokoro_provider', 'The CPU candidate requires CPUExecutionProvider only.')
        output = session.run(None, {'X': np.array([1., 2.], dtype=np.float32)})[0]
        if not np.array_equal(output, np.array([2., 4.], dtype=np.float32)):
            raise BackendError('native_kokoro_cpu_operation', 'ONNX CPU operation returned an invalid result.')
        tokenizer = Tokenizer(EspeakConfig(lib_path=str(library), data_path=str(data)))
        text = 'Hello. The test is complete.'
        phonemes = tokenizer.phonemize(text, lang='en-us')
        tokens = tokenizer.tokenize(phonemes)
        if not phonemes or phonemes == text or not tokens or not all(type(x) is int and x >= 0 for x in tokens):
            raise BackendError('native_kokoro_phonemization', 'Bundled eSpeak NG did not produce usable English phonemes.')
        if not callable(Kokoro.from_session) or not callable(KokoroBackend.synthesise):
            raise BackendError('native_kokoro_backend', 'Pinned Kokoro backend entry points are unavailable.')
        return dict(platform='Darwin', machine='arm64', versions={n: metadata.version(n) for n in EXPECTED},
            providers=providers, cpu_operation=output.tolist(), threads=2, inter_threads=1, cpu_mem_arena=False,
            espeak_library=str(library), espeak_data=str(data), phonemes=phonemes, token_count=len(tokens),
            capabilities=['preset'], cloning=False, weights_loaded=False, synthesis_qualified=False,
            installed_qualified=False)
    except BackendError:
        raise
    except (ImportError, OSError, RuntimeError) as exc:
        raise BackendError('native_kokoro_preflight', 'Kokoro CPU prerequisites failed; see the retained cause. ' + INSTALL) from exc
