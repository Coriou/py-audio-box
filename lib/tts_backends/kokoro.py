"""Kokoro preset adapter. No torch/Qwen dependency and no automatic downloads."""
import time
from numbers import Integral
from .contract import AudioResult, BackendError, Capabilities
from .assets import verified_bundle


class KokoroBackend:
    def __init__(self, assets, language='fr', threads=2):
        if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
            raise BackendError('invalid_threads', 'Thread count must be a positive integer')
        self.root, self.manifest = verified_bundle(assets, 'kokoro')
        if not {'model.onnx', 'voices.bin'}.issubset({entry['path'] for entry in self.manifest['files']}):
            raise BackendError('invalid_assets', 'Kokoro bundle must verify model.onnx and voices.bin')
        self.language = {'fr': 'fr-fr', 'en': 'en-us'}.get(language)
        if self.language is None:
            raise BackendError('unsupported_language', 'Choose fr or en explicitly')
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Release variable-length inference buffers rather than retain arena reserves.
        options.enable_cpu_mem_arena = False
        started = time.monotonic()
        session = ort.InferenceSession(str(self.root / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self._providers = tuple(session.get_providers())
        if self._providers != ('CPUExecutionProvider',):
            raise BackendError('unsupported_provider', 'Kokoro CPU requires only CPUExecutionProvider')
        self.model = Kokoro.from_session(session, str(self.root / 'voices.bin'))
        self._voices = frozenset(self.model.get_voices())
        self.load_seconds = time.monotonic() - started
        self._requests_started = 0
        self.capabilities = Capabilities('kokoro', self.manifest['checkpoint'], ('preset',), False, ('fr', 'en'))

    def synthesise(self, text, *, preset=None, reference=None, instruct=None, seed=None):
        if reference or instruct or seed is not None:
            raise BackendError('unsupported_parameter', 'Kokoro supports presets; clone, instructions and seed are unavailable')
        if not preset or preset not in self._voices:
            raise BackendError('invalid_preset', 'Choose an available Kokoro preset explicitly')
        prefix = 'f' if self.language == 'fr-fr' else 'a'
        if not preset.startswith(prefix):
            raise BackendError('preset_language_mismatch', 'Preset must match the selected language')
        self._requests_started += 1
        started = time.monotonic()
        samples, rate = self.model.create(text, voice=preset, lang=self.language)
        result = AudioResult(samples, rate, 1, {
            'backend': 'kokoro', 'runtime': 'onnxruntime', 'providers': list(self._providers),
            'cpu_mem_arena': False,
            'checkpoint': self.manifest['checkpoint'], 'text_submitted': text, 'preset': preset,
            'generation_seconds': time.monotonic() - started,
            'model_load_seconds': self.load_seconds,
            'load_seconds': self.load_seconds if self._requests_started == 1 else 0.0,
            'model_reused': self._requests_started > 1,
            'request_index': self._requests_started,
        })
        return result

    def close(self):
        self.model = None
