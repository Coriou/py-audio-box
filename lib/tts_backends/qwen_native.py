"""In-process Qwen model owner for the private native exec child only.

This module grants no launch, timeout, registration or publication authority.
The supervisor must own the calling process for its entire lifetime.
"""
import math
import os
from pathlib import Path

from .contract import AudioResult


class NativeQwenBackend:
    """Closed preset/clone routes; prompts never leave this model owner."""

    def __init__(self, *, model_name, num_threads, dtype_str, revision,
                 offline, device, attention, diagnostic_request_seconds=None, diagnostic_dtype=None):
        if diagnostic_dtype is not None and (type(diagnostic_dtype) is not str or diagnostic_dtype != 'float16' or diagnostic_request_seconds != 120 or dtype_str != diagnostic_dtype):
            raise ValueError('native_diagnostic_precision')
        expected_dtype = 'float32' if diagnostic_dtype is None else diagnostic_dtype
        if (type(model_name) is not str or not Path(model_name).is_absolute()
                or not Path(model_name).is_dir()
                or type(num_threads) is not int or not 1 <= num_threads <= 64
                or type(revision) is not str or not 0 < len(revision) <= 256
                or offline is not True
                or (device, dtype_str, attention) != ('mps', expected_dtype, 'sdpa')):
            raise ValueError('native_execution_required')
        from worker.native_budgets import REQUEST_SECONDS, DIAGNOSTIC_REQUEST_SECONDS
        if diagnostic_request_seconds is not None and (type(diagnostic_request_seconds) is not int or diagnostic_request_seconds != DIAGNOSTIC_REQUEST_SECONDS):
            raise ValueError('native_diagnostic_budget')
        self._request_seconds = REQUEST_SECONDS if diagnostic_request_seconds is None else diagnostic_request_seconds
        # These must precede the complete loader's Torch/upstream imports.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(num_threads)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from ..tts import _load_tts_model_local
        self._model = _load_tts_model_local(model_name=model_name,
            num_threads=num_threads, dtype_str=dtype_str, revision=revision,
            offline=True, device=device, attention=attention)
        self._checkpoint = model_name
        self._revision = getattr(self._model, '_audio_box_revision', None)
        if self._revision != revision:
            self.close()
            raise ValueError('result_model_identity')
        # Verify the upstream owner's selected device, not just requested flags.
        if getattr(getattr(self._model, 'device', None), 'type', None) != device:
            self.close()
            raise ValueError('native_device_mismatch')

        if diagnostic_dtype is not None and str(getattr(getattr(self._model, 'model', None), 'dtype', None)) != 'torch.' + diagnostic_dtype:
            self.close()
            raise ValueError('native_dtype_mismatch')

    def close(self):
        # No process API, implicit retry, cache flush or asynchronous cleanup.
        self._model = None

    def _parameters(self, params, timeout_s):
        if self._model is None:
            raise ValueError('native_backend_closed')
        keys = {'text', 'language', 'temperature', 'top_p', 'top_k',
                'repetition_penalty', 'max_new_tokens'}
        if type(params) is not dict or set(params) != keys:
            raise ValueError('closed_generation_required')
        for key in ('text', 'language'):
            if type(params[key]) is not str or not params[key].strip() or len(params[key].encode()) > 65536:
                raise ValueError('generation_control')
        for key, lo, hi in (('temperature', 0, 2), ('top_p', 0, 1),
                            ('repetition_penalty', 0, 10)):
            value = params[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not lo <= value <= hi or key == 'repetition_penalty' and value == 0):
                raise ValueError('generation_control')
        for key, lo, hi in (('top_k', 0, 1000), ('max_new_tokens', 1, 32768)):
            if type(params[key]) is not int or not lo <= params[key] <= hi:
                raise ValueError('generation_control')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= self._request_seconds:
            raise ValueError('native_budget_required')
        # Timeout is enforced by the owning supervisor, never a helper thread.
        return dict(params)

    def _audio(self, value):
        wavs, rate = value
        if len(wavs) != 1:
            raise ValueError('single_audio_required')
        samples = wavs[0]
        channels = 1 if getattr(samples, 'ndim', 1) == 1 else samples.shape[1]
        return AudioResult(samples, rate, channels, dict(backend='qwen',
            checkpoint=self._checkpoint, revision=self._revision, offline=True))

    def synthesise_preset(self, params, preset, instruct, timeout_s):
        values = self._parameters(params, timeout_s)
        if type(preset) is not str or not preset.strip() or len(preset) > 256:
            raise ValueError('preset_required')
        if instruct is not None:
            if type(instruct) is not str or not instruct.strip() or len(instruct.encode()) > 65536:
                raise ValueError('generation_control')
            values['instruct'] = instruct
        return self._audio(self._model.generate_custom_voice(**values, speaker=preset))

    def synthesise_clone(self, params, reference, reference_text, x_vector_only, timeout_s):
        values = self._parameters(params, timeout_s)
        if (type(reference) is not str or not Path(reference).is_absolute()
                or type(x_vector_only) is not bool
                or reference_text is not None and (type(reference_text) is not str
                    or not reference_text.strip() or len(reference_text.encode()) > 65536)
                or not x_vector_only and reference_text is None):
            raise ValueError('reference_conditioning')
        prompt = self._model.create_voice_clone_prompt(ref_audio=reference,
            ref_text=reference_text, x_vector_only_mode=x_vector_only)
        try:
            return self._audio(self._model.generate_voice_clone(**values, voice_clone_prompt=prompt))
        finally:
            prompt = None
