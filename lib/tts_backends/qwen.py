"""Qwen adapter: spawned model ownership, bounded stop/reap and clean restart."""
import multiprocessing as mp
import os
import atexit
import weakref
import contextlib
import sys
import json
from pathlib import Path
from dataclasses import dataclass
import threading
import time
import signal
from .contract import BackendError, Capabilities, AudioResult


def checkpoint_capabilities(name):
    checkpoint = str(name).rstrip('/').split('/')[-1].lower()
    local_config = Path(name) / 'config.json'
    if local_config.is_file():
        try:
            config = json.loads(local_config.read_text())
            if config.get('model_type') != 'qwen3_tts':
                raise ValueError('Not a Qwen3 TTS checkpoint')
            size = {'0b6': '0.6b', '1b7': '1.7b'}.get(config['tts_model_size'])
            mode = {'base': 'base', 'custom_voice': 'customvoice', 'voice_design': 'voicedesign'}.get(config['tts_model_type'])
            if size is None or mode is None:
                raise ValueError('Unknown checkpoint size or mode')
            checkpoint = f'{size}-{mode}'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BackendError('unknown_checkpoint', 'Local checkpoint capabilities are unknown', checkpoint=str(name)) from exc
    mode = 'custom_voice' if checkpoint.endswith('customvoice') else 'voice_design' if checkpoint.endswith('voicedesign') else 'clone' if checkpoint.endswith('base') else None
    if mode is None:
        raise BackendError('unknown_checkpoint', 'Checkpoint capabilities are unknown', checkpoint=str(name))
    direction = mode == 'voice_design' or (mode == 'custom_voice' and '1.7b' in checkpoint)
    return Capabilities('qwen', str(name), (mode,), direction)


def validate_checkpoint_request(name, mode, instruct=None):
    caps = checkpoint_capabilities(name)
    if mode not in caps.modes:
        raise BackendError('unsupported_mode', f'{name} does not support {mode}', checkpoint=str(name), mode=mode)
    if instruct and str(instruct).strip() and not caps.text_direction:
        raise BackendError('unsupported_instruction', f'{name} does not support text direction; select 1.7B-CustomVoice explicitly for instructions', checkpoint=str(name), mode=mode)
    return caps


@dataclass
class ClonePromptItem:
    """Portable prompt DTO; preserves transcript and ICL without upstream imports."""
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None


def _cpu_payload(value):
    """Never export CUDA tensors/IPC lifetimes from the model owner."""
    import torch
    from dataclasses import is_dataclass, fields, replace
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _cpu_payload(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_payload(item) for item in value)
    return value


def _worker(connection, settings):
    # Keep every upstream print away from the caller's structured stdout.
    with contextlib.redirect_stdout(sys.stderr):
        _worker_impl(connection, settings)


def _worker_impl(connection, settings):
    try:
        if settings.get("offline"):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from tts import _load_tts_model_local
        model = _load_tts_model_local(**settings)
        connection.send(('ready', {'revision': getattr(model, '_audio_box_revision', settings.get('revision'))}))
        while True:
            method, args, kwargs = connection.recv()
            value = None
            try:
                seed = kwargs.pop('_audio_box_seed', None)
                if seed is not None:
                    import torch
                    torch.manual_seed(seed)
                if method == '_service_clone':
                    reference = kwargs.pop('ref_audio')
                    ref_text = kwargs.pop('ref_text')
                    x_vector_only = kwargs.pop('x_vector_only_mode')
                    prompt = model.create_voice_clone_prompt(ref_audio=reference, ref_text=ref_text, x_vector_only_mode=x_vector_only)
                    value = model.generate_voice_clone(**kwargs, voice_clone_prompt=prompt)
                else:
                    value = getattr(model, method)(*args, **kwargs)
                if method == 'create_voice_clone_prompt':
                    # Attribute-compatible items preserve ref_text/ICL. Native
                    # Qwen dict prompts omit transcripts in qwen-tts 0.1.1.
                    value = [ClonePromptItem(item.ref_code, item.ref_spk_embedding,
                             item.x_vector_only_mode, item.icl_mode, item.ref_text)
                             for item in value]
                connection.send(('ok', _cpu_payload(value)))
            except Exception as exc:
                connection.send(('error', (type(exc).__name__, str(exc))))
            finally:
                # Sending serializes the result; the owner need not retain the
                # previous take while idle or during the next inference.
                value = None
    except BaseException as exc:
        try:
            connection.send(('error', (type(exc).__name__, str(exc))))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def _memory_snapshot():
    """Best-effort Linux cgroup evidence, never a claim about one process."""
    result = {}
    roots = [Path('/sys/fs/cgroup')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                roots.insert(0, Path('/sys/fs/cgroup') / line[3:].lstrip('/'))
    except OSError:
        pass
    for root in roots:
        try:
            events = dict(line.split() for line in (root / 'memory.events').read_text().splitlines())
            result['oom_kill'] = int(events.get('oom_kill', 0))
            limit = (root / 'memory.max').read_text().strip()
            result['memory_limit_bytes'] = int(limit) if limit != 'max' else 'unlimited'
            return result
        except (OSError, ValueError):
            continue
    return result


class QwenWorkerError(RuntimeError):
    """Transport failure with bounded, non-payload diagnostics."""
    def __init__(self, details):
        self.details = details
        summary = ', '.join(f'{key}={value}' for key, value in details.items())
        super().__init__(f'Qwen worker unavailable: {summary}')


class QwenBackend:
    """Single owner of model state. Spawn avoids inherited CUDA/PyTorch locks.

    Errors/timeouts discard the worker. The following job starts a fresh model.
    Prompt construction and synthesis execute in the same child, never the host.
    """
    def __init__(self, **settings):
        self.settings = settings
        self.name_or_path = settings['model_name']
        self._audio_box_checkpoint = self.name_or_path
        self._audio_box_revision = settings.get('revision')
        self.capabilities = checkpoint_capabilities(self.name_or_path)
        self._process = self._connection = None
        self._lock = threading.Lock()
        self._failed_reap = False
        owner_ref = weakref.ref(self)
        def on_exit():
            owner = owner_ref()
            if owner is not None:
                owner.close()
        self._exit_callback = on_exit
        atexit.register(on_exit)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        callback = self.__dict__.get('_exit_callback')
        if callback is not None:
            atexit.unregister(callback)
        process, connection = self._process, self._connection
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            if process.is_alive():
                self._failed_reap = True
                raise RuntimeError('Inference worker could not be reaped; backend is unavailable')
            process.close()
        elif process is not None:
            process.close()
        if connection is not None:
            connection.close()
        self._process = self._connection = None

    def _worker_failure(self, method, stage, before):
        process = self._process
        # EOF can precede publication of the OS exit status by a few ticks.
        if process is not None and process.pid is not None:
            process.join(0.1)
        exitcode = process.exitcode if process is not None else None
        details = {'method': method, 'stage': stage,
                   'checkpoint': str(self.name_or_path).rstrip('/').split('/')[-1],
                   'dtype': str(self.settings.get('dtype_str', self.settings.get('dtype', 'auto'))),
                   'exitcode': exitcode}
        if exitcode is not None and exitcode < 0:
            try:
                details['signal'] = signal.Signals(-exitcode).name
            except ValueError:
                details['signal'] = -exitcode
        after = _memory_snapshot()
        if 'memory_limit_bytes' in after:
            details['memory_limit_bytes'] = after['memory_limit_bytes']
        if 'oom_kill' in before and 'oom_kill' in after:
            delta = after['oom_kill'] - before['oom_kill']
            details['cgroup_oom_kill_delta'] = max(0, delta)
            if delta > 0:
                details['memory_evidence'] = 'OOM observed in shared cgroup; worker attribution unproven'
        return QwenWorkerError(details)

    def _receive(self, deadline, method, stage, before):
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError('Qwen inference exceeded wall-clock timeout')
            try:
                if self._connection.poll(min(1, remaining) if remaining is not None else 1):
                    status, value = self._connection.recv()
                    if status == 'error':
                        raise RuntimeError(f'Qwen worker {value[0]}: {value[1]}')
                    return value
            except (EOFError, OSError) as exc:
                raise self._worker_failure(method, stage, before) from exc
            if not self._process.is_alive():
                raise self._worker_failure(method, stage, before)

    def invoke(self, method, args=(), kwargs=None, timeout_s=None):
        with self._lock:
            if self._failed_reap:
                raise RuntimeError('Backend disabled after failed worker reap')
            deadline = None if timeout_s == 0 else time.monotonic() + (timeout_s if timeout_s is not None else 3600)
            before = _memory_snapshot()
            child = None
            try:
                if self._process is None:
                    atexit.register(self._exit_callback)
                    ctx = mp.get_context('spawn')
                    parent, child = ctx.Pipe()
                    self._connection = parent
                    try:
                        self._process = ctx.Process(target=_worker, args=(child, self.settings), daemon=True)
                        self._process.start()
                    except (OSError, EOFError) as exc:
                        raise self._worker_failure(method, 'spawn', before) from exc
                    finally:
                        child.close()
                        child = None
                    ready = self._receive(deadline, method, 'startup', before)
                    if isinstance(ready, dict):
                        self._audio_box_revision = ready.get("revision")
                try:
                    self._connection.send((method, args, kwargs or {}))
                except (OSError, EOFError) as exc:
                    raise self._worker_failure(method, 'send', before) from exc
                return self._receive(deadline, method, 'response', before)
            except BaseException:
                if child is not None:
                    child.close()
                self.close()
                raise

    def create_voice_clone_prompt(self, *args, **kwargs):
        from tts import bind_clone_prompt
        prompt = self.invoke('create_voice_clone_prompt', args, kwargs)
        return bind_clone_prompt(prompt, model=self)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.invoke(name, args, kwargs)

    def generate_audio(self, method, params, timeout_s=None):
        wavs, sr = self.invoke(method, kwargs=params, timeout_s=timeout_s)
        samples = wavs[0]
        channels = 1 if getattr(samples, 'ndim', 1) == 1 else samples.shape[1]
        return AudioResult(samples, sr, channels, {'backend': 'qwen', 'checkpoint': self.name_or_path, 'revision': self._audio_box_revision, 'offline': self.settings.get('offline', False)})

    # Closed service methods. The legacy dynamic dispatcher remains private to
    # legacy callers; no method name or arbitrary kwargs crosses this interface.
    def synthesise_clone(self, params, reference, reference_text, x_vector_only, timeout_s):
        return self.generate_audio('_service_clone', dict(params, ref_audio=reference,
            ref_text=reference_text, x_vector_only_mode=x_vector_only), timeout_s)

    def synthesise_preset(self, params, preset, instruct, timeout_s):
        values = dict(params, speaker=preset)
        if instruct is not None: values['instruct'] = instruct
        return self.generate_audio('generate_custom_voice', values, timeout_s)

    def synthesise_design(self, params, instruct, timeout_s):
        return self.generate_audio('generate_voice_design', dict(params, instruct=instruct), timeout_s)
