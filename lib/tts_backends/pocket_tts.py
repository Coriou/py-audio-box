"""Pocket clone adapter using exclusively provisioned local safetensors/config."""
import time
import hashlib
import os
import stat
from numbers import Integral
from .pocket_snapshot import snapshot as reference_snapshot, is_snapshot
from pathlib import Path
from .contract import AudioResult, BackendError, Capabilities
from .assets import verified_bundle


class PocketBackend:
    """Local cloning with one bounded reference-state cache.

    Supervised process callers must supply a private owned reference_temp_dir
    and remove that namespace after confirmed worker reap. Loader inputs use
    anonymous descriptors outside staging, reclaimed even on forced exit.
    close() never deletes a caller-owned directory. Construction requires an
    exclusive worker process: model loading temporarily changes cwd and sets
    PyTorch process-wide thread counts; concurrent construction is unsupported.
    """
    def __init__(self, assets, language='fr', threads=2, reference_temp_dir=None, *, managed_namespace=False):
        if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
            raise BackendError('invalid_threads', 'Thread count must be a positive integer')
        if managed_namespace and reference_temp_dir is None:raise BackendError('invalid_reference_temp_dir','Managed namespace required')
        if type(managed_namespace) is not bool:raise BackendError('invalid_reference_temp_dir','Invalid managed namespace selector')
        self.reference_temp_dir = None
        if reference_temp_dir is not None:
            directory = Path(reference_temp_dir)
            try:
                if managed_namespace:
                    from .pocket_snapshot import managed_directory
                    with managed_directory(directory):pass
                else:
                    info = directory.stat()
                    if directory.is_symlink() or not stat.S_ISDIR(info.st_mode):
                        raise ValueError('Snapshot namespace must be a real directory')
                    if os.name == 'posix' and (stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid()):
                        raise ValueError('Snapshot namespace must be owned and mode0700')
            except (OSError, ValueError) as exc:
                raise BackendError('invalid_reference_temp_dir', 'Choose an existing private owned snapshot directory') from exc
            self.reference_temp_dir = directory.resolve()
        self.root, self.manifest = verified_bundle(assets, 'pocket-tts')
        if language != self.manifest['language']:
            raise BackendError('language_mismatch', 'Provision a separate bundle for each language')
        import torch
        import yaml
        # Reject remote paths even if a user tampers with a config and its manifest.
        config = yaml.safe_load((self.root / 'config.yaml').read_text())
        verified_paths = {(self.root / entry['path']).resolve() for entry in self.manifest['files']}
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.endswith('_path') and item is not None:
                        if not isinstance(item, str):
                            raise BackendError('invalid_assets', 'Pocket asset paths must be strings')
                        path = (self.root / item).resolve()
                        if not path.is_relative_to(self.root) or not path.is_file() or path not in verified_paths:
                            raise BackendError('invalid_assets', 'Pocket config must reference local bundle assets')
                        # Keep approved portable config bytes unchanged. Loading
                        # occurs with the verified bundle as working directory.
                    else:
                        check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(config)
        from pocket_tts import TTSModel
        torch.set_num_threads(threads)
        started = time.monotonic()
        # The supported upstream loader requires a .yaml/.yml filename.
        # Reuse approved bytes; relative weights resolve inside the bundle.
        previous_directory = os.open('.', os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.chdir(self.root)
            self.model = TTSModel.load_model(config=str(self.root / 'config.yaml'))
        finally:
            try:
                os.fchdir(previous_directory)
            finally:
                os.close(previous_directory)
        self.load_seconds = time.monotonic() - started
        self.capabilities = Capabilities('pocket-tts', self.manifest['checkpoint'], ('clone',), False, (language,))
        self._states = {}
        self._requests_started = 0

    def synthesise(self, text, *, reference=None, preset=None, instruct=None, seed=None):
        if preset or instruct:
            raise BackendError('unsupported_parameter', 'Pocket requires an authorized local reference; text instructions/presets are unavailable')
        path = Path(reference or '')
        if not path.is_file() or (path.suffix.lower() != '.wav' and not is_snapshot(path)):
            raise BackendError('invalid_reference', 'Choose an authorized local WAV reference explicitly')
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        snapshot_started = time.monotonic()
        try:
            reference_bytes = path.read_bytes()
        except OSError as exc:
            raise BackendError('invalid_reference', 'Could not snapshot the chosen WAV reference') from exc
        fingerprint = hashlib.sha256(reference_bytes).hexdigest()
        reference_snapshot_seconds = time.monotonic() - snapshot_started
        self._requests_started += 1
        started = time.monotonic()
        reference_state_reused = fingerprint in self._states
        if not reference_state_reused:
            # Hash and condition exactly the same immutable snapshot. The
            # original path may change while a model or encoder is working.
            # No transient directory entries are exposed to staging scans.
            with reference_snapshot(reference_bytes) as snapshot:
                self._states = {fingerprint: self.model.get_state_for_audio_prompt(str(snapshot))}
        del reference_bytes
        conditioning_seconds = time.monotonic() - started
        started = time.monotonic()
        samples = self.model.generate_audio(self._states[fingerprint], text, copy_state=True)
        result = AudioResult(samples.detach().cpu().numpy(), self.model.sample_rate, 1, {
            'backend': 'pocket-tts', 'runtime': 'pytorch', 'device': 'cpu',
            'checkpoint': self.manifest['checkpoint'], 'reference_sha256': fingerprint,
            'text_submitted': text, 'seed': seed,
            'model_load_seconds': self.load_seconds,
            'load_seconds': self.load_seconds if self._requests_started == 1 else 0.0,
            'model_reused': self._requests_started > 1,
            'request_index': self._requests_started,
            'reference_state_reused': reference_state_reused,
            'reference_snapshot_seconds': reference_snapshot_seconds,
            'conditioning_seconds': conditioning_seconds,
            'generation_seconds': time.monotonic() - started,
        })
        return result

    def close(self):
        self._states.clear()
        self.model = None
