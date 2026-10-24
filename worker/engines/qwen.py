"""SOURCE-only Qwen seam. Trusted mappings must remain sealed until child reap.

No service activation, acquisition, public prompt deserialization or OS containment.
The backend owns its spawned model child; supervisor descendant integration is pending.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import wave
from numbers import Integral
from worker.transport import Refused
from worker.hardware import HardwareInventory, DeviceFamily, topology_allows
from .qwen_execution import QwenExecution
from .pocket import _open, _read, _reference_pcm16


def verify_inputs(e, bundles, references, inventory, *, owner_uid=None):
    if type(e) is not QwenExecution: raise Refused('typed_execution_required')
    family = DeviceFamily.CUDA if e.device.startswith('cuda:') else DeviceFamily(e.device)
    index = int(e.device.split(':')[1]) if family == DeviceFamily.CUDA else 0
    if type(inventory) is not HardwareInventory or not topology_allows(inventory.host, family): raise Refused('device_topology')
    if not any(d.family == family and d.index == index and d.supported is True and d.available is True and d.probed_at for d in inventory.devices): raise Refused('device_unavailable')
    if e.bundle_id not in bundles: raise Refused('unauthorized_bundle')
    root = Path(bundles[e.bundle_id]).absolute()
    os.close(_open(root, True, owner_uid=owner_uid))
    raw = _read(root/'manifest.json', 65536, e.manifest_sha256, owner_uid=owner_uid)
    try:
        m = json.loads(raw)
        if (m['schema_version'], m['backend'], m['checkpoint'], m['revision'], m['language']) != (1,'qwen',e.checkpoint,e.revision,e.language) or type(m['schema_version']) is not int: raise Refused('bundle_identity')
        entries = m['files']
        if type(entries) is not list or not 1 <= len(entries) <= 256: raise Refused('bundle_files')
        names = set(); total = 0
        for entry in entries:
            if type(entry) is not dict or not {'path','sha256','bytes'} <= set(entry) <= {'path','sha256','bytes','origin','revision'}: raise Refused('bundle_files')
            name = entry['path']
            if type(name) is not str or len(name)>512 or any(p in ('','.','..') for p in name.split('/')) or Path(name).is_absolute() or name in names or Path(name).suffix not in ('.json','.safetensors','.txt','.model'): raise Refused('bundle_files')
            names.add(name)
            if type(entry['bytes']) is not int or not 0 < entry['bytes'] <= 8*1024**3 or type(entry['sha256']) is not str or len(entry['sha256']) != 64: raise Refused('bundle_limit')
            total += entry['bytes']
            if total > 16*1024**3: raise Refused('bundle_limit')
            fd = _open(root/name, owner_uid=owner_uid)
            with os.fdopen(fd,'rb') as stream:
                before = os.fstat(stream.fileno()); digest = hashlib.sha256(); count = 0
                if before.st_size != entry['bytes']: raise Refused('bundle_size')
                for block in iter(lambda: stream.read(1024*1024), b''):
                    count += len(block)
                    if count > before.st_size: raise Refused('bundle_changed')
                    digest.update(block)
                after = os.fstat(stream.fileno())
                if count != before.st_size or (before.st_mtime_ns,before.st_ctime_ns) != (after.st_mtime_ns,after.st_ctime_ns) or digest.hexdigest() != entry['sha256']: raise Refused('bundle_digest')
        if not {'config.json','generation_config.json'} <= names or not any(n.endswith('.safetensors') for n in names): raise Refused('bundle_config')
        # Reject unlisted assets that upstream might otherwise select implicitly.
        actual = set()
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                p = Path(directory)/name
                if p.is_symlink(): raise Refused('bundle_files')
            actual.update(str((Path(directory)/name).relative_to(root)) for name in files)
        if actual != names | {'manifest.json'}: raise Refused('bundle_files')
        from lib.tts_backends.qwen import validate_checkpoint_request
        validate_checkpoint_request(root, e.task, e.instruct)
        c = m['execution']
        if type(c) is not dict or set(c) != {'sample_rate','languages','presets','devices'}: raise Refused('bundle_capabilities')
        if type(c['sample_rate']) is not int or c['sample_rate'] != e.sample_rate or type(c['languages']) is not list or e.language not in c['languages'] or type(c['presets']) is not list: raise Refused('bundle_capabilities')
        if e.preset is not None and e.preset not in c['presets']: raise Refused('unsupported_preset')
        if type(c['devices']) is not list or {'device':e.device,'dtype':e.dtype,'attention':e.attention} not in c['devices']: raise Refused('unsupported_execution')
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused('bundle_contract') from exc
    reference = None
    if e.task == 'clone':
        if e.reference_id not in references: raise Refused('unauthorized_reference')
        r = references[e.reference_id]
        if type(r) is not dict or set(r) != {'path','voice_id','voice_revision','checkpoint','revision','text'} or (r['voice_id'],r['voice_revision'],r['checkpoint'],r['revision'],r['text']) != (e.voice_id,e.voice_revision,e.checkpoint,e.revision,e.reference_text): raise Refused('reference_identity')
        reference = _read(r['path'],e.reference_bytes,e.reference_sha256,owner_uid=owner_uid)
        if len(reference) != e.reference_bytes: raise Refused('reference_size')
        _reference_pcm16(reference,e.reference_rate,e.reference_frames)
    return root, reference


def execute(execution, text, bundles, references, inventory, *, reference_temp_dir=None, backend_factory=None, managed_owner_uid=None):
    e = execution
    if type(e) is not QwenExecution: raise Refused('typed_execution_required')
    if type(text) is not str or not text.strip() or len(text.encode()) > e.max_text_bytes: raise Refused('text_limit')
    root, reference = verify_inputs(e,bundles,references,inventory,owner_uid=managed_owner_uid)
    if backend_factory is None:
        from lib.tts_backends.qwen import QwenBackend
        backend_factory = QwenBackend
    # Namespace validation precedes backend construction, including lazy model imports.
    if reference is not None:
        if reference_temp_dir is None: raise Refused('snapshot_namespace')
        fd = _open(reference_temp_dir,True)
        try:
            if os.fstat(fd).st_mode & 0o777 != 0o700: raise Refused('snapshot_namespace')
        finally: os.close(fd)
    backend = backend_factory(model_name=str(root),num_threads=e.threads,dtype_str=e.dtype,revision=e.revision,offline=True,device=e.device,attention=e.attention)
    try:
        params = dict(text=text,language=e.language)
        if e.task == 'clone':
            with tempfile.NamedTemporaryFile(suffix='.wav',dir=reference_temp_dir) as snapshot:
                snapshot.write(reference); snapshot.flush()
                result = backend.synthesise_clone(params, snapshot.name, e.reference_text, e.x_vector_only, e.timeout_s)
        elif e.task == 'custom_voice':
            result = backend.synthesise_preset(params,e.preset,e.instruct,e.timeout_s)
        else:
            result = backend.synthesise_design(params,e.instruct,e.timeout_s)
        if type(getattr(result, 'metadata', None)) is not dict or (result.metadata.get('checkpoint'), result.metadata.get('revision')) != (str(root), e.revision):
            raise Refused('result_model_identity')
        return _encode(e,text,result)
    finally:
        backend.close()


def _encode(execution, text, result):
    from lib.tts_backends.contract import AudioResult
    if type(result) is not AudioResult:
        raise Refused('audio_result_required')
    import numpy as np
    data = np.asarray(result.samples)
    frames = data.shape[0] if data.ndim in (1, 2) else 0
    size = 44 + frames*2
    if not 0 < frames <= execution.max_frames or size > execution.max_output_bytes:
        raise Refused('output_limit')
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (result.sample_rate, result.channels)) or (result.sample_rate, result.channels) != (execution.sample_rate, 1) or data.shape not in ((frames,), (frames, 1)):
        raise Refused('audio_format')
    # AudioResult is mutable: revalidate after receiving it, in bounded chunks.
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
        raise Refused('invalid_audio')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(execution.sample_rate)
        for start in range(0, frames, 4096):
            block = data[start:start+4096].reshape(-1).astype('float64')
            if not np.isfinite(block).all() or (np.abs(block) > 1).any():
                raise Refused('invalid_audio')
            wav.writeframesraw((block * 32767).astype('<i2').tobytes())
    output = buffer.getvalue()
    if len(output) != size:
        raise Refused('output_size')
    return dict(data=output, frames=frames, size_bytes=len(output), sha256=hashlib.sha256(output).hexdigest(), identity=dict(execution.identity(), text_sha256=hashlib.sha256(text.encode()).hexdigest()))
