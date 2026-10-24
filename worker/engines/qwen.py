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


def validate_declarations(c, language):
    """Pure operator declarations; never probes or manufactures hardware evidence."""
    if type(c) is not dict or set(c) != {'sample_rate','languages','presets','devices'}: raise Refused('bundle_capabilities')
    if type(c['sample_rate']) is not int or not 0 < c['sample_rate'] <= 192000: raise Refused('bundle_capabilities')
    for key in ('languages', 'presets'):
        values = c[key]
        if type(values) is not list or len(values) > 256 or any(type(v) is not str or not v.strip() or len(v) > 256 for v in values) or len(set(values)) != len(values): raise Refused('bundle_capabilities')
    if language not in c['languages']: raise Refused('bundle_capabilities')
    if type(c['devices']) is not list or not 1 <= len(c['devices']) <= 256: raise Refused('bundle_capabilities')
    seen = set()
    import re
    for d in c['devices']:
        if type(d) is not dict or set(d) != {'device','dtype','attention'}: raise Refused('bundle_capabilities')
        device, dtype, attention = d['device'], d['dtype'], d['attention']
        if type(device) is not str or not re.fullmatch(r'cpu|mps|cuda:(0|[1-9][0-9]{0,2})', device) or dtype not in ('float32','float16','bfloat16') or attention not in ('eager','sdpa','flash_attention_2'): raise Refused('bundle_capabilities')
        if (device == 'cpu' and dtype != 'float32') or (device == 'mps' and dtype == 'bfloat16') or (not device.startswith('cuda:') and attention == 'flash_attention_2'): raise Refused('bundle_capabilities')
        key = (device,dtype,attention)
        if key in seen: raise Refused('bundle_capabilities')
        seen.add(key)


def _managed_snapshot(root, manifest, names, caps, owner_uid):
    """Bound local JSON and forbid executable/remote loader declarations.

    Architecture/tensor semantics still belong to the actual upstream loader.
    Config bytes are never rewritten or generated from a model-name guess.
    """
    if manifest['language'] not in ('English', 'French') or any(l not in ('English','French') for l in manifest['execution']['languages']): raise Refused('bundle_language')
    if not {'tokenizer.json','tokenizer_config.json'} <= names: raise Refused('bundle_config')
    if caps.modes != ('custom_voice',) and manifest['execution']['presets']: raise Refused('bundle_capabilities')
    def inspect(value, key='', directory=''):
        if key in ('auto_map', 'custom_pipelines') or key == 'trust_remote_code' and value is not False: raise Refused('remote_code')
        # Path declarations have one meaning: a nonempty listed local asset.
        # Validate before recursion so containers cannot erase their enclosing key.
        if key.endswith(('_file', '_path')):
            if type(value) is not str or not value.strip(): raise Refused('unlisted_config_path')
            if str(Path(directory)/value) not in names or Path(value).is_absolute() or '..' in Path(value).parts: raise Refused('unlisted_config_path')
        if isinstance(value, dict):
            for k, v in value.items(): inspect(v, k, directory)
        elif isinstance(value, list):
            for v in value: inspect(v, key, directory)
        elif isinstance(value, str):
            if '://' in value: raise Refused('remote_config')
    for name in names:
        if not name.endswith('.json'): continue
        value = json.loads(_read(root/name, 16*1024*1024, owner_uid=owner_uid))
        if type(value) is not dict: raise Refused('bundle_config')
        inspect(value, directory=str(Path(name).parent) if Path(name).parent != Path('.') else '')
        if name.endswith('.safetensors.index.json'):
            weights = value.get('weight_map')
            if type(weights) is not dict or not weights or any(type(v) is not str or str(Path(name).parent/v) not in names or not v.endswith('.safetensors') for v in weights.values()): raise Refused('bundle_config')
    config = json.loads(_read(root/'config.json', 16*1024*1024, owner_uid=owner_uid))
    talker = config.get('talker_config', {})
    if type(talker) is not dict: raise Refused('bundle_config')
    # When present upstream declarations are authoritative, never inferred pins.
    languages = talker.get('lang_id')
    if languages is not None:
        if type(languages) is not dict or any(l.lower() not in languages for l in manifest['execution']['languages']): raise Refused('bundle_capabilities')
    speakers = talker.get('spk_id')
    if speakers is not None:
        if type(speakers) is not dict or any(p.lower() not in speakers for p in manifest['execution']['presets']): raise Refused('bundle_capabilities')
    for config_name in ('config.json', 'generation_config.json'):
        value = json.loads(_read(root/config_name, 16*1024*1024, owner_uid=owner_uid))
        if 'sample_rate' in value and value['sample_rate'] != manifest['execution']['sample_rate']: raise Refused('bundle_capabilities')


def verify_bundle(root, manifest_digest, checkpoint, revision, language, *, owner_uid=None, execution=None):
    """Local byte/config validation shared by provisioning and execution. No ML."""
    root = Path(root).absolute()
    os.close(_open(root, True, owner_uid=owner_uid))
    raw = _read(root/'manifest.json', 65536, manifest_digest, owner_uid=owner_uid)
    try:
        m = json.loads(raw)
        if (m['schema_version'], m['backend'], m['checkpoint'], m['revision'], m['language']) != (1,'qwen',checkpoint,revision,language) or type(m['schema_version']) is not int: raise Refused('bundle_identity')
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
        from lib.tts_backends.qwen import checkpoint_capabilities, validate_checkpoint_request
        caps = checkpoint_capabilities(root)
        c = m['execution']
        validate_declarations(c, language)
        if 'versions' in m:
            _managed_snapshot(root, m, names, caps, owner_uid)
        if execution is not None:
            e = execution
            validate_checkpoint_request(root, e.task, e.instruct)
            if c['sample_rate'] != e.sample_rate: raise Refused('bundle_capabilities')
            if e.preset is not None and e.preset not in c['presets']: raise Refused('unsupported_preset')
            if {'device':e.device,'dtype':e.dtype,'attention':e.attention} not in c['devices']: raise Refused('unsupported_execution')
    except (KeyError, TypeError, ValueError) as exc:
        raise Refused('bundle_contract') from exc
    return root, m


def verify_inputs(e, bundles, references, inventory, *, owner_uid=None):
    if type(e) is not QwenExecution: raise Refused('typed_execution_required')
    family = DeviceFamily.CUDA if e.device.startswith('cuda:') else DeviceFamily(e.device)
    index = int(e.device.split(':')[1]) if family == DeviceFamily.CUDA else 0
    if type(inventory) is not HardwareInventory or not topology_allows(inventory.host, family): raise Refused('device_topology')
    if not any(d.family == family and d.index == index and d.supported is True and d.available is True and d.probed_at for d in inventory.devices): raise Refused('device_unavailable')
    if e.bundle_id not in bundles: raise Refused('unauthorized_bundle')
    root, _ = verify_bundle(bundles[e.bundle_id], e.manifest_sha256, e.checkpoint, e.revision, e.language, owner_uid=owner_uid, execution=e)
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
        params = dict(text=text,language=e.language, **{k:getattr(e,k) for k in ('temperature','top_p','top_k','repetition_penalty','max_new_tokens')})
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


def verify_service_inputs(spec, bundles, references, inventory, *, owner_uid=None):
    """Trusted managed service storage is sealed for the full attempt."""
    owner = os.getuid() if owner_uid is None else owner_uid
    if set(bundles) != {spec.bundle_id}: raise Refused('owned_mapping_required')
    root = Path(bundles[spec.bundle_id])
    for directory, dirs, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory)/n for n in dirs+files)]:
            info = path.lstat()
            if info.st_uid != owner or info.st_mode & 0o222 or path.is_symlink(): raise Refused('sealed_bundle_required')
    expected = {spec.reference_id} if spec.task == 'clone' else set()
    if set(references) != expected: raise Refused('owned_mapping_required')
    return verify_inputs(spec,bundles,references,inventory,owner_uid=owner)
