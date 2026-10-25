"""Bounded generation only. Trusted mappings and namespace belong to integration.

Caller reserves memory and owns namespace cleanup after confirmed forced reap.
Managed owner UID is trusted integration input only (runtime fixes it to 10001);
private conditioning snapshots always require the executing child UID.
Mappings must name canonical paths (no symlink components) in immutable managed
storage, stable throughout the call; this is not isolation from same-UID writers.
No preparation, transcript interpretation, publication or qualification occurs.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
from lib.tts_backends.pocket_snapshot import snapshot as reference_snapshot
import wave
from numbers import Integral
from worker.transport import Refused
from .pocket_execution import PocketExecution


def _open(path, directory=False, *, owner_uid=None):
    """Walk every component with no-follow, retaining directory descriptors."""
    path = Path(path).absolute()
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            if part in ('.', '..'):
                raise Refused('unsafe_path')
            last = index == len(path.parts[1:])-1
            nxt = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory or not last else os.O_NONBLOCK), dir_fd=fd)
            os.close(fd)
            fd = nxt
        info = os.fstat(fd)
        if info.st_uid != (os.getuid() if owner_uid is None else owner_uid) or info.st_mode & 0o022 or (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
            raise Refused('file_ownership')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(path, limit, digest=None, *, owner_uid=None):
    fd = _open(path, owner_uid=owner_uid)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= limit:
            raise Refused('file_limit')
        data = stream.read(limit+1)
        after = os.fstat(stream.fileno())
        if len(data) != before.st_size or len(data) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise Refused('file_changed')
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise Refused('file_digest')
    return data


def _reference_pcm16(raw, rate, frames):
    """Validate complete classic RIFF; bounded bytes were authenticated by _read.

    Like ingest's chunk walker, require one fmt before one data and at most
    4096 chunks. Keep this narrower worker policy independent of service imports.
    Ancillary payloads and padding values are opaque; padding must exist.
    """
    if len(raw) < 12 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE' or struct.unpack_from('<I', raw, 4)[0] + 8 != len(raw):
        raise Refused('reference_format')
    offset = 12
    seen_fmt = seen_data = False
    count = 0
    while offset < len(raw):
        count += 1
        if count > 4096 or len(raw) - offset < 8:
            raise Refused('reference_format')
        kind, size = struct.unpack_from('<4sI', raw, offset)
        start = offset + 8
        end = start + size + (size & 1)
        if end > len(raw):
            raise Refused('reference_format')
        if kind == b'fmt ':
            if seen_fmt or seen_data or size not in (16, 18):
                raise Refused('reference_format')
            # Extensible, float and all non-16-bit declarations are unsupported.
            if struct.unpack_from('<HHIIHH', raw, start) != (1, 1, rate, rate*2, 2, 16) or (size == 18 and raw[start+16:start+18] != b'\x00\x00'):
                raise Refused('reference_format')
            seen_fmt = True
        elif kind == b'data':
            if not seen_fmt or seen_data or size % 2 or size != frames*2:
                raise Refused('reference_format')
            seen_data = True
        offset = end
    if offset != len(raw) or not seen_fmt or not seen_data:
        raise Refused('reference_format')


def execute(execution, text, bundles, references, *, reference_temp_dir, backend_factory=None, preset=None, instruct=None, managed_owner_uid=None, reference_fd=None):
    if type(execution) is not PocketExecution:
        raise Refused('typed_execution_required')
    e = execution
    if preset is not None or instruct is not None:
        raise Refused('unsupported_parameter')
    if type(text) is not str or not text or len(text.encode('utf-8')) > e.max_text_bytes:
        raise Refused('text_limit')
    reference = verify_inputs(e, bundles, references, owner_uid=managed_owner_uid, reference_fd=reference_fd)
    return _synthesise(e, text, bundles[e.bundle_id], reference, reference_temp_dir, backend_factory)


def _synthesise(e, text, bundle, reference, reference_temp_dir, backend_factory):
    """Private adapter seam: caller already verified the full binding and bytes."""
    root = Path(bundle).absolute()
    namespace = Path(reference_temp_dir).absolute()
    fd = _open(namespace, True)
    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
            raise Refused('snapshot_namespace')
    finally:
        os.close(fd)
    if backend_factory is None:
        from lib.tts_backends.pocket_tts import PocketBackend
        backend_factory = PocketBackend
    # Anonymous inputs cannot disappear beneath a concurrent staging scan.
    # The kernel also closes them after forced termination.
    with reference_snapshot(reference) as snapshot:
        backend = backend_factory(root, language=e.language, threads=e.threads, reference_temp_dir=namespace)
        try:
            result = backend.synthesise(text, reference=snapshot)
            from lib.tts_backends.contract import AudioResult
            if type(result) is not AudioResult:
                raise Refused('audio_result_required')
            import numpy as np
            data = np.asarray(result.samples)
            frames = data.shape[0] if data.ndim in (1, 2) else 0
            size = 44+frames*2
            if not 0 < frames <= e.max_frames or size > e.max_output_bytes:
                raise Refused('output_limit')
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (result.sample_rate, result.channels)) or (result.sample_rate, result.channels) != (e.sample_rate, 1) or data.shape not in ((frames,), (frames, 1)):
                raise Refused('audio_format')
            if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
                raise Refused('invalid_audio')
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(result.sample_rate)
                for start in range(0, frames, 4096):
                    block = data[start:start+4096].reshape(-1).astype('float64')
                    if not np.isfinite(block).all() or (np.abs(block) > 1).any():
                        raise Refused('invalid_audio')
                    wav.writeframesraw((block*32767).astype('<i2').tobytes())
            output = buffer.getvalue()
            if len(output) != size:
                raise Refused('output_size')
            return dict(data=output, frames=frames, size_bytes=size, sha256=hashlib.sha256(output).hexdigest(), identity=dict(e.identity(), text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
        finally:
            backend.close()


def verify_inputs(e, bundles, references, *, owner_uid=None, reference_fd=None):
    """Inert verification shared by trusted integration; no backend import."""
    from .pocket_prepare import PocketPreparation
    if type(e) not in (PocketExecution, PocketPreparation): raise Refused('typed_execution_required')
    if e.bundle_id not in bundles or e.reference_id not in references:
        raise Refused('unauthorized_identity')
    verify_bundle(e, bundles, owner_uid=owner_uid)
    if reference_fd is None:
        reference = _read(references[e.reference_id], e.reference_bytes, e.reference_sha256, owner_uid=owner_uid)
    else:
        reference = read_reference_fd(reference_fd, e, owner_uid=owner_uid)
    if len(reference) != e.reference_bytes:
        raise Refused('reference_size')
    _reference_pcm16(reference, e.reference_rate, e.reference_frames)
    return reference


def verify_bundle(e, bundles, *, owner_uid=None):
    """Verify the configured bundle without requiring a preselected reference."""
    if e.bundle_id not in bundles: raise Refused('unauthorized_identity')
    root = Path(bundles[e.bundle_id]).absolute()
    fd = _open(root, True, owner_uid=owner_uid)
    os.close(fd)
    raw = _read(root/'manifest.json', 65536, e.manifest_sha256, owner_uid=owner_uid)
    manifest = json.loads(raw)
    if (manifest.get('schema_version'), manifest.get('backend'), manifest.get('checkpoint'), manifest.get('revision'), manifest.get('language')) != (1, 'pocket-tts', e.checkpoint, e.revision, e.language) or type(manifest.get('schema_version')) is not int:
        raise Refused('bundle_identity')
    entries = manifest.get('files')
    if type(entries) is not list or not 1 <= len(entries) <= 128:
        raise Refused('bundle_files')
    names = set()
    total = 0
    for entry in entries:
        if type(entry) is not dict or not {'path', 'sha256'} <= set(entry) <= {'path', 'sha256', 'bytes', 'origin', 'revision'}:
            raise Refused('bundle_files')
        name = entry['path']
        if type(name) is not str or len(name) > 512 or any(p in ('', '.', '..') for p in name.split('/')) or Path(name).is_absolute() or name in names or Path(name).suffix not in ('.safetensors', '.yaml', '.json'):
            raise Refused('bundle_files')
        names.add(name)
        # Hash large weights in bounded chunks, never materialize model bytes.
        fd = _open(root/name, owner_uid=owner_uid)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            total += info.st_size
            if not 0 < info.st_size <= 2*1024**3 or total > 4*1024**3 or ('bytes' in entry and (type(entry['bytes']) is not int or entry['bytes'] != info.st_size)):
                raise Refused('bundle_limit')
            digest = hashlib.sha256()
            count = 0
            for block in iter(lambda: stream.read(1024*1024), b''):
                count += len(block)
                if count > info.st_size:
                    raise Refused('bundle_changed')
                digest.update(block)
            after = os.fstat(stream.fileno())
            if count != info.st_size or (info.st_mtime_ns, info.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns) or digest.hexdigest() != entry['sha256']:
                raise Refused('bundle_digest')
    if 'config.yaml' not in names:
        raise Refused('bundle_config')


def read_reference_fd(fd, e, *, owner_uid=None):
    """Read one inherited private descriptor without reopening any public path."""
    from worker.managed_reference import check_fd
    before = check_fd(fd, owner_uid=10001 if owner_uid is None else owner_uid)
    if before.st_size != e.reference_bytes: raise Refused('reference_size')
    data = bytearray()
    while len(data) <= e.reference_bytes:
        block = os.pread(fd, min(65536, e.reference_bytes+1-len(data)), len(data))
        if not block: break
        data.extend(block)
    after = check_fd(fd, owner_uid=10001 if owner_uid is None else owner_uid)
    fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    if any(getattr(before,k) != getattr(after,k) for k in fields): raise Refused('file_changed')
    if len(data) != e.reference_bytes or hashlib.sha256(data).hexdigest() != e.reference_sha256:
        raise Refused('reference_digest')
    _reference_pcm16(data, e.reference_rate, e.reference_frames)
    return bytes(data)
