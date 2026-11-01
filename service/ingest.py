"""Trusted owning-thread LOCAL byte ingestion; no paths, URLs or job activation.

Integer PCM WAV only. Publication precedes metadata: a crash may leave an
unreferenced private file, never a metadata row pointing at partial bytes.
Uncertain commit failures retain published bytes for later owner reconciliation.
"""
import hashlib
import io
import os
import stat
import struct
import wave
from uuid import uuid4

from contracts import validate as oracle
from service.voice_store import _id

MAX_BYTES = 24 * 1024 * 1024
MAX_SECONDS = 60
MAX_RATE = 48000
CHUNK = 65536
RESERVE_BYTES = 1024 * 1024


def reference_budget(metadata):
    """Larger verification budget only for bounded, directly owned uploads."""
    return (metadata.get('principal_id') is not None
            and metadata['provenance']['kind'] == 'upload'
            and metadata['media_format'] == 'wav'
            and metadata['media_subtype'] in ('PCM_16', 'PCM_24', 'PCM_32')
            and 8000 <= metadata['sample_rate'] <= MAX_RATE
            and 0 < metadata['frames'] <= MAX_SECONDS * metadata['sample_rate'])


def open_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        oracle.require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0,
                       'foreign_media')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _riff_pcm(stream, length):
    """Validate classic PCM RIFF without reading declared payload lengths.

    Exactly one fmt before one data; ancillary chunks may occur anywhere.
    No trailing bytes, duplicate fmt/data, or extensible/float formats.
    Padding must exist (its value is unspecified). At most 4096 chunks.
    """
    stream.seek(0)
    header = stream.read(12)
    oracle.require(len(header) == 12, 'invalid_audio')
    kind, size, form = struct.unpack('<4sI4s', header)
    oracle.require(kind == b'RIFF' and form == b'WAVE' and size + 8 == length,
                   'invalid_audio')
    offset = 12
    fields = data_size = None
    count = 0
    while offset < length:
        count += 1
        oracle.require(count <= 4096 and length - offset >= 8, 'invalid_audio')
        stream.seek(offset)
        header = stream.read(8)
        oracle.require(len(header) == 8, 'invalid_audio')
        kind, size = struct.unpack('<4sI', header)
        end = offset + 8 + size + (size & 1)
        oracle.require(end <= length, 'invalid_audio')
        if kind == b'fmt ':
            oracle.require(fields is None and data_size is None and size in (16, 18),
                           'invalid_audio')
            raw = stream.read(size)
            oracle.require(len(raw) == size, 'invalid_audio')
            tag, channels, rate, byte_rate, align, bits = struct.unpack('<HHIIHH', raw[:16])
            oracle.require(tag == 1 and bits in (16, 24, 32)
                           and channels in (1, 2) and 8000 <= rate <= MAX_RATE
                           and align == channels * (bits // 8)
                           and byte_rate == rate * align
                           and (size == 16 or raw[16:] == b'\x00\x00'), 'invalid_audio')
            fields = channels, bits // 8, rate, align
        elif kind == b'data':
            oracle.require(fields is not None and data_size is None, 'invalid_audio')
            oracle.require(size % fields[3] == 0, 'invalid_audio')
            data_size = size
        offset = end
    oracle.require(offset == length and fields is not None and data_size is not None,
                   'invalid_audio')
    channels, width, rate, align = fields
    return channels, width, rate, data_size // align


def _audio(stream, length):
    try:
        raw_fields = _riff_pcm(stream, length)
        stream.seek(0)
        # wave bounds its chunk reads; header/chunk sizes cannot allocate audio.
        with wave.open(stream, 'rb') as wav:
            channels, width, rate, frames = (wav.getnchannels(), wav.getsampwidth(),
                                             wav.getframerate(), wav.getnframes())
            oracle.require((channels, width, rate, frames) == raw_fields, 'invalid_audio')
            oracle.require(wav.getcomptype() == 'NONE' and width in (2, 3, 4)
                           and channels in (1, 2) and 8000 <= rate <= MAX_RATE,
                           'invalid_audio')
            oracle.require(0 < frames <= rate * MAX_SECONDS, 'resource_limit')
            expected = frames * channels * width
            oracle.require(expected <= length and expected <= MAX_BYTES, 'invalid_audio')
            consumed = 0
            while consumed < expected:
                data = wav.readframes(min(8192, frames - consumed // (channels * width)))
                oracle.require(bool(data) and len(data) % (channels * width) == 0, 'invalid_audio')
                consumed += len(data)
            oracle.require(consumed == expected, 'invalid_audio')
        return frames, rate, channels, {2: 'PCM_16', 3: 'PCM_24', 4: 'PCM_32'}[width]
    except (wave.Error, EOFError, OverflowError) as exc:
        raise ValueError('invalid_audio') from exc


def ingest_local(core, principal, source):
    """Return asset metadata from bytes or trusted blocking binary read(size).

    Principal is supplied by the authenticated owner, never read from source.
    Each call gets a new opaque ID, including identical content across owners.
    Streams are not closed. No idempotency/deduplication or caller filename.
    """
    _id(principal)
    core.db  # Enforce owning thread before any filesystem operation.
    oracle.require(core.p1, 'pre_p1_inspect_only')
    oracle.require(not core.db.in_transaction, 'invalid_request')
    if isinstance(source, bytes):
        oracle.require(len(source) <= MAX_BYTES, 'resource_limit')
        source = io.BytesIO(source)
    oracle.require(callable(getattr(source, 'read', None)), 'invalid_request')
    staging = assets = None
    temporary = 'ingest-' + uuid4().hex
    asset_id = 'upload-' + uuid4().hex
    staged = published = committed = commit_started = False
    try:
        staging = open_directory(core.root / 'staging')
        assets = open_directory(core.root / 'assets')
        space = os.fstatvfs(staging)
        oracle.require(space.f_bavail * space.f_frsize >= MAX_BYTES + RESERVE_BYTES,
                       'resource_limit')
        fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=staging)
        staged = True
        with os.fdopen(fd, 'w+b') as stream:
            digest = hashlib.sha256()
            length = 0
            while True:
                data = source.read(min(CHUNK, MAX_BYTES + 1 - length))
                oracle.require(isinstance(data, bytes), 'invalid_request')
                oracle.require(len(data) <= min(CHUNK, MAX_BYTES + 1 - length), 'resource_limit')
                if not data:
                    break
                length += len(data)
                oracle.require(length <= MAX_BYTES, 'resource_limit')
                digest.update(data)
                view = memoryview(data)
                while view:
                    count = stream.write(view)
                    if count is None or count <= 0: raise OSError('incomplete_ingest_write')
                    view = view[count:]
            stream.flush()
            frames, rate, channels, subtype = _audio(stream, length)
            metadata = dict(id=asset_id, principal_id=principal, sha256=digest.hexdigest(),
                            frames=frames, sample_rate=rate, channels=channels,
                            media_format='wav', media_subtype=subtype,
                            provenance=dict(kind='upload', source_sha256=digest.hexdigest(),
                                            source_id=asset_id, missing_fields=[]))
            oracle.validate('asset', metadata)
            os.fsync(stream.fileno())
        # Atomic no-overwrite publication. The staged hard link is removed before
        # metadata admission, restoring Core's single-link verification invariant.
        os.link(temporary, asset_id, src_dir_fd=staging, dst_dir_fd=assets,
                follow_symlinks=False)
        published = True
        os.unlink(temporary, dir_fd=staging)
        staged = False
        os.fsync(staging)
        os.fsync(assets)
        from service.core import encode
        with core.transaction():
            core.db.execute('INSERT INTO assets VALUES(?,?)', (asset_id, encode(metadata)))
            commit_started = True
        committed = True
        return metadata
    finally:
        # Known pre-commit failures remove only this invocation's random names.
        # COMMIT errors can be ambiguous: preserve bytes rather than break rows.
        try:
            if staged: os.unlink(temporary, dir_fd=staging)
            if published and not committed and not commit_started:
                os.unlink(asset_id, dir_fd=assets)
                os.fsync(assets)
            if staged: os.fsync(staging)
        finally:
            if staging is not None: os.close(staging)
            if assets is not None: os.close(assets)
