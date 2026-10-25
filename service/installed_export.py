"""Pure r378 family inputs; these validators confer no installed activation.

The old installed parsers deliberately remain closed. Callers must select this
family from trusted installer authority and separately corroborate enrollment,
Docker observations, publication schema and the live supervisor.
"""
import copy
import hashlib
import json
import os

from contracts import validate as oracle
from service import install_policy as ip
from worker.audio_export import registration, verify_converter
from worker.engines.pocket import _open

SCOPE = 'owned-kokoro-export-v2'
POLICY = 'docker-kokoro-export-split-v2'
LAUNCH = 'installed-kokoro-export-daemon-volumes-v2'
EVIDENCE = 'kokoro-export-daemon-inspection-v2'
VERSION = 2
SCALAR_BYTES = 16 * 1024**2
MEDIA_BYTES = 160 * 1024**2


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def configuration(value):
    """Validate closed new inputs without changing any v1 parser/default.

    An explicit unqualified profile is preserved as such. Hash-shaped fixture
    inputs cannot establish that images or a converter actually exist.
    """
    ip._closed(value, 'version ownership_policy profiles budget contexts bundles images mounts audio_export')
    if len(_encoded(value)) > 48000:
        raise ValueError('installed_configuration_budget')
    if type(value['version']) is not int or value['version'] != VERSION or value['ownership_policy'] != POLICY:
        raise ValueError('installed_export_family_required')
    value = json.loads(_encoded(value))
    # Reuse only the unchanged Kokoro field validator. This local projection is
    # never persisted or returned as a legacy installation/configuration.
    kokoro = {k: copy.deepcopy(v) for k, v in value.items() if k != 'audio_export'}
    kokoro.update(version=1, ownership_policy=ip.POLICY)
    ip.configuration(kokoro)
    export = value['audio_export']
    ip._closed(export, 'converter profile principals')
    converter = registration(export['converter'])
    if converter['worker_image_digest'] != value['images']['worker']:
        raise ValueError('converter_image_mismatch')
    profile = export['profile']
    oracle.validate('profile', profile)
    if (not profile['id'].startswith('audio-export-') or profile['compatibility'] or
            profile['device'] != 'cpu' or profile['topology'] != 'networkless_container'):
        raise ValueError('export_profile_required')
    principals = export['principals']
    if (type(principals) is not list or not principals or
            any(type(p) is not str or p not in value['contexts'] for p in principals) or
            principals != sorted(set(principals))):
        raise ValueError('export_principals_required')
    return value


def admission_profiles(value):
    value = configuration(value)
    return value['profiles'] + [value['audio_export']['profile']]


def export_seal(value):
    """Expected assertion only; never a receipt or proof of a running worker."""
    value = configuration(value)
    export = value['audio_export']
    return dict(configuration_digest=_digest(value),
                converter_digest=_digest(export['converter']),
                worker_image_digest=value['images']['worker'],
                profile_digest=_digest(export['profile']),
                principals_digest=_digest(export['principals']), publication_format=2)


def validate_seal(seal, value):
    expected = export_seal(value)
    ip._closed(seal, ' '.join(expected))
    if (type(seal['publication_format']) is not int or
            _encoded(seal) != _encoded(expected)):
        raise ValueError('installed_export_seal_mismatch')
    return copy.deepcopy(seal)


def verify_local_converter(value, *, owner_uid=0):
    """No process/probe: validate the declared immutable closure and exe hash.

    Completeness of the declared closure remains an acquisition/review gate.
    Return one retained executable descriptor, owned by the caller.
    """
    value = configuration(value)
    converter = value['audio_export']['converter']
    identities = set()
    paths = [converter['executable'], *converter['runtime_closure']]
    for path in paths:
        fd = _open(path, owner_uid=owner_uid)
        try:
            info = os.fstat(fd)
            identity = info.st_dev, info.st_ino
            if info.st_mode & 0o222 or identity in identities:
                raise ValueError('converter_closure_immutable_required')
            identities.add(identity)
        finally:
            os.close(fd)
    return verify_converter(converter, value['images']['worker'], owner_uid=owner_uid)
