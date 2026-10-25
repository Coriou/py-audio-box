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


# Private selection token; never reconstructed from retained markers.
FAMILY = object()


def binding(value):
    from service import daemon_volumes as dv
    if type(value) is not dict or value.get('policy') != LAUNCH:
        raise ValueError('export_launch_family_required')
    projected = copy.deepcopy(value)
    projected['policy'] = dv.POLICY
    dv.binding(projected)
    return copy.deepcopy(value)


def host_scope(value):
    from service import daemon_volumes as dv
    projected = binding(value)
    projected['policy'] = dv.POLICY
    return dv.scope(projected)


def installation(value):
    from service.persistent_init import digest
    ip._closed(value, 'deployment_id revision scope host configuration configuration_digest ownership_policy roles launch')
    config = configuration(value['configuration'])
    launch = binding(value['launch'])
    if (value['scope'] != SCOPE or value['ownership_policy'] != POLICY or
            value['configuration_digest'] != digest(config) or
            launch['configuration_digest'] != digest(config) or
            launch['installation'] != value['deployment_id'] or
            _encoded(value['host']) != _encoded(host_scope(launch))):
        raise ValueError('export_enrollment_binding')
    # Reuse unchanged role/revision/host grammar, without admitting the new
    # record to any legacy parser or production activation consumer.
    old = copy.deepcopy(value)
    old.pop('launch')
    old.update(scope=ip.SCOPE, ownership_policy=ip.POLICY)
    old['configuration'].pop('audio_export')
    old['configuration'].update(version=1, ownership_policy=ip.POLICY)
    old['configuration_digest'] = digest(old['configuration'])
    ip.installation(old)
    return value


def database(db, public, *, fresh=False):
    """Exact bounded SQL model; no DDL on the inspected database."""
    from service import persistence, recovery
    from service.reconciliation import COLUMNS
    from service.core import encode
    import time
    installation(public)
    deadline = time.monotonic()+2
    db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
    if persistence.store_format(db) != 2:
        raise ValueError('export_publication_v2_required')
    columns = dict(COLUMNS,
        p1_inventory_scans='id observed complete fingerprint summary',
        p1_inventory_objects='id first_observed last_observed scan_id record',
        p1_inventory_history='seq scan_id object_id observed record',
        p1_integrity='id revision state token', sqlite_sequence='name seq')
    n, scalar = db.execute("SELECT count(*),coalesce(sum(coalesce(length(sql),0)+length(name)+length(tbl_name)+length(type)),0) FROM sqlite_master").fetchone()
    if n>2048 or scalar>SCALAR_BYTES:
        raise ValueError('export_scalar_budget')
    schema = list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master LIMIT 2049'))
    if {r[1] for r in schema if r[0]=='table'} != set(columns):
        raise ValueError('export_exact_schema_required')
    data = {}; count = 0; size = scalar
    for table, fields in columns.items():
        expression = '+'.join('coalesce(length(CAST('+field+' AS BLOB)),0)' for field in fields.split())
        n, scalar = db.execute('SELECT count(*),coalesce(sum('+expression+'),0) FROM '+table).fetchone()
        if count+n>2048 or size+scalar>SCALAR_BYTES:
            raise ValueError('export_scalar_budget')
        rows = list(db.execute('SELECT '+','.join(fields.split())+' FROM '+table+' LIMIT 2049'))
        count += len(rows); size += len(encode(rows).encode())
        if count > 2048 or size > SCALAR_BYTES or time.monotonic()>deadline:
            raise ValueError('export_scalar_budget')
        data[table] = rows
    recovery._installed_detached_rows(dict(schema=schema,data=data), columns)
    expected = [(public['deployment_id'],1,SCOPE,encode(public),_digest(public))]
    if data['p1_installation'] != expected:
        raise ValueError('export_exact_enrollment_required')
    if list(db.execute('PRAGMA integrity_check')) != [('ok',)] or list(db.execute('PRAGMA foreign_key_check')):
        raise ValueError('export_database_integrity')
    if data['p1_recovery_plans']:
        raise ValueError('export_maintenance_refused')
    for rid, raw, digest in data['p1_registrations']:
        record = json.loads(raw)
        bound_registration(record, public)
        if (_encoded(record).decode() != raw or _digest(record) != rid or rid != digest or
                record['deployment_id'] != public['deployment_id'] or
                _encoded(record['scope']) != _encoded(public['host'])):
            raise ValueError('export_registration_canonical_binding')
    if fresh:
        fixed = dict(p1_format=[(2,'synthetic-core')], voice_catalogue_format=[(1,)],
            resource_accounting=[(1,0)], resource_protocol=[(1,2)], resource_monitor=[(1,'healthy')],
            p1_integrity=[(1,1,'healthy',None)], p1_installation=expected)
        if any(rows != fixed.get(table,[]) for table,rows in data.items()):
            raise ValueError('export_nonfresh_initialization')
    else:
        recovery._installed_detached_documents({t:[dict(zip(columns[t].split(),r)) for r in rows] for t,rows in data.items()}, publication_format=2)
    return data


class _Expectation:
    def __init__(self, public):
        self._public = _encoded(installation(public))

    def check(self, db):
        return database(db, json.loads(self._public))

    def check_root(self, root):
        public = json.loads(self._public)
        record = public['roles']['control']
        from pathlib import Path
        info = ip.directory_identity(Path(root).parent, policy=ip.SPLIT, role='control')
        if info[:2] != (record['device'], record['inode']):
            raise ValueError('export_enrollment_root_changed')


def expectation(public, family):
    if family is not FAMILY:
        raise ValueError('explicit_export_family_required')
    return _Expectation(public)


def host_registration(value):
    from service import recovery
    if type(value) is not dict or value.get('evidence') != EVIDENCE:
        raise ValueError('export_registration_evidence')
    old = copy.deepcopy(value)
    old['evidence'] = 'daemon-volumes-inspection-v1'
    recovery.validate_registration(old)
    return value


def fresh_namespace(roots):
    """Finite empty initialization inventory, not a retained-history scanner."""
    root = roots['control']/'core'
    ip.directory_identity(root, policy=ip.SPLIT, role='control')
    required = {'core.sqlite', 'control.lock', 'assets', 'owners', 'staging'}
    names = set(os.listdir(root))
    if not required <= names or names-required-{'core.sqlite-wal','core.sqlite-shm'}:
        raise ValueError('export_fresh_namespace')
    for name in names:
        path = root/name
        if name in ('assets','owners','staging'):
            ip.directory_identity(path, policy=ip.SPLIT, role='control')
            if os.listdir(path): raise ValueError('export_fresh_namespace')
        else:
            with ip.parent_fd(path) as (parent, leaf):
                ip.check(os.stat(leaf,dir_fd=parent,follow_symlinks=False),ip.SPLIT,'control')
    if os.listdir(roots['staging']): raise ValueError('export_fresh_namespace')


def bound_registration(value, public):
    host_registration(value)
    if (_encoded(value['scope']) != _encoded(public['host']) or
            value['deployment_id'] != public['deployment_id'] or
            any(c['image'] != public['configuration']['images'][c['role']] for c in value['containers'])):
        raise ValueError('export_registration_image_binding')
    return value
