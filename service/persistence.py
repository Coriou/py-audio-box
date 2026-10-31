"""Private A1 publication format; not deployment enrollment or migration."""
import hashlib
import os
import stat
import sqlite3
from functools import wraps


class PublicationQuarantine(ValueError):
    """Ambiguous namespace/provenance: retain intent and every name."""


PUBLICATION_V1 = '''CREATE TABLE publication_work(
          job_id TEXT PRIMARY KEY REFERENCES publications(job_id),
          binding TEXT NOT NULL, destination TEXT NOT NULL UNIQUE,
          temporary TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
          byte_length INTEGER NOT NULL CHECK(byte_length BETWEEN 1 AND 1048576))'''
PUBLICATION_V2 = PUBLICATION_V1.replace('1048576', '37748736')
FORMAT_DDL = 'CREATE TABLE p1_format(revision INTEGER NOT NULL, scope TEXT NOT NULL)'


def detached_format(schema, markers):
    """Named detached model; envelopes make no historical user_version claim."""
    from service.voice_store import _sql_identity
    if (len(markers) != 1 or len(markers[0]) != 2 or
            type(markers[0][0]) is not int or markers[0][0] not in (1, 2) or
            markers[0][1] != 'synthetic-core'):
        raise PublicationQuarantine('unsupported_p1_format')
    version = markers[0][0]
    expected = sqlite3.connect(':memory:')
    try:
        expected.execute(FORMAT_DDL)
        expected.execute(PUBLICATION_V1 if version == 1 else PUBLICATION_V2)
        def identity(rows):
            return sorted((k,n,t,_sql_identity(s)) for k,n,t,s in rows
                          if t in ('p1_format','publication_work') or n in ('p1_format','publication_work'))
        if identity(schema) != identity(expected.execute('SELECT type,name,tbl_name,sql FROM sqlite_master')):
            raise PublicationQuarantine('unsupported_publication_schema')
    finally:
        expected.close()
    return version


def store_format(db):
    where=" WHERE tbl_name IN ('p1_format','publication_work') OR name IN ('p1_format','publication_work')"
    count,size=db.execute('SELECT count(*),coalesce(sum(coalesce(length(sql),0)+length(name)+length(tbl_name)+length(type)),0) FROM sqlite_master'+where).fetchone()
    markers=list(db.execute('SELECT typeof(revision),length(scope) FROM p1_format LIMIT 2'))
    if count>2048 or size>16*1024**2 or len(markers)!=1 or markers[0]!=('integer',14):
        raise PublicationQuarantine('unsupported_p1_format')
    version = detached_format(list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'+where+' LIMIT 2049')),
                              list(db.execute('SELECT revision,scope FROM p1_format LIMIT 2')))
    if db.execute('PRAGMA user_version').fetchone() != (version,):
        raise PublicationQuarantine('publication_version_pair')
    return version


def initialize(db, *, fresh, creation=2):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check_recovery_open(db)
    private = tables & {'p1_format', 'publication_work'}
    if private:
        if private != {'p1_format', 'publication_work'}:
            raise PublicationQuarantine('partial_p1_format')
        store_format(db)
        expected = {
            'p1_format': ['revision', 'scope'],
            'publication_work': ['job_id', 'binding', 'destination', 'temporary', 'sha256', 'byte_length'],
        }
        for table, columns in expected.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns:
                raise PublicationQuarantine('partial_p1_schema')
        return True
    if not fresh:
        if db.execute('PRAGMA user_version').fetchone() != (1,):
            raise PublicationQuarantine('partial_p1_format')
        return False  # Never stamp an old root, even an empty old database.
    # Called inside fresh Core's transaction: DDL and both markers commit together.
    if type(creation) is not int or creation not in (1,2):
        raise ValueError('internal_publication_creation')
    db.execute(FORMAT_DDL)
    db.execute("INSERT INTO p1_format VALUES (?,'synthetic-core')",(creation,))
    db.execute(PUBLICATION_V1 if creation == 1 else PUBLICATION_V2)
    db.execute('PRAGMA user_version='+str(creation))
    return True


def mkdir_durable(path, sync_dir):
    """Persist each new name in its parent, including a newly created root."""
    if not path.exists():
        mkdir_durable(path.parent, sync_dir)
        path.mkdir(mode=0o700)
        sync_dir(path)
        sync_dir(path.parent)


def write_complete(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0 or written > len(view):
            raise OSError('incomplete_publication_write')
        view = view[written:]
    stream.flush()
    if os.fstat(stream.fileno()).st_size != len(data):
        raise OSError('incomplete_publication_length')
    os.fsync(stream.fileno())


def verify_file(path, digest, length, *, links, publication_format=1, job=None, context=None):
    if type(length) is not int or not 1 <= length <= publication_limit(publication_format, job, context):
        raise PublicationQuarantine('publication_length_limit')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != links or info.st_size != length):
            raise PublicationQuarantine('foreign_publication_inode')
        count = 0; digest_state = hashlib.sha256()
        while count < length + 1:
            chunk = os.read(fd, min(65536, length + 1 - count))
            if not chunk:
                break
            count += len(chunk); digest_state.update(chunk)
        if count != length or digest_state.hexdigest() != digest:
            raise PublicationQuarantine('publication_bytes_mismatch')
        os.fsync(fd)
        return info
    finally:
        os.close(fd)


def storage_session(method):
    """A failed storage operation needs reopen/read-state, never live retry."""
    @wraps(method)
    def guarded(core, *args, **kwargs):
        if getattr(core, '_publication_storage_failed', False):
            raise OSError('publication_reopen_required')
        try:
            job_id = args[0] if args else kwargs['job_id']
            check_publication_allowed(core.db, job_id)
            return method(core, *args, **kwargs)
        except (OSError, sqlite3.Error, PublicationQuarantine):
            core._publication_storage_failed = True
            raise
    return guarded


def initialize_inventory(core):
    """Additive private revision-1 diagnostics, only on an existing A1 root.

    This does not enroll an old/populated pre-P1 installation or create authority.
    """
    if not core.p1:
        raise PublicationQuarantine('pre_p1_inspect_only')
    expected = {
        'p1_inventory_scans': ['id','observed','complete','fingerprint','summary'],
        'p1_inventory_objects': ['id','first_observed','last_observed','scan_id','record'],
        'p1_inventory_history': ['seq','scan_id','object_id','observed','record'],
    }
    tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    existing = tables & expected.keys()
    if existing:
        if existing != expected.keys(): raise PublicationQuarantine('partial_inventory_schema')
        for table, columns in expected.items():
            if [r[1] for r in core.db.execute('PRAGMA table_info('+table+')')] != columns:
                raise PublicationQuarantine('unsupported_inventory_schema')
        initialize_integrity(core)
        return
    with core.transaction():
        core.db.execute('CREATE TABLE p1_inventory_scans(id TEXT PRIMARY KEY, observed REAL NOT NULL, complete INTEGER NOT NULL, fingerprint TEXT, summary TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_objects(id TEXT PRIMARY KEY, first_observed REAL NOT NULL, last_observed REAL NOT NULL, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), record TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_history(seq INTEGER PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), object_id TEXT NOT NULL, observed REAL NOT NULL, record TEXT NOT NULL)')

    initialize_integrity(core)


def initialize_integrity(core):
    """Revision 1: missing populated diagnostics are unresolved, never healthy."""
    db = core.db
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='p1_integrity'").fetchone()
    if exists:
        if [r[1] for r in db.execute('PRAGMA table_info(p1_integrity)')] != ['id','revision','state','token']:
            raise PublicationQuarantine('unsupported_integrity_schema')
        if db.execute('SELECT id,revision FROM p1_integrity').fetchall() != [(1,1)]:
            raise PublicationQuarantine('unsupported_integrity_revision')
        return
    populated = any(db.execute('SELECT 1 FROM '+t+' LIMIT 1').fetchone() for t in
                    ('p1_inventory_scans','p1_inventory_objects','p1_inventory_history'))
    with core.transaction():
        db.execute('CREATE TABLE p1_integrity(id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, state TEXT NOT NULL, token TEXT)')
        db.execute('INSERT INTO p1_integrity VALUES(1,1,?,NULL)', ('unresolved' if populated else 'healthy',))


# B1 records deliberately contain no resource epoch or cleanup authority.
B1_COLUMNS = {
    'p1_installation': 'id revision scope record digest',
    'p1_registrations': 'id record digest',
    'p1_recovery_plans': 'id record digest',
}


def initialize_enrollment(core, record):
    """Only the create-only initializer calls this, after all predecessors exist."""
    from service.core import encode
    from service.install_policy import installation
    if getattr(core, '_installed_family', None) is not None:
        from service.installed_export import installation as export_installation
        export_installation(record)
    else:
        installation(record)
    tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables & B1_COLUMNS.keys():
        for table, columns in B1_COLUMNS.items():
            if [r[1] for r in core.db.execute('PRAGMA table_info('+table+')')] != columns.split():
                raise PublicationQuarantine('partial_enrollment_schema')
        rows = core.db.execute('SELECT record,digest FROM p1_installation').fetchall()
        if rows != [(encode(record), hashlib.sha256(encode(record).encode()).hexdigest())]:
            raise PublicationQuarantine('enrollment_mismatch')
        return
    with core.transaction():
        core.db.execute('CREATE TABLE p1_installation(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, scope TEXT NOT NULL, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_registrations(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_recovery_plans(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('INSERT INTO p1_installation VALUES(?,1,?,?,?)',
                        (record['deployment_id'], record['scope'], encode(record), hashlib.sha256(encode(record).encode()).hexdigest()))


# Optional B2 extension. Normal constructors validate it, never install it.
B2_COLUMNS = {
    'p1_b2_format': 'id revision',
    'p1_abandon_plans': 'id record digest',
    'p1_recoveries': 'id plan_id record digest',
    'p1_recovery_steps': 'recovery_id step record digest',
    'dispatch_resolutions': 'dispatch_seq recovery_id record digest',
    'p1_retired_publications': 'job_id recovery_id record digest',
    'p1_integrity_resolutions': 'object_id recovery_id record digest',
}
B2_DDL = (
    'CREATE TABLE p1_b2_format(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL CHECK(revision=1))',
    'CREATE TABLE p1_abandon_plans(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)',
    'CREATE TABLE p1_recoveries(id TEXT PRIMARY KEY, plan_id TEXT NOT NULL UNIQUE REFERENCES p1_abandon_plans(id), record TEXT NOT NULL, digest TEXT NOT NULL)',
    'CREATE TABLE p1_recovery_steps(recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), step TEXT NOT NULL, record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(recovery_id,step))',
    'CREATE TABLE dispatch_resolutions(dispatch_seq INTEGER NOT NULL REFERENCES dispatches(seq), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(dispatch_seq,recovery_id))',
    'CREATE TABLE p1_retired_publications(job_id TEXT NOT NULL REFERENCES publications(job_id), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(job_id,recovery_id))',
    'CREATE TABLE p1_integrity_resolutions(object_id TEXT NOT NULL REFERENCES p1_inventory_objects(id), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(object_id,recovery_id))',
)


def b2_schema(db):
    names = db.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 2049").fetchall()
    if len(names)>2048: raise PublicationQuarantine('b2_schema_budget')
    tables = {r[0] for r in names}
    present = tables & B2_COLUMNS.keys()
    if not present:
        return False
    if present != B2_COLUMNS.keys():
        raise PublicationQuarantine('partial_b2_schema')
    for table, ddl in zip(B2_COLUMNS, B2_DDL):
        if db.execute('SELECT length(sql) FROM sqlite_master WHERE name=?',(table,)).fetchone()[0]>65536:
            raise PublicationQuarantine('b2_schema_budget')
        sql = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
        if ' '.join(sql.split()) != ' '.join(ddl.split()):
            raise PublicationQuarantine('unsupported_b2_schema')
    if db.execute('SELECT id,revision FROM p1_b2_format').fetchall() != [(1,1)]:
        raise PublicationQuarantine('unsupported_b2_format')
    return True


def check_recovery_open(db, root=None):
    """DB mutation authority only; cannot attest the receipt volume."""
    if not b2_schema(db):
        return None
    from service.recovery import validate_b2_database
    try:
        state = validate_b2_database(db, root)
        completed = {r['recovery_id']: r for r in state['p1_recovery_steps'] if r['step']=='core-complete'}
        for row in state['p1_recoveries']:
            if row['id'] not in completed:
                raise ValueError('b2_recovery_pending')
        return state
    except (ValueError, TypeError, KeyError) as exc:
        raise PublicationQuarantine('b2_'+str(exc)) from exc


def check_publication_allowed(db, job_id):
    state = check_recovery_open(db)
    if state and any(r['job_id']==job_id for r in state['p1_retired_publications']):
        raise PublicationQuarantine('b2_retired_publication')


def completed_dispatch_resolutions(db):
    """Read-only validated completed identities; pending records have no effect."""
    from service.recovery import validate_b2_database
    state=validate_b2_database(db)
    if not state:return []
    completed={s['recovery_id'] for s in state['p1_recovery_steps'] if s['step']=='core-complete'}
    return [r for r in state['dispatch_resolutions'] if r['recovery_id'] in completed]


def check_operator_resolved(db,job_id):
    state=check_recovery_open(db)
    if state and any(a['job_id']==job_id for r in state['p1_recoveries'] for a in r['actions']):
        raise ValueError('operator_resolved_job')


def publication_limit(publication_format, job=None, context=None):
    """Storage capacity is never producer authority. Export needs a frozen execution."""
    from contracts import validate as oracle
    from worker.audio_export import execution
    if type(publication_format) is not int or publication_format not in (1,2):
        raise PublicationQuarantine('unsupported_p1_format')
    if publication_format == 2 and job is not None and context is not None:
        plan = context['plan']
        if plan['operation'] == 'audio_export':
            oracle.frozen_plan(plan,context['bundles'])
            if job['operation'] != 'audio_export' or job['input_sha256'] != oracle.plan_hash(plan):
                raise PublicationQuarantine('publication_producer_mismatch')
            spec = execution(context)
            from worker.transport import canonical
            if canonical(context['execution']) != canonical(dict(converter=plan['resolved']['identity']['converter'])):
                raise PublicationQuarantine('publication_execution_mismatch')
            return min(37748736,spec.max_output_bytes)
    return 1048576


def publication_intent(row, *, publication=None, job=None, context=None, publication_format=1):
    """Decode only literal legacy work or an explicit, closed prepared pair.

    No DDL migration. Scalar columns always corroborate the first actual file.
    The complete manifest is retained before any filesystem write.
    """
    import json
    from jsonschema.exceptions import ValidationError
    from contracts import validate as oracle
    from worker.transport import canonical
    binding,destination,temporary,digest,length=row
    try:
        value=json.loads(binding)
        if 'version' not in value:
            oracle.validate('artifact',value)
            entries=[dict(destination=destination,temporary=temporary,sha256=digest,byte_length=length)]
            artifact=value
            if artifact['asset_ids']!=[destination]: raise ValueError()
        else:
            if set(value)!={'version','artifact','manifest','files'} or value['version']!='prepared-reference-adoption-v1': raise ValueError()
            artifact=value['artifact'];manifest=value['manifest'];entries=value['files']
            oracle.validate('artifact',artifact);oracle.validate('result_manifest',manifest)
            if (type(entries)!=list or len(entries)!=2 or len(manifest['outputs'])!=2
                    or manifest['operation']!='voice_prepare' or len(manifest['candidates'])!=1): raise ValueError()
            roles=manifest['candidates'][0]['reference_roles']
            if (roles['reference']['kind']!='prepared_reference' or
                    [a['id'] for a in manifest['outputs']]!=[roles['reference']['asset']['id'],roles['audition']['asset']['id']] or
                    roles['reference']['asset']!=manifest['outputs'][0] or roles['audition']['asset']!=manifest['outputs'][1] or
                    roles['reference']['asset']['sha256']==roles['audition']['asset']['sha256']): raise ValueError()
            if artifact['manifest_sha256']!=oracle.canonical_hash(manifest): raise ValueError()
            for key in ('job_id','generation','attempt_id','owner_id','staging_id','input_sha256'):
                if artifact[key]!=manifest[key]: raise ValueError()
            if artifact['id']!=manifest['artifact_id'] or artifact['asset_ids']!=[a['id'] for a in manifest['outputs']]: raise ValueError()
            for entry,output in zip(entries,manifest['outputs']):
                if (entry['destination']!=output['id'] or entry['sha256']!=output['sha256'] or
                        output['media_format']!='wav' or output['media_subtype']!='PCM_16'): raise ValueError()
                canonical_length=44+output['frames']*output['channels']*2
                if entry['byte_length']!=canonical_length:
                    # Only Qwen's exact authenticated, whole-upload reference may
                    # carry ancillary RIFF chunks. A detached intent without its
                    # frozen producer context cannot claim this exception.
                    if context is None or job is None or entry is not entries[0]: raise ValueError()
                    from worker.runtime_protocol import execution as producer_execution
                    spec=producer_execution(context)
                    if (spec is None or spec.engine!='qwen' or
                            context['plan']['operation']!='voice_prepare' or
                            job['operation']!='voice_prepare' or
                            job['input_sha256']!=oracle.plan_hash(context['plan']) or
                            output['sha256']!=spec.reference_sha256 or
                            entry['byte_length']!=spec.reference_bytes or
                            output['frames']!=spec.reference_frames or
                            output['sample_rate']!=spec.reference_rate or
                            output['channels']!=1 or
                            not canonical_length<=entry['byte_length']<=1048576): raise ValueError()
            if entries[0]!=dict(destination=destination,temporary=temporary,sha256=digest,byte_length=length): raise ValueError()
        if publication is not None:
            if canonical(artifact)!=canonical(json.loads(publication['artifact'])): raise ValueError()
            if 'version' in value and canonical(value['manifest'])!=canonical(json.loads(publication['manifest'])): raise ValueError()
        if job is not None:
            attempt=job['attempts'][-1]
            for key,expected in dict(job_id=job['id'],generation=job['generation'],input_sha256=job['input_sha256'],attempt_id=attempt['id'],owner_id=attempt['owner_id'],staging_id=attempt['staging_id']).items():
                if canonical(artifact[key])!=canonical(expected): raise ValueError()
        if context is not None and 'version' in value:
            oracle.reference_roles(context['plan'],value['manifest']['candidates'][0],{a['id']:a for a in value['manifest']['outputs']})
            if canonical(artifact['identity'])!=canonical(context['plan']['resolved']['identity']): raise ValueError()
        limit=publication_limit(publication_format,job,context)
        if context is not None and job is not None:
            plan=context['plan']
            if canonical(artifact['identity'])!=canonical(plan['resolved']['identity']): raise ValueError()
            if plan['operation']=='audio_export':
                source=plan['assets'][0]
                provenance=dict(source_asset_id=source['id'],source_sha256=source['sha256'],settings=plan['request'],converter=plan['resolved']['identity']['converter'])
                if canonical(artifact.get('audio_export'))!=canonical(provenance): raise ValueError()
                if publication is not None:
                    manifest=json.loads(publication['manifest'])
                    if (manifest['operation']!='audio_export' or
                            canonical(manifest.get('audio_export'))!=canonical(provenance) or
                            artifact['manifest_sha256']!=oracle.canonical_hash(manifest) or
                            artifact['asset_ids']!=[o['id'] for o in manifest['outputs']]): raise ValueError()
                    if len(entries)!=1 or len(manifest['outputs'])!=1: raise ValueError()
                    for e,o in zip(entries,manifest['outputs']):
                        if e['destination']!=o['id'] or e['sha256']!=o['sha256']: raise ValueError()
        for e in entries:
            if (set(e)!={'destination','temporary','sha256','byte_length'} or
                    type(e['byte_length'])!=int or not 1<=e['byte_length']<=limit or
                    not isinstance(e['temporary'],str) or not e['temporary'].startswith('adoption-') or
                    len(e['temporary'])!=41 or any(c not in '0123456789abcdef' for c in e['temporary'][9:])): raise ValueError()
        if len({e['destination'] for e in entries})!=len(entries) or len({e['temporary'] for e in entries})!=len(entries): raise ValueError()
        return artifact,entries
    except (ValueError,TypeError,KeyError,ValidationError,oracle.ContractError) as exc:
        raise PublicationQuarantine('publication_work_mismatch') from exc


def publication_record(record, data, *, publication_format=1):
    """Same cross-document binding for live and detached catalogue readers."""
    import json
    try:
        publication=next(r for r in data['publications'] if r['job_id']==record['job_id'])
        stored=next(r for r in data['jobs'] if r['id']==record['job_id'])
        return publication_intent(tuple(record[k] for k in ('binding','destination','temporary','sha256','byte_length')),
            publication=publication,job=json.loads(stored['job']),context=json.loads(stored['context']),publication_format=publication_format)
    except (StopIteration,KeyError,IndexError,TypeError) as exc:
        raise PublicationQuarantine('publication_work_mismatch') from exc


def public_artifact(record, data, *, publication_format=1):
    """Corroborate the independently served copy with retained producer identity."""
    import json
    from contracts import validate as oracle
    from worker.transport import canonical
    from jsonschema.exceptions import ValidationError
    try:
        artifact = json.loads(record['metadata'])
        oracle.validate('artifact', artifact)
        job_id = artifact['job_id']
        stored = next(r for r in data['jobs'] if r['id'] == job_id)
        job, context = json.loads(stored['job']), json.loads(stored['context'])
        publication = next(r for r in data['publications'] if r['job_id'] == job_id)
        work = next(r for r in data['publication_work'] if r['job_id'] == job_id)
        bound, _ = publication_record(work, data,publication_format=publication_format)
        manifest = json.loads(publication['manifest'])
        oracle.validate('result_manifest', manifest)
        plan = context['plan']; attempt = job['attempts'][-1]
        owner = dict(job_id=job['id'], generation=job['generation'],
            input_sha256=job['input_sha256'], attempt_id=attempt['id'],
            owner_id=attempt['owner_id'], staging_id=attempt['staging_id'])
        expected = dict(owner, id=manifest['artifact_id'],
            asset_ids=[o['id'] for o in manifest['outputs']],
            project_revision=plan['project']['revision'] if plan['operation']=='project_render' else 0,
            manifest_sha256=oracle.canonical_hash(manifest), identity=plan['resolved']['identity'])
        if plan['operation'] == 'audio_export':
            source = plan['assets'][0]
            expected['audio_export'] = dict(source_asset_id=source['id'], source_sha256=source['sha256'],
                settings=plan['request'], converter=plan['resolved']['identity']['converter'])
            if canonical(manifest.get('audio_export')) != canonical(expected['audio_export']): raise ValueError()
        elif 'audio_export' in manifest: raise ValueError()
        if (record['id'] != expected['id'] or job['state'] != 'succeeded' or
                canonical(job['artifact_ids']) != canonical([expected['id']]) or
                job['operation'] != plan['operation'] or manifest['operation'] != plan['operation'] or
                job['input_sha256'] != oracle.plan_hash(plan) or
                canonical({k:manifest[k] for k in owner}) != canonical(owner) or
                any(canonical(a) != canonical(expected) for a in
                    (artifact, bound, json.loads(publication['artifact'])))):
            raise ValueError()
        intent = json.loads(work['binding'])
        if 'version' in intent and canonical(intent['manifest']) != canonical(manifest): raise ValueError()
        return artifact
    except (ValueError, TypeError, KeyError, IndexError, StopIteration, ValidationError, oracle.ContractError) as exc:
        raise PublicationQuarantine('public_artifact_mismatch') from exc
