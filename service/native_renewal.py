"""Private stopped native renewal preparation. No public provisioning authority.

The commit/activation seam is deliberately unavailable pending cumulative host
capacity and successor-origin integration. Durable intents always block startup.
No rollback, cleanup, receipt manufacture or live-proof fallback is provided.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from service import native_package as n
from worker import native_protocol as p
from worker.transport import Refused, canonical

TOTAL = 16 * 1024**2
EXPORT = 2 * 1024**2
PREFIX = '001'
ACTION = ('retain owner and all evidence; further execution requires a separately '
          'implemented retention/archival policy')


def require(ok, reason='native_renewal_partial_or_conflicting'):
    if not ok: raise Refused(reason)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def document(path, cap=EXPORT):
    raw = n.read(path, cap)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: require(False))
        require(canonical(value) == raw)
    except (ValueError, TypeError, RecursionError):
        raise Refused('native_renewal_partial_or_conflicting') from None
    return value


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path, value, cap=EXPORT):
    """Exclusion held by caller. Exact owned staging resumes; partial bytes stay.

    An existing final is re-fsynced on explicit resume, including its parent.
    Neither an unknown final nor an unknown staging file is ever overwritten.
    """
    path = Path(path)
    n.directory(str(path.parent))
    raw = canonical(value)
    require(len(raw) <= cap, 'native_renewal_capacity')
    stage = path.with_name(path.name+'.stage')
    if path.exists() or path.is_symlink():
        require(n.read(path, cap) == raw and not stage.exists() and not stage.is_symlink())
    else:
        if stage.exists() or stage.is_symlink():
            require(n.read(stage, cap) == raw)
        else:
            fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                require(stream.write(raw) == len(raw))
                stream.flush(); os.fsync(stream.fileno())
        # Fsync even a completed staging write whose previous acknowledgement was lost.
        fd = os.open(stage, os.O_RDONLY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
        require(not path.exists() and not path.is_symlink())
        os.rename(stage, path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)
    sync(path.parent)
    return value


def namespace(root):
    """Closed finite preparation namespace; no recursive reconciliation exemption."""
    directory = Path(root)/'native-renewal'
    if not directory.exists() and not directory.is_symlink(): return {}
    n.directory(str(directory))
    entries = list(directory.iterdir())
    require(len(entries) <= 6)
    allowed = {PREFIX+'.'+kind+suffix for kind in ('intent','export','seal')
               for suffix in ('', '.stage')}
    require({f.name for f in entries} <= allowed)
    size = 0; result = {}
    for file in entries:
        info = file.lstat(); size += info.st_size
        require(size <= TOTAL, 'native_renewal_capacity')
        # Staging is never silently accepted by normal admission/reconciliation.
        require(not file.name.endswith('.stage'))
        value = document(file)
        if file.name.endswith('.intent'): validate_intent(value)
        elif file.name.endswith('.export'): validate_export(value)
        else: validate_seal(value)
        result[file.name] = value
    require(PREFIX+'.intent' in result)
    intent = result[PREFIX+'.intent']
    if PREFIX+'.export' in result:
        require(result[PREFIX+'.export']['intent_sha256'] == digest(intent))
    if PREFIX+'.seal' in result:
        require(PREFIX+'.export' in result)
        seal = result[PREFIX+'.seal']
        require(seal['intent_sha256'] == digest(intent) and
                seal['export_sha256'] == digest(result[PREFIX+'.export']))
    return result


def startup_guard(root):
    path = Path(root)/'native-renewal'
    if path.exists() or path.is_symlink():
        # Until the successor seam is implemented, even a complete seal cannot
        # authorize execution. Stale epoch-zero entrypoints cannot bypass intent.
        namespace(root)
        raise Refused('native_renewal_pending')


def capacity(dimension, used, limit, extra=1):
    require(type(used) is int and type(limit) is int and type(extra) is int)
    if used + extra > limit:
        raise Refused(f'native_retention_capacity: {dimension} used={used} limit={limit}; {ACTION}')


def compatible(old_config, old_templates, new_config, new_templates):
    """Closed permitted differences, with canonical primitive/type comparison.

    This fixed-version Lot A relation deliberately leaves origin derivation closed.
    All voice versions, settings, profiles, budgets and topology remain exact.
    """
    def control(value):
        value = copy.deepcopy(value)
        p._closed(value, ('version','variant','native','host','port','peer_sha256',
                         'cafile','certfile','keyfile','spool','staging'))
        for key in ('peer_sha256','cafile','certfile','keyfile'): value.pop(key)
        native = value['native']
        p._closed(native, ('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
        for key in ('configuration_sha256','lane','boot'): native.pop(key)
        return value
    def inventory(value):
        value = copy.deepcopy(value)
        p._closed(value, ('host','devices'))
        for device in value['devices']:
            p._closed(device, ('family','index','supported','available','probed_at'))
            # Only positive same-device inventory is eligible for a refresh.
            require(device['supported'] is True and device['available'] is True,
                    'native_renewal_incompatible')
            require(type(device['probed_at']) is str, 'native_renewal_incompatible')
            device.pop('probed_at')
        return value
    def templates(value, config):
        value = copy.deepcopy(value)
        p._closed(value, ('version','variant','profiles','budget','inventory','registrations'))
        original_inventory = copy.deepcopy(value['inventory'])
        value['inventory'] = inventory(original_inventory)
        for row in value['registrations']:
            p._closed(row, ('principal','profile','voice_version','context'))
            execution = row['context']['execution']
            require(canonical(execution['hardware']) == canonical(original_inventory) and
                    canonical(execution['native']) == canonical(config['native']),
                    'native_renewal_incompatible')
            execution['hardware'] = inventory(execution['hardware'])
            p._closed(execution['native'], ('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
            for key in ('configuration_sha256','lane','boot'): execution['native'].pop(key)
        return value
    require(canonical(control(old_config)) == canonical(control(new_config)) and
            canonical(templates(old_templates,old_config)) == canonical(templates(new_templates,new_config)),
            'native_renewal_incompatible')


def stopped_membership(owner, native):
    rows = owner.db.execute('SELECT binding,authority,runtime,receipt,acked,phase,outcome FROM dispatches ORDER BY binding').fetchall()
    count = owner.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
    require(count == len(rows), 'native_history_unclean')
    capacity('history', count, 63)
    from service.worker_transport import Authority
    result = []
    for binding, raw, runtime, receipt, acked, phase, outcome in rows:
        authority = Authority(*json.loads(raw))
        require(raw.encode() == authority._snapshot and binding == authority.binding and
                canonical(p.native_binding(authority.context)) == canonical(native), 'native_history_unclean')
        require(acked == 1 and phase == 'acknowledged' and outcome is not None,
                'native_history_unclean')
        p.terminal_receipt(json.loads(receipt), authority=authority, runtime_id=runtime,
                           local_import_uid=os.getuid())
        require(len(json.loads(receipt)['processes']) == 1, 'native_history_unclean')
        result.append(dict(binding=binding, snapshot=json.loads(raw), runtime_id=runtime,
                           receipt=json.loads(receipt)))
    return result


def stopped_fingerprint(owner):
    """Temporary stopped-transaction bracket, never a permanent whole-DB seal."""
    from service import recovery as r
    data = {}
    rows = size = 0
    for table in r._evidence_columns():
        exists = owner.db.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?',(table,)).fetchone()
        if not exists: continue
        # Bound counts and bytes before materializing any evidence.
        columns = [v[1] for v in owner.db.execute('PRAGMA table_info('+table+')')]
        charge = '+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in columns)
        count, total = owner.db.execute('SELECT count(*),coalesce(sum('+charge+'),0) FROM '+table).fetchone()
        rows += count; size += total
        require(rows <= 2048 and size <= TOTAL, 'native_retention_capacity: history rows/bytes; '+ACTION)
        values = [list(row) for row in owner.db.execute('SELECT * FROM '+table)]
        data[table] = sorted(values, key=canonical)
    return digest(data)


def validate_intent(value):
    p._closed(value, ('kind','request_id','epoch','predecessor','successor','membership','stopped_sha256'))
    require(value['kind'] == 'native-renewal-intent-v1' and type(value['epoch']) is int and value['epoch'] == 1)
    p._hex(value['request_id'],32); p._hex(value['stopped_sha256'],64)
    require(type(value['membership']) is list and len(value['membership']) < 64)
    for row in value['membership']:
        p._closed(row, ('binding','snapshot','runtime_id','receipt'))
        p._hex(row['binding'],64)
    # Enrollment records are exact canonical copies, not loosely decoded flags.
    for record in (value['predecessor'], value['successor']):
        p._closed(record, ('version','variant','kind','owner','configuration_file','templates_file',
                         'configuration_sha256','templates_sha256','trust_sha256','native'))
        require(type(record['version']) is int and record['version'] == 1 and
                record['kind'] == 'native-enrollment-source-v1' and record['variant'] == n.VARIANT)
        p._closed(record['owner'], ('path','device','inode','uid'))
        require(all(type(record['owner'][k]) is int for k in ('device','inode','uid')))
        n.literal(record['owner']['path'])
        for key in ('configuration_file','templates_file'): n.literal(record[key])
        for key in ('configuration_sha256','templates_sha256'): p._hex(record[key],64)
        p._closed(record['trust_sha256'], ('cafile','certfile','keyfile'))
        for trust in record['trust_sha256'].values(): p._hex(trust,64)
        validate_native(record['native'])
    require(canonical(value['predecessor']['owner']) == canonical(value['successor']['owner']))



def begin(record_file, *, configuration_file, templates_file, owner_root, request_id):
    """Persist blocking intent under the existing Core exclusion, no live old RPC."""
    from service.core import native_maintenance
    p._hex(request_id,32)
    with native_maintenance(owner_root) as owner:
        predecessor = n.document(record_file)
        require(canonical(predecessor['owner']) == canonical(dict(path=str(owner.root), device=owner.root.stat().st_dev,
            inode=owner.root.stat().st_ino, uid=os.getuid())), 'native_owner_binding')
        old_config = n.document(predecessor['configuration_file'])
        old_templates = n.document(predecessor['templates_file'])
        require(digest(old_config) == predecessor['configuration_sha256'] and
                digest(old_templates) == predecessor['templates_sha256'], 'native_enrollment_changed')
        successor, _, templates = n._enrollment_snapshot(configuration_file, templates_file, owner_root)
        require(canonical(successor['owner']) == canonical(predecessor['owner']), 'native_owner_binding')
        compatible(old_config, old_templates, n.document(configuration_file), templates)
        membership = stopped_membership(owner, predecessor['native'])
        capacity('registrations', len(old_templates['registrations'])+len(membership),64)
        intent = dict(kind='native-renewal-intent-v1', request_id=request_id, epoch=1,
                      predecessor=predecessor, successor=successor, membership=membership,
                      stopped_sha256=stopped_fingerprint(owner))
        validate_intent(intent)
        directory = owner.root/'native-renewal'
        if not directory.exists():
            os.mkdir(directory,0o700); sync(owner.root)
        n.directory(str(directory))
        # Exact bounded preflight, retaining any conflicting/partial namespace.
        require({f.name for f in directory.iterdir()} <= {
            PREFIX+'.'+k+s for k in ('intent','export','seal') for s in ('','.stage')})
        require(sum(f.lstat().st_size for f in directory.iterdir()) + len(canonical(intent)) <= TOTAL,
                'native_renewal_capacity')
        require(shutil.disk_usage(directory).free >= len(canonical(intent))+EXPORT*2,
                'native_renewal_capacity')
        return publish(directory/(PREFIX+'.intent'), intent)


def read_stopped_journal(root, native, *, retired=False):
    """Pure bounded journal reader; caller owns native.lock, no Journal creation."""
    from worker.native_supervisor import _JournalValidation, private, read_private
    from worker.native_child import validate_proposal
    root = private(root,True)
    wrapped = document(root/'native.json',1048576)
    p._closed(wrapped, ('payload','sha256'))
    require(digest(wrapped['payload']) == wrapped['sha256'], 'native_history_unclean')
    reader = _JournalValidation()
    reader.deployment = native['deployment_id']; reader.configuration = native['configuration_sha256']
    reader.value = wrapped['payload']; reader.validate(reader.value)
    require(reader.value['lane'] == native['lane'], 'native_history_unclean')
    records = reader.value['records']
    names = {'native.lock','native.json'} | {r['token'] for r in records}
    if retired: names |= {f.name for f in root.iterdir() if f.name in ('retirement','retirement.stage','retirement-export','retirement-export.stage')}
    require({f.name for f in root.iterdir()} == names, 'native_history_unclean')
    for r in records:
        require(r['phase'] == 'terminal' and r['acked'] is True and r['boot'] == native['boot'],
                'native_history_unclean')
        spec = p.native_execution(r['snapshot'][1])
        require(canonical(p.native_binding(r['snapshot'][1])) == canonical(native), 'native_history_unclean')
        directory = private(root/r['token'],True)
        expected = {'output'} if r['output'] is not None else set()
        if spec.reference_bytes: expected.add('input-'+spec.reference_sha256)
        require({f.name for f in directory.iterdir()} == expected, 'native_history_unclean')
        if spec.reference_bytes:
            raw = read_private(directory/('input-'+spec.reference_sha256),spec.reference_bytes)
            require(len(raw) == spec.reference_bytes and hashlib.sha256(raw).hexdigest() == spec.reference_sha256,
                    'native_history_unclean')
        if r['receipt']['timely'] is False:
            require(r['proposal'] is None and r['output'] is None, 'native_history_unclean')
        if r['output'] is not None:
            raw = read_private(directory/'output',1048576)
            require(len(raw) == r['output']['bytes'] and hashlib.sha256(raw).hexdigest() == r['output']['sha256'],
                    'native_history_unclean')
            validate_proposal(r['snapshot'], r['proposal'], raw, spec)
    return reader.value


@contextmanager
def host_lock(root):
    from worker.native_supervisor import private
    root = private(root,True)
    private(root/'native.lock')
    fd = os.open(root/'native.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield root
    finally: os.close(fd)


def validate_native(value):
    p._closed(value, ('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
    for key in ('deployment_id','lane','boot'): p._hex(value[key],32)
    for key in ('configuration_sha256','runtime_manifest_sha256'): p._hex(value[key],64)


def validate_export(value):
    p._closed(value, ('kind','intent_sha256','request_id','predecessor_sha256','native','root',
                      'records','charges'))
    require(value['kind'] == 'native-retirement-export-v1')
    validate_native(value['native'])
    p._hex(value['intent_sha256'],64); p._hex(value['predecessor_sha256'],64); p._hex(value['request_id'],32)
    require(type(value['records']) is list and len(value['records']) < 64)
    p._closed(value['root'], ('path','device','inode','uid'))
    require(all(type(value['root'][k]) is int for k in ('device','inode','uid')))
    p._closed(value['charges'], ('journal_records','allocation_registrations','host_bytes','journal_bytes'))
    require(all(type(v) is int and v >= 0 for v in value['charges'].values()))
    from worker.native_supervisor import _JournalValidation
    reader = _JournalValidation()
    reader.deployment=value['native']['deployment_id']; reader.configuration=value['native']['configuration_sha256']
    reader.value=dict(version='native-journal-v1', deployment_id=reader.deployment,
        configuration_sha256=reader.configuration,lane=value['native']['lane'],records=value['records'])
    reader.validate(reader.value)
    require(all(r['phase']=='terminal' and r['acked'] is True and
                r['boot'] == value['native']['boot'] and
                canonical(p.native_binding(r['snapshot'][1])) == canonical(value['native'])
                for r in value['records']))
    charges=value['charges']
    require(charges['journal_records'] == len(value['records']) and
            charges['allocation_registrations'] == len(value['records']) and
            charges['host_bytes'] == sum(2*(p.native_execution(r['snapshot'][1]).reference_bytes+
                p.native_execution(r['snapshot'][1]).max_output_bytes) for r in value['records']))
    require(charges['journal_bytes'] <= 1048576)



def retire(intent_file, *, host_root):
    """Stopped host export under native.lock; no TLS, child, ACK, or deletion."""
    intent = document(intent_file); validate_intent(intent)
    native = intent['predecessor']['native']
    with host_lock(host_root) as root:
        journal = read_stopped_journal(root,native,retired=True)
        records = journal['records']
        joined = [dict(binding=r['binding'],snapshot=r['snapshot'],runtime_id=r['id'],receipt=r['receipt']) for r in records]
        require(canonical(sorted(joined,key=lambda r:r['binding'])) == canonical(intent['membership']),
                'native_history_unclean')
        from service.worker_transport import Authority
        for r in records:
            p.terminal_receipt(r['receipt'], authority=Authority(*r['snapshot']),
                              runtime_id=r['id'],local_import_uid=os.getuid())
            require(len(r['receipt']['processes']) == 1, 'native_history_unclean')
        spec = [p.native_execution(r['snapshot'][1]) for r in records]
        info = root.stat()
        export = dict(kind='native-retirement-export-v1', intent_sha256=digest(intent),
            request_id=intent['request_id'],predecessor_sha256=digest(intent['predecessor']),
            native=native,root=dict(path=str(root),device=info.st_dev,inode=info.st_ino,uid=info.st_uid),
            records=records,charges=dict(journal_records=len(records), allocation_registrations=len(records),
                host_bytes=sum(2*(s.reference_bytes+s.max_output_bytes) for s in spec),
                journal_bytes=len(n.read(root/'native.json',1048576))))
        validate_export(export)
        require(len(canonical(export)) <= EXPORT, 'native_renewal_capacity')
        marker = dict(kind='native-retirement-v1',request_id=intent['request_id'],
            intent_sha256=digest(intent),predecessor_sha256=digest(intent['predecessor']),export_sha256=digest(export))
        # Marker goes first: any uncertain publication permanently fences launch.
        publish(root/'retirement',marker)
        publish(root/'retirement-export',export)
        return export


class RetiredEvidence:
    """Explicit closed offline source, constructed only by validating custody input.

    This cannot authorize execution and is never selected after live RPC failure.
    """
    def __init__(self, intent, export):
        validate_intent(intent); validate_export(export)
        require(export['intent_sha256'] == digest(intent) and
                export['request_id'] == intent['request_id'] and
                export['predecessor_sha256'] == digest(intent['predecessor']) and
                canonical(export['native']) == canonical(intent['predecessor']['native']))
        self.intent = copy.deepcopy(intent); self.export = copy.deepcopy(export)
        self.records = {r['binding']:r for r in export['records']}
        require(len(self.records) == len(export['records']))
        joined = [dict(binding=r['binding'],snapshot=r['snapshot'],runtime_id=r['id'],receipt=r['receipt'])
                  for r in sorted(export['records'],key=lambda r:r['binding'])]
        require(canonical(joined) == canonical(intent['membership']))

    def check_membership(self, owner, runtime):
        require(canonical(runtime.native) == canonical(self.export['native']), 'native_history_unclean')
        require(canonical(stopped_membership(owner,runtime.native)) == canonical(self.intent['membership']),
                'native_history_unclean')

    def collect(self, authority, runtime_id):
        r = self.records.get(authority.binding)
        require(r is not None and r['id'] == runtime_id and
                canonical(r['snapshot']) == authority._snapshot, 'native_history_evidence_unavailable')
        return copy.deepcopy({k:r[k] for k in ('receipt','proposal','output')})


def validate_seal(value):
    p._closed(value, ('kind','intent_sha256','export_sha256','stopped_sha256','membership_sha256'))
    require(value['kind'] == 'native-history-seal-preparation-v1')
    for key in set(value)-{'kind'}: p._hex(value[key],64)


def seal(intent_file, *, export_file):
    """Validate full existing history offline. This preparation seal is not activation."""
    from service.core import native_maintenance
    from service.native_runtime import NativeRuntime
    from service.native_activation import completed_history
    intent = document(intent_file); validate_intent(intent)
    export = document(export_file); evidence = RetiredEvidence(intent,export)
    predecessor = intent['predecessor']
    with native_maintenance(predecessor['owner']['path']) as owner:
        directory = owner.root/'native-renewal'
        require(canonical(document(directory/(PREFIX+'.intent'))) == canonical(intent))
        require(stopped_fingerprint(owner) == intent['stopped_sha256'], 'native_enrollment_changed')
        config = n.document(predecessor['configuration_file'])
        require(digest(config) == predecessor['configuration_sha256'], 'native_enrollment_changed')
        # No SSLContext, key read, or connector invocation: historical evidence is
        # explicitly retired. The connector is a closed refusal, never a fallback.
        def unavailable(_): raise Refused('native_history_evidence_unavailable')
        runtime = NativeRuntime(host=config['host'],port=config['port'],tls=None,
            peer_sha256=config['peer_sha256'],native=predecessor['native'],spool=config['spool'],
            staging=config['staging'],local_import_uid=os.getuid(),connector=unavailable)
        completed_history(owner,runtime,retired=evidence)
        require(stopped_fingerprint(owner) == intent['stopped_sha256'], 'native_enrollment_changed')
        # Do not persist the export until the entire r341 history proof passes.
        publish(directory/(PREFIX+'.export'),export)
        result = dict(kind='native-history-seal-preparation-v1',intent_sha256=digest(intent),
            export_sha256=digest(export),stopped_sha256=intent['stopped_sha256'],
            membership_sha256=digest(intent['membership']))
        return publish(directory/(PREFIX+'.seal'),result)


def commit(intent_file, *, export_file):
    # Deliberately no fake successor authority: this implementation is preparation
    # only. Keep the blocking intent and report the missing implementation seam.
    seal(intent_file,export_file=export_file)
    raise Refused('native_renewal_successor_unavailable: cumulative host capacity, committed epoch activation and origin mapping are not implemented; retain all evidence')
