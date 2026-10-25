"""Private stopped epoch transactions. No public provisioning or voice migration."""
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
    if path.parent.name=='native-renewal':
        require(sum(f.lstat().st_size for f in path.parent.iterdir() if f not in (path,stage))+len(raw)<=TOTAL,'native_renewal_capacity')
        require(shutil.disk_usage(path.parent).free>=len(raw),'native_renewal_capacity')
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


def namespace(root, *, resume=False, _begin=False):
    """Finite append-only chain; pending/staging never grants admission."""
    require(not _begin or resume)
    directory = Path(root)/'native-renewal'
    if not directory.exists() and not directory.is_symlink(): return {}
    n.directory(str(directory))
    entries = list(directory.iterdir())
    require(len(entries) <= 70)
    allowed = {f'{epoch:03d}.'+kind+suffix for epoch in range(1,8)
               for kind in ('intent','export','seal','enrollment','commit') for suffix in ('','.stage')}
    require({f.name for f in entries} <= allowed)
    require(sum(f.lstat().st_size for f in entries) <= TOTAL, 'native_renewal_capacity')
    result = {}
    validators = dict(intent=validate_intent, export=validate_export, seal=validate_seal,
                      enrollment=validate_envelope, commit=validate_commit)
    for file in entries:
        if file.name.endswith('.stage'):
            require(resume and not file.with_suffix('').exists())
            value = document(file)  # Publisher must subsequently compare exact bytes.
            if _begin and file.name.endswith('.intent.stage'):
                # Only explicit begin may see a completed intent as a candidate.
                # Chain validation below runs before request/evidence comparison.
                validate_intent(value)
                require(all(other == file or other.name[:3] < file.name[:3]
                            for other in entries))
                result[file.name.removesuffix('.stage')] = value
            continue
        value = document(file)
        validators[file.name.split('.')[1]](value)
        result[file.name] = value
    require('001.intent' in result or (resume and (directory/'001.intent.stage').exists()))
    previous = None
    pending = False
    for epoch in range(1,8):
        prefix=f'{epoch:03d}.'
        group={k[len(prefix):]:v for k,v in result.items() if k.startswith(prefix)}
        if not group: continue
        require(not pending and 'intent' in group)
        intent=group['intent']; require(intent['epoch'] == epoch)
        for source in (intent['predecessor'],intent['successor']):
            for field in ('configuration','templates'):
                require(digest(n.document(source[field+'_file']))==source[field+'_sha256'],'native_enrollment_changed')
        if epoch > 1:
            require(previous is not None and previous['epoch']==epoch-1 and canonical(intent['predecessor']) == canonical(previous['source']))
        if 'export' in group: RetiredEvidence(intent,group['export'])
        if 'seal' in group:
            require('export' in group and group['seal']['intent_sha256']==digest(intent) and
                    group['seal']['export_sha256']==digest(group['export']))
        if 'enrollment' in group:
            envelope=group['enrollment']
            require('seal' in group and envelope['epoch']==epoch and
                    envelope['seal_sha256']==digest(group['seal']) and
                    envelope['export_sha256']==digest(group['export']) and
                    canonical(envelope['source'])==canonical(intent['successor']))
        if 'commit' in group:
            require(set(group)=={'intent','export','seal','enrollment','commit'})
            old_templates=n.document(intent['predecessor']['templates_file'])
            new_templates=n.document(intent['successor']['templates_file'])
            require(canonical(group['enrollment']['origins'])==canonical(origin_mapping(old_templates,new_templates,None if previous is None else previous['origins'])))
            commit=group['commit']
            earlier_boots={v['predecessor']['native']['boot'] for k,v in result.items() if k.endswith('.intent') and k[:3]<=prefix[:3]}
            require(intent['successor']['native']['boot'] not in earlier_boots,'native_renewal_incompatible')
            require(commit['request_id']==intent['request_id'] and commit['epoch']==epoch and
                    commit['enrollment_sha256']==digest(group['enrollment']) and
                    commit['predecessor_commit_sha256']==(None if previous is None else previous['commit_sha256']))
            previous=dict(group['enrollment'],commit_sha256=digest(commit))
        else: pending=True
    return result


def tip(root, *, resume=False):
    values=namespace(root,resume=resume)
    if not values: return None
    commits=sorted(k for k in values if k.endswith('.commit'))
    intents=sorted(k for k in values if k.endswith('.intent'))
    require(commits and intents[-1][:3]==commits[-1][:3], 'native_renewal_pending')
    key=commits[-1][:3]+'.enrollment'
    return values[key]


def startup_guard(root, *, activation=False):
    if (Path(root)/'native-renewal').exists() or (Path(root)/'native-renewal').is_symlink():
        tip(root)
        require(activation, 'native_renewal_activation_required')


def source_record(path):
    value=document(path)
    if value.get('kind')=='native-successor-enrollment-v1':
        validate_envelope(value)
        return value['source']
    return value


def selected_source(record_file, root):
    current=tip(root)
    saved=document(record_file)
    if current is None: return saved
    require(canonical(saved)==canonical(current), 'native_renewal_stale_enrollment')
    return current['source']


def capacity(dimension, used, limit, extra=1):
    require(type(used) is int and type(limit) is int and type(extra) is int and used>=0 and limit>=0 and extra>=0)
    if used + extra > limit:
        raise Refused(f'native_retention_capacity: {dimension} used={used} limit={limit}; {ACTION}')


def compatible(old_config, old_templates, new_config, new_templates):
    """Closed permitted differences, with canonical primitive/type comparison.

    All template versions, settings, profiles, budgets and topology remain exact;
    origin authorization is separately validated by both execution consumers.
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


def stopped_membership(owner, native, *, subset=False):
    rows = owner.db.execute('SELECT binding,authority,runtime,receipt,acked,phase,outcome FROM dispatches ORDER BY binding').fetchall()
    count = owner.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
    require(count == len(rows), 'native_history_unclean')
    capacity('history', count, 63)
    from service.worker_transport import Authority
    result = []
    for binding, raw, runtime, receipt, acked, phase, outcome in rows:
        authority = Authority(*json.loads(raw))
        if subset and canonical(p.native_binding(authority.context)) != canonical(native): continue
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
    require(value['kind'] == 'native-renewal-intent-v1' and type(value['epoch']) is int and 1 <= value['epoch'] <= 7)
    p._hex(value['request_id'],32); p._hex(value['stopped_sha256'],64)
    require(type(value['membership']) is list and len(value['membership']) < 64)
    require(len({r['binding'] for r in value['membership']})==len(value['membership']))
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
        saved_values=namespace(owner_root,resume=True,_begin=True)
        replays=[(k,v) for k,v in saved_values.items() if k.endswith('.intent') and v['request_id']==request_id]
        if replays:
            require(len(replays)==1)
            key,existing=replays[0]
            supplied,_,_=n._enrollment_snapshot(configuration_file,templates_file,owner_root)
            require(canonical(source_record(record_file))==canonical(existing['predecessor']) and canonical(supplied)==canonical(existing['successor']))
            require(canonical(existing['predecessor']['owner']) == canonical(dict(
                path=str(owner.root), device=owner.root.stat().st_dev,
                inode=owner.root.stat().st_ino, uid=os.getuid())), 'native_owner_binding')
            if key[:3]+'.commit' not in saved_values:
                predecessor,_,old_templates=n._enrollment_snapshot(
                    existing['predecessor']['configuration_file'],
                    existing['predecessor']['templates_file'],owner_root)
                require(canonical(predecessor)==canonical(existing['predecessor']), 'native_enrollment_changed')
                compatible(n.document(predecessor['configuration_file']), old_templates,
                           n.document(configuration_file), n.document(templates_file))
                require(canonical(stopped_membership(owner,predecessor['native'],subset=True)) ==
                        canonical(existing['membership']), 'native_history_unclean')
                require(stopped_fingerprint(owner)==existing['stopped_sha256'],'native_enrollment_changed')
            return publish(owner.root/'native-renewal'/key,existing)
        pending=[v for k,v in saved_values.items() if k.endswith('.intent') and k[:3]+'.commit' not in saved_values]
        if pending:
            require(len(pending)==1 and pending[0]['request_id']==request_id)
            predecessor=source_record(record_file)
            require(canonical(predecessor)==canonical(pending[0]['predecessor']))
        else: predecessor = selected_source(record_file, owner_root)
        require(canonical(predecessor['owner']) == canonical(dict(path=str(owner.root), device=owner.root.stat().st_dev,
            inode=owner.root.stat().st_ino, uid=os.getuid())), 'native_owner_binding')
        old_config = n.document(predecessor['configuration_file'])
        old_templates = n.document(predecessor['templates_file'])
        require(digest(old_config) == predecessor['configuration_sha256'] and
                digest(old_templates) == predecessor['templates_sha256'], 'native_enrollment_changed')
        successor, _, templates = n._enrollment_snapshot(configuration_file, templates_file, owner_root)
        require(canonical(successor['owner']) == canonical(predecessor['owner']), 'native_owner_binding')
        new_config=n.document(configuration_file)
        compatible(old_config, old_templates, new_config, templates)
        require(successor['configuration_file']!=predecessor['configuration_file'] and successor['templates_file']!=predecessor['templates_file'] or canonical(successor)==canonical(predecessor),'native_renewal_incompatible')
        if canonical(successor)!=canonical(predecessor):
            require(all(new_config[k]!=old_config[k] for k in ('cafile','certfile','keyfile')),'native_renewal_incompatible')
        values=namespace(owner.root,resume=True)
        epoch=1+len([k for k in values if k.endswith('.commit')])
        capacity('epochs',epoch,7,extra=0)
        prefix=f'{epoch:03d}'
        membership = stopped_membership(owner, predecessor['native'], subset=True)
        total=owner.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
        capacity('registrations',len(old_templates['registrations'])+total,63)
        # Existing closed spool accounting is global across epochs.
        check=offline_runtime(predecessor)
        extra=max((4 if row['context']['plan']['operation']=='voice_prepare' else 2)*p.native_execution(row['context']).reference_bytes+3*p.native_execution(row['context']).max_output_bytes+96000 for row in templates['registrations'])
        check._account(extra,new=True)
        values_now=namespace(owner.root,resume=True)
        historical_bytes=sum(v['charges']['host_bytes'] for k,v in values_now.items() if k.endswith('.export') and k[:3]!=prefix)
        historical_journal=sum(v['charges']['journal_bytes'] for k,v in values_now.items() if k.endswith('.export') and k[:3]!=prefix)
        capacity('host_bytes',historical_bytes+sum(2*(p.native_execution(row['snapshot'][1]).reference_bytes+p.native_execution(row['snapshot'][1]).max_output_bytes) for row in membership),64*1024**2,extra=max(2*(p.native_execution(row['context']).reference_bytes+p.native_execution(row['context']).max_output_bytes) for row in templates['registrations']))
        capacity('journal_bytes',historical_journal+sum(len(canonical(row)) for row in membership),1048576,extra=3*p.FRAME_LIMIT)
        intent = dict(kind='native-renewal-intent-v1', request_id=request_id, epoch=epoch,
                      predecessor=predecessor, successor=successor, membership=membership,
                      stopped_sha256=stopped_fingerprint(owner))
        validate_intent(intent)
        directory = owner.root/'native-renewal'
        if not directory.exists():
            os.mkdir(directory,0o700); sync(owner.root)
        n.directory(str(directory))
        # Exact bounded preflight, retaining any conflicting/partial namespace.
        require({f.name for f in directory.iterdir()} <= {
            f'{e:03d}.'+k+s for e in range(1,8) for k in ('intent','export','seal','enrollment','commit') for s in ('','.stage')})
        require(sum(f.lstat().st_size for f in directory.iterdir()) + len(canonical(intent)) <= TOTAL,
                'native_renewal_capacity')
        require(shutil.disk_usage(directory).free >= len(canonical(intent))+EXPORT*2,
                'native_renewal_capacity')
        return publish(directory/(prefix+'.intent'), intent)


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
    registrations=host_registrations(root)
    names = {'native.lock','native.json'} | {f'registration-{b}' for b in registrations} | ({'predecessor'} if (root/'predecessor').exists() else set()) | {r['token'] for r in records}
    if (root/'predecessor').exists():
        retained=document(root/'predecessor');validate_host_predecessor(retained)
        require(canonical(retained['native'])==canonical(native),'native_renewal_capacity_binding')
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
                      'records','charges','registration_bindings'))
    require(value['kind'] == 'native-retirement-export-v1')
    validate_native(value['native'])
    p._hex(value['intent_sha256'],64); p._hex(value['predecessor_sha256'],64); p._hex(value['request_id'],32)
    require(type(value['records']) is list and len(value['records']) < 64)
    require(type(value['registration_bindings']) is list and len(value['registration_bindings'])<=64)
    for binding in value['registration_bindings']: p._hex(binding,64)
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
            charges['allocation_registrations'] == len(value['registration_bindings']) and
            len(set(value['registration_bindings']))==len(value['registration_bindings']) and
            {r['binding'] for r in value['records']} <= set(value['registration_bindings']) and
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
            records=records,registration_bindings=sorted(set(host_registrations(root)) | {r['binding'] for r in records}),
            charges=dict(journal_records=len(records), allocation_registrations=len(set(host_registrations(root)) | {r['binding'] for r in records}),
                host_bytes=sum(2*(s.reference_bytes+s.max_output_bytes) for s in spec),
                journal_bytes=len(n.read(root/'native.json',1048576))+sum(len(canonical(v)) for v in host_registrations(root).values())))
        validate_export(export)
        # Host-local headroom is knowable here, before irreversible retirement.
        prior=document(root/'predecessor')['charges'] if (root/'predecessor').exists() else dict.fromkeys(export['charges'],0)
        templates=n.document(intent['successor']['templates_file'])
        origins=0 if (root/'predecessor').exists() else len(templates['registrations'])
        capacity('registrations',prior['allocation_registrations']+origins+export['charges']['allocation_registrations'],63)
        capacity('journal_records',prior['journal_records']+len(records),63)
        capacity('journal_bytes',prior['journal_bytes']+export['charges']['journal_bytes'],1048576,extra=3*p.FRAME_LIMIT)
        capacity('host_bytes',prior['host_bytes']+export['charges']['host_bytes'],64*1024**2,extra=max(2*(p.native_execution(row['context']).reference_bytes+p.native_execution(row['context']).max_output_bytes) for row in templates['registrations']))
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
        pass  # Per-binding runtime tuples are checked by completed_history.
        require(canonical(stopped_membership(owner,self.export['native'],subset=True)) == canonical(self.intent['membership']),
                'native_history_unclean')

    def collect(self, authority, runtime_id):
        r = self.records.get(authority.binding)
        require(r is not None and r['id'] == runtime_id and
                canonical(r['snapshot']) == authority._snapshot, 'native_history_evidence_unavailable')
        return copy.deepcopy({k:r[k] for k in ('receipt','proposal','output')})


def binding_seals(owner, membership):
    """Terminal rows only, never catalogue aliases/cursors or the SQLite file."""
    result={}
    for member in membership:
        binding=member['binding']
        row=owner.db.execute('SELECT job_id,seq FROM dispatches WHERE binding=?',(binding,)).fetchone()
        require(row is not None, 'native_history_unclean')
        jid,seq=row
        evidence={}
        for table,column,key in (('jobs','id',jid),('events','job_id',jid),('dispatches','binding',binding),
                ('dispatch_commands','dispatch_id',seq),('resource_reservations','id',jid),
                ('publications','job_id',jid),('publication_work','job_id',jid)):
            evidence[table]=sorted([list(r) for r in owner.db.execute(
                'SELECT * FROM '+table+' WHERE '+column+'=?',(key,))],key=canonical)
        context=member['snapshot'][1];principal=member['snapshot'][0]['principal_id']
        evidence['versions']=[]
        for version in context['plan']['voice_versions']:
            rows=owner.db.execute('SELECT * FROM catalogue_versions WHERE principal=? AND id=?',(principal,version['id'])).fetchall()
            require(len(rows)==1 and canonical(json.loads(rows[0][-1]))==canonical(version),'native_history_unclean')
            evidence['versions'].append(list(rows[0]))
        # Seal retained immutable media/transport bytes as well as SQL. This
        # also makes an acknowledged-commit replay validate its old dependencies
        # without requiring credentials for a later current epoch.
        from service.native_runtime import _read
        config=n.document(member_source(owner,binding)['configuration_file'])
        files={}
        asset_ids={a['id'] for a in context['plan']['assets']}
        for manifest, in owner.db.execute('SELECT manifest FROM publications WHERE job_id=?',(jid,)):
            asset_ids.update(a['id'] for a in json.loads(manifest)['outputs'])
        for asset_id in sorted(asset_ids):
            pth=owner.root/'assets'/asset_id
            files['asset/'+asset_id]=hashlib.sha256(_read(pth,24*1024**2)).hexdigest()
        for label,directory in (('spool',Path(config['spool'])/binding),('stage',Path(config['staging'])/member['snapshot'][0]['attempts'][0]['staging_id'])):
            from service.native_runtime import _directory
            _directory(directory,0o700 if label=='spool' else 0o770)
            entries=list(directory.iterdir());require(len(entries)<=8)
            for file in entries:
                files[label+'/'+file.name]=hashlib.sha256(_read(file,24*1024**2)).hexdigest()
        evidence['files']=files
        result[binding]=digest(evidence)
    return result


def validate_seal(value):
    p._closed(value, ('kind','intent_sha256','export_sha256','bindings'))
    require(value['kind']=='native-history-seal-v1')
    p._hex(value['intent_sha256'],64); p._hex(value['export_sha256'],64)
    require(type(value['bindings']) is dict and len(value['bindings'])<64)
    for key,v in value['bindings'].items(): p._hex(key,64); p._hex(v,64)


def validate_envelope(value):
    p._closed(value, ('kind','epoch','source','seal_sha256','export_sha256','origins'))
    require(value['kind']=='native-successor-enrollment-v1' and type(value['epoch']) is int and 1<=value['epoch']<=7)
    p._hex(value['seal_sha256'],64);p._hex(value['export_sha256'],64)
    # Reuse the exact closed source validator without granting origin voice authority.
    validate_intent(dict(kind='native-renewal-intent-v1',request_id='0'*32,epoch=value['epoch'],
        predecessor=value['source'],successor=value['source'],membership=[],stopped_sha256='0'*64))
    validate_origins(value['origins'])


def validate_origins(value):
    require(type(value) is list and 1<=len(value)<=64)
    keys=[]
    for row in value:
        p._closed(row, ('principal','origin_sha256','current_sha256'))
        require(type(row['principal']) is str and 1<=len(row['principal'])<=256)
        p._hex(row['origin_sha256'],64); p._hex(row['current_sha256'],64)
        keys.append((row['principal'],row['origin_sha256']))
    require(len(set(keys))==len(keys))


def origin_mapping(old, new, previous=None):
    """Carry stable original row identities through exact compatible hops."""
    require(len(old['registrations'])==len(new['registrations']))
    result=[]
    for a,b in zip(old['registrations'],new['registrations']):
        origin=digest(a)
        if previous is not None:
            matches=[r for r in previous if r['principal']==a['principal'] and r['current_sha256']==origin]
            require(len(matches)==1);origin=matches[0]['origin_sha256']
        result.append(dict(principal=a['principal'],origin_sha256=origin,current_sha256=digest(b)))
    validate_origins(result)
    return result


def validate_commit(value):
    p._closed(value, ('kind','epoch','request_id','enrollment_sha256','predecessor_commit_sha256'))
    require(value['kind']=='native-renewal-commit-v1' and type(value['epoch']) is int and 1<=value['epoch']<=7)
    p._hex(value['request_id'],32);p._hex(value['enrollment_sha256'],64)
    if value['predecessor_commit_sha256'] is not None: p._hex(value['predecessor_commit_sha256'],64)


class EpochEvidence:
    """Closed chain selector; current jobs always require live collect."""
    def __init__(self, owner, runtime, *, pending=None):
        values=namespace(owner.root,resume=pending is not None)
        self.retired={}; self.current=runtime.native; self.resuming=pending is not None
        self.sealed=[]
        for name,intent in sorted(values.items()):
            if not name.endswith('.intent'): continue
            prefix=name[:3]
            if prefix+'.commit' not in values and pending is None: raise Refused('native_renewal_pending')
            if prefix+'.export' not in values: continue
            evidence=RetiredEvidence(intent,values[prefix+'.export'])
            evidence.check_membership(owner,runtime)
            for binding in evidence.records:
                require(binding not in self.retired)
                self.retired[binding]=evidence
            if prefix+'.seal' in values:
                require(canonical(binding_seals(owner,intent['membership']))==canonical(values[prefix+'.seal']['bindings']),
                        'native_history_unclean')
        if pending is not None:
            evidence=pending
            evidence.check_membership(owner,runtime)
            for binding in evidence.records:
                require(binding not in self.retired or self.retired[binding].export==evidence.export)
                self.retired[binding]=evidence

    def native(self, binding):
        return self.retired[binding].export['native'] if binding in self.retired else self.current

    def collect(self, authority, runtime_id, runtime):
        if authority.binding in self.retired: return self.retired[authority.binding].collect(authority,runtime_id)
        return runtime._rpc('collect',{},authority=authority,runtime_id=runtime_id)


def offline_runtime(source):
    from service.native_runtime import NativeRuntime
    config=n.document(source['configuration_file'])
    require(digest(config)==source['configuration_sha256'],'native_enrollment_changed')
    def unavailable(_): raise Refused('native_history_evidence_unavailable')
    return NativeRuntime(host=config['host'],port=config['port'],tls=None,
        peer_sha256=config['peer_sha256'],native=source['native'],spool=config['spool'],
        staging=config['staging'],local_import_uid=os.getuid(),connector=unavailable)


def seal(intent_file, *, export_file):
    from service.core import native_maintenance
    from service.native_activation import completed_history
    intent=document(intent_file);validate_intent(intent)
    export=document(export_file);evidence=RetiredEvidence(intent,export)
    with native_maintenance(intent['predecessor']['owner']['path']) as owner:
        directory=owner.root/'native-renewal';prefix=f"{intent['epoch']:03d}"
        require(canonical(document(directory/(prefix+'.intent')))==canonical(intent))
        require(stopped_fingerprint(owner)==intent['stopped_sha256'],'native_enrollment_changed')
        runtime=offline_runtime(intent['predecessor'])
        completed_history(owner,runtime,retired=EpochEvidence(owner,runtime,pending=evidence))
        require(stopped_fingerprint(owner)==intent['stopped_sha256'],'native_enrollment_changed')
        publish(directory/(prefix+'.export'),export)
        result=dict(kind='native-history-seal-v1',intent_sha256=digest(intent),
            export_sha256=digest(export),bindings=binding_seals(owner,intent['membership']))
        return publish(directory/(prefix+'.seal'),result)


def commit(intent_file, *, export_file, connector=None):
    from service.core import native_maintenance
    from service.native_activation import completed_history
    intent=document(intent_file);validate_intent(intent)
    with native_maintenance(intent['predecessor']['owner']['path']) as owner:
        values=namespace(owner.root,resume=True);prefix=f"{intent['epoch']:03d}"
        if prefix+'.commit' in values:
            require(canonical(values[prefix+'.intent'])==canonical(intent) and canonical(values[prefix+'.export'])==canonical(document(export_file)))
            require(canonical(binding_seals(owner,intent['membership']))==canonical(values[prefix+'.seal']['bindings']),'native_history_unclean')
            return publish(owner.root/'native-renewal'/(prefix+'.commit'),values[prefix+'.commit'])
    seal(intent_file,export_file=export_file)
    with native_maintenance(intent['predecessor']['owner']['path']) as owner:
        directory=owner.root/'native-renewal';prefix=f"{intent['epoch']:03d}"
        namespace(owner.root,resume=True)
        require(stopped_fingerprint(owner)==intent['stopped_sha256'],'native_enrollment_changed')
        source=intent['successor']
        require(source['native']['boot'] not in {v['predecessor']['native']['boot'] for k,v in namespace(owner.root,resume=True).items() if k.endswith('.intent')},'native_renewal_incompatible')
        args=dict(configuration_file=source['configuration_file'],templates_file=source['templates_file'],owner_root=str(owner.root))
        before,_,templates=n._enrollment_snapshot(**args)
        require(canonical(before)==canonical(source),'native_enrollment_changed')
        old=intent['predecessor']
        old_templates=n.document(old['templates_file'])
        require(digest(old_templates)==old['templates_sha256'],'native_enrollment_changed')
        compatible(n.document(old['configuration_file']),old_templates,n.document(source['configuration_file']),templates)
        runtime=n.enrolled_runtime(source['configuration_file'],connector=connector,_completed_history=True)
        hello=runtime.call('hello',{})
        require(hello['launches']==0 and hello['quarantined'] is False and
                canonical(hello['inventory'])==canonical(templates['inventory']) and
                all(hello[k]==v for k,v in source['native'].items()),'native_renewal_hello')
        expected=host_predecessor(intent,directory)
        require(hello.get('retention')==digest(expected),'native_renewal_capacity_binding')
        completed_history(owner,runtime,retired=EpochEvidence(owner,runtime,pending=RetiredEvidence(intent,document(export_file))))
        values=namespace(owner.root,resume=True)
        previous=values.get(f"{intent['epoch']-1:03d}.enrollment")
        envelope=dict(kind='native-successor-enrollment-v1',epoch=intent['epoch'],source=source,
            seal_sha256=digest(document(directory/(prefix+'.seal'))),export_sha256=digest(document(export_file)),
            origins=origin_mapping(old_templates,templates,None if previous is None else previous['origins']))
        validate_envelope(envelope)
        publish(directory/(prefix+'.enrollment'),envelope)
        after,_,_=n._enrollment_snapshot(**args)
        require(canonical(after)==canonical(source) and stopped_fingerprint(owner)==intent['stopped_sha256'], 'native_enrollment_changed')
        last=runtime.call('hello',{})
        require(canonical(last)==canonical(hello),'native_renewal_hello')
        after,_,_=n._enrollment_snapshot(**args)
        require(canonical(after)==canonical(source),'native_enrollment_changed')
        result=dict(kind='native-renewal-commit-v1',epoch=intent['epoch'],request_id=intent['request_id'],
            enrollment_sha256=digest(envelope),predecessor_commit_sha256=None if previous is None else
            digest(values[f"{intent['epoch']-1:03d}.commit"]))
        return publish(directory/(prefix+'.commit'),result)


def host_predecessor(intent, directory):
    """Explicitly provisioned immutable cumulative host authorization."""
    require(intent['successor']['native']['boot']!=intent['predecessor']['native']['boot'],'native_renewal_incompatible')
    exports=[]
    for epoch in range(1,intent['epoch']+1):
        value=document(Path(directory)/f'{epoch:03d}.export');validate_export(value);exports.append(value)
    origins=n.document(intent['predecessor']['templates_file'])['registrations']
    charges={key:sum(e['charges'][key] for e in exports) for key in exports[0]['charges']}
    charges['allocation_registrations']+=len(origins)
    for key,limit in (('journal_records',63),('allocation_registrations',63),('host_bytes',64*1024**2),('journal_bytes',1048576)):
        capacity(key,charges[key],limit)
    first = document(Path(directory)/'001.intent')['predecessor']
    original = n.document(first['templates_file'])
    current = n.document(intent['successor']['templates_file'])
    previous = None if intent['epoch'] == 1 else document(Path(directory)/f"{intent['epoch']-1:03d}.enrollment")['origins']
    mapping = origin_mapping(n.document(intent['predecessor']['templates_file']), current, previous)
    p.origin_template_pairs(original['registrations'], current['registrations'], mapping)
    return dict(kind='native-host-predecessor-v1',native=intent['successor']['native'],
        export_sha256=[digest(e) for e in exports],charges=charges,
        origin_templates=original['registrations'],current_templates=current['registrations'],origins=mapping)


def provision(intent_file, *, export_file, host_root):
    """Private custody transfer before constructing the zero-launch successor."""
    seal(intent_file,export_file=export_file)
    intent=document(intent_file);root=n.directory(str(host_root))
    require(not any(root.iterdir()) or {f.name for f in root.iterdir()} <= {'predecessor','predecessor.stage'})
    return publish(root/'predecessor',host_predecessor(intent,Path(intent_file).parent))


def preallocate(core, context):
    """Before creating a queued job; replay has already returned."""
    if not context.get('execution',{}).get('native'): return
    count=core.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
    capacity('history',count,63)
    templates=getattr(core,'_native_retention_templates',None)
    current=tip(core.root)
    if current is not None: templates=n.document(current['source']['templates_file'])
    if templates is not None: capacity('registrations',count+len(templates['registrations']),63)
    runtime=getattr(core,'_native_retention_runtime',None)
    if runtime is not None:
        spec=p.native_execution(context)
        runtime._account((4 if context['plan']['operation']=='voice_prepare' else 2)*spec.reference_bytes+3*spec.max_output_bytes+96000,new=True)
    # Bound prospective metadata before creating the queued job.
    from service import recovery
    rows=size=0
    for table,fields in recovery._evidence_columns().items():
        if core.db.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?',(table,)).fetchone() is None: continue
        columns=fields.split()
        charge='+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in columns)
        a,b=core.db.execute('SELECT count(*),coalesce(sum('+charge+'),0) FROM '+table).fetchone();rows+=a;size+=b
    capacity('history_rows',rows,2048,extra=64)
    capacity('history_bytes',size,TOTAL,extra=8*48000)
    from worker import native_protocol as protocol
    spec=protocol.native_execution(context)
    total=0
    for raw, in core.db.execute('SELECT context FROM jobs'):
        old=protocol.native_execution(__import__('json').loads(raw))
        total+=2*(old.reference_bytes+old.max_output_bytes)
    capacity('host_bytes',total,64*1024**2,extra=2*(spec.reference_bytes+spec.max_output_bytes))
    journal=0
    for authority,receipt in core.db.execute('SELECT authority,receipt FROM dispatches'):
        journal+=2*len(authority.encode())+len((receipt or '').encode())+2048
    for message,manifest in core.db.execute('SELECT message,manifest FROM publications'):
        journal+=len(message.encode())+len(manifest.encode())
    capacity('journal_bytes',journal,1048576,extra=3*p.FRAME_LIMIT)


def validate_host_predecessor(value):
    p._closed(value, ('kind','native','export_sha256','charges','origin_templates','current_templates','origins'))
    require(value['kind']=='native-host-predecessor-v1')
    p.origin_template_pairs(value['origin_templates'],value['current_templates'],value['origins'])
    validate_native(value['native'])
    require(type(value['export_sha256']) is list and 1<=len(value['export_sha256'])<=7)
    for item in value['export_sha256']: p._hex(item,64)
    require(len(set(value['export_sha256']))==len(value['export_sha256']))
    p._closed(value['charges'], ('journal_records','allocation_registrations','host_bytes','journal_bytes'))
    for key,limit in (('journal_records',63),('allocation_registrations',63),('host_bytes',64*1024**2),('journal_bytes',1048576)):
        capacity(key,value['charges'][key],limit)


def host_registrations(root):
    result={}
    from service.worker_transport import Authority
    files=list(Path(root).glob('registration-*'))
    require(len(files)<=64)
    for file in files:
        binding=file.name.removeprefix('registration-');p._hex(binding,64)
        snapshot=document(file,48000);authority=Authority(*snapshot)
        require(authority.binding==binding)
        result[binding]=snapshot
    return result


def member_source(owner, binding):
    directory=owner.root/'native-renewal'
    matches=[]
    for path in sorted(directory.glob('*.intent')):
        intent=document(path);validate_intent(intent)
        if any(row['binding']==binding for row in intent['membership']): matches.append(intent['predecessor'])
    require(len(matches)==1,'native_history_unclean')
    return matches[0]


def origin_authorization(root):
    """Resolve retained originals only through the validated committed chain."""
    values = namespace(root)
    if not values: return None
    current = tip(root)
    require(current is not None)
    first = values['001.intent']['predecessor']
    original = n.document(first['templates_file'])['registrations']
    rows = n.document(current['source']['templates_file'])['registrations']
    p.origin_template_pairs(original, rows, current['origins'])
    return original, rows, current['origins']
