"""Protected bounded durable evidence, not a service job database."""
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from worker.transport import canonical

MAX_RECORDS = 64
MAX_UNACKED = 8
MAX_BYTES = 1024 * 1024


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = open(self.root / 'lane.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self._load()
        except BaseException:
            self.lock.close()
            raise

    def _load(self, *, policy=None):
        self.maintenance = read_maintenance(self.root,policy=policy)
        if policy is not None:
            from service.install_policy import read_private
            encoded = read_private(self.root/'journal.json',MAX_BYTES,policy=policy,role='receipts')
        else:
            fd = os.open(self.root / 'journal.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                info=os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_size>MAX_BYTES:
                    raise ValueError('unsafe_journal_file')
                encoded = stream.read(MAX_BYTES+1)
        if len(encoded) > MAX_BYTES:
            raise ValueError('journal_limit')
        envelope = json.loads(encoded)
        if (not isinstance(envelope, dict) or set(envelope) != {'payload','sha256'} or
                hashlib.sha256(canonical(envelope['payload'])).hexdigest() != envelope['sha256']):
            raise ValueError('journal_checksum')
        self.value = envelope['payload']
        if (set(self.value) not in ({'version', 'lane', 'records'}, {'version', 'lane', 'records', 'maintenance'}) or self.value['version'] != 1 or
                not isinstance(self.value['lane'], str) or not isinstance(self.value['records'], list) or
                len(self.value['records']) > MAX_RECORDS):
            raise ValueError('invalid_journal')
        if 'maintenance' in self.value and (self.maintenance is None or self.value['maintenance']!=self.maintenance):
            raise ValueError('missing_required_maintenance_fence')
        if self.maintenance is not None and self.value.get('maintenance')!=self.maintenance:
            raise ValueError('missing_journal_maintenance_binding')
        identities = set()
        for row in self.value['records']:
            fields = {'id', 'allocation', 'phase', 'acked', 'uid', 'boot'}
            if not isinstance(row, dict) or set(row) not in (fields, fields | {'receipt', 'proposal'}):
                raise ValueError('invalid_record')
            if (row['phase'] not in ('prepared', 'started', 'terminal') or
                    type(row['acked']) is not bool or row['id'] in identities or
                    not isinstance(row['id'], str) or not row['id'].startswith(self.value['lane']+'/') or
                    type(row['uid']) is not int or not 20000 <= row['uid'] < 20000+MAX_RECORDS):
                raise ValueError('invalid_record_identity')
            identities.add(row['id'])
            allocation = row['allocation']
            if (not isinstance(allocation, dict) or set(allocation) != {'snapshot','binding','fixture'} or
                    not isinstance(allocation['snapshot'], list) or len(allocation['snapshot']) != 4 or
                    hashlib.sha256(canonical(allocation['snapshot'])).hexdigest() != allocation['binding']):
                raise ValueError('invalid_record_binding')
            if row['phase'] == 'terminal':
                receipt = row.get('receipt')
                if (not isinstance(receipt, dict) or set(receipt) != {'runtime_id','binding','reaped','timely','original_deadline','engine_uid','reap_method','reaped_children'} or
                        receipt['runtime_id'] != row['id'] or receipt['binding'] != allocation['binding'] or
                        receipt['reaped'] is not True or receipt['reap_method'] != 'pid1-waitpid-echild' or
                        type(receipt['reaped_children']) is not int or receipt['reaped_children'] < 0 or receipt['engine_uid'] != row['uid'] or type(receipt['timely']) is not bool or
                        receipt['original_deadline'] != allocation['snapshot'][0]['deadline']):
                    raise ValueError('invalid_terminal_receipt')
                proposal = row.get('proposal')
                if not row['acked'] and receipt['timely'] and proposal is None:
                    raise ValueError('missing_timely_proposal')
                if proposal is not None:
                    from contracts import validate as oracle
                    if not isinstance(proposal, dict) or set(proposal) != {'message','manifest'}:
                        raise ValueError('invalid_cached_proposal')
                    job, context, assets, profiles = allocation['snapshot']
                    oracle.worker_message(proposal['message'],job,assets,transport_owner=job['attempts'][-1]['owner_id'],context=context,current_profiles=profiles)
                    if proposal['message']['kind'] != 'result':
                        raise ValueError('invalid_cached_result')
                    oracle.validate('result_manifest',proposal['manifest'])
                    for field in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256'):
                        if proposal['message'][field] != proposal['manifest'][field]:
                            raise ValueError('invalid_cached_manifest_tuple')
            elif row['acked'] or 'receipt' in row or 'proposal' in row:
                raise ValueError('invalid_nonterminal_receipt')
        if self.maintenance is not None and self.value['lane']!=self.maintenance['lane']:
            raise ValueError('maintenance_journal_lane_mismatch')
        if len(canonical(self.value)) > MAX_BYTES:
            raise ValueError('journal_limit')

    def save(self, value):
        _durable_save(self.root,value)
        self.value = value


def journal_bytes(value):
    data=canonical(dict(payload=value,sha256=hashlib.sha256(canonical(value)).hexdigest()))
    if len(data)>MAX_BYTES: raise ValueError('journal_limit')
    return data


def read_snapshot(root, *, policy):
    """Detached validated read only; caller holds the existing lane lock.

    Never construct Journal here: its ordinary constructor opens lane.lock for
    writing. This shares the complete record/receipt validator with startup.
    """
    from service.install_policy import require, parent_fd, check, identity
    require(policy)
    root=Path(root)
    with parent_fd(root) as (parent,name):
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        try:
            before=os.fstat(fd);check(before,policy,'receipts',directory=True)
            reader=Journal.__new__(Journal);reader.root=root
            reader._load(policy=policy)
            with parent_fd(root) as (current,leaf):
                if (identity(os.fstat(fd))!=identity(before) or
                        identity(os.stat(leaf,dir_fd=current,follow_symlinks=False))!=identity(before)):
                    raise ValueError('installed_root_changed')
            return reader.value,reader.maintenance
        finally:os.close(fd)


def _durable_save(root,value,*,fault=lambda point:None,exclusive=False):
    data=journal_bytes(value)
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if exclusive else os.O_TRUNC)
    fd=os.open(root/'journal.tmp',flags,0o600)
    with os.fdopen(fd,'wb') as stream:
        if stream.write(data)!=len(data): raise OSError('journal_short_write')
        stream.flush(); fault('journal-write')
        os.fsync(stream.fileno()); fault('journal-fsync')
    os.replace(root/'journal.tmp',root/'journal.json'); fault('journal-replace')
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    fault('journal-dirsync')


def _maintenance_replace(root,value,*,fault=lambda point:None):
    """Offline caller holds both normal locks and validates proof/archive chain.

    Fixed managed names only; no normal constructor bypass of pending startup.
    The caller durably records the exact creation attempt before entering here.
    """
    if (set(value)!={'version','lane','records'} or value['version']!=1 or value['records']!=[]):
        raise ValueError('invalid_maintenance_journal')
    fence=json.loads((root/'p1-maintenance.json').read_bytes())
    maintenance_record(fence)
    if fence['step']!='pending' or fence['new_lane']!=value['lane']: raise ValueError('maintenance_rotation_binding')
    _durable_save(root,value,fault=fault,exclusive=True)


def maintenance_record(value):
    """Closed receipt-side fence shared by offline readers and startup only."""
    import re
    keys={'revision','deployment_id','recovery_id','plan_id','plan_digest','proof_digest','old_lane','new_lane','step','certificate'}
    if not isinstance(value,dict) or set(value)!=keys or type(value['revision'])!=int or value['revision']!=1:
        raise ValueError('invalid_maintenance_fence')
    for key in ('deployment_id','recovery_id','plan_id','old_lane','new_lane'):
        if not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{32}',value[key]): raise ValueError('invalid_maintenance_identity')
    for key in ('plan_digest','proof_digest'):
        if not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{64}',value[key]): raise ValueError('invalid_maintenance_digest')
    if value['old_lane']==value['new_lane'] or value['step'] not in ('pending','evidence','journal','core-complete','complete'):
        raise ValueError('invalid_maintenance_step')
    cert=value['certificate']
    if value['step']!='complete':
        if cert is not None: raise ValueError('premature_maintenance_certificate')
    else:
        cert_keys={'deployment_id','recovery_id','plan_digest','proof_digest','lane','epoch','certificate_digest'}
        if not isinstance(cert,dict) or set(cert)!=cert_keys: raise ValueError('invalid_maintenance_certificate')
        for key in ('deployment_id','recovery_id','plan_digest','proof_digest'):
            if cert[key]!=value[key]: raise ValueError('maintenance_certificate_binding')
        if cert['lane']!=value['new_lane'] or not isinstance(cert['epoch'],str) or not re.fullmatch('[0-9a-f]{32}',cert['epoch']): raise ValueError('maintenance_certificate_lane')
        if not isinstance(cert['certificate_digest'],str) or not re.fullmatch('[0-9a-f]{64}',cert['certificate_digest']): raise ValueError('maintenance_certificate_digest')
    return value


def read_maintenance(root, *, policy=None):
    """Protected managed names only, bounded no-follow/nonblocking reads."""
    import stat
    root=Path(root)
    if policy is not None:
        from service.install_policy import require, read_private
        require(policy)
    def read(name,limit=65536,*,document=True):
        if policy is not None:
            try:data=read_private(root/name,limit,policy=policy,role='receipts')
            except FileNotFoundError:return None
            return json.loads(data) if document else data
        try: fd=os.open(root/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        except FileNotFoundError: return None
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_size>limit or info.st_mode & 0o077:
                raise ValueError('unsafe_maintenance_file')
            data=stream.read(limit+1)
        if len(data)>limit: raise ValueError('maintenance_limit')
        return json.loads(data) if document else data
    marker=read('p1-b2-required.json'); fence=read('p1-maintenance.json')
    if marker is None and fence is None: return None
    if not isinstance(marker,dict) or set(marker)!={'revision','deployment_id'} or type(marker['revision'])!=int or marker['revision']!=1:
        raise ValueError('missing_required_maintenance_marker')
    if fence is None: raise ValueError('missing_required_maintenance_fence')
    maintenance_record(fence)
    if marker['deployment_id']!=fence['deployment_id']: raise ValueError('maintenance_deployment_mismatch')
    if fence['step']!='complete': raise ValueError('maintenance_pending')
    prefix='p1-recovery-'+fence['recovery_id']
    archive=read(prefix+'.certificate.json')
    journal=read(prefix+'.journal',MAX_BYTES,document=False)
    validate_maintenance_archive(fence,archive,journal)
    return fence['certificate']


def validate_maintenance_archive(fence,archive,journal):
    """Receipt-side immutable evidence, verified under lane.lock before boot."""
    import re
    def closed(value,keys):
        if not isinstance(value,dict) or set(value)!=set(keys.split()):raise ValueError('maintenance_archive_fields')
    def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
    def hash_value(value):
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('maintenance_archive_digest')
    closed(archive,'certificate steps')
    if archive['certificate']!=fence['certificate'] or journal is None:raise ValueError('maintenance_archive_binding')
    steps=archive['steps'];names=('pending','evidence','journal','core-complete')
    if not isinstance(steps,list) or len(steps)!=4:raise ValueError('maintenance_archive_steps')
    previous=None
    for name,step in zip(names,steps):
        closed(step,'recovery_id step input_digest output_digest output lane epoch')
        if step['recovery_id']!=fence['recovery_id'] or step['step']!=name or step['lane']!=fence['new_lane'] or step['epoch']!=fence['certificate']['epoch']:raise ValueError('maintenance_archive_step_binding')
        hash_value(step['input_digest']);hash_value(step['output_digest'])
        if previous is not None and previous!=step['input_digest']:raise ValueError('maintenance_archive_chain')
        if digest(dict(step=name,input_digest=step['input_digest'],output=step['output']))!=step['output_digest']:raise ValueError('maintenance_archive_output')
        closed(step['output'],{
            'pending':'recovery_id plan_digest proof_digest before_digest',
            'evidence':'archive_id snapshot_digest manifest_digest journal_state journal_digest',
            'journal':'old_lane new_lane journal',
            'core-complete':'before_digest prior_epoch new_epoch actions diagnostic_decisions highwater budget_after monitor_state rows'}[name])
        previous=step['output_digest']
    pending,evidence,rotation,core=[step['output'] for step in steps]
    if (pending!={k:fence[k] for k in ('recovery_id','plan_digest','proof_digest')}|{'before_digest':steps[0]['input_digest']} or
        core['before_digest']!=pending['before_digest'] or core['new_epoch']!=fence['certificate']['epoch'] or
        rotation!=dict(old_lane=fence['old_lane'],new_lane=fence['new_lane'],journal=dict(version=1,lane=fence['new_lane'],records=[])) or
        digest(steps[-1])!=fence['certificate']['certificate_digest']):raise ValueError('maintenance_archive_certificate')
    for key in ('snapshot_digest','manifest_digest','journal_digest'):hash_value(evidence[key])
    if evidence['journal_state']=='valid':
        original=json.loads(journal);closed(original,'payload sha256')
        payload=original['payload'];closed(payload,'version lane records maintenance' if 'maintenance' in payload else 'version lane records')
        if payload['lane']!=fence['old_lane'] or type(payload['version'])!=int or payload['version']!=1 or not isinstance(payload['records'],list) or len(payload['records'])>MAX_RECORDS or digest(payload)!=original['sha256']:raise ValueError('maintenance_archive_journal')
        actual=digest(original)
    elif evidence['journal_state'] in ('missing','unreadable'):
        if evidence['journal_state']=='missing' and journal:raise ValueError('maintenance_archive_journal')
        actual=hashlib.sha256(journal).hexdigest()
    else:raise ValueError('maintenance_archive_journal')
    if actual!=evidence['journal_digest']:raise ValueError('maintenance_archive_journal_digest')
