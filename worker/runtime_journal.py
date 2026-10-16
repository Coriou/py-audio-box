"""Protected bounded durable evidence, not a service job database."""
import fcntl
import hashlib
import json
import os
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
        fd = os.open(self.root / 'journal.json', os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            encoded = stream.read(MAX_BYTES+1)
        if len(encoded) > MAX_BYTES:
            raise ValueError('journal_limit')
        envelope = json.loads(encoded)
        if (not isinstance(envelope, dict) or set(envelope) != {'payload','sha256'} or
                hashlib.sha256(canonical(envelope['payload'])).hexdigest() != envelope['sha256']):
            raise ValueError('journal_checksum')
        self.value = envelope['payload']
        if (set(self.value) != {'version', 'lane', 'records'} or self.value['version'] != 1 or
                not isinstance(self.value['lane'], str) or not isinstance(self.value['records'], list) or
                len(self.value['records']) > MAX_RECORDS):
            raise ValueError('invalid_journal')
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
        if len(canonical(self.value)) > MAX_BYTES:
            raise ValueError('journal_limit')

    def save(self, value):
        data = canonical(dict(payload=value, sha256=hashlib.sha256(canonical(value)).hexdigest()))
        if len(data) > MAX_BYTES:
            raise ValueError('journal_limit')
        fd = os.open(self.root / 'journal.tmp', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.root / 'journal.tmp', self.root / 'journal.json')
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.value = value
