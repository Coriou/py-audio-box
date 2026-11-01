"""Daemon-local immutable model store. No acquisition, Docker or installation writes.

Publication is last; attempts and lock inodes are permanent. All paths below the
catalog and payload roots are resolved through held no-follow descriptors.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import PurePosixPath
import re
import secrets
import stat
import time

from service.daemon_client import encoded, parse, file_identity
from service import daemon_volumes as dv

SCHEMA = 'shared-model-store-v1'
CATALOG = 'pab-model-store-v1'
CATALOG_ROOT = '/model-catalog'
MODEL_ROOT = '/models'
MAX_RECORD = 1024 * 1024
MAX_FILE = 2 * 1024**3
MAX_TOTAL = 4 * 1024**3
MAX_ENTRIES = 4096
CHUNK = 256 * 1024
OWNER = (10001, 10001)
CATALOG_OWNER = (0, 0)
HEX = re.compile(r'[0-9a-f]{64}')
ATTEMPT = re.compile(r'[0-9a-f]{32}')


def closed(v, fields):
    dv.closed(v, fields)


def digest(v):
    return hashlib.sha256(encoded(v)).hexdigest()


def token(v, pattern=HEX):
    if type(v) is not str or not pattern.fullmatch(v):
        raise ValueError('shared_identity')
    return v


def remaining(end):
    if time.monotonic() >= end:
        raise ValueError('shared_deadline')


def deadline(expires):
    # Convert once, never send host monotonic timestamps across the VM boundary.
    if type(expires) not in (int, float) or not 0 < expires-time.time() <= 600:
        raise ValueError('shared_expiry')
    left=expires-time.time()
    if left <= 0: raise ValueError('shared_expiry')
    return time.monotonic() + left


def labels(key=None, attempt=None):
    result = {'speech-box.model-store': SCHEMA}
    if key is not None:
        token(key); token(attempt, ATTEMPT)
        result.update({'speech-box.manifest': key, 'speech-box.attempt': attempt})
    return result


def volume(v, key=None, attempt=None):
    closed(v, 'Name Driver Options Scope Mountpoint CreatedAt Labels')
    name = CATALOG if key is None else 'pab-model-sha256-'+token(key)+'-'+token(attempt, ATTEMPT)
    path = v['Mountpoint']
    if (v['Name'] != name or v['Driver'] != 'local' or v['Options'] not in (None, {}) or
            v['Scope'] != 'local' or v['Labels'] != labels(key, attempt) or
            not dv.canonical_mountpoint(path) or
            type(v['CreatedAt']) is not str or not 1 <= len(v['CreatedAt']) <= 128):
        raise ValueError('shared_volume')
    return v


def descriptor(value):
    # Lazy: the ordinary operator client's import/help remains stdlib-only.
    from service.provisioning import validate_expected, bundle_manifest, manifest_bytes
    value = validate_expected(value)
    if value['backend'] != 'qwen':
        raise ValueError('shared_qwen_descriptor')
    raw = manifest_bytes(bundle_manifest(value))
    entries = value['files']
    if (len(raw) > 65536 or any(e['bytes'] > MAX_FILE or PurePosixPath(e['path']).name == 'manifest.json' for e in entries) or
            sum(e['bytes'] for e in entries)+len(raw) > MAX_TOTAL):
        raise ValueError('shared_payload_budget')
    key = hashlib.sha256(raw).hexdigest()
    files = {'bundles/'+key+'/manifest.json': (len(raw), key)}
    for e in entries:
        files['bundles/'+key+'/'+e['path']] = (e['bytes'], e['sha256'])
    dirs = {''}
    for name in files:
        dirs.update(str(p) for p in PurePosixPath(name).parents if str(p) != '.')
    if len(dirs)+len(files) > MAX_ENTRIES:
        raise ValueError('shared_inventory_budget')
    return key, raw, files, dirs


def open_root(path):
    path = os.path.abspath(path)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.split('/')[1:]:
            if not part: continue
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd); raise


@contextmanager
def root_fd(path):
    """No symlink ancestor, including the selected root itself; repeat on exit."""
    fd = open_root(path)
    try:
        before = os.fstat(fd)
        yield fd
        checked = open_root(path)
        try:
            if ((os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (before.st_dev, before.st_ino) or
                    (os.fstat(checked).st_dev, os.fstat(checked).st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError('shared_root_changed')
        finally: os.close(checked)
    finally: os.close(fd)


def policy(s, directory, owner, mode):
    if ((stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)) is not True or
            (s.st_uid, s.st_gid) != owner or stat.S_IMODE(s.st_mode) != mode or
            (not directory and s.st_nlink != 1)):
        raise ValueError('shared_filesystem_policy')


def names_at(fd, limit=MAX_ENTRIES, end=None):
    names=[]
    with os.scandir(fd) as entries:
        for entry in entries:
            if end is not None: remaining(end)
            if len(names)>=limit: raise ValueError('shared_inventory_budget')
            names.append(entry.name)
    return names


def disjoint(catalog, candidate):
    a,b=(v['Mountpoint'].rstrip('/') for v in (catalog,candidate))
    if not a or not b or a==b or a.startswith(b+'/') or b.startswith(a+'/') or catalog['Name']==candidate['Name']:
        raise ValueError('shared_volume_alias')


def read_at(fd, name, limit=MAX_RECORD):
    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(child)
        policy(before, False, CATALOG_OWNER, 0o600)
        if before.st_size > limit: raise ValueError('shared_record_budget')
        raw = bytearray()
        while len(raw) <= before.st_size:
            block=os.read(child,min(65536,before.st_size+1-len(raw)))
            if not block: break
            raw.extend(block)
        raw=bytes(raw)
        if (len(raw) != before.st_size or file_identity(before) != file_identity(os.fstat(child)) or
                file_identity(before) != file_identity(os.stat(name, dir_fd=fd, follow_symlinks=False))):
            raise ValueError('shared_record_changed')
        v = parse(raw)
        if encoded(v) != raw: raise ValueError('shared_record_encoding')
        return v
    finally: os.close(child)


def write_at(fd, name, value):
    raw = encoded(value)
    if len(raw) > MAX_RECORD: raise ValueError('shared_record_budget')
    child = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        os.fchmod(child, 0o600)
        offset = 0
        while offset < len(raw): offset += os.write(child, raw[offset:])
        os.fsync(child)
    finally: os.close(child)
    os.fsync(fd)


@contextmanager
def flock(fd, end, *, wait=30):
    limit = min(end, time.monotonic()+wait)
    while True:
        remaining(limit)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(min(.02, max(0, limit-time.monotonic())))
    try: yield
    finally: fcntl.flock(fd, fcntl.LOCK_UN)


class Catalog:
    def __init__(self, path, daemon, metadata, end):
        self.path, self.daemon, self.metadata, self.end = path, dv.daemon(daemon), volume(metadata), end
        self.schema = dict(schema=SCHEMA, daemon=daemon, catalog=metadata)

    def init(self):
        # Root inode itself excludes separate clients during first normalization.
        with root_fd(self.path) as fd, flock(fd, self.end):
            remaining(self.end)
            names = set(names_at(fd,6,self.end))
            if 'schema.json' in names:
                self.validate(fd); return self.schema
            if names: raise ValueError('shared_partial_bootstrap')
            s = os.fstat(fd)
            if not stat.S_ISDIR(s.st_mode) or (s.st_uid, s.st_gid) != CATALOG_OWNER:
                raise ValueError('shared_fresh_catalog_owner')
            write_at(fd, 'normalization-intent.json', self.schema)
            os.fchmod(fd, 0o700); os.fsync(fd)
            for name in ('locks', 'attempts', 'published', 'references'):
                os.mkdir(name, 0o700, dir_fd=fd)
                d = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try: os.fchmod(d, 0o700); os.fsync(d)
                finally: os.close(d)
            os.fsync(fd)
            write_at(fd, 'schema.json', self.schema)
            self.validate(fd)
        return self.schema

    def validate(self, fd):
        policy(os.fstat(fd), True, CATALOG_OWNER, 0o700)
        if set(names_at(fd,6,self.end)) != {'normalization-intent.json','schema.json','locks','attempts','published','references'}:
            raise ValueError('shared_catalog_inventory')
        if read_at(fd, 'schema.json') != self.schema or read_at(fd, 'normalization-intent.json') != self.schema:
            raise ValueError('shared_catalog_identity')
        for name in ('locks','attempts','published','references'):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try: policy(os.fstat(child), True, CATALOG_OWNER, 0o700)
            finally: os.close(child)

    @contextmanager
    def held(self, key):
        token(key)
        with root_fd(self.path) as fd:
            self.validate(fd)
            locks = os.open('locks', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            lock = None
            locks_identity=(os.fstat(locks).st_dev,os.fstat(locks).st_ino)
            try:
                lock = os.open(key, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=locks)
                policy(os.fstat(lock), False, CATALOG_OWNER, 0o600)
                if os.fstat(lock).st_size!=0: raise ValueError('shared_lock_contents')
                os.fsync(lock); os.fsync(locks)
                identity = file_identity(os.fstat(lock))
                with flock(lock, self.end):
                    def stable():
                        remaining(self.end); self.validate(fd)
                        current=os.stat('locks',dir_fd=fd,follow_symlinks=False)
                        if (current.st_dev,current.st_ino)!=locks_identity: raise ValueError('shared_locks_directory_changed')
                        checked=open_root(self.path)
                        try:
                            if (os.fstat(checked).st_dev,os.fstat(checked).st_ino)!=(os.fstat(fd).st_dev,os.fstat(fd).st_ino):
                                raise ValueError('shared_catalog_root_changed')
                        finally: os.close(checked)
                        if (file_identity(os.fstat(lock)) != identity or
                                file_identity(os.stat(key, dir_fd=locks, follow_symlinks=False)) != identity):
                            raise ValueError('shared_lock_changed')
                    stable()
                    yield Store(self, fd, key, stable)
                    stable()
            finally:
                if lock is not None: os.close(lock)
                os.close(locks)


class Store:
    def __init__(self, catalog, fd, key, stable):
        self.catalog, self.fd, self.key, self.stable = catalog, fd, key, stable

    @contextmanager
    def directory(self, name):
        self.stable()
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            policy(os.fstat(fd), True, CATALOG_OWNER, 0o700)
            before = os.fstat(fd)
            yield fd
            if os.stat(name, dir_fd=self.fd, follow_symlinks=False).st_ino != before.st_ino:
                raise ValueError('shared_catalog_directory_changed')
            self.stable()
        finally: os.close(fd)

    def read(self, area, name):
        with self.directory(area) as fd: return read_at(fd, name)

    def append(self, attempt, event, value):
        token(attempt, ATTEMPT)
        if event not in ('intent','candidate','writer-intent','writer-created','writer-terminal','verifier-intent','verifier-created','verifier-terminal'):
            raise ValueError('shared_event')
        with self.directory('attempts') as fd:
            write_at(fd, self.key+'-'+attempt+'-'+event+'.json', value)

    def event(self, attempt, event):
        token(attempt, ATTEMPT)
        return self.read('attempts', self.key+'-'+attempt+'-'+event+'.json')

    def attempts(self):
        with self.directory('attempts') as fd:
            names = names_at(fd,MAX_ENTRIES,self.catalog.end)
            if len(names) > MAX_ENTRIES: raise ValueError('shared_attempt_budget')
            result = []
            for name in names:
                if not re.fullmatch(r'[0-9a-f]{64}-[0-9a-f]{32}-(intent|candidate|writer-intent|writer-created|writer-terminal|verifier-intent|verifier-created|verifier-terminal)\.json',name):
                    raise ValueError('shared_attempt_grammar')
                if name[:97]+'-intent.json' not in names: raise ValueError('shared_orphan_attempt_record')
                if name.startswith(self.key+'-') and name.endswith('-intent.json') and not name.endswith(('-writer-intent.json','-verifier-intent.json')):
                    attempt = name[len(self.key)+1:-len('-intent.json')]
                    token(attempt, ATTEMPT); result.append(attempt)
            return sorted(result)

    def publication(self):
        try: result = self.read('published', self.key+'.json')
        except FileNotFoundError: return None
        publication(result, self.catalog, self.key)
        return result

    def publish(self, value, fault=lambda _: None):
        publication(value, self.catalog, self.key)
        if self.publication() is not None: raise ValueError('shared_publication_exists')
        with self.directory('published') as fd:
            temp = self.key+'-'+value['attempt']+'-'+secrets.token_hex(16)+'.tmp'
            write_at(fd, temp, value); fault('publication_temp_fsync')
            self.stable()
            # The held permanent key lock excludes all conforming writers. Refuse
            # an existing final path even if malformed; never rename over it.
            try: os.stat(self.key+'.json', dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError: pass
            else: raise ValueError('shared_publication_conflict')
            os.rename(temp, self.key+'.json', src_dir_fd=fd, dst_dir_fd=fd)
            fault('publication_rename')
            os.fsync(fd); fault('publication_parent_fsync')
        if self.publication() != value: raise ValueError('shared_publication_changed')
        return value


def publication(v, catalog, key):
    closed(v, 'schema daemon catalog key attempt candidate observation writer verifier')
    if (v['schema'] != SCHEMA or v['daemon'] != catalog.daemon or v['catalog'] != catalog.metadata or v['key'] != key):
        raise ValueError('shared_publication_identity')
    token(v['attempt'], ATTEMPT); volume(v['candidate'], key, v['attempt']); disjoint(catalog.metadata,v['candidate'])
    closed(v['observation'], 'root tree_sha256')
    token(v['observation']['tree_sha256'])
    if type(v['observation']['root']) is not list or len(v['observation']['root']) != 2 or any(type(n) is not int or n < 0 for n in v['observation']['root']):
        raise ValueError('shared_root_identity')
    # Terminal evidence is re-inspected before every reuse by the controller.
    for role in ('writer','verifier'):
        closed(v[role], 'id nonce image inspection_sha256 started finished')
        for field in ('id','inspection_sha256'): token(v[role][field])
        token(v[role]['nonce'], ATTEMPT)
        if not dv.IMAGE.fullmatch(v[role]['image']): raise ValueError('shared_image')
        for field in ('started','finished'):
            if type(v[role][field]) is not str or not 1 <= len(v[role][field]) <= 64:
                raise ValueError('shared_terminal_time')
    return v


def verify_tree(path, expected, end):
    """Exact directory closure plus existing installed verifier; every byte read.

    Descriptor validation and no-follow streaming below precede _model_tree;
    its checked minimal Qwen view retains the installed pin/capability checks.
    """
    key, raw, files, dirs = descriptor(expected)
    seen_files = set(); seen_dirs = set(); identities = []; records = []
    with root_fd(path) as root:
        root_stat = os.fstat(root)
        def walk(fd, relative):
            remaining(end)
            s = os.fstat(fd); policy(s, True, OWNER, 0o555)
            seen_dirs.add(relative); identities.append((relative, file_identity(s)))
            if relative not in dirs: raise ValueError('shared_extra_directory')
            names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    remaining(end)
                    if len(names)+len(seen_files)+len(seen_dirs) >= MAX_ENTRIES:
                        raise ValueError('shared_inventory_budget')
                    names.append(entry.name)
            for name in sorted(names):
                remaining(end)
                rel = relative+'/'+name if relative else name
                before = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try: walk(child, rel)
                    finally: os.close(child)
                else:
                    if rel not in files: raise ValueError('shared_extra_file')
                    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                    try:
                        s = os.fstat(child); policy(s, False, OWNER, 0o444)
                        if file_identity(s) != file_identity(before) or s.st_size != files[rel][0]:
                            raise ValueError('shared_file_identity')
                        hashed = hashlib.sha256(); count = 0; manifest = bytearray()
                        while True:
                            remaining(end)
                            block = os.read(child, min(CHUNK, s.st_size+1-count))
                            if not block: break
                            count += len(block); hashed.update(block)
                            if rel.endswith('/manifest.json'): manifest.extend(block)
                            if count > s.st_size: raise ValueError('shared_file_extended')
                        if (count != s.st_size or hashed.hexdigest() != files[rel][1] or
                                file_identity(os.fstat(child)) != file_identity(s)):
                            raise ValueError('shared_file_hash')
                        if rel.endswith('/manifest.json') and bytes(manifest) != raw:
                            raise ValueError('shared_manifest_bytes')
                        identities.append((rel, file_identity(s))); seen_files.add(rel)
                        records.append([rel, count, hashed.hexdigest(), list(file_identity(s))])
                    finally: os.close(child)
                if file_identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) != file_identity(before):
                    raise ValueError('shared_tree_changed')
        walk(root, '')
        if seen_dirs != dirs or seen_files != set(files): raise ValueError('shared_incomplete_tree')
        # Existing installed hashing/pin policy is deliberately not duplicated or
        # weakened. This view contains no runtime profile, enrollment or authority.
        from service.host_install import _model_tree
        from service.install_policy import QWEN_POLICY
        view = dict(ownership_policy=QWEN_POLICY, bundles={key: dict(manifest_sha256=key,
                    checkpoint=expected['checkpoint'], revision=expected['revision'])})
        tree = _model_tree(path, view, end)
        for rel, identity in identities:
            current = root
            opened = []
            try:
                parts = rel.split('/') if rel else []
                for part in parts[:-1]:
                    current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current); opened.append(current)
                s = os.stat(parts[-1], dir_fd=current, follow_symlinks=False) if parts else os.fstat(root)
                if file_identity(s) != identity: raise ValueError('shared_tree_changed')
            finally:
                for fd in reversed(opened): os.close(fd)
        remaining(end)
        return dict(root=[root_stat.st_dev, root_stat.st_ino], tree_sha256=tree)


STORAGE = 'shared-model-bundle-v1'


def storage(value, binding):
    """Detached storage identity; catalog records remain authority at attach (C)."""
    closed(value, 'kind manifest_sha256 bundle_id catalog publication_sha256 reference_id')
    key = token(value['manifest_sha256'])
    from service import installed_qwen_daemon as qd
    if (value['kind'] != STORAGE or key != qd.MANIFEST_SHA256 or
            value['bundle_id'] != key or value['reference_id'] != binding['installation']):
        raise ValueError('shared_storage_binding')
    token(value['publication_sha256']); token(value['reference_id'], ATTEMPT)
    volume(value['catalog'])
    candidate = binding['volumes']['models']
    if type(candidate) is not dict or type(candidate.get('Labels')) is not dict:
        raise ValueError('shared_volume')
    attempt = candidate['Labels'].get('speech-box.attempt')
    volume(candidate, key, attempt)
    disjoint(value['catalog'], candidate)
    return value


class Installations:
    """Permanent attachment authority, usable only inside Catalog.held(K)."""
    def __init__(self, store):
        self.store = store
        self._bindings={}; self._readers=None
        self.published = store.publication()
        if self.published is None: raise ValueError('shared_not_published')

    def name(self, installation, event):
        token(installation, ATTEMPT)
        return 'install-'+self.store.key+'-'+installation+'-'+event+'.json'

    def read(self, installation, event):
        return self.store.read('references', self.name(installation,event))

    def append(self, installation, event, value):
        with self.store.directory('references') as fd:
            write_at(fd,self.name(installation,event),value)

    def reservation(self, installation, config):
        from service import install_policy as ip, installed_qwen_daemon as qd
        config=ip.family_configuration(config,qd.FAMILY)
        return dict(schema=STORAGE,installation=token(installation,ATTEMPT),key=self.store.key,
                    daemon=self.store.catalog.daemon,catalog=self.store.catalog.metadata,
                    publication_sha256=digest(self.published),configuration_digest=digest(config))

    def reserve(self, installation, config):
        value=self.reservation(installation,config)
        self.append(installation,'reserved',value)
        return self.published

    def bound(self, installation):
        from service import installed_qwen_daemon as qd, install_policy as ip
        value=self.read(installation,'bound'); closed(value,'binding configuration')
        reserved=self.read(installation,'reserved')
        cache_key=digest([value,reserved])
        if cache_key in self._bindings: return self._bindings[cache_key]
        config=ip.family_configuration(value['configuration'],qd.FAMILY)
        b=qd.binding_configuration(value['binding'],config)
        # Canonical protected reads above detached these records from callers.
        s=b['model_storage']
        if (b['installation']!=installation or b['daemon']!=self.store.catalog.daemon or
                b['volumes']['models']!=self.published['candidate'] or
                s['catalog']!=self.store.catalog.metadata or s['publication_sha256']!=digest(self.published) or
                s['manifest_sha256']!=self.store.key or
                reserved!=self.reservation(installation,config)):
            raise ValueError('shared_reference_binding')
        self._bindings[cache_key]=(b,config)
        return b,config

    def bind(self,b,config):
        installation=b['installation']
        self.append(installation,'bound',dict(binding=b,configuration=config))
        self.bound(installation)
        return dict(binding=digest(b))

    def attachment(self,b,config,role,phase,nonce):
        from service import installed_qwen_daemon as qd
        if self.bound(b['installation'])!=(b,config): raise ValueError('shared_attachment_binding')
        token(nonce,ATTEMPT)
        command=dv.command(role,config,b,phase=phase,nonce=nonce if role=='helper' else None,family=qd.FAMILY)
        value=dict(binding=digest(b),role=role,phase=phase,nonce=nonce,command_sha256=digest(command))
        self.append(b['installation'],'attach-'+nonce+'-intent',value)
        return value

    def created(self,b,nonce,c):
        from service import installed_qwen_daemon as qd
        b,config=self.bound(b['installation'])
        intent=self.read(b['installation'],'attach-'+token(nonce,ATTEMPT)+'-intent')
        dv.inspect_container(c,intent['role'],config,b,phase=intent['phase'],
            nonce=nonce if intent['role']=='helper' else None,family=qd.FAMILY)
        if c['State']['Running'] or c['State']['Status']!='created': raise ValueError('shared_fresh_attachment')
        self.append(b['installation'],'attach-'+nonce+'-created',dict(id=token(c['Id']),intent_sha256=digest(intent)))
        return dict(id=c['Id'])

    def enrolled(self,b,public):
        from service import installed_qwen_daemon as qd
        qd.installation(public)
        if public['launch']!=b: raise ValueError('shared_enrollment_binding')
        self.bound(b['installation'])
        self.append(b['installation'],'enrolled',dict(binding=digest(b),enrollment_digest=digest(public)))

    def readers(self):
        """Join every attachment to protected reservation and detached binding."""
        from service import installed_qwen_daemon as qd
        with self.store.directory('references') as fd:
            names=names_at(fd,MAX_ENTRIES,self.store.catalog.end)
            selected=[n for n in names if n.startswith('install-'+self.store.key+'-')]
            records={n:read_at(fd,n) for n in selected}
        fingerprint=digest(records)
        if self._readers is not None and self._readers[0]==fingerprint: return self._readers[1]
        result={}; bindings={}
        prefix='install-'+self.store.key+'-'
        for name,value in records.items():
            suffix=name[len(prefix):]
            match=re.fullmatch(r'([0-9a-f]{32})-(reserved|bound|enrolled|attach-[0-9a-f]{32}-(intent|created))\.json',suffix)
            if match is None: raise ValueError('shared_installation_record_grammar')
            installation,event=match[1],match[2]
            if event=='reserved':
                closed(value,'schema installation key daemon catalog publication_sha256 configuration_digest')
                if value != dict(schema=STORAGE,installation=installation,key=self.store.key,daemon=self.store.catalog.daemon,
                        catalog=self.store.catalog.metadata,publication_sha256=digest(self.published),configuration_digest=value['configuration_digest']):
                    raise ValueError('shared_reservation_identity')
                token(value['configuration_digest'])
                continue
            if installation not in bindings: bindings[installation]=self.bound(installation)
            b,config=bindings[installation]
            if event=='bound': continue
            if event=='enrolled':
                closed(value,'binding enrollment_digest')
                if value['binding']!=digest(b): raise ValueError('shared_enrolled_binding')
                token(value['enrollment_digest']); continue
            # Each pair is checked once; the created record cannot stand alone.
            if event.endswith('-created') and self.name(installation,event[:-7]+'intent') in records: continue
            nonce=event.split('-')[1]
            intent=records.get(self.name(installation,'attach-'+nonce+'-intent'))
            created=records.get(self.name(installation,'attach-'+nonce+'-created'))
            if intent is None or created is None: raise ValueError('shared_attachment_outcome_unknown')
            closed(intent,'binding role phase nonce command_sha256'); closed(created,'id intent_sha256')
            role,phase=intent['role'],intent['phase']
            if role not in ('helper','control','worker') or (phase not in dv.PHASES if role=='helper' else phase is not None):
                raise ValueError('shared_attachment_role')
            if role!='helper' or phase not in ('install','init'):
                enrolled=records.get(self.name(installation,'enrolled'))
                if enrolled is None: raise ValueError('shared_enrollment_required')
                closed(enrolled,'binding enrollment_digest')
                if enrolled['binding']!=digest(b): raise ValueError('shared_enrolled_binding')
                token(enrolled['enrollment_digest'])
            command=dv.command(role,config,b,phase=phase,nonce=nonce if role=='helper' else None,family=qd.FAMILY)
            if (intent!=dict(binding=digest(b),role=role,phase=phase,nonce=nonce,command_sha256=digest(command)) or
                    created['intent_sha256']!=digest(intent)):
                raise ValueError('shared_attachment_record')
            identity=token(created['id'])
            item=(b,config,intent)
            if identity in result and result[identity]!=item: raise ValueError('shared_duplicate_attachment')
            result[identity]=item
        self._readers=(fingerprint,result)
        return result

    def classify(self,c,readers):
        from service import installed_qwen_daemon as qd
        item=readers.get(c['Id'])
        if item is None: raise ValueError('shared_unknown_consumer')
        b,config,intent=item
        dv.inspect_container(c,intent['role'],config,b,phase=intent['phase'],
            nonce=intent['nonce'] if intent['role']=='helper' else None,family=qd.FAMILY)
        return b


def reference_name(name):
    """Closed shared reference namespace, including retained incomplete holds."""
    return bool(re.fullmatch(r'[0-9a-f]{64}-[0-9a-f]{32}-(intent|created|done)\.json',name) or
        re.fullmatch(r'install-[0-9a-f]{64}-[0-9a-f]{32}-(reserved|bound|enrolled|attach-[0-9a-f]{32}-(intent|created))\.json',name))


def source_overlap(source, roots):
    """Discovery recognizes Linux spelling aliases; admission stays literal."""
    import posixpath
    if not isinstance(source,str) or not source.startswith('/'): return False
    source='/'+posixpath.normpath(source).lstrip('/')
    return any(source==root or source.startswith(root+'/') or root.startswith(source.rstrip('/')+'/') for root in roots)
