"""Real inert payloads/catalog descriptors; ownership alone is host-adapted."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace
from unittest.mock import patch

import tempfile
import unittest
from service import shared_models as sm, shared_model_helper as sh
from service import installed_qwen_daemon as qd

ROOT=Path(__file__).resolve().parents[2]
IMAGE='sha256:'+'1'*64
DAEMON=dict(endpoint='unix:///var/run/docker.sock',id='inert-daemon',os='linux',architecture='amd64',security=[],cgroup='2')


def fixture():
    expected=json.loads((ROOT/'requirements/tts/qwen-0.6b-base.expected.json').read_text())
    data={e['path']:('tiny inert '+e['path']).encode() for e in expected['files']}
    for e in expected['files']:
        e.update(bytes=len(data[e['path']]),sha256=hashlib.sha256(data[e['path']]).hexdigest())
    return expected,data


def metadata(path,key=None,attempt=None):
    return dict(Name=sm.CATALOG if key is None else 'pab-model-sha256-'+key+'-'+attempt,
                Driver='local',Options=None,Scope='local',Mountpoint=str(path),CreatedAt='2026-10-27T00:00:00Z',Labels=sm.labels(key,attempt))


@contextmanager
def ownership():
    # Mac source evidence: preserve every real inode/mode/link/hash operation,
    # adapt only Linux UID/GID observations and privileged chown.
    real_stat,real_fstat=os.stat,os.fstat
    def projected(s):
        if stat.S_IMODE(s.st_mode) not in (0o444,0o555): return s
        return SimpleNamespace(**{name:(10001 if name in ('st_uid','st_gid') else getattr(s,name))
            for name in dir(s) if name.startswith('st_')})
    with patch.object(sm,'CATALOG_OWNER',(os.getuid(),os.getgid())), \
         patch.object(os,'fchown'), \
         patch.object(os,'stat',side_effect=lambda *a,**kw:projected(real_stat(*a,**kw))), \
         patch.object(os,'fstat',side_effect=lambda *a,**kw:projected(real_fstat(*a,**kw))):
        yield


@contextmanager
def new_bundle():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path=Path(directory).resolve()
        expected,data=fixture();key=sm.descriptor(expected)[0]
        try:
            with ownership(), patch.object(qd,'MANIFEST_SHA256',key):
                yield expected,data,key,tmp_path
        finally:
            for root,dirs,files in os.walk(tmp_path,followlinks=False):
                os.chmod(root,0o755)
                for name in files:
                    path=Path(root)/name
                    if not path.is_symlink():path.chmod(0o644)


class SharedModelCase(unittest.TestCase):
    def setUp(self):
        self.bundle=self.enterContext(new_bundle())



def imported(bundle,path=None):
    expected,data,key,tmp=bundle
    path=path or tmp/'payload';path.mkdir()
    calls=[]
    def chunk(name,offset):
        calls.append((name,offset)); return data[name.split('/',2)[2]][offset:offset+sm.CHUNK]
    sh.import_tree(path,expected,chunk,time.monotonic()+10)
    return path,calls


class SharedModelTests(SharedModelCase):
    def test_direct_volume_original_path_aliases(self):
        key = qd.MANIFEST_SHA256
        for candidate in (False, True):
            args = (key, 'b'*32) if candidate else ()
            good = metadata('/opaque-store/payload' if candidate else '/opaque-store/catalog', *args)
            self.assertIs(sm.volume(good, *args), good)
            for path in ('//opaque-store/payload', '///opaque-store/payload',
                         '/opaque-store/payload/', '/opaque-store/./payload',
                         '/opaque-store//payload', '/opaque-store/../payload',
                         'opaque-store/payload', '/', '/opaque-store/invalid\n'):
                bad = dict(good, Mountpoint=path)
                before = copy.deepcopy(bad)
                with self.subTest(candidate=candidate, path=path):
                    with self.assertRaisesRegex(ValueError, '^shared_volume$'):
                        sm.volume(bad, *args)
                    self.assertEqual(bad, before)

    def test_real_exact_tree_and_late_bytes(self):
        bundle=self.bundle
        expected,data,key,tmp=bundle
        path,calls=imported(bundle)
        first=sm.verify_tree(path,expected,time.monotonic()+10)
        assert len(calls)==11
        assert first==sm.verify_tree(path,expected,time.monotonic()+10)
        for name in data:
            file=path/'bundles'/key/name
            original=file.read_bytes()
            for index in (0,len(original)-1):
                file.chmod(0o644); bad=bytearray(original);bad[index]^=1;file.write_bytes(bad);file.chmod(0o444)
                with self.assertRaises(ValueError): sm.verify_tree(path,expected,time.monotonic()+10)
                file.chmod(0o644);file.write_bytes(original);file.chmod(0o444)
            for bad in (original[:-1],original+b'x'):
                file.chmod(0o644);file.write_bytes(bad);file.chmod(0o444)
                with self.assertRaises(ValueError): sm.verify_tree(path,expected,time.monotonic()+10)
            file.chmod(0o644);file.write_bytes(original);file.chmod(0o444)


    def test_inventory_modes_links_and_manifest_bytes(self):
        bundle=self.bundle
        expected,data,key,tmp=bundle
        path,_=imported(bundle)
        directory=path/'bundles'/key; directory.chmod(0o755)
        extra=directory/'extra';extra.mkdir();extra.chmod(0o555);directory.chmod(0o555)
        with self.assertRaisesRegex(ValueError,'extra_directory'): sm.verify_tree(path,expected,time.monotonic()+10)
        # Fixtures retain distinct adversarial trees; no production repair/deletion.
        path2,_=imported(bundle,tmp/'second')
        leaf=path2/'bundles'/key/next(iter(data)); leaf.chmod(0o644)
        with self.assertRaises(ValueError): sm.verify_tree(path2,expected,time.monotonic()+10)
        path3,_=imported(bundle,tmp/'third');d=path3/'bundles'/key;d.chmod(0o755)
        os.link(d/next(iter(data)),d/'alias')
        with self.assertRaises(ValueError): sm.verify_tree(path3,expected,time.monotonic()+10)
        path4,_=imported(bundle,tmp/'fourth');d=path4/'bundles'/key;d.chmod(0o755)
        (d/'extra').symlink_to(d/next(iter(data)));d.chmod(0o555)
        with self.assertRaises(ValueError): sm.verify_tree(path4,expected,time.monotonic()+10)
        path5,_=imported(bundle,tmp/'fifth');m=path5/'bundles'/key/'manifest.json';m.chmod(0o644)
        m.write_bytes(m.read_bytes().rstrip(b'\n'));m.chmod(0o444)
        with self.assertRaises(ValueError): sm.verify_tree(path5,expected,time.monotonic()+10)


    def test_descriptor_conflicts_and_noncanonical(self):
        bundle=self.bundle
        expected,_,_,_=bundle
        for name in ('../evil','a//b.json','/evil','config.json/sub.json','manifest.json'):
            value=copy.deepcopy(expected);value['files'][-1]['path']=name
            with self.assertRaises(ValueError): sm.descriptor(value)
        value=copy.deepcopy(expected);value['files'][-1]['path']=value['files'][0]['path']
        with self.assertRaises(ValueError): sm.descriptor(value)
        with self.assertRaises(ValueError): sm.deadline(float('inf'))
        with self.assertRaises(ValueError): sm.deadline(time.time()+601)


    def test_catalog_bootstrap_partial_and_identity(self):
        bundle=self.bundle
        _,_,key,tmp=bundle
        root=tmp/'catalog';root.mkdir();meta=metadata(root)
        catalog=sm.Catalog(root,DAEMON,meta,time.monotonic()+10)
        assert catalog.init()==catalog.init()
        with catalog.held(key) as store:
            assert store.publication() is None
            inode=(root/'locks'/key).stat().st_ino
        with catalog.held(key): assert (root/'locks'/key).stat().st_ino==inode
        bad=copy.deepcopy(DAEMON);bad['id']='other'
        with self.assertRaises(ValueError): sm.Catalog(root,bad,meta,time.monotonic()+10).init()
        partial=tmp/'partial';partial.mkdir();(partial/'retained').write_bytes(b'proof')
        with self.assertRaisesRegex(ValueError,'partial_bootstrap'): sm.Catalog(partial,DAEMON,metadata(partial),time.monotonic()+10).init()
        assert (partial/'retained').read_bytes()==b'proof'


    def test_timeout_and_source_chunk_refusal(self):
        bundle=self.bundle
        expected,_,_,tmp=bundle
        root=tmp/'partial';root.mkdir()
        with self.assertRaises(ValueError): sh.import_tree(root,expected,lambda *_:b'',time.monotonic()+10)
        with self.assertRaises(ValueError): sm.verify_tree(root,expected,time.monotonic()+10)
        another=tmp/'timeout';another.mkdir()
        with self.assertRaisesRegex(ValueError,'deadline'): sh.import_tree(another,expected,lambda *_:b'x',time.monotonic()-1)


    def test_nofollow_catalog_and_root(self):
        bundle=self.bundle
        _,_,key,tmp=bundle
        root=tmp/'catalog';root.mkdir();catalog=sm.Catalog(root,DAEMON,metadata(root),time.monotonic()+10);catalog.init()
        (root/'locks'/key).symlink_to(tmp/'outside')
        with self.assertRaises(OSError):
            with catalog.held(key): pass
        link=tmp/'link';link.symlink_to(root,target_is_directory=True)
        with self.assertRaises(OSError):
            with sm.root_fd(link): pass

    def test_special_files_ancestor_and_live_replacement(self):
        expected,data,key,tmp=self.bundle
        import socket
        for variant in ('fifo','socket','ancestor','replacement','wrong-uid','hardlink','emptydir'):
            with self.subTest(variant=variant),new_bundle() as bundle:
                expected,data,key,tmp=bundle
                path,_=imported(bundle)
                directory=path/'bundles'/key;directory.chmod(0o755)
                leaf=directory/'config.json'
                if variant=='fifo':os.mkfifo(directory/'fifo')
                elif variant=='socket':
                    sock=socket.socket(socket.AF_UNIX);self.addCleanup(sock.close);sock.bind(str(tmp/'sock'));os.rename(tmp/'sock',directory/'sock')
                elif variant=='ancestor':
                    tokenizer=directory/'speech_tokenizer';tokenizer.chmod(0o755);tokenizer.rename(directory/'moved');(directory/'moved').chmod(0o555);tokenizer.symlink_to(directory/'moved',target_is_directory=True)
                elif variant=='hardlink':os.link(leaf,tmp/'external-link')
                elif variant=='emptydir':(directory/'extra').mkdir()
                directory.chmod(0o555)
                if variant=='replacement':
                    original_read=os.read;changed=False
                    def replace(fd,size):
                        nonlocal changed
                        raw=original_read(fd,size)
                        if not changed and raw and raw.startswith(b'tiny inert config'):
                            changed=True;directory.chmod(0o755)
                            leaf.rename(directory/'retained-old');leaf.write_bytes(data['config.json']);leaf.chmod(0o444);directory.chmod(0o555)
                        return raw
                    with patch.object(os,'read',side_effect=replace),self.assertRaises(ValueError):
                        sm.verify_tree(path,expected,time.monotonic()+10)
                    assert changed
                elif variant=='wrong-uid':
                    with patch.object(sm,'OWNER',(999,999)),self.assertRaises(ValueError):sm.verify_tree(path,expected,time.monotonic()+10)
                else:
                    with self.assertRaises((ValueError,OSError)):sm.verify_tree(path,expected,time.monotonic()+10)

    def test_lock_inode_replacement_and_no_publication_overwrite(self):
        _,_,key,tmp=self.bundle
        root=tmp/'catalog';root.mkdir();catalog=sm.Catalog(root,DAEMON,metadata(root),time.monotonic()+10);catalog.init()
        with self.assertRaisesRegex(ValueError,'lock_changed'):
            with catalog.held(key) as store:
                lock=root/'locks'/key
                lock.rename(root/'locks'/'retained-lock')
                lock.write_bytes(b'');lock.chmod(0o600)
                store.stable()
