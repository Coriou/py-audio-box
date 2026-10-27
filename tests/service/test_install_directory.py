import os

import pytest

from service import install_policy as ip


def test_held_directory_observes_new_entries_from_fresh_offsets(tmp_path, monkeypatch):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original = os.listdir
    scans = []
    def listed(scan):
        assert scan != fd
        assert os.fstat(scan).st_ino == os.fstat(fd).st_ino
        scans.append(scan)
        return original(scan)
    monkeypatch.setattr(os, 'listdir', listed)
    try:
        assert ip.directory_entries(fd) == []
        (tmp_path/'intent').write_bytes(b'owned')
        assert ip.directory_entries(fd) == ['intent']
        (tmp_path/'intent').unlink()
        assert ip.directory_entries(fd) == []
        for scan in scans:
            with pytest.raises(OSError): os.fstat(scan)
        os.fstat(fd)
    finally:
        os.close(fd)


def test_failed_directory_scan_closes_only_its_own_descriptor(tmp_path, monkeypatch):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    scans = []
    def failed(scan):
        scans.append(scan)
        raise OSError('injected scan failure')
    monkeypatch.setattr(os, 'listdir', failed)
    try:
        with pytest.raises(OSError, match='injected scan failure'):
            ip.directory_entries(fd)
        with pytest.raises(OSError): os.fstat(scans[0])
        os.fstat(fd)
    finally:
        os.close(fd)
