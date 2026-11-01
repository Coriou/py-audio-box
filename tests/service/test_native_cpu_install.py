"""Factory pre-acquisition custody contracts, no platform/model operations."""
from datetime import datetime,timezone
import os
from pathlib import Path
import tempfile
import shutil
import sys
import hashlib
import zipfile
import unittest
from unittest.mock import patch

from service import native_cpu_install as install
from service.native_cpu_configuration import expected
from worker.transport import Refused


class CPUInstallTests(unittest.TestCase):
    def test_pending_record_checks_real_archive_interpreter_tls_and_lock_identity(self):
        from service import native_cpu_package as package
        from worker.transport import canonical
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();root.chmod(0o700)
            archive=root/'source.zip';package.archive(archive)
            with zipfile.ZipFile(archive) as source:source.extractall(root/'source')
            installed=root/'source/native-worker'
            python=root/'venv/bin/python';python.parent.mkdir(parents=True,mode=0o700)
            shutil.copyfile(sys.executable,python);python.chmod(0o700)
            install.create_tls(root)
            credential=dict(version=1,credentials=[dict(token='inert-source-fixture-not-a-real-secret',principal='local',scopes=['read','production-write'])])
            install.write(root/'credential.json',credential)
            pins=install.locked_versions(installed)
            record=dict(version=1,variant=package.VARIANT,root=str(root),source=str(installed),
                source_files={name:install.files.file_hash(str(installed/name)) for name in [*package.members(),'native-worker.py']},
                python=str(python),python_sha256=install.files.file_hash(str(python),executable=True),python_version=sys.version,
                dependencies=pins,deployment_id='a'*32,principal='local',installed_at=datetime.now(timezone.utc).isoformat(),port=8788,
                tls_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tls').iterdir()},
                credential_sha256=hashlib.sha256((root/'credential.json').read_bytes()).hexdigest(),qualification_session_id='b'*32,
                source_provenance=dict(commit=None,dirty=None))
            install.write(root/'setup-record.json',record)
            with patch('lib.tts_backends.kokoro_native.validate_host'),patch('lib.tts_backends.kokoro_native.validate_versions'),patch.object(install.metadata,'version',side_effect=lambda name:pins[name]):
                _,observed,_,bundle=install.read_installation(root,_pending=True)
                self.assertEqual(observed,record);self.assertFalse(bundle.exists())
                changed=installed/'worker/native_cpu_child.py';raw=changed.read_bytes();changed.write_bytes(b'inert-changed-source')
                with self.assertRaisesRegex(Refused,'source_changed'):install.read_installation(root,_pending=True)
                changed.write_bytes(raw)
                record['python_version']='inert changed interpreter'
                (root/'setup-record.json').write_bytes(canonical(record))
                with self.assertRaisesRegex(Refused,'interpreter_changed'):install.read_installation(root,_pending=True)

    def test_custodian_unknown_close_reports_reap_separately_and_keeps_lock(self):
        import contextlib,io,json
        from types import SimpleNamespace
        from unittest.mock import Mock
        owner=install.Prepared.__new__(install.Prepared);owner.closed=False;owner.errors=[RuntimeError('/private/secret-config')]
        lock=Mock();owner.supervisor=SimpleNamespace(shutdown_event=Mock(),active=None,journal=SimpleNamespace(lock=lock,
            value={'records':[dict(phase='terminal',acked=False,receipt={'reaped':True})]}))
        owner.thread=Mock();owner.thread.is_alive.return_value=True
        output=io.StringIO()
        with contextlib.redirect_stderr(output),self.assertRaisesRegex(Refused,'shutdown_unknown'):owner.close()
        self.assertFalse(owner.closed);owner.thread.join.assert_called_once_with(5);lock.close.assert_not_called()
        self.assertNotIn('secret-config',output.getvalue())
        events=[json.loads(line.split(':',1)[1]) for line in output.getvalue().splitlines()]
        self.assertEqual([e['stage'] for e in events],['stop_requested','join_observed'])
        self.assertTrue(events[-1]['thread_alive']);self.assertEqual(events[-1]['reaped'],1);self.assertEqual(events[-1]['acked'],0)

    def test_unknown_startup_close_retains_strong_owner_and_actual_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();root.chmod(0o700)
            for role in ('control','worker','spool','staging'):
                path=root/'state'/role;path.mkdir(mode=0o700,parents=True)
                path.chmod(0o770 if role=='staging' else 0o700)
            config=dict(source_files={},python_sha256='e'*64,python_version='inert',dependencies={},
                deployment_id='a'*32,installed_at=datetime.now(timezone.utc).isoformat(),principal='local')
            before=set(install._PREPARED_OWNERS)
            with (patch.object(install,'read_installation',return_value=(root,config,expected(),root/'unused-model')),
                    patch.object(install.Prepared,'_prepare',side_effect=ValueError('inert startup failure')),
                    patch.object(install.Prepared,'close',side_effect=Refused('inert cleanup unknown'))):
                with self.assertRaisesRegex(Refused,'cleanup unknown'):install.Prepared(root)
            owned=set(install._PREPARED_OWNERS)-before
            self.assertEqual(len(owned),1);identity=owned.pop();owner=install._PREPARED_OWNERS[identity]
            try:
                self.assertFalse(owner.closed);self.assertEqual(os.fstat(owner.lock).st_uid,os.getuid())
            finally:
                # Test-owned inert failure only; product never automatically retries.
                os.close(owner.lock);install._PREPARED_OWNERS.pop(identity)


if __name__=='__main__':unittest.main()
