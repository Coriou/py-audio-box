"""Injected process bytes only. Never executes FFmpeg or an application child."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from worker.audio_export import execution, execute, pcm, registration, settings, verify_output, provenance, verify_converter


def wav(rate=48000,channels=2,width=3,frames=2400):
    out=io.BytesIO()
    with wave.open(out,'wb') as f:
        f.setparams((channels,width,rate,0,'NONE','not compressed')); f.writeframes(b'\0'*(frames*channels*width))
    return out.getvalue()


def fixture():
    return json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())


class AudioExportTests(unittest.TestCase):
    def test_complete_dynamic_closure_fits_but_registration_stays_bounded(self):
        value = fixture()['context']['execution']['converter']
        value['runtime_closure'] = [
            '/usr/lib/x86_64-linux-gnu/libdependency%03d.so.1.2.3' % i
            for i in range(214)]
        self.assertGreater(len(json.dumps(value)), 4096)
        self.assertEqual(registration(value), value)
        value['runtime_closure'] = [path + 'x'*80 for path in value['runtime_closure']]
        with self.assertRaisesRegex(ValueError, 'resource_plan_limit'):
            registration(value)

    def test_closed_settings(self):
        request=fixture()['context']['plan']['request']
        for change in ({'rate':True},{'channels':1.0},{'loudness':float('nan')},{'loudness':float('inf')},{'loudness':-71},{'loudness':-4},{'path':'/tmp/a'},{'rate':22050}):
            with self.subTest(change=change),self.assertRaises(Exception): settings(dict(request,**change))
        for name in request:
            value=request.copy();del value[name]
            with self.assertRaises(Exception): settings(value)

    def test_pcm_rejects_malformed_and_header_mismatch(self):
        raw=wav()
        self.assertEqual(pcm(raw)['media_subtype'],'PCM_24')
        for data in (raw[:-1],raw+b'x',raw[:20]+b'\3\0'+raw[22:],wav(width=2),wav(channels=1),wav(rate=24000),wav(frames=0)):
            with self.assertRaises(ValueError): pcm(data,rate=48000,channels=2,bits=24)
        # Valid FFmpeg PCM extensible form.
        fmt=struct.pack('<HHIIHHHHI',65534,2,48000,288000,6,24,22,24,3)+bytes.fromhex('0100000000001000800000aa00389b71')
        body=b'WAVEfmt '+struct.pack('<I',40)+fmt+b'data'+struct.pack('<I',12)+b'\0'*12
        self.assertEqual(pcm(b'RIFF'+struct.pack('<I',len(body))+body)['frames'],2)

    def test_closed_execution_and_limits(self):
        context=fixture()['context'];execution(context)
        for mutation in ('voice','hash','source','size','identity'):
            c=copy.deepcopy(context)
            if mutation=='voice': c['plan']['voice_versions']=[{}]
            elif mutation=='hash': c['plan']['resolved']['settings_sha256']='1'*64
            elif mutation=='source': c['plan']['request']['asset_id']='other'
            elif mutation=='identity': c['execution']['converter']['executable']='/tmp/other'
            else: c['plan']['assets'][0]['frames']=10**9
            with self.assertRaises(Exception): execution(c)

    def test_executable_logic_with_fake_process_and_sealed_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();exe=root/'converter';exe.write_bytes(b'INERT executable fixture');exe.chmod(0o555)
            lib=root/'runtime.so';lib.write_bytes(b'INERT runtime closure');lib.chmod(0o444)
            c=fixture()['context'];reg=c['execution']['converter']
            reg.update(executable=str(exe),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runtime_closure=[str(lib)])
            c['plan']['resolved']['identity']['converter']=copy.deepcopy(reg)
            source=Path('contracts/fixtures/v1/short.wav');fd=os.open(source,os.O_RDONLY)
            self.addCleanup(os.close,fd)
            stage=root/'stage';stage.mkdir(mode=0o770);stage.chmod(0o770)
            calls=[]
            def fake(argv,**kw):
                calls.append((argv,kw));os.write(kw['pass_fds'][2],wav())
            raw,measured=execute(c,fd,stage,image_digest=reg['worker_image_digest'],converter_uid=os.getuid(),run=fake)
            self.assertEqual(measured['frames'],2400)
            self.assertEqual(len(calls),1)
            argv,kw=calls[0];self.assertIn('pcm_s24le',argv);self.assertNotIn('shell',kw);self.assertNotIn('start_new_session',kw)
            self.assertEqual(kw['timeout'],25)
            output=dict(measured,provenance=provenance(c['plan']))
            verify_output(c,raw,output)
            with self.assertRaises(ValueError): verify_output(c,raw,dict(output,sha256='1'*64))
            with self.assertRaises(ValueError): verify_converter(reg,'sha256:'+'1'*64,owner_uid=os.getuid())
            with self.assertRaises(FileExistsError): execute(c,fd,stage,image_digest=reg['worker_image_digest'],converter_uid=os.getuid(),run=fake)
            self.assertEqual(len(calls),1)

    def test_unavailable_converter_never_calls_process(self):
        c=fixture()['context']
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'stage';stage.mkdir(mode=0o770);stage.chmod(0o770)
            fd=os.open('contracts/fixtures/v1/short.wav',os.O_RDONLY)
            try:
                with self.assertRaises((ValueError,OSError)):
                    execute(c,fd,stage,image_digest='sha256:'+'1'*64,run=lambda *a,**k:self.fail('spawned'))
            finally: os.close(fd)
            self.assertEqual(list(stage.iterdir()),[])


class ExportWorkerBoundaryTests(unittest.TestCase):
    def test_supervisor_refuses_unregistered_converter_independently(self):
        from worker.runtime_supervisor import RuntimeSupervisor
        s=RuntimeSupervisor.__new__(RuntimeSupervisor);s.audio_converters={}
        with self.assertRaisesRegex(ValueError,'unregistered_converter'):
            s.verify_export(fixture()['context'],'alice')

    def test_child_protocol_uses_authenticated_authority_and_stages_no_synthetic_output(self):
        from unittest.mock import patch
        import time
        from service.worker_transport import Authority
        from worker.runtime_child import run_export
        from worker.transport import Refused
        f=fixture();context=f['context'];profile=f['profiles'][0]
        profile.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INERT fixture only')
        from contracts import validate as oracle
        plan=context['plan'];plan['resolved']['profile']=oracle.profile_snapshot(profile)
        job=dict(id='job-fixture',operation='audio_export',input_sha256=oracle.plan_hash(plan),principal_id='alice',idempotency_key='export',state='running',generation=1,attempts=[dict(id='attempt',generation=1,owner_id='owner',staging_id='stage',reaped=False)],artifact_ids=[],stage='loading',deadline='2026-10-25T23:59:59+00:00')
        a=Authority(job,context,['short'],[profile])
        config=dict(job=job,context=context,allowed_assets=['short'],profiles=[profile],secret='a'*64,control_uid=os.getuid(),gid=os.getgid(),reference_fd=99,converter_image_digest=plan['resolved']['identity']['converter']['worker_image_digest'])
        raw=wav();sent=[]
        with patch('worker.transport.authenticate') as auth,patch('worker.runtime_child.receive',return_value=a.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.audio_export.execute',return_value=(raw,pcm(raw))) as convert:
            run_export(None,config,Path('/inert/staging'),time.monotonic()+5)
            self.assertEqual(auth.call_args.args[1].binding,a.binding)
            self.assertEqual(convert.call_args.args[1],99)
        a.check(sent[0]['message'],'result')
        manifest=sent[0]['manifest'];self.assertEqual(manifest['operation'],'audio_export')
        self.assertEqual(manifest['audio_export']['settings'],plan['request'])
        self.assertEqual(manifest['candidates'],[])
        with patch('worker.transport.authenticate',side_effect=Refused('unbound_session')),patch('worker.audio_export.execute') as convert:
            with self.assertRaisesRegex(Refused,'unbound_session'):run_export(None,config,Path('/inert'),time.monotonic()+5)
            convert.assert_not_called()

    def test_process_failure_retains_attempt_staging_without_result(self):
        from unittest.mock import patch
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();exe=root/'converter';exe.write_bytes(b'INERT');exe.chmod(0o555)
            lib=root/'closure';lib.write_bytes(b'INERT');lib.chmod(0o444)
            c=fixture()['context'];reg=c['execution']['converter'];reg.update(executable=str(exe),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runtime_closure=[str(lib)])
            c['plan']['resolved']['identity']['converter']=copy.deepcopy(reg)
            stage=root/'stage';stage.mkdir(mode=0o770);stage.chmod(0o770)
            fd=os.open('contracts/fixtures/v1/short.wav',os.O_RDONLY)
            try:
                def fail(*a,**kw):raise subprocess.CalledProcessError(1,a[0])
                with self.assertRaises(subprocess.CalledProcessError):
                    execute(c,fd,stage,image_digest=reg['worker_image_digest'],converter_uid=os.getuid(),run=fail)
                self.assertTrue((stage/'output.wav').exists())
                self.assertEqual((stage/'output.wav').stat().st_size,0)
            finally:os.close(fd)
