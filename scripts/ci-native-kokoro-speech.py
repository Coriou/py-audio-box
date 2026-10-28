#!/usr/bin/env python3
"""One opt-in cold EN Kokoro backend test; no installed/native CPU admission."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DESCRIPTOR = ROOT/'requirements/tts/kokoro-en-macos.expected.json'
DESCRIPTOR_SHA = 'afbc1fafed69c55adff4e94ccdb936f347f99cdc60171ed0ae9a2314b080c7f3'
TEXT = 'Hello. The test is complete.'
PRESET = 'af_heart'
TOTAL = 180
ACQUIRE = 120
HOSTS = frozenset(('github.com', 'release-assets.githubusercontent.com'))


def owner_module():
    spec = importlib.util.spec_from_file_location('kokoro_speech_owner', ROOT/'scripts/ci-apple-speech.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    return owner


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    if len(raw) > 65536: raise ValueError('report_limit')
    path.write_bytes(raw)


def expected():
    raw = DESCRIPTOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESCRIPTOR_SHA:
        raise ValueError('kokoro_descriptor_identity')
    from service.provisioning import validate_expected
    return validate_expected(json.loads(raw))


def public_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname not in HOSTS or p.username or p.password or p.port not in (None,443) or p.fragment:
        raise ValueError('kokoro_download_host')
    return url


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, public_url(newurl))


def acquire(bundle):
    from service.provisioning import manifest_bytes, bundle_manifest
    descriptor = expected()
    if shutil.disk_usage(bundle).free < 1024**3: raise ValueError('requires_1GiB_free_disk')
    started = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects())
    for entry in descriptor['files']:
        # Origins, names, sizes and hashes are bound by the exact descriptor SHA.
        digest = hashlib.sha256(); remaining = entry['bytes']
        with opener.open(public_url(entry['origin']), timeout=15) as source, (bundle/entry['path']).open('xb') as dst:
            while remaining:
                if time.monotonic()-started >= ACQUIRE: raise TimeoutError('acquire_120_seconds')
                block = source.read(min(1024**2,remaining))
                if not block: raise ValueError('download_truncated')
                dst.write(block); digest.update(block); remaining -= len(block)
            if source.read(1) or digest.hexdigest() != entry['sha256']: raise ValueError('download_size_or_hash')
        (bundle/entry['path']).chmod(0o400)
    (bundle/'manifest.json').write_bytes(manifest_bytes(bundle_manifest(descriptor)))
    (bundle/'manifest.json').chmod(0o400); bundle.chmod(0o500)


def wav_evidence(raw):
    import numpy as np
    with wave.open(io.BytesIO(raw),'rb') as wav:
        if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype()) != (1,2,24000,'NONE'):
            raise ValueError('wav_format')
        frames=wav.getnframes(); pcm=wav.readframes(frames)
    samples=np.frombuffer(pcm,dtype='<i2')
    if frames <= 0 or len(raw)!=44+frames*2 or len(samples)!=frames or not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError('wav_empty_nonfinite_or_truncated')
    return dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),frames=frames,
        sample_rate=24000,channels=1,subtype='PCM_16',duration_seconds=frames/24000,
        nonzero=True,finite=True,semantic_ending='unverified')


def infer(bundle, output):
    progress={}; started=time.monotonic()
    def phase(stage, **values):
        progress.update(values,stage=stage,stage_started=time.monotonic())
        temporary=output/'progress.tmp'; save(temporary,progress); temporary.replace(output/'progress.json')
        print('phase',stage,flush=True)
    phase('cold_load',device='cpu',dtype='float32',threads=2,preset=PRESET,text=TEXT,speed=1.0)
    from worker.native_child import install_spawn_guard
    install_spawn_guard()
    from worker.engines.execution import Execution
    from worker.engines.kokoro import execute
    from lib.tts_backends.kokoro import KokoroBackend
    import inspect
    import numpy as np
    descriptor=expected()
    class Measured(KokoroBackend):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self._providers != ('CPUExecutionProvider',): raise ValueError('cpu_provider_required')
            if PRESET not in self._voices: raise ValueError('required_af_heart_absent')
            if inspect.signature(self.model.create).parameters['speed'].default != 1.0:
                raise ValueError('pinned_default_speed_mismatch')
            floating={x.name:x.type for x in self.model.sess.get_inputs() if x.type.startswith('tensor(float')}
            if not floating or any(v!='tensor(float)' for v in floating.values()): raise ValueError('float32_inputs_required')
            phase('generation',cold_load_seconds=time.monotonic()-started,providers=list(self._providers),floating_inputs=floating)
        def synthesise(self,*args,**kwargs):
            generation=time.monotonic()
            result=super().synthesise(*args,**kwargs)
            if np.asarray(result.samples).dtype != np.dtype('float32'): raise ValueError('float32_audio_required')
            phase('encoding',generation_seconds=time.monotonic()-generation)
            return result
        def close(self):
            super().close(); progress['backend_closed']=True
    manifest=(bundle/'manifest.json').read_bytes()
    from service.provisioning import manifest_bytes, bundle_manifest
    if manifest != manifest_bytes(bundle_manifest(descriptor)):
        raise ValueError('kokoro_en_manifest_identity')
    execution=Execution(bundle_id='kokoro-en',manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        checkpoint=descriptor['checkpoint'],revision=descriptor['revision'],preset=PRESET,language='en',threads=2,
        max_text_bytes=256,max_frames=24000*30,max_output_bytes=1440044,memory_bytes=6*1024**3)
    try:
        result=execute(execution,TEXT,{'kokoro-en':bundle},backend_factory=Measured)
        evidence=wav_evidence(result['data'])
        if (result['sha256'],result['frames'],result['size_bytes']) != (evidence['sha256'],evidence['frames'],evidence['size_bytes']):
            raise ValueError('engine_wav_identity')
        (output/'English.wav').write_bytes(result['data'])
        evidence.update(status='passed',execution=result['identity'],progress=progress.copy(),
            backend_total_seconds=time.monotonic()-started,
            generation_rtf=progress['generation_seconds']/evidence['duration_seconds'])
        save(output/'English.json',evidence)
        phase('complete',encoding_seconds=time.monotonic()-progress['stage_started'])
    except BaseException as exc:
        progress.update(error_type=type(exc).__name__,error=str(exc)[:2000])
        save(output/'progress.json',progress)
        raise


def run(output):
    from lib.tts_backends.kokoro_native import validate_host, validate_versions
    validate_host(); validate_versions(); descriptor=expected()
    output.mkdir(parents=True,exist_ok=False)
    owner=owner_module(); report=dict(status='failed',commit=os.environ.get('GITHUB_SHA'),
        runner_image=os.environ.get('ImageOS'),runner_version=os.environ.get('ImageVersion'),python=sys.version,
        descriptor_sha256=DESCRIPTOR_SHA,model=descriptor['checkpoint'],revision=descriptor['revision'],
        model_bytes=sum(f['bytes'] for f in descriptor['files']),requested_languages=['English'],
        options=dict(text=TEXT,preset=PRESET,device='cpu',dtype='float32',threads=2,speed=1.0),
        effective_budgets=dict(acquire_seconds=120,cold_load_seconds=60,generation_seconds=60,total_seconds=180),
        installed_qualified=False,diagnostic_only=True,cloning=False,children={})
    private=None; stage='acquire'
    env={k:v for k,v in os.environ.items() if k in ('PATH','TMPDIR','GITHUB_SHA','ImageOS','ImageVersion')}
    try:
        with tempfile.TemporaryDirectory(prefix='kokoro-speech-') as private:
            root=Path(private).resolve(); bundle=root/'bundle'; bundle.mkdir(mode=0o700)
            env.update(HOME=str(root),HF_HOME=str(root),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2')
            started=time.monotonic(); cutoff=started+TOTAL
            def child(kind,limit,progress=None):
                remaining=cutoff-time.monotonic()
                if remaining <= 0: raise TimeoutError('kokoro_total_180_seconds')
                observation=report['children'][kind]={}
                owner.owned_child([sys.executable,'-I','-B','-u',str(Path(__file__).resolve()),
                    '--child',kind,'--bundle',str(bundle),'--output',str(output.resolve())],env,
                    output/(kind+'.log'),observation,seconds=min(limit,remaining),
                    progress_path=progress,phase_profile='kokoro')
            try:
                child('acquire',ACQUIRE); report['acquire']='passed'; report['acquire_seconds']=time.monotonic()-started
                stage='English'; child('infer',cutoff-time.monotonic(),output/'progress.json')
                with (output/'English.wav').open('rb') as stream:
                    audio=stream.read(1440045)
                if len(audio)>1440044: raise ValueError('parent_wav_limit')
                evidence=wav_evidence(audio)
                receipt=json.loads((output/'English.json').read_text())
                if receipt['sha256']!=evidence['sha256']: raise ValueError('parent_wav_identity')
                report['wav']=evidence
                report['English']='passed'; report['measurement_seconds']=time.monotonic()-started
                if time.monotonic()>=cutoff: raise TimeoutError('kokoro_total_180_seconds')
                report['status']='passed'
            finally:
                bundle.chmod(0o700)
    except BaseException as exc:
        report.update(failed_stage=stage,error_type=type(exc).__name__,error=str(exc)[:2000])
        raise
    finally:
        report['private_model_tree_removed']=private is None or not Path(private).exists()
        save(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--run-en-backend',action='store_true')
    parser.add_argument('--child',choices=('acquire','infer'))
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    if args.child:
        if not args.bundle: parser.error('--bundle required for child')
        if args.child=='acquire': acquire(args.bundle)
        else: infer(args.bundle,args.output)
    elif args.run_en_backend: run(args.output)
    else: parser.error('explicit --run-en-backend opt-in required')
