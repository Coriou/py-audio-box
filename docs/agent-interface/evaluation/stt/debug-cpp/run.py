"""One bounded pinned whisper.cpp mechanics probe after image provisioning."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[5]
evaluation = root / 'docs/agent-interface/evaluation/stt'
assets = root / 'work/stt-evaluation'
out = Path(__file__).resolve().parent / 'run-v2'
image = 'ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647'
name = 'pab-stt-debug-cpp-v2-20261015'
model = assets / 'models/whisper-cpp/ggml-tiny-q5_1.bin'
expected = '818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7'
assert hashlib.sha256(model.read_bytes()).hexdigest() == expected
manifest = json.loads((assets / 'manifest.json').read_text())
samples = [next(s for s in manifest['samples'] if s['id'] == sid)
           for sid in ['fr-1679', 'en-1914']]
for sample in samples:
    local = assets / Path(sample['file']).relative_to('/assets')
    assert hashlib.sha256(local.read_bytes()).hexdigest() == sample['sha256']
if out.exists():
    raise SystemExit('Refusing repeat/overwrite')
image_metadata = subprocess.check_output(['docker', 'image', 'inspect', image], timeout=10)
if not (out.parent / 'image.json').exists():
    (out.parent / 'image.json').write_bytes(image_metadata)
if not (out.parent / 'inputs.json').exists():
    (out.parent / 'inputs.json').write_text(json.dumps(samples, indent=2))
command = ['docker', 'run', '--name', name, '--network', 'none', '--cpus', '4',
           '--memory', '6g', '--pids-limit', '256',
           '--mount', f'type=bind,src={assets},dst=/assets,readonly',
           '--mount', f'type=bind,src={out},dst=/results',
           '--entrypoint', '/usr/bin/timeout', image, '--signal=TERM',
           '--kill-after=3s', '85s', '/app/build/bin/whisper-cli',
           '--model', '/assets/models/whisper-cpp/ggml-tiny-q5_1.bin',
           '--no-gpu', '--threads', '4', '--beam-size', '5', '--best-of', '5',
           '--language', 'auto', '--max-context', '0', '--output-json']
for sample in samples:
    command += ['--file', sample['file'], '--output-file', '/results/' + sample['id']]
code = subprocess.call([
    sys.executable, str(evaluation / 'debug-inference/supervise.py'),
    '--name', name, '--seconds', '90', '--out', str(out), '--', *command])
if code:
    raise SystemExit(code)
# This CLI can print an argument error/help and still return zero.
for sample in samples:
    result = json.loads((out / (sample['id'] + '.json')).read_text())
    if not result.get('transcription'):
        raise RuntimeError('No transcription artifact for ' + sample['id'])
