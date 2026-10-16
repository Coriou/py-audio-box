"""Capture the original image SIGILL instruction, with an owned 10s watchdog."""
import json, subprocess, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
ASSETS=ROOT/'work/stt-evaluation'
IMAGE='ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647'
manifest=json.loads((ASSETS/'manifest.json').read_text())
sample=next(s for s in manifest['samples'] if s['id']=='fr-1679')
script='gcc -shared -fPIC -O0 -o /output/sigill.so /output/sigill.c -ldl && LD_PRELOAD=/output/sigill.so /usr/bin/timeout --signal=TERM --kill-after=1s 8s /app/build/bin/whisper-cli --no-gpu --threads 1 --language auto --max-context 0 --model /assets/models/whisper-cpp/ggml-tiny-q5_1.bin --file '+sample['file']
cmd=['docker','run','--name','pab-stt-debug-sigill-20261015','--network','none','--cpus','1','--memory','1g','--pids-limit','64','-v',f'{BASE}:/output','-v',f'{ASSETS}:/assets:ro','--entrypoint','/bin/sh',IMAGE,'-c',script]
raise SystemExit(subprocess.call([sys.executable,str(BASE.parent/'debug-inference/supervise.py'),'--name','pab-stt-debug-sigill-20261015','--seconds','10','--out',str(BASE/'sigill-run'),'--',*cmd]))
