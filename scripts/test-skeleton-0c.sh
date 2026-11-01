#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# One heavy-lane owner; sequential builds, explicit deadlines, no user mounts.
exec "${SPEECH_0C_PYTHON:-python3}" - <<'PY'
import subprocess
import sys
import uuid

def run(command, deadline):
    subprocess.run(command, check=True, timeout=deadline)

for target in ('control', 'worker', 'tests'):
    run(['docker','build','--memory','2g','-f','service/Dockerfile.release','--target',target,
         '-t',f'speech-skeleton-0c-{target}:0.1.0','.'], 600)
def container(arguments, deadline):
    name = 'speech-0c-test-' + uuid.uuid4().hex[:10]
    try:
        run(['docker','run','--name',name,'--rm',*arguments], deadline)
    finally:
        exists = subprocess.run(['docker','container','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        if exists.returncode == 0:
            run(['docker','rm','-f',name], 15)

container(['--network','none','--cpus','2','--memory','1g','--pids-limit','128',
     '--read-only','--tmpfs','/tmp:rw,nosuid,size=64m','speech-skeleton-0c-tests:0.1.0'],120)
container(['--network','none','--cpus','1','--memory','384m','--read-only',
     'speech-skeleton-0c-tests:0.1.0','python','-m','tests.release.measure'],30)
run([sys.executable,'tests/release/smoke.py'],180)
PY
