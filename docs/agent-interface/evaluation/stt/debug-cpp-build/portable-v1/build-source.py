"""Portable AVX2 rebuild using the corrected shared owned-container watchdog."""
import json, subprocess, sys
from pathlib import Path
import build

BASE=Path(__file__).resolve().parent
OUT=BASE/'portable-v1'
IMAGE='py-audio-box:stt-cpp-avx2-toolchain-20261015'
NAME='pab-stt-debug-cpp-portable-build-20261015'
if OUT.exists(): raise SystemExit('Refusing existing build')
metadata=subprocess.check_output(['docker','image','inspect',IMAGE],timeout=10)
cmd=['docker','run','--name',NAME,'--network','none','--cpus','4','--memory','6g','--pids-limit','256','-v',f'{OUT}:/output','--entrypoint','/bin/sh',IMAGE,'-c',build.COMMAND]
code=subprocess.call([sys.executable,str(BASE.parent/'debug-inference/supervise.py'),'--name',NAME,'--seconds','300','--out',str(OUT),'--',*cmd])
(OUT/'image.json').write_bytes(metadata)
(OUT/'build-source.py').write_text(Path(__file__).read_text())
(OUT/'build-command.sh').write_text(build.COMMAND)
if code: raise SystemExit(code)
PYTHON_NOTE='Build success requires actual CLI and CPU library artifacts.'
if not (OUT/'build/bin/whisper-cli').is_file(): raise RuntimeError(PYTHON_NOTE)
print(json.dumps(dict(build_complete=True,directory=str(OUT))))
