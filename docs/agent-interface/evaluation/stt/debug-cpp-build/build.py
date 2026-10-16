"""Bounded local portable build from the exact already downloaded image."""
import datetime, json, os, signal, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = 'ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647'
NAME = 'pab-cpp-portable-build-20261015'
COMMAND = '''set -eu
git -C /app rev-parse HEAD > /output/source-commit.txt
cmake -S /app -B /output/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /output/build --target whisper-cli -j4
'''

def docker(*args):
    return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=10)

def main():
    if (ROOT/'build-result.json').exists() or (ROOT/'build').exists():
        raise SystemExit('Refusing existing build artifacts')
    if docker('container','inspect',NAME).returncode == 0:
        raise SystemExit('Refusing existing container')
    command = ['docker','run','--name',NAME,'--network','none','--cpus','4','--memory','6g','--pids-limit','256','-v',f'{ROOT}:/output','--entrypoint','/bin/sh',IMAGE,'-c',COMMAND]
    start=time.monotonic(); timed_out=False
    with (ROOT/'build.log').open('wb') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out=True
            docker('kill',NAME)
            os.killpg(proc.pid,signal.SIGTERM)
            try: code=proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); code=proc.wait(timeout=3)
    inspected=docker('inspect',NAME)
    (ROOT/'build-container.json').write_text(inspected.stdout)
    result=dict(command=command,exit_code=code,timed_out=timed_out,wall_seconds=time.monotonic()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inspection_exit=inspected.returncode)
    (ROOT/'build-result.json').write_text(json.dumps(result,indent=2))
    if inspected.returncode==0: docker('rm',NAME)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
