"""One bounded ephemeral Ubuntu C++ compiler provisioning attempt."""
import json, subprocess, time, os, signal
from pathlib import Path
BASE=Path(__file__).resolve().parent
IMAGE='ghcr.io/ggml-org/whisper.cpp@sha256:23da2ac176ae6be2633f5766f1e532746383dce1864cbebf6c7af64887a23647'
NAME='pab-stt-debug-cpp-toolchain-20261015'
TARGET='py-audio-box:stt-cpp-avx2-toolchain-20261015'
SCRIPT='''set -eu
export DEBIAN_FRONTEND=noninteractive
timeout --kill-after=2s 70s apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 update
du -sb /var/lib/apt/lists > /output/toolchain-index-bytes.txt
test "$(du -sb /var/lib/apt/lists | cut -f1)" -lt 100000000
apt-get --print-uris -y --no-install-recommends install g++ > /output/toolchain-package-plan.txt
awk '/^\047http/ {total += $2} END {print total; if (total > 50000000) exit 1}' /output/toolchain-package-plan.txt > /output/toolchain-package-bytes.txt
timeout --kill-after=2s 70s apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 install -y --no-install-recommends g++
g++ --version
'''
def command(*args):
    return subprocess.run(['docker',*args],capture_output=True,text=True,timeout=20)
if (BASE/'toolchain-result.json').exists(): raise SystemExit('Refusing previous provisioning')
if command('inspect',NAME).returncode==0: raise SystemExit('Refusing previous container')
args=['docker','run','--name',NAME,'--cpus','2','--memory','1g','--pids-limit','128','-v',f'{BASE}:/output','--entrypoint','/bin/sh',IMAGE,'-c',SCRIPT]
start=time.monotonic(); timed_out=False
with (BASE/'toolchain-provision.log').open('wb') as log:
    proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try: code=proc.wait(timeout=150)
    except subprocess.TimeoutExpired:
        timed_out=True; command('kill',NAME)
        os.killpg(proc.pid,signal.SIGKILL); code=proc.wait(timeout=3)
inspection=command('inspect',NAME)
(BASE/'toolchain-container.json').write_text(inspection.stdout)
commit=None
if code==0:
    commit=command('commit',NAME,TARGET)
    (BASE/'toolchain-image.json').write_text(command('image','inspect',TARGET).stdout)
result=dict(exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,target=TARGET,commit_exit=commit.returncode if commit else None,commit_image=commit.stdout.strip() if commit else None)
(BASE/'toolchain-result.json').write_text(json.dumps(result,indent=2))
if inspection.returncode==0: command('rm',NAME)
print(json.dumps(result),flush=True)
