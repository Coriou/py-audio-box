"""Bounded owned Compose smoke. No builds, user-data mounts or global cleanup."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'speech-0c-smoke-' + uuid.uuid4().hex[:10]

def run(*args, timeout=60):
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=timeout)


def main():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]
    env = {**os.environ,'SPEECH_0C_PORT':str(port)}
    command = ['docker','compose','-p',PROJECT,'-f','service/compose.release.yml']
    def compose(*args):
        return subprocess.check_output([*command,*args],cwd=ROOT,env=env,text=True,timeout=60)
    config = json.loads(compose('config','--format','json'))
    worker = config['services']['fake-worker']
    assert worker['network_mode'] == 'none'
    assert not worker.get('environment') and not worker.get('ports')
    assert all(v['type']=='volume' for v in worker['volumes'])
    assert {v['target'] for v in worker['volumes']} == {'/run/speech-ipc','/staging','/assets'}
    assert next(v for v in worker['volumes'] if v['target']=='/assets')['read_only']
    initial_volumes = set(run('docker','volume','ls','-q').splitlines())
    try:
        compose('up','-d','--no-build','--pull','never')
        ids = {name:compose('ps','-q',name).strip() for name in ('control','fake-worker')}
        for name, identity in ids.items():
            item=json.loads(run('docker','inspect',identity))[0]
            assert item['Config']['User']=='10001:10001'
            assert item['HostConfig']['ReadonlyRootfs']
            assert item['HostConfig']['CapDrop']==['ALL']
            assert all(m['Type']=='volume' for m in item['Mounts'])
            assert all('docker.sock' not in m['Destination'] for m in item['Mounts'])
            if name=='fake-worker':
                assert item['HostConfig']['NetworkMode']=='none'
                assert not item['NetworkSettings']['Networks'].get('default')
        # Docker internal networks do not guarantee host-published access on this daemon.
        # Check HTTP inside the offline control, and report host reachability separately.
        host_loopback = False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/skeleton',timeout=1) as response:
                host_loopback = not json.load(response)['ready_for_speech']
        except OSError:
            pass
        code = """
import socket, pathlib, urllib.request, json, time
for _ in range(100):
 try:
  with urllib.request.urlopen('http://127.0.0.1:8780/skeleton',timeout=1) as response:
   assert json.load(response)['ready_for_speech'] is False
  break
 except OSError: time.sleep(.1)
else: raise RuntimeError('Internal control startup timed out')
assert pathlib.Path('/run/speech-ipc').stat().st_mode & 0o777 == 0o700
with socket.socket(socket.AF_UNIX) as sock:
 sock.settimeout(3); sock.connect('/run/speech-ipc/boundary.sock')
 sock.sendall(b'0C-PING\\n'); assert sock.recv(64)==b'0C-BOUNDARY-ONLY\\n'
with urllib.request.urlopen('http://127.0.0.1:8780/evaluation/') as response:
 html=response.read().decode(); assert '<div id="root">' in html and '/evaluation/assets/' in html
with urllib.request.urlopen('http://127.0.0.1:8780/skeleton') as response:
 assert json.load(response)['ready_for_speech'] is False
assert not pathlib.Path('/opt/speech/apps').exists()
"""
        run('docker','exec',ids['control'],'python','-c',code)
        code = """
import pathlib, socket
assert not pathlib.Path('/opt/speech/service').exists()
assert not pathlib.Path('/opt/speech/contracts').exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not list(pathlib.Path('/').glob('*.db'))
try:
 pathlib.Path('/assets/forbidden').write_text('x')
except OSError: pass
else: raise AssertionError('worker asset volume writable')
assert sorted(pathlib.Path('/sys/class/net').iterdir()) == [pathlib.Path('/sys/class/net/lo')]
"""
        run('docker','exec',ids['fake-worker'],'python','-c',code)
        print(json.dumps({'E2':'boundary probe passed; no jobs executed','project':PROJECT, 'host_loopback_reachable':host_loopback,
            'frontend_limit':'Internal-network HTML fetch only; host UI publication/actual browser pending 1C',
            'checks':['config','nonroot','read-only root/assets','no worker network','private UDS','compiled synthetic UI','no worker service/DB mounts']}))
    finally:
        compose('down','--volumes','--remove-orphans','--timeout','5')
        assert not compose('ps','-aq').strip()
        remaining = set(run('docker','volume','ls','-q').splitlines())
        assert not any(v.startswith(PROJECT+'_') for v in remaining)
        assert initial_volumes <= remaining  # unrelated volumes preserved
        print('Owned containers/network/volumes disposed; unrelated volumes preserved.')

if __name__=='__main__': main()
