#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Offline, sequential, bounded, disposable synthetic containers; no user mounts.
exec "${W1_PYTHON:-python3}" - <<'PY'
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

lane = open('/tmp/pab-worker-w1-docker-lane.lock', 'a')
fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
out=Path('w1-evidence');out.mkdir(exist_ok=True)
project='pab-w1-'+uuid.uuid4().hex[:12]
env=dict(os.environ,W1_CONTROL_IMAGE=project+'-control',W1_WORKER_IMAGE=project+'-worker')
compose=['docker','compose','-p',project,'-f','service/compose.w1.yml']
def run(args,timeout=30,check=True):
    return subprocess.run(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,check=check)
def dc(*args,**kw):return run(compose+list(args),**kw)
def inventory(kind):return sorted(run(['docker',kind,'ls','-q'] if kind=='volume' else ['docker','ps','-aq']).stdout.split())
before={k:inventory(k) for k in ('container','volume')}
record={'project':project,'before':before,'commands':[]}
owned_runs=[]
def logged(args,timeout):
    if args[:2]==['docker','run']:
        name=project+'-check-'+str(len(owned_runs))
        args=args[:2]+['--name',name]+args[2:]
        owned_runs.append(name)
    result=run(args,timeout,False)
    record['commands'].append({'argv':args,'exit':result.returncode,'output':result.stdout if len(result.stdout)<12000 else result.stdout[-12000:]})
    (out/'run.json').write_text(json.dumps(record,indent=2))
    if result.returncode: raise RuntimeError(result.stdout[-4000:])
    return result.stdout
try:
    base=json.loads(run(['docker','image','inspect','speech-core-1a-tests:f1-f2']).stdout)[0]['Id']
    assert base=='sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e', 'dependency image drift'
    record['dependency_image']=base
    for target in ('worker','control'):
        logged(['docker','build','--network=none','--memory','2g','-f','service/Dockerfile.w1','--target',target,'-t',env['W1_'+target.upper()+'_IMAGE'],'.'],180)
    record['images']={k:json.loads(run(['docker','image','inspect',env['W1_'+k.upper()+'_IMAGE']]).stdout)[0]['Id'] for k in ('control','worker')}
    # Exact source maps, and worker exclusion proof from the actual built filesystem.
    paths=['service','worker','contracts','clients','tests/service','tests/release']
    def local_map():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in paths for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    script="import pathlib,hashlib,json; roots="+repr(paths)+"; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for r in roots for p in pathlib.Path(r).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}))"
    mapped=json.loads(logged(['docker','run','--rm','--network','none',env['W1_CONTROL_IMAGE'],'python','-c',script],30))
    assert mapped==local_map(), 'control source mismatch'
    record['source_sha256']=mapped
    worker_check="import pathlib,json; assert not pathlib.Path('service').exists(); assert not pathlib.Path('/var/run/docker.sock').exists(); print(json.dumps({str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for r in ['worker','contracts'] for p in pathlib.Path(r).rglob('*') if p.is_file()}))"
    worker_map=json.loads(logged(['docker','run','--rm','--network','none',env['W1_WORKER_IMAGE'],'python','-c',worker_check],30))
    assert all(mapped[k]==v for k,v in worker_map.items())
    record['worker_source_sha256']=worker_map
    logged(['docker','run','--rm','--network','none','--cpus','2','--memory','1g','--pids-limit','128','--read-only','--tmpfs','/tmp:rw,nosuid,size=64m',env['W1_CONTROL_IMAGE'],'python','-m','pytest','tests/service','tests/release','-q','-p','no:cacheprovider'],120)
    record['scenarios']=[]
    for fault in ('none','session','tuple'):
        env['W1_FAULT']=fault
        expected='verified_synthetic_proposal' if fault=='none' else 'verified_refusal'
        scenario={'fault':fault}
        record['scenarios'].append(scenario)
        dc('up','-d',timeout=30)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            logs=dc('logs','--no-color','control','worker').stdout
            if expected in logs:break
            time.sleep(.1)
        else:raise RuntimeError(logs)
        ids=dc('ps','-aq').stdout.split()
        inspected=json.loads(run(['docker','inspect',*ids]).stdout)
        topology=[]
        for item in inspected:
            service=item['Config']['Labels']['com.docker.compose.service']
            mounts=[{'type':m['Type'],'name':m.get('Name'),'destination':m['Destination'],'rw':m['RW']} for m in item['Mounts']]
            topology.append(dict(service=service,id=item['Id'],image=item['Image'],user=item['Config']['User'],network=item['HostConfig']['NetworkMode'],pid_mode=item['HostConfig']['PidMode'],readonly=item['HostConfig']['ReadonlyRootfs'],mounts=mounts,cap_drop=item['HostConfig']['CapDrop']))
            assert item['HostConfig']['NetworkMode']=='none'
            assert all(m['type']=='volume' for m in mounts)
            assert {m['destination'] for m in mounts}=={'/ipc','/staging'}
            if service=='worker':
                assert item['Config']['User']=='10002:10001'
                assert not next(m['rw'] for m in mounts if m['destination']=='/ipc')
                assert item['HostConfig']['ReadonlyRootfs'] and item['HostConfig']['CapDrop']==['ALL']
        scenario['topology']=topology
        scenario['roundtrip_logs']=logs
        dc('exec','-T','control','python','-c',"from pathlib import Path; Path('/ipc/release').touch()")
        for service in ('worker','control'):
            cid=dc('ps','-aq',service).stdout.strip()
            state=json.loads(run(['docker','wait',cid],10).stdout)
            expected_exit = 2 if service == 'worker' and fault == 'session' else 0
            assert state==expected_exit, (service,state)
        scenario['terminated']=True
        dc('down','--volumes',timeout=20)
finally:
    for name in owned_runs:
        found=run(['docker','container','inspect',name],check=False)
        if found.returncode==0:
            run(['docker','stop','-t','2',name],10,False)
            state=json.loads(run(['docker','inspect',name]).stdout)[0]['State']
            assert not state['Running'], 'retain unknown test ownership'
            run(['docker','rm',name],10)
    # Stop only this random project; inspect all stopped before removing its volumes.
    dc('stop','-t','2',timeout=15,check=False)
    ids=dc('ps','-aq',check=False).stdout.split()
    if ids:
        states=json.loads(run(['docker','inspect',*ids]).stdout)
        assert all(not x['State']['Running'] for x in states), 'retain unknown ownership'
    dc('down','--volumes','--remove-orphans',timeout=20)
    record['after']={k:inventory(k) for k in ('container','volume')}
    record['prior_inventory_preserved']={k:set(before[k])<=set(record['after'][k]) for k in before}
    assert all(record['prior_inventory_preserved'].values())
    record['owned_containers_remaining']=dc('ps','-aq').stdout.split()
    record['owned_volumes_remaining']=run(['docker','volume','ls','-q','--filter','label=com.docker.compose.project='+project]).stdout.split()
    assert not record['owned_containers_remaining'] and not record['owned_volumes_remaining']
    (out/'run.json').write_text(json.dumps(record,indent=2))
print('W1 separate-container synthetic transport verified; evidence: w1-evidence/run.json')
PY
