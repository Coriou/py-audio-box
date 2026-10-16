#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${W2_PYTHON:-python3}" - <<'PY'
import fcntl, hashlib, json, os, pathlib, subprocess, time, uuid
from service.runtime_adapter import DockerHarnessRuntime
from service.worker_supervisor import Supervisor
from worker.lifecycle import State
lane=open('/tmp/pab-worker-w1-docker-lane.lock','a')
fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
project='pab-w2b-'+uuid.uuid4().hex[:12]
out=pathlib.Path('docs/agent-interface/phase-1/w2b-evidence');out.mkdir(exist_ok=True)
record={'harness_sha256':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ('scripts/test-worker-w2b.sh','service/compose.w2.yml','service/Dockerfile.w2','service/Dockerfile.w2.dockerignore')},'project':project,'commands':[], 'tier':'existing pinned dependency source-overlay; synthetic E1/E2 only'}
env=dict(os.environ,W2_CONTROL_IMAGE=project+'-control',W2_WORKER_IMAGE=project+'-worker')
compose=['docker','compose','-p',project,'-f','service/compose.w2.yml']
def run(args, timeout=30, check=True):
 r=subprocess.run(args,env=env,text=True,capture_output=True,timeout=timeout)
 record['commands'].append(dict(argv=args,exit=r.returncode,output=(r.stdout+r.stderr)[-18000:]))
 (out/'run.json').write_text(json.dumps(record,indent=2))
 if check and r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout

def dc(*args,**kw): return run(compose+list(args),**kw)
def inventory():return {k:run(args).split() for k,args in [('containers',['docker','ps','-aq']),('volumes',['docker','volume','ls','-q'])]}
before=inventory();record['before']=before
runtime=DockerHarnessRuntime(lambda args, timeout: run(['docker',*args],timeout))
identity=None
checks=[]
try:
 base=run(['docker','image','inspect','speech-core-1a-tests:f1-f2','--format','{{.Id}}']).strip()
 assert base=='sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e'
 record['dependency_image']=base
 for target in ('worker','control'):
  run(['docker','build','--network=none','--memory','2g','-f','service/Dockerfile.w2','--target',target,'-t',env['W2_'+target.upper()+'_IMAGE'],'.'],180)
 record['images']={target:run(['docker','image','inspect',env['W2_'+target.upper()+'_IMAGE'],'--format','{{.Id}}']).strip() for target in ('worker','control')}
 roots=['service','worker','contracts','clients','tests/service','tests/release']
 source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 for target in ('control','worker'):
  name=project+'-source-'+target;checks.append(name)
  script="import pathlib,hashlib,json; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in "+repr(roots)+" for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}))"
  mapped=json.loads(run(['docker','run','--name',name,'--network','none',env['W2_'+target.upper()+'_IMAGE'],'python','-c',script]))
  if target=='control': assert mapped==source
  else:
   assert not any(p.startswith('service/') for p in mapped)
   assert all(source[p]==h for p,h in mapped.items())
  record[target+'_source_sha256']=mapped
 name=project+'-regression';checks.append(name)
 run(['docker','run','--name',name,'--network','none','--cpus','2','--memory','1g','--pids-limit','128','--read-only','--tmpfs','/tmp:rw,nosuid,size=64m',env['W2_CONTROL_IMAGE'],'python','-m','pytest','tests/service','tests/release','-q','-p','no:cacheprovider'],120)
 record['scenarios']=[]
 for scenario in ('deadline_absent_control','bounded_stop'):
  record['scenarios'].append({'name':scenario})
  env['W2_IGNORE_TERM']='1' if scenario=='bounded_stop' else '0'
  dc('up','-d','init'); dc('run','--name',project+'-prepare','--no-deps','control')
  dc('create','worker')
  cid=dc('ps','-aq','worker').strip();identity=runtime.identity(cid)
  def control(action, state=State.ALIVE):
   name=project+'-'+scenario+'-'+action+'-'+uuid.uuid4().hex[:4]; checks.append(name)
   if action=='lostack':
    result=dc('run','--name',name,'--no-deps','-e','W2_ACTION='+action,'-e','W2_RUNTIME_ID='+identity,'-e','W2_STATE='+state.value,'control',check=False)
    item=json.loads(run(['docker','inspect',name]))[0]
    assert item['State']['ExitCode']==17
    return result
   return dc('run','--name',name,'--no-deps','-e','W2_ACTION='+action,'-e','W2_RUNTIME_ID='+identity,'-e','W2_STATE='+state.value,'control')
  control('bind',State.UNKNOWN)
  runtime.start(identity,3)
  assert runtime.observe(identity,2).state==State.ALIVE
  # Ready is an explicit worker barrier, not a guessed sleep.
  run(['docker','exec',cid,'python','-c',"import pathlib,time; end=time.monotonic()+3\nwhile not pathlib.Path('/ipc/ready').exists():\n if time.monotonic()>end: raise TimeoutError('ready')\n time.sleep(.01)"])
  control('stale')
  control('lostack');control('collect');control('collect') # fresh control process, same cached IDs/bytes
  inspected=json.loads(run(['docker','inspect',cid]))[0]
  topology=dict(id=inspected['Id'],image=inspected['Image'],user=inspected['Config']['User'],network=inspected['HostConfig']['NetworkMode'],pid_mode=inspected['HostConfig']['PidMode'],mounts=[dict(type=m['Type'],destination=m['Destination'],rw=m['RW']) for m in inspected['Mounts']],readonly=inspected['HostConfig']['ReadonlyRootfs'],cap_drop=inspected['HostConfig']['CapDrop'])
  assert topology['network']=='none' and topology['user']=='10002:10001' and topology['readonly'] and topology['cap_drop']==['ALL']
  assert {m['destination'] for m in topology['mounts']}=={'/ipc','/staging'} and all(m['type']=='volume' for m in topology['mounts'])
  record['scenarios'][-1]['topology']=topology
  if scenario=='bounded_stop':
   class SerializedControlStore:
    # Harness-only command proxy. All actual DB reads/writes occur inside the
    # exclusive OwnerStore control process, never in the worker/host runtime.
    def current(self): return json.loads(control('status'))
    def observe(self, evidence):
     assert evidence.runtime_id==identity
     return json.loads(control('observe',evidence.state))
   before_signals=len(record['commands'])
   stopped=Supervisor(SerializedControlStore(),runtime,timeout=3).stop()
   assert stopped['state']==State.STOPPED.value
   signals=[c['argv'] for c in record['commands'][before_signals:] if c['argv'][:2]==['docker','kill']]
   assert [a[3] for a in signals]==['TERM','KILL'],signals
   record['scenarios'][-1]['escalated_after_alive_observation']=True
  else:
   run(['docker','wait',cid],25)
  evidence=runtime.observe(identity,3); assert evidence.state==State.STOPPED
  record['scenarios'][-1]['termination']=dict(runtime_id=identity,state=evidence.state.value)
  control('verify',State.STOPPED)
  control_ids=run(['docker','ps','-aq','--filter','label=com.docker.compose.project='+project,'--filter','label=com.docker.compose.service=control']).split()
  controls=json.loads(run(['docker','inspect',*control_ids]))
  record['scenarios'][-1]['control_topology']=[]
  for item in controls:
   mounts=[dict(type=m['Type'],destination=m['Destination'],rw=m['RW']) for m in item['Mounts']]
   assert item['Config']['User']=='10001:10001' and item['HostConfig']['NetworkMode']=='none'
   assert item['HostConfig']['PidMode']=='' and item['HostConfig']['ReadonlyRootfs']
   assert {m['destination'] for m in mounts}=={'/ipc','/staging','/control'} and all(m['type']=='volume' for m in mounts)
   record['scenarios'][-1]['control_topology'].append(dict(id=item['Id'],image=item['Image'],user=item['Config']['User'],mounts=mounts,network='none',pid_mode=''))
  # Do not restart this identity; a Docker restart invalidates trusted observation.
  if scenario=='bounded_stop':
   run(['docker','start',cid]); assert runtime.observe(identity,2).state==State.UNKNOWN
   run(['docker','wait',cid],25)
   record['scenarios'][-1]['restart_quarantined']=True
  # remove only after independent daemon wait/inspect confirms termination
  assert not json.loads(run(['docker','inspect',cid]))[0]['State']['Running']
  # one-shot run containers are not removed by compose down on every Docker version
  for name in [project+'-prepare',*checks]:
   found=run(['docker','inspect',name],check=False)
   if found and json.loads(found):
    items=json.loads(found)
    assert all(not x['State']['Running'] for x in items)
    run(['docker','rm',name])
  dc('down','--volumes',timeout=20)
finally:
 dc('stop','-t','2',timeout=15,check=False)
 ids=dc('ps','-aq',check=False).split()
 for cid in ids:
  run(['docker','wait',cid],10)
  assert not json.loads(run(['docker','inspect',cid]))[0]['State']['Running']
 for name in [project+'-prepare',*checks]:
  raw=run(['docker','inspect',name],check=False)
  if raw and json.loads(raw):
   item=json.loads(raw)[0]
   if item['State']['Running']:
    run(['docker','stop','-t','2',name],10);run(['docker','wait',name],10)
   assert not json.loads(run(['docker','inspect',name]))[0]['State']['Running']
   run(['docker','rm',name])
 dc('down','--volumes',timeout=20,check=False)
 after=inventory();record['prior_inventory_preserved']={k:set(before[k])<=set(after[k]) for k in before}
 assert all(record['prior_inventory_preserved'].values())
 record['after']=after
 (out/'run.json').write_text(json.dumps(record,indent=2))
print('W2b bounded synthetic checks complete')
PY
