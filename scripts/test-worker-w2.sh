#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# W2a only: offline source overlay, not W2 container supervision acceptance.
exec "${W2_PYTHON:-python3}" - <<'PY'
import fcntl,hashlib,json,pathlib,subprocess,uuid
lane=open("/tmp/pab-worker-w1-docker-lane.lock", "a")
fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
out=pathlib.Path('docs/agent-interface/phase-1/w2-evidence/local.json')
name='pab-w2a-'+uuid.uuid4().hex[:12]
image='sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e'
record={'container':name,'dependency_image':image,'evidence_tier':'E1 source-overlay, not packaged W2 or separate-container E2','commands':[]}
def run(args,timeout=30,check=True):
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
 record['commands'].append({'argv':args,'exit':r.returncode,'output': '[inspect omitted; reduced termination state recorded]' if args[1:2]==['inspect'] else r.stdout})
 out.write_text(json.dumps(record,indent=2))
 if check and r.returncode:raise RuntimeError(r.stdout)
 return r
verifier=pathlib.Path('/tmp/'+name+'.py')
before=set(run(['docker','ps','-aq']).stdout.split())
try:
 run(['docker','create','--name',name,'--network','none','--cpus','2','--memory','1g','--pids-limit','128',image,'python','-m','pytest','tests/service/test_worker_lifecycle.py','tests/service/test_w2_source_check.py','-q','-p','no:cacheprovider'])
 roots=['service','worker','tests/service','tests/release','contracts','clients']
 record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 for root in roots:run(['docker','cp',root+'/.',name+':/opt/speech/'+root])
 verifier.write_text('import pathlib,hashlib\nexpected='+repr(record['source_sha256'])+'\ndef test_exact_supplied_source():\n for p,h in expected.items():\n  assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\n')
 run(['docker','cp',str(verifier),name+':/opt/speech/tests/service/test_w2_source_check.py'])
 result=run(['docker','start','-a',name],timeout=120,check=False)
 state=json.loads(run(['docker','inspect',name]).stdout)[0]['State']
 record['terminated_state']={k:state[k] for k in ('Running','Status','ExitCode')}
 if state['ExitCode']:raise RuntimeError(result.stdout)
finally:
 state=json.loads(run(['docker','inspect',name],check=False).stdout or '[]')
 if state and state[0]['State']['Running']:
  run(['docker','stop','-t','2',name],timeout=10)
  state=json.loads(run(['docker','inspect',name]).stdout)
 if state:
  assert not state[0]['State']['Running']
  run(['docker','rm',name])
 after=set(run(['docker','ps','-aq']).stdout.split())
 record['prior_inventory_preserved']=before<=after
 assert record['prior_inventory_preserved']
 verifier.unlink(missing_ok=True)
 out.write_text(json.dumps(record,indent=2))

PY
