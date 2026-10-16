"""Bounded host container failure injector/auditor. Never normal runtime authority."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def main():
    lane=open('/tmp/pab-worker-w1-docker-lane.lock','a')
    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
    project='pab-w2c-'+uuid.uuid4().hex[:10]
    out=Path('docs/agent-interface/phase-1/w2c-evidence');out.mkdir(exist_ok=True)
    record=dict(project=project,tier='E1/E2 offline dependency-overlay synthetic only',commands=[],scenarios=[])
    env=dict(os.environ,W2C_CONTROL_IMAGE=project+'-control',W2C_WORKER_IMAGE=project+'-worker')
    compose=['docker','compose','-p',project,'-f','service/compose.lifecycle.yml']
    names=[]
    def run(args,timeout=30,check=True):
        result=subprocess.run(args,env=env,text=True,capture_output=True,timeout=timeout)
        record['commands'].append(dict(argv=args,exit=result.returncode,output=(result.stdout+result.stderr)[-8000:] if args[:3] not in (['docker','ps','-aq'],['docker','volume','ls']) else 'inventory recorded separately'))
        (out/'run.json').write_text(json.dumps(record,indent=2))
        if check and result.returncode: raise RuntimeError(result.stdout+result.stderr)
        return result
    def dc(*args,**kw):return run(compose+list(args),**kw)
    def inventory():
        return dict(containers=run(['docker','ps','-aq']).stdout.split(),volumes=run(['docker','volume','ls','-q']).stdout.split())
    def control(*args,check=True):
        name=project+'-c-'+uuid.uuid4().hex[:8];names.append(name)
        result=dc('run','--name',name,'--no-deps','control',*args,check=check)
        if check:
            value=json.loads(result.stdout.strip().splitlines()[-1])
            run(['docker','rm',name]) # CLI has exited; remove only confirmed completed one-shot.
            return value
        run(['docker','rm',name])
        return result
    def wait_state(state,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            value=control('observe')
            if value['state']==state:return value
            time.sleep(.05)
        raise AssertionError(('state_timeout',value))
    def audit(script):
        return run(['docker','exec',worker,'python','-c',script])
    def restart():
        run(['docker','kill','--signal','KILL',worker]);run(['docker','wait',worker])
        run(['docker','start',worker]);time.sleep(.2)
    def reset():
        nonlocal worker
        if worker:
            run(['docker','exec',worker,'python','-c',"import os;os.chmod('/receipts',0o700)"],check=False)
            run(['docker','stop','-t','3',worker],check=False);run(['docker','wait',worker],check=False)
        for cid in dc('ps','-aq',check=False).stdout.split():
            state=json.loads(run(['docker','inspect',cid]).stdout)[0]['State']
            assert not state['Running']
            run(['docker','rm',cid])
        dc('down','-v')
        dc('up','-d','worker');worker=dc('ps','-aq','worker').stdout.strip()
        end=time.monotonic()+5
        while control('hello',check=False).returncode:
            if time.monotonic()>end:raise AssertionError('reset_readiness')
    def audit_busy_tree(minimum):
        # Independent audit only: none of these observations is supplied to control.
        script="""import os,json,time
from pathlib import Path
end=time.monotonic()+2
while True:
 rows=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   status=(path/'status').read_text();uid=int(next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1])
   if not 20000<=uid<20064:continue
   ignored=int(next(l for l in status.splitlines() if l.startswith('SigIgn:')).split()[1],16)
   rest=(path/'stat').read_text().rpartition(')')[2].split()
   rows.append(dict(pid=int(path.name),uid=uid,session=int(rest[3]),ignored_term=bool(ignored & (1<<14))))
  except (FileNotFoundError,ProcessLookupError):pass
 if len(rows)>=MINIMUM and all(r['ignored_term'] for r in rows) and (MINIMUM==1 or len({r['session'] for r in rows})>=2):break
 if time.monotonic()>end:raise AssertionError(('fault_child_tree_not_live',rows))
 time.sleep(.01)
print(json.dumps(rows))
""".replace('MINIMUM',str(minimum))
        rows=json.loads(audit(script).stdout)
        record.setdefault('live_tree_audits',[]).append(rows)
    def deny_receipt_writes():
        audit("import os;os.chmod('/receipts',0o500)")
    before=inventory();record['before']={k:dict(count=len(v),sha256=hashlib.sha256(json.dumps(sorted(v)).encode()).hexdigest()) for k,v in before.items()}
    worker=None
    try:
        base=run(['docker','image','inspect','speech-core-1a-tests:f1-f2','--format','{{.Id}}']).stdout.strip()
        assert base=='sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e'
        record['dependency_image']=base
        for target in ('worker','control'):
            run(['docker','build','--network=none','--memory','2g','-f','service/Dockerfile.lifecycle','--target',target,'-t',env['W2C_'+target.upper()+'_IMAGE'],'.'],180)
        record['images']={t:run(['docker','image','inspect',env['W2C_'+t.upper()+'_IMAGE'],'--format','{{.Id}}']).stdout.strip() for t in ('worker','control')}
        roots=['service','worker','contracts','clients','tests/service','tests/release']
        source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        for target in ('control','worker'):
            name=project+'-source-'+target;names.append(name)
            script="import pathlib,hashlib,json; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in "+repr(roots)+" for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}))"
            mapped=json.loads(run(['docker','run','--name',name,'--network','none','--entrypoint','python',env['W2C_'+target.upper()+'_IMAGE'],'-c',script]).stdout)
            assert all(source[p]==h for p,h in mapped.items())
            if target=='control':assert source==mapped
            else:assert not any(p.startswith('service/') for p in mapped)
            record[target+'_source_sha256']=mapped
        name=project+'-regressions';names.append(name)
        run(['docker','run','--name',name,'--network','none','--cpus','2','--memory','1g','--pids-limit','128','--read-only','--tmpfs','/tmp:rw,nosuid,size=64m','--entrypoint','python',env['W2C_CONTROL_IMAGE'],'-m','pytest','tests/service','tests/release','-q','-p','no:cacheprovider'],120)
        dc('up','-d','worker')
        worker=dc('ps','-aq','worker').stdout.strip()
        end=time.monotonic()+5
        while True:
            result=control('hello',check=False)
            if result.returncode==0:break
            if time.monotonic()>end:raise AssertionError('readiness_failed')
        topology=json.loads(run(['docker','inspect',worker]).stdout)[0]
        assert topology['Config']['Entrypoint']==['python','-m','worker.runtime_supervisor']
        host=topology['HostConfig']
        assert host['NetworkMode']=='none' and not host['PidMode'] and host['ReadonlyRootfs']
        assert set(host['CapAdd'])=={'CAP_SETUID','CAP_SETGID','CAP_KILL'}
        assert host['CapDrop'] in (['CAP_ALL'], ['ALL'])
        assert host['PidsLimit']==48 and host['Memory']==192*1024*1024 and host['NanoCpus']==1000000000
        assert host['RestartPolicy']['Name']=='no' and 'no-new-privileges:true' in host['SecurityOpt']
        assert {m['Destination'] for m in topology['Mounts']}=={'/ipc','/staging','/receipts'}
        assert all(m['Type']=='volume' for m in topology['Mounts'])
        name=project+'-namespace';names.append(name)
        script="import os,json,shutil;assert shutil.which('docker') is None;print(json.dumps({k:os.readlink('/proc/self/ns/'+k) for k in ('pid','net','mnt')}))"
        control_ns=json.loads(dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script).stdout)
        ci=json.loads(run(['docker','inspect',name]).stdout)[0]
        assert ci['Config']['User']=='10001:10001' and ci['HostConfig']['NetworkMode']=='none' and ci['HostConfig']['ReadonlyRootfs']
        assert all(m['Type']=='volume' for m in ci['Mounts'])
        assert {m['Destination'] for m in ci['Mounts']}=={'/ipc','/staging','/control'}
        assert ci['HostConfig']['CapDrop'] in (['ALL'],['CAP_ALL']) and not ci['HostConfig']['CapAdd']
        worker_ns=json.loads(audit(script).stdout)
        assert all(control_ns[k]!=worker_ns[k] for k in control_ns)
        record['namespaces']=dict(control=control_ns,worker=worker_ns)
        run(['docker','rm',name])
        record['topology']=dict(entrypoint=topology['Config']['Entrypoint'],user=topology['Config']['User'],network=host['NetworkMode'],pid_mode=host['PidMode'],caps=host['CapAdd'],mounts=[dict(destination=m['Destination'],type=m['Type'],rw=m['RW']) for m in topology['Mounts']])
        identities=[]
        for mode in ('silence','permissions'):
            dispatched=control('dispatch','--fixture',mode,'--seconds','8');identities.append(dispatched['runtime_id'])
            executed=control('execute')
            control('observe') # Same allocation reconnect; never starts a second child.
            wait_state('stopped-and-reaped')
            a=control('collect');assert a['proposal_sha256']==executed['proposal_sha256']
            # Retained original proposal is collectable after the execution cutoff.
            from datetime import datetime, timezone
            remaining=(datetime.fromisoformat(a['receipt']['original_deadline'])-datetime.now(timezone.utc)).total_seconds()
            if remaining>0:time.sleep(min(8,remaining+.05))
            b=control('collect')
            assert a==b and a['receipt']['timely'] and a['sha256']=='639dad0ac2923f5fe9e9ccfb99aa9b3084048e2e53317d1903e6e899a4f6296a'
            if mode == 'permissions':
                restart();assert control('collect')==a
            control('ack')
            record['scenarios'].append(dict(name='sequential_'+mode,result=a))
        assert len(set(identities))==2
        assert control('hello')['launches']==0
        name=project+'-old-start';names.append(name)
        script="from service.supervisor_runtime import SupervisorRuntime; r=SupervisorRuntime(); ids="+repr(identities)+"; assert all(r.start(i)['state']=='stopped-and-reaped' for i in ids); assert r.call('hello',{})['launches']==0; print('old_identity_fenced')"
        dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script)
        run(['docker','rm',name])
        audit("import os; p=[int(n) for n in os.listdir('/proc') if n.isdigit()];assert set(p)<={1,os.getpid()};print('namespace_execution_tree_empty')")
        # Authenticated wrong tuple, wrong session, oversized and partial frames.
        name=project+'-frames';names.append(name)
        script='''import socket,time,struct
from pathlib import Path
from worker.runtime_protocol import session,authenticate,send,receive,VERSION
from worker.transport import Session
key=Path('/ipc/lifecycle.key').read_text().strip()
for mode in ('wrong_session','wrong_tuple','oversized','incomplete'):
 s=socket.socket(socket.AF_UNIX);s.settimeout(3);s.connect('/ipc/lifecycle.sock');cutoff=time.monotonic()+2
 try:
  auth=Session(key,'wrong-domain',0,10001) if mode=='wrong_session' else session(key,0,10001)
  authenticate(s,auth,cutoff,control=True)
  s.settimeout(3)
  if mode=='wrong_tuple':
   send(s,dict(version=VERSION,op='start',body=dict(runtime_id='wrong/tuple/nonce')),cutoff)
   assert receive(s,cutoff)['ok'] is False
  elif mode=='oversized':
   s.sendall(struct.pack('!I',65537));assert not s.recv(1)
  elif mode=='incomplete':
   s.sendall(struct.pack('!I',20)+b'{');assert not s.recv(1)
  else:raise AssertionError('wrong_session_accepted')
 except (OSError,ValueError):
  assert mode=='wrong_session'
 finally:s.close()
print('bounded_closed_frame_refusals')
'''
        dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script)
        run(['docker','rm',name])
        record['scenarios'].append(dict(name='old_identity_and_closed_frames',passed=True))
        # Terminal proof survives a new PID1 boot; no fresh execution is involved.
        terminal=control('collect');restart()
        assert control('collect')==terminal
        record['scenarios'].append(dict(name='terminal_receipt_restart_replay',passed=True))
        for interrupt in ('postspawn','start-sent'):
            before_launch=control('hello')['launches']
            result=control('dispatch','--seconds','8','--interrupt',interrupt,check=False)
            assert result.returncode==17
            control('observe')
            wait_state('stopped-and-reaped')
            assert control('hello')['launches']==before_launch+1
            assert control('collect')['receipt']['timely']
            control('ack')
            record['scenarios'].append(dict(name='actual_control_death_'+interrupt,passed=True))
        for mode in ('busy','descendants'):
            control('dispatch','--fixture',mode,'--seconds','6')
            audit_busy_tree(3 if mode=='descendants' else 1)
            if mode=='busy':control('start') # Same bound start cannot fork twice.
            # No control process remains alive between these calls; worker owns expiry.
            time.sleep(6.3)
            wait_state('stopped-and-reaped')
            value=control('collect');assert value['receipt']['reaped_children'] >= (3 if mode=='descendants' else 1)
            assert not value['receipt']['timely'] and value['sha256'] is None
            control('ack')
            record['scenarios'].append(dict(name='absent_control_deadline_'+mode,result=value))
        control('dispatch','--fixture','descendants','--seconds','10')
        audit_busy_tree(3)
        # One blocked unauthenticated client cannot starve deadline/status processing.
        name=project+'-blocked';names.append(name)
        dc('run','-d','--name',name,'--no-deps','--entrypoint','python','control','-c',
            "import socket,time; s=socket.socket(socket.AF_UNIX);s.connect('/ipc/lifecycle.sock');time.sleep(3)")
        start=time.monotonic();control('hello');latency=time.monotonic()-start
        assert latency<5
        record['readiness_ceiling_seconds']=5
        record['readiness_observed_seconds']=latency
        run(['docker','wait',name]);run(['docker','rm',name])
        start=time.monotonic();assert control('stop')['state']=='stopped-and-reaped'
        assert time.monotonic()-start<5
        assert control('collect')['receipt']['reaped_children']>=3
        control('ack');record['scenarios'].append(dict(name='term_kill_detached_descendants',passed=True))
        control('dispatch','--fixture','busy','--seconds','10')
        lost=control('stop','--interrupt','stop-sent',check=False);assert lost.returncode==17
        wait_state('stopped-and-reaped');control('ack')
        record['scenarios'].append(dict(name='lost_stop_response',passed=True))
        control('dispatch','--fixture','busy','--seconds','12')
        old=control('observe')['runtime_id'];oldboot=control('hello')['boot']
        run(['docker','kill','--signal','KILL',worker]);run(['docker','wait',worker])
        run(['docker','start',worker]);time.sleep(.3)
        assert control('hello')['boot']!=oldboot
        assert control('observe')['state']=='unknown'
        assert control('dispatch')['runtime_id']==old
        assert control('hello')['launches']==0
        name=project+'-quarantine-admission';names.append(name)
        script="from service.supervisor_runtime import SupervisorRuntime;from service.lifecycle_cli import fixture_authority; r=SupervisorRuntime()\ntry:r.prepare(fixture_authority())\nexcept RuntimeError:print('fresh_admission_refused')\nelse:raise AssertionError('quarantine_bypassed')"
        dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script);run(['docker','rm',name])
        record['scenarios'].append(dict(name='container_restart_quarantine',passed=True))
        # Fresh owned diagnostic installs isolate destructive fault scenarios.
        reset()
        interrupted=control('dispatch','--seconds','20','--interrupt','prestart',check=False)
        assert interrupted.returncode==17 and control('hello')['launches']==0
        deny_receipt_writes()
        assert control('start',check=False).returncode!=0
        assert control('hello')['launches']==0 and control('observe')['state']=='unknown'
        record['scenarios'].append(dict(name='unwritable_start_fence_no_fork',passed=True))
        reset()
        control('dispatch','--fixture','busy','--seconds','20')
        deny_receipt_writes()
        assert control('stop')['state']=='unknown'
        assert control('ack',check=False).returncode!=0
        restart();assert control('observe')['state']=='unknown'
        assert control('hello')['launches']==0
        record['scenarios'].append(dict(name='unwritable_terminal_receipt_quarantine',passed=True))
        for corruption in ('missing','corrupt'):
            reset()
            control('dispatch','--fixture','busy','--seconds','20')
            if corruption=='missing':audit("import os;os.unlink('/receipts/journal.json')")
            else:audit("open('/receipts/journal.json','w').write('{}')")
            restart()
            assert control('hello',check=False).returncode!=0
            assert control('observe')['state']=='unknown'
            record['scenarios'].append(dict(name=corruption+'_journal_fail_closed',passed=True))
        record['status']='matrix_passed'
    except Exception as error:
        record['status']='failed';record['failure']=type(error).__name__+': '+str(error)[-1500:]
        raise
    finally:
        # Stop only this project's worker; inspect termination before any removal.
        if worker:
            run(['docker','logs','--tail','20',worker],check=False)
            run(['docker','stop','-t','3',worker],check=False)
            run(['docker','wait',worker],check=False)
        owned=dc('ps','-aq',check=False).stdout.split()
        for name in set(names+owned):
            result=run(['docker','inspect',name],check=False)
            if result.returncode:continue
            state=json.loads(result.stdout)[0]['State']
            assert not state['Running'] and state['Status'] in ('exited','dead','created')
            run(['docker','rm',name])
        dc('down','-v',check=False)
        after=inventory();record['after']={k:dict(count=len(v),sha256=hashlib.sha256(json.dumps(sorted(v)).encode()).hexdigest()) for k,v in after.items()}
        assert set(before['containers'])<=set(after['containers']) and set(before['volumes'])<=set(after['volumes'])
        assert not run(['docker','ps','-aq','--filter','label=com.docker.compose.project='+project]).stdout.strip()
        assert not run(['docker','volume','ls','-q','--filter','label=com.docker.compose.project='+project]).stdout.strip()
        record['prior_resources_preserved']=True
        record['owned_resources_remaining']=0
        record['packaging_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('scripts/test-worker-w2c.sh','service/compose.lifecycle.yml','service/Dockerfile.lifecycle','service/Dockerfile.lifecycle.dockerignore')}
        (out/'run.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
