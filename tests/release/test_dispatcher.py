"""Host-only W2d qualification; actual installed Core + PID1 round trips.

Docker is used exclusively by this test operator for installation/fault injection.
No observed host state is supplied to the dispatcher.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def installed(case):
    import threading
    from service.dispatcher import Dispatcher
    from service.supervisor_runtime import SupervisorRuntime
    def wait(fn,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.02)
        raise AssertionError('installed_timeout')
    marker=threading.Event()
    def fault(point):
        if case.startswith('crash-') and point==case[6:]:os._exit(17)
        if case in ('cancel-completion','stale','preverification') and point=='preverification' and not marker.is_set():
            marker.set();raise RuntimeError('deterministic_preverification_interruption')
    runtime=SupervisorRuntime(interrupt='start-sent' if case=='crash-start-sent' else 'none')
    if case=='stale':
        collect=runtime.collect
        def stale(a,i):
            value=collect(a,i)
            if value and value[1]:value[1].message['generation']+=1
            return value
        runtime.collect=stale
    if case=='lost-ack':
        call=runtime.call;lost=[False]
        def lost_ack(op,body,timeout=1.5):
            result=call(op,body,timeout)
            if op=='ack' and not lost[0]:lost[0]=True;raise OSError('lost_ack_response_after_supervisor_commit')
            return result
        runtime.call=lost_ack
    d=Dispatcher('/control/core',runtime=runtime,fault=fault)
    try:
        wait(lambda:d.call('status')['reconciled'])
        if case in ('recover','recover-failed','unknown','recover-terminal'):
            job=d.call('jobs','p')[0]['job']
            if case=='unknown':
                wait(lambda:d.call('status')['quarantined'])
                assert d.call('job',job['id'])['state'] in ('running','cancel_requested')
                assert d.call('submit','p','a')['id']==job['id']
                try:d.call('submit','p','must-refuse')
                except ValueError:pass
                else:raise AssertionError('unknown_admitted')
                assert not job['artifact_ids'] and not job['attempts'][-1]['reaped']
                assert Path('/staging',job['attempts'][-1]['staging_id']).is_dir()
                return dict(status=d.call('status'),job=job)
        elif case=='queue-cancel':
            first=d.call('submit','p','a',fixture='busy')
            wait(lambda:d.call('job',first['id'])['attempts'])
            job=d.call('submit','p','queued')
            assert d.call('cancel',job['id'])['state']=='cancelled'
            d.call('cancel',first['id'])
            wait(lambda:d.call('job',first['id'])['state']=='cancelled')
            return dict(job=d.call('job',job['id']),first=d.call('job',first['id']))
        else:
            fixture='descendants' if case in ('cancel','deadline','leave-running','leave-deadline') else 'silence'
            job=d.call('submit','p','a',fixture=fixture,seconds=3 if case in ('deadline','leave-deadline') else 8)
            if case in ('leave-running','leave-deadline'):
                wait(lambda:d.call('status')['phase']=='observing')
                # Leave actual descendants alive for PID1/container restart injection.
                time.sleep(.3);os._exit(17)
            if case=='cancel':
                wait(lambda:d.call('status')['phase']=='observing');time.sleep(.3)
                t=time.monotonic()
                d.call('job',job['id']);d.call('events',job['id']);d.call('cancel',job['id'])
                latency=time.monotonic()-t;assert latency<1
            if case=='cancel-completion':
                assert marker.wait(8)
                assert d.call('cancel',job['id'])['state']=='cancel_requested'
        result=wait(lambda:(j if (j:=d.call('job',job['id']))['state'] in ('succeeded','failed','cancelled') else None))
        wait(lambda:d.call('status')['phase']=='idle')
        expected='cancelled' if case=='cancel' else 'failed' if case in ('deadline','stale','recover-failed') else 'succeeded'
        assert result['state']==expected,(case,result)
        assert result['attempts'][-1]['reaped']
        if result['state']=='succeeded':
            artifact=d.call('artifact',result['artifact_ids'][0]);asset=d.call('asset',artifact['asset_ids'][0])
            assert asset['sha256']=='639dad0ac2923f5fe9e9ccfb99aa9b3084048e2e53317d1903e6e899a4f6296a'
        else:assert not result['artifact_ids']
        if case=='cancel':
            retry=d.call('submit','p','explicit-retry',retry_of=result['id'])
            assert retry['id']!=result['id'] and retry['input_sha256']==result['input_sha256']
            wait(lambda:d.call('job',retry['id'])['state']=='succeeded')
            wait(lambda:d.call('status')['phase']=='idle')
        if case=='sequential':
            second=d.call('submit','p','b',fixture='permissions')
            second=wait(lambda:(j if (j:=d.call('job',second['id']))['state']=='succeeded' else None))
            wait(lambda:d.call('status')['phase']=='idle')
            for k in ('id','owner_id','staging_id'):assert result['attempts'][0][k]!=second['attempts'][0][k]
            assert d.call('submit','p','a')['id']==result['id']
        if case=='deadline':
            old=result
            d.call('replace',old['id'])
            replaced=wait(lambda:(j if (j:=d.call('job',old['id']))['state']=='succeeded' else None))
            wait(lambda:d.call('status')['phase']=='idle')
            assert replaced['generation']==2 and replaced['input_sha256']==old['input_sha256']
            for k in ('id','owner_id','staging_id'):assert replaced['attempts'][0][k]!=replaced['attempts'][1][k]
            result=replaced
        return dict(job=result,status=d.call('status'),supervisor=runtime.call('hello',{}),cancel_latency=latency if case=='cancel' else None)
    finally:d.close()


def main():
    lane=open('/tmp/pab-worker-w1-docker-lane.lock','a')
    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
    project='pab-w2d-'+uuid.uuid4().hex[:10]
    out=Path('docs/agent-interface/phase-1/w2d-evidence');out.mkdir(exist_ok=True)
    record=dict(project=project,tier='E1/E2 fixed synthetic dependency overlay',commands=[],scenarios=[])
    env=dict(os.environ,W2D_CONTROL_IMAGE=project+'-control',W2D_WORKER_IMAGE=project+'-worker')
    compose=['docker','compose','-p',project,'-f','service/compose.dispatcher.yml']
    names=[];worker=None
    def save(): (out/'run.json').write_text(json.dumps(record,indent=2))
    def run(args,timeout=40,check=True):
        result=subprocess.run(args,env=env,text=True,capture_output=True,timeout=timeout)
        record['commands'].append(dict(argv=args,exit=result.returncode,output=(result.stdout+result.stderr)[-6000:]))
        save()
        if check and result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result
    def dc(*args,**kw):return run(compose+list(args),**kw)
    def inventory():return dict(containers=run(['docker','ps','-aq']).stdout.split(),volumes=run(['docker','volume','ls','-q']).stdout.split())
    def digest(values):return dict(count=len(values),sha256=hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest())
    def control(case,expected=0):
        name=project+'-c-'+uuid.uuid4().hex[:8];names.append(name)
        script="import json;from tests.release.test_dispatcher import installed; print(json.dumps(installed("+repr(case)+")))"
        result=dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script,check=False)
        assert result.returncode==expected,(case,result.returncode,result.stdout,result.stderr)
        run(['docker','rm',name])
        value=json.loads(result.stdout.strip().splitlines()[-1]) if expected==0 else dict(exit=expected)
        record['scenarios'].append(dict(case=case,result=value));save();return value
    def audit(script):return run(['docker','exec',worker,'python','-c',script])
    def reset():
        nonlocal worker
        if worker:
            run(['docker','stop','-t','3',worker],check=False);run(['docker','wait',worker],check=False)
        # All control commands above have exited; never remove running resources.
        for cid in dc('ps','-aq',check=False).stdout.split():
            assert not json.loads(run(['docker','inspect',cid]).stdout)[0]['State']['Running']
        dc('down','-v');dc('up','-d','worker');worker=dc('ps','-aq','worker').stdout.strip()
        time.sleep(.3)
    before=inventory();record['before']={k:digest(v) for k,v in before.items()}
    try:
        base=run(['docker','image','inspect','speech-core-1a-tests:f1-f2','--format','{{.Id}}']).stdout.strip()
        assert base=='sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e'
        record['dependency_image']=base
        for target in ('worker','control'):
            run(['docker','build','--network=none','--memory','2g','-f','service/Dockerfile.dispatcher','--target',target,'-t',env['W2D_'+target.upper()+'_IMAGE'],'.'],180)
        record['images']={t:run(['docker','image','inspect',env['W2D_'+t.upper()+'_IMAGE'],'--format','{{.Id}}']).stdout.strip() for t in ('worker','control')}
        roots=['service','worker','contracts','clients','tests/service','tests/release']
        source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        for target in ('worker','control'):
            name=project+'-source-'+target;names.append(name)
            script="import pathlib,hashlib,json;print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in "+repr(roots)+" for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}))"
            mapped=json.loads(run(['docker','run','--name',name,'--network','none','--entrypoint','python',env['W2D_'+target.upper()+'_IMAGE'],'-c',script]).stdout)
            assert all(source[p]==h for p,h in mapped.items())
            if target=='control':assert source==mapped
            else:assert not any(p.startswith('service/') for p in mapped)
            record[target+'_source_sha256']=mapped
        record['packaging_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('service/Dockerfile.dispatcher','service/Dockerfile.dispatcher.dockerignore','service/compose.dispatcher.yml','scripts/test-worker-w2d.sh'))}
        name=project+'-tests';names.append(name)
        run(['docker','run','--name',name,'--network','none','--cpus','2','--memory','1g','--pids-limit','128','--read-only','--tmpfs','/tmp:rw,nosuid,size=64m','--entrypoint','python',env['W2D_CONTROL_IMAGE'],'-m','pytest','tests/service','tests/release','-q','-p','no:cacheprovider'],180)
        reset()
        topology=json.loads(run(['docker','inspect',worker]).stdout)[0]
        assert topology['Config']['Entrypoint']==['python','-m','worker.runtime_supervisor']
        host=topology['HostConfig']
        assert host['NetworkMode']=='none' and not host['PidMode'] and host['ReadonlyRootfs']
        assert set(host['CapAdd'])=={'CAP_SETUID','CAP_SETGID','CAP_KILL'}
        assert {m['Destination'] for m in topology['Mounts']}=={'/ipc','/staging','/receipts'}
        assert all(m['Type']=='volume' for m in topology['Mounts'])
        record['topology']=dict(entrypoint=topology['Config']['Entrypoint'],network=host['NetworkMode'],mounts=[dict(destination=m['Destination'],rw=m['RW']) for m in topology['Mounts']])
        for case in ('sequential','cancel','queue-cancel','cancel-completion','deadline','stale','lost-ack','preverification'):
            reset();control(case)
        for point in ('postspawn','start-sent','after_collect','preverification','before_adoption','after_adoption','after_commit','durable_outcome','before_ack','after_ack'):
            reset();control('crash-'+point,17)
            if point=='after_commit':
                run(['docker','kill','--signal','KILL',worker]);run(['docker','wait',worker]);run(['docker','start',worker]);time.sleep(.3)
            # Original timely receipt/proposal is collected by a fresh real control.
            recovered=control('recover-failed' if point=='before_adoption' else 'recover')
            assert recovered['supervisor']['launches']==(0 if point=='after_commit' else 1)
            rows=json.loads(audit("import json;from pathlib import Path;print(json.dumps(json.loads(Path('/receipts/journal.json').read_text())['payload']['records']))").stdout)
            assert len(rows)==1 and rows[0]['phase']=='terminal' and rows[0]['acked']
            record['scenarios'][-1]['physical_allocations']=len(rows)
        reset();control('leave-deadline',17);time.sleep(3.3);control('recover-failed')
        reset();control('leave-running',17)
        run(['docker','kill','--signal','KILL',worker]);run(['docker','wait',worker]);run(['docker','start',worker]);time.sleep(.3)
        control('unknown')
        record['passed']=True
    except BaseException as exc:
        record['passed']=False;record['failure']=str(exc)
        previous=list(out.glob('failure-*.json'))
        (out/('failure-'+str(len(previous)+1)+'.json')).write_text(json.dumps(record,indent=2))
        raise
    finally:
        if worker:
            run(['docker','stop','-t','3',worker],check=False);run(['docker','wait',worker],check=False)
        for name in names:
            info=run(['docker','inspect',name],check=False)
            if info.returncode==0:
                assert not json.loads(info.stdout)[0]['State']['Running']
                run(['docker','rm',name],check=False)
        for cid in dc('ps','-aq',check=False).stdout.split():
            assert not json.loads(run(['docker','inspect',cid]).stdout)[0]['State']['Running']
        dc('down','-v',check=False)
        after=inventory()
        assert all(set(before[k])<=set(after[k]) for k in before)
        record['prior_preserved']={k:digest(before[k]) for k in before}
        record['remaining_owned_containers']=run(['docker','ps','-aq','--filter','label=com.docker.compose.project='+project]).stdout.split()
        record['remaining_owned_volumes']=run(['docker','volume','ls','-q','--filter','label=com.docker.compose.project='+project]).stdout.split()
        assert not record['remaining_owned_containers'] and not record['remaining_owned_volumes']
        save()
    print(json.dumps(dict(passed=record['passed'],scenarios=len(record['scenarios']),project=project)))


if __name__=='__main__':main()
