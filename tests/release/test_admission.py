"""R1 exact-source E1/E2 packaged synthetic admission; host harness only."""
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
    from tests.release.test_dispatcher import installed as lifecycle
    if case.startswith('authority-'):return authority_installed(case)
    if not case.startswith('resource-'):
        result=lifecycle(case)
        if case=='unknown':assert result['status']['reservations'][0]['state']=='executing'
        if case in ('recover','recover-failed','sequential','cancel'):
            assert all(r['state']=='retained' for r in result['status']['reservations'])
        return result
    from service.dispatcher import Dispatcher
    from service.admission import DEFAULT_BUDGET
    def wait(fn):
        end=time.monotonic()+12
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.02)
        raise AssertionError('resource_installed_timeout')
    if case=='resource-transient':
        import threading
        from service.supervisor_runtime import SupervisorRuntime
        class DelayedRuntime(SupervisorRuntime):
            def __init__(self):
                super().__init__();self.entered=threading.Event();self.release=threading.Event();self.stops=0
            def observe(self, identity, timeout):
                if not self.entered.is_set():
                    self.entered.set();assert self.release.wait(2)
                return super().observe(identity,timeout)
            def terminate(self, identity, timeout):
                self.stops+=1
                return super().terminate(identity,timeout)
        runtime=DelayedRuntime();pressure=[]
        def fault(point):
            if point=='prespawn':
                original=d.admission.check
                def check(**kwargs):
                    if runtime.entered.is_set() and not pressure:
                        pressure.append(True);runtime.release.set()
                        raise ValueError('resource_stage_growth')
                    return original(**kwargs)
                d.admission.check=check
        d=Dispatcher('/control/core',runtime=runtime,fault=fault)
        try:
            wait(lambda:d.call('status')['reconciled'])
            j=d.call('submit','p','a',fixture='busy',seconds=8)
            wait(lambda:d.call('status')['phase']=='idle' and runtime.stops==1)
            job=d.call('job',j['id']);status=d.call('status')
            assert job['state']=='failed' and job['attempts'][-1]['reaped'] and not job['artifact_ids']
            assert status['quarantined'] and status['reservations'][0]['state']=='retained'
            time.sleep(.2)
            assert d.call('status')['quarantined']
            try:d.call('submit','p','fresh')
            except ValueError:pass
            else:raise AssertionError('fault_latch_lost')
            supervisor=runtime.call('hello',{})
            assert supervisor['launches']==1
            return dict(status=status,job=job,stops=runtime.stops,supervisor=supervisor,
                        pressure='one-shot E1 policy injection during real UDS observe; no physical disk pressure')
        finally:runtime.release.set();d.close()
    budget=dict(DEFAULT_BUDGET)
    if case=='resource-ram':budget['ram_bytes']-=1
    if case=='resource-disk':budget['disk_bytes']=1024**2
    if case=='resource-queue':budget['queue']=1
    d=Dispatcher('/control/core',budget=budget)
    try:
        wait(lambda:d.call('status')['reconciled'])
        if case in ('resource-ram','resource-disk'):
            try:d.call('submit','p','refused')
            except ValueError:pass
            else:raise AssertionError('resource_refusal_missing')
            assert d.call('jobs','p')==[] and d.call('status')['reservations']==[]
            return dict(status=d.call('status'))
        j=d.call('submit','p','a',fixture='busy')
        wait(lambda:d.call('status')['phase']=='observing')
        if case=='resource-queue':
            try:d.call('submit','p','full')
            except ValueError:pass
            else:raise AssertionError('queue_refusal_missing')
            assert d.call('submit','p','a')['id']==j['id']
        before=d.call('status')['reservations'];assert before[0]['state']=='executing'
        d.close();d=Dispatcher('/control/core',budget=budget)
        wait(lambda:d.call('status')['reconciled'])
        assert d.call('status')['reservations']==before
        assert d.call('submit','p','a')['id']==j['id']
        d.call('cancel',j['id'])
        wait(lambda:d.call('job',j['id'])['state']=='cancelled')
        wait(lambda:d.call('status')['phase']=='idle')
        after=d.call('status')['reservations'];assert after[0]['state']=='retained'
        assert after[0]['demand']['disk_bytes']==before[0]['demand']['disk_bytes']
        second=d.call('submit','p','b')
        wait(lambda:d.call('job',second['id'])['state']=='succeeded')
        wait(lambda:d.call('status')['phase']=='idle')
        return dict(before=before,after=after,status=d.call('status'))
    finally:d.close()


def authority_installed(case):
    """E1 owner-connection policy injection; real E2 supervisor stop/reap."""
    import sqlite3
    import threading
    from service.dispatcher import Dispatcher
    from service.supervisor_runtime import SupervisorRuntime
    def wait(fn):
        end=time.monotonic()+12
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.02)
        raise AssertionError('authority_installed_timeout')
    if case=='authority-recover':
        db=sqlite3.connect('/control/core/core.sqlite')
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
        job=json.loads(db.execute('SELECT job FROM jobs').fetchone()[0]);db.close()
        output=Path('/staging')/job['attempts'][-1]['staging_id']/'output.wav'
        assert not output.exists() or output.stat().st_size!=10000
        class ReconnectRuntime(SupervisorRuntime):
            def __init__(self):super().__init__();self.stops=0
            def terminate(self,identity,timeout):
                self.stops+=1
                return super().terminate(identity,timeout)
        runtime=ReconnectRuntime();d=Dispatcher('/control/core',runtime=runtime)
        try:
            wait(lambda:d.call('status')['reconciled'])
            wait(lambda:d.call('job',job['id'])['state']=='failed')
            status=d.call('status');result=d.call('job',job['id'])
            assert status['persistence_fault'] and status['quarantined']
            assert status['reservations'][0]['state']=='executing'
            assert result['attempts'][-1]['reaped'] and not result['artifact_ids']
            assert d.call('submit','p','a')['id']==job['id']
            for op,args in [('submit',('p','fresh')),('replace',(job['id'],))]:
                try:d.call(op,*args)
                except ValueError:pass
                else:raise AssertionError('authority_progress')
            assert runtime.call('hello',{})['launches']==1
            db=sqlite3.connect('/control/core/core.sqlite')
            row=db.execute('SELECT outcome,acked FROM dispatches').fetchone()
            assert row==(None,0)
            receipt=json.loads(db.execute('SELECT receipt FROM dispatches').fetchone()[0])
            assert receipt['reaped'] is True and receipt['reap_method']=='pid1-waitpid-echild'
            assert receipt['original_deadline']==result['deadline']
            assert runtime.stops==(0 if job['state']=='failed' else 1)
            assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',)
            db.close()
            return dict(status=status,job=result,authority='checking',ack=row[1],stops=runtime.stops,receipt=receipt,lifecycle='real supervisor exact reap; charge retained')
        finally:d.close()
    if case=='authority-setup':
        d=Dispatcher('/control/core')
        try:
            wait(lambda:d.call('status')['reconciled'])
            job=d.call('submit','p','a',fixture='busy',seconds=25)
            wait(lambda:d.call('status')['phase']=='observing')
            job=d.call('job',job['id'])
        finally:d.close()
        db=sqlite3.connect('/control/core/core.sqlite')
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('healthy',);db.close()
        return dict(job=job,barrier='clean shutdown; operator may inject owned staging bytes')
    assert case=='authority-denied'
    from service.admission import Admission
    from service.resource_store import ResourceStore
    check=Admission.check;observe=ResourceStore.observe;denied=[]
    def violation(admission,**kwargs):
        db=sqlite3.connect('/control/core/core.sqlite')
        assert db.execute('SELECT state FROM resource_monitor').fetchone()==('checking',);db.close()
        try:check(admission,**kwargs)
        except ValueError as exc:
            assert str(exc)=='resource_output_bytes'
            admission.core.db.execute('PRAGMA query_only=ON');denied.append(True)
            admission.core.db.set_trace_callback(lambda sql: denied.append(sql) if sql.startswith("UPDATE resource_monitor SET state='fault'") else None)
            raise
        raise AssertionError('missing_real_10000_byte_violation')
    Admission.check=violation
    def fenced(store,fn):
        try:return observe(store,fn)
        except sqlite3.OperationalError:
            assert len(denied)==2 and denied[1].startswith("UPDATE resource_monitor SET state='fault'") and store.state()=='checking'
            print(json.dumps(dict(crash_fence='checking',fault='E1 query_only after actual operator-injected 10000-byte output observation')),flush=True)
            os._exit(29)  # No loop, close, finally, or retry after denied UPDATE.
    ResourceStore.observe=fenced
    d=Dispatcher('/control/core')
    d.thread.join(10)
    raise AssertionError(('expected_immediate_death',d.call('status'),denied))


def main():
    lane=open('/tmp/pab-worker-w1-docker-lane.lock','a')
    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
    project='pab-r1-'+uuid.uuid4().hex[:10]
    out=Path('docs/agent-interface/phase-1/r1-evidence');out.mkdir(parents=True, exist_ok=True)
    record=dict(project=project,tier='E1/E2 fixed synthetic dependency overlay',commands=[],scenarios=[])
    env=dict(os.environ,R1_CONTROL_IMAGE=project+'-control',R1_WORKER_IMAGE=project+'-worker')
    compose=['docker','compose','-p',project,'-f','service/compose.admission.yml']
    names=[];worker=None
    prefix=os.environ.get('R1_EVIDENCE_PREFIX','')
    if prefix not in ('','correction-','authority-'):raise ValueError('closed_evidence_prefix')
    def save(): (out/(prefix+'run.json')).write_text(json.dumps(record,indent=2))
    def run(args,timeout=40,check=True):
        result=subprocess.run(args,env=env,text=True,capture_output=True,timeout=timeout)
        output=(result.stdout+result.stderr)[-6000:]
        if args in (['docker','ps','-aq'],['docker','volume','ls','-q']):
            output=json.dumps(dict(count=len(result.stdout.split()),sha256=hashlib.sha256(json.dumps(sorted(result.stdout.split())).encode()).hexdigest()))
        record['commands'].append(dict(argv=args,exit=result.returncode,output=output))
        save()
        if check and result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result
    def dc(*args,**kw):return run(compose+list(args),**kw)
    def inventory():return dict(containers=run(['docker','ps','-aq']).stdout.split(),volumes=run(['docker','volume','ls','-q']).stdout.split())
    def digest(values):return dict(count=len(values),sha256=hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest())
    def control(case,expected=0):
        name=project+'-c-'+uuid.uuid4().hex[:8];names.append(name)
        script="import json;from tests.release.test_admission import installed; print(json.dumps(installed("+repr(case)+")))"
        result=dc('run','--name',name,'--no-deps','--entrypoint','python','control','-c',script,check=False)
        assert result.returncode==expected,(case,result.returncode,result.stdout,result.stderr)
        run(['docker','rm',name])
        value=json.loads(result.stdout.strip().splitlines()[-1]) if expected==0 else dict(exit=expected,barrier=result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None)
        if case in ('crash-start-sent','crash-start-overlap'):
            assert json.loads(value['barrier'])['crash_fence']==('healthy' if case=='crash-start-sent' else 'checking')
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
            run(['docker','build','--network=none','--memory','2g','-f','service/Dockerfile.admission','--target',target,'-t',env['R1_'+target.upper()+'_IMAGE'],'.'],180)
        record['images']={t:run(['docker','image','inspect',env['R1_'+t.upper()+'_IMAGE'],'--format','{{.Id}}']).stdout.strip() for t in ('worker','control')}
        roots=['service','worker','contracts','clients','tests/service','tests/release']
        source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        for target in ('worker','control'):
            name=project+'-source-'+target;names.append(name)
            script="import pathlib,hashlib,json;print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in "+repr(roots)+" for p in pathlib.Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}))"
            mapped=json.loads(run(['docker','run','--name',name,'--network','none','--entrypoint','python',env['R1_'+target.upper()+'_IMAGE'],'-c',script]).stdout)
            assert all(source[p]==h for p,h in mapped.items())
            if target=='control':assert source==mapped
            else:assert not any(p.startswith('service/') for p in mapped)
            record[target+'_source_sha256']=mapped
        record['packaging_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('service/Dockerfile.admission','service/Dockerfile.admission.dockerignore','service/compose.admission.yml','scripts/test-resource-r1.sh'))}
        name=project+'-tests';names.append(name)
        run(['docker','run','--name',name,'--network','none','--cpus','2','--memory','1g','--pids-limit','128','--read-only','--tmpfs','/tmp:rw,nosuid,size=64m','--entrypoint','python',env['R1_CONTROL_IMAGE'],'-m','pytest','tests/service','tests/release','-q','-p','no:cacheprovider'],180)
        reset()
        topology=json.loads(run(['docker','inspect',worker]).stdout)[0]
        assert topology['Config']['Entrypoint']==['python','-m','worker.runtime_supervisor']
        host=topology['HostConfig']
        assert host['NetworkMode']=='none' and not host['PidMode'] and host['ReadonlyRootfs']
        assert set(host['CapAdd'])=={'CAP_SETUID','CAP_SETGID','CAP_KILL'}
        assert {m['Destination'] for m in topology['Mounts']}=={'/ipc','/staging','/receipts'}
        assert all(m['Type']=='volume' for m in topology['Mounts'])
        record['topology']=dict(entrypoint=topology['Config']['Entrypoint'],network=host['NetworkMode'],mounts=[dict(destination=m['Destination'],rw=m['RW']) for m in topology['Mounts']])
        for case in ('resource-ram','resource-disk','resource-queue','resource-transient','sequential','cancel','queue-cancel','cancel-completion','deadline','stale','lost-ack','preverification'):
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
        reset();control('crash-start-overlap',17);control('authority-recover');control('authority-recover')
        reset();setup=control('authority-setup')
        pressure='/staging/'+setup['job']['attempts'][-1]['staging_id']+'/output.wav'
        run(['docker','exec','--user','10001:10001',worker,'python','-c',"from pathlib import Path;p=Path("+repr(pressure)+");assert not p.exists();p.write_bytes(b'x'*10000)"])
        control('authority-denied',29)
        # Restore the test pressure before either new control opens the same root.
        run(['docker','exec','--user','10001:10001',worker,'python','-c',"from pathlib import Path;p=Path("+repr(pressure)+");assert p.stat().st_size==10000;p.unlink()"])
        control('authority-recover');control('authority-recover')
        reset();control('leave-deadline',17);time.sleep(3.3);control('recover-failed')
        reset();control('leave-running',17)
        run(['docker','kill','--signal','KILL',worker]);run(['docker','wait',worker]);run(['docker','start',worker]);time.sleep(.3)
        control('unknown')
        record['passed']=True
    except BaseException as exc:
        record['passed']=False;record['failure']=str(exc)
        previous=list(out.glob(prefix+'failure-*.json'))
        (out/(prefix+'failure-'+str(len(previous)+1)+'.json')).write_text(json.dumps(record,indent=2))
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
