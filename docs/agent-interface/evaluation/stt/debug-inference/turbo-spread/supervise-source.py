"""Host-owned process group watchdog; stops/reaps only an explicit named container."""
import argparse, datetime, json, os, signal, subprocess, time, threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--out',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
out=Path(a.out);out.mkdir(exist_ok=False)
(out/'supervise-source.py').write_text(Path(__file__).read_text())
diag=Path(__file__).with_name('diagnose.py')
if diag.exists(): (out/'diagnose-source.py').write_text(diag.read_text())
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not a.name.startswith('pab-stt-debug-'): raise SystemExit('Refuse non-task name')
if subprocess.run(['docker','inspect',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise SystemExit('Container exists')
t=time.monotonic();timed_out=False
sample_done=threading.Event()
def host_samples():
    with (out/'host-load.log').open('w') as f:
        while not sample_done.is_set():
            f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
            for sample_cmd in [['ps','-Ao','pid,pcpu,comm','-r'],['pmset','-g','therm']]:
                r=subprocess.run(sample_cmd,capture_output=True,text=True,timeout=5)
                f.write(' '.join(sample_cmd)+'\n'+'\n'.join(r.stdout.splitlines()[:12])+'\n')
            f.flush()
            sample_done.wait(15)
threading.Thread(target=host_samples,daemon=True).start()
with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
    try: code=proc.wait(timeout=a.seconds)
    except subprocess.TimeoutExpired:
        timed_out=True
        subprocess.run(['docker','kill','--signal','TERM',a.name],capture_output=True,timeout=10)
        try: code=proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','kill',a.name],capture_output=True,timeout=10)
            try: code=proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                except PermissionError: proc.terminate()
                try: code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    except PermissionError: proc.kill()
                    code=proc.wait(timeout=2)
    finally:
        inspect=subprocess.run(['docker','inspect',a.name],capture_output=True,text=True,timeout=10)
        if inspect.returncode==0:
            state=json.loads(inspect.stdout)[0]
            (out/'container.json').write_text(json.dumps({'Id':state['Id'],'State':state['State'],'Image':state['Image'],'HostConfig':{k:state['HostConfig'][k] for k in ['NanoCpus','Memory','PidsLimit','NetworkMode']}},indent=2))
            if state['State']['Running']:
                subprocess.run(['docker','kill',a.name],capture_output=True,timeout=10)
                subprocess.run(['docker','wait',a.name],capture_output=True,timeout=10)
                terminal=subprocess.run(['docker','inspect',a.name],capture_output=True,text=True,timeout=10,check=True)
                state=json.loads(terminal.stdout)[0]
                (out/'container-terminal.json').write_text(json.dumps({'Id':state['Id'],'State':state['State'],'Image':state['Image']},indent=2))
                if state['State']['Running']: raise RuntimeError('Refusing removal without terminal receipt')
            removal=subprocess.run(['docker','rm',a.name],capture_output=True,text=True,timeout=10,check=True)
            (out/'removed.json').write_text(json.dumps({'removed':True,'name':a.name,'container_id':state['Id'],'had_terminal_receipt':not state['State']['Running']},indent=2))
        sample_done.set()
        (out/'supervisor.json').write_text(json.dumps({'command':cmd,'deadline_seconds':a.seconds,'timed_out':timed_out,'client_exit':proc.returncode,'client_reaped':proc.returncode is not None,'elapsed_seconds':time.monotonic()-t,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
raise SystemExit(code)
