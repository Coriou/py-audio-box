"""Actual installed factory/doctor/Core DDL; tiny inert model and host observations.

No Prepared.__new__, mocked DDL/create_app, weights, external network or hardware
qualification. Fresh isolated subprocess imports only the extracted installation.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

PROBE=r'''
import asyncio,contextlib,copy,hashlib,io,json,os,sqlite3,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
root=Path(sys.argv[1]);mode=sys.argv[2];source=root/'source/native-worker'
sys.path.insert(0,str(source));os.chdir(source)
from service import native_cpu_install as install,native_cpu_setup as setup,native_cpu_activation as activation
from service.native_cpu_configuration import VOICE,TEXT
from worker.native_cpu_supervisor import NativeCPUSupervisor
from worker.hardware import Host
from worker.transport import canonical,Refused
from worker import native_cpu_protocol as p
from contracts import validate as oracle
expected=json.loads((root/'fixture-descriptor.json').read_bytes());pins=install.locked_versions(source)
if sys.platform=='darwin':
    from importlib.metadata import version
    assert len(pins)==44 and all(version(k)==v for k,v in pins.items())
boot='00000000-0000-0000-0000-000000000001';host=Host('darwin','arm64','native')
owners=[];handles=[];fences=[];validations=[];mounts=[];signals=[]
original_init=NativeCPUSupervisor.__init__
class Sampler:
    def __init__(self,pid):self.pid=pid;self.thread=SimpleNamespace(is_alive=lambda:False)
    def start(self):pass
    def request_stop(self):pass
    def stop(self):return True
    def snapshot(self):return dict(count=1,peak=100,error=None,last=dict(pid=self.pid,identity=4,rss_bytes=100,method='proc_pid_rusage_v0',observed_at=time.monotonic()))
def supervisor_init(owner,*args,**kwargs):
    def popen(*args,**kwargs):
        class Handle:
            pid=10000+len(handles)
            code=None
            def poll(self):
                if owner.active and owner.active.get('closed'):self.code=0
                return self.code
            def wait(self,timeout):return self.poll()
        value=Handle();handles.append(value);return value
    kwargs.update(popen=popen,identity=lambda handle:'4',signal_group=lambda *v:signals.append(v),
        free_ram=lambda:16*1024**3,free_bytes=lambda:100*1024**3,sampler_factory=Sampler)
    original_init(owner,*args,**kwargs)

def exchange(owner,connection,config,cutoff,on_loaded):
    # Actual sealed tiny bundle verification, existing encoder and proposal.
    from lib.tts_backends.contract import AudioResult
    from worker.engines.kokoro import execute
    from worker.native_child import proposal
    from lib.spoken_text import speech_text
    import numpy as np
    on_loaded(min(config['cutoff'],time.monotonic()+30))
    spec=p.native_execution(config['snapshot'][1]);backend=Mock()
    backend.synthesise.return_value=AudioResult(np.asarray([.1,-.1],dtype='float32'),24000)
    result=execute(spec.model,speech_text(config['snapshot'][1]['plan']),config['bundles'],backend_factory=lambda *a,**kw:backend)
    return proposal(config['snapshot'],result,spec,protocol=p),result['data']

original_fence=activation.NativeCPUActivation.retained_check
original_history=activation.completed_history
def completed(*args,**kwargs):
    value=original_history(*args,**kwargs);validations.append(True);return value
def fence(owner,core):
    fresh=core.fresh_initialized
    if fresh:
        assert core.db.execute("SELECT name FROM sqlite_master WHERE name='jobs'").fetchall()==[]
    count=len(validations)
    original_fence(owner,core)
    if owner.prepared.selftest:
        assert type(owner.prepared.selftest) is bool
        if fresh:assert owner.session.prior_jobs==frozenset()
        else:assert len(validations)==count+1
    fences.append(dict(fresh=fresh,selftest=owner.prepared.selftest,prior=None if owner.session is None else sorted(owner.session.prior_jobs)))

from service.api import create_voice_app
original_api=create_voice_app
def mount(owner,**kwargs):
    prepared=owner.installed_activation.prepared;owners.append(prepared)
    jobs=owner.call('jobs','local');mounts.append(dict(fresh=owner.core.fresh_initialized,selftest=prepared.selftest,jobs=[j['job']['id'] for j in jobs],models=len(handles)))
    if owner.core.fresh_initialized:assert jobs==[] and handles==[] and prepared.selftest is True
    return original_api(owner,**kwargs)

async def call(client,name,payload,key=None):
    from urllib.parse import quote
    from service.catalogue import OPERATIONS
    op=OPERATIONS[name];oracle.validate(op['request'],payload);route=op['http']['route'];query=dict(payload)
    for field in tuple(query):
        marker='{'+field+'}'
        if marker in route:route=route.replace(marker,quote(str(query.pop(field)),safe=''))
    response=await asyncio.wait_for(client.post(route,content=canonical(payload),headers={'Content-Type':'application/json','Idempotency-Key':key}) if op['http']['method']=='POST' else client.get(route,params=query),5)
    return response

async def generate(client,key):
    voice=(await call(client,'voice_get',dict(voice_id=VOICE))).json()
    version=(await call(client,'voice_version_get',dict(voice_version_id=voice['version_ids'][-1]))).json()
    request=dict(voice_version_id=version['id'],text=TEXT,identity=version['identity'],fallback='forbid',variants=1,reuse='never')
    response=await call(client,'speech_generate',request,key)
    if response.status_code!=200:return response,None
    submitted=response.json();cursor='';deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        response=await call(client,'job_wait',dict(job_id=submitted['job_id'],cursor=cursor,wait_seconds=1))
        assert response.status_code==200,response.text
        result=response.json();cursor=result['next_cursor'];job=result['job']
        if job['state'] in ('succeeded','failed','cancelled'):
            assert job['state']=='succeeded',job
            return response,job
    raise AssertionError('source fixture job deadline')

async def app_jobs(*,selftest,count):
    import httpx
    from worker.native_cpu_qualification import current
    app=setup.create_app(root,_selftest=selftest)
    async with app.router.lifespan_context(app):
        owner=app.state.dispatcher;prepared=owner.installed_activation.prepared
        assert type(prepared) is install.Prepared and prepared.selftest is selftest
        assert current() is None
        if not selftest:
            assert prepared.qualification is None and owner.installed_activation.session is None and prepared.supervisor.qualification is None
        token=json.loads((root/'credential.json').read_bytes())['credentials'][0]['token']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://source-fixture',headers={'Authorization':'Bearer '+token},trust_env=False) as client:
            status=(await client.get('/v1/status')).json();assert status['ready_for_speech'] is (not selftest),status
            before=owner.call('jobs','local')
            if selftest:assert prepared.qualification.prior_jobs==frozenset(j['job']['id'] for j in before)
            for i in range(count):
                response,job=await generate(client,('selftest-cap-' if selftest else 'ordinary-')+str(i))
                assert job is not None and response.status_code==200,response.text
            if selftest and count==6:
                response,job=await generate(client,'seventh-new-refused')
                assert response.status_code!=200 and job is None
                assert 'native CPU retained-history capacity' in response.json()['message'],response.text
                assert len(owner.call('jobs','local'))==len(before)+6
            end=time.monotonic()+5
            while owner.call('status')['phase']!='idle':
                assert time.monotonic()<end;await asyncio.sleep(.01)
    assert app.state.installed_lifecycle['closed'] and prepared.closed and not prepared.thread.is_alive() and prepared.supervisor.active is None

original_version=install.metadata.version
with contextlib.ExitStack() as stack:
    for name,value in (
        ('lib.tts_backends.kokoro_native.validate_host',lambda:None),
        ('lib.tts_backends.kokoro_native.validate_versions',lambda:None),
        ('worker.hardware.host_metadata',lambda mode:host),
        ('worker.native_cpu_availability.host_metadata',lambda mode:host),
        ('worker.native_cpu_qualification.boot_identity',lambda:boot),
        ('service.native_cpu_install.expected',lambda:copy.deepcopy(expected)),
        ('service.native_cpu_install.metadata.version',lambda name:pins[name] if name in pins else original_version(name)),
        ('worker.native_cpu_supervisor.NativeCPUSupervisor.__init__',supervisor_init),
        ('worker.native_cpu_supervisor.NativeCPUSupervisor.exchange',exchange),
        ('service.native_cpu_activation.NativeCPUActivation.retained_check',fence),
        ('service.native_cpu_activation.completed_history',completed),
        ('service.api.create_voice_app',mount)):
        stack.enter_context(patch(name,new=value))
    if mode in ('fresh-command','retained-caps'):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):assert setup.main(['doctor','--root',str(root),'--selftest']) is None
        value=json.loads(output.getvalue());assert value['ready_for_speech'] and value['history_jobs_used']==1 and len(handles)==1
        assert mounts[0]==dict(fresh=True,selftest=True,jobs=[],models=0)
        if mode=='fresh-command':
            asyncio.run(app_jobs(selftest=False,count=1));assert len(handles)==2
        else:
            # One retained job + six NEW jobs fits the unchanged128MiB ledger.
            # Two retained + six hit its independent disk guard before slot6.
            asyncio.run(app_jobs(selftest=True,count=6));assert len(handles)==7
            assert any(not f['fresh'] and f['selftest'] and len(f['prior'])==1 for f in fences)
        assert not signals and fences[0]==dict(fresh=True,selftest=True,prior=[])
    elif mode=='retained-missing-table':
        asyncio.run(app_jobs(selftest=True,count=0));assert not handles
        dbpath=root/'state/control/core.sqlite'
        with sqlite3.connect(dbpath) as db:db.execute('DROP TABLE dispatches')
        try:asyncio.run(app_jobs(selftest=True,count=0))
        except sqlite3.OperationalError as error:assert 'no such table: dispatches' in str(error)
        else:raise AssertionError('corrupt retained Core was accepted')
        assert not handles and owners[0].closed
        with sqlite3.connect(dbpath) as db:
            assert db.execute("SELECT name FROM sqlite_master WHERE name='dispatches'").fetchall()==[]
            assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
    else:raise AssertionError('source probe mode')
    assert not install._PREPARED_OWNERS
for name,module in tuple(sys.modules.items()):
    file=getattr(module,'__file__',None)
    if name.split('.')[0] in ('service','worker','clients','contracts','lib') and file:assert Path(file).resolve().is_relative_to(source)
print(json.dumps(dict(mode=mode,actual_factory=True,unmocked_core_ddl=True,fresh_flags=True,models_inert=True,weights_loaded=False,backend_calls=len(handles),fences=fences)))
'''


class FreshSelftestFactoryTests(unittest.TestCase):
    def probe(self,mode):
        from service import native_cpu_package as package,native_cpu_install as install
        from service.provisioning import provision
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();root.chmod(0o700)
            archive=root/'source.zip';package.archive(archive);package.verify_archive(archive)
            with zipfile.ZipFile(archive) as source:source.extractall(root/'source')
            installed=root/'source/native-worker';python=root/'venv/bin/python';python.parent.mkdir(parents=True)
            shutil.copyfile(sys.executable,python);python.chmod(0o700)
            install.create_tls(root)
            install.write(root/'credential.json',dict(version=1,credentials=[dict(token='f'*96,principal='local',scopes=['read','production-write'])]))
            data={'model.onnx':b'not model weights','voices.bin':b'not voice weights'}
            descriptor=dict(schema_version=1,backend='kokoro',language='en',checkpoint='inert-factory-fixture',revision='inert-v1',
                license='inert source fixture only',access='inert source fixture, never hardware qualification',
                versions={name:'fixture' for name in ('kokoro-onnx','onnxruntime','espeakng-loader','phonemizer')},
                files=[dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),origin='https://fixture.invalid/'+name,revision='fixture') for name,raw in data.items()])
            provision(root/'models',descriptor,runtime_versions=descriptor['versions'],downloader=lambda url,path:path.write_bytes(data[url.rsplit('/',1)[1]]))
            install.write(root/'fixture-descriptor.json',descriptor)
            (root/'state').mkdir(mode=0o700)
            for role in ('control','worker','spool','staging'):
                path=root/'state'/role;path.mkdir(mode=0o770 if role=='staging' else 0o700);path.chmod(0o770 if role=='staging' else 0o700)
            config=dict(version=1,variant=package.VARIANT,root=str(root),source=str(installed),
                source_files={name:install.files.file_hash(str(installed/name)) for name in [*package.members(),'native-worker.py']},
                python=str(python),python_sha256=install.files.file_hash(str(python),executable=True),python_version=sys.version,
                dependencies=install.locked_versions(installed),deployment_id='a'*32,principal='local',installed_at=datetime.now(timezone.utc).isoformat(),port=8788,
                tls_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tls').iterdir()},
                credential_sha256=hashlib.sha256((root/'credential.json').read_bytes()).hexdigest(),qualification_session_id='b'*32,source_provenance=dict(commit=None,dirty=None))
            install.write(root/'install.json',config)
            env={k:v for k,v in os.environ.items() if k in ('PATH','SYSTEMROOT','TMPDIR')};env.update(HOME=str(root),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1')
            result=subprocess.run([sys.executable,'-I','-B','-c',PROBE,str(root),mode],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr[-10000:]);value=json.loads(result.stdout)
            self.assertTrue(value['actual_factory']);self.assertTrue(value['unmocked_core_ddl']);self.assertFalse(value['weights_loaded'])
            return value

    def test_actual_extracted_fresh_doctor_then_ordinary_without_session(self):
        value=self.probe('fresh-command');self.assertEqual(value['backend_calls'],2)

    def test_actual_retained_validated_ids_allow_six_new_and_refuse_seventh(self):
        value=self.probe('retained-caps');self.assertEqual(value['backend_calls'],7)

    def test_actual_retained_missing_table_refuses_without_ddl_repair(self):
        value=self.probe('retained-missing-table');self.assertEqual(value['backend_calls'],0)


if __name__=='__main__':unittest.main()
