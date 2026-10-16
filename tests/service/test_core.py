"""Real SQLite/process tests for the bounded synthetic core; no ML/network."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from contracts.validate import ContractError
from service.core import Core

FIXTURES=Path(__file__).resolve().parents[2]/'contracts/fixtures/v1'

@pytest.fixture
def context():
    return json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['context']

@pytest.fixture
def core(tmp_path):
    value=Core(tmp_path/'data')
    yield value
    if value.active: value.finish(cancel=True)
    value.close()


def test_persisted_idempotency_conflict_scope(core,context):
    first=core.submit_synthetic('p','key',context)
    changed=copy.deepcopy(context);changed['plan']['request']['text']='changed'
    with pytest.raises(ContractError,match='idempotency_conflict'):
        core.submit_synthetic('p','key',changed)
    assert core.submit_synthetic('p','key',context)==first
    assert core.submit_synthetic('other','key',context)['id']!=first['id']
    root=core.root
    core.close(); reopened=Core(root)
    assert reopened.submit_synthetic('p','key',context)==first
    reopened.close()
    # fixture still owns a valid connection for teardown
    core.__init__(root)


def test_one_owner_and_lane(core,context):
    with pytest.raises(ValueError,match='control_busy'): Core(core.root)
    job=core.submit_synthetic('p','a',context)
    core.start(delay=.2)
    assert core.job(job['id'])['state']=='running'
    with pytest.raises(ValueError,match='lane_busy'): core.start()
    assert core.finish()['state']=='succeeded'
    assert [e['state'] for e in core.events(job['id'])]==['queued','running','succeeded']
    assert len(core.jobs('p',limit=1))==1


def test_cas_and_history_no_inference(core):
    project=json.loads((FIXTURES/'production-30.json').read_text())
    core.create_project(project)
    block=project['blocks'][0]['id']
    revised=core.edit_text(project['id'],1,block,'  Exact\n é ')
    assert revised['blocks'][0]['text']=='  Exact\n é '
    with pytest.raises(ContractError,match='revision_conflict'):
        core.edit_text(project['id'],1,block,'lost update')
    assert core.project(project['id'],1)==project
    assert core.project(project['id'])==revised
    assert core.jobs('p')==[]


def test_cancel_reap_and_new_retry_lineage(core,context):
    job=core.submit_synthetic('p','a',context)
    core.start(delay=10)
    assert core.request_cancel(job['id'])['state']=='cancel_requested'
    result=core.finish(cancel=True)
    assert result['state']=='cancelled' and result['attempts'][-1]['reaped']
    retry=core.submit_synthetic('p','b',context,retry_of=job['id'])
    assert retry['id']!=job['id'] and retry['input_sha256']==job['input_sha256']
    assert core.db.execute('SELECT retry_of FROM jobs WHERE id=?',(retry['id'],)).fetchone()[0]==job['id']
    core.start(); assert core.finish()['state']=='succeeded'


def test_verified_completion_wins_cancel(core,context):
    job=core.submit_synthetic('p','a',context);core.start()
    core.active[1].wait(timeout=5)
    core.request_cancel(job['id'])
    assert core.finish()['state']=='succeeded'


def proposal(core,context):
    job=core.submit_synthetic('p','a',context);core.start()
    _,process,connection,lock=core.active
    process.wait(timeout=5)
    raw=connection.recv(65536)
    connection.close();lock.close();core.active=None
    return job,json.loads(raw)

@pytest.mark.parametrize('field,value',[('owner_id','foreign'),('generation',99),('staging_id','foreign'),('input_sha256','0'*64)])
def test_foreign_result_fences(core,context,field,value):
    job,result=proposal(core,context)
    result['message'][field]=value
    with pytest.raises(ContractError): core._publish(job['id'],result)
    assert core.job(job['id'])['state']=='running'
    core.reconcile(); assert core.job(job['id'])['state']=='failed'

@pytest.mark.parametrize('corruption',['missing','symlink','provenance','truncated','hash'])
def test_actual_file_and_provenance_verification(core,context,corruption):
    job,result=proposal(core,context)
    path=core.root/'staging'/core.job(job['id'])['attempts'][-1]['staging_id']/'output.wav'
    if corruption=='missing': path.unlink()
    if corruption=='symlink':
        path.unlink();path.symlink_to('/etc/hosts')
    if corruption=='provenance': result['manifest']['outputs'][0]['provenance']['source_id']='foreign'
    if corruption=='truncated': path.write_bytes(path.read_bytes()[:48])
    if corruption=='hash': result['manifest']['outputs'][0]['sha256']='0'*64
    with pytest.raises((OSError,ValueError,ContractError)): core._publish(job['id'],result)
    assert not core.db.execute('SELECT * FROM artifacts').fetchall()
    core.reconcile()


def test_disk_full_no_false_success(core,context,monkeypatch):
    job,result=proposal(core,context)
    import builtins
    real=builtins.open
    def failing(path,*args,**kwargs):
        if Path(path).parent==core.root/'assets': raise OSError(28,'synthetic disk full')
        return real(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'open',failing)
        with pytest.raises(OSError): core._publish(job['id'],result)
    core.reconcile()
    assert core.job(job['id'])['state']=='failed'
    assert not core.db.execute('SELECT * FROM artifacts').fetchall()


@pytest.mark.parametrize('point,state',[('before_adoption','failed'),('after_adoption','succeeded'),('after_commit','succeeded')])
def test_control_process_death_and_restart(tmp_path,point,state):
    root=tmp_path/'data'
    script='''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]));context=json.loads(Path(sys.argv[2]).read_text())['context']
c.submit_synthetic('p','a',context);c.start()
def die(point):
    if point==sys.argv[3]: os._exit(73)
c.finish(fault=die)
'''
    process=subprocess.run([sys.executable,'-c',script,str(root),str(FIXTURES/'transcript-speech_generate.json'),point],timeout=10)
    assert process.returncode==73
    core=Core(root)
    try:
        assert not core.reconcile()
        job=core.jobs('p')[0]['job']
        assert job['state']==state
        assert not core.reconcile() # repeat recovery is idempotent
        if state=='succeeded':
            asset=json.loads(core.db.execute('SELECT metadata FROM assets').fetchone()[0])
            assert (root/'assets'/asset['id']).is_file()
    finally: core.close()


def test_control_death_live_worker_quarantine_then_recover(tmp_path):
    root=tmp_path/'data'
    script='''
import json,os,sys
from pathlib import Path
from service.core import Core
c=Core(Path(sys.argv[1]));x=json.loads(Path(sys.argv[2]).read_text())['context']
c.submit_synthetic('p','a',x);c.start(delay=1);os._exit(74)
'''
    assert subprocess.run([sys.executable,'-c',script,str(root),str(FIXTURES/'transcript-speech_generate.json')],timeout=5).returncode==74
    core=Core(root)
    try:
        assert core.reconcile()
        with pytest.raises(ValueError,match='lane_quarantined'): core.start()
        deadline=time.monotonic()+5
        while core.reconcile() and time.monotonic()<deadline: time.sleep(.05)
        assert not core.reconcile()
        assert core.jobs('p')[0]['job']['state']=='failed'
    finally: core.close()


def test_terminal_stale_result_cannot_publish(core,context):
    job,result=proposal(core,context)
    core.reconcile()
    with pytest.raises(ContractError): core._publish(job['id'],result)


def test_queue_cancel_without_worker(core,context):
    job=core.submit_synthetic('p','a',context)
    core.request_cancel(job['id']);core.reconcile()
    assert core.job(job['id'])['state']=='cancelled'
    assert core.start() is None


def test_frozen_input_and_missing_published_media(core,context):
    job=core.submit_synthetic('p','a',context)
    original=copy.deepcopy(context)
    context['plan']['request']['text']='draft changed after admission'
    assert core._load(job['id'])[1]==original
    core.start();result=core.finish()
    artifact=core.artifact(result['artifact_ids'][0])
    asset=core.asset(artifact['asset_ids'][0])
    (core.root/'assets'/asset['id']).unlink()
    with pytest.raises(OSError): core.artifact(artifact['id'])
    # Corruption is detected on retrieval, never replaced by invented bytes.


def test_foreign_stage_refused(core,context):
    job,result=proposal(core,context)
    stage=core.root/'staging'/core.job(job['id'])['attempts'][-1]['staging_id']
    stage.chmod(0o755)
    with pytest.raises(ContractError,match='foreign_stage'): core._publish(job['id'],result)
    core.reconcile()


def test_explicit_unimplemented_render_and_reuse(core,context):
    render=json.loads((FIXTURES/'transcript-project_render.json').read_text())['context']
    with pytest.raises(NotImplementedError): core.submit_synthetic('p','render',render)
    context['plan']['request']['reuse']='exact'
    with pytest.raises(NotImplementedError): core.submit_synthetic('p','reuse',context)
    assert core.jobs('p')==[]


def test_deadline_kills_and_reaps(core,context):
    context['plan']['resolved']['profile']['limits']['deadline_seconds']=1
    core.submit_synthetic('p','a',context);core.start(delay=10)
    result=core.finish()
    assert result['state']=='failed' and result['stage']=='deadline'
    assert result['attempts'][-1]['reaped']
