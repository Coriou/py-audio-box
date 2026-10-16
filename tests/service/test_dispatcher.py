"""E1 deterministic policy/SQLite proofs; fake receipts are not installed E2."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import wave

import pytest
from service.dispatcher import Dispatcher
from service.worker_transport import TransportObservation
from worker.lifecycle import Observation, State


def wait(d, predicate, timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError(('timeout',d.call('status')))


def ready(d):
    return wait(d,lambda:d.call('status')['reconciled'])


class Runtime:
    def __init__(self,staging):
        self.staging=staging;self.prepares=0;self.starts=0;self.acks=0
        self.state=State.ALIVE;self.authority=None;self.block=threading.Event();self.ack_fail=False
        self.timely=True;self.corrupt=False
    def call(self,op,body):
        if op=='hello':return dict(quarantined=False)
        assert op=='ack';self.acks+=1
        if self.ack_fail:self.ack_fail=False;raise OSError('lost_ack_response')
        return dict(acked=True)
    def prepare(self,a,fixture):
        self.prepares+=1;self.authority=a;return 'lane/boot/allocation-'+str(self.prepares)
    def start(self,identity):
        self.starts+=1;return dict(state='alive')
    def observe(self,identity,timeout):
        self.block.wait(.2)
        return Observation(identity,self.state)
    def terminate(self,identity,timeout):self.state=State.STOPPED
    def collect(self,a,identity):
        receipt=dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',timely=self.timely,original_deadline=a.job['deadline'])
        if not self.timely:return receipt,None
        stage=self.staging/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770,exist_ok=True);stage.chmod(0o770)
        buf=io.BytesIO()
        with wave.open(buf,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(b'\0'*9600)
        data=buf.getvalue();(stage/'output.wav').write_bytes(data)
        digest=hashlib.sha256(data).hexdigest()
        output=dict(id='asset-test-'+str(self.prepares),sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[]))
        message=a.message();message.update(kind='result',stage='completed',result_asset_ids=[output['id']])
        manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation='speech_generate',artifact_id='artifact-test-'+str(self.prepares),outputs=[output],candidates=[])
        if self.corrupt:message['generation']+=1
        return receipt,TransportObservation('proposal','E1',message=message,manifest=manifest)


@pytest.fixture
def pair(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();runtime=Runtime(stage)
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=runtime);ready(d)
    yield d,runtime
    d.close()


def running(d,r,key='a'):
    job=d.call('submit','p',key)
    wait(d,lambda:r.starts)
    return job


def test_serialized_responsive_cancel_unknown_and_retry(pair):
    d,r=pair;job=running(d,r)
    t=time.monotonic();assert d.call('job',job['id'])['state']=='running'
    assert d.call('cancel',job['id'])['state']=='cancel_requested'
    assert time.monotonic()-t<.5
    r.state=State.UNKNOWN
    wait(d,lambda:d.call('status')['quarantined'])
    assert d.call('job',job['id'])['state']=='cancel_requested'
    assert d.call('submit','p','a')['id']==job['id']
    with pytest.raises(ValueError):d.call('replace',job['id'])
    assert r.prepares==r.starts==1 and r.acks==0
    r.timely=False;r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='cancelled' else None))
    assert result['attempts'][-1]['reaped']
    wait(d,lambda:d.call('status')['phase']=='idle')
    retry=d.call('submit','p','retry',retry_of=job['id'])
    assert retry['id']!=job['id'] and retry['input_sha256']==job['input_sha256']


def test_timely_completion_wins_cancel_and_lost_ack(pair):
    d,r=pair;job=running(d,r);r.ack_fail=True
    d.call('cancel',job['id']);r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='succeeded' else None))
    wait(d,lambda:d.call('status')['phase']=='idle')
    assert r.acks==2 and r.starts==1
    assert d.call('artifact',result['artifact_ids'][0])['input_sha256']==job['input_sha256']
    assert d.call('submit','p','a')['id']==job['id']


@pytest.mark.parametrize('timely,corrupt',[(False,False),(True,True)])
def test_late_or_stale_proposal_cannot_publish(pair,timely,corrupt):
    d,r=pair;job=running(d,r);r.timely=timely;r.corrupt=corrupt;r.state=State.STOPPED
    result=wait(d,lambda:(j if (j:=d.call('job',job['id']))['state']=='failed' else None))
    assert result['artifact_ids']==[] and result['attempts'][-1]['reaped']


def test_replacement_only_confirmed_terminal_increments_generation(pair):
    d,r=pair;job=running(d,r);r.timely=False;r.state=State.STOPPED
    wait(d,lambda:d.call('job',job['id'])['state']=='failed')
    wait(d,lambda:d.call('status')['phase']=='idle')
    old=d.call('job',job['id']);r.state=State.ALIVE
    new=d.call('replace',job['id'])
    assert new['generation']==2 and new['input_sha256']==old['input_sha256']
    for key in ('id','owner_id','staging_id'):assert new['attempts'][0][key]!=new['attempts'][1][key]
    assert new['attempts'][0]['reaped'] and not new['attempts'][1]['reaped']


@pytest.mark.parametrize('point',['prespawn','before_prepare','after_prepare','preregistration','prestart','before_start','after_start','postspawn','observation_reap','preverification','before_ack','after_ack','durable_outcome'])
def test_interruption_reconnect_no_duplicate_execution(tmp_path,point):
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage);hit=threading.Event();release=threading.Event()
    def fault(where):
        if where==point and not hit.is_set():
            hit.set();release.wait(3);raise RuntimeError('E1_control_interruption')
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r,fault=fault);ready(d)
    job=d.call('submit','p','a')
    if point in ('observation_reap','preverification','before_ack','after_ack','durable_outcome'):
        wait(d,lambda:r.starts);r.state=State.STOPPED
    assert hit.wait(4);release.set();d.close()
    before=r.starts
    d=Dispatcher(tmp_path/'core',staging=stage,runtime=r)
    try:
        ready(d);time.sleep(.3)
        assert r.starts==before
        stored=d.call('job',job['id'])
        assert len(stored['attempts'])==1 and stored['input_sha256']==job['input_sha256']
        if point in ('prespawn','before_prepare','after_prepare','preregistration'):
            assert d.call('status')['quarantined'] and not stored['artifact_ids']
    finally:d.close()


def test_unjournaled_foundation_attempt_cannot_admit(tmp_path):
    from service.core import Core
    root=tmp_path/'core';core=Core(root)
    fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
    job=core.submit_synthetic('p','a',fixture['context'])
    with core.transaction():
        job['state']='cancel_requested';core._save(job)
    core.close()
    stage=tmp_path/'stage';stage.mkdir();r=Runtime(stage)
    d=Dispatcher(root,staging=stage,runtime=r)
    try:
        ready(d);assert d.call('status')['quarantined']
        with pytest.raises(ValueError):d.call('submit','p','new')
        assert r.starts==0
    finally:d.close()


def test_admission_settings_and_job_are_one_transaction(tmp_path):
    from service.core import Core
    from service.dispatch_store import DispatchStore
    root=tmp_path/'core';core=Core(root);DispatchStore(core)
    context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    # A real SQLite failure after job INSERT rolls admission back as a whole.
    core.db.execute("CREATE TRIGGER fail_settings BEFORE INSERT ON dispatch_settings BEGIN SELECT RAISE(ABORT,'settings_failure'); END")
    import sqlite3
    try:
        with pytest.raises(sqlite3.IntegrityError):core.submit_synthetic('p','a',context,_dispatch_settings=('silence',8))
        assert core.jobs('p')==[]
        assert core.db.execute('SELECT * FROM events').fetchall()==[]
        core.db.execute('DROP TRIGGER fail_settings')
        job=core.submit_synthetic('p','a',context,_dispatch_settings=('busy',8))
        assert core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(job['id'],)).fetchone()==('busy',8)
    finally:core.close()


def test_queued_cancel_terminal_and_request_are_atomic(tmp_path):
    from service.core import Core
    import sqlite3
    core=Core(tmp_path/'core')
    context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
    job=core.submit_synthetic('p','a',context)
    core.db.execute("CREATE TRIGGER fail_cancel BEFORE INSERT ON events WHEN NEW.state='cancelled' BEGIN SELECT RAISE(ABORT,'cancel_commit_failure'); END")
    try:
        with pytest.raises(sqlite3.IntegrityError):core.request_cancel(job['id'],_cancel_unstarted=True)
        assert core.job(job['id'])['state']=='queued'
        assert [e['state'] for e in core.events(job['id'])]==['queued']
        core.db.execute('DROP TRIGGER fail_cancel')
        assert core.request_cancel(job['id'],_cancel_unstarted=True)['state']=='cancelled'
        assert [e['state'] for e in core.events(job['id'])]==['queued','cancel_requested','cancelled']
    finally:core.close()
