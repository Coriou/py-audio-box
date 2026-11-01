import copy
import os
import secrets
import socket
import struct
import threading
import time

import pytest
from service.worker_transport import exchange
from tests.release.test_worker_boundary import fixture_authority, verify
from worker.isolated_synthetic import run
from worker.transport import Session, Refused, Uncertain, authenticate, receive, send


def pair():
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


@pytest.mark.parametrize('payload,close,expected', [
    (b'', True, Uncertain), (b'\0\0', True, Uncertain),
    (struct.pack('!I', 10)+b'{}', True, Uncertain),
    (struct.pack('!I', 65537), False, Refused),
    (struct.pack('!I', 0), False, Refused),
    (struct.pack('!I', 3)+b'xxx', False, Refused),
    (struct.pack('!I', 13)+b'{"x":1,"x":2}', False, Refused),
])
def test_bad_frames(payload, close, expected):
    a,b = pair()
    with a,b:
        b.sendall(payload)
        if close: b.shutdown(socket.SHUT_WR)
        with pytest.raises(expected): receive(a, time.monotonic()+.1)


def test_incomplete_frame_absolute_timeout():
    a,b = pair()
    with a,b:
        b.sendall(struct.pack('!I', 100)+b'x')
        start = time.monotonic()
        with pytest.raises((TimeoutError, Uncertain)): receive(a, start+.05)
        assert time.monotonic()-start < .5


def test_outgoing_limit():
    a,b=pair()
    with a,b:
        with pytest.raises(Refused): send(a, 'x'*65536, time.monotonic()+1)


def test_unbound_session_and_disconnect():
    authority=fixture_authority()
    a,b=pair()
    with a,b:
        result=exchange(a,authority,Session(secrets.token_hex(32),'wrong',os.getuid(),os.getgid()),time.monotonic()+.1)
        assert result.status=='refused'
    if not hasattr(socket,'SO_PEERCRED'): return
    a,b=pair()
    with a,b:
        b.shutdown(socket.SHUT_WR)
        result=exchange(a,authority,Session(secrets.token_hex(32),authority.binding,os.getuid(),os.getgid()),time.monotonic()+.1)
        assert result.status=='uncertain' and result.message is None


@pytest.mark.skipif(not hasattr(socket,'SO_PEERCRED'),reason='Linux peer credentials required')
@pytest.mark.parametrize('fault', ['none','worker_peer','control_peer','secret','binding','tuple','deadline','plan','manifest','disconnect'])
def test_authenticated_exchange(tmp_path,fault):
    authority=fixture_authority()
    secret=secrets.token_hex(32)
    stage=tmp_path/authority.job['attempts'][-1]['staging_id'];stage.mkdir()
    config=dict(job=authority.job,context=authority.context,allowed_assets=authority.allowed_assets,
                profiles=authority.profiles,secret=secret,control_uid=os.getuid(),gid=os.getgid())
    config=copy.deepcopy(config)
    if fault=='secret': config['secret']=secrets.token_hex(32)
    if fault=='binding': config['job']['attempts'][-1]['owner_id']='other-owner'
    if fault=='control_peer': config['control_uid']=os.getuid()+1
    a,b=pair();errors=[]
    cutoff=time.monotonic()+1
    def worker():
        try:
            with b:
                if fault in ('tuple','deadline','plan','manifest','disconnect'):
                    session=Session(secret,authority.binding,os.getuid(),os.getgid())
                    authenticate(b,session,cutoff,control=False)
                    message=receive(b,cutoff)
                    if fault=='disconnect': return
                    # Authentic peer with a forged result, rather than unauthenticated garbage.
                    fixture=__import__('json').loads(open('contracts/fixtures/v1/transcript-speech_generate.json').read())
                    proposal=dict(message=message,manifest=fixture['verified_manifest'])
                    proposal['message'].update(kind='result',stage='completed',result_asset_ids=[o['id'] for o in proposal['manifest']['outputs']])
                    if fault=='tuple': proposal['message']['attempt_id']='wrong-attempt'
                    if fault=='deadline': proposal['message']['deadline']='2099-01-01T00:00:00Z'
                    if fault=='plan': proposal['message']['plan']['request']['text']='changed'
                    if fault=='manifest': proposal['manifest']['owner_id']='wrong-owner'
                    send(b,proposal,cutoff)
                else: run(b,config,tmp_path,cutoff)
        except (OSError,ValueError) as error: errors.append(type(error).__name__)
    thread=threading.Thread(target=worker);thread.start()
    with a:
        result=exchange(a,authority,Session(secret,authority.binding,os.getuid()+(fault=='worker_peer'),os.getgid()),cutoff)
    thread.join(2);assert not thread.is_alive()
    if fault=='none':
        verify(result,authority,tmp_path,os.getuid());assert not errors
    elif fault=='disconnect': assert result.status=='uncertain'
    else: assert result.status in ('refused','uncertain') and result.message is None
    if fault in ('worker_peer','control_peer','secret','binding'):
        assert not (stage/'output.wav').exists()

@pytest.mark.skipif(not hasattr(socket,'SO_PEERCRED'),reason='Linux peer credentials required')
def test_worker_refuses_forged_control_proof():
    authority=fixture_authority()
    session=Session(secrets.token_hex(32),authority.binding,os.getuid(),os.getgid())
    a,b=pair();errors=[];cutoff=time.monotonic()+1
    def worker():
        try:
            with b: authenticate(b,session,cutoff,control=False)
        except Refused as error: errors.append(str(error))
    thread=threading.Thread(target=worker);thread.start()
    with a:
        hello=receive(a,cutoff)
        assert hello['role']=='worker'
        send(a,dict(role='control',nonce=secrets.token_hex(32),proof='0'*64),cutoff)
    thread.join(2)
    assert not thread.is_alive() and errors==['session_mismatch']


@pytest.mark.skipif(not hasattr(socket,'SO_PEERCRED'),reason='Linux peer credentials required')
def test_replayed_hello_cannot_answer_fresh_challenge():
    from worker.transport import proof
    authority=fixture_authority()
    session=Session(secrets.token_hex(32),authority.binding,os.getuid(),os.getgid())
    a,b=pair();result=[];cutoff=time.monotonic()+1
    wn=secrets.token_hex(32)
    def control():
        try:
            with a: authenticate(a,session,cutoff,control=True)
        except Refused as error: result.append(str(error))
    thread=threading.Thread(target=control);thread.start()
    with b:
        send(b,dict(role='worker',nonce=wn,proof=proof(session,'hello',wn,'')),cutoff)
        receive(b,cutoff)
        send(b,dict(role='worker',nonce=wn,proof=proof(session,'worker',wn,'old-challenge')),cutoff)
    thread.join(2)
    assert not thread.is_alive() and result==['session_mismatch']

@pytest.mark.skipif(not hasattr(socket,'SO_PEERCRED'),reason='Linux peer credentials required')
@pytest.mark.parametrize('field', ['attempt_id','owner_id','generation','deadline','plan','extra'])
def test_worker_rejects_authenticated_wrong_execute(tmp_path,field):
    from contracts.validate import ContractError
    from jsonschema.exceptions import ValidationError
    authority=fixture_authority();secret=secrets.token_hex(32)
    config=dict(job=authority.job,context=authority.context,allowed_assets=authority.allowed_assets,
                profiles=authority.profiles,secret=secret,control_uid=os.getuid(),gid=os.getgid())
    (tmp_path/authority.job['attempts'][-1]['staging_id']).mkdir()
    a,b=pair();errors=[];cutoff=time.monotonic()+1
    def worker():
        try:
            with b: run(b,config,tmp_path,cutoff)
        except (ContractError,ValidationError,Refused) as error: errors.append(type(error).__name__)
    thread=threading.Thread(target=worker);thread.start()
    with a:
        authenticate(a,Session(secret,authority.binding,os.getuid(),os.getgid()),cutoff,control=True)
        message=copy.deepcopy(authority.message())
        if field=='generation': message[field]+=1
        elif field=='deadline': message[field]='2099-01-01T00:00:00Z'
        elif field=='plan': message[field]['request']['text']='changed'
        elif field=='extra': message[field]='not-allowed'
        else: message[field]='wrong-tuple'
        send(a,message,cutoff)
    thread.join(2)
    assert not thread.is_alive() and errors
    assert not list(tmp_path.rglob('output.wav'))


def test_authority_cannot_be_retargeted_through_accessors():
    authority = fixture_authority()
    binding = authority.binding
    message = authority.message()
    authority.job['attempts'][-1]['owner_id'] = 'changed'
    authority.context['plan']['request']['text'] = 'changed'
    authority.profiles[0]['id'] = 'changed'
    assert authority.binding == binding and authority.message() == message
