"""Real AF_UNIX authorization checks with a fake model, no weights or Torch."""
import hashlib
import multiprocessing as mp
import shutil
import socket
import tempfile
import time
import uuid

import pytest

from worker.qwen_resident import Client, frame, read, serve


class FakeBackend:
    events = None

    def __init__(self, **settings):
        self.settings = settings
        self.events.put(('load', settings['model_name']))

    def synthesise_clone(self, params, reference, text, x_vector_only, timeout):
        from lib.tts_backends.contract import AudioResult
        import numpy as np
        if params.get('fail'): raise RuntimeError('fake_backend_fault')
        assert reference == b'RIFFfixture' and text == 'trusted transcript' and x_vector_only is False
        self.events.put(('clone', self.settings['model_name']))
        return AudioResult(np.array([0.0,.25,-.25],dtype='float32'),24000,1,
                           dict(checkpoint=self.settings['model_name'],revision=self.settings['revision']))

    def synthesise_preset(self, params, preset, instruct, timeout):
        from lib.tts_backends.contract import AudioResult
        import numpy as np
        assert preset == 'Ryan'
        self.events.put(('preset', self.settings['model_name']))
        return AudioResult(np.array([0.0,.5],dtype='float32'),24000,1,
                           dict(checkpoint=self.settings['model_name'],revision=self.settings['revision']))

    def close(self):
        self.events.put(('close', self.settings['model_name']))


def _serve(address, control, events, ready):
    FakeBackend.events = events
    def transition(current,requested,active):
        assert active == 0
        return 'reuse' if current == requested else 'load' if current is None else 'unload_then_load'
    roots={name:dict(path=path,revision='verified-revision') for name,path in
           (('clone','/models/base'),('Ryan','/models/custom'))}
    serve(address,roots,control,backend_factory=FakeBackend,transition=transition,
          ready=lambda: ready.send(True))


def _settings(name):
    return dict(model_name=name,num_threads=2,dtype_str='float32',revision='verified-revision',
                offline=True,device='cpu',attention='sdpa')


def _start():
    ctx = mp.get_context('fork')
    events = ctx.Queue()
    parent, child = ctx.Pipe(duplex=False)
    control, owner_control = socket.socketpair()
    directory = tempfile.mkdtemp(prefix='qwen-resident-test-')
    address = directory+'/socket'
    process = ctx.Process(target=_serve,args=(address,owner_control,events,child))
    process.start(); child.close(); owner_control.close()
    assert parent.poll(2) and parent.recv() is True
    parent.close()
    return process, address, control, events, directory


def _stop(process, control, directory):
    if process.is_alive(): process.terminate()
    process.join(2)
    if process.is_alive(): process.kill(); process.join(2)
    assert not process.is_alive()
    process.close(); control.close(); shutil.rmtree(directory)


def _grant(control, name, nonce, cutoff, *, params=None):
    clone=name=='/models/base'
    authority=dict(nonce=nonce,cutoff=cutoff,method='clone' if clone else 'custom_voice',
                   settings=_settings(name),params=params or {'text':'hello'},
                   reference_sha256=hashlib.sha256(b'RIFFfixture').hexdigest() if clone else None,
                   reference_text='trusted transcript' if clone else None,
                   x_vector_only=False if clone else None,preset=None if clone else 'Ryan',instruct=None)
    frame(control,dict(op='grant',authority=authority),time.monotonic()+2)
    assert read(control,time.monotonic()+2)=={'ok':True}


def _revoke(control):
    frame(control,dict(op='revoke',authority=None),time.monotonic()+2)
    assert read(control,time.monotonic()+2)=={'ok':True}


def _clone(address, nonce, cutoff, **kw):
    client=Client(address=address,nonce=nonce,cutoff=cutoff,**_settings(kw.get('name','/models/base')))
    return client.synthesise_clone(kw.get('params',{'text':'hello'}),b'RIFFfixture','trusted transcript',False,10)


def test_serial_grants_reuse_then_switch_with_confirmed_close():
    process,address,control,events,directory=_start()
    try:
        for _ in range(2):
            nonce=uuid.uuid4().hex; cutoff=time.monotonic()+2
            _grant(control,'/models/base',nonce,cutoff)
            assert _clone(address,nonce,cutoff).samples.tolist()==[0,.25,-.25]
            _revoke(control)
        nonce=uuid.uuid4().hex; cutoff=time.monotonic()+2
        _grant(control,'/models/custom',nonce,cutoff)
        client=Client(address=address,nonce=nonce,cutoff=cutoff,**_settings('/models/custom'))
        assert client.synthesise_preset({'text':'hello'},'Ryan',None,1).samples.tolist()==[0,.5]
        assert process.is_alive()
        assert [events.get(timeout=1) for _ in range(6)] == [
            ('load','/models/base'),('clone','/models/base'),('clone','/models/base'),
            ('close','/models/base'),('load','/models/custom'),('preset','/models/custom')]
    finally:
        _stop(process,control,directory)


@pytest.mark.parametrize('fault', ['replay','stale','wrong_model','extended_cutoff','wrong_params','wrong_reference'])
def test_socket_authority_rejects_replay_stale_and_changed_call(fault):
    process,address,control,events,directory=_start()
    try:
        nonce=uuid.uuid4().hex; cutoff=time.monotonic()+4
        _grant(control,'/models/base',nonce,cutoff)
        if fault in ('replay','stale'):
            assert _clone(address,nonce,cutoff).sample_rate==24000
            if fault=='stale': _revoke(control)
        if fault=='wrong_model':
            client=Client(address=address,nonce=nonce,cutoff=cutoff,**_settings('/models/custom'))
            call=lambda:client.synthesise_preset({'text':'hello'},'Ryan',None,1)
        elif fault=='extended_cutoff':
            call=lambda:_clone(address,nonce,cutoff+60)
        elif fault=='wrong_params':
            call=lambda:_clone(address,nonce,cutoff,params={'text':'different'})
        elif fault=='wrong_reference':
            client=Client(address=address,nonce=nonce,cutoff=cutoff,**_settings('/models/base'))
            call=lambda:client.synthesise_clone({'text':'hello'},b'RIFFother','trusted transcript',False,10)
        else:
            call=lambda:_clone(address,nonce,cutoff)
        with pytest.raises((RuntimeError,EOFError,ConnectionResetError)):
            call()
        process.join(2)
        assert not process.is_alive()
    finally:
        _stop(process,control,directory)


def test_backend_fault_invalidates_owner():
    process,address,control,events,directory=_start()
    try:
        nonce=uuid.uuid4().hex; cutoff=time.monotonic()+2
        _grant(control,'/models/base',nonce,cutoff,params={'fail':True})
        with pytest.raises(RuntimeError,match='resident_model_failed'):
            _clone(address,nonce,cutoff,params={'fail':True})
        process.join(2)
        assert not process.is_alive()
    finally:
        _stop(process,control,directory)
def test_resident_applies_own_limits_before_model_load(monkeypatch):
    import os
    import resource
    import socket
    import sys
    import worker.qwen_resident as resident
    import worker.transport as transport
    gate_read, gate_write = os.pipe()
    parent, child = socket.socketpair()
    calls=[]
    try:
        os.write(gate_write,b'1')
        monkeypatch.setattr(sys,'argv',['resident',str(child.fileno()),str(gate_read)])
        monkeypatch.setattr(transport,'receive',lambda *_:dict(address='inert',roots={},pids=64))
        monkeypatch.setattr(resident.resource,'setrlimit',lambda kind,limit:calls.append((kind,limit)))
        monkeypatch.setattr(resident,'serve',lambda *args,**kwargs:calls.append(('serve',None)))
        resident.main()
        assert calls==[(resource.RLIMIT_CORE,(0,0)),(resource.RLIMIT_NPROC,(64,64)),('serve',None)]
    finally:
        os.close(gate_write)
        parent.close()
        try:child.close()
        except OSError:pass
