"""Source-level model-owner tests; no weights, Torch or installed runtime claims."""
import multiprocessing as mp
import time
import uuid
import tempfile
import shutil

import pytest

from worker.qwen_resident import Client, serve


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


def _serve(address, secret, events, ready):
    FakeBackend.events = events
    def transition(current,requested,active):
        assert active == 0
        return 'reuse' if current == requested else 'load' if current is None else 'unload_then_load'
    roots={name:dict(path=path,revision='verified-revision') for name,path in
           (('clone','/models/base'),('Ryan','/models/custom'))}
    serve(address,secret,roots,backend_factory=FakeBackend,transition=transition,
          ready=lambda: ready.send(True))


def _settings(name):
    return dict(model_name=name,num_threads=2,dtype_str='float32',revision='verified-revision',
                offline=True,device='cpu',attention='sdpa')


def _start():
    ctx = mp.get_context('fork')
    events = ctx.Queue()
    parent, child = ctx.Pipe(duplex=False)
    directory = tempfile.mkdtemp(prefix='qwen-resident-test-')
    address = directory+'/socket'
    secret = uuid.uuid4().hex
    process = ctx.Process(target=_serve,args=(address,secret,events,child))
    process.start(); child.close()
    assert parent.poll(2) and parent.recv() is True
    parent.close()
    return process, address, secret, events, directory


def _stop(process):
    if process.is_alive(): process.terminate()
    process.join(2)
    if process.is_alive(): process.kill(); process.join(2)
    assert not process.is_alive()
    process.close()


def test_two_requests_reuse_then_switch_confirmed_close():
    process,address,secret,events,directory = _start()
    try:
        for _ in range(2):
            client = Client(address=address,secret=secret,cutoff=time.monotonic()+2,**_settings('/models/base'))
            result = client.synthesise_clone({'text':'hello'},b'RIFFfixture','trusted transcript',False,1)
            assert result.sample_rate == 24000 and result.samples.tolist() == [0,.25,-.25]
            client.close()
        client = Client(address=address,secret=secret,cutoff=time.monotonic()+2,**_settings('/models/custom'))
        assert client.synthesise_preset({'text':'hello'},'Ryan',None,1).samples.tolist() == [0,.5]
        assert process.is_alive()
        assert [events.get(timeout=1) for _ in range(6)] == [
            ('load','/models/base'),('clone','/models/base'),('clone','/models/base'),
            ('close','/models/base'),('load','/models/custom'),('preset','/models/custom')]
    finally:
        _stop(process)
        shutil.rmtree(directory)


def test_expired_or_bad_authority_invalidates_owner():
    process,address,secret,events,directory = _start()
    try:
        client = Client(address=address,secret=secret,cutoff=time.monotonic()+2,**_settings('/models/base'))
        client.synthesise_clone({},b'RIFFfixture','trusted transcript',False,1)
        bad = Client(address=address,secret='wrong',cutoff=time.monotonic()+2,**_settings('/models/base'))
        with pytest.raises(RuntimeError,match='resident_model_failed'):
            bad.synthesise_clone({},b'RIFFfixture','trusted transcript',False,1)
        process.join(2)
        assert not process.is_alive()
        assert [events.get(timeout=1) for _ in range(3)] == [
            ('load','/models/base'),('clone','/models/base'),('close','/models/base')]
        with pytest.raises((ConnectionRefusedError,TimeoutError,OSError)):
            client.synthesise_clone({},b'RIFFfixture','trusted transcript',False,1)
    finally:
        _stop(process)
        shutil.rmtree(directory)


def test_cancellation_reaps_owner():
    process,address,secret,events,directory = _start()
    try:
        client = Client(address=address,secret=secret,cutoff=time.monotonic()+2,**_settings('/models/base'))
        client.synthesise_clone({},b'RIFFfixture','trusted transcript',False,1)
    finally:
        _stop(process)
        shutil.rmtree(directory)


def test_expiry_before_connect_and_backend_fault():
    process,address,secret,events,directory = _start()
    try:
        expired = Client(address=address,secret=secret,cutoff=time.monotonic()-1,**_settings('/models/base'))
        with pytest.raises(TimeoutError,match='resident_deadline'):
            expired.synthesise_clone({},b'RIFFfixture','trusted transcript',False,1)
        assert process.is_alive()
        client = Client(address=address,secret=secret,cutoff=time.monotonic()+2,**_settings('/models/base'))
        with pytest.raises(RuntimeError,match='resident_model_failed'):
            client.synthesise_clone({'fail':True},b'RIFFfixture','trusted transcript',False,1)
        process.join(2)
        assert not process.is_alive()
        assert [events.get(timeout=1) for _ in range(2)] == [('load','/models/base'),('close','/models/base')]
    finally:
        _stop(process)
        shutil.rmtree(directory)
