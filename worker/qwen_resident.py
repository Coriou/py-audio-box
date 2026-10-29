"""Opt-in Linux Qwen model owner. AF_UNIX only; one serial model and one request at a time.

The supervisor owns this process separately from each authenticated attempt child.
No staging, journal, lifecycle key or publication path is opened here.
"""
import base64
import json
import math
import hashlib
import select
import socket
import sys
import time

MAX_FRAME = 32 * 1024 * 1024


def frame(connection, value, cutoff):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_FRAME: raise ValueError('resident_frame_limit')
    connection.settimeout(max(.001, cutoff-time.monotonic()))
    connection.sendall(len(data).to_bytes(4, 'big') + data)


def read(connection, cutoff):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            connection.settimeout(max(.001, cutoff-time.monotonic()))
            if time.monotonic() >= cutoff: raise TimeoutError('resident_deadline')
            part = connection.recv(n-len(data))
            if not part: raise EOFError('resident_closed')
            data.extend(part)
        return bytes(data)
    size = int.from_bytes(exact(4), 'big')
    if size > MAX_FRAME: raise ValueError('resident_frame_limit')
    return json.loads(exact(size))


def _result_wire(result):
    import numpy as np
    from lib.tts_backends.contract import AudioResult
    if type(result) is not AudioResult: raise ValueError('audio_result_required')
    samples = np.asarray(result.samples)
    if samples.size > 24_000_000: raise ValueError('resident_output_limit')
    return dict(samples=base64.b64encode(np.asarray(samples, dtype='<f4').tobytes()).decode(),
                shape=list(samples.shape), sample_rate=int(result.sample_rate),
                channels=int(result.channels), metadata=result.metadata)


def _result_from_wire(value):
    import numpy as np
    from lib.tts_backends.contract import AudioResult
    if type(value) is not dict or set(value) != {'samples','shape','sample_rate','channels','metadata'}: raise ValueError('resident_result')
    shape = value['shape']
    if type(shape) is not list or len(shape) not in (1,2) or any(type(x) is not int or x <= 0 for x in shape): raise ValueError('resident_result')
    count = 1
    for x in shape: count *= x
    if count > 24_000_000: raise ValueError('resident_output_limit')
    data = base64.b64decode(value['samples'], validate=True)
    if len(data) != count*4: raise ValueError('resident_result')
    return AudioResult(np.frombuffer(data, dtype='<f4').reshape(shape), value['sample_rate'], value['channels'], value['metadata'])


class Client:
    """Per-attempt backend factory. close() closes no model; supervisor controls lifetime."""
    def __init__(self, *, address, nonce, cutoff, **settings):
        self.address, self.nonce, self.cutoff, self.settings = address, nonce, cutoff, settings

    def _call(self, method, params, *, reference=None, reference_text=None, x_vector_only=None, preset=None, instruct=None, timeout_s=None):
        cutoff = min(self.cutoff, time.monotonic() + timeout_s) if timeout_s is not None else self.cutoff
        if time.monotonic() >= cutoff: raise TimeoutError('resident_deadline')
        body = dict(nonce=self.nonce, method=method, settings=self.settings, params=params,
                    reference=base64.b64encode(reference).decode() if reference is not None else None,
                    reference_text=reference_text, x_vector_only=x_vector_only, preset=preset,
                    instruct=instruct, cutoff=cutoff)
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(max(.001, cutoff-time.monotonic()))
            connection.connect(self.address)
            frame(connection, body, cutoff)
            result = read(connection, cutoff)
        if time.monotonic() >= cutoff: raise TimeoutError('resident_deadline')
        if type(result) is not dict or set(result) != {'ok','value'} or result['ok'] is not True:
            raise RuntimeError('resident_model_failed')
        return _result_from_wire(result['value'])

    def synthesise_clone(self, params, reference, reference_text, x_vector_only, timeout_s):
        if type(reference) is not bytes: raise ValueError('managed_reference_bytes_required')
        return self._call('clone',params,reference=reference,reference_text=reference_text,x_vector_only=x_vector_only,timeout_s=timeout_s)

    def synthesise_preset(self, params, preset, instruct, timeout_s):
        return self._call('custom_voice',params,preset=preset,instruct=instruct,timeout_s=timeout_s)

    def close(self):
        pass


def serve(address, roots, control, *, backend_factory=None, transition=None, ready=None):
    """One model only. Any failed request ends service so custody is never reused."""
    from lib.tts_backends.qwen import QwenBackend
    if transition is None:
        from service.installed_qwen_daemon import resident_transition
        transition = resident_transition
    if (type(roots) is not dict or set(roots) != {'clone','Ryan'} or
            any(type(value) is not dict or set(value) != {'path','revision'} for value in roots.values()) or
            len({value['path'] for value in roots.values()}) != 2):
        raise ValueError('resident_exact_bundles_required')
    factory = backend_factory or QwenBackend
    backend = None
    identity = None
    current_model = None
    grant = None
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(address)
        listener.listen(1)
        if ready is not None: ready()
        try:
            while True:
                readable, _, _ = select.select([listener, control], [], [])
                if control in readable:
                    command = read(control, time.monotonic()+2)
                    if type(command) is not dict or set(command) != {'op','authority'} or command['op'] not in ('grant','revoke'):
                        raise ValueError('resident_control')
                    if command['op'] == 'grant':
                        if grant is not None or type(command['authority']) is not dict:
                            raise ValueError('resident_control')
                        grant = command['authority']
                    else:
                        grant = None
                    frame(control, {'ok':True}, time.monotonic()+2)
                    continue
                connection, _ = listener.accept()
                with connection:
                    cutoff = time.monotonic()+1800
                    try:
                        request = read(connection, cutoff)
                        if type(request) is not dict or set(request) != {'nonce','method','settings','params','reference','reference_text','x_vector_only','preset','instruct','cutoff'} or grant is None or request['nonce'] != grant.get('nonce'):
                            raise ValueError('resident_authority')
                        # Consume before validation: a captured nonce cannot be replayed.
                        authorized = grant
                        grant = None
                        if (type(request['cutoff']) not in (int,float) or not math.isfinite(request['cutoff']) or
                                request['cutoff'] > authorized['cutoff'] or
                                any(json.dumps(request[k],sort_keys=True,allow_nan=False) !=
                                    json.dumps(authorized[k],sort_keys=True,allow_nan=False) for k in
                                    ('method','settings','params','reference_text','x_vector_only','preset','instruct'))):
                            raise ValueError('resident_authority')
                        if authorized['reference_sha256'] is None:
                            if request['reference'] is not None: raise ValueError('resident_reference_authority')
                        else:
                            raw = base64.b64decode(request['reference'], validate=True)
                            if hashlib.sha256(raw).hexdigest() != authorized['reference_sha256']:
                                raise ValueError('resident_reference_authority')
                        cutoff = min(cutoff, authorized['cutoff'], request['cutoff'])
                        if time.monotonic() >= cutoff: raise TimeoutError('resident_deadline')
                        method = request['method']; settings = request['settings']
                        requested_model = 'clone' if method == 'clone' else 'Ryan' if method == 'custom_voice' else None
                        if (requested_model is None or type(settings) is not dict or
                                set(settings) != {'model_name','num_threads','dtype_str','revision','offline','device','attention'} or
                                (settings['model_name'],settings['revision']) != (roots[requested_model]['path'],roots[requested_model]['revision']) or
                                (settings['device'],settings['dtype_str'],settings['attention'],settings['offline']) != ('cpu','float32','sdpa',True)):
                            raise ValueError('resident_model_authority')
                        decision = transition(current_model, requested_model, 0)
                        if type(request['params']) is not dict: raise ValueError('resident_params')
                        selected = (settings['model_name'], settings['revision'], settings['device'], settings['dtype_str'], settings['attention'])
                        if selected != identity:
                            if decision not in ('load','unload_then_load'): raise ValueError('resident_transition')
                            if backend is not None:
                                backend.close()  # Confirmed reap before second checkpoint loads.
                                backend = None
                            identity = None
                            current_model = None
                            backend = factory(**settings)
                            identity = selected
                            current_model = requested_model
                        elif decision != 'reuse':
                            raise ValueError('resident_transition')
                        remaining = cutoff-time.monotonic()
                        if remaining <= 0: raise TimeoutError('resident_deadline')
                        if method == 'clone':
                            if request['reference'] is None or request['preset'] is not None: raise ValueError('resident_clone')
                            reference = base64.b64decode(request['reference'],validate=True)
                            if len(reference) > 1048576: raise ValueError('resident_reference_limit')
                            result = backend.synthesise_clone(request['params'],reference,request['reference_text'],request['x_vector_only'],remaining)
                        else:
                            if request['reference'] is not None or request['preset'] is None: raise ValueError('resident_preset')
                            result = backend.synthesise_preset(request['params'],request['preset'],request['instruct'],remaining)
                        if time.monotonic() >= cutoff: raise TimeoutError('resident_deadline')
                        frame(connection,dict(ok=True,value=_result_wire(result)),cutoff)
                    except BaseException:
                        try: frame(connection,dict(ok=False,value=None),time.monotonic()+.1)
                        except BaseException: pass
                        raise
        finally:
            if backend is not None: backend.close()


def main():
    from worker.transport import receive, send
    import os
    gate = int(sys.argv[2])
    try:
        if os.read(gate, 1) != b'1': raise ValueError('resident_cgroup_assignment_required')
    finally:
        os.close(gate)
    control = socket.socket(fileno=int(sys.argv[1]))
    config = receive(control,time.monotonic()+2)
    if type(config) is not dict or set(config) != {'address','roots'} or type(config['roots']) is not dict:
        raise ValueError('resident_configuration')
    def ready(): send(control,dict(ready=True),time.monotonic()+2)
    try: serve(config['address'],config['roots'],control,ready=ready)
    finally: control.close()


if __name__ == '__main__': main()
